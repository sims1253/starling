"""Export a Qwen3 LLM prefill segment (layers [L0, L1)) of a Qwen3-ASR HF
checkpoint as a TPU artifact source graph, for a fixed bucket of SB tokens:

  input  x  [1, SB, hidden] f32  (residual stream; layer 0: prompt embeddings)
  output h  [1, SB, hidden] f32  (residual stream after layer L1-1)
         kv [2*(L1-L0), n_kv, SB, head_dim] f32: per layer K (q/k-normed and
            RoPE'd, positions 0..SB-1) then V — the fast engine's cache rows.

Causal attention; rows past the real prompt are padding at the end, so they
never influence the real rows. Weights fp16 (cast after conversion) unless the
segment only converts in lightweight mode; the converter's
keep_stablehlo_constant metadata key is renamed (the phone's compiler rejects it).

python -I export_prefill.py MODEL SB L0 L1 OUT.tflite
Prints one JSON line with the key, input/output byte sizes in the tflite's
output order, and the file size.
"""
import glob
import json
import os
import sys
import tempfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

model, SB, L0, L1, out_path = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
snap = model if os.path.isdir(model) else glob.glob(os.path.expanduser(
    f"~/.cache/huggingface/hub/models--{model.replace('/', '--')}/snapshots/*/"))[0]
from safetensors import safe_open  # noqa: E402
from transformers import AutoConfig  # noqa: E402

torch.set_grad_enabled(False)
tc = AutoConfig.from_pretrained(snap).text_config
HD, NH, KV, HDIM, FF = tc.hidden_size, tc.num_attention_heads, tc.num_key_value_heads, tc.head_dim, tc.intermediate_size
eps = tc.rms_norm_eps
theta = tc.rope_parameters["rope_theta"] if hasattr(tc, "rope_parameters") and tc.rope_parameters else tc.rope_theta
W = {}
pre = "model.language_model.layers."
for f in sorted(glob.glob(os.path.join(snap, "*.safetensors"))):
    with safe_open(f, "pt") as h:
        for k in h.keys():
            if k.startswith(pre) and L0 <= int(k[len(pre):].split(".")[0]) < L1:
                W[k[len(pre):]] = h.get_tensor(k).float()


class Layer(nn.Module):
    def __init__(self, i):
        super().__init__()
        g = lambda n: nn.Parameter(W.pop(f"{i}.{n}"), requires_grad=False)  # noqa: E731
        self.ln1, self.ln2 = g("input_layernorm.weight"), g("post_attention_layernorm.weight")
        self.q, self.k, self.v, self.o = (g(f"self_attn.{n}_proj.weight") for n in "qkvo")
        self.qn, self.kn = g("self_attn.q_norm.weight"), g("self_attn.k_norm.weight")
        self.gate, self.up, self.down = (g(f"mlp.{n}_proj.weight") for n in ("gate", "up", "down"))


def rms(x, w):
    return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + eps) * w


inv = 1.0 / (theta ** (torch.arange(0, HDIM, 2).float() / HDIM))
ang = torch.arange(SB).float()[:, None] * inv[None]
COS = torch.cat([ang.cos(), ang.cos()], -1)[None, None]
SIN = torch.cat([ang.sin(), ang.sin()], -1)[None, None]
MASK = torch.triu(torch.full((SB, SB), -1e4), 1)[None, None]


def rot(x):
    return torch.cat([-x[..., HDIM // 2:], x[..., :HDIM // 2]], -1)


class Segment(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([Layer(i) for i in range(L0, L1)])
        self.register_buffer("cos", COS)
        self.register_buffer("sin", SIN)
        self.register_buffer("mask", MASK)

    def forward(self, h):
        kvs = []
        for ly in self.layers:
            x = rms(h, ly.ln1)
            q = F.linear(x, ly.q).reshape(1, SB, NH, HDIM).transpose(1, 2)
            k = F.linear(x, ly.k).reshape(1, SB, KV, HDIM).transpose(1, 2)
            v = F.linear(x, ly.v).reshape(1, SB, KV, HDIM).transpose(1, 2)
            q = rms(q, ly.qn)
            k = rms(k, ly.kn)
            q = q * self.cos + rot(q) * self.sin
            k = k * self.cos + rot(k) * self.sin
            kvs += [k[0], v[0]]
            kk = k.repeat_interleave(NH // KV, dim=1)
            vv = v.repeat_interleave(NH // KV, dim=1)
            s = torch.matmul(q, kk.transpose(2, 3)) * (HDIM ** -0.5) + self.mask
            o = torch.matmul(torch.softmax(s, -1), vv).transpose(1, 2).reshape(1, SB, NH * HDIM)
            h = h + F.linear(o, ly.o)
            x = rms(h, ly.ln2)
            h = h + F.linear(F.silu(F.linear(x, ly.gate)) * F.linear(x, ly.up), ly.down)
        return h, torch.stack(kvs, 0)


seg = Segment().eval()
x = torch.randn(1, SB, HD) * 0.02
nparams = sum(p.numel() for p in seg.parameters())
import litert_torch  # noqa: E402
from ai_edge_litert import interpreter as tfl  # noqa: E402
from ai_edge_quantizer import quantizer, recipe_manager, qtyping  # noqa: E402
from ai_edge_quantizer.algorithm_manager import AlgorithmName  # noqa: E402

with tempfile.TemporaryDirectory() as td:
    f32 = os.path.join(td, "f32.tflite")
    light = nparams * 4 > 1_200_000_000
    litert_torch.convert(seg, (x,), lightweight_conversion=light).export(f32)
    del seg
    src = f32
    if not light:
        rm = recipe_manager.RecipeManager()
        rm.add_weight_only_config(".*", qtyping.TFLOperationName.ALL_SUPPORTED, 16,
                                  algorithm_key=AlgorithmName.FLOAT_CASTING)
        src = os.path.join(td, "fp16.tflite")
        quantizer.Quantizer(f32, rm.get_quantization_recipe()).quantize(None, serialize_to_path=src)
    outs = tfl.Interpreter(model_path=src).get_output_details()
    out_bytes = [int(np.prod(o["shape"])) * 4 for o in outs]
    b = bytearray(open(src, "rb").read())
    key = b"keep_stablehlo_constant"
    assert b.count(key) <= 1, b.count(key)
    open(out_path, "wb").write(b.replace(key, b"x" * len(key)))

ckey = f"qwen3-llm-prefill/1 sb={SB} layers={L0}-{L1} hd={HD} nh={NH} kv={KV} hdim={HDIM} ff={FF}"
print(json.dumps({"key": ckey, "in": [SB * HD * 4], "out": out_bytes, "light": light,
                  "tflite": out_path, "bytes": os.path.getsize(out_path)}))
