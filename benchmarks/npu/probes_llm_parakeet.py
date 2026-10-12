"""Compile-coverage probes for the TPU path: (a) Qwen3-ASR-0.6B LLM prefill
(real weights, first L layers, S=112 tokens, outputs hidden + per-layer K/V),
(b) one FastConformer (Parakeet) block + 8x dw-striding subsampling at real dims
with random weights. Exported with metadata stripped (the on-device compiler
rejects the converter's metadata).  python -I probes_llm_parakeet.py OUTDIR [L]"""
import glob
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

OUT = os.path.join(sys.argv[1], "own")
L = int(sys.argv[2]) if len(sys.argv) > 2 else 4
os.makedirs(OUT, exist_ok=True)
torch.set_grad_enabled(False)
torch.manual_seed(0)
import litert_torch  # noqa: E402
from ai_edge_quantizer.utils import tfl_flatbuffer_utils as u  # noqa: E402
from ai_edge_litert.tools import flatbuffer_utils as fu  # noqa: E402


def export(m, args, name):
    p = f"{OUT}/{name}_raw.tflite"
    litert_torch.convert(m.eval(), args).export(p)
    if os.environ.get("RAW_ONLY"):
        print(name, "raw", os.path.getsize(p) / 1e6, "MB", flush=True)
        return
    mod = u.read_model(p)
    mod.signatureDefs[0].signatureKey = b"subgraph_0_fn"
    mod.metadata = None
    q = f"{OUT}/{name}_signometa.tflite"
    open(q, "wb").write(fu.convert_object_to_bytearray(mod))
    os.remove(p)
    sizes_in = [int(torch.tensor(a.shape).prod()) * 4 for a in args]
    with torch.no_grad():
        outs = m(*args)
    outs = outs if isinstance(outs, (tuple, list)) else (outs,)
    sizes_out = [int(o.numel()) * 4 for o in outs]
    print(f"{name}: {os.path.getsize(q)/1e6:.1f} MB in={','.join(map(str, sizes_in))} "
          f"out={','.join(map(str, sizes_out))}", flush=True)


# ---------------- (a) Qwen3 LLM prefill ----------------
SNAP = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen3-ASR-0.6B-hf/snapshots/*/"))[0]
from transformers import Qwen3ASRForConditionalGeneration  # noqa: E402

full = Qwen3ASRForConditionalGeneration.from_pretrained(SNAP, dtype=torch.float32).eval()
lm = full.model.language_model
cfg = full.config.text_config
S, H, KV, HD = 112, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
inv = 1.0 / (cfg.rope_parameters["rope_theta"] ** (torch.arange(0, HD, 2).float() / HD))
ang = torch.arange(S).float()[:, None] * inv[None]
COS = torch.cat([ang.cos(), ang.cos()], -1)[None, None]   # [1,1,S,HD]
SIN = torch.cat([ang.sin(), ang.sin()], -1)[None, None]
MASK = torch.triu(torch.full((S, S), -1e4), 1)[None, None]


def rms(x, w, eps):
    return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + eps) * w


def rot(x):
    a, b = x[..., :HD // 2], x[..., HD // 2:]
    return torch.cat([-b, a], -1)


class Prefill(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.layers = lm.layers[:n]
        self.register_buffer("cos", COS)
        self.register_buffer("sin", SIN)
        self.register_buffer("mask", MASK)

    def forward(self, h):                      # [1,S,1024] embeddings (audio rows already merged)
        kvs = []
        for ly in self.layers:
            a = ly.self_attn
            x = rms(h, ly.input_layernorm.weight, ly.input_layernorm.variance_epsilon)
            q = a.q_proj(x).reshape(1, S, H, HD).transpose(1, 2)
            k = a.k_proj(x).reshape(1, S, KV, HD).transpose(1, 2)
            v = a.v_proj(x).reshape(1, S, KV, HD).transpose(1, 2)
            q = rms(q, a.q_norm.weight, a.q_norm.variance_epsilon)
            k = rms(k, a.k_norm.weight, a.k_norm.variance_epsilon)
            q = q * self.cos + rot(q) * self.sin
            k = k * self.cos + rot(k) * self.sin
            kvs += [k, v]
            kk = k.repeat_interleave(H // KV, dim=1)
            vv = v.repeat_interleave(H // KV, dim=1)
            s = torch.matmul(q, kk.transpose(2, 3)) / math.sqrt(HD) + self.mask
            o = torch.matmul(torch.softmax(s, -1), vv).transpose(1, 2).reshape(1, S, H * HD)
            h = h + a.o_proj(o)
            x = rms(h, ly.post_attention_layernorm.weight, ly.post_attention_layernorm.variance_epsilon)
            m = ly.mlp
            h = h + m.down_proj(F.silu(m.gate_proj(x)) * m.up_proj(x))
        return (h, *kvs)


export(Prefill(L), (torch.randn(1, S, 1024) * 0.02,), f"qwen3_prefill_L{L}")
del full, lm
if os.environ.get("RAW_ONLY"):
    raise SystemExit(0)

# ---------------- (b) FastConformer block (Parakeet) ----------------
D, NH, T = 1024, 8, 93


class Subsample(nn.Module):        # dw_striding 8x: conv 3x3 s2, then 2x (dw 3x3 s2 + pw 1x1), ReLU
    def __init__(self, c=256):
        super().__init__()
        self.c1 = nn.Conv2d(1, c, 3, 2, 1)
        self.d2, self.p2 = nn.Conv2d(c, c, 3, 2, 1, groups=c), nn.Conv2d(c, c, 1)
        self.d3, self.p3 = nn.Conv2d(c, c, 3, 2, 1, groups=c), nn.Conv2d(c, c, 1)
        self.out = nn.Linear(c * 16, D)

    def forward(self, mel):                    # [1,1,T*8,128]
        x = F.relu(self.c1(mel))
        x = F.relu(self.p2(self.d2(x)))
        x = F.relu(self.p3(self.d3(x)))        # [1,c,T,16]
        x = x.permute(0, 2, 1, 3).reshape(1, x.shape[2], -1)
        return self.out(x)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ff1a, self.ff1b = nn.Linear(D, 4 * D), nn.Linear(4 * D, D)
        self.ff2a, self.ff2b = nn.Linear(D, 4 * D), nn.Linear(4 * D, D)
        self.ln = nn.ModuleList([nn.LayerNorm(D) for _ in range(5)])
        self.q, self.k, self.v, self.o, self.pos = (nn.Linear(D, D) for _ in range(5))
        self.bu = nn.Parameter(torch.randn(NH, D // NH) * 0.02)
        self.bv = nn.Parameter(torch.randn(NH, D // NH) * 0.02)
        self.pw1, self.dw, self.pw2 = nn.Conv1d(D, 2 * D, 1), nn.Conv1d(D, D, 9, padding=4, groups=D), nn.Conv1d(D, D, 1)
        self.bn = nn.BatchNorm1d(D).eval()

    def mhsa(self, x, p):                      # Transformer-XL relative attention with rel_shift
        dk = D // NH
        q = self.q(x).reshape(1, T, NH, dk)
        k = self.k(x).reshape(1, T, NH, dk).transpose(1, 2)
        v = self.v(x).reshape(1, T, NH, dk).transpose(1, 2)
        pe = self.pos(p).reshape(1, 2 * T - 1, NH, dk).transpose(1, 2)
        ac = torch.matmul((q + self.bu).transpose(1, 2), k.transpose(2, 3))
        bd = torch.matmul((q + self.bv).transpose(1, 2), pe.transpose(2, 3))      # [1,NH,T,2T-1]
        bd = F.pad(bd, (1, 0)).reshape(1, NH, 2 * T, T)[:, :, 1:].reshape(1, NH, T, 2 * T - 1)[..., :T]
        a = torch.softmax((ac + bd) / math.sqrt(dk), -1)
        return self.o(torch.matmul(a, v).transpose(1, 2).reshape(1, T, D))

    def forward(self, x, p):
        x = x + 0.5 * self.ff1b(F.silu(self.ff1a(self.ln[0](x))))
        x = x + self.mhsa(self.ln[1](x), p)
        c = self.ln[2](x).transpose(1, 2)
        c = F.glu(self.pw1(c), dim=1)
        c = self.pw2(F.silu(self.bn(self.dw(c))))
        x = x + c.transpose(1, 2)
        x = x + 0.5 * self.ff2b(F.silu(self.ff2a(self.ln[3](x))))
        return self.ln[4](x)


export(Subsample(), (torch.randn(1, 1, T * 8, 128),), "parakeet_subsample")
export(Block(), (torch.randn(1, T, D), torch.randn(1, 2 * T - 1, D)), "parakeet_block")
