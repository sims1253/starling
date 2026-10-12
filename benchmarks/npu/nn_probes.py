"""Single-op / single-layer probe models for the NNAPI google-edgetpu driver.
python -I nn_probes.py OUTDIR   -> OUTDIR/probes/*.tflite (+ _a8w8 / _a16w8 quantized)"""
import glob
import os
import sys

import numpy as np
import torch
import torch.nn as nn

OUT = os.path.join(sys.argv[1], "probes")
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(0)
torch.set_grad_enabled(False)
C = 2.0 * 0.7978845608028654


def gelu(x):
    return x * torch.sigmoid(C * (x + 0.044715 * x * x * x))


class FC(nn.Module):
    def __init__(self):
        super().__init__()
        self.l = nn.Linear(896, 3584)

    def forward(self, x):
        return self.l(x)


class BMM(nn.Module):
    def forward(self, q, k):
        return torch.matmul(q, k.transpose(2, 3))


class Softmax(nn.Module):
    def forward(self, s):
        return torch.softmax(s, dim=-1)


class LN(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn(896))
        self.b = nn.Parameter(torch.randn(896))

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        d = x - mu
        return d * torch.rsqrt((d * d).mean(-1, keepdim=True) + 1e-5) * self.w + self.b


class Gelu(nn.Module):
    def forward(self, x):
        return gelu(x)


class Conv(nn.Module):
    def __init__(self):
        super().__init__()
        self.c = nn.Conv2d(480, 480, 3, 2, padding=1)

    def forward(self, x):
        return self.c(x)


class Heads(nn.Module):
    def forward(self, x):
        return x.reshape(1, 104, 14, 64).transpose(1, 2) * 0.125


SNAP = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen3-ASR-0.6B-hf/snapshots/*/"))[0]


def real_layer():
    from safetensors import safe_open
    from transformers.models.qwen3_asr.modeling_qwen3_asr import Qwen3ASRAudioEncoderLayer
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(SNAP).audio_config
    L = Qwen3ASRAudioEncoderLayer(cfg).eval()
    f = safe_open(os.path.join(SNAP, "model.safetensors"), "pt")
    pre = [k for k in f.keys() if ".layers.0." in k and "audio" in k][0].split(".layers.0.")[0] + ".layers.0."
    sd = {k[len(pre):]: f.get_tensor(k).float() for k in f.keys() if k.startswith(pre)}
    L.load_state_dict(sd)
    return L


class Layer(nn.Module):
    def __init__(self, L):
        super().__init__()
        self.L = L

    def forward(self, h, kbias):
        L, a = self.L, self.L.self_attn

        def ln(x, m):
            mu = x.mean(-1, keepdim=True)
            d = x - mu
            return d * torch.rsqrt((d * d).mean(-1, keepdim=True) + m.eps) * m.weight + m.bias
        y = ln(h, L.self_attn_layer_norm)
        q = a.q_proj(y).reshape(1, 104, 14, 64).transpose(1, 2)
        k = a.k_proj(y).reshape(1, 104, 14, 64).transpose(1, 2)
        v = a.v_proj(y).reshape(1, 104, 14, 64).transpose(1, 2)
        s = torch.matmul(q * a.scaling, k.transpose(2, 3)) + kbias
        o = torch.matmul(torch.softmax(s, dim=-1), v).transpose(1, 2).reshape(1, 104, 896)
        h = h + a.out_proj(o)
        return h + L.fc2(gelu(L.fc1(ln(h, L.final_layer_norm))))


class LayerNoMask(Layer):
    def forward(self, h):
        return super().forward(h, torch.zeros(1, 1, 1, 104))


Lr = real_layer()
probes = {
    "fc": (FC(), (torch.randn(104, 896),)),
    "bmm": (BMM(), (torch.randn(1, 14, 104, 64), torch.randn(1, 14, 104, 64))),
    "softmax": (Softmax(), (torch.randn(1, 14, 104, 104),)),
    "ln": (LN(), (torch.randn(1, 104, 896),)),
    "gelu": (Gelu(), (torch.randn(104, 3584),)),
    "conv": (Conv(), (torch.randn(8, 480, 32, 25),)),
    "heads": (Heads(), (torch.randn(1, 104, 896),)),
    "layer": (Layer(Lr), (torch.randn(1, 104, 896), torch.zeros(1, 1, 1, 104))),
    "layernm": (LayerNoMask(Lr), (torch.randn(1, 104, 896),)),
}
import litert_torch  # noqa: E402
from ai_edge_quantizer import quantizer, recipe  # noqa: E402
from ai_edge_litert import interpreter as tfl  # noqa: E402

for name, (m, args) in probes.items():
    path = f"{OUT}/{name}.tflite"
    litert_torch.convert(m.eval(), args).export(path)
    it = tfl.Interpreter(model_path=path)
    sig = it.get_signature_list()["serving_default"]
    for tag, rp in (("a8w8", recipe.static_wi8_ai8), ("a16w8", recipe.static_wi8_ai16)):
        q = quantizer.Quantizer(path, rp())
        cal = [{n: (a + 0.1 * torch.randn_like(a)).numpy() for n, a in zip(sig["inputs"], args)} for _ in range(4)]
        if name == "layer":   # a real mask range for the key-bias input
            for c in cal:
                c[sig["inputs"][1]] = np.where(np.arange(104) >= 97, -1e4, 0).astype(np.float32).reshape(1, 1, 1, 104)
        res = q.calibrate({"serving_default": cal})
        qp = f"{OUT}/{name}_{tag}.tflite"
        if os.path.exists(qp):
            os.remove(qp)
        q.quantize(res, serialize_to_path=qp)
    print("wrote", name, flush=True)
