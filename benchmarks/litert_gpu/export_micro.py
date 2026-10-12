"""Micro models for pricing LiteRT GPU (ML Drift) kernels at Starling's real shapes.

python -I export_micro.py OUTDIR

Writes OUTDIR/fc_M{M}_K{K}_N{N}.tflite: one FullyConnected y = x @ W^T (+ no bias),
x [1, M, K], random fp32 weights (the GPU accelerator stores them in its own
precision, fp16 in fp16 mode); OUTDIR/fc16_M{M}_K{K}_N{N}.tflite: 16 independent
FullyConnected ops of that shape on the same x, summed, so one run keeps the
GPU busy long enough for its clock to ramp (a lone FC runs at the idle clock); and OUTDIR/qwen3_layer_S{S}.tflite: one
Qwen3-0.6B decoder layer at prefill size (RMSNorm, q/k/v, q/k RMSNorm, RoPE,
causal GQA attention 16q/8kv x 128, o, RMSNorm, SwiGLU 3072), random weights.
"""
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

import litert_torch

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(0)
torch.set_grad_enabled(False)

# (M, K, N): the task's 104x1024 @ 1024x4096; the Qwen3-ASR encoder linears
# (d 896, ffn 3584, 104 tokens per 8 s window); the Qwen3-0.6B LLM linears at
# the short-clip prefill (S = 112) and a medium-clip-ish prefill (S = 320):
# qkv 1024 -> 4096, o 2048 -> 1024, gate+up 1024 -> 6144, down 3072 -> 1024.
SHAPES = [
    (104, 1024, 4096),
    (104, 896, 896), (104, 896, 3584), (104, 3584, 896),
    (112, 1024, 4096), (112, 2048, 1024), (112, 1024, 6144), (112, 3072, 1024),
    (320, 1024, 4096), (320, 1024, 6144), (320, 3072, 1024),
]


class FC(nn.Module):
    def __init__(self, k, n):
        super().__init__()
        self.lin = nn.Linear(k, n, bias=False)
        nn.init.uniform_(self.lin.weight, -1 / math.sqrt(k), 1 / math.sqrt(k))

    def forward(self, x):
        return self.lin(x)


class FC16(nn.Module):
    def __init__(self, k, n, reps=16):
        super().__init__()
        self.fcs = nn.ModuleList([FC(k, n) for _ in range(reps)])

    def forward(self, x):
        y = self.fcs[0](x)
        for f in self.fcs[1:]:
            y = y + f(x)
        return y


for M, K, N in SHAPES:
    for tag, cls in (("fc", FC), ("fc16", FC16)):
        path = f"{OUT}/{tag}_M{M}_K{K}_N{N}.tflite"
        if os.path.exists(path):
            continue
        m = cls(K, N).eval()
        litert_torch.convert(m, (torch.randn(1, M, K),)).export(path)
        print("wrote", path)


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(1 + 0.1 * torch.randn(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w


class Qwen3Layer(nn.Module):
    D, H, KV, HD, FF = 1024, 16, 8, 128, 3072

    def __init__(self, S):
        super().__init__()
        D, H, KV, HD, FF = self.D, self.H, self.KV, self.HD, self.FF

        def lin(i, o):
            m = nn.Linear(i, o, bias=False)
            nn.init.uniform_(m.weight, -1 / math.sqrt(i), 1 / math.sqrt(i))
            return m

        self.n1, self.n2 = RMSNorm(D), RMSNorm(D)
        self.qn, self.kn = RMSNorm(HD), RMSNorm(HD)
        self.q, self.k, self.v, self.o = lin(D, H * HD), lin(D, KV * HD), lin(D, KV * HD), lin(H * HD, D)
        self.g, self.u, self.d = lin(D, FF), lin(D, FF), lin(FF, D)
        inv = 1.0 / (1e6 ** (torch.arange(0, HD, 2).float() / HD))
        ang = torch.arange(S).float()[:, None] * inv[None]
        emb = torch.cat([ang, ang], -1)
        self.register_buffer("cos", emb.cos()[None, None])
        self.register_buffer("sin", emb.sin()[None, None])
        self.register_buffer("mask", torch.triu(torch.full((S, S), -1e4), 1)[None, None])

    @staticmethod
    def rot(x):
        h = x.shape[-1] // 2
        return torch.cat([-x[..., h:], x[..., :h]], -1)

    def forward(self, x):                      # [1, S, 1024]
        S = x.shape[1]
        H, KV, HD = self.H, self.KV, self.HD
        y = self.n1(x)
        q = self.qn(self.q(y).reshape(1, S, H, HD)).transpose(1, 2)
        k = self.kn(self.k(y).reshape(1, S, KV, HD)).transpose(1, 2)
        v = self.v(y).reshape(1, S, KV, HD).transpose(1, 2)
        q = q * self.cos + self.rot(q) * self.sin
        k = k * self.cos + self.rot(k) * self.sin
        # GQA by slicing + concat (repeat_interleave lowers to a 5-D
        # BROADCAST_TO the GPU delegate rejects).
        r = H // KV
        k = torch.cat([k[:, i:i + 1] for i in range(KV) for _ in range(r)], dim=1)
        v = torch.cat([v[:, i:i + 1] for i in range(KV) for _ in range(r)], dim=1)
        s = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(HD) + self.mask
        a = torch.matmul(torch.softmax(s, -1), v).transpose(1, 2).reshape(1, S, H * HD)
        x = x + self.o(a)
        y = self.n2(x)
        return x + self.d(F.silu(self.g(y)) * self.u(y))


for S in (112,):
    path = f"{OUT}/qwen3_layer_S{S}.tflite"
    if not os.path.exists(path):
        m = Qwen3Layer(S).eval()
        litert_torch.convert(m, (torch.randn(1, S, 1024),)).export(path)
        print("wrote", path)


class Layers(nn.Module):
    def __init__(self, S, n):
        super().__init__()
        self.layers = nn.ModuleList([Qwen3Layer(S) for _ in range(n)])

    def forward(self, x):
        for L in self.layers:
            x = L(x)
        return x


# Eight chained layers: long enough per run for the GPU clock to ramp.
for S, n in ((112, 8),):
    path = f"{OUT}/qwen3_layer{n}_S{S}.tflite"
    if not os.path.exists(path):
        litert_torch.convert(Layers(S, n).eval(), (torch.randn(1, S, 1024),)).export(path)
        print("wrote", path)
