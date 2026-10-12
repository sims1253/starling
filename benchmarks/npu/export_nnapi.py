"""NNAPI-friendly export of the Qwen3-ASR-0.6B encoder window (800 mel frames, 104
tokens, + projector, additive key-mask input). Same math as export_qwen3_enc.py, but
only ops in the NNAPI operation set: LayerNorm as MEAN/SUB/MUL/RSQRT and GELU in its
tanh form written as x * sigmoid(2z) (NNAPI has no GELU/ERF; the sigmoid spelling keeps
the converter from fusing it back into GELU).

python -I export_nnapi.py OUTDIR      (needs OUTDIR/mel_w800.f32, kbias_w800.f32)
Writes OUTDIR/qwen3_enc_w800_nn_f32.tflite and ref_w800_nn.f32 (torch output of this graph).
"""
import glob
import os
import sys

import numpy as np
import torch
import torch.nn as nn

OUT = sys.argv[1]
SNAP = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen3-ASR-0.6B-hf/snapshots/*/"))[0]
from transformers import Qwen3ASRForConditionalGeneration  # noqa: E402

torch.set_grad_enabled(False)
model = Qwen3ASRForConditionalGeneration.from_pretrained(SNAP, dtype=torch.float32).eval()
enc, proj = model.model.audio_tower, model.model.multi_modal_projector
del model.model.language_model, model.lm_head
C = 2.0 * 0.7978845608028654


def gelu(x):
    return x * torch.sigmoid(C * (x + 0.044715 * x * x * x))


def ln(x, m):
    mu = x.mean(-1, keepdim=True)
    d = x - mu
    var = (d * d).mean(-1, keepdim=True)
    return d * torch.rsqrt(var + m.eps) * m.weight + m.bias


class Window(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc, self.proj = enc, proj
        self.register_buffer("pos", enc.positional_embedding.positional_embedding[:13].clone())

    def forward(self, mel, kbias):
        e = self.enc
        x = mel.reshape(128, 8, 100).permute(1, 0, 2).reshape(8, 1, 128, 100)
        x = gelu(e.conv2d1(x))
        x = gelu(e.conv2d2(x))
        x = gelu(e.conv2d3(x))
        x = x.permute(0, 3, 1, 2).reshape(8, 13, 480 * 16)
        h = (e.conv_out(x) + self.pos).reshape(1, 104, 896)
        for L in e.layers:
            a = L.self_attn
            y = ln(h, L.self_attn_layer_norm)
            q = a.q_proj(y).reshape(1, 104, 14, 64).transpose(1, 2)
            k = a.k_proj(y).reshape(1, 104, 14, 64).transpose(1, 2)
            v = a.v_proj(y).reshape(1, 104, 14, 64).transpose(1, 2)
            s = torch.matmul(q * a.scaling, k.transpose(2, 3)) + kbias
            p = torch.softmax(s, dim=-1)
            o = torch.matmul(p, v).transpose(1, 2).reshape(1, 104, 896)
            h = h + a.out_proj(o)
            h = h + L.fc2(gelu(L.fc1(ln(h, L.final_layer_norm))))
        h = ln(h, e.ln_post)
        return self.proj.linear_2(gelu(self.proj.linear_1(h)))


mel = torch.from_numpy(np.fromfile(f"{OUT}/mel_w800.f32", np.float32).reshape(1, 128, 800))
kb = torch.from_numpy(np.fromfile(f"{OUT}/kbias_w800.f32", np.float32).reshape(1, 1, 1, 104))
w = Window().eval()
o = w(mel, kb)[0]
o.numpy().astype(np.float32).tofile(f"{OUT}/ref_w800_nn.f32")
hf = torch.from_numpy(np.fromfile(f"{OUT}/ref_hf.f32", np.float32).reshape(-1, 1024))
A = hf.shape[0]
cos = torch.nn.functional.cosine_similarity(o[:A], hf, dim=-1)
print(f"nnapi-friendly torch vs HF: rel-L2 {((o[:A]-hf).norm()/hf.norm()).item():.3e} cos min {cos.min().item():.6f}")

import litert_torch  # noqa: E402

em = litert_torch.convert(w, (mel, kb))
path = f"{OUT}/qwen3_enc_w800_nn_f32.tflite"
em.export(path)
print("wrote", path, os.path.getsize(path) / 1e6, "MB")
