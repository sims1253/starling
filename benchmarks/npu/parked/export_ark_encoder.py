# PARKED (2026-10-13): the Pixel 10 Pro TPU compiler cannot build these graphs
# (30 s full-length tensors overflow on-chip memory; sequence slices fail or
# exhaust the phone's memory). Kept for reference only, not wired into the
# engine or the compile scripts. See docs/fast-engine.md, "Pixel TPU".
"""Export one segment of ARK-ASR's audio encoder (Whisper with RoPE + LayerNorm
+ 4-frame merge + MLP adapter) as a TPU artifact source graph, for a fixed
bucket of P encoder positions (2P mel frames, 30 s = 1500 positions):

  segment L0 = 0:   inputs mel [1, n_mels, 2P] f32 (zero past the audio),
                    fmask [1, 1, 2P] f32 (1 on real mel frames: zeroes the
                    conv1 output past them, like the GPU's zero-padded conv),
                    kbias [1, 1, 1, P] f32 (0 real / -1e4 padding positions)
  later segments:   inputs h [1, P, d] f32, kbias
  output:           h [1, P, d] f32, or, for the last segment, the adapter
                    rows [1, P / merge, llm hidden] f32.

Global attention like the model (padding keys masked), in blocks of <= 256
positions (tpu_attention.py; P must split evenly). fp16 weights (int8 with
NPU_WEIGHTS=int8, tpu_export.py).

python -I export_ark_encoder.py MODEL P L0 L1 OUT.tflite
Prints one JSON line: key, input/output byte sizes (tflite order), size.
"""
import glob
import json
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

model, P, L0, L1, out_path = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
snap = model if os.path.isdir(model) else glob.glob(os.path.expanduser(
    f"~/.cache/huggingface/hub/models--{model.replace('/', '--')}/snapshots/*/"))[0]
from safetensors import safe_open  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tpu_export  # noqa: E402
import tpu_attention  # noqa: E402

torch.set_grad_enabled(False)
cfg = json.load(open(os.path.join(snap, "config.json")))
wc = cfg["whisper_config"]
D, H, NL, FF, NM = wc["d_model"], wc["encoder_attention_heads"], wc["encoder_layers"], wc["encoder_ffn_dim"], wc["num_mel_bins"]
HDIM = D // H
MERGE, OUT = int(cfg["merge_factor"]), cfg["hidden_size"]
last = L1 == NL
W = {}
for f in sorted(glob.glob(os.path.join(snap, "*.safetensors"))):
    with safe_open(f, "pt") as h:
        for k in h.keys():
            if not k.startswith("audio_encoder."):
                continue
            n = k[len("audio_encoder."):]
            if n.startswith("whisper.layers."):
                li = int(n.split(".")[2])
                if not (L0 <= li < L1):
                    continue
            elif n.startswith("whisper.conv") and L0 != 0:
                continue
            elif (n.startswith("layer_norm") or n.startswith("adapting")) and not last:
                continue
            W[n] = h.get_tensor(k).float()
assert W, "no audio_encoder.* tensors (wrong checkpoint?)"

# RoPE over the first head_dim/2 dims of each head, interleaved pairs (ARK's
# RotaryEmbedding(head_dim // 2), base 10000). The phone's compiler rejects
# the interleaving reshapes, so q/k rows are permuted per head (pairs ->
# halves; q.k is unchanged) and the rotation is rotate-half, as a constant
# [HDIM, HDIM] matmul over the whole head (cos 1 / sin 0 on the pass-through
# dims) inside tpu_attention's blocked attention.
RD = HDIM // 2
inv = 1.0 / (10000 ** (torch.arange(0, RD, 2).float() / RD))
fr = torch.outer(torch.arange(P).float(), inv)            # [P, RD/2]
COS = torch.cat([fr.cos(), fr.cos(), torch.ones(P, HDIM - RD)], -1)     # [P, HDIM]
SIN = torch.cat([fr.sin(), fr.sin(), torch.zeros(P, HDIM - RD)], -1)
PERM = torch.cat([torch.arange(0, RD, 2), torch.arange(1, RD, 2), torch.arange(RD, HDIM)])
ROWS = torch.cat([h * HDIM + PERM for h in range(H)])
ROT = torch.zeros(HDIM, HDIM)                             # x @ ROT = (rotate_half(x[..., :RD]), 0)
ROT[torch.arange(RD // 2) + RD // 2, torch.arange(RD // 2)] = -1.0
ROT[torch.arange(RD // 2), torch.arange(RD // 2) + RD // 2] = 1.0


def P_(n):
    return nn.Parameter(W.pop(n), requires_grad=False)


class Seg(nn.Module):
    def __init__(self):
        super().__init__()
        if L0 == 0:
            self.c1w, self.c1b, self.c2w, self.c2b = (P_(f"whisper.conv{i}.{t}") for i in (1, 2) for t in ("weight", "bias"))
        self.ly = []
        for i in range(L0, L1):
            p = f"whisper.layers.{i}."
            g = {n: P_(p + n) for n in ["self_attn.q_proj.weight", "self_attn.q_proj.bias", "self_attn.k_proj.weight",
                                        "self_attn.v_proj.weight", "self_attn.v_proj.bias", "self_attn.out_proj.weight",
                                        "self_attn.out_proj.bias", "self_attn_layer_norm.weight",
                                        "self_attn_layer_norm.bias", "final_layer_norm.weight", "final_layer_norm.bias",
                                        "fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"]}
            for n in ("self_attn.q_proj.weight", "self_attn.q_proj.bias", "self_attn.k_proj.weight"):
                g[n] = nn.Parameter(g[n].data[ROWS].contiguous(), requires_grad=False)
            mod = nn.Module()
            for n, v in g.items():
                setattr(mod, n.replace(".", "_"), v)
            self.ly.append(mod)
        self.ly = nn.ModuleList(self.ly)
        if last:
            self.lnw, self.lnb = P_("layer_norm.weight"), P_("layer_norm.bias")
            self.a0w, self.a0b, self.a2w, self.a2b = (P_(f"adapting.{i}.{t}") for i in (0, 2) for t in ("weight", "bias"))

    def forward(self, *args):
        if L0 == 0:
            mel, fmask, kbias = args
            x = F.gelu(F.conv1d(mel, self.c1w, self.c1b, padding=1)) * fmask
            h = F.gelu(F.conv1d(x, self.c2w, self.c2b, stride=2, padding=1)).transpose(1, 2)   # [1, P, D]
        else:
            h, kbias = args
        for m in self.ly:
            r = h
            y = F.layer_norm(h, (D,), m.self_attn_layer_norm_weight, m.self_attn_layer_norm_bias)
            q = F.linear(y, m.self_attn_q_proj_weight, m.self_attn_q_proj_bias)
            k = F.linear(y, m.self_attn_k_proj_weight)
            v = F.linear(y, m.self_attn_v_proj_weight, m.self_attn_v_proj_bias)
            o = tpu_attention.attention(q, k, v, H, HDIM, HDIM ** -0.5, rope=(COS, SIN, ROT), bias=kbias)
            h = r + F.linear(o, m.self_attn_out_proj_weight, m.self_attn_out_proj_bias)
            y = F.layer_norm(h, (D,), m.final_layer_norm_weight, m.final_layer_norm_bias)
            h = h + F.linear(F.gelu(F.linear(y, m.fc1_weight, m.fc1_bias)), m.fc2_weight, m.fc2_bias)
        if not last:
            return h
        h = F.layer_norm(h, (D,), self.lnw, self.lnb).reshape(1, P // MERGE, D * MERGE)
        return F.linear(F.gelu(F.linear(h, self.a0w, self.a0b)), self.a2w, self.a2b)


if __name__ == "__main__":
    seg = Seg().eval()
    kb = torch.zeros(1, 1, 1, P)
    args = (torch.randn(1, NM, 2 * P) * 0.5, torch.ones(1, 1, 2 * P), kb) if L0 == 0 else (torch.randn(1, P, D), kb)
    ins, outs = tpu_export.export(seg, args, out_path)
    ckey = f"ark-enc/1 P={P} layers={L0}-{L1} d={D} L={NL} H={H} ff={FF} mels={NM} merge={MERGE} out={OUT}"
    print(json.dumps({"key": ckey, "in": ins, "out": outs, "tflite": out_path, "bytes": os.path.getsize(out_path)}))
