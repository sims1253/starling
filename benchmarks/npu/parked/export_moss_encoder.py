# PARKED (2026-10-13): MOSS-Transcribe's encoder segments compile and run on the
# Pixel 10 Pro TPU, but the last segment (ln_post, projection, gated-MLP adapter)
# returns wrong rows (rel-L2 0.98, also with the adapter's up projection scaled
# by 1024), and compiling the adapter alone panicked the phone's kernel. Kept
# for reference only, not wired into the engine or the compile scripts. See
# docs/fast-engine.md, "Pixel TPU".
"""Export one segment of MOSS-Transcribe's audio encoder (Qwen3-Omni audio
tower + gated-MLP adapter) for one attention window of n_window_infer mel
frames, as a TPU artifact source graph. The tower is Qwen3-ASR's
(export_encoder.py) at 32 x 1280, too big for one graph, so it is cut into
layer segments:

  segment L0 = 0:  inputs mel [1, n_mels, frames] f32, kbias [1, 1, 1, tokens]
                   f32 (0 valid / -1e4 padding tokens of a partial window)
  later segments:  inputs h [1, tokens, d] f32, kbias
  output:          h [1, tokens, d] f32, or, for the last segment, the adapter
                   rows [1, tokens, llm hidden] f32 (ln_post, proj1, GELU,
                   proj2, adapter), times OSCALE: the adapter's SwiGLU
                   products are ~1e-4, half of them below fp16's normal
                   range, which the TPU flushes to zero; its up projection
                   is scaled by OSCALE (exact, a power of two) and the
                   engine divides the rows by it.

fp16 weights (int8 with NPU_WEIGHTS=int8, tpu_export.py).

python -I export_moss_encoder.py MODEL L0 L1 OUT.tflite
"""
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

model, L0, L1, out_path = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
snap = model if os.path.isdir(model) else glob.glob(os.path.expanduser(
    f"~/.cache/huggingface/hub/models--{model.replace('/', '--')}/snapshots/*/"))[0]
from safetensors import safe_open  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tpu_export  # noqa: E402

torch.set_grad_enabled(False)
cfg = json.load(open(os.path.join(snap, "config.json")))
ac = cfg["audio_config"]
D, H, NL, FF, NM = ac["d_model"], ac["encoder_attention_heads"], ac["encoder_layers"], ac["encoder_ffn_dim"], ac["num_mel_bins"]
DS, ODIM = ac["downsample_hidden_size"], ac["output_dim"]
AH, OUT = cfg["adapter_hidden_size"], cfg["language_config"]["hidden_size"]
FR = ac["n_window_infer"]                    # mel frames per attention window
CH = ac["n_window"] * 2                      # frames per conv chunk (100)
NCH = FR // CH
half = lambda x: (x - 1) // 2 + 1  # noqa: E731
TPC = half(half(half(CH)))                   # tokens per chunk (13)
TOK = TPC * NCH
FQ = half(half(half(NM)))
HD = D // H
OSCALE = 1024
last = L1 == NL
W = {}
for f in sorted(glob.glob(os.path.join(snap, "*.safetensors"))):
    with safe_open(f, "pt") as h:
        for k in h.keys():
            if k.startswith("model.audio_model.layers."):
                if L0 <= int(k.split(".")[3]) < L1:
                    W[k[len("model.audio_model."):]] = h.get_tensor(k).float()
            elif k.startswith("model.audio_model.conv") and L0 == 0:
                W[k[len("model.audio_model."):]] = h.get_tensor(k).float()
            elif (k.startswith("model.audio_model.ln_post") or k.startswith("model.audio_model.proj")) and last:
                W[k[len("model.audio_model."):]] = h.get_tensor(k).float()
            elif k.startswith("model.audio_adapter.") and last:
                W[k[len("model.audio_"):]] = h.get_tensor(k).float()
assert W, "no model.audio_model.* tensors (wrong checkpoint?)"


def sinusoids(length, channels, max_timescale=10000):
    inc = np.log(max_timescale) / (channels // 2 - 1)
    inv = torch.exp(-inc * torch.arange(channels // 2).float())
    t = torch.arange(length)[:, None].float() * inv[None, :]
    return torch.cat([torch.sin(t), torch.cos(t)], dim=1)


def Pm(n):
    return nn.Parameter(W.pop(n), requires_grad=False)


class Seg(nn.Module):
    def __init__(self):
        super().__init__()
        if L0 == 0:
            for i in (1, 2, 3):
                setattr(self, f"c{i}w", Pm(f"conv2d{i}.weight"))
                setattr(self, f"c{i}b", Pm(f"conv2d{i}.bias"))
            self.cow = Pm("conv_out.weight")
            self.register_buffer("pos", sinusoids(ac["max_source_positions"], D)[:TPC].clone())
        ly = []
        for li in range(L0, L1):
            m = nn.Module()
            for n in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.out_proj", "fc1", "fc2",
                      "self_attn_layer_norm", "final_layer_norm"]:
                setattr(m, n.replace(".", "_") + "_w", Pm(f"layers.{li}.{n}.weight"))
                setattr(m, n.replace(".", "_") + "_b", Pm(f"layers.{li}.{n}.bias"))
            ly.append(m)
        self.ly = nn.ModuleList(ly)
        if last:
            self.lnw, self.lnb = Pm("ln_post.weight"), Pm("ln_post.bias")
            self.p1w, self.p1b, self.p2w, self.p2b = Pm("proj1.weight"), Pm("proj1.bias"), Pm("proj2.weight"), Pm("proj2.bias")
            self.gw, self.dw = Pm("adapter.gate_proj.weight"), Pm("adapter.down_proj.weight")
            self.uw = nn.Parameter(W.pop("adapter.up_proj.weight") * OSCALE, requires_grad=False)

    def forward(self, x, kbias):
        if L0 == 0:
            x = x.reshape(NM, NCH, CH).permute(1, 0, 2).reshape(NCH, 1, NM, CH)
            x = F.gelu(F.conv2d(x, self.c1w, self.c1b, stride=2, padding=1))
            x = F.gelu(F.conv2d(x, self.c2w, self.c2b, stride=2, padding=1))
            x = F.gelu(F.conv2d(x, self.c3w, self.c3b, stride=2, padding=1))
            x = x.permute(0, 3, 1, 2).reshape(NCH, TPC, DS * FQ)
            h = (F.linear(x, self.cow) + self.pos).reshape(1, TOK, D)
        else:
            h = x
        for m in self.ly:
            y = F.layer_norm(h, (D,), m.self_attn_layer_norm_w, m.self_attn_layer_norm_b)
            q = F.linear(y, m.self_attn_q_proj_w, m.self_attn_q_proj_b).reshape(1, TOK, H, HD).transpose(1, 2)
            k = F.linear(y, m.self_attn_k_proj_w, m.self_attn_k_proj_b).reshape(1, TOK, H, HD).transpose(1, 2)
            v = F.linear(y, m.self_attn_v_proj_w, m.self_attn_v_proj_b).reshape(1, TOK, H, HD).transpose(1, 2)
            s = torch.matmul(q * HD ** -0.5, k.transpose(2, 3)) + kbias
            o = torch.matmul(torch.softmax(s, dim=-1), v).transpose(1, 2).reshape(1, TOK, D)
            h = h + F.linear(o, m.self_attn_out_proj_w, m.self_attn_out_proj_b)
            y = F.layer_norm(h, (D,), m.final_layer_norm_w, m.final_layer_norm_b)
            h = h + F.linear(F.gelu(F.linear(y, m.fc1_w, m.fc1_b)), m.fc2_w, m.fc2_b)
        if not last:
            return h
        h = F.layer_norm(h, (D,), self.lnw, self.lnb)
        h = F.linear(F.gelu(F.linear(h, self.p1w, self.p1b)), self.p2w, self.p2b)
        return F.linear(F.silu(F.linear(h, self.gw)) * F.linear(h, self.uw), self.dw)


if __name__ == "__main__":
    seg = Seg().eval()
    assert not W, f"unused tensors: {sorted(W)[:5]}"
    kb = torch.zeros(1, 1, 1, TOK)
    args = (torch.randn(1, NM, FR) * 0.5, kb) if L0 == 0 else (torch.randn(1, TOK, D), kb)
    ins, outs = tpu_export.export(seg, args, out_path)
    ckey = (f"moss-enc/1 frames={FR} tokens={TOK} layers={L0}-{L1} d={D} L={NL} H={H} ff={FF} ds={DS} odim={ODIM} "
            f"ah={AH} out={OUT} oscale={OSCALE}")
    print(json.dumps({"key": ckey, "in": ins, "out": outs, "tflite": out_path, "bytes": os.path.getsize(out_path)}))
