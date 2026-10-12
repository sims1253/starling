"""Export the Qwen3-ASR-0.6B audio encoder (+projector) for one static
attention window (800 mel frames = 8 s, 104 tokens) to LiteRT .tflite.

Usage: python -I export_qwen3_enc.py OUTDIR WAV
Writes:
  OUTDIR/qwen3_enc_w800_f32.tflite     fp32 graph
  OUTDIR/mel_w800.f32                  [1,128,800] mel of WAV (zero padded)
  OUTDIR/ref_w800.f32                  [104,1024] torch fp32 reference (static wrapper)
  OUTDIR/ref_hf.f32                    [A,1024] HF get_audio_features on the real length
"""
import glob
import math
import os
import sys

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT, WAV = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
SNAP = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen3-ASR-0.6B-hf/snapshots/*/"))[0]

from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration  # noqa: E402

torch.set_grad_enabled(False)
model = Qwen3ASRForConditionalGeneration.from_pretrained(
    SNAP, dtype=torch.float32, attn_implementation="eager").eval()
proc = AutoProcessor.from_pretrained(SNAP)
enc = model.model.audio_tower
proj = model.model.multi_modal_projector
# Drop the LLM to save memory.
del model.model.language_model, model.lm_head

pcm, sr = sf.read(WAV, dtype="float32")
assert sr == 16000, sr
if pcm.ndim > 1:
    pcm = pcm.mean(1)
feats = proc.feature_extractor(pcm, sampling_rate=16000, return_tensors="pt",
                               return_attention_mask=True, padding=True)
mel = feats["input_features"]            # [1,128,Tpad]
mask = feats["attention_mask"]           # [1,Tpad]
T = int(mask.sum())
print("mel", tuple(mel.shape), "valid frames", T)

ref = model.get_audio_features(mel, mask).pooler_output
ref = ref.reshape(-1, ref.shape[-1])
print("HF features", tuple(ref.shape))
ref.numpy().astype(np.float32).tofile(f"{OUT}/ref_hf.f32")

W = 800  # n_window_infer: one attention window
mel_w = torch.zeros(1, 128, W)
mel_w[:, :, :min(T, W)] = mel[:, :, :min(T, W)]
mel_w.numpy().astype(np.float32).tofile(f"{OUT}/mel_w800.f32")


class Window(nn.Module):
    """One full 800-frame window: 8 chunks x 13 tokens, full attention."""

    def __init__(self, enc, proj, gelu_tanh=False):
        super().__init__()
        self.enc, self.proj = enc, proj
        self.approx = "tanh" if gelu_tanh else "none"
        self.register_buffer("pos", enc.positional_embedding.positional_embedding[:13].clone())

    def forward(self, mel, kbias=None):           # [1,128,800], [1,1,1,104] additive key bias
        e = self.enc
        x = mel.reshape(128, 8, 100).permute(1, 0, 2).reshape(8, 1, 128, 100)
        x = F.gelu(e.conv2d1(x), approximate=self.approx)
        x = F.gelu(e.conv2d2(x), approximate=self.approx)
        x = F.gelu(e.conv2d3(x), approximate=self.approx)   # [8,480,16,13]
        x = x.permute(0, 3, 1, 2).reshape(8, 13, 480 * 16)
        x = e.conv_out(x) + self.pos                         # [8,13,896]
        h = x.reshape(1, 104, 896)
        for L in e.layers:
            a = L.self_attn
            r = h
            y = L.self_attn_layer_norm(h)
            q = a.q_proj(y).reshape(1, 104, 14, 64).transpose(1, 2)
            k = a.k_proj(y).reshape(1, 104, 14, 64).transpose(1, 2)
            v = a.v_proj(y).reshape(1, 104, 14, 64).transpose(1, 2)
            s = torch.matmul(q * a.scaling, k.transpose(2, 3))
            if kbias is not None:
                s = s + kbias
            p = torch.softmax(s, dim=-1)
            o = torch.matmul(p, v).transpose(1, 2).reshape(1, 104, 896)
            h = r + a.out_proj(o)
            r = h
            y = L.final_layer_norm(h)
            y = L.fc2(F.gelu(L.fc1(y), approximate=self.approx))
            h = r + y
        h = e.ln_post(h)
        p = self.proj
        return p.linear_2(F.gelu(p.linear_1(h), approximate=self.approx))  # [1,104,1024]


win = Window(enc, proj).eval()
out = win(mel_w)[0]
out.numpy().astype(np.float32).tofile(f"{OUT}/ref_w800.f32")
A = ref.shape[0]
d = (out[:A] - ref).abs()
cos = F.cosine_similarity(out[:A], ref, dim=-1)
print(f"static window vs HF on the {A} valid tokens: max|d| {d.max():.3e} "
      f"mean|d| {d.mean():.3e} min cos {cos.min():.6f} (tail tokens see the zero pad)")

import litert_torch  # noqa: E402

# Masked variant: keys beyond the valid token count get -1e4 (partial windows).
kb = torch.zeros(1, 1, 1, 104)
kb[..., A:] = -1e4
outm = win(mel_w, kb)[0]
outm.numpy().astype(np.float32).tofile(f"{OUT}/ref_w800_mask.f32")
kb.numpy().astype(np.float32).tofile(f"{OUT}/kbias_w800.f32")
d = (outm[:A] - ref).abs()
cos = F.cosine_similarity(outm[:A], ref, dim=-1)
print(f"masked static window vs HF on {A} tokens: max|d| {d.max():.3e} min cos {cos.min():.6f}")
if os.environ.get("ONLY_MASK"):
    em = litert_torch.convert(win, (mel_w, kb))
    em.export(f"{OUT}/qwen3_enc_w800_f32mask.tflite")
    print("wrote masked")
    raise SystemExit(0)

for tag, gt in (("f32", False), ("f32tanh", True)):
    m = Window(enc, proj, gelu_tanh=gt).eval()
    em = litert_torch.convert(m, (mel_w,))
    path = f"{OUT}/qwen3_enc_w800_{tag}.tflite"
    em.export(path)
    print("wrote", path, os.path.getsize(path) / 1e6, "MB")
