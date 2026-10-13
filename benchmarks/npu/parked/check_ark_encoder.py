# PARKED (2026-10-13): the Pixel 10 Pro TPU compiler cannot build these graphs
# (30 s full-length tensors overflow on-chip memory; sequence slices fail or
# exhaust the phone's memory). Kept for reference only, not wired into the
# engine or the compile scripts. See docs/fast-engine.md, "Pixel TPU".
"""Host check of export_ark_encoder.py's segment graphs (torch, chained over
all layers, padded to P with the masks) against the HF ARK audio encoder on
the unpadded mel of a WAV.  python -I check_ark_encoder.py MODEL WAV [P] [SEG]"""
import glob
import json
import os
import runpy
import sys

import numpy as np
import soundfile as sf
import torch

model, wav = sys.argv[1], sys.argv[2]
P = int(sys.argv[3]) if len(sys.argv) > 3 else 1500
SEG = int(sys.argv[4]) if len(sys.argv) > 4 else 8
snap = model if os.path.isdir(model) else glob.glob(os.path.expanduser(
    f"~/.cache/huggingface/hub/models--{model.replace('/', '--')}/snapshots/*/"))[0]
from transformers import AutoModelForCausalLM, WhisperFeatureExtractor  # noqa: E402

torch.set_grad_enabled(False)
m = AutoModelForCausalLM.from_pretrained(snap, trust_remote_code=True, dtype=torch.float32).eval()
fe = WhisperFeatureExtractor.from_pretrained(snap)
pcm, sr = sf.read(wav, dtype="float32")
mel = torch.from_numpy(fe(pcm, sampling_rate=16000, return_tensors="np", padding="longest")["input_features"])
T = mel.shape[-1] - (mel.shape[-1] % 2)
mel = mel[..., :T]
ref = m.audio_encoder(mel)[0]                    # [T/2 / merge, hidden]
NL = json.load(open(os.path.join(snap, "config.json")))["whisper_config"]["encoder_layers"]
src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "export_ark_encoder.py")
Tp = T // 2
x = torch.zeros(1, mel.shape[1], 2 * P)
x[..., :T] = mel
fm = torch.zeros(1, 1, 2 * P)
fm[..., :T] = 1
kb = torch.full((1, 1, 1, P), -1e4)
kb[..., :Tp] = 0
h = None
for l0 in range(0, NL, SEG):
    sys.argv = ["x", snap, str(P), str(l0), str(min(NL, l0 + SEG)), "/dev/null"]
    g = runpy.run_path(src, run_name="__seg__")
    seg = g["Seg"]().eval()
    h = seg(x, fm, kb) if l0 == 0 else seg(h, kb)
mine = h[0, :ref.shape[0]]
rel = ((mine - ref).norm() / ref.norm()).item()
cos = torch.nn.functional.cosine_similarity(mine, ref, dim=-1)
print(f"T={T} frames, {ref.shape[0]} adapter rows: rel-L2 {rel:.2e}, cos min {cos.min().item():.6f}")
