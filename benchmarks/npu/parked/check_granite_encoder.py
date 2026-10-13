# PARKED (2026-10-13): the Pixel 10 Pro TPU compiler cannot build these graphs
# (30 s full-length tensors overflow on-chip memory; sequence slices fail or
# exhaust the phone's memory). Kept for reference only, not wired into the
# engine or the compile scripts. See docs/fast-engine.md, "Pixel TPU".
"""Host check of export_granite_encoder.py's segment graphs (torch, chained,
padded to P with the masks) against HF Granite Speech's encoder + projector
on stacked input frames: a graph check, so seeded random frames as long as
the WAV's (Granite's feature extractor needs torchaudio).
python -I check_granite_encoder.py MODEL WAV [P] [SEG]"""
import glob
import json
import os
import runpy
import sys

import soundfile as sf
import torch

model, wav = sys.argv[1], sys.argv[2]
P = int(sys.argv[3]) if len(sys.argv) > 3 else 1600
SEG = int(sys.argv[4]) if len(sys.argv) > 4 else 8
snap = model if os.path.isdir(model) else glob.glob(os.path.expanduser(
    f"~/.cache/huggingface/hub/models--{model.replace('/', '--')}/snapshots/*/"))[0]
from safetensors import safe_open  # noqa: E402
from transformers import AutoConfig  # noqa: E402
from transformers.models.granite_speech.modeling_granite_speech import (  # noqa: E402
    GraniteSpeechCTCEncoder, GraniteSpeechEncoderProjector)

torch.set_grad_enabled(False)
cfg = AutoConfig.from_pretrained(snap)
pcm, sr = sf.read(wav, dtype="float32")
assert sr == 16000
torch.manual_seed(0)
x = torch.randn(1, len(pcm) // 320, cfg.encoder_config.input_dim)   # [1, T, 160] (50 frames/s)
T = x.shape[1]
enc = GraniteSpeechCTCEncoder(cfg.encoder_config).float().eval()
proj = GraniteSpeechEncoderProjector(cfg).float().eval()
esd, psd = {}, {}
for f in sorted(glob.glob(os.path.join(snap, "*.safetensors"))):
    with safe_open(f, "pt") as h:
        for k in h.keys():
            if k.startswith("encoder."):
                esd[k[len("encoder."):]] = h.get_tensor(k).float()
            elif k.startswith("projector."):
                psd[k[len("projector."):]] = h.get_tensor(k).float()
print(enc.load_state_dict(esd, strict=False), proj.load_state_dict(psd, strict=False))
ref = proj(enc(x).last_hidden_state)[0]                               # [ceil(T/15)*3, hidden]
del enc, proj, esd, psd
NL = cfg.encoder_config.num_layers
C = cfg.encoder_config.context_size
src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "export_granite_encoder.py")
xi = torch.zeros(1, P, x.shape[2])
xi[:, :T] = x[0]
fm = torch.zeros(1, P, 1)
fm[:, :T] = 1
kb = torch.full((P,), -1e4)
kb[:T] = 0
kb = kb.reshape(P // C, 1, 1, C)
h = xi
for l0 in range(0, NL, SEG):
    sys.argv = ["x", snap, str(P), str(l0), str(min(NL, l0 + SEG)), "/dev/null"]
    g = runpy.run_path(src, run_name="__seg__")
    h = g["Seg"]().eval()(h, fm, kb)
mine = h[0, :ref.shape[0]]
rel = ((mine - ref).norm() / ref.norm()).item()
cos = torch.nn.functional.cosine_similarity(mine, ref, dim=-1)
print(f"T={T} frames, {ref.shape[0]} rows: rel-L2 {rel:.2e}, cos min {cos.min().item():.6f}")
