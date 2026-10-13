# PARKED (2026-10-13): MOSS-Transcribe's encoder segments compile and run on the
# Pixel 10 Pro TPU, but the last segment (ln_post, projection, gated-MLP adapter)
# returns wrong rows (rel-L2 0.98, also with the adapter's up projection scaled
# by 1024), and compiling the adapter alone panicked the phone's kernel. Kept
# for reference only, not wired into the engine or the compile scripts. See
# docs/fast-engine.md, "Pixel TPU".
"""Host check of export_moss_encoder.py's segment graphs (torch, chained) on
one attention window against HF's Qwen3-Omni audio encoder + MOSS's adapter,
for a full window and a partial one (padding tokens masked): a graph check on
seeded random log-mel frames.
python -I check_moss_encoder.py MODEL [SEG]"""
import glob
import json
import os
import runpy
import sys

import torch

model = sys.argv[1]
SEG = int(sys.argv[2]) if len(sys.argv) > 2 else 8
snap = model if os.path.isdir(model) else glob.glob(os.path.expanduser(
    f"~/.cache/huggingface/hub/models--{model.replace('/', '--')}/snapshots/*/"))[0]
from safetensors import safe_open  # noqa: E402
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeAudioEncoderConfig  # noqa: E402
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeAudioEncoder  # noqa: E402

torch.set_grad_enabled(False)
cfg = json.load(open(os.path.join(snap, "config.json")))
ac = cfg["audio_config"]
enc = Qwen3OmniMoeAudioEncoder(Qwen3OmniMoeAudioEncoderConfig(**ac)).float().eval()
sd, ad = {}, {}
for f in sorted(glob.glob(os.path.join(snap, "*.safetensors"))):
    with safe_open(f, "pt") as h:
        for k in h.keys():
            if k.startswith("model.audio_model."):
                sd[k[len("model.audio_model."):]] = h.get_tensor(k).float()
            elif k.startswith("model.audio_adapter."):
                ad[k[len("model.audio_adapter."):]] = h.get_tensor(k).float()
print(enc.load_state_dict(sd, strict=False))
del sd
FR, NM, NL = ac["n_window_infer"], ac["num_mel_bins"], ac["encoder_layers"]
src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "export_moss_encoder.py")
torch.manual_seed(0)
for T in (FR, 600):
    mel = torch.randn(NM, T) * 0.5
    a = enc(input_features=mel, feature_lens=torch.tensor([T])).last_hidden_state
    F = torch.nn.functional
    ref = F.linear(F.silu(F.linear(a, ad["gate_proj.weight"])) * F.linear(a, ad["up_proj.weight"]), ad["down_proj.weight"])
    x = torch.zeros(1, NM, FR)
    x[0, :, :T] = mel
    h = x
    kb = None
    for l0 in range(0, NL, SEG):
        sys.argv = ["x", snap, str(l0), str(min(NL, l0 + SEG)), "/dev/null"]
        g = runpy.run_path(src, run_name="__seg__")
        if kb is None:
            kb = torch.full((1, 1, 1, g["TOK"]), -1e4)
            kb[..., :ref.shape[0]] = 0
        h = g["Seg"]().eval()(h, kb)
    mine = h[0, :ref.shape[0]] / g["OSCALE"]
    rel = ((mine - ref).norm() / ref.norm()).item()
    cos = F.cosine_similarity(mine, ref, dim=-1)
    print(f"T={T} frames, {ref.shape[0]} rows: rel-L2 {rel:.2e}, cos min {cos.min().item():.6f}")
