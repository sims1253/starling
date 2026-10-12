"""Host transcript of short.wav with the HF Qwen3-ASR-0.6B decoder, feeding audio
embeddings from: the HF encoder (reference) and each given raw [104,1024] f32 file
(first A valid rows).  python -I transcribe.py WAV feat.f32 ..."""
import glob
import os
import sys

import numpy as np
import soundfile as sf
import torch
from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

SNAP = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen3-ASR-0.6B-hf/snapshots/*/"))[0]
torch.set_grad_enabled(False)
proc = AutoProcessor.from_pretrained(SNAP)
model = Qwen3ASRForConditionalGeneration.from_pretrained(SNAP, dtype=torch.float32).eval()
pcm, sr = sf.read(sys.argv[1], dtype="float32")
inputs = proc.apply_transcription_request(audio=[pcm], return_tensors="pt") \
    if hasattr(proc, "apply_transcription_request") else None
orig = model.model.get_audio_features
ref = orig(inputs["input_features"], inputs["input_features_mask"]).pooler_output
A = ref.reshape(-1, 1024).shape[0]


def run(feats, name):
    if feats is not None:
        def fake(*a, **k):
            o = orig(*a, **k)
            o.pooler_output = feats.reshape(o.pooler_output.shape).to(o.pooler_output.dtype)
            return o
        model.model.get_audio_features = fake
    else:
        model.model.get_audio_features = orig
    out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    txt = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    print(f"{name:28s} | {txt}")


run(None, "HF encoder (reference)")
for p in sys.argv[2:]:
    raw = np.fromfile(p, np.float32).reshape(-1, 1024)[:A]
    run(torch.from_numpy(raw.copy()), os.path.basename(p))
