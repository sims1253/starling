"""Quantize the fp32 encoder-window tflite with ai-edge-quantizer.

python -I quantize.py OUTDIR WAV...   (WAVs give the calibration windows)
"""
import glob
import os
import sys

import numpy as np
import soundfile as sf

OUT = sys.argv[1]
WAVS = sys.argv[2:]
SNAP = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen3-ASR-0.6B-hf/snapshots/*/"))[0]
from transformers import AutoProcessor  # noqa: E402
from ai_edge_quantizer import quantizer, recipe  # noqa: E402
from ai_edge_litert import interpreter as tfl  # noqa: E402

fe = AutoProcessor.from_pretrained(SNAP).feature_extractor
wins = []
for w in WAVS:
    pcm, sr = sf.read(w, dtype="float32")
    if pcm.ndim > 1:
        pcm = pcm.mean(1)
    mel = fe(pcm, sampling_rate=16000, return_tensors="np", padding=True)["input_features"]
    for s in range(0, mel.shape[2], 800):
        m = np.zeros((1, 128, 800), np.float32)
        seg = mel[:, :, s:s + 800]
        m[:, :, :seg.shape[2]] = seg
        wins.append(m)
print("calibration windows:", len(wins))

f32 = f"{OUT}/qwen3_enc_w800_f32.tflite"
it = tfl.Interpreter(model_path=f32)
sig = it.get_signature_list()
print(sig)
skey = list(sig.keys())[0]
iname = sig[skey]["inputs"][0]
calib = {skey: [{iname: w} for w in wins]}

which = os.environ.get("RECIPES", "a8w8,a16w8,wi8").split(",")
for tag in which:
    rp = {"a8w8": recipe.static_wi8_ai8, "a16w8": recipe.static_wi8_ai16,
          "wi8": recipe.weight_only_wi8_afp32, "dyn8": recipe.dynamic_wi8_afp32}[tag]()
    q = quantizer.Quantizer(f32, rp)
    res = q.calibrate(calib, num_threads=4) if q.need_calibration else None
    path = f"{OUT}/qwen3_enc_w800_{tag}.tflite"
    if os.path.exists(path):
        os.remove(path)
    q.quantize(res, serialize_to_path=path)
    print("wrote", path, os.path.getsize(path) / 1e6, "MB", flush=True)
