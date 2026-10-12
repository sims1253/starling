"""Numerics of the phone's encoder outputs against the torch references from
benchmarks/npu/export_qwen3_enc.py.

python -I compare.py REFDIR LOGDIR...

REFDIR holds ref_w800_mask.f32 (the masked 800-frame window in torch fp32,
[104, 1024]) and ref_hf.f32 (HF get_audio_features on the real 743 frames,
[97, 1024]). Every *_out0.f32 of shape [104, 1024] in LOGDIR is compared on
the 97 valid tokens.
"""
import glob
import os
import sys

import numpy as np

ref_dir, logs = sys.argv[1], sys.argv[2:]
ref = np.fromfile(f"{ref_dir}/ref_w800_mask.f32", np.float32).reshape(-1, 1024)
hf = np.fromfile(f"{ref_dir}/ref_hf.f32", np.float32).reshape(-1, 1024)
A = hf.shape[0]


def stats(r, o):
    cos = (r * o).sum(1) / (np.linalg.norm(r, axis=1) * np.linalg.norm(o, axis=1) + 1e-12)
    rel = np.linalg.norm(r - o) / np.linalg.norm(r)
    return f"rel-L2 {rel:.2e}  cos min {cos.min():.5f} mean {cos.mean():.6f}  max|d| {np.abs(r - o).max():.2e}"


print(f"{'torch masked window vs HF':34s} {stats(hf, ref[:A])}")
for d in logs:
    for f in sorted(glob.glob(f"{d}/*_out0.f32")):
        o = np.fromfile(f, np.float32)
        if o.size != ref.size:
            continue
        o = o.reshape(-1, 1024)
        if not np.isfinite(o).all():
            print(f"{os.path.basename(f)[:-9]:34s} NON-FINITE ({(~np.isfinite(o)).sum()} values)")
            continue
        tag = os.path.basename(f)[:-9]
        print(f"{tag:34s} vs torch window: {stats(ref[:A], o[:A])}")
        print(f"{'':34s} vs HF:           {stats(hf, o[:A])}")
