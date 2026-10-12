"""Compare phone outputs (raw, possibly quantized) to the torch fp32 reference.
python -I cmp_phone.py LOGDIR"""
import sys

import numpy as np

LOG = sys.argv[1]
ref = np.fromfile("/tmp/npu-spike/out/ref_w800.f32", np.float32).reshape(-1, 1024)
hf = np.fromfile("/tmp/npu-spike/out/ref_hf.f32", np.float32).reshape(-1, 1024)
Q = {"a16w8": (4.235263986629434e-06, 0, np.int16), "a8w8": (0.0010653070639818907, -3, np.int8)}


def stats(r, o, name):
    d = np.abs(r - o)
    cos = (r * o).sum(1) / (np.linalg.norm(r, axis=1) * np.linalg.norm(o, axis=1) + 1e-12)
    rel = np.linalg.norm(r - o) / np.linalg.norm(r)
    print(f"{name:22s} max|d| {d.max():.3e} rel-L2 {rel:.3e} cos min {cos.min():.5f} mean {cos.mean():.6f}")


def load(path, v):
    if v in Q:
        sc, zp, dt = Q[v]
        return (np.fromfile(path, dt).astype(np.float32) - zp) * sc
    return np.fromfile(path, np.float32)


print("vs static-window torch fp32 reference, all 104 tokens")
for v in ["f32", "wi8", "a16w8", "a8w8"]:
    for a in ["npu", "cpu"]:
        try:
            o = load(f"{LOG}/{v}_{a}_out0.f32", v).reshape(-1, 1024)
        except FileNotFoundError:
            continue
        stats(ref, o, f"{v} {a}")
for v in ["wi8", "a16w8", "a8w8"]:
    stats(ref, np.fromfile(f"/tmp/npu-spike/out/host_{v}.f32", np.float32).reshape(-1, 1024), f"{v} host-cpu")
print("vs HF get_audio_features (97 valid tokens; first 96 tokens, tail chunk differs by padding)")
stats(hf[:96], ref[:96], "torch static window")
for v in ["f32", "a16w8"]:
    try:
        stats(hf[:96], load(f"{LOG}/{v}_npu_out0.f32", v).reshape(-1, 1024)[:96], f"{v} npu")
    except FileNotFoundError:
        pass
