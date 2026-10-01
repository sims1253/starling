"""Export SELECTIVE per-layer low-rank K factors for granite (#59, v3).

The uniform-rank frontier (see kv_spectral.md) fails the 0.2-point WER gate
at every rank with a latency win. This exporter compresses only the layers
whose runtime-K is genuinely low-rank, leaving the rest untouched: the
per-layer held-out projection error stays <= ~0.005 (vs 0.04+ mean for
uniform compression), so the total encoder perturbation drops ~50x.

Ranks are chosen from CALIBRATION data (runtime K dumps on FLEURS train
clips, held-out clip 7) — never from the WER test set; the FLEURS test gate
remains the independent quality test.

v3 file format (little-endian):
    char magic[8] = "STLGKVF3"
    u32 version = 3, u32 n_layers, u32 hidden, u32 n_heads
    u32 rank_k[n_layers]          (0 = layer unchanged)
    u32 rank_v[n_layers]
    per layer with rank > 0: f32 f1[n_heads*rank*hidden], f32 f2[hidden*n_heads*rank]

Basis: runtime-fitted PCA (top-r right singular vectors of the layer's
runtime K over the fit clips), folded f1 = (W^T B)^T, f2 = B — the same
construction as export_kv_lowrank_runtime.py.

Usage:
    uv run python benchmarks/export_kv_lowrank_selective.py \
        --gguf models/granite-2b-dynq4.imx.gguf \
        --dumps .auto/kdumps/clip0 ... --map 0:32,6:32,9:8,10:24,11:32,13:32 \
        --out benchmarks/results/kvfactors_granite_sel.bin
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from export_kv_lowrank import dequant_2d  # noqa: E402

import gguf  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--dumps", nargs="+", required=True,
                    help="STARLING_GRANITE_DUMP_K prefixes; the LAST is held out")
    ap.add_argument("--map", required=True,
                    help="layer:rank pairs, e.g. 0:32,6:32,9:8")
    ap.add_argument("--layers", type=int, default=16)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    n_layers, n_heads, hidden = args.layers, args.heads, args.hidden
    head_dim = hidden // n_heads
    rank_map = {}
    for part in args.map.split(","):
        li, r = part.split(":")
        rank_map[int(li)] = int(r)
    assert all(0 <= li < n_layers and 0 < r <= head_dim for li, r in rank_map.items())

    for p in args.dumps:
        n = len(glob.glob(f"{p}.L*.f32"))
        assert n == n_layers, f"{p}: expected {n_layers} layer dumps, got {n}"

    fit_prefixes, held_prefix = args.dumps[:-1], args.dumps[-1]
    reader = gguf.GGUFReader(args.gguf)
    import struct
    rk_tab = [rank_map.get(li, 0) for li in range(n_layers)]
    rv_tab = [0] * n_layers
    f1k = [np.zeros(0, dtype=np.float32)] * n_layers
    f2k = [np.zeros(0, dtype=np.float32)] * n_layers
    written = {}
    for li, r in rank_map.items():
        def load_K(prefixes):
            mats = []
            for p in prefixes:
                a = np.fromfile(f"{p}.L{li}.f32", dtype=np.float32)
                m = a.reshape(-1, hidden).T
                mats.append(m[:, np.any(m != 0.0, axis=0)])
            k = np.concatenate(mats, axis=1)
            return k.reshape(n_heads, head_dim, -1).transpose(2, 0, 1)

        K_fit, K_held = load_K(fit_prefixes), load_K([held_prefix])
        t = next(x for x in reader.tensors if x.name == f"enc.blk.{li}.attn_kv.weight")
        w = dequant_2d(t)
        wk = w[: w.shape[0] // 2]
        f1 = np.zeros((n_heads * r, hidden), dtype=np.float32)
        f2 = np.zeros((hidden, n_heads * r), dtype=np.float32)
        err = en = 0.0
        for h in range(n_heads):
            Xf = K_fit[:, h, :].astype(np.float64)
            _, _, Vt = np.linalg.svd(Xf, full_matrices=False)
            B = Vt[: r].T
            Xh = K_held[:, h, :].astype(np.float64)
            err += float((((Xh @ B) @ B.T - Xh) ** 2).sum())
            en += float((Xh ** 2).sum())
            W_h = wk[h * head_dim:(h + 1) * head_dim].astype(np.float64)
            M = W_h.T @ B
            f1[h * r:(h + 1) * r, :] = M.T.astype(np.float32)
            f2[h * head_dim:(h + 1) * head_dim, h * r:(h + 1) * r] = B.astype(np.float32)
        f1k[li], f2k[li] = f1, f2
        written[li] = err / en
        print(f"  layer {li:2d} rank {r:2d}: held-out K proj rel-MSE={err / en:.5f}",
              flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        f.write(b"STLGKVF3")
        f.write(struct.pack("<4I", 3, n_layers, hidden, n_heads))
        f.write(struct.pack(f"<{n_layers}I", *rk_tab))
        f.write(struct.pack(f"<{n_layers}I", *rv_tab))
        for li in range(n_layers):
            if rk_tab[li] > 0:
                f1k[li].tofile(f)
                f2k[li].tofile(f)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB) map={rank_map}")
    print("worst compressed layer:",
          max(written.items(), key=lambda kv: kv[1]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
