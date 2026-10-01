"""Export SELECTIVE per-layer low-rank K factors for granite (#59, v3).

The uniform-rank frontier (see kv_spectral.md) fails the 0.2-point WER gate
at every rank with a latency win. This exporter compresses only the layers
whose runtime-K is genuinely low-rank, leaving the rest untouched: the
per-layer held-out projection error stays <= ~0.005 (vs 0.04+ mean for
uniform compression), so the total encoder perturbation drops ~50x.

Ranks are chosen from CALIBRATION data (runtime K dumps on FLEURS train
clips; the LAST --dumps prefix is held out for the printed sanity MSE) —
never from the WER test set; the FLEURS test gate remains the independent
quality test.

v3 file format (little-endian):
    char magic[8] = "STLGKVF3"
    u32 version = 3, u32 n_layers, u32 hidden, u32 n_heads
    u32 rank_k[n_layers]          (0 = layer unchanged)
    u32 rank_v[n_layers]
    layer-major payload — per layer: the K pair f32 f1[n_heads*rank*hidden],
    f32 f2[hidden*n_heads*rank] when rank_k > 0, then the V pair likewise
    (this exporter writes K only; every rank must be <= head_dim)

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
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from export_kv_lowrank import (  # noqa: E402
    check_dump_prefixes,
    dequant_2d,
    fit_pca_fold,
    load_runtime_k,
)

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
    bad = [(li, r) for li, r in rank_map.items()
           if not (0 <= li < n_layers and 0 < r <= head_dim)]
    if bad:
        raise ValueError(f"invalid --map entries (layer or rank out of range): {bad}")
    check_dump_prefixes(args.dumps, n_layers)

    fit_prefixes, held_prefix = args.dumps[:-1], args.dumps[-1]
    reader = gguf.GGUFReader(args.gguf)
    import struct
    rk_tab = [rank_map.get(li, 0) for li in range(n_layers)]
    rv_tab = [0] * n_layers
    f1k = [np.zeros(0, dtype=np.float32)] * n_layers
    f2k = [np.zeros(0, dtype=np.float32)] * n_layers
    written = {}
    for li, r in rank_map.items():
        K_fit = load_runtime_k(fit_prefixes, li, hidden, n_heads)
        K_held = load_runtime_k([held_prefix], li, hidden, n_heads)
        t = next(x for x in reader.tensors if x.name == f"enc.blk.{li}.attn_kv.weight")
        w = dequant_2d(t)
        wk = w[: w.shape[0] // 2]
        f1, f2, e = fit_pca_fold(K_fit, K_held, wk, n_heads, r)
        f1k[li], f2k[li] = f1, f2
        written[li] = e["err_held"] / e["en_held"]
        print(f"  layer {li:2d} rank {r:2d}: held-out K proj rel-MSE={written[li]:.5f}",
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
