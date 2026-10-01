"""Export RUNTIME-fitted low-rank K factors for the granite encoder (#59).

Third basis provenance, after export_kv_lowrank.py (weight-space SVD) and
export_kv_lowrank_activation.py (bf16 reference activations): the basis is
fit on K captured from the NATIVE runtime itself. STARLING_GRANITE_DUMP_K=
<prefix> makes the engine write one <prefix>.L<layer>.f32 per layer
(row-major [hidden, T_pad]; padded columns are exactly zero), so the fit
sees the Q4_K weights, the bf16 rounding boundaries and the real block
padding of the engine that will consume it.

    B_h = top-r right singular vectors of runtime K_h  [128, r]
    f1  = (W_h^T B)^T   (z = n @ f1^T lives in the r-dim basis)
    f2  = B             (k_h ~= z @ B^T)

Same v1 factor-file format as the weight-space exporter.

Measured outcome (2026-10-01, this campaign): identical per-layer K
projection error to the bf16 fit (~0.04 mean, layers 3/7 worst at ~0.11)
and the same WER outcome (+0.28 at r=32, gate limit 0.2). Basis provenance
is NOT what makes activation-fitted bases fail the WER gate; see
benchmarks/kv_spectral.md for the OOD-tail discussion.

Usage:
    uv run python benchmarks/export_kv_lowrank_runtime.py \
        --gguf models/granite-2b-dynq4.imx.gguf \
        --dumps .auto/kdumps/clip0 ... --rank-k 32 \
        --out benchmarks/results/kvfactors_granite_rt_k32.bin
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from export_kv_lowrank import (  # noqa: E402
    attn_kv_weight,
    check_dump_prefixes,
    check_rank,
    dequant_2d,
    fit_pca_fold,
    load_runtime_k,
)

import gguf  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--dumps", nargs="+", required=True,
                    help="STARLING_GRANITE_DUMP_K prefixes (one per clip)")
    ap.add_argument("--rank-k", type=int, default=32)
    ap.add_argument("--layers", type=int, default=16)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    n_layers, n_heads, hidden = args.layers, args.heads, args.hidden
    head_dim = hidden // n_heads
    check_rank(args.rank_k, head_dim, "--rank-k")
    check_dump_prefixes(args.dumps, n_layers)

    # Held-out check: the last dump prefix. Fit: all the others.
    fit_prefixes = args.dumps[:-1]
    held_prefix = args.dumps[-1]

    reader = gguf.GGUFReader(args.gguf)
    import struct
    f1k_l, f2k_l = [], []
    for li in range(n_layers):
        K_fit = load_runtime_k(fit_prefixes, li, hidden, n_heads)   # [T, H, D]
        K_held = load_runtime_k([held_prefix], li, hidden, n_heads)
        w = dequant_2d(attn_kv_weight(reader, li))
        wk = w[: w.shape[0] // 2]
        f1, f2, e = fit_pca_fold(K_fit, K_held, wk, n_heads, args.rank_k)
        f1k_l.append(f1)
        f2k_l.append(f2)
        print(f"  layer {li:2d}: runtime-K proj rel-MSE "
              f"fit={e['err_fit'] / e['en_fit']:.4f} "
              f"held-out={e['err_held'] / e['en_held']:.4f}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        f.write(b"STLGKVF1")
        f.write(struct.pack("<6I", 1, n_layers, hidden, n_heads, args.rank_k, 0))
        for i in range(n_layers):
            f1k_l[i].tofile(f)
            f2k_l[i].tofile(f)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB) rank_k={args.rank_k}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
