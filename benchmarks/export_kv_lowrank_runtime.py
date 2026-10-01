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
                    help="STARLING_GRANITE_DUMP_K prefixes (one per clip)")
    ap.add_argument("--rank-k", type=int, default=32)
    ap.add_argument("--layers", type=int, default=16)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    n_layers, n_heads, hidden = args.layers, args.heads, args.hidden
    head_dim = hidden // n_heads
    for p in args.dumps:
        n = len(glob.glob(f"{p}.L*.f32"))
        assert n == n_layers, f"{p}: expected {n_layers} layer dumps, got {n}"

    # Held-out check: the last dump prefix. Fit: all the others.
    fit_prefixes = args.dumps[:-1]
    held_prefix = args.dumps[-1]

    reader = gguf.GGUFReader(args.gguf)
    import struct
    f1k_l, f2k_l = [], []
    for li in range(n_layers):
        def load_K(prefixes):
            mats = []
            for p in prefixes:
                a = np.fromfile(f"{p}.L{li}.f32", dtype=np.float32)
                m = a.reshape(-1, hidden).T              # [hidden, T_pad]
                keep = np.any(m != 0.0, axis=0)          # padded cols are exact 0
                mats.append(m[:, keep])
            k = np.concatenate(mats, axis=1)             # [hidden, T_total]
            return k.reshape(n_heads, head_dim, -1).transpose(2, 0, 1)  # [T, H, D]

        K_fit, K_held = load_K(fit_prefixes), load_K([held_prefix])
        t = next(x for x in reader.tensors if x.name == f"enc.blk.{li}.attn_kv.weight")
        w = dequant_2d(t)
        wk = w[: w.shape[0] // 2]
        f1 = np.zeros((n_heads * args.rank_k, hidden), dtype=np.float32)
        f2 = np.zeros((hidden, n_heads * args.rank_k), dtype=np.float32)
        err_fit = err_held = en_fit = en_held = 0.0
        for h in range(n_heads):
            Xf = K_fit[:, h, :].astype(np.float64)
            _, _, Vt = np.linalg.svd(Xf, full_matrices=False)
            B = Vt[: args.rank_k].T
            Xh = K_held[:, h, :].astype(np.float64)
            err_held += float((((Xh @ B) @ B.T - Xh) ** 2).sum())
            en_held += float((Xh ** 2).sum())
            err_fit += float((((Xf @ B) @ B.T - Xf) ** 2).sum())
            en_fit += float((Xf ** 2).sum())
            W_h = wk[h * head_dim:(h + 1) * head_dim].astype(np.float64)
            M = W_h.T @ B
            f1[h * args.rank_k:(h + 1) * args.rank_k, :] = M.T.astype(np.float32)
            f2[h * head_dim:(h + 1) * head_dim,
               h * args.rank_k:(h + 1) * args.rank_k] = B.astype(np.float32)
        f1k_l.append(f1)
        f2k_l.append(f2)
        print(f"  layer {li:2d}: runtime-K proj rel-MSE fit={err_fit / en_fit:.4f} "
              f"held-out={err_held / en_held:.4f}", flush=True)

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
