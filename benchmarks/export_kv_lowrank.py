"""Export low-rank K/V projection factors for the granite encoder (#59).

Takes the encoder's fused ``attn_kv`` weight (K||V halves) from a GGUF,
dequantizes it with gguf-py's K-quant block math (the same formulas the
runtime's dequant uses), and factors each head's projection by SVD:

    W_h [128, hidden]  ~=  B_h [128, r] @ C_h [r, hidden]

so the runtime computes ``k_h = (n @ C_h^T) @ B_h^T`` as two GEMMs. This is a
WEIGHT-space factorization (no audio involved), mirroring the draft-head
study in bench_ctc_draft_rank.py. The output is a fixed binary artifact the
engine consumes via STARLING_GRANITE_KVFACT=<file> (cpp/granite/kv_factors.hpp
documents the format); factors are float32 and the file is the pinned
artifact any run must name (do not regenerate silently between measurements).

File format (little-endian):
    char magic[8] = "STLGKVF1"
    u32 version = 1, u32 n_layers, u32 hidden, u32 n_heads
    u32 rank_k, u32 rank_v          (0 = half left unchanged; <= head_dim)
    layer-major payload — per layer: f32 f1k[n_heads*rank_k*hidden],
    f32 f2k[hidden*n_heads*rank_k], then f32 f1v[...], f32 f2v[...]
    (each half present only when its rank > 0)

Usage:
    uv run python benchmarks/export_kv_lowrank.py \
        --gguf models/granite-2b-dynq4.imx.gguf --rank-k 32 --rank-v 0 \
        --out benchmarks/results/kvfactors_granite_k32.bin
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import gguf
import gguf.quants
import numpy as np

MAGIC = b"STLGKVF1"
VERSION = 1


def dequant_2d(t) -> np.ndarray:
    """Dequantize one GGUF 2-D tensor (ne = [ne0, ne1]) to float32 [ne1, ne0]
    (PyTorch Linear [out, in] layout)."""
    ne0, ne1 = int(t.shape[0]), int(t.shape[1])
    qtype = gguf.GGMLQuantizationType(t.tensor_type)
    if qtype == gguf.GGMLQuantizationType.F32:
        return np.array(t.data, dtype=np.float32).reshape(ne1, ne0)
    if qtype == gguf.GGMLQuantizationType.BF16:
        raw = np.ascontiguousarray(t.data, dtype=np.uint8).reshape(-1).view(np.uint16)
        return (raw.astype(np.uint32) << 16).view(np.float32).reshape(ne1, ne0)
    cls = getattr(gguf.quants, qtype.name, None)
    if cls is None:
        raise ValueError(f"{t.name}: unsupported quant type {qtype}")
    data = np.array(t.data, dtype=np.uint8).reshape(ne1, -1)
    out = np.empty((ne1, ne0), dtype=np.float32)
    for i in range(ne1):
        out[i] = cls.dequantize_rows(data[i]).astype(np.float32)[:ne0]
    return out


def check_rank(rank: int, head_dim: int, what: str = "rank") -> None:
    """The engine rejects per-head ranks outside [0, head_dim]; fail here first."""
    if not 0 <= rank <= head_dim:
        raise ValueError(f"{what} must be in [0, {head_dim}] (head_dim); got {rank}")


def check_dump_prefixes(prefixes: list[str], n_layers: int) -> None:
    """STARLING_GRANITE_DUMP_K prefixes: >= 2 (fit set + held-out), each with
    one K dump per layer."""
    if len(prefixes) < 2:
        raise ValueError("need at least two --dumps prefixes (fit set + held-out)")
    for p in prefixes:
        n = len(glob.glob(f"{p}.L*.f32"))
        if n != n_layers:
            raise ValueError(f"{p}: expected {n_layers} layer dumps, got {n}")


def load_runtime_k(prefixes: list[str], layer: int, hidden: int, n_heads: int) -> np.ndarray:
    """Concatenate one layer's runtime K dumps (row-major [hidden, T_pad] each)
    into [T, n_heads, head_dim], dropping the block padding. The engine pads
    with exact-zero TRAILING columns; an all-zero column before the last real
    frame would be a contract break, not padding, so it raises."""
    mats = []
    for p in prefixes:
        m = np.fromfile(f"{p}.L{layer}.f32", dtype=np.float32).reshape(-1, hidden).T
        keep = np.any(m != 0.0, axis=0)
        n_valid = int(np.flatnonzero(keep)[-1]) + 1 if keep.any() else 0
        if not keep[:n_valid].all():
            raise ValueError(f"{p}.L{layer}.f32: all-zero K column inside the valid "
                             "frames (padding must be trailing)")
        mats.append(m[:, :n_valid])
    k = np.concatenate(mats, axis=1)                          # [hidden, T_total]
    return k.reshape(n_heads, hidden // n_heads, -1).transpose(2, 0, 1)


def fit_pca_fold(k_fit: np.ndarray, k_held: np.ndarray, w_half: np.ndarray,
                 n_heads: int, rank: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Activation-space factors shared by the activation/runtime/selective
    exporters. Per head: B_h = top-``rank`` right singular vectors of the fit
    activations k_fit[:, h, :] ([T, head_dim]), folded into the weight as
    z = n @ (W_h^T B_h), k_h ~= z @ B_h^T. Returns (f1, f2, errs) in the
    factor-file layouts; errs holds uncentered projection residual/energy
    sums (``err_fit``/``en_fit``/``err_held``/``en_held``)."""
    head_dim = w_half.shape[0] // n_heads
    hidden = w_half.shape[1]
    f1 = np.zeros((n_heads * rank, hidden), dtype=np.float32)
    f2 = np.zeros((hidden, n_heads * rank), dtype=np.float32)
    errs = {"err_fit": 0.0, "en_fit": 0.0, "err_held": 0.0, "en_held": 0.0}
    for h in range(n_heads):
        X = k_fit[:, h, :].astype(np.float64)
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        B = Vt[:rank].T                                       # [head_dim, r]
        Xh = k_held[:, h, :].astype(np.float64)
        errs["err_held"] += float((((Xh @ B) @ B.T - Xh) ** 2).sum())
        errs["en_held"] += float((Xh ** 2).sum())
        errs["err_fit"] += float((((X @ B) @ B.T - X) ** 2).sum())
        errs["en_fit"] += float((X ** 2).sum())
        W_h = w_half[h * head_dim:(h + 1) * head_dim].astype(np.float64)
        M = W_h.T @ B                                         # [in=hidden, r]
        f1[h * rank:(h + 1) * rank, :] = M.T.astype(np.float32)
        f2[h * head_dim:(h + 1) * head_dim, h * rank:(h + 1) * rank] = B.astype(np.float32)
    return f1, f2, errs


def factor_half(w_half: np.ndarray, n_heads: int, rank: int) -> tuple[np.ndarray, np.ndarray]:
    """Factor a [n_heads*hd, hidden] half per head by SVD. Returns (f1, f2):
    f1 rows = per-head C_h [rank, hidden] (stacked [n_heads*rank, hidden]);
    f2 = block-diagonal [hidden, n_heads*rank] of per-head B_h^T blocks."""
    hd = w_half.shape[0] // n_heads
    hidden = w_half.shape[1]
    check_rank(rank, hd)
    f1 = np.zeros((n_heads * rank, hidden), dtype=np.float32)
    f2 = np.zeros((hidden, n_heads * rank), dtype=np.float32)
    for h in range(n_heads):
        w = w_half[h * hd:(h + 1) * hd].astype(np.float64)
        u, s, vt = np.linalg.svd(w, full_matrices=False)
        f1[h * rank:(h + 1) * rank, :] = (s[:rank, None] * vt[:rank, :]).astype(np.float32)
        f2[h * hd:(h + 1) * hd, h * rank:(h + 1) * rank] = u[:, :rank].astype(np.float32)
    return f1, f2


def reconstruct(f1: np.ndarray, f2: np.ndarray, n_heads: int, rank: int,
                hd: int) -> np.ndarray:
    rec = np.zeros((n_heads * hd, f1.shape[1]), dtype=np.float32)
    for h in range(n_heads):
        rec[h * hd:(h + 1) * hd] = (
            f2[h * hd:(h + 1) * hd, h * rank:(h + 1) * rank]
            @ f1[h * rank:(h + 1) * rank, :])
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--rank-k", type=int, default=32)
    ap.add_argument("--rank-v", type=int, default=0)
    ap.add_argument("--layers", type=int, default=16)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    reader = gguf.GGUFReader(args.gguf)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_layers, n_heads = args.layers, args.heads
    hidden = None
    f1k_l, f2k_l, f1v_l, f2v_l = [], [], [], []
    rel = {"k": [], "v": []}
    for li in range(n_layers):
        name = f"enc.blk.{li}.attn_kv.weight"
        t = next((x for x in reader.tensors if x.name == name), None)
        if t is None:
            raise KeyError(f"tensor {name} not found in GGUF")
        w = dequant_2d(t)                                   # [2*hidden, hidden]
        if hidden is None:
            hidden = w.shape[1]
            if w.shape[0] != 2 * n_heads * (hidden // n_heads):
                raise ValueError(f"{name}: unexpected shape {w.shape}")
            check_rank(args.rank_k, hidden // n_heads, "--rank-k")
            check_rank(args.rank_v, hidden // n_heads, "--rank-v")
        hd = hidden // n_heads
        wk, wv = w[: w.shape[0] // 2], w[w.shape[0] // 2:]
        if args.rank_k > 0:
            f1, f2 = factor_half(wk, n_heads, args.rank_k)
            rel["k"].append(float(np.linalg.norm(
                reconstruct(f1, f2, n_heads, args.rank_k, hd) - wk)
                / np.linalg.norm(wk)))
            f1k_l.append(f1)
            f2k_l.append(f2)
        if args.rank_v > 0:
            f1, f2 = factor_half(wv, n_heads, args.rank_v)
            rel["v"].append(float(np.linalg.norm(
                reconstruct(f1, f2, n_heads, args.rank_v, hd) - wv)
                / np.linalg.norm(wv)))
            f1v_l.append(f1)
            f2v_l.append(f2)

    import struct
    with open(out_path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<6I", VERSION, n_layers, hidden, n_heads,
                            args.rank_k, args.rank_v))
        for i in range(n_layers):
            if args.rank_k > 0:
                f1k_l[i].tofile(f)
                f2k_l[i].tofile(f)
            if args.rank_v > 0:
                f1v_l[i].tofile(f)
                f2v_l[i].tofile(f)

    for half in ("k", "v"):
        if rel[half]:
            print(f"{half.upper()} weight-space rel-Frobenius error: "
                  f"mean={np.mean(rel[half]):.4f} max={np.max(rel[half]):.4f}")
    print(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
