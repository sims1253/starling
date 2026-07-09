"""De-risk microbench: NVFP4 fused dequant-GEMV on ARK-ASR shapes.

Measures the fp4 fused GEMV vs cuBLAS bf16 F.linear at M=1 on the exact ARK
weight shapes, in BOTH hot-L2 (naive timing) and cold-L2 (flush the cache to
mimic the real decode loop where 5GB of weights evict L2 every token). Also
checks the bf16-vs-fp4 reconstruction error.

If fp4 wins clearly in cold-L2 (the real regime) AND the per-element error
is tolerable, the NVFP4 decode integration is worth doing.

  TRUST_REMOTE_CODE=1 uv run python benchmarks/bench_ark_fp4_gemv.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from starling.granite.fp4 import quantize_fp4_packed, _fp4_linear_fused

# ARK-ASR-3B Qwen2.5 decoder GEMM shapes (OUT, K).
# qkv is fused [q(2048);k(256);v(256)] -> 2560; gate+up fused -> 22016.
SHAPES = [
    ("qkv",     2560, 2048),
    ("gu",     22016, 2048),
    ("o_proj",  2048, 2048),
    ("down",    2048, 11008),
    ("lm_head", 151936, 2048),
]

# A 128 MB float32 flush buffer to evict L2 between iters (cold-L2 regime).
FLUSH_BYTES = 128 * 1024 * 1024  # 128 MB


def _cuda_ms(fn, warmup=3, iters=20, flush=None):
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        if flush is not None:
            flush.zero_()  # scribble 128MB -> evicts the GEMV weights from L2
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return float(np.median(times))


def _recon_error(w_bf16, codes, scales):
    """Max-abs and mean-abs error of fp4 reconstruction vs bf16 weight."""
    OUT, K = w_bf16.shape
    w32 = w_bf16.float()
    blocks = w32.view(OUT, K // 16, 16)
    # reconstruct: the quantizer divides block by amax(fp8) and scales to 6;
    # to invert we need the same fp8-rounded amax the kernel uses.
    from starling.granite.fp4 import _SCALE_DTYPE
    amax = blocks.amax(dim=-1).abs().clamp(min=1e-12).to(_SCALE_DTYPE).float()
    # codes (OUT, K//2) -> (OUT, K)
    lo = (codes & 0xF)
    hi = (codes >> 4) & 0xF
    codes_flat = torch.empty(OUT, K, dtype=torch.uint8, device=w_bf16.device)
    codes_flat[:, 0::2] = lo
    codes_flat[:, 1::2] = hi
    levels = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=w_bf16.device, dtype=torch.float32)
    idx = (codes_flat & 0x7).long()
    sign = (codes_flat >> 3) & 1
    mag = levels[idx]
    normed = torch.where(sign.bool(), -mag, mag)  # (OUT, K) in [-6,6]
    rec = normed.view(OUT, K // 16, 16) * amax.unsqueeze(-1) / 6.0
    rec = rec.view(OUT, K)
    diff = (rec - w32).abs()
    return float(diff.max()), float(diff.mean()), float(w32.abs().mean())


def main() -> int:
    torch.manual_seed(0)
    flush = torch.empty(FLUSH_BYTES // 4, dtype=torch.float32, device="cuda")
    print(f"{'shape':<10s} {'OUT':>7s} {'K':>6s} {'bf16 ms':>9s} {'fp4 ms':>8s} "
          f"{'spd hot':>8s} {'fp4 cold':>9s} {'bf16 cold':>10s} {'spd cold':>9s} "
          f"{'max-err':>8s} {'mean-err':>9s} {'rel':>6s}")
    print("-" * 115)

    for name, OUT, K in SHAPES:
        w = torch.randn(OUT, K, dtype=torch.bfloat16, device="cuda") * 0.1
        x = torch.randn(K, dtype=torch.bfloat16, device="cuda") * 0.1
        codes, scales = quantize_fp4_packed(w)

        def _bf16():
            return F.linear(x.unsqueeze(0), w).squeeze(0)

        def _fp4():
            return _fp4_linear_fused(x, (codes, scales))

        bf16_hot = _cuda_ms(_bf16, flush=None)
        fp4_hot = _cuda_ms(_fp4, flush=None)
        bf16_cold = _cuda_ms(_bf16, flush=flush)
        fp4_cold = _cuda_ms(_fp4, flush=flush)
        spd_hot = bf16_hot / fp4_hot if fp4_hot > 0 else 0
        spd_cold = bf16_cold / fp4_cold if fp4_cold > 0 else 0
        maxe, meane, wmean = _recon_error(w, codes, scales)
        rel = meane / wmean
        print(f"{name:<10s} {OUT:>7d} {K:>6d} {bf16_hot:>8.3f}m {fp4_hot:>7.3f}m "
              f"{spd_hot:>7.2f}x {fp4_cold:>8.3f}m {bf16_cold:>9.3f}m {spd_cold:>8.2f}x "
              f"{maxe:>7.4f} {meane:>8.5f} {rel:>5.1%}")
    print("\n'cold' = L2 flushed between iters (mimics the real decode loop where")
    print("5GB of weights evict L2 every token). 'hot' = repeated calls hit L2 cache.")
    print("spd = bf16_ms / fp4_ms (>1 means fp4 wins).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
