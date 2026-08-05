"""Benchmark a fused NVFP4 dequant-GEMV kernel against cuBLAS bf16 GEMV.

The decode-step profile (bench_decode_profile.py) shows GEMVs are 51% of step
time and weight-bandwidth-bound. NVFP4 packing (nibble-packed e2m1 codes + fp8
block scales) cuts weight bytes by 3.56x (0.5625 bytes/elem vs 2.0 for bf16).
The question: can a fused Triton dequant-GEMV realize that bandwidth win on
the RTX 5090 (sm120)?

Compares on granite decode shapes (batch=1, M=1 GEMV):
  1. F.linear(x, w_bf16)   -- cuBLAS GEMV baseline (reads 2*K*OUT weight bytes)
  2. fp4 fused GEMV         -- Triton kernel (reads 0.56*K*OUT weight bytes)

The kernel streams packed fp4 codes + fp8 block scales, dequantizes in fp32
registers (no bf16 materialisation), and accumulates the dot-product in fp32.
This is the same pattern as Jetha Chan's A4Q QK-kernel applied to the GEMV
instead of attention -- and it's the TODO #1 from fp4.py.

Usage:
  uv run python benchmarks/bench_fp4_gemv.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from starling.granite.fp4 import BLOCK_SIZE  # noqa: E402

# =========================================================================== #
# NVFP4 packing: nibble-packed e2m1 codes + fp8e4m3 block scales.
#
# Storage for a weight (OUT, K) where K % 16 == 0:
#   codes:  (OUT, K // 2)  uint8  -- two e2m1 codes per byte (low=even, high=odd)
#   scales: (OUT, K // 16) float8_e4m3fn  -- one scale per 16-element block
#
# Total bytes = OUT * K / 2 + OUT * K / 16 = OUT * K * 0.5625
# vs bf16:     = OUT * K * 2.0
# compression = 2.0 / 0.5625 = 3.56x
# =========================================================================== #


def quantize_fp4_packed(w: torch.Tensor):
    """Quantize (OUT, K) bf16 weight to nibble-packed NVFP4.

    Returns (codes, scales):
      codes:  (OUT, K // 2) uint8, two e2m1 nibbles per byte
      scales: (OUT, K // 16) float8_e4m3fn, one per 16-elem block

    The e2m1 quantization matches starling.granite.fp4.quantize_fp4: each
    16-element block is scaled by its amax (quantized to fp8), then each
    element is mapped to the nearest e2m1 level in [-6, 6]. Reconstruction:
    w ~= scale * level / 6.0  (the /6.0 maps e2m1 max 6.0 back to the block amax).
    """
    OUT, K = w.shape
    assert K % BLOCK_SIZE == 0, f"K={K} must be multiple of {BLOCK_SIZE}"
    w32 = w.float()
    blocks = w32.view(OUT, K // BLOCK_SIZE, BLOCK_SIZE)
    amax = blocks.amax(dim=-1).abs().clamp(min=1e-12)  # (OUT, K//16)
    scales_fp8 = amax.to(torch.float8_e4m3fn)  # (OUT, K//16)
    scales_f32 = scales_fp8.float()  # (OUT, K//16)
    normed = (blocks / scales_f32.unsqueeze(-1) * 6.0).clamp(-6.0, 6.0)

    # Quantize to e2m1 codes (0..15). Bit 3 = sign, bits[2:0] = mag level idx.
    # Levels: (0, 0.5, 1, 1.5, 2, 3, 4, 6) -- matches standard e2m1 magnitude.
    levels = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
                          device=w.device, dtype=torch.float32)
    sign = (normed < 0).to(torch.uint8) << 3  # (OUT, K//16, 16)
    mag = normed.abs()
    idx = (mag.unsqueeze(-1) - levels).abs().argmin(dim=-1).to(torch.uint8)
    codes_flat = (idx | sign).view(OUT, K)  # (OUT, K) uint8, one code per byte

    # Nibble-pack: two codes per byte. Even elements -> low nibble, odd -> high.
    lo = codes_flat[:, 0::2]  # (OUT, K//2)
    hi = codes_flat[:, 1::2]
    packed = (lo & 0xF) | ((hi & 0xF) << 4)  # (OUT, K//2) uint8
    return packed.contiguous(), scales_fp8.contiguous()


# =========================================================================== #
# Fused NVFP4 dequant-GEMV Triton kernel.
#
# Computes out[i] = sum_k dequant(codes[i,k], scales[i,k//16]) * x[k]
# for a single input vector x (M=1 decode step). Mirrors the structure of
# llm_kernels._gemv_normscale_kernel but dequantizes fp4 weights on the fly.
#
# Dequant: each e2m1 code (4 bits) decodes to a signed level in {-6,-4,-3,
# -2,-1.5,-1,-0.5,0,0.5,1,1.5,2,3,4,6}, then w = scale_fp8 * level / 6.0.
# =========================================================================== #


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": bm}, num_warps=w, num_stages=s)
        for bm in (16, 32, 64, 128)
        for w in (4, 8, 16)
        for s in (1, 2, 3, 4)
    ],
    key=["OUT_N", "K"],
)
@triton.jit
def _fp4_gemv_kernel(
    X_ptr,          # (K,) bf16 input vector
    CODES_ptr,      # (OUT, K // 2) uint8 nibble-packed codes
    SCALES_ptr,     # (OUT, K // 16) fp8e4m3 block scales
    OUT_ptr,        # (OUT,) bf16 output
    K: tl.constexpr,
    K_BYTES: tl.constexpr,
    K_BLOCKS: tl.constexpr,
    OUT_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,   # must be multiple of 32 (2 bytes, 2 scale-blocks)
):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_M
    rows = row_start + tl.arange(0, BLOCK_M)
    rmask = rows < OUT_N

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # Iterate over K in tiles of BLOCK_K elements.
    # BLOCK_K elements = BLOCK_K/2 bytes of codes, BLOCK_K/16 scale blocks.
    BLOCK_BYTES: tl.constexpr = BLOCK_K // 2

    for kb in range(0, K_BYTES, BLOCK_BYTES):
        byte_cols = kb + tl.arange(0, BLOCK_BYTES)
        bmask = byte_cols < K_BYTES

        # Load code bytes: (BLOCK_M, BLOCK_BYTES)
        byte_off = rows[:, None] * K_BYTES + byte_cols[None, :]
        bm = bmask[None, :] & rmask[:, None]
        raw = tl.load(CODES_ptr + byte_off, mask=bm, other=0)

        # Extract nibbles -> 2 codes per byte
        code_lo = raw & 0xF           # even elements (low nibble)
        code_hi = (raw >> 4) & 0xF    # odd elements (high nibble)

        # --- e2m1 dequant (inline for both lo and hi) ---
        # Bit 3 = sign, bits[2:0] = magnitude index into 8 levels.
        sign_lo = (code_lo >> 3) & 1
        mag_lo = code_lo & 0x7
        # mag level: idx 0..7 -> (0, 0.5, 1, 1.5, 2, 3, 4, 6)
        level_lo = tl.where(mag_lo == 0, 0.0,
                  tl.where(mag_lo == 1, 0.5,
                  tl.where(mag_lo == 2, 1.0,
                  tl.where(mag_lo == 3, 1.5,
                  tl.where(mag_lo == 4, 2.0,
                  tl.where(mag_lo == 5, 3.0,
                  tl.where(mag_lo == 6, 4.0, 6.0))))))).to(tl.float32)
        level_lo = tl.where(sign_lo == 1, -level_lo, level_lo)

        sign_hi = (code_hi >> 3) & 1
        mag_hi = code_hi & 0x7
        level_hi = tl.where(mag_hi == 0, 0.0,
                  tl.where(mag_hi == 1, 0.5,
                  tl.where(mag_hi == 2, 1.0,
                  tl.where(mag_hi == 3, 1.5,
                  tl.where(mag_hi == 4, 2.0,
                  tl.where(mag_hi == 5, 3.0,
                  tl.where(mag_hi == 6, 4.0, 6.0))))))).to(tl.float32)
        level_hi = tl.where(sign_hi == 1, -level_hi, level_hi)

        # Load block scales: element 2*byte_cols[j] -> block (2*byte_cols[j])//16
        elem_lo = 2 * byte_cols       # (BLOCK_BYTES,)
        elem_hi = 2 * byte_cols + 1
        blk_lo = elem_lo // 16        # scale-block index
        blk_hi = elem_hi // 16
        # Scales: (OUT, K_BLOCKS) fp8. Load as (BLOCK_M, BLOCK_BYTES) fp8 -> fp32.
        scale_off_lo = rows[:, None] * K_BLOCKS + blk_lo[None, :]
        scale_off_hi = rows[:, None] * K_BLOCKS + blk_hi[None, :]
        # tl.load from fp8 tensor: Triton handles the dtype from the pointer.
        scale_lo = tl.load(SCALES_ptr + scale_off_lo, mask=bm, other=0.0).to(tl.float32)
        scale_hi = tl.load(SCALES_ptr + scale_off_hi, mask=bm, other=0.0).to(tl.float32)

        w_lo = level_lo * scale_lo * (1.0 / 6.0)   # dequantized weight (even k)
        w_hi = level_hi * scale_hi * (1.0 / 6.0)   # dequantized weight (odd k)

        # Load input vector slices
        x_lo = tl.load(X_ptr + elem_lo, mask=bmask, other=0.0).to(tl.float32)
        x_hi = tl.load(X_ptr + elem_hi, mask=bmask, other=0.0).to(tl.float32)

        # Accumulate per output row
        acc += tl.sum(w_lo * x_lo[None, :], axis=1)
        acc += tl.sum(w_hi * x_hi[None, :], axis=1)

    tl.store(OUT_ptr + rows, acc.to(OUT_ptr.dtype.element_ty), mask=rmask)


def fp4_gemv(x: torch.Tensor, codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Fused NVFP4 dequant-GEMV: out = dequant(codes, scales) @ x.

    Args:
        x: (K,) bf16 input vector (M=1 decode step).
        codes: (OUT, K//2) uint8 nibble-packed e2m1 codes.
        scales: (OUT, K//16) float8_e4m3fn block scales.

    Returns:
        (OUT,) bf16 output.
    """
    OUT, K半 = codes.shape
    K = K半 * 2
    assert K % 16 == 0
    K_BLOCKS = K // 16
    x1 = x.reshape(-1).contiguous()
    out = torch.empty((OUT,), dtype=x.dtype, device=codes.device)
    BLOCK_K = 128  # 64 bytes, 8 scale blocks per tile
    grid = lambda meta: (triton.cdiv(OUT, meta["BLOCK_M"]),)
    _fp4_gemv_kernel[grid](
        x1, codes, scales, out,
        K=K, K_BYTES=K // 2, K_BLOCKS=K_BLOCKS, OUT_N=OUT,
        BLOCK_K=BLOCK_K,
    )
    return out


# =========================================================================== #
# Benchmark
# =========================================================================== #

SHAPES = [
    ("qkv",     3072, 2048),   # fused q+k+v proj
    ("o_proj",  2048, 2048),
    ("gu",      4096, 2048),   # gate+up
    ("down",    2048, 4096),
    ("lm_head", 100353, 2048),
]


def _bench(fn, reps=200, warmup=20, flush_l2=False, flush_buf=None):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps):
        if flush_l2 and flush_buf is not None:
            # Read ~256MB to evict the weight from L2 before each GEMV,
            # simulating the real decode loop where 2.6GB of weights stream
            # through HBM and nothing stays resident.
            flush_buf.sum()
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / reps * 1000.0  # us


@torch.inference_mode()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    args = ap.parse_args()

    print(f"GPU: {torch.cuda.get_device_name()} (sm{torch.cuda.get_device_capability()})")
    print(f"\nFused NVFP4 dequant-GEMV vs cuBLAS bf16 GEMV (M=1, median of {args.reps} reps)")
    print("\n[HOT L2] -- weight resident in cache (optimistic; not the real decode regime)\n")
    print(f'{"name":10s} {"OUT":>7s} {"K":>6s}  {"bf16 us":>8s} {"fp4 us":>8s}  {"speedup":>8s}  {"max_err":>8s}  {"bw_bf16":>8s} {"bw_fp4":>8s}')
    print("-" * 95)

    # L2-flush buffer: ~256MB read each iter evicts the weight from the 96MB L2.
    flush_buf = torch.empty(32 * 1024 * 1024, dtype=torch.float32, device="cuda")  # 128MB

    for name, OUT, K in SHAPES:
        w = torch.randn(OUT, K, dtype=torch.bfloat16, device="cuda") * 0.05
        x = torch.randn(K, dtype=torch.bfloat16, device="cuda")

        # bf16 baseline (hot L2)
        ref = torch.nn.functional.linear(x, w)  # (OUT,)
        us_bf16 = _bench(lambda: torch.nn.functional.linear(x, w), reps=args.reps)

        # fp4 fused (hot L2)
        codes, scales = quantize_fp4_packed(w)
        out_fp4 = fp4_gemv(x, codes, scales)
        us_fp4 = _bench(lambda: fp4_gemv(x, codes, scales), reps=args.reps)
        max_err = (out_fp4.float() - ref.float()).abs().max().item()
        speedup = us_bf16 / us_fp4

        bf16_bytes = 2 * OUT * K
        fp4_bytes = OUT * K // 2 + OUT * K // 16
        bw_bf16 = bf16_bytes / (us_bf16 * 1e-6) / 1e9  # GB/s
        bw_fp4 = fp4_bytes / (us_fp4 * 1e-6) / 1e9

        print(f"{name:10s} {OUT:7d} {K:6d}  {us_bf16:8.2f} {us_fp4:8.2f}  {speedup:7.2f}x  {max_err:8.4f}  {bw_bf16:7.0f}G {bw_fp4:7.0f}G")

    # ---- cold-cache regime: flush L2 before each GEMV ----------------------
    print("\n[COLD L2] -- weight evicted each step (real decode: 2.6GB weights, 96MB L2)\n")
    print(f'{"name":10s} {"OUT":>7s} {"K":>6s}  {"bf16 us":>8s} {"fp4 us":>8s}  {"speedup":>8s}  {"bw_bf16":>8s} {"bw_fp4":>8s}  {"bf16 %peak":>10s} {"fp4 %peak":>10s}')
    print("-" * 105)

    # RTX 5090: ~1792 GB/s HBM peak. Report % peak to show how bound each path is.
    HBM_PEAK_GBS = 1792.0
    for name, OUT, K in SHAPES:
        w = torch.randn(OUT, K, dtype=torch.bfloat16, device="cuda") * 0.05
        x = torch.randn(K, dtype=torch.bfloat16, device="cuda")
        codes, scales = quantize_fp4_packed(w)

        us_bf16 = _bench(lambda: torch.nn.functional.linear(x, w), reps=args.reps // 2,
                         flush_l2=True, flush_buf=flush_buf)
        us_fp4 = _bench(lambda: fp4_gemv(x, codes, scales), reps=args.reps // 2,
                        flush_l2=True, flush_buf=flush_buf)

        speedup = us_bf16 / us_fp4
        bf16_bytes = 2 * OUT * K
        fp4_bytes = OUT * K // 2 + OUT * K // 16
        bw_bf16 = bf16_bytes / (us_bf16 * 1e-6) / 1e9
        bw_fp4 = fp4_bytes / (us_fp4 * 1e-6) / 1e9

        print(f"{name:10s} {OUT:7d} {K:6d}  {us_bf16:8.2f} {us_fp4:8.2f}  {speedup:7.2f}x  "
              f"{bw_bf16:7.0f}G {bw_fp4:7.0f}G  {100*bw_bf16/HBM_PEAK_GBS:9.0f}% {100*bw_fp4/HBM_PEAK_GBS:9.0f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
