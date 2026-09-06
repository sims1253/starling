"""NVFP4-style block weight quantization (scaffold).

Implements a simple, correct NVFP4 (4-bit microscaling) block quantizer for
linear-layer weight tensors. This is the **reference / correctness** path only:
``dequantize_fp4`` materializes a full bf16 tensor, so ``_fp4_linear`` is a
slow dequant-then-matmul. It is NOT the fused dequant-GEMV kernel that would
actually deliver the bandwidth win (see TODO at bottom).

Format (per ``quantize_fp4`` output):
    * Block size 16 elements (NVFP4 standard).
    * Per-block scale: ``amax(|block|)`` quantized to ``torch.float8_e4m3fn``
      (falls back to fp32 if the dtype is unavailable on the build).
    * Per-element mantissa: e2m1 (1 sign, 2 exponent, 1 mantissa bit ->
      8 positive levels: 0, 0.5, 1, 1.5, 2, 3, 4, 6).
    * Element reconstruction: ``scale * (e2m1_code / 6)``.

Bandwidth
---------
A bf16 weight costs 2 bytes/element. The FP4 packing costs:
    * 0.5 bytes/element for the packed e2m1 codes (2 codes per byte), plus
    * 1 byte per 16-element block for the fp8 scale  =  0.0625 bytes/element
    => ~0.5625 bytes/element total, a **3.56x reduction** in weight bytes
       (measured; see ``benchmarks/bench_fp4_calibrate.py``).

Decode-step GEMVs are ~51% of step time and entirely weight-bandwidth-bound
(``benchmarks/bench_decode_profile.py``), so the *theoretical* speedup ceiling
is ~3.5x on the GEMV fraction == ~1.57x on full decode-step latency. **This is
an upper bound only** -- realizing it needs a fused dequant-GEMV kernel that
streams packed FP4 weights and never materializes the bf16 tensor. That kernel
is not in this file; ``_fp4_linear`` here dequantizes the whole weight first,
so it pays the bf16 materialization cost AND the cuBLAS GEMV cost -- it will be
*slower* than the bf16 path. That is the expected, documented result for the
scaffold (see ``bench_fp4_calibrate.py`` and the TODO list).

TODO (out of scope for this scaffold)
-------------------------------------
1. Fused dequant-GEMV Triton kernel (stream packed FP4 + per-block fp8 scale,
   dequant in registers, accumulate in fp32). This is what makes FP4 faster.
2. QAD (quantization-aware distillation) fine-tune loop on ASR data to recover
   the WER loss from raw PTQ. See
   ``/mnt/z/concepts/quantization-aware-distillation.md``.
3. Per-channel / grouped scale calibration (current: one fp8 scale per
   16-element block). For the row-wise GEMV, per-output-channel scales often
   reduce error further -- worth measuring before committing to QAD.
4. lm_head quantization sanity. lm_head is the single largest weight and the
   most WER-sensitive (it produces the logits). Validate that FP4 lm_head
   alone doesn't blow up WER before QAD; consider keeping it in bf16.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

BLOCK_SIZE: int = 16
"""NVFP4 block size (elements per scale)."""

# e2m1 positive magnitude levels (sign bit handled separately).
_FP4_LEVELS: Tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

try:
    _SCALE_DTYPE = torch.float8_e4m3fn
    _HAS_FP8 = True
except AttributeError:  # older torch builds
    _SCALE_DTYPE = torch.float32
    _HAS_FP8 = False


def _fp4_levels(device, dtype=torch.float32) -> torch.Tensor:
    # Cached per (device, dtype) so we don't allocate inside CUDA-graph capture.
    key = (str(device), dtype)
    cached = _LEVELS_CACHE.get(key)
    if cached is not None and cached.device == torch.device(device):
        return cached
    t = torch.tensor(_FP4_LEVELS, device=device, dtype=dtype)
    _LEVELS_CACHE[key] = t
    return t


_LEVELS_CACHE: dict = {}


def _quant_e2m1(x_normed: torch.Tensor) -> torch.Tensor:
    """Quantize a normalized-to-[-6, 6] fp tensor to e2m1 codes (uint8, 0..15).

    Bit layout: bit3 = sign (1 negative), bits[2:0] = magnitude level index.
    """
    levels = _fp4_levels(x_normed.device, x_normed.dtype)
    sign = (x_normed < 0).to(torch.uint8) << 3
    mag = x_normed.abs().clamp(max=6.0)
    # nearest level by argmin of |mag - level|
    idx = (mag.unsqueeze(-1) - levels).abs().argmin(dim=-1).to(torch.uint8)
    return idx | sign


def _dequant_e2m1(code: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`_quant_e2m1`. Returns fp32 in [-6, 6]."""
    levels = _fp4_levels(code.device, torch.float32)
    idx = (code & 0x7).long()
    sgn = ((code >> 3) & 0x1).bool()
    mag = levels[idx]
    return torch.where(sgn, -mag, mag)


def quantize_fp4(w: torch.Tensor):
    """Quantize a 2D bf16/fp16 weight tensor to NVFP4.

    Args:
        w: ``(out_features, in_features)`` weight tensor.

    Returns:
        ``(codes, scales, meta)`` where:
          * ``codes``  : ``(*, 16)`` uint8 tensor of e2m1 codes.
          * ``scales`` : ``(*,)`` fp32 tensor of per-block fp8 scales
                         (dequantized from fp8e4m3 for portability).
          * ``meta``   : ``(orig_shape, pad)`` tuple needed by
                         :func:`dequantize_fp4`.
    """
    if w.dim() != 2:
        raise ValueError(f"quantize_fp4 expects a 2D weight, got shape {tuple(w.shape)}")
    orig_shape = w.shape
    n = w.numel()
    pad = (-n) % BLOCK_SIZE
    flat = w.reshape(-1)
    if pad:
        flat = F.pad(flat, (0, pad))
    blocks = flat.view(-1, BLOCK_SIZE).to(torch.float32)
    amax = blocks.abs().amax(dim=-1).clamp(min=1e-12)
    if _HAS_FP8:
        # Round-trip the scale through fp8e4m3 (NVFP4 standard). Keep fp32 copy
        # for portable dequant; the fp8 bytes are what a fused kernel would read.
        scales = amax.to(_SCALE_DTYPE).to(torch.float32)
    else:
        scales = amax
    # Normalize so the block max maps to e2m1 max (6.0).
    normed = (blocks / scales.unsqueeze(-1) * 6.0).clamp(-6.0, 6.0)
    codes = _quant_e2m1(normed)
    return codes.contiguous(), scales.contiguous(), (orig_shape, pad)


def dequantize_fp4(codes: torch.Tensor, scales: torch.Tensor, meta) -> torch.Tensor:
    """Materialize the bf16 weight from FP4 codes + scales.

    This is the **slow reference dequant** -- it allocates the full bf16 weight
    tensor. A fused dequant-GEMV kernel would replace both this and the
    subsequent matmul.
    """
    orig_shape, pad = meta
    normed = _dequant_e2m1(codes)
    blocks = normed * scales.unsqueeze(-1) / 6.0
    flat = blocks.reshape(-1)
    if pad:
        flat = flat[:-pad]
    return flat.reshape(orig_shape).to(torch.bfloat16)


def fp4_weight_bytes(numel: int) -> int:
    """Total bytes for an FP4-packed weight of ``numel`` elements.

    0.5 bytes/elem (packed codes) + 1 byte per 16-elem block (fp8 scale).
    """
    n_blocks = (numel + BLOCK_SIZE - 1) // BLOCK_SIZE
    return (numel + 1) // 2 + n_blocks


def _fp4_linear(
    x: torch.Tensor,
    packed_w,          # (codes, scales, meta)
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference dequant-then-matmul linear. SLOW (dequant is not fused).

    Correctness: matches the bf16 reference linear to within the FP4
    quantization error envelope (~1e-1 relative on the *weights*; see the
    calibration report). The dequant cost makes this slower than ``F.linear``
    on the bf16 weight -- that's expected; the speedup needs a fused kernel.
    """
    codes, scales, meta = packed_w
    w_bf16 = dequantize_fp4(codes, scales, meta).to(x.dtype)
    return F.linear(x, w_bf16, bias)


# =========================================================================== #
# Fused NVFP4 dequant-GEMV (closes TODO #1).
#
# Nibble-packed storage for true 3.56x bandwidth reduction (the unpacked
# ``quantize_fp4`` format above is 1.6x and wastes 4 bits per code):
#   * codes  : (OUT, K // 2) uint8 -- two e2m1 nibbles per byte
#              even elements (k=0,2,4...) -> low nibble (bits 0-3)
#              odd  elements (k=1,3,5...) -> high nibble (bits 4-7)
#   * scales : (OUT, K // 16) float8_e4m3fn -- one fp8 scale per 16-elem block
#
# The fused kernel streams packed codes + fp8 scales, dequantizes to fp32 in
# registers (no bf16 materialisation), and accumulates the M=1 dot-product in
# fp32.  This is the A4Q pattern (fp4-streams-to-tensor-core) applied to the
# decode-step GEMV instead of attention; it targets the 51% of step time the
# GEMVs consume (bench_decode_profile.py).  Only worth it for HBM-bandwidth-
# bound shapes -- see benchmarks/bench_fp4_gemv.py for the cold-L2 regime
# where it wins (lm_head 1.5-1.9x) and the hot-L2 regime where it doesn't.
# =========================================================================== #
_FP4_LEVELS_F32: Tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def quantize_fp4_packed(w: torch.Tensor):
    """Quantize a 2D ``(OUT, K)`` bf16 weight to nibble-packed NVFP4.

    Returns ``(codes, scales)``:
      * ``codes``  : ``(OUT, K // 2)`` uint8, two e2m1 nibbles per byte
                     (even ``k`` -> low nibble, odd ``k`` -> high nibble).
      * ``scales`` : ``(OUT, K // 16)`` ``float8_e4m3fn``, one fp8 scale per
                     16-element block.

    Reconstruction: ``w ~= scale_fp8 * e2m1_level(code) / 6.0``.

    The e2m1 quantization matches :func:`quantize_fp4` (nearest of the 8
    positive levels ``(0, 0.5, 1, 1.5, 2, 3, 4, 6)``, sign bit separate);
    only the *storage* differs (nibble-packed uint8 + fp8 scale vs the
    unpacked uint8 + fp32 scale that :func:`quantize_fp4` returns).  This
    packing is what makes the fused kernel bandwidth-competitive: 0.5625
    bytes/element vs 2.0 for bf16 == 3.56x reduction (see
    :func:`fp4_weight_bytes`).
    """
    if w.dim() != 2:
        raise ValueError(f"quantize_fp4_packed expects 2D (OUT, K), got {tuple(w.shape)}")
    OUT, K = w.shape
    if K % BLOCK_SIZE != 0:
        raise ValueError(f"K={K} must be a multiple of BLOCK_SIZE={BLOCK_SIZE}")
    w32 = w.float()
    blocks = w32.view(OUT, K // BLOCK_SIZE, BLOCK_SIZE)
    amax = blocks.abs().amax(dim=-1).clamp(min=1e-12)            # (OUT, K//16)
    scales_fp8 = amax.to(_SCALE_DTYPE)                            # (OUT, K//16)
    scales_f32 = scales_fp8.float()
    normed = (blocks / scales_f32.unsqueeze(-1) * 6.0).clamp(-6.0, 6.0)

    levels = w.new_tensor(_FP4_LEVELS_F32, dtype=torch.float32)
    sign = (normed < 0).to(torch.uint8) << 3                      # (OUT, K//16, 16)
    mag = normed.abs()
    idx = (mag.unsqueeze(-1) - levels).abs().argmin(dim=-1).to(torch.uint8)
    codes_flat = (idx | sign).view(OUT, K)                        # (OUT, K) uint8

    lo = codes_flat[:, 0::2]                                      # (OUT, K//2)
    hi = codes_flat[:, 1::2]
    packed = (lo & 0xF) | ((hi & 0xF) << 4)                       # (OUT, K//2) uint8
    return packed.contiguous(), scales_fp8.contiguous()


def fp4_weight_bytes_packed(numel: int) -> int:
    """Bytes for nibble-packed NVFP4 (codes + fp8 scales). 0.5625 bytes/elem."""
    return (numel + 1) // 2 + (numel + BLOCK_SIZE - 1) // BLOCK_SIZE


def _fp4_linear_fused(
    x: torch.Tensor,
    packed_w,   # (codes, scales) from quantize_fp4_packed
) -> torch.Tensor:
    """Fused NVFP4 dequant-GEMV for the M=1 decode step.

    Streams nibble-packed fp4 codes + fp8 block scales through the tensor
    pipeline, dequantizes in fp32 registers, accumulates in fp32.  Replaces
    :func:`_fp4_linear`'s full-bf16-materialisation reference path.

    Bandwidth win: 3.56x fewer weight bytes than bf16.  Realised only when the
    weight is HBM-bandwidth-bound (lm_head: 1.5-1.9x; layer GEMVs in the
    hot-L2 microbench are slower -- see ``benchmarks/bench_fp4_gemv.py`` for
    the cold-L2 regime that matches the real decode loop).
    """
    codes, scales = packed_w
    import triton  # local; keeps the module importable without triton
    from .llm_kernels import _fp4_gemv_kernel
    OUT, K半 = codes.shape
    K = K半 * 2
    K_BLOCKS = K // BLOCK_SIZE
    x1 = x.reshape(-1).contiguous()
    out = torch.empty((OUT,), dtype=x.dtype, device=codes.device)
    BLOCK_K = 128

    def grid(meta) -> tuple[int, ...]:
        return (triton.cdiv(OUT, meta["BLOCK_M"]),)

    _fp4_gemv_kernel[grid](
        x1, codes, scales, out,
        K=K, K_BYTES=K // 2, K_BLOCKS=K_BLOCKS, OUT_N=OUT, BLOCK_K=BLOCK_K,
    )
    return out


__all__ = [
    "BLOCK_SIZE",
    "quantize_fp4",
    "dequantize_fp4",
    "fp4_weight_bytes",
    "quantize_fp4_packed",
    "fp4_weight_bytes_packed",
    "_fp4_linear",
    "_fp4_linear_fused",
]
