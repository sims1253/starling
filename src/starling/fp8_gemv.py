"""Shared FP8 (e4m3) weight-only quantization for M=1 decoder GEMVs.

Decode is memory-bandwidth bound on the LLM weights: the per-layer projection
GEMVs (qkv/o/gateup/down) are ~57% of the captured decode step (per
``benchmarks/bench_decode_profile.py``), each a pure weight read at batch=1.
Casting those weights to fp8e4m3 halves the weight traffic -- this module
provides a **fused dequant-GEMV Triton kernel** that realizes that bandwidth
win without ``torch._scaled_mm``'s per-token activation-quant overhead (which
makes ``_scaled_mm`` ~9x *slower* than bf16 at M=1; see "Why not _scaled_mm"
below).

The fused kernel streams the fp8 weight, dequantizes each element to fp32 in
registers (via the hardware fp8->fp32 cast), and accumulates the dot product in
fp32.  The activation stays bf16 -- no per-token activation quantization, no
extra launches.  Pattern mirrors ``llm_kernels._fp4_gemv_kernel`` but simpler
(1-byte e4m3 codes vs nibble-packed e2m1; one per-output-channel scale vs
per-block scales).

Scaling
-------
Per-output-channel symmetric absmax scaling (``scale[o] = max|W[o,:]| / 448``).
The weight is stored row-major ``(N, K)`` fp8e4m3 (natural for the row-streaming
GEMV).  The activation is read as bf16 and cast to fp32 inside the kernel.

Consumers deliberately keep their ``lm_head`` in bf16 because vocabulary-wide
argmaxes have near-tie logits that fp8 rounding can flip. See
:attr:`starling.flags.OptFlags.fp8_weights`. Granite and MOSS share this
implementation because both decoder paths execute the same M=1 projection
pattern during autoregressive generation.

Why not torch._scaled_mm?
-------------------------
``_scaled_mm`` dispatches a general cutlass GEMM that is not tuned for M=1, and
it requires the activation pre-quantized to fp8 (an amax reduce + div + clamp +
cast per call).  At batch=1 those quant launches dominate: measured 174 us for
a qkv GEMV vs 22 us for bf16 ``F.linear`` (8.7x slower).  The fused kernel here
avoids both problems -- it is a purpose-built GEMV that dequantizes on the fly.

Public API
----------
``quantize_weight_e4m3(weight) -> (w_fp8, scale)``  (row-major ``(N,K)`` fp8 + ``(N,)`` fp32)
``fp8_linear(x, w_fp8, scale) -> y``  (``x @ W^T`` with an fp8 weight, M=1 GEMV)
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = 448.0


def quantize_weight_e4m3(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize an ``(N, K)`` linear weight to fp8e4m3, per-output-channel.

    Returns ``(w_fp8, scale)`` where ``w_fp8`` is a row-major ``(N, K)`` fp8e4m3
    tensor (the layout the fused GEMV streams) and ``scale`` is the ``(N,)``
    fp32 per-output-channel dequant scale.
    """
    amax = weight.abs().amax(dim=1).clamp(min=1e-8)  # (N,)
    scale = amax / FP8_MAX                            # (N,)
    w_fp8 = (weight / scale[:, None]).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE).contiguous()
    return w_fp8, scale.float()


# =========================================================================== #
# Fused fp8 dequant-GEMV Triton kernel
#
# out[o] = scale[o] * sum_k dequant(w_fp8[o,k]) * x[k]
#   w_fp8: (OUT, K) fp8e4m3, row-major
#   scale: (OUT,) fp32 per-output-channel
#   x:     (K,) bf16
# Each program handles BLOCK_M output rows, streaming K in BLOCK_K tiles.
# fp8 -> fp32 is a hardware cast (tl.load(...).to(tl.float32)).
# =========================================================================== #
_FP8_GEMV_CONFIGS = [
    triton.Config({"BLOCK_M": bm, "BLOCK_K": bk}, num_warps=w, num_stages=s)
    for bm in (16, 32, 64, 128)
    for bk in (128, 256, 512, 1024)
    for w in (4, 8)
    for s in (1, 2, 3)
]


@triton.autotune(_FP8_GEMV_CONFIGS, key=["OUT_N", "K"])
@triton.jit
def _fp8_gemv_kernel(
    X_ptr,          # (K,) bf16 input vector
    W_ptr,          # (OUT, K) fp8e4m3 weight (row-major)
    SCALE_ptr,      # (OUT,) fp32 per-output-channel scale
    OUT_ptr,        # (OUT,) bf16 output
    K: tl.constexpr,
    OUT_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_M
    rows = row_start + tl.arange(0, BLOCK_M)
    rmask = rows < OUT_N
    scale = tl.load(SCALE_ptr + rows, mask=rmask, other=0.0)  # (BLOCK_M,) fp32

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        cols = k0 + tl.arange(0, BLOCK_K)
        cmask = cols < K
        # load fp8 weight block: (BLOCK_M, BLOCK_K) fp8 -> fp32 (hardware cast)
        w_off = rows[:, None] * K + cols[None, :]
        wmask = cmask[None, :] & rmask[:, None]
        w = tl.load(W_ptr + w_off, mask=wmask, other=0.0).to(tl.float32)
        x = tl.load(X_ptr + cols, mask=cmask, other=0.0).to(tl.float32)
        acc += tl.sum(w * x[None, :], axis=1)
    acc = acc * scale
    tl.store(OUT_ptr + rows, acc.to(OUT_ptr.dtype.element_ty), mask=rmask)


def fp8_linear(x: torch.Tensor, w_fp8: torch.Tensor, w_scale: torch.Tensor) -> torch.Tensor:
    """``x @ W^T`` with an fp8 weight via the fused dequant-GEMV (M=1).

    Args:
        x: ``(1, K)`` or ``(K,)`` bf16 activation.
        w_fp8: ``(N, K)`` row-major fp8 weight from :func:`quantize_weight_e4m3`.
        w_scale: ``(N,)`` or ``(1, N)`` fp32 per-output-channel scale.

    Returns:
        ``(1, N)`` bf16 result, matching ``F.linear(x, W)`` to fp8 precision.
    """
    K = w_fp8.shape[1]
    N = w_fp8.shape[0]
    x1 = x.reshape(-1)[:K].contiguous()
    scale1d = w_scale.reshape(-1)
    out = torch.empty(N, dtype=torch.bfloat16, device=w_fp8.device)
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_M"]),)
    _fp8_gemv_kernel[grid](x1, w_fp8, scale1d, out, K=K, OUT_N=N)
    return out.view(1, N)
