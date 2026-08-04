"""Unified Triton backend for Starling's fused decode kernels.

This is the **single** Triton implementation of every fused op Starling uses on
the LLM decode hot path.  It consolidates -- without behavioral change -- the
three previously-duplicated per-model copies of these kernels:

* ``starling/granite/llm_kernels.py`` (the canonical source of the elementwise
  kernels, RoPE, the GEMM-epilogue CODA fusion, and the fused NVFP4 dequant-GEMV
  kernel ``_fp4_gemv_kernel``),
* ``starling/fp8_gemv.py`` (the FP8 e4m3 weight-only dequant-GEMV), and
* the ``granite/fp4.py`` NVFP4 GEMV launch logic (``_fp4_linear_fused``).

Per-model ``llm_kernels.py`` modules and the ``starling._kernels`` dispatch
package (see :mod:`starling._kernels`) now re-export from this module, so no
consumer code changes.  On Windows (no Triton wheel) the ``torch`` backend
(:mod:`starling._kernels.torch_backend`) provides the same public names backed
by stock-PyTorch fused ops; the two are A/B-comparable via
``benchmarks/bench_kernels``.

Every kernel here is byte-for-byte identical to its canonical source (same
``@triton.jit`` bodies, same ``@triton.autotune`` config lists, same reduction
order and bf16 truncation points).  The only deliberate API changes versus the
original per-model files are:

1. ``fused_residual_scale(x, y, alpha)`` is renamed to
   :func:`residual_add` with signature ``residual_add(x, y, alpha=1.0)`` so it
   serves as the single unified residual: Granite calls it with
   ``alpha=residual_multiplier`` (~0.22), Moss/Qwen3 call it with the default
   ``alpha=1.0`` (which *is* plain ``x + y``).  The underlying
   ``_residual_scale_kernel`` is unchanged.
2. The fused NVFP4 dequant-GEMV launch logic from
   ``granite/fp4.py:_fp4_linear_fused`` is exposed as
   :func:`fp4_gemv_fused` taking the ``(codes, scales)`` tensors directly
   (rather than a 2-tuple) per the cross-backend interface in
   :mod:`starling._kernels.base`.

Public API (the union of all consumer calls across granite/higgs/moss):

Elementwise decode kernels (all LLM models):
    :func:`fused_rmsnorm`, :func:`fused_silu_mul`, :func:`residual_add`,
    :func:`fused_rope`

Granite GEMM-epilogue fusion (experimental, off by default):
    :func:`compute_rstd`, :func:`fused_gemv_normscale`

FP8 weight-only decode GEMV (opt-in via ``OptFlags.fp8_weights``):
    :func:`quantize_weight_e4m3`, :func:`fp8_linear`

FP4 fused GEMV (granite, experimental):
    :func:`fp4_gemv_fused`

Autotune control:
    :data:`AUTOTUNE`, :func:`set_autotune`, :func:`autotune_enabled`

Constants (re-exported from :mod:`starling._kernels.base`):
    :data:`FP8_DTYPE`, :data:`FP8_MAX`
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from .base import FP8_DTYPE, FP8_MAX

# =========================================================================== #
# Autotune toggle (Deliverable 1: "autotuned Triton" baseline).
#
# When ``AUTOTUNE`` is True the three decode-critical elementwise kernels
# (RMSNorm, SwiGLU silu*mul, residual scale-add) are wrapped in
# ``@triton.autotune`` over ``(num_warps, num_stages)`` so the CUDA-graph-
# captured decode picks the fastest launch config per feature dim on the RTX
# 5090. When False the kernels use Triton's default config -- this is the
# byte-exact fallback (identical to the pre-autotune path) used to measure the
# autotune delta.
#
# ``BLOCK_N`` stays launcher-computed (= ``next_power_of_2(N)``) so reduction
# coverage is always exact; ONLY ``num_warps``/``num_stages`` are swept, which
# never changes the elementwise arithmetic. For RMSNorm the ``tl.sum`` reduction
# order can in principle depend on ``num_warps``, but the resulting fp32 rstd
# delta is far below bf16 truncation granularity for the Granite hidden-state
# magnitudes -- verified bit-exact against the PyTorch reference (see
# ``test_fused_kernels_match_reference``). The OFF path (``.fn``) is exactly
# the original default-config launch, so it is guaranteed byte-exact.
# =========================================================================== #
AUTOTUNE: bool = os.environ.get("MEGAPAR_LLM_AUTOTUNE", "1") not in (
    "0", "", "false", "False",
)

# Config sweep: num_warps x num_stages. These kernels are single-program
# (grid=(M,) with M=1 at decode), so the sweep targets per-block parallelism /
# pipelining. 12 configs per (kernel, N); tuned once and cached.
_AT_CONFIGS = [
    triton.Config({}, num_warps=w, num_stages=s)
    for w in (1, 2, 4, 8)
    for s in (1, 2, 3)
]


def set_autotune(enabled: bool) -> None:
    """Enable/disable LLM-kernel autotuning at runtime (process-global)."""
    global AUTOTUNE
    AUTOTUNE = bool(enabled)


def autotune_enabled() -> bool:
    """Return whether LLM-kernel autotuning is active."""
    return AUTOTUNE


# =========================================================================== #
# Fused RMSNorm  (GraniteRMSNorm: no mean subtraction, fp32 variance)
#
# Input x: (M, N)  with N = hidden_size (2048).  For decode M = batch*seq = 1.
# Reference:
#   variance = mean(x.to(f32)^2, dim=-1)
#   x_normed = x * rsqrt(variance + eps)
#   output   = weight * x_normed
# =========================================================================== #
@triton.autotune(_AT_CONFIGS, key=["N"])
@triton.jit
def _rmsnorm_kernel(
    X_ptr, W_ptr, Y_ptr,
    eps,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    offs = row * N + cols
    dtype = Y_ptr.dtype.element_ty  # bf16

    x = tl.load(X_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    # RMS = mean(x^2) = sum(x^2) / N
    var = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    # Match GraniteRMSNorm exactly: normalize in fp32, truncate to bf16,
    # THEN multiply by weight in bf16.  Computing the weight product in fp32
    # and truncating once gives a different result (0.125 diff on real inputs).
    x_normed = (x * rstd).to(dtype)  # truncate to bf16 (matches model)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0)  # bf16 weight
    y = x_normed * w  # bf16 * bf16 (Triton uses fp32 internal, truncates to bf16)
    tl.store(Y_ptr + offs, y, mask=mask)


# OFF path: the raw JIT function under the autotuner (default config == the
# original pre-autotune launch). Guaranteed byte-exact fallback.
_rmsnorm_kernel_raw = _rmsnorm_kernel.fn


def fused_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm over the last dim, fp32 internally, bf16 in/out.

    ``x`` is ``(*, N)`` with ``N == weight.numel()``; one program per leading
    row.  For decode this is a single row of 2048 elements.
    """
    N = weight.numel()
    M = x.numel() // N
    x2 = x.reshape(M, N)
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    y = torch.empty_like(x2)
    BLOCK_N = triton.next_power_of_2(N)
    kern = _rmsnorm_kernel if AUTOTUNE else _rmsnorm_kernel_raw
    kern[(M,)](x2, weight, y, eps, N=N, BLOCK_N=BLOCK_N)
    return y.view_as(x)


# =========================================================================== #
# Fused RoPE  (apply rotary position embedding to Q and K simultaneously)
#
# Q: (B, n_q_heads, 1, head_dim)   K: (B, n_kv_heads, 1, head_dim)
# cos, sin: (B, 1, 1, head_dim) or broadcastable to (1, head_dim)
#
# rotate_half(x) = cat(-x[d/2:], x[:d/2])
# q_out = q * cos + rotate_half(q) * sin
# k_out = k * cos + rotate_half(k) * sin
#
# One program per (head) across both Q and K, total = n_q + n_kv programs.
# =========================================================================== #
@triton.jit
def _rope_kernel(
    Q_ptr, K_ptr, QO_ptr, KO_ptr,
    COS_ptr, SIN_ptr,
    n_q_heads,
    head_dim: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)  # 0 .. n_q_heads-1 -> Q,  n_q_heads .. n_q+n_kv-1 -> K
    cols = tl.arange(0, BLOCK_D)
    mask = cols < head_dim
    half = head_dim // 2
    dtype = QO_ptr.dtype.element_ty  # bf16

    # cos/sin are the same for all heads at this position
    cos = tl.load(COS_ptr + cols, mask=mask, other=0.0)
    sin = tl.load(SIN_ptr + cols, mask=mask, other=0.0)

    if pid < n_q_heads:
        src_ptr = Q_ptr + pid * head_dim
        dst_ptr = QO_ptr + pid * head_dim
    else:
        kid = pid - n_q_heads
        src_ptr = K_ptr + kid * head_dim
        dst_ptr = KO_ptr + kid * head_dim

    x = tl.load(src_ptr + cols, mask=mask, other=0.0)
    # rotate_half(x)[i] = -x[i+half] for i < half,  x[i-half] for i >= half
    lo = cols < half
    rot_idx = tl.where(lo, cols + half, cols - half)
    x_rot = tl.load(src_ptr + rot_idx, mask=mask, other=0.0)
    x_rot = tl.where(lo, -x_rot, x_rot)

    # Match PyTorch bf16 intermediate truncation: truncate each product to
    # bf16 BEFORE adding, then truncate the sum.  Computing in fp32 throughout
    # and truncating once gives different rounding than the eager reference.
    prod1 = (x * cos).to(dtype)
    prod2 = (x_rot * sin).to(dtype)
    out = prod1 + prod2
    tl.store(dst_ptr + cols, out, mask=mask)


def fused_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embedding to Q and K in one kernel launch.

    Args:
        q: ``(B, n_q_heads, 1, head_dim)`` bf16.
        k: ``(B, n_kv_heads, 1, head_dim)`` bf16.
        cos, sin: ``(B, 1, 1, head_dim)`` or ``(1, 1, head_dim)`` bf16/fp32.

    Returns:
        ``(q_rotated, k_rotated)`` same shapes/dtype as inputs.
    """
    B, n_q, _, hd = q.shape
    n_kv = k.shape[1]
    assert q.dtype == k.dtype
    # Flatten to (B * heads, hd) per tensor
    q_flat = q.reshape(B * n_q, hd)
    k_flat = k.reshape(B * n_kv, hd)
    # cos/sin: take the single position (seq=1)
    cos_flat = cos.reshape(-1, hd)[0:1].reshape(hd)  # (hd,) for B=1
    sin_flat = sin.reshape(-1, hd)[0:1].reshape(hd)

    q_out = torch.empty_like(q_flat)
    k_out = torch.empty_like(k_flat)
    total_heads = n_q + n_kv
    BLOCK_D = triton.next_power_of_2(hd)
    _rope_kernel[(total_heads,)](
        q_flat, k_flat, q_out, k_out, cos_flat, sin_flat,
        n_q, head_dim=hd, BLOCK_D=BLOCK_D,  # type: ignore
    )
    return q_out.view_as(q), k_out.view_as(k)


# =========================================================================== #
# Fused SiLU * Mul  (SwiGLU activation: silu(gate) * up)
#
# gate, up: (M, N)  with N = intermediate_size (4096).  For decode M = 1.
# out = silu(gate) * up = (gate / (1 + exp(-gate))) * up
# =========================================================================== #
@triton.autotune(_AT_CONFIGS, key=["N"])
@triton.jit
def _silu_mul_kernel(
    GATE_ptr, UP_ptr, OUT_ptr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    offs = row * N + cols
    dtype = OUT_ptr.dtype.element_ty  # bf16

    g = tl.load(GATE_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    # SiLU(g) = g * sigmoid(g); compute in fp32 then truncate to bf16 BEFORE
    # multiplying by up (matches PyTorch's ATen intermediate truncation).
    silu_g = g * (1.0 / (1.0 + tl.exp(-g)))
    silu_g_bf = silu_g.to(dtype)

    u = tl.load(UP_ptr + offs, mask=mask, other=0.0)  # bf16
    out = silu_g_bf * u  # bf16 * bf16
    tl.store(OUT_ptr + offs, out, mask=mask)


# OFF path: raw JIT (default config) -- original byte-exact launch.
_silu_mul_kernel_raw = _silu_mul_kernel.fn


def fused_silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """SiLU(gate) * up fused into one kernel, fp32 internally.

    ``gate`` and ``up`` are ``(*, N)`` with the same shape; one program per
    leading row.
    """
    N = gate.shape[-1]
    M = gate.numel() // N
    g2 = gate.reshape(M, N)
    u2 = up.reshape(M, N)
    if not g2.is_contiguous():
        g2 = g2.contiguous()
    if not u2.is_contiguous():
        u2 = u2.contiguous()
    out = torch.empty_like(g2)
    BLOCK_N = triton.next_power_of_2(N)
    kern = _silu_mul_kernel if AUTOTUNE else _silu_mul_kernel_raw
    kern[(M,)](g2, u2, out, N=N, BLOCK_N=BLOCK_N)
    return out.view_as(gate)


# =========================================================================== #
# Fused residual scale-add  (x + alpha * y)  -- the unified residual kernel.
#
# x, y: (M, N)  with N = hidden_size (2048).  alpha = residual_multiplier.
#
# This is the single residual kernel for every model: Granite passes
# ``alpha=residual_multiplier`` (~0.22); Moss/Qwen3 pass ``alpha=1.0`` (which
# is plain ``x + y``).  The implementation is the Granite autotuned
# ``_residual_scale_kernel`` unchanged.
# =========================================================================== #
@triton.autotune(_AT_CONFIGS, key=["N"])
@triton.jit
def _residual_scale_kernel(
    X_ptr, Y_ptr, Z_ptr, ALPHA,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    offs = row * N + cols
    dtype = Z_ptr.dtype.element_ty  # bf16

    # Match model's ``residual + delta * multiplier``: compute the scaled delta
    # in fp32, truncate to bf16, THEN add the residual (also bf16).
    y = tl.load(Y_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    scaled = (ALPHA * y).to(dtype)  # bf16 (truncated product)

    x = tl.load(X_ptr + offs, mask=mask, other=0.0)  # bf16 residual
    z = x + scaled  # bf16 + bf16
    tl.store(Z_ptr + offs, z, mask=mask)


# OFF path: raw JIT (default config) -- original byte-exact launch.
_residual_scale_kernel_raw = _residual_scale_kernel.fn


def residual_add(
    x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0
) -> torch.Tensor:
    """``x + alpha * y`` fused into one kernel, fp32 internally.

    Unified residual used by all models.  Granite calls this with
    ``alpha=residual_multiplier`` (~0.22); Moss/Qwen3 call it with the default
    ``alpha=1.0`` (plain ``x + y``).  Backed by the autotuned
    ``_residual_scale_kernel``.

    Args:
        x: the residual (``(*, N)`` bf16).
        y: the delta to scale-add (``(*, N)`` bf16).
        alpha: scalar multiplier on ``y``; defaults to ``1.0``.

    Returns:
        ``(*, N)`` bf16, same shape as ``x``.
    """
    N = x.shape[-1]
    M = x.numel() // N
    x2 = x.reshape(M, N)
    y2 = y.reshape(M, N)
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    if not y2.is_contiguous():
        y2 = y2.contiguous()
    z = torch.empty_like(x2)
    BLOCK_N = triton.next_power_of_2(N)
    kern = _residual_scale_kernel if AUTOTUNE else _residual_scale_kernel_raw
    kern[(M,)](x2, y2, z, alpha, N=N, BLOCK_N=BLOCK_N)
    return z.view_as(x)


# =========================================================================== #
# GEMM-epilogue fusion (CODA Pattern 1): RMSNorm folded into the next GEMV.
#
# At decode (batch=1, seq=1) the QKV / gate-up projections are GEMVs (M=1).
# The RMSNorm scale ``r = rsqrt(mean(x^2)+eps)`` commutes with the matmul, so
#   rmsnorm(x) @ W^T  ==  (x @ (gamma .* W)^T) * r
# We compute ``r`` with a one-program scalar kernel, prescale ``W`` by ``gamma``
# once at load time (see FusedLLMMega._fuse_epilogue_weights), and fold the
# ``* r`` into the GEMV accumulator. Eliminates the standalone _rmsnorm_kernel
# launch before each of the two projections per layer (80 launches/step).
#
# Numerical note: the unfused path truncates the *normalized* hidden to bf16
# before the GEMV; this path accumulates in fp32 and truncates once. Per-layer
# max-abs diff ~0.25 (qkv) / ~0.5 (gate-up) on magnitude-70..97 projections,
# i.e. ~1-3 bf16 ULP. NOT byte-exact; gated by OptFlags.gemm_epilogue_fusion.
# =========================================================================== #
@triton.jit
def _rstd_kernel(X_ptr, RSTD_ptr, eps, N: tl.constexpr, BLOCK_N: tl.constexpr):
    # Single-program scalar kernel: writes rstd = rsqrt(mean(x^2)+eps) for one row.
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(X_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    tl.store(RSTD_ptr, 1.0 / tl.sqrt(var + eps))


def compute_rstd(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Row rstd = rsqrt(mean(x^2)+eps) as a (1,) fp32 scalar. x is (N,) or (1,N)."""
    N = x.shape[-1]
    x1 = x.reshape(-1).contiguous()
    rstd = torch.empty((1,), dtype=torch.float32, device=x.device)
    _rstd_kernel[(1,)](x1, rstd, eps, N=N, BLOCK_N=triton.next_power_of_2(N))  # type: ignore
    return rstd


@triton.jit
def _gemv_normscale_kernel(
    X_ptr, W_ptr, RSTD_ptr, OUT_ptr, OUT: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """GEMV (M=1) with RMSNorm scale folded into the epilogue.

    W is (OUT, K), already prescaled by gamma. Computes
    ``out[i] = rstd * sum_k W[i,k]*x[k]``.
    """
    pid = tl.program_id(0)
    row_start = pid * BLOCK_M
    rows = row_start + tl.arange(0, BLOCK_M)
    rmask = rows < OUT
    rstd = tl.load(RSTD_ptr)
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        cols = k0 + tl.arange(0, BLOCK_K)
        cmask = cols < K
        x = tl.load(X_ptr + cols, mask=cmask, other=0.0).to(tl.float32)
        w = tl.load(W_ptr + rows[:, None] * K + cols[None, :],
                    mask=cmask[None, :] & rmask[:, None], other=0.0).to(tl.float32)
        acc += tl.sum(w * x[None, :], axis=1)
    acc = acc * rstd
    tl.store(OUT_ptr + rows, acc.to(OUT_ptr.dtype.element_ty), mask=rmask)


def fused_gemv_normscale(x: torch.Tensor, w_scaled: torch.Tensor,
                         rstd: torch.Tensor) -> torch.Tensor:
    """GEMV (M=1) of x @ w_scaled^T with the RMSNorm rstd folded into the epilogue.

    x: (1, K) or (K,); w_scaled: (OUT, K) already *= gamma; rstd: (1,) fp32.
    Returns (OUT,) bf16.
    """
    K = w_scaled.shape[1]
    OUT = w_scaled.shape[0]
    x1 = x.reshape(-1).contiguous()
    out = torch.empty((OUT,), dtype=w_scaled.dtype, device=w_scaled.device)
    BLOCK_M = 64
    BLOCK_K = min(triton.next_power_of_2(K), 1024)
    grid = (triton.cdiv(OUT, BLOCK_M),)
    _gemv_normscale_kernel[grid](
        x1, w_scaled, rstd, out, OUT=OUT, K=K, BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,  # type: ignore
    )
    return out


# =========================================================================== #
# FP8 (e4m3) weight-only quantization + fused dequant-GEMV (M=1 decode).
#
# Ported verbatim from ``starling/fp8_gemv.py``.  Per-output-channel symmetric
# absmax scaling (scale = amax / 448), weight stored row-major (N, K) fp8e4m3.
# The fused GEMV streams the fp8 weight, dequantizes each element to fp32 in
# registers (hardware fp8->fp32 cast), and accumulates in fp32.  Activation
# stays bf16 -- no per-token activation quant, no extra launches.
# =========================================================================== #
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

    def grid(meta) -> tuple[int, ...]:
        return (triton.cdiv(N, meta["BLOCK_M"]),)

    _fp8_gemv_kernel[grid](x1, w_fp8, scale1d, out, K=K, OUT_N=N)
    return out.view(1, N)


# =========================================================================== #
# Fused NVFP4 dequant-GEMV.
#
# Reads nibble-packed e2m1 codes (uint8, 2/byte) + fp8e4m3 block scales and
# accumulates out[i] = sum_k dequant(code[i,k], scale[i,k//16]) * x[k] in fp32.
# This is the bandwidth win that makes NVFP4 weights worth it on the decode
# step (GEMVs are 51% of step time per bench_decode_profile.py).  Pattern
# mirrors A4Q -- packed fp4 streams into the pipeline and dequantizes in
# registers; here for a plain GEMV rather than a paged attention QK^T.
#
# Storage layout (see starling.granite.fp4.quantize_fp4_packed):
#   codes  : (OUT, K // 2)  uint8 -- even k -> low nibble, odd k -> high nibble
#   scales : (OUT, K // 16) float8_e4m3fn -- one per 16-element block
# Reconstruction:  w ~= scale_fp8 * e2m1_level(code) / 6.0
#
# The ``_fp4_gemv_kernel`` body + ``_FP4_GEMV_CONFIGS`` are ported verbatim from
# ``starling/granite/llm_kernels.py``; the launch logic of
# :func:`fp4_gemv_fused` is ported from ``starling/granite/fp4.py:
# _fp4_linear_fused``.  ``BLOCK_SIZE`` is the NVFP4 block size (16 elements per
# scale), kept here so the launcher is self-contained.
# =========================================================================== #
BLOCK_SIZE: int = 16
"""NVFP4 block size (elements per fp8 scale)."""

_FP4_GEMV_CONFIGS = [
    triton.Config({"BLOCK_M": bm}, num_warps=w, num_stages=s)
    for bm in (16, 32, 64, 128)
    for w in (4, 8, 16)
    for s in (1, 2, 3, 4)
]


@triton.autotune(_FP4_GEMV_CONFIGS, key=["OUT_N", "K"])
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
    BLOCK_K: tl.constexpr,   # multiple of 32 (2 bytes -> 4 codes -> 2 scale blocks)
):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_M
    rows = row_start + tl.arange(0, BLOCK_M)
    rmask = rows < OUT_N

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    BLOCK_BYTES: tl.constexpr = BLOCK_K // 2

    for kb in range(0, K_BYTES, BLOCK_BYTES):
        byte_cols = kb + tl.arange(0, BLOCK_BYTES)
        bmask = byte_cols < K_BYTES

        # Load code bytes: (BLOCK_M, BLOCK_BYTES)
        byte_off = rows[:, None] * K_BYTES + byte_cols[None, :]
        bm = bmask[None, :] & rmask[:, None]
        raw = tl.load(CODES_ptr + byte_off, mask=bm, other=0)

        # Extract nibbles -> 2 codes per byte.
        code_lo = raw & 0xF           # even k (low nibble)
        code_hi = (raw >> 4) & 0xF    # odd  k (high nibble)

        # --- e2m1 dequant (bit 3 = sign, bits[2:0] = magnitude level index) ---
        # Levels: idx 0..7 -> (0, 0.5, 1, 1.5, 2, 3, 4, 6).
        sign_lo = (code_lo >> 3) & 1
        mag_lo = code_lo & 0x7
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

        # Block scales: element 2*byte_cols[j] -> block (2*byte_cols[j])//16.
        elem_lo = 2 * byte_cols
        elem_hi = 2 * byte_cols + 1
        blk_lo = elem_lo // 16
        blk_hi = elem_hi // 16
        scale_off_lo = rows[:, None] * K_BLOCKS + blk_lo[None, :]
        scale_off_hi = rows[:, None] * K_BLOCKS + blk_hi[None, :]
        scale_lo = tl.load(SCALES_ptr + scale_off_lo, mask=bm, other=0.0).to(tl.float32)
        scale_hi = tl.load(SCALES_ptr + scale_off_hi, mask=bm, other=0.0).to(tl.float32)

        w_lo = level_lo * scale_lo * (1.0 / 6.0)
        w_hi = level_hi * scale_hi * (1.0 / 6.0)

        x_lo = tl.load(X_ptr + elem_lo, mask=bmask, other=0.0).to(tl.float32)
        x_hi = tl.load(X_ptr + elem_hi, mask=bmask, other=0.0).to(tl.float32)

        acc += tl.sum(w_lo * x_lo[None, :], axis=1)
        acc += tl.sum(w_hi * x_hi[None, :], axis=1)

    tl.store(OUT_ptr + rows, acc.to(OUT_ptr.dtype.element_ty), mask=rmask)


def fp4_gemv_fused(
    x: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    """Fused NVFP4 dequant-GEMV for the M=1 decode step.

    Streams nibble-packed fp4 codes + fp8 block scales through the tensor
    pipeline, dequantizes in fp32 registers, accumulates in fp32.  This is the
    bandwidth win (3.56x fewer weight bytes than bf16) realized for the
    decode-step GEMV.  Launch logic ported from
    ``starling/granite/fp4.py:_fp4_linear_fused``.

    Args:
        x: ``(1, K)`` or ``(K,)`` bf16 activation.
        codes: ``(OUT, K // 2)`` uint8 nibble-packed e2m1 codes (even ``k`` ->
            low nibble, odd ``k`` -> high nibble) from
            :func:`starling.granite.fp4.quantize_fp4_packed`.
        scales: ``(OUT, K // 16)`` ``float8_e4m3fn`` per-16-element-block scales
            from :func:`starling.granite.fp4.quantize_fp4_packed`.

    Returns:
        ``(OUT,)`` bf16 result.
    """
    OUT, K_bytes = codes.shape
    K = K_bytes * 2
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
    # constants (re-exported from base)
    "FP8_DTYPE",
    "FP8_MAX",
    # autotune control
    "AUTOTUNE",
    "set_autotune",
    "autotune_enabled",
    # elementwise decode kernels
    "fused_rmsnorm",
    "fused_silu_mul",
    "residual_add",
    "fused_rope",
    # Granite GEMM-epilogue fusion
    "compute_rstd",
    "fused_gemv_normscale",
    # FP8 weight-only decode GEMV
    "quantize_weight_e4m3",
    "fp8_linear",
    # FP4 fused GEMV
    "fp4_gemv_fused",
]
