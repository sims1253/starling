"""Triton fused kernels for the Granite LLM decode path (Phase C).

These kernels replace the small elementwise "glue" ops inside each decoder
layer with single-launch fused variants to cut memory traffic and kernel
launches during CUDA-graph-captured single-token decode.

All GEMMs (q/k/v/o_proj, gate/up/down_proj, lm_head) stay as cuBLAS bf16
matmuls; only the memory-bound elementwise ops are fused.  Every kernel uses
**fp32 internal accumulation** so bf16 outputs match the stock PyTorch ops
to within rounding (tolerance: ``LLM_LOGIT_ATOL`` = 0.05 max abs logit diff).

Kernels:
    * :func:`fused_rmsnorm`     - GraniteRMSNorm (no mean subtraction) in one
      kernel; replaces pow/mean/rsqrt/mul/cast chain.
    * :func:`fused_rope`        - rotary embedding applied to Q and K in one
      kernel (rotate_half + cos/sin broadcast); replaces ~8 separate ops.
    * :func:`fused_silu_mul`    - SwiGLU ``silu(gate) * up`` in one kernel;
      replaces silu + mul (and the intermediate allocation between them).
    * :func:`fused_residual_scale` - ``x + alpha * y`` for the Granite residual
      connections (``residual_multiplier`` = 0.22).

The decode tensors are tiny (batch=1, seq=1): hidden (1,1,2048), per-head
(1,1,128), intermediate (1,1,4096).  Each kernel uses a single program (or one
per head) with a constexpr block covering the full feature dimension, so there
is zero wasted work and minimal launch overhead.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl


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
        n_q, head_dim=hd, BLOCK_D=BLOCK_D,
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
# Fused residual scale-add  (x + alpha * y)  for Granite residual connections
#
# x, y: (M, N)  with N = hidden_size (2048).  alpha = residual_multiplier.
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


def fused_residual_scale(
    x: torch.Tensor, y: torch.Tensor, alpha: float
) -> torch.Tensor:
    """x + alpha * y fused into one kernel, fp32 internally."""
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
