"""Hand-written Triton fused kernels for the conformer encoder elementwise glue.

All GEMMs and cuDNN convolutions stay as torch ops; these kernels only fuse the
small elementwise "glue" ops (LayerNorm, SiLU, residual adds, BatchNorm+SiLU)
that are otherwise one CUDA launch each. Variance/accumulation is done in fp32
internally so bf16 outputs match the stock PyTorch ops to within rounding.

Elementwise binary/unary kernels are STRIDE-AWARE so they accept the
non-contiguous views the conv module produces (``down_conv(h).permute(0,2,1)``)
without forcing an extra copy.

Each public function is a thin launcher that:
  * flattens / reshapes the operand to the kernel's expected 2D layout,
  * launches the kernel with a static block size, and
  * returns a freshly-allocated output tensor (never mutates inputs).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# =========================================================================== #
# LayerNorm (last-dim), fp32 accumulation, bf16 in/out.
# Input ``x`` is treated as (M, N) with N = weight.numel(); the input MUST be
# contiguous in the last dim (it always is in our encoder: comes from a Linear
# or a residual-add output).
# =========================================================================== #
@triton.jit
def _layernorm_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    eps, N: tl.constexpr, BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    rx = row.to(tl.int64)
    offs = rx * N + cols
    x = tl.load(X_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    xc = x - mean
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = xc * rstd * w + b
    tl.store(Y_ptr + offs, y.to(Y_ptr.dtype.element_ty), mask=mask)


def fused_layernorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """LayerNorm over the last dim, fp32 internally.

    ``x`` is treated as ``(*, N)`` where ``N == weight.numel()``; the kernel
    is launched with one program per leading row. Input is made row-contiguous
    (it always is in practice).
    """
    N = weight.numel()
    M = x.numel() // N
    x2 = x.reshape(M, N)
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    y = torch.empty_like(x2)
    BLOCK_N = triton.next_power_of_2(N)
    _layernorm_kernel[(M,)](
        x2, weight, bias, y, eps, N=N, BLOCK_N=BLOCK_N,
    )
    return y.view_as(x)


# =========================================================================== #
# Stride-aware 3D elementwise kernels for (1, T, D) tensors.
# Used for SiLU and the residual adds, which may receive non-contiguous views
# (e.g. the conv module's ``down_conv(h).permute(0, 2, 1)`` output).
# =========================================================================== #
@triton.jit
def _silu_kernel_3d(
    X_ptr, Y_ptr,
    sx1, sx2, N, D: tl.constexpr, BLOCK: tl.constexpr,
):
    # grid over N = T*D elements (batch dim is 0, size 1)
    pid = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    idx = pid * BLOCK + cols
    mask = idx < N
    # decompose linear idx -> (t, d) for batch=0
    t = idx // D
    d = idx % D
    offs = t * sx1 + d * sx2
    x = tl.load(X_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = x * (1.0 / (1.0 + tl.exp(-x)))
    tl.store(Y_ptr + idx, y.to(Y_ptr.dtype.element_ty), mask=mask)  # Y is contiguous


def fused_silu(x: torch.Tensor) -> torch.Tensor:
    """SiLU on a (1, T, D) tensor (stride-aware on the input)."""
    assert x.ndim == 3
    T, D = x.shape[1], x.shape[2]
    n = x.numel()
    y = torch.empty((1, T, D), dtype=x.dtype, device=x.device)
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    s = x.stride()
    _silu_kernel_3d[grid](x, y, s[1], s[2], n, D=D, BLOCK=BLOCK)
    return y


@triton.jit
def _add_kernel_3d(
    X_ptr, Y_ptr, Z_ptr,
    sx1, sx2, sy1, sy2, N, D: tl.constexpr, BLOCK: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    idx = pid * BLOCK + cols
    mask = idx < N
    t = idx // D
    d = idx % D
    x = tl.load(X_ptr + t * sx1 + d * sx2, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(Y_ptr + t * sy1 + d * sy2, mask=mask, other=0.0).to(tl.float32)
    tl.store(Z_ptr + idx, (x + y).to(Z_ptr.dtype.element_ty), mask=mask)


def fused_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """x + y for (1, T, D) tensors (stride-aware on both inputs)."""
    assert x.ndim == 3 and x.shape == y.shape
    T, D = x.shape[1], x.shape[2]
    n = x.numel()
    z = torch.empty((1, T, D), dtype=x.dtype, device=x.device)
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    sx, sy = x.stride(), y.stride()
    _add_kernel_3d[grid](x, y, z, sx[1], sx[2], sy[1], sy[2], n, D=D, BLOCK=BLOCK)
    return z


@triton.jit
def _residual_scale_add_kernel_3d(
    X_ptr, Y_ptr, Z_ptr, ALPHA,
    sx1, sx2, sy1, sy2, N, D: tl.constexpr, BLOCK: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    idx = pid * BLOCK + cols
    mask = idx < N
    t = idx // D
    d = idx % D
    x = tl.load(X_ptr + t * sx1 + d * sx2, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(Y_ptr + t * sy1 + d * sy2, mask=mask, other=0.0).to(tl.float32)
    z = x + ALPHA * y
    tl.store(Z_ptr + idx, z.to(Z_ptr.dtype.element_ty), mask=mask)


def fused_residual_scale_add(
    x: torch.Tensor, y: torch.Tensor, alpha: float
) -> torch.Tensor:
    """x + alpha * y for (1, T, D) tensors (stride-aware on both inputs)."""
    assert x.ndim == 3 and x.shape == y.shape
    T, D = x.shape[1], x.shape[2]
    n = x.numel()
    z = torch.empty((1, T, D), dtype=x.dtype, device=x.device)
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    sx, sy = x.stride(), y.stride()
    _residual_scale_add_kernel_3d[grid](
        x, y, z, alpha, sx[1], sx[2], sy[1], sy[2], n, D=D, BLOCK=BLOCK
    )
    return z


# =========================================================================== #
# BatchNorm1d (eval) + SiLU, per-channel over the length dim.
# Input layout: (1, C, L) -> the kernel normalises each channel over L using
# the cached running stats, then applies SiLU. Output is contiguous (1, C, L).
# =========================================================================== #
@triton.jit
def _bn_silu_kernel(
    X_ptr, W_ptr, B_ptr, RM_ptr, RV_ptr, Y_ptr,
    eps, L: tl.constexpr, BLOCK_L: tl.constexpr,
):
    c = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_L)
    mask = cols < L
    offs = c * L + cols
    rm = tl.load(RM_ptr + c).to(tl.float32)
    rv = tl.load(RV_ptr + c).to(tl.float32)
    w = tl.load(W_ptr + c).to(tl.float32)
    b = tl.load(B_ptr + c).to(tl.float32)
    # Use the hardware rsqrt (matches torch's native batch_norm, which uses
    # rsqrt internally rather than 1.0/sqrt).
    rstd = tl.rsqrt(rv + eps)
    x = tl.load(X_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = (x - rm) * rstd * w + b
    y = y * (1.0 / (1.0 + tl.exp(-y)))
    tl.store(Y_ptr + offs, y.to(Y_ptr.dtype.element_ty), mask=mask)


def fused_batchnorm_silu(
    x: torch.Tensor,  # (1, C, L)
    weight: torch.Tensor,  # (C,)
    bias: torch.Tensor,  # (C,)
    running_mean: torch.Tensor,  # (C,)
    running_var: torch.Tensor,  # (C,)
    eps: float,
) -> torch.Tensor:
    C, L = x.shape[1], x.shape[2]
    x2 = x.reshape(C, L)
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    y = torch.empty_like(x2)
    BLOCK_L = triton.next_power_of_2(L)
    _bn_silu_kernel[(C,)](
        x2, weight, bias, running_mean, running_var, y, eps, L=L, BLOCK_L=BLOCK_L,
    )
    return y.view(1, C, L)
