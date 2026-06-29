"""Triton fused elementwise kernels for the MOSS Qwen3 decode path.

Adapted from ``starling.granite.llm_kernels``.  These replace the small
memory-bound elementwise ops inside each Qwen3 decoder layer with
single-launch fused variants to cut memory traffic and kernel launches during
CUDA-graph-captured single-token decode.

All GEMMs (q/k/v/o_proj, gate/up/down_proj, lm_head, q_norm/k_norm) stay as
cuBLAS bf16 matmuls; only the elementwise glue is fused.  Every kernel uses
fp32 internal accumulation so bf16 outputs match the stock PyTorch ops.

Kernels:
    * :func:`fused_rmsnorm`     - Qwen3RMSNorm (fp32 variance, mean-subtracted
      is NOT used by Qwen3 -- it's the same no-mean form as granite) in one
      kernel.
    * :func:`fused_silu_mul`    - SwiGLU ``silu(gate) * up``.
    * :func:`fused_residual`    - ``x + y`` (Qwen3 residuals are plain adds,
      no Granite residual_multiplier).
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

AUTOTUNE: bool = os.environ.get("MOSS_LLM_AUTOTUNE", "1") not in (
    "0", "", "false", "False",
)

_AT_CONFIGS = [
    triton.Config({}, num_warps=w, num_stages=s)
    for w in (1, 2, 4, 8)
    for s in (1, 2, 3)
]


@triton.autotune(_AT_CONFIGS, key=["N"])
@triton.jit
def _rmsnorm_kernel(X_ptr, W_ptr, Y_ptr, eps, N: tl.constexpr, BLOCK_N: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    offs = row * N + cols
    dtype = Y_ptr.dtype.element_ty  # bf16
    x = tl.load(X_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    x_normed = (x * rstd).to(dtype)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0)
    y = x_normed * w
    tl.store(Y_ptr + offs, y, mask=mask)


_rmsnorm_kernel_raw = _rmsnorm_kernel.fn


def fused_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm over the last dim, fp32 internally, bf16 in/out."""
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


@triton.autotune(_AT_CONFIGS, key=["N"])
@triton.jit
def _silu_mul_kernel(GATE_ptr, UP_ptr, OUT_ptr, N: tl.constexpr, BLOCK_N: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    offs = row * N + cols
    dtype = OUT_ptr.dtype.element_ty  # bf16
    g = tl.load(GATE_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    silu_g = g * (1.0 / (1.0 + tl.exp(-g)))
    silu_g_bf = silu_g.to(dtype)
    u = tl.load(UP_ptr + offs, mask=mask, other=0.0)
    out = silu_g_bf * u
    tl.store(OUT_ptr + offs, out, mask=mask)


_silu_mul_kernel_raw = _silu_mul_kernel.fn


def fused_silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """SiLU(gate) * up fused, fp32 internally."""
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


@triton.autotune(_AT_CONFIGS, key=["N"])
@triton.jit
def _residual_kernel(X_ptr, Y_ptr, Z_ptr, N: tl.constexpr, BLOCK_N: tl.constexpr):
    """z = x + y (bf16 + bf16).  Qwen3 has no residual multiplier."""
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    offs = row * N + cols
    dtype = Z_ptr.dtype.element_ty  # bf16
    x = tl.load(X_ptr + offs, mask=mask, other=0.0)
    y = tl.load(Y_ptr + offs, mask=mask, other=0.0)
    z = x + y
    tl.store(Z_ptr + offs, z, mask=mask)


_residual_kernel_raw = _residual_kernel.fn


def fused_residual(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """x + y fused (bf16)."""
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
    kern = _residual_kernel if AUTOTUNE else _residual_kernel_raw
    kern[(M,)](x2, y2, z, N=N, BLOCK_N=BLOCK_N)
    return z.view_as(x)
