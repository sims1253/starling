"""Granite fused decode kernels -- thin shim over the active backend.

Historically this file held the canonical hand-written Triton kernels for the
Granite decode path (RMSNorm, RoPE, SwiGLU, residual-scale, the experimental
GEMM-epilogue rstd/gemv_normscale fusion, and the fused NVFP4 dequant-GEMV).
Those kernels are now unified in :mod:`starling._kernels` behind a pluggable
backend (Triton on Linux, stock-PyTorch on Windows).  This module re-exports
them so existing consumer code is unchanged::

    from . import llm_kernels as _k
    ... = _k.fused_rmsnorm(...)
    ... = _k.fused_residual_scale(x, y, alpha=residual_multiplier)
    ... = _k.compute_rstd(...)
    ... = _k.fused_gemv_normscale(...)

The ``_fp4_gemv_kernel`` symbol is exposed lazily via :pep:`562` ``__getattr__``
because it is a Triton-only object (a ``@triton.jit`` kernel).  Importing it at
module top level would drag in the triton backend and break
``import starling.granite.llm_kernels`` on triton-less machines (Windows).  The
sole consumer, :func:`starling.granite.fp4._fp4_linear_fused`, already imports
it lazily inside the function; that call site keeps working, and the fused NVFP4
path simply raises ``ImportError`` on Windows (the reference
:func:`_fp4_linear` dequant path is the fallback there).  See
:mod:`starling._kernels` for the backend dispatch.
"""

from __future__ import annotations

from .._kernels import (
    AUTOTUNE,
    autotune_enabled,
    compute_rstd,
    fused_gemv_normscale,
    fused_rmsnorm,
    fused_rope,
    fused_silu_mul,
    set_autotune,
)
from .._kernels import residual_add
from .._kernels import residual_add as fused_residual_scale

__all__ = [
    "fused_rmsnorm",
    "fused_silu_mul",
    "fused_residual_scale",
    "fused_rope",
    "compute_rstd",
    "fused_gemv_normscale",
    "residual_add",
    "set_autotune",
    "autotune_enabled",
    "AUTOTUNE",
]


def __getattr__(name: str):
    # ``_fp4_gemv_kernel`` is a Triton @jit object -- defer its import so this
    # module stays importable without triton (Windows).  It is only needed by
    # the fused NVFP4 dequant-GEMV path (starling.granite.fp4._fp4_linear_fused).
    if name == "_fp4_gemv_kernel":
        from .._kernels.triton_backend import _fp4_gemv_kernel

        return _fp4_gemv_kernel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
