"""MOSS Qwen3 fused decode kernels -- thin shim over the active backend.

Historically this file held hand-written Triton kernels for the MOSS decode
path (RMSNorm, SwiGLU, residual).  Those kernels are now unified in
:mod:`starling._kernels` behind a pluggable backend (Triton on Linux,
stock-PyTorch on Windows).  This module re-exports them so the existing
consumer code is unchanged::

    from . import llm_kernels as _k
    ... = _k.fused_rmsnorm(...)
    ... = _k.fused_silu_mul(...)
    ... = _k.fused_residual(...)

Qwen3/MOSS residuals are plain ``x + y`` (no Granite ``residual_multiplier``),
so ``fused_residual`` aliases the backend's ``residual_add`` with its default
``alpha=1.0``.  See :mod:`starling._kernels` for the backend dispatch.
"""

from __future__ import annotations

from .._kernels import (
    fused_rmsnorm,
    fused_rope,
    fused_silu_mul,
)
from .._kernels import residual_add as fused_residual
from .._kernels import residual_add

__all__ = [
    "fused_rmsnorm",
    "fused_silu_mul",
    "fused_residual",
    "fused_rope",
    "residual_add",
]
