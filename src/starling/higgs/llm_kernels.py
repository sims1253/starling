"""Higgs-Audio Qwen3 fused decode kernels -- thin shim over the active backend.

Historically this file was a verbatim copy of the Granite Triton kernels
(RMSNorm, RoPE, SwiGLU, residual-scale).  Those kernels are now unified in
:mod:`starling._kernels` behind a pluggable backend (Triton on Linux,
stock-PyTorch on Windows), removing the copy-divergence risk.  This module
re-exports them so the existing consumer code is unchanged::

    from . import llm_kernels as _k
    ... = _k.fused_rmsnorm(...)
    ... = _k.fused_silu_mul(...)
    ... = _k.fused_residual_scale(...)

Granite/Higgs residuals use ``x + alpha*y`` (``alpha = residual_multiplier``),
exposed here as both ``fused_residual_scale`` (the historical name) and
``residual_add`` (the unified backend name).  See :mod:`starling._kernels`.
"""

from __future__ import annotations

from .._kernels import (
    fused_rmsnorm,
    fused_rope,
    fused_silu_mul,
)
from .._kernels import residual_add
from .._kernels import residual_add as fused_residual_scale

__all__ = [
    "fused_rmsnorm",
    "fused_silu_mul",
    "fused_residual_scale",
    "fused_rope",
    "residual_add",
]
