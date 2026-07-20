"""Cross-platform kernel backend dispatch.

This package is the single entry point for Starling's fused decode kernels.
Per-model ``llm_kernels.py`` modules are thin re-export shims::

    # src/starling/moss/llm_kernels.py
    from .._kernels import (            # resolves to the active backend
        fused_rmsnorm,
        fused_silu_mul,
        residual_add as fused_residual,
        fused_rope,
    )

so consumer code (``from . import llm_kernels as _k``) is unchanged regardless
of platform.

Backend selection
------------------
Resolved once (lazily, at first ``__getattr__`` access) from:

1. :func:`set_backend` (explicit programmatic override), else
2. the ``STARLING_KERNEL_BACKEND`` env var (``auto``|``triton``|``torch``|
   ``cuda``), else
3. ``"auto"`` = triton if :func:`have_triton` else torch.

On Linux with triton installed, ``auto`` → triton (max performance).  On
Windows (no triton wheel), ``auto`` → torch (stock PyTorch fused ops).

The benchmark harness (:mod:`benchmarks.bench_kernels`) imports the backend
modules directly (``from starling._kernels import triton_backend,
torch_backend``) to A/B them, bypassing this dispatch.
"""

from __future__ import annotations

import os

from .base import FP8_DTYPE, FP8_MAX, have_cuda_compile, have_triton

__all__ = [
    # dispatch
    "have_triton",
    "have_cuda_compile",
    "get_backend",
    "get_backend_name",
    "set_backend",
    # constants
    "FP8_DTYPE",
    "FP8_MAX",
    # fused ops (delegated to the active backend via __getattr__)
    "fused_rmsnorm",
    "fused_silu_mul",
    "residual_add",
    "fused_rope",
    "compute_rstd",
    "fused_gemv_normscale",
    "quantize_weight_e4m3",
    "fp8_linear",
    "fp4_gemv_fused",
    "set_autotune",
    "autotune_enabled",
    "AUTOTUNE",
]

# Names resolved lazily via __getattr__ against the active backend module.
# Must match the public function/attribute names defined by every backend.
_PUBLIC = frozenset({
    "fused_rmsnorm",
    "fused_silu_mul",
    "residual_add",
    "fused_rope",
    "compute_rstd",
    "fused_gemv_normscale",
    "quantize_weight_e4m3",
    "fp8_linear",
    "fp4_gemv_fused",
    "set_autotune",
    "autotune_enabled",
    "AUTOTUNE",
})

_ACTIVE_NAME: str | None = None
_ACTIVE = None


def get_backend_name() -> str:
    """Return the effective backend name (resolving ``auto``)."""
    if _ACTIVE_NAME is not None:
        name = _ACTIVE_NAME
    else:
        name = os.environ.get("STARLING_KERNEL_BACKEND", "auto").strip().lower()
    if name == "auto":
        # Preference order: triton (Linux, max perf + autotuning) > cuda
        # (Windows w/ CUDA toolkit, full fused-kernel perf via load_inline) >
        # torch (pure-PyTorch correctness fallback, anywhere).
        if have_triton():
            name = "triton"
        elif have_cuda_compile():
            name = "cuda"
        else:
            name = "torch"
    return name


def get_backend():
    """Return the active kernel backend module (importing it on first call)."""
    global _ACTIVE
    if _ACTIVE is not None:
        return _ACTIVE
    name = get_backend_name()
    if name == "triton":
        if not have_triton():
            raise ImportError(
                "kernel backend 'triton' requested but triton is not "
                "importable (no Windows wheel?). Set STARLING_KERNEL_BACKEND=torch "
                "or install triton."
            )
        from . import triton_backend

        _ACTIVE = triton_backend
    elif name == "torch":
        from . import torch_backend

        _ACTIVE = torch_backend
    elif name == "cuda":
        try:
            from . import cuda_backend

            _ACTIVE = cuda_backend
        except Exception as e:  # nvcc / CUDA toolkit missing or compile failed
            import logging

            logging.getLogger(__name__).warning(
                "CUDA kernel backend unavailable (%s); falling back to torch "
                "backend. Set STARLING_KERNEL_BACKEND=torch to silence this.",
                e,
            )
            from . import torch_backend

            _ACTIVE = torch_backend
    else:
        raise ValueError(
            f"unknown kernel backend {name!r}; expected auto/triton/torch/cuda"
        )
    return _ACTIVE


def set_backend(name: str | None) -> None:
    """Override the kernel backend.

    Pass ``"triton"``, ``"torch"``, ``"cuda"``, or ``"auto"``.  Pass ``None`` to
    reset to env-var / auto resolution.  Forces re-resolution on next access.
    """
    global _ACTIVE_NAME, _ACTIVE
    _ACTIVE_NAME = name.lower() if name else None
    _ACTIVE = None


def __getattr__(name: str):
    # Lazy delegation: ``from starling._kernels import fused_rmsnorm`` resolves
    # to the active backend's ``fused_rmsnorm`` at first access.  This avoids
    # importing any backend module at package import time (so the package is
    # importable before the backends exist, and on triton-less machines).
    if name in _PUBLIC:
        return getattr(get_backend(), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
