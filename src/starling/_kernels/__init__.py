"""Select fused kernels on first access.

``set_backend`` overrides ``STARLING_KERNEL_BACKEND`` (default: ``auto``).
Auto prefers Triton, then CUDA C++, then torch. CUDA compilation completes
before the backend is exposed; auto falls back to torch if it fails. Explicit
backend requests propagate initialization errors.

Model shims import functions from this package. Set the backend before importing
those shims: changing the selection cannot replace functions already imported.
Benchmarks can import backend modules directly to bypass selection.
"""

from __future__ import annotations

import logging
import os
import threading

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
_SELECTION_LOCK = threading.Lock()


def get_backend_name() -> str:
    """Initialize the backend if needed and return its active name."""
    return get_backend().__name__.rsplit(".", 1)[-1].removesuffix("_backend")


def get_backend():
    """Return the initialized backend, cached until ``set_backend`` resets it."""
    global _ACTIVE
    active = _ACTIVE
    if active is not None:
        return active
    with _SELECTION_LOCK:
        if _ACTIVE is not None:
            return _ACTIVE
        requested = (_ACTIVE_NAME if _ACTIVE_NAME is not None else
                     os.environ.get("STARLING_KERNEL_BACKEND", "auto").strip().lower())
        name = requested
        if name == "auto":
            name = "triton" if have_triton() else "cuda" if have_cuda_compile() else "torch"
        if name == "triton":
            from . import triton_backend

            _ACTIVE = triton_backend
        elif name == "torch":
            from . import torch_backend

            _ACTIVE = torch_backend
        elif name == "cuda":
            try:
                from . import cuda_backend

                cuda_backend._ext()
            except Exception as e:
                if requested != "auto":
                    raise
                logging.getLogger(__name__).warning(
                    "CUDA kernel initialization failed (%s); using torch kernels.", e,
                )
                from . import torch_backend

                _ACTIVE = torch_backend
            else:
                _ACTIVE = cuda_backend
        else:
            raise ValueError(
                f"unknown kernel backend {name!r}; expected auto/triton/torch/cuda"
            )
        return _ACTIVE


def set_backend(name: str | None) -> None:
    """Override the kernel backend.

    Pass ``"triton"``, ``"torch"``, ``"cuda"``, or ``"auto"``.  Pass ``None`` to
    reset to env-var / auto resolution.  Forces re-resolution on next access; already imported functions keep their backend.
    """
    global _ACTIVE_NAME, _ACTIVE
    with _SELECTION_LOCK:
        _ACTIVE_NAME = name.strip().lower() if name is not None else None
        _ACTIVE = None


def __getattr__(name: str):
    # Lazy delegation: ``from starling._kernels import fused_rmsnorm`` resolves
    # to the active backend's ``fused_rmsnorm`` at first access.  This avoids
    # importing any backend module at package import time (so the package is
    # importable before the backends exist, and on triton-less machines).
    if name in _PUBLIC:
        return getattr(get_backend(), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
