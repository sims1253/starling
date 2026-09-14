"""Shared dtypes and availability probes for fused kernel backends."""

from __future__ import annotations

from pathlib import Path

import torch

# Canonical FP8 (e4m3) constants shared by all backends.  Defined here so that
# every backend module and every consumer reads the same source of truth.
FP8_DTYPE = torch.float8_e4m3fn
"""The fp8 e4m3 dtype used for weight-only quantization (``torch.float8_e4m3fn``)."""

FP8_MAX = 448.0
"""Max representable finite value for e4m3 (used as the symmetric-quant ceiling)."""


def have_triton() -> bool:
    """Return whether Triton imports successfully."""
    try:
        import triton
        import triton.language  # noqa: F401
    except Exception:
        return False
    return True


def have_cuda_compile() -> bool:
    """Check for a CUDA device and nvcc; initialization still has to compile."""
    if not torch.cuda.is_available():
        return False
    from torch.utils.cpp_extension import CUDA_HOME, IS_WINDOWS

    return CUDA_HOME is not None and (
        Path(CUDA_HOME) / "bin" / ("nvcc.exe" if IS_WINDOWS else "nvcc")
    ).is_file()
