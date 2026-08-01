"""Kernel backend interface and shared constants for cross-platform fused ops.

Starling's fused decode kernels (RMSNorm, SwiGLU, residual, RoPE, FP8/FP4
dequant-GEMV) were originally hand-written Triton.  Triton has no official
Windows wheels, so this package provides a **uniform interface with pluggable
backends** so the same consumer code runs on:

* **Linux**  -- the ``triton`` backend (existing hand-tuned kernels, max perf).
* **Windows** -- the ``torch`` backend (stock-PyTorch fused ops, no triton dep),
                 or a future ``cuda`` backend (``load_inline`` CUDA C++).

Backend selection happens once (at first access) via the
``STARLING_KERNEL_BACKEND`` env var (``auto``|``triton``|``torch``|``cuda``;
default ``auto`` = triton if importable else torch).  Per-model
``llm_kernels.py`` modules are thin re-export shims over this package
(:mod:`starling._kernels`), so **no consumer code changes** -- ``from . import
llm_kernels as _k`` still works.

The interface
-------------
Each backend module (``triton_backend``, ``torch_backend``, ``cuda_backend``)
must provide every public function listed below with matching signatures.
All operate on bf16 tensors with fp32 internal accumulation and return bf16,
matching the stock PyTorch reference ops to within rounding.

Elementwise (the decode hot path for every LLM-based model):

* :func:`fused_rmsnorm(x, weight, eps) <fused_rmsnorm>` -- RMSNorm, last dim.
* :func:`fused_silu_mul(gate, up) <fused_silu_mul>` -- SwiGLU ``silu(gate)*up``.
* :func:`residual_add(x, y, alpha=1.0) <residual_add>` -- ``x + alpha*y``
  (granite passes ``alpha=residual_multiplier``; moss/qwen3 pass ``alpha=1.0``).
* :func:`fused_rope(q, k, cos, sin) <fused_rope>` -- rotary embedding on Q+K.

Granite GEMM-epilogue fusion (experimental, off by default):

* :func:`compute_rstd(x, eps) <compute_rstd>` -- scalar ``rsqrt(mean(x^2)+eps)``.
* :func:`fused_gemv_normscale(x, w_scaled, rstd) <fused_gemv_normscale>` -- M=1
  GEMV with the rstd scale folded into the epilogue.

FP8 weight-only decode GEMV (opt-in via ``OptFlags.fp8_weights``):

* :func:`quantize_weight_e4m3(weight) <quantize_weight_e4m3>`
* :func:`fp8_linear(x, w_fp8, w_scale) <fp8_linear>`

Runtime autotune control (triton backend only; no-ops elsewhere):

* :func:`set_autotune(enabled) <set_autotune>`, :func:`autotune_enabled() <autotune_enabled>`
"""

from __future__ import annotations

import torch

# Canonical FP8 (e4m3) constants shared by all backends.  Defined here so that
# every backend module and every consumer reads the same source of truth.
FP8_DTYPE = torch.float8_e4m3fn
"""The fp8 e4m3 dtype used for weight-only quantization (``torch.float8_e4m3fn``)."""

FP8_MAX = 448.0
"""Max representable finite value for e4m3 (used as the symmetric-quant ceiling)."""


def have_triton() -> bool:
    """Return ``True`` if the ``triton`` package is importable.

    Cached after the first call.  This is the single gate that decides whether
    the ``auto`` backend selects triton (Linux) or torch (Windows).
    """
    try:
        import triton
        import triton.language  # noqa: F401
    except Exception:
        return False
    return True


def have_cuda_compile() -> bool:
    """Return ``True`` if a CUDA toolkit + GPU are present (for ``load_inline``).

    Used by the ``auto`` backend selector: on Windows (no triton wheel) we prefer
    the CUDA C++ backend (full fused-kernel perf) over the torch fallback when a
    CUDA toolchain is available.  This checks CUDA availability without importing
    the cuda backend (which would JIT-compile on import).
    """
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False
