"""Guarded ``torch.compile`` wrapper for cross-platform safety.

``torch.compile`` with the Inductor backend emits Triton kernels on GPU.  When
triton is unavailable (Windows has no official wheels) compilation either fails
outright or silently produces a broken/degraded graph.  This module provides
:func:`torch_compile`, a drop-in wrapper that returns the function **unchanged**
when triton is absent, so call sites can use it unconditionally::

    from starling._kernels._compile import torch_compile

    self._step = torch_compile(self._step, mode="max-autotune-no-cudagraphs")

All existing ``torch.compile`` call sites in Starling are opt-in (gated behind
``compile_decode`` / encoder ``"compiled"`` mode), so on Windows they simply
become eager no-ops -- correctness is unaffected, only the compile speedup is
skipped.
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar

from .base import have_triton

F = TypeVar("F", bound=Callable[..., Any])


def torch_compile(fn: F, mode: str | None = None, **kwargs: Any) -> F:
    """``torch.compile(fn, ...)`` when triton is available, else ``fn`` unchanged."""
    if not have_triton():
        return fn
    import torch

    return torch.compile(fn, mode=mode, **kwargs)  # type: ignore[return-value]
