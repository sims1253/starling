"""Starling GPU isolation primitives (Task 1).

The flock-based, native-child-aware GPU lock lives in :mod:`starling.gpu.session`.
This package is importable on its own (no torch); ``starling.gpu.session`` is
stdlib-only so it also loads standalone via ``importlib`` for hermetic tests.
"""

from .session import (
    GpuLockBusy,
    GpuLockTimeout,
    GpuSession,
    _parse_token,
    _query_gpu_uuids,
    _resolve_lock_key,
)

__all__ = [
    "GpuSession",
    "GpuLockBusy",
    "GpuLockTimeout",
]
