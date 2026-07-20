"""Backward-compatible shim over the flock-based :class:`GpuSession`.

Historically this module *was* the GPU lock: an ``O_CREAT|O_EXCL`` file keyed by
the ``CUDA_VISIBLE_DEVICES`` string, recording only the Python parent PID. That
design could not represent native (subprocess) GPU holders, so killing the
Python parent released the lock while the native child still held VRAM — see the
hazard analysis in :mod:`starling.gpu.session`.

The real lock now lives in :mod:`starling.gpu.session` (``fcntl.flock`` +
fd-inheritance + GPU-UUID keying). This module keeps the **public signatures**
(``with_gpu_lock`` / ``acquire_gpu_lock`` / ``release_gpu_lock`` /
``GpuLockBusy``) so all existing call sites are unchanged, and delegates to
``GpuSession`` under the hood. ``GpuLockBusy`` is the *same class* as
``starling.gpu.session.GpuLockBusy`` so ``except GpuLockBusy`` works everywhere.

Usage (unchanged)::

    from starling.parakeet.gpu_lock import with_gpu_lock
    with with_gpu_lock(session="bench", model="parakeet-tdt-0.6b-v3",
                       eta_min=5, note="decode benchmark"):
        ...  # timed benchmark here
"""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from pathlib import Path

# Same exception class as the real lock (isinstance-equivalent across modules).
from starling.gpu.session import GpuLockBusy, GpuSession

# --- compat attributes (kept so external readers / old tests still resolve) --
# Computed CHEAPLY (no nvidia-smi) from the CVD string; the REAL lock key (the
# GPU-UUID set) is resolved lazily inside GpuSession.acquire. Kept so any code
# that displays or reads ``gpu_lock.LOCK_PATH`` still resolves a sane path.
_cvdfallback = (
    os.environ.get("CUDA_VISIBLE_DEVICES", "default") or "default"
).replace("/", "_").replace(",", "-")
LOCK_PATH = Path(
    os.environ.get(
        "STARLING_GPU_LOCK",
        str(Path("/tmp") / f"starling-gpu-{_cvdfallback}.flock"),
    )
)
# Kept as a constant for any code that reads it; the flock releases on death so
# age-based staleness no longer applies (heartbeat staleness lives in GpuSession).
STALE_SEC = 10 * 60

_LOCAL = threading.local()


def _sessions() -> dict:
    sess = getattr(_LOCAL, "sessions", None)
    if sess is None:
        sess = {}
        _LOCAL.sessions = sess
    return sess


def acquire_gpu_lock(
    *,
    session: str,
    model: str,
    eta_min: int = 5,
    note: str = "",
    wait: bool = True,
    poll_sec: float = 0.2,
    max_wait_sec: float = 600.0,
) -> str:
    """Acquire the GPU lock; return an opaque ``owner_id`` token.

    Delegates to :class:`GpuSession`. Raises :class:`GpuLockBusy` if a fresh
    lock is held and ``wait=False``. The returned token must be passed to
    :func:`release_gpu_lock`.
    """
    gs = GpuSession(
        session=session, model=model, eta_min=eta_min, note=note,
        wait=wait, poll_sec=poll_sec, max_wait_sec=max_wait_sec,
        install_signal_handlers=False,
    )
    gs.acquire()  # raises GpuLockBusy / GpuLockTimeout on contention
    owner_id = gs.owner_id or ""
    _sessions()[owner_id] = gs
    _LOCAL.last_owner = owner_id
    return owner_id


def release_gpu_lock(owner_id: str | None = None) -> bool:
    """Release the lock for ``owner_id`` (or the last acquirer).

    Returns ``True`` if a matching session was found and released, ``False``
    otherwise (already released / unknown owner) — a safe no-op.
    """
    sess = _sessions()
    key = owner_id or getattr(_LOCAL, "last_owner", None)
    gs = sess.pop(key, None) if key else None
    if gs is None:
        return False
    try:
        gs.release()
    finally:
        if getattr(_LOCAL, "last_owner", None) == key:
            _LOCAL.last_owner = None
    return True


@contextmanager
def with_gpu_lock(*, session: str, model: str, eta_min: int = 5, note: str = ""):
    """Context-manager wrapper around :func:`acquire_gpu_lock`/`release_gpu_lock`."""
    owner_id = acquire_gpu_lock(
        session=session, model=model, eta_min=eta_min, note=note)
    try:
        yield
    finally:
        release_gpu_lock(owner_id)
