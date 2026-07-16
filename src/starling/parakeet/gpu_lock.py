"""File-based GPU lock for benchmark isolation.

Concurrent timed benchmarks on the same GPU corrupt each other's numbers.
This module provides a `.gpu.lock` file-protocol so only one benchmark runs
at a time: acquire before a timed region, release after. Stale locks
(older than ``STALE_SEC``) are considered crashed and may be stolen.

Usage:
    from starling.parakeet.gpu_lock import with_gpu_lock
    with with_gpu_lock(session="bench", model="parakeet-tdt-0.6b-v3",
                       eta_min=5, note="decode benchmark"):
        ...  # timed benchmark here
"""

from __future__ import annotations

import json
import os
import socket
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

_GPU_KEY = os.environ.get("CUDA_VISIBLE_DEVICES", "default").replace("/", "_").replace(",", "-")
LOCK_PATH = Path(
    os.environ.get(
        "STARLING_GPU_LOCK",
        str(Path(tempfile.gettempdir()) / f"starling-gpu-{_GPU_KEY}.lock"),
    )
)
STALE_SEC = 10 * 60  # 10 minutes
_LOCAL = threading.local()


class GpuLockBusy(RuntimeError):
    """Raised when a fresh lock is held by another session and wait=False."""


def _read_lock() -> dict | None:
    if not LOCK_PATH.exists():
        return None
    try:
        return json.loads(LOCK_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _is_stale(entry: dict | None, now: float | None = None) -> bool:
    if entry is None:
        return True
    # A local PID is authoritative: crashed holders are reclaimable
    # immediately, while long-running live holders are never stolen by age.
    if entry.get("hostname") == socket.gethostname() and "pid" in entry:
        try:
            os.kill(int(entry["pid"]), 0)
        except (OSError, TypeError, ValueError):
            return True
        return False

    now = time.time() if now is None else now
    started = entry.get("started_at", 0)
    try:
        return (now - float(started)) >= STALE_SEC
    except (TypeError, ValueError):
        return True


def acquire_gpu_lock(
    *,
    session: str,
    model: str,
    eta_min: int = 5,
    note: str = "",
    wait: bool = True,
    poll_sec: float = 5.0,
    max_wait_sec: float = 600.0,
) -> str:
    """Acquire `.gpu.lock`. If a fresh lock exists, wait (or raise if wait=False).

    Stale locks are stolen; the takeover is noted so the previous holder can see it.
    Uses an atomic O_CREAT|O_EXCL create to avoid a TOCTOU race between sessions.
    """
    deadline = time.time() + max_wait_sec
    while True:
        now = time.time()
        existing = _read_lock()
        if existing is not None and not _is_stale(existing, now):
            # fresh lock held by someone else
            if not wait:
                raise GpuLockBusy(f"GPU locked by {existing.get('session')!r}")
            if now > deadline:
                raise TimeoutError(
                    f"timed out after {max_wait_sec}s waiting for GPU lock held by "
                    f"{existing.get('session')!r}"
                )
            time.sleep(poll_sec)
            continue
        # try to create atomically; if a stale lock file is on disk, remove it first
        if existing is not None and _is_stale(existing, now):
            try:
                LOCK_PATH.unlink()
            except FileNotFoundError:
                pass
        try:
            fd = os.open(str(LOCK_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            # someone created it between our unlink and create; loop and re-check
            continue
        owner_id = uuid.uuid4().hex
        payload = {
            "session": session,
            "model": model,
            "started_at": now,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "owner_id": owner_id,
            "eta_min": eta_min,
            "note": note,
        }
        if existing is not None and _is_stale(existing, now):
            payload["stole_from"] = existing.get("session")
            payload["stale_started_at"] = existing.get("started_at")
        with os.fdopen(fd, "w") as fh:
            json.dump(payload, fh)
        _LOCAL.owner_id = owner_id
        return owner_id


def release_gpu_lock(owner_id: str | None = None) -> bool:
    """Release the lock only when it is still owned by this acquisition."""
    expected = owner_id or getattr(_LOCAL, "owner_id", None)
    entry = _read_lock()
    if not expected or entry is None or entry.get("owner_id") != expected:
        return False
    try:
        LOCK_PATH.unlink()
    except FileNotFoundError:
        return False
    if getattr(_LOCAL, "owner_id", None) == expected:
        _LOCAL.owner_id = None
    return True


@contextmanager
def with_gpu_lock(*, session: str, model: str, eta_min: int = 5, note: str = ""):
    """Context manager wrapper around acquire/release."""
    owner_id = acquire_gpu_lock(session=session, model=model, eta_min=eta_min, note=note)
    try:
        yield
    finally:
        release_gpu_lock(owner_id)
