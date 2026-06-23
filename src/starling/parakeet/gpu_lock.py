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
import time
from contextlib import contextmanager
from pathlib import Path

LOCK_PATH = Path(__file__).resolve().parents[3] / ".gpu.lock"
STALE_SEC = 10 * 60  # 10 minutes


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
) -> None:
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
        payload = {
            "session": session,
            "model": model,
            "started_at": now,
            "eta_min": eta_min,
            "note": note,
        }
        if existing is not None and _is_stale(existing, now):
            payload["stole_from"] = existing.get("session")
            payload["stale_started_at"] = existing.get("started_at")
        with os.fdopen(fd, "w") as fh:
            json.dump(payload, fh)
        return


def release_gpu_lock() -> None:
    """Release `.gpu.lock` (best-effort unlink)."""
    try:
        LOCK_PATH.unlink()
    except FileNotFoundError:
        pass


@contextmanager
def with_gpu_lock(*, session: str, model: str, eta_min: int = 5, note: str = ""):
    """Context manager wrapper around acquire/release."""
    acquire_gpu_lock(session=session, model=model, eta_min=eta_min, note=note)
    try:
        yield
    finally:
        release_gpu_lock()
