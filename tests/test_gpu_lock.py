"""Shim-level tests for ``starling.parakeet.gpu_lock``.

The lock mechanism changed in Task 1 from an ``O_CREAT|O_EXCL`` file (with an
``_is_stale`` age check and file-content owner matching) to a ``fcntl.flock``
delegating to :class:`starling.gpu.session.GpuSession`. These tests assert the
*equivalent invariants* under the new mechanism while exercising the unchanged
public surface (``with_gpu_lock`` / ``acquire_gpu_lock`` / ``release_gpu_lock``
/ ``GpuLockBusy``) that the 19 call sites depend on.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from starling.parakeet import gpu_lock


def test_release_is_owner_safe(tmp_path, monkeypatch) -> None:
    """``release_gpu_lock`` only releases a session you actually hold.

    Replaces the old "does not unlink another owner's lock" test: ownership is
    now tracked by the in-memory ``owner_id`` token (not the file contents), so
    a double release and a bogus owner are safe no-ops that return ``False``.
    """
    monkeypatch.setenv("STARLING_GPU_LOCK_DIR", str(tmp_path))
    monkeypatch.setattr("starling.gpu.session._query_gpu_uuids",
                        lambda: ["GPU-SHIM"])

    owner = gpu_lock.acquire_gpu_lock(session="first", model="test", wait=False)
    assert isinstance(owner, str) and owner
    # Releasing a bogus owner does nothing and does not free the real lock.
    assert gpu_lock.release_gpu_lock("not-my-owner") is False
    # The real lock is still held.
    with pytest.raises(gpu_lock.GpuLockBusy):
        gpu_lock.acquire_gpu_lock(session="second", model="test", wait=False)
    # Releasing the real owner frees it.
    assert gpu_lock.release_gpu_lock(owner) is True
    # A second release of the same owner is a safe no-op.
    assert gpu_lock.release_gpu_lock(owner) is False
    # And the lock is now acquirable again.
    owner2 = gpu_lock.acquire_gpu_lock(session="third", model="test", wait=False)
    gpu_lock.release_gpu_lock(owner2)


def test_live_local_holder_is_never_stolen_regardless_of_age(
        tmp_path, monkeypatch) -> None:
    """A live holder is never stolen however old it gets (flock has no age limit).

    Replaces the old "_is_stale is False for an old live holder" test: under
    ``flock`` a held lock is simply held — there is no age-based staleness, so a
    contender always sees ``GpuLockBusy`` until the holder releases.
    """
    monkeypatch.setenv("STARLING_GPU_LOCK_DIR", str(tmp_path))
    monkeypatch.setattr("starling.gpu.session._query_gpu_uuids",
                        lambda: ["GPU-SHIM"])

    owner = gpu_lock.acquire_gpu_lock(session="holder", model="m", wait=False)
    try:
        time.sleep(0.05)  # let it get "old"
        with pytest.raises(gpu_lock.GpuLockBusy):
            gpu_lock.acquire_gpu_lock(session="contender", model="m", wait=False)
    finally:
        gpu_lock.release_gpu_lock(owner)


def test_dead_holder_releases_the_lock_immediately(tmp_path, monkeypatch) -> None:
    """A holder that dies releases the lock at once (flock closes with the fd).

    Replaces the old "_is_stale is True for a dead holder immediately" test:
    under ``flock`` there is nothing to steal — process death closes the fd and
    the lock is free instantly, with no stale-file cleanup.
    """
    monkeypatch.setattr("starling.gpu.session._query_gpu_uuids",
                        lambda: ["GPU-SHIM"])
    holder = textwrap.dedent(f"""
        import os, sys
        os.environ["STARLING_GPU_LOCK_DIR"] = {str(tmp_path)!r}
        sys.path.insert(0, {str(Path(__file__).resolve().parent.parent / "src")!r})
        from starling.parakeet import gpu_lock
        o = gpu_lock.acquire_gpu_lock(session="dying", model="m", wait=False)
        print("HELD", flush=True)
        # exit immediately -- no release(); simulates a crash / SIGKILL
    """)
    proc = subprocess.run([sys.executable, "-c", holder],
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert "HELD" in proc.stdout

    # The holder is dead and never released -- but flock closed with its fd, so
    # we can acquire immediately with no stale-file stealing.
    owner = gpu_lock.acquire_gpu_lock(session="after", model="m", wait=False)
    gpu_lock.release_gpu_lock(owner)
