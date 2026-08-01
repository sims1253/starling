"""Tests for the flock-based :class:`GpuSession` GPU lock (Task 1).

These are 100 % CPU/filesystem tests — no CUDA is required to land the lock.
The multiprocess tests (2 and 6) load ``session.py`` *directly via importlib
file path* so the holder subprocess never imports the ``starling`` package
(which would pull in torch); ``session.py`` is deliberately stdlib-only so it
loads standalone.

The load-bearing invariant under test is #2: a native child spawned with
``pass_fds`` *inherits* the flock file descriptor, so SIGKILLing the Python
parent does NOT release the GPU lock while the native child still holds VRAM.
That closes the orphaned-native-child VRAM hole that the old
``gpu_lock.py`` could not represent.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from starling.gpu.session import GpuLockBusy, GpuSession

_REPO = Path(__file__).resolve().parent.parent
_SESSION_PY = _REPO / "src" / "starling" / "gpu" / "session.py"


def _holder_script(body: str) -> str:
    """Wrap ``body`` in a torch-free loader that imports session.py by path.

    ``body`` is a 0-indented snippet (pass it through ``textwrap.dedent``)
    and may use ``GpuSession``, ``GpuLockBusy``, ``os``, ``sys``, ``time``,
    ``json``. The loader is built without ``textwrap.dedent`` so the body's
    own indentation is never merged with the wrapper's.
    """
    loader = (
        "import importlib.util, sys, os, json, time\n"
        f"_spec = importlib.util.spec_from_file_location('starling_gpu_session', {str(_SESSION_PY)!r})\n"
        "_m = importlib.util.module_from_spec(_spec)\n"
        "sys.modules['starling_gpu_session'] = _m\n"
        "_spec.loader.exec_module(_m)\n"
        "GpuSession = _m.GpuSession\n"
        "GpuLockBusy = _m.GpuLockBusy\n"
    )
    # Triple-quoted snippets conventionally begin with a formatting newline;
    # strip only newlines so the holder's first stdout line is protocol data.
    return loader + body.lstrip("\r\n")


def _poll_until_free(lock_dir: Path, uuid: str, timeout: float = 5.0) -> bool:
    """Return True if a wait=False acquire succeeds within ``timeout`` seconds."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with GpuSession(session="probe", lock_dir=str(lock_dir),
                            uuid=uuid, wait=False, install_signal_handlers=False):
                return True
        except GpuLockBusy:
            time.sleep(0.05)
    return False


# --------------------------------------------------------------------------- #
# 1. flock serializes concurrent acquirers (threads, each own fd)
# --------------------------------------------------------------------------- #
def test_flock_serializes_concurrent_acquirers(tmp_path) -> None:
    """N threads each opening their own fd must be strictly serialized."""
    import threading

    uuid = "test-uuid-serialize"
    intervals: list[tuple[float, float]] = []
    lock = threading.Lock()
    barrier = threading.Barrier(4)

    def worker():
        barrier.wait()  # release all threads together to maximize contention
        with GpuSession(session="w", lock_dir=str(tmp_path), uuid=uuid,
                        max_wait_sec=30, install_signal_handlers=False):
            t1 = time.monotonic()
            time.sleep(0.20)
            t2 = time.monotonic()
        with lock:
            intervals.append((t1, t2))

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not any(t.is_alive() for t in threads), "workers deadlocked"

    # No two held-intervals may overlap: flock grants strictly one at a time.
    intervals.sort()
    for (a_acq, a_rel), (b_acq, _b_rel) in zip(intervals, intervals[1:]):
        assert b_acq >= a_rel - 0.02, (
            f"overlapping acquire: ({a_acq:.3f},{a_rel:.3f}) vs {b_acq:.3f}")


# --------------------------------------------------------------------------- #
# 2. THE invariant: a native child inherits the flock fd, so killing the
#    Python parent does not release the lock while the child lives.
# --------------------------------------------------------------------------- #
def test_native_child_inherits_flock_fd(tmp_path) -> None:
    holder = _holder_script(textwrap.dedent("""
        lock_dir = sys.argv[1]
        uuid = sys.argv[2]
        with GpuSession(session="holder", lock_dir=lock_dir, uuid=uuid,
                        install_signal_handlers=False) as s:
            child = s.spawn(["sleep", "60"])
            print("HOLDER", os.getpid(), flush=True)
            print("CHILD", child.pid, flush=True)
            time.sleep(300)  # we get SIGKILLed by the test
    """))

    proc = subprocess.Popen(
        [sys.executable, "-c", holder, str(tmp_path), "uuid-orphan"],
        stdout=subprocess.PIPE, text=True)
    try:
        holder_pid = child_pid = None
        for _ in range(200):
            line = proc.stdout.readline()
            if not line:
                break
            # Some development environments install startup hooks that emit
            # cosmetic blank lines before user code. They are not protocol.
            if not line.strip():
                continue
            tag, pid = line.split()
            if tag == "HOLDER":
                holder_pid = int(pid)
            elif tag == "CHILD":
                child_pid = int(pid)
            if holder_pid and child_pid:
                break
        assert holder_pid and child_pid, "holder did not report pids"

        # Lock is held by the holder.
        with pytest.raises(GpuLockBusy):
            with GpuSession(session="probe", lock_dir=str(tmp_path),
                            uuid="uuid-orphan", wait=False,
                            install_signal_handlers=False):
                pass

        # Kill the Python parent ONLY. The native sleep child survives and
        # still holds the inherited flock fd -> the lock must REMAIN held.
        os.kill(holder_pid, signal.SIGKILL)
        proc.wait(timeout=10)
        time.sleep(0.2)
        with pytest.raises(GpuLockBusy):
            with GpuSession(session="probe", lock_dir=str(tmp_path),
                            uuid="uuid-orphan", wait=False,
                            install_signal_handlers=False):
                pass

        # Now kill the native child -> its fd closes -> flock releases.
        os.kill(child_pid, signal.SIGKILL)
        try:
            os.waitpid(child_pid, 0)
        except ChildProcessError:
            pass
        assert _poll_until_free(tmp_path, "uuid-orphan", timeout=5.0), \
            "lock not released after the native child died"
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


def test_graceful_parent_release_keeps_inherited_child_lock(tmp_path) -> None:
    """Closing the parent's fd must not explicitly unlock the child's flock."""
    session = GpuSession(session="parent", lock_dir=str(tmp_path),
                         uuid="uuid-graceful", install_signal_handlers=False)
    session.acquire()
    child = session.spawn(["sleep", "60"])
    try:
        session.release()
        with pytest.raises(GpuLockBusy):
            with GpuSession(session="probe", lock_dir=str(tmp_path),
                            uuid="uuid-graceful", wait=False,
                            install_signal_handlers=False):
                pass
    finally:
        child.kill()
        child.wait(timeout=5)
    assert _poll_until_free(tmp_path, "uuid-graceful", timeout=5.0)


# --------------------------------------------------------------------------- #
# 3. UUID keying collapses CUDA_VISIBLE_DEVICES variants on a 1-GPU box
# --------------------------------------------------------------------------- #
def test_uuid_collapses_cvd_variants(monkeypatch) -> None:
    from starling.gpu import session

    # Pretend the box has exactly one GPU.
    monkeypatch.setattr(session, "_query_gpu_uuids", lambda: ["GPU-AAAA"])

    def key_for(cvd):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", cvd) if cvd is not None \
            else monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        return session._resolve_lock_key()

    # All three CVD spellings of "this single GPU" must map to one key/file.
    k0 = key_for("0")
    k_unset = key_for(None)
    k01 = key_for("0,1")
    assert k0 == k_unset == k01 == "GPU-AAAA", (k0, k_unset, k01)

    # And the lock file path is identical.
    monkeypatch.setattr(session, "_query_gpu_uuids", lambda: ["GPU-AAAA"])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    p0 = session._lock_file_path("GPU-AAAA", str(session._default_lock_dir()))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    p1 = session._lock_file_path("GPU-AAAA", str(session._default_lock_dir()))
    assert p0 == p1


def test_lock_key_distinguishes_different_gpus(monkeypatch) -> None:
    from starling.gpu import session
    monkeypatch.setattr(session, "_query_gpu_uuids",
                        lambda: ["GPU-AAAA", "GPU-BBBB"])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    k0 = session._resolve_lock_key()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    k1 = session._resolve_lock_key()
    assert k0 == "GPU-AAAA"
    assert k1 == "GPU-BBBB"
    assert k0 != k1


def test_multi_gpu_session_fails_closed_instead_of_allowing_overlap(
        tmp_path, monkeypatch) -> None:
    """Set locks need per-device acquisition; reject them until implemented."""
    monkeypatch.setattr("starling.gpu.session._query_gpu_uuids",
                        lambda: ["GPU-AAAA", "GPU-BBBB"])
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(RuntimeError, match="exactly one visible GPU"):
        GpuSession(session="multi", lock_dir=str(tmp_path), wait=False).acquire()
    with pytest.raises(RuntimeError, match="exactly one visible GPU"):
        GpuSession(session="multi-explicit", lock_dir=str(tmp_path),
                   uuid="GPU-AAAA,GPU-BBBB", wait=False).acquire()


def test_missing_flock_fails_closed_unless_explicitly_disabled(
        tmp_path, monkeypatch) -> None:
    """Unsupported platforms must never silently allow concurrent benchmarks."""
    from starling.gpu import session
    monkeypatch.setattr(session, "fcntl", None)
    with pytest.raises(RuntimeError, match="requires POSIX fcntl.flock"):
        GpuSession(session="unsupported", lock_dir=str(tmp_path),
                   uuid="GPU-X", wait=False).acquire()

    monkeypatch.setenv("STARLING_GPU_LOCK_DISABLE", "1")
    with GpuSession(session="explicit-opt-out", lock_dir=str(tmp_path),
                    uuid="GPU-X", wait=False):
        pass


# --------------------------------------------------------------------------- #
# 4. token v2 schema roundtrip; stale v:1 tokens are rejected
# --------------------------------------------------------------------------- #
def test_token_v2_schema_roundtrip(tmp_path) -> None:
    with GpuSession(session="rt", model="parakeet", eta_min=7, note="hello",
                    lock_dir=str(tmp_path), uuid="uuid-rt",
                    install_signal_handlers=False) as s:
        token = s.read_token()
    assert token is not None, "no token written under the flock"
    assert token["v"] == 2
    for field in ("session", "model", "pid", "hostname", "uuid", "started_at",
                  "heartbeat_at", "eta_min", "note", "owner_id", "schema"):
        assert field in token, f"token missing {field}"
    assert token["session"] == "rt"
    assert token["model"] == "parakeet"
    assert token["eta_min"] == 7
    assert token["note"] == "hello"
    assert token["uuid"] == "uuid-rt"
    # heartbeat is recent (within a couple seconds of release).
    assert time.time() - token["heartbeat_at"] < 5.0


def test_stale_v1_token_is_rejected(tmp_path) -> None:
    from starling.gpu import session
    bogus = tmp_path / "stale.flock"
    bogus.write_text(json.dumps({"v": 1, "session": "ancient"}))
    parsed = session._parse_token(bogus)
    assert parsed is None, "a v:1 token must not be trusted"


# --------------------------------------------------------------------------- #
# 5. the gpu_lock shim keeps the 19-call-site signature unchanged
# --------------------------------------------------------------------------- #
def test_shim_with_gpu_lock_signature_unchanged(tmp_path, monkeypatch) -> None:
    from starling.parakeet import gpu_lock
    monkeypatch.setenv("STARLING_GPU_LOCK_DIR", str(tmp_path))
    # Force a deterministic single-GPU key so the shim + GpuSession agree.
    monkeypatch.setattr("starling.gpu.session._query_gpu_uuids",
                        lambda: ["GPU-SHIM"])

    # Context-manager form (the common call shape).
    with gpu_lock.with_gpu_lock(session="shim", model="parakeet",
                                eta_min=3, note="cm"):
        # ``with_gpu_lock`` keeps the historical signature (no ``wait`` kwarg,
        # blocks on contention), so probe contention via acquire_gpu_lock.
        with pytest.raises(gpu_lock.GpuLockBusy):
            gpu_lock.acquire_gpu_lock(session="other", model="x",
                                      eta_min=1, note="contender", wait=False)

    # acquire/release form with an owner_id, as some call sites use it.
    owner = gpu_lock.acquire_gpu_lock(
        session="shim2", model="moss", eta_min=2, note="ar", wait=False)
    assert isinstance(owner, str)
    assert gpu_lock.release_gpu_lock(owner) is True
    # releasing an already-released owner is a safe no-op.
    assert gpu_lock.release_gpu_lock(owner) is False


# --------------------------------------------------------------------------- #
# 6. starling-gpu-run holds the lock for the child's whole lifetime
# --------------------------------------------------------------------------- #
def test_runner_exec_under_lock(tmp_path) -> None:
    _RUN_PY = _REPO / "src" / "starling" / "gpu" / "run.py"
    ready = tmp_path / "ready"
    child = (
        f"open({str(ready)!r}, 'w').write('x'); "
        "import time; time.sleep(1.5)")
    runner_loader = textwrap.dedent(f"""
        import importlib.util, sys
        _spec = importlib.util.spec_from_file_location(
            "starling_gpu_run", {str(_RUN_PY)!r})
        _m = importlib.util.module_from_spec(_spec)
        sys.modules["starling_gpu_run"] = _m
        _spec.loader.exec_module(_m)
        _m.main()
    """)
    argv = [
        sys.executable, "-c", runner_loader,
        "--session", "runner", "--eta", "1",
        "--lock-dir", str(tmp_path), "--uuid", "uuid-runner",
        "--", sys.executable, "-c", child,
    ]
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)
    try:
        # wait for the exec'd child to signal readiness
        for _ in range(200):
            if ready.exists():
                break
            if proc.poll() is not None:
                break
            time.sleep(0.05)
        assert ready.exists(), (
            f"runner child never started. stderr=\n{proc.stderr.read()}")

        # While the child runs, the lock is held for its lifetime.
        with pytest.raises(GpuLockBusy):
            with GpuSession(session="probe", lock_dir=str(tmp_path),
                            uuid="uuid-runner", wait=False,
                            install_signal_handlers=False):
                pass

        proc.wait(timeout=15)
        # After the child exits, the inherited fd closes -> lock is free.
        assert _poll_until_free(tmp_path, "uuid-runner", timeout=5.0), \
            "runner did not release the lock when its child exited"
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


def test_runner_allows_nested_legacy_lock_without_deadlock(tmp_path) -> None:
    """A wrapped benchmark may safely call its existing with_gpu_lock internally."""
    _RUN_PY = _REPO / "src" / "starling" / "gpu" / "run.py"
    nested = _holder_script(textwrap.dedent("""
        lock_dir = sys.argv[1]
        uuid = sys.argv[2]
        with GpuSession(session="nested", lock_dir=lock_dir, uuid=uuid,
                        max_wait_sec=1, install_signal_handlers=False):
            print("NESTED_OK", flush=True)
    """))
    runner_loader = textwrap.dedent(f"""
        import importlib.util, sys
        _spec = importlib.util.spec_from_file_location(
            "starling_gpu_run", {str(_RUN_PY)!r})
        _m = importlib.util.module_from_spec(_spec)
        sys.modules["starling_gpu_run"] = _m
        _spec.loader.exec_module(_m)
        raise SystemExit(_m.main())
    """)
    proc = subprocess.run([
        sys.executable, "-c", runner_loader,
        "--session", "outer", "--lock-dir", str(tmp_path),
        "--uuid", "uuid-nested", "--",
        sys.executable, "-c", nested, str(tmp_path), "uuid-nested",
    ], capture_output=True, text=True, timeout=10)
    assert proc.returncode == 0, proc.stderr
    assert "NESTED_OK" in proc.stdout
