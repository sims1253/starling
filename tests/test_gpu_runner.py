"""GPU runner process ownership and explicit opt-out."""

import os
import subprocess
import sys

import pytest

from starling.gpu import run


def _runner(*extra):
    return [sys.executable, "-m", "starling.gpu.run", "--session", "test", *extra, "--"]


def test_disabled_runner_executes_command_without_gpu_discovery(monkeypatch):
    monkeypatch.setenv("STARLING_GPU_LOCK_DISABLE", "1")
    result = subprocess.run(
        _runner() + [sys.executable, "-c", "print('RUNNER_OK')"],
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "RUNNER_OK"


def test_disabled_runner_without_setsid(monkeypatch):
    monkeypatch.setenv("STARLING_GPU_LOCK_DISABLE", "1")
    monkeypatch.delattr(run.os, "setsid", raising=False)
    called = []

    def exec_command(command, args):
        called.append((command, args))
        raise FileNotFoundError(command)

    monkeypatch.setattr(run.os, "execvp", exec_command)
    with pytest.raises(FileNotFoundError):
        run.main(["--session", "test", "--", "missing-command"])
    assert called == [("missing-command", ["missing-command"])]


@pytest.mark.skipif(os.name != "posix", reason="POSIX descriptor inheritance")
def test_nested_runner_keeps_published_descriptor_open(tmp_path, monkeypatch):
    monkeypatch.delenv("STARLING_GPU_LOCK_DISABLE", raising=False)
    monkeypatch.setenv("STARLING_GPU_LOCK_DIR", str(tmp_path))
    command = (
        "from starling.gpu.session import GpuSession; "
        "s = GpuSession(session='inner', uuid='GPU-NESTED', "
        "max_wait_sec=.2, poll_sec=.01); "
        "s.acquire(); print('NESTED_OK'); s.release()"
    )
    runner = _runner("--uuid", "GPU-NESTED")
    result = subprocess.run(
        runner + runner + [sys.executable, "-c", command],
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "NESTED_OK"


@pytest.mark.skipif(os.name != "posix", reason="POSIX flock")
def test_failed_exec_releases_lock(tmp_path, monkeypatch):
    from starling.gpu.session import GpuSession

    monkeypatch.delenv("STARLING_GPU_LOCK_DISABLE", raising=False)
    monkeypatch.setenv("STARLING_GPU_LOCK_DIR", str(tmp_path))
    monkeypatch.setattr(run.os, "setsid", lambda: None)
    # main publishes the FD in this process's environment before exec.
    for key in ("STARLING_GPU_LOCK_FD", "STARLING_GPU_LOCK_KEY", "STARLING_GPU_LOCK_OWNER"):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(FileNotFoundError):
        run.main(["--session", "test", "--uuid", "GPU-FAILED", "--", str(tmp_path / "absent")])
    with GpuSession(session="next", uuid="GPU-FAILED", lock_dir=str(tmp_path), wait=False):
        pass
