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


@pytest.mark.skipif(os.name != "posix", reason="POSIX descriptor inheritance")
@pytest.mark.parametrize("adapter", ["acquire", "context"])
def test_explicit_runner_key_reaches_legacy_adapter_without_nvidia(
    tmp_path, monkeypatch, adapter,
):
    monkeypatch.delenv("STARLING_GPU_LOCK_DISABLE", raising=False)
    monkeypatch.setenv("STARLING_GPU_LOCK_DIR", str(tmp_path))
    # Both runner and child use absolute Python paths. No NVIDIA discovery tool
    # is available, even when the test host happens to have one installed.
    monkeypatch.setenv("PATH", str(tmp_path))
    command = """
import os
from starling.gpu import session
from starling.parakeet import gpu_lock

def no_discovery():
    raise AssertionError('explicit identity must bypass NVIDIA discovery')
session._query_gpu_uuids = no_discovery
"""
    if adapter == "acquire":
        command += """
owner = gpu_lock.acquire_gpu_lock(session='inner', model='test', uuid='metal:test-device',
                                  wait=False)
assert owner == os.environ['STARLING_GPU_LOCK_OWNER']
assert gpu_lock.release_gpu_lock(owner)
"""
    else:
        command += """
with gpu_lock.with_gpu_lock(session='inner', model='test', uuid='metal:test-device'):
    assert gpu_lock._LOCAL.last_owner == os.environ['STARLING_GPU_LOCK_OWNER']
"""
    command += """
# Releasing the borrowed reference must leave the runner's lock held.
for key in ('STARLING_GPU_LOCK_FD', 'STARLING_GPU_LOCK_KEY', 'STARLING_GPU_LOCK_OWNER'):
    os.environ.pop(key, None)
try:
    gpu_lock.acquire_gpu_lock(session='probe', model='test', uuid='metal:test-device',
                              wait=False)
except gpu_lock.GpuLockBusy:
    print('NESTED_OK')
else:
    raise AssertionError('nested release freed the runner lock')
"""
    result = subprocess.run(
        _runner("--uuid", "metal:test-device") + [sys.executable, "-c", command],
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "NESTED_OK"


def test_gpu_tools_import_without_inference_dependencies():
    command = """
import importlib.abc
import sys

class NoInferenceImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'triton', 'transformers'}:
            raise AssertionError(f'GPU lock tools imported {fullname}')

sys.meta_path.insert(0, NoInferenceImports())
import starling.gpu.run
import starling.gpu.session
import starling.parakeet.gpu_lock
"""
    result = subprocess.run(
        [sys.executable, "-c", command], capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr


def test_root_encoder_export_is_preserved():
    command = """
import starling
assert 'FusedEncoder' in dir(starling)
assert 'FusedEncoder' in starling.__all__
from starling import FusedEncoder
from starling.granite.encoder_mega import FusedEncoder as DirectEncoder
assert FusedEncoder is DirectEncoder
assert starling.FusedEncoder is DirectEncoder
try:
    starling.no_such_attribute
except AttributeError:
    pass
else:
    raise AssertionError('unknown root attribute did not raise AttributeError')
"""
    result = subprocess.run(
        [sys.executable, "-c", command], capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(os.name != "posix", reason="POSIX descriptor inheritance")
def test_runner_subprocess_helper_preserves_explicit_device_lock(tmp_path, monkeypatch):
    import signal
    import socket

    from starling.gpu.session import GpuLockBusy, GpuSession

    monkeypatch.delenv("STARLING_GPU_LOCK_DISABLE", raising=False)
    monkeypatch.setenv("STARLING_GPU_LOCK_DIR", str(tmp_path))
    parent_socket, child_socket = socket.socketpair()
    parent_socket.settimeout(15)
    fd = child_socket.fileno()
    child_code = (
        "import os, socket; "
        f"channel = socket.socket(fileno={fd}); "
        "channel.sendall(b'R'); "
        "assert channel.recv(1) == b'X'; "
        "os.close(int(os.environ['STARLING_GPU_LOCK_FD'])); "
        "channel.sendall(b'D')"
    )
    helper_code = (
        "import sys; "
        "from starling.gpu import session; "
        "session._query_gpu_uuids = lambda: []; "
        "from starling.parakeet.gpu_lock import spawn_gpu_subprocess; "
        f"spawn_gpu_subprocess([sys.executable, '-c', {child_code!r}], "
        f"uuid='device-test', pass_fds=({fd},))"
    )
    proc = subprocess.Popen(
        _runner("--uuid", "device-test") + [sys.executable, "-c", helper_code],
        pass_fds=(fd,), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    child_socket.close()
    try:
        assert parent_socket.recv(1) == b"R"
        # The runner/helper exits; the child alone must keep the device locked.
        assert proc.wait(timeout=15) == 0
        with pytest.raises(GpuLockBusy):
            with GpuSession(session="probe", uuid="device-test", wait=False):
                pass
        parent_socket.sendall(b"X")
        assert parent_socket.recv(1) == b"D"
        with GpuSession(session="after", uuid="device-test", wait=False):
            pass
    finally:
        parent_socket.close()
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        proc.wait(timeout=15)
        proc.stderr.close()
