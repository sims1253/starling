from __future__ import annotations

import json
import socket
import time

from starling.parakeet import gpu_lock


def test_release_does_not_unlink_another_owners_lock(tmp_path, monkeypatch) -> None:
    path = tmp_path / "gpu.lock"
    monkeypatch.setattr(gpu_lock, "LOCK_PATH", path)
    owner = gpu_lock.acquire_gpu_lock(session="first", model="test", wait=False)
    entry = json.loads(path.read_text())
    entry["owner_id"] = "new-owner"
    path.write_text(json.dumps(entry))

    assert gpu_lock.release_gpu_lock(owner) is False
    assert path.exists()


def test_live_local_holder_is_not_stale_just_because_it_is_old() -> None:
    entry = {
        "hostname": socket.gethostname(),
        "pid": __import__("os").getpid(),
        "started_at": time.time() - gpu_lock.STALE_SEC * 10,
    }
    assert gpu_lock._is_stale(entry) is False


def test_dead_local_holder_is_stale_immediately() -> None:
    entry = {
        "hostname": socket.gethostname(),
        "pid": 2**31 - 1,
        "started_at": time.time(),
    }
    assert gpu_lock._is_stale(entry) is True
