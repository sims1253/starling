"""CPU tests for the WebSocket smoke client's commit retries."""

import importlib.util
import json
from pathlib import Path
import socket
from types import SimpleNamespace

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "starling_smoke_client", Path(__file__).resolve().parents[1] / "scripts" / "test_server.py"
)
client = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(client)


@pytest.mark.parametrize("reply, expected, commits", [
    ("final", True, 2), ("cancelled", False, 1), ("always_busy", False, 240),
])
def test_commit_retries_busy_with_one_deadline(monkeypatch, reply, expected, commits):
    now = [0.0]
    sent = []
    frames = []

    class Socket:
        closed = False

        def settimeout(self, timeout):
            assert 0 < timeout <= 120

        def recv(self, size):
            if not frames:
                raise socket.timeout()
            chunk = frames[0][:size]
            frames[0] = frames[0][size:]
            if not frames[0]:
                frames.pop(0)
            return chunk

        def close(self):
            self.closed = True

    sock = Socket()

    def send(_sock, opcode, payload):
        if opcode != 0x1:
            return
        sent.append(json.loads(payload))
        if reply == "final" and len(sent) == 2:
            message = {"type": "final", "text": "done"}
        else:
            message = {"type": "error", "message": "cancelled" if reply == "cancelled" else "server busy"}
        data = json.dumps(message).encode()
        frames.append(bytes([0x81, len(data)]) + data)

    monkeypatch.setattr(client, "_ws_connect", lambda *args: sock)
    monkeypatch.setattr(client, "_ws_send", send)
    monkeypatch.setattr(client, "time", SimpleNamespace(
        monotonic=lambda: now[0], perf_counter=lambda: now[0],
        sleep=lambda seconds: now.__setitem__(0, now[0] + seconds),
    ))
    assert client.test_stream("localhost", 1, np.zeros(0), 16000, 100) is expected
    assert sent == [{"type": "commit"}] * commits
    assert now[0] <= 60.1
    assert sock.closed
