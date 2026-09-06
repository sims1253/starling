"""Exercise streaming through FastAPI's ASGI transport without a model or GPU."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from starling import server as S


@pytest.fixture
def server(monkeypatch):
    server = S.StarlingServer(config=S.ServerConfig(
        stream_chunk_seconds=1, stream_overlap_seconds=.25,
        min_chunk_seconds=.1, partial_interval_seconds=0,
    ))
    loads = []

    def load():
        loads.append(True)
        server._loaded = True

    monkeypatch.setattr(server, "load", load)
    server.test_loads = loads
    return server


@pytest.mark.parametrize("chunk_seconds", [0, 1])
def test_websocket_first_request_loads_once_and_commits(server, monkeypatch, chunk_seconds):
    server.config.stream_chunk_seconds = chunk_seconds

    def transcribe(samples, rid, **kwargs):
        assert server.loaded
        return S.TranscribeResult(text="hello", segments=[{"text": "hello"}])

    monkeypatch.setattr(server, "_run_queued_sync", transcribe)
    with TestClient(S.create_app(server=server, load_on_startup=False)) as client:
        assert server.test_loads == []
        with client.websocket_connect('/stream') as ws:
            ws.send_json({"type": "ping"})
            assert ws.receive_json() == {"type": "pong"}
            assert server.test_loads == []
            ws.send_bytes(np.zeros(S.SAMPLE_RATE // 2, dtype=np.int16).tobytes())
            assert ws.receive_json()["type"] == "partial"
            ws.send_json({"type": "commit"})
            final = ws.receive_json()
            assert final["type"] == "final"
            assert final["text"] == "hello"
            assert final["duration_s"] == .5
            ws.send_json({"type": "commit"})
            assert ws.receive_json()["text"] == ""
    assert len(server.test_loads) == 1


def test_websocket_busy_commit_retains_audio(server, monkeypatch):
    monkeypatch.setattr("starling.stream_chunk._FLUSH_BACKOFF_SECONDS", 0)
    busy = True

    def transcribe(samples, rid, **kwargs):
        if busy:
            raise S._Busy()
        return S.TranscribeResult(text="retained audio")

    monkeypatch.setattr(server, "_run_queued_sync", transcribe)
    with TestClient(S.create_app(server=server, load_on_startup=False)) as client:
        with client.websocket_connect('/stream') as ws:
            ws.send_bytes(np.zeros(S.SAMPLE_RATE // 2, dtype=np.int16).tobytes())
            ws.send_json({"type": "commit"})
            assert ws.receive_json() == {"type": "error", "message": "server busy"}
            busy = False
            ws.send_json({"type": "commit"})
            final = ws.receive_json()
            assert final["text"] == "retained audio"
            assert final["duration_s"] == .5
            ws.send_json({"type": "commit"})
            assert ws.receive_json()["duration_s"] == 0


def test_invalid_commands_preserve_connection_and_disconnect_is_clean(server, caplog):
    with TestClient(S.create_app(server=server, load_on_startup=False)) as client:
        with client.websocket_connect('/stream') as ws:
            for command in ['{', 'null', '[]', '1', '"commit"']:
                ws.send_text(command)
                assert ws.receive_json()["type"] == "error"
            ws.send_json({"type": "reset"})
            assert ws.receive_json() == {"type": "reset_ack"}
            ws.send_json({"type": "ping"})
            assert ws.receive_json() == {"type": "pong"}
    assert server.test_loads == []
    assert not [r for r in caplog.records if r.levelname == "ERROR"]


def test_lifespan_owns_eager_load(server):
    app = S.create_app(server=server)
    assert server.test_loads == []
    with TestClient(app):
        assert len(server.test_loads) == 1


def test_cli_missing_server_dependencies_fails_before_model_load(monkeypatch):
    import builtins

    original_import = builtins.__import__

    def import_without_uvicorn(name, *args, **kwargs):
        if name == 'uvicorn':
            raise ModuleNotFoundError('uvicorn')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', import_without_uvicorn)
    monkeypatch.setattr(S.StarlingServer, 'load', lambda self: pytest.fail('loaded model'))
    with pytest.raises(SystemExit, match='uv sync --extra server'):
        S.run([])
    with pytest.raises(SystemExit) as error:
        S.run(['--help'])
    assert error.value.code == 0
