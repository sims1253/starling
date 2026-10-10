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


# ---------------------------------------------------------------------------
# Invalid binary audio frames are refused loudly (issue #173), mirroring the
# native frame-validity policy (issue #145): one error frame per episode, the
# take is invalidated until reset, and commit never returns a successful final
# that silently omits audio.
# ---------------------------------------------------------------------------
def _transcribe_hello(samples, rid, **kwargs):
    return S.TranscribeResult(text="hello", segments=[{"text": "hello"}])


@pytest.mark.parametrize(
    "frame, reject_message, reason",
    [
        # RIFF/WAVE magic but truncated before fmt: fails the WAV decoder.
        (b"RIFF\x00\x00\x00\x00WAVEjunk",
         "malformed WAV frame rejected; audio ignored until reset",
         "malformed_wav"),
        # Whole int16 samples plus one dangling byte: a split sample.
        (np.zeros(S.SAMPLE_RATE // 2, dtype=np.int16).tobytes() + b"\x00",
         "odd-length PCM frame rejected (split sample); audio ignored until reset",
         "odd_pcm_length"),
    ],
    ids=["malformed-wav", "odd-pcm"],
)
def test_stream_invalid_frame_rejected_and_take_invalidated(
        server, monkeypatch, frame, reject_message, reason):
    monkeypatch.setattr(server, "_run_queued_sync", _transcribe_hello)
    half = np.zeros(S.SAMPLE_RATE // 2, dtype=np.int16).tobytes()
    with TestClient(S.create_app(server=server, load_on_startup=False)) as client:
        with client.websocket_connect('/stream') as ws:
            ws.send_bytes(frame)
            assert ws.receive_json() == {"type": "error", "message": reject_message}

            # One error frame per episode: a second refused frame stays silent
            # (the ping proves nothing else was queued).
            ws.send_bytes(frame)
            ws.send_json({"type": "ping"})
            assert ws.receive_json() == {"type": "pong"}

            # The invalidated take cannot silently succeed: commit is refused
            # with the machine-readable reason instead of an empty final.
            ws.send_json({"type": "commit"})
            assert ws.receive_json() == {
                "type": "error",
                "message": f"take invalidated ({reason}); reset and resend",
            }

            # reset restores normal operation: valid audio is accepted again.
            ws.send_json({"type": "reset"})
            assert ws.receive_json() == {"type": "reset_ack"}
            ws.send_bytes(half)
            assert ws.receive_json()["type"] == "partial"
            ws.send_json({"type": "commit"})
            final = ws.receive_json()
            assert final["type"] == "final"
            assert final["text"] == "hello"
            assert final["duration_s"] == .5


def test_stream_valid_invalid_valid_sequence(server, monkeypatch):
    """Valid audio before a rejection is kept; audio after it is ignored
    until reset (the issue's regression sequence)."""
    monkeypatch.setattr(server, "_run_queued_sync", _transcribe_hello)
    half = np.zeros(S.SAMPLE_RATE // 2, dtype=np.int16).tobytes()
    with TestClient(S.create_app(server=server, load_on_startup=False)) as client:
        with client.websocket_connect('/stream') as ws:
            ws.send_bytes(half)  # accepted
            assert ws.receive_json()["type"] == "partial"

            ws.send_bytes(b"RIFF\x00\x00\x00\x00WAVEjunk")  # invalidates
            data = ws.receive_json()
            assert data["type"] == "error"
            assert "malformed WAV frame rejected" in data["message"]

            ws.send_bytes(half)  # ignored: the take is already invalidated
            ws.send_json({"type": "ping"})
            assert ws.receive_json() == {"type": "pong"}  # no partial queued

            ws.send_json({"type": "commit"})
            assert ws.receive_json() == {
                "type": "error",
                "message": "take invalidated (malformed_wav); reset and resend",
            }

            ws.send_json({"type": "reset"})
            assert ws.receive_json() == {"type": "reset_ack"}
            ws.send_bytes(half)
            assert ws.receive_json()["type"] == "partial"
            ws.send_json({"type": "commit"})
            final = ws.receive_json()
            assert final["type"] == "final"
            assert final["text"] == "hello"
            assert final["duration_s"] == .5


# ---------------------------------------------------------------------------
# Opt-in stream instrumentation (issue #226): ``trace=1`` attaches the call
# ledger; without it the frames are unchanged.
# ---------------------------------------------------------------------------
def test_stream_trace_is_opt_in(server, monkeypatch):
    monkeypatch.setattr(server, "_run_queued_sync", _transcribe_hello)
    half = np.zeros(S.SAMPLE_RATE // 2, dtype=np.int16).tobytes()
    with TestClient(S.create_app(server=server, load_on_startup=False)) as client:
        with client.websocket_connect('/stream') as ws:
            ws.send_bytes(half)
            assert "trace" not in ws.receive_json()
            ws.send_json({"type": "commit"})
            assert "trace" not in ws.receive_json()


@pytest.mark.parametrize("chunk_seconds", [0, 1])
def test_stream_trace_ledger(server, monkeypatch, chunk_seconds):
    server.config.stream_chunk_seconds = chunk_seconds
    monkeypatch.setattr(server, "_run_queued_sync", _transcribe_hello)
    half = np.zeros(S.SAMPLE_RATE // 2, dtype=np.int16).tobytes()
    with TestClient(S.create_app(server=server, load_on_startup=False)) as client:
        with client.websocket_connect('/stream?trace=1') as ws:
            ws.send_bytes(half)
            partial = ws.receive_json()
            assert partial["trace"]["audio_s"] == .5
            assert partial["trace"]["covered_s"] == .5
            assert partial["trace"]["totals"]["engine_calls"] == 1
            ws.send_bytes(half)  # new final samples after the preview
            assert ws.receive_json()["type"] == "partial"
            ws.send_json({"type": "commit"})
            trace = ws.receive_json()["trace"]
    calls = trace["calls"]
    if chunk_seconds:
        # 1.0 s buffered: the second step finalizes one full window, whose
        # committed text is the partial (no preview right behind a commit);
        # the stop flushes only the 0.25 s past the boundary (advance 0.75 s).
        assert [c["kind"] for c in calls] == ["preview", "window", "flush_tail"]
        assert trace["by_kind"]["window"]["engine_audio_s"] == 1.0
        assert trace["stop"]["path"] == "tail"
        assert trace["stop"]["unfinalized_s"] == .25
        assert trace["stop"]["totals"]["engine_audio_s"] == .25
        assert calls[-1] == {**calls[-1], "kind": "flush_tail",
                             "start_s": .75, "end_s": 1.0, "result": "ok"}
    else:
        # Whole-buffer mode re-transcribes the take: labeled, never "tail".
        assert [c["kind"] for c in calls] == ["preview", "preview", "full_take"]
        assert trace["stop"]["path"] == "full_take"
        assert calls[-1]["kind"] == "full_take"
        assert trace["stop"]["totals"]["engine_audio_s"] == 1.0
    assert trace["covered_s"] == 1.0


# ---------------------------------------------------------------------------
# Preview cadence and coalescing (issue #357).
# ---------------------------------------------------------------------------
def test_stream_per_connection_preview_cadence(server, monkeypatch):
    monkeypatch.setattr(server, "_run_queued_sync", _transcribe_hello)
    half = np.zeros(S.SAMPLE_RATE // 2, dtype=np.int16).tobytes()
    with TestClient(S.create_app(server=server, load_on_startup=False)) as client:
        with client.websocket_connect('/stream?min_partial_seconds=0.75&trace=1') as ws:
            ws.send_bytes(half)  # 0.5 s: below this connection's minimum
            ws.send_json({"type": "ping"})
            assert ws.receive_json() == {"type": "pong"}
            ws.send_bytes(half)
            partial = ws.receive_json()
            assert partial["type"] == "partial"
            assert partial["trace"]["preview"]["min_s"] == .75
            assert partial["trace"]["preview"]["interval_s"] == 0


@pytest.mark.parametrize("query", ["min_partial_seconds=-1", "partial_interval_seconds=nan",
                                   "min_partial_seconds=abc"])
def test_stream_invalid_cadence_is_refused(server, query):
    from starlette.websockets import WebSocketDisconnect

    with TestClient(S.create_app(server=server, load_on_startup=False)) as client:
        with client.websocket_connect(f'/stream?{query}') as ws:
            error = ws.receive_json()
            assert error["type"] == "error"
            assert error["message"].startswith("invalid stream parameter")
            with pytest.raises(WebSocketDisconnect):
                ws.receive_json()


def test_stream_burst_coalesces_previews_and_keeps_audio(server, monkeypatch):
    import time as _time

    server.config.stream_chunk_seconds = 1
    lens = []

    def slow(samples, rid, **kwargs):
        lens.append(len(samples))
        _time.sleep(0.02)
        return S.TranscribeResult(text="w")

    monkeypatch.setattr(server, "_run_queued_sync", slow)
    frame = np.zeros(S.SAMPLE_RATE // 20, dtype=np.int16).tobytes()  # 50 ms
    with TestClient(S.create_app(server=server, load_on_startup=False)) as client:
        with client.websocket_connect('/stream?trace=1') as ws:
            for _ in range(60):
                ws.send_bytes(frame)
            ws.send_json({"type": "commit"})
            while (msg := ws.receive_json())["type"] != "final":
                assert msg["type"] == "partial"
    previews = [n for n in lens if n < S.SAMPLE_RATE]
    assert len(previews) < 30
    assert msg["duration_s"] == 3.0
    assert msg["trace"]["covered_s"] == 3.0
    assert msg["trace"]["audio_s"] == 3.0


@pytest.mark.parametrize("message,reply", [({"type": "ping"}, "pong"),
                                           ({"type": "bogus"}, "error")])
def test_stream_ping_does_not_coalesce_the_preview(server, monkeypatch, message, reply):
    # A ping (or an unrecognized message) queued while a window commits
    # carries no audio: the preview of the newest tail still runs, and the
    # reply follows it.
    import time as _time

    server.config.stream_chunk_seconds = 1
    lens = []

    def slow_window(samples, rid, **kwargs):
        lens.append(len(samples))
        if len(samples) == S.SAMPLE_RATE:
            _time.sleep(0.3)  # the ping arrives during the window commit
        return S.TranscribeResult(text="w")

    monkeypatch.setattr(server, "_run_queued_sync", slow_window)
    audio = np.zeros(S.SAMPLE_RATE * 11 // 10, dtype=np.int16).tobytes()  # 1.1 s
    with TestClient(S.create_app(server=server, load_on_startup=False)) as client:
        with client.websocket_connect('/stream?min_partial_seconds=0.5'
                                      '&partial_interval_seconds=0') as ws:
            ws.send_bytes(audio)
            _time.sleep(0.1)
            ws.send_json(message)
            types = []
            while (msg := ws.receive_json())["type"] != reply:
                types.append(msg["type"])
    assert "partial" in types
    assert [n for n in lens if n < S.SAMPLE_RATE] != []


def test_stream_tiny_queue_bounds_keep_every_frame(server, monkeypatch):
    # Queue and byte budgets smaller than the burst, with a slow engine: the
    # receiver waits for space (frames blocked on a full queue keep their
    # byte reservation) and the final still covers every frame.
    import time as _time

    server.config.stream_chunk_seconds = 1
    monkeypatch.setattr(S, "STREAM_QUEUE_MAX_FRAMES", 2)
    frame = np.zeros(S.SAMPLE_RATE // 10, dtype=np.int16).tobytes()  # 100 ms
    monkeypatch.setattr(S, "STREAM_QUEUE_MAX_BYTES", 3 * len(frame))

    def slow(samples, rid, **kwargs):
        _time.sleep(0.01)
        return S.TranscribeResult(text="w")

    monkeypatch.setattr(server, "_run_queued_sync", slow)
    with TestClient(S.create_app(server=server, load_on_startup=False)) as client:
        with client.websocket_connect('/stream') as ws:
            for _ in range(40):
                ws.send_bytes(frame)
            ws.send_json({"type": "commit"})
            while (msg := ws.receive_json())["type"] != "final":
                assert msg["type"] == "partial"
    assert msg["duration_s"] == 4.0


def _abortable_preview(started):
    """A fake transcribe whose previews run until their cancel flag fires
    (polled like a backend checkpoint), 10 s otherwise; windows and flushes
    answer at once with their length."""
    import time as _time

    def transcribe(samples, rid, *, streaming=False, cancel=None):
        if cancel is not None:  # only previews carry a preempt flag
            started.set()
            t0 = _time.monotonic()
            while _time.monotonic() - t0 < 10:
                if cancel.is_set():
                    raise S._Cancelled()
                _time.sleep(0.001)
            return S.TranscribeResult(text="stale preview")
        return S.TranscribeResult(text=f"w{len(samples)}")

    return transcribe


def test_stream_stop_preempts_running_preview(server, monkeypatch):
    # Stop after newer audio while a preview runs: the preview is cancelled
    # and the flush covers all audio without waiting for it (issue #357).
    import threading
    import time as _time

    started = threading.Event()
    monkeypatch.setattr(server, "_run_queued_sync", _abortable_preview(started))
    with TestClient(S.create_app(server=server, load_on_startup=False)) as client:
        with client.websocket_connect('/stream?trace=1') as ws:
            ws.send_bytes(np.zeros(9600, dtype=np.int16).tobytes())
            assert started.wait(5)
            t_stop = _time.monotonic()
            ws.send_bytes(np.zeros(3200, dtype=np.int16).tobytes())
            ws.send_json({"type": "commit"})
            while (msg := ws.receive_json())["type"] != "final":
                assert "stale" not in msg.get("text", "")
            stop_s = _time.monotonic() - t_stop
    assert stop_s < 2.0
    assert msg["text"] == "w12800"
    assert msg["duration_s"] == .8
    results = [c["result"] for c in msg["trace"]["calls"]]
    assert "preempted" in results
    assert msg["trace"]["totals"]["preempted"] == 1
    assert msg["trace"]["stop"]["path"] == "tail"


def test_stream_commit_without_new_audio_preempts_preview(server, monkeypatch):
    # Stop with no audio after the preview started: this server has no
    # exact-tail reuse (the native one lets the preview finish and reuses
    # it), so the preview is cancelled and the flush decodes the tail once.
    import threading

    started = threading.Event()
    release = threading.Event()
    calls = []

    def transcribe(samples, rid, *, streaming=False, cancel=None):
        calls.append(("preview" if cancel is not None else "other", len(samples)))
        if cancel is not None:
            started.set()
            while not release.wait(0.001):
                if cancel.is_set():
                    raise S._Cancelled()
        return S.TranscribeResult(text=f"w{len(samples)}")

    monkeypatch.setattr(server, "_run_queued_sync", transcribe)
    with TestClient(S.create_app(server=server, load_on_startup=False)) as client:
        with client.websocket_connect('/stream?trace=1') as ws:
            ws.send_bytes(np.zeros(9600, dtype=np.int16).tobytes())
            assert started.wait(5)
            ws.send_json({"type": "commit"})
            import time as _time
            _time.sleep(0.05)
            release.set()
            while (msg := ws.receive_json())["type"] != "final":
                pass
    assert msg["text"] == "w9600"
    assert msg["trace"]["totals"]["preempted"] == 1
    assert calls == [("preview", 9600), ("other", 9600)]
    assert msg["trace"]["stop"]["path"] == "tail"


def test_stream_reset_preempts_running_preview(server, monkeypatch):
    import threading
    import time as _time

    started = threading.Event()
    monkeypatch.setattr(server, "_run_queued_sync", _abortable_preview(started))
    with TestClient(S.create_app(server=server, load_on_startup=False)) as client:
        with client.websocket_connect('/stream') as ws:
            ws.send_bytes(np.zeros(9600, dtype=np.int16).tobytes())
            assert started.wait(5)
            t0 = _time.monotonic()
            ws.send_json({"type": "reset"})
            while (msg := ws.receive_json())["type"] != "reset_ack":
                assert "stale" not in msg.get("text", "")
    assert _time.monotonic() - t0 < 2.0


def test_stream_empty_commit_reports_committed_stop(server):
    with TestClient(S.create_app(server=server, load_on_startup=False)) as client:
        with client.websocket_connect('/stream?trace=1') as ws:
            ws.send_json({"type": "commit"})
            final = ws.receive_json()
    assert final["type"] == "final"
    stop = final["trace"]["stop"]
    assert stop["path"] == "committed"
    assert stop["t0_ms"] == 0.0 and stop["t1_ms"] == 0.0


def test_frame_samples_follow_the_decoded_audio():
    import io
    import wave

    def wav(rate, channels, width, frames):
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(channels)
            w.setsampwidth(width)
            w.setframerate(rate)
            w.writeframes(b"\x00" * frames * channels * width)
        return buf.getvalue()

    assert S._frame_samples(b"\x00" * 3200) == 1600                # raw PCM16
    assert S._frame_samples(wav(16000, 1, 1, 6400)) == 6400        # 0.4 s mono 8-bit
    assert S._frame_samples(wav(16000, 2, 2, 3200)) == 3200        # 0.2 s stereo PCM16
    assert S._frame_samples(wav(8000, 1, 2, 1600)) == 3200         # resampled to 16 kHz
    assert S._frame_samples(b"RIFF\x00\x00\x00\x00WAVEjunk") == 0  # refused


def test_preempt_event_latches_its_predicate():
    state = {"go": False}
    ev = S._PreemptEvent(lambda: state["go"])
    assert not ev.is_set() and not ev.wait(0.01) and not ev.fired
    state["go"] = True
    assert ev.is_set() and ev.fired
    state["go"] = False
    assert ev.is_set()  # latched, like a cancel


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
