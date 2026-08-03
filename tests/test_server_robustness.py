"""CPU-only regression tests for recently-fixed robustness bugs (review #12).

Each test exercises a pure / faked code path in :mod:`starling.server`,
:mod:`starling.stream_chunk`, or :mod:`starling.flags`. No GPU, no loaded
model, no network, no live server socket. These guard against regressions of
the specific fixes called out in the review:

* A. malformed audio -> HTTP 400 (not a 500 / dead socket)
* B. multipart parsing selects the audio field by name / filename / content-type
* C. WebSocket frame cap + client-mask enforcement
* D. streaming buffer is bounded (committed prefix trimmed)
* E. ``ChunkStreamer.flush`` retries the tail on a busy transcriber (bounded)
* F. ``flags()`` applies overrides, restores defaults, preserves all fields
* G. ``StarlingServer.warmup`` dedupes concurrent calls
"""

from __future__ import annotations

import io
import os
import struct
import sys
import threading

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, "..", "src"))
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from starling.server import (  # noqa: E402
    MAX_WS_FRAME_BYTES,
    SAMPLE_RATE,
    STREAM_TRIM_MIN_SAMPLES,
    ServerConfig,
    StarlingServer,
    StreamSession,
    TranscribeResult,
    _extract_multipart_payload,
    _transcribe_payload_sync,
    _ws_read_frame,
)
from starling.stream_chunk import (  # noqa: E402
    ChunkStreamer,
    _FLUSH_TAIL_MAX_RETRIES,
)


# ---------------------------------------------------------------------------
# small helpers shared across tests
# ---------------------------------------------------------------------------
def _wav_bytes(samples: np.ndarray, sampwidth: int = 2, framerate: int = SAMPLE_RATE,
               n_channels: int = 1) -> bytes:
    """Serialize a 1-D float array to a real, parseable WAV byte string."""
    buf = io.BytesIO()
    import wave

    with wave.open(buf, "wb") as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(framerate)
        if sampwidth == 2:
            raw = (np.clip(samples, -1.0, 1.0) * 32768.0).astype("<i2").tobytes()
        elif sampwidth == 4:
            raw = (np.clip(samples, -1.0, 1.0) * 2147483648.0).astype("<i4").tobytes()
        elif sampwidth == 1:
            raw = ((np.clip(samples, -1.0, 1.0) * 128.0) + 128.0).astype(np.uint8).tobytes()
        else:
            raise ValueError(sampwidth)
        wf.writeframes(raw)
    return buf.getvalue()


class _FakeServer:
    """Minimal stand-in for StarlingServer for ``_transcribe_payload_sync``.

    Only the surface used by the transport adapter is faked; the decode path
    runs against the *real* ``_wav_bytes_to_float32`` to exercise the fix.
    """

    def __init__(self, exc=None, *, result_text: str = "ok") -> None:
        # ``exc``: if set, ``transcribe_bytes_sync`` raises it (so the decode
        # is simulated as failing). If None, a canned TranscribeResult is
        # returned (meaning the bytes decoded cleanly).
        self._exc = exc
        self._result_text = result_text
        # a real queue_depth()/registry is not needed for the 400 path
        self._n_waiters = 0
        self._lock = threading.Lock()

    def transcribe_bytes_sync(self, wav_bytes, request_id=None):  # noqa: ANN001
        if self._exc is not None:
            raise self._exc
        return TranscribeResult(text=self._result_text)

    def queue_depth(self) -> int:
        return 0


# ---------------------------------------------------------------------------
# A. Malformed input -> 400
# ---------------------------------------------------------------------------
def test_transcribe_payload_returns_400_on_value_error() -> None:
    """A ValueError raised while decoding WAV maps to a 400, not a 500."""
    server = _FakeServer(exc=ValueError("bad sample width"))

    status, response = _transcribe_payload_sync(
        server, b"definitely not a wav body", "req-1"
    )

    assert status == 400
    assert "error" in response
    assert response["text"] == ""


def test_transcribe_payload_returns_400_on_wave_error() -> None:
    """A wave.Error raised while decoding WAV maps to a 400."""
    import wave

    server = _FakeServer(exc=wave.Error("truncated header"))

    status, response = _transcribe_payload_sync(
        server, b"RIFF\x00\x00\x00\x00WAVEjunk", "req-2"
    )

    assert status == 400
    assert "error" in response


def test_transcribe_payload_passes_through_200_on_clean_wav() -> None:
    """Sanity check: a clean decode still returns 200 with the transcript."""
    server = _FakeServer(result_text="hello world")
    payload = _wav_bytes(np.zeros(SAMPLE_RATE, dtype=np.float32))

    status, response = _transcribe_payload_sync(server, payload, "req-3")

    assert status == 200
    assert response["text"] == "hello world"
    assert response["request_id"] == "req-3"


# ---------------------------------------------------------------------------
# B. Multipart parsing respects field names
# ---------------------------------------------------------------------------
def _mp_part(headers: list[tuple[str, str]], payload: bytes) -> bytes:
    """Build one multipart part body (headers + blank line + payload)."""
    head = "".join(f"{k}: {v}\r\n" for k, v in headers)
    return head.encode() + b"\r\n" + payload


def test_multipart_picks_named_audio_field_over_preceding_text() -> None:
    """A leading ``metadata`` text field must NOT shadow the ``audio`` part."""
    boundary = "----boundary"
    meta = _mp_part(
        [("Content-Disposition", 'form-data; name="metadata"')],
        b'{"lang":"en"}',
    )
    audio = _mp_part(
        [
            ("Content-Disposition", 'form-data; name="audio"; filename="x.wav"'),
            ("Content-Type", "application/octet-stream"),
        ],
        b"AUDIO-BYTES",
    )
    body = b"--" + boundary.encode() + b"\r\n" + meta + b"\r\n" \
        + b"--" + boundary.encode() + b"\r\n" + audio + b"\r\n" \
        + b"--" + boundary.encode() + b"--\r\n"
    ctype = f'multipart/form-data; boundary="{boundary}"'

    out = _extract_multipart_payload(body, ctype)

    assert out == b"AUDIO-BYTES"


def test_multipart_picks_file_field_by_name() -> None:
    """A single part named ``file`` with a filename is selected by name."""
    boundary = "----b"
    part = _mp_part(
        [("Content-Disposition", 'form-data; name="file"; filename="a.wav"')],
        b"FILE-PAYLOAD",
    )
    body = b"--" + boundary.encode() + b"\r\n" + part + b"\r\n" \
        + b"--" + boundary.encode() + b"--\r\n"
    ctype = f"multipart/form-data; boundary={boundary}"

    out = _extract_multipart_payload(body, ctype)

    assert out == b"FILE-PAYLOAD"


def test_multipart_selects_part_by_audio_content_type() -> None:
    """A part whose Content-Type is ``audio/wav`` is preferred over plain text."""
    boundary = "----b"
    text_part = _mp_part(
        [("Content-Disposition", 'form-data; name="note"')],
        b"just a note",
    )
    audio_part = _mp_part(
        [
            ("Content-Disposition", 'form-data; name="clip"'),
            ("Content-Type", "audio/wav"),
        ],
        b"WAVDATA",
    )
    body = b"--" + boundary.encode() + b"\r\n" + text_part + b"\r\n" \
        + b"--" + boundary.encode() + b"\r\n" + audio_part + b"\r\n" \
        + b"--" + boundary.encode() + b"--\r\n"
    ctype = f"multipart/form-data; boundary={boundary}"

    out = _extract_multipart_payload(body, ctype)

    assert out == b"WAVDATA"


def test_multipart_without_boundary_returns_raw_body() -> None:
    """No boundary in Content-Type -> the raw body is returned (raw-WAV path)."""
    raw = _wav_bytes(np.zeros(8, dtype=np.float32))

    out = _extract_multipart_payload(raw, "application/octet-stream")

    assert out == raw


# ---------------------------------------------------------------------------
# C. WS frame cap + mask validation
# ---------------------------------------------------------------------------
def _ws_frame(*, payload: bytes, opcode: int = 0x1, mask: bool = True,
              mask_key: bytes = b"\x01\x02\x03\x04") -> bytes:
    """Build one RFC 6455 client->server frame's wire bytes.

    All client frames are masked per RFC 6455 §5.1 unless ``mask=False``.
    """
    b0 = 0x80 | (opcode & 0x0F)  # FIN set
    n = len(payload)
    out = bytearray()
    if n < 126:
        b1 = (0x80 if mask else 0x00) | n
        out += struct.pack(">BB", b0, b1)
    elif n < 65536:
        b1 = (0x80 if mask else 0x00) | 126
        out += struct.pack(">BBH", b0, b1, n)
    else:
        b1 = (0x80 if mask else 0x00) | 127
        out += struct.pack(">BBQ", b0, b1, n)
    if mask:
        assert len(mask_key) == 4
        out += mask_key
        out += bytes(b ^ mask_key[i % 4] for i, b in enumerate(payload))
    else:
        out += payload
    return bytes(out)


def _rfile(data: bytes):
    """A file-like rfile backed by BytesIO (has ``.read(n)``)."""
    return io.BufferedReader(io.BytesIO(data))


def test_ws_read_frame_valid_masked_text_frame() -> None:
    payload = b"hello stream"
    frame = _ws_frame(payload=payload, opcode=0x1)

    opcode, out = _ws_read_frame(_rfile(frame))

    assert opcode == 0x1
    assert out == payload


def test_ws_read_frame_rejects_unmasked_client_frame() -> None:
    """An unmasked client->server frame is a protocol violation -> ConnectionError."""
    frame = _ws_frame(payload=b"sneaky", opcode=0x1, mask=False)

    with pytest.raises(ConnectionError, match="unmasked"):
        _ws_read_frame(_rfile(frame))


def test_ws_read_frame_rejects_oversized_frame_header() -> None:
    """A frame claiming a payload > MAX_WS_FRAME_BYTES raises before reading body."""
    huge = MAX_WS_FRAME_BYTES + 1
    # 127-length extended header with a gigantic declared length, masked.
    b0 = 0x80 | 0x1  # FIN + text
    b1 = 0x80 | 127   # masked + 64-bit length
    header = struct.pack(">BBQ", b0, b1, huge) + b"\x01\x02\x03\x04"

    with pytest.raises(ValueError, match="too large"):
        _ws_read_frame(_rfile(header))


def test_ws_read_frame_allows_max_size_header() -> None:
    """A frame declaring exactly MAX_WS_FRAME_BYTES passes the size check.

    We only feed the header (no real payload) so the reader will then hit EOF
    reading the body; the point is the size ValueError must NOT fire.
    """
    b0 = 0x80 | 0x1
    b1 = 0x80 | 127
    header = struct.pack(">BBQ", b0, b1, MAX_WS_FRAME_BYTES) + b"\x01\x02\x03\x04"

    with pytest.raises(ConnectionError, match="closed mid-frame"):
        _ws_read_frame(_rfile(header))
    # If the size check had fired we'd have seen ValueError instead; reaching
    # the EOF "closed mid-frame" ConnectionError proves the cap allowed it.


# ---------------------------------------------------------------------------
# D. Streaming buffer is bounded
# ---------------------------------------------------------------------------
class _FakeChunker:
    """A controllable stand-in for ChunkStreamer used by StreamSession trim tests.

    Exposes the two attributes StreamSession._maybe_trim_samples reads:
    ``boundary`` (read/written) and nothing else needed for the trim path.
    """

    def __init__(self, boundary: int = 0) -> None:
        self.boundary = boundary


def _stream_session_with_chunker(chunker) -> StreamSession:
    """Build a StreamSession whose chunker is replaced by ``chunker``.

    We bypass the real ChunkStreamer construction (and the GPU-touching
    transcribe path) by constructing the session against a config with
    streaming disabled, then swapping in our fake chunker.
    """
    server = StarlingServer(config=ServerConfig(stream_chunk_seconds=0.0))
    sess = StreamSession(server=server)
    sess.chunker = chunker
    return sess


def _pcm_chunk(n_samples: int) -> bytes:
    """n_samples of int16 PCM zeros -> 2*n_samples bytes."""
    return np.zeros(n_samples, dtype="<i2").tobytes()


def test_stream_session_trims_committed_prefix() -> None:
    """Once the chunker has committed a large prefix, appending trims it."""
    chunker = _FakeChunker(boundary=0)
    sess = _stream_session_with_chunker(chunker)

    # Grow the buffer well past STREAM_TRIM_MIN_SAMPLES without any committed
    # prefix -> buffer should grow normally (no trim yet).
    grow = STREAM_TRIM_MIN_SAMPLES * 3
    sess.append_pcm(_pcm_chunk(grow))
    assert len(sess.samples) == grow
    assert chunker.boundary == 0  # nothing committed

    # Now simulate the chunker having finalized the first chunk of audio.
    committed = STREAM_TRIM_MIN_SAMPLES + 1000
    chunker.boundary = committed

    # Append a small further chunk -> the trim should fire, dropping the
    # committed prefix and resetting the chunker's boundary to 0.
    sess.append_pcm(_pcm_chunk(500))

    assert chunker.boundary == 0
    assert len(sess.samples) == (grow - committed) + 500


def test_stream_session_does_not_trim_when_nothing_committed() -> None:
    """With boundary == 0 the buffer must grow without being trimmed."""
    chunker = _FakeChunker(boundary=0)
    sess = _stream_session_with_chunker(chunker)

    total = STREAM_TRIM_MIN_SAMPLES * 4
    sess.append_pcm(_pcm_chunk(total))

    assert chunker.boundary == 0
    assert len(sess.samples) == total  # nothing trimmed


# ---------------------------------------------------------------------------
# E. flush retries on busy
# ---------------------------------------------------------------------------
def _chunker() -> ChunkStreamer:
    return ChunkStreamer(
        sample_rate=SAMPLE_RATE,
        chunk_seconds=1.0,
        overlap_seconds=0.25,
        min_seconds=0.5,
        partial_interval_seconds=10.0,
    )


def test_flush_commits_tail_when_succeeds_within_retries(monkeypatch) -> None:
    """tx returning None then succeeding commits the tail text."""
    monkeypatch.setattr("starling.stream_chunk._FLUSH_TAIL_BACKOFF_SECONDS", 0.0)
    chunker = _chunker()

    calls = {"n": 0}

    def tx(_window):  # noqa: ANN001
        calls["n"] += 1
        return None if calls["n"] == 1 else "tail text"

    # Feed audio that is shorter than one full window so it all lands in the
    # tail path (flush's retry loop), not _finalize_full_windows.
    samples = np.zeros(int(0.5 * SAMPLE_RATE), dtype=np.float32)
    out = chunker.flush(samples, tx)

    assert out == "tail text"
    assert calls["n"] == 2  # one busy, one success
    assert chunker.boundary == len(samples)  # tail finalized


def test_flush_drops_tail_when_always_busy_and_logs_warning(monkeypatch, caplog) -> None:
    """Always-busy tx: committed text returned WITHOUT the tail, warning logged."""
    monkeypatch.setattr("starling.stream_chunk._FLUSH_TAIL_BACKOFF_SECONDS", 0.0)
    chunker = _chunker()

    calls = {"n": 0}

    def tx(_window):  # noqa: ANN001
        calls["n"] += 1
        return None  # always busy

    # Pre-seed some committed text so we can assert it survives the dropped tail.
    chunker.committed = ["already", "committed"]

    samples = np.zeros(int(0.5 * SAMPLE_RATE), dtype=np.float32)
    with caplog.at_level("WARNING", logger="starling.stream_chunk"):
        out = chunker.flush(samples, tx)

    assert out == "already committed"  # tail dropped, committed kept
    assert calls["n"] == _FLUSH_TAIL_MAX_RETRIES  # bounded retry count
    assert any("dropped untranscribed tail" in rec.message for rec in caplog.records)


def test_flush_never_hangs_on_persistent_busy(monkeypatch) -> None:
    """The retry loop must be bounded: flush returns in finite time, not hang."""
    monkeypatch.setattr("starling.stream_chunk._FLUSH_TAIL_BACKOFF_SECONDS", 0.0)
    chunker = _chunker()
    samples = np.zeros(int(0.5 * SAMPLE_RATE), dtype=np.float32)

    done = threading.Event()
    result: dict = {}

    def runner() -> None:
        result["out"] = chunker.flush(samples, lambda _w: None)
        done.set()

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    # Generous but finite: if flush were unbounded this would time out.
    assert done.wait(timeout=5.0), "flush hung instead of bounding retries"
    assert result["out"] == ""


# ---------------------------------------------------------------------------
# F. flags() contextmanager
# ---------------------------------------------------------------------------
def test_flags_applies_override_and_restores() -> None:
    """Inside the block the override is visible; after, the default is back."""
    from starling import flags as flags_mod

    before = flags_mod.get_default_flags()
    original_tol = before.tolerance_mode

    with flags_mod.flags(tolerance_mode=True) as f:
        assert f.tolerance_mode is True
        assert flags_mod.get_default_flags().tolerance_mode is True

    # restored after the block
    assert flags_mod.get_default_flags().tolerance_mode is original_tol


def test_flags_ignores_unknown_keys() -> None:
    """An unknown override key does not crash (lenient behavior preserved)."""
    from starling import flags as flags_mod

    with flags_mod.flags(nonexistent_flag=True) as f:
        # known defaults still present
        assert f.multistep_graph is True


def test_flags_preserves_all_existing_fields_on_override() -> None:
    """Regression guard: overriding one field must not silently drop the others."""
    import dataclasses

    from starling import flags as flags_mod

    saved = flags_mod.get_default_flags()
    saved_snapshot = {fld.name: getattr(saved, fld.name) for fld in dataclasses.fields(saved)}

    with flags_mod.flags(tolerance_mode=True) as f:
        for fld in dataclasses.fields(f):
            name = fld.name
            expected = (True if name == "tolerance_mode" else saved_snapshot[name])
            assert getattr(f, name) == expected, f"field {name!r} not preserved"

    # global fully restored
    for fld in dataclasses.fields(saved):
        assert getattr(flags_mod.get_default_flags(), fld.name) == saved_snapshot[fld.name]


def test_flags_concurrent_overrides_do_not_crash() -> None:
    """Two threads entering flags() concurrently with different overrides succeed."""
    from starling import flags as flags_mod

    errors: list[BaseException] = []

    def worker(value: bool) -> None:
        try:
            with flags_mod.flags(tolerance_mode=value) as f:
                assert f.tolerance_mode is value
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(v,)) for v in (True, False)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors


# ---------------------------------------------------------------------------
# G. Warmup dedup
# ---------------------------------------------------------------------------
def _make_warmable_server(monkeypatch) -> tuple[StarlingServer, dict]:
    """Build a loaded StarlingServer whose GPU work is faked + counted.

    Returns (server, counters) where counters['transcribe'] counts how many
    times the real GPU-side ``_transcribe_np`` body ran.
    """
    import starling.parakeet.gpu_lock as gpu_lock

    class _Backend:
        _cancel_event = None
        _deadline = float("inf")

        def set_graph_mode(self, **kwargs):  # noqa: ANN001, ARG002
            pass

    counters = {"transcribe": 0}
    server = StarlingServer(backend=_Backend(), _loaded=True)

    def fake_transcribe(self, samples, *, streaming=False):  # noqa: ANN001, ARG002
        counters["transcribe"] += 1
        return TranscribeResult(text="warm")

    monkeypatch.setattr(
        StarlingServer, "_transcribe_np", fake_transcribe
    )
    # Make with_gpu_lock a fast no-op (no real flock / nvidia-smi).
    monkeypatch.setattr(gpu_lock, "acquire_gpu_lock", lambda **kwargs: "owner")
    monkeypatch.setattr(gpu_lock, "release_gpu_lock", lambda owner=None: True)
    return server, counters


def test_warmup_dedupes_concurrent_calls(monkeypatch) -> None:
    """Two concurrent warmup() calls run the GPU body exactly once.

    The dedup guards *concurrent in-flight* calls. To exercise it reliably we
    hold the first caller inside the (faked) GPU work on an Event until the
    second caller has had a chance to observe ``_warmup_in_progress`` and bail.
    Without this latch the first call can finish and clear the flag before the
    second checks it, making the assertion race-dependent (the dedup is still
    correct — it only guarantees dedup while a call is genuinely in flight).
    """
    server, counters = _make_warmable_server(monkeypatch)

    in_gpu_work = threading.Event()  # the fake body sets this when it starts
    release_gpu_work = threading.Event()  # the test releases it after a beat
    barrier = threading.Barrier(2)

    real_fake_transcribe = StarlingServer._transcribe_np

    def blocking_fake_transcribe(self, samples, *, streaming=False):  # noqa: ANN001, ARG002
        in_gpu_work.set()  # signal that the first call is inside the GPU work
        release_gpu_work.wait(timeout=5.0)  # hold until the test releases us
        counters["transcribe"] += 1
        return TranscribeResult(text="warm")

    monkeypatch.setattr(StarlingServer, "_transcribe_np", blocking_fake_transcribe)

    def call_warmup() -> None:
        barrier.wait()  # line up both threads, then race into warmup()
        server.warmup()

    t1 = threading.Thread(target=call_warmup)
    t2 = threading.Thread(target=call_warmup)
    t1.start()
    t2.start()

    # Wait until one thread has entered the GPU body (flag is now set), then
    # give the other thread a moment to observe the flag and dedup out.
    assert in_gpu_work.wait(timeout=5.0), "first caller never entered GPU work"
    release_gpu_work.set()  # let the in-flight call finish

    t1.join(timeout=10.0)
    t2.join(timeout=10.0)

    # Restore the original fake so later tests in the session get the simple counter.
    monkeypatch.setattr(StarlingServer, "_transcribe_np", real_fake_transcribe)

    assert counters["transcribe"] == 1  # deduped: GPU work ran once
    assert server._warmup_in_progress is False  # flag reset afterward


def test_warmup_second_call_after_first_completes_runs_again(monkeypatch) -> None:
    """A later warmup (after the first finished) is allowed to run again.

    The dedup is only for *concurrent* in-flight warmups, not a once-ever guard.
    """
    server, counters = _make_warmable_server(monkeypatch)

    server.warmup()
    server.warmup()

    assert counters["transcribe"] == 2
    assert server._warmup_in_progress is False


def test_warmup_noop_when_not_loaded(monkeypatch) -> None:
    """warmup() on an unloaded server is a no-op (no GPU work)."""
    import starling.parakeet.gpu_lock as gpu_lock

    server = StarlingServer()  # not loaded, backend None
    assert server.loaded is False

    def boom(**_kwargs):  # pragma: no cover - must not be called
        raise AssertionError("warmup should not reach GPU lock when unloaded")

    monkeypatch.setattr(gpu_lock, "acquire_gpu_lock", boom)
    server.warmup()  # must short-circuit before touching the GPU lock
