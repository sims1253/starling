"""CPU-only regression tests for recently-fixed robustness bugs (review #12).

Each test exercises a pure / faked code path in :mod:`starling.server`,
:mod:`starling.stream_chunk`, or :mod:`starling.flags`. No GPU, no loaded
model, no network, no live server socket. These guard against regressions of
the specific fixes called out in the review:

* A. malformed audio -> HTTP 400 (not a 500 / dead socket)
* B. multipart parsing selects the audio field by name / filename / content-type
* D. streaming buffer is bounded (committed prefix trimmed)
* E. ``ChunkStreamer.flush`` retries the tail on a busy transcriber (bounded)
* F. ``flags()`` applies overrides, restores defaults, preserves all fields
* G. ``StarlingServer.warmup`` dedupes concurrent calls
"""

from __future__ import annotations

import io
import os
import sys
import threading
import time
from typing import Any, NoReturn

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, "..", "src"))
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from starling.server import (  # noqa: E402
    SAMPLE_RATE,
    STREAM_TRIM_MIN_SAMPLES,
    ServerConfig,
    StarlingServer,
    StreamSession,
    TranscribeResult,
    _extract_multipart_payload,
    _transcribe_payload_sync,
    _wav_bytes_to_float32,
)
from starling.stream_chunk import (  # noqa: E402
    ChunkStreamer,
    _FLUSH_MAX_RETRIES,
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
            # Scale by the positive max-representable value so a clipped +1.0
            # maps to +32767 rather than overflowing int16 to -32768.
            raw = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
        elif sampwidth == 4:
            raw = (np.clip(samples, -1.0, 1.0) * 2147483647.0).astype("<i4").tobytes()
        elif sampwidth == 1:
            raw = ((np.clip(samples, -1.0, 1.0) * 128.0) + 128.0).astype(np.uint8).tobytes()
        else:
            raise ValueError(sampwidth)
        wf.writeframes(raw)
    return buf.getvalue()


class _FakeServer:
    """Minimal stand-in for StarlingServer for ``_transcribe_payload_sync``.

    Only the surface used by the transport adapter is faked: ``decode_wav_bytes``
    raises the injected ``exc`` (simulating a decode failure) when set, and
    ``_run_queued_sync`` returns a canned :class:`TranscribeResult` on the clean
    path. NOTE: these tests exercise only the exception-to-status mapping in
    ``_transcribe_payload_sync``; they do NOT exercise the real
    ``_wav_bytes_to_float32`` -- see ``test_wav_bytes_to_float32_*`` below for
    direct decoder coverage.
    """

    def __init__(self, exc: BaseException | None = None, *, result_text: str = "ok") -> None:
        # ``exc``: if set, ``decode_wav_bytes`` raises it (simulating a decode
        # failure). If None, decode + inference succeed with ``result_text``.
        self._exc = exc
        self._result_text = result_text

    def decode_wav_bytes(self, _wav_bytes: bytes) -> np.ndarray:
        # Argument mirrors StarlingServer.decode_wav_bytes' signature so this
        # fake is duck-typed to the transport adapter; the bytes are unused
        # (the injected ``_exc`` decides the path, not the payload).
        if self._exc is not None:
            raise self._exc
        return np.zeros(SAMPLE_RATE, dtype=np.float32)

    def _ensure_loaded(self) -> None:
        # No-op: the fake has no real backend to load. Lets _transcribe_payload_sync
        # call _ensure_loaded() before queueing (mirrors the real StarlingServer).
        pass

    def _run_queued_sync(
        self, _samples: np.ndarray, _request_id: str, *, _streaming: bool = False
    ) -> TranscribeResult:
        # Signature matches StarlingServer._run_queued_sync for duck-typing; the
        # arguments are intentionally unused (a canned result is always returned).
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


def test_transcribe_payload_loads_backend_before_queueing_inference() -> None:
    """A valid request to an (unloaded) server triggers _ensure_loaded() before
    inference is queued -- regression: the decode/inference split dropped the
    lazy-load call, so an unloaded server would crash inside _run_queued_sync
    instead of loading first.

    Records the call order so a load-after-queue regression (e.g. moving the
    _ensure_loaded() call into _run_queued_sync, or after it) fails here rather
    than silently passing on a single boolean."""
    server = _FakeServer(result_text="loaded-and-transcribed")
    calls: list[str] = []

    def _track_load() -> None:
        calls.append("load")

    def _track_queue(  # noqa: ANN001
        _samples: np.ndarray, _request_id: str, *, _streaming: bool = False
    ) -> TranscribeResult:
        # The queued fake asserts loading already happened -- the strongest
        # ordering check: if _ensure_loaded hasn't run yet, this raises.
        assert calls, "_run_queued_sync ran but _ensure_loaded never did"
        assert calls[-1] == "load", (
            "_run_queued_sync started before _ensure_loaded completed"
        )
        calls.append("queue")
        return TranscribeResult(text=server._result_text)

    server._ensure_loaded = _track_load  # type: ignore[method-assign]
    server._run_queued_sync = _track_queue  # type: ignore[method-assign]
    payload = _wav_bytes(np.zeros(SAMPLE_RATE, dtype=np.float32))

    status, response = _transcribe_payload_sync(server, payload, "req-load")

    assert status == 200
    assert response["text"] == "loaded-and-transcribed"
    # Explicit order assertion: load precedes queue.
    assert calls == ["load", "queue"]


def test_wav_bytes_to_float32_raises_on_truncated_body() -> None:
    """A truncated WAV body raises ValueError/wave.Error from the real decoder.

    This exercises the actual ``_wav_bytes_to_float32`` decode path (the
    section-A tests above only check the exception-to-status mapping in the
    transport adapter via the faked decode).
    """
    import wave

    with pytest.raises((ValueError, wave.Error)):
        _wav_bytes_to_float32(b"RIFF\x00\x00\x00\x00WAVEjunk")


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


def test_multipart_all_zero_score_parts_returns_last_part() -> None:
    """Multiple parts that all score 0 (no audio/file name, no filename, no
    audio content-type) fall back to the LAST non-empty part."""
    boundary = "----b"
    part_a = _mp_part(
        [("Content-Disposition", 'form-data; name="note"')],
        b"first note",
    )
    part_b = _mp_part(
        [("Content-Disposition", 'form-data; name="note2"')],
        b"second note",
    )
    body = b"--" + boundary.encode() + b"\r\n" + part_a + b"\r\n" \
        + b"--" + boundary.encode() + b"\r\n" + part_b + b"\r\n" \
        + b"--" + boundary.encode() + b"--\r\n"
    ctype = f"multipart/form-data; boundary={boundary}"

    out = _extract_multipart_payload(body, ctype)

    assert out == b"second note"  # last part wins


def test_multipart_boundary_with_no_usable_parts_returns_original_body() -> None:
    """A Content-Type carrying a boundary whose body yields no candidate parts
    returns the original body unchanged."""
    boundary = "----b"
    body = b"--" + boundary.encode() + b"--\r\n"
    ctype = f"multipart/form-data; boundary={boundary}"

    out = _extract_multipart_payload(body, ctype)

    assert out == body


def test_multipart_zero_score_fallback_ignores_trailing_empty_part() -> None:
    """The all-zero-score fallback returns the last NON-EMPTY part, not a
    trailing empty part (regression: last_payload was clobbered by empties)."""
    boundary = "----b"
    note_part = _mp_part(
        [("Content-Disposition", 'form-data; name="note"')],
        b"the real payload",
    )
    empty_part = _mp_part(
        [("Content-Disposition", 'form-data; name="trailer"')],
        b"",  # trailing empty part -- must not become the fallback
    )
    body = b"--" + boundary.encode() + b"\r\n" + note_part + b"\r\n" \
        + b"--" + boundary.encode() + b"\r\n" + empty_part + b"\r\n" \
        + b"--" + boundary.encode() + b"--\r\n"
    ctype = f"multipart/form-data; boundary={boundary}"

    out = _extract_multipart_payload(body, ctype)

    assert out == b"the real payload"  # last non-empty, not the empty trailer


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


def _stream_session_with_chunker(chunker: Any) -> StreamSession:
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


def test_stream_session_does_not_trim_small_committed_prefix() -> None:
    """A committed boundary below the min threshold AND below half the buffer
    leaves the boundary and samples unchanged (the small-prefix guard)."""
    chunker = _FakeChunker(boundary=0)
    sess = _stream_session_with_chunker(chunker)

    # Grow the buffer well past the threshold so a small committed prefix is
    # both < STREAM_TRIM_MIN_SAMPLES and < half the buffer.
    grow = STREAM_TRIM_MIN_SAMPLES * 4
    sess.append_pcm(_pcm_chunk(grow))
    small = STREAM_TRIM_MIN_SAMPLES // 2  # below min threshold
    chunker.boundary = small

    sess.append_pcm(_pcm_chunk(500))

    # Guard fired: nothing trimmed.
    assert chunker.boundary == small
    assert len(sess.samples) == grow + 500


def test_stream_session_does_not_trim_when_boundary_equals_buffer() -> None:
    """A boundary equal to the buffer length (b >= len) leaves the full buffer
    and boundary unchanged (the b >= len(samples) guard)."""
    chunker = _FakeChunker(boundary=0)
    sess = _stream_session_with_chunker(chunker)

    n = STREAM_TRIM_MIN_SAMPLES * 3
    sess.append_pcm(_pcm_chunk(n))
    chunker.boundary = n  # boundary == buffer length

    sess.append_pcm(_pcm_chunk(0))

    assert chunker.boundary == n
    assert len(sess.samples) == n  # full buffer retained


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


def test_flush_commits_tail_when_succeeds_within_retries(monkeypatch) -> None:  # noqa: ANN001
    """tx returning None then succeeding commits the tail text."""
    monkeypatch.setattr("starling.stream_chunk._FLUSH_BACKOFF_SECONDS", 0.0)
    chunker = _chunker()

    calls = {"n": 0}

    def tx(_window: np.ndarray) -> str | None:
        calls["n"] += 1
        return None if calls["n"] == 1 else "tail text"

    # Feed audio that is shorter than one full window so it all lands in the
    # tail path (flush's retry loop), not _finalize_full_windows.
    samples = np.zeros(int(0.5 * SAMPLE_RATE), dtype=np.float32)
    out = chunker.flush(samples, tx)

    assert out == "tail text"
    assert calls["n"] == 2  # one busy, one success
    assert chunker.boundary == len(samples)  # tail finalized


def test_flush_retains_tail_when_always_busy(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr("starling.stream_chunk._FLUSH_BACKOFF_SECONDS", 0.0)
    chunker = _chunker()
    chunker.committed = ["already", "committed"]
    calls = []
    samples = np.zeros(int(0.5 * SAMPLE_RATE), dtype=np.float32)
    assert chunker.flush(samples, lambda window: calls.append(len(window))) is None
    assert len(calls) == _FLUSH_MAX_RETRIES
    assert chunker.boundary == 0
    assert chunker.committed == ["already", "committed"]
    assert chunker.flush(samples, lambda window: "tail text") == "already committed tail text"
    assert chunker.boundary == len(samples)


def test_flush_accepts_empty_string_result(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr("starling.stream_chunk._FLUSH_BACKOFF_SECONDS", 0.0)
    chunker = _chunker()
    chunker.committed = ["already", "committed"]
    samples = np.zeros(int(0.5 * SAMPLE_RATE), dtype=np.float32)
    assert chunker.flush(samples, lambda window: "") == "already committed"
    assert chunker.boundary == len(samples)


def test_flush_never_hangs_on_persistent_busy(monkeypatch) -> None:  # noqa: ANN001
    """The retry loop must be bounded: flush returns in finite time, not hang."""
    monkeypatch.setattr("starling.stream_chunk._FLUSH_BACKOFF_SECONDS", 0.0)
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
    assert result["out"] is None


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


def test_flags_reject_unknown_override_without_changing_defaults() -> None:
    from starling import flags as flags_mod

    saved = flags_mod.get_default_flags()
    with pytest.raises(TypeError, match="multistep_grap"):
        with flags_mod.flags(multistep_grap=False):
            pytest.fail("unknown flag was accepted")
    assert flags_mod.get_default_flags() is saved


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
    """Two threads entering flags() concurrently with different overrides succeed.

    Captures the default global flags before the threads start and asserts they
    match that snapshot (including tolerance_mode) after both joins -- so a
    cross-scope restore leak would be caught, not just a crash.
    """
    from starling import flags as flags_mod

    snapshot = flags_mod.get_default_flags()

    errors: list[BaseException] = []

    def worker(value: bool) -> None:  # noqa: FBT001 -- positional: Thread(args=...)
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
    # The process-global default must be fully restored after both scopes exit,
    # including tolerance_mode -- overlapping scopes must not leak overrides.
    assert flags_mod.get_default_flags() == snapshot


def test_flags_overlapping_scopes_do_not_clobber_restore() -> None:
    """Concurrent scopes cannot restore each other's snapshots.

    The RLock is held across the entire yielded body, so two ``flags()`` scopes
    are serialized: the second scope waits for the first to fully exit (snapshot
    + body + restore) before it can itself snapshot. Each scope therefore sees
    a consistent global default, and after both exit the global is back to the
    pre-test default -- overlapping scopes never leak or clobber each other.
    """
    from starling import flags as flags_mod

    snapshot = flags_mod.get_default_flags()
    errors: list[BaseException] = []
    active = {"n": 0, "max": 0}
    active_lock = threading.Lock()

    def worker(value: bool) -> None:  # noqa: FBT001 -- positional: Thread(args=...)
        try:
            with flags_mod.flags(tolerance_mode=value) as f:
                # Each scope observes its own locally-built override.
                assert f.tolerance_mode is value
                # Track overlap: if scopes ever interleave, max > 1.
                with active_lock:
                    active["n"] += 1
                    active["max"] = max(active["max"], active["n"])
                # Small delay to widen the concurrency window.
                time.sleep(0.05)
                with active_lock:
                    active["n"] -= 1
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(v,)) for v in (True, False)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    # Scopes are serialized (never overlap) -> max concurrency inside a scope is 1.
    assert active["max"] == 1
    # The process-global default is fully restored after both scopes exit.
    assert flags_mod.get_default_flags() == snapshot


# ---------------------------------------------------------------------------
# G. Warmup dedup
# ---------------------------------------------------------------------------
def _make_warmable_server(monkeypatch) -> tuple[StarlingServer, dict]:  # noqa: ANN001
    """Build a loaded StarlingServer whose GPU work is faked + counted.

    Returns (server, counters) where counters['transcribe'] counts how many
    times the real GPU-side ``_transcribe_np`` body ran.
    """
    import starling.parakeet.gpu_lock as gpu_lock

    class _Backend:
        _cancel_event = None
        _deadline = float("inf")

        def set_graph_mode(self, **_kwargs: Any) -> None:
            pass

    counters = {"transcribe": 0}
    server = StarlingServer(backend=_Backend(), _loaded=True)

    def fake_transcribe(self, _samples: np.ndarray, *, _streaming: bool = False) -> TranscribeResult:  # noqa: ANN001, ARG001
        counters["transcribe"] += 1
        return TranscribeResult(text="warm")

    monkeypatch.setattr(
        StarlingServer, "_transcribe_np", fake_transcribe
    )
    # Make with_gpu_lock a fast no-op (no real flock / nvidia-smi).
    monkeypatch.setattr(gpu_lock, "acquire_gpu_lock", lambda **kwargs: "owner")
    monkeypatch.setattr(gpu_lock, "release_gpu_lock", lambda owner=None: True)
    return server, counters


def test_warmup_dedupes_concurrent_calls(monkeypatch) -> None:  # noqa: ANN001
    """The second warmup returns while the first is still in GPU work."""
    server, counters = _make_warmable_server(monkeypatch)
    in_gpu_work = threading.Event()
    release_gpu_work = threading.Event()
    second_done = threading.Event()

    def blocking_fake_transcribe(self, _samples: np.ndarray, *, _streaming: bool = False) -> TranscribeResult:  # noqa: ANN001, ARG001
        in_gpu_work.set()
        assert release_gpu_work.wait(timeout=10.0), "GPU work was never released"
        counters["transcribe"] += 1
        return TranscribeResult(text="warm")

    monkeypatch.setattr(StarlingServer, "_transcribe_np", blocking_fake_transcribe)

    def second_warmup() -> None:
        server.warmup()
        second_done.set()

    first = threading.Thread(target=server.warmup, daemon=True)
    second = threading.Thread(target=second_warmup, daemon=True)
    first.start()
    try:
        assert in_gpu_work.wait(timeout=5.0), "first caller never entered GPU work"
        second.start()
        assert second_done.wait(timeout=5.0), "second caller did not deduplicate"
        assert counters["transcribe"] == 0
    finally:
        release_gpu_work.set()
        first.join(timeout=5.0)
        if second.ident is not None:
            second.join(timeout=5.0)

    assert not first.is_alive() and not second.is_alive()
    assert counters["transcribe"] == 1
    assert server._warmup_in_progress is False


def test_warmup_second_call_after_first_completes_runs_again(monkeypatch) -> None:  # noqa: ANN001
    """A later warmup (after the first finished) is allowed to run again.

    The dedup is only for *concurrent* in-flight warmups, not a once-ever guard.
    """
    server, counters = _make_warmable_server(monkeypatch)

    server.warmup()
    server.warmup()

    assert counters["transcribe"] == 2
    assert server._warmup_in_progress is False


def test_warmup_noop_when_not_loaded(monkeypatch) -> None:  # noqa: ANN001
    """warmup() on an unloaded server is a no-op (no GPU work)."""
    import starling.parakeet.gpu_lock as gpu_lock

    server = StarlingServer()  # not loaded, backend None
    assert server.loaded is False

    def boom(**_kwargs: Any) -> NoReturn:  # pragma: no cover - must not be called
        raise AssertionError("warmup should not reach GPU lock when unloaded")

    monkeypatch.setattr(gpu_lock, "acquire_gpu_lock", boom)
    server.warmup()  # must short-circuit before touching the GPU lock


@pytest.mark.parametrize("path", ["/inference", "/transcribe"])
@pytest.mark.parametrize("multipart", [False, True])
def test_http_aliases_accept_wav_uploads(path, multipart, monkeypatch):
    import asyncio
    import json
    from starling import server as module

    wav = _wav_bytes(np.zeros(160, dtype=np.float32))
    body = wav
    content_type = "audio/wav"
    if multipart:
        body = (b'--audio-boundary\r\n'
                b'Content-Disposition: form-data; name="file"; filename="clip.wav"\r\n'
                b'Content-Type: audio/wav\r\n\r\n' + wav + b'\r\n--audio-boundary--\r\n')
        content_type = "multipart/form-data; boundary=audio-boundary"
    server = StarlingServer()
    monkeypatch.setattr(server, "_ensure_loaded", lambda: None)

    def transcribe(samples, request_id):
        assert len(samples) == 160
        assert request_id == "upload-id"
        return TranscribeResult(text="accepted")

    monkeypatch.setattr(server, "_run_queued_sync", transcribe)
    headers = {"content-length": str(len(body)), "content-type": content_type,
               "x-request-id": "upload-id"}
    fastapi = pytest.importorskip("fastapi")
    app = module.create_app(server=server, load_on_startup=False)

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    request = fastapi.Request({"type": "http", "method": "POST", "path": path,
                               "headers": [(k.encode(), v.encode()) for k, v in headers.items()]},
                              receive)
    route = next(r for r in app.routes if getattr(r, "path", None) == path)
    result = asyncio.run(route.endpoint(request))
    status, response = result.status_code, json.loads(result.body)
    assert status == 200
    assert response["text"] == "accepted"
    assert response["request_id"] == "upload-id"


@pytest.mark.parametrize("stage", ["_ensure_loaded", "_run_queued_sync"])
@pytest.mark.parametrize("error_type", [RuntimeError, ValueError])
def test_http_engine_errors_are_json(stage, error_type, monkeypatch, caplog):
    import asyncio
    import json
    from starling import server as module

    server = StarlingServer()
    monkeypatch.setattr(server, "_ensure_loaded", lambda: None)

    def fail(*args):
        raise error_type("private engine details")

    monkeypatch.setattr(server, stage, fail)
    body = _wav_bytes(np.zeros(160, dtype=np.float32))
    headers = {"content-type": "audio/wav", "content-length": str(len(body)),
               "x-request-id": "failed-request"}
    pytest.importorskip("fastapi")
    app = module.create_app(server=server, load_on_startup=False)
    messages = []

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message):
        messages.append(message)

    asyncio.run(app({"type": "http", "asgi": {"version": "3.0"},
                     "http_version": "1.1", "method": "POST", "scheme": "http",
                     "path": "/inference", "query_string": b"", "root_path": "",
                     "headers": [(k.encode(), v.encode()) for k, v in headers.items()]},
                    receive, send))
    status = messages[0]["status"]
    content_type = dict(messages[0]["headers"])[b"content-type"].decode()
    payload = b"".join(message.get("body", b"") for message in messages[1:])
    assert status == 500
    assert content_type == "application/json"
    assert json.loads(payload) == {"error": "transcription failed", "text": "",
                                   "request_id": "failed-request"}
    record = next(record for record in caplog.records if "failed-request" in record.message)
    assert record.exc_info[1].args == ("private engine details",)
