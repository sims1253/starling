"""Unit tests for fixed-window overlapping-chunk streaming (no model needed).

The transcription is faked deterministically: the sample buffer is filled with
its own indices (``samples[i] == i``), so a fake transcriber can read a window's
absolute start from ``window[0]`` and return the ground-truth words whose time
falls in that window. This exercises the real chunk/finalize/stitch logic and
verifies the reconstructed transcript equals the ground truth despite overlap.
"""

from __future__ import annotations

import numpy as np
import pytest

from starling.stream_chunk import ChunkStreamer, stitch_words, stream_window_config_error

SR = 16000


# --------------------------------------------------------------------------- #
# stitch_words
# --------------------------------------------------------------------------- #
def test_stitch_exact_overlap():
    assert stitch_words(["a", "b", "c"], ["b", "c", "d", "e"]) == ["a", "b", "c", "d", "e"]


def test_stitch_no_common_run_concatenates():
    assert stitch_words(["a", "b"], ["c", "d"]) == ["a", "b", "c", "d"]


def test_stitch_empty_sides():
    assert stitch_words([], ["a", "b"]) == ["a", "b"]
    assert stitch_words(["a", "b"], []) == ["a", "b"]


def test_stitch_normalizes_case_and_punctuation():
    # 2-word overlap "hello world" vs "Hello World." -> deduped by normalization
    out = stitch_words(["say", "hello", "world."], ["Hello", "World", "again"])
    assert out == ["say", "hello", "world.", "again"]


def test_stitch_non_ascii_words_survive_and_dedupe():
    # Shared fixture with the C++ port (issue #118): \w is Unicode-aware here,
    # so non-ASCII words must never lose words to empty normalization keys —
    # the native port regressed on exactly these fixtures before it became
    # UTF-8 aware.
    assert stitch_words(["привет", "мир"], ["совсем", "другое"]) == [
        "привет", "мир", "совсем", "другое",
    ]
    assert stitch_words(["你好", "世界"], ["再见", "朋友"]) == ["你好", "世界", "再见", "朋友"]
    assert stitch_words(
        ["привет", "мир", "тут"], ["мир", "тут", "ок"]
    ) == ["привет", "мир", "тут", "ок"]


def test_stitch_empty_keys_never_match():
    # Words that normalize to "" (pure punctuation) must never count as an
    # overlap run — defense in depth for issue #118, shared fixture with the
    # C++ port: matching empty keys would drop the new chunk's leading words.
    assert stitch_words(["--", ";;"], ["..", ",,", "word"]) == ["--", ";;", "..", ",,", "word"]


def test_stitch_tolerates_one_word_error_in_overlap():
    # overlap region "quick brown fox" vs "quick BROWN-ish fox": longest run
    # ("fox") still splices without dropping the tail or duplicating it wholesale
    committed = ["the", "quick", "brown", "fox"]
    new = ["quick", "brownish", "fox", "jumps", "over"]
    out = stitch_words(committed, new)
    assert out[-2:] == ["jumps", "over"]  # tail always appended
    assert out.count("jumps") == 1 and out.count("over") == 1


# --------------------------------------------------------------------------- #
# ChunkStreamer end-to-end (faked transcription over a word timeline)
# --------------------------------------------------------------------------- #
def _make_tx(words_with_times, sr=SR, record=None):
    def tx(window):
        start_s = float(window[0]) / sr
        end_s = start_s + len(window) / sr
        if record is not None:
            record.append(len(window) / sr)
        return " ".join(w for (w, t) in words_with_times if start_s <= t < end_s)
    return tx


def _timeline(n_words=40, spacing=0.7):
    # unique words so exact-overlap dedup is unambiguous
    return [(f"w{i:03d}", 0.3 + i * spacing) for i in range(n_words)]


def test_chunkstreamer_reconstructs_full_transcript():
    words = _timeline(40, 0.7)          # ~28s of speech
    truth = [w for (w, _) in words]
    total_s = words[-1][1] + 1.0
    samples = np.arange(int(total_s * SR), dtype=np.float32)  # value == index
    cs = ChunkStreamer(sample_rate=SR, chunk_seconds=12, overlap_seconds=2,
                       min_seconds=5, partial_interval_seconds=3)
    tx = _make_tx(words)
    # simulate streaming: reveal the buffer in 0.5s increments, step each time
    now = 0.0
    for end in range(SR // 2, len(samples) + 1, SR // 2):
        now += 0.5
        partial = cs.step(samples[:end], now, tx)
        if partial is not None:
            expected = [w for w, t in words if t < end / SR]
            assert partial.split() == expected
    final_text = cs.flush(samples, tx)
    assert final_text.split() == truth, f"reconstructed != truth:\n{final_text}"


def test_chunkstreamer_window_is_bounded():
    words = _timeline(60, 0.7)          # ~42s -> would overflow a naive buffer
    total_s = words[-1][1] + 1.0
    samples = np.arange(int(total_s * SR), dtype=np.float32)
    seen: list[float] = []
    cs = ChunkStreamer(sample_rate=SR, chunk_seconds=12, overlap_seconds=2,
                       min_seconds=5, partial_interval_seconds=3)
    tx = _make_tx(words, record=seen)
    now = 0.0
    for end in range(SR // 2, len(samples) + 1, SR // 2):
        now += 0.5
        cs.step(samples[:end], now, tx)
    cs.flush(samples, tx)
    # no transcribe ever sees more than one chunk of audio (prompt bounded)
    assert max(seen) <= 12.0 + 1e-6, f"window exceeded chunk: {max(seen)}s"


def test_chunkstreamer_reconstructs_across_many_chunks():
    words = _timeline(120, 0.5)         # ~60s, many finalized windows
    truth = [w for (w, _) in words]
    total_s = words[-1][1] + 1.0
    samples = np.arange(int(total_s * SR), dtype=np.float32)
    cs = ChunkStreamer(sample_rate=SR, chunk_seconds=10, overlap_seconds=2,
                       min_seconds=4, partial_interval_seconds=2)
    tx = _make_tx(words)
    now = 0.0
    for end in range(SR // 2, len(samples) + 1, SR // 2):
        now += 0.5
        cs.step(samples[:end], now, tx)
    assert cs.flush(samples, tx).split() == truth


def test_chunkstreamer_busy_does_not_advance():
    words = _timeline(40, 0.7)
    total_s = words[-1][1] + 1.0
    samples = np.arange(int(total_s * SR), dtype=np.float32)
    cs = ChunkStreamer(sample_rate=SR, chunk_seconds=12, overlap_seconds=2,
                       min_seconds=5, partial_interval_seconds=3)
    # transcriber always "busy" -> None: no boundary advance, no emission
    out = cs.step(samples, 10.0, lambda w: None)
    assert out is None
    assert cs.boundary == 0 and cs.committed == []


def test_streamsession_chunked_integration():
    """Full StreamSession -> ChunkStreamer -> _tx path (fake, model-less server)."""
    from starling import server as S

    words = _timeline(40, 0.7)
    truth = [w for (w, _) in words]
    total = int((words[-1][1] + 1.0) * SR)
    buf = np.arange(total, dtype=np.float32)  # value == index

    class FakeServer:
        def __init__(self):
            self.config = S.ServerConfig(
                model="moss", stream_chunk_seconds=12, stream_overlap_seconds=2,
                min_chunk_seconds=5, partial_interval_seconds=3,
            )

        def _run_queued_sync(self, window, _rid, *, streaming=False):
            assert streaming is True, "chunked stream path should request streaming mode"
            start_s = float(window[0]) / SR
            end_s = start_s + len(window) / SR
            txt = " ".join(w for (w, t) in words if start_s <= t < end_s)
            return S.TranscribeResult(text=txt)

    sess = S.StreamSession(server=FakeServer())
    assert sess.chunker is not None, "chunker should be created from config"
    now = 0.0
    for end in range(SR // 2, total + 1, SR // 2):
        now += 0.5
        sess.samples = buf[:end]
        sess.stream_step(now)
    assert sess.stream_flush().split() == truth


def test_busy_full_window_waits_for_next_step():
    cs = ChunkStreamer(sample_rate=1, chunk_seconds=12, overlap_seconds=2,
                       min_seconds=5, partial_interval_seconds=0)
    samples = np.arange(30)
    seen = []

    def tx(window):
        seen.append((int(window[0]), len(window)))
        return None if len(seen) == 1 else "hello world"

    assert cs.step(samples, 10, tx) is None
    assert seen == [(0, 12)]
    assert cs.step(samples, 11, tx) == "hello world"
    assert seen == [(0, 12), (0, 12), (10, 12), (20, 10)]


def test_flush_retries_full_windows_without_repeating_committed_audio(monkeypatch):
    monkeypatch.setattr("starling.stream_chunk._FLUSH_BACKOFF_SECONDS", 0)
    cs = ChunkStreamer(sample_rate=1, chunk_seconds=12, overlap_seconds=2,
                       min_seconds=5, partial_interval_seconds=0)
    samples = np.arange(30)
    seen = []
    busy = True

    def tx(window):
        seen.append((int(window[0]), len(window)))
        return None if busy and window[0] == 10 else "hello world"

    assert cs.flush(samples, tx) is None
    assert cs.boundary == 10
    assert cs.committed == ["hello", "world"]
    busy = False
    assert cs.flush(samples, tx) == "hello world"
    assert seen.count((0, 12)) == 1
    assert seen[-2:] == [(10, 12), (20, 10)]
    assert all(length <= 12 for _, length in seen)
    assert cs.boundary == len(samples)



def test_flush_recovers_from_busy_full_window_in_same_commit(monkeypatch):
    monkeypatch.setattr("starling.stream_chunk._FLUSH_BACKOFF_SECONDS", 0)
    cs = ChunkStreamer(sample_rate=1, chunk_seconds=12, overlap_seconds=2,
                       min_seconds=5, partial_interval_seconds=0)
    seen = []

    def tx(window):
        seen.append((int(window[0]), len(window)))
        return None if len(seen) == 1 else "hello world"

    assert cs.flush(np.arange(30), tx) == "hello world"
    assert seen == [(0, 12), (0, 12), (10, 12), (20, 10)]


# --------------------------------------------------------------------------- #
# Stream window config validation (issue #146)
# --------------------------------------------------------------------------- #
def _cfg(**overrides):
    base = dict(sample_rate=SR, chunk_seconds=12.0, overlap_seconds=3.0,
                min_seconds=5.0, partial_interval_seconds=3.0)
    base.update(overrides)
    return base


def test_stream_window_config_error_cases():
    # The shipped defaults and the legacy whole-buffer switch are valid.
    assert stream_window_config_error(**_cfg()) is None
    assert stream_window_config_error(**_cfg(chunk_seconds=0.0)) is None
    # Everything else is rejected with an error message.
    assert stream_window_config_error(**_cfg(overlap_seconds=-3.0)) is not None
    assert stream_window_config_error(**_cfg(overlap_seconds=12.0)) is not None
    assert stream_window_config_error(**_cfg(overlap_seconds=13.0)) is not None
    assert stream_window_config_error(**_cfg(chunk_seconds=-12.0)) is not None
    assert stream_window_config_error(**_cfg(chunk_seconds=1e-9)) is not None
    assert stream_window_config_error(**_cfg(chunk_seconds=1e12)) is not None
    assert stream_window_config_error(**_cfg(min_seconds=-5.0)) is not None
    assert stream_window_config_error(**_cfg(min_seconds=1e12)) is not None
    assert stream_window_config_error(**_cfg(partial_interval_seconds=-3.0)) is not None
    assert stream_window_config_error(**_cfg(chunk_seconds=float("nan"))) is not None
    assert stream_window_config_error(**_cfg(overlap_seconds=float("inf"))) is not None
    assert stream_window_config_error(**_cfg(min_seconds=float("nan"))) is not None
    assert stream_window_config_error(**_cfg(partial_interval_seconds=float("inf"))) is not None
    assert stream_window_config_error(**_cfg(sample_rate=0)) is not None


def test_chunkstreamer_rejects_invalid_config():
    # The issue #146 repro (12 s window, -3 s overlap) used to build a chunker
    # whose advance skipped audio; invalid configs must now fail at
    # construction.
    for overrides in (
        dict(overlap_seconds=-3.0),     # negative overlap (issue #146)
        dict(overlap_seconds=12.0),     # overlap == chunk
        dict(overlap_seconds=13.0),     # overlap > chunk
        dict(chunk_seconds=0.0),        # legacy mode is CLI-only
        dict(chunk_seconds=-12.0),
        dict(chunk_seconds=1e-9),       # sub-sample window
        dict(chunk_seconds=1e12),       # sample-count overflow
        dict(min_seconds=-5.0),
        dict(partial_interval_seconds=-1.0),
        dict(chunk_seconds=float("nan")),
        dict(overlap_seconds=float("inf")),
        dict(min_seconds=float("nan")),
        dict(sample_rate=0),
    ):
        with pytest.raises(ValueError):
            ChunkStreamer(**_cfg(**overrides))
    # A valid config still constructs with the documented geometry.
    cs = ChunkStreamer(**_cfg())
    assert cs.chunk == 12 * SR
    assert cs.overlap == 3 * SR
    assert cs.advance == 9 * SR


def test_chunkstreamer_overlap_clamped_to_half_chunk():
    # Overlap above half the chunk keeps its documented clamp (half the
    # chunk): 9 s requested on a 12 s window overlaps 6 s and advances 6 s.
    cs = ChunkStreamer(**_cfg(overlap_seconds=9.0))
    assert cs.overlap == 6 * SR
    assert cs.advance == 6 * SR


def test_chunkstreamer_windows_stay_in_range():
    # Issue #146 invariant: every transcribe receives a nonempty window of at
    # most one chunk, with its interval inside the buffered samples (the
    # buffer holds its own indices, so window[0] is the absolute start).
    cs = ChunkStreamer(**_cfg(min_seconds=5.0, partial_interval_seconds=0.0))
    samples = np.arange(30 * SR, dtype=np.float32)
    calls: list[int] = []

    def tx(window):
        assert 0 < len(window) <= cs.chunk
        start = int(window[0])
        assert 0 <= start and start + len(window) <= len(samples)
        calls.append(len(window))
        return "w"

    assert cs.step(samples, 1.0, tx) is not None
    assert cs.flush(samples, tx) is not None
    assert calls == [12 * SR, 12 * SR, 12 * SR, 3 * SR]


def test_chunkstreamer_never_transcribes_empty_window():
    # min=0 with an exactly-consumed buffer must not hand the transcriber an
    # empty window (the partial-tail guard).
    cs = ChunkStreamer(sample_rate=SR, chunk_seconds=1.0, overlap_seconds=0.0,
                       min_seconds=0.0, partial_interval_seconds=0.0)
    calls: list[int] = []

    def tx(window):
        assert len(window) > 0
        calls.append(len(window))
        return "w"

    samples = np.zeros(SR, dtype=np.float32)  # exactly one window
    assert cs.step(samples, 1.0, tx) is not None
    assert calls == [SR]  # the full window only; no empty partial


# --------------------------------------------------------------------------- #
# Server CLI flag validation (mirror of the native starling-serve checks)
# --------------------------------------------------------------------------- #
def _validated_stream_args(argv):
    from starling import server as S

    args = S._build_arg_parser().parse_args(argv)
    S._validate_stream_args(args)
    return args


def test_server_stream_flags_accept_valid_config():
    args = _validated_stream_args([])
    assert args.stream_chunk_seconds == 12.0
    assert args.stream_overlap_seconds == 3.0
    # 0 selects the legacy whole-buffer mode and stays accepted.
    assert _validated_stream_args(["--stream-chunk-seconds", "0"]).stream_chunk_seconds == 0.0


def test_server_stream_flags_reject_invalid_config():
    for argv in (
        ["--stream-overlap-seconds", "-3"],                                   # issue #146
        ["--stream-chunk-seconds", "12", "--stream-overlap-seconds", "12"],   # overlap == chunk
        ["--stream-chunk-seconds", "5", "--stream-overlap-seconds", "6"],     # overlap > chunk
        ["--stream-chunk-seconds", "1e-9"],                                   # sub-sample window
        ["--stream-chunk-seconds", "-5"],
        ["--stream-chunk-seconds", "1e18"],                                   # counter overflow
        ["--min-chunk-seconds", "-1"],
        ["--partial-interval-seconds", "-1"],
    ):
        with pytest.raises(SystemExit, match="error:"):
            _validated_stream_args(argv)


def test_server_stream_flags_reject_non_finite_and_junk():
    # float() already rejects trailing junk; the finite-aware type now also
    # rejects nan/inf tokens (native CLI: parse_double_strict).
    for flag in ("--stream-overlap-seconds", "--stream-chunk-seconds",
                 "--min-chunk-seconds", "--partial-interval-seconds",
                 "--max-chunk-seconds"):
        with pytest.raises(SystemExit):
            _validated_stream_args([flag, "nan"])
        with pytest.raises(SystemExit):
            _validated_stream_args([flag, "inf"])
        with pytest.raises(SystemExit):
            _validated_stream_args([flag, "3abc"])
