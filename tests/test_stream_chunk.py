"""Unit tests for fixed-window overlapping-chunk streaming (no model needed).

The transcription is faked deterministically: the sample buffer is filled with
its own indices (``samples[i] == i``), so a fake transcriber can read a window's
absolute start from ``window[0]`` and return the ground-truth words whose time
falls in that window. This exercises the real chunk/finalize/stitch logic and
verifies the reconstructed transcript equals the ground truth despite overlap.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pytest

from starling.stream_chunk import (
    STITCH_TIME_TOLERANCE_SECONDS,
    ChunkStreamer,
    TimedWord,
    Transcript,
    max_plausible_words,
    stitch_timed,
    stitch_words,
    stream_window_config_error,
    suppress_loops,
    suppress_loops_kept,
    suppress_preview_loops,
    suppress_preview_loops_kept,
    voiced_seconds,
)

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
    # 2-word overlap "hello world" vs "Hello World." -> deduped by
    # normalization; the cut is the middle (lower) matched word, so the
    # overlap after it comes from the new window (issue #357).
    out = stitch_words(["say", "hello", "world."], ["Hello", "World", "again"])
    assert out == ["say", "hello", "World", "again"]


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
    calls: list[tuple[str, int]] = []

    def tx(window):
        assert 0 < len(window) <= cs.chunk
        start = int(window[0])
        assert 0 <= start and start + len(window) <= len(samples)
        assert start == cs.call_start
        calls.append((cs.call_kind, len(window)))
        return "w"

    assert cs.step(samples, 1.0, tx) is not None
    assert cs.flush(samples, tx) is not None
    # Three windows, a preview of the 3 s overlap tail (the take is past the
    # 5 s first-partial minimum, issue #357), and the 3 s flush tail. The
    # audio is loud and each window says one word, so every window is also
    # decoded again (issue #357); those calls stay in range too.
    assert [n for k, n in calls if k != "redecode"] == [12 * SR, 12 * SR, 12 * SR, 3 * SR, 3 * SR]
    assert any(k == "redecode" for k, _ in calls)


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


# --------------------------------------------------------------------------- #
# preview cadence and coalescing (issue #357; lockstep with the C++ tests)
# --------------------------------------------------------------------------- #
def _small(**kw):
    cfg = dict(sample_rate=SR, chunk_seconds=1.0, overlap_seconds=0.25,
               min_seconds=0.5, partial_interval_seconds=0.0)
    cfg.update(kw)
    return ChunkStreamer(**cfg)


def test_preview_minimum_gates_the_take_not_the_tail():
    cs = _small()
    lens: list[int] = []

    def tx(window):
        lens.append(len(window))
        return "w"

    assert cs.step(np.zeros(7999, np.float32), 1.0, tx) is None
    assert lens == []
    assert cs.step(np.zeros(8000, np.float32), 2.0, tx) is not None
    assert lens == [8000]
    # A window commit emits its committed text and restarts the interval
    # (it covers the whole buffer), so no preview runs right behind it.
    lens.clear()
    assert cs.step(np.zeros(16000 + 1600, np.float32), 3.0, tx) is not None
    assert lens == [16000]
    # The tail is the overlap plus 0.1 s, below the minimum, yet still
    # previewed once the interval passes: commits never stall previews.
    assert cs.step(np.zeros(16000 + 1600, np.float32), 4.0, tx) is not None
    assert lens == [16000, 16000 + 1600 - 12000]


def test_preview_coalesced_when_newer_audio_is_queued():
    cs = _small()
    lens: list[int] = []

    def tx(window):
        lens.append(len(window))
        return "w x"

    pending = [True]
    assert cs.step(np.zeros(20000, np.float32), 1.0, tx, lambda: pending[0]) is None
    assert lens == [16000] and cs.boundary == 12000 and cs.coalesced == 1
    pending[0] = False
    # The next step previews the newest tail and carries the owed update.
    assert cs.step(np.zeros(22000, np.float32), 1.1, tx, lambda: pending[0]) is not None
    assert lens == [16000, 10000]
    assert cs.flush(np.zeros(22000, np.float32), tx) is not None
    assert lens[-1] == 22000 - 12000


def test_window_commit_does_not_force_a_throttled_preview():
    # A window commit inside the interval emits the committed text at once
    # but does not decode the tail (the preview duty bound holds).
    cs = _small(partial_interval_seconds=10.0)
    lens: list[int] = []

    def tx(window):
        lens.append(len(window))
        return "w x"

    assert cs.step(np.zeros(8000, np.float32), 20.0, tx) is not None
    assert lens == [8000]
    assert cs.step(np.zeros(20000, np.float32), 21.0, tx) == "w x"
    assert lens == [8000, 16000]
    assert cs.step(np.zeros(21000, np.float32), 21.5, tx) is None
    # The commit restarted the interval: 10 s after the commit, not after
    # the last preview (20.0).
    assert cs.step(np.zeros(21000, np.float32), 30.5, tx) is None
    assert cs.step(np.zeros(21000, np.float32), 31.5, tx) is not None
    assert lens == [8000, 16000, 21000 - 12000]


def test_window_decode_time_counts_toward_the_preview_gap():
    # A 150 ms window decode at a 0.1 s interval: the step clock moves past
    # the window, so a step 0.2 s (fake) after the last one began is still
    # inside the gap the commit restarted.
    import time as _time

    cs = _small(min_seconds=0.1, partial_interval_seconds=0.1)
    lens: list[int] = []

    def tx(window):
        lens.append(len(window))
        if len(window) == 16000:
            _time.sleep(0.15)
        return "w"

    samples = np.zeros(16000 + 800, np.float32)
    assert cs.step(samples, 10.0, tx) is not None  # window commit only
    assert lens == [16000]
    assert cs.step(samples, 10.2, tx) is None      # 0.05 s after the window
    assert cs.step(samples, 10.3, tx) is not None  # gap over: preview
    assert lens == [16000, 16800 - 12000]


def test_preview_interval_adapts_to_preview_cost():
    import time as _time

    cs = _small(min_seconds=0.1)
    calls = []

    def tx(window):
        calls.append(len(window))
        _time.sleep(0.03)
        return "w"

    assert cs.step(np.zeros(4000, np.float32), 10.0, tx) is not None
    assert 0.06 <= cs.effective_interval < 1.0
    assert cs.step(np.zeros(5000, np.float32), 10.01, tx) is None
    assert cs.step(np.zeros(5000, np.float32), 10.001 + cs.effective_interval, tx) is not None
    assert len(calls) == 2
    cs.reset()
    assert cs.effective_interval == 0.0


def test_preview_policy_validation():
    from starling.stream_chunk import preview_policy_error

    ok = dict(sample_rate=SR, min_seconds=1.0, interval_seconds=0.5)
    assert preview_policy_error(**ok) is None
    for bad in ({"min_seconds": -1.0}, {"interval_seconds": -0.5},
                {"min_seconds": float("nan")}, {"interval_seconds": float("inf")},
                {"min_seconds": 1e9}):
        assert preview_policy_error(**{**ok, **bad}) is not None
    cs = _small()
    with pytest.raises(ValueError):
        cs.set_preview_policy(-1.0, 0.0)
    cs.set_preview_policy(2.0, 0.25)
    assert cs.min == 2 * SR and cs.partial_interval == 0.25


# ---- window plausibility and loops (issue #357) -------------------------------


def test_voiced_seconds_counts_loud_frames_only():
    silence = np.zeros(2 * SR, dtype=np.float32)
    loud = np.full(3 * SR, 0.1, dtype=np.float32)
    assert voiced_seconds(silence, SR) == 0.0
    assert voiced_seconds(loud, SR) == 3.0
    assert voiced_seconds(np.concatenate([silence, loud, silence]), SR) == 3.0
    # Below -60 dBFS is never speech, whatever the floor.
    assert voiced_seconds(np.full(2 * SR, 1e-4, dtype=np.float32), SR) == 0.0
    # A span of continuous speech is not its own floor (floor <= -45 dBFS).
    assert voiced_seconds(np.full(2 * SR, 0.05, dtype=np.float32), SR) == 2.0
    assert voiced_seconds(np.zeros(100, dtype=np.float32), SR) == 0.0


def test_suppress_loops_collapses_only_what_does_not_fit():
    # Within the bound (7 words/s + 4) nothing changes, repeats included.
    assert suppress_loops("no no no no no".split(), 1.0) == "no no no no no".split()
    # Over it, the longest run goes first, and only until the text fits:
    # the real "no no no no no" survives next to a loop.
    words = "no no no no no I said".split() + ["la"] * 100
    assert suppress_loops(words, 2.0) == "no no no no no I said la la".split()
    loop = "so a little bit of a little bit of a little bit of a little bit of it".split()
    assert suppress_loops(loop, 0.5) == "so a little bit of a little".split()  # bound 7
    # Punctuation and case do not hide a loop.
    assert suppress_loops("Again, again again. again again".split(), 0.0) == ["Again,", "again"]
    # No repeats left: cut at the bound.
    assert suppress_loops([f"w{i}" for i in range(30)], 1.0) == [f"w{i}" for i in range(11)]
    assert suppress_loops([], 0.0) == []


def test_max_plausible_words_bounds_a_window():
    assert max_plausible_words(12.0) == 88
    assert max_plausible_words(0.0) == 4


def test_preview_never_shows_a_loop():
    # A 900-word looping preview (seen live on the long take, PR #430) is
    # cut to what its audio can hold; the next preview is not affected.
    cs = ChunkStreamer(**_cfg(min_seconds=0.0, partial_interval_seconds=0.0))
    samples = np.zeros(6 * SR, dtype=np.float32)
    loop = "a little bit of " * 225
    text = cs.step(samples, 1.0, lambda w: loop)
    assert text == "a little bit of a little bit of"
    assert len(text.split()) <= max_plausible_words(6.0)


def test_real_repetition_in_a_window_survives():
    # Repeated speech within what the audio can hold is never collapsed.
    cs = ChunkStreamer(**_cfg(min_seconds=0.0, partial_interval_seconds=0.0))
    samples = np.full(2 * SR, 0.1, dtype=np.float32)
    said = "no no no no no I said no"
    assert cs.step(samples, 1.0, lambda w: said) == said
    assert cs.flush(samples, lambda w: said) == said


def test_redecode_needs_audio_before_the_boundary_to_be_kept():
    # The buffer may drop audio only before retain_from: the boundary minus
    # the largest re-decode shift (0.75 x the 3 s overlap).
    cs = ChunkStreamer(**_cfg())
    assert cs.lookback == int(2.25 * SR)
    assert cs.retain_from == 0
    cs.boundary = 9 * SR
    assert cs.retain_from == 9 * SR - cs.lookback


# ---- word timestamps (issue #357) -------------------------------------------


def _timed_tx(answers: dict):
    """A transcriber answering by (call kind, window start in s) with
    "word@start-end ..." scripts (times from the window start)."""
    def tx(window: np.ndarray):
        return _transcript(answers[(cs_ref[0].call_kind, cs_ref[0].call_start / SR)].split())
    cs_ref: list = []
    return tx, cs_ref


def test_preview_is_stitched_by_time():
    # A window heard "in fact" at its edge; the preview of the tail heard
    # "fact" again at the same time: the preview shows it once.
    cs = ChunkStreamer(**_cfg(min_seconds=0.0, partial_interval_seconds=0.0))
    tx, ref = _timed_tx({
        ("window", 0.0): "so@1-1.2 in@10.4-10.6 fact@10.8-11.1",
        ("preview", 9.0): "fact@1.84-2.1 there's@2.5-2.8 more@3.2-3.4",
    })
    ref.append(cs)
    samples = np.zeros(13 * SR, dtype=np.float32)  # silence is never sparse
    # The step commits the window, then previews the tail from 9 s.
    assert cs.step(samples, 1.0, tx) == "so in fact there's more"
    assert cs.committed == ["so", "in", "fact"]


def test_word_times_that_do_not_match_the_text_are_ignored():
    # Words that are not the text's split() leave the window untimed: the
    # text alignment, which keeps a single shared word twice.
    cs = ChunkStreamer(**_cfg())
    calls = iter([
        _transcript("so@1-1.2 in@10.4-10.6 fact@10.8-11.1".split()),
        Transcript("fact there's more", (TimedWord("fact", 1.84, 2.1),)),
    ])
    out = cs.flush(np.zeros(20 * SR, dtype=np.float32), lambda w: next(calls))
    assert out == "so in fact fact there's more"
    assert cs.spans[-3:] == [None, None, None]


def test_loop_suppression_keeps_word_times_with_their_words():
    # A looping flush tail whose re-decodes loop too keeps the suppressed
    # words, each still at its own time.
    cs = ChunkStreamer(**_cfg())
    loop = " ".join(f"la@{0.1 * k:.1f}-{0.1 * k + 0.05:.2f}" for k in range(40)) + " end@4.5-4.8"
    out = cs.flush(np.zeros(5 * SR, dtype=np.float32), lambda w: _transcript(loop.split()))
    assert out == "la la end"
    assert cs.spans == [(0, 800), (1600, 2400), (72000, 76800)]


def test_suppress_kept_indices_match_the_words():
    words = "and then a little bit of a little bit of a little bit of a so on".split()
    assert [words[i] for i in suppress_loops_kept(words, 1)] == suppress_loops(words, 1)
    assert ([words[i] for i in suppress_preview_loops_kept(words, 10)]
            == suppress_preview_loops(words, 10))


# ---- stitch parity fixture (issue #357) --------------------------------------

_CASES = Path(__file__).resolve().parent / "fixtures" / "stream_stitch_cases.txt"


def _parse_cases() -> list[dict]:
    """Blocks of tests/fixtures/stream_stitch_cases.txt (format in its header)."""
    cases: list[dict] = []
    for raw in _CASES.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        tag, _, rest = line.partition(" ")
        if tag in ("stitch", "tstitch", "suppress", "preview", "stream"):
            cases.append({"op": tag, "args": rest.split(), "tx": []})
        elif tag in ("<", ">", "="):
            cases[-1][tag] = rest.split()
        elif tag == "audio":
            cases[-1]["audio"] = [float(x) for x in rest.split()]
        elif tag == "tx":
            kind, start, length, *text = rest.split()
            cases[-1]["tx"].append((kind, start, length, " ".join(text)))
        elif tag == "calls":
            cases[-1]["calls"] = rest.split()
        else:
            raise AssertionError(f"bad fixture line: {raw!r}")
    return cases


def _timed(token: str) -> tuple[str, float, float]:
    """A fixture word "word@start-end" (seconds)."""
    word, _, times = token.rpartition("@")
    start, _, end = times.partition("-")
    return word, float(start), float(end or start)


def _transcript(tokens: list[str]) -> Transcript:
    """Scripted engine text with word timestamps (every token "word@start-end")."""
    words = tuple(TimedWord(*_timed(t)) for t in tokens)
    return Transcript(" ".join(w.word for w in words), words)


def _stitch_timed_case(case: dict) -> list[str]:
    """tstitch: the chunker's timed stitch over the given shared audio; no
    shared word concatenates."""
    committed = [_timed(t) for t in case.get("<", [])]
    new = [_timed(t) for t in case.get(">", [])]
    cut = stitch_timed([w for w, _, _ in committed],
                       [int(math.floor(a * SR + 0.5)) for _, a, _ in committed],
                       [w for w, _, _ in new],
                       [int(math.floor(a * SR + 0.5)) for _, a, _ in new],
                       lo=int(math.floor(float(case["args"][1]) * SR + 0.5)),
                       hi=int(math.floor(float(case["args"][2]) * SR + 0.5)),
                       tolerance=int(STITCH_TIME_TOLERANCE_SECONDS * SR))
    keep, skip = cut if cut is not None else (len(committed), 0)
    return [w for w, _, _ in committed[:keep]] + [w for w, _, _ in new[skip:]]


def _seconds(sample: int) -> str:
    return f"{sample / SR:g}"


def _replay_stream(case: dict) -> tuple[list[str], list[str]]:
    chunk_s, overlap_s = (float(x) for x in case["args"][1:3])
    audio = case["audio"]
    samples = np.concatenate([np.full(int(round(audio[i] * SR)), audio[i + 1], dtype=np.float32)
                              for i in range(0, len(audio), 2)])
    cs = ChunkStreamer(sample_rate=SR, chunk_seconds=chunk_s, overlap_seconds=overlap_s,
                       min_seconds=0.0, partial_interval_seconds=0.0)
    calls: list[str] = []

    used: set[int] = set()

    def tx(window: np.ndarray) -> Optional[Union[str, Transcript]]:
        start, n = cs.call_start, len(window)
        calls.append(f"{cs.call_kind}@{_seconds(start)}+{_seconds(n)}")
        hits = [k for k, (kind, a, length, _) in enumerate(case["tx"])
                if k not in used and kind in ("*", cs.call_kind)
                and (a == "*" or int(round(float(a) * SR)) == start)
                and (length == "*" or int(round(float(length) * SR)) == n)]
        assert hits, f"no scripted text for {calls[-1]}"
        if len(hits) > 1:
            used.add(hits[0])
        text = case["tx"][hits[0]][3]
        if text == "BUSY":
            return None
        return _transcript(text.split()) if "@" in text else text

    final = cs.flush(samples, tx)
    assert final is not None
    return final.split(), calls


@pytest.mark.parametrize("case", _parse_cases(), ids=lambda c: f"{c['op']}-{c['args'][0]}")
def test_stitch_fixture_case(case):
    # The native server replays the same file (stream_session_test.cpp,
    # test_stitch_fixture_cases); both must match every expected line.
    if case["op"] == "stitch":
        assert stitch_words(case.get("<", []), case.get(">", [])) == case["="]
    elif case["op"] == "tstitch":
        assert _stitch_timed_case(case) == case.get("=", [])
    elif case["op"] == "suppress":
        assert suppress_loops(case["<"], float(case["args"][1])) == case["="]
    elif case["op"] == "preview":
        assert suppress_preview_loops(case["<"], float(case["args"][1])) == case["="]
    else:
        final, calls = _replay_stream(case)
        assert final == case["="]
        assert calls == case["calls"]
