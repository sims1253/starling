"""Unit tests for fixed-window overlapping-chunk streaming (no model needed).

The transcription is faked deterministically: the sample buffer is filled with
its own indices (``samples[i] == i``), so a fake transcriber can read a window's
absolute start from ``window[0]`` and return the ground-truth words whose time
falls in that window. This exercises the real chunk/finalize/stitch logic and
verifies the reconstructed transcript equals the ground truth despite overlap.
"""

from __future__ import annotations

import numpy as np

from starling.stream_chunk import ChunkStreamer, stitch_words

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
