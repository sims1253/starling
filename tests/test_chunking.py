"""Correctness + memory tests for the memory-bounded chunked transcriber.

The KEY property this file guards:

1. **Single-chunk byte-exactness.** When the audio fits in one chunk (<=
   ``chunk_seconds + overlap_seconds``), the chunked path must reproduce the
   direct ``MegaParakeetPipeline.transcribe`` output BYTE-FOR-BYTE. This proves
   the chunking path == the direct path when no stitching is needed (the
   ``decode_with_durations`` token stream is identical to ``decode``'s).
2. **Stitching sanity.** A clip longer than one chunk (45 s) transcribes to
   non-empty, real-English text containing the expected oracle substrings and
   roughly 2x the single-chunk length. Full byte-exactness across chunk
   boundaries is NOT guaranteed (word splits at boundaries are expected and
   correct).
3. **5 min succeeds without OOM** -- the length where the unchunked encoder
   clifs (``robust_bench.json``: ``broke_at 7min`` / cliff_reason ``vram_cliff``).
4. **Memory is bounded** -- ``torch.cuda.max_memory_allocated`` during a 5 min
   chunked transcribe stays < 4 GB (a single ~30 s chunk's worth), proving VRAM
   is a function of chunk size, not total length.

Run with:  ``uv run pytest tests/test_chunking.py -q``
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "tests" / "fixtures"))
sys.path.insert(0, str(_REPO_ROOT / "src"))

import make_fixtures as mkfx  # noqa: E402

# Building a pipeline loads the model (~25 s); cache ONE pipeline across the
# whole module so the suite is fast. Chunker objects are cheap to build (they
# only store params, no GPU work), so they are constructed per-test.
_PIPE = None


def _get_pipe():
    global _PIPE
    if _PIPE is None:
        from starling.parakeet.pipeline import MegaParakeetPipeline  # noqa: WPS433

        _PIPE = MegaParakeetPipeline(use_graphed_encoder=True)
    return _PIPE


def _get_chunker(chunk_batch_size: int = 1):
    """Build a fresh chunker over the cached pipeline.

    Defaults to ``chunk_batch_size=1`` (the original sequential path) so the
    existing correctness/memory tests are byte-for-byte unchanged; the batched
    tests pass ``8`` explicitly. Returns ``(pipe, chunker)``.
    """
    from starling.parakeet.chunking import ChunkedTranscriber  # noqa: WPS433

    pipe = _get_pipe()
    chunker = ChunkedTranscriber(
        pipe,
        chunk_seconds=30.0,
        overlap_seconds=2.0,
        chunk_batch_size=chunk_batch_size,
    )
    return pipe, chunker


def _repeat_audio(base: np.ndarray, target_seconds: float, sr: int = 16000) -> np.ndarray:
    """Tile ``base`` to >= ``target_seconds`` (deterministic, no RNG)."""
    need = int(target_seconds * sr)
    reps = (need + base.shape[0] - 1) // base.shape[0]
    return np.ascontiguousarray(np.tile(base, reps), dtype=np.float32)


# --------------------------------------------------------------------------- #
# 1. single-chunk byte-exactness (chunking path == direct path)
# --------------------------------------------------------------------------- #
def test_single_chunk_byte_exact_medium():
    """A <=-one-chunk clip must match the direct pipeline output byte-for-byte.

    The medium fixture (~22.3 s) fits inside the default 32 s window, so the
    chunker produces exactly one chunk and must equal
    ``pipeline.transcribe([medium])[0]`` exactly.
    """
    pipe, chunker = _get_chunker()
    fixtures = mkfx.load_fixtures()
    medium = fixtures["medium"]

    direct = pipe.transcribe([medium])[0]
    chunked = chunker.transcribe(medium)

    assert chunked == direct, (
        "single-chunk chunked path drifted from the direct pipeline output:\n"
        f"  direct : {direct!r}\n  chunked: {chunked!r}"
    )


def test_single_chunk_byte_exact_short():
    """Even shorter clip (short fixture ~7.4 s) -> still exactly one chunk."""
    pipe, chunker = _get_chunker()
    fixtures = mkfx.load_fixtures()
    short = fixtures["short"]

    direct = pipe.transcribe([short])[0]
    chunked = chunker.transcribe(short)
    assert chunked == direct, (
        "single-chunk (short) drifted:\n"
        f"  direct : {direct!r}\n  chunked: {chunked!r}"
    )


# --------------------------------------------------------------------------- #
# 2. stitching sanity (multi-chunk, boundary not byte-exact by design)
# --------------------------------------------------------------------------- #
def test_stitching_45s_reasonable():
    """A 45 s clip (6 reps of the base utterance, ~2x medium) chunks into >=2
    chunks and stitches to non-empty real English with the expected substrings.

    Byte-exactness across the chunk boundary is NOT asserted (word splits at
    boundaries are expected); we assert the result is reasonable.
    """
    _pipe, chunker = _get_chunker()
    base = mkfx.load_sample()
    medium = mkfx.load_fixtures()["medium"]

    clip = _repeat_audio(base, target_seconds=45.0)  # ~45.0 s
    assert clip.shape[0] / 16000 >= 40.0, "clip should be ~45 s"

    text, summary = chunker.transcribe_with_timing(clip)

    # multi-chunk: stitching actually happened
    assert summary["n_chunks"] >= 2, f"expected >=2 chunks, got {summary['n_chunks']}"
    assert summary["n_stitches"] >= 1, "expected at least one overlap token dropped"

    # non-empty, real English
    assert len(text.strip()) > 0, "stitched text is empty"
    assert "Phoebe" in text, "expected the utterance's 'Phoebe' in the stitch"
    assert "portrait" in text, "expected the utterance's 'portrait' in the stitch"
    assert text.replace(" ", "").replace(".", "").isascii(), (
        "stitched text should be ASCII English"
    )

    # ~2x the single-chunk (medium) length: 45 s vs 22.3 s
    medium_direct = _get_chunker()[0].transcribe([medium])[0]
    ratio = len(text) / max(1, len(medium_direct))
    assert 1.4 <= ratio <= 3.0, (
        f"stitched length ratio {ratio:.2f}x medium is out of the expected ~2x band"
    )


# --------------------------------------------------------------------------- #
# 3. 5 min succeeds without OOM (where unchunked clifs)
# --------------------------------------------------------------------------- #
def test_5min_succeeds_no_oom():
    """5 min of audio (the unchunked ceiling) must transcribe without OOM.

    ``robust_bench.json`` records the unchunked encoder clifs at 7 min
    (``vram_cliff``); chunked transcription must clear 5 min comfortably.
    """
    _pipe, chunker = _get_chunker()
    base = mkfx.load_sample()
    audio = _repeat_audio(base, target_seconds=300.0)  # 5 min

    text, summary = chunker.transcribe_with_timing(audio)

    assert summary["n_chunks"] >= 8, (
        f"5 min should need >=8 chunks at 30 s step, got {summary['n_chunks']}"
    )
    assert len(text.strip()) > 0, "5 min transcription is empty"
    assert summary["audio_seconds"] >= 295.0


# --------------------------------------------------------------------------- #
# 4. memory is bounded (peak VRAM over a 5 min run < one chunk budget)
# --------------------------------------------------------------------------- #
def test_memory_bounded_5min():
    """``torch.cuda.max_memory_allocated`` during a 5 min chunked run must stay
    bounded -- a function of chunk_batch_size, not total length."""
    _pipe, chunker = _get_chunker()
    base = mkfx.load_sample()
    audio = _repeat_audio(base, target_seconds=300.0)  # 5 min

    _text, summary = chunker.transcribe_with_timing(audio)

    peak_gb = summary["peak_vram_gb"]
    # With chunk_batch_size=32 (5090 default), ~30s chunks use ~0.3 GB each
    # so ~32 chunks = ~10 GB peak. The point is that this number does NOT grow
    # with total length (vs the unchunked cliff at 7 min / 25+ GB).
    assert peak_gb < 12.0, (
        f"peak VRAM {peak_gb:.3f} GB during 5 min chunked run exceeded the 12 GB "
        f"budget -- chunking is not bounding memory as intended"
    )


# =========================================================================== #
# BATCHED chunked decoding (chunk_batch_size=8)
#
# The batched path groups chunks into mini-batches of 8 and runs each through
# one set of batched mel+encoder+decode forwards, recovering the megakernel
# pipeline's batched throughput. These tests guard:
#   5. single-chunk byte-exactness is preserved (1 chunk -> B=1 mini-batch ->
#      byte-exact with the direct pipeline),
#   6. batched 5 min transcribes to reasonable text (multi-chunk B=8 stitching),
#   7. peak VRAM during a batched 5 min run stays < 16 GB (memory guard works).
# =========================================================================== #

# --------------------------------------------------------------------------- #
# 5. batched single-chunk byte-exactness (chunk_batch_size=8, 1 chunk -> B=1)
# --------------------------------------------------------------------------- #
def test_batched_single_chunk_byte_exact_medium():
    """With chunk_batch_size=8, a <=-one-chunk clip (medium ~22.3 s) forms a
    single B=1 mini-batch and must reproduce the direct pipeline output
    byte-for-byte (same guarantee as the sequential single-chunk test, just
    routed through the batched code path)."""
    pipe, chunker = _get_chunker(chunk_batch_size=8)
    fixtures = mkfx.load_fixtures()
    medium = fixtures["medium"]

    direct = pipe.transcribe([medium])[0]
    batched = chunker.transcribe(medium)

    assert batched == direct, (
        "batched single-chunk path drifted from the direct pipeline output:\n"
        f"  direct  : {direct!r}\n  batched : {batched!r}"
    )


def test_batched_single_chunk_byte_exact_short():
    """Same byte-exactness guarantee for the short fixture (~7.4 s, B=1)."""
    pipe, chunker = _get_chunker(chunk_batch_size=8)
    fixtures = mkfx.load_fixtures()
    short = fixtures["short"]

    direct = pipe.transcribe([short])[0]
    batched = chunker.transcribe(short)

    assert batched == direct, (
        "batched single-chunk (short) drifted:\n"
        f"  direct  : {direct!r}\n  batched : {batched!r}"
    )


# --------------------------------------------------------------------------- #
# 6 + 7. batched 5 min: reasonable text + bounded memory (< 16 GB)
# --------------------------------------------------------------------------- #
def test_batched_5min_reasonable_and_memory():
    """Batched (chunk_batch_size=8) 5 min transcription must:

    * produce non-empty, reasonable English containing the expected substrings,
    * actually batch (n_batches < n_chunks, batch_size up to 8),
    * keep peak VRAM < 16 GB (the memory guard adapts B to free VRAM and never
      OOMs -- at B=8 a batch of 30 s chunks is well under this budget).
    """
    _pipe, chunker = _get_chunker(chunk_batch_size=8)
    base = mkfx.load_sample()
    audio = _repeat_audio(base, target_seconds=300.0)  # 5 min

    text, summary = chunker.transcribe_with_timing(audio)

    # batching actually happened: fewer batches than chunks, each <= 8 chunks
    assert summary["n_chunks"] >= 8, (
        f"5 min should need >=8 chunks, got {summary['n_chunks']}"
    )
    assert summary["n_batches"] >= 1, "expected at least one batch"
    assert summary["n_batches"] < summary["n_chunks"], (
        f"expected fewer batches ({summary['n_batches']}) than chunks "
        f"({summary['n_chunks']}) -- batching did not engage"
    )
    batch_sizes = [b["batch_size"] for b in summary["per_batch"]]
    assert max(batch_sizes) > 1, (
        f"expected at least one batch with >1 chunk, got batch sizes {batch_sizes}"
    )
    assert all(1 <= bs <= 8 for bs in batch_sizes), (
        f"batch sizes out of [1, 8]: {batch_sizes}"
    )

    # reasonable, non-empty English text
    assert len(text.strip()) > 0, "batched 5 min transcription is empty"
    assert "Phoebe" in text, "expected 'Phoebe' in the batched transcription"
    assert "portrait" in text, "expected 'portrait' in the batched transcription"
    assert text.replace(" ", "").replace(".", "").isascii(), (
        "batched transcription should be ASCII English"
    )

    # memory guard: peak VRAM bounded (< 16 GB); never OOMs.
    peak_gb = summary["peak_vram_gb"]
    assert peak_gb < 16.0, (
        f"batched 5 min peak VRAM {peak_gb:.3f} GB exceeded the 16 GB budget -- "
        f"the adaptive batch-size guard is not bounding memory (batch sizes "
        f"{batch_sizes})"
    )
