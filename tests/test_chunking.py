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
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "tests" / "fixtures"))
sys.path.insert(0, str(_REPO_ROOT / "src"))

import make_fixtures as mkfx  # noqa: E402

# Building a pipeline loads the model (~25 s); cache ONE pipeline + chunker
# across the whole module so the suite is fast.
_PIPE = None
_CHUNKER = None


def _get_chunker():
    global _PIPE, _CHUNKER
    if _CHUNKER is None:
        from megapar.parakeet.pipeline import MegaParakeetPipeline  # noqa: WPS433
        from megapar.parakeet.chunking import ChunkedTranscriber  # noqa: WPS433

        _PIPE = MegaParakeetPipeline(use_graphed_encoder=True)
        _CHUNKER = ChunkedTranscriber(_PIPE, chunk_seconds=30.0, overlap_seconds=2.0)
    return _PIPE, _CHUNKER


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
    bounded (< 4 GB) -- a single ~30 s chunk's worth, independent of total
    length."""
    _pipe, chunker = _get_chunker()
    base = mkfx.load_sample()
    audio = _repeat_audio(base, target_seconds=300.0)  # 5 min

    _text, summary = chunker.transcribe_with_timing(audio)

    peak_gb = summary["peak_vram_gb"]
    # One ~30 s chunk is ~1.5-2 GB; allow generous headroom to 4 GB. The point
    # is that this number does NOT grow with total length (vs the unchunked
    # cliff at 7 min / 25 GB).
    assert peak_gb < 4.0, (
        f"peak VRAM {peak_gb:.3f} GB during 5 min chunked run exceeded the 4 GB "
        f"budget -- chunking is not bounding memory as intended"
    )
