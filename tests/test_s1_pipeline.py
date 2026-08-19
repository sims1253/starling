"""S1-mini pipeline gates: golden byte-exactness + prompt-contract behavior.

The goldens under ``golden/s1/`` are the eager stock-transformers path (the
model-card quickstart) on the transcript fixtures; the CUDA-graph
``NormalizePipeline`` must reproduce the generated token ids EXACTLY (greedy
is deterministic; the graph changes only timing).

GPU-gated: skips when CUDA or the HF snapshot is unavailable.

Run with:  uv run pytest tests/test_s1_pipeline.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "tests" / "fixtures"))
sys.path.insert(0, str(_REPO_ROOT / "src"))

import s1_transcripts as fx  # noqa: E402

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():  # noqa: SIM108 -- mirrors test_qwen3_pipeline
    pytest.skip("CUDA required for S1-mini pipeline tests", allow_module_level=True)
if not _REPO_ROOT.joinpath("golden", "s1").exists():
    pytest.skip(
        "golden/s1 absent (run: uv run python -m starling.s1.golden)",
        allow_module_level=True,
    )


def _pipeline():
    from starling.s1.pipeline import NormalizePipeline

    return NormalizePipeline.from_pretrained()


# --------------------------------------------------------------------- #
# CPU-only: prompt contract
# --------------------------------------------------------------------- #
def test_control_line_validates_values() -> None:
    from starling.s1.config import control_line

    assert control_line() == (
        "[Styling: semi-formal] [Structure: prose] [Context: general]")
    assert control_line("casual", "lists", "email") == (
        "[Styling: casual] [Structure: lists] [Context: email]")
    with pytest.raises(ValueError):
        control_line(styling="pirate")
    with pytest.raises(ValueError):
        control_line(structure="table")
    with pytest.raises(ValueError):
        control_line(context="meeting")


def test_chunk_transcript_short_input_passthrough() -> None:
    from starling.s1.pipeline import chunk_transcript

    t = "um so like one two three"
    assert chunk_transcript(t, max_words=600) == [t]
    # no-punctuation run longer than the budget splits at word boundaries
    long_t = " ".join(["word"] * 10)
    parts = chunk_transcript(long_t, max_words=4)
    assert parts == ["word word word word", "word word word word", "word word"]
    # punctuated text splits at sentence boundaries when they fit the budget
    assert chunk_transcript("a b. c d. e f.", max_words=2) == ["a b.", "c d.", "e f."]
    # ... and falls back to word windows when a sentence exceeds the budget
    assert chunk_transcript("a b. c d.", max_words=1) == ["a", "b.", "c", "d."]


def test_max_new_tokens_formula() -> None:
    from starling.s1.config import max_new_tokens_for

    assert max_new_tokens_for(100) == 162  # 1.3*100 + 32


# --------------------------------------------------------------------- #
# GPU: byte-exact vs golden
# --------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def pipeline():
    pipe = _pipeline()
    yield pipe
    del pipe
    import torch

    torch.cuda.empty_cache()


@pytest.mark.parametrize("tier", ["short", "medium", "long"])
def test_generated_ids_match_golden(pipeline, tier: str) -> None:
    """The CUDA-graph pipeline reproduces the stock greedy ids exactly."""

    from starling.s1.golden import GREEDY_IDS, load_golden

    ref_ids = load_golden(GREEDY_IDS.format(tier=tier))  # 1-D generated ids
    _, ids = pipeline.normalize(fx.LENGTH_TIERS[tier])
    n = ids.shape[1]
    assert (ids[0][:n] == ref_ids[:n]).all(), (
        f"{tier}: generated ids diverge from golden "
        f"({int((ids[0][:n] != ref_ids[:n]).sum())}/{n} mismatched)"
    )


@pytest.mark.parametrize("tier", ["short", "medium", "long"])
def test_transcript_matches_golden_text(pipeline, tier: str) -> None:
    from starling.s1.golden import GREEDY_TEXT, load_golden_text

    ref = load_golden_text(GREEDY_TEXT.format(tier=tier))
    text, _ = pipeline.normalize(fx.LENGTH_TIERS[tier])
    assert text == ref


@pytest.mark.parametrize(
    "transcript,styling,structure,context,expected",
    fx.QUALITY_CASES,
)
def test_quality_cases(pipeline, transcript, styling, structure, context, expected) -> None:
    """Curated cases with expected outputs (model-card provenance, re-verified
    against the shipped weights)."""
    got, _ = pipeline.normalize(
        transcript, styling=styling, structure=structure, context=context)
    assert got == expected


def test_overlong_transcript_rejected(pipeline) -> None:
    from starling.s1.pipeline import chunk_transcript

    huge = " ".join(["word"] * 1200)  # ~1500+ tokens with the template
    with pytest.raises(ValueError, match="trained max"):
        pipeline.normalize(huge)
    # ... and the documented remedy chunks it under the limit
    parts = chunk_transcript(huge, max_words=600)
    assert all(p == parts[0] or len(p.split()) <= 600 for p in parts)
