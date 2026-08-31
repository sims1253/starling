"""S1-mini pipeline gates: golden byte-exactness on the GPU path.

The goldens under ``golden/s1/`` are the eager stock-transformers path (the
model-card quickstart) on the transcript fixtures; the CUDA-graph
``NormalizePipeline`` must reproduce the generated token ids EXACTLY (greedy
is deterministic; the graph changes only timing).

GPU-gated: skips when CUDA or the HF snapshot is unavailable. The CPU-only
prompt-contract tests (control line, chunking, budget formula) live in
tests/test_s1_config.py, which CI runs on every platform.

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
    # Length pin: a dropped trailing stop token would otherwise pass the
    # prefix compare (the EOS decodes to nothing under skip_special_tokens).
    assert n == ref_ids.numel(), (
        f"{tier}: generated {n} tokens, golden has {ref_ids.numel()} "
        "(truncated stop token?)"
    )
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
