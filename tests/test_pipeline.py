"""Correctness gate for the end-to-end MegaPipeline.

The fused encoder and fused LLM decoder are both byte-exact vs the eager
reference, so the full pipeline must reproduce the golden greedy transcript and
token ids EXACTLY.  The merge step (build_inputs_embeds) is checked separately
against ``golden/inputs_embeds.pt`` to catch merge/scatter bugs early.

Run with:  uv run pytest tests/test_pipeline.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from starling.granite.audio import build_inputs, load_sample_audio  # noqa: E402
from starling.granite.golden import load_golden, load_golden_text  # noqa: E402
from starling.granite.pipeline import MegaPipeline  # noqa: E402

# Loading the speech model is expensive (~5s); cache across tests.
_MODEL = None
_PROC = None
_INPUTS: dict | None = None


def _get_model_and_processor():
    global _MODEL, _PROC
    if _MODEL is None:
        from starling.granite.loader import load_model_and_processor

        _MODEL, _PROC = load_model_and_processor(attn_impl="eager")
    return _MODEL, _PROC


def _get_inputs() -> dict:
    global _INPUTS
    if _INPUTS is None:
        _, proc = _get_model_and_processor()
        wav, sr = load_sample_audio()
        _INPUTS = build_inputs(proc, wav)
    return _INPUTS


@pytest.fixture(scope="module")
def pipeline():
    model, proc = _get_model_and_processor()
    return MegaPipeline(model, proc, encoder_mode="cudagraph", use_fused_llm=True)


# --------------------------------------------------------------------------- #
# merge correctness (catches scatter/dtype bugs before running the decoder)
# --------------------------------------------------------------------------- #
def test_inputs_embeds_matches_golden(pipeline):
    """Constructed inputs_embeds must match golden within 1e-3 (byte-exact)."""
    inputs = _get_inputs()
    golden_ie = load_golden("inputs_embeds.pt").to("cuda", torch.bfloat16)

    with torch.inference_mode():
        _enc, audio_embeds = pipeline.encode_audio(inputs["input_features"])
        mine = pipeline.build_inputs_embeds(
            inputs["input_ids"],
            audio_embeds,
            inputs.get("input_features_mask"),
        )

    assert mine.shape == golden_ie.shape, (
        f"shape mismatch: {tuple(mine.shape)} != {tuple(golden_ie.shape)}"
    )
    assert mine.dtype == torch.bfloat16, f"dtype {mine.dtype} != bf16"
    diff = (mine.float() - golden_ie.float()).abs().max().item()
    print(f"[inputs_embeds] max abs diff vs golden = {diff:.6e}")
    assert diff < 1e-3, f"inputs_embeds max abs diff {diff:.4e} >= 1e-3"


# --------------------------------------------------------------------------- #
# end-to-end correctness
# --------------------------------------------------------------------------- #
def test_generated_tokens_match_golden(pipeline):
    """Generated ids must equal golden greedy_ids[:, 271:] EXACTLY."""
    inputs = _get_inputs()
    golden_gen = load_golden("greedy_ids.pt")[0, 271:]  # (100,)

    _text, ids = pipeline.transcribe(
        inputs["input_features"],
        inputs["input_ids"],
        inputs.get("input_features_mask"),
        max_new_tokens=100,
    )

    assert ids.shape == (1, golden_gen.shape[0]), (
        f"expected {golden_gen.shape[0]} tokens, got {ids.shape[1]}"
    )
    assert (ids[0] == golden_gen).all(), (
        f"token mismatch: first diff at "
        f"{(ids[0] != golden_gen).nonzero()[0].item() if (ids[0] != golden_gen).any() else -1}"
    )


def test_transcript_matches_golden(pipeline):
    """Decoded transcript must match the golden ASSISTANT response exactly."""
    inputs = _get_inputs()
    golden_text = load_golden_text().strip()

    text, _ids = pipeline.transcribe(
        inputs["input_features"],
        inputs["input_ids"],
        inputs.get("input_features_mask"),
        max_new_tokens=100,
    )

    # golden_text is the full chat-templated decode (USER: ... ASSISTANT: ...).
    # transcribe returns only the generated response body.
    assert "ASSISTANT:" in golden_text, "golden text must contain ASSISTANT marker"
    golden_response = golden_text.split("ASSISTANT:", 1)[1].strip()
    assert text.strip() == golden_response, (
        f"transcript mismatch:\n  golden: {golden_response[:100]!r}\n"
        f"  ours:   {text.strip()[:100]!r}"
    )


if __name__ == "__main__":
    # Allow running directly: .venv/bin/python tests/test_pipeline.py
    model, proc = _get_model_and_processor()
    pipe = MegaPipeline(model, proc, encoder_mode="cudagraph", use_fused_llm=True)
    test_inputs_embeds_matches_golden(pipe)
    print("[manual] test_inputs_embeds_matches_golden PASSED")
    test_generated_tokens_match_golden(pipe)
    print("[manual] test_generated_tokens_match_golden PASSED")
    test_transcript_matches_golden(pipe)
    print("[manual] test_transcript_matches_golden PASSED")
