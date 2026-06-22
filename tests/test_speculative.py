"""Correctness gate for the self-speculative decoding path.

The speculative path uses the encoder's BPE CTC head to draft tokens, then
greedy-verifies them against the LLM.  Because greedy-verify-of-a-greedy-oracle
produces byte-identical output, the speculative transcript must match the
non-speculative path AND the golden reference EXACTLY.

Run with:  uv run pytest tests/test_speculative.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from megapar.audio import build_inputs, load_sample_audio  # noqa: E402
from megapar.golden import load_golden, load_golden_text  # noqa: E402
from megapar.pipeline import MegaPipeline  # noqa: E402
from megapar.speculative import CTCBPEDraft, load_out_llm  # noqa: E402

# Reuse the model cached by test_pipeline.py.
_MODEL = None
_PROC = None
_INPUTS: dict | None = None


def _get_model_and_processor():
    global _MODEL, _PROC
    if _MODEL is None:
        from megapar.loader import load_model_and_processor

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
# draft sanity (recognizably the golden utterance, NOT garbage)
# --------------------------------------------------------------------------- #
def test_draft_is_sensible(pipeline):
    """The CTC BPE draft must be recognizably the golden utterance.

    The draft lacks capitalization and punctuation (the CTC head predicts raw
    BPE content), so we check WORD-LEVEL overlap, not exact match.
    """
    inputs = _get_inputs()
    out_llm = load_out_llm(device="cuda", dtype=torch.bfloat16)
    draft_ext = CTCBPEDraft(pipeline.fused_encoder, out_llm)

    with torch.inference_mode():
        mid_h, enc_hidden = draft_ext.encode_with_mid(inputs["input_features"])
        draft = draft_ext.draft(enc_hidden, mid_h)

    assert len(draft) > 10, f"draft too short: {len(draft)} tokens"

    draft_text = pipeline.processor.tokenizer.decode(draft, skip_special_tokens=True)
    golden_text = load_golden_text().strip()
    golden_resp = golden_text.split("ASSISTANT:", 1)[1].strip()

    # Word-level overlap: the draft must share significant vocabulary with the
    # golden response (same utterance, different formatting).
    draft_words = set(draft_text.lower().split())
    golden_words = set(golden_resp.lower().split())
    overlap = draft_words & golden_words
    overlap_ratio = len(overlap) / max(len(golden_words), 1)

    print(f"[draft] {len(draft)} tokens, word overlap = {overlap_ratio:.1%}")
    print(f"[draft] draft (first 120 chars): {draft_text[:120]!r}")

    assert overlap_ratio > 0.5, (
        f"draft word overlap too low: {overlap_ratio:.1%}. "
        f"Draft may be garbage.\n  draft: {draft_text[:200]!r}\n"
        f"  golden: {golden_resp[:200]!r}"
    )


# --------------------------------------------------------------------------- #
# byte-exact correctness (greedy-verify guarantee)
# --------------------------------------------------------------------------- #
def test_speculative_matches_greedy(pipeline):
    """Speculative output must match non-speculative AND golden EXACTLY.

    This is the self-speculative guarantee for greedy decoding: every emitted
    token is the LLM's greedy argmax at its position, so the output is
    byte-identical to standard greedy decoding.
    """
    inputs = _get_inputs()
    golden_gen = load_golden("greedy_ids.pt")[0, 271:]  # (100,)

    # Non-speculative baseline.
    _text_nonspec, ids_nonspec = pipeline.transcribe(
        inputs["input_features"],
        inputs["input_ids"],
        inputs.get("input_features_mask"),
        max_new_tokens=100,
        speculative=False,
    )

    # Speculative path.
    text_spec, ids_spec = pipeline.transcribe(
        inputs["input_features"],
        inputs["input_ids"],
        inputs.get("input_features_mask"),
        max_new_tokens=100,
        speculative=True,
    )

    # --- token-level exact match vs golden ---
    n = min(ids_spec.shape[1], golden_gen.shape[0])
    assert (ids_spec[0, :n] == golden_gen[:n]).all(), (
        f"speculative token mismatch vs golden at "
        f"{(ids_spec[0, :n] != golden_gen[:n]).nonzero()[0].item()}"
    )

    # --- token-level exact match vs non-speculative ---
    n2 = min(ids_spec.shape[1], ids_nonspec.shape[1])
    assert (ids_spec[0, :n2] == ids_nonspec[0, :n2]).all(), (
        f"speculative vs non-speculative mismatch at "
        f"{(ids_spec[0, :n2] != ids_nonspec[0, :n2]).nonzero()[0].item()}"
    )

    # --- transcript string match vs golden ---
    golden_text = load_golden_text().strip()
    golden_resp = golden_text.split("ASSISTANT:", 1)[1].strip()
    assert text_spec.strip() == golden_resp, (
        f"speculative transcript mismatch:\n  golden: {golden_resp[:100]!r}\n"
        f"  spec:   {text_spec.strip()[:100]!r}"
    )


if __name__ == "__main__":
    model, proc = _get_model_and_processor()
    pipe = MegaPipeline(model, proc, encoder_mode="cudagraph", use_fused_llm=True)
    test_draft_is_sensible(pipe)
    print("[manual] test_draft_is_sensible PASSED")
    test_speculative_matches_greedy(pipe)
    print("[manual] test_speculative_matches_greedy PASSED")
