"""Correctness tests for the Audex-2B megakernel vs the golden reference.

Mirror the qwen3/ark correctness gates: byte-exact comparison against the eager
``transformers`` reference captured in ``golden/audex/`` (run
``python -m starling.audex.golden`` first).
"""

from __future__ import annotations

import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, "..", "src"))
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA required for Audex-2B megakernel tests", allow_module_level=True)


_PIPE = None
_PROC = None


def _pipe():
    global _PIPE
    if _PIPE is None:
        from starling.audex.pipeline import MegaPipeline

        _PIPE = MegaPipeline.from_pretrained()
    return _PIPE


def _golden_inputs():
    from starling.audex.audio import build_inputs, load_wav
    from starling.audex.golden import _fixture_wav

    pipe = _pipe()
    wav, sr = load_wav(_fixture_wav())
    inputs = build_inputs(
        pipe.tokenizer, pipe.feature_extractor, wav
    )
    return inputs, wav, sr


def test_golden_artefacts_present():
    from starling.audex.golden import _all_exist

    if not _all_exist():
        pytest.skip("golden artefacts missing; run `python -m starling.audex.golden`")


def test_encoder_byte_exact_vs_golden():
    """Graphed encoder last_hidden_state matches golden bit-for-bit."""
    from starling.audex.golden import ENCODER_LAST_HIDDEN, load_golden

    inputs, *_ = _golden_inputs()
    enc_lhs = _pipe().fused_encoder(inputs["input_features"])
    ref = load_golden(ENCODER_LAST_HIDDEN).to(enc_lhs.device, enc_lhs.dtype)
    diff = (enc_lhs.float() - ref.float()).abs().max().item()
    assert enc_lhs.shape == ref.shape, f"{enc_lhs.shape} vs {ref.shape}"
    assert diff == 0.0, f"encoder max-abs diff {diff} != 0.0"


def test_inputs_embeds_byte_exact_vs_golden():
    """Merged inputs_embeds matches golden bit-for-bit."""
    from starling.audex.golden import INPUTS_EMBEDS, load_golden

    inputs, *_ = _golden_inputs()
    pipe = _pipe()
    audio_embeds = pipe.encode_audio(inputs["input_features"])
    mine = pipe.build_inputs_embeds(inputs["input_ids"], audio_embeds)
    ref = load_golden(INPUTS_EMBEDS).to(mine.device, mine.dtype)
    diff = (mine.float() - ref.float()).abs().max().item()
    assert diff == 0.0, f"inputs_embeds max-abs diff {diff} != 0.0"


def test_transcript_exact_match_vs_golden():
    """End-to-end transcript matches the golden greedy decode exactly."""
    from starling.audex.golden import load_golden_text

    inputs, wav, sr = _golden_inputs()
    text, ids = _pipe().transcribe(wav, max_new_tokens=200)
    golden = load_golden_text().strip()
    assert text.strip() == golden, f"\ngot:  {text!r}\nwant: {golden!r}"
    assert ids.shape[1] >= 5


def test_no_think_tags_in_output():
    """Non-thinking ASR output must not contain <think> tags."""
    inputs, wav, sr = _golden_inputs()
    text, _ = _pipe().transcribe(wav, max_new_tokens=200)
    assert "<think>" not in text
    assert "</think>" not in text
