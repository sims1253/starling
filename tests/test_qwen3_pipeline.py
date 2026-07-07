"""Correctness tests for the Qwen3-ASR megakernel vs the golden reference.

These mirror the granite/parakeet correctness gates: byte-exact (or
tolerance-bounded) comparison against the eager ``transformers`` reference
captured in ``golden/qwen3/`` (run ``python -m starling.qwen3.golden`` first).

The fixtures live in ``tests/fixtures/`` (shared, gitignored). The model is
loaded once per module and torn down in conftest.
"""

from __future__ import annotations

import os
import sys

import pytest

# Ensure the in-worktree starling package is importable when this test file is
# collected from any worktree / venv state. The shared venv's editable install
# may point at a sibling worktree, so prefer this file's own src/.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, "..", "src"))
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA required for Qwen3-ASR megakernel tests", allow_module_level=True)


# Module-level model cache (torn down by conftest after the module's last test).
_PIPE = None
_PROC = None


def _pipe():
    global _PIPE
    if _PIPE is None:
        from starling.qwen3.pipeline import MegaPipeline

        _PIPE = MegaPipeline.from_pretrained(encoder_mode="eager")
    return _PIPE


def _golden_inputs():
    from starling.qwen3.audio import build_inputs, load_wav
    from starling.qwen3.golden import _fixture_wav

    pipe = _pipe()
    wav, sr = load_wav(_fixture_wav())
    return build_inputs(pipe.processor, wav, sr=sr), wav, sr


def test_golden_artefacts_present():
    from starling.qwen3.golden import _all_exist

    if not _all_exist():
        pytest.skip("golden artefacts missing; run `python -m starling.qwen3.golden`")


def test_encoder_byte_exact_vs_golden():
    """Graphed-eager encoder last_hidden_state matches golden bit-for-bit."""
    from starling.qwen3.golden import ENCODER_LAST_HIDDEN, load_golden

    inputs, *_ = _golden_inputs()
    enc_lhs = _pipe().fused_encoder(inputs["input_features"], inputs["input_features_mask"])
    ref = load_golden(ENCODER_LAST_HIDDEN).to(enc_lhs.device, enc_lhs.dtype)
    diff = (enc_lhs.float() - ref.float()).abs().max().item()
    assert enc_lhs.shape == ref.shape, f"{enc_lhs.shape} vs {ref.shape}"
    assert diff == 0.0, f"encoder max-abs diff {diff} != 0.0 (expected byte-exact)"


def test_inputs_embeds_byte_exact_vs_golden():
    """Merged inputs_embeds matches golden bit-for-bit (encoder + merge)."""
    from starling.qwen3.golden import INPUTS_EMBEDS, load_golden

    inputs, *_ = _golden_inputs()
    pipe = _pipe()
    audio_embeds = pipe.encode_audio(inputs["input_features"], inputs["input_features_mask"])
    mine = pipe.build_inputs_embeds(inputs["input_ids"], audio_embeds)
    ref = load_golden(INPUTS_EMBEDS).to(mine.device, mine.dtype)
    diff = (mine.float() - ref.float()).abs().max().item()
    assert diff == 0.0, f"inputs_embeds max-abs diff {diff} != 0.0"


def test_prefill_graph_matches_eager():
    """Shape-keyed graphed prefill must produce the same first token as eager."""
    from starling.qwen3.golden import INPUTS_EMBEDS, load_golden

    inputs_embeds = load_golden(INPUTS_EMBEDS).to("cuda", torch.bfloat16)
    llm = _pipe().llm

    eager = llm.prefill(inputs_embeds, use_graph=False)
    graphed = llm.prefill(inputs_embeds, use_graph=True)

    assert torch.equal(eager, graphed), (
        f"prefill token mismatch: eager={eager.item()} graphed={graphed.item()}"
    )


def test_transcript_exact_match_vs_golden():
    """End-to-end transcript matches the golden greedy decode exactly."""
    from starling.qwen3.golden import load_golden_text

    inputs, *_ = _golden_inputs()
    text, ids = _pipe().transcribe(
        inputs["input_features"],
        inputs["input_ids"],
        inputs.get("input_features_mask"),
        max_new_tokens=200,
    )
    assert text.strip() == load_golden_text().strip()
    # token count sanity: golden produced 35 tokens for the short clip
    assert ids.shape[1] >= 5


def test_generated_ids_match_golden():
    """Generated token ids match the golden greedy ids exactly."""
    from starling.qwen3.golden import GREEDY_IDS, load_golden

    inputs, *_ = _golden_inputs()
    _, ids = _pipe().transcribe(
        inputs["input_features"],
        inputs["input_ids"],
        inputs.get("input_features_mask"),
        max_new_tokens=200,
    )
    ref = load_golden(GREEDY_IDS).to(ids.device).squeeze(0)
    mine = ids.squeeze(0)
    n = min(mine.shape[0], ref.shape[0])
    assert bool((mine[:n] == ref[:n]).all().item()), "decoded token ids diverge from golden"
