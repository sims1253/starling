"""Flag validation, scoped overrides, and GPU pipeline smoke tests.

Run with: uv run pytest tests/test_flags.py -q
"""

from __future__ import annotations

import difflib
import sys
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from starling.flags import OptFlags, flags, get_default_flags  # noqa: E402
from starling.granite.golden import load_golden, load_golden_text  # noqa: E402
from starling.granite.loader import load_model_and_processor  # noqa: E402

_MODEL = None
_PROC = None


def _get_model_and_processor():
    global _MODEL, _PROC
    if _MODEL is None:
        _MODEL, _PROC = load_model_and_processor(attn_impl="eager")
    return _MODEL, _PROC


def _golden_generated() -> torch.Tensor:
    return load_golden("greedy_ids.pt")[0, 271:]


# Historical fixture thresholds tolerate the observed long-decode drift.
# This is a smoke check, not token parity or a corpus-level accuracy gate.
_LONG_DECODE_PREFIX_FLOOR = 30
_LONG_DECODE_TRANSCRIPT_FLOOR = 0.90


def _leading_match_len(actual: torch.Tensor, expected: torch.Tensor) -> int:
    """Length of the leading matching token prefix (truncated to the shorter)."""
    n = min(len(actual), len(expected))
    eq = actual[:n] == expected[:n]
    return n if bool(eq.all()) else int((~eq).nonzero()[0].item())


def _transcript_similarity(actual: str, expected: str) -> float:
    """Normalized (case/space-insensitive) difflib transcript similarity ratio."""
    return difflib.SequenceMatcher(
        None, " ".join(actual.lower().split()), " ".join(expected.lower().split())
    ).ratio()


def _golden_response_text() -> str:
    """The golden ASSISTANT response body (everything after the ASSISTANT marker)."""
    golden_text = load_golden_text().strip()
    assert "ASSISTANT:" in golden_text, "golden text must contain ASSISTANT marker"
    return golden_text.split("ASSISTANT:", 1)[1].strip()


# --------------------------------------------------------------------------- #
# flag defaults + validation
# --------------------------------------------------------------------------- #
def test_default_flags():
    """Approximate paths require explicit opt-in."""
    f = OptFlags()
    assert f.multistep_graph is True, "multistep_graph defaults True"
    assert f.batched_encoder is False, "batched_encoder defaults False"
    assert f.tolerance_mode is False, "tolerance_mode defaults False"
    assert f.fused_qkv is True, "fused_qkv defaults True"
    assert f.sdpa_attention is False, "sdpa_attention defaults False"
    assert f.flash_attention is False, "flash_attention defaults False"
    assert f.fp8_attention is False, "fp8_attention defaults False"
    assert f.fp8_weights is False, "fp8_weights defaults False (requires tolerance)"
    assert f.rope_alloc_free is True, "rope_alloc_free defaults True"
    assert f.lm_head_scale_fold is True, "lm_head_scale_fold defaults True"
    assert f.gemm_epilogue_fusion is False, "gemm_epilogue_fusion defaults False (experimental)"
    assert f.chunk_prefill_overlap is True, "chunk_prefill_overlap defaults True"
    assert f.nvfp4_weights is False, "nvfp4_weights defaults False (requires tolerance)"
    assert f.nvfp4_lm_head_only is False, "nvfp4_lm_head_only defaults False (requires tolerance)"


@pytest.mark.parametrize("name", [
    "batched_encoder", "sdpa_attention", "flash_attention", "fp8_attention",
    "fp8_weights", "gemm_epilogue_fusion", "nvfp4_weights", "nvfp4_lm_head_only",
])
def test_approximate_path_requires_tolerance(name):
    with pytest.raises(ValueError, match=f"{name}=True requires tolerance_mode=True"):
        OptFlags(**{name: True})
    assert getattr(OptFlags(tolerance_mode=True, **{name: True}), name)


@pytest.mark.parametrize("name", [
    "fp8_weights", "nvfp4_weights", "nvfp4_lm_head_only", "gemm_epilogue_fusion",
])
def test_weight_paths_enable_fused_qkv(name):
    assert OptFlags(tolerance_mode=True, fused_qkv=False, **{name: True}).fused_qkv


def test_fp8_attention_implies_flash():
    assert OptFlags(fp8_attention=True, tolerance_mode=True).flash_attention


# --------------------------------------------------------------------------- #
# context manager scoping
# --------------------------------------------------------------------------- #
def test_flags_context_restores():
    """The flags() context manager must restore the previous default on exit."""
    saved = get_default_flags()
    assert saved.tolerance_mode is False

    with flags(tolerance_mode=True) as scoped:
        assert scoped.tolerance_mode is True
        assert get_default_flags().tolerance_mode is True

    # restored after exit
    assert get_default_flags().tolerance_mode is False
    assert get_default_flags() is saved or (
        get_default_flags().tolerance_mode == saved.tolerance_mode
        and get_default_flags().multistep_graph == saved.multistep_graph
    )


def test_flags_context_partial_override():
    """flags() only overrides the given keys; others inherit the current default."""
    with flags(multistep_graph=False) as scoped:
        assert scoped.multistep_graph is False
        # batched_encoder not overridden -> inherits default
        assert scoped.batched_encoder is False
        assert scoped.tolerance_mode is False


def test_flags_context_restores_on_exception():
    """The context manager restores even if an exception is raised inside."""
    assert get_default_flags().tolerance_mode is False
    with pytest.raises(RuntimeError):
        with flags(tolerance_mode=True):
            assert get_default_flags().tolerance_mode is True
            raise RuntimeError("boom")
    assert get_default_flags().tolerance_mode is False


# --------------------------------------------------------------------------- #
# pipeline wiring: multistep_graph selects the right decoder
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_pipeline_multistep_graph_wiring():
    """multistep_graph=True -> MultiStepLLMMega; False -> FusedLLMMega."""
    from starling.granite.pipeline import MegaPipeline
    from starling.granite.multistep import MultiStepLLMMega
    from starling.granite.llm_mega import FusedLLMMega

    model, proc = _get_model_and_processor()

    pipe_on = MegaPipeline(
        model, proc, flags=OptFlags(multistep_graph=True)
    )
    assert isinstance(pipe_on.llm, MultiStepLLMMega), (
        "multistep_graph=True should use MultiStepLLMMega"
    )

    pipe_off = MegaPipeline(
        model, proc, flags=OptFlags(multistep_graph=False)
    )
    assert isinstance(pipe_off.llm, FusedLLMMega), (
        "multistep_graph=False should use FusedLLMMega"
    )


# --------------------------------------------------------------------------- #
# end-to-end smoke test
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_default_flags_end_to_end_smoke():
    """Check the fixture's leading tokens and normalized text similarity."""
    from starling.granite.pipeline import MegaPipeline
    from starling.granite.audio import build_inputs, load_sample_audio

    model, proc = _get_model_and_processor()
    pipe = MegaPipeline(model, proc, encoder_mode="cudagraph")

    wav, sr = load_sample_audio()
    inputs = build_inputs(proc, wav)
    golden_gen = _golden_generated()

    text, ids = pipe.transcribe(
        inputs["input_features"],
        inputs["input_ids"],
        inputs.get("input_features_mask"),
        max_new_tokens=100,
    )

    prefix = _leading_match_len(ids[0], golden_gen)
    assert prefix >= _LONG_DECODE_PREFIX_FLOOR, (
        f"leading token match too short ({prefix}/{min(ids.shape[1], len(golden_gen))}); "
        f"expected >={_LONG_DECODE_PREFIX_FLOOR}. "
        f"golden={golden_gen[:12].tolist()} mine={ids[0][:12].tolist()}"
    )
    sim = _transcript_similarity(text, _golden_response_text())
    assert sim >= _LONG_DECODE_TRANSCRIPT_FLOOR, (
        f"transcript similarity {sim:.3f} < {_LONG_DECODE_TRANSCRIPT_FLOOR}: "
        f"golden={_golden_response_text()[:100]!r} ours={text[:100]!r}"
    )
