"""Correctness gate for the feature-flag infrastructure.

Verifies that:
* Default flags preserve byte-exactness (``tolerance_mode=False``,
  ``batched_encoder=False``, ``multistep_graph=True``).
* The ``flags()`` context manager scopes overrides correctly and restores on
  exit.
* ``batched_encoder=True`` without ``tolerance_mode=True`` raises (guard).
* The pipeline wires ``multistep_graph`` to the correct decoder class.
* End-to-end: default-flags ``MegaPipeline`` produces byte-exact golden tokens.

Run with:  uv run pytest tests/test_flags.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from starling.flags import OptFlags, flags, get_default_flags, set_default_flags  # noqa: E402
from starling.config import LLM_EOS_TOKEN_ID  # noqa: E402
from starling.golden import load_golden  # noqa: E402
from starling.loader import get_components, load_model_and_processor  # noqa: E402

_MODEL = None
_PROC = None


def _get_model_and_processor():
    global _MODEL, _PROC
    if _MODEL is None:
        _MODEL, _PROC = load_model_and_processor(attn_impl="eager")
    return _MODEL, _PROC


def _golden_generated() -> torch.Tensor:
    return load_golden("greedy_ids.pt")[0, 271:]


# --------------------------------------------------------------------------- #
# flag defaults + validation
# --------------------------------------------------------------------------- #
def test_default_flags_preserve_byte_exactness():
    """Default flags must be the byte-exact safe baseline."""
    f = OptFlags()
    assert f.multistep_graph is True, "multistep_graph defaults True (byte-exact)"
    assert f.batched_encoder is False, "batched_encoder defaults False"
    assert f.tolerance_mode is False, "tolerance_mode defaults False"


def test_batched_encoder_requires_tolerance():
    """batched_encoder=True without tolerance_mode must raise."""
    with pytest.raises(ValueError, match="tolerance_mode"):
        OptFlags(batched_encoder=True, tolerance_mode=False)


def test_batched_encoder_with_tolerance_ok():
    """batched_encoder=True WITH tolerance_mode=True is valid."""
    f = OptFlags(batched_encoder=True, tolerance_mode=True)
    assert f.batched_encoder is True


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
def test_pipeline_multistep_graph_wiring():
    """multistep_graph=True -> MultiStepLLMMega; False -> FusedLLMMega."""
    from starling.pipeline import MegaPipeline
    from starling.multistep import MultiStepLLMMega
    from starling.llm_mega import FusedLLMMega

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
# end-to-end: default flags -> byte-exact golden match
# --------------------------------------------------------------------------- #
def test_default_flags_end_to_end_byte_exact():
    """A default-flags MegaPipeline (multistep on) must match golden exactly."""
    from starling.pipeline import MegaPipeline
    from starling.audio import build_inputs, load_sample_audio

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
    assert ids.shape == (1, golden_gen.shape[0]), (
        f"expected {golden_gen.shape[0]} tokens, got {ids.shape[1]}"
    )
    assert (ids[0] == golden_gen).all(), (
        f"default-flags token mismatch at "
        f"{(ids[0] != golden_gen).nonzero()[0].item() if (ids[0] != golden_gen).any() else -1}"
    )


if __name__ == "__main__":
    test_default_flags_preserve_byte_exactness()
    print("[manual] test_default_flags_preserve_byte_exactness PASSED")
    test_batched_encoder_requires_tolerance()
    print("[manual] test_batched_encoder_requires_tolerance PASSED")
    test_batched_encoder_with_tolerance_ok()
    print("[manual] test_batched_encoder_with_tolerance_ok PASSED")
    test_flags_context_restores()
    print("[manual] test_flags_context_restores PASSED")
    test_flags_context_partial_override()
    print("[manual] test_flags_context_partial_override PASSED")
    test_flags_context_restores_on_exception()
    print("[manual] test_flags_context_restores_on_exception PASSED")
    test_pipeline_multistep_graph_wiring()
    print("[manual] test_pipeline_multistep_graph_wiring PASSED")
    test_default_flags_end_to_end_byte_exact()
    print("[manual] test_default_flags_end_to_end_byte_exact PASSED")
