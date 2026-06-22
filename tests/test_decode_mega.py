"""Byte-exact correctness tests for the parakeet TDT decoders (eager + graphed).

Both the hand-rolled eager decoder (:mod:`decode_eager`) and the CUDA-graph-
captured decoder (:mod:`decode_mega`) must reproduce the deterministic greedy
transcript in ``outputs/oracle.json`` BYTE-FOR-BYTE on the short/medium/long
fixtures.

Run with:  uv run pytest tests/test_decode_mega.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "tests" / "fixtures"))
sys.path.insert(0, str(_REPO_ROOT / "src"))

import make_fixtures as mkfx  # noqa: E402

ORACLE_PATH = _REPO_ROOT / "outputs" / "oracle.json"

# Loading the model is expensive (~25s); cache it across all tests in the module.
_STATE: dict = {}


def _get_model_and_processor():
    if not _STATE:
        import torch  # noqa: WPS433
        from transformers import AutoModelForTDT, AutoProcessor  # noqa: WPS433

        MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        model = AutoModelForTDT.from_pretrained(
            MODEL_ID, dtype=torch.bfloat16, device_map="cuda"
        )
        model.eval()
        _STATE["model"] = model
        _STATE["processor"] = processor
    return _STATE["model"], _STATE["processor"]


def _oracle():
    if not ORACLE_PATH.exists():
        pytest.skip(
            f"oracle missing: {ORACLE_PATH} (run benchmarks/bench_rtf.py first)"
        )
    return {e["name"]: e for e in json.loads(ORACLE_PATH.read_text())}


def _prepare(processor, audio):
    """processor + H2D + bf16 cast (matches the baseline's prepare_inputs)."""
    import torch  # noqa: WPS433

    inputs = processor([audio], sampling_rate=16000).to("cuda")
    inputs["input_features"] = inputs["input_features"].to(torch.bfloat16)
    return inputs


FIXTURE_NAMES = ["short", "medium", "long"]


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_eager_decode_matches_oracle(name):
    """Eager greedy_decode must reproduce the oracle transcript byte-for-byte."""
    from megapar.parakeet.decode_eager import greedy_decode

    oracle = _oracle()
    model, processor = _get_model_and_processor()
    fixtures = mkfx.load_fixtures()
    inputs = _prepare(processor, fixtures[name])

    texts = greedy_decode(
        model,
        inputs["input_features"],
        inputs["attention_mask"],
        processor,
    )
    text = texts[0]
    expected = oracle[name]["text"]
    assert text == expected, (
        f"[eager/{name}] transcript drift:\n  oracle: {expected!r}\n"
        f"  mine:   {text!r}"
    )
    # token count must also match (the decoder emitted the right number of ids)
    # oracle num_tokens counts non-pad ids; our decode emitted exactly the
    # transcript, so a text match already implies the token sequence matches.


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_graphed_decode_matches_oracle(name):
    """CUDA-graph-captured greedy_decode_graphed must match the oracle too."""
    from megapar.parakeet.decode_mega import greedy_decode_graphed

    oracle = _oracle()
    model, processor = _get_model_and_processor()
    fixtures = mkfx.load_fixtures()
    inputs = _prepare(processor, fixtures[name])

    texts = greedy_decode_graphed(
        model,
        inputs["input_features"],
        inputs["attention_mask"],
        processor,
    )
    text = texts[0]
    expected = oracle[name]["text"]
    assert text == expected, (
        f"[graphed/{name}] transcript drift:\n  oracle: {expected!r}\n"
        f"  mine:   {text!r}"
    )
