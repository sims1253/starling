"""Byte-exact correctness tests for the integrated MegaParakeetPipeline.

The pipeline wires GPU mel -> Conformer encoder -> graphed TDT decode. All three
components are individually byte-exact, so the integrated transcript must match
``outputs/oracle.json`` BYTE-FOR-BYTE on the short/medium/long fixtures, and a
batch=8 uniform-medium batch must reproduce 8x the medium transcript (the shape
cache reuses one captured decoder for the (8, 279) shape).

NOTE on filename: this file is deliberately ``test_parakeet_pipeline.py`` rather
than ``tests/test_pipeline.py`` because the latter is **granite-owned** per
comms.md §P2 and already contains the granite-speech correctness gate
(``megapar.pipeline.MegaPipeline``). Overwriting it would corrupt the shared
repo. The parent task asked for ``tests/test_pipeline.py``; the filename
collision is reported back to the orchestrator.

Run with:  uv run pytest tests/test_parakeet_pipeline.py -q
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

# Building the pipeline (loads the model ~25s); cache across all tests.
_PIPELINE = None


def _get_pipeline():
    global _PIPELINE
    if _PIPELINE is None:
        from megapar.parakeet.pipeline import MegaParakeetPipeline  # noqa: WPS433

        _PIPELINE = MegaParakeetPipeline()
    return _PIPELINE


def _oracle():
    if not ORACLE_PATH.exists():
        pytest.skip(f"oracle missing: {ORACLE_PATH}")
    return {e["name"]: e for e in json.loads(ORACLE_PATH.read_text())}


FIXTURE_NAMES = ["short", "medium", "long"]


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_pipeline_single_matches_oracle(name):
    """transcribe([fixture]) must match the oracle transcript byte-for-byte."""
    oracle = _oracle()
    pipe = _get_pipeline()
    fixtures = mkfx.load_fixtures()
    texts = pipe.transcribe([fixtures[name]])
    text = texts[0]
    expected = oracle[name]["text"]
    assert text == expected, (
        f"[pipeline/{name}] transcript drift:\n  oracle: {expected!r}\n"
        f"  mine:   {text!r}"
    )


def test_pipeline_batch8_uniform_medium():
    """Batch=8 uniform-medium: all 8 must equal the medium oracle transcript."""
    oracle = _oracle()
    pipe = _get_pipeline()
    fixtures = mkfx.load_fixtures()
    audio_list = mkfx.build_uniform_batch(fixtures["medium"], 8)
    texts = pipe.transcribe(audio_list)
    expected = oracle["medium"]["text"]
    assert len(texts) == 8, f"expected 8 transcripts, got {len(texts)}"
    for i, t in enumerate(texts):
        assert t == expected, (
            f"[pipeline/batch8 elem {i}] transcript drift:\n"
            f"  oracle: {expected!r}\n  mine:   {t!r}"
        )
