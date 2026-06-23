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

Both encoder modes are exercised: the stock eager
``model.get_audio_features`` path (``use_graphed_encoder=False``) and the
CUDA-graphed :class:`GraphedEncoder` path (``use_graphed_encoder=True``). The
graphed path is byte-exact with eager (max_diff 0.0), so both must reproduce
the oracle transcript byte-for-byte.

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

# Building a pipeline (loads the model ~25s); cache across all tests, one per
# encoder mode so the byte-exactness A/B covers both paths.
_PIPELINES: dict[bool, "object"] = {}


def _get_pipeline(use_graphed_encoder: bool):
    key = bool(use_graphed_encoder)
    if key not in _PIPELINES:
        from megapar.parakeet.pipeline import MegaParakeetPipeline  # noqa: WPS433

        _PIPELINES[key] = MegaParakeetPipeline(use_graphed_encoder=key)
    return _PIPELINES[key]


# Exercise both the stock eager encoder and the CUDA-graphed encoder. The graphed
# path is byte-exact with eager (max_diff 0.0), so the integrated transcript must
# match the oracle in both modes.
ENCODER_MODES = [False, True]


def _oracle():
    if not ORACLE_PATH.exists():
        pytest.skip(f"oracle missing: {ORACLE_PATH}")
    return {e["name"]: e for e in json.loads(ORACLE_PATH.read_text())}


FIXTURE_NAMES = ["short", "medium", "long"]


@pytest.mark.parametrize("use_graphed_encoder", ENCODER_MODES)
@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_pipeline_single_matches_oracle(name, use_graphed_encoder):
    """transcribe([fixture]) must match the oracle transcript byte-for-byte."""
    oracle = _oracle()
    pipe = _get_pipeline(use_graphed_encoder)
    fixtures = mkfx.load_fixtures()
    texts = pipe.transcribe([fixtures[name]])
    text = texts[0]
    expected = oracle[name]["text"]
    mode = "graphed" if use_graphed_encoder else "eager"
    assert text == expected, (
        f"[pipeline/{name}/{mode}] transcript drift:\n  oracle: {expected!r}\n"
        f"  mine:   {text!r}"
    )


@pytest.mark.parametrize("use_graphed_encoder", ENCODER_MODES)
def test_pipeline_batch8_uniform_medium(use_graphed_encoder):
    """Batch=8 uniform-medium: all 8 must equal the medium oracle transcript."""
    oracle = _oracle()
    pipe = _get_pipeline(use_graphed_encoder)
    fixtures = mkfx.load_fixtures()
    audio_list = mkfx.build_uniform_batch(fixtures["medium"], 8)
    texts = pipe.transcribe(audio_list)
    expected = oracle["medium"]["text"]
    mode = "graphed" if use_graphed_encoder else "eager"
    assert len(texts) == 8, f"expected 8 transcripts, got {len(texts)}"
    for i, t in enumerate(texts):
        assert t == expected, (
            f"[pipeline/batch8/{mode} elem {i}] transcript drift:\n"
            f"  oracle: {expected!r}\n  mine:   {t!r}"
        )
