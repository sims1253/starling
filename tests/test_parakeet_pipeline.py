"""Byte-exact correctness tests for the integrated MegaParakeetPipeline.

The pipeline wires GPU mel -> Conformer encoder -> graphed TDT decode. All three
components are individually byte-exact, so the integrated transcript must match
``outputs/oracle.json`` BYTE-FOR-BYTE on the short/medium/long fixtures, and a
batch=8 uniform-medium batch must reproduce 8x the medium transcript (the shape
cache reuses one captured decoder for the (8, 279) shape).

NOTE on filename: this file is deliberately ``test_parakeet_pipeline.py``
rather than ``tests/test_pipeline.py``; the latter holds the granite-speech
correctness gate (``starling.granite.pipeline.MegaPipeline``), while this file holds
the parakeet pipeline correctness gate.

Both encoder modes are exercised: the stock eager
``model.get_audio_features`` path (``use_graphed_encoder=False``) and the
CUDA-graphed :class:`GraphedEncoder` path (``use_graphed_encoder=True``). The
graphed path is byte-exact with eager (max_diff 0.0), so both must reproduce
the oracle transcript byte-for-byte.

The compiled mode (``encoder_mode="compiled"``: torch.compile + BatchNorm1d
fold) is NOT guaranteed byte-exact; its correctness gate is a text-level
transcript match vs the oracle plus a pooler max_abs sanity bound (see
``test_compiled_encoder_*``).

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
_PIPELINES: dict[str, "object"] = {}


def _get_pipeline(use_graphed_encoder: bool):
    """Backward-compat entry: bool flag -> graphed/eager mode pipeline (cached)."""
    mode = "graphed" if use_graphed_encoder else "eager"
    return _get_pipeline_mode(mode)


def _get_pipeline_mode(encoder_mode: str):
    """Return a cached pipeline for the given ``encoder_mode`` string."""
    if encoder_mode not in _PIPELINES:
        from starling.parakeet.pipeline import MegaParakeetPipeline  # noqa: WPS433

        _PIPELINES[encoder_mode] = MegaParakeetPipeline(encoder_mode=encoder_mode)
    return _PIPELINES[encoder_mode]


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


# ---------------------------------------------------------------------- #
# compiled mode (torch.compile + BatchNorm1d fold) -- NOT byte-exact
# correctness gate: text-level transcript match + pooler max_abs sanity bound
# ---------------------------------------------------------------------- #
# The compiled pooler deviates from the byte-exact graphed reference because
# torch.compile reorders reductions / upcasts attention and the conv-module
# BN fold bakes the per-channel gain into the depthwise weight. Measured
# (RTX 5090, bf16): max_abs ~ 0.06 (short) / 0.28 (medium) / 1.3 (long) -- it
# grows with sequence length (more frames = more worst-case elements), but
# mean_abs stays ~2e-3 and the greedy TDT transcript matches the oracle on all
# fixtures, so the transcript is the real correctness gate and this bound is a
# loose stability guard against catastrophic drift, not an accuracy budget.
MAX_ABS_SANITY = 2.0  # compiled-vs-graphed pooler max_abs upper bound (loose)


def _encoder_pooler(pipe, audio):
    """Run mel + encoder on one audio; return the bf16 pooler (B, T_enc, 640)."""
    feats, mask = pipe.mel([audio])
    feats = feats.to(pipe.dtype)
    pooler, _ = pipe._run_encoder(feats, mask)
    return pooler


@pytest.mark.compile
@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_compiled_transcript_matches_oracle(name):
    """compiled encoder: transcript must match the oracle (text-level).

    Gated behind the ``compile`` marker: the compiled pipeline uses
    ``torch.compile`` (max-autotune) which benchmarks kernels on the GPU and is
    slow. Run with ``pytest --runcompile``.
    """
    oracle = _oracle()
    pipe = _get_pipeline_mode("compiled")
    fixtures = mkfx.load_fixtures()
    text = pipe.transcribe([fixtures[name]])[0]
    expected = oracle[name]["text"]
    assert text == expected, (
        f"[pipeline/{name}/compiled] transcript drift:\n"
        f"  oracle: {expected!r}\n  mine:   {text!r}"
    )


@pytest.mark.compile
def test_compiled_batch8_uniform_medium():
    """compiled encoder: batch=8 uniform-medium all match the oracle transcript.

    Gated behind the ``compile`` marker (slow ``torch.compile``). Run with
    ``pytest --runcompile``.
    """
    oracle = _oracle()
    pipe = _get_pipeline_mode("compiled")
    fixtures = mkfx.load_fixtures()
    audio_list = mkfx.build_uniform_batch(fixtures["medium"], 8)
    texts = pipe.transcribe(audio_list)
    expected = oracle["medium"]["text"]
    assert len(texts) == 8, f"expected 8 transcripts, got {len(texts)}"
    for i, t in enumerate(texts):
        assert t == expected, (
            f"[pipeline/batch8/compiled elem {i}] transcript drift:\n"
            f"  oracle: {expected!r}\n  mine:   {t!r}"
        )


@pytest.mark.compile
@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_compiled_pooler_near_exact_vs_graphed(name):
    """compiled encoder pooler must be near the graphed (reference) pooler.

    The compiled path (BN fold + torch.compile) is NOT guaranteed byte-exact,
    but must stay within a sane max_abs bound of the byte-exact graphed path so
    the transcript cannot drift. If the compiled pooler is byte-exact
    (max_abs == 0.0) this test still passes (the bound is an upper limit).

    Gated behind the ``compile`` marker (slow ``torch.compile``). Run with
    ``pytest --runcompile``.
    """
    g_pipe = _get_pipeline_mode("graphed")
    c_pipe = _get_pipeline_mode("compiled")
    fixtures = mkfx.load_fixtures()
    g_pooler = _encoder_pooler(g_pipe, fixtures[name])
    c_pooler = _encoder_pooler(c_pipe, fixtures[name])
    assert g_pooler.shape == c_pooler.shape, (
        f"[{name}] compiled pooler shape {tuple(c_pooler.shape)} != "
        f"graphed {tuple(g_pooler.shape)}"
    )
    max_abs = (g_pooler.float() - c_pooler.float()).abs().max().item()
    assert max_abs < MAX_ABS_SANITY, (
        f"[pipeline/{name}/compiled] pooler max_abs={max_abs:.3e} "
        f">= sanity bound {MAX_ABS_SANITY:.0e}; transcript may drift"
    )
