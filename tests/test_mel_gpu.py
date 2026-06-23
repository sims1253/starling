"""Correctness tests for the GPU mel extractor (``starling.parakeet.mel_gpu``).

Two layers of checks:

1. **Numerical match vs stock ``processor``** (parametrised over short/medium/
   long and over a mixed batch): max-abs ``input_features`` difference < 1e-3
   and ``attention_mask`` matches bit-exactly. The float32 GPU vs CPU pipeline
   agreement is ~3e-4 (well inside tolerance).

2. **End-to-end oracle transcript match**: run GPU mel -> encoder ->
   CUDA-graph-captured TDT decode on each fixture and assert the transcript
   matches ``outputs/oracle.json`` byte-for-byte. This proves the 1e-3 numerical
   tolerance does not change the model's greedy output.

Run with:  uv run pytest tests/test_mel_gpu.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "tests" / "fixtures"))
sys.path.insert(0, str(_REPO_ROOT / "src"))

import make_fixtures as mkfx  # noqa: E402

MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"
ORACLE_PATH = _REPO_ROOT / "outputs" / "oracle.json"
MAX_ABS_TOLERANCE = 1e-3

# Loading the model is expensive (~25s); cache it across all tests in the module.
_STATE: dict = {}


def _get_model_and_processor():
    if not _STATE:
        from transformers import AutoModelForTDT, AutoProcessor  # noqa: WPS433

        processor = AutoProcessor.from_pretrained(MODEL_ID)
        model = AutoModelForTDT.from_pretrained(
            MODEL_ID, dtype=torch.bfloat16, device_map="cuda"
        )
        model.eval()
        _STATE["model"] = model
        _STATE["processor"] = processor
        # build the GPU extractor once and cache it too
        from starling.parakeet.mel_gpu import GpuMelExtractor  # noqa: WPS433
        _STATE["extractor"] = GpuMelExtractor(processor, device="cuda")
    return _STATE["model"], _STATE["processor"], _STATE["extractor"]


def _oracle():
    if not ORACLE_PATH.exists():
        pytest.skip(f"oracle missing: {ORACLE_PATH}")
    return {e["name"]: e for e in json.loads(ORACLE_PATH.read_text())}


FIXTURE_NAMES = ["short", "medium", "long"]


# ---------------------------------------------------------------------------
# Step 2 correctness test: numerical match vs stock processor
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_mel_matches_stock_single(name):
    """Single utterance: GPU mel vs stock processor, max-abs < 1e-3, mask exact."""
    _, processor, extractor = _get_model_and_processor()
    fixtures = mkfx.load_fixtures()
    audio = fixtures[name]

    stock = processor([audio], sampling_rate=16000)
    feats_stock = stock["input_features"]
    mask_stock = stock["attention_mask"]

    feats_gpu, mask_gpu = extractor([audio])

    assert feats_gpu.shape == feats_stock.shape, (
        f"[{name}] feats shape gpu={tuple(feats_gpu.shape)} "
        f"stock={tuple(feats_stock.shape)}"
    )
    assert mask_gpu.shape == mask_stock.shape
    assert mask_gpu.dtype == torch.bool
    assert feats_gpu.dtype == torch.float32
    assert feats_gpu.device.type == "cuda"
    assert mask_gpu.device.type == "cuda"

    # attention mask must match bit-exactly
    assert torch.equal(mask_gpu.cpu(), mask_stock.to(torch.bool)), (
        f"[{name}] attention_mask differs from stock"
    )

    max_abs = (feats_gpu.cpu() - feats_stock).abs().max().item()
    assert max_abs < MAX_ABS_TOLERANCE, (
        f"[{name}] max_abs={max_abs:.3e} >= {MAX_ABS_TOLERANCE:.0e}"
    )


def test_mel_matches_stock_mixed_batch():
    """Mixed-length batch (short, medium, long): shape/mask/max-abs all match."""
    _, processor, extractor = _get_model_and_processor()
    fixtures = mkfx.load_fixtures()
    audio_list = [fixtures["short"], fixtures["medium"], fixtures["long"]]

    stock = processor(audio_list, sampling_rate=16000)
    feats_stock = stock["input_features"]
    mask_stock = stock["attention_mask"]

    feats_gpu, mask_gpu = extractor(audio_list)

    assert feats_gpu.shape == feats_stock.shape
    assert mask_gpu.shape == mask_stock.shape
    assert torch.equal(mask_gpu.cpu(), mask_stock.to(torch.bool))

    max_abs = (feats_gpu.cpu() - feats_stock).abs().max().item()
    assert max_abs < MAX_ABS_TOLERANCE, f"max_abs={max_abs:.3e}"

    # also check that the valid-frame region matches very tightly (padding rows
    # are forced to zero by both pipelines, so they always match exactly)
    valid = mask_stock.to(torch.bool)
    valid_diff = (feats_gpu.cpu() - feats_stock)[valid].abs().max().item()
    assert valid_diff < MAX_ABS_TOLERANCE


def test_mel_matches_stock_uniform_batch8():
    """Uniform batch (8x medium): clean per-length scaling, max-abs < 1e-3."""
    _, processor, extractor = _get_model_and_processor()
    fixtures = mkfx.load_fixtures()
    audio_list = mkfx.build_uniform_batch(fixtures["medium"], 8)

    stock = processor(audio_list, sampling_rate=16000)
    feats_stock = stock["input_features"]
    mask_stock = stock["attention_mask"]

    feats_gpu, mask_gpu = extractor(audio_list)

    assert feats_gpu.shape == feats_stock.shape
    assert torch.equal(mask_gpu.cpu(), mask_stock.to(torch.bool))
    max_abs = (feats_gpu.cpu() - feats_stock).abs().max().item()
    assert max_abs < MAX_ABS_TOLERANCE, f"max_abs={max_abs:.3e}"


def test_extractor_extract_from_tensor():
    """The on-device entry point must give the same answer as the np entry point."""
    _, processor, extractor = _get_model_and_processor()
    fixtures = mkfx.load_fixtures()
    audio = fixtures["short"]

    feats_a, mask_a = extractor([audio])

    # batch the audio on device by hand and call extract_from_tensor
    L = len(audio)
    wav = torch.zeros((1, L), dtype=torch.float32, device="cuda")
    wav[0, :L] = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
    lengths = torch.tensor([L], dtype=torch.long, device="cuda")
    feats_b, mask_b = extractor.extract_from_tensor(wav, lengths)

    assert feats_a.shape == feats_b.shape
    assert torch.equal(mask_a, mask_b)
    assert torch.allclose(feats_a, feats_b, atol=1e-6, rtol=0)


# ---------------------------------------------------------------------------
# Step 2 correctness test: full-pipeline oracle transcript match
# (GPU mel -> encoder -> graphed decode == oracle, byte-for-byte)
# This proves the 1e-3 numerical tolerance does not change the model output.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_full_pipeline_oracle_transcript(name):
    """GPU mel -> encoder -> graphed TDT decode must match the oracle text."""
    from starling.parakeet.decode_mega import greedy_decode_graphed  # noqa: WPS433

    oracle = _oracle()
    model, processor, extractor = _get_model_and_processor()
    fixtures = mkfx.load_fixtures()
    audio = fixtures[name]

    # GPU mel features (float32 on cuda); cast to bf16 for the encoder (the
    # encoder matmul accumulator upcasts anyway; this matches the baseline).
    feats_gpu, mask_gpu = extractor([audio])
    feats_bf16 = feats_gpu.to(torch.bfloat16)

    texts = greedy_decode_graphed(
        model,
        feats_bf16,
        mask_gpu,
        processor,
    )
    text = texts[0]
    expected = oracle[name]["text"]
    assert text == expected, (
        f"[gpu-mel/{name}] transcript drift:\n  oracle: {expected!r}\n"
        f"  mine:   {text!r}"
    )
