"""Correctness gate for the fused encoder megakernel.

Loads the golden ``encoder_last_hidden.pt`` (eager reference), runs every
``FusedEncoder`` mode on the sample ``input_features`` (1247 mel frames), and
asserts:
  * max(abs(out - golden)) < ENCODER_ATOL  (2e-2)
  * mean(abs(out - golden)) < 5e-3

Run with:  uv run pytest tests/test_encoder_mega.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from megapar.config import ENCODER_ATOL  # noqa: E402
from megapar.encoder_mega import FusedEncoder  # noqa: E402
from megapar.golden import load_golden  # noqa: E402
from megapar.loader import get_components, load_model_and_processor  # noqa: E402
from megapar.audio import build_inputs, load_sample_audio  # noqa: E402

_MEAN_TOL = 5e-3

# Cached across tests (model load is ~25s).
_MODEL_CACHE: dict = {}
_INPUT_FEATURES: torch.Tensor | None = None
_GOLDEN: torch.Tensor | None = None


def _get_encoder():
    if "encoder" not in _MODEL_CACHE:
        model, _ = _MODEL_CACHE.setdefault("model_pair", (None, None))
        if model is None:
            model, _ = load_model_and_processor(attn_impl="eager")
            _MODEL_CACHE["model_pair"] = (model, _)
        _MODEL_CACHE["encoder"] = get_components(model)["encoder"]
    return _MODEL_CACHE["encoder"]


def _get_input_features() -> torch.Tensor:
    global _INPUT_FEATURES
    if _INPUT_FEATURES is None:
        model, processor = _MODEL_CACHE["model_pair"]
        wav, sr = load_sample_audio()
        inputs = build_inputs(processor, wav)
        _INPUT_FEATURES = inputs["input_features"].to(torch.bfloat16).cuda()
    return _INPUT_FEATURES


def _get_golden() -> torch.Tensor:
    global _GOLDEN
    if _GOLDEN is None:
        _GOLDEN = load_golden("encoder_last_hidden.pt").cuda()
    return _GOLDEN


def _check_against_golden(out: torch.Tensor, label: str) -> None:
    golden = _get_golden()
    assert out.shape == golden.shape, f"{label}: shape {out.shape} != golden {golden.shape}"
    assert out.dtype == torch.bfloat16, f"{label}: dtype {out.dtype} != bf16"
    diff = (out.float() - golden.float()).abs()
    max_d = diff.max().item()
    mean_d = diff.mean().item()
    print(f"[{label}] max abs diff = {max_d:.6e}  mean abs diff = {mean_d:.6e}")
    assert max_d < ENCODER_ATOL, (
        f"{label}: max abs diff {max_d:.4e} >= ENCODER_ATOL {ENCODER_ATOL:.4e}"
    )
    assert mean_d < _MEAN_TOL, (
        f"{label}: mean abs diff {mean_d:.4e} >= {_MEAN_TOL:.4e}"
    )


@pytest.fixture(scope="module")
def encoder():
    return _get_encoder()


@pytest.fixture(scope="module")
def input_features():
    # Touch model cache first so the processor is available.
    _get_encoder()
    return _get_input_features()


def test_eager_matches_golden(encoder, input_features):
    fe = FusedEncoder(encoder, mode="eager").cuda()
    with torch.inference_mode():
        out = fe(input_features)
    _check_against_golden(out, "eager")


def test_compile_matches_golden(encoder, input_features):
    fe = FusedEncoder(encoder, mode="compile", compile_mode="max-autotune").cuda()
    # warmup the compiled fn once (compilation happens on first call)
    with torch.inference_mode():
        _ = fe(input_features)
        torch.cuda.synchronize()
        out = fe(input_features)
    _check_against_golden(out, "compile(max-autotune)")


def test_compile_reduce_overhead_matches_golden(encoder, input_features):
    fe = FusedEncoder(
        encoder, mode="compile", compile_mode="reduce-overhead", compile_fullgraph=True
    ).cuda()
    with torch.inference_mode():
        _ = fe(input_features)
        torch.cuda.synchronize()
        out = fe(input_features)
    _check_against_golden(out, "compile(reduce-overhead)")


def test_triton_matches_golden(encoder, input_features):
    fe = FusedEncoder(encoder, mode="triton", compile_mode="max-autotune").cuda()
    with torch.inference_mode():
        _ = fe(input_features)
        torch.cuda.synchronize()
        out = fe(input_features)
    _check_against_golden(out, "triton(max-autotune)")


if __name__ == "__main__":
    # Allow running as a script for a quick numeric report.
    enc = _get_encoder()
    feats = _get_input_features()
    for mode, kw in [
        ("eager", {}),
        ("compile", {"compile_mode": "max-autotune"}),
        ("compile", {"compile_mode": "reduce-overhead"}),
        ("triton", {"compile_mode": "max-autotune"}),
    ]:
        tag = f"{mode}/{kw.get('compile_mode', '-')}"
        try:
            fe = FusedEncoder(enc, mode=mode, **kw).cuda()
            with torch.inference_mode():
                _ = fe(feats)
                torch.cuda.synchronize()
                out = fe(feats)
            _check_against_golden(out, tag)
            print(f"  -> {tag}: PASS")
        except Exception as e:  # noqa: BLE001
            print(f"  -> {tag}: FAIL ({e!r})")
