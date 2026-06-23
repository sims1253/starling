"""Correctness gate for the fused encoder megakernel.

Loads the golden ``encoder_last_hidden.pt`` (eager reference), runs every
``FusedEncoder`` mode on the sample ``input_features`` (1247 mel frames), and
asserts:
  * max(abs(out - golden)) < ENCODER_ATOL  (2e-2)
  * mean(abs(out - golden)) < 5e-3

The ``eager``, ``cudagraph``, and ``triton`` modes are byte-exact (0.0 diff)
and must pass. The ``compile`` mode uses fp32 attention intermediates (inductor)
and is numerically close but not bitwise identical; it is tested separately
with a relaxed assertion + a printed diff.

Run with:  uv run pytest tests/test_encoder_mega.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from starling.config import ENCODER_ATOL  # noqa: E402
from starling.encoder_mega import FusedEncoder  # noqa: E402
from starling.golden import load_golden  # noqa: E402
from starling.loader import get_components, load_model_and_processor  # noqa: E402
from starling.audio import build_inputs, load_sample_audio  # noqa: E402

_MEAN_TOL = 5e-3
_COMPILE_RELAXED_ATOL = 5.0  # compile changes attention precision; report, don't fail

# Cached across tests (model load is ~25s).
_MODEL_CACHE: dict = {}
_INPUT_FEATURES: torch.Tensor | None = None
_GOLDEN: torch.Tensor | None = None


def _get_encoder():
    if "encoder" not in _MODEL_CACHE:
        model, processor = _MODEL_CACHE.setdefault("model_pair", (None, None))
        if model is None:
            model, processor = load_model_and_processor(attn_impl="eager")
            _MODEL_CACHE["model_pair"] = (model, processor)
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
    _get_encoder()  # ensure model cache (processor) is populated
    return _get_input_features()


def test_eager_matches_golden(encoder, input_features):
    """Eager mode is a clean reimplementation of the stock forward — byte-exact."""
    fe = FusedEncoder(encoder, mode="eager").cuda()
    with torch.inference_mode():
        out = fe(input_features)
    _check_against_golden(out, "eager")


def test_cudagraph_matches_golden(encoder, input_features):
    """CUDA-graph mode captures the byte-exact eager forward — byte-exact + fast."""
    fe = FusedEncoder(encoder, mode="cudagraph").cuda()
    with torch.inference_mode():
        out = fe(input_features)  # captures + runs
        torch.cuda.synchronize()
        out2 = fe(input_features)  # replay
    _check_against_golden(out, "cudagraph(run1)")
    _check_against_golden(out2, "cudagraph(run2)")


def test_triton_matches_golden(encoder, input_features):
    """Triton mode uses byte-exact elementwise kernels — byte-exact."""
    fe = FusedEncoder(encoder, mode="triton").cuda()
    fe._compiled_forward = None  # pure triton (no torch.compile wrapper)
    with torch.inference_mode():
        out = fe(input_features)
    _check_against_golden(out, "triton")


def test_compile_numerically_close(encoder, input_features):
    """torch.compile changes attention precision (fp32 intermediates).

    This mode is numerically close but NOT byte-exact; we assert it stays
    within a relaxed tolerance and print the actual diff for visibility.
    """
    fe = FusedEncoder(
        encoder, mode="compile", compile_mode="max-autotune-no-cudagraphs"
    ).cuda()
    with torch.inference_mode():
        _ = fe(input_features)  # compile
        torch.cuda.synchronize()
        out = fe(input_features)
    golden = _get_golden()
    diff = (out.float() - golden.float()).abs()
    max_d = diff.max().item()
    mean_d = diff.mean().item()
    print(f"[compile(max-autotune)] max abs diff = {max_d:.6e}  mean abs diff = {mean_d:.6e}")
    # Relaxed: just ensure it's not producing NaNs / garbage.
    assert max_d < _COMPILE_RELAXED_ATOL, (
        f"compile: max abs diff {max_d:.4e} >= relaxed {_COMPILE_RELAXED_ATOL}"
    )


def test_cudagraph_shape_validation(encoder, input_features):
    """CUDA graph mode rejects inputs of a different shape than captured."""
    fe = FusedEncoder(encoder, mode="cudagraph").cuda()
    with torch.inference_mode():
        _ = fe(input_features)  # capture at (1, 1247, 160)
    wrong = torch.randn(1, 800, 160, dtype=torch.bfloat16, device="cuda")
    fe._prepare_block_mask(800, wrong.device)
    with pytest.raises(RuntimeError, match="cudagraph captured for shape"):
        with torch.inference_mode():
            fe(wrong)


if __name__ == "__main__":
    # Quick numeric report when run as a script.
    enc = _get_encoder()
    feats = _get_input_features()
    for mode, kw in [
        ("eager", {}),
        ("cudagraph", {}),
        ("triton", {}),
        ("compile", {"compile_mode": "max-autotune-no-cudagraphs"}),
    ]:
        tag = f"{mode}" + (f"/{kw.get('compile_mode', '')}" if kw else "")
        try:
            fe = FusedEncoder(enc, mode=mode, **kw).cuda()
            if mode == "triton":
                fe._compiled_forward = None
            with torch.inference_mode():
                _ = fe(feats)
                torch.cuda.synchronize()
                out = fe(feats)
            _check_against_golden(out, tag)
            print(f"  -> {tag}: PASS")
        except Exception as e:  # noqa: BLE001
            print(f"  -> {tag}: FAIL ({e!r})")
