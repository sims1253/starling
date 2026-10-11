"""GPU-gated parity tests for the Voxtral ggml engine vs the golden reference.

Module-level CUDA/golden skips, then compares engine transcripts vs golden/voxtral_reference.json (captured by
scripts/goldens/voxtral_golden.py) on the short/medium/long fixtures.

Collects and skips cleanly on boxes without CUDA, the model, or goldens
(this box: CPU-only, tiny GGUF only — everything below skips at import).
"""

from __future__ import annotations

import ctypes
import json
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_TOOLS = os.path.normpath(os.path.join(_HERE, "..", "tools"))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA required for Voxtral ggml parity tests", allow_module_level=True)

REPO_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
GOLDEN_PATH = os.path.join(REPO_ROOT, "golden", "voxtral_reference.json")
FIXTURES = os.path.join(REPO_ROOT, "tests", "fixtures")

if not os.path.exists(GOLDEN_PATH):
    pytest.skip(
        f"golden {GOLDEN_PATH} missing; run scripts/goldens/voxtral_golden.py",
        allow_module_level=True,
    )


def _load_golden():
    with open(GOLDEN_PATH) as f:
        return json.load(f)


def _engine():
    """The in-process ggml voxtral engine (tiny GGUF on CPU-only boxes)."""
    from starling_ggml import VOXTRAL, GgmlModel, available

    if not available():
        pytest.skip("libstarling_ggml unavailable (build cpp/ first)")
    for cand in (
        os.environ.get("STARLING_VOXTRAL_GGUF", ""),
        os.path.join(REPO_ROOT, "models", "voxtral-mini-4b-realtime-bf16-exact.gguf"),
    ):
        if cand and os.path.exists(cand):
            return GgmlModel(VOXTRAL, cand)
    pytest.skip("voxtral GGUF absent (set STARLING_VOXTRAL_GGUF)")


@pytest.fixture(scope="module")
def engine():
    eng = _engine()
    yield eng
    eng.close()


def _wav(name: str):
    import numpy as np
    import soundfile as sf

    path = os.path.join(FIXTURES, f"{name}.wav")
    if not os.path.exists(path):
        pytest.skip(f"fixture {path} not found")
    wav, sr = sf.read(path)
    if getattr(wav, "ndim", 1) > 1:
        wav = wav[:, 0]
    assert sr == 16000
    return np.ascontiguousarray(wav, dtype=np.float32)


@pytest.mark.parametrize("fixture", ["short", "medium", "long"])
def test_engine_transcript_matches_golden(engine, fixture):
    """Engine transcript matches the stock-generate golden text exactly."""
    golden = _load_golden()
    if fixture not in golden.get("fixtures", {}):
        pytest.skip(f"golden has no entry for {fixture!r}")
    wav = _wav(fixture)
    pcm = wav.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    text = engine.transcribe_pcm(pcm, len(wav))
    entry = golden["fixtures"][fixture]
    assert text == entry["text"], f"\ngot:  {text!r}\nwant: {entry['text']!r}"


# This module compares the native engine to the stock golden
# (scripts/goldens/voxtral_golden.py) without keeping a Python model copy
# resident alongside the native weights and graphs.
