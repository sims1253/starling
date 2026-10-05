"""Correctness tests for the Qwen3-ASR-0.6B megakernel vs the golden reference.

Mirror of ``tests/test_ark06_pipeline.py``: the golden file
(``golden/qwen3_06_reference.json``, captured by ``scripts/make_qwen3_06_golden.py``
with the stock-numerics path — eager encoder + the model's own decoder layers)
is gitignored; the tests ``pytest.skip`` when it is absent so the suite runs
green on a fresh clone without the 0.6B model downloaded.

The pipeline under test is the 1.7B track's ``MegaPipeline`` pointed at the
0.6B loader (every dim derived from the loaded model), exercised through the
same serve chunk policy the golden captured with.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

torch = pytest.importorskip("torch")

# Ensure the in-worktree starling package is importable when this test file is
# collected from any worktree / venv state (mirrors test_qwen3_pipeline.py).
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, "..", "src"))
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for Qwen3-ASR-0.6B megakernel tests",
)

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = REPO_ROOT / "golden" / "qwen3_06_reference.json"
FIXTURES = REPO_ROOT / "tests" / "fixtures"

# The serve chunk policy the golden was captured under (mirrors
# ModelBackend._transcribe_chunked / the C++ decode entry).
SAMPLE_RATE = 16000
MAX_NEW_TOKENS = 200


def _load_golden():
    if not GOLDEN_PATH.exists():
        pytest.skip(
            f"golden reference {GOLDEN_PATH} not found; run scripts/make_qwen3_06_golden.py"
        )
    return json.loads(GOLDEN_PATH.read_text())


def _decode_budget(duration_s: float) -> int:
    estimated = max(1, int(np.ceil(duration_s * 5.0)) + 32)
    return min(MAX_NEW_TOKENS, estimated)


@pytest.fixture(scope="module")
def golden():
    return _load_golden()


@pytest.fixture(scope="module")
def pipeline(golden):
    """Build and prewarm the 0.6B MegaPipeline once for the whole module.

    Depends on `golden` so a missing gitignored golden skips BEFORE the
    checkpoint downloads/loads (the ARK06 fixture ordering)."""
    from starling.qwen3_06.pipeline import MegaPipeline

    pipe = MegaPipeline.from_pretrained(encoder_mode="cudagraph")
    return pipe


def _wav(name: str) -> np.ndarray:
    path = FIXTURES / f"{name}.wav"
    if not path.exists():
        pytest.skip(f"fixture {path} not found; run tests/fixtures/make_fixtures.py")
    data, sr = sf.read(str(path))
    assert sr == SAMPLE_RATE, f"fixture {path} is {sr} Hz"
    if data.ndim > 1:
        data = data[:, 0]
    return np.ascontiguousarray(data, dtype=np.float32)


@needs_cuda
@pytest.mark.parametrize("fixture", ["short", "medium", "long"])
def test_transcribe_matches_golden_text(pipeline, golden, fixture):
    """The fused pipeline reproduces the stock-numerics golden transcript.

    Chunked exactly like the serving path (30 s chunks, duration-scaled
    budgets, whitespace-collapsed join); the comparison is exact-text on
    the final transcript.
    """
    if fixture not in golden["fixtures"]:
        pytest.skip(f"golden has no entry for {fixture!r}")
    from starling.qwen3.audio import build_inputs

    wav_np = _wav(fixture)
    n_samples = wav_np.shape[0]
    chunk_samples = max(1, round(30.0 * SAMPLE_RATE))

    texts: list[str] = []
    for start in range(0, n_samples, chunk_samples):
        piece = wav_np[start : min(start + chunk_samples, n_samples)]
        budget = _decode_budget(len(piece) / SAMPLE_RATE)
        wav = torch.from_numpy(piece).float().unsqueeze(0).contiguous()
        inp = build_inputs(pipeline.processor, wav, sr=SAMPLE_RATE)
        text, _ = pipeline.transcribe(
            inp["input_features"],
            inp["input_ids"],
            inp.get("input_features_mask"),
            max_new_tokens=budget,
        )
        texts.append(text)
    out = " ".join(" ".join(texts).split())
    golden_text = golden["fixtures"][fixture]["text"]
    assert out == golden_text, (
        f"{fixture}: fused qwen3_06 transcript diverges from the golden reference:\n"
        f"  golden: {golden_text[:160]!r}\n"
        f"  fused:  {out[:160]!r}"
    )
