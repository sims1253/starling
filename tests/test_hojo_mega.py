"""Correctness gates for the Hojo-ASR-V1 megakernel.

The pipeline must reproduce the golden reference token sequence EXACTLY on
short/medium/long. The decoder is beam-4 (driven via the stock
``decoder_model.generate``), so the only acceptable result is an exact match of
both the ``gen_ids`` and the decoded transcript text.

NOTE ON ENVIRONMENT
-------------------
Hojo-ASR-V1 runs under its OWN isolated venv ``.venv-hojo`` (transformers
4.57.6, torch 2.7.1+cu128) because the model depends on ``hojo-asr`` and the
Qwen3-Omni / Qwen3 modeling that ships with that transformers version. Run via
uv targeting that venv (``--no-project`` so uv does not sync to the project's
pinned env):

    uv run --no-project --python .venv-hojo/bin/python python -m pytest tests/test_hojo_mega.py -q

Use ``uv run ... python -m pytest``, not bare ``uv run pytest``. The model loads
slowly (~10 s); be patient. The whole module gates on transformers 4.57 + CUDA
+ the model bundle being present.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import transformers

# Gate on the isolated env + CUDA.
_TFV = tuple(int(p) for p in transformers.__version__.split("+", 1)[0].split(".")[:2])
if _TFV != (4, 57):
    pytest.skip(
        "hojo tests require the isolated .venv-hojo transformers 4.57 env",
        allow_module_level=True,
    )
if not torch.cuda.is_available():
    pytest.skip("hojo tests require CUDA", allow_module_level=True)

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from starling.hojo.pipeline import HojoMega  # noqa: E402

_FIXTURES_DIR = _REPO_ROOT / "tests" / "fixtures"
_MODEL_DIR = _REPO_ROOT / ".hf-cache" / "hojo-asr-v1"
if not (_FIXTURES_DIR / "short.wav").exists():
    pytest.skip("hojo tests require tests/fixtures/*.wav", allow_module_level=True)
if not _MODEL_DIR.exists():
    pytest.skip("hojo tests require .hf-cache/hojo-asr-v1 model bundle", allow_module_level=True)

# Module-level cache so the ~5GB model loads once for the whole module.
_PIPE: HojoMega | None = None


def _pipe() -> HojoMega:
    global _PIPE
    if _PIPE is None:
        _PIPE = HojoMega.from_pretrained(folder_path=str(_MODEL_DIR), device="cuda:0")
    return _PIPE


def _golden() -> dict:
    return json.loads((_REPO_ROOT / "golden" / "hojo_reference.json").read_text())


def _fixture_audio(name: str) -> np.ndarray:
    import soundfile as sf
    a, sr = sf.read(str(_FIXTURES_DIR / f"{name}.wav"))
    if a.ndim > 1:
        a = a[:, 0]  # match reference's waveform[0:1, :] -> channel 0
    return np.asarray(a, dtype=np.float32)


@pytest.mark.parametrize("name", ["short", "medium", "long"])
def test_gen_ids_byte_exact(name: str) -> None:
    """HojoMega reproduces the golden ``gen_ids`` exactly (the authoritative gate)."""
    g = _golden()["fixtures"][name]
    pipe = _pipe()
    _, gen_ids = pipe.transcribe_gen_ids(_fixture_audio(name), sample_rate=16000)
    assert gen_ids == g["gen_ids"], (
        f"{name}: gen_ids diverged from golden\n"
        f"  got ({len(gen_ids)}): {gen_ids[:20]}\n"
        f"  ref ({g['gen_ids_len']}): {g['gen_ids'][:20]}"
    )


@pytest.mark.parametrize("name", ["short", "medium", "long"])
def test_transcribe_text_exact(name: str) -> None:
    """HojoMega.transcribe matches the golden transcript text."""
    g = _golden()["fixtures"][name]
    pipe = _pipe()
    text = pipe.transcribe(_fixture_audio(name), sample_rate=16000)
    assert text.strip() == g["text"].strip(), (
        f"{name}: transcript text mismatch\n got: {text!r}\n ref: {g['text']!r}"
    )


def test_short_transcript_value() -> None:
    """Short fixture produces the exact known transcript (definition-of-done gate)."""
    pipe = _pipe()
    text = pipe.transcribe(_fixture_audio("short"), sample_rate=16000)
    expected = (
        "well i don't wish to see it any more observed phebe turning away "
        "her eyes it is certainly very like the old portrait"
    )
    assert text == expected, f"\n got: {text!r}\n ref: {expected!r}"
