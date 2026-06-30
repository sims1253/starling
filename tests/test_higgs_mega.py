"""Correctness gates for the higgs-audio-v3-stt megakernel.

The graph-captured Qwen3 decode (single- and multi-step) must reproduce the
golden reference token sequence EXACTLY (greedy = greedy; the only change is
when the argmax runs / when the host syncs, not the arithmetic).

NOTE ON ENVIRONMENT
-------------------
higgs-audio runs under its OWN isolated venv ``.venv-higgs`` (transformers 4.51)
because the model's trust_remote_code modeling breaks under the repo's shared
``transformers 5.13`` venv (see ``src/starling/higgs/loader.py``). These tests
therefore run under that venv, not ``uv run pytest``:

    .venv-higgs/bin/python -m pytest tests/test_higgs_mega.py -q

The fixtures (``tests/fixtures/*.wav``) live in the main repo checkout; the test
resolves them from either the worktree or the main repo.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch
import soundfile as sf

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MAIN_REPO = Path("/home/m0hawk/Documents/starling")
_FIXTURES_DIR = (
    _MAIN_REPO / "tests" / "fixtures"
    if (_MAIN_REPO / "tests" / "fixtures" / "short.wav").exists()
    else _REPO_ROOT / "tests" / "fixtures"
)
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "ref"))

from starling.higgs.config import EOS_TOKEN_IDS  # noqa: E402
from starling.higgs.loader import load_model_and_tokenizer, make_collator  # noqa: E402
from starling.higgs.llm_mega import LLMMega  # noqa: E402
from starling.higgs.multistep import MultiStepLLMMega  # noqa: E402
import transcribe as ref  # noqa: E402  (upstream prompt/sample builder)

# Module-level caches so the ~5GB model loads once for the whole module.
_MODEL = None
_TOK = None
_COLL = None


def _load():
    global _MODEL, _TOK, _COLL
    if _MODEL is None:
        _MODEL, _TOK = load_model_and_tokenizer()
        _COLL = make_collator(_MODEL)
    return _MODEL, _TOK, _COLL


def _golden() -> dict:
    return json.loads((_REPO_ROOT / "golden" / "higgs_golden.json").read_text())


def _build_batch(coll, tok, audio_np: np.ndarray) -> dict:
    input_ids = ref._build_input_tokens(tok, ref.DEFAULT_PROMPT, enable_thinking=True)
    sample = ref._build_sample(audio_np, input_ids, sample_rate=16000)
    batch = asdict(coll([sample]))
    return {k: (v.to("cuda").contiguous() if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()}


def _fixture_audio(name: str) -> np.ndarray:
    a, sr = sf.read(str(_FIXTURES_DIR / f"{name}.wav"))
    if a.ndim > 1:
        a = a.mean(axis=1)
    return np.asarray(a, dtype=np.float32)


# Run only on short for the default (fast) suite; medium/long are heavier.
@pytest.mark.parametrize("name", ["short", "medium", "long"])
def test_single_step_byte_exact(name: str) -> None:
    """LLMMega (single-step graph) reproduces the golden token ids exactly."""
    model, tok, coll = _load()
    golden = _golden()["fixtures"][name]
    batch = _build_batch(coll, tok, _fixture_audio(name))
    llm = LLMMega(model, max_cache_len=2048)
    res = llm.generate(
        batch, max_new_tokens=len(golden["gen_ids"]) + 2,
        eos_token_ids=EOS_TOKEN_IDS, tokenizer=tok,
    )
    assert res.ids[0].tolist() == golden["gen_ids"], (
        f"{name}: single-step decode diverged from golden"
    )


@pytest.mark.parametrize("name,k", [("short", 4), ("medium", 8), ("long", 16)])
def test_multi_step_byte_exact(name: str, k: int) -> None:
    """MultiStepLLMMega (K-step graph) reproduces the golden token ids exactly."""
    model, tok, coll = _load()
    golden = _golden()["fixtures"][name]
    batch = _build_batch(coll, tok, _fixture_audio(name))
    llm = MultiStepLLMMega(model, max_cache_len=2048, steps_per_replay=k)
    res = llm.generate(
        batch, max_new_tokens=len(golden["gen_ids"]) + 2,
        eos_token_ids=EOS_TOKEN_IDS, tokenizer=tok,
    )
    assert res.ids[0].tolist() == golden["gen_ids"], (
        f"{name} K={k}: multi-step decode diverged from golden"
    )


@pytest.mark.parametrize("name", ["short", "medium", "long"])
def test_fused_decode_byte_exact(name: str) -> None:
    """FusedLLMMega (fused Triton elementwise kernels) reproduces golden exactly.

    The fused path (RMSNorm / SwiGLU / residual / QK-norm as single-launch
    Triton kernels) must still match the eager reference bit-for-bit.
    """
    from starling.higgs.fused_decode import FusedLLMMega

    model, tok, coll = _load()
    golden = _golden()["fixtures"][name]
    batch = _build_batch(coll, tok, _fixture_audio(name))
    llm = FusedLLMMega(model, max_cache_len=2048)
    res = llm.generate(
        batch, max_new_tokens=len(golden["gen_ids"]) + 2,
        eos_token_ids=EOS_TOKEN_IDS, tokenizer=tok,
    )
    assert res.ids[0].tolist() == golden["gen_ids"], (
        f"{name}: fused decode diverged from golden"
    )


def test_transcribe_matches_golden_text() -> None:
    """End-to-end pipeline text matches the golden transcript (short clip)."""
    from starling.higgs.pipeline import HiggsMega

    model, tok, _ = _load()
    golden = _golden()["fixtures"]["short"]
    pipe = HiggsMega(model, tok, decoder="multi", max_cache_len=2048, steps_per_replay=8)
    text = pipe.transcribe(_fixture_audio("short"), sample_rate=16000, max_new_tokens=64)
    assert text.strip() == golden["text"].strip(), (
        f"transcribed text mismatch:\n got: {text!r}\n ref: {golden['text']!r}"
    )
