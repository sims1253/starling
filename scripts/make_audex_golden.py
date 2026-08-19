"""Generate the Nemotron-Labs-Audex-2B golden reference (``golden/audex_reference.json``).

Runs the STOCK-numerics Python path (the stock Qwen2AudioEncoder forward +
the model's own decoder layers -- ``use_fused_llm=False``, i.e. the same op
sequence the C++ port mirrors) on the short/medium/long fixtures and records
the emitted token ids, transcript text, prompt lengths, and decode budgets.
The output gates the C++/GGML engine, so it records the raw ``ids`` stream
per chunk -- not just the text -- because the port needs byte-exact text
parity and the ids localize any divergence.

The capture mirrors ``AudexBackend.transcribe`` (the Python server path via
``ModelBackend._transcribe_chunked``) EXACTLY, chunk policy included:

  * max_chunk = min(30 s, (max_new_tokens - 32) / 5)   (= 30 s at 200 tokens
    -- exactly one 30 s clip per chunk, so every prompt carries the FIXED
    750 <so_embedding> audio slots)
  * audio <= max_chunk: single shot, budget = min(200, ceil(dur * 5) + 32)
  * longer audio: contiguous 30 s waveform chunks, each padded to a full
    30 s clip at the mel level (padding="max_length"), per-chunk budget =
    min(200, ceil(chunk_dur * 5) + 32)
  * per-chunk text is ``MegaPipeline._decode_response`` (the quote
    extraction); texts joined with whitespace-collapsed ``" ".join``

Greedy decode, eos <|im_end|> (11) -- the serving path's stop.

The output is gitignored (it requires the ~6 GB model); re-run this after
pulling a new model revision to refresh the reference.

Usage (from the repo root, GPU):
    uv run python scripts/make_audex_golden.py
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import soundfile as sf

# Purge preamble: the main .venv's .pth imports starling from the MAIN
# checkout at interpreter startup; this script must run against THIS
# worktree's sources (whose loader resolves .hf-cache next to the repo).
import sys
for _k in [k for k in list(sys.modules) if k == "starling" or k.startswith("starling.")]:
    del sys.modules[_k]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "tests" / "fixtures"
GOLDEN_PATH = REPO_ROOT / "golden" / "audex_reference.json"
FIXTURE_NAMES = ("short", "medium", "long")

SAMPLE_RATE = 16000
MAX_NEW_TOKENS = 200
MAX_CACHE_LEN = 4096
CHUNK_SECONDS = 30.0
EOS_TOKEN_ID = 11


def decode_budget(duration_s: float) -> int:
    """Mirror ModelBackend._decode_budget: scale the cap to the chunk length."""
    estimated = max(1, math.ceil(duration_s * 5.0) + 32)
    return min(max(1, MAX_NEW_TOKENS), estimated)


def effective_chunk_seconds() -> float:
    token_limited = max(0.1, (MAX_NEW_TOKENS - 32) / 5.0)
    return min(CHUNK_SECONDS, token_limited)


def join_chunk_texts(texts: list[str]) -> str:
    """Mirror ModelBackend._transcribe_chunked's join."""
    return " ".join(" ".join(texts).split())


def transcribe_chunk(pipe: Any, wav, budget: int) -> tuple[str, list[int]]:
    """One greedy transcribe through the stock-numerics pipeline."""
    text, ids = pipe.transcribe(wav, max_new_tokens=budget)
    return text, ids[0].cpu().tolist()


def load_fixture(path: Path):
    """(n,) float32 16 kHz waveform on cpu."""
    samples, sr = sf.read(str(path))
    assert sr == SAMPLE_RATE, f"fixture {path} is {sr} Hz"
    if samples.ndim > 1:
        samples = samples[:, 0]
    return samples.astype("float32").copy()


def main() -> int:
    from starling.audex.config import MODEL_ID
    from starling.audex.pipeline import MegaPipeline
    from starling.parakeet.gpu_lock import with_gpu_lock

    with with_gpu_lock(
        session="ggml-goldens",
        model="Nemotron-Labs-Audex-2B",
        eta_min=25,
        note="capturing audex C++ reference goldens",
    ):
        print("[audex-golden] loading model (eager, bf16) ...")
        # STOCK numerics: the stock Qwen2AudioEncoder forward + the model's
        # own decoder layers. This is the op-for-op oracle the C++ engine
        # mirrors (the fused/multistep paths are accelerations of exactly
        # this computation).
        pipe = MegaPipeline.from_pretrained(
            max_cache_len=MAX_CACHE_LEN, use_fused_llm=False
        )

        max_chunk = effective_chunk_seconds()
        chunk_samples = max(1, round(max_chunk * SAMPLE_RATE))
        out: dict[str, Any] = {
            "model": MODEL_ID,
            "policy": {
                "sample_rate": SAMPLE_RATE,
                "max_new_tokens": MAX_NEW_TOKENS,
                "max_cache_len": MAX_CACHE_LEN,
                "chunk_seconds": max_chunk,
                "audio_tokens_per_clip": 750,
                "eos_token_id": EOS_TOKEN_ID,
                "encoder": "stock Qwen2AudioEncoder (eager)",
                "llm": "model-layers (stock numerics)",
                "text": "quote extraction (_decode_response)",
            },
            "fixtures": {},
        }

        for name in FIXTURE_NAMES:
            wav = load_fixture(FIXTURES / f"{name}.wav")
            duration = len(wav) / SAMPLE_RATE
            t0 = time.perf_counter()

            chunks: list[dict[str, Any]] = []
            n_samples = len(wav)
            for start in range(0, n_samples, chunk_samples):
                end = min(start + chunk_samples, n_samples)
                chunk_wav = wav[start:end].copy()
                chunk_dur = (end - start) / SAMPLE_RATE
                budget = decode_budget(chunk_dur)
                prompt_len = 23 + 750  # fixed: 4-token prefix + 750 audio + 19-token suffix
                assert prompt_len + budget <= MAX_CACHE_LEN + 1, (
                    f"{name}: budget would overflow the static KV cache"
                )
                text, ids = transcribe_chunk(pipe, chunk_wav, budget)
                chunks.append(
                    {
                        "start_s": start / SAMPLE_RATE,
                        "end_s": end / SAMPLE_RATE,
                        "prompt_len": prompt_len,
                        "budget": budget,
                        "text": text,
                        "ids": ids,
                    }
                )

            full_text = join_chunk_texts([c["text"] for c in chunks])
            out["fixtures"][name] = {
                "fixture": f"tests/fixtures/{name}.wav",
                "seconds": duration,
                "text": full_text,
                "chunks": chunks,
            }
            import torch

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            print(
                f"[audex-golden] {name}: {len(chunks)} chunk(s), "
                f"{sum(len(c['ids']) for c in chunks)} tokens in {elapsed:.1f}s: "
                f"{full_text[:80]!r}"
            )

    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_PATH.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"[audex-golden] wrote {GOLDEN_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
