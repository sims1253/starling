"""Generate the Qwen3-ASR-1.7B golden reference (``golden/qwen3_reference.json``).

Runs the STOCK-numerics Python path (eager encoder + the model's own decoder
layers -- ``use_fused_llm=False``, i.e. the same op sequence the C++ port
mirrors) on the short/medium/long fixtures and records the emitted token ids,
transcript text, prompt lengths, and decode budgets. The output gates the
C++/GGML engine, so it records the raw ``ids`` stream per chunk -- not just
the text -- because the port needs byte-exact text parity and the ids
localize any divergence.

The capture mirrors ``Qwen3Backend.transcribe`` (the Python server path via
``ModelBackend._transcribe_chunked``) EXACTLY, chunk policy included:

  * max_chunk = min(30 s, (max_new_tokens - 32) / 5)   (= 30 s at 200 tokens)
  * audio <= max_chunk: single shot, budget = min(200, ceil(dur * 5) + 32)
  * longer audio: contiguous 30 s waveform chunks, the LAST chunk passed
    through SHORT (qwen3 does not zero-pad the tail -- the mel-level chunk
    padding handles it), per-chunk budget = min(200, ceil(chunk_dur * 5) + 32)
  * per-chunk text is ``processor.decode(ids, return_format=
    "transcription_only")`` (the ``<asr_text>`` extraction); texts joined
    with whitespace-collapsed ``" ".join``

Greedy decode, eos <|im_end|> (151645) -- the serving path's stop.

The output is gitignored (it requires the ~4 GB model); re-run this after
pulling a new model revision to refresh the reference.

Usage (from the repo root, GPU):
    uv run python scripts/make_qwen3_golden.py
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import soundfile as sf
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "tests" / "fixtures"
GOLDEN_PATH = REPO_ROOT / "golden" / "qwen3_reference.json"
FIXTURE_NAMES = ("short", "medium", "long")

SAMPLE_RATE = 16000
MAX_NEW_TOKENS = 200
MAX_CACHE_LEN = 4096
CHUNK_SECONDS = 30.0
EOS_TOKEN_ID = 151645


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


def extract_transcription(text: str) -> str:
    """Mirror processor.decode(..., return_format="transcription_only")."""
    from transformers.models.qwen3_asr.processing_qwen3_asr import _parse_single_output

    return _parse_single_output(text)["transcription"]


def transcribe_chunk(pipe: Any, processor: Any, wav: torch.Tensor, budget: int) -> tuple[str, list[int]]:
    """One greedy transcribe through the stock-numerics pipeline."""
    from starling.qwen3.audio import build_inputs

    inputs = build_inputs(processor, wav, sr=SAMPLE_RATE)
    text, ids = pipe.transcribe(
        inputs["input_features"],
        inputs["input_ids"],
        inputs.get("input_features_mask"),
        max_new_tokens=budget,
    )
    return text, ids[0].cpu().tolist()


def load_fixture(path: Path) -> torch.Tensor:
    """(1, n) float32 16 kHz waveform on cuda."""
    samples, sr = sf.read(str(path))
    assert sr == SAMPLE_RATE, f"fixture {path} is {sr} Hz"
    if samples.ndim > 1:
        samples = samples[:, 0]
    return torch.from_numpy(samples.astype("float32")).unsqueeze(0).contiguous().to("cuda")


def main() -> int:
    from starling.qwen3.audio import build_inputs
    from starling.qwen3.config import MODEL_ID
    from starling.qwen3.loader import load_model_and_processor
    from starling.qwen3.pipeline import MegaPipeline
    from starling.parakeet.gpu_lock import with_gpu_lock

    with with_gpu_lock(
        session="ggml-goldens",
        model="Qwen3-ASR-1.7B-hf",
        eta_min=20,
        note="capturing qwen3 C++ reference goldens",
    ):
        print("[qwen3-golden] loading model (eager, bf16) ...")
        model, processor = load_model_and_processor(attn_impl="eager")
        # STOCK numerics: eager encoder + the model's own decoder layers. This
        # is the op-for-op oracle the C++ engine mirrors (the fused/multistep
        # paths are accelerations of exactly this computation).
        pipe = MegaPipeline(
            model,
            processor,
            max_cache_len=MAX_CACHE_LEN,
            encoder_mode="eager",
            use_fused_llm=False,
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
                "pad_last_chunk": False,
                "eos_token_id": EOS_TOKEN_ID,
                "encoder_mode": "eager",
                "llm": "model-layers (stock numerics)",
                "text": "transcription_only (<asr_text> extraction)",
            },
            "fixtures": {},
        }

        for name in FIXTURE_NAMES:
            wav = load_fixture(FIXTURES / f"{name}.wav")
            duration = wav.shape[1] / SAMPLE_RATE
            t0 = time.perf_counter()

            chunks: list[dict[str, Any]] = []
            n_samples = wav.shape[1]
            for start in range(0, n_samples, chunk_samples):
                end = min(start + chunk_samples, n_samples)
                chunk_wav = wav[:, start:end].contiguous()
                chunk_dur = (end - start) / SAMPLE_RATE
                budget = decode_budget(chunk_dur)
                prompt_len = None
                text, ids = transcribe_chunk(pipe, processor, chunk_wav, budget)
                # ids/text consistency through the transcription_only path.
                decoded = extract_transcription(
                    processor.tokenizer.batch_decode(
                        torch.tensor([ids]), skip_special_tokens=True
                    )[0]
                )
                assert decoded == text, f"{name}: ids detokenize mismatch"
                prompt_len = int(
                    build_inputs(processor, chunk_wav, sr=SAMPLE_RATE)["input_ids"].shape[1]
                )
                assert prompt_len + budget <= MAX_CACHE_LEN + 1, (
                    f"{name}: budget would overflow the static KV cache"
                )
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
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            print(
                f"[qwen3-golden] {name}: {len(chunks)} chunk(s), "
                f"{sum(len(c['ids']) for c in chunks)} tokens in {elapsed:.1f}s: "
                f"{full_text[:80]!r}"
            )

    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_PATH.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"[qwen3-golden] wrote {GOLDEN_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
