"""Generate the granite-speech-4.1-2b golden reference (``golden/granite_reference.json``).

Runs the STOCK-numerics Python path (eager encoder + the model's own decoder
layers -- ``OptFlags(multistep_graph=False)`` + ``use_fused_llm=False``, i.e.
the same op sequence the C++ port mirrors) on the short/medium/long fixtures
and records the emitted token ids, transcript text, prompt lengths, and decode
budgets. The output gates the C++/GGML engine, so it records the raw ``ids``
stream per chunk -- not just the text -- because the port needs byte-exact
text parity and the ids localize any divergence.

The capture mirrors ``GraniteBackend.transcribe`` (the Python server path)
EXACTLY, chunk policy included:

  * max_chunk = min(30 s, (max_new_tokens - 32) / 5)   (= 30 s at 200 tokens)
  * audio <= max_chunk: single shot, budget = min(200, ceil(dur * 5) + 32)
  * longer audio: 30 s waveform chunks (``chunk_audio``, the last chunk
    zero-padded to the full chunk length), per-chunk budget =
    max(1, min(budget(chunk_dur), 640 - prompt_len - 1))
  * texts joined with ``" ".join(t.strip())`` and whitespace-collapsed

Greedy decode only (``speculative=False``; the CTC speculative path is a
follow-up and is byte-identical to greedy by construction).

The output is gitignored (it requires the ~5 GB model); re-run this after
pulling a new model revision to refresh the reference.

Usage (from the repo root, GPU):
    uv run python scripts/make_granite_golden.py
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
GOLDEN_PATH = REPO_ROOT / "golden" / "granite_reference.json"
FIXTURE_NAMES = ("short", "medium", "long")

SAMPLE_RATE = 16000
MAX_NEW_TOKENS = 200
MAX_CACHE_LEN = 640
CHUNK_SECONDS = 30.0


def decode_budget(duration_s: float) -> int:
    """Mirror ModelBackend._decode_budget: scale the cap to the chunk length."""
    estimated = max(1, math.ceil(duration_s * 5.0) + 32)
    return min(max(1, MAX_NEW_TOKENS), estimated)


def effective_chunk_seconds() -> float:
    token_limited = max(0.1, (MAX_NEW_TOKENS - 32) / 5.0)
    return min(CHUNK_SECONDS, token_limited)


def join_chunk_texts(texts: list[str]) -> str:
    """Mirror granite.long_audio._join_chunk_texts with zero overlap."""
    joined = " ".join(t.strip() for t in texts if t and t.strip())
    return " ".join(joined.split())


def transcribe_chunk(pipe: Any, processor: Any, wav: torch.Tensor, budget: int) -> tuple[str, list[int]]:
    """One greedy transcribe through the stock-numerics pipeline."""
    from starling.granite.audio import build_inputs

    inputs = build_inputs(processor, wav)
    text, ids = pipe.transcribe(
        inputs["input_features"],
        inputs["input_ids"],
        inputs.get("input_features_mask"),
        max_new_tokens=budget,
        speculative=False,
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
    from starling.config import MODEL_ID
    from starling.flags import OptFlags
    from starling.granite.audio import build_inputs
    from starling.granite.loader import load_model_and_processor
    from starling.granite.long_audio import chunk_audio
    from starling.granite.pipeline import MegaPipeline
    from starling.parakeet.gpu_lock import with_gpu_lock

    with with_gpu_lock(
        session="ggml-goldens",
        model="granite-speech-4.1-2b",
        eta_min=20,
        note="capturing granite C++ reference goldens",
    ):
        print("[granite-golden] loading model (eager, bf16) ...")
        model, processor = load_model_and_processor(attn_impl="eager")
        # STOCK numerics: eager encoder + the model's own decoder layers. This
        # is the op-for-op oracle the C++ engine mirrors (the fused/multistep
        # paths are accelerations of exactly this computation).
        pipe = MegaPipeline(
            model,
            processor,
            encoder_mode="eager",
            use_fused_llm=False,
            flags=OptFlags(multistep_graph=False),
        )

        max_chunk = effective_chunk_seconds()
        out: dict[str, Any] = {
            "model": MODEL_ID,
            "policy": {
                "sample_rate": SAMPLE_RATE,
                "max_new_tokens": MAX_NEW_TOKENS,
                "max_cache_len": MAX_CACHE_LEN,
                "chunk_seconds": max_chunk,
                "pad_last_chunk": True,
                "speculative": False,
                "encoder_mode": "eager",
                "llm": "model-layers (stock numerics)",
            },
            "fixtures": {},
        }

        for name in FIXTURE_NAMES:
            wav = load_fixture(FIXTURES / f"{name}.wav")
            duration = wav.shape[1] / SAMPLE_RATE
            t0 = time.perf_counter()

            chunks: list[dict[str, Any]] = []
            if duration <= max_chunk:
                budget = decode_budget(duration)
                text, ids = transcribe_chunk(pipe, processor, wav, budget)
                decoded = processor.tokenizer.batch_decode(
                    torch.tensor([ids]), skip_special_tokens=True
                )[0]
                assert decoded == text, f"{name}: ids detokenize mismatch"
                chunks.append(
                    {
                        "start_s": 0.0,
                        "end_s": duration,
                        "prompt_len": None,  # filled from the ids below
                        "budget": budget,
                        "text": text,
                        "ids": ids,
                    }
                )
            else:
                for chunk_wav, start, end, _idx in chunk_audio(wav, SAMPLE_RATE, max_chunk):
                    inputs_ids_len = int(build_inputs(processor, chunk_wav)["input_ids"].shape[1])
                    budget = max(
                        1,
                        min(decode_budget(end - start), MAX_CACHE_LEN - inputs_ids_len - 1),
                    )
                    text, ids = transcribe_chunk(pipe, processor, chunk_wav, budget)
                    decoded = processor.tokenizer.batch_decode(
                        torch.tensor([ids]), skip_special_tokens=True
                    )[0]
                    assert decoded == text, f"{name}: ids detokenize mismatch"
                    chunks.append(
                        {
                            "start_s": start,
                            "end_s": end,
                            "prompt_len": inputs_ids_len,
                            "budget": budget,
                            "text": text,
                            "ids": ids,
                        }
                    )

            full_text = join_chunk_texts([c["text"] for c in chunks])
            if chunks[0]["prompt_len"] is None:
                chunks[0]["prompt_len"] = int(build_inputs(processor, wav)["input_ids"].shape[1])

            out["fixtures"][name] = {
                "fixture": f"tests/fixtures/{name}.wav",
                "seconds": duration,
                "text": full_text,
                "chunks": chunks,
            }
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            print(
                f"[granite-golden] {name}: {len(chunks)} chunk(s), "
                f"{sum(len(c['ids']) for c in chunks)} tokens in {elapsed:.1f}s: "
                f"{full_text[:80]!r}"
            )

    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_PATH.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"[granite-golden] wrote {GOLDEN_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
