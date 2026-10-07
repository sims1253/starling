#!/usr/bin/env python3
"""Generate the Qwen3-ASR-0.6B golden reference (``golden/qwen3_06_reference.json``).

Same capture as ``scripts/make_qwen3_golden.py`` (stock-numerics Python path:
eager encoder + the model's own decoder layers, the exact chunk policy the C++
engine and the Python server mirror) with the 0.6B loader and hub id. The
chunk-policy helpers are IMPORTED from the 1.7B generator so the two captures
share one policy implementation.

The output gates the C++/GGML engine (``qwen3_06.intree.text`` in
tests/native_parity_manifest.json) and the fused-pipeline tests
(tests/test_qwen3_06_pipeline.py). It records the raw ``ids`` stream per chunk
so a divergence localizes to the first flipped token.

The output is gitignored (it requires the ~1.6 GB model); re-run after pulling
a new model revision to refresh the reference.

Usage (from the repo root, GPU):
    uv run python scripts/make_qwen3_06_golden.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
# Chunk-policy helpers shared with the 1.7B capture (one policy, two models).
from make_qwen3_golden import (  # noqa: E402
    EOS_TOKEN_ID,
    FIXTURES,
    FIXTURE_NAMES,
    MAX_CACHE_LEN,
    MAX_NEW_TOKENS,
    SAMPLE_RATE,
    decode_budget,
    effective_chunk_seconds,
    extract_transcription,
    join_chunk_texts,
    load_fixture,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = REPO_ROOT / "golden" / "qwen3_06_reference.json"


def main() -> int:
    from starling.qwen3.audio import build_inputs
    from starling.qwen3_06.config import MODEL_ID, MODEL_REVISION
    from starling.qwen3_06.loader import load_model_and_processor
    from starling.qwen3_06.pipeline import MegaPipeline
    from starling.parakeet.gpu_lock import with_gpu_lock

    with with_gpu_lock(
        session="ggml-goldens",
        model="Qwen3-ASR-0.6B-hf",
        eta_min=20,
        note="capturing qwen3_06 C++ reference goldens",
    ):
        print("[qwen3_06-golden] loading model (eager, bf16) ...")
        # Pinned revision comes from the 0.6B loader default (MODEL_REVISION),
        # matching the converter snapshot — no moving-main drift (#353).
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
            "revision": MODEL_REVISION,
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
                inputs = build_inputs(processor, chunk_wav, sr=SAMPLE_RATE)
                # Guard BEFORE the decode: an overflowing budget would be
                # consumed by pipe.transcribe (it does not clamp
                # max_new_tokens internally) and corrupt the capture.
                prompt_len = int(inputs["input_ids"].shape[1])
                if prompt_len + budget > MAX_CACHE_LEN + 1:
                    raise SystemExit(
                        f"{name}: budget would overflow the static KV cache "
                        f"(prompt {prompt_len} + budget {budget} > "
                        f"max_cache_len {MAX_CACHE_LEN} + 1)"
                    )
                text, ids = pipe.transcribe(
                    inputs["input_features"],
                    inputs["input_ids"],
                    inputs.get("input_features_mask"),
                    max_new_tokens=budget,
                )
                ids = ids[0].cpu().tolist()
                # ids/text consistency through the transcription_only path
                # (mirrors the 1.7B capture so the two goldens cannot drift).
                decoded = extract_transcription(
                    processor.tokenizer.batch_decode(
                        torch.tensor([ids]), skip_special_tokens=True
                    )[0]
                )
                assert decoded == text, f"{name}: ids detokenize mismatch"
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
                f"[qwen3_06-golden] {name}: {len(chunks)} chunk(s), "
                f"{sum(len(c['ids']) for c in chunks)} tokens in {elapsed:.1f}s: "
                f"{full_text[:80]!r}"
            )

    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_PATH.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"[qwen3_06-golden] wrote {GOLDEN_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
