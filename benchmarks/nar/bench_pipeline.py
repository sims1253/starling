#!/usr/bin/env python3
"""Benchmark: stock transformers vs the starling.nar megakernel.

Granite-Speech-4.1-2b-NAR is non-autoregressive (one bidirectional forward), so
both stock and starling are single-pass. The starling win comes from CUDA-graph
capture of the encoder trunk + the torch.compiled LLM editor forward, removing
host launch overhead across the ~16 encoder + 40 LLM layers.

Reports wall-clock transcribe time + RTFx for short/medium/long fixtures,
best-of-N warm runs (model load + graph capture excluded). All starling outputs
are byte-identical to stock eager.

Usage:
    uv run python benchmarks/nar/bench_pipeline.py
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

MODEL_ID = "ibm-granite/granite-speech-4.1-2b-nar"
FIXTURES = [
    ("short", _REPO_ROOT / "tests" / "fixtures" / "short.wav"),
    ("medium", _REPO_ROOT / "tests" / "fixtures" / "medium.wav"),
    ("long", _REPO_ROOT / "tests" / "fixtures" / "long.wav"),
]
WARMUP = 6
ITERS = 7


def _load():
    import soundfile as sf
    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, attn_implementation="sdpa", trust_remote_code=True
    ).to("cuda")
    model.eval()

    tiers = {}
    for tier, path in FIXTURES:
        wav, sr = sf.read(str(path))
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        wav_t = torch.from_numpy(wav).float().to("cuda")
        inputs = processor(audios=wav_t, device="cuda")
        tiers[tier] = {
            "feats": inputs["input_features"].to(torch.bfloat16),
            "attn": inputs["attention_mask"],
            "dur": len(wav) / sr,
        }
    return model, processor, tiers


def _bench(fn) -> float:
    """Median wall ms over ITERS timed runs (after WARMUP warmups)."""
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(ITERS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times)


def main() -> int:
    from starling.nar import NarMega
    from starling.parakeet.gpu_lock import with_gpu_lock

    model, processor, tiers = _load()
    mega = NarMega(model)

    rows = []
    with with_gpu_lock(
        session="nar-bench", model=MODEL_ID, eta_min=10, note="nar pipeline bench"
    ):
        for tier in ("short", "medium", "long"):
            st = tiers[tier]
            feats, attn, dur = st["feats"], st["attn"], st["dur"]

            # --- stock eager ---
            def _stock():
                with torch.inference_mode():
                    model.transcribe(feats, attention_mask=attn)

            stock_ms = _bench(_stock)

            # --- starling (compile + graph capture happen on the warmup calls) ---
            def _mega():
                with torch.inference_mode():
                    mega.transcribe(feats, attn)

            mega_ms = _bench(_mega)

            # correctness check (cheap)
            with torch.inference_mode():
                mega.transcribe(feats, attn)
                preds, _ = mega.transcribe(feats, attn)
                stock_preds = model.transcribe(feats, attention_mask=attn).preds[0].tolist()
            exact = preds[0] == stock_preds

            rows.append((tier, dur, stock_ms, mega_ms, exact))

    print(f"\n{'tier':<7} {'audio':>6} {'stock':>9} {'starling':>10} {'speedup':>8} {'RTFx':>7} {'exact':>6}")
    print("-" * 60)
    for tier, dur, stock_ms, mega_ms, exact in rows:
        speedup = stock_ms / mega_ms
        rtfx = dur / (mega_ms / 1000.0)
        print(
            f"{tier:<7} {dur:>5.1f}s {stock_ms:>7.0f}ms {mega_ms:>8.0f}ms "
            f"{speedup:>7.1f}x {rtfx:>6.0f}x {str(exact):>6}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
