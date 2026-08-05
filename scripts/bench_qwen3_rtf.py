"""RTF benchmark for the Qwen3-ASR megakernel vs stock transformers.

Single RTX 5090, bf16, model load excluded. Measures short/medium/long tiers
against the same audio used by the granite/parakeet tables, producing the
``starling`` vs ``stock transformers`` RTFx numbers.

Uses the shared GPU lock so concurrent benches don't corrupt timing.
"""

from __future__ import annotations

import json
import statistics
import time

import torch

from starling.qwen3.audio import build_inputs, load_wav
from starling.qwen3.config import GOLDEN_DIR, MODEL_ID, REPO_ROOT
from starling.qwen3.loader import load_model_and_processor
from starling.qwen3.pipeline import MegaPipeline


def _fixture_path(name: str) -> str:
    p = REPO_ROOT / "tests" / "fixtures" / name
    if p.exists():
        return str(p)
    p = REPO_ROOT.parent / "starling" / "tests" / "fixtures" / name
    return str(p)


@torch.inference_mode()
def bench_mega(pipe: MegaPipeline, inputs, repeats: int = 5) -> list[float]:
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        pipe.transcribe(
            inputs["input_features"],
            inputs["input_ids"],
            inputs.get("input_features_mask"),
            max_new_tokens=400,
        )
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return times


@torch.inference_mode()
def bench_stock(model, processor, inputs, repeats: int = 3) -> list[float]:
    """Stock transformers generate() reference."""
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.generate(
            input_ids=inputs["input_ids"],
            input_features=inputs["input_features"],
            input_features_mask=inputs.get("input_features_mask"),
            max_new_tokens=400,
            do_sample=False,
            num_beams=1,
        )
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return times


def main() -> int:
    from starling.parakeet.gpu_lock import with_gpu_lock

    tiers = [("short", "short.wav"), ("medium", "medium.wav"), ("long", "long.wav")]
    results = {}

    print(f"[bench] loading {MODEL_ID} ...")
    model, processor = load_model_and_processor(attn_impl="eager")

    with with_gpu_lock(session="qwen3-asr", model=MODEL_ID, eta_min=20, note="RTF benchmark"):
        # (1) stock reference numbers
        print("[bench] --- stock transformers ---")
        for label, fname in tiers:
            wav, sr = load_wav(_fixture_path(fname))
            inputs = build_inputs(processor, wav, sr=sr)
            audio_s = wav.shape[1] / sr
            t = bench_stock(model, processor, inputs, repeats=3)
            ms = statistics.median(t)
            rtfx = audio_s / (ms / 1000.0)
            results.setdefault(label, {})["audio_s"] = round(audio_s, 2)
            results[label]["stock_ms"] = round(ms, 1)
            results[label]["stock_rtfx"] = round(rtfx, 1)
            print(f"[bench] {label:7s} {audio_s:5.1f}s: stock {ms:8.1f}ms ({rtfx:6.1f}x)")

        # (2) mega numbers (separate pipeline; frees stock KV state implicitly)
        del model
        torch.cuda.empty_cache()
        print("[bench] --- starling (mega) ---")
        # cudagraph encoder (custom windowed-attention kernel) is byte-exact +
        # ~3-4x faster than eager; report both for transparency.
        for enc_mode in ["cudagraph", "eager"]:
            pipe = MegaPipeline.from_pretrained(encoder_mode=enc_mode, max_cache_len=4096)
            for label, fname in tiers:
                wav, sr = load_wav(_fixture_path(fname))
                inputs = build_inputs(processor, wav, sr=sr)
                audio_s = wav.shape[1] / sr
                t = bench_mega(pipe, inputs, repeats=5)
                ms = statistics.median(t)
                rtfx = audio_s / (ms / 1000.0)
                results[label][f"mega_{enc_mode}_ms"] = round(ms, 1)
                results[label][f"mega_{enc_mode}_rtfx"] = round(rtfx, 1)
                print(f"[bench] {label:7s} {audio_s:5.1f}s: mega[{enc_mode}] {ms:7.1f}ms ({rtfx:6.1f}x)")
            del pipe
            torch.cuda.empty_cache()

    out = GOLDEN_DIR / "bench_rtf.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n[bench] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
