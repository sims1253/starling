"""Deliverable 1 benchmark: autotuned Triton vs hand-picked (OFF) decode kernels.

Measures the impact of ``@triton.autotune`` (num_warps/num_stages sweep) on the
three LLM-decode-critical elementwise kernels (RMSNorm, SwiGLU, residual
scale-add) inside the CUDA-graph-captured Granite decode.

Reports (autotune ON vs OFF):
  * per-token decode GPU ms (CUDA events, pure graph replay)
  * wall-clock tok/s + RTFx (full generate loop)
  * batched B=16 RTFx

Byte-exact in both modes (verified separately): the only difference is the
launch config, never the arithmetic.

Run:  .venv/bin/python scripts/bench_autotune.py
"""
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from starling import llm_kernels as K
from starling.audio import build_inputs, load_sample_audio
from starling.golden import load_golden
from starling.llm_mega import FusedLLMMega
from starling.loader import get_components, load_model_and_processor
from starling.parakeet.gpu_lock import with_gpu_lock

MAX_NEW_TOKENS = 100
ITERS = 10
WARMUP = 3


def _wall_ms(fn, warm=WARMUP, iters=ITERS):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(ts), min(ts)


def _cuda_ms(fn, warm=WARMUP, iters=ITERS):
    torch.cuda.synchronize()
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return statistics.median(times)


def main() -> int:
    with with_gpu_lock(
        session="granite-mega",
        model="granite-speech-4.1-2b",
        eta_min=10,
        note="bench_autotune: autotune ON vs OFF decode benchmark",
    ):
        print("loading model + processor ...", flush=True)
        model, proc = load_model_and_processor(attn_impl="eager")
        comps = get_components(model)
        lm = comps["language_model"]
        lm_head = model.lm_head

        wav, sr = load_sample_audio()
        inputs = build_inputs(proc, wav)
        audio_seconds = wav.shape[1] / sr
        inputs_embeds = load_golden("inputs_embeds.pt").to("cuda", torch.bfloat16)
        n_tok = inputs["input_ids"].shape[1]
        print(f"audio {audio_seconds:.1f}s, prompt {n_tok} tokens, "
              f"{MAX_NEW_TOKENS} new tokens\n", flush=True)

        results = {}
        for label, autotune_on in [("autotune-OFF", False), ("autotune-ON", True)]:
            K.set_autotune(autotune_on)
            # Fresh decoder so graph capture runs under the chosen mode.
            dec = FusedLLMMega(lm, lm_head, max_cache_len=640)
            rep = dec.bench(inputs_embeds, max_new_tokens=MAX_NEW_TOKENS, decode_iters=ITERS)

            def _gen():
                dec.generate(inputs_embeds, max_new_tokens=MAX_NEW_TOKENS)
            wall_med, wall_min = _wall_ms(_gen, warm=WARMUP, iters=ITERS)
            tps = MAX_NEW_TOKENS / (wall_med / 1000.0)
            rtfx = audio_seconds / (wall_med / 1000.0)

            entry = {
                "gpu_ms_per_tok": rep.decode_ms_per_token,
                "prefill_ms": rep.prefill_ms,
                "wall_ms_total": wall_med,
                "wall_ms_min": wall_min,
                "wall_tok_per_s": tps,
                "rtfx_single_stream": rtfx,
                "configs": {
                    "rmsnorm": str(K._rmsnorm_kernel.best_config) if autotune_on else "default",
                    "silu_mul": str(K._silu_mul_kernel.best_config) if autotune_on else "default",
                    "residual": str(K._residual_scale_kernel.best_config) if autotune_on else "default",
                },
            }
            results[label] = entry
            print(f"[{label}] GPU {rep.decode_ms_per_token:.3f} ms/tok | "
                  f"wall {wall_med:.1f} ms ({tps:.1f} tok/s, RTFx {rtfx:.1f}x) | "
                  f"prefill {rep.prefill_ms:.2f} ms")
            if autotune_on:
                print(f"         best configs: {entry['configs']}")

        on = results["autotune-ON"]
        off = results["autotune-OFF"]
        gpu_speedup = off["gpu_ms_per_tok"] / on["gpu_ms_per_tok"]
        wall_speedup = off["wall_ms_total"] / on["wall_ms_total"]
        print(f"\n  GPU ms/tok speedup (autotune): {gpu_speedup:.3f}x  "
              f"({off['gpu_ms_per_tok']:.3f} -> {on['gpu_ms_per_tok']:.3f})")
        print(f"  wall tok/s speedup (autotune): {wall_speedup:.3f}x  "
              f"({off['wall_tok_per_s']:.1f} -> {on['wall_tok_per_s']:.1f} tok/s)")
        print(f"  RTFx: {off['rtfx_single_stream']:.1f}x -> {on['rtfx_single_stream']:.1f}x")

        # Batched B=16 (fused batched decode path uses the same kernels).
        print("\n--- batched B=16 ---", flush=True)
        from starling.batched import BatchedPipeline

        for label, autotune_on in [("autotune-OFF", False), ("autotune-ON", True)]:
            K.set_autotune(autotune_on)
            feats = inputs["input_features"].to(torch.bfloat16)
            ids = inputs["input_ids"]
            mask = inputs.get("input_features_mask")
            pipe = BatchedPipeline(
                model, proc, max_batch_size=16, encoder_mode="cudagraph",
            )
            B = 16
            fl = [feats] * B
            il = [ids] * B
            ml = [mask] * B

            def _batch():
                pipe.transcribe_batch(fl, il, ml, max_new_tokens=MAX_NEW_TOKENS)
            wall_med, _ = _wall_ms(_batch, warm=2, iters=4)
            rtfx = B * audio_seconds / (wall_med / 1000.0)
            results[f"batched_B16_{label}"] = {
                "wall_ms_total": wall_med,
                "rtfx": rtfx,
                "tok_per_s": B * MAX_NEW_TOKENS / (wall_med / 1000.0),
            }
            print(f"[B16 {label}] wall {wall_med:.1f} ms, RTFx {rtfx:.1f}x, "
                  f"{B * MAX_NEW_TOKENS / (wall_med / 1000.0):.0f} tok/s")

        # Peak VRAM (autotune adds no persistent buffers).
        torch.cuda.reset_peak_memory_stats()
        K.set_autotune(True)
        dec = FusedLLMMega(lm, lm_head, max_cache_len=640)
        dec.generate(inputs_embeds, max_new_tokens=MAX_NEW_TOKENS)
        vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        results["peak_vram_mb"] = vram_mb
        print(f"\n  peak VRAM (autotune ON): {vram_mb:.0f} MB")

        # Persist for the final aggregate.
        import json
        out_path = Path(__file__).resolve().parent.parent / "outputs" / "autotune_bench.json"
        out_path.parent.mkdir(exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"\n  results -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
