#!/usr/bin/env python3
"""Config sweep: find the strongest batch size for each model on long-form audio.

Phase 1: sweep B on 60min audio (2 repeats each) to find the RTFx peak.
Phase 2: run 5 repeats at the winning B across 30/60/90min.

Also tests a "max throughput" granite config: tolerance_mode batched encoder
(NOT byte-exact, but shows the speed ceiling).

Run:  cd /home/m0hawk/Documents/starling && uv run benchmarks/bench_sweep.py
"""

from __future__ import annotations

import json
import statistics
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tests" / "fixtures"))

SR = 16000
OUTPUTS = REPO / "outputs"

GRANITE_BATCH_SIZES = [16, 32, 48, 64]
PARAKEET_BATCH_SIZES = [32, 48]
SWEEP_DURATION = 3600  # 60min for the sweep
SWEEP_REPEATS = 2
FINAL_DURATIONS = [1800, 3600, 5400]  # 30/60/90min
FINAL_REPEATS = 5


def _suppress():
    for mod in ("transformers",):
        try:
            warnings.filterwarnings("ignore", module=mod)
        except Exception:
            pass
    warnings.filterwarnings("ignore", message=".*max_length.*")
    warnings.filterwarnings("ignore", message=".*RNN module weights.*")


def tile_granite_audio(seconds: int) -> torch.Tensor:
    from starling.granite.audio import load_sample_audio
    wav, sr = load_sample_audio()
    reps = max(1, (seconds * sr + wav.shape[1] - 1) // wav.shape[1])
    return wav.repeat(1, reps)[:, :seconds * sr].contiguous()


def tile_parakeet_audio(seconds: int) -> np.ndarray:
    import make_fixtures as mkfx
    base = mkfx.load_sample()
    reps = max(1, (seconds * SR + len(base) - 1) // len(base))
    return np.concatenate([base] * reps)[:seconds * SR].astype(np.float32)


def stats(values: list[float]) -> dict:
    return {
        "mean": round(statistics.mean(values), 3),
        "std": round(statistics.stdev(values), 3) if len(values) > 1 else 0.0,
        "min": round(min(values), 3),
        "max": round(max(values), 3),
        "values": [round(v, 3) for v in values],
    }


# ---------------------------------------------------------------------------
# Granite sweep
# ---------------------------------------------------------------------------
def sweep_granite(batch_sizes: list[int], duration: int, repeats: int) -> dict:
    from starling.granite.batched import BatchedPipeline
    from starling.granite.long_audio import transcribe_long_batched
    from starling.granite.loader import load_model_and_processor

    print("\n" + "=" * 72)
    print(f"GRANITE SWEEP ({duration//60}min, {repeats} repeats per B)")
    print("=" * 72)

    model, proc = load_model_and_processor(attn_impl="eager")
    wav = tile_granite_audio(duration)

    results = {}
    best_rtfx = 0
    best_b = batch_sizes[0]

    for B in batch_sizes:
        pipe = BatchedPipeline(model, proc, max_batch_size=B, encoder_mode="cudagraph")

        # warmup
        from starling.granite.audio import build_inputs
        warm_wav = wav[:, :int(30 * SR)]
        wi = build_inputs(proc, warm_wav)
        pipe.run_batch(
            [wi["input_features"].bfloat16()] * min(B, 4),
            [wi["input_ids"]] * min(B, 4),
            [wi.get("input_features_mask")] * min(B, 4),
            max_new_tokens=30,
        )
        torch.cuda.synchronize()

        rtfxs, walls, vrams = [], [], []
        for rep in range(repeats):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            res = transcribe_long_batched(
                pipe, proc, wav, SR,
                chunk_seconds=30.0, overlap_seconds=2.0,
                max_new_tokens=200,
            )
            torch.cuda.synchronize()
            wall = time.perf_counter() - t0
            vram = torch.cuda.max_memory_allocated() / 1e9
            walls.append(wall)
            rtfxs.append(duration / wall)
            vrams.append(vram)

        mean_rtfx = statistics.mean(rtfxs)
        mean_vram = statistics.mean(vrams)
        print(f"  B={B:<4d}  wall={statistics.mean(walls):.2f}s  "
              f"RTFx={mean_rtfx:.1f}x  VRAM={mean_vram:.2f}GB  "
              f"chunks={res.n_chunks}", flush=True)

        results[str(B)] = {
            "batch_size": B,
            "wall": stats(walls),
            "rtfx": stats(rtfxs),
            "vram_gb": stats(vrams),
        }
        if mean_rtfx > best_rtfx:
            best_rtfx = mean_rtfx
            best_b = B

        del pipe
        torch.cuda.empty_cache()

    results["_best_b"] = best_b
    results["_best_rtfx"] = round(best_rtfx, 1)
    print(f"\n  >>> GRANITE BEST: B={best_b} ({best_rtfx:.1f}x RTFx)")
    return results


# ---------------------------------------------------------------------------
# Parakeet sweep
# ---------------------------------------------------------------------------
def sweep_parakeet(batch_sizes: list[int], duration: int, repeats: int) -> dict:
    from starling.parakeet.pipeline import MegaParakeetPipeline
    from starling.parakeet.chunking import ChunkedTranscriber

    print("\n" + "=" * 72)
    print(f"PARAKEET SWEEP ({duration//60}min, {repeats} repeats per B)")
    print("=" * 72)

    audio = tile_parakeet_audio(duration)

    results = {}
    best_rtfx = 0
    best_b = batch_sizes[0]

    for B in batch_sizes:
        pipe = MegaParakeetPipeline(dtype=torch.bfloat16)
        chunker = ChunkedTranscriber(
            pipe, chunk_seconds=30.0, overlap_seconds=2.0, chunk_batch_size=B
        )

        # warmup (first call captures graphs)
        warm_audio = tile_parakeet_audio(30)
        chunker.transcribe(warm_audio, sr=SR)
        torch.cuda.synchronize()

        rtfxs, walls, vrams = [], [], []
        for rep in range(repeats):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            text, summary = chunker.transcribe_with_timing(audio, sr=SR)
            torch.cuda.synchronize()
            wall = time.perf_counter() - t0
            vram = torch.cuda.max_memory_allocated() / 1e9
            walls.append(wall)
            rtfxs.append(duration / wall)
            vrams.append(vram)

        mean_rtfx = statistics.mean(rtfxs)
        mean_vram = statistics.mean(vrams)
        print(f"  B={B:<4d}  wall={statistics.mean(walls):.2f}s  "
              f"RTFx={mean_rtfx:.1f}x  VRAM={mean_vram:.2f}GB  "
              f"chunks={summary['n_chunks']}", flush=True)

        results[str(B)] = {
            "batch_size": B,
            "wall": stats(walls),
            "rtfx": stats(rtfxs),
            "vram_gb": stats(vrams),
        }
        if mean_rtfx > best_rtfx:
            best_rtfx = mean_rtfx
            best_b = B

        del chunker, pipe
        torch.cuda.empty_cache()

    results["_best_b"] = best_b
    results["_best_rtfx"] = round(best_rtfx, 1)
    print(f"\n  >>> PARAKEET BEST: B={best_b} ({best_rtfx:.1f}x RTFx)")
    return results


# ---------------------------------------------------------------------------
# Final run at winning config
# ---------------------------------------------------------------------------
def final_granite(B: int, durations: list[int], repeats: int) -> dict:
    from starling.granite.batched import BatchedPipeline
    from starling.granite.long_audio import transcribe_long_batched
    from starling.granite.loader import load_model_and_processor

    print(f"\n{'=' * 72}")
    print(f"GRANITE FINAL (B={B}, {repeats} repeats)")
    print(f"{'=' * 72}")

    model, proc = load_model_and_processor(attn_impl="eager")
    pipe = BatchedPipeline(model, proc, max_batch_size=B, encoder_mode="cudagraph")

    # warmup
    wav_warm = tile_granite_audio(30)
    from starling.granite.audio import build_inputs
    wi = build_inputs(proc, wav_warm)
    pipe.run_batch(
        [wi["input_features"].bfloat16()] * min(B, 4),
        [wi["input_ids"]] * min(B, 4),
        [wi.get("input_features_mask")] * min(B, 4),
        max_new_tokens=30,
    )
    torch.cuda.synchronize()

    results = {}
    for dur in durations:
        wav = tile_granite_audio(dur)
        walls, rtfxs, vrams, toks = [], [], [], []
        for rep in range(repeats):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            res = transcribe_long_batched(
                pipe, proc, wav, SR,
                chunk_seconds=30.0, overlap_seconds=2.0, max_new_tokens=200,
            )
            torch.cuda.synchronize()
            wall = time.perf_counter() - t0
            walls.append(wall)
            rtfxs.append(dur / wall)
            vrams.append(torch.cuda.max_memory_allocated() / 1e9)
            toks.append(res.total_tokens)
            print(f"  [granite] {dur//60}min rep {rep+1}/{repeats}: "
                  f"wall={wall:.2f}s RTFx={dur/wall:.1f}x", flush=True)

        results[f"{dur}s"] = {
            "duration_s": dur, "batch_size": B,
            "wall": stats(walls), "rtfx": stats(rtfxs),
            "vram_gb": stats(vrams), "tokens": stats(toks),
        }
        print(f"  [granite] {dur//60}min MEAN: "
              f"wall={statistics.mean(walls):.2f}s "
              f"RTFx={statistics.mean(rtfxs):.1f}x "
              f"(+/-{statistics.stdev(rtfxs):.1f})", flush=True)

    del pipe, model
    torch.cuda.empty_cache()
    return results


def final_parakeet(B: int, durations: list[int], repeats: int) -> dict:
    from starling.parakeet.pipeline import MegaParakeetPipeline
    from starling.parakeet.chunking import ChunkedTranscriber

    print(f"\n{'=' * 72}")
    print(f"PARAKEET FINAL (B={B}, {repeats} repeats)")
    print(f"{'=' * 72}")

    pipe = MegaParakeetPipeline(dtype=torch.bfloat16)
    chunker = ChunkedTranscriber(
        pipe, chunk_seconds=30.0, overlap_seconds=2.0, chunk_batch_size=B
    )

    # warmup
    warm_audio = tile_parakeet_audio(30)
    chunker.transcribe(warm_audio, sr=SR)
    torch.cuda.synchronize()

    results = {}
    for dur in durations:
        audio = tile_parakeet_audio(dur)
        walls, rtfxs, vrams, toks = [], [], [], []
        for rep in range(repeats):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            text, summary = chunker.transcribe_with_timing(audio, sr=SR)
            torch.cuda.synchronize()
            wall = time.perf_counter() - t0
            walls.append(wall)
            rtfxs.append(dur / wall)
            vrams.append(torch.cuda.max_memory_allocated() / 1e9)
            toks.append(summary.get("n_tokens_surviving", 0))
            print(f"  [parakeet] {dur//60}min rep {rep+1}/{repeats}: "
                  f"wall={wall:.2f}s RTFx={dur/wall:.1f}x", flush=True)

        results[f"{dur}s"] = {
            "duration_s": dur, "batch_size": B,
            "wall": stats(walls), "rtfx": stats(rtfxs),
            "vram_gb": stats(vrams), "tokens": stats(toks),
        }
        print(f"  [parakeet] {dur//60}min MEAN: "
              f"wall={statistics.mean(walls):.2f}s "
              f"RTFx={statistics.mean(rtfxs):.1f}x "
              f"(+/-{statistics.stdev(rtfxs):.1f})", flush=True)

    del chunker, pipe
    torch.cuda.empty_cache()
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    _suppress()

    from starling.parakeet.gpu_lock import with_gpu_lock

    results = {
        "device": torch.cuda.get_device_name(0),
        "sweep_duration_s": SWEEP_DURATION,
        "sweep_repeats": SWEEP_REPEATS,
        "final_repeats": FINAL_REPEATS,
    }

    with with_gpu_lock(
        session="config-sweep",
        model="both",
        eta_min=30,
        note="config sweep + final benchmark at winning B",
    ):
        # ---- Phase 1: Sweep ----
        print("\n" + "#" * 72)
        print("# PHASE 1: CONFIG SWEEP")
        print("#" * 72)
        gran_sweep = sweep_granite(GRANITE_BATCH_SIZES, SWEEP_DURATION, SWEEP_REPEATS)
        para_sweep = sweep_parakeet(PARAKEET_BATCH_SIZES, SWEEP_DURATION, SWEEP_REPEATS)

        results["sweep"] = {"granite": gran_sweep, "parakeet": para_sweep}

        gran_best = gran_sweep["_best_b"]
        para_best = para_sweep["_best_b"]
        print(f"\n{'#' * 72}")
        print(f"# SWEEP WINNERS: granite B={gran_best}, parakeet B={para_best}")
        print(f"{'#' * 72}")

        # ---- Phase 2: Final at winning config ----
        print("\n" + "#" * 72)
        print("# PHASE 2: FINAL BENCHMARK AT WINNING CONFIG")
        print("#" * 72)
        results["final"] = {
            "granite": final_granite(gran_best, FINAL_DURATIONS, FINAL_REPEATS),
            "parakeet": final_parakeet(para_best, FINAL_DURATIONS, FINAL_REPEATS),
        }

    # ---- save ----
    OUTPUTS.mkdir(exist_ok=True)
    out_path = OUTPUTS / "bench_sweep.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[bench] wrote {out_path}")

    # ---- summary ----
    print(f"\n{'=' * 80}")
    print(f"STRONGEST CONFIG RESULTS ({FINAL_REPEATS} repeats, mean +/- stddev)")
    print(f"{'=' * 80}")
    for model_name in ("granite", "parakeet"):
        model_final = results["final"][model_name]
        sweep = results["sweep"][model_name]
        best_b = sweep["_best_b"]
        print(f"\n--- {model_name.upper()} (B={best_b}) ---")
        print(f"{'duration':>10} {'wall (s)':>16} {'RTFx':>16} {'VRAM (GB)':>12}")
        print("-" * 58)
        for dur_key, data in model_final.items():
            dur = data["duration_s"]
            dur_label = f"{dur//60}min"
            wall_str = f"{data['wall']['mean']:.3f} +/- {data['wall']['std']:.3f}"
            rtfx_str = f"{data['rtfx']['mean']:.1f} +/- {data['rtfx']['std']:.1f}"
            vram_str = f"{data['vram_gb']['mean']:.2f}"
            print(f"{dur_label:>10} {wall_str:>16} {rtfx_str:>16} {vram_str:>12}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
