"""Batched-inference benchmark for Granite-Speech-4.1-2b.

Measures aggregate throughput (RTFx = sum(audio_seconds) / wall_time) of the
batched pipeline (:class:`megapar.batched.BatchedPipeline`) at
B = {1, 2, 4, 8, 16} on 30 s audio chunks (B independent copies).

For each B it reports:
  * aggregate RTFx (the leaderboard metric) and aggregate tok/s;
  * peak VRAM (must fit in 32 GB);
  * per-step decode latency (flat => GPU underutilised, RTFx scales with B;
    rising => GPU saturating);
  * the encode / prefill / decode time breakdown (the encoder runs per-stream,
    decode runs batched -- that is where the GEMV -> GEMM win lives).

It also prints the batch=1 references (mega non-spec / spec) and the Open ASR
Leaderboard target (RTFx ~= 231) for comparison.

Run:  .venv/bin/python scripts/bench_batched.py
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from megapar.audio import build_inputs, load_sample_audio
from megapar.batched import BatchedPipeline
from megapar.long_audio import synthesize_long_audio
from megapar.loader import load_model_and_processor
from megapar.parakeet.gpu_lock import with_gpu_lock
from megapar.pipeline import MegaPipeline

LEADERBOARD_RTFX = 231.0
"""Open ASR Leaderboard RTFx for this model (achieved via batching)."""

CHUNK_SECONDS = 30.0
MAX_NEW_TOKENS = 200
BATCH_SIZES = [1, 2, 4, 8, 16]
ITERS = 5
WARMUP = 2


def _median_min(xs: list[float]) -> tuple[float, float]:
    return statistics.median(xs), min(xs)


def bench_batched(pipe, feats, ids, mask, B, audio_seconds):
    """Run B copies through the batched pipeline; return timing + token stats."""
    feats_list = [feats] * B
    ids_list = [ids] * B
    mask_list = [mask] * B

    # warmup (captures the CUDA graph + stabilises cuBLAS).
    for _ in range(WARMUP):
        pipe.transcribe_batch(
            feats_list, ids_list, mask_list, max_new_tokens=MAX_NEW_TOKENS
        )
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    walls = []
    encode_ms_l = []
    res_last = None
    for _ in range(ITERS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        res = pipe.run_batch(
            feats_list, ids_list, mask_list, max_new_tokens=MAX_NEW_TOKENS
        )
        torch.cuda.synchronize()
        walls.append(time.perf_counter() - t0)
        # encode = total - (prefill + decode); the LLM prefill/decode are timed
        # inside BatchedLLMMega, the per-stream encoder is the remainder.
        encode_ms_l.append(
            max(walls[-1] * 1000.0 - res.prefill_ms - res.decode_ms, 0.0)
        )
        res_last = res

    wall_med, wall_min = _median_min(walls)
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1e6
    cur_vram_mb = torch.cuda.memory_allocated() / 1e6
    total_tokens = res_last.total_tokens
    n_per = res_last.n_tokens_per_stream[0]
    decode_ms = res_last.decode_ms
    prefill_ms = res_last.prefill_ms
    encode_ms = statistics.median(encode_ms_l)
    n_steps = max(n_per - 1, 1)  # first token from prefill, rest from decode
    per_step_ms = decode_ms / n_steps

    return {
        "B": B,
        "audio_seconds": audio_seconds,
        "wall_ms_med": round(wall_med * 1000.0, 2),
        "wall_ms_min": round(wall_min * 1000.0, 2),
        "rtfx": round(B * audio_seconds / wall_med, 1),
        "rtfx_min": round(B * audio_seconds / wall_min, 1),
        "agg_tok_per_s": round(total_tokens / wall_med, 1),
        "decode_tok_per_s": round(total_tokens / max(decode_ms / 1000.0, 1e-9), 1),
        "total_tokens": total_tokens,
        "tokens_per_stream": n_per,
        "encode_ms": round(encode_ms, 2),
        "prefill_ms": round(prefill_ms, 2),
        "decode_ms": round(decode_ms, 2),
        "per_step_ms": round(per_step_ms, 3),
        "peak_vram_mb": round(peak_vram_mb, 1),
        "current_vram_mb": round(cur_vram_mb, 1),
    }


def bench_single_ref(pipe, feats, ids, mask, audio_seconds, speculative, label):
    """Batch=1 reference via the existing MegaPipeline (spec / non-spec)."""
    for _ in range(WARMUP):
        pipe.transcribe(
            feats, ids, input_features_mask=mask,
            max_new_tokens=MAX_NEW_TOKENS, speculative=speculative,
        )
    torch.cuda.synchronize()
    walls = []
    n_tok = 0
    for _ in range(ITERS):
        t0 = time.perf_counter()
        _t, _ids = pipe.transcribe(
            feats, ids, input_features_mask=mask,
            max_new_tokens=MAX_NEW_TOKENS, speculative=speculative,
        )
        torch.cuda.synchronize()
        walls.append(time.perf_counter() - t0)
        n_tok = int(_ids.shape[1])
    wall_med = statistics.median(walls)
    return {
        "label": label,
        "wall_ms_med": round(wall_med * 1000.0, 2),
        "rtfx": round(audio_seconds / wall_med, 1),
        "tok_per_s": round(n_tok / wall_med, 1),
        "n_tokens": n_tok,
    }


def main() -> int:
    print("=" * 78)
    print("Batched-inference benchmark: Granite-Speech-4.1-2b (RTX 5090, bf16)")
    print(f"chunk = {CHUNK_SECONDS:.0f}s, max_new_tokens = {MAX_NEW_TOKENS}, "
          f"iters = {ITERS}, warmup = {WARMUP}")
    print(f"leaderboard target RTFx = {LEADERBOARD_RTFX:.0f}x")
    print("=" * 78)

    with with_gpu_lock(
        session="granite-mega",
        model="granite-speech-4.1-2b",
        eta_min=15,
        note="batched inference benchmark B=1..16",
    ):
        print("loading model + processor ...", flush=True)
        model, proc = load_model_and_processor("eager")
        weights_mb = torch.cuda.memory_allocated() / 1e6
        print(f"  weights VRAM = {weights_mb:.0f} MB")

        # 30 s audio chunk (tiled sample -> identical mel shape for every B).
        wav, sr = load_sample_audio()
        wav_chunk, _ = synthesize_long_audio(CHUNK_SECONDS, base_wav=wav, sr=sr)
        inputs = build_inputs(proc, wav_chunk)
        feats = inputs["input_features"].bfloat16()
        ids = inputs["input_ids"]
        mask = inputs.get("input_features_mask")
        prompt_len = int(ids.shape[1])
        audio_seconds = CHUNK_SECONDS
        print(f"chunk {audio_seconds:.0f}s: prompt {prompt_len} tok, "
              f"mel {tuple(feats.shape)}\n", flush=True)

        # ---- batch=1 references (mega non-spec / spec) --------------------
        mega = MegaPipeline(model, proc, encoder_mode="cudagraph", use_fused_llm=True)
        ref_nonspec = bench_single_ref(
            mega, feats, ids, mask, audio_seconds, False, "mega batch=1 (non-spec)"
        )
        ref_spec = bench_single_ref(
            mega, feats, ids, mask, audio_seconds, True, "mega batch=1 (spec)"
        )
        del mega
        torch.cuda.empty_cache()

        print(f"{'reference':<26}{'wall ms':>10}{'RTFx':>10}{'tok/s':>10}")
        print("-" * 56)
        for r in (ref_nonspec, ref_spec):
            print(f"{r['label']:<26}{r['wall_ms_med']:>10.1f}{r['rtfx']:>10.1f}"
                  f"{r['tok_per_s']:>10.1f}")
        print("-" * 56)

        # ---- batched B = 1..16 --------------------------------------------
        results = []
        print(f"\n{'B':>3}{'wall ms':>10}{'agg RTFx':>10}{'agg tok/s':>11}"
              f"{'dec tok/s':>11}{'enc ms':>9}{'pf ms':>8}{'dec ms':>9}"
              f"{'us/step':>9}{'VRAM MB':>10}")
        print("-" * 90)
        for B in BATCH_SIZES:
            pipe = BatchedPipeline(
                model, proc, max_batch_size=B, encoder_mode="cudagraph"
            )
            r = bench_batched(pipe, feats, ids, mask, B, audio_seconds)
            results.append(r)
            print(f"{B:>3}{r['wall_ms_med']:>10.1f}{r['rtfx']:>10.1f}"
                  f"{r['agg_tok_per_s']:>11.1f}{r['decode_tok_per_s']:>11.1f}"
                  f"{r['encode_ms']:>9.1f}{r['prefill_ms']:>8.1f}{r['decode_ms']:>9.1f}"
                  f"{r['per_step_ms']*1000:>9.1f}{r['peak_vram_mb']:>10.0f}")
            del pipe
            torch.cuda.empty_cache()
        print("-" * 90)

        # ---- analysis -----------------------------------------------------
        best = max(results, key=lambda r: r["rtfx"])
        b1 = next(r for r in results if r["B"] == 1)
        print(f"\nSweet spot: B={best['B']} -> aggregate RTFx={best['rtfx']:.1f}x "
              f"({best['agg_tok_per_s']:.0f} tok/s), peak VRAM={best['peak_vram_mb']:.0f} MB")
        print(f"  vs batch=1 non-spec RTFx={b1['rtfx']:.1f}x "
              f"({best['rtfx']/b1['rtfx']:.1f}x speedup)")
        print(f"  vs mega batch=1 spec RTFx={ref_spec['rtfx']:.1f}x "
              f"({best['rtfx']/ref_spec['rtfx']:.1f}x speedup)")
        print(f"  vs leaderboard {LEADERBOARD_RTFX:.0f}x "
              f"({'BEATS' if best['rtfx'] >= LEADERBOARD_RTFX else 'below'} leaderboard)")

        # per-step latency trend (GPU saturation).
        print("\nPer-step decode latency vs B (flat = GPU underutilised, "
              "rising = saturating):")
        base_step = b1["per_step_ms"]
        for r in results:
            bar = "#" * int(r["per_step_ms"] / base_step * 10) if base_step else 0
            print(f"  B={r['B']:>2}: {r['per_step_ms']*1000:>7.1f} us/step  "
                  f"({r['per_step_ms']/base_step:>4.2f}x B=1)  {bar}")
        # linear scaling check: RTFx(B) / RTFx(1) should track B until saturation.
        print("\nRTFx scaling vs ideal (linear = B, until GPU saturates):")
        for r in results:
            scale = r["rtfx"] / b1["rtfx"]
            eff = scale / r["B"] if r["B"] > 0 else 0
            print(f"  B={r['B']:>2}: RTFx={r['rtfx']:>7.1f}x  "
                  f"({scale:>4.1f}x over B=1, {eff*100:>5.1f}% of linear)")

        out = {
            "config": {
                "chunk_seconds": CHUNK_SECONDS,
                "max_new_tokens": MAX_NEW_TOKENS,
                "iters": ITERS,
                "warmup": WARMUP,
                "leaderboard_rtfx": LEADERBOARD_RTFX,
                "weights_vram_mb": round(weights_mb, 1),
                "prompt_len": prompt_len,
            },
            "references": {"nonspec": ref_nonspec, "spec": ref_spec},
            "batched": results,
            "sweet_spot_B": best["B"],
            "sweet_spot_rtfx": best["rtfx"],
        }
        out_path = Path(__file__).resolve().parent.parent / "outputs" / "batched_bench.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2))
        print(f"\n[bench] wrote {out_path}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
