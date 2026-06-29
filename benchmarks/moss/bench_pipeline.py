"""RTF + per-stage benchmark for the MOSS-Transcribe megakernel pipeline.

Reports wall-clock transcribe time, RTFx, per-token decode throughput, and
per-stage breakdown (encoder / merge / prefill / decode) for the
short/medium/long fixtures, with graph capture + warmup excluded from the timed
region. Optional ``--compare-eager`` re-runs the stock eager greedy decoder for
a head-to-head.

Run:  ``uv run python -m benchmarks.moss.bench_pipeline [--compare-eager]``
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import median

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))


def _load_wav(name: str):
    import soundfile as sf

    wav, sr = sf.read(str(REPO / "tests" / "fixtures" / f"{name}.wav"))
    if wav.ndim > 1:
        wav = wav.mean(1)
    return wav, sr


def _cuda_timer(fn, warmup: int = 3, iters: int = 10) -> float:
    torch.cuda.synchronize()
    for _ in range(warmup):
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
    return median(times)


def main() -> int:
    from starling.moss.encoder_graph import GraphedAudioEncoder
    from starling.moss.fused_decode import FusedMossLLMMega
    from starling.moss.loader import load_model_and_processor
    from starling.moss.multistep import FusedMossMultiStepMega
    from starling.moss.reference import (
        audio_features,
        build_inputs_embeds,
        greedy_generate,
    )

    ap = argparse.ArgumentParser()
    ap.add_argument("--compare-eager", action="store_true", help="also run stock eager")
    ap.add_argument("--K", type=int, default=16, help="multistep steps per replay")
    args = ap.parse_args()

    print("[bench] loading model ...")
    model, proc = load_model_and_processor()
    inner = model.model

    from starling.moss.pipeline import MossMegaPipeline

    pipe = MossMegaPipeline(model, proc, max_cache_len=2048, steps_per_replay=args.K)
    enc = pipe.fused_encoder

    results = {}
    for name in ("short", "medium", "long"):
        wav, sr = _load_wav(name)
        seconds = wav.shape[0] / sr
        inp = proc(wav.astype("float32"))
        inp = {k: (v.cuda() if isinstance(v, torch.Tensor) else v) for k, v in inp.items()}

        gold_len = 200
        gp = REPO / "golden" / f"moss_{name}_ids.pt"
        if gp.exists():
            gold_len = int(torch.load(gp).shape[1])

        # ---- warmup: capture both graphs once for this mel length ----
        with torch.inference_mode():
            _ = pipe.transcribe(
                inp["audio_data"], inp["audio_data_seqlens"], inp["input_ids"],
                inp["audio_input_mask"], max_new_tokens=8,
            )

        # ---- per-stage steady-state timing (CUDA events) ----
        # NOTE: use a SEPARATE decoder for stage probing so the pipeline's own
        # decoder (used for the timed transcribe below) is not left in a
        # captured/warmed state that perturbs its numbers.
        probe = FusedMossMultiStepMega(
            inner.language_model, model.lm_head, max_cache_len=2048,
            steps_per_replay=args.K,
        )
        with torch.inference_mode():
            enc_ms = _cuda_timer(
                lambda: enc(inp["audio_data"], inp["audio_data_seqlens"]), iters=8
            )
            feats = enc(inp["audio_data"], inp["audio_data_seqlens"])
            merge_ms = _cuda_timer(
                lambda: build_inputs_embeds(
                    model, inp["input_ids"], feats, inp["audio_input_mask"]
                ),
                iters=20,
            )
            emb = build_inputs_embeds(
                model, inp["input_ids"], feats, inp["audio_input_mask"]
            )
            T = emb.shape[1]
            probe._reset_cache_pos(0)
            ft = probe.prefill(emb)
            probe.capture(ft, T)
            probe._reset_to_chunk_start(T, ft)

            def _k():
                probe._ms_graph.replay()
                probe._reset_to_chunk_start(T, ft)

            replay_ms = _cuda_timer(_k, iters=15)
            decode_ms_per_tok = replay_ms / probe.K
        del probe

        # ---- full wall-clock transcribe (capture excluded) ----
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            text, ids = pipe.transcribe(
                inp["audio_data"], inp["audio_data_seqlens"], inp["input_ids"],
                inp["audio_input_mask"], max_new_tokens=gold_len,
            )
        torch.cuda.synchronize()
        total_ms = (time.perf_counter() - t0) * 1000.0
        rtfx = seconds / (total_ms / 1000.0)
        r_n = int(ids.shape[1])

        row = {
            "seconds": round(seconds, 2),
            "tokens": r_n,
            "total_ms": round(total_ms, 1),
            "rtfx": round(rtfx, 1),
            "ms_per_tok": round(total_ms / max(r_n, 1), 2),
            "stage_ms": {
                "encoder": round(enc_ms, 2),
                "merge": round(merge_ms, 2),
                "decode_per_tok": round(decode_ms_per_tok, 3),
            },
            "decode_tok_per_s": round(1000.0 / decode_ms_per_tok, 0) if decode_ms_per_tok else 0,
        }

        if args.compare_eager:
            with torch.inference_mode():
                feats = enc._forward_eager(inp["audio_data"], inp["audio_data_seqlens"])
                emb = build_inputs_embeds(
                    model, inp["input_ids"], feats, inp["audio_input_mask"]
                )
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                _ = greedy_generate(model, emb, max_new_tokens=gold_len, max_cache_len=2048)
            torch.cuda.synchronize()
            eager_ms = (time.perf_counter() - t0) * 1000.0
            row["eager_ms"] = round(eager_ms, 1)
            row["eager_rtfx"] = round(seconds / (eager_ms / 1000.0), 1)
            row["speedup_vs_eager"] = round(eager_ms / total_ms, 1)

        results[name] = row
        print(f"\n[bench] {name}: {seconds:.1f}s, {r_n} tokens")
        print(f"  total {total_ms:.0f}ms  RTFx {rtfx:.0f}x  ({total_ms/r_n:.2f}ms/tok)")
        print(f"  stages: encoder {enc_ms:.0f}ms  merge {merge_ms:.1f}ms  "
              f"decode {decode_ms_per_tok:.3f}ms/tok ({row['decode_tok_per_s']:.0f} tok/s)")
        if args.compare_eager:
            print(f"  eager {eager_ms:.0f}ms ({row['eager_rtfx']:.0f}x)  -> "
                  f"{row['speedup_vs_eager']:.1f}x faster")

    out = REPO / "outputs"
    out.mkdir(exist_ok=True)
    (out / "moss_bench.json").write_text(json.dumps(results, indent=2))
    print(f"\n[bench] saved -> {out}/moss_bench.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
