"""Feature-flag + multi-step benchmark for Granite-Speech-4.1-2b.

Measures the impact of the multi-step CUDA-graph capture (Deliverable 1) and
the tolerance-mode batched encoder (Deliverable 2) on:

1. **Single-stream tok/s**: per-token-sync (FusedLLMMega) vs multi-step
   (K=8, K=16).  Shows the host<->device sync-overhead reduction.
2. **Per-token decode ms**: pure CUDA-graph replay time (CUDA events) vs
   full-loop per-token time (wall clock).  The gap is the sync/launch overhead
   that multi-step eliminates.
3. **Batched RTFx at B=16**: current batched baseline (per-step sync).
4. **Tolerance-mode batched encoder**: the RTFx gain AND the actual numerical
   diff (max/mean abs vs per-stream encode) AND whether the decoded transcript
   still matches (greedy-chaos may or may not flip).

All timed runs acquire the GPU lock (comms.md P1).

Run:  .venv/bin/python scripts/bench_flags.py
"""
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from megapar.audio import build_inputs, load_sample_audio
from megapar.batched import BatchedPipeline
from megapar.flags import OptFlags
from megapar.loader import get_components, load_model_and_processor
from megapar.llm_mega import FusedLLMMega
from megapar.multistep import MultiStepLLMMega
from megapar.parakeet.gpu_lock import with_gpu_lock

MAX_NEW_TOKENS = 100
ITERS = 8
WARMUP = 3


def _median_min(xs: list[float]) -> tuple[float, float]:
    return statistics.median(xs), min(xs)


def _wall_ms(fn, warm: int = WARMUP, iters: int = ITERS) -> tuple[float, float]:
    """Median + min wall-clock ms for ``fn`` (with GPU sync)."""
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000.0)
    return _median_min(ts)


def _cuda_event_ms(fn, warm: int = WARMUP, iters: int = ITERS) -> float:
    """Median GPU-only ms for ``fn`` via CUDA events (excludes host overhead)."""
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


# =========================================================================== #
# 1. Single-stream: per-token-sync vs multi-step
# =========================================================================== #
def bench_single_stream(model, proc, comps, inputs_embeds, audio_seconds):
    """Compare single-step (FusedLLMMega) vs multi-step (K=8, K=16)."""
    print("\n" + "=" * 78)
    print("1. SINGLE-STREAM DECODE: per-token-sync vs multi-step")
    print("=" * 78)
    lm = comps["language_model"]
    lm_head = model.lm_head

    configs = [
        ("single-step (FusedLLMMega)", FusedLLMMega(lm, lm_head, max_cache_len=640)),
        ("multi-step K=8", MultiStepLLMMega(lm, lm_head, max_cache_len=640, steps_per_replay=8)),
        ("multi-step K=16", MultiStepLLMMega(lm, lm_head, max_cache_len=640, steps_per_replay=16)),
    ]

    print(f"\n{'config':<30}{'GPU ms/tok':>12}{'wall ms':>12}{'wall tok/s':>12}"
          f"{'RTFx':>10}{'speedup':>10}")
    print("-" * 86)

    base_tps = None
    for label, dec in configs:
        # GPU-only per-token decode (CUDA events, pure replay).
        rep = dec.bench(inputs_embeds, max_new_tokens=MAX_NEW_TOKENS, decode_iters=10)
        gpu_ms_pt = rep.decode_ms_per_token

        # Full-loop wall time.
        def _gen():
            dec.generate(inputs_embeds, max_new_tokens=MAX_NEW_TOKENS)
        wall_med, wall_min = _wall_ms(_gen, warm=WARMUP, iters=ITERS)
        n = MAX_NEW_TOKENS
        tps = n / (wall_med / 1000.0)
        if base_tps is None:
            base_tps = tps
        speedup = tps / base_tps

        print(f"{label:<30}{gpu_ms_pt:>12.3f}{wall_med:>12.1f}{tps:>12.1f}"
              f"{audio_seconds / (wall_med / 1000.0):>10.2f}x{speedup:>10.2f}x")

    print(f"\n  GPU ms/tok = pure CUDA-graph replay / token (excludes host sync).")
    print(f"  wall ms    = full generate() loop (includes all host<->device syncs).")


# =========================================================================== #
# 2. Sync overhead breakdown
# =========================================================================== #
def bench_sync_overhead(model, proc, comps, inputs_embeds):
    """Show the per-token sync overhead as (wall_per_tok - gpu_per_tok)."""
    print("\n" + "=" * 78)
    print("2. SYNC-OVERHEAD BREAKDOWN (wall_per_tok - gpu_per_tok)")
    print("=" * 78)
    lm = comps["language_model"]
    lm_head = model.lm_head

    configs = [
        ("single-step", FusedLLMMega(lm, lm_head, max_cache_len=640), 1),
        ("multi-step K=8", MultiStepLLMMega(lm, lm_head, max_cache_len=640, steps_per_replay=8), 8),
        ("multi-step K=16", MultiStepLLMMega(lm, lm_head, max_cache_len=640, steps_per_replay=16), 16),
    ]

    print(f"\n{'config':<22}{'GPU ms/tok':>12}{'wall ms/tok':>14}"
          f"{'overhead us':>14}{'syncs/100tok':>16}")
    print("-" * 78)
    for label, dec, K in configs:
        rep = dec.bench(inputs_embeds, max_new_tokens=MAX_NEW_TOKENS, decode_iters=10)
        gpu_pt = rep.decode_ms_per_token

        def _gen():
            dec.generate(inputs_embeds, max_new_tokens=MAX_NEW_TOKENS)
        wall_med, _ = _wall_ms(_gen, warm=WARMUP, iters=ITERS)
        wall_pt = wall_med / MAX_NEW_TOKENS
        overhead_us = (wall_pt - gpu_pt) * 1000.0
        n_syncs = -(-MAX_NEW_TOKENS // K)  # ceil(100/K) syncs for decode
        print(f"{label:<22}{gpu_pt:>12.3f}{wall_pt:>14.3f}"
              f"{overhead_us:>14.1f}{n_syncs:>16}")


# =========================================================================== #
# 3. Batched B=16
# =========================================================================== #
def bench_batched(model, proc, feats, ids, mask, audio_seconds):
    """Batched RTFx at B=16 (current baseline)."""
    print("\n" + "=" * 78)
    print("3. BATCHED B=16 (current baseline, per-step sync decode)")
    print("=" * 78)
    B = 16
    pipe = BatchedPipeline(model, proc, max_batch_size=B, encoder_mode="cudagraph")
    feats_list = [feats] * B
    ids_list = [ids] * B
    mask_list = [mask] * B

    def _batch():
        pipe.transcribe_batch(feats_list, ids_list, mask_list, max_new_tokens=MAX_NEW_TOKENS)
    wall_med, wall_min = _wall_ms(_batch, warm=2, iters=4)
    rtfx = B * audio_seconds / (wall_med / 1000.0)
    print(f"\n  B={B}: wall={wall_med:.1f}ms, RTFx={rtfx:.1f}x, "
          f"tok/s={B * MAX_NEW_TOKENS / (wall_med / 1000.0):.1f}")
    return pipe


# =========================================================================== #
# 4. Tolerance-mode batched encoder
# =========================================================================== #
def bench_tolerance_encoder(model, proc, comps, feats, ids, mask, audio_seconds):
    """Measure the batched-encoder RTFx gain + numerical diff + transcript match."""
    print("\n" + "=" * 78)
    print("4. TOLERANCE-MODE BATCHED ENCODER (B=16)")
    print("=" * 78)
    B = 16

    # (a) Byte-exact per-stream encode (baseline).
    pipe_exact = BatchedPipeline(
        model, proc, max_batch_size=B, encoder_mode="cudagraph",
        flags=OptFlags(batched_encoder=False, tolerance_mode=False),
    )
    feats_list = [feats] * B
    ids_list = [ids] * B
    mask_list = [mask] * B

    def _batch_exact():
        pipe_exact.run_batch(feats_list, ids_list, mask_list, max_new_tokens=MAX_NEW_TOKENS)
    wall_exact, _ = _wall_ms(_batch_exact, warm=2, iters=4)

    # (b) Tolerance-mode batched encode.
    with torch.inference_mode():
        pipe_tol = BatchedPipeline(
            model, proc, max_batch_size=B, encoder_mode="cudagraph",
            flags=OptFlags(batched_encoder=True, tolerance_mode=True),
        )

        def _batch_tol():
            pipe_tol.run_batch(feats_list, ids_list, mask_list, max_new_tokens=MAX_NEW_TOKENS)
        wall_tol, _ = _wall_ms(_batch_tol, warm=2, iters=4)

    rtfx_exact = B * audio_seconds / (wall_exact / 1000.0)
    rtfx_tol = B * audio_seconds / (wall_tol / 1000.0)
    speedup = wall_exact / wall_tol

    print(f"\n  per-stream encode (byte-exact): wall={wall_exact:.1f}ms, RTFx={rtfx_exact:.1f}x")
    print(f"  batched encode (tolerance):     wall={wall_tol:.1f}ms, RTFx={rtfx_tol:.1f}x")
    print(f"  speedup: {speedup:.2f}x")

    # (c) Numerical diff: batched encoder hidden vs per-stream.
    print("\n  --- numerical diff (batched vs per-stream encode) ---")
    with torch.inference_mode():
        raw_enc = comps["encoder"]
        dtype = pipe_tol.dtype
        # Per-stream encode (byte-exact reference) via the FusedEncoder.
        enc_per = pipe_exact.fused_encoder(feats.to(dtype))  # (1, T, 1024) tensor
        # Batched encode (raw encoder returns a ModelOutput).
        feats_batched = torch.cat([feats.to(dtype)] * B, dim=0)
        enc_batch_out = raw_enc(feats_batched, return_dict=True)
        enc_batch = enc_batch_out.last_hidden_state  # (B, T, 1024)
        enc_batch_s0 = enc_batch[0:1]
        diff = (enc_per.float() - enc_batch_s0.float()).abs()
        max_abs = diff.max().item()
        mean_abs = diff.mean().item()
        print(f"  encoder hidden max-abs diff  = {max_abs:.4f}")
        print(f"  encoder hidden mean-abs diff = {mean_abs:.6f}")

    # (d) Transcript match (does greedy-chaos flip?).
    with torch.inference_mode():
        texts_exact = pipe_exact.transcribe_batch(
            feats_list, ids_list, mask_list, max_new_tokens=MAX_NEW_TOKENS
        )
        texts_tol = pipe_tol.transcribe_batch(
            feats_list, ids_list, mask_list, max_new_tokens=MAX_NEW_TOKENS
        )
    n_match = sum(1 for a, b in zip(texts_exact, texts_tol) if a.strip() == b.strip())
    print(f"\n  transcript match: {n_match}/{B} streams identical "
          f"(byte-exact vs tolerance)")
    if n_match < B:
        for i in range(B):
            if texts_exact[i].strip() != texts_tol[i].strip():
                print(f"    stream {i} DIFFERS:")
                print(f"      exact: {texts_exact[i].strip()[:100]!r}")
                print(f"      tol:   {texts_tol[i].strip()[:100]!r}")
                break

    verdict = "WORTH IT" if (speedup > 1.05 and n_match == B) else \
              "MARGINAL" if speedup > 1.02 else "NOT WORTH IT"
    print(f"\n  VERDICT: {verdict} (speedup={speedup:.2f}x, "
          f"transcripts {'match' if n_match == B else 'differ'})")


# =========================================================================== #
# main
# =========================================================================== #
def main() -> int:
    with with_gpu_lock(
        session="granite-mega",
        model="granite-speech-4.1-2b",
        eta_min=10,
        note="bench_flags: multi-step + tolerance-mode benchmark",
    ):
        print("loading model + processor ...", flush=True)
        model, proc = load_model_and_processor(attn_impl="eager")
        comps = get_components(model)

        wav, sr = load_sample_audio()
        inputs = build_inputs(proc, wav)
        audio_seconds = wav.shape[1] / sr
        feats = inputs["input_features"].to(torch.bfloat16)
        ids = inputs["input_ids"]
        mask = inputs.get("input_features_mask")

        # Precompute golden inputs_embeds for the single-stream LLM benchmarks
        # (avoids re-running the encoder each time).
        from megapar.golden import load_golden
        inputs_embeds = load_golden("inputs_embeds.pt").to("cuda", torch.bfloat16)

        print(f"audio {audio_seconds:.1f}s, prompt {ids.shape[1]} tokens, "
              f"{MAX_NEW_TOKENS} new tokens\n", flush=True)

        bench_single_stream(model, proc, comps, inputs_embeds, audio_seconds)
        bench_sync_overhead(model, proc, comps, inputs_embeds)
        bench_batched(model, proc, feats, ids, mask, audio_seconds)
        bench_tolerance_encoder(model, proc, comps, feats, ids, mask, audio_seconds)

        print("\n" + "=" * 78)
        print("DONE")
        print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
