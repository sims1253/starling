# Autoresearch: fast Vulkan engines on the Pixel 10 Pro

## Objective
Make the model-specialized Vulkan fast engines (`cpp/fast/`, PR #287) as fast
and power-efficient as possible **on the Pixel 10 Pro** (Tensor G5, PowerVR
DXT-48-1536). Full brief: `benchmarks/fast_engine/AUTORESEARCH.md`.
Log every experiment in `benchmarks/fast_engine/RESEARCH_LOG.md`.

## Metrics
- **Primary**: `phone_ms` (ms, lower) — median Parakeet medium.wav total +
  median MOSS short.wav total, on the phone, after warm-up.
- **Secondary**: `parakeet_encoder_ms`, `parakeet_decode_ms`, `moss_enc_prefill_ms`,
  `moss_decode_ms`, `moss_per_tok_ms`, mel times, `transcript_match`.

## How to Run
`./.auto/measure.sh` — builds (cmake/ninja, build-android), pushes, runs,
emits METRIC lines. Env: `NO_BUILD=1`, `RUNS=n`, `REFRESH_TUNE=1`
(delete on-device autotune cache first — REQUIRED after kernel changes),
`CAPTURE=1` (reset transcript baseline).

## Files in Scope
`cpp/fast/**` (kernels, shaders, engines, runtime, weights) and
`benchmarks/fast_engine/*` (harness, log). Docs `docs/fast-engine.md`.

## Off Limits
`third_party/ggml` (submodule has unrelated dirty content — never commit it),
`apps/`, `packages/`, anything outside the fast engine.

## Constraints — quality gates (revert on failure)
1. Fixture transcripts (short/medium/long) unchanged for both models —
   measure.sh enforces short+medium; check `long.wav` manually per milestone.
2. FLEURS-en 100-clip WER within 0.2 pts of ggml (Parakeet 5.33 / MOSS 7.92)
   — at milestones & for any numerics change.
3. `fast_weights_test` passes; `STARLING_FAST=OFF` build builds (milestones).
4. Desktop RADV regression ≤ 10% (milestones / risky changes).
Thermals: median of ≥3 runs, brief says alternate order if numbers drift.

## Key facts (measured, see AUTORESEARCH.md)
- PowerVR: subgroup 128, 32 KiB shared, 128 MiB storage buffer range,
  shaderFloat16 yes. Tile-based GPU — per-dispatch timestamps are misattributed;
  A/B wall time or dedicated single-kernel recordings instead.
- Baseline (branch head): Parakeet medium 2.9 s (enc 2.6 s = 88%!), MOSS short
  7.1 s (enc+prefill 3.7 s, decode 3.16 s ≈ 100 ms/token).
- Encoder ~130 GFLOPS only; MOSS decode ~11 GB/s (weights ~1.1 GB/token).
- 464 dispatches for the Parakeet encoder; barriers may flush tiles.
- Tune cache `starling-fast-tune-<vendor>-<dev>-<driver>-h.txt` is keyed on
  device+driver+f16 flag ONLY → delete it (REFRESH_TUNE=1) after kernel changes.

## What's Been Tried
(see RESEARCH_LOG.md for details — keep this synced)

## Loop Rules
One hypothesis per iteration → smallest change → build+measure → keep (commit
with numbers) or revert. Append RESEARCH_LOG.md rows. Never stop.
