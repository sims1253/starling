# Autoresearch: co-design the Pixel weight layout and decode kernel (#317)

## Objective

Cut MOSS decode ms/token on the Pixel 10 Pro (PowerVR DXT-48-1536) by
designing the weight layout together with the decode GEMV kernel. Decode
GEMVs on this GPU are instruction-issue bound (W4 ≈ W8 ≈ 33 G weights/s), so
the lever is work per weight: int8 activations with OpSDot
(`GL_EXT_integer_dot_product`, probe: 154 G i8-MAC/s vs 77.5 G f32), leaner
scale storage, pre-permuted nibbles. The Phase 0 quantizer (#319, merged in
this branch's history) quantizes MOSS source weights into any descriptor and
scores weight-only WER on the desktop before phone time is spent.

Branch: `autoresearch/pixel-layout-2026-09-25` (from `feat/pixel-layout-quantizer`).
Worktree: `/home/m0hawk/Documents/starling-pixel` (the main checkout is not
mine to touch; `third_party/ggml` there carries local perf patches — this
worktree mirrors that dirty state, keep it).

## Metrics

- **Primary**: `moss_decode` — MOSS decode ms/token on the phone, short
  fixture, median of ≥3 runs per A/B round, alternating rounds in one thermal
  window (lower is better).
- **Secondary**: `pk_medium` (Parakeet medium total ms — G5 regression canary),
  `transcripts_match` (1 = fixture transcripts identical, G1),
  `en_wer` (desktop weight-only eval, en_us 100), `pack_mb`.

## How to Run

`./.auto/measure.sh` — builds the Android bench, pushes it as the candidate,
A/Bs it against `starling-bench-base` on the phone (same thermal window),
prints `METRIC name=value` lines and the transcript gate. When a candidate is
KEPT, log_experiment commits it; the next run must refresh the base binary
(measure.sh detects a keep via `.auto/base_dirty` and re-pushes; verify by
checking git log before trusting a base).

Phone: wireless adb (`avahi-browse -rpt _adb-tls-connect._tcp` → `adb connect
ip:port`; the port changes when wireless debugging restarts). Models and
fixtures already live in `/data/local/tmp/starling`. Let the phone cool
between long sessions; record `adb shell dumpsys thermalservice | head` when
numbers drift.

## Gates (any failure ⇒ revert)

- **G1** kernel-only change on an unchanged layout: fixture transcripts
  (short/medium/long, MOSS) identical to the base binary.
- **G2** layout/numerics change: FLEURS en_us 100 within +0.2 pt of the
  current fast engine (7.87 % GGUF path; the Phase 0 rebuild measured 7.31 %
  packed). Pre-screen weight-only with `benchmarks/fast_engine/layout_eval.py`
  (see the Phase 0 log section: ±0.5 pt is numerics-draw noise; rank by
  deltas ≥ 1 pt, confirm survivors with the fast engine on the full 100).
  German and Tamil reported alongside (their > 100 % WER regime is noisy —
  qualitative only).
- **G3** `fast_weights_test` passes; the default build (STARLING_FAST=OFF)
  compiles.
- **G4** desktop RADV regress ≤ 10 % (`STARLING_FAST_MICRO` GEMV probe on
  `build-vk/starling-bench`).
- **G5** Parakeet encoder / MOSS prefill GEMMs don't regress beyond noise
  (`gemm.comp`/`dequant_row.glsl` read the same layouts — a new layout must
  be GEMM-readable or apply only to decode-only tensors; `pk_medium` in
  measure.sh watches this).

## Fixed (never edit)

MOSS source weights + hash, `models/moss-full.imx`, the #319 scorer
(`layout_eval.py` semantics), FLEURS lists, WER thresholds, fixture
transcripts, the GGUF distribution files.

## Writable

`cpp/fast/layout.*` (descriptors), `cpp/fast/weights.cpp` (packing — GGUF
path stays byte-stable), `cpp/fast/shaders/gemv.comp` + new variants,
`dequant_row.glsl`/`gemm.comp` variants, `cpp/fast/kernels.cpp` dispatch,
`cpp/fast/packed_file.*`, `cpp/tools/starling_layout_quant.cpp`,
`cpp/fast/fast.cmake` (shader list), `benchmarks/fast_engine/layout_rules/`.

Pattern: new paths default on for PowerVR (vendor 0x1010) with an env
override (`STARLING_FAST_W4U` style); the ggml fallback stays untouched.

## What's Been Tried

Phase 0 (RESEARCH_LOG "#319 Phase 0"): infra landed; free-offset asymmetric
W4 loses ~2 pt WER despite better MSE — symmetric only; bf16-store eval
noise biases W8 candidates (use native ggml blocks); ±0.5 pt en deltas are
numerics-draw noise.

Phase 1 (RESEARCH_LOG "#317 Phase 1", this branch): **int8 activations via
OpSDot are dead on PowerVR** — dotPacked4x8EXT costs the same issue slots
as the f32 dot(vec4) it replaces (3 kernel variants measured, idea stopped);
f16 dots run at 4 MAC/slot (2.6x f32) but the GEMV plateau (~37-46 G w/s,
every layout) is latency/occupancy-structural, not op-count; skinny tiled
GEMM is flat M=1..16 but 6.8x a GEMV pass → **speculative decoding (#311)
needs a dedicated M-token GEMV kernel** (~1.6-1.8x fewer ops/token at
M=4-8) — that is the remaining big lever. Layout conclusion + acceptance
table at the end of the log; notes posted to #311/#316/#318.

Session protocol notes: phone screen OFF (compositor steals the GPU, ±100%
swings), order-balanced A/B, ±2.5% noise floor, absolute numbers drift per
boot — same-window deltas only.

P1-5..P1-9 + aftermath (see RESEARCH_LOG): scale-interleave dead (free at
saturation); **gemv_w4um KEPT** (1.76-2.11x/token at M=2 — #311's verify
primitive); rows=16 KEPT (-1.9% verified twice); drafter study closed the
speculative path (n-gram 1.000, copy-draft 1.03-1.05 tokens/pass);
micro-vs-context gap attributed (pipeline switches + DRAM-cold);
F16-lm_head REJECTED (+9.4%, in-context bandwidth ~43 GB/s); energy
delivered (2.42 mWh/transcription); #325 driver investigation: VK_TIMEOUT
hang (not OOM), wedge guards landed (marker + preflight + enriched errors);
final certification: tree audited, all gates green on the PR head
(`48a04e0`+), G4 final −1.3%/+0.9%. LOOP COMPLETE — resume only with a new
hypothesis from #311 (learned drafter) or new hardware/driver.

## Ideas backlog

See `.auto/ideas.md` (ranked; from the #317 brief). Side measurement #0
(skinny-GEMM M sweep for #311/speculative decoding) runs first — it needs no
code change, just GEMM timing probes at M=1..16 on the phone.
