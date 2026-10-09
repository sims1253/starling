# Autoresearch brief: make the fast engines fast on the Pixel 10 Pro

You are an autonomous performance engineer. Your job is to make Starling's
model-specialized Vulkan engines (`cpp/fast/`, PR #287, branch
`feat/fast-vulkan-engines`) as fast and power-efficient as possible **on a
Google Pixel 10 Pro connected over adb**. Run a disciplined experiment loop
and keep only changes that are measured wins and pass the quality gates.

## Target hardware (measured)

- SoC: Tensor G5. GPU: **Imagination PowerVR D-Series DXT-48-1536 MC1**
  (Vulkan 1.4, vendor 0x1010). Subgroup size **128**, max compute shared
  memory **32 KiB**, `maxStorageBufferRange` **128 MiB**, `shaderFloat16`
  yes, unified memory that is CPU-cached (mapped uploads are fast).
- CPU: 1× Cortex-X4 + 5× A725 + 2× A520, ARMv9.2 with `asimddp`, `i8mm`,
  `sve2`, `bf16`. The app runs ggml on 6 threads (`STARLING_GGML_THREADS=6`).
- **PowerVR is a tile-based GPU: per-dispatch timestamps
  (`STARLING_FAST_PROFILE=1`) are misattributed** — the time of a barrier
  segment lands on its first dispatch (e.g. `norm` shows 30 %, `ff_down` 0 %).
  To measure one kernel, time a recording that repeats only that kernel, or
  compare end-to-end wall times with the kernel changed.

## Baseline on the Pixel (start of this brief)

Build: `feat/fast-vulkan-engines`, f16 GEMM products on, autotuned tile
64,64,4,4, GEMV rows 8. Wall time per transcription after warm-up:

| | ggml CPU (6 thr) | fast |
| --- | --- | --- |
| Parakeet 22.3 s (medium) | 7.2 s (enc 1.30 s, dec 5.9 s) | 2.79 s (mel 23 ms, **enc 2.56 s**, dec 0.20 s) |
| Parakeet 74.4 s (long) | — | 9.0 s (enc 8.6 s) |
| MOSS 7.4 s (short) | 7.3 s (enc 1.3 s, gen 5.9 s) | 7.0 s (**enc+prefill 3.6 s, decode 3.25 s = 100 ms/token**) |
| Load (Parakeet / MOSS) | — | ~2 s / ~5 s after the mapped-upload fix |

The GPU encoder reaches only ~130 GFLOPS (the ggml CPU encoder beats it),
and MOSS decode moves ~11 GB/s of weights on LPDDR5X that can do several
times that. Both are far from the hardware: that is the opportunity.

## Quality gates (a change that fails any gate is reverted)

1. Fixtures `short`/`medium`/`long` (`tests/fixtures/*.wav`) transcribe to the
   same text as before the change, for Parakeet and MOSS, on the phone.
2. Before merging a batch of wins: FLEURS-en 100-clip WER on the desktop
   (`export_fleurs.py` + `wer_engines.py`) stays within 0.2 points of the
   ggml engine (currently Parakeet 5.33 % vs 5.47 %, MOSS 7.92 % vs 7.92 %).
   Numerics changes (f16 accumulation, int8 activations, fewer bits) must
   pass this gate explicitly.
3. `fast_weights_test` passes; the default build (`STARLING_FAST=OFF`) builds.
4. No regression on the desktop RADV path of more than 10 % (it is the CI and
   development device; phone wins may trade a little there, not more).

## Harness

- Build + push + compare: `benchmarks/fast_engine/android_bench.sh`
  (`--no-build` to reuse; `RUNS`, `EXTRA_ENV="STARLING_FAST_TILE=..."`).
- Fast iteration: `cmake --build build-android --target starling-bench`, then
  `adb push build-android/starling-bench /data/local/tmp/starling/` and run
  `adb shell "cd /data/local/tmp/starling && LD_LIBRARY_PATH=. STARLING_ENGINE=fast
  STARLING_FAST_TIMING=1 STARLING_FAST_CACHE_DIR=. ./starling-bench --model parakeet
  --gguf parakeet-tdt-0.6b-v3-q4_k_m-shrink16.gguf --warmup --runs 3 medium.wav"`.
  Models and fixtures are already in `/data/local/tmp/starling`.
- Knobs without rebuilding: `STARLING_FAST_TILE=BM,BN,TM,TN`,
  `STARLING_FAST_GEMV_ROWS`, `STARLING_FAST_F16`, `STARLING_FAST_KSTEP`,
  `STARLING_FAST_TUNE=1` (re-run the autotuner; delete the cached
  `starling-fast-tune-*.txt` after changing kernels).
- Thermals: the phone throttles. Let it cool between long runs, take the
  median of ≥3 runs, alternate A/B order, and record `adb shell dumpsys
  thermalservice | head` when numbers drift. Report power where possible
  (`adb shell dumpsys batterystats` / on-device power rails) — energy per
  transcription matters as much as latency.

### Phone measurement protocol (#325; evidence in RESEARCH_LOG P2-*)

Working settings, not device limits — each states its evidence and its
status:

- **Settle, then verify.** Waiting ~5 min after boot is a heuristic. What
  makes a window valid is stable thermals (`dumpsys thermalservice`) and
  baseline timings back in the session's band — check those, don't trust
  the timer.
- **Model loads per boot.** "~8 loads, then the driver degrades" (the
  Phase 1 loop header and the P2 protocol addenda) was observed under harnesses that
  also SIGKILLed benches; P2-12–P2-16 later ran many more load/kill cycles
  without persistent degradation once teardown was clean. A conservative
  per-session load budget is a fine experiment setting; it is not an
  established limit.
- **Wake state.** A locked phone without any wake source stalls each decode
  round-trip ~0.7 s through system suspend (P2-3/P2-4). `svc power stayon
  true` only keeps a plugged-in phone awake; it is **not** a wake lock, and
  what kept the P2-8 discharging windows stall-free is **unverified**. The
  verified wake source for screen-off windows is the shell wake lock below.
- **Never run GPU work on a phone that may suspend.** Kernel suspend with
  GPU work outstanding wedges the PowerVR driver: screen off without a wake
  lock wedged 5/6 trials, awake 0/13, forced deep Doze under a wake lock 0/3
  (RESEARCH_LOG P3-1–P3-4). Suspend, not Doze, is the trigger, so a
  plugged-in phone with the screen off is exposed too. The phone bench
  scripts hold a shell-uid partial wake lock for the whole session, which
  Doze does not disable (`phone_common.sh` `wake_hold`, built on demand from
  `wakehold/WakeHold.java`; needs a JDK and the Android SDK). A new script
  that runs benches with the screen off does the same: `wake_hold` before
  the first bench, `wake_release || true` in its EXIT trap, `wake_held`
  before each measurement window.
- **Run `wedge_forensics.sh watch start` at the start of every phone
  session**, so an incident's minutes are still on the device.
- **Energy runs** need a discharging battery: verify the charge counter
  moves between two reads before opening a window.
- **Wifi adb.** Stream bench output to device-side files; the TLS transport
  stalls under sustained shell output.

### GPU failures and phone restarts

The evidence rotates away within minutes (logcat) to hours (kernel log), so
this procedure outranks the experiment in progress.

**Recognize it.** A fence timeout (`vkWaitForFences failed (VkResult 2)`),
`device lost`, a wedge marker refusing a load, a bench hung past its
timeout, decode times in the degraded band (~2× the session's median,
bimodal), or a restart you did not cause (boot reason not `reboot,shell` /
`reboot,userrequested`).

1. **Capture first**, before any retry, recovery load or reboot:
   `wedge_forensics.sh event <label>` while the phone is up, or
   `wedge_forensics.sh post-reboot <label> --bugreport` as soon as adb is
   back after a restart (only the bugreport keeps the previous boot's
   kernel log).
2. **Record it** in `RESEARCH_LOG.md`: the forensics directory, the boot
   reason, and what the capture cannot see — binary and commit, model,
   `STARLING_FAST_KSTEP`, loads and dirty deaths (SIGKILL/timeouts) this
   boot, uptime, battery and charging, what ran in the minutes before.
   Write "unknown" where unknown.
3. **Then** follow rule 3 of the root `AUTORESEARCH.md`. Never delete a
   wedge marker to get past it; wait out its 15 minutes or reboot.

**What the evidence decides:**

| hypothesis | look for |
| --- | --- |
| kernel suspend with GPU work outstanding (reproduced, P3-3/P3-4) | `mWakefulness` and suspend blockers in `summary.txt`; GPU rail ~0 mW during the hang in `gpu-rail.txt` |
| unclean device death (process killed with a live VkDevice; P2-12) | dirty deaths this boot before the incident; clean `vk teardown` lines on the clean ones |
| GPU job too long → failed hardware recovery | `pvr`/`rogue`/`HWR`/`lockup` lines in `gpu-lines.txt` before the hang; KSTEP of the run |
| thermal | thermal status and temperatures in `summary.txt` |
| low battery / power management (all early wedges at ≤ 44 %) | battery level and charging state |
| memory pressure | `lowmemorykiller`/`lmkd` kills, `MemAvailable` |
| driver/firmware | build fingerprint; the boot-time `RGXValidateFWHeaderVersion2: KM and FW version mismatch` line seen on this unit (2026-10) |

A restart with a boot reason such as `kernel_panic` or `watchdog`, with
driver lines before it, is the report Imagination/Google need: attach the
forensics directory and the bugreport.

## Experiment loop

For each iteration:
1. State one hypothesis with the expected gain and why (cite the measurement).
2. Make the smallest change that tests it (a new shader variant or spec
   constant beats rewriting a working kernel; keep the old path selectable).
3. Build, run the gates that the change can affect, measure A/B on the phone.
4. Keep it (commit with the measured numbers in the message) or revert it.
5. Append a row to `benchmarks/fast_engine/RESEARCH_LOG.md`: hypothesis,
   change, before/after (median, spread), gates, verdict.
Stop an idea after two failed variants; move to the next. Re-rank ideas as
measurements come in. Commit often; push to the PR branch at milestones.

## Idea backlog (ranked by expected payoff; verify, do not assume)

**Encoder GEMM on PowerVR (largest single item)**
- The GEMM (`shaders/gemm.comp`) was tuned on AMD (subgroup 64, LDS-bound).
  PowerVR has subgroup 128, 32 KiB shared memory and different register
  files. Try: larger K tiles per barrier, register double-buffering of the
  next A/B tile, fewer barriers, `f16vec4` math, shared-memory layouts
  without bank conflicts for 128-wide subgroups, workgroup = one subgroup.
- Try `VK_KHR_cooperative_matrix` if the driver exposes it (check
  `vulkaninfo` on the device via `adb shell` or a small probe).
- Try `VK_KHR_shader_integer_dot_product` (int8 activations × int4/int8
  weights, dp4a-style) if reported as accelerated.
- Subgroup operations (arithmetic, shuffle) for reductions and operand
  broadcast; the runtime currently avoids them for portability — gate them
  on the reported `subgroupSupportedOperations`.
- Fuse LayerNorm into the following GEMM's A-tile load (the row statistics
  are two scalars per row), removing 5 dispatches + barriers per layer.
- Dispatch/barrier overhead: 464 dispatches for the Parakeet encoder; on a
  tiler each barrier may flush tiles. Measure a recording of empty dispatches
  to price a barrier, then fuse accordingly (q/k/v into one GEMM with three
  outputs, attention scores + softmax + PV into one flash-style kernel,
  depthwise conv into the pw2 A-load).
- Parakeet attention materializes S (T² f32) and BD (2T² f32) per head; a
  fused rel-pos flash attention removes that traffic.

**MOSS decode (bandwidth-bound: ~1.1 GB weights per token)**
- GEMV (`shaders/gemv.comp`): measure achieved GB/s per matrix; try more rows
  per workgroup with subgroup reductions, 128-bit loads of 2–4 groups per
  lane, f16 math for the dequant, and int8 dot with per-token int8 x.
- Fewer dispatches per token (currently ~146): fuse o-proj + residual +
  RMSNorm + gate/up where possible; one command buffer per K tokens already
  exists (`STARLING_FAST_KSTEP`).
- The Q8 tied lm_head is 28 % of the bytes: try the q4e4 model file (Q4
  embeddings) and report its WER; consider a smaller-precision lm_head copy
  with exact re-scoring of the top candidates.
- Speculative decoding for transcription (draft from Parakeet's transcript
  or n-grams, verify K tokens per weight pass) — large potential, needs care.

**CPU side / system**
- Parakeet TDT decoder (0.2 s per 22 s): i8mm (SMMLA) GEMV, two-thread row
  split on the X4 + one A725 with a spin barrier, and keep weights hot.
- Mel on the GPU (DFT-as-GEMM) inside the encoder recording — only if the
  CPU mel shows up in the phone's profile.
- Load time: repack in parallel already; consider caching the repacked blob
  on disk (mmap) to skip repacking entirely on later loads.
- Energy: prefer fewer, larger submissions; let the CPU sleep on fences;
  check GPU/CPU frequency residency during a transcription.

## Deliverable

A sequence of commits on `feat/fast-vulkan-engines`, each a measured win, a
`RESEARCH_LOG.md` with every experiment (including failures), updated numbers
in `docs/fast-engine.md` (Pixel section), and a final summary: before/after
latency and energy per model, WER gates, what did not work and why.
