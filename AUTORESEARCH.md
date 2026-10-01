# AUTORESEARCH — standing constraints for agent-run optimization loops

How we optimize Starling with agents: **an issue names the target, an agent
(pi with the pi-autoresearch loop, or any coding agent) works the issue, and
this file is the standing context every such loop inherits.** Nothing here
is per-issue; issues stay short.

For the Pixel fast-engine campaign there is a second, device-specific brief
with measured baselines, hardware notes and its own quality gates:
`benchmarks/fast_engine/AUTORESEARCH.md`.
When the two overlap, the constraints below win.

## What an optimization issue must state

Target (model + binary), device (notebook / Pixel 10 Pro), the metric
(latency, energy per transcription, memory), the accept bar, and the code
region the agent may touch. Everything else lives here.

## Benchmark data — the only inputs

- Audio fixtures: `tests/fixtures/{short,medium,long}.wav`.
- Models: the GGUFs in `$STARLING_MODELS_DIR` (or
  `/data/local/tmp/starling/` on the phone). Never edit a model or fixture
  to make a number move.
- Word-error evaluation: FLEURS-en 100 clips via
  `benchmarks/fast_engine/export_fleurs.py` + `wer_engines.py`
  (`pixel_baseline_transcripts.txt` holds the phone goldens).

## Evaluation — independent of whatever you changed

1. **Correctness is an exact-transcript contract.** Build the baseline
   binary BEFORE the first change; after every kept change the fixtures must
   transcribe to identical text on both binaries. A faster but different
   transcript is a failure. (`benchmarks/experiments/serve_contract_smoke.py`
   does this over HTTP for serve binaries; `phone_gates.sh` on the Pixel.)
2. **Performance is an alternating A/B against the baseline binary, median
   of ≥3 runs, screen off on the phone.** For CPU/notebook work use the
   paired comparator (`benchmarks/experiments/run_experiment.py`): it
   reports a confidence interval — a single faster run is noise, not a win.
3. **Numerics changes** (f16 accumulation, int8, fewer bits) additionally
   keep FLEURS-en WER within 0.2 points of the ggml engine.
4. **Desktop RADV must not regress >10 %** — it is the CI and development
   device; phone wins may trade a little there, not more.

## Hard constraints

1. Never modify the measurement scripts, fixtures, models, goldens, or this
   file inside an optimization loop. Propose changes to the measurement
   stack as a separate issue/PR.
2. Record the baseline before the first attempt; every claim is relative to
   it.
3. Pixel safety (#325): posture depends on whether the loop is attended.
   **Attended** (the operator is around and says keep going): wedges are
   **data, not session-enders** — on a wedge (hang, fence timeout, marker):
   capture forensics (`.auto/wedge-forensics.sh`: logcat tail, thermals, GPU
   devfreq state, marker content → timestamped host file), append the event
   to `RESEARCH_LOG.md`, attempt recovery **without rebooting** (idle a few
   minutes, probe one load; reboot is the fallback, not a defeat), and keep
   working — the operator stops the loop when the phone is needed.
   **Unattended** (overnight loops): after a wedge, capture forensics and
   STOP phone work — post-wedge measurements are invalid anyway, and an
   unsupervised loop must not keep feeding a wedged driver; resume on
   operator intervention or a fresh day. In both modes: kill stray benches
   (`phone_common.sh: kill_benches` — TERM first, KILL only as fallback)
   before every new measurement, and record `dumpsys thermalservice` when
   numbers drift. Measurement hygiene: budget **≤ 8 model loads per boot**
   for *clean-window* measurements (the driver degrades beyond that; batch
   transcript gates to one load per model+binary); **settle ≥ 5 min after
   boot** before the first round; keep the phone **screen-off with an
   awake-hold** (`svc power stayon true`) or unlocked — on a locked phone
   without a hold, system suspend stalls each decode round-trip ~0.7 s and
   the numbers are void; energy runs need a **discharging** battery —
   verify the charge counter moves between two reads before opening a
   window. On wifi adb, stream bench output to device-side files and bound
   every adb call with a host timeout (the TLS transport stalls under
   sustained shell output).
4. Failures are data: append every attempt — kept or reverted — to the
   relevant `RESEARCH_LOG.md` with its numbers. "No improvement" is a valid
   result; say so.
5. No auto-merge. A loop ends with a branch/PR carrying the measured
   numbers; a human merges.

## Harness pointers

- Phone: `benchmarks/fast_engine/android_bench.sh` (build+push+compare),
  `phone_ab.sh` (A/B), `phone_gates.sh` (correctness), `phone_energy.sh`.
- Notebook/CPU: `benchmarks/experiments/run_experiment.py` (paired A/B with
  CI), `benchmarks/experiments/serve_contract_smoke.py` (exact-contract
  check), `starling-bench` for quick timing.
- Engine knobs that need no rebuild: `STARLING_FAST_TILE`, `*_GEMV_ROWS`,
  `*_F16`, `*_KSTEP`, `*_TUNE` (see the fast-engine brief).

## pi setup (per issue)

Point pi at the issue; this file is the standing prompt context. Suggested
`.auto/` trio:

- `prompt.md`: the issue's target + "obey AUTORESEARCH.md".
- `measure.sh`: the A/B command for the target device from the pointers
  above (alternating, median of ≥3).
- `checks.sh`: the exact-transcript check (`serve_contract_smoke.py` /
  `phone_gates.sh`) against the pre-built baseline.
