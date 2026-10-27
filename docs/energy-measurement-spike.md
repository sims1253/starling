# Energy-measurement spike (#476)

Time-boxed research issue. Companion to the per-model speed check (#475)
and Insights (#308). Conclusion: **an in-app "energy per minute" card is
not feasible in v1; report time + peak RSS and label energy as unknown.**

## Per-platform method survey

### Linux notebook — RAPL (`/sys/class/powercap`)

- Available domains: `intel-rapl:0` (package) and `intel-rapl:0:0` (PP0,
  cores). Reading `energy_uj` requires membership in the `powercap` group
  (or root); on many distros the files are `root:root 0444` only.
- Scope: package energy includes uncore/DRAM on some SKUs; PP0 is cores
  only. Neither isolates "this transcription" from background load, so a
  rail-grade number needs a quiescent machine and repeated runs.
- Variance: run-to-run variance is dominated by thermal state; a quiescent
  machine and repeated runs are required. (No measured figure available.)
- Cost: permission/setup friction is real (group membership, reboot). Not
  shippable as a zero-config feature.

### Android (Pixel 10 Pro primary target)

- `BatteryManager` exposes `BATTERY_PROPERTY_CURRENT_NOW` (µA) on some
  devices; Pixel firmware typically returns a coarse estimate or does not
  support it (`getIntProperty()` returns `Integer.MIN_VALUE` for an
  unsupported property per the API contract).
  `BATTERY_PROPERTY_ENERGY_COUNTER` (nanowatt-hours) is even less
  consistently populated.
- `batterystats` is a *model*, not a rail measurement: it attributes
  estimated energy to apps/packages from CPU time + screen + radio models.
  Per #292 working rules, batterystats output must never be presented as
  joules.
- Rail measurement requires external hardware (Monsoon/ Otii) or a
  device-specific kernel debugfs path; not available to a shipped app.
- `benchmarks/fast_engine/phone_energy.sh` collects a charge-counter-based
  energy per transcription on the phone (primary), with the batterystats
  power model reported alongside as an explicitly estimated cross-check.

### iOS

- No public API exposes energy for a third-party app. `os_signpost` /
  Instruments are developer-only. `UIDevice.batteryLevel` is a coarse
  (5 %) percentage with no rate. In-app energy reporting on iOS is
  effectively impossible without private APIs.

## Prototype outcome

### Pixel 10 Pro: measured (2026-10-27)

`benchmarks/fast_engine/phone_energy.sh`, run twice back to back, with
Parakeet TDT 0.6B v3 q4pw-fullimx (the mobile default). Each session has
three windows: 400 transcriptions of `short.wav` (7.43 s of audio) on the
fast engine, a duration-matched idle control, then 400 on the ggml CPU
path. Setup: one model load per window, screen off, shell wake lock,
battery discharging (29 → 22 %), `starling-bench` at `28d1ff66`, 6
threads. The fast engine ran its GPU encoder: there were no TPU artifacts
in the bench cache directory. Primary method: the charge counter (a
coulomb gauge, not a rail), idle drain scaled to each window and
subtracted, and a nominal 3.87 V to convert µAh to mWh.

| session | idle drain | fast: raw / net | ggml: raw / net | median wall fast / ggml |
|---|---|---|---|---|
| 1 | 32 500 µAh / 230 s (141 µAh/s) | 90 000 / 57 500 µAh | 62 500 / 35 370 µAh over 192 s | 573 / 482 ms |
| 2 | 12 500 µAh / 227 s (55 µAh/s) | 75 000 / 62 500 µAh | 65 000 / 56 134 µAh over 161 s | 570 / 392 ms |

Per transcription that is: fast **0.56 / 0.60 mWh** net (0.87 / 0.73 raw)
and ggml **0.34 / 0.54 mWh** net (0.60 / 0.63 raw). Per audio minute
(× 60 / 7.43): fast 4.5–4.9 mWh (16–18 J) and ggml 2.8–4.4 mWh (10–16 J)
net. The raw drain bounds them from above at 5.9–7.0 and 4.9–5.1 mWh.

What this says about the method:

- **The idle baseline is the largest moving part.** It moved 2.6× between
  two consecutive sessions (141 vs 55 µAh/s) with the same screen-off,
  wake-locked state. Background apps and radios are not under our control.
  The net ggml estimate differed by 59 % between the sessions and the fast
  one by 9 %. Raw drain and window lengths changed too, so two sessions
  neither isolate the cause nor give an uncertainty bound.
- **Gauge resolution.** Every counter reading in both sessions is a
  multiple of 2500 µAh, the step already recorded on this device in
  `benchmarks/fast_engine/RESEARCH_LOG.md` (#317). Quantization alone
  is therefore up to ±5000 µAh per window delta, ±0.05 mWh per
  transcription at 400 per window; idle drift (above) adds to that. A
  single dictation take (seconds) drains less than one step, so the gauge
  cannot give a per-take reading.
- **`current_now` is readable** on this Pixel
  (`/sys/class/power_supply/battery/current_now`, µA; −620 mA at the
  time). That corrects the survey above for this device. It is still a
  fuel-gauge estimate of whole-device current, so the idle-attribution
  problem stays.
- batterystats did not give a per-window figure here: the script's
  cross-check printed history lines, not an estimate. The power model
  remains unused as evidence.
- Side result: on this phone, with the GPU encoder, Parakeet short clips
  cost **no more on the ggml CPU path than on the fast engine** (ratio
  0.62× and 0.90×), and run faster (392–482 vs 570 ms median). It is not
  a recommendation to switch engines: it is one clip length, one model,
  and the TPU encoder path was not active.

### Linux notebook RAPL: still outstanding

Not measured. This run was on a WSL2 desktop with no
`/sys/class/powercap`, and the notebook was not available. The RAPL
section above is still a survey, and the notebook prototype that #476
asks for remains open.

### Conclusion

The phone can produce a session-level energy-per-audio-minute estimate only
as a lab measurement: minutes of sustained load and an idle control, with
estimates that differed by up to 59 % between two sessions. That is not a
per-take number a user card could show, and the survey alone already
declined v1.

## Recommendation

- **v1: ship time + peak RSS only** (already in `starling-bench`), and
  label energy as **unknown** in the UI. No fake joules: if a platform
  cannot report rail-grade energy, show "unknown", never zero.
- **v2 candidate**: a research-only RAPL path behind an opt-in developer
  flag on Linux notebooks, reporting joules with method + variance, for
  developers who can grant the permission. Not a default user-facing card.
- Insight-events schema: no energy field in v1. If a v2 energy card ever
  ships, bump insight-events to v2 with an `energy_uj` field and a
  `energy_method` enum (`rapl_package`, `batterystats_model`,
  `unknown`); until then the field is absent, not zero.

## Privacy contract

Holds: no raw text, counts only (`packages/contracts/insight-events/README.md`).
Energy numbers are aggregate per take; no per-user data leaves the device.
