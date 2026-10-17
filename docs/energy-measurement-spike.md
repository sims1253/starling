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
- Variance: ~2–5 % run-to-run on a quiet laptop; thermal state dominates.
- Cost: permission/setup friction is real (group membership, reboot). Not
  shippable as a zero-config feature.

### Android (Pixel 10 Pro primary target)

- `BatteryManager` exposes `BATTERY_PROPERTY_CURRENT_NOW` (µA) on some
  devices, but Pixel firmware typically returns `-1` (unknown) or a coarse
  estimate. `BATTERY_PROPERTY_ENERGY_COUNTER` (µWh) is even less
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

No rail-grade probe has been run yet; the figures below are the expected
characteristics from the survey, not measured results. A notebook RAPL
probe (package domain) and a phone probe via `phone_energy.sh` (charge
counter primary, batterystats cross-check) are still to be executed and
their raw points filed before any in-app energy card is reconsidered.

- Notebook RAPL: run-to-run variance is dominated by thermal state; even a
  quiescent machine needs repeated runs and a screen-off protocol.
  Permission/setup friction (group membership) makes it unsuitable as a
  default.
- Phone: batterystats deltas track CPU time more than audio length and are
  a model, not a rail measurement; their reproducibility is unverified.

Neither supports a trustworthy "joules per audio-minute" card without the
pending measurements, and the survey alone is enough to decline v1.

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