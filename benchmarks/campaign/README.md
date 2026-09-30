# Campaign runner: bounded hypothesis → experiment → review (issue #176)

This directory wraps a coding agent in the discipline the fast-engine
research loop (`benchmarks/fast_engine/AUTORESEARCH.md`) already uses
manually: one preregistered hypothesis per attempt, independent sealed
gates, keep only measured wins, record every failure, stop on hardware
trouble. It is built on `benchmarks/experiments/` (#168) for paired
statistical comparison and on `benchmarks/fast_engine/` for phone
measurement, and it reuses the existing Pullfrog workflow as the
agent-driven entry point — there is no second agent framework.

Stdlib-only Python 3.10+, like `benchmarks/experiments/`. Tests:
`python3 -m unittest discover -s benchmarks/campaign -p 'test_*.py'`
(hermetic: toy git repo, toy engine, no network, no models).

## The one-command hermetic pilot

```bash
python3 benchmarks/campaign/campaign.py pilot --out build-exp/campaign-pilot
```

Builds a toy repo (a script that sleeps `WORK_MS` and prints text), seals
a real task, extracts a real trusted evaluator, and runs five scripted
attempts past every negative control from the issue: a real speedup
(kept), an incorrect patch (fails correctness with a first-divergence
line), a gate-tampering patch (rejected at authority), a fake speedup that
truncates output (fails correctness, not perf), and a no-op (fails the
perf bar). The report lands in `build-exp/campaign-pilot/report.md`.

## The real-repo CPU pilot (no models, no GPU)

The same pipeline against this repository, using the deterministic
contract-fixture engine and the sealed experiment comparator as the perf
gate (profile `fixture--notebook-cpu`; allowed paths default `cpp/serve/**`):

```bash
python3 benchmarks/campaign/campaign.py preview --profile fixture--notebook-cpu \
    --baseline <sha-or-branch>
python3 benchmarks/campaign/campaign.py start   --profile fixture--notebook-cpu \
    --baseline <sha> --out build-exp/campaign-serve \
    --agent-cmd "uv run python my_agent.py"        # or omit for agent-driven mode
python3 benchmarks/campaign/campaign.py run     --campaign build-exp/campaign-serve
python3 benchmarks/campaign/campaign.py report  --campaign build-exp/campaign-serve
```

The baseline revision must contain `benchmarks/campaign/` itself (the
trusted evaluator is extracted from the baseline, not your working tree).

## Overnight walkthrough

Notebook (Ryzen 5 PRO 5650U / Vega 7 / RADV), e.g. Parakeet:

```bash
export STARLING_MODELS_DIR=~/models          # holds the pinned GGUFs
uv run python tests/fixtures/make_fixtures.py  # short/medium/long.wav (gitignored)
export STARLING_FIXTURES_DIR=$PWD/tests/fixtures   # sealed as artifacts at start
export STARLING_CAMPAIGN_BUILD_JOBS=4        # build parallelism cap (memory-bound notebook)
python3 benchmarks/campaign/campaign.py list
python3 benchmarks/campaign/campaign.py preview --profile parakeet--notebook \
    --baseline origin/master --out ~/campaigns/pk-$(date +%F) --wall-clock 8h
python3 benchmarks/campaign/campaign.py start --profile parakeet--notebook \
    --baseline origin/master --out ~/campaigns/pk-$(date +%F) \
    --wall-clock 8h --agent-cmd "<your agent command>"
systemd-run --user --scope python3 benchmarks/campaign/campaign.py run \
    --campaign ~/campaigns/pk-$(date +%F)      # or: nohup … &
```

Pixel 10 Pro (adb, screen-off protocol, #325 wedge guard):

```bash
python3 benchmarks/campaign/campaign.py preview --profile parakeet--pixel \
    --baseline origin/master --out ~/campaigns/pkpx-$(date +%F) --adb-serial <serial>
python3 benchmarks/campaign/campaign.py start --profile parakeet--pixel \
    --baseline origin/master --adb-serial <serial> --out ~/campaigns/pkpx-$(date +%F) \
    --wall-clock 8h --agent-cmd "<your agent command>"
python3 benchmarks/campaign/campaign.py run --campaign ~/campaigns/pkpx-$(date +%F)
```

Morning:

```bash
python3 benchmarks/campaign/campaign.py status --campaign DIR   # attempt table
python3 benchmarks/campaign/campaign.py report  --campaign DIR  # report.md + report.json
```

After a crash/power loss (or a `SIGTERM` — the runner checkpoints
`interrupted`, kills its children's process groups and releases its locks):

```bash
python3 benchmarks/campaign/campaign.py resume --campaign DIR
```

Promotion check — revalidates the exact best commit against the original
baseline on THIS device, then (only here) runs the held-out gates:

```bash
python3 benchmarks/campaign/campaign.py finalize --campaign DIR --heldout $STARLING_HELDOUT_DIR
```

Without `--heldout`, the report says `promotion: blocked (held-out corpus
not provided)`; a device the finalize never ran on stays
`blocked: not validated on <device>` — never substituted. Promotion also
needs a PINNED held-out corpus (`start --heldout-pin <sha256>` or the
profile's `heldout.sha256`; an unpinned corpus could be swapped for an easier
one) and the profile's `heldout_gates`. The committed model profiles do not
define held-out gates yet (the per-language FLEURS WER gate needs its
corpus chosen and pinned first), so their promotion honestly reports
`blocked: no held-out gates defined for this profile` until one is added.

## Subcommands and exit codes

`list` · `preview` (dry-run, mutates nothing, exit 2 lists every missing
prerequisite) · `start` (preflight → worktree → seal → baseline build →
control attempt 0) · `run` (agent loop) · `attempt` (agent-driven mode:
evaluate the worktree's current change once) · `resume` · `status` ·
`report` · `finalize` · `pilot`.

Exit codes: 0 ok · 2 preflight/usage · 3 stopped by a safety monitor
(thermal, battery, adb, wedge, driver failure, memory) · 4 evaluator
tampered / seal mismatch · 130 interrupted.

## Trust boundary (and its honest limit)

- The **candidate/agent** may write only the task's `allowed_paths`
  (checked at the authority stage against the committed diff; empty diffs
  are recorded `no_change`, inconclusive). Everything under
  `benchmarks/campaign/**`, `benchmarks/experiments/**`,
  `benchmarks/fast_engine/phone_*.sh`, `benchmarks/sonar/**`,
  `tests/fixtures/**`, `.github/**` and the profile's evaluator paths is
  ALWAYS protected, regardless of the task spec.
- The **evaluator** (gates, thresholds, goldens) is extracted at `start`
  from the BASELINE revision — not the candidate, not your working tree —
  hashed, and re-verified before every attempt and at finalize. A mismatch
  stops the campaign (`evaluator_tampered`, exit 4), never a silent
  re-extract.
- **Credentials**: gate/build children receive no provider, cloud,
  release or held-out credentials; the agent keeps provider keys (the
  orchestration boundary) but loses held-out, release/signing and GitHub
  credentials. Scrubbed variable NAMES are recorded, never values.
- **Held-out data** is only read by `finalize`, is never placed in the
  campaign dir or worktree, and its env var is stripped from agent and
  gate environments.
- Gitignored paths are outside the authority check (it inspects the
  committed diff), and reverts use `git clean -fd`, which keeps ignored build
  caches. Profile builds therefore configure their own `build-campaign/`
  from scratch-safe CMake flags, but an agent that plants files in ignored
  paths is not detected — another reason to sandbox the agent.
- Honest limit: the harness DETECTS out-of-bounds writes, it does not
  sandbox filesystem READS. For real read isolation of held-out data, run
  the agent in a container/sandbox; the runner is designed so held-out
  data is not reachable through anything it passes the agent (path, env,
  task description), but a determined process on the same host could look
  elsewhere on disk.

## Safety monitors

Before each attempt and between gates the runner probes: adb state, the
#325 GPU-wedge marker (`starling-fast-gpu-wedged`), wedge/driver-failure
patterns in the last gate's log, battery level while discharging, memory
pressure and temperature (phone battery/thermal severity, notebook thermal
zones). Hot → cooldown wait (bounded by `cooldown_max_s`), still hot →
stop. The runner NEVER reboots the phone, NEVER deletes the wedge marker,
NEVER retries a load after a wedge; a wedge stop is terminal until a human
inspects, reboots and clears the marker (then `resume` re-probes).

One host-wide flock (`$STARLING_CAMPAIGN_LOCK`, default
`<tmp>/starling-campaign.lock`) serializes builds/GPU work/agents; pixel
campaigns add a per-serial flock. It is deliberately a DIFFERENT lock from
`$STARLING_EXPERIMENT_LOCK`, which the perf gate's `run_experiment.py`
child takes itself — no deadlock.

## Pullfrog usage (agent-driven mode)

Pullfrog runs on a GitHub-hosted runner, so it can only PROPOSE: it never
holds the campaign, the evaluator, the device or the held-out data. The
split keeps the trust boundary intact:

1. Start the campaign locally without `--agent-cmd` (agent-driven mode).
2. Ask the existing workflow for one hypothesis as a branch based on the
   campaign's current best commit (`campaign.py status` prints it):

   ```bash
   gh workflow run pullfrog.yml -f name=campaign-pk -f prompt="Follow benchmarks/campaign/AGENT.md. Starting from commit <best-sha>, make ONE change inside <allowed paths> testing this hypothesis: <...>. Push it to branch campaign-proposal/pk-1 and stop. Do not run benchmarks."
   ```

3. On the machine holding the campaign, apply only the proposal's diff to
   the campaign worktree and let the TRUSTED evaluator judge it:

   ```bash
   git fetch origin campaign-proposal/pk-1
   git diff <best-sha> FETCH_HEAD | git -C ~/campaigns/pk-2026-09-30/worktree apply
   python3 benchmarks/campaign/campaign.py attempt --campaign ~/campaigns/pk-2026-09-30 \
       --hypothesis "<the proposal's hypothesis>"
   ```

The proposal's claims (numbers, "tests pass") are never evidence; only the
local gates are. An edit to a protected path is rejected at the authority
stage like any other candidate.

The workflow file itself is vendor-managed and stays unmodified; the
runner never merges, pushes or changes default models, and neither does
the suggested draft-PR command in the report (a human runs it, or not).

## Profiles

`profiles/<model>--<device>.json` (schema `starling-campaign-profile/1`):
readiness, artifacts (hash-pinned or hashed-and-sealed at start), build,
gates, workloads (honestly `unavailable` with an owner issue where no tool
exists — the interactive/streaming workloads of #226/#310 are listed, not
faked), objectives, quality policy (exact numerical contract for kernel
changes; +0.2 abs WER points per language for quant changes, and never
promotion from the 100-clip English smoke set alone), held-out config,
default budgets and monitor thresholds. `list` prints the table; `start`
refuses a `blocked` profile with its `blocked_on`/reason. Notebook profiles
gate correctness on exact transcripts of the fixture clips versus the
ORIGINAL baseline binary (`gates/serve_contract_smoke.py --baseline-binary`),
so a truncated or altered transcript fails however fast it is; perf uses the
sealed `benchmarks/experiments` comparator. Pixel profiles run the generic
`gates/phone_bench_ab.sh` (static `starling-bench`, alternating rounds,
transcript match + latency delta with min/max). Gates with `"phase":
"finalize"` are measurements that run only at `finalize` and cannot decide a
keep: MOSS-on-Pixel's `phone_energy.sh` gate is one (two extra model loads
per run would spend the #325 per-boot load budget on every attempt; exit 3
when charging -> inconclusive, never zero).
`phone_gates.sh` itself is not wired as a gate because it builds into its
own `build-android/` tree — inside the trusted evaluator that would mutate
the sealed tree; `phone_bench_ab.sh` is its generalized, build-free form.

## Files

- `campaign.py` — CLI + state machine + checkpointing + worktree management.
- `spec.py` — task/profile schemas, validation, sealing, `**`-glob matcher.
- `gates.py` — gate execution: process groups, timeouts, credential
  scrubbing, METRIC parsing, rules, verdicts.
- `monitors.py` — device discovery + safety probes (injectable for tests).
- `report.py` — report.md / report.json rendering.
- `toy.py`, `toy_agent.py` — the hermetic toy fixture + scripted agent.
- `gates/` — trusted gate scripts (serve contract smoke, experiments A/B
  adapter, generic phone A/B gate).
- `profiles/` — the committed per-model × per-device profiles.
- `gen_profiles.py` — regenerates `profiles/` (edit profiles there).
- `test_*.py` — the acceptance suite.

## Non-goals

No autonomous deployment, no multi-agent platform, no unrestricted
optimizer, no auto-merge, no push, no default-model changes. A bounded,
resumable, audited loop with independent acceptance criteria — and an
honest no-improvement outcome as a valid result.
