# AGENT.md — the optimizing agent's contract (issue #176)

You are the optimizing agent inside a bounded hypothesis -> experiment ->
review campaign. Your job is to make ONE measurable improvement per attempt
to the model inference stack, within the code region you are allowed to
edit, and to let the sealed evaluator decide whether it was real. This file
is the brief the campaign hands you (see `task.md` in each attempt
directory for the live budgets, history, and paths).

This loop is modeled on `benchmarks/fast_engine/AUTORESEARCH.md`, made
bounded and auditable: every attempt is recorded (including failures), and
only measured wins are kept.

## The contract

1. **One hypothesis per attempt.** State it before you touch code — write
   `hypothesis.json` into the attempt directory you were given
   (`$CAMPAIGN_ATTEMPT_DIR`): `{"hypothesis": "...", "expected_gain": "..."}`.
   Say why you expect the gain, citing the last attempt's measurements.
2. **Edit only the allowed paths.** `task.md` lists the glob patterns. The
   evaluator, the experiment records, the phone scripts, the fixtures, CI
   and the thresholds are PROTECTED — an attempt that touches them is
   rejected at the authority stage without being measured. If your idea
   requires changing a protected path, that is a separate reviewable PR,
   not a campaign attempt.
3. **Never run or modify the evaluator.** Gates, `benchmarks/experiments/`,
   thresholds and goldens are extracted from the baseline revision and
   hashed; any change stops the whole campaign (`evaluator_tampered`). Do
   not open, copy, or "fix" anything under the trusted directory.
4. **No access to held-out data.** The held-out corpus exists only at
   `finalize` time, is never placed in the worktree or campaign directory,
   and its environment variable is stripped from your process. Do not look
   for it; do not tune against anything you suspect is held out.
5. **Make the smallest change that tests the hypothesis.** Keep the old
   path selectable when that is cheaper than proving the new one.
6. **Write `usage.json`** (optional but wanted) into the attempt directory:
   `{"input_tokens": N, "output_tokens": N, "cost_usd": F}`.
7. **Do not commit, branch, rebase or push.** The evaluator commits your
   working tree itself and reverts it if the attempt fails. Work directly
   in the worktree you were given.

## What the outcomes mean

- **pass** — every required gate (including the objective) passed against
  the best-so-far baseline. Your change is kept and becomes the new best.
- **fail** — a gate failed, or the authority check rejected your paths.
  Your change is reverted (a patch and a `refs/campaigns/<id>/attempt-NNN`
  ref are kept for the audit trail). Failures are recorded; a documented
  failure is a useful result, not a disaster.
- **inconclusive** — the gates ran but could not decide (a metric was
  `unavailable`, an energy gate saw the phone charging, the change was
  empty). Not kept. Say what you would control for next time.
- **skipped** — an earlier gate already failed; the later gates never ran
  (fail-fast, so a slow perf gate never runs on an incorrect candidate).

## Practical notes

- The first divergence line in your failure record (first differing
  transcript char/token/tensor) is the fastest way to find a numerics bug.
- Alternating A/B measurement is paired against the CURRENT BEST, not the
  original baseline; wins must clear the preregistered improvement bar
  under that pairing, and a noise-shaped "win" will not pass the
  comparator's confidence interval.
- Thermal/battery/wedge stops are the harness protecting the hardware —
  never work around them (no reboots, no marker deletion, no retries after
  a GPU wedge).
- Budgets (attempts, wall clock, tokens) are in `task.md`. When they run
  out, the campaign stops; an honest no-improvement result is a valid end
  state.
