# Program decisions

Consequential choices, each with the alternative that was rejected. Append
only; never silently amend a recorded decision.

## D1 — 2026-09-20 — No upstream writes from agent sessions
User instruction (AGENTS.md): never file PRs upstream. Earlier package
publication attempts hit 403. **Decision:** all integrated work lands on local
program branches (`program/*`); ready-to-publish records are kept in
STATUS.md/tasks.json for the human to push. **Rejected:** using `gh` to open
PRs/issues on sims1253/starling from agent sessions.

## D2 — 2026-09-20 — Branch topology
`program/wave-a` branches from master `12b6548` in the main checkout for
control plane + master-side fixes (B/E10/E16/E06/E28 contract work). GPUI work
continues in the `starling-gpui` worktree on branches off PR #193 head
`4569e67` (G01–G07, E15, E17 extraction starts there). Integration branch
`program/integration` (from master) merges both streams in small verified
increments. The dirty ggml submodule working tree is user work and stays
untouched; engine experiments that need a clean submodule use their own
worktrees with `git submodule update` per-branch, never resetting the main
checkout's submodule state.

## D3 — 2026-09-20 — Supplied packages stay untracked
`starling-final-package/`, `starling-serving-optimization/`, `bun.lock` are
added to `.git/info/exclude` (local, not committed) so program commits cannot
accidentally absorb them. Their reference implementations (router.py,
metrics.py + fixtures/contracts) are the executable behavioral specs; porting
happens into product code with the fixtures as tests, not by vendoring the
Python.

## D4 — 2026-09-20 — Serving first wave
S01 (MOSS device-side KV clearing), S02 (TDT cache budget), S11 (exact-tail
reuse), S09 (packed CPU dispatch investigation), S03 (EOS semantics, separated
from speed work) per the addendum's evidence-ranked order. S03's acceptance is
a termination-contract test, deliberately not bundled into any timing change.
No speedup claims without paired benchmark runs through the existing harness.

## D5 — 2026-09-20 — Status truthfulness vocabulary
Every tasks.json item carries one status from the fixed vocabulary; a compiled
change is `implemented`, merged into a program branch is `integrated`, and only
executed-on-named-hardware evidence yields `verified_on_target`. Historical
cumulative speedups from unmerged branches are not counted as gains.
