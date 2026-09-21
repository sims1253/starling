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
2026-09-21 addendum: `filed` marks not-yet-started open work consolidated into
a GitHub issue via the per-item `github_issue` field; it is orthogonal to
progress states — an item that starts work moves to `in_progress` (or beyond)
with `github_issue` retained.

## D6 — 2026-09-20 — E17 sequencing: library-first, envelope before IPC
Adopted the design note (e576a7a): `starling-runtime` in-process library (Mode A)
first, thin user-scoped host (Mode B) behind the identical versioned envelope
later; increments I0 (envelope contracts) → I1 (capture hardening) → I2
(storage v2 + migration) → I3 (runtime crate Mode A) → I4 (IPC host) → I5
(documents/context/delivery). IPC crate choice and per-OS auth are I4 decisions,
not now. **Rejected:** building the service host first; ad-hoc internal API
before the envelope.

## D7 — 2026-09-20 — Streaming resampler for I1: stateful port, no new deps
I1's streaming capture needs a stateful anti-aliased resampler; G06 landed the
stateless whole-recording port of the master #122 windowed-sinc kernel with no
new dependencies. Decision: extend that kernel to a streaming/stateful form
rather than adopting rubato or another crate; revisit only with measured
spectral/latency evidence showing it inadequate. **Rejected:** new resampler
dependency without evidence; keeping linear interpolation anywhere on the path.

## D8 — 2026-09-20 — Upstream pushes and PRs authorized by user
The user explicitly authorized pushing program branches and opening PRs on
sims1253/starling so automated reviews run on them ("Let them push to PRs").
This supersedes D1's default for this session's program branches. Still
forbidden: merging, force-pushing shared branches, pushing to master, and
claiming publication without a successful command result. Coordinator pushes
verified branches; agents never push directly.

## D11 — 2026-09-20 — Storage v2 uses rusqlite with the bundled SQLite
I2 needs a transactional metadata store; the design (§4) specifies SQLite
in WAL mode. Decision: `rusqlite` with the `bundled` feature — no system
sqlite dev dependency on user machines, one cargo cache, version pinned by
the lockfile. **Rejected:** system sqlite (packaging/ABI variance per OS);
a hand-rolled journal-only metadata format (re-invents transactions badly).
WAL checkpoint policy stays tunable per the design's slow-disk caveat.

## D12 — 2026-09-20 — All G-items and R-items from the port review are closed
as of gpui/fixes-wave-a10 (8153a35): G01 (I1 ring+journal), G02 (per-record
isolation + orphan recovery), G03-G07, R01-R21. Remaining GPUI-port work is
product scope (E15 parity items, streaming transport parity), not review debt.

## D13 — 2026-09-20 — Engine branches frozen pending the 5090 CUDA gate
#196-#199 and program/gpu-validation are frozen at 4623e4c / 22e7d1b /
adb0893 / 7340eb6 / 52f393d until the user's 5090 agent validates and
merges. R27/R29 (P3 nits) deliberately NOT landed now so the validation
branch the agent tests is exactly the PR heads. New work continues from
master (program/wave-b).

## D14 — 2026-09-20 — No backwards compatibility of any kind
User directive: the product is pre-release; there is no user data to be
compatible with. Consequences: (1) storage-v2 becomes THE store — the v1
migration flow (dry-run/apply/rollback UI), the dual-store facade, the
STARLING_STORAGE_V2 flag, and the persisted-choice machinery are unnecessary
complexity to be removed, not maintained; (2) legacy-field fallback paths
(e.g. the Electron threadJoinedAt createdAt-ordering fallback) may be
deleted rather than preserved; (3) no schema coexistence: bump/replace
freely. Source evidence survives as raw journals where the crash-recovery
policy demands it — that is durability, not compatibility.
