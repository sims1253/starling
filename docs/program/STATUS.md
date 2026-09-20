# Program status

Last updated: 2026-09-20 09:30 (session 1). Resumable state for any successor
session. See PROGRAM.md for scope, DECISIONS.md for choices, tasks.json for
the item-level inventory.

## Baseline verification (executed this session, model-free)

| Suite | Command | Result |
|---|---|---|
| JS/TS unit | `vp run test` (root) | 89/89 passed (8 files) |
| Rust dictation crate | `cargo test -p starling-dictation` (in `starling-gpui/apps/desktop-gpui`) | 85/85 passed |
| Python server units | `uv run python -m pytest tests/test_server_{routing,robustness,lifecycle,websocket,client}.py -q` | 82 passed |
| Python stream/patches | `uv run python -m pytest tests/test_stream_chunk.py tests/test_ggml_patches.py -q` | 38 passed |
| C++ stream session | `./build/stream_session_test` | 78/78 passed |
| C++ audio parser | `./build/audio_parser_test` | 40/40 passed |
| C++ dtype guard | `./build/higgs_dtype_guard_test` | OK |
| Package self-checks | `validate_package.py`, publisher tests, `reference/` tests, serving `analysis_checks` | all passed (26+13+13, 16+70 subtests) |

Notes: use `uv run python` (never `.venv/bin/python` directly — it fails with
`init_fs_encoding` on this machine). Rust via
`~/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin/cargo` (rustup shims
misbehave in this environment). C++ tests are plain executables, not ctest.

## Resource policy (binding after user feedback 2026-09-20 morning)

This is a 14 GiB notebook. Parallel test barrages + multiple agents caused
heavy swap and user-visible unresponsiveness (no reboot/OOM kills confirmed).
Policy: **one** background compute task at a time, **one** subagent at a time,
targeted scopes, serial builds. Full engine rebuilds and GPU benchmarks are
validation gaps on this machine unless run by the user.

## Reconciled facts

- PR #193 (`gpui-port` @ 4569e67) is OPEN/mergeable; worktree
  `starling-gpui/` holds it. Rust workspace: `apps/desktop-gpui` =
  `crates/dictation` (logic, 85 tests) + `crates/app` (GPUI UI, 0 tests).
  CI runs **no Rust** today (gap for E12/E15/E16).
- All seven G-findings confirmed in code (see GPUI recon summary below).
- gpui port currently: blocking HTTP transcription only (no streaming/WS),
  file-based history (schema v1, `recording.wav` + `manifest.json`),
  linear-interp resampling, single window, global-hotkey via X11 polling,
  no overlay/tray/IME-tested input/a11y/packaging/migration.
- Master baseline engine/builds healthy (see table). ggml submodule carries
  uncommitted Starling patch series — user work, preserved; engine-track
  agents must use separate worktrees and must not reset it.
- Serving addendum REVIEW.md maps S01–S19 to exact files
  (device_cache.cpp, qwen_decode.cpp, tdt_multistep.cpp, relpos_attention.cpp,
  model_loader.cpp, stream_session.cpp, serve/server.cpp); use those
  pointers directly instead of re-exploring.

## Integrated commits

- `program/wave-a` (from master 12b6548): control plane only so far
  (docs/program/*). No code changes integrated yet.

## Active assignments

- G03+G05+G06 bundle → one implementation agent on branch
  `gpui/fixes-wave-a` in `starling-gpui/` worktree (see tasks.json).

## Next actions (in order)

1. Integrate verified G03/G05/G06 into the gpui worktree branch; run
   `cargo test -p starling-dictation` myself as the independent check.
2. G04 (playback watcher) + G07 (PCM rounding cross-language contract) as the
   next single-agent bundle; G01/G02 need E17 ownership design first.
3. E06 fidelity corpus + E28 insight-event contract: fixture-driven test work
   on `program/wave-a` (no model needed).
4. E17 native-runtime ownership extraction design note (contracts + state
   machines) before G01/G02/E01/E02 implementation.
5. Serving S01/S03 implementation in a dedicated worktree with one serialized
   CPU-only build + `stream_session_test` + parity tests; S11 design note.
   GPU timing evidence stays a validation gap on this notebook.

## Validation gaps (honest)

- No GPU benchmark, packaged-app run, Android/iOS build, or real-model
  evaluation executed this session (hardware/scope constraints).
- Python `.venv` direct interpreter broken on this machine; uv required.
- Engine-map deep exploration was cut for resource reasons; serving work
  relies on REVIEW.md's pinned file pointers plus targeted reads.

## Pending human decisions

- None blocking. Standing constraints: no upstream PRs/issues from agents;
  no merges to master; no releases; local-only processing default.

## Update 2026-09-20 (~11:30) — PRs opened per user authorization (D8)

- PR #194: gpui/fixes-wave-a2 → gpui-port (G03-G07 + PORT.md; 95/95 + 11/11 rerun).
- PR #195 (draft): program/wave-a → master (E28+E06 contracts 95/95 rerun, B11
  verified 41/41+53/53, E17 design, control plane). Living branch — push updates.
- PR #196: program/s01-moss-kv-clear → master (S01 committed f84e2a7; agent
  verification pending at open time).
- PR monitor (45 min) now covers all open PRs; new bot reviews are harvested
  into docs/program/pr/ and triaged as R## items.
- G04/G07 integrated (95/95, 11/11 on gpui/fixes-wave-a2). R02/R03 dispatched
  (impl-8, gpui/fixes-wave-a3). Active: B11 reporting, S01 reporting, E10, I0.

## Update 2026-09-20 (afternoon) — Wave A substantially complete on contracts + Electron

Integrated + independently verified + pushed (PR #195 unless noted):
E28+E06 contracts (95), E10 capabilities (16), E17 design + I0 protocol (125),
B11 (41+53), B02/B03 (103+tsc), R14 review fixes (105/54), R15/R16 (146),
E24 multilingual (58+66). GPUI: G03-G07 + R02/R03 + Rust CI workflow on
PR #194 lineage (gpui/fixes-wave-a2 fast-forward; latest 96/96+17/17).
Engine: S01 on PR #196 (CPU-verified; GPU gaps).

Active: I1-phase1 capture ring (gpui worktree), S03 EOS contract (engine,
heavy), mode-routing contracts E18/E19/E26 (suite passing 133 cross-check),
B04/B05 (Electron), E16 reproducibility audit.

Pending dispatch queues: R13 after I1-phase1 (GPUI review fixes incl. stale
connection badge); B06 after B04/B05; B07/B08 Android after S03 frees the
heavy slot (gradle needs a serialized window too); I1-phase2 journal +
G02/I2 storage v2 after I1-phase1; S02/S11 after S03.

Review loop: OCR bots re-review pushed branches; harvest/triage is automated
(45-min monitor) + on-demand; every finding gets fixed or evidence-rebutted
(R14 pattern) — no blind suggestions applied.

Validation gaps (unchanged + new): no GPU timing/quality runs (S01 acceptance
2-3 open), no real-model A/B, no packaged-app/device runs, no Android/iOS
builds this session, ASR quality on E06/E24 fixtures is model-gated by design.

## Update 2026-09-20 (evening) — local iGPU validation + hardware split

- AMD Cezanne iGPU (Vulkan 1.4/RADV) discovered on this notebook: all four
  engine suites pass on the VULKAN device backend (program/gpu-validation,
  build-vk): device_cache_clear (backend=Vulkan0), greedy_termination,
  tdt_graph_budget (graphs + pointer stability on-device), stream_session
  235/235. S01/S02/S03 "source-read only" device gaps are CLOSED for Vulkan;
  CUDA timing/quality remains for the user's RTX 5090 (runbook:
  VALIDATION.md on program/gpu-validation, pushed).
- I1 COMPLETE (ring + journal): 133/133 + 23/23, PR #194 @ 9343bb1.
- B01: first agent died on API rate limit; continuation agent dispatched.
- program/gpu-validation branch = S01+S02+S03 merged + runbook.

## Update 2026-09-20 (night) — review debt fully cleared; E29 + I2 landed

- ALL engine review batches resolved: R20-R24 integrated and pushed (#198:
  R20/R22/R23 lineage; #199: S11 + R24). Every bot finding to date is either
  fixed with tests or evidence-rebutted; none outstanding.
- I2 storage-v2 core integrated (06a80bb -> #194): SQLite/WAL schema, 4-step
  crash protocol w/ per-step fault injection, reconciliation matrix, sample-
  domain-verified dry-run migration + rollback. Cutover UI remains.
- E29 Insights phase 1 integrated (369e806 -> #195): 287/287 desktop tests;
  E28 oracle semantics as TS + emitter + usage/quality views.
- Serving first wave S01/S02/S03/S11: implemented, CPU+Vulkan(iGPU)-verified,
  real-model smoke (parakeet q8/q4, moss q4) recorded. 5090 evidence run
  pending (user).
- Rate limiter: 4 agent deaths today, each recovered by coordinator takeover;
  dispatch concurrency held at 1-2.
- Next: I3 runtime crate (dispatched), then Android B07/B08 (gradle window),
  E02 cutover UI, E15/E12 packaging parity, S13/S14 quant track.

## Update 2026-09-20 (late night) — CI fully green, review loop at zero

- PR #195 ALL checks green (Desktop 3x OS, browser-and-api, build-cpu,
  engine guards, SONAR, android): lint gate satisfied (56 anti-slop errors
  fixed via schema-parse/SAFETY/named-type idioms, 4ffe6fd), fake-indexeddb
  declared at its package boundary, conditional engine-target CI wiring.
- PR #194 ALL green: X11/Wayland link libs, headless player-test guards,
  Schema-based G07 fixture parser, fmt-clean TS.
- Engine PRs #198/#199 green through review batch-4 (R25/R26 integrated:
  373/373 S11 lineage, 235/235 + budget PASS S02 lineage).
- Open review comments: ZERO across all PRs. Bots confirmed fixes in-thread.
- Process lessons recorded: pipeline-exit-code false greens (`cmd | tail;
  echo $?`) eliminated from coordinator verification; CI-exact invocations
  (pnpm run lint/fmt:check from repo root) replace scoped manual runs;
  temp-file staging in shared checkouts uses untracked names only (a real
  tracked test was nearly lost to a staging copy - restored, gates re-run).
- I3 runtime crate integrated (161+42+50). Lint-cleanup agent final state:
  0 warnings 0 errors.

## Update 2026-09-20 (final) — MERGES

Per user authorization: #194 -> gpui-port, #195 -> master, #193 -> master
(after resolving a stale-local-ref near-miss and one audio.test.ts conflict
caused by an accidental fmt sweep of master's copy; merged tree verified
161+42+50 before push). Master now carries the full GPUI port + program
wave A. Engine PRs #196/#197/#198/#199 deliberately held for the user's
5090 agent (CUDA gate + merge authority): program/gpu-validation refreshed
with AGENT-5090.md + VALIDATION.md. R28 landed on all four engine branches
(conditional CI wiring).
