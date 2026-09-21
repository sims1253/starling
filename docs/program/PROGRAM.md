# Starling integrated implementation program

Coordinator control plane. Generated 2026-09-20 from the consolidated application
package (`starling-final-package` v3.0) and the serving addendum
(`starling-serving-optimization`), reconciled against the live repository.

## Baselines (verified 2026-09-20)

| Ref | Commit | State |
|---|---|---|
| master | `12b6548c96ecfa1c989abb7afd39e90080cfb92f` | checked out as worktree `starling/` (branch `program/wave-a` created from it) |
| PR #193 head | `4569e67268adeb65119aad9f5b83fbab134e0833` | OPEN, mergeable, branch `gpui-port`, worktree `starling-gpui/` |
| ggml submodule | `e91ded11bdcd78c42f9c8d3978ff6686eb4c1226` (v0.23.0) + uncommitted Starling patch series in `third_party/ggml` (alloc, cpu, cuda fattn/norm/pad, vulkan + `norm_affine.comp`) | preserved untouched |

Both packages pass their self-validation: reference behavior tests 26/26,
publisher tests 13/13 each, serving analysis checks 16+70 subtests. No GitHub
writes have been made from this session; per user instruction no PRs will be
filed upstream by agents. Local program branches carry integrated work.

## Outcomes

1. **Native desktop direction.** GPUI + renderer-independent native runtime,
   evolved from PR #193's `crates/dictation` foundation. Electron remains the
   migration reference. Tauri excluded. No second port commissioned.
2. **Three workflows:** Dictate (focus-safe system-wide), Edit (selection-based,
   conflict-checked), Capture (named documents, immutable revisions).
3. **Modes + context:** typed versioned mode engine, app/site/project rules,
   leading voice phrases ("code this"), permission-scoped context snapshots,
   Mode Studio.
4. **Insights:** local-first usage/voice/quality surface with idempotent
   versioned events and deletion-aware derived data.
5. **Serving/engine:** S01–S19 program — MOSS + Parakeet optimization,
   quantization provenance, honest benchmarking through the existing harness.
6. **Mobile/interop:** Android completion (E13/E22), iOS flow (E14),
   CLI/MCP sharing (E23), optional sync (E25).

## Architecture boundaries (target)

- `crates/dictation` (PR #193) → grows into the renderer-independent native
  runtime: capture ownership, durable store, documents/revisions, mode engine,
  delivery. UI (GPUI or Electron) is a client.
- Native runtime separates: capture state machine, inference job queue,
  context/mode decisions, document store, external delivery — each with
  versioned commands/events.
- Engine track works in `cpp/` (`moss/`, `parakeet/`, `serve/`, `runtime/`,
  `lib/`) on the existing replay/cache infrastructure; extends, never forks it.
- Storage: append-only audio + transactional metadata, crash-consistency to a
  documented durable sample boundary; migration from Electron IndexedDB and
  GPUI store with field preservation and rollback.

## Integration waves (from ROADMAP.md, adapted)

- **Wave A (now):** G01–G07 GPUI hardening; E17 native ownership extraction;
  E06 fidelity corpus; E10/E18/E19/E28 contracts; E15 GPUI screens vs fixtures;
  B01–B11 independent Electron/Android fixes; Cargo/GPUI CI (E12/E15/E16).
  Serving first wave: S01, S02, S11, S09, S03 (semantics separated).
- **Wave B:** E01/E02 native capture + durable history; E03/E04 platform
  input/overlay; E05 model setup; E09 endpoint/pairing; E15 parity; E21
  documents; verified migration. Serving: S04/S19, S05/S06/S07/S10 kernels.
- **Wave C:** E07/E08/E20/E26/E27/E30 authoring+context; E28/E29 Insights;
  E24 multilingual. Serving: S13/S14 quant, S15/S16 fusion+gating.
- **Wave D:** E13/E14/E22/E23/E25 mobile/interop. Serving: S12/S17/S18
  (after workload evidence).

Dependency edges are integration ordering, not serial staffing: contracts and
fixture-driven UI/tests proceed ahead of service completion.

## Release matrix proposal

Honest per-platform state, populated as evidence lands (see STATUS.md):
desktop Linux (X11/Wayland), macOS, Windows; Android IME/service/activity;
iOS host+keyboard; server CPU/Vulkan/CUDA/Metal. Claims require named-target
executed evidence; missing hardware is a recorded validation gap.

## Control plane files

- `tasks.json` — all 67 items (48 app + 19 serving), status vocabulary:
  pending / filed / in_progress / implemented / integrated / verified_on_target /
  review_ready / blocked / no_go / inconclusive / obsolete_duplicate / done.
- `DECISIONS.md` — consequential choices and rejected alternatives.
- `STATUS.md` — resumable state: integrated commits, active assignments,
  validation gaps, next actions, pending human decisions.
