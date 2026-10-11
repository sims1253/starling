# Mode routing, staging and processing contract

Schemas are the structural half; the reference implementations are the
semantic half. The Rust crate `starling-processing`
(`apps/desktop-gpui/crates/processing`) and the Kotlin ports in the Android
app (`apps/mobile/app/src/main/java/dev/starling/mobile/processing/`) read the
fixtures in this directory in place. The original Python oracles
(`tests/mode_routing.py`, `tests/staging.py`, `tests/spoken_commands.py`,
`tests/spoken_instructions.py`) are at tag `python-research-final`; schema
`description`s that cite them refer to that tag.

| File | Role | Implemented in (Rust under `apps/desktop-gpui/crates/`; Kotlin under `processing/`) |
|---|---|---|
| `mode.schema.json`, `profiles.schema.json` | One typed mode entry; a profiles document (default + scoped rules). | `processing/src/routing.rs` (`resolve`, `validate_config`), `processing/src/contract.rs` (`validate_processing`); `ModeRouting.kt` |
| `decision.schema.json` | One routing decision per take. | `processing/src/routing.rs`; `ModeRouting.kt` |
| `context-snapshot.schema.json` | The frozen activation-time context. | `runtime/src/machine/context.rs` |
| `draft.schema.json` | A staged take: typed regions over one text, immutable raw attempts, proposals pinned to a revision, deliveries. | `processing/src/staging.rs`; `StagedDraft.kt` |
| `provider.schema.json` | What a processing provider can do and where text goes. | `processing/src/contract.rs` (`validate_provider`, `processing_route`) |
| `transform-request.schema.json`, `transform-result.schema.json` | One model step and its answer. | `processing/src/contract.rs` |
| `spoken-instruction.schema.json` | One trailing spoken-instruction record per finalized take: the delimiter/payload split with code-point spans. | `processing/src/instructions.rs`; `SpokenInstructions.kt` |

Fixtures: `fixtures/staging.json` (the #293 scenarios: revised partials,
typing during a partial replacement, stale proposal, concurrent retry, delete
during processing, crash/reconnect, duplicate final delivery, and more),
`fixtures/processing-routes.json`, `fixtures/providers.json`,
`fixtures/transform-requests.json`, `fixtures/transform-results.json`, plus
the routing fixtures. The spoken-command and spoken-instruction tables
(`spoken-commands.json`, `spoken-instructions.json`) and their fixture
corpora (`fixtures/spoken-commands.json`, `fixtures/spoken-instructions.json`)
are contract data the Rust ports (`transforms.rs`, `instructions.rs`) and
the Kotlin ports (`SpokenCommands.kt`, `SpokenInstructions.kt`) read in place.

Rules the processing side adds (#292 working rules):

- Raw recognition attempts are immutable. A transform result is a proposal
  tied to the revision its request read; it never changes the draft by
  itself, and a stale proposal is taken only by an explicit user choice.
- Transcript text is data: `input` is the text to transform, never
  instructions. Only an explicit trailing instruction travels as
  `instruction`, and only rewrite/translate may carry one.
- A mode names an authoring route, never a provider. `processing_route`
  picks at most one provider and never falls back; `local_only` never
  reaches a remote provider.
- Offsets are Unicode code points.

Run: `cargo test -p starling-processing` in `apps/desktop-gpui` and
`./gradlew :app:testDebugUnitTest` in `apps/mobile` (the `*Conformance*` tests).
