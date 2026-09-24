# Mode routing, staging and processing contract

Schemas are the structural half; the executable oracles in `tests/` are the
semantic half. Both the Python tests and the Rust ports read the fixtures in
this directory in place.

| File | Role | Oracle |
|---|---|---|
| `mode.schema.json`, `profiles.schema.json` | One typed mode entry; a profiles document (default + scoped rules). | `tests/mode_routing.py` (`resolve`, `validate_config`, `validate_processing`) |
| `decision.schema.json` | One routing decision per take. | `decision_from_resolve` |
| `context-snapshot.schema.json` | The frozen activation-time context. | `selection_decision`, `snapshot_expired` |
| `draft.schema.json` | A staged take: typed regions over one text, immutable raw attempts, proposals pinned to a revision, deliveries. | `tests/staging.py` |
| `provider.schema.json` | What a processing provider can do and where text goes. | `validate_provider`, `processing_route` |
| `transform-request.schema.json`, `transform-result.schema.json` | One model step and its answer. | `check_request`, `check_result` |

Fixtures: `fixtures/staging.json` (the #293 scenarios: revised partials,
typing during a partial replacement, stale proposal, concurrent retry, delete
during processing, crash/reconnect, duplicate final delivery, and more),
`fixtures/processing-routes.json`, `fixtures/providers.json`,
`fixtures/transform-requests.json`, `fixtures/transform-results.json`, plus
the routing fixtures.

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

Run: `uv run --extra dev python -m pytest tests/test_mode_routing.py tests/test_staging.py -q`
and `cargo test -p starling-processing` in `apps/desktop-gpui`.
