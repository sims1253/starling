# Runtime protocol contract (E17-I0)

This is the **executable contract** for the E17 native runtime's wire protocol:
the versioned envelope, the v1 command/event sets of the five state machines,
conformance fixtures, and the semantic rules an implementation must reproduce.
It is frozen contract data plus a stdlib oracle — **no runtime implementation
exists yet**. Increment I3 (`starling-runtime` crate, Mode A) and I4 (service
host + IPC) implement against this; nothing here should be read as describing
a running service.

Design source: `docs/program/design/e17-native-runtime.md` §2 (envelope + the
five machines). The schemas are structural; `tests/runtime_protocol.py` is the
executable oracle; `tests/test_runtime_protocol.py` replays every fixture
through it.

## Files

| File | Role |
|---|---|
| `envelope.schema.json` | The wire envelope every message shares (structure, version, NACK + seq rules in `$comment`). |
| `commands.schema.json` | All 17 v1 commands, payload discriminated by `type`, `additionalProperties: false` everywhere. |
| `events.schema.json` | All 23 v1 events (22 machine events + `runtime.nack`), same discipline. |
| `fixtures/*.json` | Valid command→event traces, one set per machine, covering every state and transition named in §2. |
| `fixtures/invalid/*.json` | Invalid cases: unknown version, non-monotonic `seq`, command/event illegal in the current state, unfrozen route. |

Run the conformance suite with:

```
uv run python -m pytest tests/test_runtime_protocol.py -q
```

## Envelope

```json
{ "v": 1, "id": "cmd_3f9a", "ts": "2026-09-20T10:00:00Z", "corr": "take_77",
  "seq": 412, "type": "capture.stop", "payload": { "drain": true } }
```

- Required: `v`, `id`, `ts`, `type`, `payload`. Optional: `corr`, `seq`.
  `payload` is an object on every message, `{}` when the type has no fields.
- **Versioning**: `v` names the payload grammar. A receiver that cannot parse
  `v` must not apply the payload; it answers with the event
  `runtime.nack {reason: "unsupported_version"}` carrying `corr` = the rejected
  message's `id`. For a v1 validator any `v != 1` is a schema failure.
- **Ordering**: `seq` is strictly increasing per stream. A stream is the
  correlation channel named by `corr` (for example a take); messages without
  `corr` belong to their direction stream (`__commands__` / `__events__`).
  Duplicates and reorders must be surfaced, never silently absorbed.
- **Correlation**: a command that awaits an outcome event (e.g. `jobs.submit`
  → `jobs.queued` | `jobs.rejected`) and the event that resolves it carry the
  same `corr`; the oracle enforces this.
- `type` is `machine.message` (`^[a-z][a-z0-9]*\.[a-z][A-Za-z0-9]*$`); unknown
  types are invalid, not ignored.

## The five machines

Common rules (§2): each machine is a single owned actor/task; the UI is a
projection (commands in, events + snapshots out, never shared mutable state);
queues are bounded with explicit rejection events. In the tables below,
"internal" edges carry **no wire message** — they are runtime-owned progress
(scheduler dispatch, expiry timers, journal recovery). Fixtures step over them
explicitly with `{"$advance": "<state>"}` directives so every named transition
is exercised without inventing wire events §2 does not define.

### capture (writer task + actor)

| State | Meaning |
|---|---|
| `Idle` | no take |
| `Acquiring` | device open / route selection (start latency measurable separately) |
| `Recording` | samples flowing, only `Recording`/`Draining` accept input |
| `Draining` | final sample index declared, writer draining ring, fsync pending |
| `Persisted` | committed storage v2 capture row referenced |
| `Interrupted{reason, ackIndex}` | device error / storage fault / process death found on recovery |
| `Recovering` | journal replay |

Transitions: `capture.start` from `Idle|Persisted` → `Acquiring`;
`capture.started` → `Recording`; `capture.stop` (from `Recording`) →
`Draining`; `capture.stopped` (from `Draining|Recovering`) → `Persisted`;
`capture.abort` from `Acquiring|Recording|Draining` → `Idle` (no v1 event);
`capture.error{fatal:true}` from any active state → `Interrupted` (non-fatal
errors surface without a transition); internal: `Interrupted → Recovering` on
restart. `capture.stopped` carries `sampleDurationMs` (acknowledged samples /
actual rate) separately from `wallClockMs` (includes start latency + drain
wait). **Invariant: no `jobs.*` message participates in any capture exit —
Draining never waits on inference (test-enforced).**

### jobs (scheduler; supervised workers)

Per job: `Queued → Dispatched → Loading → Recognizing → Completed | Failed |
Cancelled` for a recognition job, `Queued → Dispatched → Loading →
Transforming → Completed | Failed | Cancelled` for a transform job (#294),
plus `Rejected` at admission (queue full, resource limits, duplicate
submission) and the scheduler's pre-job `Idle`.

`jobs.submit{captureRef, route, budget}` and `jobs.transform{request}` are
each answered by exactly one of `jobs.queued` (→ `Queued`) or
`jobs.rejected{reason}` (→ `Rejected`) on the same `corr`.
`Dispatched/Loading/Recognizing/Transforming` are internal;
`jobs.progress{partial, stabilityHint}` is legal from
`Loading|Recognizing|Transforming` (stability reported independently of
recording completeness; for a transform job the partial is streamed output);
`jobs.completed` from `Recognizing` only and always carries raw recognition
text; `jobs.transformed{result}` from `Transforming` only, carrying a
completed transform result (a proposal for the take's draft, never a rewrite
of the recognition); `jobs.failed{reason, retryable}` demotes from any active
state (a crashed worker never touches capture or history; a failed provider
call reports the result's typed failure reason); `jobs.cancel{jobId}` from
any active state → `Cancelled` (no v1 event). `jobs.setLimits` (bounded
queue, max concurrent, per-route caps) is legal in every state. Job identity
in v1 travels on the submit/queued `corr`. The transform request is the
mode-routing `transform-request.schema.json` record, embedded verbatim in
`commands.schema.json` (and the result in `events.schema.json`); the tests
assert the copies cannot drift.

### context / mode (context service)

`Observing → SnapshotTaken → ModeDecided → RouteFrozen → Expired | Released`.

`context.snapshot{source}` resolves via `context.targetSnapshot{descriptor,
digest, capabilities, selectionRange, offsetEncoding, expiry}`; `mode.set`
(manual) resolves via `mode.decision{modeId, source, matchedPrefixSpan,
explanation, payloadView}`; `mode.routeFrozen{route, decidedAt}` is legal only
from `ModeDecided` and is what freezes the audio route before the first frame
can leave the runtime — a later spoken phrase cannot un-send audio. Internal:
snapshot/mode expiry (`SnapshotTaken|ModeDecided → Expired`), take completion
(`RouteFrozen → Released`), next cycle (`Expired|Released → Observing`).
`context.expire` is the explicit command form. Snapshots are short-lived with
explicit expiry; mode payloads select among already-authorized sources and can
never elevate permissions or change routes. (§2.3 says `routeFrozen` "carries
the explanation surfaced in UI"; the §2 field list gives it `{route,
decidedAt}` — v1 keeps the listed fields, the explanation travels on
`mode.decision`.)

### docs / revisions (document service)

Per head update: `Validating → Committed | Conflicted`, between updates
`Steady`. Documents carry `{docId, name, headRevision, turnSeq}`; revisions
are immutable `{revId, baseRevision, sourceAttemptIds[],
instructionTemplateId, text, status, provenance}`.

`docs.updateHead{docId, expectedBase, newRevision}` is a compare-and-swap: base
mismatch answers `docs.headConflict{expected, actual, candidatePreserved}`
and the candidate revision is retained for explicit user choice (`Conflicted`
persists until the user acts); a match answers
`docs.headUpdated{docId, headRevision}` → `Committed` (internal → `Steady`).
`docs.appendTurn{docId, takeRef}` resolves via `docs.turnAppended{turnSeq}`;
turn order is the explicit `turnSeq`, never `createdAt`. `docs.get{docId,
page}` is legal whenever no update is in flight (page 0 is the first page) and
defines no v1 event (served via snapshot/correlation response).

### delivery (delivery service)

`Prepared → Revalidating → SubmittedUnconfirmed → Confirmed | Failed{reason}
| Conflict{targetChanged} | Cancelled`, plus the pre-delivery `Idle`.

`delivery.prepare{revisionId, targetRef}` resolves via
`delivery.prepared{deliveryId, compareToken}`; `delivery.apply{deliveryId}`
revalidates target identity/range/version immediately before apply
(`Prepared → Revalidating`), then `delivery.submittedUnconfirmed`, then
`delivery.confirmed{evidenceLevel}` — synthetic key acceptance is not proof
text landed, so the evidence level is stated. `delivery.conflict{
expectedTarget, actualTarget}` when the target changed since freeze.
`delivery.cancel` (from `Prepared|Revalidating|SubmittedUnconfirmed`) →
`Cancelled`. `delivery.copyFallback{deliveryId}` is legal from
`Failed|Conflict` and closes nothing in v1. **Invariant: no Enter injection —
there is no command that sends, executes, auto-confirms or presses anything;
`apply` is user-initiated and names its delivery.** Correlation/delivery IDs
prevent duplicate insertion from duplicate final callbacks.

## What I3 (Rust runtime crate, Mode A) must satisfy

- Implement the five machines so that these fixtures replay green in-process
  (E17 AC: independent test fixtures): same states, same
  command→allowed-from→event tables, same internal edges, same NACK and seq
  rules. The oracle in `tests/runtime_protocol.py` is the reference semantics.
- Speak the envelope as the only internal API: unknown `v` → `runtime.nack`,
  payload never applied; per-stream strictly increasing `seq`; bounded queues
  with `jobs.rejected` instead of unbounded waiting.
- Hold the tested invariants: Draining never waits on jobs; docs CAS retains
  candidates; delivery never auto-sends; route freezes before audio leaves.

## What I4 (IPC host) must satisfy

- Transport the envelope bytes unchanged over authenticated OS-local IPC
  (UDS / named pipe); the same fixtures replay green over IPC — the wire is a
  transport swap, not a new API.
- Enforce envelope-level rules at the connection boundary: reject unknown `v`
  with `runtime.nack{unsupported_version}` (corr = rejected id), enforce seq
  monotonicity per stream, rate/size limits and stale-owner leases per §1/§4
  of the design note (out of scope for the envelope itself).

## Interpretations beyond the design text

The design note names commands/events and their fields; where it stopped
short, v1 freezes the following choices (each is test-visible, none expands
§2's sets):

1. `capture.stop` keeps the sketch's optional `drain` boolean; `capture.abort`
   and `jobs.cancel`/`delivery.cancel` define no v1 reply event — the state
   snapshot is the acknowledgement. `delivery.cancel` takes an optional
   `deliveryId` (§2 lists it bare).
2. Bookkeeping states not named in §2: jobs `Idle` (pre-admission), docs
   `Steady` (between head updates), delivery `Idle` (before prepare).
3. Outcome-pending commands (`jobs.submit`, `jobs.transform`, `context.snapshot`, `mode.set`,
   `docs.appendTurn`, `delivery.prepare`) resolve via exactly one correlated
   event; the event must carry the command's `corr`.
4. `jobs.queued`/`jobs.rejected` payloads are `{}` per §2; job identity
   travels on `corr`. `jobs.setLimits` requires `maxQueued` + `maxConcurrent`
   (per-route caps optional). `budget` is a named token from explicit config.
5. Unspecified value vocabularies are tokens, not free strings, and not
   enumerated in v1: `jobs.progress.stabilityHint`, `completionEvidence`,
   `delivery.confirmed.evidenceLevel`, revision `status`/`provenance`,
   `capture.error.code`. Only `jobs.rejected.reason` (queue_full,
   resource_limits, duplicate_submission) and `mode.decision.source` (manual,
   phrase, rule) are enumerated — §2 names their values.
6. `docs.headConflict.candidatePreserved` is a boolean (the retained revision
   is the `newRevision` of the failed `docs.updateHead`).
7. The route-freeze-before-audio-leave invariant is executed as: every
   `jobs.submit` must reference a route frozen by an earlier
   `mode.routeFrozen` (audio-leave proxy — §2 defines no explicit
   audio-leave event; `jobs.submit` is the first message that can carry audio
   out of the runtime). Fixtures share one synthetic clock to make the
   ordering checkable across machines.
8. `{"$advance": ...}` fixture directives mark runtime-internal transitions
   (no wire message): jobs dispatch chain, context expiry/release, docs
   Committed→Steady, capture Interrupted→Recovering.
