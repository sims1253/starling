# E17 — Native runtime design note

Status: design for coordination, 2026-09-20. Baselines: master `12b6548`, PR #193 `4569e67`.
Inputs: E17/E01/E02/E03, ARCHITECTURE.md, PR193_REVIEW.md, G01/G02, tasks.json R01/R09,
`crates/dictation` at PR #193 (storage.rs, recorder.rs, client.rs). Zero code changed here.

## 1. Target architecture

**Decision: build the runtime as an in-process library first (`crates/starling-runtime`), then
wrap it in a thin user-scoped service host behind the same API. The production end-state is the
user-scoped process from E17/ARCHITECTURE.md; the library/host split is sequencing, not a
compromise.**

Rationale:
- Every state machine, queue bound and command/event contract must exist before a process
  boundary means anything. IPC over an ad-hoc internal API is a rewrite; IPC over the versioned
  envelope is a transport swap.
- G01/G02 capture and storage hardening are urgent and process-independent: durability comes
  from removing UI-lifetime ownership (writer task + journal), not from a separate process.
  Shipping them inside the GPUI app immediately unblocks E01/E02.
- The library embeds directly in unit/conformance tests without sockets; the same fixtures then
  run against the host over IPC (E17 AC: "independent test fixtures").

Hosting modes share one contract:
- **Mode A (interim, shippable)**: GPUI links `starling-runtime`; runtime owns threads/tasks
  (capture writer, jobs, storage); UI holds a `RuntimeClient` speaking the envelope over an
  in-process channel. Renderer kill still loses the *process*, but acknowledged journal samples
  survive on disk and recover (E02 recovery contract).
- **Mode B (target)**: `starling-runtime-host` binary, one per user session; OS-local
  authenticated IPC (UDS on Linux/macOS, named pipe with ACL on Windows); stale-owner
  detection, message size/rate limits; GPUI and the Electron comparison adapter are both just
  clients. Supervised worker processes for the C++ engine attach to the runtime, not the UI.

What would force full Mode B earlier: (a) the Electron adapter must drive the same live runtime
concurrently with GPUI; (b) renderer-crash isolation becomes a shipped acceptance criterion
rather than journal-recovery; (c) mobile/desktop share one runtime over local sockets.
What would delay Mode B: metering/waveform latency over IPC (mitigated by snapshot streaming,
not request/response) and Windows pipe-ACL friction — neither blocks the contract.

Code disposition (PR #193 `crates/dictation`):
- Reuse as-is: `client.rs` (protocol enum, endpoint validation, bounded timeouts, error
  taxonomy, tested parsers — becomes the provider adapter behind the job executor);
  `Attenuator`, `ClipCounters` (G03-correct pre-DSP evidence), downmix/conversion helpers,
  WAV validation, `MemorySessionStore` test twin, `SessionStatus`/manifest field model.
- Reuse behind new owners: CPAL stream construction (`pick_input_config`, `open_stream`) —
  callback body replaced per §3; `FileSessionStore` becomes a read-only migration source.
- Discard/replace: `Shared` mutex plumbing (R01), unbounded `mpsc` chunk channel (G01),
  `stop()`-drains-everything lifecycle (R09/G01), eager `Arc<Vec<u8>>` WAV loading,
  `write_atomic` fixed `.tmp` name, list-aborts-on-first-error (G02).

## 2. Five state machines

Common rules: each machine is a single owned actor/task in the runtime (capture writer,
job scheduler, context service, document service, delivery service). The UI is a projection:
it sends commands and receives events + snapshots, never shared mutable state. All queues are
bounded with explicit rejection events; ordering is per-stream monotonic `seq`.

Envelope (sketch):

```rust
enum Command { V1(CommandV1) }   // unknown variant -> Nack{unsupported_version}
enum Event   { V1(EventV1) }
struct Envelope<T> { v: u8, id: MsgId, ts: Utc, corr: Option<Id>, seq: Option<u64>, payload: T }
```

```json
{ "v": 1, "id": "cmd_3f9a", "ts": "2026-09-20T10:00:00Z", "corr": "take_77",
  "seq": 412, "type": "capture.stop", "payload": { "drain": true } }
{ "v": 1, "id": "evt_8c21", "corr": "take_77", "seq": 413,
  "type": "capture.stopped", "payload": { "finalSampleIndex": 128512,
  "acknowledgedSamples": 128000, "gaps": [], "journalId": "j_02ab",
  "sampleDurationMs": 8032, "wallClockMs": 8410 } }
```

### 2.1 Capture (owner: capture actor + RT writer task)
States: `Idle → Acquiring → Recording → Draining → Persisted`, plus
`Interrupted{reason, ackIndex}` (device error, storage fault, process death found on recovery)
and `Recovering` (journal replay). `Acquiring` covers device open/route selection and makes
start latency measurable separately from sample duration.
Commands: `capture.start{policy}` | `capture.stop` | `capture.abort`.
Events: `capture.started{device, actualRate, channels}` | `capture.progress{ackSamples,
clipRatio, level}` (throttled) | `capture.gap{startSample, endSample}` |
`capture.error{code, fatal}` | `capture.stopped{finalSampleIndex, acknowledgedSamples, gaps}`.
Invariants: only `Recording`/`Draining` accept input; `Draining` has a declared final sample
index; `Persisted` references a committed storage v2 capture row.

### 2.2 Inference jobs (owner: job scheduler; workers supervised)
States: `Queued → Dispatched → Loading → Recognizing → [Transforming] → Completed | Failed |
Cancelled`, plus `Rejected` at admission (queue full, resource limits, duplicate submission).
Commands: `jobs.submit{captureRef, route, budget}` | `jobs.cancel{jobId}` | `jobs.setLimits`.
Events: `jobs.queued | jobs.rejected{reason} | jobs.progress{partial, stabilityHint} |
jobs.completed{attemptId, text, backend, timing, completionEvidence} | jobs.failed{reason,
retryable}`.
Invariants: capture never waits in `Draining` for a job (E17 AC); bounded queue + max
concurrent + per-route limits are explicit config; a crashed worker demotes its jobs to
`Failed{retryable}` without touching capture or history; partial-transcript stability is
reported independently of recording completeness.

### 2.3 Context / mode decisions (owner: context service)
States: `Observing → SnapshotTaken{target, selection, grants} → ModeDecided{mode, routeIntent}
→ RouteFrozen{providers, policy} → Expired | Released`.
Commands: `context.snapshot{source}` | `mode.set{mode, source: manual}` | `context.expire`.
Events: `context.targetSnapshot{descriptor, digest, capabilities, selectionRange, offsetEncoding,
expiry}` | `mode.decision{modeId, source: manual|phrase|rule, matchedPrefixSpan, explanation,
payloadView}` | `mode.routeFrozen{route, decidedAt}`.
Invariants: audio route freezes before the first frame can leave the runtime (a later spoken
phrase cannot un-send audio); context is data — mode payloads can select among
already-authorized sources but never elevate permissions or change routes; snapshots are
short-lived with explicit expiry; `routeFrozen` carries the explanation surfaced in UI.

### 2.4 Documents / revisions (owner: document service)
States (per head update): `Validating → Committed | Conflicted`; documents carry
`{docId, name, headRevision, turnSeq}`; revisions are immutable
`{revId, baseRevision, sourceAttemptIds[], instructionTemplateId, text, status, provenance}`.
Commands: `docs.updateHead{docId, expectedBase, newRevision}` | `docs.appendTurn{docId,
takeRef}` | `docs.get{docId, page}`.
Events: `docs.headUpdated{docId, headRevision}` | `docs.headConflict{expected, actual,
candidatePreserved}` | `docs.turnAppended{turnSeq}`.
Invariants: head update is compare-and-swap on `expectedBase` (E17 AC: verify base
identities); conflicts retain the candidate revision for explicit user choice; turn order is
the explicit `turnSeq`, never `createdAt`; re-recognition creates a new attempt against the
same capture without mutating capture timestamps or erasing prior results.

### 2.5 External delivery (owner: delivery service)
States: `Prepared{revisionRef, frozenTarget, policy} → Revalidating → SubmittedUnconfirmed →
Confirmed | Failed{reason} | Conflict{targetChanged} | Cancelled`.
Commands: `delivery.prepare{revisionId, targetRef}` | `delivery.apply{deliveryId}` |
`delivery.cancel` | `delivery.copyFallback{deliveryId}`.
Events: `delivery.prepared{deliveryId, compareToken}` | `delivery.submittedUnconfirmed` |
`delivery.confirmed{evidenceLevel}` | `delivery.failed{reason, fallbackSuggested}` |
`delivery.conflict{expectedTarget, actualTarget}`.
Invariants: target identity/range/version revalidated immediately before apply; no Enter
injection (cannot auto-send/execute); correlation/delivery IDs prevent duplicate insertion
from duplicate final callbacks; `confirmed` states its evidence level (synthetic key
acceptance is not proof text landed); undo metadata records what Starling inserted so Undo
revalidates instead of blind Ctrl+Z; best-effort paths are labeled as such.

## 3. Capture lifecycle hardening contract (G01, R01, R09)

Fixes, mapped to defects:
- **R01 (mutexes on RT thread)**: the audio callback touches no `Mutex` and no channel. It
  writes converted mono f32 into a preallocated single-producer/single-consumer ring
  (fixed capacity, e.g. 1–2 s at device rate) plus lock-free atomics: `written_seq`,
  `callback_alive`, clip/peak counters (pre-DSP, per G03). Downmix/conversion/attenuator state
  live in callback-local storage; no allocation after stream start.
- **G01 (unbounded channel, memory-only until Stop)**: a dedicated writer task owns the
  journal. It drains the ring continuously, appends frames to the per-take audio journal with
  sequence numbers, and periodically fsyncs (cadence: every ≤250 ms or ≤64 KiB, tunable) then
  publishes the durable boundary record and emits `capture.progress{ackSamples}`. Only
  fsynced samples are "acknowledged". Overflow policy: a full ring overwrites oldest (keeps
  capture live, bounds memory); the writer detects the sequence discontinuity and emits
  `capture.gap{start, end}` — gaps are flagged, never silently joined or fabricated (E02).
- **R09 (poisoned mutex silent drain, no quiesce)**: with the mutexes gone the failure mode is
  removed structurally; `stop` becomes an explicit handshake. `capture.stop` → actor sets
  `stopping`, declares `finalSampleIndex = ring.written_seq`, drops the CPAL stream and waits
  for `callback_alive == false` (callback quiesce), then the writer drains remaining ring
  entries, fsyncs, finalizes the journal trailer (length + content hash), publishes the
  metadata transaction (§4) and emits `capture.stopped`. A timeout on quiesce degrades to
  `capture.error{quiesce_timeout}` with acknowledged samples intact — never a silent empty
  result.
- **Error surfacing (E01/G01)**: the CPAL error callback posts a typed message to the actor →
  `capture.error` event + transition to `Interrupted`; the UI timer cannot keep pretending
  capture is live. No `eprintln` anywhere on the capture path.
- **Stop vs wall clock**: `capture.stopped` reports `sampleDurationMs` (derived from
  acknowledged samples / actual rate) separately from `wallClockMs` (includes start latency
  and drain wait); UIs display them distinctly (G01 AC5).
- **Close/restart policy**: on UI close with a live take, Mode A persists what is
  acknowledged and marks the take `interrupted` with a known tail boundary; nothing deletes
  the only source. In Mode B the runtime keeps recording; a reattached renderer reconstructs
  state and error history from snapshots + event replay (E17 AC1).

Acceptance hooks: kill renderer mid-capture → recover all acknowledged frames with labeled
tail; inject device error, sustained storage stall (expect gap events, live capture, bounded
memory), disk-full (capture degrades to `Interrupted`, source preserved), quiesce timeout.

## 4. Storage v2

Layout per user data root (single root, replacing the two divergent ones after migration):
`audio/<captureId>.sj` append-only sample journals (raw device-rate PCM + periodic boundary
records + final trailer), `starling.db` SQLite (WAL) transactional metadata, `staging/`,
`quarantine/`, `leases/`.

Schema direction (tables, essential columns):
- `captures(id, created_utc, tz, device, actual_rate, policy, frame_count, ack_sample_index,
  journal_hash, status, retention_class, extra_json)`
- `recognition_attempts(id, capture_id, backend, model_hash, language, options_json, text,
  partial_or_final, status, timing_json, extra_json)`
- `context_snapshots(id, capture_id, descriptor_digest, capabilities, selection_json,
  expiry_utc)` — short-lived; full-context replay is opt-in and excluded from exports.
- `mode_decisions(id, capture_id, mode_id, mode_version, source, route_json, explanation)`
- `documents(doc_id, name, head_revision, turn_seq)` /
  `revisions(rev_id, doc_id, base_rev, sources_json, text, status, provenance, disposition)`
- `deliveries(delivery_id, revision_id, target_json, compare_token, status, ack_level,
  undo_json, failure_json)`
- `tombstones(id, kind, deleted_utc, retention)`; `meta(key, value)` incl. schema version.
`extra_json` per row preserves unknown/newer fields verbatim (never dropped, never rewritten
from a lower schema reader — higher versions are read-only).

Crash-consistency protocol (per take):
1. `staging/<id>/journal` created; frames + boundary records appended and fsynced on the §3
   cadence — acknowledged = boundary fsynced.
2. Finalize: trailer with length + hash; fsync file, fsync dir (POSIX); rename into
   `audio/`.
3. SQLite transaction: insert `captures` row referencing the finalized journal; commit;
   WAL checkpoint per policy. **The durable ack to the capture actor is emitted only after
   this commit.**
4. GC staging. Recovery reconciles both sides independently: journal tail past the last valid
   boundary → truncate + `gap` flag; finalized audio with no row → orphan; row without
   finalized audio → `interrupted`.

Damaged-record isolation (G02): `list` is metadata-only (never loads PCM), paged; each record
read returns a per-record `RecordReport`; corrupt manifest/JSON, missing audio, unsupported
version, or identity/path mismatch quarantines that record (flagged or moved to `quarantine/`,
recoverable, never auto-deleted) while good records stay visible. WAV-only GPUI directories
become recoverable takes with duration computed from verified bytes. File reads are bounded;
playback/export/transcription load audio lazily on demand.

Ownership: a runtime lease (`pid` + boot id + heartbeat in `leases/`) makes a second process a
client instead of a competitor; stale leases are breakable after heartbeat expiry — no more
"startup resets every transcribing row" across instances; the fixed `.tmp` name is replaced by
unique temporaries under the lease owner.

Migration (additive, versioned, reversible):
- Sources: Electron IndexedDB via the versioned export bridge (dry-run → verified import;
  compare counts + hashes, not row counts) and GPUI `FileSessionStore` v1 directories
  (manifest schema 1: keep `transcript_history`, `attempt_count`, `last_error`, ownership and
  quarantine semantics intact).
- Import copies; sources are left untouched until the user accepts the verified result, then
  kept as backup per retention. Rollback = discard imported rows; originals were never
  rewritten. Unknown fields land in `extra_json`. Both existing roots are preserved until
  their content is verified inside v2 (no silent merge or deletion).

## 5. Integration sequence (dependency order, independently shippable)

| Incr | Delivers | Consumes from this design | Size |
|---|---|---|---|
| I0 | `packages/contracts` v1 envelope + command/event enums + conformance fixtures | §2 envelope; machine command/event sets | M |
| I1 | G01/R01/R09 capture hardening inside existing GPUI app | §3 contract; writer/journal/boundary pieces of §4 | L |
| I2 | G02 isolation + orphan recovery on current file store; storage v2 core (SQLite + journals + crash protocol) + GPUI-store migration | §4 | L |
| I3 | `starling-runtime` crate, Mode A: five machines behind the envelope; job executor reusing `client.rs`; admission/cancel; GPUI embeds it | §1 Mode A, §2 machines, §2.2 limits | L |
| I4 | Service host + authenticated UDS/pipe IPC; GPUI switches to IPC client; renderer-kill and reconnect acceptance; supervised C++ engine workers | §1 Mode B, §4 lease | L |
| I5 | Documents/revisions + context/mode + delivery machines wired to E03; Electron IndexedDB migration bridge | §2.3–2.5, §4 migration | L |

Order rationale: E01 consumes §3 via I1 without waiting for the runtime crate; E02 consumes §4
via I2; E03 needs I5 (targets frozen at capture start ride on I3's context service). Each
increment has its own acceptance evidence (kill/fault injection for I1/I2, conformance
fixtures replayed in-process for I3 and over IPC for I4).

## 6. Open questions (coordinator/user decisions needed)

1. Resampler: adopt an anti-aliased stateful library (e.g. rubato) vs porting master #122
   filter — requires measured spectral/latency evidence (G06). Blocks I1 finalization only.
2. IPC transport crate choice (raw std UnixListener/named pipes vs `interprocess`) and the
   per-OS client authentication model — needed by I4, not before.
3. SQLite bundled vs system lib, and WAL checkpoint policy on slow disks — I2.
4. Does the Electron comparison adapter drive the live runtime (forcing Mode B earlier), or
   compare exported results only until I5?
5. Secure-storage mechanism per OS for provider keys (runtime-held, session-only fallback) —
   I3/I5 boundary decision.
6. Journal boundary cadence defaults (250 ms / 64 KiB) need on-target measurement; tune after
   I1 fault-injection runs.
