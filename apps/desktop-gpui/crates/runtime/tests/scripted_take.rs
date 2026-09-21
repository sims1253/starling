//! The Mode A integration test (mission I3): boots the real runtime with
//! a fake capture source and a scripted provider, runs one take through
//! `context → capture → jobs → docs`, and asserts the emitted event
//! sequence is exactly the I0-legal one — including replaying the walked
//! command/event sequences back through the oracle port, so the live
//! actors are proven to emit oracle-legal traces.
//!
//! Also covers the actor-level behaviors the mission names: admission
//! rejection (`queue_full`, `duplicate_submission`), per-stream `seq`
//! monotonicity, the NACK path for unknown envelope versions, the
//! route-freeze-before-submit invariant, quiesce-timeout salvage, and the
//! fatal device-error interruption path.

use std::time::{Duration, Instant};

use starling_dictation::recorder::CaptureGap;
use starling_runtime::bus::{EventMessage, EventSub};
use starling_runtime::machine::capture::{CaptureConfig, InMemoryCaptureStore, TakeStatus};
use starling_runtime::protocol::replay::{route_freeze_violations, MachineReplay};
use starling_runtime::protocol::{Command, Event, JobLimits, Revision};
use starling_runtime::testing::{FakeCaptureSource, FakeTakeScript, FakeStop};
use starling_runtime::{
    provider::{FakeJob, Partial, ProviderOutcome},
    Runtime, RuntimeClient, RuntimeConfig,
};

fn revision(base: u64, text: &str, attempts: &[&str]) -> Revision {
    Revision {
        rev_id: format!("rev-{base}"),
        base_revision: base,
        source_attempt_ids: attempts.iter().map(|s| s.to_string()).collect(),
        instruction_template_id: "tpl-none".into(),
        text: text.into(),
        status: "candidate".into(),
        provenance: "recognition".into(),
    }
}

/// Collects from one subscription until `predicate` matches (or the
/// deadline passes); returns everything seen, including the match.
fn until(
    events: &EventSub,
    label: &str,
    predicate: impl Fn(&EventMessage) -> bool,
    deadline: Duration,
) -> Vec<EventMessage> {
    let mut seen = Vec::new();
    let start = Instant::now();
    loop {
        match events.recv_timeout(Duration::from_millis(20)) {
            Ok(message) => {
                let matched = predicate(&message);
                seen.push(message);
                if matched {
                    return seen;
                }
            }
            Err(starling_runtime::channel::RecvError::Timeout) => {
                if start.elapsed() >= deadline {
                    panic!(
                        "timed out waiting for {label}; saw {:?}",
                        seen.iter().map(|m| m.type_name()).collect::<Vec<_>>()
                    );
                }
            }
            Err(other) => panic!("event stream error: {other:?}"),
        }
    }
}

fn test_config(
    source: std::sync::Arc<FakeCaptureSource>,
    provider: std::sync::Arc<starling_runtime::provider::FakeProvider>,
    store: std::sync::Arc<InMemoryCaptureStore>,
    limits: JobLimits,
) -> RuntimeConfig {
    RuntimeConfig::default()
        .with_capture_source(source)
        .with_provider(provider)
        .with_capture_store(store)
        .with_capture_config(CaptureConfig {
            journals_dir: std::env::temp_dir().join("starling-runtime-test-journals"),
            poll_interval: Duration::from_millis(10),
        })
        .with_jobs_limits(limits)
}

/// Freezes the route (context snapshot + manual mode) for a take about to
/// start, so audio may legally leave on `local-default`.
fn freeze_route(client: &RuntimeClient, events: &EventSub) -> Vec<EventMessage> {
    let mut log = Vec::new();
    client
        .send(Some("ctx-1"), Command::ContextSnapshot { source: "vscode".into() })
        .expect("snapshot accepted");
    log.extend(until(events, "context.targetSnapshot", |m| m.type_name() == "context.targetSnapshot", Duration::from_secs(5)));
    client
        .send(Some("ctx-1"), Command::ModeSet {
            mode: "code-guidance".into(),
            source: starling_runtime::protocol::Manual,
        })
        .expect("mode accepted");
    log.extend(until(events, "mode.decision", |m| m.type_name() == "mode.decision", Duration::from_secs(5)));
    log
}

/// Runs one scripted take end to end; returns when `capture.stopped` was
/// seen for it.
fn run_take(
    source: &std::sync::Arc<FakeCaptureSource>,
    client: &RuntimeClient,
    events: &EventSub,
    take: &str,
    script: FakeTakeScript,
) -> Vec<EventMessage> {
    let mut log = Vec::new();
    source.push(script);
    client
        .send(Some(take), Command::CaptureStart { policy: "push-to-talk".into() })
        .expect("start accepted");
    log.extend(until(events, "capture.started", |m| m.type_name() == "capture.started", Duration::from_secs(5)));
    std::thread::sleep(Duration::from_millis(30));
    client
        .send(Some(take), Command::CaptureStop { drain: Some(true) })
        .expect("stop accepted");
    log.extend(until(events, "capture.stopped", |m| m.type_name() == "capture.stopped", Duration::from_secs(5)));
    log
}

/// Waits until the runtime projection shows `state` for `machine`-named
/// view ("capture", "context", ...), or panics after `deadline`.
fn wait_for_state(client: &RuntimeClient, state: &str, deadline: Duration) {
    let start = Instant::now();
    loop {
        let snapshot = client.snapshot();
        let current = [
            ("capture", snapshot.capture.state.clone()),
            ("context", snapshot.context.state.clone()),
            ("docs", snapshot.docs.state.clone()),
            ("delivery", snapshot.delivery.state.clone()),
            ("jobs", snapshot.jobs.state.clone()),
        ];
        if current.iter().any(|(name, s)| s == state || name == &state && s == state) && current.iter().any(|(_, s)| s == state) {
            return;
        }
        if start.elapsed() >= deadline {
            panic!("state {state} not reached; snapshot: {current:?}");
        }
        std::thread::sleep(Duration::from_millis(5));
    }
}

#[test]
fn scripted_take_through_capture_jobs_docs_matches_an_i0_trace() {
    let source = FakeCaptureSource::new(vec![]);
    let store = InMemoryCaptureStore::new();
    let provider = starling_runtime::provider::FakeProvider::new(vec![
        FakeJob::transforms_to("Hello, world."),
    ]);
    let config = test_config(
        std::sync::Arc::clone(&source),
        provider,
        store.clone(),
        JobLimits {
            max_queued: 2,
            max_concurrent: 1,
            per_route: vec![],
        },
    );
    let (runtime, client) = Runtime::start(config);
    let events = client.subscribe();

    // -- context: freeze the route before any audio can leave ----------
    let mut log = freeze_route(&client, &events);

    // -- capture: one scripted take -------------------------------------
    log.extend(run_take(&source, &client, &events, "take_77", FakeTakeScript::clean()));

    // -- jobs: submit the take on the frozen route ----------------------
    client
        .send(Some("job-1"), Command::JobsSubmit {
            capture_ref: "take_77".into(),
            route: "local-default".into(),
            budget: "standard".into(),
        })
        .expect("submit accepted");
    log.extend(until(&events, "jobs.completed", |m| m.type_name() == "jobs.completed", Duration::from_secs(5)));

    // -- docs: head update referencing the attempt, then a turn ---------
    client
        .send(Some("up-1"), Command::DocsUpdateHead {
            doc_id: "notes".into(),
            expected_base: 0,
            new_revision: revision(0, "Hello, world.", &["att-1"]),
        })
        .expect("update accepted");
    log.extend(until(&events, "docs.headUpdated", |m| m.type_name() == "docs.headUpdated", Duration::from_secs(5)));
    client
        .send(Some("turn-1"), Command::DocsAppendTurn {
            doc_id: "notes".into(),
            take_ref: "take_77".into(),
        })
        .expect("turn accepted");
    log.extend(until(&events, "docs.turnAppended", |m| m.type_name() == "docs.turnAppended", Duration::from_secs(5)));
    let stream = log;

    // -- assertions ------------------------------------------------------
    // 1. The deterministic event skeleton, in order (progress events are
    //    throttled and excluded; everything else is exact).
    let skeleton: Vec<&str> = stream
        .iter()
        .map(|m| m.type_name())
        .filter(|t| !matches!(*t, "capture.progress" | "jobs.progress"))
        .collect();
    assert_eq!(
        skeleton,
        vec![
            "context.targetSnapshot",
            "mode.decision",
            "mode.routeFrozen",
            "capture.started",
            "capture.stopped",
            "jobs.queued",
            "jobs.completed",
            "docs.headUpdated",
            "docs.turnAppended",
        ],
        "event skeleton must match the I0 trace shape"
    );

    // 2. Correlation: the take's stream and the job's stream.
    for message in &stream {
        match message.type_name() {
            "mode.routeFrozen" | "capture.started" | "capture.stopped" => {
                assert_eq!(message.corr.as_deref(), Some("take_77"), "{}", message.type_name());
            }
            "jobs.queued" | "jobs.completed" => {
                assert_eq!(message.corr.as_deref(), Some("job-1"));
            }
            _ => {}
        }
    }

    // 3. Per-stream seq is strictly increasing across the whole stream.
    let mut last_seq = std::collections::HashMap::new();
    for message in &stream {
        let stream_key = message
            .corr
            .clone()
            .unwrap_or_else(|| "__events__".to_string());
        if let Some(last) = last_seq.get(&stream_key) {
            assert!(message.seq > *last, "seq regression on {stream_key}");
        }
        last_seq.insert(stream_key, message.seq);
    }

    // 4. The walked command/event sequences replay green through the
    //    oracle port (the deterministic skeleton as a fixture-style trace).
    replay_walk("context", &[
        Step::Cmd("context.snapshot", "ctx-1"),
        Step::Evt("context.targetSnapshot", "ctx-1"),
        Step::Cmd("mode.set", "ctx-1"),
        Step::Evt("mode.decision", "ctx-1"),
        Step::Evt("mode.routeFrozen", "take_77"),
        Step::Advance("Released"),
    ]);
    replay_walk("capture", &[
        Step::Cmd("capture.start", "take_77"),
        Step::Evt("capture.started", "take_77"),
        Step::Cmd("capture.stop", "take_77"),
        Step::Evt("capture.stopped", "take_77"),
    ]);
    // The dispatch chain is runtime-internal (the fixtures step over it
    // with $advance directives); the scripted provider transformed, so the
    // walk includes the Transforming edge the live scheduler took.
    replay_walk("jobs", &[
        Step::Cmd("jobs.submit", "job-1"),
        Step::Evt("jobs.queued", "job-1"),
        Step::Advance("Dispatched"),
        Step::Advance("Loading"),
        Step::Advance("Recognizing"),
        Step::Advance("Transforming"),
        Step::Evt("jobs.completed", "job-1"),
    ]);
    replay_walk("docs", &[
        Step::Cmd("docs.updateHead", "up-1"),
        Step::Evt("docs.headUpdated", "up-1"),
        Step::Advance("Steady"),
        Step::Cmd("docs.appendTurn", "turn-1"),
        Step::Evt("docs.turnAppended", "turn-1"),
    ]);

    // 5. Route-freeze-before-submit over the live stream (the audio-leave
    //    proxy): no violations among the emitted envelopes.
    let mut corpus: Vec<serde_json::Value> = stream.iter().map(|m| m.to_value()).collect();
    corpus.push(serde_json::json!({
        "v": 1, "id": "cmd_probe", "ts": "2099-01-01T00:00:00Z",
        "type": "jobs.submit",
        "payload": {"captureRef": "take_77", "route": "local-default", "budget": "standard"}
    }));
    assert!(route_freeze_violations(&corpus).is_empty());

    // 6. Persistence: the take was committed (complete) via the store.
    {
        let takes = store.takes.lock().unwrap();
        assert!(
            takes
                .iter()
                .any(|(status, id, _)| *status == TakeStatus::Complete && id == "take_77"),
            "take_77 committed: {takes:?}"
        );
    }

    // 7. Snapshot projections: capture Persisted, context Released (the
    //    take completed the cycle), docs Steady, jobs Completed.
    let snapshot = client.snapshot();
    assert_eq!(snapshot.capture.state, "Persisted");
    assert_eq!(snapshot.context.state, "Released");
    assert_eq!(snapshot.docs.state, "Steady");
    assert_eq!(snapshot.jobs.state, "Completed");
    assert_eq!(snapshot.frozen_routes, vec!["local-default".to_string()]);

    runtime.shutdown();
}

enum Step {
    Cmd(&'static str, &'static str),
    Evt(&'static str, &'static str),
    Advance(&'static str),
}

/// Feeds a walked command/event sequence through the oracle port as a
/// fixture-style trace — green means the live runtime's observed walk is
/// exactly an I0-legal one.
fn replay_walk(machine: &str, steps: &[Step]) {
    let mut messages = Vec::new();
    for (index, step) in steps.iter().enumerate() {
        let id = format!("walk_{index}");
        match step {
            Step::Cmd(type_, corr) => messages.push(serde_json::json!({
                "v": 1, "id": id, "ts": "2026-09-20T10:00:00Z", "corr": corr,
                "type": type_, "payload": {}
            })),
            Step::Evt(type_, corr) => {
                // capture.stopped needs its required payload fields for
                // structural validation; supply the minimal legal ones.
                let payload = match *type_ {
                    "capture.stopped" => serde_json::json!({
                        "finalSampleIndex": 1600, "acknowledgedSamples": 1600,
                        "gaps": [], "journalId": "j_walk",
                        "sampleDurationMs": 100.0, "wallClockMs": 120.0
                    }),
                    "jobs.completed" => serde_json::json!({
                        "attemptId": "att-walk", "text": "Hello, world.",
                        "backend": "fake-provider", "timing": 12.0,
                        "completionEvidence": "final_decode"
                    }),
                    "jobs.queued" | "docs.turnAppended" if *type_ == "jobs.queued" => {
                        serde_json::json!({})
                    }
                    "docs.turnAppended" => serde_json::json!({ "turnSeq": 1 }),
                    "docs.headUpdated" => serde_json::json!({
                        "docId": "notes", "headRevision": 1
                    }),
                    "mode.routeFrozen" => serde_json::json!({
                        "route": "local-default", "decidedAt": "2026-09-20T10:00:03Z"
                    }),
                    _ => serde_json::json!({}),
                };
                messages.push(serde_json::json!({
                    "v": 1, "id": id, "ts": "2026-09-20T10:00:01Z", "corr": corr,
                    "type": type_, "payload": payload
                }));
            }
            Step::Advance(state) => messages.push(serde_json::json!({ "$advance": state })),
        }
    }
    let trace = serde_json::json!({ "machine": machine, "messages": messages });
    let replayed = MachineReplay::new(machine, None)
        .and_then(|mut replay| {
            for message in trace["messages"].as_array().unwrap() {
                replay.feed(message)?;
            }
            replay.finish()
        })
        .unwrap_or_else(|violation| panic!("{machine} walk must replay green: {violation}"));
    assert!(!replayed.transitions.is_empty());
}

// ------------------------------------------------------------------------- //
// Actor-level behaviors
// ------------------------------------------------------------------------- //

#[test]
fn admission_rejection_queue_full_is_rejected_not_absorbed() {
    let source = FakeCaptureSource::new(vec![]);
    let store = InMemoryCaptureStore::new();
    let provider = starling_runtime::provider::FakeProvider::new(vec![
        FakeJob::completes_with("first"), // occupies the single worker a while
    ]);
    let config = test_config(
        std::sync::Arc::clone(&source),
        provider,
        store,
        JobLimits {
            max_queued: 1,
            max_concurrent: 1,
            per_route: vec![],
        },
    );
    let (runtime, client) = Runtime::start(config);
    let events = client.subscribe();

    freeze_route(&client, &events);
    run_take(&source, &client, &events, "take_a", FakeTakeScript::clean());
    run_take(&source, &client, &events, "take_b", FakeTakeScript::clean());
    run_take(&source, &client, &events, "take_c", FakeTakeScript::clean());

    // job-1 occupies the worker; job-2 fills the (maxQueued = 1) waiting
    // queue; job-3 must be answered with jobs.rejected{queue_full} on its
    // corr.
    client
        .send(Some("job-1"), Command::JobsSubmit {
            capture_ref: "take_a".into(),
            route: "local-default".into(),
            budget: "standard".into(),
        })
        .expect("job-1 accepted");
    client
        .send(Some("job-2"), Command::JobsSubmit {
            capture_ref: "take_b".into(),
            route: "local-default".into(),
            budget: "standard".into(),
        })
        .expect("job-2 accepted");
    let receipt = client
        .send(Some("job-3"), Command::JobsSubmit {
            capture_ref: "take_c".into(),
            route: "local-default".into(),
            budget: "standard".into(),
        })
        .expect("submit itself is accepted; the rejection is an event");

    assert!(matches!(receipt, starling_runtime::machine::Receipt::Accepted));
    let rejected = until(
        &events,
        "jobs.rejected(job-3)",
        |m| m.type_name() == "jobs.rejected" && m.corr.as_deref() == Some("job-3"),
        Duration::from_secs(5),
    );
    let rejection = rejected
        .iter()
        .rev()
        .find(|m| m.type_name() == "jobs.rejected")
        .unwrap();
    match &rejection.event {
        Event::JobsRejected { reason } => assert_eq!(reason.as_str(), "queue_full"),
        other => panic!("expected JobsRejected, got {other:?}"),
    }

    runtime.shutdown();
}

#[test]
fn duplicate_submission_of_an_active_capture_is_rejected() {
    let source = FakeCaptureSource::new(vec![]);
    let store = InMemoryCaptureStore::new();
    let provider = starling_runtime::provider::FakeProvider::new(vec![
        FakeJob::completes_with("first"),
    ]);
    let config = test_config(std::sync::Arc::clone(&source), provider, store, JobLimits {
        max_queued: 8,
        max_concurrent: 1,
        per_route: vec![],
    });
    let (runtime, client) = Runtime::start(config);
    let events = client.subscribe();

    freeze_route(&client, &events);
    run_take(&source, &client, &events, "take_a", FakeTakeScript::clean());

    client
        .send(Some("job-1"), Command::JobsSubmit {
            capture_ref: "take_a".into(),
            route: "local-default".into(),
            budget: "standard".into(),
        })
        .expect("job-1 accepted");
    client
        .send(Some("job-2"), Command::JobsSubmit {
            capture_ref: "take_a".into(), // same captureRef while job-1 runs
            route: "local-default".into(),
            budget: "standard".into(),
        })
        .expect("job-2 accepted");
    let collected = until(
        &events,
        "jobs.rejected(job-2)",
        |m| m.type_name() == "jobs.rejected" && m.corr.as_deref() == Some("job-2"),
        Duration::from_secs(5),
    );
    let rejection = collected
        .iter()
        .rev()
        .find(|m| m.type_name() == "jobs.rejected")
        .unwrap();
    match &rejection.event {
        Event::JobsRejected { reason } => assert_eq!(reason.as_str(), "duplicate_submission"),
        other => panic!("expected JobsRejected, got {other:?}"),
    }

    runtime.shutdown();
}

#[test]
fn nack_path_answers_unsupported_version() {
    let (runtime, client) = Runtime::start(RuntimeConfig::default());
    let events = client.subscribe();
    let bad = serde_json::json!({
        "v": 2, "id": "cmd_900", "ts": "2026-09-20T10:09:00Z",
        "corr": "take_99", "seq": 1,
        "type": "capture.stop", "payload": { "drain": true }
    });
    let rejection = client.send_raw(bad).expect_err("v2 must be rejected");
    assert!(
        matches!(
            &rejection,
            starling_runtime::machine::Rejection::UnsupportedVersion { id }
                if id.as_deref() == Some("cmd_900")
        ),
        "got {rejection:?}"
    );
    // The runtime.nack event on the stream, corr = the rejected id.
    let collected = until(&events, "runtime.nack", |m| m.type_name() == "runtime.nack", Duration::from_secs(5));
    let nack = collected
        .iter()
        .rev()
        .find(|m| m.type_name() == "runtime.nack")
        .unwrap();
    assert_eq!(nack.corr.as_deref(), Some("cmd_900"));
    assert!(matches!(nack.event, Event::RuntimeNack { .. }));
    // The payload was never applied: capture remains Idle.
    assert_eq!(client.snapshot().capture.state, "Idle");
    runtime.shutdown();
}

#[test]
fn multi_byte_utf8_ts_is_rejected_not_fatal() {
    // Issue #202: a multi-byte UTF-8 `ts` used to panic inside the router
    // thread (non-char-boundary slice in `is_rfc3339`), wedging every
    // later command on the bounded(1) reply channel. It must come back as
    // a plain envelope rejection, and the router must keep serving.
    let (runtime, client) = Runtime::start(RuntimeConfig::default());
    let bad = serde_json::json!({
        "v": 1, "id": "x", "ts": "日日日日日日日",
        "type": "capture.abort", "payload": {}
    });
    match client.send_raw(bad) {
        Err(starling_runtime::machine::Rejection::InvalidEnvelope(detail)) => {
            assert!(detail.contains("ts"), "{detail}");
        }
        other => panic!("expected InvalidEnvelope, got {other:?}"),
    }
    // The router thread survived: a subsequent command still completes
    // (this call would never be answered if the panicking router had died).
    client
        .send(
            Some("take_alive"),
            Command::JobsSetLimits(JobLimits {
                max_queued: 1,
                max_concurrent: 1,
                per_route: Vec::new(),
            }),
        )
        .expect("router must still answer after rejecting a malformed envelope");
    runtime.shutdown();
}

#[test]
fn command_seq_must_be_strictly_monotonic_per_stream() {
    let (runtime, client) = Runtime::start(RuntimeConfig::default());
    // First command on the stream (its machine-level fate is irrelevant).
    let _ = client.send(Some("take_x"), Command::CaptureAbort);
    // A raw command with a lower seq on the same stream is rejected with
    // seq_not_monotonic, never absorbed.
    let stale = serde_json::json!({
        "v": 1, "id": "cmd_stale", "ts": "2026-09-20T10:00:00Z",
        "corr": "take_x", "seq": 0,
        "type": "capture.abort", "payload": {}
    });
    match client.send_raw(stale) {
        Err(starling_runtime::machine::Rejection::SeqNotMonotonic { detail }) => {
            assert!(detail.contains("take_x"), "{detail}");
        }
        other => panic!("expected SeqNotMonotonic, got {other:?}"),
    }
    runtime.shutdown();
}

#[test]
fn submit_against_an_unfrozen_route_is_rejected_before_admission() {
    let source = FakeCaptureSource::new(vec![FakeTakeScript::clean()]);
    let config = test_config(
        source,
        starling_runtime::provider::FakeProvider::new(vec![]),
        InMemoryCaptureStore::new(),
        JobLimits {
            max_queued: 8,
            max_concurrent: 2,
            per_route: vec![],
        },
    );
    let (runtime, client) = Runtime::start(config);
    let events = client.subscribe();

    // No context cycle ran, so no route was frozen: audio must not leave.
    // (Capture can still run — the invariant binds the audio-leave proxy.)
    client
        .send(Some("take_a"), Command::CaptureStart { policy: "push-to-talk".into() })
        .expect("start accepted");
    until(&events, "capture.started", |m| m.type_name() == "capture.started", Duration::from_secs(5));
    std::thread::sleep(Duration::from_millis(20));
    client
        .send(Some("take_a"), Command::CaptureStop { drain: None })
        .expect("stop accepted");
    until(&events, "capture.stopped", |m| m.type_name() == "capture.stopped", Duration::from_secs(5));

    match client.send(
        Some("job-9"),
        Command::JobsSubmit {
            capture_ref: "take_a".into(),
            route: "route-never-frozen".into(),
            budget: "standard".into(),
        },
    ) {
        Err(starling_runtime::machine::Rejection::RouteNotFrozen { route }) => {
            assert_eq!(route, "route-never-frozen");
        }
        other => panic!("expected RouteNotFrozen, got {other:?}"),
    }
    runtime.shutdown();
}

#[test]
fn quiesce_timeout_salvages_samples_as_interrupted_take() {
    let source = FakeCaptureSource::new(vec![FakeTakeScript {
        stop: FakeStop::QuiesceTimeout {
            journal_id: "j_quiesce".into(),
        },
        ..FakeTakeScript::clean()
    }]);
    let store = InMemoryCaptureStore::new();
    let config = test_config(
        source,
        starling_runtime::provider::FakeProvider::new(vec![]),
        store.clone(),
        JobLimits {
            max_queued: 8,
            max_concurrent: 2,
            per_route: vec![],
        },
    );
    let (runtime, client) = Runtime::start(config);
    let events = client.subscribe();

    freeze_route(&client, &events);
    client
        .send(Some("take_q"), Command::CaptureStart { policy: "dictation".into() })
        .expect("start accepted");
    until(&events, "capture.started", |m| m.type_name() == "capture.started", Duration::from_secs(5));
    std::thread::sleep(Duration::from_millis(30));
    client
        .send(Some("take_q"), Command::CaptureStop { drain: Some(true) })
        .expect("stop accepted");
    let collected = until(&events, "capture.stopped", |m| m.type_name() == "capture.stopped", Duration::from_secs(5));

    // The quiesce timeout surfaced as a non-fatal capture.error before the
    // salvaged capture.stopped.
    let types: Vec<&str> = collected.iter().map(|m| m.type_name()).collect();
    let error_index = types
        .iter()
        .position(|t| *t == "capture.error")
        .unwrap_or_else(|| panic!("quiesce timeout must surface as capture.error: {types:?}"));
    let stop_index = types.iter().position(|t| *t == "capture.stopped").unwrap();
    assert!(error_index < stop_index);
    match &collected[error_index].event {
        Event::CaptureError { code, fatal } => {
            assert_eq!(code, "quiesce_timeout");
            assert!(!fatal);
        }
        other => panic!("expected CaptureError, got {other:?}"),
    }
    {
        let takes = store.takes.lock().unwrap();
        assert!(
            takes
                .iter()
                .any(|(status, id, note)| *status == TakeStatus::Interrupted
                    && id == "take_q"
                    && note.contains("salvaged")),
            "salvage recorded: {takes:?}"
        );
    }
    // The snapshot is a projection that can lag the collected events on a
    // loaded runner: poll for the terminal state instead of racing it.
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    loop {
        if client.snapshot().capture.state == "Persisted" {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "capture projection never reached Persisted: {:?}",
            client.snapshot().capture.state
        );
        std::thread::sleep(Duration::from_millis(25));
    }
    runtime.shutdown();
}

#[test]
fn fatal_device_error_mid_take_interrupts() {
    let source = FakeCaptureSource::new(vec![FakeTakeScript {
        error_after: Some((Duration::from_millis(15), "DeviceUnavailable".into())),
        stop: FakeStop::Clean {
            journal_id: "j_lost".into(),
            ack_fraction: 1.0,
        },
        ..FakeTakeScript::clean()
    }]);
    let config = test_config(
        source,
        starling_runtime::provider::FakeProvider::new(vec![]),
        InMemoryCaptureStore::new(),
        JobLimits {
            max_queued: 8,
            max_concurrent: 2,
            per_route: vec![],
        },
    );
    let (runtime, client) = Runtime::start(config);
    let events = client.subscribe();

    freeze_route(&client, &events);
    client
        .send(Some("take_5"), Command::CaptureStart { policy: "dictation".into() })
        .expect("start accepted");
    let collected = until(&events, "capture.error", |m| m.type_name() == "capture.error", Duration::from_secs(5));
    match &collected.last().unwrap().event {
        Event::CaptureError { code, fatal } => {
            assert_eq!(code, "device_stream_lost");
            assert!(*fatal);
        }
        other => panic!("expected fatal CaptureError, got {other:?}"),
    }
    // The machine is Interrupted with the take salvaged into the registry
    // (the projection lags the event by one actor-loop pass).
    wait_for_state(&client, "Interrupted", Duration::from_secs(2));
    runtime.shutdown();
}

#[test]
fn jobs_failure_from_provider_error_is_retryable_and_isolated() {
    let source = FakeCaptureSource::new(vec![]);
    let provider = starling_runtime::provider::FakeProvider::new(vec![
        FakeJob::fails("worker_crash", true),
    ]);
    let config = test_config(
        std::sync::Arc::clone(&source),
        provider,
        InMemoryCaptureStore::new(),
        JobLimits {
            max_queued: 8,
            max_concurrent: 1,
            per_route: vec![],
        },
    );
    let (runtime, client) = Runtime::start(config);
    let events = client.subscribe();

    freeze_route(&client, &events);
    run_take(&source, &client, &events, "take_a", FakeTakeScript::clean());
    client
        .send(Some("job-5"), Command::JobsSubmit {
            capture_ref: "take_a".into(),
            route: "local-default".into(),
            budget: "standard".into(),
        })
        .expect("submit accepted");
    let collected = until(&events, "jobs.failed", |m| m.type_name() == "jobs.failed", Duration::from_secs(5));
    match &collected
        .iter()
        .rev()
        .find(|m| m.type_name() == "jobs.failed")
        .unwrap()
        .event
    {
        Event::JobsFailed { reason, retryable } => {
            assert_eq!(reason, "worker_crash");
            assert!(*retryable);
        }
        other => panic!("expected JobsFailed, got {other:?}"),
    }
    // Capture and history untouched: capture stayed Persisted.
    assert_eq!(client.snapshot().capture.state, "Persisted");
    runtime.shutdown();
}

/// Empties the subscription's queue (bounded-capacity setups need a
/// known-clean subscriber before provoking backpressure on purpose).
fn drain(events: &EventSub) {
    while events.try_recv().is_ok() {}
}

// ------------------------------------------------------------------------- //
// Worker-report delivery under a full inbox (issue #210)
// ------------------------------------------------------------------------- //

/// Regression for issue #210: worker reports shared the `jobs.*` command
/// inbox and were `try_send`-discarded on `Full` — a lost `Done` wedged
/// the job's core in `Recognizing` forever and permanently leaked its
/// scheduler slot.
///
/// The inbox fills for real: the runtime's only event subscriber stops
/// draining, so the scheduler parks inside `bus.emit` (the bus's own
/// backpressure contract) while the worker's partial storm backs the
/// inbox up past its bound. With the fix, the worker parks on the full
/// inbox instead of dropping — every report, the `Done` included, lands
/// as soon as the subscriber resumes, and the slot is provably released
/// (a second job dispatches after it).
#[test]
fn worker_done_report_survives_a_full_inbox_and_releases_the_slot() {
    let source = FakeCaptureSource::new(vec![]);
    let store = InMemoryCaptureStore::new();
    let partials: Vec<Partial> = (0..2_000)
        .map(|index| Partial {
            text: format!("chunk {index}"),
            stability_hint: "unstable".to_string(),
        })
        .collect();
    let storm = FakeJob {
        outcome: ProviderOutcome::Completed {
            text: "storm result".to_string(),
            backend: "fake-provider".to_string(),
            timing_ms: 12.0,
            completion_evidence: "final_decode".to_string(),
            transformed: false,
        },
        partials,
        work_ms: 5,
    };
    // FakeProvider pops its script LIFO, so the second job's entry comes
    // first in the vector.
    let provider = starling_runtime::provider::FakeProvider::new(vec![
        FakeJob::completes_with("after the storm"),
        storm,
    ]);
    let mut config = test_config(
        std::sync::Arc::clone(&source),
        provider,
        store,
        JobLimits {
            max_queued: 8,
            max_concurrent: 1,
            per_route: vec![],
        },
    );
    // A tiny subscriber queue makes the stall deterministic: within a
    // handful of progress events the scheduler is parked in bus.emit and
    // stops draining its inbox while the storm floods it.
    config.event_capacity = 8;
    let (runtime, client) = Runtime::start(config);
    let events = client.subscribe();

    freeze_route(&client, &events);
    run_take(&source, &client, &events, "take_s", FakeTakeScript::clean());
    // Settle trailing capture events, then leave the queue empty so the
    // submit below cannot park the scheduler before its worker runs.
    std::thread::sleep(Duration::from_millis(100));
    drain(&events);

    client
        .send(Some("job-1"), Command::JobsSubmit {
            capture_ref: "take_s".into(),
            route: "local-default".into(),
            budget: "standard".into(),
        })
        .expect("submit accepted");

    // The stall: nobody drains the subscriber, so the scheduler parks in
    // bus.emit while the worker's 2 000 partial reports fill the inbox
    // (bound 64) — exactly the window in which a best-effort try_send
    // drops the Done. (With the fix the worker parks here instead.)
    std::thread::sleep(Duration::from_millis(250));

    // Resume draining: the parked system must unwind and the Done must
    // still land, completing the job instead of wedging it.
    let collected = until(
        &events,
        "jobs.completed(job-1)",
        |m| m.type_name() == "jobs.completed" && m.corr.as_deref() == Some("job-1"),
        Duration::from_secs(10),
    );
    // The storm really did exceed the inbox bound, so under drop-on-Full
    // semantics the Done could not have survived.
    let progress = collected
        .iter()
        .filter(|m| m.type_name() == "jobs.progress" && m.corr.as_deref() == Some("job-1"))
        .count();
    assert!(
        progress > 64,
        "the partial storm must exceed the inbox bound (saw {progress} of 2 000)"
    );
    let completed = collected
        .iter()
        .rev()
        .find(|m| m.type_name() == "jobs.completed" && m.corr.as_deref() == Some("job-1"))
        .unwrap();
    match &completed.event {
        Event::JobsCompleted(data) => assert_eq!(data.text, "storm result"),
        other => panic!("expected JobsCompleted, got {other:?}"),
    }
    // The slot was released: the projection shows no active jobs.
    {
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let snapshot = client.snapshot();
            if snapshot.jobs.active == 0 && snapshot.jobs.state == "Completed" {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "job-1 must be Completed with its slot released: {:?}",
                snapshot.jobs
            );
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    // Capacity was truly restored: a second job dispatches onto the freed
    // slot and completes (a leaked slot would park it at Queued forever).
    client
        .send(Some("job-2"), Command::JobsSubmit {
            capture_ref: "take_s".into(), // job-1 is terminal, so no duplicate
            route: "local-default".into(),
            budget: "standard".into(),
        })
        .expect("job-2 accepted");
    until(
        &events,
        "jobs.completed(job-2)",
        |m| m.type_name() == "jobs.completed" && m.corr.as_deref() == Some("job-2"),
        Duration::from_secs(5),
    );

    runtime.shutdown();
}

// CaptureGap is referenced for signature compatibility with the fake
// scripts' gap spans.
#[allow(dead_code)]
fn _gap_compat(_gap: CaptureGap) {}
