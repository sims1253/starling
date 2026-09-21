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
//! fatal device-error interruption path — plus the capture-machine
//! recovery regressions (issues #211/#212): a failed device open settles
//! back to Idle for the retry, and stop-handshake errors never drop the
//! take or its gap evidence.

use std::time::{Duration, Instant};

use starling_dictation::recorder::{CaptureGap, RecorderFault};
use starling_runtime::bus::{EventMessage, EventSub};
use starling_runtime::machine::capture::{
    CaptureSource, CaptureStore, CaptureConfig, InMemoryCaptureStore, TakeRecord, TakeStatus,
    V1FileCaptureStore,
};
use starling_runtime::protocol::replay::{route_freeze_violations, MachineReplay};
use starling_runtime::machine::Rejection;
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
    store: std::sync::Arc<dyn CaptureStore>,
    limits: JobLimits,
) -> RuntimeConfig {
    RuntimeConfig::default()
        .with_capture_source(source)
        .with_provider(provider)
        .with_capture_store(store)
        .with_capture_config(CaptureConfig {
            journals_dir: std::env::temp_dir().join("starling-runtime-test-journals"),
            poll_interval: Duration::from_millis(10),
            ..CaptureConfig::default()
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
/// seen for it. Sample accumulation is proven, not assumed: the take waits
/// for its first `capture.progress` (emitted by the capture actor's poll
/// tick once the fake's acknowledged count is nonzero) instead of sleeping
/// a fixed window a loaded runner may not fit inside (#217).
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
    log.extend(until(
        events,
        "capture.progress (samples accumulated)",
        |m| m.type_name() == "capture.progress" && m.corr.as_deref() == Some(take),
        Duration::from_secs(5),
    ));
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

/// Polls the runtime projection until `predicate` holds, or panics with
/// `what` after `deadline`. The projection is a non-consuming observation:
/// unlike the event subscription, polling it never drains (and so never
/// unparks) anything — which is what makes it safe to use as a gate in
/// backpressure scenarios (#210, #217).
fn wait_for_projection(
    client: &RuntimeClient,
    what: &str,
    predicate: impl Fn(&starling_runtime::RuntimeSnapshot) -> bool,
    deadline: Duration,
) {
    let start = Instant::now();
    loop {
        let snapshot = client.snapshot();
        if predicate(&snapshot) {
            return;
        }
        assert!(
            start.elapsed() < deadline,
            "{what}; capture {}, jobs {:?}",
            snapshot.capture.state,
            snapshot.jobs.jobs,
        );
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
    // Let the take accumulate samples via an event-based wait rather than
    // a fixed sleep (#217): the first capture.progress proves the fake's
    // acknowledged count went nonzero before the stop.
    until(
        &events,
        "capture.progress (samples accumulated)",
        |m| m.type_name() == "capture.progress" && m.corr.as_deref() == Some("take_a"),
        Duration::from_secs(5),
    );
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
    // The salvage below keeps whatever the fake accumulated: prove samples
    // exist via the first capture.progress instead of a fixed sleep a
    // loaded runner may not fit inside (#217).
    until(
        &events,
        "capture.progress (samples accumulated)",
        |m| m.type_name() == "capture.progress" && m.corr.as_deref() == Some("take_q"),
        Duration::from_secs(5),
    );
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
        error_after: Some((
            Duration::from_millis(15),
            RecorderFault::Device("DeviceUnavailable".into()),
        )),
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
///
/// Verification is timing-independent (#217): the stall window is gated on
/// the (non-consuming) projection showing the storm in flight, and the
/// outcome is asserted by exact counts — all 2 000 partials must precede
/// the `Done` on the FIFO path, so a single dropped report anywhere fails
/// the test; a wedged `Done` fails the completion wait loudly.
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
    // submit below cannot park the scheduler before its worker runs. The
    // settle is gated on the capture projection reaching its terminal
    // state — once it has, the capture poll ticker (the only late emitter)
    // has stopped, so a single drain leaves the queue genuinely empty. No
    // fixed sleep to mis-time on a loaded runner (#217).
    wait_for_projection(
        &client,
        "capture projection never settled after take_s",
        |s| s.capture.state == "Persisted",
        Duration::from_secs(5),
    );
    drain(&events);

    client
        .send(Some("job-1"), Command::JobsSubmit {
            capture_ref: "take_s".into(),
            route: "local-default".into(),
            budget: "standard".into(),
        })
        .expect("submit accepted");

    // Gate the stall on the storm provably being in flight: the projection
    // shows the worker dispatched and job-1's core in Recognizing (it
    // advances there at dispatch, before the worker spawns). Sleeping a
    // fixed window straight after the submit — the old shape — could
    // elapse before a loaded runner had even scheduled the submit chain.
    wait_for_projection(
        &client,
        "job-1 never dispatched onto its worker",
        |s| {
            s.jobs.active == 1
                && s.jobs
                    .jobs
                    .iter()
                    .any(|job| job.job == "job-1" && job.state == "Recognizing")
        },
        Duration::from_secs(5),
    );

    // The stall: nobody drains the subscriber, so the scheduler parks in
    // bus.emit (the bus's own backpressure contract) once 8 events queue
    // up, while the worker's 2 000 partial reports — fired back-to-back
    // after the fake's single 5 ms work delay — fill the inbox (bound 64)
    // and park in send_blocking. Exactly the window in which a best-effort
    // try_send drops the Done. The hold only needs to cover the thread
    // wake-ups for those bounded-queue pushes (microseconds of work once
    // Recognizing is observable); 250 ms is three orders of magnitude of
    // margin, and the assertions below verify the fix independently of
    // this window being hit (see the count assertion).
    std::thread::sleep(Duration::from_millis(250));

    // Resume draining: the parked system must unwind and the Done must
    // still land, completing the job instead of wedging it.
    let collected = until(
        &events,
        "jobs.completed(job-1)",
        |m| m.type_name() == "jobs.completed" && m.corr.as_deref() == Some("job-1"),
        Duration::from_secs(10),
    );
    // Deterministic delivery proof: every one of the 2 000 partials must
    // reach this subscriber before the Done — they share the worker →
    // bounded inbox → scheduler → bus → subscriber FIFO path, and
    // jobs.progress is legal throughout (the core sits in Recognizing from
    // dispatch until the Done). A single report lost anywhere — the exact
    // #210 failure mode — fails this count. This replaces the old
    // `progress > 64` bound, which only showed the storm was large and
    // could not vouch that any report ever faced a Full inbox.
    let progress = collected
        .iter()
        .filter(|m| m.type_name() == "jobs.progress" && m.corr.as_deref() == Some("job-1"))
        .count();
    assert_eq!(
        progress, 2_000,
        "every partial report must survive the bounded inbox and land before the Done"
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

// ------------------------------------------------------------------------- //
// Capture-machine recovery (issues #211 / #212)
// ------------------------------------------------------------------------- //

/// A test store that records the full persisted `TakeRecord`s (the
/// in-memory store keeps only ids), so the salvage regressions can assert
/// gap evidence and acknowledged boundaries.
#[derive(Default)]
struct RecordingStore {
    records: std::sync::Mutex<Vec<(TakeStatus, TakeRecord, String)>>,
}

impl RecordingStore {
    fn new() -> std::sync::Arc<Self> {
        std::sync::Arc::new(Self::default())
    }

    fn interrupted(&self, id: &str) -> Option<TakeRecord> {
        self.records
            .lock()
            .unwrap()
            .iter()
            .find(|(status, record, _)| *status == TakeStatus::Interrupted && record.id == id)
            .map(|(_, record, _)| record.clone())
    }

    fn completed(&self, id: &str) -> Option<TakeRecord> {
        self.records
            .lock()
            .unwrap()
            .iter()
            .find(|(status, record, _)| *status == TakeStatus::Complete && record.id == id)
            .map(|(_, record, _)| record.clone())
    }
}

impl CaptureStore for RecordingStore {
    fn commit_take(&self, take: &TakeRecord) -> Result<(), String> {
        self.records
            .lock()
            .unwrap()
            .push((TakeStatus::Complete, take.clone(), String::new()));
        Ok(())
    }
    fn mark_interrupted(&self, take: &TakeRecord, note: &str) -> Result<(), String> {
        self.records
            .lock()
            .unwrap()
            .push((TakeStatus::Interrupted, take.clone(), note.to_string()));
        Ok(())
    }
    fn describe(&self) -> String {
        "recording".to_string()
    }
}

/// Polls until the capture projection shows `state`, or panics.
fn wait_for_capture_state(client: &RuntimeClient, state: &str, deadline: Duration) {
    wait_for_machine_state(client, "capture", state, deadline);
}

/// Polls until the context projection shows `state`, or panics.
fn wait_for_context_state(client: &RuntimeClient, state: &str, deadline: Duration) {
    wait_for_machine_state(client, "context", state, deadline);
}

fn wait_for_machine_state(
    client: &RuntimeClient,
    machine: &str,
    state: &str,
    deadline: Duration,
) {
    let start = Instant::now();
    loop {
        let current = match machine {
            "capture" => client.snapshot().capture.state.clone(),
            "context" => client.snapshot().context.state.clone(),
            other => panic!("unknown machine {other:?}"),
        };
        if current == state {
            return;
        }
        assert!(
            start.elapsed() < deadline,
            "{machine} never reached {state:?} (now {current:?})"
        );
        std::thread::sleep(Duration::from_millis(25));
    }
}

/// Issue #211: a failed device open must not wedge the capture machine in
/// `Interrupted` for the process lifetime — the fatal error settles back
/// to Idle and the next `capture.start` retries the device.
#[test]
fn failed_device_open_returns_to_idle_and_a_retry_succeeds() {
    // No script queued: the fake source's first start fails the device
    // open (its honest "no microphone" error).
    let source = FakeCaptureSource::new(vec![]);
    let store = InMemoryCaptureStore::new();
    let config = test_config(
        std::sync::Arc::clone(&source),
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
        .send(Some("take_9"), Command::CaptureStart { policy: "dictation".into() })
        .expect("start itself is accepted; the failure surfaces as an event");
    let collected = until(
        &events,
        "capture.error",
        |m| m.type_name() == "capture.error",
        Duration::from_secs(5),
    );
    match &collected.last().unwrap().event {
        Event::CaptureError { code, fatal } => {
            assert_eq!(code, "device_open_failed");
            assert!(*fatal, "the open failure is fatal to the take");
        }
        other => panic!("expected CaptureError, got {other:?}"),
    }

    // The machine settled back to Idle (a runtime-internal edge the UI
    // sees via the snapshot, not the wire), and the route freeze the
    // failed take took was released.
    wait_for_capture_state(&client, "Idle", Duration::from_secs(5));
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let context = client.snapshot().context.state.clone();
        if context == "Released" {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "the failed take never released its route freeze (context {context:?})"
        );
        std::thread::sleep(Duration::from_millis(25));
    }

    // The retry: a device appears (a script is queued) and the next take
    // runs end to end. With the #211 wedge this start was rejected
    // IllegalInState{Interrupted}.
    source.push(FakeTakeScript::clean());
    client
        .send(Some("take_10"), Command::CaptureStart { policy: "dictation".into() })
        .expect("retry start accepted — the machine left Interrupted");
    until(&events, "capture.started", |m| m.type_name() == "capture.started", Duration::from_secs(5));
    client
        .send(Some("take_10"), Command::CaptureStop { drain: Some(true) })
        .expect("stop accepted");
    until(&events, "capture.stopped", |m| m.type_name() == "capture.stopped", Duration::from_secs(5));
    {
        let takes = store.takes.lock().unwrap();
        assert!(
            takes
                .iter()
                .any(|(status, id, _)| *status == TakeStatus::Complete && id == "take_10"),
            "take_10 committed: {takes:?}"
        );
    }
    runtime.shutdown();
}

/// Issue #212: a device error on the stop handshake consumed the session
/// and used to drop the whole take. The interrupted record is still
/// registered — metadata-only (the error carries no audio) but with the
/// gap evidence and the last acknowledged boundary intact.
#[test]
fn device_error_on_stop_still_registers_an_interrupted_take() {
    let source = FakeCaptureSource::new(vec![FakeTakeScript {
        gaps: vec![(
            Duration::from_millis(5),
            CaptureGap { start_sample: 160, end_sample: 320 },
        )],
        stop: FakeStop::DeviceError("DeviceUnavailable".into()),
        ..FakeTakeScript::clean()
    }]);
    let store = RecordingStore::new();
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
        .send(Some("take_d"), Command::CaptureStart { policy: "dictation".into() })
        .expect("start accepted");
    until(&events, "capture.started", |m| m.type_name() == "capture.started", Duration::from_secs(5));
    std::thread::sleep(Duration::from_millis(40));
    client
        .send(Some("take_d"), Command::CaptureStop { drain: Some(true) })
        .expect("stop accepted");
    let collected = until(&events, "capture.error", |m| m.type_name() == "capture.error", Duration::from_secs(5));
    match &collected.last().unwrap().event {
        Event::CaptureError { code, fatal } => {
            assert_eq!(code, "device_error_on_stop");
            assert!(*fatal);
        }
        other => panic!("expected CaptureError, got {other:?}"),
    }

    let deadline = Instant::now() + Duration::from_secs(5);
    let record = loop {
        if let Some(record) = store.interrupted("take_d") {
            break record;
        }
        assert!(
            Instant::now() < deadline,
            "take_d never registered as interrupted after the stop error"
        );
        std::thread::sleep(Duration::from_millis(25));
    };
    assert!(record.samples.is_empty(), "no audio rides a device stop error");
    assert!(
        record.acknowledged_samples > 0,
        "the last acknowledged boundary is kept: {:?}",
        record.acknowledged_samples
    );
    assert!(
        record
            .gaps
            .iter()
            .any(|gap| gap.start_sample == 160 && gap.end_sample == 320),
        "the surfaced gap span travels with the record: {:?}",
        record.gaps
    );
    if record.final_sample_index > record.acknowledged_samples {
        assert!(
            record.gaps.iter().any(|gap| gap.start_sample == record.acknowledged_samples
                && gap.end_sample == record.final_sample_index),
            "the unacknowledged tail is recorded as a gap: {:?}",
            record.gaps
        );
    }
    // The contract's terminal state for a lost device (capture-interrupted
    // fixture): the fatal error entered Interrupted and no stopped event
    // follows.
    wait_for_capture_state(&client, "Interrupted", Duration::from_secs(5));
    // The route freeze died with the take, not with the machine: the
    // context cycle is free to run again (review follow-up on #212: the
    // device-error salvage used to leak the freeze — the #211 wedge one
    // layer down).
    wait_for_context_state(&client, "Released", Duration::from_secs(5));
    runtime.shutdown();
}

/// Issue #212: `capture.abort` salvages "whatever the recorder
/// acknowledged ... source preserved, never deleted" — a stop() error used
/// to discard the whole take. A quiesce-timeout abort keeps the preserved
/// samples; a device-side error keeps the metadata-only record.
#[test]
fn aborted_take_with_a_failing_stop_handshake_is_still_salvaged() {
    let source = FakeCaptureSource::new(vec![
        FakeTakeScript {
            gaps: vec![(
                Duration::from_millis(5),
                CaptureGap { start_sample: 160, end_sample: 320 },
            )],
            stop: FakeStop::QuiesceTimeout { journal_id: "j_abort_q".into() },
            ..FakeTakeScript::clean()
        },
        FakeTakeScript {
            stop: FakeStop::DeviceError("DeviceUnavailable".into()),
            ..FakeTakeScript::clean()
        },
        FakeTakeScript::clean(),
    ]);
    let store = RecordingStore::new();
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

    // take_a: abort whose stop times out — the preserved samples ride in
    // the error and must still be salvaged.
    client
        .send(Some("take_a"), Command::CaptureStart { policy: "push-to-talk".into() })
        .expect("start accepted");
    until(&events, "capture.started", |m| m.type_name() == "capture.started", Duration::from_secs(5));
    std::thread::sleep(Duration::from_millis(40));
    client
        .send(Some("take_a"), Command::CaptureAbort)
        .expect("abort accepted");
    let deadline = Instant::now() + Duration::from_secs(5);
    let salvaged = loop {
        if let Some(record) = store.interrupted("take_a") {
            break record;
        }
        assert!(Instant::now() < deadline, "take_a never salvaged");
        std::thread::sleep(Duration::from_millis(25));
    };
    assert!(!salvaged.samples.is_empty(), "quiesce-salvaged samples kept: {:?}", salvaged.samples.len());
    assert_eq!(
        salvaged.journal.as_ref().expect("journal linkage").id,
        "j_abort_q"
    );
    assert!(
        salvaged
            .gaps
            .iter()
            .any(|gap| gap.start_sample == 160 && gap.end_sample == 320),
        "gap evidence kept: {:?}",
        salvaged.gaps
    );
    // abort returned the machine to Idle, and the route freeze went with
    // the take: the context cycle can run again.
    wait_for_capture_state(&client, "Idle", Duration::from_secs(5));
    wait_for_context_state(&client, "Released", Duration::from_secs(5));

    // take_b: abort whose stop fails device-side — metadata-only, still
    // registered.
    client
        .send(Some("take_b"), Command::CaptureStart { policy: "push-to-talk".into() })
        .expect("start accepted");
    until(&events, "capture.started", |m| m.type_name() == "capture.started", Duration::from_secs(5));
    std::thread::sleep(Duration::from_millis(40));
    client
        .send(Some("take_b"), Command::CaptureAbort)
        .expect("abort accepted");
    let metadata_only = loop {
        if let Some(record) = store.interrupted("take_b") {
            break record;
        }
        assert!(Instant::now() < deadline, "take_b never registered as interrupted");
        std::thread::sleep(Duration::from_millis(25));
    };
    assert!(metadata_only.samples.is_empty());
    assert!(metadata_only.acknowledged_samples > 0);

    // The context cycle really is free: a fresh snapshot/mode decision
    // completes (from RouteFrozen, context.snapshot would be rejected as
    // IllegalInState — the wedge this regression pins down), and the next
    // take runs end to end.
    freeze_route(&client, &events);
    client
        .send(Some("take_c"), Command::CaptureStart { policy: "push-to-talk".into() })
        .expect("start after aborted takes accepted");
    until(&events, "capture.started", |m| m.type_name() == "capture.started", Duration::from_secs(5));
    std::thread::sleep(Duration::from_millis(40));
    client
        .send(Some("take_c"), Command::CaptureStop { drain: Some(true) })
        .expect("stop accepted");
    until(&events, "capture.stopped", |m| m.type_name() == "capture.stopped" && m.corr.as_deref() == Some("take_c"), Duration::from_secs(5));
    let completed = loop {
        if let Some(record) = store.completed("take_c") {
            break record;
        }
        assert!(Instant::now() < deadline, "take_c never committed");
        std::thread::sleep(Duration::from_millis(25));
    };
    assert!(!completed.samples.is_empty());
    runtime.shutdown();
}

/// Issue #212: the gap spans surfaced as `capture.gap` events during a
/// fatal mid-take fault used to be dropped from the persisted interrupted
/// record (hard-coded empty); they must travel with it.
#[test]
fn fatal_mid_take_salvage_keeps_gap_evidence() {
    let source = FakeCaptureSource::new(vec![FakeTakeScript {
        gaps: vec![(
            Duration::from_millis(5),
            CaptureGap { start_sample: 160, end_sample: 320 },
        )],
        error_after: Some((
            Duration::from_millis(15),
            RecorderFault::Device("DeviceUnavailable".into()),
        )),
        stop: FakeStop::Clean {
            journal_id: "j_lost2".into(),
            ack_fraction: 1.0,
        },
        ..FakeTakeScript::clean()
    }]);
    let store = RecordingStore::new();
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
    let deadline = Instant::now() + Duration::from_secs(5);
    let record = loop {
        if let Some(record) = store.interrupted("take_5") {
            break record;
        }
        assert!(Instant::now() < deadline, "take_5 never salvaged");
        std::thread::sleep(Duration::from_millis(25));
    };
    assert!(
        record
            .gaps
            .iter()
            .any(|gap| gap.start_sample == 160 && gap.end_sample == 320),
        "the surfaced gap span reaches the persisted record: {:?}",
        record.gaps
    );
    assert!(!record.samples.is_empty(), "the clean-stop salvage keeps the audio");
    assert_eq!(
        record.journal.as_ref().expect("journal linkage").id,
        "j_lost2"
    );
    runtime.shutdown();
}

// ---------------------------------------------------------------------------
// Issue #216: runtime actor hygiene
// ---------------------------------------------------------------------------

/// Part 2: terminal jobs leave the scheduler's map (no one-`MachineCore`
/// -per-submit leak across a long session) while the wire view keeps
/// reporting the last terminal state, and the duplicate-submission guard
/// still catches in-flight duplicates — terminal ones are forgotten, by
/// design (the guard compares active states only).
#[test]
fn terminal_jobs_retire_from_the_map_while_the_wire_view_and_duplicate_guard_hold() {
    let source = FakeCaptureSource::new(vec![]);
    let provider = starling_runtime::provider::FakeProvider::new(vec![
        FakeJob::completes_with("first"),
        FakeJob::completes_with("second"),
        FakeJob::completes_with("third"),
        FakeJob::completes_with("fourth"),
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
    run_take(&source, &client, &events, "take_r", FakeTakeScript::clean());

    // Job one completes and is retired: the map empties, the wire view
    // keeps its terminal state.
    client
        .send(Some("job-r1"), Command::JobsSubmit {
            capture_ref: "take_r".into(),
            route: "local-default".into(),
            budget: "standard".into(),
        })
        .expect("job-r1 accepted");
    until(
        &events,
        "jobs.completed(job-r1)",
        |m| m.type_name() == "jobs.completed" && m.corr.as_deref() == Some("job-r1"),
        Duration::from_secs(5),
    );
    wait_for_projection(
        &client,
        "job-r1 must be Completed and retired from the map",
        |s| s.jobs.state == "Completed" && s.jobs.jobs.is_empty(),
        Duration::from_secs(5),
    );

    // A retired job is gone, not merely finished: cancel answers
    // UnknownJob (the documented shape of terminal removal).
    assert!(matches!(
        client.send(Some("job-r1"), Command::JobsCancel { job_id: "job-r1".into() }),
        Err(Rejection::UnknownJob { .. })
    ));

    // The terminal job is forgotten for admission too: the same
    // captureRef submits again without a duplicate_submission rejection.
    client
        .send(Some("job-r2"), Command::JobsSubmit {
            capture_ref: "take_r".into(),
            route: "local-default".into(),
            budget: "standard".into(),
        })
        .expect("job-r2 accepted");
    until(
        &events,
        "jobs.completed(job-r2)",
        |m| m.type_name() == "jobs.completed" && m.corr.as_deref() == Some("job-r2"),
        Duration::from_secs(5),
    );
    wait_for_projection(
        &client,
        "job-r2 must be Completed and retired from the map",
        |s| s.jobs.state == "Completed" && s.jobs.jobs.is_empty(),
        Duration::from_secs(5),
    );

    // The guard still fires while a job is in flight: an active duplicate
    // is rejected (max_concurrent 1 keeps job-r3 in the machine).
    client
        .send(Some("job-r3"), Command::JobsSubmit {
            capture_ref: "take_r".into(),
            route: "local-default".into(),
            budget: "standard".into(),
        })
        .expect("job-r3 accepted");
    client
        .send(Some("job-r4"), Command::JobsSubmit {
            capture_ref: "take_r".into(), // same captureRef while job-r3 runs
            route: "local-default".into(),
            budget: "standard".into(),
        })
        .expect("job-r4 accepted");
    let collected = until(
        &events,
        "jobs.rejected(job-r4)",
        |m| m.type_name() == "jobs.rejected" && m.corr.as_deref() == Some("job-r4"),
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
    until(
        &events,
        "jobs.completed(job-r3)",
        |m| m.type_name() == "jobs.completed" && m.corr.as_deref() == Some("job-r3"),
        Duration::from_secs(5),
    );

    // Four jobs later the map is as empty as after the first: the map
    // holds in-flight jobs only.
    wait_for_projection(
        &client,
        "the jobs map must not accumulate terminal entries",
        |s| s.jobs.jobs.is_empty(),
        Duration::from_secs(5),
    );

    runtime.shutdown();
}

/// Part 3: the WAV encode runs on the worker, not the scheduler loop. A
/// multi-minute-sized take (30M samples → a ~60 MB WAV, hundreds of
/// milliseconds of encode) is submitted and cancelled immediately: the
/// cancel's receipt must come back from an idle scheduler, not queue
/// behind the whole encode (the pre-#216 shape stalled every `jobs.*`
/// command for the encode's duration). The functional outcome pins the
/// rest: the job settles Cancelled, the late completion lands nowhere,
/// and the worker really did encode and ship the full-sized WAV.
#[test]
fn cancel_is_answered_while_the_wav_encode_is_in_flight() {
    let source = FakeCaptureSource::new(vec![]);
    let provider =
        starling_runtime::provider::FakeProvider::new(vec![FakeJob::completes_with("late")]);
    let provider_handle = std::sync::Arc::clone(&provider);
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
    run_take(
        &source,
        &client,
        &events,
        "take_big",
        FakeTakeScript {
            // The fake outruns the cap well before the capture actor's
            // first 10 ms poll tick, so the take's size is pinned at 30M
            // samples (a ~60 MB WAV) regardless of runner load.
            samples_per_second: 4_000_000_000,
            sample_cap: 30_000_000,
            ..FakeTakeScript::clean()
        },
    );

    client
        .send(Some("job-big"), Command::JobsSubmit {
            capture_ref: "take_big".into(),
            route: "local-default".into(),
            budget: "standard".into(),
        })
        .expect("submit accepted");

    // The cancel races the worker's encode of the ~60 MB WAV. With the
    // encode on the scheduler loop the receipt queues behind the entire
    // encode (hundreds of ms); with the encode on the worker it is
    // answered in scheduler time. The bound is set well under half the
    // encode's expected duration.
    let cancel_started = Instant::now();
    let cancelled = client.send(Some("job-big"), Command::JobsCancel { job_id: "job-big".into() });
    let cancel_latency = cancel_started.elapsed();
    cancelled.expect("cancel is legal while the worker encodes");
    assert!(
        cancel_latency < Duration::from_millis(150),
        "cancel took {cancel_latency:?} — the scheduler appears stalled on the WAV encode"
    );

    // The job settles Cancelled and the worker's late completion lands
    // nowhere.
    wait_for_state(&client, "Cancelled", Duration::from_secs(5));
    std::thread::sleep(Duration::from_millis(100));
    let mut drained = Vec::new();
    while let Ok(message) = events.try_recv() {
        drained.push(message);
    }
    assert!(
        !drained
            .iter()
            .any(|m| m.type_name() == "jobs.completed" && m.corr.as_deref() == Some("job-big")),
        "a cancelled job must not complete: {:?}",
        drained.iter().map(|m| m.type_name()).collect::<Vec<_>>()
    );

    // The encode really ran at this scale — the worker shipped the
    // full-sized WAV to the provider before its report was dropped.
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let requests = provider_handle.requests();
        if requests
            .iter()
            .any(|(id, size)| id == "job-big" && *size > 40_000_000)
        {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "the worker never encoded the full-sized WAV: {:?}",
            requests
        );
        std::thread::sleep(Duration::from_millis(10));
    }

    runtime.shutdown();
}

/// The real v1-file capture store with a completion flag: the durable
/// write (WAV encode + fsync'd session layout) is the real one, and the
/// flag gives the responsiveness regression an order-based witness of the
/// commit landing — no timing guesses.
struct FlaggedV1Store {
    inner: V1FileCaptureStore,
    landed: std::sync::Mutex<Vec<String>>,
}

impl CaptureStore for FlaggedV1Store {
    fn commit_take(&self, take: &TakeRecord) -> Result<(), String> {
        let result = self.inner.commit_take(take);
        if result.is_ok() {
            self.landed.lock().unwrap().push(take.capture_id.clone());
        }
        result
    }
    fn mark_interrupted(&self, take: &TakeRecord, note: &str) -> Result<(), String> {
        let result = self.inner.mark_interrupted(take, note);
        if result.is_ok() {
            self.landed.lock().unwrap().push(take.capture_id.clone());
        }
        result
    }
    fn describe(&self) -> String {
        self.inner.describe()
    }
}

/// Issue #249, the capture-side twin of #216: with the v1-file store, the
/// stop handshake used to run the take's whole WAV encode inline on the
/// capture actor — a multi-minute take stalled every `capture.*` command
/// for the duration. The persist now runs on a per-take worker, so:
///
/// - the machine observably enters `Draining` while the encode is still
///   running (pre-#249 the projection jumped `Recording → Persisted`
///   inside one blocked command — `Draining` was never publishable);
/// - a `capture.*` command sent mid-encode is answered immediately, while
///   the durable commit has not landed (the order-based property);
/// - and the take still completes with the unchanged contract: durable
///   store row first, `capture.stopped` after, machine `Persisted`.
#[test]
fn capture_commands_stay_answered_while_a_long_v1_persist_encodes() {
    let source = FakeCaptureSource::new(vec![]);
    let dir = tempfile::tempdir().expect("v1 store temp dir");
    let store = std::sync::Arc::new(FlaggedV1Store {
        inner: V1FileCaptureStore::open(dir.path()).expect("v1 store opens"),
        landed: std::sync::Mutex::new(Vec::new()),
    });
    let config = test_config(
        std::sync::Arc::clone(&source),
        starling_runtime::provider::FakeProvider::new(vec![]),
        store.clone(),
        JobLimits {
            max_queued: 8,
            max_concurrent: 1,
            per_route: vec![],
        },
    );
    let (runtime, client) = Runtime::start(config);
    let events = client.subscribe();
    freeze_route(&client, &events);

    // A multi-minute take (30M samples ≈ a 60 MB WAV the v1 store must
    // encode and durably write), pinned the same way the #216 jobs test
    // pins its take.
    source.push(FakeTakeScript {
        stop: FakeStop::Clean { journal_id: "j_v1big".into(), ack_fraction: 1.0 },
        samples_per_second: 4_000_000_000,
        sample_cap: 30_000_000,
        ..FakeTakeScript::clean()
    });
    client
        .send(Some("take_v1"), Command::CaptureStart { policy: "push-to-talk".into() })
        .expect("start accepted");
    until(&events, "capture.started", |m| m.type_name() == "capture.started", Duration::from_secs(5));
    until(
        &events,
        "capture.progress (samples accumulated)",
        |m| m.type_name() == "capture.progress" && m.corr.as_deref() == Some("take_v1"),
        Duration::from_secs(5),
    );
    client
        .send(Some("take_v1"), Command::CaptureStop { drain: Some(true) })
        .expect("stop accepted");

    // The gate: the machine sits in Draining while the persist worker
    // encodes. This itself pins the fix — the old inline shape never
    // published Draining at all (the whole encode ran inside the stop's
    // command handling), so this wait would time out there.
    wait_for_capture_state(&client, "Draining", Duration::from_secs(5));

    // The probe: a capture.start raced against the encode. Pre-#249 it
    // queued behind the whole persist and only then got its table
    // rejection; now the actor answers from a live loop.
    let probe_started = Instant::now();
    let probed = client.send(Some("take_v1b"), Command::CaptureStart { policy: "dictation".into() });
    let probe_latency = probe_started.elapsed();
    assert!(
        matches!(&probed, Err(Rejection::IllegalInState { state, .. }) if state == "Draining"),
        "a start during the drain is answered with the table rejection, got {probed:?}"
    );
    assert!(
        probe_latency < Duration::from_millis(500),
        "the start took {probe_latency:?} to be rejected — the capture actor appears stalled on the WAV encode"
    );

    // The order-based property: the answer came while the durable commit
    // was still in flight — nothing has landed in the store yet, and no
    // capture.stopped is on the wire (it follows the commit, §4).
    assert!(
        store.landed.lock().unwrap().is_empty(),
        "the v1 persist had already landed when the probe was answered — the probe did not race the encode"
    );
    let mut buffered = Vec::new();
    while let Ok(message) = events.try_recv() {
        buffered.push(message);
    }
    assert!(
        !buffered.iter().any(|m| m.type_name() == "capture.stopped"),
        "capture.stopped preceded the probe's answer: {:?}",
        buffered.iter().map(|m| m.type_name()).collect::<Vec<_>>()
    );

    // The take still completes exactly as before: durable row first,
    // capture.stopped after it, machine Persisted, take usable.
    until(
        &events,
        "capture.stopped (take_v1)",
        |m| m.type_name() == "capture.stopped" && m.corr.as_deref() == Some("take_v1"),
        Duration::from_secs(10),
    );
    wait_for_capture_state(&client, "Persisted", Duration::from_secs(5));
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let landed = store.landed.lock().unwrap().clone();
        if landed.iter().any(|id| id == "j_v1big") {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "the v1 store row never landed: {landed:?}"
        );
        std::thread::sleep(Duration::from_millis(10));
    }
    runtime.shutdown();
}

/// A store whose commits take a fixed while — a deterministic stand-in
/// for the v1 store's encode window, for the interleavings only the
/// off-actor persist makes reachable. `failing` names takes whose
/// commits fail after the delay (the storage-fault injection); `attempts`
/// records every commit that ran, succeeded or not, so tests can wait
/// for a persist to have happened without guessing timings.
struct SlowStore {
    delay: Duration,
    failing: Vec<String>,
    attempts: std::sync::Mutex<Vec<String>>,
    landed: std::sync::Mutex<Vec<(TakeStatus, String)>>,
}

impl SlowStore {
    /// Blocks for the delay, then records the attempt.
    fn slow_attempt(&self, id: &str) {
        std::thread::sleep(self.delay);
        self.attempts.lock().unwrap().push(id.to_string());
    }

    fn waited_for_attempt(&self, id: &str, deadline: Instant) {
        loop {
            if self.attempts.lock().unwrap().iter().any(|attempted| attempted == id) {
                return;
            }
            assert!(
                Instant::now() < deadline,
                "the persist for {id} never ran: {:?}",
                self.attempts.lock().unwrap()
            );
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}

impl CaptureStore for SlowStore {
    fn commit_take(&self, take: &TakeRecord) -> Result<(), String> {
        self.slow_attempt(&take.id);
        if self.failing.iter().any(|id| id == &take.id) {
            return Err("simulated storage fault".to_string());
        }
        self.landed
            .lock()
            .unwrap()
            .push((TakeStatus::Complete, take.id.clone()));
        Ok(())
    }
    fn mark_interrupted(&self, take: &TakeRecord, _note: &str) -> Result<(), String> {
        self.slow_attempt(&take.id);
        self.landed
            .lock()
            .unwrap()
            .push((TakeStatus::Interrupted, take.id.clone()));
        Ok(())
    }
    fn describe(&self) -> String {
        "slow".to_string()
    }
}

/// The one interleaving the off-actor persist newly exposes (issue #249):
/// `capture.abort` has always been table-legal from `Draining`, but the
/// inline encode meant no abort could ever arrive there — it queued
/// behind the whole persist. Now it is answered immediately, the machine
/// returns to Idle, and the already-in-flight durable write still lands:
/// silently (no `capture.*` event is legal from Idle), with the take
/// registered, the route released, and the machine ready for the next
/// take — no wedge, no phantom `capture.stopped`.
#[test]
fn abort_during_a_pending_persist_lands_the_take_silently() {
    let source = FakeCaptureSource::new(vec![]);
    let store = std::sync::Arc::new(SlowStore {
        delay: Duration::from_millis(400),
        failing: Vec::new(),
        attempts: std::sync::Mutex::new(Vec::new()),
        landed: std::sync::Mutex::new(Vec::new()),
    });
    let config = test_config(
        std::sync::Arc::clone(&source),
        starling_runtime::provider::FakeProvider::new(vec![]),
        store.clone(),
        JobLimits {
            max_queued: 8,
            max_concurrent: 1,
            per_route: vec![],
        },
    );
    let (runtime, client) = Runtime::start(config);
    let events = client.subscribe();
    freeze_route(&client, &events);

    source.push(FakeTakeScript::clean());
    client
        .send(Some("take_ab"), Command::CaptureStart { policy: "push-to-talk".into() })
        .expect("start accepted");
    until(&events, "capture.started", |m| m.type_name() == "capture.started", Duration::from_secs(5));
    until(
        &events,
        "capture.progress (samples accumulated)",
        |m| m.type_name() == "capture.progress" && m.corr.as_deref() == Some("take_ab"),
        Duration::from_secs(5),
    );
    client
        .send(Some("take_ab"), Command::CaptureStop { drain: Some(true) })
        .expect("stop accepted");
    wait_for_capture_state(&client, "Draining", Duration::from_secs(5));

    // The abort races the (deterministically) slow persist and must win.
    let abort_started = Instant::now();
    let aborted = client.send(Some("take_ab"), Command::CaptureAbort);
    let abort_latency = abort_started.elapsed();
    aborted.expect("abort is table-legal from Draining while the persist runs");
    assert!(
        abort_latency < Duration::from_millis(200),
        "abort took {abort_latency:?} — the capture actor appears stalled on the persist"
    );
    wait_for_capture_state(&client, "Idle", Duration::from_secs(2));

    // The durable write was already in flight; aborting a finished take
    // cannot un-write it. It lands — with no wire announcement (none is
    // legal from Idle) — and the route release still runs.
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let landed = store.landed.lock().unwrap().clone();
        if landed.iter().any(|(status, id)| *status == TakeStatus::Complete && id == "take_ab") {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "the pending persist never landed after the abort: {landed:?}"
        );
        std::thread::sleep(Duration::from_millis(10));
    }
    wait_for_context_state(&client, "Released", Duration::from_secs(5));
    std::thread::sleep(Duration::from_millis(100));
    let mut drained = Vec::new();
    while let Ok(message) = events.try_recv() {
        drained.push(message);
    }
    assert!(
        !drained.iter().any(|m| m.corr.as_deref() == Some("take_ab")
            && matches!(m.type_name(), "capture.stopped" | "capture.error")),
        "an aborted mid-persist take must not be announced on the wire: {:?}",
        drained.iter().map(|m| m.type_name()).collect::<Vec<_>>()
    );

    // And the machine is not wedged behind the landed-in-the-dark take:
    // the next take runs end to end.
    source.push(FakeTakeScript::clean());
    client
        .send(Some("take_next"), Command::CaptureStart { policy: "push-to-talk".into() })
        .expect("start after the silent persist accepted");
    until(&events, "capture.started (take_next)", |m| m.type_name() == "capture.started", Duration::from_secs(5));
    until(
        &events,
        "capture.progress (take_next)",
        |m| m.type_name() == "capture.progress" && m.corr.as_deref() == Some("take_next"),
        Duration::from_secs(5),
    );
    client
        .send(Some("take_next"), Command::CaptureStop { drain: Some(true) })
        .expect("stop accepted");
    until(
        &events,
        "capture.stopped (take_next)",
        |m| m.type_name() == "capture.stopped" && m.corr.as_deref() == Some("take_next"),
        Duration::from_secs(5),
    );
    wait_for_capture_state(&client, "Persisted", Duration::from_secs(5));
    runtime.shutdown();
}

/// A store whose commits block until the test opens the gate — the hung
/// store write (an fsync on a full disk, say) that the bounded shutdown
/// drain exists for.
struct GatedStore {
    opened: std::sync::Mutex<bool>,
    signal: std::sync::Condvar,
}

impl GatedStore {
    fn new() -> std::sync::Arc<Self> {
        std::sync::Arc::new(GatedStore {
            opened: std::sync::Mutex::new(false),
            signal: std::sync::Condvar::new(),
        })
    }

    /// Lets every blocked (and future) commit through.
    fn open(&self) {
        let mut opened = self.opened.lock().unwrap();
        *opened = true;
        self.signal.notify_all();
    }

    fn hold(&self) {
        let mut opened = self.opened.lock().unwrap();
        while !*opened {
            opened = self.signal.wait(opened).unwrap();
        }
    }
}

impl CaptureStore for GatedStore {
    fn commit_take(&self, _take: &TakeRecord) -> Result<(), String> {
        self.hold();
        Ok(())
    }
    fn mark_interrupted(&self, _take: &TakeRecord, _note: &str) -> Result<(), String> {
        self.hold();
        Ok(())
    }
    fn describe(&self) -> String {
        "gated".to_string()
    }
}

/// Review on #252, defect 1: `Runtime::shutdown` joins the capture actor,
/// and the actor's drain waited for in-flight persists without a bound —
/// a worker wedged in a hung store write kept shutdown from ever
/// returning, and this PR makes that window seconds long by design. The
/// drain is bounded by `persist_drain_timeout` (250 ms here): shutdown
/// returns, names what it abandoned on stderr, and the take's durable
/// journal remains startup recovery's path.
#[test]
fn shutdown_returns_while_a_store_write_hangs() {
    let source = FakeCaptureSource::new(vec![]);
    let capture_source: std::sync::Arc<dyn CaptureSource> = source.clone();
    let store = GatedStore::new();
    let config = RuntimeConfig::default()
        .with_capture_source(capture_source)
        .with_provider(starling_runtime::provider::FakeProvider::new(vec![]))
        .with_capture_store(store.clone())
        .with_capture_config(CaptureConfig {
            journals_dir: std::env::temp_dir().join("starling-runtime-test-journals"),
            poll_interval: Duration::from_millis(10),
            persist_drain_timeout: Duration::from_millis(250),
        })
        .with_jobs_limits(JobLimits {
            max_queued: 8,
            max_concurrent: 1,
            per_route: vec![],
        });
    let (runtime, client) = Runtime::start(config);
    let events = client.subscribe();
    freeze_route(&client, &events);

    source.push(FakeTakeScript::clean());
    client
        .send(Some("take_hang"), Command::CaptureStart { policy: "push-to-talk".into() })
        .expect("start accepted");
    until(&events, "capture.started", |m| m.type_name() == "capture.started", Duration::from_secs(5));
    until(
        &events,
        "capture.progress (samples accumulated)",
        |m| m.type_name() == "capture.progress" && m.corr.as_deref() == Some("take_hang"),
        Duration::from_secs(5),
    );
    client
        .send(Some("take_hang"), Command::CaptureStop { drain: Some(true) })
        .expect("stop accepted");
    wait_for_capture_state(&client, "Draining", Duration::from_secs(5));

    // The persist is gated shut. Shutdown must still return — after the
    // bounded drain, not after the (never-arriving) report.
    let shutdown_started = Instant::now();
    let shutting_down = std::thread::spawn(move || runtime.shutdown());
    let deadline = Instant::now() + Duration::from_secs(3);
    while !shutting_down.is_finished() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(10));
    }
    assert!(
        shutting_down.is_finished(),
        "shutdown never returned while a store write hung — the drain is unbounded"
    );
    assert!(
        shutdown_started.elapsed() < Duration::from_secs(2),
        "shutdown took {:?} with a 250 ms drain bound",
        shutdown_started.elapsed()
    );
    // Cleanup: let the hung worker finish; its report lands on a closed
    // inbox by design (surfaced on stderr, dropped).
    store.open();
    shutting_down.join().expect("shutdown thread");
}

/// Review on #252, defects 2 and 4 (the reproduced swallow): take_a
/// stops, the persist is handed off, the user aborts (table-legal from
/// Draining) and a new take starts and stops — all before take_a's slow
/// report lands. The stale report must land registry-only: emitting its
/// `capture.stopped` would consume the Draining that belongs to take_b,
/// and take_b's own announcement would then be refused as illegal from
/// Persisted — silently swallowed.
#[test]
fn stale_persist_report_lands_nothing_on_the_next_take() {
    let source = FakeCaptureSource::new(vec![]);
    let store = std::sync::Arc::new(SlowStore {
        delay: Duration::from_millis(600),
        failing: Vec::new(),
        attempts: std::sync::Mutex::new(Vec::new()),
        landed: std::sync::Mutex::new(Vec::new()),
    });
    let config = test_config(
        std::sync::Arc::clone(&source),
        starling_runtime::provider::FakeProvider::new(vec![]),
        store.clone(),
        JobLimits {
            max_queued: 8,
            max_concurrent: 1,
            per_route: vec![],
        },
    );
    let (runtime, client) = Runtime::start(config);
    let events = client.subscribe();
    freeze_route(&client, &events);

    // take_a: stopped, then aborted mid-persist, then replaced.
    source.push(FakeTakeScript::clean());
    client
        .send(Some("take_a"), Command::CaptureStart { policy: "push-to-talk".into() })
        .expect("start accepted");
    until(&events, "capture.started", |m| m.type_name() == "capture.started", Duration::from_secs(5));
    until(
        &events,
        "capture.progress (take_a)",
        |m| m.type_name() == "capture.progress" && m.corr.as_deref() == Some("take_a"),
        Duration::from_secs(5),
    );
    client
        .send(Some("take_a"), Command::CaptureStop { drain: Some(true) })
        .expect("stop accepted");
    wait_for_capture_state(&client, "Draining", Duration::from_secs(5));
    client
        .send(Some("take_a"), Command::CaptureAbort)
        .expect("abort is table-legal from Draining while the persist runs");

    // take_b replaces take_a well before the slow report lands.
    source.push(FakeTakeScript::clean());
    client
        .send(Some("take_b"), Command::CaptureStart { policy: "push-to-talk".into() })
        .expect("start after the abort accepted");
    until(
        &events,
        "capture.started (take_b)",
        |m| m.type_name() == "capture.started" && m.corr.as_deref() == Some("take_b"),
        Duration::from_secs(5),
    );
    until(
        &events,
        "capture.progress (take_b)",
        |m| m.type_name() == "capture.progress" && m.corr.as_deref() == Some("take_b"),
        Duration::from_secs(5),
    );
    client
        .send(Some("take_b"), Command::CaptureStop { drain: Some(true) })
        .expect("stop accepted");
    wait_for_capture_state(&client, "Draining", Duration::from_secs(5));

    // take_a's stale report lands during take_b's Draining. With the
    // scoping fix it emits nothing; without it, its capture.stopped
    // consumed the Draining and take_b's own stopped was swallowed (the
    // wait below would time out).
    until(
        &events,
        "capture.stopped (take_b, not swallowed by take_a's stale report)",
        |m| m.type_name() == "capture.stopped" && m.corr.as_deref() == Some("take_b"),
        Duration::from_secs(5),
    );
    wait_for_capture_state(&client, "Persisted", Duration::from_secs(5));
    store.waited_for_attempt("take_a", Instant::now() + Duration::from_secs(5));
    store.waited_for_attempt("take_b", Instant::now() + Duration::from_secs(5));

    // Nothing was ever announced for the aborted take, and both takes'
    // store rows landed — the stale one registry-only, but never dropped.
    std::thread::sleep(Duration::from_millis(100));
    let mut drained = Vec::new();
    while let Ok(message) = events.try_recv() {
        drained.push(message);
    }
    assert!(
        !drained
            .iter()
            .any(|m| m.type_name() == "capture.stopped" && m.corr.as_deref() == Some("take_a")),
        "a stale persist report must not announce its aborted take: {:?}",
        drained.iter().map(|m| m.type_name()).collect::<Vec<_>>()
    );
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let landed = store.landed.lock().unwrap().clone();
        let a = landed.iter().any(|(status, id)| *status == TakeStatus::Complete && id == "take_a");
        let b = landed.iter().any(|(status, id)| *status == TakeStatus::Complete && id == "take_b");
        if a && b {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "both takes' store rows must land (stale registry-only, current announced): {landed:?}"
        );
        std::thread::sleep(Duration::from_millis(10));
    }
    runtime.shutdown();
}

/// Review on #252, defect 2 (the poison variant): a stale report whose
/// commit FAILED used to be judged legal from the *new* take's
/// `Recording` and emitted its fatal `storage_commit_failed` —
/// interrupting the new take, whose stop was then refused outright. The
/// failed stale report must land registry-only, and the live take must
/// complete untouched.
#[test]
fn a_failed_stale_persist_cannot_poison_the_next_take() {
    let source = FakeCaptureSource::new(vec![]);
    let store = std::sync::Arc::new(SlowStore {
        delay: Duration::from_millis(600),
        failing: vec!["take_a".into()],
        attempts: std::sync::Mutex::new(Vec::new()),
        landed: std::sync::Mutex::new(Vec::new()),
    });
    let config = test_config(
        std::sync::Arc::clone(&source),
        starling_runtime::provider::FakeProvider::new(vec![]),
        store.clone(),
        JobLimits {
            max_queued: 8,
            max_concurrent: 1,
            per_route: vec![],
        },
    );
    let (runtime, client) = Runtime::start(config);
    let events = client.subscribe();
    freeze_route(&client, &events);

    // take_a: stopped (commit will fail), aborted mid-persist.
    source.push(FakeTakeScript::clean());
    client
        .send(Some("take_a"), Command::CaptureStart { policy: "push-to-talk".into() })
        .expect("start accepted");
    until(&events, "capture.started", |m| m.type_name() == "capture.started", Duration::from_secs(5));
    until(
        &events,
        "capture.progress (take_a)",
        |m| m.type_name() == "capture.progress" && m.corr.as_deref() == Some("take_a"),
        Duration::from_secs(5),
    );
    client
        .send(Some("take_a"), Command::CaptureStop { drain: Some(true) })
        .expect("stop accepted");
    wait_for_capture_state(&client, "Draining", Duration::from_secs(5));
    client
        .send(Some("take_a"), Command::CaptureAbort)
        .expect("abort is table-legal from Draining while the persist runs");

    // take_b goes live (Recording) and stays recording while take_a's
    // failed report arrives.
    source.push(FakeTakeScript::clean());
    client
        .send(Some("take_b"), Command::CaptureStart { policy: "push-to-talk".into() })
        .expect("start after the abort accepted");
    until(
        &events,
        "capture.started (take_b)",
        |m| m.type_name() == "capture.started" && m.corr.as_deref() == Some("take_b"),
        Duration::from_secs(5),
    );
    until(
        &events,
        "capture.progress (take_b)",
        |m| m.type_name() == "capture.progress" && m.corr.as_deref() == Some("take_b"),
        Duration::from_secs(5),
    );
    store.waited_for_attempt("take_a", Instant::now() + Duration::from_secs(5));

    // The live take completes untouched: its stop is accepted and its
    // capture.stopped lands — no fatal error for take_a poisoned it.
    client
        .send(Some("take_b"), Command::CaptureStop { drain: Some(true) })
        .expect("the live take's stop must still be legal");
    until(
        &events,
        "capture.stopped (take_b)",
        |m| m.type_name() == "capture.stopped" && m.corr.as_deref() == Some("take_b"),
        Duration::from_secs(5),
    );
    wait_for_capture_state(&client, "Persisted", Duration::from_secs(5));
    std::thread::sleep(Duration::from_millis(100));
    let mut drained = Vec::new();
    while let Ok(message) = events.try_recv() {
        drained.push(message);
    }
    assert!(
        !drained
            .iter()
            .any(|m| m.type_name() == "capture.error" && m.corr.as_deref() == Some("take_a")),
        "a failed stale persist must not emit a fatal error against the live take: {:?}",
        drained.iter().map(|m| m.type_name()).collect::<Vec<_>>()
    );
    runtime.shutdown();
}

/// Review on #252, defect 3: the abort path's route release used to ride
/// with the persist report — but the store commit is precisely the slow
/// encode, so the freeze outlived the aborted take by the full persist
/// duration and `context.snapshot` was wedged out of `RouteFrozen`. The
/// release happens at the abort decision point now: the context cycle is
/// free while the persist is still in flight.
#[test]
fn an_aborts_route_release_does_not_wait_for_its_persist() {
    let source = FakeCaptureSource::new(vec![]);
    let store = std::sync::Arc::new(SlowStore {
        delay: Duration::from_millis(600),
        failing: Vec::new(),
        attempts: std::sync::Mutex::new(Vec::new()),
        landed: std::sync::Mutex::new(Vec::new()),
    });
    let config = test_config(
        std::sync::Arc::clone(&source),
        starling_runtime::provider::FakeProvider::new(vec![]),
        store.clone(),
        JobLimits {
            max_queued: 8,
            max_concurrent: 1,
            per_route: vec![],
        },
    );
    let (runtime, client) = Runtime::start(config);
    let events = client.subscribe();
    freeze_route(&client, &events);

    source.push(FakeTakeScript::clean());
    client
        .send(Some("take_c"), Command::CaptureStart { policy: "push-to-talk".into() })
        .expect("start accepted");
    until(&events, "capture.started", |m| m.type_name() == "capture.started", Duration::from_secs(5));
    until(
        &events,
        "capture.progress (take_c)",
        |m| m.type_name() == "capture.progress" && m.corr.as_deref() == Some("take_c"),
        Duration::from_secs(5),
    );
    client
        .send(Some("take_c"), Command::CaptureAbort)
        .expect("abort accepted");

    // The route releases at the decision point — long before the slow
    // persist lands (order-based: nothing has landed yet).
    wait_for_context_state(&client, "Released", Duration::from_secs(2));
    assert!(
        store.landed.lock().unwrap().is_empty(),
        "the route released only after the persist landed — the freeze outlived the aborted take"
    );

    // And the context cycle really is free mid-persist: a fresh snapshot
    // is answered (from RouteFrozen it would be the wedge's rejection).
    client
        .send(Some("ctx-2"), Command::ContextSnapshot { source: "vscode".into() })
        .expect("context.snapshot answered while the abort's persist runs");
    until(
        &events,
        "context.targetSnapshot",
        |m| m.type_name() == "context.targetSnapshot",
        Duration::from_secs(2),
    );

    // The persist still lands afterwards — deferred, not dropped.
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let landed = store.landed.lock().unwrap().clone();
        if landed
            .iter()
            .any(|(status, id)| *status == TakeStatus::Interrupted && id == "take_c")
        {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "the aborted take's persist never landed: {landed:?}"
        );
        std::thread::sleep(Duration::from_millis(10));
    }
    runtime.shutdown();
}

/// pullfrog on #252 (reproduced): an abort arriving while a STOP's persist
/// is in flight finds the take already consumed — the abort's decision-point
/// release must still run there, or the context stays RouteFrozen for the
/// whole persist window.
#[test]
fn an_abort_during_a_stops_persist_still_releases_the_route_at_the_decision_point() {
    let source = FakeCaptureSource::new(vec![]);
    let store = std::sync::Arc::new(SlowStore {
        delay: Duration::from_millis(600),
        failing: Vec::new(),
        attempts: std::sync::Mutex::new(Vec::new()),
        landed: std::sync::Mutex::new(Vec::new()),
    });
    let config = test_config(
        std::sync::Arc::clone(&source),
        starling_runtime::provider::FakeProvider::new(vec![]),
        store.clone(),
        JobLimits {
            max_queued: 8,
            max_concurrent: 1,
            per_route: vec![],
        },
    );
    let (runtime, client) = Runtime::start(config);
    let events = client.subscribe();
    freeze_route(&client, &events);

    // A clean take that STOPs — the machine enters Draining while the
    // slow persist encodes.
    source.push(FakeTakeScript::clean());
    client
        .send(Some("take_s"), Command::CaptureStart { policy: "push-to-talk".into() })
        .expect("start accepted");
    until(&events, "capture.started", |m| m.type_name() == "capture.started", Duration::from_secs(5));
    until(
        &events,
        "capture.progress (take_s)",
        |m| m.type_name() == "capture.progress" && m.corr.as_deref() == Some("take_s"),
        Duration::from_secs(5),
    );
    client
        .send(Some("take_s"), Command::CaptureStop { drain: Some(true) })
        .expect("stop accepted");
    wait_for_capture_state(&client, "Draining", Duration::from_secs(5));

    // The abort lands inside the stop's persist window (table-legal from
    // Draining): the take is already consumed, but the route must release
    // at the decision point — long before the slow store lands anything.
    client
        .send(Some("take_s"), Command::CaptureAbort)
        .expect("abort accepted during the stop's persist");
    wait_for_context_state(&client, "Released", Duration::from_secs(2));
    assert!(
        store.landed.lock().unwrap().is_empty(),
        "the route released only after the persist landed — the freeze outlived \
         the aborted stop for the whole persist window"
    );

    runtime.shutdown();
}

/// Part 4: fatality follows the typed fault origin, never the message
/// text. A device fault whose message happens to mention the journal is
/// fatal (the old substring heuristic let such a take continue from a
/// dead microphone); a real journal fault is surfaced once, non-fatal,
/// and the take completes.
#[test]
fn a_device_fault_whose_text_mentions_the_journal_is_fatal() {
    let source = FakeCaptureSource::new(vec![FakeTakeScript {
        error_after: Some((
            Duration::from_millis(15),
            RecorderFault::Device(
                "device stream error; journal flush pointer invalid".into(),
            ),
        )),
        stop: FakeStop::Clean {
            journal_id: "j_devj".into(),
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
        .send(Some("take_dj"), Command::CaptureStart { policy: "dictation".into() })
        .expect("start accepted");
    let collected = until(
        &events,
        "capture.error",
        |m| m.type_name() == "capture.error",
        Duration::from_secs(5),
    );
    match &collected.last().unwrap().event {
        Event::CaptureError { code, fatal } => {
            assert_eq!(code, "device_stream_lost");
            assert!(*fatal, "a device fault is fatal even when its text says journal");
        }
        other => panic!("expected fatal CaptureError, got {other:?}"),
    }
    wait_for_state(&client, "Interrupted", Duration::from_secs(2));
    runtime.shutdown();
}

#[test]
fn a_journal_fault_is_surfaced_once_non_fatal_and_the_take_completes() {
    let source = FakeCaptureSource::new(vec![FakeTakeScript {
        error_after: Some((
            Duration::from_millis(15),
            RecorderFault::Journal("journal write failed: disk full".into()),
        )),
        stop: FakeStop::Clean {
            journal_id: "j_jfault".into(),
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
        .send(Some("take_j"), Command::CaptureStart { policy: "dictation".into() })
        .expect("start accepted");
    let collected = until(
        &events,
        "capture.error{journal_fault}",
        |m| m.type_name() == "capture.error",
        Duration::from_secs(5),
    );
    let fault = collected
        .iter()
        .find(|m| m.type_name() == "capture.error")
        .unwrap();
    match &fault.event {
        Event::CaptureError { code, fatal } => {
            assert_eq!(code, "journal_fault");
            assert!(!*fatal, "a journal fault is non-fatal: capture continues in memory");
        }
        other => panic!("expected CaptureError, got {other:?}"),
    }

    // The take survives the fault: a clean stop still completes it.
    client
        .send(Some("take_j"), Command::CaptureStop { drain: Some(true) })
        .expect("stop accepted");
    until(
        &events,
        "capture.stopped",
        |m| m.type_name() == "capture.stopped",
        Duration::from_secs(5),
    );
    wait_for_state(&client, "Persisted", Duration::from_secs(2));
    runtime.shutdown();
}
