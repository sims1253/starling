//! The I4 acceptance suite over the real IPC transport (loopback UDS):
//! protocol conformance, peer-auth enforcement, size/rate limits,
//! renderer-kill survival + reconnect, multi-client sequencing, and the
//! slow-consumer posture (the runtime never stalls on one renderer).
//!
//! Every test boots the real host (`server::serve` — lease, endpoint,
//! runtime, threads) and talks to it through the real socket with the
//! real client library (or a raw socket where a transport violation has
//! to be hand-crafted). Nothing here mocks the transport.

#![cfg(unix)] // the Windows named-pipe transport cannot run on this box;
              // see the PR's "Not executed here" section.

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use starling_runtime::machine::capture::{CaptureConfig, V2CaptureStore};
use starling_runtime::machine::Rejection;
use starling_runtime::protocol::replay::{is_directive, MachineReplay};
use starling_runtime::protocol::{Command, Kind as MessageKind, Revision};
use starling_runtime::testing::{FakeCaptureSource, FakeTakeScript};
use starling_runtime::{provider::FakeProvider, Runtime, RuntimeConfig};
use starling_runtime_host::auth::{ExpectUid, PeerPolicy};
use starling_runtime_host::client::{ClientError, HostClient};
use starling_runtime_host::frame::{Frame, FrameReader, TransportErrorCode};
use starling_runtime_host::limits::RateLimit;
use starling_runtime_host::{serve, HostConfig, HostError, HostHandle};

// --------------------------------------------------------------------- //
// Helpers
// --------------------------------------------------------------------- //

fn ipc_config(
    root: &Path,
    source: Arc<FakeCaptureSource>,
    provider: Arc<FakeProvider>,
) -> HostConfig {
    let mut config = HostConfig::new(root, root.join("endpoints"));
    config.runtime = config
        .runtime
        .with_capture_source(source)
        .with_provider(provider)
        .with_capture_store(Arc::new(
            V2CaptureStore::open(root).expect("v2 store opens"),
        ))
        .with_capture_config(CaptureConfig {
            journals_dir: root.join("journals"),
            poll_interval: Duration::from_millis(10),
            ..CaptureConfig::default()
        });
    config
}

fn boot(config: HostConfig) -> (HostHandle, HostClient) {
    let host = serve(config).expect("host serves");
    let client = connect_with_retry(host.socket_path());
    (host, client)
}

fn connect_with_retry(path: &Path) -> HostClient {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        match HostClient::connect(path) {
            Ok(client) => return client,
            Err(err) => {
                if Instant::now() >= deadline {
                    panic!("no host at {} within 5s: {err}", path.display());
                }
                std::thread::sleep(Duration::from_millis(20));
            }
        }
    }
}

/// Collects events until `predicate` matches (or the deadline panics).
fn until(
    client: &HostClient,
    label: &str,
    predicate: impl Fn(&starling_runtime_host::client::EventWire) -> bool,
    deadline: Duration,
) -> Vec<starling_runtime_host::client::EventWire> {
    let mut seen = Vec::new();
    let start = Instant::now();
    loop {
        match client.recv_event_timeout(Duration::from_millis(20)) {
            Ok(event) => {
                let matched = predicate(&event);
                seen.push(event);
                if matched {
                    return seen;
                }
            }
            Err(starling_runtime::channel::RecvError::Timeout) => {
                if start.elapsed() >= deadline {
                    panic!(
                        "timed out waiting for {label}; saw {:?}",
                        seen.iter().map(|e| e.type_name()).collect::<Vec<_>>()
                    );
                }
            }
            Err(other) => panic!("event stream error: {other:?}"),
        }
    }
}

fn revision(base: u64, text: &str) -> Revision {
    Revision {
        rev_id: format!("rev-{base}"),
        base_revision: base,
        source_attempt_ids: vec!["att-1".into()],
        instruction_template_id: "tpl-none".into(),
        text: text.into(),
        status: "candidate".into(),
        provenance: "recognition".into(),
    }
}

/// Freezes the audio route (context snapshot + manual mode) over IPC.
fn freeze_route(client: &HostClient, ctx: &str) {
    client
        .send(
            Some(ctx),
            Command::ContextSnapshot {
                source: "vscode".into(),
            },
        )
        .expect("snapshot accepted");
    until(
        client,
        "context.targetSnapshot",
        |e| e.type_name() == "context.targetSnapshot",
        Duration::from_secs(5),
    );
    client
        .send(
            Some(ctx),
            Command::ModeSet {
                mode: "code-guidance".into(),
                source: starling_runtime::protocol::Manual,
            },
        )
        .expect("mode accepted");
    until(
        client,
        "mode.decision",
        |e| e.type_name() == "mode.decision",
        Duration::from_secs(5),
    );
}

// --------------------------------------------------------------------- //
// Handshake, snapshot, transport defaults
// --------------------------------------------------------------------- //

#[test]
fn hello_reports_owner_identity_and_limits() {
    let root = tempfile::tempdir().unwrap();
    let config = HostConfig::new(root.path(), root.path().join("endpoints"));
    let (mut host, client) = boot(config);

    assert_eq!(client.info.protocol, 1);
    assert_eq!(client.info.owner_id, host.owner_id());
    assert_eq!(client.info.pid, std::process::id());
    assert_eq!(
        client.info.max_frame_bytes as usize,
        starling_runtime_host::frame::DEFAULT_MAX_FRAME_BYTES
    );
    assert_eq!(client.info.rate_max, RateLimit::default().max);

    host.shutdown();
}

#[test]
fn snapshot_travels_round_trip() {
    let root = tempfile::tempdir().unwrap();
    let config = HostConfig::new(root.path(), root.path().join("endpoints"));
    let (mut host, client) = boot(config);

    let snapshot = client.snapshot().expect("snapshot");
    assert_eq!(snapshot["capture"]["state"], "Idle");
    assert_eq!(snapshot["context"]["state"], "Observing");
    assert_eq!(snapshot["jobs"]["state"], "Idle");
    assert_eq!(snapshot["docs"]["state"], "Steady");

    host.shutdown();
}

// --------------------------------------------------------------------- //
// Protocol conformance over the transport
// --------------------------------------------------------------------- //

/// The I0 contract's vendored fixtures (byte-identical copies in the
/// runtime crate's test tree — the frozen corpus this suite borrows
/// rather than re-vendors a third time).
fn fixture_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../runtime/tests/fixtures")
}

fn invalid_fixtures() -> Vec<(String, serde_json::Value)> {
    let dir = fixture_dir().join("invalid");
    let mut fixtures = Vec::new();
    for entry in std::fs::read_dir(&dir).expect("invalid fixtures dir") {
        let path = entry.expect("entry").path();
        if path.extension().is_some_and(|ext| ext == "json") {
            let name = path.file_name().unwrap().to_string_lossy().to_string();
            let value =
                serde_json::from_str(&std::fs::read_to_string(&path).expect("fixture reads"))
                    .expect("fixture parses");
            fixtures.push((name, value));
        }
    }
    fixtures.sort_by(|a, b| a.0.cmp(&b.0));
    fixtures
}

/// Transparency is the claim: for every invalid I0 fixture, sending its
/// command sequence over the socket yields **exactly the receipts an
/// in-process `RuntimeClient` twin produces** — the transport adds no
/// behavior and subtracts none. (Event-direction invalid messages cannot
/// be sent by a client at all; those are skipped here and remain covered
/// by the in-process oracle replay.)
#[test]
fn invalid_fixture_rejections_over_ipc_match_the_in_process_runtime() {
    assert_eq!(invalid_fixtures().len(), 9, "nine invalid fixtures");
    for (name, fixture) in invalid_fixtures() {
        let root = tempfile::tempdir().unwrap();
        let source = FakeCaptureSource::new(vec![]);
        let provider = FakeProvider::new(vec![]);
        let (mut host, client) = boot(ipc_config(
            root.path(),
            Arc::clone(&source),
            Arc::clone(&provider),
        ));
        let twin_config = RuntimeConfig::default()
            .with_capture_source(source)
            .with_provider(provider);
        let (runtime, twin) = Runtime::start(twin_config);

        let messages = fixture["messages"]
            .as_array()
            .expect("messages array")
            .iter()
            .filter(|message| !is_directive(message))
            .filter(|message| {
                starling_runtime::protocol::replay::message_kind(
                    message
                        .get("type")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or(""),
                ) == Some(MessageKind::Command)
            })
            .cloned()
            .collect::<Vec<_>>();
        assert!(
            !messages.is_empty(),
            "{name}: fixture must carry at least one command"
        );

        for message in &messages {
            let over_ipc = match client.send_raw(message.clone()) {
                Ok(receipt) => Ok(receipt),
                Err(ClientError::Rejected(rejection)) => Err(rejection),
                Err(err) => panic!("{name}: transport failed: {err}"),
            };
            let in_process = twin.send_raw(message.clone());
            assert_eq!(
                over_ipc, in_process,
                "{name}: IPC receipt must equal the in-process receipt"
            );
        }

        // The unknown-version fixture additionally owes its NACK on the
        // event stream (corr = the rejected id) — the contract's one
        // wire-visible rejection.
        if fixture["replay"].as_str() == Some("unknown_version") {
            let id = messages[0]["id"].as_str().expect("id");
            until(
                &client,
                "runtime.nack",
                |e| e.type_name() == "runtime.nack" && e.corr() == Some(id),
                Duration::from_secs(5),
            );
        }

        runtime.shutdown();
        host.shutdown();
    }
}

#[test]
fn unsupported_version_is_nacked_over_ipc() {
    let root = tempfile::tempdir().unwrap();
    let source = FakeCaptureSource::new(vec![]);
    let provider = FakeProvider::new(vec![]);
    let (mut host, client) = boot(ipc_config(root.path(), source, provider));

    let envelope = serde_json::json!({
        "v": 2, "id": "cmd_900", "ts": "2026-09-20T10:09:00Z",
        "type": "capture.stop", "payload": { "drain": true }
    });
    match client.send_raw(envelope) {
        Err(ClientError::Rejected(Rejection::UnsupportedVersion { id })) => {
            assert_eq!(id.as_deref(), Some("cmd_900"));
        }
        other => panic!("expected a refusal, got {other:?}"),
    }
    let nack = until(
        &client,
        "runtime.nack",
        |e| e.type_name() == "runtime.nack",
        Duration::from_secs(5),
    )
    .pop()
    .unwrap();
    assert_eq!(nack.corr(), Some("cmd_900"));
    assert_eq!(nack.payload()["reason"], "unsupported_version");

    host.shutdown();
}

#[test]
fn stale_seq_is_rejected_over_the_raw_path() {
    let root = tempfile::tempdir().unwrap();
    let source = FakeCaptureSource::new(vec![]);
    let provider = FakeProvider::new(vec![]);
    let (mut host, client) = boot(ipc_config(root.path(), source, provider));

    // setLimits is legal in every state and outcome-free: two sends on
    // one stream expose pure seq mechanics. seq 5, then 5 again — the
    // router's per-stream frontier must refuse the repeat with the typed
    // rejection, exactly as in-process.
    let limits_at = |id: &str, seq: u64| {
        serde_json::json!({
            "v": 1, "id": id, "ts": "2026-09-20T10:00:00Z",
            "corr": "limits", "seq": seq, "type": "jobs.setLimits",
            "payload": { "maxQueued": 8, "maxConcurrent": 2 }
        })
    };
    client
        .send_raw(limits_at("cmd_first", 5))
        .expect("seq 5 accepted");
    match client.send_raw(limits_at("cmd_second", 5)) {
        Err(ClientError::Rejected(Rejection::SeqNotMonotonic { detail })) => {
            assert!(detail.contains("seq 5"), "{detail}")
        }
        other => panic!("expected a seq refusal, got {other:?}"),
    }

    host.shutdown();
}

/// The full live capture/context/jobs/docs walk over the socket, with
/// every command envelope reconstructed from the host-assigned `seq`
/// and every observed event interleaved in arrival order, replayed
/// through the oracle port: what the wire actually carried is
/// oracle-legal — conformance over the real transport, not just next to
/// it.
#[test]
fn a_live_take_over_ipc_replays_green_through_the_oracle() {
    let root = tempfile::tempdir().unwrap();
    let source = FakeCaptureSource::new(vec![FakeTakeScript::clean()]);
    let provider = FakeProvider::new(vec![starling_runtime::provider::FakeJob::completes_with(
        "hello over ipc",
    )]);
    let (mut host, client) = boot(ipc_config(root.path(), source, provider));

    // Per-machine traces of wire envelopes, commands and events
    // interleaved in the order they crossed the socket.
    let mut traces: Vec<(&'static str, Vec<serde_json::Value>)> = [
        ("capture", Vec::new()),
        ("context", Vec::new()),
        ("jobs", Vec::new()),
        ("docs", Vec::new()),
    ]
    .into_iter()
    .collect();

    fn command_of(client: &HostClient, corr: &str, command: Command) -> serde_json::Value {
        let type_name = command.type_name();
        let payload = command.payload_value();
        let (_, seq) = client.send_and_seq(Some(corr), command).expect("accepted");
        serde_json::json!({
            "v": 1, "id": format!("c{}", seq.unwrap_or(0)), "ts": "2026-09-20T10:00:00Z",
            "corr": corr, "seq": seq,
            "type": type_name, "payload": payload
        })
    }

    fn observe(
        traces: &mut Vec<(&'static str, Vec<serde_json::Value>)>,
        client: &HostClient,
        label: &str,
        predicate: impl Fn(&starling_runtime_host::client::EventWire) -> bool,
    ) {
        let seen = until(client, label, predicate, Duration::from_secs(5));
        for event in seen {
            // `mode.*` events are the context machine's (the protocol
            // routes ModeSet to "context" despite the type prefix).
            let machine = match event.type_name().split('.').next().unwrap_or_default() {
                "mode" => "context",
                other => other,
            };
            for (name, trace) in traces.iter_mut() {
                if *name == machine {
                    trace.push(event.0.clone());
                }
            }
        }
    }

    // context: snapshot + manual mode.
    let envelope = command_of(
        &client,
        "ctx-1",
        Command::ContextSnapshot {
            source: "vscode".into(),
        },
    );
    traces
        .iter_mut()
        .find(|(n, _)| *n == "context")
        .unwrap()
        .1
        .push(envelope);
    observe(&mut traces, &client, "context.targetSnapshot", |e| {
        e.type_name() == "context.targetSnapshot"
    });
    let envelope = command_of(
        &client,
        "ctx-1",
        Command::ModeSet {
            mode: "code-guidance".into(),
            source: starling_runtime::protocol::Manual,
        },
    );
    traces
        .iter_mut()
        .find(|(n, _)| *n == "context")
        .unwrap()
        .1
        .push(envelope);
    observe(&mut traces, &client, "mode.decision", |e| {
        e.type_name() == "mode.decision"
    });

    // capture: a take.
    let envelope = command_of(
        &client,
        "take_ipc",
        Command::CaptureStart {
            policy: "push-to-talk".into(),
        },
    );
    traces
        .iter_mut()
        .find(|(n, _)| *n == "capture")
        .unwrap()
        .1
        .push(envelope);
    observe(&mut traces, &client, "capture.progress", |e| {
        e.type_name() == "capture.progress" && e.corr() == Some("take_ipc")
    });
    let envelope = command_of(
        &client,
        "take_ipc",
        Command::CaptureStop { drain: Some(true) },
    );
    traces
        .iter_mut()
        .find(|(n, _)| *n == "capture")
        .unwrap()
        .1
        .push(envelope);
    observe(&mut traces, &client, "capture.stopped", |e| {
        e.type_name() == "capture.stopped"
    });

    // jobs: submit the take on the frozen route. The scheduler's
    // internal walk (Queued → Dispatched → Loading → Recognizing) is
    // modeled by the oracle's advance directives, exactly as the
    // in-process suite does.
    let envelope = command_of(
        &client,
        "job-1",
        Command::JobsSubmit {
            capture_ref: "take_ipc".into(),
            route: "local-default".into(),
            budget: "standard".into(),
        },
    );
    traces
        .iter_mut()
        .find(|(n, _)| *n == "jobs")
        .unwrap()
        .1
        .push(envelope);
    observe(&mut traces, &client, "jobs.completed", |e| {
        e.type_name() == "jobs.completed"
    });
    {
        let trace = &mut traces.iter_mut().find(|(n, _)| *n == "jobs").unwrap().1;
        let queued_at = trace
            .iter()
            .position(|message| message["type"] == "jobs.queued")
            .expect("jobs.queued observed");
        let advances = ["Dispatched", "Loading", "Recognizing"]
            .iter()
            .map(|state| serde_json::json!({ "$advance": state }))
            .collect::<Vec<_>>();
        for (offset, advance) in advances.into_iter().enumerate() {
            trace.insert(queued_at + 1 + offset, advance);
        }
    }

    // docs: head update referencing the attempt, then a turn.
    let envelope = command_of(
        &client,
        "doc-1",
        Command::DocsUpdateHead {
            doc_id: "notes".into(),
            expected_base: 0,
            new_revision: revision(1, "Hello over ipc."),
        },
    );
    traces
        .iter_mut()
        .find(|(n, _)| *n == "docs")
        .unwrap()
        .1
        .push(envelope);
    observe(&mut traces, &client, "docs.headUpdated", |e| {
        e.type_name() == "docs.headUpdated"
    });
    let envelope = command_of(
        &client,
        "doc-1",
        Command::DocsAppendTurn {
            doc_id: "notes".into(),
            take_ref: "take_ipc".into(),
        },
    );
    traces
        .iter_mut()
        .find(|(n, _)| *n == "docs")
        .unwrap()
        .1
        .push(envelope);
    observe(&mut traces, &client, "docs.turnAppended", |e| {
        e.type_name() == "docs.turnAppended"
    });

    drop(client);
    for (machine, trace) in traces {
        let mut replay = MachineReplay::new(machine, None).expect("spec");
        for message in &trace {
            replay
                .feed(message)
                .unwrap_or_else(|violation| panic!("{machine} wire trace: {violation}"));
        }
        replay.finish().unwrap_or_else(|violation| {
            panic!("{machine} wire trace ends unresolved: {violation}")
        });
        assert!(!trace.is_empty(), "{machine} observed something");
    }

    host.shutdown();
}

// --------------------------------------------------------------------- //
// The renderer-kill acceptance
// --------------------------------------------------------------------- //

/// #220's acceptance, over the real socket: connect, drive a capture,
/// kill the connection mid-take, reconnect, and prove the take survives
/// (durable in storage v2) and the session resumes (the new client stops
/// the take, sees `capture.stopped`, and the machine lands in
/// `Persisted`). Also proves the host serves sequential clients.
#[test]
fn renderer_kill_mid_take_survives_and_reconnects() {
    let root = tempfile::tempdir().unwrap();
    let source = FakeCaptureSource::new(vec![FakeTakeScript::clean()]);
    let provider = FakeProvider::new(vec![]);
    let (mut host, renderer) = boot(ipc_config(root.path(), source, provider));

    freeze_route(&renderer, "ctx-1");
    renderer
        .send(
            Some("take_kill"),
            Command::CaptureStart {
                policy: "push-to-talk".into(),
            },
        )
        .expect("start accepted");
    until(
        &renderer,
        "capture.started",
        |e| e.type_name() == "capture.started",
        Duration::from_secs(5),
    );
    let pre_kill = until(
        &renderer,
        "capture.progress",
        |e| e.type_name() == "capture.progress" && e.corr() == Some("take_kill"),
        Duration::from_secs(5),
    );
    assert!(
        pre_kill
            .iter()
            .any(|e| e.payload()["ackSamples"].as_u64().unwrap_or(0) > 0),
        "audio was acknowledged before the kill"
    );

    // Kill the renderer. Not graceful, not asked: the process holding
    // the socket simply goes away (drop shuts the fd — a renderer kill).
    drop(renderer);

    // The runtime keeps recording (Mode B: a dead renderer costs nothing
    // but the projection). A new renderer connects and resumes.
    let successor = connect_with_retry(host.socket_path());
    let snapshot = successor.snapshot().expect("snapshot after reconnect");
    assert_eq!(
        snapshot["capture"]["state"], "Recording",
        "the take must still be live after the renderer died"
    );

    successor
        .send(
            Some("take_kill"),
            Command::CaptureStop { drain: Some(true) },
        )
        .expect("stop accepted over the new connection");
    let stopped = until(
        &successor,
        "capture.stopped",
        |e| e.type_name() == "capture.stopped" && e.corr() == Some("take_kill"),
        Duration::from_secs(10),
    )
    .into_iter()
    .find(|e| e.type_name() == "capture.stopped")
    .unwrap();
    let acknowledged = stopped.payload()["acknowledgedSamples"].as_u64().unwrap();
    assert!(
        acknowledged > 0,
        "acknowledged samples survived the renderer kill"
    );
    assert!(!stopped.payload()["journalId"].as_str().unwrap().is_empty());

    // And the next take works — the session resumed, the host healthy.
    let resumed = successor.snapshot().expect("snapshot after stop");
    assert_eq!(resumed["capture"]["state"], "Persisted");

    // Durable proof: the row is in storage v2, not in any process.
    let store = starling_dictation::store_v2::StoreV2::open(root.path()).expect("store opens");
    let page = store.list_records(0, 10).expect("list");
    assert_eq!(page.total, 1, "exactly one take persisted");
    match &page.records[0] {
        starling_dictation::store_v2::ListedCapture::Capture(listing) => {
            assert_eq!(
                listing.record.status,
                starling_dictation::store_v2::CaptureStatus::Complete
            );
        }
        starling_dictation::store_v2::ListedCapture::Damaged(damaged) => {
            panic!("the surviving take must be healthy: {damaged:?}")
        }
    }

    drop(successor);
    host.shutdown();
}

/// The host keeps serving while a client is between connections is the
/// trivial case; the sharper one is two clients at once, one driving and
/// one observing — the fan-out every renderer relies on.
#[test]
fn two_clients_fan_out_events() {
    let root = tempfile::tempdir().unwrap();
    let source = FakeCaptureSource::new(vec![]);
    let provider = FakeProvider::new(vec![]);
    let (mut host, driver) = boot(ipc_config(root.path(), source, provider));

    let observer = connect_with_retry(host.socket_path());
    driver
        .send(
            Some("doc-1"),
            Command::DocsUpdateHead {
                doc_id: "notes".into(),
                expected_base: 0,
                new_revision: revision(1, "Seen by two renderers."),
            },
        )
        .expect("update accepted");
    for client in [&driver, &observer] {
        until(
            client,
            "docs.headUpdated",
            |e| e.type_name() == "docs.headUpdated" && e.corr() == Some("doc-1"),
            Duration::from_secs(5),
        );
    }

    drop(observer);
    drop(driver);
    host.shutdown();
}

/// Worker lifetime is host lifetime, not renderer lifetime (§1 Mode B:
/// "supervised engine workers attach to the runtime, not the UI"): a
/// submitted job keeps running after its submitting client dies, and its
/// completion is observable by another client.
#[test]
fn a_job_survives_its_submitting_clients_death() {
    let root = tempfile::tempdir().unwrap();
    let source = FakeCaptureSource::new(vec![FakeTakeScript::clean()]);
    let mut slow_job = starling_runtime::provider::FakeJob::completes_with("late but durable");
    slow_job.work_ms = 600;
    let provider = FakeProvider::new(vec![slow_job]);
    let (mut host, renderer) = boot(ipc_config(root.path(), source, provider));

    freeze_route(&renderer, "ctx-1");
    renderer
        .send(
            Some("take_j"),
            Command::CaptureStart {
                policy: "push-to-talk".into(),
            },
        )
        .expect("start accepted");
    until(
        &renderer,
        "capture.started",
        |e| e.type_name() == "capture.started",
        Duration::from_secs(5),
    );
    until(
        &renderer,
        "capture.progress",
        |e| e.type_name() == "capture.progress",
        Duration::from_secs(5),
    );
    renderer
        .send(Some("take_j"), Command::CaptureStop { drain: Some(true) })
        .expect("stop accepted");
    until(
        &renderer,
        "capture.stopped",
        |e| e.type_name() == "capture.stopped",
        Duration::from_secs(10),
    );

    renderer
        .send(
            Some("job-late"),
            Command::JobsSubmit {
                capture_ref: "take_j".into(),
                route: "local-default".into(),
                budget: "standard".into(),
            },
        )
        .expect("submit accepted");

    // The submitting renderer dies mid-recognition; a peer client (and
    // the runtime's supervised worker) carries on.
    drop(renderer);
    let peer = connect_with_retry(host.socket_path());
    until(
        &peer,
        "jobs.completed",
        |e| e.type_name() == "jobs.completed",
        Duration::from_secs(10),
    );

    drop(peer);
    host.shutdown();
}

// --------------------------------------------------------------------- //
// The slow-consumer posture
// --------------------------------------------------------------------- //

/// A renderer that stops reading is closed (`slow_consumer`) while the
/// runtime and the other client continue untouched — the Mode B answer
/// to "the UI is not a durability dependency".
#[test]
fn a_stalled_client_is_closed_and_the_runtime_continues() {
    let root = tempfile::tempdir().unwrap();
    let source = FakeCaptureSource::new(vec![FakeTakeScript::clean()]);
    let provider = FakeProvider::new(vec![]);
    let mut config = ipc_config(root.path(), source, provider);
    config = config.with_outbound_capacity(8);
    // A 1ms poll makes the fake produce ~1000 progress events a second:
    // the default ~212 KiB unix send buffer (what the host's writer
    // thread fills against) is exhausted in a couple of seconds, which
    // is what backs the queue up to its cap.
    config.runtime = std::mem::take(&mut config.runtime).with_capture_config(CaptureConfig {
        journals_dir: root.path().join("journals"),
        poll_interval: Duration::from_millis(1),
        ..CaptureConfig::default()
    });
    let (mut host, healthy) = boot(config);

    // A raw socket that never reads, with a shrunken receive buffer so
    // the kernel path fills in test time rather than after 200 KiB of
    // events.
    use std::os::fd::AsRawFd;
    let stalled = std::os::unix::net::UnixStream::connect(host.socket_path()).unwrap();
    unsafe {
        let size: libc::c_int = 2048;
        libc::setsockopt(
            stalled.as_raw_fd(),
            libc::SOL_SOCKET,
            libc::SO_RCVBUF,
            &size as *const libc::c_int as *const libc::c_void,
            std::mem::size_of::<libc::c_int>() as libc::socklen_t,
        );
    }
    let mut stalled = stalled;

    freeze_route(&healthy, "ctx-1");
    healthy
        .send(
            Some("take_slow"),
            Command::CaptureStart {
                policy: "push-to-talk".into(),
            },
        )
        .expect("start accepted");
    until(
        &healthy,
        "capture.started",
        |e| e.type_name() == "capture.started",
        Duration::from_secs(5),
    );

    // Progress events (~100/s at a 10ms poll) pour into both sockets;
    // the stalled one fills its kernel buffer, its outbound queue, and
    // is closed by the host. Meanwhile the healthy client sees progress.
    until(
        &healthy,
        "capture.progress",
        |e| e.type_name() == "capture.progress",
        Duration::from_secs(5),
    );

    // Phase 1 — do not touch the stalled socket at all. At ~1000
    // progress events a second, the kernel's per-socket buffer (~40 KiB
    // against a shrunk receiver, measured) fills within a second; the
    // host's writer parks, the 8-frame outbound queue fills behind it,
    // and the event pump evicts the connection. Meanwhile the healthy
    // client keeps receiving (its queue is drained here only so it
    // stays out of the picture — it is not the client under test).
    let evict_by = Instant::now() + Duration::from_secs(4);
    while Instant::now() < evict_by {
        while let Ok(_event) = healthy.try_recv_event() {}
        // EOF cannot be observed yet: ~40 KiB of backlog sits ahead of
        // it. Poll the healthy path for continued service instead; the
        // stalled side is proven by phase 2.
        if let Ok(last) = healthy.try_recv_event() {
            let _ = last;
        }
        std::thread::sleep(Duration::from_millis(25));
    }
    // Phase 2 — the eviction happened while nothing drained the socket;
    // draining it now can only reveal the backlog and then the EOF the
    // host's close planted behind it. A healthy (never-evicted)
    // connection would instead keep producing data forever.
    stalled
        .set_read_timeout(Some(Duration::from_millis(50)))
        .unwrap();
    let drained_deadline = Instant::now() + Duration::from_secs(6);
    loop {
        let mut scratch = [0u8; 4096];
        match std::io::Read::read(&mut stalled, &mut scratch) {
            // Drained the backlog; the stream was closed behind it.
            Ok(0) => break,
            Ok(_) => continue,
            Err(err)
                if err.kind() == std::io::ErrorKind::WouldBlock
                    || err.kind() == std::io::ErrorKind::TimedOut =>
            {
                assert!(
                    Instant::now() < drained_deadline,
                    "the stalled socket drained no EOF in 6s — the host never closed it"
                );
                // Keep the healthy client alive-side checked too.
                while let Ok(_event) = healthy.try_recv_event() {}
            }
            Err(err) => panic!("stalled client read failed: {err}"),
        }
    }
    // Reaching Ok(0) at all is the assertion: a never-evicted stalled
    // connection keeps producing data (the capture is still running) and
    // never returns EOF.

    // The runtime never noticed: the take stops cleanly for the healthy
    // client, and the host still answers snapshots.
    healthy
        .send(
            Some("take_slow"),
            Command::CaptureStop { drain: Some(true) },
        )
        .expect("stop accepted after the stalled peer was evicted");
    until(
        &healthy,
        "capture.stopped",
        |e| e.type_name() == "capture.stopped",
        Duration::from_secs(10),
    );
    assert_eq!(healthy.snapshot().unwrap()["capture"]["state"], "Persisted");

    drop(healthy);
    host.shutdown();
}

// --------------------------------------------------------------------- //
// Peer authentication over the real socket
// --------------------------------------------------------------------- //

/// The kernel-supplied credentials of a real connection are exactly the
/// process's own: the same-user policy admits them. This is the
/// positive, on-the-wire proof that `SO_PEERCRED` was actually read
/// (an unread credential would be `None` and fail closed).
#[test]
fn same_user_credentials_pass_over_the_socket() {
    let root = tempfile::tempdir().unwrap();
    let source = FakeCaptureSource::new(vec![]);
    let provider = FakeProvider::new(vec![]);
    let policy: Arc<dyn PeerPolicy> = Arc::new(ExpectUid(current_test_uid()));
    let mut config = ipc_config(root.path(), source, provider);
    config.peer_policy = policy;
    let (mut host, client) = boot(config);
    assert_eq!(client.info.pid, std::process::id());
    drop(client);
    host.shutdown();
}

/// A policy expecting a different uid refuses the real connection: the
/// enforcement path (kernel credential → policy → close) runs against
/// genuine credentials, no second OS user needed.
#[test]
fn foreign_uid_is_refused_on_the_socket() {
    let root = tempfile::tempdir().unwrap();
    let source = FakeCaptureSource::new(vec![]);
    let provider = FakeProvider::new(vec![]);
    let policy: Arc<dyn PeerPolicy> = Arc::new(ExpectUid(current_test_uid() ^ 0x7fff));
    let mut config = ipc_config(root.path(), source, provider);
    config.peer_policy = policy;
    let mut host = serve(config).expect("host serves");

    match HostClient::connect(host.socket_path()) {
        Err(ClientError::Protocol(detail)) => {
            assert!(detail.contains("auth_failed"), "{detail}");
            assert!(
                detail.contains("peer uid"),
                "the refusal states the credential evidence: {detail}"
            );
        }
        Ok(client) => {
            drop(client);
            panic!("expected an auth refusal, the connection was admitted");
        }
        Err(other) => panic!("expected an auth refusal, got {other}"),
    }
    // The refused connection is gone; the host still serves the next,
    // equally-foreign-by-policy client identically (closed), and its
    // socket stays up.
    assert!(HostClient::connect(host.socket_path()).is_err());

    host.shutdown();
}

#[cfg(unix)]
fn current_test_uid() -> u32 {
    unsafe { libc::geteuid() as u32 }
}

// --------------------------------------------------------------------- //
// Framing limits over the raw socket
// --------------------------------------------------------------------- //

fn raw_connect(host: &HostHandle) -> std::os::unix::net::UnixStream {
    std::os::unix::net::UnixStream::connect(host.socket_path()).unwrap()
}

fn read_transport_error(
    stream: &mut std::os::unix::net::UnixStream,
) -> (TransportErrorCode, String) {
    let mut reader = FrameReader::new(&mut *stream, usize::MAX);
    // Skip the hello.
    loop {
        match reader.read_frame() {
            Ok(Frame::Hello { .. }) => continue,
            Ok(Frame::TransportError { code, detail }) => return (code, detail),
            Ok(other) => panic!("expected a transport error, got {other:?}"),
            Err(err) => panic!("expected a transport error, read {err:?}"),
        }
    }
}

fn write_raw_frame(stream: &mut std::os::unix::net::UnixStream, body: &[u8]) {
    use std::io::Write;
    stream
        .write_all(&(body.len() as u32).to_be_bytes())
        .unwrap();
    stream.write_all(body).unwrap();
    stream.flush().unwrap();
}

#[test]
fn oversized_frames_are_refused_without_being_read() {
    let root = tempfile::tempdir().unwrap();
    let source = FakeCaptureSource::new(vec![]);
    let provider = FakeProvider::new(vec![]);
    let mut config = ipc_config(root.path(), source, provider);
    config = config.with_max_frame_bytes(2048);
    let mut host = serve(config).expect("host serves");

    let mut stream = raw_connect(&host);
    // Header claims 4096 bytes; we deliberately never send a body. If
    // the host buffered the body before checking, it would wait (and
    // this test would hang or see a socket read), not refuse promptly.
    use std::io::Write;
    stream.write_all(&4096u32.to_be_bytes()).unwrap();
    stream.flush().unwrap();

    let (code, detail) = read_transport_error(&mut stream);
    assert_eq!(code, TransportErrorCode::MessageTooLarge);
    assert!(detail.contains("4096"), "{detail}");
    // The connection is closed after the error.
    let mut scratch = [0u8; 16];
    let deadline = Instant::now() + Duration::from_secs(2);
    loop {
        match std::io::Read::read(&mut stream, &mut scratch) {
            Ok(0) => break,
            Ok(_) => continue,
            Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                assert!(Instant::now() < deadline, "socket never closed");
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(err) => panic!("read after refusal: {err}"),
        }
    }

    host.shutdown();
}

#[test]
fn malformed_frames_close_with_a_clear_error() {
    let root = tempfile::tempdir().unwrap();
    let source = FakeCaptureSource::new(vec![]);
    let provider = FakeProvider::new(vec![]);
    let (mut host, _client) = boot(ipc_config(root.path(), source, provider));

    let mut stream = raw_connect(&host);
    write_raw_frame(&mut stream, b"this is not json");
    let (code, detail) = read_transport_error(&mut stream);
    assert_eq!(code, TransportErrorCode::MalformedFrame);
    assert!(detail.contains("frame"), "{detail}");

    host.shutdown();
}

#[test]
fn host_to_client_frames_from_a_client_are_a_protocol_violation() {
    let root = tempfile::tempdir().unwrap();
    let source = FakeCaptureSource::new(vec![]);
    let provider = FakeProvider::new(vec![]);
    let (mut host, _client) = boot(ipc_config(root.path(), source, provider));

    let mut stream = raw_connect(&host);
    // A well-formed frame in the wrong direction: hello is host→client.
    let frame = serde_json::to_vec(&Frame::Hello {
        protocol: 1,
        owner_id: "l_fake".into(),
        pid: 1,
        max_frame_bytes: 1,
        rate_max: 1,
        rate_window_ms: 1,
    })
    .unwrap();
    write_raw_frame(&mut stream, &frame);
    let (code, _) = read_transport_error(&mut stream);
    assert_eq!(code, TransportErrorCode::ProtocolViolation);

    host.shutdown();
}

#[test]
fn rate_limit_closes_a_flooding_connection() {
    let root = tempfile::tempdir().unwrap();
    let source = FakeCaptureSource::new(vec![]);
    let provider = FakeProvider::new(vec![]);
    let mut config = ipc_config(root.path(), source, provider);
    config = config.with_command_rate(RateLimit::new(3, Duration::from_secs(2)));
    let (mut host, client) = boot(config);

    let limits = || {
        Command::JobsSetLimits(starling_runtime::protocol::JobLimits {
            max_queued: 4,
            max_concurrent: 1,
            per_route: vec![],
        })
    };
    client.send(None, limits()).expect("frame 1 in budget");
    client.send(None, limits()).expect("frame 2 in budget");
    client.send(None, limits()).expect("frame 3 in budget");
    // Frame 4 inside one 2s window: over budget → refused + closed.
    match client.send(None, limits()) {
        Err(ClientError::Closed(reason)) => {
            assert!(reason.contains("rate_limited"), "{reason}");
        }
        other => panic!("expected a rate-limited close, got {other:?}"),
    }
    assert!(client.is_closed());

    // The host itself is untouched: a fresh client is served.
    let fresh = connect_with_retry(host.socket_path());
    fresh
        .send(None, limits())
        .expect("a fresh client is in a fresh window");
    drop(fresh);
    host.shutdown();
}

#[test]
fn connection_cap_answers_too_many_connections() {
    let root = tempfile::tempdir().unwrap();
    let source = FakeCaptureSource::new(vec![]);
    let provider = FakeProvider::new(vec![]);
    let mut config = ipc_config(root.path(), source, provider);
    config.max_connections = 2;
    let mut host = serve(config).expect("host serves");

    let first = connect_with_retry(host.socket_path());
    let second = connect_with_retry(host.socket_path());
    let _ = second;
    match HostClient::connect(host.socket_path()) {
        Err(ClientError::Protocol(detail)) => {
            assert!(detail.contains("too_many_connections"), "{detail}");
        }
        Ok(client) => {
            drop(client);
            panic!("expected a connection-cap refusal, the connection was admitted");
        }
        Err(other) => panic!("expected a connection-cap refusal, got {other}"),
    }
    drop(first);
    // A slot freed is a slot served.
    let third = connect_with_retry(host.socket_path());
    drop(third);
    host.shutdown();
}

// --------------------------------------------------------------------- //
// Host lifecycle
// --------------------------------------------------------------------- //

#[test]
fn a_second_host_on_the_same_root_is_a_client_not_a_competitor() {
    let root = tempfile::tempdir().unwrap();
    let source = FakeCaptureSource::new(vec![]);
    let provider = FakeProvider::new(vec![]);
    let (mut host, client) = boot(ipc_config(root.path(), source, provider));

    // Same data root: the lease must make this one a client. (The
    // endpoint directory is per-config; ownership is the data root's, so
    // a second serve on the same root refuses.)
    let source = FakeCaptureSource::new(vec![]);
    let provider = FakeProvider::new(vec![]);
    let second = serve(ipc_config(root.path(), source, provider));
    match second {
        Err(HostError::OwnerLive { owner_id, .. }) => {
            assert_eq!(
                owner_id,
                host.owner_id(),
                "the live owner is the first host"
            );
        }
        Err(other) => panic!("expected OwnerLive, got {other}"),
        Ok(_) => panic!("expected OwnerLive, a second host won the lease"),
    }
    // Never two owners: the first host still serves.
    client
        .send(
            None,
            Command::JobsSetLimits(starling_runtime::protocol::JobLimits {
                max_queued: 4,
                max_concurrent: 1,
                per_route: vec![],
            }),
        )
        .expect("first host unaffected");
    drop(client);
    host.shutdown();
}

#[test]
fn graceful_shutdown_releases_the_lease_and_the_endpoint() {
    let root = tempfile::tempdir().unwrap();
    let source = FakeCaptureSource::new(vec![]);
    let provider = FakeProvider::new(vec![]);
    let (mut host, client) = boot(ipc_config(root.path(), source, provider));

    let socket = host.socket_path().to_path_buf();
    host.shutdown();
    assert!(!socket.exists(), "the endpoint is removed on shutdown");
    assert!(
        HostClient::connect(&socket).is_err(),
        "nothing serves after shutdown"
    );
    // The bye reached the client, which knows why it closed.
    assert!(client.is_closed());
    assert!(
        client.close_reason().contains("goodbye"),
        "{}",
        client.close_reason()
    );

    // The lease was released: a fresh host on the same root owns it now.
    let source = FakeCaptureSource::new(vec![]);
    let provider = FakeProvider::new(vec![]);
    let mut successor =
        serve(ipc_config(root.path(), source, provider)).expect("re-serve after shutdown");
    assert_ne!(successor.owner_id(), host.owner_id());
    successor.shutdown();
}
