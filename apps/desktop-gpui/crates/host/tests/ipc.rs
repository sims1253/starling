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
use starling_runtime_host::frame::{Frame, FrameError, FrameReader, TransportErrorCode};
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
    // No exact-count pin: the corpus is a sibling crate's frozen test
    // tree; adding a fixture there must not break this test from the
    // outside. Empty would mean the borrow broke — that is the pin.
    let fixtures = invalid_fixtures();
    assert!(!fixtures.is_empty(), "invalid fixture corpus is empty");
    for (name, fixture) in fixtures {
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

    /// Monotonic fallback id for the (unexpected) seq-less send: a
    /// static counter can never collide, where a constant "c0" could
    /// (two seq-less commands, or a replay that checks id uniqueness).
    fn command_of(client: &HostClient, corr: &str, command: Command) -> serde_json::Value {
        static ANON: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let type_name = command.type_name();
        let payload = command.payload_value();
        let (_, seq) = client.send_and_seq(Some(corr), command).expect("accepted");
        let id = match seq {
            Some(seq) => format!("c{seq}"),
            None => format!(
                "c-anon-{}",
                ANON.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            ),
        };
        serde_json::json!({
            "v": 1, "id": id, "ts": "2026-09-20T10:00:00Z",
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

/// Events the fan-out must offer before a stalled peer is considered
/// provably overflowed: an order of magnitude more frame bytes than the
/// test's 8-frame outbound queue plus both kernel socket buffers (the
/// shrunk receiver and the host-side sender) can absorb — so the
/// conclusion holds at any event rate, not at one measured one.
const FILL_EVENTS: usize = 4096;

/// Failure bound for the fill phase. Generous on purpose (slow CI): the
/// fill argument is the event count above, never this clock.
const FILL_BUDGET: Duration = Duration::from_secs(60);

/// Failure bound for draining the backlog down to the host-planted EOF.
const DRAIN_BUDGET: Duration = Duration::from_secs(12);

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
    // A 1ms poll keeps progress events flowing; how *fast* does not
    // matter — the fill trigger below counts the events the fan-out
    // offered (an observable condition), not seconds on the wall, so a
    // slow box merely takes longer instead of failing.
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
    // SAFETY: `size` is a valid, initialized `c_int` on the stack for
    // the duration of the call, and `stalled` is an open fd. SO_RCVBUF
    // is advisory (Linux doubles it, rmem_min clamps it) — the assert
    // keeps a silent no-op shrink from quietly invalidating the fill
    // timing assumptions below.
    let rc = unsafe {
        let size: libc::c_int = 2048;
        libc::setsockopt(
            stalled.as_raw_fd(),
            libc::SOL_SOCKET,
            libc::SO_RCVBUF,
            &size as *const libc::c_int as *const libc::c_void,
            std::mem::size_of::<libc::c_int>() as libc::socklen_t,
        )
    };
    assert_eq!(
        rc, 0,
        "failed to shrink the receive buffer; the timing below relies on it"
    );
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

    // Progress events pour into both connections (one fan-out); the
    // healthy one drains them here, the stalled one absorbs what its
    // kernel buffers can hold and is closed by the host.
    until(
        &healthy,
        "capture.progress",
        |e| e.type_name() == "capture.progress",
        Duration::from_secs(5),
    );

    // Phase 1 — do not touch the stalled socket at all. The fill
    // trigger is an observable condition, not a wall-clock guess: every
    // event this healthy client drains was also offered to the stalled
    // connection (same fan-out), so once FILL_EVENTS have been produced
    // — an order of magnitude more frame bytes than the 8-frame queue
    // plus both kernel socket buffers can absorb, whatever the event
    // rate — the stalled side's writer must have parked and the event
    // pump evicted the connection. A slower box simply takes longer;
    // FILL_BUDGET is the failure bound, never the fill assumption.
    let fill_deadline = Instant::now() + FILL_BUDGET;
    let mut offered = 0usize;
    while offered < FILL_EVENTS && Instant::now() < fill_deadline {
        while let Ok(_event) = healthy.try_recv_event() {
            offered += 1;
        }
        std::thread::sleep(Duration::from_millis(25));
    }
    assert!(
        offered >= FILL_EVENTS,
        "only {offered} events in {FILL_BUDGET:?} — this box produces events too \
         slowly for the fill argument to hold"
    );
    // Phase 2 — the eviction happened while nothing drained the socket;
    // draining it now can only reveal the backlog and then the EOF the
    // host's close planted behind it. A healthy (never-evicted)
    // connection would instead keep producing data forever.
    stalled
        .set_read_timeout(Some(Duration::from_millis(50)))
        .unwrap();
    let drained_deadline = Instant::now() + DRAIN_BUDGET;
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
                    "the stalled socket drained no EOF in {DRAIN_BUDGET:?} — the host never closed it"
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

/// [`write_raw_frame`] for floods that expect the host to tear the
/// connection down mid-stream: `false` when a write failed (the socket
/// ended under us), which callers treat as "already answered".
fn write_raw_frame_lossy(stream: &mut std::os::unix::net::UnixStream, body: &[u8]) -> bool {
    use std::io::Write;
    let wire = (body.len() as u32).to_be_bytes();
    stream.write_all(&wire).is_ok() && stream.write_all(body).is_ok()
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
    // The connection is closed after the error. The read poll makes the
    // WouldBlock branch below actually reachable (the stream otherwise
    // blocks forever on the first read instead of failing at the
    // deadline — a regression would hang, not fail).
    stream
        .set_read_timeout(Some(Duration::from_millis(50)))
        .unwrap();
    let mut scratch = [0u8; 16];
    let deadline = Instant::now() + Duration::from_secs(2);
    loop {
        match std::io::Read::read(&mut stream, &mut scratch) {
            Ok(0) => break,
            Ok(_) => continue,
            Err(err)
                if err.kind() == std::io::ErrorKind::WouldBlock
                    || err.kind() == std::io::ErrorKind::TimedOut =>
            {
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

// ---------------------------------------------------------------------
// Startup reconciliation (the host is the §4 recovering owner)
// ---------------------------------------------------------------------

/// The §4 recovery caller: the host is the Mode B owner, so a crashed
/// predecessor's staging journal (a verified-prefix take that never
/// finalized — what a SIGKILL mid-take leaves) is salvaged by *host
/// startup*, before any client connects, and the report is surfaced on
/// the handle. #257's client-mode deferral names the owner as the
/// salvager; this is that owner running it.
#[test]
fn startup_reconcile_salvages_a_crashed_predecessors_staging_journal() {
    use starling_dictation::store_v2::{StoreV2, TakeMeta};

    let root = tempfile::tempdir().unwrap();
    // Fabricate the crash residue exactly as store_v2's own kill tests
    // do: frames + an fsynced boundary (the verified prefix), then an
    // unconfirmed tail, then "crash" (drop without finalize/promote).
    let crashed_id = {
        let store = StoreV2::open(root.path()).expect("store opens");
        let mut take = store
            .begin_take(TakeMeta::for_device("mic"))
            .expect("begin the interrupted take");
        let confirmed: Vec<f32> = (0..800).map(|i| i as f32 * 0.01).collect();
        take.append_frames(&confirmed).expect("append");
        take.write_boundary().expect("boundary");
        take.append_frames(&[0.5f32; 120]).expect("unconfirmed tail");
        let id = take.id().to_string();
        drop(take); // "crash"
        id
    };
    assert!(
        root.path().join("staging").join(format!("{crashed_id}.sj")).exists(),
        "the residue is in place before the host starts"
    );

    let source = FakeCaptureSource::new(vec![]);
    let provider = FakeProvider::new(vec![]);
    let (mut host, client) = boot(ipc_config(root.path(), source, provider));

    // The report says the take was recovered with a torn tail.
    let report = host.startup_reconciliation();
    assert_eq!(report.recovered_torn.len(), 1, "{report:?}");
    assert_eq!(report.recovered_torn[0].id, crashed_id);
    assert!(report.recovered_torn[0].torn_tail_bytes > 0);

    // And the salvage is durable, not just reported: sealed audio in
    // audio/, an interrupted row, staging empty.
    let store = StoreV2::open(root.path()).expect("store opens for verification");
    let record = store
        .get_capture(&crashed_id)
        .expect("get")
        .expect("the interrupted row exists");
    assert_eq!(
        record.status,
        starling_dictation::store_v2::CaptureStatus::Interrupted
    );
    assert!(
        root.path().join("audio").join(format!("{crashed_id}.sj")).exists(),
        "promoted out of staging into audio/"
    );

    // The host serves normally on top of the recovered state.
    assert_eq!(client.snapshot().unwrap()["capture"]["state"], "Idle");

    drop(client);
    host.shutdown();
}

/// The clean root is the normal case: startup reconcile runs owner-mode
/// and finds nothing (the report is surfaced, empty).
#[test]
fn startup_reconcile_on_a_clean_root_reports_nothing() {
    let root = tempfile::tempdir().unwrap();
    let source = FakeCaptureSource::new(vec![]);
    let provider = FakeProvider::new(vec![]);
    let (mut host, _client) = boot(ipc_config(root.path(), source, provider));
    let report = host.startup_reconciliation();
    assert!(!report.has_findings(), "{report:?}");
    host.shutdown();
}

// ---------------------------------------------------------------------
// Bounded shutdown against a wedged peer
// ---------------------------------------------------------------------

/// A raw client that never reads, with a shrunk receive buffer, parks
/// the host's writer mid-frame while its outbound queue still has room
/// (the runtime is idle — no further event arrives to evict the
/// connection). Graceful shutdown must still complete: after the drain
/// bound the host kills the socket, joins its threads, releases the
/// lease and removes the endpoint. Before the bound this join hung
/// forever on exactly this shape.
#[test]
fn shutdown_completes_despite_a_writer_parked_on_a_silent_peer() {
    let root = tempfile::tempdir().unwrap();
    let source = FakeCaptureSource::new(vec![]);
    let provider = FakeProvider::new(vec![]);
    let (mut host, healthy) = boot(ipc_config(root.path(), source, provider));
    drop(healthy); // only the wedged connection remains

    // Raw socket with a 2 KiB receive buffer (armed before connect so
    // the kernel allocates it small), then never read from again.
    use std::os::fd::AsRawFd;
    use std::os::unix::net::UnixStream;
    let mut wedged = UnixStream::connect(host.socket_path()).unwrap();
    {
        // SAFETY: `size` is a valid, initialized `c_int` for the
        // duration of the call, and the socket is an open fd. SO_RCVBUF
        // is a hint (the kernel doubles and clamps it) — the assert
        // keeps a silent no-op from undermining the parking below.
        let size: libc::c_int = 2048;
        let rc = unsafe {
            libc::setsockopt(
                wedged.as_raw_fd(),
                libc::SOL_SOCKET,
                libc::SO_RCVBUF,
                &size as *const libc::c_int as *const libc::c_void,
                std::mem::size_of::<libc::c_int>() as libc::socklen_t,
            )
        };
        assert_eq!(rc, 0, "failed to shrink the receive buffer");
    }

    // A burst of snapshot requests: each reply is queued for the writer.
    // The unix sender-side buffer (~212 KiB) plus the shrunk receive
    // buffer absorbs only the first few hundred KiB of replies, so the
    // writer parks mid-write with the queue (capacity 1024) still far
    // under its cap — no delivery failure, no eviction, just a wedged
    // connection nobody rescues.
    let request =
        serde_json::to_vec(&Frame::GetSnapshot { req: "wedged".into() }).expect("serializes");
    for _ in 0..512 {
        write_raw_frame(&mut wedged, &request);
    }
    // Give the host's reader a moment to answer them all.
    std::thread::sleep(Duration::from_millis(500));

    let socket = host.socket_path().to_path_buf();
    let started = Instant::now();
    host.shutdown();
    let elapsed = started.elapsed();

    // The bound ended the drain (the writer was parked for the whole
    // window), and shutdown finished in bounded time overall.
    assert!(
        elapsed >= Duration::from_secs(2),
        "the drain window elapsed ({elapsed:?})"
    );
    assert!(
        elapsed < Duration::from_secs(20),
        "shutdown completed within the bound ({elapsed:?})"
    );
    // And it completed *fully*: endpoint removed, lease released (a
    // successor owns the root), nothing left serving.
    assert!(!socket.exists(), "the endpoint is removed despite the wedge");
    let source = FakeCaptureSource::new(vec![]);
    let provider = FakeProvider::new(vec![]);
    let mut successor =
        serve(ipc_config(root.path(), source, provider)).expect("the lease was released");
    successor.shutdown();
}

// ---------------------------------------------------------------------
// Receipts must not wait behind an undrained event stream
// ---------------------------------------------------------------------

#[test]
fn receipts_flow_while_the_application_stops_draining_events() {
    let root = tempfile::tempdir().unwrap();
    let source = FakeCaptureSource::new(vec![FakeTakeScript::clean()]);
    let provider = FakeProvider::new(vec![]);
    let mut config = ipc_config(root.path(), source, provider);
    config.runtime = std::mem::take(&mut config.runtime).with_capture_config(CaptureConfig {
        journals_dir: root.path().join("journals"),
        poll_interval: Duration::from_millis(1),
        ..CaptureConfig::default()
    });
    let (mut host, client) = boot(config);

    freeze_route(&client, "ctx-drain");
    client
        .send(
            Some("take_drain"),
            Command::CaptureStart {
                policy: "push-to-talk".into(),
            },
        )
        .expect("start accepted");
    until(
        &client,
        "capture.started",
        |e| e.type_name() == "capture.started",
        Duration::from_secs(5),
    );

    // Stop draining events entirely while the runtime keeps producing
    // (~1000 progress events a second). The client-side backlog grows;
    // the host side stays healthy because the reader keeps consuming the
    // socket. Two seconds ≈ 2000 events — under the channel and backlog
    // caps, so the connection must stay honestly healthy.
    std::thread::sleep(Duration::from_secs(2));

    // The receipt arrives immediately — not after a 10 s reply timeout.
    assert!(!client.is_closed(), "no starvation-induced close");
    let sent_at = Instant::now();
    client
        .send(
            Some("take_drain"),
            Command::CaptureStop { drain: Some(true) },
        )
        .expect("the receipt arrived while events went undrained");
    assert!(
        sent_at.elapsed() < Duration::from_secs(2),
        "the receipt waited {}s behind undrained events",
        sent_at.elapsed().as_secs_f32()
    );

    // And draining resumes cleanly: the backlog (not just new events)
    // keeps flowing — the delayed events were retained, not dropped.
    until(
        &client,
        "capture.stopped",
        |e| e.type_name() == "capture.stopped",
        Duration::from_secs(10),
    );
    drop(client);
    host.shutdown();
}

// ---------------------------------------------------------------------
// Command shape validation and outbound-overflow close
// ---------------------------------------------------------------------

/// A command envelope that is not an object, or an object without a
/// string `id`, is refused with `malformed_frame` instead of routed (the
/// command would execute while its receipt could match no pending
/// request — the client would sit out its reply timeout for an executed
/// command).
#[test]
fn a_command_envelope_without_a_string_id_is_refused() {
    let root = tempfile::tempdir().unwrap();
    let source = FakeCaptureSource::new(vec![]);
    let provider = FakeProvider::new(vec![]);
    let (mut host, _client) = boot(ipc_config(root.path(), source, provider));

    for bad in [
        serde_json::json!([1, 2, 3]),
        serde_json::json!("a string"),
        serde_json::json!({ "v": 1, "type": "jobs.setLimits" }),
        serde_json::json!({ "v": 1, "id": 7, "type": "jobs.setLimits" }),
    ] {
        let mut stream = raw_connect(&host);
        let frame = serde_json::to_vec(&Frame::Command { envelope: bad }).unwrap();
        write_raw_frame(&mut stream, &frame);
        let (code, detail) = read_transport_error(&mut stream);
        assert_eq!(code, TransportErrorCode::MalformedFrame, "{detail}");
    }

    host.shutdown();
}

/// An outbound queue that overflows closes the connection — the
/// close-on-overflow machinery the snapshot and receipt paths share. A
/// flood of snapshot replies against a non-reading peer (shrunk receive
/// buffer) parks the host's writer mid-reply; the next reply cannot be
/// queued, and the connection is killed outright — socket and all, so a
/// writer parked in write_all can never wedge the connection open. No
/// explanation frame is owed through a full kernel buffer (a peer that
/// stopped reading cannot be handed one), so the deterministic
/// assertion is the prompt close itself: the client learns the
/// connection ended — never a silent wedge, and never a bare reply
/// timeout for a command that already ran.
#[test]
fn an_outbound_queue_overflow_closes_with_slow_consumer() {
    let root = tempfile::tempdir().unwrap();
    let source = FakeCaptureSource::new(vec![]);
    let provider = FakeProvider::new(vec![]);
    let mut config = ipc_config(root.path(), source, provider);
    config = config.with_outbound_capacity(2);
    let mut host = serve(config).expect("host serves");

    use std::os::fd::AsRawFd;
    let mut stalled = std::os::unix::net::UnixStream::connect(host.socket_path()).unwrap();
    {
        // SAFETY: as in the shutdown test — valid c_int, open fd; the
        // assert keeps the shrink honest.
        let size: libc::c_int = 2048;
        let rc = unsafe {
            libc::setsockopt(
                stalled.as_raw_fd(),
                libc::SOL_SOCKET,
                libc::SO_RCVBUF,
                &size as *const libc::c_int as *const libc::c_void,
                std::mem::size_of::<libc::c_int>() as libc::socklen_t,
            )
        };
        assert_eq!(rc, 0, "failed to shrink the receive buffer");
    }
    stalled
        .set_read_timeout(Some(Duration::from_millis(50)))
        .unwrap();
    // The connection was served before it was flooded: read the hello
    // (the only deterministic proof of service — after the close, a
    // reset can discard everything still queued, so nothing later can
    // stand in for it).
    let hello_by = Instant::now() + Duration::from_secs(5);
    loop {
        let outcome = {
            let mut reader = FrameReader::new(&mut stalled, usize::MAX);
            match reader.read_frame() {
                Ok(Frame::Hello { .. }) => Ok(()),
                // Idle poll slice before the host's accept+greet lands.
                Err(FrameError::Io(err))
                    if err.kind() == std::io::ErrorKind::WouldBlock
                        || err.kind() == std::io::ErrorKind::TimedOut =>
                {
                    Err::<(), std::io::Error>(err)
                }
                other => panic!("expected the hello first, got {other:?}"),
            }
        };
        match outcome {
            Ok(()) => break,
            Err(_) if Instant::now() < hello_by => continue,
            Err(err) => panic!("no hello from the host: {err}"),
        }
    }
    let request =
        serde_json::to_vec(&Frame::GetSnapshot { req: "flood".into() }).expect("serializes");
    // The flood must decisively exceed every kernel buffer in front of
    // the host's writer: 512 snapshot replies are a few hundred KiB
    // against a ~212 KiB sender buffer plus the shrunk receive window,
    // so the writer parks mid-reply and the 2-frame queue overflows on
    // the very next reply — the eviction under test. (A short flood is
    // nondeterministic: a runner whose socket buffers happen to hold it
    // all never parks the writer, never overflows, and never closes.)
    // A failed write mid-flood is the early branch — the host already
    // ended the connection, which is exactly the assertion.
    let mut sent = 0usize;
    for _ in 0..512 {
        if !write_raw_frame_lossy(&mut stalled, &request) {
            break;
        }
        sent += 1;
    }
    assert!(sent > 0, "the flood never reached the host");

    // The flood parks the writer mid-reply; the next reply cannot be
    // queued and the connection is torn down (socket and all — the
    // parked writer must be unblocked, not left wedged). The assertion
    // is that the host ENDS the connection — as EOF, or as ECONNRESET /
    // EPIPE (the close can arrive while our unread flood still sits in
    // the host's receive queue, which the kernel reports as a reset
    // instead of a clean EOF; a reset discards whatever was still
    // queued, which is why the hello above is the proof of service).
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        assert!(
            Instant::now() < deadline,
            "the wedged connection was never closed in 15s"
        );
        let mut chunk = [0u8; 4096];
        match std::io::Read::read(&mut stalled, &mut chunk) {
            Ok(0) => break,                                  // clean EOF
            Ok(_) => continue,                               // late replies leaving the kernel
            Err(err)
                if err.kind() == std::io::ErrorKind::WouldBlock
                    || err.kind() == std::io::ErrorKind::TimedOut =>
            {
                continue // idle poll slice
            }
            Err(err)
                if err.kind() == std::io::ErrorKind::ConnectionReset
                    || err.kind() == std::io::ErrorKind::BrokenPipe =>
            {
                break
            }
            Err(err) => panic!("reading the wedged connection's end: {err}"),
        }
    }
    // Reaching any of the three ends is the assertion: the hello proved
    // the connection served, and the close — not a wedge, not a bare
    // reply timeout — is what a stopped reader is owed. Any explanation
    // frame is best-effort by design (see the terminate/close split in
    // server.rs) and not asserted.

    host.shutdown();
}
