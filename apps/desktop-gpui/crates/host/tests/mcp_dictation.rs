//! The agent-dictation ask surface over the real IPC transport and the
//! real host, with the scripted fake capture source and fake provider.
//! `started_takes` on the fake source is the microphone.

use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use starling_runtime::channel::RecvError;
use starling_runtime::machine::capture::{
    CaptureConfig, CaptureSession, CaptureSource, CaptureStore, TakeRecord, V2CaptureStore,
};
use starling_runtime::machine::context::{ContextProvider, StubContextProvider};
use starling_runtime::machine::Rejection;
use starling_runtime::protocol::{Command, TargetSnapshotData};
use starling_runtime::provider::{FakeJob, FakeProvider};
use starling_runtime::testing::{FakeCaptureSource, FakeTakeScript};
use starling_runtime_host::agent::ALLOWLIST_FILE;
use starling_runtime_host::client::{ClientError, HostClient, UiWire};
use starling_runtime_host::frame::{AskOutcome, NoAnswerReason};
use starling_runtime_host::{serve, HostConfig, HostHandle};

const CLIENT: &str = "claude-code";
const TOKEN: &str = "tok-1";

fn allowlist_at(root: &Path) -> std::path::PathBuf {
    let path = root.join(ALLOWLIST_FILE);
    std::fs::write(
        &path,
        format!(r#"{{"version":1,"clients":[{{"name":"{CLIENT}","token":"{TOKEN}"}}]}}"#),
    )
    .unwrap();
    path
}

fn host_config(
    root: &Path,
    allowlist: Option<std::path::PathBuf>,
    source: Arc<FakeCaptureSource>,
    provider: Arc<FakeProvider>,
) -> HostConfig {
    let mut config = HostConfig::new(root, root.join("endpoints"))
        .with_agent_allowlist(allowlist)
        .with_insecure_test_app_role();
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

fn connect(path: &Path) -> HostClient {
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

/// An allowlisted agent connection (hello done).
fn agent(path: &Path) -> HostClient {
    let client = connect(path);
    let welcomed = client.agent_hello(CLIENT, TOKEN).expect("allowlisted");
    assert_eq!(welcomed, CLIENT);
    client
}

fn boot(
    root: &Path,
    source: Arc<FakeCaptureSource>,
    provider: Arc<FakeProvider>,
) -> (HostHandle, std::path::PathBuf) {
    serve_at(host_config(
        root,
        Some(allowlist_at(root)),
        source,
        provider,
    ))
}

fn serve_at(config: HostConfig) -> (HostHandle, std::path::PathBuf) {
    let host = serve(config).expect("host serves");
    let path = host.socket_path().to_path_buf();
    (host, path)
}

fn until_closed(client: &HostClient, what: &str) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while !client.is_closed() {
        assert!(Instant::now() < deadline, "the host never refused {what}");
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// Waits for an ask result with this req (or panics with everything
/// that arrived instead).
fn until_ask(client: &HostClient, req: &str) -> AskOutcome {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        match client.recv_ask_timeout(Duration::from_millis(20)) {
            Ok(result) if result.req == req => return result.outcome,
            Ok(other) => panic!(
                "ask {req}: got a result for {} first: {:?}",
                other.req, other.outcome
            ),
            Err(RecvError::Timeout) => {
                assert!(Instant::now() < deadline, "timed out waiting for ask {req}");
            }
            Err(RecvError::Closed) => panic!("ask stream closed while waiting for {req}"),
        }
    }
}

/// Waits for a prompt frame matching `predicate` (or panics).
fn until_ui(client: &HostClient, label: &str, predicate: impl Fn(&UiWire) -> bool) -> UiWire {
    let deadline = Instant::now() + Duration::from_secs(10);
    let mut seen = Vec::new();
    loop {
        match client.recv_ui_timeout(Duration::from_millis(20)) {
            Ok(frame) => {
                let matched = predicate(&frame);
                seen.push(format!("{frame:?}"));
                if matched {
                    return frame;
                }
            }
            Err(RecvError::Timeout) => {
                assert!(
                    Instant::now() < deadline,
                    "timed out waiting for {label}; saw {seen:?}"
                );
            }
            Err(RecvError::Closed) => panic!("ui stream closed while waiting for {label}"),
        }
    }
}

/// Waits for a runtime event matching `predicate` (events flow to every
/// client; the app side observes the capture path here).
fn until_event(
    client: &HostClient,
    label: &str,
    predicate: impl Fn(&starling_runtime_host::client::EventWire) -> bool,
) {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        match client.recv_event_timeout(Duration::from_millis(20)) {
            Ok(event) if predicate(&event) => return,
            Ok(_) => {}
            Err(RecvError::Timeout) => {
                assert!(
                    Instant::now() < deadline,
                    "timed out waiting for event {label}"
                );
            }
            Err(RecvError::Closed) => panic!("event stream closed waiting for {label}"),
        }
    }
}

/// The capture machine's state, via the runtime snapshot any client can
/// request.
fn capture_state(client: &HostClient) -> String {
    client
        .snapshot()
        .expect("snapshot")
        .get("capture")
        .and_then(|view| view.get("state"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default()
        .to_string()
}

fn ask_id_of(frame: &UiWire) -> String {
    match frame {
        UiWire::Show { ask_id, .. } | UiWire::Hide { ask_id, .. } => ask_id.clone(),
    }
}

fn one_clean_take() -> Arc<FakeCaptureSource> {
    FakeCaptureSource::new(vec![FakeTakeScript::clean()])
}

#[test]
fn a_spoken_answer_flows_back_as_the_tool_result() {
    let root = tempfile::tempdir().unwrap();
    let source = FakeCaptureSource::new(vec![FakeTakeScript::clean(), FakeTakeScript::clean()]);
    let provider = FakeProvider::new(vec![FakeJob::completes_with("Take the first option.")]);
    let (mut host, path) = boot(root.path(), source.clone(), provider);

    let app = connect(&path);
    let agent = agent(&path);

    agent
        .ask_user(
            "1",
            &["Should I fix the parser or the writer?".into()],
            30_000,
        )
        .expect("ask sent");

    // The prompt reaches the app with the questions intact.
    let shown = until_ui(&app, "ShowPrompt", |frame| {
        matches!(frame, UiWire::Show { .. })
    });
    let UiWire::Show { questions, .. } = &shown else {
        unreachable!()
    };
    assert_eq!(
        questions,
        &["Should I fix the parser or the writer?".to_string()]
    );

    // Before the ack: no capture has been started.
    assert!(
        source.started_takes.lock().unwrap().is_empty(),
        "the gate must hold the mic until the app acks"
    );

    app.prompt_ack(&ask_id_of(&shown), true).unwrap();
    until_event(&app, "capture.started", |e| {
        e.type_name() == "capture.started"
    });
    assert_eq!(source.started_takes.lock().unwrap().len(), 1);

    // The user finishes speaking; the host stops, persists, transcribes.
    app.prompt_done(&ask_id_of(&shown)).unwrap();
    until_event(&app, "jobs.completed", |e| {
        e.type_name() == "jobs.completed"
    });

    match until_ask(&agent, "1") {
        AskOutcome::Answered { text, backend } => {
            assert_eq!(text, "Take the first option.");
            assert!(!backend.is_empty(), "the jobs machine names its backend");
        }
        other => panic!("expected the spoken answer, got {other:?}"),
    }
    // The overlay is cleared on every resolution.
    until_ui(&app, "HidePrompt", |frame| {
        matches!(frame, UiWire::Hide { .. })
    });

    host.shutdown();
}

#[test]
fn no_ack_within_the_bound_fails_without_opening_the_mic() {
    let root = tempfile::tempdir().unwrap();
    let source = one_clean_take();
    let provider = FakeProvider::new(vec![]);
    let (mut host, path) = boot(root.path(), source.clone(), provider);

    let app = connect(&path);
    let agent = agent(&path);
    // A timeout below the ack bound clamps the bound to it.
    let asked = Instant::now();
    agent
        .ask_user("1", &["Ready?".into()], 1_500)
        .expect("ask sent");
    until_ui(&app, "ShowPrompt", |frame| {
        matches!(frame, UiWire::Show { .. })
    });

    match until_ask(&agent, "1") {
        AskOutcome::Error { code, message } => {
            assert_eq!(code, "no_prompt_ack", "{message}");
            assert!(message.contains("never"), "{message}");
        }
        other => panic!("expected the visibility-gate failure, got {other:?}"),
    }
    assert!(
        asked.elapsed() >= Duration::from_millis(1_400),
        "the prompt got its whole bound, not part of it: {:?}",
        asked.elapsed()
    );
    assert!(
        source.started_takes.lock().unwrap().is_empty(),
        "an un-acked prompt must never open the microphone"
    );
    assert_eq!(capture_state(&app), "Idle");

    host.shutdown();
}

#[test]
fn capture_starts_only_after_the_app_acks_the_prompt() {
    // The positive control for the gate, with the timing asserted
    // directly: the mic opens strictly after the ack (never before).
    let root = tempfile::tempdir().unwrap();
    let source = one_clean_take();
    let provider = FakeProvider::new(vec![FakeJob::completes_with("yes")]);
    let (mut host, path) = boot(root.path(), source.clone(), provider);

    let app = connect(&path);
    let agent = agent(&path);
    agent.ask_user("1", &["Ready?".into()], 20_000).unwrap();
    let shown = until_ui(&app, "ShowPrompt", |frame| {
        matches!(frame, UiWire::Show { .. })
    });

    // Give the broker every chance to misbehave, then assert it did
    // not: 250 ms of an un-acked prompt must not open the mic.
    std::thread::sleep(Duration::from_millis(250));
    assert!(source.started_takes.lock().unwrap().is_empty());
    assert_eq!(capture_state(&app), "Idle");

    app.prompt_ack(&ask_id_of(&shown), true).unwrap();
    until_event(&app, "capture.started", |e| {
        e.type_name() == "capture.started"
    });
    assert_eq!(source.started_takes.lock().unwrap().len(), 1);

    // Wind the take down cleanly.
    app.prompt_done(&ask_id_of(&shown)).unwrap();
    match until_ask(&agent, "1") {
        AskOutcome::Answered { text, .. } => assert_eq!(text, "yes"),
        other => panic!("expected the answer, got {other:?}"),
    }

    host.shutdown();
}

#[test]
fn an_ask_without_any_app_connection_fails_fast() {
    let root = tempfile::tempdir().unwrap();
    let source = one_clean_take();
    let provider = FakeProvider::new(vec![]);
    let (mut host, path) = boot(root.path(), source.clone(), provider);

    // No app connects; only the agent.
    let agent = agent(&path);
    agent
        .ask_user("1", &["Anyone there?".into()], 20_000)
        .unwrap();
    match until_ask(&agent, "1") {
        AskOutcome::Error { code, message } => assert_eq!(code, "no_app", "{message}"),
        other => panic!("expected the fast no-listener failure, got {other:?}"),
    }
    assert!(source.started_takes.lock().unwrap().is_empty());

    host.shutdown();
}

#[test]
fn concurrent_asks_queue_one_prompt_at_a_time() {
    let root = tempfile::tempdir().unwrap();
    let source = FakeCaptureSource::new(vec![FakeTakeScript::clean(), FakeTakeScript::clean()]);
    // The fake provider pops its script from the end (`Vec::pop`), so
    // the entries read back-to-front: the first submitted job sees the
    // last entry.
    let provider = FakeProvider::new(vec![
        FakeJob::completes_with("second answer"),
        FakeJob::completes_with("first answer"),
    ]);
    let (mut host, path) = boot(root.path(), source.clone(), provider);

    let app = connect(&path);
    let agent = agent(&path);
    // Two concurrent asks from the same agent (the MCP layer permits
    // concurrent tools/call); they must serialize, never overlap.
    agent.ask_user("a", &["First?".into()], 30_000).unwrap();
    agent.ask_user("b", &["Second?".into()], 30_000).unwrap();

    let first = until_ui(&app, "the first prompt", |frame| {
        matches!(frame, UiWire::Show { .. })
    });
    assert_eq!(
        match &first {
            UiWire::Show { questions, .. } => questions.first().map(String::as_str),
            _ => None,
        },
        Some("First?")
    );

    // While the first is un-resolved, the second prompt must not be
    // shown (serialized prompts, never overlapping captures).
    std::thread::sleep(Duration::from_millis(250));
    assert!(
        app.try_recv_ui().is_err(),
        "no second prompt before the first resolves"
    );

    app.prompt_ack(&ask_id_of(&first), true).unwrap();
    until_event(&app, "capture.started", |e| {
        e.type_name() == "capture.started"
    });
    app.prompt_done(&ask_id_of(&first)).unwrap();
    match until_ask(&agent, "a") {
        AskOutcome::Answered { text, .. } => assert_eq!(text, "first answer"),
        other => panic!("first ask: {other:?}"),
    }
    // Only now does the second prompt appear.
    let second = until_ui(&app, "the second prompt", |frame| {
        matches!(frame, UiWire::Show { .. })
    });
    assert!(
        ask_id_of(&second) != ask_id_of(&first),
        "each ask has its own broker-scoped id"
    );
    app.prompt_ack(&ask_id_of(&second), true).unwrap();
    until_event(&app, "second capture.started", |e| {
        e.type_name() == "capture.started"
    });
    app.prompt_done(&ask_id_of(&second)).unwrap();
    match until_ask(&agent, "b") {
        AskOutcome::Answered { text, .. } => assert_eq!(text, "second answer"),
        other => panic!("second ask: {other:?}"),
    }
    // One mic open per take, serialized: exactly two.
    assert_eq!(source.started_takes.lock().unwrap().len(), 2);

    host.shutdown();
}

#[test]
fn agent_cancel_mid_recording_stops_the_capture() {
    let root = tempfile::tempdir().unwrap();
    let source = one_clean_take();
    let provider = FakeProvider::new(vec![]);
    let (mut host, path) = boot(root.path(), source.clone(), provider);

    let app = connect(&path);
    let agent = agent(&path);
    agent
        .ask_user("1", &["Keep going?".into()], 30_000)
        .unwrap();
    let shown = until_ui(&app, "ShowPrompt", |frame| {
        matches!(frame, UiWire::Show { .. })
    });
    app.prompt_ack(&ask_id_of(&shown), true).unwrap();
    until_event(&app, "capture.started", |e| {
        e.type_name() == "capture.started"
    });
    assert_eq!(capture_state(&app), "Recording");

    // The agent gives up mid-recording.
    agent.ask_cancel("1", "user bailed").unwrap();
    match until_ask(&agent, "1") {
        AskOutcome::NoAnswer { reason } => assert_eq!(reason, NoAnswerReason::AgentCancelled),
        other => panic!("expected the typed cancel, got {other:?}"),
    }
    // The mic stopped: the machine is Idle again, one open ever.
    let deadline = Instant::now() + Duration::from_secs(5);
    while capture_state(&app) != "Idle" {
        assert!(Instant::now() < deadline, "capture never returned to Idle");
        std::thread::sleep(Duration::from_millis(20));
    }
    assert_eq!(source.started_takes.lock().unwrap().len(), 1);
    until_ui(&app, "HidePrompt", |frame| {
        matches!(frame, UiWire::Hide { .. })
    });

    host.shutdown();
}

#[test]
fn timeout_mid_recording_returns_no_answer() {
    let root = tempfile::tempdir().unwrap();
    let source = one_clean_take();
    let provider = FakeProvider::new(vec![]);
    let (mut host, path) = boot(root.path(), source.clone(), provider);

    let app = connect(&path);
    let agent = agent(&path);
    // A budget that expires mid-recording (the fake take records
    // forever until stopped).
    agent.ask_user("1", &["Ready?".into()], 1_500).unwrap();
    let shown = until_ui(&app, "ShowPrompt", |frame| {
        matches!(frame, UiWire::Show { .. })
    });
    app.prompt_ack(&ask_id_of(&shown), true).unwrap();
    until_event(&app, "capture.started", |e| {
        e.type_name() == "capture.started"
    });

    match until_ask(&agent, "1") {
        AskOutcome::NoAnswer { reason } => assert_eq!(reason, NoAnswerReason::Timeout),
        other => panic!("expected the timeout no-answer, got {other:?}"),
    }
    let deadline = Instant::now() + Duration::from_secs(5);
    while capture_state(&app) != "Idle" {
        assert!(Instant::now() < deadline, "capture never returned to Idle");
        std::thread::sleep(Duration::from_millis(20));
    }
    assert_eq!(
        source.started_takes.lock().unwrap().len(),
        1,
        "one open, then aborted"
    );

    host.shutdown();
}

#[test]
fn user_dismissal_mid_recording_returns_no_answer() {
    let root = tempfile::tempdir().unwrap();
    let source = one_clean_take();
    let provider = FakeProvider::new(vec![]);
    let (mut host, path) = boot(root.path(), source.clone(), provider);

    let app = connect(&path);
    let agent = agent(&path);
    agent.ask_user("1", &["Ready?".into()], 30_000).unwrap();
    let shown = until_ui(&app, "ShowPrompt", |frame| {
        matches!(frame, UiWire::Show { .. })
    });
    app.prompt_ack(&ask_id_of(&shown), true).unwrap();
    until_event(&app, "capture.started", |e| {
        e.type_name() == "capture.started"
    });

    // The app dismisses the prompt mid-take.
    app.prompt_ack(&ask_id_of(&shown), false).unwrap();
    match until_ask(&agent, "1") {
        AskOutcome::NoAnswer { reason } => assert_eq!(reason, NoAnswerReason::UserCancelled),
        other => panic!("expected the dismissal, got {other:?}"),
    }
    let deadline = Instant::now() + Duration::from_secs(5);
    while capture_state(&app) != "Idle" {
        assert!(Instant::now() < deadline, "capture never returned to Idle");
        std::thread::sleep(Duration::from_millis(20));
    }

    host.shutdown();
}

#[test]
fn agent_disconnect_mid_recording_cancels_the_take() {
    // The deliberate disconnect rule: the ask is bound to its agent
    // connection; that connection dying cancels it. The AskResult has
    // nowhere to go (the connection is gone) — what must happen is the
    // mic stops and the app's prompt hides, host-enforced (this is what
    // covers a SIGKILLed MCP server, whose cancel frames never sent).
    let root = tempfile::tempdir().unwrap();
    let source = one_clean_take();
    let provider = FakeProvider::new(vec![]);
    let (mut host, path) = boot(root.path(), source.clone(), provider);

    let app = connect(&path);
    {
        let agent = agent(&path);
        agent
            .ask_user("1", &["Still there?".into()], 30_000)
            .unwrap();
        let shown = until_ui(&app, "ShowPrompt", |frame| {
            matches!(frame, UiWire::Show { .. })
        });
        app.prompt_ack(&ask_id_of(&shown), true).unwrap();
        until_event(&app, "capture.started", |e| {
            e.type_name() == "capture.started"
        });
        // EOF: the agent process dies mid-recording.
        drop(agent);
    }

    until_ui(&app, "HidePrompt after the disconnect", |frame| {
        matches!(frame, UiWire::Hide { .. })
    });
    let deadline = Instant::now() + Duration::from_secs(5);
    while capture_state(&app) != "Idle" {
        assert!(
            Instant::now() < deadline,
            "a disconnected agent's take kept the mic open"
        );
        std::thread::sleep(Duration::from_millis(20));
    }
    assert_eq!(source.started_takes.lock().unwrap().len(), 1);

    // And the slot is free: a fresh agent can ask again.
    source.push(FakeTakeScript::clean());
    let agent = agent(&path);
    agent.ask_user("2", &["Again?".into()], 30_000).unwrap();
    let shown = until_ui(&app, "the follow-up prompt", |frame| {
        matches!(frame, UiWire::Show { .. })
    });
    app.prompt_ack(&ask_id_of(&shown), true).unwrap();
    until_event(&app, "capture.started again", |e| {
        e.type_name() == "capture.started"
    });

    host.shutdown();
}

#[test]
fn app_disconnect_mid_recording_cancels_the_take() {
    // With the acking app gone no visible prompt is left, so the take
    // must stop instead of running to the budget.
    let root = tempfile::tempdir().unwrap();
    let source = one_clean_take();
    let provider = FakeProvider::new(vec![]);
    let (mut host, path) = boot(root.path(), source.clone(), provider);

    let observer = connect(&path);
    let agent = agent(&path);
    {
        let app = connect(&path);
        agent
            .ask_user("1", &["Still there?".into()], 60_000)
            .unwrap();
        let shown = until_ui(&app, "ShowPrompt", |frame| {
            matches!(frame, UiWire::Show { .. })
        });
        app.prompt_ack(&ask_id_of(&shown), true).unwrap();
        until_event(&app, "capture.started", |e| {
            e.type_name() == "capture.started"
        });
    }

    match until_ask(&agent, "1") {
        AskOutcome::NoAnswer { reason } => assert_eq!(reason, NoAnswerReason::UserCancelled),
        other => panic!("expected the dismissal, got {other:?}"),
    }
    let deadline = Instant::now() + Duration::from_secs(5);
    while capture_state(&observer) != "Idle" {
        assert!(Instant::now() < deadline, "the take outlived its prompt");
        std::thread::sleep(Duration::from_millis(20));
    }

    host.shutdown();
}

#[test]
fn agent_hello_is_checked_against_the_allowlist() {
    let root = tempfile::tempdir().unwrap();
    let source = one_clean_take();
    let provider = FakeProvider::new(vec![]);
    let allowlist = allowlist_at(root.path());
    let mut host = serve(host_config(
        root.path(),
        Some(allowlist),
        source.clone(),
        provider.clone(),
    ))
    .expect("host serves");
    let path = host.socket_path().to_path_buf();

    // Unknown client: refused, connection closed, default deny.
    let stranger = connect(&path);
    match stranger.agent_hello("codex", "tok-1") {
        Err(ClientError::Closed(reason)) => {
            assert!(reason.contains("auth_failed"), "{reason}");
            assert!(reason.contains("not allowlisted"), "{reason}");
        }
        other => panic!("the unknown client must be refused, got {other:?}"),
    }
    assert!(stranger.is_closed());

    // Known client, wrong token: refused the same way.
    let wrong = connect(&path);
    assert!(wrong.agent_hello(CLIENT, "wrong").is_err());
    assert!(wrong.is_closed());

    // And with no allowlist configured at all, nobody is admitted.
    let root2 = tempfile::tempdir().unwrap();
    let mut host2 =
        serve(host_config(root2.path(), None, source.clone(), provider)).expect("host serves");
    let denied = connect(host2.socket_path());
    assert!(
        denied.agent_hello(CLIENT, TOKEN).is_err(),
        "no allowlist file means deny-all"
    );

    host.shutdown();
    host2.shutdown();
}

#[test]
fn asks_from_non_agent_connections_are_refused() {
    let root = tempfile::tempdir().unwrap();
    let source = one_clean_take();
    let provider = FakeProvider::new(vec![]);
    let (mut host, path) = boot(root.path(), source.clone(), provider);

    // A plain renderer connection (no agent hello) sending AskUser is a
    // protocol violation: the ask surface is opt-in per connection.
    let renderer = connect(&path);
    renderer
        .ask_user("1", &["Sneaky?".into()], 10_000)
        .expect("the frame itself sends fine");
    let deadline = Instant::now() + Duration::from_secs(5);
    while !renderer.is_closed() {
        assert!(Instant::now() < deadline, "the host never refused the ask");
        std::thread::sleep(Duration::from_millis(20));
    }
    assert!(renderer.close_reason().contains("protocol_violation"));
    assert!(source.started_takes.lock().unwrap().is_empty());

    host.shutdown();
}

#[test]
fn invalid_asks_are_answered_with_typed_errors() {
    let root = tempfile::tempdir().unwrap();
    let source = one_clean_take();
    let provider = FakeProvider::new(vec![]);
    let (mut host, path) = boot(root.path(), source.clone(), provider);

    let app = connect(&path);
    let agent = agent(&path);

    // Empty questions (raw-frame path; the MCP layer refuses the same
    // shape with -32602 before it ever reaches the wire).
    agent.ask_user("e", &[], 10_000).unwrap();
    match until_ask(&agent, "e") {
        AskOutcome::Error { code, .. } => assert_eq!(code, "invalid_questions"),
        other => panic!("empty questions: {other:?}"),
    }
    // A blank question.
    agent.ask_user("b", &["   ".into()], 10_000).unwrap();
    match until_ask(&agent, "b") {
        AskOutcome::Error { code, .. } => assert_eq!(code, "invalid_questions"),
        other => panic!("blank question: {other:?}"),
    }
    // An oversized question.
    agent.ask_user("o", &["x".repeat(2_001)], 10_000).unwrap();
    match until_ask(&agent, "o") {
        AskOutcome::Error { code, .. } => assert_eq!(code, "invalid_questions"),
        other => panic!("oversized question: {other:?}"),
    }
    // An out-of-bounds timeout.
    agent.ask_user("t", &["Ready?".into()], 10).unwrap();
    match until_ask(&agent, "t") {
        AskOutcome::Error { code, .. } => assert_eq!(code, "invalid_timeout"),
        other => panic!("tiny timeout: {other:?}"),
    }
    // Nothing was prompted and nothing was captured through any of it.
    assert!(source.started_takes.lock().unwrap().is_empty());
    let _ = app;

    host.shutdown();
}

#[test]
fn a_burst_of_asks_overflows_the_queue_with_a_typed_error() {
    let root = tempfile::tempdir().unwrap();
    let source = FakeCaptureSource::new(vec![FakeTakeScript::clean()]);
    let provider = FakeProvider::new(vec![]);
    let (mut host, path) = boot(root.path(), source.clone(), provider);

    let app = connect(&path);
    let agent = agent(&path);
    // One live (un-acked) ask…
    agent.ask_user("live", &["Held?".into()], 30_000).unwrap();
    let held = until_ui(&app, "the held prompt", |frame| {
        matches!(frame, UiWire::Show { .. })
    });
    // …then the queue's full worth of waiting asks…
    for index in 0..4 {
        agent
            .ask_user(&format!("w{index}"), &[format!("Q{index}?")], 30_000)
            .unwrap();
    }
    // …and one past the bound: refused immediately and typed.
    agent
        .ask_user("overflow", &["One too many?".into()], 30_000)
        .unwrap();
    match until_ask(&agent, "overflow") {
        AskOutcome::Error { code, message } => {
            assert_eq!(code, "queue_full", "{message}");
        }
        other => panic!("expected queue_full, got {other:?}"),
    }
    // The queued asks are intact: resolve the live one (decline it)
    // and the next prompt appears.
    app.prompt_ack(&ask_id_of(&held), false).unwrap();
    match until_ask(&agent, "live") {
        AskOutcome::NoAnswer { reason } => assert_eq!(reason, NoAnswerReason::Declined),
        other => panic!("decline: {other:?}"),
    }
    until_ui(&app, "the first queued prompt", |frame| {
        matches!(frame, UiWire::Show { .. })
    });

    host.shutdown();
}

#[test]
fn late_prompt_frames_for_resolved_asks_are_ignored() {
    // A stale Done/Ack must not wedgie the broker: the ask it names is
    // gone, the frames are dropped, and the next ask works.
    let root = tempfile::tempdir().unwrap();
    let source = FakeCaptureSource::new(vec![FakeTakeScript::clean()]);
    let provider = FakeProvider::new(vec![FakeJob::completes_with("fine")]);
    let (mut host, path) = boot(root.path(), source.clone(), provider);

    let app = connect(&path);
    let agent = agent(&path);
    agent.ask_user("1", &["First?".into()], 5_000).unwrap();
    let shown = until_ui(&app, "the prompt", |frame| {
        matches!(frame, UiWire::Show { .. })
    });
    app.prompt_ack(&ask_id_of(&shown), false).unwrap(); // declined → resolved
    match until_ask(&agent, "1") {
        AskOutcome::NoAnswer { reason } => assert_eq!(reason, NoAnswerReason::Declined),
        other => panic!("decline: {other:?}"),
    }
    // The stale done (the app raced the resolution).
    app.prompt_done(&ask_id_of(&shown)).unwrap();
    app.prompt_done("ask_999").unwrap();
    std::thread::sleep(Duration::from_millis(50));

    // A fresh ask flows normally.
    agent.ask_user("2", &["Second?".into()], 20_000).unwrap();
    let shown = until_ui(&app, "the second prompt", |frame| {
        matches!(frame, UiWire::Show { .. })
    });
    app.prompt_ack(&ask_id_of(&shown), true).unwrap();
    until_event(&app, "capture.started", |e| {
        e.type_name() == "capture.started"
    });
    app.prompt_done(&ask_id_of(&shown)).unwrap();
    match until_ask(&agent, "2") {
        AskOutcome::Answered { text, .. } => assert_eq!(text, "fine"),
        other => panic!("second ask: {other:?}"),
    }

    host.shutdown();
}

#[test]
fn a_done_before_the_visibility_ack_is_ignored_not_failed() {
    // An early Done (before any ack) is the app misbehaving: it must
    // be ignored — the ack bound still applies — never resolved as a
    // capture failure, and the ask must stay live for an honest ack.
    let root = tempfile::tempdir().unwrap();
    let source = one_clean_take();
    let provider = FakeProvider::new(vec![FakeJob::completes_with("yes")]);
    let (mut host, path) = boot(root.path(), source.clone(), provider);

    let app = connect(&path);
    let agent = agent(&path);
    agent.ask_user("1", &["Ready?".into()], 20_000).unwrap();
    let shown = until_ui(&app, "ShowPrompt", |frame| {
        matches!(frame, UiWire::Show { .. })
    });

    // Done with no ack in sight.
    app.prompt_done(&ask_id_of(&shown)).unwrap();
    std::thread::sleep(Duration::from_millis(250));
    assert!(
        source.started_takes.lock().unwrap().is_empty(),
        "an early done must not open the mic or resolve the ask"
    );
    assert_eq!(capture_state(&app), "Idle");

    // The ask survived: an honest decline still resolves it.
    app.prompt_ack(&ask_id_of(&shown), false).unwrap();
    match until_ask(&agent, "1") {
        AskOutcome::NoAnswer { reason } => assert_eq!(reason, NoAnswerReason::Declined),
        other => panic!("expected the decline after the ignored done, got {other:?}"),
    }

    host.shutdown();
}

#[test]
fn a_forged_ack_from_a_connection_that_never_saw_the_prompt_is_ignored() {
    // The binding: only a connection the prompt was fanned out to may
    // ack it. A renderer connecting after the fan-out guesses the ask
    // id and forges `visible: true` — the gate must stay shut, and the
    // honest ack must still work.
    let root = tempfile::tempdir().unwrap();
    let source = FakeCaptureSource::new(vec![FakeTakeScript::clean(), FakeTakeScript::clean()]);
    let provider = FakeProvider::new(vec![FakeJob::completes_with("yes")]);
    let (mut host, path) = boot(root.path(), source.clone(), provider);

    let app = connect(&path);
    let agent = agent(&path);
    agent.ask_user("1", &["Ready?".into()], 20_000).unwrap();
    let shown = until_ui(&app, "ShowPrompt", |frame| {
        matches!(frame, UiWire::Show { .. })
    });

    // Connected after the fan-out: never shown this prompt.
    let forger = connect(&path);
    forger.prompt_ack(&ask_id_of(&shown), true).unwrap();
    std::thread::sleep(Duration::from_millis(250));
    assert!(
        source.started_takes.lock().unwrap().is_empty(),
        "a forged ack must not open the microphone"
    );
    assert_eq!(capture_state(&app), "Idle");

    // The honest ack from the connection that was shown the prompt.
    app.prompt_ack(&ask_id_of(&shown), true).unwrap();
    until_event(&app, "capture.started", |e| {
        e.type_name() == "capture.started"
    });
    assert_eq!(source.started_takes.lock().unwrap().len(), 1);
    app.prompt_done(&ask_id_of(&shown)).unwrap();
    match until_ask(&agent, "1") {
        AskOutcome::Answered { text, .. } => assert_eq!(text, "yes"),
        other => panic!("expected the answer after the honest ack, got {other:?}"),
    }

    host.shutdown();
}

#[test]
fn an_agent_connection_cannot_forge_prompt_frames() {
    // An agent-flagged connection never receives a ShowPrompt, so its
    // ack/done can only be a forge attempt: the host closes the
    // connection as a protocol violation, and the disconnect rule
    // then cancels its live ask.
    let root = tempfile::tempdir().unwrap();
    let source = one_clean_take();
    let provider = FakeProvider::new(vec![]);
    let (mut host, path) = boot(root.path(), source.clone(), provider);

    let app = connect(&path);
    let agent = agent(&path);
    agent.ask_user("1", &["Ready?".into()], 20_000).unwrap();
    let shown = until_ui(&app, "ShowPrompt", |frame| {
        matches!(frame, UiWire::Show { .. })
    });

    // The forged ack sends fine on the wire; the host's verdict
    // follows.
    agent.prompt_ack(&ask_id_of(&shown), true).unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    while !agent.is_closed() {
        assert!(
            Instant::now() < deadline,
            "the host never refused the forged ack"
        );
        std::thread::sleep(Duration::from_millis(20));
    }
    assert!(agent.close_reason().contains("protocol_violation"));

    // The mic never opened, and the disconnect rule hid the prompt.
    assert!(source.started_takes.lock().unwrap().is_empty());
    until_ui(&app, "HidePrompt after the forged ack", |frame| {
        matches!(frame, UiWire::Hide { .. })
    });

    host.shutdown();
}

#[test]
fn a_duplicate_in_flight_ask_token_is_refused_with_duplicate_req() {
    let root = tempfile::tempdir().unwrap();
    let source = one_clean_take();
    let provider = FakeProvider::new(vec![]);
    let (mut host, path) = boot(root.path(), source, provider);

    let app = connect(&path);
    let agent = agent(&path);
    agent.ask_user("dup", &["Ready?".into()], 30_000).unwrap();
    let shown = until_ui(&app, "the live prompt", |frame| {
        matches!(frame, UiWire::Show { .. })
    });

    // The same token while the first is still in flight: refused with
    // the accurate code (a duplicate correlation token, not an
    // invalid-questions shape).
    agent.ask_user("dup", &["Ready?".into()], 30_000).unwrap();
    match until_ask(&agent, "dup") {
        AskOutcome::Error { code, message } => {
            assert_eq!(code, "duplicate_req", "{message}");
        }
        other => panic!("expected duplicate_req, got {other:?}"),
    }

    // The original ask is untouched and still resolvable.
    app.prompt_ack(&ask_id_of(&shown), false).unwrap();
    match until_ask(&agent, "dup") {
        AskOutcome::NoAnswer { reason } => assert_eq!(reason, NoAnswerReason::Declined),
        other => panic!("expected the decline of the original ask, got {other:?}"),
    }

    host.shutdown();
}

#[test]
fn the_budget_ends_at_capture_stop_so_a_captured_take_survives_slow_transcription() {
    // The timeout is user-facing: once the microphone closes it stops
    // ticking, so a transcription slower than the whole budget still
    // answers instead of discarding the captured take on the clock.
    let root = tempfile::tempdir().unwrap();
    let source = one_clean_take();
    // Transcription outlasts the ask's entire budget.
    let provider = FakeProvider::new(vec![FakeJob {
        work_ms: 2_500,
        ..FakeJob::completes_with("worth the wait")
    }]);
    let (mut host, path) = boot(root.path(), source.clone(), provider);

    let app = connect(&path);
    let agent = agent(&path);
    agent.ask_user("1", &["Ready?".into()], 1_500).unwrap();
    let shown = until_ui(&app, "ShowPrompt", |frame| {
        matches!(frame, UiWire::Show { .. })
    });
    app.prompt_ack(&ask_id_of(&shown), true).unwrap();
    until_event(&app, "capture.started", |e| {
        e.type_name() == "capture.started"
    });
    // The mic closes well before the budget expires.
    app.prompt_done(&ask_id_of(&shown)).unwrap();

    match until_ask(&agent, "1") {
        AskOutcome::Answered { text, .. } => assert_eq!(text, "worth the wait"),
        other => panic!("a captured take must not be discarded on the clock, got {other:?}"),
    }
    assert_eq!(source.started_takes.lock().unwrap().len(), 1);

    host.shutdown();
}

#[test]
fn the_stdio_server_exits_zero_promptly_on_stdin_eof() {
    let root = tempfile::tempdir().unwrap();
    let source = one_clean_take();
    let provider = FakeProvider::new(vec![]);
    let (mut host, path) = boot(root.path(), source, provider);
    let _app = connect(&path);

    let mut child = std::process::Command::new(env!("CARGO_BIN_EXE_mcp-dictation"))
        .arg("--socket")
        .arg(&path)
        .arg("--client")
        .arg(CLIENT)
        .env("STARLING_MCP_TOKEN", TOKEN)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("the mcp-dictation binary builds and spawns");

    {
        let mut stdin = child.stdin.take().expect("stdin piped");
        writeln!(
            stdin,
            r#"{{"jsonrpc":"2.0","id":1,"method":"initialize","params":{{}}}}"#
        )
        .unwrap();
    }

    let mut child_stdout = child.stdout.take().expect("stdout piped");
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(child.wait());
    });
    let status = rx
        .recv_timeout(Duration::from_secs(10))
        .expect("the server exited after stdin EOF instead of hanging")
        .expect("the wait itself succeeded");
    assert!(status.success(), "clean EOF exits 0, got {status:?}");
    let mut stdout = String::new();
    std::io::Read::read_to_string(&mut child_stdout, &mut stdout).unwrap();
    let reply: serde_json::Value =
        serde_json::from_str(stdout.lines().next().expect("the handshake was answered")).unwrap();
    assert_eq!(reply["id"], 1);
    assert!(reply["result"]["protocolVersion"].is_string(), "{reply}");

    host.shutdown();
}

#[test]
fn without_the_app_role_every_ask_fails_closed() {
    let root = tempfile::tempdir().unwrap();
    let source = one_clean_take();
    let mut config = host_config(
        root.path(),
        Some(allowlist_at(root.path())),
        source.clone(),
        FakeProvider::new(vec![]),
    );
    config.insecure_test_app_role = false;
    let (mut host, path) = serve_at(config);

    // The agent's own second, plain connection: same user, no app role.
    let plain = connect(&path);
    let agent = agent(&path);
    agent.ask_user("1", &["Ready?".into()], 20_000).unwrap();
    match until_ask(&agent, "1") {
        AskOutcome::Error { code, message } => assert_eq!(code, "no_app", "{message}"),
        other => panic!("expected the fail-closed refusal, got {other:?}"),
    }
    assert!(plain.try_recv_ui().is_err(), "no prompt reaches it");

    plain.prompt_ack("ask_1", true).unwrap();
    until_closed(&plain, "an ack without the app role");
    assert!(plain.close_reason().contains("protocol_violation"));
    assert!(source.started_takes.lock().unwrap().is_empty());

    host.shutdown();
}

#[test]
fn an_agent_connection_cannot_send_runtime_commands() {
    let root = tempfile::tempdir().unwrap();
    let source = one_clean_take();
    let (mut host, path) = boot(root.path(), source.clone(), FakeProvider::new(vec![]));

    let agent = agent(&path);
    let start = Command::CaptureStart {
        policy: "push-to-talk".into(),
    };
    match agent.send(Some("take_direct"), start) {
        Err(ClientError::Closed(reason)) => {
            assert!(reason.contains("protocol_violation"), "{reason}")
        }
        other => panic!("an agent's capture.start must be refused, got {other:?}"),
    }
    assert!(source.started_takes.lock().unwrap().is_empty());

    host.shutdown();
}

/// A context provider slow enough for connection messages to overtake
/// the route freeze.
struct SlowContext;

impl ContextProvider for SlowContext {
    fn snapshot(&self, source: &str) -> Result<TargetSnapshotData, String> {
        std::thread::sleep(Duration::from_millis(500));
        StubContextProvider::default().snapshot(source)
    }
}

#[test]
fn an_app_leaving_while_the_route_freezes_never_opens_the_mic() {
    let root = tempfile::tempdir().unwrap();
    let source = one_clean_take();
    let mut config = host_config(
        root.path(),
        Some(allowlist_at(root.path())),
        source.clone(),
        FakeProvider::new(vec![]),
    );
    config.runtime = config.runtime.with_context_provider(Arc::new(SlowContext));
    let (mut host, path) = serve_at(config);

    let agent = agent(&path);
    {
        let app = connect(&path);
        agent.ask_user("1", &["Ready?".into()], 30_000).unwrap();
        let shown = until_ui(&app, "ShowPrompt", |frame| {
            matches!(frame, UiWire::Show { .. })
        });
        app.prompt_ack(&ask_id_of(&shown), true).unwrap();
    }
    match until_ask(&agent, "1") {
        AskOutcome::NoAnswer { reason } => assert_eq!(reason, NoAnswerReason::UserCancelled),
        other => panic!("expected the dismissal, got {other:?}"),
    }
    std::thread::sleep(Duration::from_millis(300));
    assert!(source.started_takes.lock().unwrap().is_empty());

    // The abandoned context was released, so the next ask can freeze a
    // route of its own and record.
    let app = connect(&path);
    agent.ask_user("2", &["Again?".into()], 30_000).unwrap();
    let shown = until_ui(&app, "the second prompt", |frame| {
        matches!(frame, UiWire::Show { .. })
    });
    app.prompt_ack(&ask_id_of(&shown), true).unwrap();
    until_event(&app, "capture.started", |e| {
        e.type_name() == "capture.started"
    });

    host.shutdown();
}

#[test]
fn a_tiny_event_buffer_cannot_wedge_the_broker() {
    // With a one-slot subscription the capture actor blocks on the
    // broker's subscription while the broker waits on its receipts,
    // unless the subscription is drained independently.
    let root = tempfile::tempdir().unwrap();
    let source = FakeCaptureSource::new(vec![FakeTakeScript::clean(); 3]);
    let mut config = host_config(
        root.path(),
        Some(allowlist_at(root.path())),
        source.clone(),
        FakeProvider::new(vec![]),
    );
    config.runtime.event_capacity = 1;
    let (mut host, path) = serve_at(config);

    let app = connect(&path);
    let agent = agent(&path);
    for round in 0..3 {
        let req = round.to_string();
        agent.ask_user(&req, &["Ready?".into()], 30_000).unwrap();
        let shown = until_ui(&app, "ShowPrompt", |frame| {
            matches!(frame, UiWire::Show { .. })
        });
        app.prompt_ack(&ask_id_of(&shown), true).unwrap();
        until_event(&app, "capture.started", |e| {
            e.type_name() == "capture.started"
        });
        std::thread::sleep(Duration::from_millis(100));
        agent.ask_cancel(&req, "changed my mind").unwrap();
        match until_ask(&agent, &req) {
            AskOutcome::NoAnswer { reason } => {
                assert_eq!(reason, NoAnswerReason::AgentCancelled)
            }
            other => panic!("round {round}: expected the cancel, got {other:?}"),
        }
    }

    host.shutdown();
}

struct FailingStore;

impl CaptureStore for FailingStore {
    fn commit_take(&self, _: &TakeRecord) -> Result<(), String> {
        Err("disk full".to_string())
    }
    fn mark_interrupted(&self, _: &TakeRecord, _: &str) -> Result<(), String> {
        Ok(())
    }
    fn describe(&self) -> String {
        "failing".to_string()
    }
}

#[test]
fn a_failed_store_commit_fails_the_ask() {
    let root = tempfile::tempdir().unwrap();
    let mut config = host_config(
        root.path(),
        Some(allowlist_at(root.path())),
        one_clean_take(),
        FakeProvider::new(vec![]),
    );
    config.runtime = config.runtime.with_capture_store(Arc::new(FailingStore));
    let (mut host, path) = serve_at(config);

    let app = connect(&path);
    let agent = agent(&path);
    agent.ask_user("1", &["Ready?".into()], 30_000).unwrap();
    let shown = until_ui(&app, "ShowPrompt", |frame| {
        matches!(frame, UiWire::Show { .. })
    });
    app.prompt_ack(&ask_id_of(&shown), true).unwrap();
    until_event(&app, "capture.started", |e| {
        e.type_name() == "capture.started"
    });
    app.prompt_done(&ask_id_of(&shown)).unwrap();
    match until_ask(&agent, "1") {
        AskOutcome::Error { code, message } => assert_eq!(code, "capture_failed", "{message}"),
        other => panic!("expected the storage failure, got {other:?}"),
    }

    host.shutdown();
}

#[test]
fn the_broker_never_ends_a_take_it_did_not_start() {
    let root = tempfile::tempdir().unwrap();
    let source = FakeCaptureSource::new(vec![FakeTakeScript::clean(), FakeTakeScript::clean()]);
    let (mut host, path) = boot(root.path(), source.clone(), FakeProvider::new(vec![]));

    let app = connect(&path);
    let agent = agent(&path);
    agent.ask_user("1", &["Ready?".into()], 30_000).unwrap();
    let shown = until_ui(&app, "ShowPrompt", |frame| {
        matches!(frame, UiWire::Show { .. })
    });
    let ask_id = ask_id_of(&shown);
    app.prompt_ack(&ask_id, true).unwrap();
    until_event(&app, "capture.started", |e| {
        e.type_name() == "capture.started"
    });

    // The ask's corr is reserved: no client can address its take by
    // name, or reuse the name for a take of its own.
    let renderer = connect(&path);
    assert!(
        ask_id.starts_with("ask_") && ask_id.len() > "ask_".len() + 8,
        "{ask_id}"
    );
    for command in [
        Command::CaptureStop { drain: Some(true) },
        Command::CaptureStart {
            policy: "push-to-talk".into(),
        },
    ] {
        match renderer.send(Some(&ask_id), command) {
            Err(ClientError::Rejected(Rejection::InvalidEnvelope(detail))) => {
                assert!(detail.contains("reserved"), "{detail}")
            }
            other => panic!("a reserved corr must be refused, got {other:?}"),
        }
    }
    assert_eq!(capture_state(&renderer), "Recording");

    // Another client ends the ask's take without naming it and starts
    // its own.
    renderer
        .send(None, Command::CaptureAbort)
        .expect("abort accepted");
    renderer
        .send(
            Some("take_other"),
            Command::CaptureStart {
                policy: "push-to-talk".into(),
            },
        )
        .expect("start accepted");
    until_event(&renderer, "capture.started", |e| {
        e.type_name() == "capture.started" && e.corr() == Some("take_other")
    });

    agent.ask_cancel("1", "never mind").unwrap();
    match until_ask(&agent, "1") {
        AskOutcome::NoAnswer { reason } => assert_eq!(reason, NoAnswerReason::AgentCancelled),
        other => panic!("expected the cancel, got {other:?}"),
    }
    std::thread::sleep(Duration::from_millis(200));
    assert_eq!(
        capture_state(&renderer),
        "Recording",
        "the other take survives"
    );

    host.shutdown();
}

#[test]
fn cancelling_a_queued_ask_answers_it() {
    let root = tempfile::tempdir().unwrap();
    let (mut host, path) = boot(root.path(), one_clean_take(), FakeProvider::new(vec![]));

    let app = connect(&path);
    let agent = agent(&path);
    agent.ask_user("live", &["First?".into()], 30_000).unwrap();
    let shown = until_ui(&app, "the live prompt", |frame| {
        matches!(frame, UiWire::Show { .. })
    });
    agent
        .ask_user("queued", &["Second?".into()], 30_000)
        .unwrap();
    agent.ask_cancel("queued", "no longer needed").unwrap();
    match until_ask(&agent, "queued") {
        AskOutcome::NoAnswer { reason } => assert_eq!(reason, NoAnswerReason::AgentCancelled),
        other => panic!("expected the queued cancel, got {other:?}"),
    }

    // The live ask is untouched, and the cancelled one never prompts.
    app.prompt_ack(&ask_id_of(&shown), false).unwrap();
    match until_ask(&agent, "live") {
        AskOutcome::NoAnswer { reason } => assert_eq!(reason, NoAnswerReason::Declined),
        other => panic!("expected the decline, got {other:?}"),
    }
    std::thread::sleep(Duration::from_millis(200));
    while let Ok(frame) = app.try_recv_ui() {
        assert!(!matches!(frame, UiWire::Show { .. }), "{frame:?}");
    }

    host.shutdown();
}

/// Holds the capture actor inside `start` until opened.
struct GatedSource {
    inner: Arc<FakeCaptureSource>,
    entered: std::sync::atomic::AtomicBool,
    open: std::sync::Mutex<bool>,
    opened: std::sync::Condvar,
}

impl GatedSource {
    fn release(&self) {
        *self.open.lock().unwrap() = true;
        self.opened.notify_all();
    }
}

/// Opens the gate when dropped, so a failing test cannot leave the
/// capture actor parked and the host's shutdown hanging.
struct ReleaseOnDrop(Arc<GatedSource>);

impl Drop for ReleaseOnDrop {
    fn drop(&mut self) {
        self.0.release();
    }
}

impl CaptureSource for GatedSource {
    fn start(&self, journals: &Path, policy: &str) -> Result<Box<dyn CaptureSession>, String> {
        self.entered
            .store(true, std::sync::atomic::Ordering::SeqCst);
        let mut open = self.open.lock().unwrap();
        while !*open {
            open = self.opened.wait(open).unwrap();
        }
        drop(open);
        self.inner.start(journals, policy)
    }
}

#[test]
fn an_abort_refused_by_a_full_inbox_is_retried_before_the_next_ask() {
    let root = tempfile::tempdir().unwrap();
    let gate = Arc::new(GatedSource {
        inner: one_clean_take(),
        entered: std::sync::atomic::AtomicBool::new(false),
        open: std::sync::Mutex::new(false),
        opened: std::sync::Condvar::new(),
    });
    let mut config = host_config(
        root.path(),
        Some(allowlist_at(root.path())),
        one_clean_take(),
        FakeProvider::new(vec![]),
    );
    config.runtime.command_capacity = 1;
    config.runtime = config.runtime.with_capture_source(gate.clone());
    let (mut host, path) = serve_at(config);
    // Declared after the host so it is dropped, and the gate opened,
    // before the host shuts down.
    let _release = ReleaseOnDrop(Arc::clone(&gate));

    let app = connect(&path);
    let agent = agent(&path);
    agent.ask_user("1", &["Ready?".into()], 30_000).unwrap();
    let shown = until_ui(&app, "the first prompt", |frame| {
        matches!(frame, UiWire::Show { .. })
    });
    app.prompt_ack(&ask_id_of(&shown), true).unwrap();
    // capture.start is accepted, then the actor parks in the gated source.
    let deadline = Instant::now() + Duration::from_secs(5);
    while !gate.entered.load(std::sync::atomic::Ordering::SeqCst) {
        assert!(Instant::now() < deadline, "the take never started");
        std::thread::sleep(Duration::from_millis(20));
    }

    // Two senders race for the parked actor's one inbox slot: one is
    // queued, the other refused at once, which proves the slot is taken.
    let (results, outcomes) = std::sync::mpsc::channel();
    let fillers: Vec<_> = ["take_filler_a", "take_filler_b"]
        .into_iter()
        .map(|corr| {
            let filler = connect(&path);
            let results = results.clone();
            std::thread::spawn(move || {
                let _ = results.send(filler.send(Some(corr), Command::CaptureAbort));
            })
        })
        .collect();
    match outcomes.recv_timeout(Duration::from_secs(5)) {
        Ok(Err(ClientError::Rejected(Rejection::InboxFull))) => {}
        other => panic!("expected one filler refused by the full inbox, got {other:?}"),
    }

    agent.ask_user("2", &["Next?".into()], 30_000).unwrap();
    agent.ask_cancel("1", "never mind").unwrap();
    match until_ask(&agent, "1") {
        AskOutcome::NoAnswer { reason } => assert_eq!(reason, NoAnswerReason::AgentCancelled),
        other => panic!("expected the cancel, got {other:?}"),
    }
    // The abort could not land, so the next ask stays queued.
    std::thread::sleep(Duration::from_millis(300));
    while let Ok(frame) = app.try_recv_ui() {
        assert!(
            !matches!(frame, UiWire::Show { .. }),
            "admitted too early: {frame:?}"
        );
    }

    gate.release();
    for filler in fillers {
        let _ = filler.join();
    }
    until_ui(&app, "the next prompt", |frame| {
        matches!(frame, UiWire::Show { .. })
    });
    let deadline = Instant::now() + Duration::from_secs(5);
    while capture_state(&app) != "Idle" {
        assert!(Instant::now() < deadline, "the retried abort never landed");
        std::thread::sleep(Duration::from_millis(20));
    }

    host.shutdown();
}
