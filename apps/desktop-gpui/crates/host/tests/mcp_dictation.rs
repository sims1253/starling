//! The agent-dictation slice of issue #309, over the real IPC transport
//! with the real host (`server::serve`), the scripted fake capture
//! source, and the fake provider — no real audio, no real agent.
//!
//! What each acceptance from the issue is proven by:
//!
//! - **Visibility gate** — `capture_starts_only_after_the_app_acks_the_prompt`
//!   and `no_ack_within_the_bound_fails_without_opening_the_mic`: the
//!   microphone (the fake source's `started_takes`) opens only after
//!   the app acked, and an un-acked ask fails `no_prompt_ack` with the
//!   mic never touched.
//! - **Allowlist (default deny)** — `agent_hello_is_checked_against_the_allowlist`
//!   and `asks_from_non_agent_connections_are_refused`.
//! - **Queueing** — `concurrent_asks_queue_one_prompt_at_a_time`.
//! - **Cancel** — `agent_cancel_mid_recording_stops_the_capture`.
//! - **Timeout** — `timeout_mid_recording_returns_no_answer`.
//! - **Disconnect** — `agent_disconnect_mid_recording_cancels_the_take`
//!   (the deliberate rule: the ask dies with its connection).
//! - **End to end** — `a_spoken_answer_flows_back_as_the_tool_result`.

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use starling_runtime::channel::RecvError;
use starling_runtime::machine::capture::{CaptureConfig, V2CaptureStore};
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
        .with_agent_allowlist(allowlist);
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
    let allowlist = allowlist_at(root);
    let host = serve(host_config(root, Some(allowlist), source, provider)).expect("host serves");
    let path = host.socket_path().to_path_buf();
    (host, path)
}

/// Waits for an ask result with this req (or panics with everything
/// that arrived instead).
fn until_ask(client: &HostClient, req: &str) -> AskOutcome {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        match client.recv_ask_timeout(Duration::from_millis(20)) {
            Ok(result) if result.req == req => return result.outcome,
            Ok(other) => panic!("ask {req}: got a result for {} first: {:?}", other.req, other.outcome),
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
        .ask_user("1", &["Should I fix the parser or the writer?".into()], 30_000)
        .expect("ask sent");

    // The prompt reaches the app with the questions intact.
    let shown = until_ui(&app, "ShowPrompt", |frame| matches!(frame, UiWire::Show { .. }));
    let UiWire::Show { questions, .. } = &shown else { unreachable!() };
    assert_eq!(questions, &["Should I fix the parser or the writer?".to_string()]);

    // Before the ack: no capture has been started.
    assert!(
        source.started_takes.lock().unwrap().is_empty(),
        "the gate must hold the mic until the app acks"
    );

    app.prompt_ack(&ask_id_of(&shown), true).unwrap();
    until_event(&app, "capture.started", |e| e.type_name() == "capture.started");
    assert_eq!(source.started_takes.lock().unwrap().len(), 1);

    // The user finishes speaking; the host stops, persists, transcribes.
    app.prompt_done(&ask_id_of(&shown)).unwrap();
    until_event(&app, "jobs.completed", |e| e.type_name() == "jobs.completed");

    match until_ask(&agent, "1") {
        AskOutcome::Answered { text, backend } => {
            assert_eq!(text, "Take the first option.");
            assert!(!backend.is_empty(), "the jobs machine names its backend");
        }
        other => panic!("expected the spoken answer, got {other:?}"),
    }
    // The overlay is cleared on every resolution.
    until_ui(&app, "HidePrompt", |frame| matches!(frame, UiWire::Hide { .. }));

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
    // Short timeout: the ack bound clamps to the remaining budget, so
    // this also exercises the bound, not just the overall timeout.
    agent
        .ask_user("1", &["Ready?".into()], 1_500)
        .expect("ask sent");
    until_ui(&app, "ShowPrompt", |frame| matches!(frame, UiWire::Show { .. }));
    // No ack. Ever.

    match until_ask(&agent, "1") {
        AskOutcome::Error { code, message } => {
            assert_eq!(code, "no_prompt_ack", "{message}");
            assert!(message.contains("never"), "{message}");
        }
        other => panic!("expected the visibility-gate failure, got {other:?}"),
    }
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
    let shown = until_ui(&app, "ShowPrompt", |frame| matches!(frame, UiWire::Show { .. }));

    // Give the broker every chance to misbehave, then assert it did
    // not: 250 ms of an un-acked prompt must not open the mic.
    std::thread::sleep(Duration::from_millis(250));
    assert!(source.started_takes.lock().unwrap().is_empty());
    assert_eq!(capture_state(&app), "Idle");

    app.prompt_ack(&ask_id_of(&shown), true).unwrap();
    until_event(&app, "capture.started", |e| e.type_name() == "capture.started");
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
    agent.ask_user("1", &["Anyone there?".into()], 20_000).unwrap();
    match until_ask(&agent, "1") {
        AskOutcome::Error { code, message } => {
            assert_eq!(code, "no_prompt_ack", "{message}");
            assert!(message.contains("no app connection"), "{message}");
        }
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

    let first = until_ui(&app, "the first prompt", |frame| matches!(frame, UiWire::Show { .. }));
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
    until_event(&app, "capture.started", |e| e.type_name() == "capture.started");
    app.prompt_done(&ask_id_of(&first)).unwrap();
    match until_ask(&agent, "a") {
        AskOutcome::Answered { text, .. } => assert_eq!(text, "first answer"),
        other => panic!("first ask: {other:?}"),
    }
    // Only now does the second prompt appear.
    let second = until_ui(&app, "the second prompt", |frame| matches!(frame, UiWire::Show { .. }));
    assert!(
        ask_id_of(&second) != ask_id_of(&first),
        "each ask has its own broker-scoped id"
    );
    app.prompt_ack(&ask_id_of(&second), true).unwrap();
    until_event(&app, "second capture.started", |e| e.type_name() == "capture.started");
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
    agent.ask_user("1", &["Keep going?".into()], 30_000).unwrap();
    let shown = until_ui(&app, "ShowPrompt", |frame| matches!(frame, UiWire::Show { .. }));
    app.prompt_ack(&ask_id_of(&shown), true).unwrap();
    until_event(&app, "capture.started", |e| e.type_name() == "capture.started");
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
    until_ui(&app, "HidePrompt", |frame| matches!(frame, UiWire::Hide { .. }));

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
    let shown = until_ui(&app, "ShowPrompt", |frame| matches!(frame, UiWire::Show { .. }));
    app.prompt_ack(&ask_id_of(&shown), true).unwrap();
    until_event(&app, "capture.started", |e| e.type_name() == "capture.started");

    match until_ask(&agent, "1") {
        AskOutcome::NoAnswer { reason } => assert_eq!(reason, NoAnswerReason::Timeout),
        other => panic!("expected the timeout no-answer, got {other:?}"),
    }
    let deadline = Instant::now() + Duration::from_secs(5);
    while capture_state(&app) != "Idle" {
        assert!(Instant::now() < deadline, "capture never returned to Idle");
        std::thread::sleep(Duration::from_millis(20));
    }
    assert_eq!(source.started_takes.lock().unwrap().len(), 1, "one open, then aborted");

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
    let shown = until_ui(&app, "ShowPrompt", |frame| matches!(frame, UiWire::Show { .. }));
    app.prompt_ack(&ask_id_of(&shown), true).unwrap();
    until_event(&app, "capture.started", |e| e.type_name() == "capture.started");

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
        agent.ask_user("1", &["Still there?".into()], 30_000).unwrap();
        let shown = until_ui(&app, "ShowPrompt", |frame| matches!(frame, UiWire::Show { .. }));
        app.prompt_ack(&ask_id_of(&shown), true).unwrap();
        until_event(&app, "capture.started", |e| e.type_name() == "capture.started");
        // EOF: the agent process dies mid-recording.
        drop(agent);
    }

    until_ui(&app, "HidePrompt after the disconnect", |frame| {
        matches!(frame, UiWire::Hide { .. })
    });
    let deadline = Instant::now() + Duration::from_secs(5);
    while capture_state(&app) != "Idle" {
        assert!(Instant::now() < deadline, "a disconnected agent's take kept the mic open");
        std::thread::sleep(Duration::from_millis(20));
    }
    assert_eq!(source.started_takes.lock().unwrap().len(), 1);

    // And the slot is free: a fresh agent can ask again.
    source.push(FakeTakeScript::clean());
    let agent = agent(&path);
    agent.ask_user("2", &["Again?".into()], 30_000).unwrap();
    let shown = until_ui(&app, "the follow-up prompt", |frame| matches!(frame, UiWire::Show { .. }));
    app.prompt_ack(&ask_id_of(&shown), true).unwrap();
    until_event(&app, "capture.started again", |e| e.type_name() == "capture.started");

    host.shutdown();
}

#[test]
fn agent_hello_is_checked_against_the_allowlist() {
    let root = tempfile::tempdir().unwrap();
    let source = one_clean_take();
    let provider = FakeProvider::new(vec![]);
    let allowlist = allowlist_at(root.path());
    let mut host = serve(host_config(root.path(), Some(allowlist), source.clone(), provider.clone()))
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
    let mut host2 = serve(host_config(root2.path(), None, source.clone(), provider)).expect("host serves");
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
    agent
        .ask_user("o", &["x".repeat(2_001)], 10_000)
        .unwrap();
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
    let held = until_ui(&app, "the held prompt", |frame| matches!(frame, UiWire::Show { .. }));
    // …then the queue's full worth of waiting asks…
    for index in 0..4 {
        agent
            .ask_user(&format!("w{index}"), &[format!("Q{index}?")], 30_000)
            .unwrap();
    }
    // …and one past the bound: refused immediately and typed.
    agent.ask_user("overflow", &["One too many?".into()], 30_000).unwrap();
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
    until_ui(&app, "the first queued prompt", |frame| matches!(frame, UiWire::Show { .. }));

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
    let shown = until_ui(&app, "the prompt", |frame| matches!(frame, UiWire::Show { .. }));
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
    let shown = until_ui(&app, "the second prompt", |frame| matches!(frame, UiWire::Show { .. }));
    app.prompt_ack(&ask_id_of(&shown), true).unwrap();
    until_event(&app, "capture.started", |e| e.type_name() == "capture.started");
    app.prompt_done(&ask_id_of(&shown)).unwrap();
    match until_ask(&agent, "2") {
        AskOutcome::Answered { text, .. } => assert_eq!(text, "fine"),
        other => panic!("second ask: {other:?}"),
    }

    host.shutdown();
}
