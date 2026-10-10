//! #220 step B: the host transcribes over the real transport. A take the
//! app records is stored with the intent to transcribe it and transcribed
//! by the host — live text while it records, the stream's final (or the
//! stored take uploaded) once stored — and every window hears how it
//! went, the owner as the one to act on it. A retry is a new result for
//! the window that asked. A take the store holds the intent for is
//! transcribed once, however the host that stored it ended.

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use starling_dictation::store_v2::{CommitMark, StoreV2, TakeMeta};
use starling_runtime::machine::capture::{CaptureConfig, V2CaptureStore};
use starling_runtime::protocol::Command;
use starling_runtime::testing::{FakeCaptureSource, FakeTakeScript};
use starling_runtime_host::client::{HostClient, TakeWire};
use starling_runtime_host::engine::EngineChoice;
use starling_runtime_host::frame::{TranscribeWith, TranscriptionState};
use starling_runtime_host::{serve, HostConfig, HostHandle};

#[path = "common/fake_engine.rs"]
mod fake_engine;
use fake_engine::{FakeEngine, Reply, StreamMode};

fn config(root: &Path, scripts: Vec<FakeTakeScript>, engine: &FakeEngine) -> HostConfig {
    let mut config = HostConfig::new(root, root.join("endpoints"))
        .with_engine(EngineChoice::Manual {
            endpoint: engine.endpoint(),
            model: "fake-model".to_string(),
        });
    config.runtime = config
        .runtime
        .with_capture_source(FakeCaptureSource::new(scripts))
        .with_capture_store(Arc::new(V2CaptureStore::open(root).expect("v2 store opens")))
        .with_capture_config(CaptureConfig {
            journals_dir: root.join("journals"),
            poll_interval: Duration::from_millis(10),
            ..CaptureConfig::default()
        });
    config
}

fn connect(host: &HostHandle) -> HostClient {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        match HostClient::connect(host.socket_path()) {
            Ok(client) => return client,
            Err(err) => {
                assert!(Instant::now() < deadline, "no host: {err}");
                std::thread::sleep(Duration::from_millis(20));
            }
        }
    }
}

fn watching(host: &HostHandle) -> HostClient {
    let client = connect(host);
    client.take_watch().expect("watching");
    client
}

fn until_take(
    client: &HostClient,
    label: &str,
    seen: &mut Vec<TakeWire>,
    predicate: impl Fn(&TakeWire) -> bool,
) -> TakeWire {
    let start = Instant::now();
    loop {
        match client.recv_take_timeout(Duration::from_millis(20)) {
            Ok(frame) => {
                seen.push(frame.clone());
                if predicate(&frame) {
                    return frame;
                }
            }
            Err(starling_runtime::channel::RecvError::Timeout) => {
                assert!(
                    start.elapsed() < Duration::from_secs(15),
                    "timed out waiting for {label}; saw {seen:#?}"
                );
            }
            Err(other) => panic!("take feed error: {other:?} ({})", client.close_reason()),
        }
    }
}

fn record(client: &HostClient, take: &str) {
    client
        .send(Some(take), Command::CaptureStart { policy: "push-to-talk".into() })
        .expect("start accepted");
    let mut seen = Vec::new();
    until_take(client, "a status tick", &mut seen, |frame| {
        matches!(frame, TakeWire::Live { take: name, status: Some(_), .. } if name == take)
    });
    std::thread::sleep(Duration::from_millis(200));
    client
        .send(Some(take), Command::CaptureStop { drain: Some(true) })
        .expect("stop accepted");
}

/// The final transcription frame for `stored_id`, with every frame seen.
fn finished(client: &HostClient, stored_id: &str, seen: &mut Vec<TakeWire>) -> (TranscriptionState, bool) {
    let frame = until_take(client, "the transcription's end", seen, |frame| {
        matches!(frame, TakeWire::Transcription { stored_id: id, state, .. } if id == stored_id && state.is_final())
    });
    let TakeWire::Transcription { state, yours, .. } = frame else {
        unreachable!()
    };
    (state, yours)
}

fn stored_id_of(client: &HostClient, take: &str, seen: &mut Vec<TakeWire>) -> String {
    let frame = until_take(client, "the stored row", seen, |frame| {
        matches!(frame, TakeWire::Persisted { take: name, .. } if name == take)
    });
    match frame {
        TakeWire::Persisted {
            stored_id: Some(id),
            ..
        } => id,
        other => panic!("not stored: {other:?}"),
    }
}

fn completed_attempts(root: &Path, id: &str) -> Vec<String> {
    let store = StoreV2::open(root).expect("store");
    store
        .attempts_for(id)
        .expect("attempts")
        .into_iter()
        .filter(|attempt| attempt.is_final_transcript())
        .map(|attempt| attempt.text)
        .collect()
}

#[test]
fn a_take_is_transcribed_by_the_host_and_its_owner_is_the_one_to_act_on_it() {
    let root = tempfile::tempdir().unwrap();
    let engine = FakeEngine::start(vec![Reply::Text("hello from the host".into())], StreamMode::Refuse);
    let mut host = serve(config(root.path(), vec![FakeTakeScript::clean()], &engine)).expect("serves");
    let owner = watching(&host);
    let other = watching(&host);

    record(&owner, "take-1");
    let mut seen = Vec::new();
    let id = stored_id_of(&owner, "take-1", &mut seen);
    let (state, yours) = finished(&owner, &id, &mut seen);
    assert_eq!(
        state,
        TranscriptionState::Completed {
            text: "hello from the host".into(),
            kept_earlier: false
        }
    );
    assert!(yours, "the owner acts on its take's result");
    let started = seen.iter().position(|frame| {
        matches!(frame, TakeWire::Transcription { state: TranscriptionState::Started { backend }, yours: true, .. } if backend == "openai:fake-model")
    });
    let stored = seen.iter().position(|frame| matches!(frame, TakeWire::Persisted { .. }));
    assert!(started.is_some() && stored < started, "stored, then started: {seen:#?}");

    let mut theirs = Vec::new();
    let (state, yours) = finished(&other, &id, &mut theirs);
    assert!(matches!(state, TranscriptionState::Completed { .. }));
    assert!(!yours, "another window only shows it");

    assert_eq!(completed_attempts(root.path(), &id), vec!["hello from the host"]);
    assert!(!StoreV2::open(root.path()).unwrap().transcription_wanted(&id).unwrap());
    assert_eq!(engine.batch_requests(), 1);
    host.shutdown();
}

#[test]
fn live_text_streams_while_the_take_records_and_the_final_comes_from_the_stream() {
    let root = tempfile::tempdir().unwrap();
    let engine = FakeEngine::start(
        Vec::new(),
        StreamMode::Echo {
            partial: "hello".into(),
            final_text: "hello there".into(),
        },
    );
    let mut host = serve(config(root.path(), vec![FakeTakeScript::clean()], &engine)).expect("serves");
    let app = watching(&host);
    app.send(Some("take-live"), Command::CaptureStart { policy: "push-to-talk".into() })
        .expect("start");
    let mut seen = Vec::new();
    let preview = until_take(&app, "live text", &mut seen, |frame| {
        matches!(frame, TakeWire::LiveText { take, partial: Some(_), .. } if take == "take-live")
    });
    let TakeWire::LiveText {
        partial: Some(partial),
        ..
    } = preview
    else {
        unreachable!()
    };
    assert_eq!(partial.text, "hello");
    // The owner's ticks carry its level meter.
    until_take(&app, "a level meter", &mut seen, |frame| {
        matches!(frame, TakeWire::Live { meter: Some(meter), .. } if !meter.is_empty())
    });
    app.send(Some("take-live"), Command::CaptureStop { drain: Some(true) })
        .expect("stop");
    let id = stored_id_of(&app, "take-live", &mut seen);
    let (state, _) = finished(&app, &id, &mut seen);
    assert_eq!(
        state,
        TranscriptionState::Completed {
            text: "hello there".into(),
            kept_earlier: false
        }
    );
    assert_eq!(engine.stream_sessions(), 1);
    assert!(engine.stream_audio_frames() > 0);
    assert_eq!(engine.batch_requests(), 0, "the stream's final, no upload");
    assert_eq!(completed_attempts(root.path(), &id), vec!["hello there"]);
    host.shutdown();
}

#[test]
fn a_retry_adds_a_result_for_the_window_that_asked() {
    let root = tempfile::tempdir().unwrap();
    let engine = FakeEngine::start(vec![Reply::Text("first".into())], StreamMode::Refuse);
    let server = FakeEngine::start(
        vec![Reply::Text("second".into()), Reply::Text(" ".into())],
        StreamMode::Refuse,
    );
    let mut host = serve(config(root.path(), vec![FakeTakeScript::clean()], &engine)).expect("serves");
    let owner = watching(&host);
    let asker = watching(&host);
    record(&owner, "take-r");
    let mut seen = Vec::new();
    let id = stored_id_of(&owner, "take-r", &mut seen);
    finished(&owner, &id, &mut seen);

    let with = TranscribeWith::Server {
        endpoint: server.endpoint(),
        model: "big".into(),
    };
    asker.transcribe("r_1", &id, with.clone()).expect("asked");
    let mut theirs = Vec::new();
    let frame = until_take(&asker, "the retry's end", &mut theirs, |frame| {
        matches!(frame, TakeWire::Transcription { req: Some(req), state, .. } if req == "r_1" && state.is_final())
    });
    let TakeWire::Transcription { state, yours, .. } = frame else {
        unreachable!()
    };
    assert_eq!(
        state,
        TranscriptionState::Completed {
            text: "second".into(),
            kept_earlier: false
        }
    );
    assert!(yours, "the result is for the window that asked");
    assert!(theirs.iter().any(|frame| matches!(
        frame,
        TakeWire::Transcription { state: TranscriptionState::Started { backend }, .. } if backend == "openai:big"
    )));
    let mut owners = Vec::new();
    let frame = until_take(&owner, "the retry, seen by the owner", &mut owners, |frame| {
        matches!(frame, TakeWire::Transcription { req: Some(req), state, .. } if req == "r_1" && state.is_final())
    });
    assert!(matches!(frame, TakeWire::Transcription { yours: false, .. }));

    // A blank retry is kept, and the take keeps showing its words.
    asker.transcribe("r_2", &id, with).expect("asked");
    let frame = until_take(&asker, "the blank retry", &mut theirs, |frame| {
        matches!(frame, TakeWire::Transcription { req: Some(req), state, .. } if req == "r_2" && state.is_final())
    });
    assert!(
        matches!(
            frame,
            TakeWire::Transcription { state: TranscriptionState::Completed { kept_earlier: true, .. }, .. }
        ),
        "{frame:?}"
    );
    assert_eq!(completed_attempts(root.path(), &id), vec!["first", "second", " "]);
    host.shutdown();
}

#[test]
fn a_retry_that_cannot_run_leaves_the_take_alone() {
    let root = tempfile::tempdir().unwrap();
    let engine = FakeEngine::start(vec![Reply::Text("kept".into())], StreamMode::Refuse);
    let mut host = serve(config(root.path(), vec![FakeTakeScript::clean()], &engine)).expect("serves");
    let app = watching(&host);
    record(&app, "take-n");
    let mut seen = Vec::new();
    let id = stored_id_of(&app, "take-n", &mut seen);
    finished(&app, &id, &mut seen);
    // The host serves the user's server: no built-in model to wait for.
    app.transcribe(
        "r_model",
        &id,
        TranscribeWith::Model {
            model_id: "moss".into(),
        },
    )
    .unwrap();
    let frame = until_take(&app, "the refusal", &mut seen, |frame| {
        matches!(frame, TakeWire::Transcription { req: Some(req), state, .. } if req == "r_model" && state.is_final())
    });
    assert!(
        matches!(&frame, TakeWire::Transcription { state: TranscriptionState::Refused { message }, .. } if message.contains("unchanged")),
        "{frame:?}"
    );
    let store = StoreV2::open(root.path()).unwrap();
    assert_eq!(store.attempts_for(&id).unwrap().len(), 1, "nothing was attempted");
    drop(store);
    host.shutdown();
}

#[test]
fn a_failed_transcription_is_recorded_with_the_take_and_retried_only_on_request() {
    let root = tempfile::tempdir().unwrap();
    let engine = FakeEngine::start(vec![Reply::Status("500 Internal Server Error")], StreamMode::Refuse);
    let mut host = serve(config(root.path(), vec![FakeTakeScript::clean()], &engine)).expect("serves");
    let app = watching(&host);
    record(&app, "take-f");
    let mut seen = Vec::new();
    let id = stored_id_of(&app, "take-f", &mut seen);
    let (state, yours) = finished(&app, &id, &mut seen);
    assert!(yours);
    assert!(
        matches!(&state, TranscriptionState::Failed { transport: false, .. }),
        "{state:?}"
    );
    let store = StoreV2::open(root.path()).unwrap();
    let attempts = store.attempts_for(&id).unwrap();
    assert_eq!(attempts.len(), 1);
    assert_eq!(attempts[0].status, "failed");
    assert!(!store.transcription_wanted(&id).unwrap(), "a failure ends the intent");
    drop(store);
    host.shutdown();
}

/// A take whose host died after storing it — before its transcription was
/// claimed — or while transcribing it: the next host transcribes it once,
/// and the one after that leaves it alone.
#[test]
fn a_take_left_waiting_by_a_dead_host_is_transcribed_once_by_the_next() {
    let root = tempfile::tempdir().unwrap();
    // Stored with its intent and claimed by nobody (the host died right
    // after the commit) …
    let unclaimed = {
        let mut store = StoreV2::open(root.path()).unwrap();
        let mut meta = TakeMeta::for_device("test-device");
        meta.transcribe = true;
        let mut take = store.begin_take(meta).unwrap();
        take.append_and_seal(&vec![0.1f32; 16_000]).unwrap();
        take.finalize().unwrap().commit_marked(&mut store, CommitMark::Complete).unwrap().record.id
    };
    // … and one claimed by a host that died mid-request (its attempt
    // started, its marker's lock gone with it).
    let claimed = {
        let mut store = StoreV2::open(root.path()).unwrap();
        let mut meta = TakeMeta::for_device("test-device");
        meta.transcribe = true;
        let mut take = store.begin_take(meta).unwrap();
        take.append_and_seal(&vec![0.2f32; 16_000]).unwrap();
        let id = take.finalize().unwrap().commit_marked(&mut store, CommitMark::Complete).unwrap().record.id;
        store.claim_transcription(&id, "openai:fake-model", None).unwrap();
        id
    };
    let engine = FakeEngine::start(
        vec![Reply::Text("one".into()), Reply::Text("two".into())],
        StreamMode::Refuse,
    );
    let mut host = serve(config(root.path(), Vec::new(), &engine)).expect("serves");
    let deadline = Instant::now() + Duration::from_secs(15);
    while completed_attempts(root.path(), &unclaimed).is_empty()
        || completed_attempts(root.path(), &claimed).is_empty()
    {
        assert!(Instant::now() < deadline, "the waiting takes were not transcribed");
        std::thread::sleep(Duration::from_millis(20));
    }
    host.shutdown();
    let mut texts = completed_attempts(root.path(), &unclaimed);
    texts.extend(completed_attempts(root.path(), &claimed));
    texts.sort();
    assert_eq!(texts, vec!["one", "two"], "each transcribed once");

    // The next host finds nothing left to do.
    let mut again = serve(config(root.path(), Vec::new(), &engine)).expect("serves again");
    std::thread::sleep(Duration::from_millis(500));
    again.shutdown();
    assert_eq!(engine.batch_requests(), 2);
    assert_eq!(completed_attempts(root.path(), &unclaimed).len(), 1);
    assert_eq!(completed_attempts(root.path(), &claimed).len(), 1);
}

/// A take that stopped but whose host died before storing it (its journal
/// finalized, its commit never made) comes back at the next start and is
/// transcribed there, as its own commit would have had it.
#[test]
fn a_take_stopped_but_not_stored_when_its_host_died_is_transcribed_by_the_next() {
    let root = tempfile::tempdir().unwrap();
    let id = {
        let scratch = tempfile::tempdir().unwrap();
        let store = StoreV2::open(scratch.path()).unwrap();
        let mut take = store.begin_take(TakeMeta::for_device("test-device")).unwrap();
        let id = take.id().to_string();
        take.append_frames(&vec![0.1f32; 16_000]).unwrap();
        take.finalize().unwrap();
        let journals = root.path().join("journals");
        std::fs::create_dir_all(&journals).unwrap();
        let path = journals.join(format!("{id}.sj"));
        std::fs::rename(scratch.path().join("staging").join(format!("{id}.sj")), &path).unwrap();
        std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_modified(std::time::SystemTime::now() - Duration::from_secs(120))
            .unwrap();
        id
    };
    let engine = FakeEngine::start(vec![Reply::Text("recovered words".into())], StreamMode::Refuse);
    let mut host = serve(config(root.path(), Vec::new(), &engine)).expect("serves");
    let deadline = Instant::now() + Duration::from_secs(15);
    while completed_attempts(root.path(), &id).is_empty() {
        assert!(Instant::now() < deadline, "the recovered take was not transcribed");
        std::thread::sleep(Duration::from_millis(20));
    }
    host.shutdown();
    assert_eq!(completed_attempts(root.path(), &id), vec!["recovered words"]);
    assert_eq!(engine.batch_requests(), 1);
}

/// Two windows on one host that dies mid-transcription: the take is
/// transcribed once by the next host, never by a window.
#[test]
fn two_windows_and_a_host_shut_down_mid_transcription_still_transcribe_once() {
    let root = tempfile::tempdir().unwrap();
    let held = FakeEngine::start(vec![Reply::Held("never".into())], StreamMode::Refuse);
    let mut host = serve(config(root.path(), vec![FakeTakeScript::clean()], &held)).expect("serves");
    let owner = watching(&host);
    let other = watching(&host);
    record(&owner, "take-x");
    let mut seen = Vec::new();
    let id = stored_id_of(&owner, "take-x", &mut seen);
    until_take(&owner, "the transcription starting", &mut seen, |frame| {
        matches!(frame, TakeWire::Transcription { state: TranscriptionState::Started { .. }, .. })
    });
    drop(other);
    host.shutdown();
    held.release();
    let engine = FakeEngine::start(vec![Reply::Text("once".into())], StreamMode::Refuse);
    let mut next = serve(config(root.path(), Vec::new(), &engine)).expect("serves again");
    let deadline = Instant::now() + Duration::from_secs(15);
    while completed_attempts(root.path(), &id).is_empty() {
        assert!(Instant::now() < deadline, "not transcribed after the restart");
        std::thread::sleep(Duration::from_millis(20));
    }
    next.shutdown();
    assert_eq!(completed_attempts(root.path(), &id), vec!["once"]);
}

/// The host **process** killed (SIGKILL — no shutdown runs) while it
/// transcribes a take: the next host process transcribes the take once,
/// and the dead host's attempt reads failed, never as a second result.
/// Unix: the host binary finds its settings through `XDG_CONFIG_HOME`.
#[cfg(unix)]
#[test]
fn a_host_process_killed_mid_transcription_leaves_the_take_to_the_next_one() {
    use std::process::{Command as ProcessCommand, Stdio};
    const HOST_BIN: &str = env!("CARGO_BIN_EXE_starling-runtime-host");

    let scratch = tempfile::tempdir().unwrap();
    let root = scratch.path().join("root");
    let runtime_dir = scratch.path().join("run");
    let engine = FakeEngine::start(
        vec![Reply::Held("never".into()), Reply::Text("once".into())],
        StreamMode::Refuse,
    );
    // The desktop settings the host binary follows: the user's own server.
    let config_home = scratch.path().join("config");
    std::fs::create_dir_all(config_home.join("starling-gpui")).unwrap();
    let mut settings = starling_dictation::settings::Settings::default_settings();
    settings.engine.mode = starling_dictation::settings::EngineMode::Manual;
    settings.endpoint = engine.endpoint();
    settings.model = "fake-model".to_string();
    std::fs::write(
        config_home.join("starling-gpui").join("settings.json"),
        serde_json::to_vec(&settings).unwrap(),
    )
    .unwrap();
    // A take stored for transcription that no host has claimed yet.
    let id = {
        let mut store = StoreV2::open(&root).unwrap();
        let mut meta = TakeMeta::for_device("test-device");
        meta.transcribe = true;
        let mut take = store.begin_take(meta).unwrap();
        take.append_and_seal(&vec![0.1f32; 16_000]).unwrap();
        take.finalize().unwrap().commit_marked(&mut store, CommitMark::Complete).unwrap().record.id
    };
    let spawn = || {
        ProcessCommand::new(HOST_BIN)
            .arg("--root")
            .arg(&root)
            .arg("--runtime-dir")
            .arg(&runtime_dir)
            .env("XDG_CONFIG_HOME", &config_home)
            .env("XDG_DATA_HOME", scratch.path().join("data"))
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("host binary spawns")
    };
    let wait_for = |what: &str, done: &dyn Fn() -> bool| {
        let deadline = Instant::now() + Duration::from_secs(20);
        while !done() {
            assert!(Instant::now() < deadline, "timed out waiting for {what}");
            std::thread::sleep(Duration::from_millis(20));
        }
    };

    let socket = HostConfig::new(&root, &runtime_dir).socket_path();
    let windows = |label: &str| -> Vec<HostClient> {
        let deadline = Instant::now() + Duration::from_secs(20);
        (0..2)
            .map(|_| loop {
                if let Ok(client) = HostClient::connect(&socket) {
                    client.take_watch().expect("watching");
                    return client;
                }
                assert!(Instant::now() < deadline, "no host for {label}");
                std::thread::sleep(Duration::from_millis(20));
            })
            .collect()
    };

    // Two windows follow the host that dies mid-transcription …
    let mut first = spawn();
    let before = windows("the first host");
    wait_for("the first host's request", &|| engine.batch_requests() == 1);
    first.kill().expect("killed");
    first.wait().expect("reaped");
    drop(before);

    // … and reconnect to the next one, which transcribes it once (at its
    // startup — possibly before the windows are back).
    let mut second = spawn();
    let after = windows("the second host");
    wait_for("the transcript", &|| !completed_attempts(&root, &id).is_empty());
    drop(after);
    second.kill().expect("killed");
    second.wait().expect("reaped");
    engine.release();

    assert_eq!(completed_attempts(&root, &id), vec!["once"], "transcribed once");
    let store = StoreV2::open(&root).unwrap();
    let attempts = store.attempts_for(&id).unwrap();
    assert_eq!(attempts.len(), 2, "the dead host's attempt and the one that finished");
    assert!(attempts.iter().any(|attempt| attempt.status == "failed"));
    assert!(!store.transcription_wanted(&id).unwrap());
    assert_eq!(engine.batch_requests(), 2);
}

/// An import stored with its intent and asked for at once (`due`): run
/// once, whether the request or the host's own look at the store gets to
/// it first, and never again by this host or the next; the window that
/// asked is the one to act on it.
#[test]
fn an_import_asked_for_is_transcribed_once() {
    let root = tempfile::tempdir().unwrap();
    let id = {
        let mut store = StoreV2::open(root.path()).unwrap();
        let mut meta = TakeMeta::for_device("import");
        meta.transcribe = true;
        let mut take = store.begin_take(meta).unwrap();
        take.append_and_seal(&vec![0.1f32; 16_000]).unwrap();
        take.finalize().unwrap().commit_marked(&mut store, CommitMark::Complete).unwrap().record.id
    };
    let engine = FakeEngine::start(
        vec![Reply::Text("imported".into()), Reply::Text("again".into())],
        StreamMode::Refuse,
    );
    let mut host = serve(config(root.path(), Vec::new(), &engine)).expect("serves");
    let app = watching(&host);
    app.transcribe_due(&id).unwrap();
    app.transcribe_due(&id).unwrap();
    let deadline = Instant::now() + Duration::from_secs(15);
    while completed_attempts(root.path(), &id).is_empty() {
        assert!(Instant::now() < deadline, "the import was not transcribed");
        std::thread::sleep(Duration::from_millis(20));
    }
    // Asked again once it ran: nothing is due, nothing runs.
    app.transcribe_due(&id).unwrap();
    std::thread::sleep(Duration::from_millis(300));
    host.shutdown();
    let mut next = serve(config(root.path(), Vec::new(), &engine)).expect("serves again");
    std::thread::sleep(Duration::from_millis(500));
    next.shutdown();
    assert_eq!(completed_attempts(root.path(), &id), vec!["imported"]);
    assert_eq!(engine.batch_requests(), 1, "transcribed once");
    assert!(!StoreV2::open(root.path()).unwrap().transcription_wanted(&id).unwrap());
}

/// A retry the host accepted but has not started (both job slots busy)
/// keeps its take's audio from every process's upkeep — also once the
/// window that asked is gone.
#[test]
fn a_queued_retry_holds_its_audio_after_its_window_left() {
    let root = tempfile::tempdir().unwrap();
    let ids: Vec<String> = {
        let mut store = StoreV2::open(root.path()).unwrap();
        (0..3)
            .map(|_| {
                let mut take = store.begin_take(TakeMeta::for_device("t")).unwrap();
                take.append_and_seal(&vec![0.1f32; 32_000]).unwrap();
                take.finalize().unwrap().commit_marked(&mut store, CommitMark::Complete).unwrap().record.id
            })
            .collect()
    };
    let held = FakeEngine::start(
        vec![Reply::Held("a".into()), Reply::Held("b".into()), Reply::Held("c".into())],
        StreamMode::Refuse,
    );
    let mut host = serve(config(root.path(), Vec::new(), &held)).expect("serves");
    let with = TranscribeWith::Server {
        endpoint: held.endpoint(),
        model: "fake-model".into(),
    };
    {
        let asker = watching(&host);
        for (index, id) in ids.iter().enumerate() {
            asker.transcribe(&format!("r_{index}"), id, with.clone()).unwrap();
        }
        let deadline = Instant::now() + Duration::from_secs(10);
        while held.batch_requests() < 2 {
            assert!(Instant::now() < deadline, "the first two retries did not start");
            std::thread::sleep(Duration::from_millis(20));
        }
    }
    std::thread::sleep(Duration::from_millis(200));
    // Another process's upkeep would compress (or retire) any of them now.
    let other = StoreV2::open(root.path()).unwrap();
    let candidates: Vec<String> = other
        .compression_candidates(usize::MAX)
        .unwrap()
        .into_iter()
        .map(|job| job.id)
        .collect();
    for id in &ids {
        assert!(!candidates.contains(id), "{id} is in use (running or queued)");
    }
    drop(other);
    held.release();
    host.shutdown();
}
