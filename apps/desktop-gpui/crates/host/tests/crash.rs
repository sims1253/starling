//! Crashes, with real process deaths (#220, E17 I4 §4). A killed
//! **host** — mid-take, after a take stopped but before it was stored,
//! mid-transcription — costs no acknowledged audio and no transcript:
//! the next host brings the take back (cut takes as interrupted, with a
//! note on what was lost; stopped ones whole) and transcribes what was
//! due exactly once. A killed **renderer** — during the handshake, with a
//! command in flight, mid-take, during the host's shutdown — costs only
//! its projection: its take is adopted by the next window or stopped,
//! stored and transcribed by the host; its connection slot returns; the
//! endpoint and the data root are never wedged. Never two owners.
//!
//! Renderers are `renderer-double`, a real child process killed at the
//! phase its mode names. Hosts are the real `starling-runtime-host`
//! binary, or — where the host has to record — this test binary re-run
//! as [`host_child_process`]: the real `serve` with a microphone that
//! journals like the recorder does. `Child::kill` is SIGKILL on unix and
//! TerminateProcess on Windows: no handler, no cleanup.

mod common;

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Child, Command as ProcessCommand, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use common::fake_engine::{FakeEngine, Reply, StreamMode};
use common::reaped::Reaped;
use common::*;
use starling_dictation::journal::{FileSink, JournalWriter};
use starling_dictation::recorder::{CaptureGap, CapturedTake, RecorderError, RecorderFault};
use starling_dictation::store_v2::{read_audio_journal, CaptureStatus, StoreV2};
use starling_runtime::machine::capture::{CaptureSession, CaptureSource};
use starling_runtime::machine::Receipt;
use starling_runtime::protocol::{Command, JobLimits, Revision};
use starling_runtime::testing::FakeTakeScript;
use starling_runtime_host::client::{HostClient, TakeWire};
use starling_runtime_host::frame::{TakeOwner, TranscribeWith};
use starling_runtime_host::{serve, HostConfig, HostError};

/// The renderer double (a bin target of this crate).
const DOUBLE: &str = env!("CARGO_BIN_EXE_renderer-double");
/// The real host binary.
const HOST_BIN: &str = env!("CARGO_BIN_EXE_starling-runtime-host");
/// Set for [`host_child_process`]: the data root it serves.
const CHILD_ROOT: &str = "STARLING_CRASH_SUITE_HOST_ROOT";

fn socket_of(root: &Path) -> PathBuf {
    HostConfig::new(root, root.join("endpoints")).socket_path()
}

/// The real host binary on `root`, with no engine of the user's.
fn host_binary(root: &Path) -> Reaped {
    Reaped(
        ProcessCommand::new(HOST_BIN)
            .args(["--engine", "none"])
            .arg("--root")
            .arg(root)
            .arg("--runtime-dir")
            .arg(root.join("endpoints"))
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("host binary spawns"),
    )
}

fn hard_kill(child: &mut Child) {
    child.kill().expect("SIGKILL");
    let _ = child.wait();
}

fn set_limits(client: &HostClient, max_queued: u32) -> Result<Receipt, starling_runtime_host::client::ClientError> {
    client.send(
        None,
        Command::JobsSetLimits(JobLimits {
            max_queued,
            max_concurrent: 1,
            per_route: vec![],
        }),
    )
}

fn note_revision(text: &str) -> Revision {
    Revision {
        rev_id: "rev-1".into(),
        base_revision: 0,
        source_attempt_ids: vec![],
        instruction_template_id: "tpl-none".into(),
        text: text.into(),
        status: "candidate".into(),
        provenance: "manual".into(),
    }
}

// --------------------------------------------------------------------- //
// The host child: the real serve(), recording into the real journal tree
// --------------------------------------------------------------------- //

/// A microphone that journals a steady tone into the host's journal tree
/// the way the recorder does: frames appended as they arrive, a boundary
/// fsynced on the recorder's cadence — acknowledged is what survives a
/// kill. It never stops cleanly: the scenarios kill its host instead.
struct JournalingMicrophone;

struct JournalingSession {
    captured: Arc<AtomicU64>,
    acknowledged: Arc<AtomicU64>,
    running: Arc<AtomicBool>,
}

impl CaptureSource for JournalingMicrophone {
    fn start(&self, journals_dir: &Path, _policy: &str) -> Result<Box<dyn CaptureSession>, String> {
        std::fs::create_dir_all(journals_dir).map_err(|err| err.to_string())?;
        let mut journal: JournalWriter<FileSink> =
            JournalWriter::create(journals_dir, 16_000).map_err(|err| err.to_string())?;
        let session = JournalingSession {
            captured: Arc::default(),
            acknowledged: Arc::default(),
            running: Arc::new(AtomicBool::new(true)),
        };
        let (captured, acknowledged, running) = (
            Arc::clone(&session.captured),
            Arc::clone(&session.acknowledged),
            Arc::clone(&session.running),
        );
        std::thread::spawn(move || {
            let mut index = 0u64;
            while running.load(Ordering::SeqCst) {
                let block: Vec<f32> = (index..index + 160).map(|i| 0.2 * (i as f32 * 0.05).sin()).collect();
                index += 160;
                journal.append_frames(&block).expect("journal append");
                captured.store(index, Ordering::SeqCst);
                if journal.boundary_due() {
                    let confirmed = journal.write_boundary().expect("journal boundary");
                    acknowledged.store(confirmed, Ordering::SeqCst);
                }
                std::thread::sleep(Duration::from_millis(10));
            }
        });
        Ok(Box::new(session))
    }
}

impl CaptureSession for JournalingSession {
    fn sample_rate(&self) -> u32 {
        16_000
    }
    fn captured_sample_count(&self) -> u64 {
        self.captured.load(Ordering::SeqCst)
    }
    fn acknowledged_samples(&self) -> u64 {
        self.acknowledged.load(Ordering::SeqCst)
    }
    fn source_clip_ratio(&self) -> f64 {
        0.0
    }
    fn gaps(&self) -> Vec<CaptureGap> {
        Vec::new()
    }
    fn capture_fault(&self) -> Option<RecorderFault> {
        None
    }
    fn latest_window(&self, _n: usize) -> Vec<f32> {
        Vec::new()
    }
    fn stop(self: Box<Self>) -> Result<CapturedTake, RecorderError> {
        self.running.store(false, Ordering::SeqCst);
        Err(RecorderError::Device("the crash suite's microphone is never stopped".into()))
    }
}

/// Not a scenario: the host process the scenarios below spawn (this test
/// binary re-run with [`CHILD_ROOT`] set) and kill. It serves that root
/// with the [`JournalingMicrophone`] until it dies.
#[test]
#[ignore = "the host child process the crash scenarios spawn and kill"]
fn host_child_process() {
    let Some(root) = std::env::var_os(CHILD_ROOT).map(PathBuf::from) else {
        return;
    };
    let _host = serve(host_config(&root, Arc::new(JournalingMicrophone), None)).expect("the child host serves");
    loop {
        std::thread::sleep(Duration::from_secs(1));
    }
}

fn host_child(root: &Path) -> Reaped {
    Reaped(
        ProcessCommand::new(std::env::current_exe().expect("this test binary"))
            .args(["--ignored", "--exact", "host_child_process", "--nocapture"])
            .env(CHILD_ROOT, root)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("the host child spawns"),
    )
}

/// Samples the take journaling in `root`'s tree has confirmed on disk.
fn confirmed_in_journal(root: &Path) -> usize {
    std::fs::read_dir(root.join("journals"))
        .into_iter()
        .flatten()
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("sj"))
        .filter_map(|path| read_audio_journal(&path).ok())
        .map(|audio| audio.samples.len())
        .max()
        .unwrap_or(0)
}

// --------------------------------------------------------------------- //
// Host deaths
// --------------------------------------------------------------------- //

/// The host process dies mid-take. Everything its journal had confirmed
/// comes back at the next start as an interrupted take whose note says
/// what survived — announced to the app, never transcribed on its own (a
/// cut take is the user's call), and retryable.
#[test]
fn a_host_killed_mid_take_brings_the_take_back_interrupted_with_a_note() {
    let scratch = tempfile::tempdir().unwrap();
    let root = scratch.path().join("root");
    std::fs::create_dir_all(&root).unwrap();
    let mut child = host_child(&root);
    let app = connect_to(&socket_of(&root));
    app.take_watch().expect("watching");
    start(&app, "take-k");
    wait_for("a second confirmed on disk", || confirmed_in_journal(&root) >= 16_000);
    let confirmed = confirmed_in_journal(&root);
    hard_kill(&mut child);
    drop(app);

    let engine = FakeEngine::start(vec![Reply::Text("retried words".into())], StreamMode::Refuse);
    let mut host = serve(scripted(&root, Vec::new(), Some(&engine))).expect("the next host serves");
    let app = connect(&host);
    let recovery = app.take_watch().expect("watching").expect("the startup recovery is reported");
    assert!(recovery.notice.contains("Recovered 1 recording"), "{recovery:?}");

    let store = StoreV2::open(&root).unwrap();
    let page = store.list_records(0, 10).unwrap();
    assert_eq!(page.total, 1, "the take came back once");
    let id = page.records[0].id().to_string();
    let record = store.get_capture(&id).unwrap().expect("row");
    assert_eq!(record.status, CaptureStatus::Interrupted);
    let note = record.recovery_note().expect("a note on what survived");
    assert!(note.contains("closed while this take was recording"), "{note}");
    assert!(note.contains("confirmed on disk"), "{note}");
    let samples = store.load_audio(&id).unwrap().samples.len();
    assert!(samples >= confirmed, "acknowledged audio survives the kill: {samples} < {confirmed}");
    assert!(!store.transcription_wanted(&id).unwrap(), "a cut take is not transcribed on its own");
    drop(store);
    std::thread::sleep(Duration::from_millis(300));
    assert_eq!(engine.batch_requests(), 0);

    // Retryable like any take.
    app.transcribe(
        "r_1",
        &id,
        TranscribeWith::Server {
            endpoint: engine.endpoint(),
            model: "fake-model".into(),
        },
    )
    .unwrap();
    assert_eq!(retried(&app, "r_1", &mut Vec::new()).0, completed("retried words"));
    drop(app);
    host.shutdown();
}

/// A take that stopped but whose host died before storing it (its journal
/// finalized, its commit never made) comes back whole at the next start
/// — once its writer could no longer be saving it — and is transcribed
/// there, as its own commit would have had it; the host after that finds
/// nothing left to do.
#[test]
fn a_take_stopped_but_not_stored_when_its_host_died_comes_back_whole_and_is_transcribed_once() {
    let root = tempfile::tempdir().unwrap();
    let journals = root.path().join("journals");
    std::fs::create_dir_all(&journals).unwrap();
    let samples: Vec<f32> = (0..16_000).map(|i| 0.1 * (i as f32 * 0.03).sin()).collect();
    let id = {
        let mut journal: JournalWriter<FileSink> = JournalWriter::create(&journals, 16_000).unwrap();
        journal.append_frames(&samples).unwrap();
        journal.finalize().unwrap();
        let path = journal.path().to_path_buf();
        let id = journal.id().to_string();
        drop(journal);
        // Old enough that no live process could still be saving it.
        std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_modified(std::time::SystemTime::now() - Duration::from_secs(120))
            .unwrap();
        id
    };
    let engine = FakeEngine::start(
        vec![Reply::Text("recovered words".into()), Reply::Text("again".into())],
        StreamMode::Refuse,
    );
    for _ in 0..2 {
        let mut host = serve(scripted(root.path(), Vec::new(), Some(&engine))).expect("serves");
        wait_for("the recovered take transcribed", || {
            !completed_attempts(root.path(), &id).is_empty()
        });
        std::thread::sleep(Duration::from_millis(300));
        host.shutdown();
    }
    assert_eq!(completed_attempts(root.path(), &id), vec!["recovered words"]);
    assert_eq!(engine.batch_requests(), 1, "transcribed exactly once");
    let store = StoreV2::open(root.path()).unwrap();
    assert_eq!(store.load_audio(&id).unwrap().samples, samples, "whole, bit for bit");
    let note = store.get_capture(&id).unwrap().unwrap().recovery_note().expect("note");
    assert!(note.contains("complete recording"), "{note}");
    assert_eq!(store.list_records(0, 10).unwrap().total, 1);
}

/// The host **process** killed while it transcribes a take: the next host
/// process transcribes it once, and the dead host's attempt reads failed,
/// never as a second result. Unix: the host binary finds its settings
/// through `XDG_CONFIG_HOME`.
#[cfg(unix)]
#[test]
fn a_host_process_killed_mid_transcription_leaves_the_take_to_the_next_one() {
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
    let id = stored_take(&root, true, 0.1);
    let spawn = || {
        Reaped(
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
                .expect("host binary spawns"),
        )
    };
    let socket = HostConfig::new(&root, &runtime_dir).socket_path();
    let windows = || -> Vec<HostClient> {
        (0..2)
            .map(|_| {
                let client = connect_to(&socket);
                client.take_watch().expect("watching");
                client
            })
            .collect()
    };

    // Two windows follow the host that dies mid-transcription …
    let mut first = spawn();
    let before = windows();
    wait_for("the first host's request", || engine.batch_requests() == 1);
    hard_kill(&mut first);
    drop(before);

    // … and reconnect to the next one, which transcribes it once (at its
    // startup — possibly before the windows are back).
    let mut second = spawn();
    let after = windows();
    wait_for("the transcript", || !completed_attempts(&root, &id).is_empty());
    drop(after);
    hard_kill(&mut second);
    engine.release();

    assert_eq!(completed_attempts(&root, &id), vec!["once"], "transcribed once");
    let store = StoreV2::open(&root).unwrap();
    let attempts = store.attempts_for(&id).unwrap();
    assert_eq!(attempts.len(), 2, "the dead host's attempt and the one that finished");
    assert!(attempts.iter().any(|attempt| attempt.status == "failed"));
    assert!(!store.transcription_wanted(&id).unwrap());
    assert_eq!(engine.batch_requests(), 2);
}

/// Takes a dead host left waiting — stored with the intent but claimed by
/// nobody (it died right after the commit), or claimed by a host that
/// died mid-request — and one whose host shut down with two windows open
/// mid-transcription: the next host transcribes each once, never a
/// window, and the host after it leaves them alone.
#[test]
fn takes_a_dead_host_left_waiting_are_transcribed_once_by_the_next() {
    let root = tempfile::tempdir().unwrap();
    // A host shut down while its windows waited on a transcription.
    let held = FakeEngine::start(vec![Reply::Held("never".into())], StreamMode::Refuse);
    let mut host = serve(scripted(root.path(), vec![FakeTakeScript::clean()], Some(&held))).expect("serves");
    let owner = watching(&host);
    let other = watching(&host);
    let interrupted = {
        record(&owner, "take-x");
        let mut seen = Vec::new();
        let id = stored_id_of(&owner, "take-x", &mut seen);
        until_take(&owner, "the transcription starting", &mut seen, |frame| {
            matches!(frame, TakeWire::Transcription { stored_id, state: starling_runtime_host::frame::TranscriptionState::Started { .. }, .. } if *stored_id == id)
        });
        id
    };
    drop(other);
    host.shutdown();
    held.release();
    drop(owner);
    // Stored with the intent by a host that died right after storing it,
    // and claimed by one that died mid-request (its claim's lock went
    // with it).
    let unclaimed = stored_take(root.path(), true, 0.1);
    let claimed = stored_take(root.path(), true, 0.2);
    StoreV2::open(root.path())
        .unwrap()
        .claim_transcription(&claimed, "openai:fake-model", None)
        .unwrap();

    let engine = FakeEngine::start(
        vec![
            Reply::Text("one".into()),
            Reply::Text("one".into()),
            Reply::Text("one".into()),
        ],
        StreamMode::Refuse,
    );
    let mut next = serve(scripted(root.path(), Vec::new(), Some(&engine))).expect("serves again");
    wait_for("the waiting takes transcribed", || {
        [&unclaimed, &claimed, &interrupted]
            .iter()
            .all(|id| !completed_attempts(root.path(), id).is_empty())
    });
    next.shutdown();
    let mut after = serve(scripted(root.path(), Vec::new(), Some(&engine))).expect("serves again");
    std::thread::sleep(Duration::from_millis(500));
    after.shutdown();
    for id in [&unclaimed, &claimed, &interrupted] {
        assert_eq!(completed_attempts(root.path(), id), vec!["one"], "{id}: transcribed once");
    }
    assert_eq!(engine.batch_requests(), 3);
}

/// An app process that died with a recognition attempt of its own
/// started: the host fails it at startup, ready to retry.
#[test]
fn an_attempt_a_dead_app_left_started_is_failed_at_the_next_start() {
    let root = tempfile::tempdir().unwrap();
    let id = stored_take(root.path(), false, 0.1);
    StoreV2::open(root.path())
        .unwrap()
        .begin_recognition(&id, "engine:test", None)
        .unwrap();
    let mut host = serve(scripted(root.path(), Vec::new(), None)).expect("serves");
    let app = watching(&host);
    let attempts = StoreV2::open(root.path()).unwrap().attempts_for(&id).unwrap();
    assert_eq!(attempts.last().unwrap().status, "failed", "{attempts:?}");
    drop(app);
    host.shutdown();
}

/// The ownership ladder with real processes: host A (the binary) serves a
/// renderer that writes a document; A is SIGKILLed (the data-root lock is
/// released by the OS, the socket file left behind); host B — the binary
/// again — takes over the same endpoint under a new owner id, and an
/// in-process successor after it reads the document back from storage.
#[test]
fn a_killed_host_is_succeeded_on_its_endpoint_and_its_renderers_data_survives() {
    let root = tempfile::tempdir().unwrap();
    let socket = socket_of(root.path());
    let mut first = host_binary(root.path());
    let renderer = connect_to(&socket);
    let owner_a = renderer.info.owner_id.clone();
    assert_eq!(renderer.info.pid, first.id());
    renderer
        .send(
            Some("doc-1"),
            Command::DocsUpdateHead {
                doc_id: "notes".into(),
                expected_base: 0,
                new_revision: note_revision("Written before the host died."),
            },
        )
        .expect("the renderer wrote through the host process");
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        assert!(Instant::now() < deadline, "the document was never written");
        if let Ok(event) = renderer.recv_event_timeout(Duration::from_millis(20)) {
            if event.type_name() == "docs.headUpdated" {
                break;
            }
        }
    }

    hard_kill(&mut first);
    wait_for("the renderer to notice the host died", || renderer.is_closed());
    drop(renderer);
    #[cfg(unix)]
    assert!(socket.exists(), "the killed host leaves its socket file behind");

    let mut second = host_binary(root.path());
    let client = connect_to(&socket);
    assert_ne!(client.info.owner_id, owner_a, "a new run, not the old one inherited");
    assert_eq!(client.info.pid, second.id());
    assert_eq!(client.snapshot().unwrap()["docs"]["state"], "Steady");
    drop(client);
    hard_kill(&mut second);

    let mut config = scripted(root.path(), Vec::new(), None);
    config.runtime = config.runtime.with_document_store(Arc::new(
        starling_runtime::machine::docs::V2DocumentStore::open(root.path()).expect("documents"),
    ));
    let mut successor = serve(config).expect("the successor owns the root");
    let reconnected = connect(&successor);
    match reconnected
        .send(Some("doc-2"), Command::DocsGet { doc_id: "notes".into(), page: 0 })
        .expect("get accepted")
    {
        Receipt::Served(view) => {
            assert_eq!(view["found"], true, "{view}");
            assert_eq!(view["headRevision"], 1, "{view}");
            assert_eq!(view["revisions"][0]["text"], "Written before the host died.");
        }
        other => panic!("expected a served view, got {other:?}"),
    }
    drop(reconnected);
    successor.shutdown();
}

/// A second host process against a live owner exits 0 with
/// `already-running` and the owner's endpoint — the launcher contract —
/// and the owner serves on.
#[test]
fn a_second_host_binary_reports_already_running_and_exits_zero() {
    let root = tempfile::tempdir().unwrap();
    let mut owner = serve(scripted(root.path(), Vec::new(), None)).expect("in-process owner serves");
    let client = connect(&owner);
    let mut child = Reaped(
        ProcessCommand::new(HOST_BIN)
            .args(["--engine", "none"])
            .arg("--root")
            .arg(root.path())
            .arg("--runtime-dir")
            .arg(root.path().join("endpoints"))
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("second host runs"),
    );
    let deadline = Instant::now() + Duration::from_secs(15);
    let status = loop {
        match child.try_wait().expect("try_wait the second host") {
            Some(status) => break status,
            None => {
                assert!(Instant::now() < deadline, "the second host did not exit within 15s");
                std::thread::sleep(Duration::from_millis(25));
            }
        }
    };
    let (mut stdout, mut stderr) = (String::new(), String::new());
    child.stdout.take().expect("piped").read_to_string(&mut stdout).expect("drain stdout");
    child.stderr.take().expect("piped").read_to_string(&mut stderr).expect("drain stderr");
    assert!(status.success(), "a live owner means client mode, not failure: {stderr}");
    let status: serde_json::Value = serde_json::from_str(stdout.trim())
        .unwrap_or_else(|err| panic!("stdout is one JSON line ({stdout:?}): {err}"));
    assert_eq!(status["status"], "already-running");
    assert_eq!(status["socket"].as_str().unwrap(), owner.socket_path().to_str().unwrap());
    set_limits(&client, 4).expect("the owner still serves after a second host tried the root");
    drop(client);
    owner.shutdown();
}

/// What a dead host may leave at its endpoint: a socket file that was
/// never a socket is taken over; a live listener that does not hold the
/// root is refused, and the root this attempt briefly held is released.
#[cfg(unix)]
#[test]
fn endpoint_residue_is_taken_over_and_a_squatter_is_refused() {
    let root = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(root.path().join("endpoints")).unwrap();
    let socket = socket_of(root.path());
    std::fs::write(&socket, b"not a socket").unwrap();
    let mut host = serve(scripted(root.path(), Vec::new(), None))
        .unwrap_or_else(|err| panic!("residue must be taken over, refused with {err}"));
    HostClient::connect(host.socket_path()).expect("serves on the taken-over path");
    host.shutdown();

    let squatter = std::os::unix::net::UnixListener::bind(&socket).unwrap();
    match serve(scripted(root.path(), Vec::new(), None)) {
        Err(HostError::ForeignServer(path)) => assert_eq!(path, socket),
        Err(other) => panic!("expected ForeignServer, got {other}"),
        Ok(host) => {
            drop(host);
            panic!("expected ForeignServer, a host bound over a squatter")
        }
    }
    drop(squatter);
    let _ = std::fs::remove_file(&socket);
    let mut successor = serve(scripted(root.path(), Vec::new(), None)).expect("serves once the endpoint is free");
    successor.shutdown();
}

/// The binary's usage surface: help on stdout with exit 0, an unknown
/// argument or engine source is a usage error (stderr, exit 2) — never a
/// silent fallback to (or away from) the user's engine.
#[test]
fn the_binary_documents_its_interface() {
    let output = ProcessCommand::new(HOST_BIN).arg("--help").output().unwrap();
    assert_eq!(output.status.code(), Some(0), "--help is not an error");
    let help = String::from_utf8_lossy(&output.stdout);
    for flag in ["--root", "--engine", "--runtime-dir"] {
        assert!(help.contains(flag), "{help}");
    }
    let output = ProcessCommand::new(HOST_BIN).arg("--bogus").output().unwrap();
    assert_eq!(output.status.code(), Some(2), "a usage error exits 2");
    assert!(String::from_utf8_lossy(&output.stderr).contains("--bogus"));
    let output = ProcessCommand::new(HOST_BIN).args(["--engine", "bundled"]).output().unwrap();
    assert_eq!(output.status.code(), Some(2), "an unknown engine source exits 2");
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("--engine takes settings or none"),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

// --------------------------------------------------------------------- //
// Renderer deaths
// --------------------------------------------------------------------- //

/// The renderer double, killed on drop: `ready` is awaited bounded, a
/// failing setup surfaces the double's stderr.
struct DoubleRenderer {
    child: Child,
}

impl DoubleRenderer {
    fn spawn(socket: &Path, mode: &str, take: Option<&str>) -> DoubleRenderer {
        let mut command = ProcessCommand::new(DOUBLE);
        command.arg("--socket").arg(socket).arg("--mode").arg(mode);
        if let Some(take) = take {
            command.arg("--take").arg(take);
        }
        let child = command
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("renderer double spawns");
        let mut double = DoubleRenderer { child };
        double.wait_ready();
        double
    }

    /// The double's single `ready` line: the phase after which the kill
    /// lands exactly where the test means it.
    fn wait_ready(&mut self) {
        let (tx, rx) = std::sync::mpsc::channel::<String>();
        let mut stdout = self.child.stdout.take().expect("piped stdout");
        std::thread::spawn(move || {
            let mut line = String::new();
            let _ = std::io::BufRead::read_line(&mut std::io::BufReader::new(&mut stdout), &mut line);
            let _ = tx.send(line);
        });
        let deadline = Instant::now() + Duration::from_secs(20);
        loop {
            match rx.recv_timeout(Duration::from_millis(50)) {
                Ok(line) if line.trim() == "ready" => return,
                Ok(other) => self.die_with_stderr(format!("renderer double printed {other:?} instead of ready")),
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                    if let Some(status) = self.child.try_wait().expect("try wait the double") {
                        self.die_with_stderr(format!("renderer double exited early: {status}"));
                    }
                    if Instant::now() > deadline {
                        self.die_with_stderr("renderer double never reached ready".to_string());
                    }
                }
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    self.die_with_stderr("renderer double's stdout closed".to_string())
                }
            }
        }
    }

    fn die_with_stderr(&mut self, message: String) -> ! {
        let mut stderr = String::new();
        if let Some(mut pipe) = self.child.stderr.take() {
            let _ = pipe.read_to_string(&mut stderr);
        }
        hard_kill(&mut self.child);
        panic!("{message}; stderr: {stderr}");
    }

    fn kill(&mut self) {
        hard_kill(&mut self.child);
    }
}

impl Drop for DoubleRenderer {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// A renderer killed between connect and its first frame must not hold
/// its connection slot past the kernel's EOF — in a crashloop of eight at
/// the harshest phase, with the cap at 2 (the double holds one slot, the
/// probe the other), a single leaked slot refuses the next cycle's probe.
/// The idle deadline is far off, so only the connection's own death can
/// return the slot.
#[test]
fn a_crashlooping_renderer_never_leaks_connection_slots() {
    let root = tempfile::tempdir().unwrap();
    let mut config = scripted(root.path(), Vec::new(), None);
    config.max_connections = 2;
    config.first_frame_idle = Duration::from_secs(60);
    let mut host = serve(config).expect("host serves");
    let socket = host.socket_path().to_path_buf();
    for cycle in 0..8 {
        let mut renderer = DoubleRenderer::spawn(&socket, "raw", None);
        let probe = connect_to(&socket);
        set_limits(&probe, 2 + cycle).unwrap_or_else(|err| panic!("cycle {cycle}: the host stopped serving: {err}"));
        drop(probe);
        renderer.kill();
        // Let the EOF land before the next cycle's admission so the kill,
        // not a scheduling race, is what each cycle exercises.
        std::thread::sleep(Duration::from_millis(120));
    }
    let last = connect_to(&socket);
    assert_eq!(last.snapshot().unwrap()["jobs"]["limits"]["maxQueued"], 2 + 7);
    drop(last);
    host.shutdown();
}

/// A renderer that dies between sending a command and reading its
/// receipt: the command still executes, the orphaned receipt costs
/// nothing, and the host keeps serving.
#[test]
fn a_renderer_killed_with_a_command_in_flight_costs_only_the_receipt() {
    let root = tempfile::tempdir().unwrap();
    let mut host = serve(scripted(root.path(), Vec::new(), None)).expect("host serves");
    let socket = host.socket_path().to_path_buf();
    let mut renderer = DoubleRenderer::spawn(&socket, "command", None);
    let probe = connect_to(&socket);
    wait_for("the renderer's command routed", || {
        probe.snapshot().expect("probe snapshot")["jobs"]["limits"]["maxQueued"] == 3
    });
    drop(probe);
    renderer.kill();
    let successor = connect_to(&socket);
    let snapshot = successor.snapshot().expect("snapshot");
    assert_eq!(snapshot["jobs"]["limits"]["maxQueued"], 3, "the killed renderer's command executed");
    assert_eq!(snapshot["jobs"]["limits"]["maxConcurrent"], 1);
    drop(successor);
    host.shutdown();
}

/// A renderer process killed mid-take: the take records on, the next
/// window finds it unowned, adopts it and stops it, and its row is a
/// healthy, complete take in storage.
#[test]
fn a_renderer_killed_mid_take_leaves_the_take_for_the_next_window_to_adopt() {
    let root = tempfile::tempdir().unwrap();
    let mut host = serve(scripted(root.path(), vec![FakeTakeScript::clean()], None)).expect("host serves");
    let socket = host.socket_path().to_path_buf();
    // Ready means: route frozen, take started, audio acknowledged.
    let mut renderer = DoubleRenderer::spawn(&socket, "take", Some("take_kill"));
    renderer.kill();

    let window = watching(&host);
    assert_eq!(window.snapshot().unwrap()["capture"]["state"], "Recording", "the take outlives its renderer");
    let tick = until_take(&window, "the running take", &mut Vec::new(), is_live_tick("take_kill"));
    assert!(matches!(tick, TakeWire::Live { owner: TakeOwner::Nobody, .. }), "{tick:?}");
    window.take_tap("take_kill", 0).unwrap();
    until_take(&window, "the adoption", &mut Vec::new(), |frame| {
        matches!(frame, TakeWire::Live { owner: TakeOwner::You, ended: None, .. })
    });
    stop(&window, "take_kill");
    let id = stored_id_of(&window, "take_kill", &mut Vec::new());
    let store = StoreV2::open(root.path()).expect("store opens");
    assert_eq!(store.list_records(0, 10).unwrap().total, 1, "exactly one take persisted");
    assert_eq!(store.get_capture(&id).unwrap().unwrap().status, CaptureStatus::Complete);
    drop(store);
    drop(window);
    host.shutdown();
}

/// A renderer killed mid-take and never relaunched: the host records on
/// only for the orphan grace, then stops, stores and transcribes the take
/// itself — once — and, with nobody left, goes idle.
#[test]
fn a_take_whose_renderer_never_returns_is_stopped_stored_and_transcribed_by_the_host() {
    let root = tempfile::tempdir().unwrap();
    let engine = FakeEngine::start(vec![Reply::Text("kept by the host".into())], StreamMode::Refuse);
    let config = scripted(root.path(), vec![FakeTakeScript::clean()], Some(&engine))
        .with_orphan_grace(Duration::from_millis(300));
    let mut host = serve(config).expect("host serves");
    let socket = host.socket_path().to_path_buf();
    let mut renderer = DoubleRenderer::spawn(&socket, "take", Some("take_orphan"));
    renderer.kill();

    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        let store = StoreV2::open(root.path()).expect("store opens");
        let page = store.list_records(0, 10).expect("list");
        if let Some(record) = page.records.first() {
            let attempts = store.attempts_for(record.id()).expect("attempts");
            if attempts.iter().any(|attempt| attempt.is_final_transcript()) {
                assert_eq!(page.total, 1, "stored once");
                assert_eq!(attempts.len(), 1, "transcribed once");
                assert_eq!(attempts[0].text, "kept by the host");
                break;
            }
        }
        assert!(Instant::now() < deadline, "the host never stored and transcribed the take");
        std::thread::sleep(Duration::from_millis(50));
    }
    wait_for("the host to go idle", || host.idle());
    host.shutdown();
}

/// A wedged renderer (a snapshot flood it never reads, the host's writer
/// parked mid-reply) is SIGKILLed inside the host's shutdown drain: the
/// shutdown still completes in bounded time with the full teardown —
/// endpoint removed, data root released for a successor.
#[cfg(unix)] // the fill mode shrinks SO_RCVBUF
#[test]
fn shutdown_completes_when_the_renderer_is_killed_mid_drain() {
    let root = tempfile::tempdir().unwrap();
    let mut host = serve(scripted(root.path(), Vec::new(), None)).expect("host serves");
    let socket = host.socket_path().to_path_buf();
    let mut renderer = DoubleRenderer::spawn(&socket, "fill", None);
    // Let the host's reader answer the flood so the writer is parked.
    std::thread::sleep(Duration::from_millis(500));
    let started = Instant::now();
    let drain = std::thread::spawn(move || host.shutdown());
    std::thread::sleep(Duration::from_millis(300));
    renderer.kill();
    drain.join().expect("shutdown thread");
    assert!(started.elapsed() < Duration::from_secs(45), "shutdown within its bound");
    wait_for("the endpoint removed", || !endpoint_present(&socket));
    let successor = serve(scripted(root.path(), Vec::new(), None)).expect("the root was released");
    drop(successor);
}
