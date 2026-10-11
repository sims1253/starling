//! Shared helpers for the host's integration suites — each test binary
//! pulls this in with `mod common;`, so the suites cannot drift apart
//! on the helpers they share: a host on a scratch root recording
//! scripted takes ([`host_config`]), clients that follow its take feed,
//! and the scripted transcription server ([`fake_engine`]).
#![allow(dead_code)]

pub mod fake_engine;
pub mod reaped;

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use starling_dictation::store_v2::{CommitMark, StoreV2, TakeMeta};
use starling_runtime::machine::capture::{CaptureConfig, CaptureSource, V2CaptureStore};
use starling_runtime::protocol::Command;
use starling_runtime::testing::{FakeCaptureSource, FakeTakeScript};
use starling_runtime_host::client::{HostClient, TakeWire};
use starling_runtime_host::engine::EngineChoice;
use starling_runtime_host::frame::TranscriptionState;
use starling_runtime_host::{HostConfig, HostHandle};

use fake_engine::FakeEngine;

/// Whether the endpoint still exists: the socket file on unix; on
/// Windows (no filesystem entry — `\\.\pipe\…` lives in the kernel
/// namespace) whether a probe still finds a server bound to the name.
pub fn endpoint_present(socket: &std::path::Path) -> bool {
    #[cfg(unix)]
    {
        socket.exists()
    }
    #[cfg(windows)]
    {
        starling_runtime_host::platform::probe(socket)
            != starling_runtime_host::platform::Probe::Dead
    }
}

/// A host on `root` (its store, its journal tree) recording from `source`
/// and, with `engine`, transcribing on that scripted server as the
/// user's own server.
pub fn host_config(root: &Path, source: Arc<dyn CaptureSource>, engine: Option<&FakeEngine>) -> HostConfig {
    let mut config = HostConfig::new(root, root.join("endpoints"));
    if let Some(engine) = engine {
        config = config.with_engine(EngineChoice::Manual {
            endpoint: engine.endpoint(),
            model: "fake-model".to_string(),
        });
    }
    config.runtime = config
        .runtime
        .with_capture_source(source)
        .with_capture_store(Arc::new(V2CaptureStore::open(root).expect("v2 store opens")))
        .with_capture_config(CaptureConfig {
            journals_dir: root.join("journals"),
            poll_interval: Duration::from_millis(10),
            ..CaptureConfig::default()
        });
    config
}

/// [`host_config`] recording `scripts` in order.
pub fn scripted(root: &Path, scripts: Vec<FakeTakeScript>, engine: Option<&FakeEngine>) -> HostConfig {
    host_config(root, FakeCaptureSource::new(scripts), engine)
}

pub fn connect_to(socket: &Path) -> HostClient {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        match HostClient::connect(socket) {
            Ok(client) => return client,
            Err(err) => {
                assert!(Instant::now() < deadline, "no host at {}: {err}", socket.display());
                std::thread::sleep(Duration::from_millis(20));
            }
        }
    }
}

pub fn connect(host: &HostHandle) -> HostClient {
    connect_to(host.socket_path())
}

/// A client following the take feed, as every app window does.
pub fn watching(host: &HostHandle) -> HostClient {
    let client = connect(host);
    client.take_watch().expect("watching");
    client
}

/// The next take frame matching `predicate`, collecting every frame seen.
pub fn until_take(
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
                    start.elapsed() < Duration::from_secs(20),
                    "timed out waiting for {label}; saw {seen:#?}"
                );
            }
            Err(other) => panic!("take feed error: {other:?} ({})", client.close_reason()),
        }
    }
}

/// Waits (bounded) until `done` holds.
pub fn wait_for(what: &str, done: impl Fn() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(20);
    while !done() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(20));
    }
}

pub fn start(client: &HostClient, take: &str) {
    client
        .send(Some(take), Command::CaptureStart { policy: "push-to-talk".into() })
        .expect("start accepted");
}

pub fn stop(client: &HostClient, take: &str) {
    client
        .send(Some(take), Command::CaptureStop { drain: Some(true) })
        .expect("stop accepted");
}

/// Records `take` for a moment past its first status tick, then stops it.
pub fn record(client: &HostClient, take: &str) {
    start(client, take);
    until_take(client, "a status tick", &mut Vec::new(), is_live_tick(take));
    std::thread::sleep(Duration::from_millis(200));
    stop(client, take);
}

pub fn is_live_tick(take: &str) -> impl Fn(&TakeWire) -> bool + '_ {
    move |frame| matches!(frame, TakeWire::Live { take: name, ended: None, status: Some(_), .. } if name == take)
}

pub fn is_end(take: &str) -> impl Fn(&TakeWire) -> bool + '_ {
    move |frame| matches!(frame, TakeWire::Live { take: name, ended: Some(_), .. } if name == take)
}

/// The stored-row notice; for a tapping app it always follows the end.
pub fn is_persisted(take: &str) -> impl Fn(&TakeWire) -> bool + '_ {
    move |frame| matches!(frame, TakeWire::Persisted { take: name, .. } if name == take)
}

/// The stored id of `take`, from its stored-row notice.
pub fn stored_id_of(client: &HostClient, take: &str, seen: &mut Vec<TakeWire>) -> String {
    match until_take(client, "the stored row", seen, is_persisted(take)) {
        TakeWire::Persisted {
            stored_id: Some(id),
            ..
        } => id,
        other => panic!("not stored: {other:?}"),
    }
}

/// The final transcription frame for `stored_id`: its state, and whether
/// the receiving window is the one to act on it.
pub fn finished(client: &HostClient, stored_id: &str, seen: &mut Vec<TakeWire>) -> (TranscriptionState, bool) {
    let frame = until_take(client, "the transcription's end", seen, |frame| {
        matches!(frame, TakeWire::Transcription { stored_id: id, req: None, state, .. } if id == stored_id && state.is_final())
    });
    let TakeWire::Transcription { state, yours, .. } = frame else {
        unreachable!()
    };
    (state, yours)
}

/// The final frame of the retry `req` asked for.
pub fn retried(client: &HostClient, req: &str, seen: &mut Vec<TakeWire>) -> (TranscriptionState, bool) {
    let frame = until_take(client, "the retry's end", seen, |frame| {
        matches!(frame, TakeWire::Transcription { req: Some(r), state, .. } if r == req && state.is_final())
    });
    let TakeWire::Transcription { state, yours, .. } = frame else {
        unreachable!()
    };
    (state, yours)
}

pub fn completed(text: &str) -> TranscriptionState {
    TranscriptionState::Completed {
        text: text.into(),
        kept_earlier: false,
    }
}

/// The texts of `id`'s finished transcripts, oldest first.
pub fn completed_attempts(root: &Path, id: &str) -> Vec<String> {
    StoreV2::open(root)
        .expect("store")
        .attempts_for(id)
        .expect("attempts")
        .into_iter()
        .filter(|attempt| attempt.is_final_transcript())
        .map(|attempt| attempt.text)
        .collect()
}

/// A take stored beside the host the way a host stores it, one second
/// long, with the intent to transcribe it when `transcribe`.
pub fn stored_take(root: &Path, transcribe: bool, level: f32) -> String {
    let mut store = StoreV2::open(root).unwrap();
    let mut meta = TakeMeta::for_device("test-device");
    meta.transcribe = transcribe;
    let mut take = store.begin_take(meta).unwrap();
    take.append_and_seal(&vec![level; 16_000]).unwrap();
    take.finalize()
        .unwrap()
        .commit_marked(&mut store, CommitMark::Complete)
        .unwrap()
        .record
        .id
}
