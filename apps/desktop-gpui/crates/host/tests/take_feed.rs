//! #220: the take feed over the real transport — the app's projection of
//! the takes the host records. A watching, tapping app gets the take's
//! health and its every sample, then the end and the stored row; a take
//! nobody follows is stopped and stored by the host and handed to the
//! next app that watches; an app that reconnects mid-take adopts it; a
//! second window never takes over a take its live owner records.

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use starling_runtime::machine::capture::{CaptureConfig, V2CaptureStore};
use starling_runtime::protocol::Command;
use starling_runtime::testing::{FakeCaptureSource, FakeTakeScript};
use starling_runtime_host::client::{HostClient, TakeWire};
use starling_runtime_host::frame::TakeOwner;
use starling_runtime_host::{serve, HostConfig, HostHandle};

fn config(root: &Path, source: Arc<FakeCaptureSource>) -> HostConfig {
    let mut config = HostConfig::new(root, root.join("endpoints"));
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

/// The next take frame matching `predicate`, collecting every frame seen.
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
                    start.elapsed() < Duration::from_secs(10),
                    "timed out waiting for {label}"
                );
            }
            Err(other) => panic!("take feed error: {other:?} ({})", client.close_reason()),
        }
    }
}

fn audio_of(frames: &[TakeWire], take: &str) -> Vec<f32> {
    let mut samples = Vec::new();
    for frame in frames {
        if let TakeWire::Live {
            take: name,
            audio: Some((start, chunk)),
            ..
        } = frame
        {
            if name == take {
                assert_eq!(*start as usize, samples.len(), "audio arrives in order, once");
                samples.extend_from_slice(chunk);
            }
        }
    }
    samples
}

fn start(client: &HostClient, take: &str) {
    client
        .send(Some(take), Command::CaptureStart { policy: "push-to-talk".into() })
        .expect("start accepted");
}

fn stop(client: &HostClient, take: &str) {
    client
        .send(Some(take), Command::CaptureStop { drain: Some(true) })
        .expect("stop accepted");
}

fn is_live_tick(take: &str) -> impl Fn(&TakeWire) -> bool + '_ {
    move |frame| matches!(frame, TakeWire::Live { take: name, ended: None, status: Some(_), .. } if name == take)
}

fn is_end(take: &str) -> impl Fn(&TakeWire) -> bool + '_ {
    move |frame| matches!(frame, TakeWire::Live { take: name, ended: Some(_), .. } if name == take)
}

/// The stored-row notice; for a tapping app it always follows the end.
fn is_persisted(take: &str) -> impl Fn(&TakeWire) -> bool + '_ {
    move |frame| matches!(frame, TakeWire::Persisted { take: name, .. } if name == take)
}

#[test]
fn a_tapping_app_gets_every_sample_then_the_end_then_the_stored_row() {
    let root = tempfile::tempdir().unwrap();
    let source = FakeCaptureSource::new(vec![FakeTakeScript::clean()]);
    let mut host = serve(config(root.path(), source)).expect("host serves");
    let app = connect(&host);
    assert_eq!(app.take_watch().expect("watching"), None, "nothing recovered");

    start(&app, "take-a");
    app.take_tap("take-a", 0).unwrap();
    let mut seen = Vec::new();
    let tick = until_take(&app, "a live tick", &mut seen, is_live_tick("take-a"));
    let TakeWire::Live { rate, owner, .. } = tick else { unreachable!() };
    assert_eq!(rate, 16_000);
    assert_eq!(owner, TakeOwner::You, "the starting connection owns the take");
    std::thread::sleep(Duration::from_millis(200));
    stop(&app, "take-a");
    let end = until_take(&app, "the end", &mut seen, is_end("take-a"));
    let TakeWire::Live { ended: Some(total), .. } = end else { unreachable!() };
    let tapped = audio_of(&seen, "take-a");
    assert_eq!(tapped.len() as u64, total, "the end comes after the last sample");
    let persisted = until_take(&app, "the stored row", &mut seen, is_persisted("take-a"));
    let TakeWire::Persisted { stored_id: Some(id), interrupted, error, orphan, .. } = persisted
    else {
        panic!("expected a stored row: {persisted:?}");
    };
    assert!(!interrupted && error.is_none() && !orphan);
    let store = starling_dictation::store_v2::StoreV2::open(root.path()).unwrap();
    let stored = store.load_audio(&id).expect("the stored take's audio");
    assert_eq!(stored.samples, tapped, "the app streamed exactly the stored take");
    drop(app);
    host.shutdown();
}

#[test]
fn a_take_nobody_follows_is_stored_by_the_host_and_handed_to_the_next_app() {
    let root = tempfile::tempdir().unwrap();
    let source = FakeCaptureSource::new(vec![FakeTakeScript::clean()]);
    let config = config(root.path(), source).with_orphan_grace(Duration::from_millis(300));
    let mut host = serve(config).expect("host serves");
    let app = connect(&host);
    app.take_watch().unwrap();
    start(&app, "take-o");
    let mut seen = Vec::new();
    until_take(&app, "a live tick", &mut seen, is_live_tick("take-o"));
    // The app goes away mid-take (the renderer-kill suite does this with
    // a real process kill).
    drop(app);
    let deadline = Instant::now() + Duration::from_secs(10);
    while !host_settled(root.path()) {
        assert!(Instant::now() < deadline, "the host never stored the orphaned take");
        std::thread::sleep(Duration::from_millis(50));
    }
    let next = connect(&host);
    next.take_watch().unwrap();
    let mut seen = Vec::new();
    let persisted = until_take(&next, "the orphan", &mut seen, is_persisted("take-o"));
    let TakeWire::Persisted { stored_id: Some(_), orphan: true, interrupted: false, .. } = persisted
    else {
        panic!("expected the orphan, stored complete: {persisted:?}");
    };
    // It goes to one app only.
    let other = connect(&host);
    other.take_watch().unwrap();
    std::thread::sleep(Duration::from_millis(200));
    assert!(
        !std::iter::from_fn(|| other.try_recv_take().ok())
            .any(|frame| matches!(frame, TakeWire::Persisted { .. })),
        "a second app does not get the same orphan"
    );
    drop(next);
    drop(other);
    let deadline = Instant::now() + Duration::from_secs(5);
    while !host.idle() {
        assert!(Instant::now() < deadline, "the host never went idle");
        std::thread::sleep(Duration::from_millis(20));
    }
    host.shutdown();
}

fn host_settled(root: &Path) -> bool {
    starling_dictation::store_v2::StoreV2::open(root)
        .and_then(|store| store.list_records(0, 10))
        .is_ok_and(|page| page.total == 1)
}

#[test]
fn an_app_that_reconnects_mid_take_adopts_it_and_replays_it_from_the_start() {
    let root = tempfile::tempdir().unwrap();
    let source = FakeCaptureSource::new(vec![FakeTakeScript::clean()]);
    let mut host = serve(config(root.path(), source)).expect("host serves");
    let first = connect(&host);
    first.take_watch().unwrap();
    start(&first, "take-r");
    let mut seen = Vec::new();
    until_take(&first, "a live tick", &mut seen, is_live_tick("take-r"));
    std::thread::sleep(Duration::from_millis(150));
    drop(first);

    let relaunched = connect(&host);
    relaunched.take_watch().unwrap();
    let mut seen = Vec::new();
    let tick = until_take(&relaunched, "the running take", &mut seen, is_live_tick("take-r"));
    assert!(
        matches!(tick, TakeWire::Live { owner: TakeOwner::Nobody, .. }),
        "its owner is gone: {tick:?}"
    );
    relaunched.take_tap("take-r", 0).unwrap();
    let owned = until_take(&relaunched, "owned after tapping", &mut seen, |frame| {
        matches!(frame, TakeWire::Live { take, owner: TakeOwner::You, ended: None, .. } if take == "take-r")
    });
    assert!(matches!(owned, TakeWire::Live { .. }));
    stop(&relaunched, "take-r");
    let mut seen = Vec::new();
    until_take(&relaunched, "the end", &mut seen, is_end("take-r"));
    let persisted = until_take(&relaunched, "the stored row", &mut seen, is_persisted("take-r"));
    assert!(
        matches!(persisted, TakeWire::Persisted { orphan: false, stored_id: Some(_), .. }),
        "the adopting app owns the stored take: {persisted:?}"
    );
    drop(relaunched);
    host.shutdown();
}

#[test]
fn a_second_window_sees_a_take_its_live_owner_records_as_owned() {
    let root = tempfile::tempdir().unwrap();
    let source = FakeCaptureSource::new(vec![FakeTakeScript::clean()]);
    let mut host = serve(config(root.path(), source)).expect("host serves");
    let owner = connect(&host);
    owner.take_watch().unwrap();
    let second = connect(&host);
    second.take_watch().unwrap();
    start(&owner, "take-s");
    let mut seen = Vec::new();
    let tick = until_take(&second, "the owner's take", &mut seen, is_live_tick("take-s"));
    assert!(matches!(tick, TakeWire::Live { owner: TakeOwner::Another, .. }), "{tick:?}");
    // The second window tapping (to show levels) does not take it over:
    // the owner still stops it and gets a non-orphan stored row.
    second.take_tap("take-s", 0).unwrap();
    stop(&owner, "take-s");
    let mut seen = Vec::new();
    let persisted = until_take(&owner, "the stored row", &mut seen, is_persisted("take-s"));
    assert!(matches!(persisted, TakeWire::Persisted { orphan: false, .. }), "{persisted:?}");
    drop(owner);
    drop(second);
    host.shutdown();
}

#[test]
fn a_microphone_that_will_not_open_is_reported_to_watchers() {
    let root = tempfile::tempdir().unwrap();
    let source = FakeCaptureSource::new(vec![]);
    let mut host = serve(config(root.path(), source)).expect("host serves");
    let app = connect(&host);
    app.take_watch().unwrap();
    start(&app, "take-f");
    let mut seen = Vec::new();
    let failed = until_take(&app, "the start failure", &mut seen, |frame| {
        matches!(frame, TakeWire::StartFailed { take, .. } if take == "take-f")
    });
    let TakeWire::StartFailed { message, problem, .. } = failed else { unreachable!() };
    assert!(message.contains("No microphone"), "{message}");
    assert_eq!(problem, None, "the fake source does not classify its failure");
    drop(app);
    host.shutdown();
}

#[test]
fn a_host_with_a_client_is_not_idle() {
    let root = tempfile::tempdir().unwrap();
    let source = FakeCaptureSource::new(vec![]);
    let mut host = serve(config(root.path(), source)).expect("host serves");
    assert!(host.idle());
    let app = connect(&host);
    assert!(!host.idle(), "a connected client keeps the host");
    drop(app);
    let deadline = Instant::now() + Duration::from_secs(5);
    while !host.idle() {
        assert!(Instant::now() < deadline, "the host never went idle");
        std::thread::sleep(Duration::from_millis(20));
    }
    host.shutdown();
}

#[test]
fn of_two_windows_adopting_an_unowned_take_only_the_first_tap_gets_it() {
    let root = tempfile::tempdir().unwrap();
    let source = FakeCaptureSource::new(vec![FakeTakeScript::clean()]);
    let mut host = serve(config(root.path(), source)).expect("host serves");
    {
        let gone = connect(&host);
        start(&gone, "take-race");
        let mut seen = Vec::new();
        gone.take_watch().unwrap();
        until_take(&gone, "a live tick", &mut seen, is_live_tick("take-race"));
    }
    let first = connect(&host);
    first.take_watch().unwrap();
    let second = connect(&host);
    second.take_watch().unwrap();
    let mut seen = Vec::new();
    let tick = until_take(&second, "the unowned take", &mut seen, is_live_tick("take-race"));
    assert!(matches!(tick, TakeWire::Live { owner: TakeOwner::Nobody, .. }), "{tick:?}");
    first.take_tap("take-race", 0).unwrap();
    std::thread::sleep(Duration::from_millis(100));
    second.take_tap("take-race", 0).unwrap();
    let mine = until_take(&first, "the first tap's ownership", &mut Vec::new(), |frame| {
        matches!(frame, TakeWire::Live { owner: TakeOwner::You, ended: None, .. })
    });
    assert!(matches!(mine, TakeWire::Live { .. }));
    until_take(&second, "the second tap sees another owner", &mut Vec::new(), |frame| {
        matches!(frame, TakeWire::Live { owner: TakeOwner::Another, ended: None, .. })
    });
    drop(first);
    drop(second);
    host.shutdown();
}

#[test]
fn a_second_window_cannot_end_a_take_its_live_owner_records() {
    let root = tempfile::tempdir().unwrap();
    let source = FakeCaptureSource::new(vec![FakeTakeScript::clean()]);
    let mut host = serve(config(root.path(), source)).expect("host serves");
    let owner = connect(&host);
    owner.take_watch().unwrap();
    start(&owner, "take-mine");
    let mut seen = Vec::new();
    until_take(&owner, "a live tick", &mut seen, is_live_tick("take-mine"));
    let other = connect(&host);
    other.take_watch().unwrap();
    for command in [Command::CaptureStop { drain: Some(true) }, Command::CaptureAbort] {
        match other.send(Some("take-mine"), command) {
            Err(starling_runtime_host::client::ClientError::Rejected(rejection)) => {
                assert!(format!("{rejection:?}").contains("another connection"), "{rejection:?}");
            }
            other => panic!("expected a refusal, got {other:?}"),
        }
    }
    assert_eq!(host_capture_state(&owner), "Recording", "the take records on");
    stop(&owner, "take-mine");
    until_take(&owner, "the stored row", &mut seen, is_persisted("take-mine"));
    drop(owner);
    drop(other);
    host.shutdown();
}

fn host_capture_state(client: &HostClient) -> String {
    client.snapshot().unwrap()["capture"]["state"]
        .as_str()
        .unwrap_or_default()
        .to_string()
}

#[test]
fn an_orphan_no_app_claimed_survives_the_host_exiting() {
    let root = tempfile::tempdir().unwrap();
    let source = FakeCaptureSource::new(vec![FakeTakeScript::clean()]);
    let config_a = config(root.path(), source).with_orphan_grace(Duration::from_millis(200));
    let mut host = serve(config_a).expect("host serves");
    {
        let gone = connect(&host);
        start(&gone, "take-later");
        std::thread::sleep(Duration::from_millis(100));
    }
    let deadline = Instant::now() + Duration::from_secs(10);
    while !host_settled(root.path()) || !host.idle() {
        assert!(Instant::now() < deadline, "the host never stored the orphan");
        std::thread::sleep(Duration::from_millis(50));
    }
    // The host goes away (idle exit) before any app returns.
    host.shutdown();
    drop(host);
    let mut host = serve(config(root.path(), FakeCaptureSource::new(vec![]))).expect("serves again");
    let app = connect(&host);
    app.take_watch().unwrap();
    let mut seen = Vec::new();
    let persisted = until_take(&app, "the remembered orphan", &mut seen, |frame| {
        matches!(frame, TakeWire::Persisted { orphan: true, .. })
    });
    let TakeWire::Persisted { stored_id: Some(id), .. } = persisted else {
        panic!("{persisted:?}");
    };
    let store = starling_dictation::store_v2::StoreV2::open(root.path()).unwrap();
    assert!(store.get_capture(&id).unwrap().is_some());
    assert!(
        !root.path().join(starling_runtime_host::takes::UNCLAIMED_FILE).exists(),
        "handed over, so no longer remembered"
    );
    drop(app);
    host.shutdown();
}

#[test]
fn an_app_watching_fails_the_attempts_a_dead_app_left_started() {
    let root = tempfile::tempdir().unwrap();
    let id = {
        let mut store = starling_dictation::store_v2::StoreV2::open(root.path()).unwrap();
        let wav = starling_dictation::audio::encode_wav_16k(&starling_dictation::audio::PcmAudio {
            samples: vec![0.1; 1600],
            sample_rate: 16_000,
            channels: 1,
        })
        .unwrap();
        store
            .save_wav_capture(&wav, starling_dictation::store_v2::TakeMeta::for_device("t"))
            .unwrap()
            .record
            .id
    };
    let mut host = serve(config(root.path(), FakeCaptureSource::new(vec![]))).expect("serves");
    // An app process transcribing the take dies with its attempt started.
    {
        let mut dead_app = starling_dictation::store_v2::StoreV2::open(root.path()).unwrap();
        dead_app.begin_recognition(&id, "engine:test", None).unwrap();
    }
    let app = connect(&host);
    app.take_watch().unwrap();
    let store = starling_dictation::store_v2::StoreV2::open(root.path()).unwrap();
    let attempts = store.attempts_for(&id).unwrap();
    assert_ne!(attempts.last().unwrap().status, "started", "{attempts:?}");
    drop(app);
    host.shutdown();
}
