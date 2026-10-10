//! #220: the take feed over the real transport — the app's projection of
//! the takes the host records. A watching, tapping app gets the take's
//! health and its every sample, then the end and the stored row; a take
//! nobody follows is stopped, stored and transcribed by the host; an app
//! that reconnects mid-take adopts it; a second window never takes over a
//! take its live owner records.

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
fn a_take_nobody_follows_is_stored_and_transcribed_by_the_host() {
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
    // The host transcribes it itself (this one has no engine, so the take
    // records why it has no transcript), and nothing is left for an app.
    let store = starling_dictation::store_v2::StoreV2::open(root.path()).unwrap();
    let id = store.list_records(0, 1).unwrap().records[0].id().to_string();
    let deadline = Instant::now() + Duration::from_secs(10);
    while store.transcription_wanted(&id).unwrap() {
        assert!(Instant::now() < deadline, "the host never transcribed the orphan");
        std::thread::sleep(Duration::from_millis(50));
    }
    let attempts = store.attempts_for(&id).unwrap();
    assert_eq!(attempts.len(), 1);
    assert_eq!(attempts[0].status, "failed");
    drop(store);
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
    // The second tap goes out once the first one owns the take.
    until_take(&first, "the first tap's ownership", &mut Vec::new(), |frame| {
        matches!(frame, TakeWire::Live { owner: TakeOwner::You, ended: None, .. })
    });
    second.take_tap("take-race", 0).unwrap();
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
    for (corr, command) in [
        (Some("take-mine"), Command::CaptureStop { drain: Some(true) }),
        (Some("take-mine"), Command::CaptureAbort),
        (None, Command::CaptureStop { drain: Some(true) }),
        (None, Command::CaptureAbort),
    ] {
        match other.send(corr, command) {
            Err(starling_runtime_host::client::ClientError::Rejected(rejection)) => {
                assert!(
                    matches!(
                        &rejection,
                        starling_runtime::machine::Rejection::IllegalInState { detail, .. }
                            if detail.contains("another connection")
                    ),
                    "{rejection:?}"
                );
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

/// A refused start says what holds the microphone, typed in its receipt
/// (#220): another connection's take, the connection's own, or nothing
/// the feed knows of.
#[test]
fn a_refused_start_says_what_holds_the_microphone() {
    use starling_runtime_host::client::ClientError;
    use starling_runtime_host::frame::TakeBusy;
    let root = tempfile::tempdir().unwrap();
    let source = FakeCaptureSource::new(vec![FakeTakeScript::clean(), FakeTakeScript::clean()]);
    let mut host = serve(config(root.path(), source)).expect("host serves");
    let owner = connect(&host);
    owner.take_watch().unwrap();
    start(&owner, "take-mine");
    let mut seen = Vec::new();
    until_take(&owner, "a live tick", &mut seen, is_live_tick("take-mine"));
    let other = connect(&host);
    other.take_watch().unwrap();
    let again = || Command::CaptureStart {
        policy: "push-to-talk".into(),
    };
    for (client, take, yours) in [(&other, "take-theirs", false), (&owner, "take-again", true)] {
        match client.send_reporting_busy(Some(take), again()) {
            Err((ClientError::Rejected(_), busy)) => {
                assert_eq!(busy, Some(TakeBusy::Recording { yours }), "{take}")
            }
            other => panic!("expected a refusal, got {other:?}"),
        }
    }
    // A refusal for any other reason names nothing: the plain send path
    // is unchanged.
    match other.send(Some("take-theirs"), again()) {
        Err(ClientError::Rejected(rejection)) => assert!(
            matches!(
                rejection,
                starling_runtime::machine::Rejection::IllegalInState { .. }
            ),
            "{rejection:?}"
        ),
        other => panic!("expected a refusal, got {other:?}"),
    }
    stop(&owner, "take-mine");
    until_take(&owner, "the stored row", &mut seen, is_persisted("take-mine"));
    // Free again: the other window's start is taken.
    other
        .send_reporting_busy(Some("take-theirs"), again())
        .expect("a start once the microphone is free");
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
fn refused_starts_during_a_slow_device_open_never_take_the_owner_away() {
    let root = tempfile::tempdir().unwrap();
    let slow = FakeTakeScript {
        open_delay: Duration::from_millis(800),
        ..FakeTakeScript::clean()
    };
    let source = FakeCaptureSource::new(vec![slow]);
    let mut host = serve(config(root.path(), Arc::clone(&source))).expect("host serves");
    let owner = Arc::new(connect(&host));
    owner.take_watch().unwrap();
    let opening = std::thread::spawn({
        let owner = Arc::clone(&owner);
        move || start(&owner, "take-slow")
    });
    // While the device opens, other windows press record: each start is
    // refused, and there are more of them than the hub remembers.
    let deadline = Instant::now() + Duration::from_secs(5);
    while source.started_takes.lock().unwrap().is_empty() {
        assert!(Instant::now() < deadline, "the take never started opening its device");
        std::thread::sleep(Duration::from_millis(5));
    }
    let others: Vec<HostClient> = (0..6).map(|_| connect(&host)).collect();
    std::thread::scope(|scope| {
        for (index, other) in others.iter().enumerate() {
            scope.spawn(move || {
                let _ = other.send(
                    Some(&format!("take-burst-{index}")),
                    Command::CaptureStart { policy: "push-to-talk".into() },
                );
            });
        }
    });
    opening.join().unwrap();
    let mut seen = Vec::new();
    let tick = until_take(&owner, "the slow take's tick", &mut seen, is_live_tick("take-slow"));
    assert!(matches!(tick, TakeWire::Live { owner: TakeOwner::You, .. }), "{tick:?}");
    // Another window can neither adopt it nor end it.
    let other = &others[0];
    other.take_watch().unwrap();
    other.take_tap("take-slow", 0).unwrap();
    until_take(other, "another window's view", &mut Vec::new(), |frame| {
        matches!(frame, TakeWire::Live { take, owner: TakeOwner::Another, ended: None, .. } if take == "take-slow")
    });
    assert!(other
        .send(Some("take-slow"), Command::CaptureStop { drain: Some(true) })
        .is_err());
    stop(&owner, "take-slow");
    let persisted = until_take(&owner, "the stored row", &mut seen, is_persisted("take-slow"));
    assert!(matches!(persisted, TakeWire::Persisted { orphan: false, .. }), "{persisted:?}");
    drop(others);
    drop(owner);
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

#[test]
fn starting_the_next_take_does_not_cut_off_the_last_ones_tail() {
    let root = tempfile::tempdir().unwrap();
    // A first take long enough that its replay takes several ticks.
    let long = FakeTakeScript {
        samples_per_second: 2_000_000,
        sample_cap: 1_500_000,
        ..FakeTakeScript::clean()
    };
    let source = FakeCaptureSource::new(vec![long, FakeTakeScript::clean()]);
    let mut host = serve(config(root.path(), source)).expect("host serves");
    let app = connect(&host);
    app.take_watch().unwrap();
    start(&app, "take-one");
    let mut seen = Vec::new();
    until_take(&app, "the first take", &mut seen, is_live_tick("take-one"));
    std::thread::sleep(Duration::from_millis(900));
    stop(&app, "take-one");
    let deadline = Instant::now() + Duration::from_secs(20);
    while host_capture_state(&app) != "Persisted" {
        assert!(Instant::now() < deadline, "the first take was never stored");
        std::thread::sleep(Duration::from_millis(20));
    }
    // What the feed said before tapping (the end every watcher hears).
    std::thread::sleep(Duration::from_millis(100));
    while app.try_recv_take().is_ok() {}
    // Replay the first take, and start and tap the second one at once.
    app.take_tap("take-one", 0).unwrap();
    start(&app, "take-two");
    app.take_tap("take-two", 0).unwrap();
    let mut seen = Vec::new();
    let end = until_take(&app, "the first take's end", &mut seen, is_end("take-one"));
    let TakeWire::Live { ended: Some(total), .. } = end else { unreachable!() };
    assert!(total > 200_000, "a long take: {total}");
    assert_eq!(audio_of(&seen, "take-one").len() as u64, total, "every sample, then the end");
    assert!(
        seen.iter().any(|frame| matches!(frame, TakeWire::Live { take, audio: Some(_), .. } if take == "take-two")),
        "the second take's audio flowed meanwhile"
    );
    stop(&app, "take-two");
    until_take(&app, "the second take's row", &mut seen, is_persisted("take-two"));
    drop(app);
    host.shutdown();
}

#[test]
fn a_take_a_command_only_client_records_is_the_watching_apps_to_act_on() {
    let root = tempfile::tempdir().unwrap();
    let source = FakeCaptureSource::new(vec![FakeTakeScript::clean()]);
    let mut host = serve(config(root.path(), source)).expect("host serves");
    let app = connect(&host);
    app.take_watch().unwrap();
    let tool = connect(&host);
    start(&tool, "take-tool");
    let mut seen = Vec::new();
    until_take(&app, "the tool's take", &mut seen, is_live_tick("take-tool"));
    stop(&tool, "take-tool");
    let heard = until_take(&app, "the stored row", &mut seen, is_persisted("take-tool"));
    assert!(matches!(heard, TakeWire::Persisted { orphan: false, .. }), "{heard:?}");
    // The tool follows no feed: the watching app is the one to show the
    // result.
    let result = until_take(&app, "the transcription's end", &mut seen, |frame| {
        matches!(frame, TakeWire::Transcription { take: Some(take), state, .. } if take == "take-tool" && state.is_final())
    });
    assert!(matches!(result, TakeWire::Transcription { yours: true, .. }), "{result:?}");
    drop(tool);
    drop(app);
    host.shutdown();
}

#[test]
fn a_corr_a_finished_take_used_is_free_for_the_next_start() {
    let root = tempfile::tempdir().unwrap();
    let source = FakeCaptureSource::new(vec![FakeTakeScript::clean(), FakeTakeScript::clean()]);
    let mut host = serve(config(root.path(), source)).expect("host serves");
    let first = connect(&host);
    first.take_watch().unwrap();
    let anon = starling_runtime::machine::capture::ANON_TAKE;
    first
        .send(None, Command::CaptureStart { policy: "push-to-talk".into() })
        .expect("start accepted");
    let mut seen = Vec::new();
    until_take(&first, "the first take", &mut seen, is_live_tick(anon));
    first
        .send(None, Command::CaptureStop { drain: Some(true) })
        .expect("stop accepted");
    until_take(&first, "the first stored row", &mut seen, is_persisted(anon));
    let second = connect(&host);
    second.take_watch().unwrap();
    second
        .send(None, Command::CaptureStart { policy: "push-to-talk".into() })
        .expect("start accepted");
    let tick = until_take(&second, "the second take", &mut Vec::new(), is_live_tick(anon));
    assert!(matches!(tick, TakeWire::Live { owner: TakeOwner::You, .. }), "{tick:?}");
    second
        .send(None, Command::CaptureStop { drain: Some(true) })
        .expect("its starter stops it");
    drop(first);
    drop(second);
    host.shutdown();
}
