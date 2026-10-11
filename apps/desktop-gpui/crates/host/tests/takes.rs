//! Takes, end to end over the real transport (#220): an app records a
//! take through the host, follows it on the take feed — its health, its
//! every sample, its live text — and the host stores and transcribes it
//! against a scripted transcription server, handing the result to the
//! window that dictated it. The scenarios script what goes wrong on the
//! way: a live stream that drops or never finishes, an engine that
//! errors, hangs up or answers nothing, retries on the user's server or
//! another model, and windows that come, go and compete for a take.

mod common;

use std::sync::Arc;
use std::time::Duration;

use common::fake_engine::{FakeEngine, Reply, StreamMode};
use common::*;
use starling_dictation::storage::ListedRecord;
use starling_dictation::store_v2::StoreV2;
use starling_runtime::protocol::Command;
use starling_runtime::testing::{FakeCaptureSource, FakeTakeScript};
use starling_runtime_host::client::{ClientError, HostClient, TakeWire};
use starling_runtime_host::frame::{TakeBusy, TakeOwner, TranscribeWith, TranscriptionState};
use starling_runtime_host::history::HistoryClient;
use starling_runtime_host::serve;

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

fn echo() -> StreamMode {
    StreamMode::Echo {
        partial: "hello".into(),
        final_text: "hello there".into(),
    }
}

fn host_capture_state(client: &HostClient) -> String {
    client.snapshot().unwrap()["capture"]["state"]
        .as_str()
        .unwrap_or_default()
        .to_string()
}

/// The happy take: live ticks with the owner's level meter and the
/// stream's live text while it records, every sample to the tapping
/// owner, then the end, the stored row, and the stream's final as the
/// transcript — for the owner to act on, for another window to show.
#[test]
fn a_dictated_take_streams_live_text_then_is_stored_transcribed_and_handed_to_its_window() {
    let root = tempfile::tempdir().unwrap();
    let engine = FakeEngine::start(Vec::new(), echo());
    let mut host = serve(scripted(root.path(), vec![FakeTakeScript::clean()], Some(&engine))).expect("serves");
    let owner = connect(&host);
    assert_eq!(owner.take_watch().expect("watching"), None, "nothing recovered");
    let other = watching(&host);

    start(&owner, "take-a");
    let mut seen = Vec::new();
    let preview = until_take(&owner, "live text", &mut seen, |frame| {
        matches!(frame, TakeWire::LiveText { take, partial: Some(_), .. } if take == "take-a")
    });
    let TakeWire::LiveText { partial: Some(partial), .. } = preview else { unreachable!() };
    assert_eq!(partial.text, "hello");
    // The owner's ticks carry its level meter (until it taps the audio).
    let tick = until_take(&owner, "a level meter", &mut seen, |frame| {
        matches!(frame, TakeWire::Live { meter: Some(meter), .. } if !meter.is_empty())
    });
    let TakeWire::Live { rate, owner: whose, .. } = tick else { unreachable!() };
    assert_eq!(rate, 16_000);
    assert_eq!(whose, TakeOwner::You, "the starting connection owns the take");
    owner.take_tap("take-a", 0).unwrap();
    let theirs = until_take(&other, "the owner's take", &mut Vec::new(), is_live_tick("take-a"));
    assert!(matches!(theirs, TakeWire::Live { owner: TakeOwner::Another, .. }), "{theirs:?}");

    stop(&owner, "take-a");
    let end = until_take(&owner, "the end", &mut seen, is_end("take-a"));
    let TakeWire::Live { ended: Some(total), .. } = end else { unreachable!() };
    let tapped = audio_of(&seen, "take-a");
    assert_eq!(tapped.len() as u64, total, "the end comes after the last sample");
    let persisted = until_take(&owner, "the stored row", &mut seen, is_persisted("take-a"));
    let TakeWire::Persisted { stored_id: Some(id), interrupted, error, orphan, .. } = persisted else {
        panic!("expected a stored row: {persisted:?}");
    };
    assert!(!interrupted && error.is_none() && !orphan);
    let (state, yours) = finished(&owner, &id, &mut seen);
    assert_eq!(state, completed("hello there"));
    assert!(yours, "the owner acts on its take's result");
    let (state, yours) = finished(&other, &id, &mut Vec::new());
    assert_eq!(state, completed("hello there"));
    assert!(!yours, "another window only shows it");

    let store = StoreV2::open(root.path()).unwrap();
    assert_eq!(store.load_audio(&id).unwrap().samples, tapped, "the app streamed exactly the stored take");
    assert!(!store.transcription_wanted(&id).unwrap());
    drop(store);
    assert_eq!(completed_attempts(root.path(), &id), vec!["hello there"]);
    assert_eq!(engine.session_samples(), vec![tapped.len()], "the stream heard every sample, once");
    assert_eq!(engine.batch_requests(), 0, "the stream's final, no upload");
    host.shutdown();
}

/// A live stream that drops mid-take is reopened, and the new session
/// hears the take from its first sample: its final is the transcript.
#[test]
fn a_stream_dropped_mid_take_is_reopened_and_hears_the_whole_take() {
    let root = tempfile::tempdir().unwrap();
    let engine = FakeEngine::start(
        Vec::new(),
        StreamMode::Drop {
            failing: 1,
            after_frames: 3,
            partial: "hello".into(),
            final_text: "hello again".into(),
        },
    );
    let mut host = serve(scripted(root.path(), vec![FakeTakeScript::clean()], Some(&engine))).expect("serves");
    let app = watching(&host);
    start(&app, "take-d");
    wait_for("the reopened stream", || engine.stream_sessions() == 2);
    std::thread::sleep(Duration::from_millis(300));
    stop(&app, "take-d");
    let mut seen = Vec::new();
    let id = stored_id_of(&app, "take-d", &mut seen);
    let (state, yours) = finished(&app, &id, &mut seen);
    assert_eq!(state, completed("hello again"));
    assert!(yours);
    let stored = StoreV2::open(root.path()).unwrap().load_audio(&id).unwrap().samples.len();
    let sessions = engine.session_samples();
    assert_eq!(sessions.len(), 2, "{sessions:?}");
    assert_eq!(sessions[1], stored, "the new session heard the take from its start");
    assert_eq!(engine.batch_requests(), 0, "the reopened stream finished it");
    host.shutdown();
}

/// A stream that keeps dropping gives up after its bounded reconnects,
/// says so in the live text, and the stored take is uploaded instead —
/// once, after it was stored.
#[test]
fn a_stream_that_keeps_failing_falls_back_to_uploading_the_stored_take() {
    let root = tempfile::tempdir().unwrap();
    let engine = FakeEngine::start(
        vec![Reply::Text("from the upload".into())],
        StreamMode::Drop {
            failing: usize::MAX,
            after_frames: 2,
            partial: "hello".into(),
            final_text: "never".into(),
        },
    );
    let mut host = serve(scripted(root.path(), vec![FakeTakeScript::clean()], Some(&engine))).expect("serves");
    let app = watching(&host);
    start(&app, "take-f");
    let mut seen = Vec::new();
    let degraded = until_take(&app, "live text giving up", &mut seen, |frame| {
        matches!(frame, TakeWire::LiveText { take, degraded: Some(_), .. } if take == "take-f")
    });
    assert!(matches!(degraded, TakeWire::LiveText { .. }));
    stop(&app, "take-f");
    let id = stored_id_of(&app, "take-f", &mut seen);
    let (state, _) = finished(&app, &id, &mut seen);
    assert_eq!(state, completed("from the upload"));
    let started = seen.iter().position(|frame| {
        matches!(frame, TakeWire::Transcription { state: TranscriptionState::Started { backend }, yours: true, .. } if backend == "openai:fake-model")
    });
    let stored = seen.iter().position(|frame| matches!(frame, TakeWire::Persisted { .. }));
    assert!(started.is_some() && stored < started, "stored, then started: {seen:#?}");
    assert_eq!(engine.stream_sessions(), 3, "the first session and two reconnects");
    assert_eq!(engine.batch_requests(), 1);
    assert_eq!(completed_attempts(root.path(), &id), vec!["from the upload"]);
    host.shutdown();
}

/// The engine-fault matrix on one take: an error status, a server that
/// hangs up mid-request, and blank answers each keep the take and its
/// audio; every retry adds an attempt beside the earlier ones, answered to
/// the window that asked; a blank result never hides real words; and a
/// retry that cannot run (a built-in model while the host serves the
/// user's server) leaves the take alone.
#[test]
fn engine_faults_keep_the_take_and_every_retry_adds_an_attempt() {
    let root = tempfile::tempdir().unwrap();
    let engine = FakeEngine::start(
        vec![
            Reply::Status("500 Internal Server Error"),
            Reply::HangUp,
            Reply::Text(" ".into()),
            Reply::Text("real words".into()),
            Reply::Text(" ".into()),
        ],
        StreamMode::Refuse,
    );
    let mut host = serve(scripted(root.path(), vec![FakeTakeScript::clean()], Some(&engine))).expect("serves");
    let owner = watching(&host);
    let asker = watching(&host);
    record(&owner, "take-x");
    let mut seen = Vec::new();
    let id = stored_id_of(&owner, "take-x", &mut seen);
    let samples = StoreV2::open(root.path()).unwrap().load_audio(&id).unwrap().samples;

    // The take's own transcription: an error status.
    let (state, yours) = finished(&owner, &id, &mut seen);
    assert!(yours);
    assert!(matches!(&state, TranscriptionState::Failed { transport: false, .. }), "{state:?}");
    let store = StoreV2::open(root.path()).unwrap();
    assert_eq!(store.attempts_for(&id).unwrap()[0].status, "failed");
    assert!(!store.transcription_wanted(&id).unwrap(), "a failure ends the intent");
    drop(store);

    let server = TranscribeWith::Server {
        endpoint: engine.endpoint(),
        model: "big".into(),
    };
    let mut theirs = Vec::new();
    let retry = |req: &str, theirs: &mut Vec<TakeWire>| {
        asker.transcribe(req, &id, server.clone()).expect("asked");
        let (state, yours) = retried(&asker, req, theirs);
        assert!(yours, "{req}: the result is for the window that asked");
        let (seen_by_owner, owners) = retried(&owner, req, &mut Vec::new());
        assert_eq!(seen_by_owner, state, "{req}");
        assert!(!owners, "{req}: the owner only shows another window's retry");
        state
    };
    // The engine dies mid-request: a transport failure.
    let state = retry("r_hangup", &mut theirs);
    assert!(matches!(&state, TranscriptionState::Failed { transport: true, .. }), "{state:?}");
    // A blank answer with no words to keep.
    assert_eq!(retry("r_blank", &mut theirs), completed(" "));
    // Real words, on the server the retry named.
    assert_eq!(retry("r_words", &mut theirs), completed("real words"));
    assert!(theirs.iter().any(|frame| matches!(
        frame,
        TakeWire::Transcription { state: TranscriptionState::Started { backend }, .. } if backend == "openai:big"
    )));
    // A blank answer after them is kept, and the take keeps its words.
    let state = retry("r_blank_again", &mut theirs);
    assert!(
        matches!(&state, TranscriptionState::Completed { kept_earlier: true, .. }),
        "{state:?}"
    );
    let history = HistoryClient(&asker);
    let listed = history
        .list()
        .unwrap()
        .into_iter()
        .find_map(|record| match record {
            ListedRecord::Session(summary) if summary.id == id => Some(summary),
            _ => None,
        })
        .expect("listed");
    assert_eq!(listed.results.len(), 3, "every result is kept beside the others");
    assert_eq!(listed.model_label.as_deref(), Some("openai:big"), "the label of the words shown");
    assert_eq!(listed.transcript.expect("words").text, "real words", "a blank never hides words");

    // The host serves the user's server: no built-in model to wait for.
    asker
        .transcribe("r_model", &id, TranscribeWith::Model { model_id: "moss".into() })
        .unwrap();
    let (state, _) = retried(&asker, "r_model", &mut theirs);
    assert!(
        matches!(&state, TranscriptionState::Refused { message } if message.contains("unchanged")),
        "{state:?}"
    );

    let store = StoreV2::open(root.path()).unwrap();
    assert_eq!(store.attempts_for(&id).unwrap().len(), 5, "every run is an attempt; the refusal is none");
    assert_eq!(store.load_audio(&id).unwrap().samples, samples, "the audio is untouched throughout");
    drop(store);
    assert_eq!(completed_attempts(root.path(), &id), vec![" ", "real words", " "]);
    assert_eq!(engine.batch_requests(), 5);
    host.shutdown();
}

/// A retry the host accepted but has not started (both job slots busy)
/// keeps its take's audio from every process's upkeep — also once the
/// window that asked is gone.
#[test]
fn a_queued_retry_holds_its_audio_after_its_window_left() {
    let root = tempfile::tempdir().unwrap();
    let ids: Vec<String> = (0..3).map(|_| stored_take(root.path(), false, 0.1)).collect();
    let held = FakeEngine::start(
        vec![Reply::Held("a".into()), Reply::Held("b".into()), Reply::Held("c".into())],
        StreamMode::Refuse,
    );
    let mut host = serve(scripted(root.path(), Vec::new(), Some(&held))).expect("serves");
    let with = TranscribeWith::Server {
        endpoint: held.endpoint(),
        model: "fake-model".into(),
    };
    {
        let asker = watching(&host);
        for (index, id) in ids.iter().enumerate() {
            asker.transcribe(&format!("r_{index}"), id, with.clone()).unwrap();
        }
        wait_for("the first two retries to start", || held.batch_requests() >= 2);
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

/// Another window sees a take its live owner records as owned: tapping
/// it (to show levels) does not take it over, it can neither stop nor
/// abort it, and its own start is refused naming what holds the
/// microphone — until the owner's take is stored.
#[test]
fn another_window_can_watch_but_never_take_over_or_end_a_live_take() {
    let root = tempfile::tempdir().unwrap();
    let mut host = serve(scripted(
        root.path(),
        vec![FakeTakeScript::clean(), FakeTakeScript::clean()],
        None,
    ))
    .expect("serves");
    let owner = watching(&host);
    let other = watching(&host);
    start(&owner, "take-mine");
    let mut seen = Vec::new();
    until_take(&owner, "a live tick", &mut seen, is_live_tick("take-mine"));
    let tick = until_take(&other, "the owner's take", &mut Vec::new(), is_live_tick("take-mine"));
    assert!(matches!(tick, TakeWire::Live { owner: TakeOwner::Another, .. }), "{tick:?}");
    other.take_tap("take-mine", 0).unwrap();
    for (corr, command) in [
        (Some("take-mine"), Command::CaptureStop { drain: Some(true) }),
        (Some("take-mine"), Command::CaptureAbort),
        (None, Command::CaptureStop { drain: Some(true) }),
        (None, Command::CaptureAbort),
    ] {
        match other.send(corr, command) {
            Err(ClientError::Rejected(rejection)) => assert!(
                matches!(
                    &rejection,
                    starling_runtime::machine::Rejection::IllegalInState { detail, .. }
                        if detail.contains("another connection")
                ),
                "{rejection:?}"
            ),
            other => panic!("expected a refusal, got {other:?}"),
        }
    }
    let again = || Command::CaptureStart { policy: "push-to-talk".into() };
    for (client, take, yours) in [(&other, "take-theirs", false), (&owner, "take-again", true)] {
        match client.send_reporting_busy(Some(take), again()) {
            Err((ClientError::Rejected(_), busy)) => {
                assert_eq!(busy, Some(TakeBusy::Recording { yours }), "{take}")
            }
            other => panic!("expected a refusal, got {other:?}"),
        }
    }
    assert_eq!(host_capture_state(&owner), "Recording", "the take records on");

    stop(&owner, "take-mine");
    let persisted = until_take(&owner, "the stored row", &mut seen, is_persisted("take-mine"));
    assert!(matches!(persisted, TakeWire::Persisted { orphan: false, .. }), "{persisted:?}");
    other
        .send_reporting_busy(Some("take-theirs"), again())
        .expect("a start once the microphone is free");
    drop(owner);
    drop(other);
    host.shutdown();
}

#[test]
fn refused_starts_during_a_slow_device_open_never_take_the_owner_away() {
    let root = tempfile::tempdir().unwrap();
    let slow = FakeTakeScript {
        open_delay: Duration::from_millis(800),
        ..FakeTakeScript::clean()
    };
    let source = FakeCaptureSource::new(vec![slow]);
    let mut host = serve(host_config(root.path(), source.clone(), None)).expect("host serves");
    let owner = Arc::new(connect(&host));
    owner.take_watch().unwrap();
    let opening = std::thread::spawn({
        let owner = Arc::clone(&owner);
        move || start(&owner, "take-slow")
    });
    // While the device opens, other windows press record: each start is
    // refused, and there are more of them than the hub remembers.
    wait_for("the take to start opening its device", || {
        !source.started_takes.lock().unwrap().is_empty()
    });
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

/// A take whose window went away is unowned; of the windows that come
/// back, the first to tap it adopts it (replayed from its first sample),
/// the second sees it owned, and the adopter's stop stores it as its own.
#[test]
fn an_unowned_take_is_adopted_by_the_first_window_to_tap_it() {
    let root = tempfile::tempdir().unwrap();
    let engine = FakeEngine::start(vec![Reply::Text("adopted".into())], StreamMode::Refuse);
    let mut host = serve(scripted(root.path(), vec![FakeTakeScript::clean()], Some(&engine))).expect("serves");
    {
        let gone = watching(&host);
        start(&gone, "take-r");
        until_take(&gone, "a live tick", &mut Vec::new(), is_live_tick("take-r"));
        std::thread::sleep(Duration::from_millis(150));
    }
    let first = watching(&host);
    let second = watching(&host);
    for window in [&first, &second] {
        let tick = until_take(window, "the running take", &mut Vec::new(), is_live_tick("take-r"));
        assert!(matches!(tick, TakeWire::Live { owner: TakeOwner::Nobody, .. }), "its owner is gone: {tick:?}");
    }
    first.take_tap("take-r", 0).unwrap();
    let mut seen = Vec::new();
    until_take(&first, "owned after tapping", &mut seen, |frame| {
        matches!(frame, TakeWire::Live { take, owner: TakeOwner::You, ended: None, .. } if take == "take-r")
    });
    second.take_tap("take-r", 0).unwrap();
    until_take(&second, "the second tap sees another owner", &mut Vec::new(), |frame| {
        matches!(frame, TakeWire::Live { owner: TakeOwner::Another, ended: None, .. })
    });
    stop(&first, "take-r");
    let end = until_take(&first, "the end", &mut seen, is_end("take-r"));
    let TakeWire::Live { ended: Some(total), .. } = end else { unreachable!() };
    assert_eq!(audio_of(&seen, "take-r").len() as u64, total, "replayed from the start");
    let persisted = until_take(&first, "the stored row", &mut seen, is_persisted("take-r"));
    let TakeWire::Persisted { orphan: false, stored_id: Some(id), .. } = persisted else {
        panic!("the adopting app owns the stored take: {persisted:?}");
    };
    let (state, yours) = finished(&first, &id, &mut seen);
    assert_eq!((state, yours), (completed("adopted"), true));
    drop(first);
    drop(second);
    host.shutdown();
}

#[test]
fn a_take_a_command_only_client_records_is_the_watching_apps_to_act_on() {
    let root = tempfile::tempdir().unwrap();
    let mut host = serve(scripted(root.path(), vec![FakeTakeScript::clean()], None)).expect("host serves");
    let app = watching(&host);
    let tool = connect(&host);
    start(&tool, "take-tool");
    let mut seen = Vec::new();
    until_take(&app, "the tool's take", &mut seen, is_live_tick("take-tool"));
    stop(&tool, "take-tool");
    let heard = until_take(&app, "the stored row", &mut seen, is_persisted("take-tool"));
    assert!(matches!(heard, TakeWire::Persisted { orphan: false, .. }), "{heard:?}");
    // The tool follows no feed: the watching app is the one to show the
    // result (this host has no engine: the take says why it has none).
    let result = until_take(&app, "the transcription's end", &mut seen, |frame| {
        matches!(frame, TakeWire::Transcription { take: Some(take), state, .. } if take == "take-tool" && state.is_final())
    });
    assert!(matches!(result, TakeWire::Transcription { yours: true, .. }), "{result:?}");
    drop(tool);
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
    let mut host = serve(scripted(root.path(), vec![long, FakeTakeScript::clean()], None)).expect("host serves");
    let app = watching(&host);
    start(&app, "take-one");
    until_take(&app, "the first take", &mut Vec::new(), is_live_tick("take-one"));
    std::thread::sleep(Duration::from_millis(900));
    stop(&app, "take-one");
    wait_for("the first take stored", || host_capture_state(&app) == "Persisted");
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
fn a_corr_a_finished_take_used_is_free_for_the_next_start() {
    let root = tempfile::tempdir().unwrap();
    let mut host = serve(scripted(
        root.path(),
        vec![FakeTakeScript::clean(), FakeTakeScript::clean()],
        None,
    ))
    .expect("host serves");
    let first = watching(&host);
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
    let second = watching(&host);
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

/// A microphone that will not open is reported to every watcher; a host
/// with a client is busy, and idle again once its clients left.
#[test]
fn a_microphone_that_will_not_open_is_reported_and_the_host_idles_once_left_alone() {
    let root = tempfile::tempdir().unwrap();
    let mut host = serve(scripted(root.path(), vec![], None)).expect("host serves");
    assert!(host.idle());
    let app = watching(&host);
    assert!(!host.idle(), "a connected client keeps the host");
    start(&app, "take-f");
    let failed = until_take(&app, "the start failure", &mut Vec::new(), |frame| {
        matches!(frame, TakeWire::StartFailed { take, .. } if take == "take-f")
    });
    let TakeWire::StartFailed { message, problem, .. } = failed else { unreachable!() };
    assert!(message.contains("No microphone"), "{message}");
    assert_eq!(problem, None, "the fake source does not classify its failure");
    drop(app);
    wait_for("the host to go idle", || host.idle());
    host.shutdown();
}
