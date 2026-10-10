//! #220: the capture observer — the seam the runtime host's take feed
//! hangs on. A take reports its start (with a monitor whose audio is the
//! take's own), its end (with everything it kept), and its persist (with
//! the stored row's id), in that order; a take with nothing to keep ends
//! with no record and no persist; a device that will not open reports
//! the source's own error text.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use starling_runtime::bus::{EventMessage, EventSub};
use starling_runtime::machine::capture::{
    CaptureConfig, CaptureObserver, LiveTakeMonitor, TakeRecord, V2CaptureStore,
};
use starling_runtime::protocol::Command;
use starling_runtime::testing::{FakeCaptureSource, FakeStop, FakeTakeScript};
use starling_runtime::RuntimeConfig;

#[derive(Debug, Clone, PartialEq)]
enum Seen {
    Started { corr: String, monitored: bool },
    StartFailed { corr: String, detail: String },
    Ended { corr: String, samples: Option<usize> },
    Persisted { corr: String, stored: Result<Option<String>, String> },
}

#[derive(Default)]
struct Recording {
    seen: Mutex<Vec<Seen>>,
    monitor: Mutex<Option<Arc<dyn LiveTakeMonitor>>>,
    ended: Mutex<Option<Arc<TakeRecord>>>,
}

impl CaptureObserver for Recording {
    fn take_started(&self, corr: &str, monitor: Option<Arc<dyn LiveTakeMonitor>>) {
        self.seen.lock().unwrap().push(Seen::Started {
            corr: corr.to_string(),
            monitored: monitor.is_some(),
        });
        *self.monitor.lock().unwrap() = monitor;
    }
    fn take_start_failed(&self, corr: &str, detail: &str) {
        self.seen.lock().unwrap().push(Seen::StartFailed {
            corr: corr.to_string(),
            detail: detail.to_string(),
        });
    }
    fn take_ended(&self, corr: &str, record: Option<&Arc<TakeRecord>>) {
        self.seen.lock().unwrap().push(Seen::Ended {
            corr: corr.to_string(),
            samples: record.map(|record| record.samples.len()),
        });
        *self.ended.lock().unwrap() = record.cloned();
    }
    fn take_persisted(
        &self,
        corr: &str,
        _record: &Arc<TakeRecord>,
        stored: Result<Option<String>, String>,
    ) {
        self.seen.lock().unwrap().push(Seen::Persisted {
            corr: corr.to_string(),
            stored,
        });
    }
}

fn until(events: &EventSub, label: &str, predicate: impl Fn(&EventMessage) -> bool) {
    let start = Instant::now();
    loop {
        match events.recv_timeout(Duration::from_millis(20)) {
            Ok(message) if predicate(&message) => return,
            Ok(_) => {}
            Err(starling_runtime::channel::RecvError::Timeout) => {
                assert!(start.elapsed() < Duration::from_secs(5), "timed out waiting for {label}");
            }
            Err(other) => panic!("event stream error: {other:?}"),
        }
    }
}

fn wait_for(observer: &Recording, count: usize) -> Vec<Seen> {
    let start = Instant::now();
    loop {
        let seen = observer.seen.lock().unwrap().clone();
        if seen.len() >= count || start.elapsed() > Duration::from_secs(5) {
            return seen;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

fn runtime(
    root: &std::path::Path,
    scripts: Vec<FakeTakeScript>,
    observer: Arc<Recording>,
) -> (starling_runtime::Runtime, starling_runtime::RuntimeClient) {
    runtime_on(Arc::new(V2CaptureStore::open(root).expect("v2 store")), root, scripts, observer)
}

fn runtime_on(
    store: Arc<V2CaptureStore>,
    root: &std::path::Path,
    scripts: Vec<FakeTakeScript>,
    observer: Arc<Recording>,
) -> (starling_runtime::Runtime, starling_runtime::RuntimeClient) {
    let config = RuntimeConfig::default()
        .with_capture_source(FakeCaptureSource::new(scripts))
        .with_capture_store(store)
        .with_capture_observer(observer)
        .with_capture_config(CaptureConfig {
            journals_dir: root.join("journals"),
            poll_interval: Duration::from_millis(10),
            ..CaptureConfig::default()
        });
    starling_runtime::Runtime::start(config)
}

#[test]
fn a_take_reports_start_end_and_its_stored_row_in_order() {
    let root = tempfile::tempdir().unwrap();
    let observer = Arc::new(Recording::default());
    let (runtime, client) = runtime(root.path(), vec![FakeTakeScript::clean()], Arc::clone(&observer));
    let events = runtime.subscribe();
    client
        .send(Some("t1"), Command::CaptureStart { policy: "push-to-talk".into() })
        .unwrap();
    until(&events, "progress", |m| m.type_name() == "capture.progress");
    let monitor = observer.monitor.lock().unwrap().clone().expect("a monitor");
    let live = monitor.samples_from(0, usize::MAX);
    assert!(!live.is_empty(), "the monitor serves the take's audio while it records");
    client
        .send(Some("t1"), Command::CaptureStop { drain: Some(true) })
        .unwrap();
    until(&events, "stopped", |m| m.type_name() == "capture.stopped");

    let seen = wait_for(&observer, 3);
    assert_eq!(
        seen[0],
        Seen::Started { corr: "t1".into(), monitored: true }
    );
    assert!(matches!(&seen[1], Seen::Ended { corr, samples: Some(n) } if corr == "t1" && *n > 0));
    let Seen::Persisted { corr, stored: Ok(Some(id)) } = &seen[2] else {
        panic!("expected a named stored row, saw {seen:?}");
    };
    assert_eq!(corr, "t1");
    // The fake's journal does not exist on disk, so the take was stored
    // from its samples under a fresh id that names the journal it
    // replaces.
    let store = starling_dictation::store_v2::StoreV2::open(root.path()).unwrap();
    assert!(store.get_capture(id).unwrap().is_some(), "the named row exists");
    let ended = observer.ended.lock().unwrap().clone().expect("the ended record");
    assert_eq!(
        &ended.samples[..live.len()],
        &live[..],
        "what the monitor served is the take's own audio"
    );
    runtime.shutdown();
}

#[test]
fn a_take_with_nothing_to_keep_ends_without_a_record_or_persist() {
    let root = tempfile::tempdir().unwrap();
    let observer = Arc::new(Recording::default());
    let script = FakeTakeScript {
        stop: FakeStop::Empty,
        ..FakeTakeScript::clean()
    };
    let (runtime, client) = runtime(root.path(), vec![script], Arc::clone(&observer));
    let events = runtime.subscribe();
    client
        .send(Some("t2"), Command::CaptureStart { policy: "push-to-talk".into() })
        .unwrap();
    until(&events, "started", |m| m.type_name() == "capture.started");
    client
        .send(Some("t2"), Command::CaptureStop { drain: Some(true) })
        .unwrap();
    until(&events, "stopped", |m| m.type_name() == "capture.stopped");
    std::thread::sleep(Duration::from_millis(50));
    let seen = observer.seen.lock().unwrap().clone();
    assert_eq!(
        seen,
        vec![
            Seen::Started { corr: "t2".into(), monitored: true },
            Seen::Ended { corr: "t2".into(), samples: None },
        ]
    );
    runtime.shutdown();
}

#[test]
fn a_device_that_will_not_open_reports_the_sources_reason() {
    let root = tempfile::tempdir().unwrap();
    let observer = Arc::new(Recording::default());
    let (runtime, client) = runtime(root.path(), vec![], Arc::clone(&observer));
    let events = runtime.subscribe();
    client
        .send(Some("t3"), Command::CaptureStart { policy: "push-to-talk".into() })
        .unwrap();
    until(&events, "capture.error", |m| m.type_name() == "capture.error");
    let seen = wait_for(&observer, 1);
    assert!(
        matches!(&seen[..], [Seen::StartFailed { corr, detail }] if corr == "t3" && detail.contains("No microphone")),
        "{seen:?}"
    );
    runtime.shutdown();
}

#[test]
fn a_take_that_ended_interrupted_does_not_refuse_the_next_one() {
    // #220: the host runs for hours; a lost device on one take must not
    // leave the machine refusing every later capture.start.
    let root = tempfile::tempdir().unwrap();
    let observer = Arc::new(Recording::default());
    let lost = FakeTakeScript {
        stop: FakeStop::DeviceError("the device went away".into()),
        ..FakeTakeScript::clean()
    };
    let (runtime, client) =
        runtime(root.path(), vec![lost, FakeTakeScript::clean()], Arc::clone(&observer));
    let events = runtime.subscribe();
    client
        .send(Some("t4"), Command::CaptureStart { policy: "push-to-talk".into() })
        .unwrap();
    until(&events, "started", |m| m.type_name() == "capture.started");
    client
        .send(Some("t4"), Command::CaptureStop { drain: Some(true) })
        .unwrap();
    until(&events, "the fatal error", |m| {
        m.type_name() == "capture.error" && m.to_value()["payload"]["fatal"] == true
    });
    let deadline = Instant::now() + Duration::from_secs(5);
    while runtime.snapshot().capture.state != "Interrupted" {
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(10));
    }
    client
        .send(Some("t5"), Command::CaptureStart { policy: "push-to-talk".into() })
        .expect("the next take is accepted");
    until(&events, "the next take started", |m| {
        m.type_name() == "capture.started" && m.corr.as_deref() == Some("t5")
    });
    runtime.shutdown();
}

#[test]
fn a_stop_that_reports_a_failed_device_keeps_the_take_as_interrupted() {
    let root = tempfile::tempdir().unwrap();
    let observer = Arc::new(Recording::default());
    let faulted = FakeTakeScript {
        stop: FakeStop::FaultedClean {
            journal_id: "j_fault".into(),
            fault: "stream error".into(),
        },
        ..FakeTakeScript::clean()
    };
    let (runtime, client) = runtime(root.path(), vec![faulted], Arc::clone(&observer));
    let events = runtime.subscribe();
    client
        .send(Some("t6"), Command::CaptureStart { policy: "push-to-talk".into() })
        .unwrap();
    until(&events, "progress", |m| m.type_name() == "capture.progress");
    client
        .send(Some("t6"), Command::CaptureStop { drain: Some(true) })
        .unwrap();
    until(&events, "stopped", |m| m.type_name() == "capture.stopped");
    let seen = wait_for(&observer, 3);
    let Some(Seen::Persisted { stored: Ok(Some(id)), .. }) = seen.get(2) else {
        panic!("expected a stored row: {seen:?}");
    };
    let store = starling_dictation::store_v2::StoreV2::open(root.path()).unwrap();
    let record = store.get_capture(id).unwrap().expect("stored");
    assert_eq!(
        record.status,
        starling_dictation::store_v2::CaptureStatus::Interrupted,
        "never presented as complete"
    );
    runtime.shutdown();
}

/// #220: a store told which takes to transcribe commits each complete
/// one with that intent, in the commit itself; a cancelled take (stored
/// interrupted) and a take it was not told about carry none.
#[test]
fn a_clean_take_is_stored_with_the_intent_to_transcribe_it() {
    use starling_runtime::machine::capture::CaptureStore;
    let root = tempfile::tempdir().unwrap();
    let observer = Arc::new(Recording::default());
    let store = Arc::new(V2CaptureStore::open(root.path()).expect("v2 store"));
    store.transcribe_takes(Arc::new(|take: &TakeRecord| !take.id.starts_with("ask_")));
    let (runtime, client) = runtime_on(
        store,
        root.path(),
        vec![FakeTakeScript::clean(), FakeTakeScript::clean(), FakeTakeScript::clean()],
        Arc::clone(&observer),
    );
    let events = runtime.subscribe();
    let mut stored = Vec::new();
    for (corr, command) in [
        ("t_kept", Command::CaptureStop { drain: Some(true) }),
        ("t_cancelled", Command::CaptureAbort),
        ("ask_agent", Command::CaptureStop { drain: Some(true) }),
    ] {
        client
            .send(Some(corr), Command::CaptureStart { policy: "push-to-talk".into() })
            .unwrap();
        until(&events, "progress", |m| m.type_name() == "capture.progress");
        client.send(Some(corr), command).unwrap();
        let start = Instant::now();
        loop {
            let id = observer.seen.lock().unwrap().iter().find_map(|seen| match seen {
                Seen::Persisted { corr: seen, stored: Ok(Some(id)) } if seen == corr => {
                    Some(id.clone())
                }
                _ => None,
            });
            if let Some(id) = id {
                stored.push(id);
                break;
            }
            assert!(start.elapsed() < Duration::from_secs(5), "{corr} was not stored");
            std::thread::sleep(Duration::from_millis(10));
        }
    }
    let store = starling_dictation::store_v2::StoreV2::open(root.path()).unwrap();
    assert!(store.transcription_wanted(&stored[0]).unwrap(), "the clean take");
    assert!(!store.transcription_wanted(&stored[1]).unwrap(), "the cancelled take");
    assert!(!store.transcription_wanted(&stored[2]).unwrap(), "a take it was not told about");
    assert_eq!(store.transcriptions_due(Duration::ZERO).unwrap(), vec![stored[0].clone()]);
    runtime.shutdown();
}
