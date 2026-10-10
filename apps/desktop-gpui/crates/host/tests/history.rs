//! #220: the app's history through the host, over the real transport.
//! Every store request the app makes — the list, a take's audio, imports,
//! processing documents, insights and corrections, deletes, retention
//! classes, audio holds — is answered by the host on its own store
//! handle; answers and uploads too large for one frame go in chunks; what
//! a connection was handed goes with it; the audio upkeep runs in the
//! host, reports to watching apps and waits while a take records.

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use starling_dictation::audio::{decode_pcm16_wav, encode_wav_16k, PcmAudio};
use starling_dictation::storage::{ListedRecord, SessionStatus, SessionSummary, TranscriptionResult};
use starling_dictation::store_v2::{CorrectionDecision, CorrectionRecord, StoreV2, TakeMeta};
use starling_runtime::machine::capture::{CaptureConfig, V2CaptureStore};
use starling_runtime::protocol::Command;
use starling_runtime::testing::{FakeCaptureSource, FakeTakeScript};
use starling_runtime_host::client::{ClientError, HostClient, TakeWire};
use starling_runtime_host::history::{
    AudioFormat, HistoryClient, ProposalRow, RowStatus, StoreCall, StoreReply, StoreRequest,
};
use starling_runtime_host::{serve, HostConfig, HostHandle};

fn config(root: &Path, source: Arc<FakeCaptureSource>) -> HostConfig {
    let mut config = HostConfig::new(root, root.join("endpoints"))
        // Passes run when a test asks for one.
        .with_upkeep(Duration::from_secs(3600), Duration::from_secs(3600));
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

fn host_at(root: &Path) -> HostHandle {
    serve(config(root, FakeCaptureSource::new(Vec::new()))).expect("host serves")
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

fn wav(samples: usize) -> Vec<u8> {
    encode_wav_16k(&PcmAudio {
        samples: (0..samples)
            .map(|i| ((i as f32) * 0.03).sin() * 0.25 + (i % 7) as f32 * 0.001)
            .collect(),
        sample_rate: 16_000,
        channels: 1,
    })
    .expect("wav")
}

/// A take stored beside the host, the way another process would.
fn stored(root: &Path, samples: usize) -> String {
    StoreV2::open(root)
        .expect("store")
        .save_wav_capture(&wav(samples), TakeMeta::for_device("test"))
        .expect("save")
        .record
        .id
}

fn transcribed(root: &Path, samples: usize, text: &str) -> (String, String) {
    let id = stored(root, samples);
    let mut store = StoreV2::open(root).expect("store");
    let attempt = store.begin_recognition(&id, "engine:test", None).expect("begin");
    store
        .finish_recognition_transcript(
            &id,
            &TranscriptionResult {
                text: text.to_string(),
                segments: Vec::new(),
                duration_seconds: None,
                request_id: None,
            },
        )
        .expect("transcript");
    (id, attempt)
}

fn summary(history: &HistoryClient<&HostClient>, id: &str) -> Option<SessionSummary> {
    history.list().expect("list").into_iter().find_map(|record| match record {
        ListedRecord::Session(summary) if summary.id == id => Some(summary),
        _ => None,
    })
}

/// Whether take `id`'s audio is still the journal (not compressed).
fn uncompressed(root: &Path, id: &str) -> bool {
    let path = StoreV2::open(root).unwrap().audio_journal_path(id).expect("audio");
    path.extension().and_then(|ext| ext.to_str()) == Some("sj")
}

/// The next upkeep report on `client`'s take feed.
fn next_upkeep(client: &HostClient, within: Duration) -> Option<(String, bool)> {
    let deadline = Instant::now() + within;
    while Instant::now() < deadline {
        if let Ok(TakeWire::Upkeep { report, retired }) =
            client.recv_take_timeout(Duration::from_millis(20))
        {
            return Some((report, retired));
        }
    }
    None
}

#[test]
fn the_history_list_and_a_takes_audio_come_from_the_host() {
    let root = tempfile::tempdir().unwrap();
    let mut host = host_at(root.path());
    let (id, _) = transcribed(root.path(), 32_000, "hello from the store");
    let client = connect(&host);
    let history = HistoryClient(&client);

    let take = summary(&history, &id).expect("listed");
    assert_eq!(take.status, SessionStatus::Transcribed);
    assert_eq!(take.transcript.expect("text").text, "hello from the store");
    assert_eq!(take.duration_ms, Some(2000.0));

    // A take's audio is many frames long: fetched in chunks, intact.
    let audio = history.audio(&id, AudioFormat::Wav).expect("wav").expect("present");
    assert_eq!(audio, wav(32_000));
    let flac = history.audio(&id, AudioFormat::Flac).expect("flac").expect("present");
    assert!(flac.starts_with(b"fLaC"));
    assert!(history.audio("c_nobody", AudioFormat::Wav).expect("answered").is_none());

    drop(client);
    host.shutdown();
}

#[test]
fn answers_and_uploads_larger_than_a_frame_go_in_chunks() {
    let root = tempfile::tempdir().unwrap();
    let mut host = serve(
        config(root.path(), FakeCaptureSource::new(Vec::new())).with_max_frame_bytes(16 * 1024),
    )
    .expect("host serves");
    for index in 0..40 {
        transcribed(root.path(), 1600, &format!("take {index} {}", "words ".repeat(40)));
    }
    let client = connect(&host);
    let history = HistoryClient(&client);
    // The list itself is larger than a frame.
    match client.store(StoreRequest::List).expect("answered") {
        StoreReply::Large { bytes, .. } => assert!(bytes > 16 * 1024),
        other => panic!("expected a large answer, got {other:?}"),
    }
    assert_eq!(history.list().expect("list").len(), 40);

    // An import many frames long goes up in parts.
    let imported = wav(48_000);
    let id = history.import(&imported, false).expect("import");
    assert_eq!(history.audio(&id, AudioFormat::Wav).unwrap().unwrap(), imported);
    assert_eq!(summary(&history, &id).expect("listed").status, SessionStatus::Captured);

    // A kept answer read to its end is gone; one never asked for is no
    // answer at all.
    let StoreReply::Bytes { blob, bytes } = client
        .store(StoreRequest::Audio { id: id.clone(), format: AudioFormat::Wav })
        .unwrap()
    else {
        panic!("expected bytes");
    };
    let mut offset = 0;
    while offset < bytes {
        let StoreReply::Chunk { data } = client
            .store(StoreRequest::Fetch { blob: blob.clone(), offset })
            .unwrap()
        else {
            panic!("expected a chunk");
        };
        use base64::Engine;
        offset += base64::engine::general_purpose::STANDARD.decode(data).unwrap().len() as u64;
    }
    assert!(matches!(
        client.store(StoreRequest::Fetch { blob, offset: 0 }).unwrap(),
        StoreReply::Failed { .. }
    ));
    // An upload out of order is refused, never stitched.
    assert!(matches!(
        client
            .store(StoreRequest::Upload {
                upload: "up_x".into(),
                offset: 10,
                data: "AAAA".into(),
            })
            .unwrap(),
        StoreReply::Failed { .. }
    ));
    drop(client);
    host.shutdown();
}

#[test]
fn an_import_with_its_intent_is_due_for_transcription() {
    let root = tempfile::tempdir().unwrap();
    let mut host = host_at(root.path());
    let client = connect(&host);
    let history = HistoryClient(&client);
    let id = history.import(&wav(8000), true).expect("import");
    assert!(StoreV2::open(root.path()).unwrap().transcription_wanted(&id).unwrap());
    let plain = history.import(&wav(8000), false).expect("import");
    assert!(!StoreV2::open(root.path()).unwrap().transcription_wanted(&plain).unwrap());
    // Nothing decodable is refused with the store's reason.
    let err = history.import(b"not a wav", false).expect_err("refused");
    assert!(err.to_string().contains("wav") || err.to_string().contains("RIFF"), "{err}");
    drop(client);
    host.shutdown();
}

#[test]
fn processing_documents_insights_and_corrections_round_trip() {
    let root = tempfile::tempdir().unwrap();
    let mut host = host_at(root.path());
    let (id, attempt) = transcribed(root.path(), 1600, "um so hello there");
    let client = connect(&host);
    let history = HistoryClient(&client);

    assert_eq!(
        history.latest_raw(&id).unwrap(),
        Some((attempt.clone(), "um so hello there".to_string()))
    );
    assert_eq!(history.processing_doc(&id).unwrap(), None);
    let doc = history.start_processing_doc(&id, &attempt, "um so hello there").unwrap();
    assert_eq!((doc.head_revision, doc.head_is_raw), (1, true));
    let proposal = ProposalRow {
        request_id: "p1".into(),
        base_revision: 1,
        text: "So, hello there.".into(),
        status: RowStatus::Proposed,
        label: "S1-mini · this computer".into(),
        failure: None,
        stop_to_result_ms: Some(12.5),
        origin: None,
    };
    history.save_proposal(&id, &proposal).unwrap();
    let accepted = ProposalRow {
        status: RowStatus::Accepted,
        ..proposal.clone()
    };
    history
        .commit_processing_head(&id, 2, "So, hello there.", false, &attempt, Some(&accepted), Some("p1"))
        .unwrap();
    let doc = history.processing_doc(&id).unwrap().expect("doc");
    assert_eq!(doc.head_text, "So, hello there.");
    assert_eq!(doc.accepted_request.as_deref(), Some("p1"));
    assert_eq!(doc.proposals, vec![accepted]);

    history
        .record_insight(&id, "ev1", "processing_recorded", "2026-10-10T10:00:00Z", "{}")
        .unwrap();
    let record = CorrectionRecord {
        capture_id: id.clone(),
        request_id: "p1".into(),
        raw_attempt_id: attempt.clone(),
        raw_text: "um so hello there".into(),
        processed_text: "So, hello there.".into(),
        final_text: Some("So, hello there.".into()),
        decision: CorrectionDecision::Accepted,
        decision_utc: "2026-10-10T10:00:00Z".into(),
        mode_id: None,
        mode_version: None,
        provider_id: None,
        provider_kind: None,
        provider_model: None,
        locality: None,
        transform_kinds: None,
        language: None,
        asr_backend: None,
        asr_model_hash: None,
        timings_json: None,
        settings_strength: None,
        extra_json: None,
    };
    assert!(history.record_correction(&record).unwrap());
    assert!(history
        .revise_correction(&id, "p1", CorrectionDecision::Reverted, "2026-10-10T10:01:00Z", "um")
        .unwrap());
    let store = StoreV2::open(root.path()).unwrap();
    assert_eq!(store.insight_events_for(&id).unwrap().len(), 1);
    let records = store.correction_records_for(&id).unwrap();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].decision, CorrectionDecision::Reverted);

    // A late write for a deleted take lands nowhere and says so.
    history.delete(&id).unwrap();
    history.delete(&id).expect("a second delete is no error");
    assert!(summary(&history, &id).is_none());
    assert!(matches!(
        history.save_proposal(&id, &proposal),
        Err(starling_dictation::storage::StorageError::NotFound(_))
    ));
    assert!(history.audio(&id, AudioFormat::Wav).unwrap().is_none());
    drop(client);
    host.shutdown();
}

#[test]
fn archiving_a_take_changes_its_class() {
    let root = tempfile::tempdir().unwrap();
    let mut host = host_at(root.path());
    let (id, _) = transcribed(root.path(), 1600, "keep me");
    let client = connect(&host);
    let history = HistoryClient(&client);
    history.set_archival(&id, true).unwrap();
    assert!(summary(&history, &id).unwrap().archival);
    history.set_archival(&id, false).unwrap();
    assert!(!summary(&history, &id).unwrap().archival);
    assert!(matches!(
        history.set_archival("c_nobody", true),
        Err(starling_dictation::storage::StorageError::NotFound(_))
    ));
    drop(client);
    host.shutdown();
}

/// Every watching window hears when another one changed the history
/// list: an import, a retention class, a delete.
#[test]
fn other_windows_hear_that_the_history_changed() {
    let root = tempfile::tempdir().unwrap();
    let mut host = host_at(root.path());
    let other = connect(&host);
    other.take_watch().unwrap();
    let client = connect(&host);
    let history = HistoryClient(&client);
    let heard = |what: &str| {
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            assert!(Instant::now() < deadline, "no history change heard after {what}");
            if let Ok(TakeWire::HistoryChanged) = other.recv_take_timeout(Duration::from_millis(20)) {
                return;
            }
        }
    };
    let id = history.import(&wav(1600), false).unwrap();
    heard("an import");
    history.set_archival(&id, true).unwrap();
    heard("archiving");
    history.delete(&id).unwrap();
    heard("a delete");
    // Reads change nothing.
    history.list().unwrap();
    assert!(other.recv_take_timeout(Duration::from_millis(300)).is_err());
    drop(client);
    drop(other);
    host.shutdown();
}

#[test]
fn a_held_take_is_left_alone_until_released_or_its_connection_ends() {
    let root = tempfile::tempdir().unwrap();
    let mut host = host_at(root.path());
    let watcher = connect(&host);
    watcher.take_watch().unwrap();
    let client = connect(&host);
    let history = HistoryClient(&client);
    let held = stored(root.path(), 32_000);
    let hold = history.hold_audio(&held).expect("held");
    // Each pass has something else to compress, so it reports.
    stored(root.path(), 32_000);
    history.upkeep(true).unwrap();
    let (report, retired) = next_upkeep(&watcher, Duration::from_secs(20)).expect("a pass");
    assert!(report.starts_with("Compressed 1 recording"), "{report}");
    assert!(!retired);
    assert!(uncompressed(root.path(), &held), "a held take is not compressed");
    assert_eq!(history.upkeep(false).unwrap().as_deref(), Some(report.as_str()));

    history.release_hold(&hold).unwrap();
    stored(root.path(), 32_000);
    history.upkeep(true).unwrap();
    let (report, _) = next_upkeep(&watcher, Duration::from_secs(20)).expect("a pass");
    assert!(report.starts_with("Compressed 2 recordings"), "{report}");
    assert!(!uncompressed(root.path(), &held));

    // A hold whose connection ended counts for nothing.
    let other = stored(root.path(), 32_000);
    history.hold_audio(&other).expect("held");
    drop(history);
    drop(client);
    let deadline = Instant::now() + Duration::from_secs(10);
    while uncompressed(root.path(), &other) {
        assert!(Instant::now() < deadline, "the dropped connection's hold stayed");
        let asker = connect(&host);
        HistoryClient(&asker).upkeep(true).unwrap();
        let _ = next_upkeep(&watcher, Duration::from_secs(5));
    }
    drop(watcher);
    host.shutdown();
}

#[test]
fn upkeep_waits_while_a_take_records() {
    let root = tempfile::tempdir().unwrap();
    let source = FakeCaptureSource::new(vec![FakeTakeScript::clean()]);
    let mut host = serve(config(root.path(), source)).expect("host serves");
    let app = connect(&host);
    app.take_watch().unwrap();
    let waiting = stored(root.path(), 32_000);
    app.send(Some("take-1"), Command::CaptureStart { policy: "push-to-talk".into() })
        .expect("start");
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        assert!(Instant::now() < deadline, "no live tick");
        if let Ok(TakeWire::Live { status: Some(_), .. }) =
            app.recv_take_timeout(Duration::from_millis(20))
        {
            break;
        }
    }
    let history = HistoryClient(&app);
    history.upkeep(true).unwrap();
    assert!(next_upkeep(&app, Duration::from_secs(2)).is_none(), "no pass while recording");
    assert!(uncompressed(root.path(), &waiting));
    app.send(Some("take-1"), Command::CaptureStop { drain: Some(true) })
        .expect("stop");
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        assert!(Instant::now() < deadline, "the take was not stored");
        if let Ok(TakeWire::Persisted { .. }) = app.recv_take_timeout(Duration::from_millis(20)) {
            break;
        }
    }
    history.upkeep(true).unwrap();
    let (report, _) = next_upkeep(&app, Duration::from_secs(20)).expect("a pass once it stopped");
    assert!(report.starts_with("Compressed"), "{report}");
    assert!(!uncompressed(root.path(), &waiting));
    drop(app);
    host.shutdown();
}

#[test]
fn agent_connections_cannot_reach_the_store() {
    let root = tempfile::tempdir().unwrap();
    let allowlist = root.path().join("agents.json");
    std::fs::write(
        &allowlist,
        r#"{"version":1,"clients":[{"name":"claude-code","token":"tok-1"}]}"#,
    )
    .unwrap();
    let mut host = serve(
        config(root.path(), FakeCaptureSource::new(Vec::new())).with_agent_allowlist(Some(allowlist)),
    )
    .expect("host serves");
    let agent = connect(&host);
    agent.agent_hello("claude-code", "tok-1").expect("allowlisted");
    match agent.store(StoreRequest::List) {
        Err(ClientError::Closed(reason)) | Err(ClientError::Protocol(reason)) => {
            assert!(reason.contains("protocol_violation") || reason.contains("closed"), "{reason}")
        }
        Err(ClientError::Timeout) => {}
        other => panic!("expected the connection closed, got {other:?}"),
    }
    assert!(agent.is_closed());
    drop(agent);
    host.shutdown();
}

#[test]
fn the_wav_a_take_reads_back_decodes_to_its_samples() {
    // The playback path: what the app plays is the stored samples.
    let root = tempfile::tempdir().unwrap();
    let mut host = host_at(root.path());
    let id = stored(root.path(), 4000);
    let client = connect(&host);
    let audio = HistoryClient(&client).audio(&id, AudioFormat::Wav).unwrap().unwrap();
    assert_eq!(decode_pcm16_wav(&audio).unwrap().samples.len(), 4000);
    // Every request is a StoreCall like any other.
    assert!(client.chunk_bytes() > 0);
    drop(client);
    host.shutdown();
}
