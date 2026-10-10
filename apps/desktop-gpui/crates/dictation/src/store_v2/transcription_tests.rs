//! Transcription intents and claims (#220).

use super::*;
use tempfile::TempDir;

fn store_in(dir: &TempDir) -> StoreV2 {
    StoreV2::open(dir.path().join("v2")).expect("open v2 store")
}

fn ramp(len: usize) -> Vec<f32> {
    (0..len).map(|i| (i % 997) as f32 * 0.0001).collect()
}

/// A take stored through the full protocol; `transcribe` records the
/// intent with it.
fn stored(store: &mut StoreV2, transcribe: bool, mark: CommitMark) -> String {
    let mut meta = TakeMeta::for_device("test-device");
    meta.transcribe = transcribe;
    let mut take = store.begin_take(meta).expect("begin take");
    take.append_and_seal(&ramp(400)).expect("append");
    take.finalize()
        .expect("finalize")
        .commit_marked(store, mark)
        .expect("commit")
        .record
        .id
}

fn claimed(claim: TranscriptionClaim) -> String {
    match claim {
        TranscriptionClaim::Claimed { attempt_id } => attempt_id,
        other => panic!("expected a claim, got {other:?}"),
    }
}

fn transcript(text: &str) -> crate::storage::TranscriptionResult {
    crate::storage::TranscriptionResult {
        text: text.to_string(),
        segments: Vec::new(),
        duration_seconds: None,
        request_id: None,
    }
}

#[test]
fn a_take_stored_to_be_transcribed_is_claimed_once_and_settled_for_good() {
    let dir = TempDir::new().expect("tempdir");
    let mut store = store_in(&dir);
    let id = stored(&mut store, true, CommitMark::Complete);
    assert!(store.transcription_wanted(&id).unwrap());
    assert_eq!(store.transcriptions_due(std::time::Duration::ZERO).unwrap(), vec![id.clone()]);

    let attempt = claimed(store.claim_transcription(&id, "engine:test", None).unwrap());
    assert_eq!(
        store.claim_transcription(&id, "engine:test", None).unwrap(),
        TranscriptionClaim::Held,
        "one claim per take"
    );
    assert!(store.transcriptions_due(std::time::Duration::ZERO).unwrap().is_empty(), "a live claim is not due");

    store
        .finish_attempt_transcript(&attempt, &transcript("hello"))
        .expect("settle");
    assert!(!store.transcription_wanted(&id).unwrap());
    assert_eq!(
        store.claim_transcription(&id, "engine:test", None).unwrap(),
        TranscriptionClaim::NotWanted
    );
    let attempts = store.attempts_for(&id).unwrap();
    assert_eq!(attempts.len(), 1);
    assert_eq!(attempts[0].status, "completed");
    assert_eq!(attempts[0].text, "hello");
}

#[test]
fn a_failed_attempt_ends_the_intent_too() {
    let dir = TempDir::new().expect("tempdir");
    let mut store = store_in(&dir);
    let id = stored(&mut store, true, CommitMark::Complete);
    let attempt = claimed(store.claim_transcription(&id, "engine:test", None).unwrap());
    store
        .finish_attempt(&attempt, RecognitionOutcome::Failed { message: "engine away" })
        .expect("settle");
    assert!(!store.transcription_wanted(&id).unwrap());
    assert!(store.transcriptions_due(std::time::Duration::ZERO).unwrap().is_empty());
}

#[test]
fn only_complete_takes_asked_for_carry_an_intent() {
    let dir = TempDir::new().expect("tempdir");
    let mut store = store_in(&dir);
    let plain = stored(&mut store, false, CommitMark::Complete);
    let interrupted = stored(
        &mut store,
        true,
        CommitMark::Interrupted {
            note: "cut short".to_string(),
        },
    );
    for id in [&plain, &interrupted] {
        assert!(!store.transcription_wanted(id).unwrap());
        assert_eq!(
            store.claim_transcription(id, "engine:test", None).unwrap(),
            TranscriptionClaim::NotWanted
        );
    }
    assert!(store.transcriptions_due(std::time::Duration::ZERO).unwrap().is_empty());
}

#[test]
fn a_claim_whose_claimant_died_is_taken_over_and_transcribed_once() {
    let dir = TempDir::new().expect("tempdir");
    let mut first = store_in(&dir);
    let id = stored(&mut first, true, CommitMark::Complete);
    let abandoned = claimed(first.claim_transcription(&id, "engine:test", None).unwrap());
    // The claimant goes away mid-request: its marker's lock goes with it.
    drop(first);

    let mut second = store_in(&dir);
    assert_eq!(second.transcriptions_due(std::time::Duration::ZERO).unwrap(), vec![id.clone()]);
    let attempt = claimed(second.claim_transcription(&id, "engine:test", None).unwrap());
    assert_ne!(attempt, abandoned);
    second
        .finish_attempt_transcript(&attempt, &transcript("once"))
        .expect("settle");

    let attempts = second.attempts_for(&id).unwrap();
    let completed: Vec<_> = attempts.iter().filter(|a| a.status == "completed").collect();
    assert_eq!(completed.len(), 1, "transcribed exactly once");
    let failed = attempts.iter().find(|a| a.id == abandoned).expect("old attempt");
    assert_eq!(failed.status, "failed");
    assert!(failed
        .extra_json
        .as_deref()
        .is_some_and(|extra| extra.contains(transcription::ABANDONED_CLAIM_NOTE)));
    // The dead claimant's attempt cannot be settled over the new result.
    assert!(matches!(
        second.finish_attempt(&abandoned, RecognitionOutcome::Failed { message: "late" }),
        Err(StoreV2Error::NotFound(_))
    ));
}

#[test]
fn of_two_live_claimants_only_one_gets_the_take() {
    let dir = TempDir::new().expect("tempdir");
    let mut first = store_in(&dir);
    let mut second = store_in(&dir);
    let id = stored(&mut first, true, CommitMark::Complete);
    let attempt = claimed(second.claim_transcription(&id, "engine:test", None).unwrap());
    assert_eq!(
        first.claim_transcription(&id, "engine:test", None).unwrap(),
        TranscriptionClaim::Held
    );
    assert!(first.transcriptions_due(std::time::Duration::ZERO).unwrap().is_empty());
    second
        .finish_attempt_transcript(&attempt, &transcript("one"))
        .unwrap();
    assert_eq!(
        first.claim_transcription(&id, "engine:test", None).unwrap(),
        TranscriptionClaim::NotWanted
    );
}

#[test]
fn settling_an_attempt_leaves_a_retry_beside_it_running() {
    let dir = TempDir::new().expect("tempdir");
    let mut store = store_in(&dir);
    let id = stored(&mut store, true, CommitMark::Complete);
    let claim = claimed(store.claim_transcription(&id, "engine:a", None).unwrap());
    // A retry started meanwhile (a newer row): the claim's settle must not
    // land on it.
    let retry = store.begin_recognition(&id, "engine:b", None).unwrap();
    store
        .finish_attempt_transcript(&claim, &transcript("first"))
        .unwrap();
    let attempts = store.attempts_for(&id).unwrap();
    let by_id = |id: &str| attempts.iter().find(|a| a.id == id).unwrap().clone();
    assert_eq!(by_id(&claim).status, "completed");
    assert_eq!(by_id(&retry).status, "started");
    assert!(!store.transcription_wanted(&id).unwrap());
    store
        .finish_attempt_transcript(&retry, &transcript("second"))
        .unwrap();
    assert_eq!(store.attempts_for(&id).unwrap().len(), 2);
}

#[test]
fn a_deleted_take_has_nothing_left_to_transcribe() {
    let dir = TempDir::new().expect("tempdir");
    let mut store = store_in(&dir);
    let id = stored(&mut store, true, CommitMark::Complete);
    let attempt = claimed(store.claim_transcription(&id, "engine:test", None).unwrap());
    store.delete_capture(&id).expect("delete");
    assert!(!store.transcription_wanted(&id).unwrap());
    assert!(matches!(
        store.finish_attempt_transcript(&attempt, &transcript("late")),
        Err(StoreV2Error::NotFound(_))
    ));
    assert!(store.transcriptions_due(std::time::Duration::ZERO).unwrap().is_empty());
}

#[test]
fn a_take_waiting_to_be_transcribed_is_not_compressed() {
    let dir = TempDir::new().expect("tempdir");
    let mut store = store_in(&dir);
    let mut meta = TakeMeta::for_device("test-device");
    meta.transcribe = true;
    let mut take = store.begin_take(meta).expect("begin take");
    // Long enough for FLAC.
    take.append_and_seal(&ramp(32_000)).expect("append");
    let id = take.finish(&mut store).expect("commit").record.id;
    let candidates = |store: &StoreV2| {
        store
            .compression_candidates(usize::MAX)
            .unwrap()
            .into_iter()
            .map(|job| job.id)
            .collect::<Vec<_>>()
    };
    assert!(!candidates(&store).contains(&id));
    let attempt = claimed(store.claim_transcription(&id, "engine:test", None).unwrap());
    assert!(!candidates(&store).contains(&id), "in flight");
    store
        .finish_attempt_transcript(&attempt, &transcript("done"))
        .unwrap();
    assert!(candidates(&store).contains(&id), "settled: compressible again");
}

#[test]
fn an_import_can_ask_for_its_transcription_after_it_is_stored() {
    let dir = TempDir::new().expect("tempdir");
    let mut store = store_in(&dir);
    let id = stored(&mut store, false, CommitMark::Complete);
    store.request_transcription(&id).expect("request");
    store.request_transcription(&id).expect("asking twice is one intent");
    assert_eq!(store.transcriptions_due(std::time::Duration::ZERO).unwrap(), vec![id.clone()]);
    assert!(matches!(
        store.request_transcription("c_missing"),
        Err(StoreV2Error::NotFound(_))
    ));
}

#[test]
fn a_recheck_leaves_intents_younger_than_it_asks_for() {
    let dir = TempDir::new().expect("tempdir");
    let mut store = store_in(&dir);
    let id = stored(&mut store, true, CommitMark::Complete);
    assert!(store
        .transcriptions_due(std::time::Duration::from_secs(60))
        .unwrap()
        .is_empty());
    assert_eq!(
        store.transcriptions_due(std::time::Duration::ZERO).unwrap(),
        vec![id]
    );
}
