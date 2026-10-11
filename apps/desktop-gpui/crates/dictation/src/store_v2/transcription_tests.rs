//! Transcription-intent races (#220): two live claimants, and a store
//! cut off between noting its intent and writing its row. Claims, retries
//! and dead claimants are proven end to end in the host's suites.
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

/// #220: a crash between a take's promotion and its metadata commit (or
/// just before the promotion) loses neither the take nor its intent to
/// transcribe it — reconcile adopts it with the intent. A take whose
/// commit carried none gets none.
#[test]
fn a_commit_cut_off_by_a_crash_keeps_its_intent() {
    for promoted in [true, false] {
        for wanted in [true, false] {
            let dir = TempDir::new().expect("tempdir");
            let id = {
                let store = store_in(&dir);
                let mut meta = TakeMeta::for_device("test-device");
                meta.transcribe = wanted;
                let mut take = store.begin_take(meta).expect("begin take");
                take.append_and_seal(&ramp(400)).expect("append");
                let finalized = take.finalize().expect("finalize");
                // What commit_marked does up to the crash.
                if wanted {
                    store.note_pending_intent(&finalized.id).expect("note");
                }
                if promoted {
                    store.promote_from_staging(&finalized.id).expect("promote");
                }
                finalized.id
            };
            let mut store = store_in(&dir);
            let report = store.reconcile().expect("reconcile");
            assert!(store.get_capture(&id).unwrap().is_some(), "{report:?}");
            assert_eq!(
                store.transcription_wanted(&id).unwrap(),
                wanted,
                "promoted: {promoted}, wanted: {wanted}"
            );
            assert!(!store.pending_intent(&id).unwrap(), "the note is done with");
        }
    }
}

