//! Journal-recovery races (#356) that need the store's own fault
//! injection (a recovery that dies at a chosen step) or a peer acting in
//! the middle of a sweep. Recovery itself is proven end to end in
//! `tests/store_recovery.rs` and the host's crash suite.
use super::*;
use tempfile::TempDir;

fn store_in(dir: &TempDir) -> StoreV2 {
    StoreV2::open(dir.path().join("v2")).expect("open v2 store")
}

fn journals(dir: &TempDir) -> PathBuf {
    let tree = dir.path().join("v2").join("journals");
    std::fs::create_dir_all(&tree).expect("journals tree");
    tree
}

fn ramp(len: usize, offset: u32) -> Vec<f32> {
    (0..len)
        .map(|i| ((offset as usize + i) % 997) as f32 * 0.0001)
        .collect()
}

/// Backdate a file so it reads as written `age` ago.
fn age(path: &Path, age: std::time::Duration) {
    let file = OpenOptions::new().write(true).open(path).expect("open");
    file.set_modified(std::time::SystemTime::now() - age)
        .expect("set mtime");
}

fn old() -> std::time::Duration {
    FINALIZED_ADOPTION_GRACE + std::time::Duration::from_secs(5)
}

fn seal_count(store: &StoreV2) -> i64 {
    store
        .conn
        .query_row("SELECT COUNT(*) FROM recovery_seals", [], |row| row.get(0))
        .expect("count seals")
}

/// #220/#356: a recovery that dies at any step of adopting a cut-short
/// journal — after noting the seal, after sealing the journal (it now
/// looks finished), after moving it into `audio/` (before its row) —
/// leaves a take the next start brings back as interrupted, holding
/// exactly its confirmed audio, with no transcription intent and a note
/// that never says it finished; the seal is gone once the row commits.
#[test]
fn a_cut_journal_whose_recovery_dies_at_any_step_still_reads_as_cut() {
    for torn_tail in [false, true] {
        for step in [RecoveryStep::SealNoted, RecoveryStep::Sealed, RecoveryStep::Moved] {
            let case = format!("torn tail {torn_tail}, died after {step:?}");
            let dir = TempDir::new().expect("tempdir");
            let tree = journals(&dir);
            let confirmed = ramp(4_800, 1);
            {
                let mut writer = JournalWriter::create_named(&tree, "j_cut".to_string(), 16_000)
                    .expect("writer");
                writer.append_frames(&confirmed).expect("append");
                writer.write_boundary().expect("boundary");
                if torn_tail {
                    writer.append_frames(&ramp(800, 9)).expect("tail");
                }
            }
            {
                let mut store = store_in(&dir);
                store.crash_after = Some(step);
                let report = store
                    .recover_capture_journals_where(&tree, |_| true, true)
                    .expect("scan");
                assert_eq!(report.failed.len(), 1, "{case}: {report:?}");
                assert!(store.get_capture("j_cut").unwrap().is_none(), "{case}");
                assert_eq!(seal_count(&store), 1, "{case}: noted before anything else");
            }

            // The next start: reconcile, then the journal scan (no grace
            // wait: a journal recovery sealed has no recorder left).
            let mut store = store_in(&dir);
            store.reconcile().expect("reconcile");
            store
                .recover_capture_journals_where(&tree, |_| true, true)
                .expect("rescan");
            assert_eq!(capture_count(&store), 1, "{case}");
            let record = store.get_capture("j_cut").unwrap().expect("row");
            assert_eq!(record.status, CaptureStatus::Interrupted, "{case}");
            assert_eq!(record.frame_count, 4_800, "{case}: the confirmed extent");
            let audio = store.load_audio("j_cut").expect("audio");
            assert_eq!(audio.samples, confirmed, "{case}");
            assert!(audio.finalized && audio.torn_tail_bytes == 0, "{case}: one valid trailer");
            assert!(!store.transcription_wanted("j_cut").unwrap(), "{case}: waits for the user");
            let note = record.recovery_note().expect("note");
            assert!(
                !note.contains("finished cleanly") && !note.contains("after this take stopped"),
                "{case}: {note}"
            );
            if torn_tail {
                assert!(
                    note.contains("discarded") || note.contains("torn tail"),
                    "{case}: the torn tail is named: {note}"
                );
            }
            assert_eq!(seal_count(&store), 0, "{case}: the seal is done with");
            assert!(!tree.join("j_cut.sj").exists(), "{case}");
        }
    }
}

/// The control for the crash steps: a journal its recorder finalized is
/// never noted as sealed, and one whose adoption dies after its move
/// still comes back with the intent to transcribe it.
#[test]
fn a_complete_journal_whose_adoption_dies_after_its_move_keeps_its_intent() {
    let dir = TempDir::new().expect("tempdir");
    let tree = journals(&dir);
    let path = {
        let mut writer =
            JournalWriter::create_named(&tree, "j_stopped".to_string(), 16_000).expect("writer");
        writer.append_frames(&ramp(4_800, 3)).expect("append");
        writer.finalize().expect("finalize");
        writer.path().to_path_buf()
    };
    age(&path, old());
    {
        let mut store = store_in(&dir);
        store.crash_after = Some(RecoveryStep::Moved);
        let report = store
            .recover_capture_journals_where(&tree, |_| true, true)
            .expect("scan");
        assert_eq!(report.failed.len(), 1, "{report:?}");
        assert_eq!(seal_count(&store), 0, "a complete journal is not sealed by recovery");
    }
    let mut store = store_in(&dir);
    store.reconcile().expect("reconcile");
    let record = store.get_capture("j_stopped").unwrap().expect("row");
    assert_eq!(record.frame_count, 4_800);
    assert!(store.transcription_wanted("j_stopped").unwrap());
}

/// Store a take from samples in place of recorder journal `journal_id`,
/// the way the app's and the runtime's save-from-memory paths do.
fn store_in_place_of(store: &mut StoreV2, journal_id: &str, samples: &[f32]) -> String {
    let mut meta = TakeMeta::for_device("");
    meta.supersedes_journal = Some(journal_id.to_string());
    let mut take = store.begin_take_at_rate(16_000, meta).expect("begin");
    take.append_and_seal(samples).expect("append");
    take.finalize()
        .expect("finalize")
        .commit_marked(store, CommitMark::Complete)
        .expect("commit")
        .record
        .id
}

fn capture_count(store: &StoreV2) -> i64 {
    store
        .conn
        .query_row("SELECT COUNT(*) FROM captures", [], |row| row.get(0))
        .expect("count")
}

#[test]
fn an_adoption_left_in_audio_without_a_row_is_not_a_second_take() {
    // Adoption moved the journal into audio/ and its commit failed; the
    // take was then stored from memory in its place. Reconcile must not
    // turn the moved journal into an orphan session beside it.
    let dir = TempDir::new().expect("tempdir");
    let mut store = store_in(&dir);
    let tree = journals(&dir);
    let audio = dir.path().join("v2").join(AUDIO_DIR);
    std::fs::create_dir_all(&audio).expect("audio dir");
    {
        let mut writer =
            JournalWriter::create_named(&audio, "j_half_adopted".to_string(), 16_000)
                .expect("writer");
        writer.append_frames(&ramp(1_600, 0)).expect("append");
        writer.finalize().expect("finalize");
    }
    store_in_place_of(&mut store, "j_half_adopted", &ramp(4_800, 0));

    let report = store.reconcile().expect("reconcile");
    assert!(report.orphan_sessions.is_empty(), "{report:?}");
    assert_eq!(report.superseded_journals, vec!["j_half_adopted".to_string()]);
    assert!(!audio.join("j_half_adopted.sj").exists());
    assert!(tree.join(SUPERSEDED_SUBDIR).join("j_half_adopted.sj").exists());
    assert_eq!(capture_count(&store), 1);
}

// --- #356 review round 6: a journal is moved aside only once a stored
// copy is read back holding every confirmed sample of it ---

// --- #356 review round 7 ---

#[test]
fn sweeping_a_superseded_copy_never_deletes_the_live_take_of_that_name() {
    // A power loss brings back the source name of an adopted journal;
    // recovery proves the stored take holds it and moves it aside. The
    // sweep that removes that copy must not leave a stamp reconcile reads
    // as the take's own deletion.
    let dir = TempDir::new().expect("tempdir");
    let mut store = store_in(&dir);
    let tree = journals(&dir);
    let samples = ramp(4_800, 5);
    let path = {
        let mut writer =
            JournalWriter::create_named(&tree, "j_live_take".to_string(), 16_000).expect("writer");
        writer.append_frames(&samples).expect("append");
        writer.finalize().expect("finalize");
        writer.path().to_path_buf()
    };
    store.adopt_journal(&path, None).expect("adopt");
    std::fs::copy(store.audio_path("j_live_take"), &path).expect("name comes back");
    age(&path, old());
    let report = store.recover_capture_journals(&tree).expect("scan");
    assert_eq!(report.superseded, vec!["j_live_take".to_string()], "{report:?}");

    assert_eq!(store.sweep_retention().expect("sweep").swept.len(), 1);
    let report = store.reconcile().expect("reconcile");
    assert!(report.completed_deletes.is_empty(), "{report:?}");
    store.sweep_retention().expect("sweep again");
    assert_eq!(store.load_audio("j_live_take").expect("still playable").samples, samples);
}

// --- #342 sweep in upkeep, review round 1 ---

#[test]
fn a_peer_compression_landing_before_the_sweep_lock_keeps_the_copy() {
    // The proof runs under the write lock the removal holds: a peer that
    // compresses the 48 kHz take after the sweep listed the tree, and
    // before it locks, leaves a copy no longer proven — kept.
    let dir = TempDir::new().expect("tempdir");
    let mut store = store_in(&dir);
    let tree = journals(&dir);
    let samples = ramp(4_800, 4);
    let path = {
        let mut writer =
            JournalWriter::create_named(&tree, "j_48k".to_string(), 48_000).expect("writer");
        writer.append_frames(&samples).expect("append");
        writer.finalize().expect("finalize");
        writer.path().to_path_buf()
    };
    store.adopt_journal(&path, None).expect("adopt");
    std::fs::copy(store.audio_path("j_48k"), &path).expect("name comes back");
    age(&path, old());
    let report = store.recover_capture_journals(&tree).expect("scan");
    assert_eq!(report.superseded, vec!["j_48k".to_string()], "{report:?}");

    let mut peer = store_in(&dir);
    store.before_sweep_lock = Some(TestHook(Box::new(move || {
        peer.compress_audio("j_48k").expect("peer compresses");
    })));
    let report = store.sweep_retention().expect("sweep");
    assert!(report.swept.is_empty(), "{report:?}");
    assert_eq!(report.retained.len(), 1, "{report:?}");
    assert_eq!(store.load_audio("j_48k").expect("audio").sample_rate, 16_000);
    let kept = tree.join(SUPERSEDED_SUBDIR).join("j_48k.sj");
    assert_eq!(read_audio_journal(&kept).expect("kept whole").samples, samples);
}

#[test]
fn a_take_starting_during_the_proof_stops_the_sweep() {
    let dir = TempDir::new().expect("tempdir");
    let mut store = store_in(&dir);
    let tree = journals(&dir);
    let samples = ramp(4_800, 5);
    let path = {
        let mut writer =
            JournalWriter::create_named(&tree, "j_proof".to_string(), 16_000).expect("writer");
        writer.append_frames(&samples).expect("append");
        writer.finalize().expect("finalize");
        writer.path().to_path_buf()
    };
    store.adopt_journal(&path, None).expect("adopt");
    std::fs::copy(store.audio_path("j_proof"), &path).expect("name comes back");
    age(&path, old());
    let report = store.recover_capture_journals(&tree).expect("scan");
    assert_eq!(report.superseded, vec!["j_proof".to_string()], "{report:?}");

    // Asked once before the proof (not yet), once after it (recording).
    let asked = std::cell::Cell::new(0);
    let report = store
        .sweep_retention_until(|| {
            asked.set(asked.get() + 1);
            asked.get() > 1
        })
        .expect("sweep");
    assert_eq!(asked.get(), 2);
    assert!(report.stopped && report.swept.is_empty(), "{report:?}");
    assert!(tree.join(SUPERSEDED_SUBDIR).join("j_proof.sj").exists());
    // Proven, so the next sweep removes it.
    assert_eq!(store.sweep_retention().expect("sweep").swept.len(), 1);
}
