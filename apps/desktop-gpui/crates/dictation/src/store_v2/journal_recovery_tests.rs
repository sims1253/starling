//! Startup recovery of the recorder's live-capture tree (#356): takes
//! whose app stopped mid-recording, or between stop and save, come back
//! as interrupted takes with their confirmed audio; live and freshly
//! finished journals are left to their owners; nothing is deleted.

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

#[test]
fn a_take_killed_mid_recording_comes_back_with_its_confirmed_audio() {
    let dir = TempDir::new().expect("tempdir");
    let mut store = store_in(&dir);
    let tree = journals(&dir);
    let confirmed = ramp(32_000, 0);
    let path = {
        let mut writer =
            JournalWriter::create_named(&tree, "j_killed".to_string(), 16_000).expect("writer");
        writer.append_frames(&confirmed).expect("append");
        writer.write_boundary().expect("boundary");
        // Written after the last boundary: never confirmed, and torn off
        // by the kill.
        writer.append_frames(&ramp(800, 7)).expect("tail");
        writer.path().to_path_buf()
        // The writer drops here without a trailer: the process died.
    };

    let report = store.recover_capture_journals(&tree).expect("scan");
    assert_eq!(report.recovered.len(), 1, "{report:?}");
    assert!(report.failed.is_empty() && report.deferred.is_empty(), "{report:?}");
    assert_eq!(report.recovered[0].samples, 32_000);
    assert!(!path.exists(), "the journal moved into the store");

    let record = store.get_capture("j_killed").expect("row").expect("present");
    assert_eq!(record.status, CaptureStatus::Interrupted);
    // Everything recovered is confirmed: the ack watermark is the whole
    // recovered take, never more.
    assert_eq!(record.frame_count, 32_000);
    assert_eq!(record.ack_sample_index, 32_000);
    let note = record.recovery_note().expect("note");
    assert!(note.contains("Recovered 2.0 s"), "{note}");
    assert!(note.contains("confirmed on disk"), "{note}");
    assert!(note.contains("discarded"), "the torn tail is named: {note}");

    // Playable/exportable: the stored audio is exactly the confirmed
    // samples.
    let audio = store.load_audio("j_killed").expect("audio");
    assert_eq!(audio.samples, confirmed);

    // Retryable: a recognition attempt starts and settles on it.
    store
        .begin_recognition("j_killed", "engine:test", None)
        .expect("retry begins");
    store
        .finish_recognition(
            "j_killed",
            RecognitionOutcome::Completed {
                text: "words",
                extra_json: None,
            },
        )
        .expect("retry finishes");

    // A second launch finds nothing left to recover.
    let again = store.recover_capture_journals(&tree).expect("rescan");
    assert!(again.recovered.is_empty(), "{again:?}");
}

#[test]
fn a_journal_still_being_written_is_left_to_its_writer() {
    let dir = TempDir::new().expect("tempdir");
    let mut store = store_in(&dir);
    let tree = journals(&dir);
    let mut writer =
        JournalWriter::create_named(&tree, "j_live".to_string(), 16_000).expect("writer");
    writer.append_frames(&ramp(1_600, 0)).expect("append");
    writer.write_boundary().expect("boundary");
    let path = writer.path().to_path_buf();
    // Even an old file is live while its writer holds it.
    age(&path, old());

    let report = store.recover_capture_journals(&tree).expect("scan");
    assert_eq!(report.deferred, vec!["j_live".to_string()]);
    assert!(report.recovered.is_empty());
    assert!(path.exists(), "a live take's journal is never touched");
    assert!(store.get_capture("j_live").expect("read").is_none());

    // Once the writer is gone, the next launch recovers it.
    drop(writer);
    let report = store.recover_capture_journals(&tree).expect("rescan");
    assert_eq!(report.recovered.len(), 1, "{report:?}");
}

#[test]
fn a_freshly_finished_journal_is_left_for_its_save_then_recovered_whole() {
    let dir = TempDir::new().expect("tempdir");
    let mut store = store_in(&dir);
    let tree = journals(&dir);
    let samples = ramp(4_800, 3);
    let path = {
        let mut writer =
            JournalWriter::create_named(&tree, "j_stopped".to_string(), 16_000).expect("writer");
        writer.append_frames(&samples).expect("append");
        writer.finalize().expect("finalize");
        writer.path().to_path_buf()
    };

    // Just stopped: a live instance may be saving it right now.
    let report = store.recover_capture_journals(&tree).expect("scan");
    assert_eq!(report.deferred, vec!["j_stopped".to_string()]);
    assert!(path.exists());

    // Still there a while later: its app stopped between stop and save.
    age(&path, old());
    let report = store.recover_capture_journals(&tree).expect("rescan");
    assert_eq!(report.recovered.len(), 1, "{report:?}");
    let record = store.get_capture("j_stopped").expect("row").expect("present");
    assert_eq!(record.status, CaptureStatus::Interrupted);
    let note = record.recovery_note().expect("note");
    assert!(note.contains("complete recording (0.3 s)"), "{note}");
    assert_eq!(store.load_audio("j_stopped").expect("audio").samples, samples);
    assert!(report.summary().contains("Recovered 1 recording"), "{}", report.summary());
}

#[test]
fn unreadable_and_empty_files_are_kept_and_other_entries_ignored() {
    let dir = TempDir::new().expect("tempdir");
    let mut store = store_in(&dir);
    let tree = journals(&dir);
    std::fs::write(tree.join("j_junk.sj"), b"not a journal at all").expect("junk");
    std::fs::write(tree.join("notes.txt"), b"hand-dropped").expect("stray");
    std::fs::create_dir_all(tree.join("deleted")).expect("legacy tombstones");
    std::fs::write(tree.join("deleted").join("j_gone.sj"), b"tombstoned").expect("gone");
    // Header only: the take died before its first boundary.
    let empty = {
        let writer =
            JournalWriter::create_named(&tree, "j_empty".to_string(), 16_000).expect("writer");
        writer.path().to_path_buf()
    };
    age(&empty, old());

    let report = store.recover_capture_journals(&tree).expect("scan");
    assert!(report.recovered.is_empty(), "{report:?}");
    assert_eq!(report.unrecognized.len(), 1, "{report:?}");
    assert!(!tree.join("j_junk.sj").exists());
    assert_eq!(
        std::fs::read(tree.join("j_junk.sj.unrecognized")).expect("kept aside"),
        b"not a journal at all"
    );
    assert!(empty.exists(), "an empty journal stays where it is");
    assert!(tree.join("notes.txt").exists());
    assert!(tree.join("deleted").join("j_gone.sj").exists());
    assert!(report.summary().contains(".unrecognized"), "{}", report.summary());

    // The aside file is not rescanned.
    let again = store.recover_capture_journals(&tree).expect("rescan");
    assert!(again.unrecognized.is_empty(), "{again:?}");
}

#[test]
fn a_superseded_journal_is_kept_but_never_recovered_twice() {
    let dir = TempDir::new().expect("tempdir");
    let mut store = store_in(&dir);
    let tree = journals(&dir);
    let path = {
        let mut writer =
            JournalWriter::create_named(&tree, "j_faulted".to_string(), 16_000).expect("writer");
        writer.append_frames(&ramp(1_600, 0)).expect("append");
        writer.write_boundary().expect("boundary");
        writer.path().to_path_buf()
    };
    age(&path, old());

    supersede_capture_journal(&path).expect("supersede");
    assert!(!path.exists());
    assert!(tree.join(SUPERSEDED_SUBDIR).join("j_faulted.sj").exists());
    // Superseding what is already gone is fine.
    supersede_capture_journal(&path).expect("idempotent");
    // A second journal of the same name never replaces the kept one.
    let again = {
        let mut writer =
            JournalWriter::create_named(&tree, "j_faulted".to_string(), 16_000).expect("writer");
        writer.append_frames(&ramp(800, 0)).expect("append");
        writer.write_boundary().expect("boundary");
        writer.path().to_path_buf()
    };
    let kept = std::fs::read(tree.join(SUPERSEDED_SUBDIR).join("j_faulted.sj")).expect("kept");
    supersede_capture_journal(&again).expect("supersede again");
    assert!(!again.exists());
    assert_eq!(
        std::fs::read(tree.join(SUPERSEDED_SUBDIR).join("j_faulted.sj")).expect("still kept"),
        kept
    );
    assert!(tree.join(SUPERSEDED_SUBDIR).join("j_faulted.1.sj").exists());

    let report = store.recover_capture_journals(&tree).expect("scan");
    assert!(report.recovered.is_empty(), "{report:?}");
}

#[test]
fn a_missing_journal_tree_is_nothing_to_recover() {
    let dir = TempDir::new().expect("tempdir");
    let mut store = store_in(&dir);
    let report = store
        .recover_capture_journals(&dir.path().join("no-such-tree"))
        .expect("scan");
    assert!(report.recovered.is_empty() && report.summary().is_empty());
}

#[test]
fn a_journal_is_locked_from_the_moment_its_name_appears() {
    // The scan must never find a live journal unlocked — not even in the
    // instant between creating the file and writing its header.
    let dir = TempDir::new().expect("tempdir");
    let tree = journals(&dir);
    let writer =
        JournalWriter::create_named(&tree, "j_new".to_string(), 16_000).expect("writer");
    let probe = File::open(writer.path()).expect("open");
    assert_eq!(try_flock_exclusive(&probe).expect("probe"), FlockEvidence::Held);
    let names: Vec<_> = std::fs::read_dir(&tree)
        .expect("tree")
        .flatten()
        .map(|entry| entry.file_name().to_string_lossy().to_string())
        .collect();
    assert_eq!(names, vec!["j_new.sj".to_string()], "no creation scratch left behind");
    // And a name already taken is never overwritten.
    drop(writer);
    assert!(JournalWriter::create_named(&tree, "j_new".to_string(), 16_000).is_err());
}

#[test]
fn the_retention_sweep_empties_superseded_journals() {
    let dir = TempDir::new().expect("tempdir");
    let mut store = store_in(&dir);
    let tree = journals(&dir);
    let path = {
        let mut writer =
            JournalWriter::create_named(&tree, "j_partial".to_string(), 16_000).expect("writer");
        writer.append_frames(&ramp(1_600, 0)).expect("append");
        writer.write_boundary().expect("boundary");
        writer.path().to_path_buf()
    };
    supersede_capture_journal(&path).expect("supersede");
    let report = store.sweep_retention().expect("sweep");
    assert_eq!(report.swept.len(), 1, "{report:?}");
    assert!(!tree.join(SUPERSEDED_SUBDIR).join("j_partial.sj").exists());
}

#[test]
fn a_second_look_considers_only_the_journals_it_is_asked_about() {
    // The delayed recheck must not make a candidate of a take recorded
    // since startup: that take is this launch's own to save.
    let dir = TempDir::new().expect("tempdir");
    let mut store = store_in(&dir);
    let tree = journals(&dir);
    for id in ["j_deferred", "j_recorded_since"] {
        let path = {
            let mut writer =
                JournalWriter::create_named(&tree, id.to_string(), 16_000).expect("writer");
            writer.append_frames(&ramp(1_600, 0)).expect("append");
            writer.write_boundary().expect("boundary");
            writer.path().to_path_buf()
        };
        age(&path, old());
    }
    let report = store
        .recover_capture_journals_where(&tree, |id| id == "j_deferred")
        .expect("scan");
    assert_eq!(report.recovered.len(), 1, "{report:?}");
    assert_eq!(report.recovered[0].id, "j_deferred");
    assert!(tree.join("j_recorded_since.sj").exists());
}

#[test]
fn an_unreadable_journal_folder_is_an_error_not_an_empty_one() {
    let dir = TempDir::new().expect("tempdir");
    let mut store = store_in(&dir);
    let not_a_dir = dir.path().join("journals-file");
    std::fs::write(&not_a_dir, b"in the way").expect("file");
    assert!(store.recover_capture_journals(&not_a_dir).is_err());
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
fn a_faulted_take_keeps_its_journal_locked_until_it_is_stored() {
    // The writer goes on a journal fault while the take records on in
    // memory: its lock must stay, or another instance's startup scan
    // adopts the partial journal as a second take.
    let dir = TempDir::new().expect("tempdir");
    let mut store = store_in(&dir);
    let tree = journals(&dir);
    let mut writer =
        JournalWriter::create_named(&tree, "j_faulted_live".to_string(), 16_000).expect("writer");
    writer.append_frames(&ramp(1_600, 0)).expect("append");
    writer.write_boundary().expect("boundary");
    let path = writer.path().to_path_buf();
    let liveness = writer.liveness();
    let saving = liveness.clone();
    drop(writer);
    age(&path, old());

    let report = store.recover_capture_journals(&tree).expect("scan");
    assert_eq!(report.deferred, vec!["j_faulted_live".to_string()], "{report:?}");
    assert!(path.exists());

    // One clone let go is not the take stored: every holder counts.
    drop(liveness);
    let report = store.recover_capture_journals(&tree).expect("rescan");
    assert_eq!(report.deferred, vec!["j_faulted_live".to_string()], "{report:?}");

    // Released by its save (or its process gone): recoverable again.
    saving.release();
    let report = store.recover_capture_journals(&tree).expect("rescan");
    assert_eq!(report.recovered.len(), 1, "{report:?}");
}

#[test]
fn a_journal_whose_take_was_stored_in_its_place_is_moved_aside_not_adopted() {
    // The app stored the take from memory and died before moving the
    // recorder's journal aside: the next launch finishes the move.
    let dir = TempDir::new().expect("tempdir");
    let mut store = store_in(&dir);
    let tree = journals(&dir);
    let path = {
        let mut writer =
            JournalWriter::create_named(&tree, "j_saved".to_string(), 16_000).expect("writer");
        writer.append_frames(&ramp(1_600, 0)).expect("append");
        writer.write_boundary().expect("boundary");
        writer.path().to_path_buf()
    };
    age(&path, old());
    let stored = store_in_place_of(&mut store, "j_saved", &ramp(4_800, 0));
    assert_eq!(
        store.journal_superseded_by("j_saved").expect("read"),
        Some(stored.clone())
    );

    let report = store.recover_capture_journals(&tree).expect("scan");
    assert!(report.recovered.is_empty(), "{report:?}");
    assert_eq!(report.superseded, vec!["j_saved".to_string()]);
    assert!(report.summary().is_empty(), "housekeeping: {}", report.summary());
    assert!(!path.exists());
    assert!(tree.join(SUPERSEDED_SUBDIR).join("j_saved.sj").exists());
    assert_eq!(capture_count(&store), 1, "one take, not two");

    // Deleting the take does not bring a leftover journal back either.
    let leftover = {
        let mut writer =
            JournalWriter::create_named(&tree, "j_saved".to_string(), 16_000).expect("writer");
        writer.append_frames(&ramp(800, 0)).expect("append");
        writer.write_boundary().expect("boundary");
        writer.path().to_path_buf()
    };
    age(&leftover, old());
    store.conn.execute("DELETE FROM captures", []).expect("delete rows");
    let report = store.recover_capture_journals(&tree).expect("rescan");
    assert!(report.recovered.is_empty(), "{report:?}");
    assert_eq!(capture_count(&store), 0);
}

#[test]
fn a_journal_already_adopted_is_moved_aside_when_its_old_name_comes_back() {
    // A power loss can undo the adoption's move out of the tree while the
    // committed row survives: the stored take already holds the samples.
    let dir = TempDir::new().expect("tempdir");
    let mut store = store_in(&dir);
    let tree = journals(&dir);
    let samples = ramp(3_200, 9);
    let path = {
        let mut writer =
            JournalWriter::create_named(&tree, "j_adopted".to_string(), 16_000).expect("writer");
        writer.append_frames(&samples).expect("append");
        writer.finalize().expect("finalize");
        writer.path().to_path_buf()
    };
    store.adopt_journal(&path, None).expect("adopt");
    std::fs::copy(store.audio_path("j_adopted"), &path).expect("name comes back");
    age(&path, old());

    let report = store.recover_capture_journals(&tree).expect("scan");
    assert!(report.recovered.is_empty() && report.failed.is_empty(), "{report:?}");
    assert_eq!(report.superseded, vec!["j_adopted".to_string()]);
    assert!(!path.exists());
    assert_eq!(capture_count(&store), 1);
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

#[test]
fn creation_scratch_no_writer_holds_is_swept() {
    let dir = TempDir::new().expect("tempdir");
    let mut store = store_in(&dir);
    let tree = journals(&dir);
    // A writer died between creating its scratch name and publishing it.
    let stale = tree.join("j_died.sj.creating");
    std::fs::write(&stale, b"").expect("stale scratch");
    // A writer is creating one right now: its lock is held.
    let busy = tree.join("j_busy.sj.creating");
    let holder = File::create(&busy).expect("busy scratch");
    assert_eq!(try_flock_exclusive(&holder).expect("lock"), FlockEvidence::Free);

    let report = store.recover_capture_journals(&tree).expect("scan");
    assert!(report.failed.is_empty(), "{report:?}");
    assert!(!stale.exists(), "the stale scratch name is gone");
    assert!(busy.exists(), "a held one is left to its writer");
    drop(holder);
}
