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

/// #220: a take that stopped but whose process died before storing it is
/// recovered with the intent to transcribe it — when the caller asks for
/// that — and a take cut short by the crash never is.
#[test]
fn a_recovered_complete_take_is_due_for_transcription_and_a_cut_one_is_not() {
    let dir = TempDir::new().expect("tempdir");
    let mut store = store_in(&dir);
    let tree = journals(&dir);
    let stopped = {
        let mut writer =
            JournalWriter::create_named(&tree, "j_stopped".to_string(), 16_000).expect("writer");
        writer.append_frames(&ramp(4_800, 3)).expect("append");
        writer.finalize().expect("finalize");
        writer.path().to_path_buf()
    };
    age(&stopped, old());
    {
        let mut writer =
            JournalWriter::create_named(&tree, "j_killed".to_string(), 16_000).expect("writer");
        writer.append_frames(&ramp(4_800, 5)).expect("append");
        writer.write_boundary().expect("boundary");
    }
    let report = store
        .recover_capture_journals_where(&tree, |_| true, true)
        .expect("scan");
    assert_eq!(report.recovered.len(), 2, "{report:?}");
    assert!(store.transcription_wanted("j_stopped").unwrap());
    assert!(!store.transcription_wanted("j_killed").unwrap());

    // Without the ask, neither is.
    let dir = TempDir::new().expect("tempdir");
    let mut store = store_in(&dir);
    let tree = journals(&dir);
    let stopped = {
        let mut writer =
            JournalWriter::create_named(&tree, "j_stopped".to_string(), 16_000).expect("writer");
        writer.append_frames(&ramp(4_800, 3)).expect("append");
        writer.finalize().expect("finalize");
        writer.path().to_path_buf()
    };
    age(&stopped, old());
    store.recover_capture_journals(&tree).expect("scan");
    assert!(!store.transcription_wanted("j_stopped").unwrap());
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
    let stored = store_in_place_of(&mut store, "j_faulted", &ramp(4_800, 0));
    let stored = store.audio_journal_path(&stored).expect("stored audio");

    assert!(supersede_journal_held_by(&path, &stored).expect("supersede"));
    assert!(!path.exists());
    assert!(tree.join(SUPERSEDED_SUBDIR).join("j_faulted.sj").exists());
    // Superseding what is already gone is fine.
    assert!(!supersede_journal_held_by(&path, &stored).expect("idempotent"));
    // A second journal of the same name never replaces the kept one.
    let again = {
        let mut writer =
            JournalWriter::create_named(&tree, "j_faulted".to_string(), 16_000).expect("writer");
        writer.append_frames(&ramp(800, 0)).expect("append");
        writer.write_boundary().expect("boundary");
        writer.path().to_path_buf()
    };
    let kept = std::fs::read(tree.join(SUPERSEDED_SUBDIR).join("j_faulted.sj")).expect("kept");
    assert!(supersede_journal_held_by(&again, &stored).expect("supersede again"));
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
    let stored = store_in_place_of(&mut store, "j_partial", &ramp(1_600, 0));
    let stored = store.audio_journal_path(&stored).expect("stored audio");
    assert!(supersede_journal_held_by(&path, &stored).expect("supersede"));
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
        .recover_capture_journals_where(&tree, |id| id == "j_deferred", false)
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

    // Deleting the take does not bring a leftover journal back either:
    // the deleted take's quarantined audio still proves it a copy.
    let leftover = {
        let mut writer =
            JournalWriter::create_named(&tree, "j_saved".to_string(), 16_000).expect("writer");
        writer.append_frames(&ramp(800, 0)).expect("append");
        writer.write_boundary().expect("boundary");
        writer.path().to_path_buf()
    };
    age(&leftover, old());
    store.delete_capture(&stored).expect("delete");
    let report = store.recover_capture_journals(&tree).expect("rescan");
    assert!(report.recovered.is_empty(), "{report:?}");
    assert_eq!(report.superseded, vec!["j_saved".to_string()]);
    assert!(tree.join(SUPERSEDED_SUBDIR).join("j_saved.1.sj").exists());
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

    // Deleted since: a copy of its journal never brings it back.
    store.delete_capture("j_adopted").expect("delete");
    std::fs::copy(
        tree.join(SUPERSEDED_SUBDIR).join("j_adopted.sj"),
        &path,
    )
    .expect("name comes back again");
    age(&path, old());
    store.reconcile().expect("reconcile");
    let report = store.recover_capture_journals(&tree).expect("rescan");
    assert!(report.recovered.is_empty(), "{report:?}");
    assert_eq!(report.deleted, vec!["j_adopted".to_string()]);
    assert_eq!(capture_count(&store), 0);
}

/// A replacement save in progress: its staging journal holds `samples`
/// when the process dies, before its commit.
fn die_while_storing_in_place_of(store: &StoreV2, journal_id: &str, samples: &[f32]) -> String {
    let mut meta = TakeMeta::for_device("");
    meta.supersedes_journal = Some(journal_id.to_string());
    let mut take = store.begin_take_at_rate(16_000, meta).expect("begin");
    take.append_and_seal(samples).expect("append");
    take.id().to_string()
}

fn faulted_journal(tree: &Path, id: &str, samples: &[f32]) -> PathBuf {
    let mut writer = JournalWriter::create_named(tree, id.to_string(), 16_000).expect("writer");
    writer.append_frames(samples).expect("append");
    writer.write_boundary().expect("boundary");
    let path = writer.path().to_path_buf();
    drop(writer);
    age(&path, old());
    path
}

#[test]
fn a_replacement_cut_short_before_its_commit_still_replaces_a_shorter_journal() {
    // The app died while storing the take from memory: reconcile turns
    // the replacement's staging journal into the take, and the recorder's
    // partial journal is not adopted beside it.
    let dir = TempDir::new().expect("tempdir");
    let store = store_in(&dir);
    let tree = journals(&dir);
    let path = faulted_journal(&tree, "j_mid_save", &ramp(1_600, 0));
    let staged = die_while_storing_in_place_of(&store, "j_mid_save", &ramp(4_800, 0));
    drop(store);

    let mut store = store_in(&dir);
    store.reconcile().expect("reconcile");
    assert!(store.get_capture(&staged).expect("read").is_some(), "the replacement is the take");
    let report = store.recover_capture_journals(&tree).expect("scan");
    assert!(report.recovered.is_empty(), "{report:?}");
    assert_eq!(report.superseded, vec!["j_mid_save".to_string()]);
    assert!(!path.exists());
    assert_eq!(capture_count(&store), 1, "one take, not two");
}

#[test]
fn a_replacement_holding_less_than_the_journal_replaces_nothing() {
    // Cut short with less audio than the recorder confirmed: both stay
    // takes — a duplicate the user can delete beats audio lost.
    let dir = TempDir::new().expect("tempdir");
    let store = store_in(&dir);
    let tree = journals(&dir);
    faulted_journal(&tree, "j_longer", &ramp(4_800, 0));
    die_while_storing_in_place_of(&store, "j_longer", &ramp(800, 0));
    drop(store);

    let mut store = store_in(&dir);
    store.reconcile().expect("reconcile");
    let report = store.recover_capture_journals(&tree).expect("scan");
    assert_eq!(report.recovered.len(), 1, "{report:?}");
    assert_eq!(store.load_audio("j_longer").expect("audio").samples.len(), 4_800);
    assert_eq!(capture_count(&store), 2);
}

#[test]
fn a_replacement_that_never_stored_replaces_nothing() {
    // The save from memory failed and rolled back: the journal is the
    // take's only copy and is recovered.
    let dir = TempDir::new().expect("tempdir");
    let mut store = store_in(&dir);
    let tree = journals(&dir);
    faulted_journal(&tree, "j_only_copy", &ramp(1_600, 0));
    let staged = die_while_storing_in_place_of(&store, "j_only_copy", &ramp(4_800, 0));
    assert!(store.discard_staging(&staged).expect("roll back"));
    assert_eq!(store.journal_superseded_by("j_only_copy").expect("read"), None);

    let report = store.recover_capture_journals(&tree).expect("scan");
    assert_eq!(report.recovered.len(), 1, "{report:?}");
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
    age(&stale, old());
    // A writer is creating one right now: its lock is held.
    let busy = tree.join("j_busy.sj.creating");
    let holder = File::create(&busy).expect("busy scratch");
    assert_eq!(try_flock_exclusive(&holder).expect("lock"), FlockEvidence::Free);
    age(&busy, old());
    // Just created, its creator about to lock it: not touched either.
    let fresh = tree.join("j_fresh.sj.creating");
    std::fs::write(&fresh, b"").expect("fresh scratch");

    let report = store.recover_capture_journals(&tree).expect("scan");
    assert!(report.failed.is_empty(), "{report:?}");
    assert!(!stale.exists(), "the stale scratch name is gone");
    assert!(busy.exists(), "a held one is left to its writer");
    assert!(fresh.exists(), "a fresh one is left to its creator");
    drop(holder);
}

// --- #356 review round 6: a journal is moved aside only once a stored
// copy is read back holding every confirmed sample of it ---

/// The single take stored in place of `journal_id`, with its stored audio
/// file, after the journal was left in the tree.
fn replaced(store: &mut StoreV2, journal_id: &str, samples: &[f32]) -> (String, PathBuf) {
    let id = store_in_place_of(store, journal_id, samples);
    let path = store.audio_journal_path(&id).expect("stored audio");
    (id, path)
}

/// Recovery adopted the journal whole: nothing it confirmed is lost.
fn adopted_whole(store: &StoreV2, report: &JournalRecovery, id: &str, samples: &[f32]) {
    assert!(report.superseded.is_empty() && report.deleted.is_empty(), "{report:?}");
    assert_eq!(report.recovered.len(), 1, "{report:?}");
    assert_eq!(report.recovered[0].id, id);
    assert_eq!(store.load_audio(id).expect("audio").samples, samples);
}

#[test]
fn a_replacement_whose_audio_is_missing_supersedes_nothing() {
    let dir = TempDir::new().expect("tempdir");
    let mut store = store_in(&dir);
    let tree = journals(&dir);
    let confirmed = ramp(1_600, 0);
    let journal = faulted_journal(&tree, "j_missing", &confirmed);
    let (_, stored) = replaced(&mut store, "j_missing", &ramp(4_800, 0));
    std::fs::remove_file(&stored).expect("the replacement's audio goes");

    // Neither the save's own move nor recovery treats the row as proof.
    assert!(!supersede_journal_held_by(&journal, &stored).expect("check"));
    assert!(journal.exists());
    let report = store.recover_capture_journals(&tree).expect("scan");
    adopted_whole(&store, &report, "j_missing", &confirmed);
    assert!(!tree.join(SUPERSEDED_SUBDIR).exists());
}

#[test]
fn a_damaged_replacement_supersedes_nothing() {
    let dir = TempDir::new().expect("tempdir");
    let mut store = store_in(&dir);
    let tree = journals(&dir);
    let confirmed = ramp(1_600, 0);
    let journal = faulted_journal(&tree, "j_damaged", &confirmed);
    let (_, stored) = replaced(&mut store, "j_damaged", &ramp(4_800, 0));
    std::fs::write(&stored, b"no longer a journal").expect("damage");

    assert!(!supersede_journal_held_by(&journal, &stored).expect("check"));
    let report = store.recover_capture_journals(&tree).expect("scan");
    adopted_whole(&store, &report, "j_damaged", &confirmed);
}

#[test]
fn a_replacement_whose_verified_audio_is_shorter_than_its_row_supersedes_nothing() {
    let dir = TempDir::new().expect("tempdir");
    let mut store = store_in(&dir);
    let tree = journals(&dir);
    let confirmed = ramp(1_600, 0);
    let journal = faulted_journal(&tree, "j_cut", &confirmed);
    let (id, stored) = replaced(&mut store, "j_cut", &ramp(4_800, 0));
    // The file loses its end: the row still claims 4,800 samples.
    let file = OpenOptions::new().write(true).open(&stored).expect("open");
    file.set_len(std::fs::metadata(&stored).expect("size").len() * 3 / 10)
        .expect("truncate");
    drop(file);
    assert_eq!(store.get_capture(&id).expect("row").expect("present").frame_count, 4_800);
    let verified = read_audio_journal(&stored).map_or(0, |audio| audio.samples.len());
    assert!(verified < confirmed.len(), "{verified}");

    assert!(!supersede_journal_held_by(&journal, &stored).expect("check"));
    let report = store.recover_capture_journals(&tree).expect("scan");
    adopted_whole(&store, &report, "j_cut", &confirmed);
}

#[test]
fn deleting_a_shorter_replacement_does_not_suppress_the_longer_journal() {
    let dir = TempDir::new().expect("tempdir");
    let mut store = store_in(&dir);
    let tree = journals(&dir);
    let confirmed = ramp(4_800, 0);
    faulted_journal(&tree, "j_longer_than_deleted", &confirmed);
    let (id, _) = replaced(&mut store, "j_longer_than_deleted", &ramp(800, 0));
    store.delete_capture(&id).expect("delete the replacement");

    let report = store.recover_capture_journals(&tree).expect("scan");
    adopted_whole(&store, &report, "j_longer_than_deleted", &confirmed);
    assert_eq!(capture_count(&store), 1);
}

#[test]
fn equal_rate_replacement_a_tenth_of_a_second_short_supersedes_nothing() {
    // No duration slack: 1,440 samples do not hold 1,600 at 16 kHz.
    let dir = TempDir::new().expect("tempdir");
    let mut store = store_in(&dir);
    let tree = journals(&dir);
    let confirmed = ramp(1_600, 0);
    let journal = faulted_journal(&tree, "j_slack", &confirmed);
    let (_, stored) = replaced(&mut store, "j_slack", &ramp(1_440, 0));

    assert!(!supersede_journal_held_by(&journal, &stored).expect("check"));
    let report = store.recover_capture_journals(&tree).expect("scan");
    adopted_whole(&store, &report, "j_slack", &confirmed);
    assert_eq!(capture_count(&store), 2, "a duplicate, not a loss");
}

#[test]
fn a_replacement_that_differs_in_one_sample_or_rate_supersedes_nothing() {
    let dir = TempDir::new().expect("tempdir");
    let mut store = store_in(&dir);
    let tree = journals(&dir);
    let confirmed = ramp(1_600, 0);
    let journal = faulted_journal(&tree, "j_differs", &confirmed);
    let mut other = ramp(4_800, 0);
    other[1_000] += 0.25;
    let (_, stored) = replaced(&mut store, "j_differs", &other);
    assert!(!supersede_journal_held_by(&journal, &stored).expect("check"));

    // The same samples at another rate are not proven the same audio.
    let mut meta = TakeMeta::for_device("");
    meta.supersedes_journal = Some("j_differs".to_string());
    let mut take = store.begin_take_at_rate(48_000, meta).expect("begin");
    take.append_and_seal(&ramp(4_800, 0)).expect("append");
    let id = take
        .finalize()
        .expect("finalize")
        .commit_marked(&mut store, CommitMark::Complete)
        .expect("commit")
        .record
        .id;
    let stored = store.audio_journal_path(&id).expect("stored audio");
    assert!(!supersede_journal_held_by(&journal, &stored).expect("check"));
    let report = store.recover_capture_journals(&tree).expect("scan");
    adopted_whole(&store, &report, "j_differs", &confirmed);
}

#[test]
fn a_replacement_saved_through_wav_or_compressed_to_flac_holds_its_journal() {
    // The app stores a faulted take from its 16 kHz PCM16 WAV, and upkeep
    // later compresses it: both are the journal's audio at rest.
    let dir = TempDir::new().expect("tempdir");
    let mut store = store_in(&dir);
    let tree = journals(&dir);
    let take: Vec<f32> = (0..4_800).map(|i| ((i * 37) % 2_001) as f32 / 1_000.0 - 1.0).collect();
    let journal = faulted_journal(&tree, "j_wav", &take[..1_600]);
    let wav = crate::audio::encode_wav_16k(&crate::audio::PcmAudio {
        samples: take.clone(),
        sample_rate: 16_000,
        channels: 1,
    })
    .expect("wav");
    let pcm = decode_pcm16_wav(&wav).expect("decode");
    let (id, stored) = replaced(&mut store, "j_wav", &pcm.samples);
    assert!(store.compress_audio(&id).is_ok());
    let stored_now = store.audio_journal_path(&id).expect("stored audio");
    assert_ne!(stored_now, stored, "compressed to FLAC");

    assert!(supersede_journal_held_by(&journal, &stored_now).expect("supersede"));
    assert!(tree.join(SUPERSEDED_SUBDIR).join("j_wav.sj").exists());
}

#[test]
fn a_journal_waits_for_its_replacement_when_reconcile_has_not_committed_it() {
    // The app died mid-save; this launch's reconcile failed before it
    // committed the replacement. Recovery must not adopt the journal now
    // — once reconcile succeeds the replacement would be a second take,
    // and the adopted journal's own row would hide the proof forever.
    let dir = TempDir::new().expect("tempdir");
    let store = store_in(&dir);
    let tree = journals(&dir);
    let journal = faulted_journal(&tree, "j_waits", &ramp(1_600, 0));
    let staged = die_while_storing_in_place_of(&store, "j_waits", &ramp(4_800, 0));
    drop(store);

    let mut store = store_in(&dir);
    // No reconcile: it failed this launch.
    let report = store.recover_capture_journals(&tree).expect("scan");
    assert_eq!(report.deferred, vec!["j_waits".to_string()], "{report:?}");
    assert!(journal.exists() && capture_count(&store) == 0);

    // The next launch's reconcile succeeds.
    store.reconcile().expect("reconcile");
    assert!(store.get_capture(&staged).expect("read").is_some());
    let report = store.recover_capture_journals(&tree).expect("rescan");
    assert_eq!(report.superseded, vec!["j_waits".to_string()], "{report:?}");
    assert_eq!(capture_count(&store), 1, "one take, not two");

    // A pending replacement holding less than the journal is no reason
    // to wait: the journal is adopted at once, and both stay.
    let confirmed = ramp(4_800, 0);
    faulted_journal(&tree, "j_no_wait", &confirmed);
    die_while_storing_in_place_of(&store, "j_no_wait", &ramp(800, 0));
    let report = store.recover_capture_journals(&tree).expect("scan");
    adopted_whole(&store, &report, "j_no_wait", &confirmed);
    store.reconcile().expect("reconcile");
    assert_eq!(capture_count(&store), 3, "a duplicate where nothing proves it one");
}

#[test]
fn reconcile_commits_a_rowless_replacement_before_judging_its_journal() {
    // Both a half-adopted recorder journal and its replacement sit in
    // audio/ without rows: whichever reconcile meets first, the journal
    // is judged against the committed replacement, never adopted beside
    // it.
    let dir = TempDir::new().expect("tempdir");
    let mut store = store_in(&dir);
    let tree = journals(&dir);
    let audio = dir.path().join("v2").join(AUDIO_DIR);
    std::fs::create_dir_all(&audio).expect("audio dir");
    // Sorts after every minted id and before: try both orders.
    for journal_id in ["0_half_adopted", "z_half_adopted"] {
        {
            let mut writer = JournalWriter::create_named(&audio, journal_id.to_string(), 16_000)
                .expect("writer");
            writer.append_frames(&ramp(1_600, 0)).expect("append");
            writer.finalize().expect("finalize");
        }
        let staged = die_while_storing_in_place_of(&store, journal_id, &ramp(4_800, 0));
        store.promote_from_staging(&staged).expect("promoted, never committed");

        let report = store.reconcile().expect("reconcile");
        assert_eq!(report.orphan_sessions, vec![staged.clone()], "{report:?}");
        assert_eq!(report.superseded_journals, vec![journal_id.to_string()]);
        assert!(tree.join(SUPERSEDED_SUBDIR).join(format!("{journal_id}.sj")).exists());
    }
    assert_eq!(capture_count(&store), 2);
}

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

#[test]
fn a_swept_copy_is_no_reason_to_set_a_returning_journal_aside_unproven() {
    // The superseded copy is swept; later the journal's name comes back
    // while its replacement's audio is gone. Nothing proves it a copy any
    // more: it is adopted, not filed as a deleted take.
    let dir = TempDir::new().expect("tempdir");
    let mut store = store_in(&dir);
    let tree = journals(&dir);
    let confirmed = ramp(1_600, 0);
    let journal = faulted_journal(&tree, "j_returns", &confirmed);
    let (_, stored) = replaced(&mut store, "j_returns", &ramp(4_800, 0));
    let kept = std::fs::read(&journal).expect("journal bytes");
    assert!(supersede_journal_held_by(&journal, &stored).expect("supersede"));
    store.sweep_retention().expect("sweep");

    std::fs::write(&journal, kept).expect("name comes back");
    age(&journal, old());
    std::fs::remove_file(&stored).expect("the replacement's audio goes");
    let report = store.recover_capture_journals(&tree).expect("scan");
    adopted_whole(&store, &report, "j_returns", &confirmed);
}

#[test]
fn a_replacement_mixing_both_quantizations_is_no_proof() {
    // Each stored sample matches the journal's through one path or the
    // other, but the take as a whole matches neither: not the same audio.
    let dir = TempDir::new().expect("tempdir");
    let mut store = store_in(&dir);
    let tree = journals(&dir);
    let at = |q: i16| f32::from(q) / 32_767.0;
    assert_eq!(crate::audio::pcm16(at(20_000)), 20_000);
    let confirmed: Vec<f32> = (0..1_600).map(|i| at(if i % 2 == 0 { 20_000 } else { 21_000 })).collect();
    let journal = faulted_journal(&tree, "j_mixed", &confirmed);
    // Through the WAV round trip: 20000 → 19999 and 21000 → 20999.
    let round_trip = |q: i16| crate::audio::pcm16(f32::from(q) / 32_768.0);
    assert_eq!((round_trip(20_000), round_trip(21_000)), (19_999, 20_999));
    let mixed: Vec<f32> = (0..4_800).map(|i| at(if i % 2 == 0 { 19_999 } else { 21_000 })).collect();
    let (_, stored) = replaced(&mut store, "j_mixed", &mixed);
    assert!(!supersede_journal_held_by(&journal, &stored).expect("check"));

    // Either path whole is the journal's audio.
    for path in [|q: i16| q, |q: i16| crate::audio::pcm16(f32::from(q) / 32_768.0)] {
        let whole: Vec<f32> = (0..4_800)
            .map(|i| at(path(if i % 2 == 0 { 20_000 } else { 21_000 })))
            .collect();
        let (_, stored) = replaced(&mut store, "j_mixed", &whole);
        let audio = read_audio_journal(&stored).expect("read");
        assert!(holds_journal(&audio, &confirmed, 16_000));
    }
    // The journal names the last replacement committed: a whole one.
    let report = store.recover_capture_journals(&tree).expect("scan");
    assert_eq!(report.superseded, vec!["j_mixed".to_string()], "{report:?}");
}

#[test]
fn a_journal_whose_namesake_take_lost_its_audio_comes_back_under_a_fresh_name() {
    // An adopted take's audio is gone and its journal's name came back:
    // the row stays (marked interrupted), and the journal is adopted
    // beside it instead of failing at every launch.
    let dir = TempDir::new().expect("tempdir");
    let mut store = store_in(&dir);
    let tree = journals(&dir);
    let samples = ramp(4_800, 2);
    let path = {
        let mut writer =
            JournalWriter::create_named(&tree, "j_namesake".to_string(), 16_000).expect("writer");
        writer.append_frames(&samples).expect("append");
        writer.finalize().expect("finalize");
        writer.path().to_path_buf()
    };
    store.adopt_journal(&path, None).expect("adopt");
    std::fs::copy(store.audio_path("j_namesake"), &path).expect("name comes back");
    std::fs::remove_file(store.audio_path("j_namesake")).expect("stored audio lost");
    age(&path, old());
    store.reconcile().expect("reconcile");

    let report = store.recover_capture_journals(&tree).expect("scan");
    assert!(report.failed.is_empty(), "{report:?}");
    assert_eq!(report.recovered.len(), 1, "{report:?}");
    assert_eq!(report.recovered[0].id, "j_namesake-recovered");
    assert_eq!(store.load_audio("j_namesake-recovered").expect("audio").samples, samples);
    assert!(store.get_capture("j_namesake").expect("read").is_some(), "the row stays");
    assert!(!path.exists());
}

// --- #356 review round 8 ---

#[test]
fn a_superseded_copy_the_take_no_longer_proves_is_kept_by_the_sweep() {
    // Proven when moved aside; then compression keeps the 48 kHz take as
    // 16 kHz request audio, and the same-rate proof no longer holds: the
    // copy stays.
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

    assert!(store.compress_audio("j_48k").is_ok());
    assert_eq!(store.load_audio("j_48k").expect("audio").sample_rate, 16_000);
    let report = store.sweep_retention().expect("sweep");
    assert!(report.swept.is_empty(), "{report:?}");
    assert_eq!(report.retained.len(), 1, "{report:?}");
    let kept = tree.join(SUPERSEDED_SUBDIR).join("j_48k.sj");
    assert_eq!(read_audio_journal(&kept).expect("kept whole").samples, samples);
}

#[test]
fn a_reconcile_that_keeps_failing_holds_the_journal_back_only_a_few_passes() {
    let dir = TempDir::new().expect("tempdir");
    let mut store = store_in(&dir);
    let tree = journals(&dir);
    let confirmed = ramp(1_600, 0);
    let journal = faulted_journal(&tree, "j_stuck", &confirmed);
    die_while_storing_in_place_of(&store, "j_stuck", &ramp(4_800, 0));

    // No reconcile commits the replacement, pass after pass.
    for _ in 0..PENDING_REPLACEMENT_PASSES {
        let report = store.recover_capture_journals(&tree).expect("scan");
        assert_eq!(report.deferred, vec!["j_stuck".to_string()], "{report:?}");
        assert!(journal.exists());
    }
    let report = store.recover_capture_journals(&tree).expect("scan");
    adopted_whole(&store, &report, "j_stuck", &confirmed);
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
fn a_stopped_sweep_removes_nothing_more() {
    let dir = TempDir::new().expect("tempdir");
    let mut store = store_in(&dir);
    let quarantine = store.root().join(QUARANTINE_DIR);
    std::fs::create_dir_all(&quarantine).expect("quarantine");
    for id in ["c_a", "c_b"] {
        std::fs::write(quarantine.join(format!("{id}.sj")), b"deleted bytes").expect("write");
    }
    let report = store
        .sweep_retention_until(|| !quarantine.join("c_a.sj").exists())
        .expect("sweep");
    assert!(report.stopped, "{report:?}");
    assert_eq!(report.swept.len(), 1, "{report:?}");
    assert!(quarantine.join("c_b.sj").exists());
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
