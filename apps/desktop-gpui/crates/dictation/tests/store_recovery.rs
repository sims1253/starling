//! The store's durability contract, through its public entry points:
//! a take killed between any two steps of its save comes back holding
//! exactly the audio that was confirmed on disk, with a note on what was
//! lost; damaged or foreign files are kept aside, never repaired into
//! takes; deleted takes stay deleted; FLAC at rest is bit-exact and no
//! step of compressing a take ever costs its audio; and the retention
//! policy removes only the audio its limits reach.
//!
//! The race windows that need the store's own fault injection (a
//! recovery that dies mid-salvage) or a second handle interleaved inside
//! one call stay beside the store (`store_v2/race_tests.rs`).

use std::path::{Path, PathBuf};
use std::time::Duration;

use starling_dictation::audio::{decode_pcm16_wav, encode_wav_16k_parts};
use starling_dictation::journal::{FileSink, JournalWriter};
use starling_dictation::storage::iso_utc;
use starling_dictation::store_v2::{
    AudioAtRest, CaptureStatus, ClassLimits, CommitMark, CompressionOutcome,
    HoldReason, ListedCapture, RecognitionOutcome, RetentionPolicy, RetireReason, StoreV2,
    StoreV2Error, TakeMeta, ARCHIVAL_CLASS, FINALIZED_ADOPTION_GRACE, STANDARD_CLASS,
    SUPERSEDED_SUBDIR,
};
use tempfile::TempDir;

// --------------------------------------------------------------------- //
// Helpers
// --------------------------------------------------------------------- //

/// A deterministic sample block.
fn ramp(len: usize, offset: usize) -> Vec<f32> {
    (0..len).map(|i| ((offset + i) % 997) as f32 * 0.0001).collect()
}

/// Speech-ish signal: an enveloped tone plus a little noise.
fn speechy(len: usize, seed: u32) -> Vec<f32> {
    let mut state = seed | 1;
    (0..len)
        .map(|i| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let t = i as f32 / 16_000.0;
            let envelope = 0.6 + 0.4 * (2.0 * std::f32::consts::PI * 10.0 * t).sin();
            0.35 * envelope * (2.0 * std::f32::consts::PI * 220.0 * t).sin()
                + (state as f32 / u32::MAX as f32 - 0.5) * 0.01
        })
        .collect()
}

fn open(dir: &TempDir) -> StoreV2 {
    StoreV2::open(dir.path().join("v2")).expect("open the store")
}

fn file_in(store: &StoreV2, tree: &str, id: &str, ext: &str) -> PathBuf {
    store.root().join(tree).join(format!("{id}.{ext}"))
}

/// The recorder's journal tree beside the store.
fn recorder_tree(store: &StoreV2) -> PathBuf {
    let tree = store.root().join("journals");
    std::fs::create_dir_all(&tree).expect("journal tree");
    tree
}

fn backdate_file(path: &Path, by: Duration) {
    std::fs::OpenOptions::new()
        .write(true)
        .open(path)
        .expect("open")
        .set_modified(std::time::SystemTime::now() - by)
        .expect("set mtime");
}

/// Older than any live save could be.
fn old() -> Duration {
    FINALIZED_ADOPTION_GRACE + Duration::from_secs(5)
}

/// A recorder journal of `confirmed` samples under a fsynced boundary;
/// `tail` more are appended unconfirmed; `finalized` seals it as a clean
/// stop does. The writer is returned so a test can keep it alive.
fn recorder_journal(
    tree: &Path,
    confirmed: &[f32],
    tail: usize,
    finalized: bool,
) -> (String, PathBuf, JournalWriter<FileSink>) {
    let mut writer = JournalWriter::create(tree, 16_000).expect("journal");
    writer.append_frames(confirmed).expect("append");
    if finalized {
        writer.finalize().expect("finalize");
    } else {
        writer.write_boundary().expect("boundary");
        if tail > 0 {
            writer.append_frames(&ramp(tail, 7)).expect("unconfirmed tail");
        }
    }
    (writer.id().to_string(), writer.path().to_path_buf(), writer)
}

/// A take stored from samples in place of recorder journal `journal_id`,
/// the way the save-from-memory paths do.
fn stored_in_place_of(store: &mut StoreV2, journal_id: &str, rate: u32, samples: &[f32]) -> String {
    let mut meta = TakeMeta::for_device("");
    meta.supersedes_journal = Some(journal_id.to_string());
    let mut take = store.begin_take_at_rate(rate, meta).expect("begin");
    take.append_and_seal(samples).expect("append");
    take.finalize()
        .expect("finalize")
        .commit_marked(store, CommitMark::Complete)
        .expect("commit")
        .record
        .id
}

fn take_at(store: &mut StoreV2, rate: u32, samples: &[f32]) -> String {
    let mut take = store.begin_take_at_rate(rate, TakeMeta::for_device("test")).expect("begin");
    take.append_and_seal(samples).expect("append");
    take.finish(store).expect("finish").record.id
}

fn transcribe(store: &mut StoreV2, id: &str) {
    store.begin_recognition(id, "engine:test", None).expect("begin");
    store
        .finish_recognition(id, RecognitionOutcome::Completed { text: "hello there", extra_json: None })
        .expect("finish");
}

/// The request WAV a transcription of this take receives right now.
fn request_wav(store: &StoreV2, id: &str) -> Vec<u8> {
    let audio = store.load_audio(id).expect("load audio");
    encode_wav_16k_parts(&audio.samples, audio.sample_rate, 1).expect("wav")
}

/// Every take the store would recover at startup: its own reconcile,
/// then the recorder's tree (complete takes come back due for
/// transcription, as the host asks).
fn recover(store: &mut StoreV2) {
    store.reconcile().expect("reconcile");
    let tree = recorder_tree(store);
    store.recover_capture_journals_where(&tree, |_| true, true).expect("journal scan");
}

fn row_count(store: &StoreV2) -> usize {
    store.list_records(0, 100).expect("list").total
}

// --------------------------------------------------------------------- //
// Crash protocol
// --------------------------------------------------------------------- //

/// Where the process died.
#[derive(Clone, Copy, Debug)]
enum Kill {
    /// §4 step 1: frames and a fsynced boundary in staging, then bytes a
    /// crash may or may not have reached disk.
    StagingTornTail,
    /// Step 2a: the trailer written in staging, never promoted.
    StagingFinalized,
    /// Step 2b: promoted into `audio/`, the row never committed.
    PromotedNoRow,
    /// Step 3 done: the row committed — the durable ack.
    Committed,
    /// The recorder's tree: mid-recording, with and without a torn tail.
    RecorderMidTake { torn: bool },
    /// The recorder's tree: stopped (finalized), never stored.
    RecorderStopped,
}

/// One take killed at each step: after recovery it is a row holding
/// exactly the confirmed samples, interrupted with a note unless its save
/// finished, its journal sealed; a stopped take is due for transcription
/// and a cut one is not; recovering again changes nothing.
#[test]
fn a_take_killed_between_any_two_steps_comes_back_with_its_confirmed_audio() {
    struct Expect {
        status: CaptureStatus,
        note: Option<&'static [&'static str]>,
        transcribe: bool,
    }
    let interrupted = |note: &'static [&'static str], transcribe| Expect {
        status: CaptureStatus::Interrupted,
        note: Some(note),
        transcribe,
    };
    let cases = [
        (Kill::StagingTornTail, interrupted(&["gap flagged"], false)),
        (Kill::StagingFinalized, interrupted(&["finished cleanly"], false)),
        (Kill::PromotedNoRow, interrupted(&["never committed"], false)),
        (
            Kill::Committed,
            Expect {
                status: CaptureStatus::Complete,
                note: None,
                transcribe: false,
            },
        ),
        (
            Kill::RecorderMidTake { torn: true },
            interrupted(&["while this take was recording", "confirmed on disk", "discarded"], false),
        ),
        (
            Kill::RecorderMidTake { torn: false },
            interrupted(&["while this take was recording", "confirmed on disk"], false),
        ),
        (Kill::RecorderStopped, interrupted(&["complete recording (2.0 s)"], true)),
    ];
    for (index, (kill, expect)) in cases.into_iter().enumerate() {
        let dir = TempDir::new().expect("tempdir");
        let confirmed = ramp(32_000, index);
        let id = {
            let mut store = open(&dir);
            let tree = recorder_tree(&store);
            match kill {
                Kill::StagingTornTail => {
                    let mut take = store.begin_take(TakeMeta::for_device("mic")).expect("begin");
                    take.append_frames(&confirmed).expect("append");
                    assert_eq!(take.write_boundary().expect("boundary"), 32_000);
                    take.append_frames(&ramp(120, 1)).expect("unconfirmed tail");
                    take.id().to_string()
                }
                Kill::StagingFinalized => {
                    let mut take = store.begin_take(TakeMeta::for_device("mic")).expect("begin");
                    take.append_frames(&confirmed).expect("append");
                    take.finalize().expect("finalize").id
                }
                Kill::PromotedNoRow => {
                    let mut take = store.begin_take(TakeMeta::for_device("mic")).expect("begin");
                    take.append_frames(&confirmed).expect("append");
                    let id = take.finalize().expect("finalize").id;
                    store.promote_from_staging(&id).expect("promote");
                    id
                }
                Kill::Committed => {
                    let mut take = store.begin_take(TakeMeta::for_device("mic")).expect("begin");
                    take.append_frames(&confirmed).expect("append");
                    take.finish(&mut store).expect("finish").record.id
                }
                Kill::RecorderMidTake { torn } => {
                    let (id, path, writer) = recorder_journal(&tree, &confirmed, if torn { 800 } else { 0 }, false);
                    drop(writer); // the process died: no trailer
                    backdate_file(&path, old());
                    id
                }
                Kill::RecorderStopped => {
                    let (id, path, writer) = recorder_journal(&tree, &confirmed, 0, true);
                    drop(writer);
                    backdate_file(&path, old());
                    id
                }
            }
            // The store handle goes with the process.
        };

        let mut store = open(&dir);
        recover(&mut store);
        let case = format!("{kill:?}");
        let record = store.get_capture(&id).expect("read").unwrap_or_else(|| panic!("{case}: no row"));
        assert_eq!(record.status, expect.status, "{case}");
        assert_eq!(record.frame_count, 32_000, "{case}: the confirmed extent, never more");
        let audio = store.load_audio(&id).expect("audio");
        assert_eq!(audio.samples, confirmed, "{case}");
        assert!(audio.finalized && audio.torn_tail_bytes == 0, "{case}: one valid trailer");
        match (expect.note, record.recovery_note()) {
            (None, None) => {}
            (Some(parts), Some(note)) => {
                for part in parts {
                    assert!(note.contains(part), "{case}: {note:?} lacks {part:?}");
                }
            }
            (want, got) => panic!("{case}: note {got:?}, wanted {want:?}"),
        }
        assert_eq!(store.transcription_wanted(&id).unwrap(), expect.transcribe, "{case}");
        let staging = std::fs::read_dir(store.root().join("staging")).expect("staging").count();
        assert_eq!(staging, 0, "{case}: staging drained");
        assert_eq!(std::fs::read_dir(recorder_tree(&store)).expect("tree").flatten().filter(|e| e.path().is_file()).count(), 0, "{case}: the tree drained");

        // A second start finds nothing more to do.
        let report = store.reconcile().expect("reconcile again");
        assert!(!report.has_findings(), "{case}: {}", report.summary());
        recover(&mut store);
        assert_eq!(row_count(&store), 1, "{case}: one take, once");
    }
}

/// A journal is left alone while it may still be someone's: while its
/// writer (or anyone holding its liveness) lives, and for a while after
/// its recorder finalized it — a live save may be under way. After that
/// it is recovered like any other.
#[test]
fn a_journal_someone_may_still_be_saving_is_left_to_them() {
    let dir = TempDir::new().expect("tempdir");
    let mut store = open(&dir);
    let tree = recorder_tree(&store);
    // Recording: the writer holds its lock (even on an old file).
    let (live, live_path, writer) = recorder_journal(&tree, &ramp(1_600, 0), 0, false);
    backdate_file(&live_path, old());
    // Faulted: the writer is gone, the take records on in memory.
    let (faulted, faulted_path, faulted_writer) = recorder_journal(&tree, &ramp(1_600, 1), 0, false);
    let saving = faulted_writer.liveness();
    drop(faulted_writer);
    backdate_file(&faulted_path, old());
    // Just stopped: its save may be running right now.
    let (stopped, stopped_path, stopped_writer) = recorder_journal(&tree, &ramp(1_600, 2), 0, true);
    drop(stopped_writer);

    let report = store.recover_capture_journals(&tree).expect("scan");
    let mut deferred = report.deferred.clone();
    deferred.sort();
    let mut expected = vec![live.clone(), faulted.clone(), stopped.clone()];
    expected.sort();
    assert_eq!(deferred, expected, "{report:?}");
    assert!(report.recovered.is_empty());
    assert!(live_path.exists() && faulted_path.exists() && stopped_path.exists());

    // Their owners go away; time passes.
    drop(writer);
    saving.release();
    backdate_file(&stopped_path, old());
    let report = store.recover_capture_journals(&tree).expect("rescan");
    assert_eq!(report.recovered.len(), 3, "{report:?}");
    assert!(report.summary().contains("Recovered 3 recordings"), "{}", report.summary());
}

/// A recorder journal whose take was already stored from memory in its
/// place is moved aside — never a second take — but only when the stored
/// audio, read back, holds every sample the journal confirmed. Anything
/// less and the journal comes back as its own take: a duplicate the user
/// can delete beats audio lost.
#[test]
fn a_journal_is_set_aside_only_when_its_stored_replacement_holds_every_sample() {
    #[derive(Debug)]
    enum Replacement {
        Holds,
        HoldsAsFlac,
        Shorter,
        OneSampleDiffers,
        OtherRate,
        Damaged,
    }
    for replacement in [
        Replacement::Holds,
        Replacement::HoldsAsFlac,
        Replacement::Shorter,
        Replacement::OneSampleDiffers,
        Replacement::OtherRate,
        Replacement::Damaged,
    ] {
        let dir = TempDir::new().expect("tempdir");
        let mut store = open(&dir);
        let tree = recorder_tree(&store);
        let confirmed = ramp(1_600, 0);
        let (journal, path, writer) = recorder_journal(&tree, &confirmed, 0, false);
        drop(writer);
        backdate_file(&path, old());
        let take = ramp(4_800, 0);
        let stored = match replacement {
            Replacement::Holds | Replacement::HoldsAsFlac | Replacement::Damaged => {
                stored_in_place_of(&mut store, &journal, 16_000, &take)
            }
            Replacement::Shorter => stored_in_place_of(&mut store, &journal, 16_000, &take[..800]),
            Replacement::OneSampleDiffers => {
                let mut other = take.clone();
                other[1_000] += 0.25;
                stored_in_place_of(&mut store, &journal, 16_000, &other)
            }
            Replacement::OtherRate => stored_in_place_of(&mut store, &journal, 48_000, &take),
        };
        match replacement {
            Replacement::HoldsAsFlac => {
                assert!(matches!(store.compress_audio(&stored).unwrap(), CompressionOutcome::Compressed { .. }))
            }
            Replacement::Damaged => std::fs::write(file_in(&store, "audio", &stored, "sj"), b"no longer audio").unwrap(),
            _ => {}
        }
        let set_aside = matches!(replacement, Replacement::Holds | Replacement::HoldsAsFlac);

        let report = store.recover_capture_journals(&tree).expect("scan");
        let case = format!("{replacement:?}");
        assert!(!path.exists(), "{case}: the journal left the tree");
        if set_aside {
            assert!(report.recovered.is_empty(), "{case}: {report:?}");
            assert_eq!(report.superseded, vec![journal.clone()], "{case}");
            assert!(tree.join(SUPERSEDED_SUBDIR).join(format!("{journal}.sj")).exists(), "{case}");
            assert_eq!(row_count(&store), 1, "{case}: one take, not two");
            // The retention sweep may remove the proven copy.
            let sweep = store.sweep_retention().expect("sweep");
            assert_eq!(sweep.swept.len(), 1, "{case}: {sweep:?}");
        } else {
            assert_eq!(report.recovered.len(), 1, "{case}: {report:?}");
            assert_eq!(store.load_audio(&journal).expect("adopted").samples, confirmed, "{case}");
            assert_eq!(row_count(&store), 2, "{case}");
        }
    }
}

/// Files that are not what they claim to be are kept where a human can
/// find them, reported, and never turned into takes; a database from a
/// newer build is refused without being touched.
#[test]
fn damaged_or_foreign_files_are_kept_and_reported_never_made_into_takes() {
    let dir = TempDir::new().expect("tempdir");
    let mut store = open(&dir);
    // Staging: garbage, and a journal that died before its first frame.
    let garbage = file_in(&store, "staging", "c_garbage", "sj");
    std::fs::write(&garbage, b"not a journal").unwrap();
    drop(store.begin_take(TakeMeta::for_device("mic")).expect("an empty take"));
    // Committed takes whose audio went missing or bad.
    let missing = take_at(&mut store, 16_000, &ramp(400, 1));
    std::fs::remove_file(file_in(&store, "audio", &missing, "sj")).unwrap();
    let corrupt = take_at(&mut store, 16_000, &ramp(400, 2));
    std::fs::write(file_in(&store, "audio", &corrupt, "sj"), b"garbage bytes").unwrap();
    // The recorder's tree: junk under a journal's name, a stray file, and
    // a journal with no confirmed audio.
    let tree = recorder_tree(&store);
    std::fs::write(tree.join("j_junk.sj"), b"not a journal at all").unwrap();
    std::fs::write(tree.join("notes.txt"), b"hand-dropped").unwrap();
    let empty = JournalWriter::create(&tree, 16_000).unwrap().path().to_path_buf();
    backdate_file(&empty, old());

    let report = store.reconcile().expect("reconcile");
    assert_eq!(report.unreadable.len(), 1, "{report:?}");
    assert_eq!(report.unreadable[0].0, "c_garbage");
    assert_eq!(report.empty_journals.len(), 1, "{report:?}");
    assert_eq!(report.marked_interrupted, vec![missing.clone()]);
    assert!(garbage.exists(), "kept in place");
    let journals = store.recover_capture_journals(&tree).expect("scan");
    assert!(journals.recovered.is_empty(), "{journals:?}");
    assert_eq!(journals.unrecognized.len(), 1, "{journals:?}");
    assert_eq!(std::fs::read(tree.join("j_junk.sj.unrecognized")).unwrap(), b"not a journal at all");
    assert!(empty.exists() && tree.join("notes.txt").exists());
    assert!(store.recover_capture_journals(&tree).unwrap().unrecognized.is_empty(), "set aside once");

    // The listing never aborts: each damaged take lists with its problem.
    let page = store.list_records(0, 10).expect("listing");
    assert_eq!(page.total, 2);
    for (id, problem) in [(&missing, "missing"), (&corrupt, "header")] {
        let listing = page
            .records
            .iter()
            .find_map(|record| match record {
                ListedCapture::Capture(listing) if listing.record.id == *id => Some(listing),
                _ => None,
            })
            .expect("listed");
        assert!(listing.problems.iter().any(|p| p.contains(problem)), "{:?}", listing.problems);
    }

    // A database a newer build wrote is refused as it is.
    drop(store);
    let db = dir.path().join("v2").join("starling.db");
    let conn = rusqlite::Connection::open(&db).unwrap();
    conn.execute("UPDATE meta SET value = '99' WHERE key = 'schema_version'", []).unwrap();
    drop(conn);
    assert!(matches!(
        StoreV2::open(dir.path().join("v2")),
        Err(StoreV2Error::SchemaTooNew { found: 99, .. })
    ));
    let conn = rusqlite::Connection::open(&db).unwrap();
    let version: String = conn
        .query_row("SELECT value FROM meta WHERE key = 'schema_version'", [], |row| row.get(0))
        .unwrap();
    assert_eq!(version, "99", "the refusal changed nothing");
}

/// R21: a deleted take stays deleted — through a restored file, a delete
/// that died between moving the audio and its transaction, and after the
/// sweep removed the tombstoned bytes.
#[test]
fn deleted_takes_never_come_back() {
    let dir = TempDir::new().expect("tempdir");
    let mut store = open(&dir);
    let deleted = take_at(&mut store, 16_000, &ramp(800, 0));
    let bytes = std::fs::read(file_in(&store, "audio", &deleted, "sj")).unwrap();
    store.delete_capture(&deleted).expect("delete");
    store.delete_capture(&deleted).expect("a second delete is no error");
    // A restored backup puts the audio back.
    std::fs::write(file_in(&store, "audio", &deleted, "sj"), &bytes).unwrap();
    // A delete killed between its quarantine rename and its transaction.
    let half_deleted = take_at(&mut store, 16_000, &ramp(800, 1));
    std::fs::rename(
        file_in(&store, "audio", &half_deleted, "sj"),
        file_in(&store, "quarantine", &half_deleted, "sj"),
    )
    .unwrap();
    // A compressed take deleted: its FLAC is the quarantined copy.
    let compressed = take_at(&mut store, 16_000, &speechy(16_000, 3));
    assert!(matches!(store.compress_audio(&compressed).unwrap(), CompressionOutcome::Compressed { .. }));
    store.delete_capture(&compressed).expect("delete");

    let report = store.reconcile().expect("reconcile");
    assert!(report.completed_deletes.contains(&deleted), "{report:?}");
    assert!(report.completed_deletes.contains(&half_deleted), "{report:?}");
    for id in [&deleted, &half_deleted, &compressed] {
        assert!(store.get_capture(id).unwrap().is_none(), "{id} resurrected");
    }
    assert!(!file_in(&store, "audio", &deleted, "sj").exists(), "the file went back to quarantine");

    let sweep = store.sweep_retention().expect("sweep");
    assert_eq!(sweep.swept.len(), 3, "{sweep:?}");
    // The tombstone outlives the bytes.
    std::fs::write(file_in(&store, "audio", &deleted, "sj"), &bytes).unwrap();
    store.reconcile().expect("reconcile");
    assert!(store.get_capture(&deleted).unwrap().is_none());
    assert_eq!(row_count(&store), 0);
}

// --------------------------------------------------------------------- //
// FLAC at rest
// --------------------------------------------------------------------- //

/// Port of `tests/fidelity_audio.py::synthesize_samples`: int16 speech
/// stand-ins (enveloped 220 Hz tone) and digital-zero silence.
fn synthesize(segments: &[serde_json::Value], rate: u32) -> Vec<i16> {
    let mut samples = Vec::new();
    for segment in segments {
        let seconds = segment["duration_seconds"].as_f64().expect("duration");
        let count = (seconds * f64::from(rate)).round() as usize;
        if segment["kind"] == "silence" {
            samples.extend(std::iter::repeat_n(0i16, count));
            continue;
        }
        for i in 0..count {
            let t = i as f64 / f64::from(rate);
            let envelope = 0.6 + 0.4 * (2.0 * std::f64::consts::PI * 10.0 * t).sin();
            let value = 12_000.0 * envelope * (2.0 * std::f64::consts::PI * 220.0 * t).sin();
            samples.push(value.clamp(-32_768.0, 32_767.0) as i16);
        }
    }
    samples
}

fn pcm16_wav(pcm: &[i16], rate: u32) -> Vec<u8> {
    let mut wav = Vec::new();
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(36 + pcm.len() as u32 * 2).to_le_bytes());
    wav.extend_from_slice(b"WAVEfmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&rate.to_le_bytes());
    wav.extend_from_slice(&(rate * 2).to_le_bytes());
    wav.extend_from_slice(&2u16.to_le_bytes());
    wav.extend_from_slice(&16u16.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&(pcm.len() as u32 * 2).to_le_bytes());
    for sample in pcm {
        wav.extend_from_slice(&sample.to_le_bytes());
    }
    wav
}

/// Compressing a take changes nothing a transcription receives: the
/// request WAV from the FLAC is byte-identical to the one the journal
/// produced — on the fidelity corpus, real speech, raw f32 captures at
/// device rates, and full-scale input.
#[test]
fn flac_at_rest_is_bit_exact() {
    let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../..");
    let corpus: serde_json::Value = serde_json::from_slice(
        &std::fs::read(repo.join("packages/contracts/fidelity-corpus/corpus.json")).expect("corpus"),
    )
    .expect("corpus json");
    let mut inputs: Vec<(String, u32, Vec<f32>)> = Vec::new();
    let fixtures = corpus["fixtures"].as_array().expect("fixtures");
    assert!(fixtures.len() >= 8, "the whole corpus, long passage included");
    for fixture in fixtures {
        let synthesis = &fixture["synthesis"];
        let rate = synthesis["sample_rate"].as_u64().expect("rate") as u32;
        let pcm = synthesize(synthesis["segments"].as_array().expect("segments"), rate);
        let decoded = decode_pcm16_wav(&pcm16_wav(&pcm, rate)).expect("wav");
        inputs.push((fixture["id"].as_str().unwrap().to_string(), decoded.sample_rate, decoded.samples));
    }
    let speech = decode_pcm16_wav(&std::fs::read(repo.join("tests/fixtures/2086-149220-0033.wav")).expect("LibriSpeech fixture"))
        .expect("decode fixture");
    inputs.push(("librispeech".into(), speech.sample_rate, speech.samples));
    inputs.push(("f32 16k".into(), 16_000, speechy(40_000, 3)));
    inputs.push(("f32 48k".into(), 48_000, speechy(96_000, 5)));
    inputs.push(("f32 44.1k".into(), 44_100, speechy(88_200, 7)));
    let hot: Vec<f32> = (0..20_000).map(|i| [1.5, -1.5, 1.0, -1.0][i % 4]).collect();
    inputs.push(("full scale".into(), 16_000, hot));

    let dir = TempDir::new().expect("tempdir");
    let mut store = open(&dir);
    for (name, rate, samples) in inputs {
        let id = take_at(&mut store, rate, &samples);
        let from_journal = request_wav(&store, &id);
        assert_eq!(from_journal, encode_wav_16k_parts(&samples, rate, 1).unwrap(), "{name}: stored as captured");
        assert!(
            matches!(store.compress_audio(&id).expect(&name), CompressionOutcome::Compressed { .. }),
            "{name}"
        );
        assert_eq!(store.audio_at_rest(&id).unwrap(), AudioAtRest::Flac, "{name}");
        assert!(!file_in(&store, "audio", &id, "sj").exists(), "{name}: the journal is gone");
        assert!(request_wav(&store, &id) == from_journal, "{name}: request WAV differs after compression");
    }
}

/// Each window of the compression protocol, killed: a temporary never
/// published leaves the journal; a FLAC published before the journal's
/// unlink is completed by reconcile; a published FLAC damaged on disk
/// never costs the journal; and a take pinned by an attempt is not
/// compressed under it. Every step reads back the same audio.
#[test]
fn no_step_of_compressing_a_take_costs_its_audio() {
    let dir = TempDir::new().expect("tempdir");
    let mut store = open(&dir);

    // Killed before publishing: only the journal (plus a temporary).
    let unpublished = take_at(&mut store, 16_000, &speechy(16_000, 3));
    let before = request_wav(&store, &unpublished);
    let job = store.compression_candidates(1).unwrap().remove(0);
    std::mem::forget(starling_dictation::store_v2::prepare_compression(&job).expect("prepare"));
    assert!(!file_in(&store, "audio", &unpublished, "flac").exists());
    assert_eq!(request_wav(&open(&dir), &unpublished), before);
    assert!(matches!(store.compress_audio(&unpublished).unwrap(), CompressionOutcome::Compressed { .. }));
    assert_eq!(request_wav(&store, &unpublished), before);

    // Killed after publishing, before the journal's unlink: both files.
    let published = take_at(&mut store, 48_000, &speechy(48_000, 4));
    let before = request_wav(&store, &published);
    let journal = file_in(&store, "audio", &published, "sj");
    let journal_bytes = std::fs::read(&journal).unwrap();
    assert!(matches!(store.compress_audio(&published).unwrap(), CompressionOutcome::Compressed { .. }));
    std::fs::write(&journal, &journal_bytes).unwrap();
    assert_eq!(store.audio_at_rest(&published).unwrap(), AudioAtRest::Journal, "reads resolve to the journal");
    assert_eq!(request_wav(&open(&dir), &published), before);
    let report = store.reconcile().expect("reconcile");
    assert_eq!(report.completed_compressions, vec![published.clone()]);
    assert!(!journal.exists());
    assert_eq!(request_wav(&store, &published), before);

    // The published FLAC damaged on disk: nothing is removed.
    let damaged = take_at(&mut store, 16_000, &speechy(16_000, 5));
    let before = request_wav(&store, &damaged);
    let journal = file_in(&store, "audio", &damaged, "sj");
    let journal_bytes = std::fs::read(&journal).unwrap();
    assert!(matches!(store.compress_audio(&damaged).unwrap(), CompressionOutcome::Compressed { .. }));
    std::fs::write(&journal, &journal_bytes).unwrap();
    let flac = file_in(&store, "audio", &damaged, "flac");
    let mut bytes = std::fs::read(&flac).unwrap();
    let middle = bytes.len() / 2;
    bytes[middle] ^= 0xff;
    std::fs::write(&flac, bytes).unwrap();
    let report = store.reconcile().expect("reconcile");
    assert!(report.completed_compressions.is_empty());
    assert!(report.unreadable.iter().any(|(id, _)| id == &damaged), "{:?}", report.unreadable);
    assert!(journal.exists() && flac.exists(), "nothing removed");
    assert_eq!(request_wav(&store, &damaged), before);
    assert!(matches!(store.compress_audio(&damaged).unwrap(), CompressionOutcome::Compressed { .. }));
    assert_eq!(request_wav(&store, &damaged), before);

    // An attempt in flight pins the journal.
    let pinned = take_at(&mut store, 16_000, &speechy(16_000, 7));
    store.begin_recognition(&pinned, "engine:test", None).unwrap();
    assert!(matches!(store.compress_audio(&pinned).unwrap(), CompressionOutcome::Skipped(_)));
    assert!(store.compression_candidates(10).unwrap().iter().all(|job| job.id != pinned));
}

// --------------------------------------------------------------------- //
// Retention
// --------------------------------------------------------------------- //

fn backdate(dir: &TempDir, id: &str, days: i64) {
    let created = iso_utc(time::OffsetDateTime::now_utc() - time::Duration::days(days));
    rusqlite::Connection::open(dir.path().join("v2").join("starling.db"))
        .unwrap()
        .execute("UPDATE captures SET created_utc = ?2 WHERE id = ?1", rusqlite::params![id, created])
        .unwrap();
}

/// A transcribed take created `days` ago.
fn aged(store: &mut StoreV2, dir: &TempDir, days: i64) -> String {
    let id = take_at(store, 16_000, &speechy(16_000, days as u32 + 1));
    transcribe(store, &id);
    backdate(dir, &id, days);
    id
}

fn limits(age_days: Option<u32>, bytes: Option<u64>) -> ClassLimits {
    ClassLimits {
        max_age_days: age_days,
        max_total_bytes: bytes,
    }
}

fn policy(class: &str, limits: ClassLimits) -> RetentionPolicy {
    let mut policy = RetentionPolicy::default();
    policy.limits.insert(class.to_string(), limits);
    policy
}

/// Retention is off until the user sets a limit. Then an age limit
/// removes old audio and nothing else — the row and its transcript stay,
/// reads say why — while audio that is untranscribed, in use or younger
/// than the grace is held and said so; a size limit keeps the newest
/// audio that fits; classes have their own limits; a deleted take is the
/// sweep's, not the policy's.
#[test]
fn retention_removes_only_the_audio_its_limits_reach() {
    let dir = TempDir::new().expect("tempdir");
    let mut store = open(&dir);
    let now = time::OffsetDateTime::now_utc();
    let old = aged(&mut store, &dir, 90);
    let fresh = aged(&mut store, &dir, 2);
    let untranscribed = take_at(&mut store, 16_000, &speechy(16_000, 12));
    backdate(&dir, &untranscribed, 90);
    let busy = aged(&mut store, &dir, 90);
    store.begin_recognition(&busy, "engine:test", None).unwrap();
    let deleted = aged(&mut store, &dir, 90);
    store.delete_capture(&deleted).unwrap();

    let off = store.apply_retention_policy(&RetentionPolicy::default(), now).unwrap();
    assert!(off.retired.is_empty() && off.held.is_empty(), "off by default");

    let report = store.apply_retention_policy(&policy(STANDARD_CLASS, limits(Some(30), None)), now).unwrap();
    let retired: Vec<_> = report.retired.iter().map(|item| item.id.clone()).collect();
    assert_eq!(retired, vec![old.clone()]);
    assert_eq!(report.retired[0].reason, RetireReason::Age);
    let held = |id: &str| report.held.iter().find(|item| item.id == id).map(|item| item.reason);
    assert_eq!(held(&untranscribed), Some(HoldReason::Untranscribed));
    assert_eq!(held(&busy), Some(HoldReason::InUse));
    assert!(file_in(&store, "audio", &fresh, "sj").exists());
    assert!(file_in(&store, "quarantine", &deleted, "sj").exists(), "the sweep's, not the policy's");
    // Only the audio went.
    assert!(matches!(store.audio_at_rest(&old).unwrap(), AudioAtRest::Retired { .. }));
    assert_eq!(store.get_capture(&old).unwrap().unwrap().status, CaptureStatus::Complete);
    assert!(store.attempts_for(&old).unwrap().iter().any(|attempt| attempt.is_final_transcript()));
    let err = store.load_audio(&old).expect_err("no audio");
    assert!(err.to_string().contains("retention policy"), "{err}");
    assert!(!store.reconcile().unwrap().has_findings(), "a retired take is not a crash finding");

    // The grace protects fresh takes from any limit.
    let fresh = aged(&mut store, &dir, 0);
    let report = store
        .apply_retention_policy(&policy(STANDARD_CLASS, limits(Some(0), None)), now + time::Duration::seconds(5))
        .unwrap();
    assert!(report.held.iter().any(|item| item.id == fresh && item.reason == HoldReason::Recent), "{report:?}");

    // A size limit keeps the newest audio that fits.
    let dir = TempDir::new().expect("tempdir");
    let mut store = open(&dir);
    let oldest = aged(&mut store, &dir, 30);
    let middle = aged(&mut store, &dir, 20);
    let newest = aged(&mut store, &dir, 10);
    let each = std::fs::metadata(file_in(&store, "audio", &newest, "sj")).unwrap().len();
    let report = store
        .apply_retention_policy(&policy(STANDARD_CLASS, limits(None, Some(each * 2 + each / 2))), now)
        .unwrap();
    assert_eq!(report.retired.iter().map(|item| item.id.clone()).collect::<Vec<_>>(), vec![oldest]);
    assert_eq!(report.retired[0].reason, RetireReason::Size);
    assert!(file_in(&store, "audio", &middle, "sj").exists() && file_in(&store, "audio", &newest, "sj").exists());

    // An archived take follows the archival limits.
    store.set_retention_class(&middle, ARCHIVAL_CLASS).unwrap();
    let mut both = policy(STANDARD_CLASS, limits(Some(365), None));
    both.limits.insert(ARCHIVAL_CLASS.to_string(), limits(Some(15), None));
    let report = store.apply_retention_policy(&both, now).unwrap();
    assert_eq!(report.retired.iter().map(|item| item.id.clone()).collect::<Vec<_>>(), vec![middle]);
    assert_eq!(report.retired[0].class, ARCHIVAL_CLASS);
    assert!(matches!(store.set_retention_class("c_missing", ARCHIVAL_CLASS), Err(StoreV2Error::NotFound(_))));
}
