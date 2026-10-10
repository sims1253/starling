//! Lossless at-rest audio and the retention policy (#342): FLAC fidelity
//! on the fidelity corpus and the LibriSpeech fixture, the compression
//! crash windows, pins, and the retention sweep cases.

use super::*;
use crate::audio::{decode_pcm16_wav, encode_wav_16k_parts};
use tempfile::TempDir;

fn store_in(dir: &TempDir) -> StoreV2 {
    StoreV2::open(dir.path().join("v2")).expect("open v2 store")
}

fn take_at(store: &mut StoreV2, rate: u32, samples: &[f32]) -> String {
    let mut take = store
        .begin_take_at_rate(rate, TakeMeta::for_device("test-device"))
        .expect("begin take");
    take.append_and_seal(samples).expect("append");
    take.finish(store).expect("finish").record.id
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

/// The request WAV a transcription of this take receives right now.
fn request_wav(store: &StoreV2, id: &str) -> Vec<u8> {
    let audio = store.load_audio(id).expect("load audio");
    encode_wav_16k_parts(&audio.samples, audio.sample_rate, 1).expect("wav")
}

fn files(store: &StoreV2, id: &str) -> (bool, bool) {
    (store.audio_path(id).exists(), store.flac_path(id).exists())
}

fn temps(store: &StoreV2) -> Vec<PathBuf> {
    std::fs::read_dir(store.root().join(AUDIO_DIR))
        .expect("audio dir")
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some(FLAC_TEMP_EXT))
        .collect()
}

fn compressed(outcome: CompressionOutcome) {
    assert!(
        matches!(outcome, CompressionOutcome::Compressed { .. }),
        "{outcome:?}"
    );
}

fn complete(store: &mut StoreV2, id: &str) -> String {
    store
        .begin_recognition(id, "engine:test", None)
        .expect("begin");
    store
        .finish_recognition(
            id,
            RecognitionOutcome::Completed {
                text: "hello there",
                extra_json: None,
            },
        )
        .expect("finish");
    store
        .attempts_for(id)
        .expect("attempts")
        .into_iter()
        .find(AttemptRecord::is_final_transcript)
        .expect("final attempt")
        .id
}

/// A transcribed take created `days_ago` days back.
fn aged_take(store: &mut StoreV2, days_ago: i64, len: usize) -> String {
    let id = take_at(store, 16_000, &speechy(len, days_ago as u32 + 1));
    complete(store, &id);
    backdate(store, &id, days_ago);
    id
}

fn backdate(store: &StoreV2, id: &str, days_ago: i64) {
    let created = iso_utc(time::OffsetDateTime::now_utc() - time::Duration::days(days_ago));
    store
        .conn
        .execute(
            "UPDATE captures SET created_utc = ?2 WHERE id = ?1",
            params![id, created],
        )
        .expect("backdate");
}

fn policy(class: &str, limits: ClassLimits) -> RetentionPolicy {
    let mut policy = RetentionPolicy::default();
    policy.limits.insert(class.to_string(), limits);
    policy
}

fn age(days: u32) -> ClassLimits {
    ClassLimits {
        max_age_days: Some(days),
        max_total_bytes: None,
    }
}

fn retired_ids(report: &RetentionReport) -> Vec<String> {
    report.retired.iter().map(|item| item.id.clone()).collect()
}

fn held(report: &RetentionReport, id: &str) -> Option<HoldReason> {
    report
        .held
        .iter()
        .find(|item| item.id == id)
        .map(|item| item.reason)
}

fn correction_for(capture: &str, attempt: &str) -> CorrectionRecord {
    CorrectionRecord {
        capture_id: capture.to_string(),
        request_id: "r1".to_string(),
        raw_attempt_id: attempt.to_string(),
        raw_text: "hello there".to_string(),
        processed_text: "Hello there.".to_string(),
        final_text: Some("Hello there.".to_string()),
        decision: CorrectionDecision::Accepted,
        decision_utc: "2026-09-24T10:00:00Z".to_string(),
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
    }
}

// ---- FLAC fidelity ----------------------------------------------------

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

/// Compress one take and check the acceptance claim: the stored samples
/// are exactly the request PCM16, and a retry's request WAV from FLAC is
/// byte-identical to the one the journal produced.
fn assert_flac_fidelity(store: &mut StoreV2, rate: u32, samples: &[f32], name: &str) {
    let id = take_at(store, rate, samples);
    let from_journal = request_wav(store, &id);
    assert_eq!(
        from_journal,
        encode_wav_16k_parts(samples, rate, 1).expect("direct"),
        "{name}: the journal stores the take as captured"
    );
    compressed(store.compress_audio(&id).expect(name));
    assert_eq!(files(store, &id), (false, true), "{name}");
    let from_flac = request_wav(store, &id);
    assert!(
        from_flac == from_journal,
        "{name}: request WAV differs after compression"
    );
    let expected = request_pcm16(samples, rate).expect("pcm16");
    let file = File::open(store.flac_path(&id)).expect("flac");
    assert_eq!(
        flac::decode(io::BufReader::new(file)).expect("decode"),
        expected,
        "{name}"
    );
}

#[test]
fn flac_is_sample_exact_on_the_fidelity_corpus() {
    let corpus: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../../../packages/contracts/fidelity-corpus/corpus.json"
    ))
    .expect("corpus");
    let dir = TempDir::new().expect("tempdir");
    let mut store = store_in(&dir);
    let fixtures = corpus["fixtures"].as_array().expect("fixtures");
    assert!(
        fixtures.len() >= 8,
        "the whole corpus, long passage included"
    );
    for fixture in fixtures {
        let name = fixture["id"].as_str().expect("id");
        let synthesis = &fixture["synthesis"];
        let rate = synthesis["sample_rate"].as_u64().expect("rate") as u32;
        let pcm = synthesize(synthesis["segments"].as_array().expect("segments"), rate);
        // The take as a WAV import would store it (decode_pcm16_wav).
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
        for sample in &pcm {
            wav.extend_from_slice(&sample.to_le_bytes());
        }
        let decoded = decode_pcm16_wav(&wav).expect("wav");
        assert_flac_fidelity(&mut store, decoded.sample_rate, &decoded.samples, name);
    }
}

#[test]
fn flac_is_sample_exact_on_real_speech_and_device_rates() {
    let dir = TempDir::new().expect("tempdir");
    let mut store = store_in(&dir);
    let wav = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../../tests/fixtures/2086-149220-0033.wav"
    ))
    .expect("LibriSpeech fixture");
    let speech = decode_pcm16_wav(&wav).expect("decode fixture");
    assert_flac_fidelity(
        &mut store,
        speech.sample_rate,
        &speech.samples,
        "librispeech",
    );
    // Raw f32 captures, not PCM16-derived: at 16 kHz and at device rates
    // the request path resamples (48 kHz, 44.1 kHz).
    assert_flac_fidelity(&mut store, 16_000, &speechy(40_000, 3), "f32 16k");
    assert_flac_fidelity(&mut store, 48_000, &speechy(96_000, 5), "f32 48k");
    assert_flac_fidelity(&mut store, 44_100, &speechy(88_200, 7), "f32 44.1k");
    // Hot input: the clamp and both full-scale ends survive.
    let hot: Vec<f32> = (0..20_000)
        .map(|i| match i % 4 {
            0 => 1.5,
            1 => -1.5,
            2 => 1.0,
            _ => -1.0,
        })
        .collect();
    assert_flac_fidelity(&mut store, 16_000, &hot, "full scale");
}

// ---- compression protocol ------------------------------------------------

#[test]
fn compression_lists_and_plays_like_the_journal_did() {
    let dir = TempDir::new().expect("tempdir");
    let mut store = store_in(&dir);
    let id = take_at(&mut store, 16_000, &speechy(32_000, 1));
    let journal_bytes = std::fs::metadata(store.audio_path(&id))
        .expect("journal")
        .len();
    let CompressionOutcome::Compressed {
        journal_bytes: reported,
        flac_bytes,
    } = store.compress_audio(&id).expect("compress")
    else {
        panic!("not compressed");
    };
    assert_eq!(reported, journal_bytes);
    assert!(
        flac_bytes * 3 < journal_bytes,
        "{flac_bytes} vs {journal_bytes}"
    );
    assert_eq!(store.audio_at_rest(&id).expect("state"), AudioAtRest::Flac);
    assert!(store.audio_journal_exists(&id).expect("exists"));
    let page = store.list_records(0, 10).expect("list");
    let ListedCapture::Capture(listing) = &page.records[0] else {
        panic!("damaged");
    };
    assert!(listing.problems.is_empty(), "{:?}", listing.problems);
    assert_eq!(listing.audio, AudioAtRest::Flac);
    // Nothing left to do, and nothing for reconcile to repair.
    assert!(
        store
            .compression_candidates(10)
            .expect("candidates")
            .is_empty()
    );
    let report = store.reconcile().expect("reconcile");
    assert!(!report.has_findings(), "{}", report.summary());
    assert_eq!(files(&store, &id), (false, true));
}

#[test]
fn a_read_that_resolved_the_journal_falls_over_to_its_flac() {
    let dir = TempDir::new().expect("tempdir");
    let mut store = store_in(&dir);
    let id = take_at(&mut store, 16_000, &speechy(16_000, 2));
    let before = request_wav(&store, &id);
    let path = store.audio_journal_path(&id).expect("path");
    assert_eq!(path.extension().and_then(|ext| ext.to_str()), Some("sj"));
    compressed(store.compress_audio(&id).expect("compress"));
    let audio = read_audio_journal(&path).expect("raced read");
    assert_eq!(
        encode_wav_16k_parts(&audio.samples, audio.sample_rate, 1).expect("wav"),
        before
    );
}

#[test]
fn a_crash_before_publishing_leaves_the_journal_and_scratch_only() {
    let dir = TempDir::new().expect("tempdir");
    let mut store = store_in(&dir);
    let id = take_at(&mut store, 16_000, &speechy(16_000, 3));
    let before = request_wav(&store, &id);
    let job = store
        .compression_candidates(1)
        .expect("candidates")
        .remove(0);
    // Killed after the temp was written and verified, before the rename.
    let prepared = prepare_compression(&job).expect("prepare");
    let temp = prepared.temp.clone();
    std::mem::forget(prepared);
    assert_eq!(files(&store, &id), (true, false));
    assert_eq!(request_wav(&store, &id), before);

    // A fresh temp belongs to a possibly-live compressor: kept.
    store.reconcile().expect("reconcile");
    assert!(temp.exists());
    // An old one is scratch.
    let old = std::time::SystemTime::now() - FLAC_TEMP_GRACE - std::time::Duration::from_secs(1);
    File::options()
        .write(true)
        .open(&temp)
        .expect("temp")
        .set_modified(old)
        .expect("mtime");
    let report = store.reconcile().expect("reconcile");
    assert!(!report.has_findings(), "{}", report.summary());
    assert!(temps(&store).is_empty());
    assert_eq!(files(&store, &id), (true, false));
    // The take compresses normally afterwards.
    compressed(store.compress_audio(&id).expect("compress"));
    assert_eq!(request_wav(&store, &id), before);
}

#[test]
fn a_crash_after_publishing_is_completed_by_reconcile() {
    let dir = TempDir::new().expect("tempdir");
    let mut store = store_in(&dir);
    let id = take_at(&mut store, 48_000, &speechy(48_000, 4));
    let before = request_wav(&store, &id);
    let job = store
        .compression_candidates(1)
        .expect("candidates")
        .remove(0);
    let prepared = prepare_compression(&job).expect("prepare");
    // Killed between the rename and the journal unlink.
    std::fs::rename(&prepared.temp, store.flac_path(&id)).expect("publish");
    assert_eq!(files(&store, &id), (true, true));
    // Both files: the journal is what reads resolve to meanwhile.
    assert_eq!(
        store.audio_at_rest(&id).expect("state"),
        AudioAtRest::Journal
    );
    assert_eq!(request_wav(&store, &id), before);

    let report = store.reconcile().expect("reconcile");
    assert_eq!(report.completed_compressions, vec![id.clone()]);
    assert!(!report.has_findings(), "{}", report.summary());
    assert_eq!(files(&store, &id), (false, true));
    assert_eq!(request_wav(&store, &id), before);
}

#[test]
fn a_damaged_published_flac_never_costs_the_journal() {
    let dir = TempDir::new().expect("tempdir");
    let mut store = store_in(&dir);
    let id = take_at(&mut store, 16_000, &speechy(16_000, 5));
    let before = request_wav(&store, &id);
    let job = store
        .compression_candidates(1)
        .expect("candidates")
        .remove(0);
    let prepared = prepare_compression(&job).expect("prepare");
    std::fs::rename(&prepared.temp, store.flac_path(&id)).expect("publish");
    // The published copy is damaged (disk trouble after the publish).
    let mut bytes = std::fs::read(store.flac_path(&id)).expect("flac");
    let middle = bytes.len() / 2;
    bytes[middle] ^= 0xff;
    std::fs::write(store.flac_path(&id), bytes).expect("damage");

    let report = store.reconcile().expect("reconcile");
    assert!(report.completed_compressions.is_empty());
    assert!(
        report
            .unreadable
            .iter()
            .any(|(unreadable, _)| unreadable == &id),
        "{:?}",
        report.unreadable
    );
    assert_eq!(files(&store, &id), (true, true), "nothing removed");
    assert_eq!(request_wav(&store, &id), before);
    // A new compression replaces the damaged copy.
    compressed(store.compress_audio(&id).expect("compress"));
    assert_eq!(files(&store, &id), (false, true));
    assert_eq!(request_wav(&store, &id), before);
}

#[test]
fn every_compression_step_leaves_complete_audio() {
    // Walk the protocol step by step; at each point a fresh store sees
    // complete audio for the take.
    let dir = TempDir::new().expect("tempdir");
    let mut store = store_in(&dir);
    let id = take_at(&mut store, 16_000, &speechy(24_000, 6));
    let before = request_wav(&store, &id);
    let check = |label: &str| {
        let reopened = StoreV2::open(dir.path().join("v2")).expect("reopen");
        assert_eq!(request_wav(&reopened, &id), before, "{label}");
    };
    let job = store
        .compression_candidates(1)
        .expect("candidates")
        .remove(0);
    check("before");
    let prepared = prepare_compression(&job).expect("prepare");
    check("temp written");
    std::fs::rename(&prepared.temp, store.flac_path(&id)).expect("publish");
    check("published");
    std::fs::remove_file(store.audio_path(&id)).expect("unlink");
    check("journal removed");
}

#[test]
fn audio_pinned_by_an_attempt_is_never_compressed_under_it() {
    let dir = TempDir::new().expect("tempdir");
    let mut store = store_in(&dir);
    let id = take_at(&mut store, 16_000, &speechy(16_000, 7));
    store
        .begin_recognition(&id, "engine:test", None)
        .expect("begin");
    assert!(
        store
            .compression_candidates(10)
            .expect("candidates")
            .is_empty()
    );
    assert!(matches!(
        store.compress_audio(&id).expect("compress"),
        CompressionOutcome::Skipped(_)
    ));
    store
        .finish_recognition(
            &id,
            RecognitionOutcome::Failed {
                message: "engine crashed",
            },
        )
        .expect("finish");

    // A retry that starts while the encode runs off the lock wins.
    let job = store
        .compression_candidates(1)
        .expect("candidates")
        .remove(0);
    let prepared = prepare_compression(&job).expect("prepare");
    store
        .begin_recognition(&id, "engine:test", None)
        .expect("retry");
    assert!(matches!(
        store.commit_compression(prepared).expect("commit"),
        CompressionOutcome::Skipped(_)
    ));
    assert_eq!(files(&store, &id), (true, false));
    assert!(temps(&store).is_empty(), "the abandoned temp is removed");
}

#[test]
fn a_take_still_recording_is_not_a_candidate() {
    let dir = TempDir::new().expect("tempdir");
    let mut store = store_in(&dir);
    let mut take = store
        .begin_take(TakeMeta::for_device("test-device"))
        .expect("begin");
    take.append_frames(&speechy(16_000, 8)).expect("append");
    take.write_boundary().expect("boundary");
    assert!(
        store
            .compression_candidates(10)
            .expect("candidates")
            .is_empty()
    );
    let id = take.finish(&mut store).expect("finish").record.id;
    assert_eq!(
        store.compression_candidates(10).expect("candidates").len(),
        1
    );
    compressed(store.compress_audio(&id).expect("compress"));
}

#[test]
fn deleting_during_compression_and_after_it_keeps_r21_semantics() {
    let dir = TempDir::new().expect("tempdir");
    let mut store = store_in(&dir);
    // Deleted while the encode runs off the lock.
    let racing = take_at(&mut store, 16_000, &speechy(16_000, 9));
    let job = store
        .compression_candidates(1)
        .expect("candidates")
        .remove(0);
    let prepared = prepare_compression(&job).expect("prepare");
    store.delete_capture(&racing).expect("delete");
    assert!(matches!(
        store.commit_compression(prepared).expect("commit"),
        CompressionOutcome::Skipped(_)
    ));
    assert!(temps(&store).is_empty());
    assert!(store.quarantine_path(&racing).exists());

    // Deleted after compression: the FLAC is quarantined, stays dead, and
    // the sweep removes it.
    let done = take_at(&mut store, 16_000, &speechy(16_000, 10));
    compressed(store.compress_audio(&done).expect("compress"));
    store.delete_capture(&done).expect("delete");
    assert!(store.quarantine_flac_path(&done).exists());
    assert_eq!(files(&store, &done), (false, false));
    store.reconcile().expect("reconcile");
    assert!(
        store.get_capture(&done).expect("row").is_none(),
        "never resurrected"
    );
    let report = store.sweep_retention().expect("sweep");
    let swept: Vec<&str> = report.swept.iter().map(|file| file.id.as_str()).collect();
    assert!(
        swept.contains(&racing.as_str()) && swept.contains(&done.as_str()),
        "{swept:?}"
    );
    assert!(!store.quarantine_flac_path(&done).exists());

    // A quarantined FLAC whose delete crashed before the row went is
    // completed, not resurrected.
    let crashed = take_at(&mut store, 16_000, &speechy(16_000, 11));
    compressed(store.compress_audio(&crashed).expect("compress"));
    std::fs::rename(
        store.flac_path(&crashed),
        store.quarantine_flac_path(&crashed),
    )
    .expect("quarantine rename");
    let report = store.reconcile().expect("reconcile");
    assert!(report.completed_deletes.contains(&crashed));
    assert!(store.get_capture(&crashed).expect("row").is_none());
}

// ---- retention policy --------------------------------------------------

#[test]
fn retention_is_off_by_default() {
    let dir = TempDir::new().expect("tempdir");
    let mut store = store_in(&dir);
    let id = aged_take(&mut store, 4_000, 16_000);
    let policy = RetentionPolicy::default();
    assert!(!policy.is_active());
    let report = store
        .apply_retention_policy(&policy, time::OffsetDateTime::now_utc())
        .expect("apply");
    assert!(report.retired.is_empty() && report.held.is_empty());
    assert_eq!(
        store.audio_at_rest(&id).expect("state"),
        AudioAtRest::Journal
    );
}

#[test]
fn the_age_limit_retires_old_audio_and_keeps_everything_else() {
    let dir = TempDir::new().expect("tempdir");
    let mut store = store_in(&dir);
    let old = aged_take(&mut store, 40, 16_000);
    let fresh = aged_take(&mut store, 2, 16_000);
    compressed(store.compress_audio(&old).expect("compress"));
    let attempts_before = store.attempts_for(&old).expect("attempts").len();

    let report = store
        .apply_retention_policy(
            &policy(STANDARD_CLASS, age(30)),
            time::OffsetDateTime::now_utc(),
        )
        .expect("apply");
    assert_eq!(retired_ids(&report), vec![old.clone()]);
    assert_eq!(report.retired[0].reason, RetireReason::Age);
    assert!(report.retired_bytes > 0);
    assert_eq!(files(&store, &old), (false, false));
    assert_eq!(files(&store, &fresh), (true, false));

    // Only the audio went: the row and its transcript stay, the listing
    // says retired without calling it a problem, reads explain why.
    assert!(store.get_capture(&old).expect("row").is_some());
    assert_eq!(
        store.attempts_for(&old).expect("attempts").len(),
        attempts_before
    );
    assert!(matches!(
        store.audio_at_rest(&old).expect("state"),
        AudioAtRest::Retired { .. }
    ));
    let err = store.load_audio(&old).expect_err("no audio");
    assert!(err.to_string().contains("retention policy"), "{err}");
    let page = store.list_records(0, 10).expect("list");
    for record in &page.records {
        let ListedCapture::Capture(listing) = record else {
            panic!("damaged");
        };
        assert!(listing.problems.is_empty(), "{:?}", listing.problems);
    }
    // Reconcile neither flags it interrupted nor resurrects anything.
    let report = store.reconcile().expect("reconcile");
    assert!(!report.has_findings(), "{}", report.summary());
    assert_eq!(
        store.get_capture(&old).expect("row").expect("row").status,
        CaptureStatus::Complete
    );
    // Not a compression candidate anymore, and idempotent.
    let candidates = store.compression_candidates(10).expect("candidates");
    assert!(candidates.iter().all(|job| job.id != old));
    let again = store
        .apply_retention_policy(
            &policy(STANDARD_CLASS, age(30)),
            time::OffsetDateTime::now_utc(),
        )
        .expect("apply again");
    assert!(again.retired.is_empty());
}

#[test]
fn referenced_in_use_untranscribed_and_recent_audio_is_held_and_said() {
    let dir = TempDir::new().expect("tempdir");
    let mut store = store_in(&dir);
    // Referenced by a correction record (#304).
    let corrected = aged_take(&mut store, 90, 16_000);
    let attempt = store.attempts_for(&corrected).expect("attempts")[0]
        .id
        .clone();
    store
        .upsert_correction_record(&correction_for(&corrected, &attempt))
        .expect("correction");
    // Referenced by a document revision naming its attempt.
    let documented = aged_take(&mut store, 90, 16_000);
    let doc_attempt = store.attempts_for(&documented).expect("attempts")[0]
        .id
        .clone();
    store.upsert_document("doc_1", "notes", 1, 1).expect("doc");
    store
        .store_document_revision(&RevisionRow {
            rev_id: "doc_1#1".to_string(),
            doc_id: "doc_1".to_string(),
            base_revision: None,
            sources_json: Some(format!(r#"{{"attempts":["{doc_attempt}"]}}"#)),
            text: "hello".to_string(),
            status: "committed".to_string(),
            provenance: None,
            disposition: Some("committed".to_string()),
        })
        .expect("revision");
    // Never transcribed: its audio is all there is.
    let failed = take_at(&mut store, 16_000, &speechy(16_000, 12));
    store
        .begin_recognition(&failed, "engine:test", None)
        .expect("begin");
    store
        .finish_recognition(&failed, RecognitionOutcome::Failed { message: "oom" })
        .expect("fail");
    backdate(&store, &failed, 90);
    // A retry in flight.
    let busy = aged_take(&mut store, 90, 16_000);
    store
        .begin_recognition(&busy, "engine:test", None)
        .expect("retry");
    // Plain and old: the only one that goes.
    let plain = aged_take(&mut store, 90, 16_000);

    let now = time::OffsetDateTime::now_utc();
    let report = store
        .apply_retention_policy(&policy(STANDARD_CLASS, age(30)), now)
        .expect("apply");
    assert_eq!(retired_ids(&report), vec![plain.clone()]);
    assert_eq!(
        held(&report, &corrected),
        Some(HoldReason::Referenced {
            revisions: 0,
            corrections: 1
        })
    );
    assert_eq!(
        held(&report, &documented),
        Some(HoldReason::Referenced {
            revisions: 1,
            corrections: 0
        })
    );
    assert_eq!(held(&report, &failed), Some(HoldReason::Untranscribed));
    assert_eq!(held(&report, &busy), Some(HoldReason::InUse));

    // Only after the user agreed does referenced audio go — and its
    // correction records and revisions stay.
    let mut agreed = policy(STANDARD_CLASS, age(30));
    agreed.include_referenced = true;
    let report = store.apply_retention_policy(&agreed, now).expect("apply");
    let mut retired = retired_ids(&report);
    retired.sort();
    let mut expected = vec![corrected.clone(), documented.clone()];
    expected.sort();
    assert_eq!(retired, expected);
    assert_eq!(
        store
            .correction_records_for(&corrected)
            .expect("records")
            .len(),
        1
    );
    assert_eq!(
        store
            .get_document("doc_1")
            .expect("doc")
            .expect("doc")
            .revisions
            .len(),
        1
    );
    assert_eq!(held(&report, &failed), Some(HoldReason::Untranscribed));
    assert_eq!(held(&report, &busy), Some(HoldReason::InUse));

    // The grace protects fresh takes from any limit.
    let fresh = aged_take(&mut store, 0, 16_000);
    let report = store
        .apply_retention_policy(
            &policy(STANDARD_CLASS, age(0)),
            now + time::Duration::seconds(5),
        )
        .expect("apply");
    assert_eq!(held(&report, &fresh), Some(HoldReason::Recent));
}

#[test]
fn the_size_limit_keeps_the_newest_audio_that_fits() {
    let dir = TempDir::new().expect("tempdir");
    let mut store = store_in(&dir);
    let oldest = aged_take(&mut store, 30, 16_000);
    let middle = aged_take(&mut store, 20, 16_000);
    let newest = aged_take(&mut store, 10, 16_000);
    let each = store.audio_bytes(&newest);
    let limits = ClassLimits {
        max_age_days: None,
        max_total_bytes: Some(each * 2 + each / 2),
    };
    let report = store
        .apply_retention_policy(
            &policy(STANDARD_CLASS, limits),
            time::OffsetDateTime::now_utc(),
        )
        .expect("apply");
    assert_eq!(retired_ids(&report), vec![oldest.clone()]);
    assert_eq!(report.retired[0].reason, RetireReason::Size);
    assert!(report.over_limit.is_empty());
    assert_eq!(files(&store, &middle), (true, false));
    assert_eq!(files(&store, &newest), (true, false));

    // A held take still occupies its class's budget: the limit is
    // reported as unreachable instead of retiring something else.
    store
        .begin_recognition(&middle, "engine:test", None)
        .expect("pin");
    let tight = ClassLimits {
        max_age_days: None,
        max_total_bytes: Some(each / 2),
    };
    let report = store
        .apply_retention_policy(
            &policy(STANDARD_CLASS, tight),
            time::OffsetDateTime::now_utc(),
        )
        .expect("apply");
    assert_eq!(retired_ids(&report), vec![newest.clone()]);
    assert_eq!(held(&report, &middle), Some(HoldReason::InUse));
    assert_eq!(report.over_limit.len(), 1);
}

#[test]
fn class_changes_move_a_take_under_the_other_limits() {
    let dir = TempDir::new().expect("tempdir");
    let mut store = store_in(&dir);
    let standard = aged_take(&mut store, 100, 16_000);
    let archived = aged_take(&mut store, 100, 16_000);
    store
        .set_retention_class(&archived, ARCHIVAL_CLASS)
        .expect("archive");
    assert!(matches!(
        store.set_retention_class("c_missing", ARCHIVAL_CLASS),
        Err(StoreV2Error::NotFound(_))
    ));
    assert!(store.set_retention_class(&standard, " ").is_err());

    let now = time::OffsetDateTime::now_utc();
    let mut both = policy(STANDARD_CLASS, age(365));
    both.limits.insert(ARCHIVAL_CLASS.to_string(), age(30));
    let report = store.apply_retention_policy(&both, now).expect("apply");
    assert_eq!(retired_ids(&report), vec![archived.clone()]);
    assert_eq!(report.retired[0].class, ARCHIVAL_CLASS);

    // Moving the standard take into the archive class subjects it to the
    // archive limit on the next run.
    store
        .set_retention_class(&standard, ARCHIVAL_CLASS)
        .expect("archive");
    let report = store.apply_retention_policy(&both, now).expect("apply");
    assert_eq!(retired_ids(&report), vec![standard.clone()]);
    // A class without limits is left alone.
    let other = aged_take(&mut store, 100, 16_000);
    store.set_retention_class(&other, "keep").expect("class");
    let report = store.apply_retention_policy(&both, now).expect("apply");
    assert!(report.retired.is_empty());
    assert_eq!(files(&store, &other), (true, false));
}

#[test]
fn limits_reached_mid_take_never_touch_the_take_being_recorded() {
    let dir = TempDir::new().expect("tempdir");
    let mut store = store_in(&dir);
    let old = aged_take(&mut store, 50, 16_000);
    // A take is recording: its journal is in staging, acknowledged.
    let mut take = store
        .begin_take(TakeMeta::for_device("test-device"))
        .expect("begin");
    take.append_frames(&speechy(32_000, 13)).expect("append");
    take.write_boundary().expect("boundary");
    let zero = ClassLimits {
        max_age_days: Some(0),
        max_total_bytes: Some(0),
    };
    let report = store
        .apply_retention_policy(
            &policy(STANDARD_CLASS, zero),
            time::OffsetDateTime::now_utc(),
        )
        .expect("apply");
    assert_eq!(retired_ids(&report), vec![old]);
    // The take finishes and commits untouched; once committed it is a
    // fresh take and the grace holds it.
    let id = take.finish(&mut store).expect("finish").record.id;
    assert_eq!(store.load_audio(&id).expect("audio").samples.len(), 32_000);
    complete(&mut store, &id);
    let report = store
        .apply_retention_policy(
            &policy(STANDARD_CLASS, zero),
            time::OffsetDateTime::now_utc(),
        )
        .expect("apply");
    assert!(report.retired.is_empty());
    assert_eq!(held(&report, &id), Some(HoldReason::Recent));
}

#[test]
fn deleted_and_retired_takes_stay_distinct() {
    let dir = TempDir::new().expect("tempdir");
    let mut store = store_in(&dir);
    let deleted = aged_take(&mut store, 60, 16_000);
    let retired = aged_take(&mut store, 60, 16_000);
    store.delete_capture(&deleted).expect("delete");
    let report = store
        .apply_retention_policy(
            &policy(STANDARD_CLASS, age(30)),
            time::OffsetDateTime::now_utc(),
        )
        .expect("apply");
    // The deleted take is the sweep's, not the policy's.
    assert_eq!(retired_ids(&report), vec![retired.clone()]);
    assert!(store.quarantine_path(&deleted).exists());
    // Deleting a retired take works like any delete.
    store.delete_capture(&retired).expect("delete retired");
    store.reconcile().expect("reconcile");
    assert!(store.get_capture(&retired).expect("row").is_none());
    let report = store.sweep_retention().expect("sweep");
    assert_eq!(report.swept.len(), 1);
}

#[test]
fn a_crash_between_stamp_and_unlink_is_finished_by_reconcile() {
    let dir = TempDir::new().expect("tempdir");
    let mut store = store_in(&dir);
    let id = aged_take(&mut store, 60, 16_000);
    compressed(store.compress_audio(&id).expect("compress"));
    // The stamp committed; the process died before the unlink.
    store
        .conn
        .execute(
            "INSERT INTO tombstones(id, kind, deleted_utc, retention)
             VALUES (?1, 'audio', ?2, 'swept')",
            params![format!("{AUDIO_TOMBSTONE_PREFIX}{id}"), now_iso()],
        )
        .expect("stamp");
    assert_eq!(files(&store, &id), (false, true));
    assert!(matches!(
        store.audio_at_rest(&id).expect("state"),
        AudioAtRest::Retired { .. }
    ));
    assert!(store.load_audio(&id).is_err(), "already reads as retired");
    let report = store.reconcile().expect("reconcile");
    assert_eq!(report.completed_retirements, vec![id.clone()]);
    assert!(!report.has_findings(), "{}", report.summary());
    assert_eq!(files(&store, &id), (false, false));
    assert!(store.get_capture(&id).expect("row").is_some());
}

#[test]
fn an_in_process_pin_holds_compression_and_retention() {
    // A retry pins before it loads the audio and keeps the pin until its
    // attempt is marked started: nothing acts in between.
    let dir = TempDir::new().expect("tempdir");
    let mut store = store_in(&dir);
    let id = aged_take(&mut store, 90, 16_000);
    store.pin_audio(&id);
    store.pin_audio(&id);
    assert!(
        store
            .compression_candidates(10)
            .expect("candidates")
            .is_empty()
    );
    let now = time::OffsetDateTime::now_utc();
    let report = store
        .apply_retention_policy(&policy(STANDARD_CLASS, age(30)), now)
        .expect("apply");
    assert!(report.retired.is_empty());
    assert_eq!(held(&report, &id), Some(HoldReason::InUse));

    // A compression prepared before the pin does not publish under it.
    store.unpin_audio(&id);
    store.unpin_audio(&id);
    let job = store
        .compression_candidates(1)
        .expect("candidates")
        .remove(0);
    let prepared = prepare_compression(&job).expect("prepare");
    store.pin_audio(&id);
    assert!(matches!(
        store.commit_compression(prepared).expect("commit"),
        CompressionOutcome::Skipped(_)
    ));
    assert_eq!(files(&store, &id), (true, false));

    // Pins nest; the last release frees the take.
    store.pin_audio(&id);
    store.unpin_audio(&id);
    assert!(
        store
            .compression_candidates(10)
            .expect("candidates")
            .is_empty()
    );
    store.unpin_audio(&id);
    store.unpin_audio(&id); // an extra release does nothing
    compressed(store.compress_audio(&id).expect("compress"));
    let report = store
        .apply_retention_policy(&policy(STANDARD_CLASS, age(30)), now)
        .expect("apply");
    assert_eq!(retired_ids(&report), vec![id]);
}

#[test]
fn retention_decides_after_a_peer_connection_commits() {
    // Another connection on the same root (another process) is starting a
    // transcription of the take. The hold check and the stamp take the
    // write lock first, so they wait for the peer and see its attempt
    // instead of retiring the audio under it.
    let dir = TempDir::new().expect("tempdir");
    let mut store = store_in(&dir);
    let id = aged_take(&mut store, 90, 16_000);
    let peer = Connection::open(store.root().join(DB_FILE)).expect("peer");
    peer.busy_timeout(std::time::Duration::from_secs(5))
        .expect("busy timeout");
    peer.execute_batch("BEGIN IMMEDIATE")
        .expect("peer write lock");
    peer.execute(
        "INSERT INTO recognition_attempts
             (id, capture_id, backend, text, partial_or_final, status)
         VALUES ('peer-attempt', ?1, 'engine:peer', '', 'final', 'started')",
        params![id],
    )
    .expect("peer attempt");

    let sweeper = std::thread::spawn(move || {
        let report = store
            .apply_retention_policy(
                &policy(STANDARD_CLASS, age(30)),
                time::OffsetDateTime::now_utc(),
            )
            .expect("apply");
        (store, report)
    });
    std::thread::sleep(std::time::Duration::from_millis(300));
    peer.execute_batch("COMMIT").expect("peer commit");
    let (store, report) = sweeper.join().expect("sweeper");
    assert!(report.retired.is_empty(), "{report:?}");
    assert_eq!(held(&report, &id), Some(HoldReason::InUse));
    assert_eq!(
        store.audio_at_rest(&id).expect("state"),
        AudioAtRest::Journal
    );
}

#[test]
fn retired_audio_is_never_transcribed_again() {
    // A retry that loaded the audio just before the policy removed it
    // (another instance's sweep) cannot start its attempt.
    let dir = TempDir::new().expect("tempdir");
    let mut store = store_in(&dir);
    let id = aged_take(&mut store, 90, 16_000);
    let report = store
        .apply_retention_policy(
            &policy(STANDARD_CLASS, age(30)),
            time::OffsetDateTime::now_utc(),
        )
        .expect("apply");
    assert_eq!(retired_ids(&report), vec![id.clone()]);
    let err = store
        .begin_recognition(&id, "engine:test", None)
        .expect_err("refused");
    assert!(err.to_string().contains("retention policy"), "{err}");
    assert!(
        store
            .attempts_for(&id)
            .expect("attempts")
            .iter()
            .all(|attempt| attempt.status != "started"),
        "no attempt row was left behind"
    );
}

#[test]
fn a_policy_changed_mid_run_stops_before_removing_more() {
    let dir = TempDir::new().expect("tempdir");
    let mut store = store_in(&dir);
    let older = aged_take(&mut store, 120, 16_000);
    let old = aged_take(&mut store, 90, 16_000);
    let limited = policy(STANDARD_CLASS, age(30));
    // The user lifts the limit right after the first removal.
    let reads = std::cell::Cell::new(0);
    let report = store
        .apply_live_retention_policy(
            || {
                reads.set(reads.get() + 1);
                if reads.get() <= 2 {
                    limited.clone()
                } else {
                    RetentionPolicy::default()
                }
            },
            time::OffsetDateTime::now_utc(),
        )
        .expect("apply");
    assert!(report.policy_changed);
    // Walked newest first: `old` went under the limit as it stood then.
    assert_eq!(retired_ids(&report), vec![old]);
    assert_eq!(
        store.audio_at_rest(&older).expect("state"),
        AudioAtRest::Journal
    );
}

#[test]
fn a_class_change_by_a_peer_mid_run_is_respected() {
    // A peer moves a due take into an unlimited class while the sweep
    // waits for the write lock: the take is not removed.
    let dir = TempDir::new().expect("tempdir");
    let mut store = store_in(&dir);
    let id = aged_take(&mut store, 90, 16_000);
    let peer = Connection::open(store.root().join(DB_FILE)).expect("peer");
    peer.busy_timeout(std::time::Duration::from_secs(5))
        .expect("busy timeout");
    peer.execute_batch("BEGIN IMMEDIATE")
        .expect("peer write lock");
    peer.execute(
        "UPDATE captures SET retention_class = ?2 WHERE id = ?1",
        params![id, ARCHIVAL_CLASS],
    )
    .expect("peer class change");

    let sweeper = std::thread::spawn(move || {
        let report = store
            .apply_retention_policy(
                &policy(STANDARD_CLASS, age(30)),
                time::OffsetDateTime::now_utc(),
            )
            .expect("apply");
        (store, report)
    });
    std::thread::sleep(std::time::Duration::from_millis(300));
    peer.execute_batch("COMMIT").expect("peer commit");
    let (store, report) = sweeper.join().expect("sweeper");
    assert!(report.retired.is_empty(), "{report:?}");
    assert_eq!(
        store.audio_at_rest(&id).expect("state"),
        AudioAtRest::Journal
    );
}

#[test]
fn a_size_limit_recounts_after_a_peer_moves_a_newer_take_away() {
    // Two old takes under a limit that fits one. While the sweep waits to
    // retire the older, a peer moves the newer into the archival class:
    // the class now fits, so the older keeps its audio.
    let dir = TempDir::new().expect("tempdir");
    let mut store = store_in(&dir);
    let older = aged_take(&mut store, 120, 16_000);
    let newer = aged_take(&mut store, 90, 16_000);
    let limit = store.audio_bytes(&older).max(store.audio_bytes(&newer));
    let peer = Connection::open(store.root().join(DB_FILE)).expect("peer");
    peer.busy_timeout(std::time::Duration::from_secs(5))
        .expect("busy timeout");
    peer.execute_batch("BEGIN IMMEDIATE")
        .expect("peer write lock");
    peer.execute(
        "UPDATE captures SET retention_class = ?2 WHERE id = ?1",
        params![newer, ARCHIVAL_CLASS],
    )
    .expect("peer class change");

    let sweeper = std::thread::spawn(move || {
        let report = store
            .apply_retention_policy(
                &policy(
                    STANDARD_CLASS,
                    ClassLimits {
                        max_age_days: None,
                        max_total_bytes: Some(limit),
                    },
                ),
                time::OffsetDateTime::now_utc(),
            )
            .expect("apply");
        (store, report)
    });
    std::thread::sleep(std::time::Duration::from_millis(300));
    peer.execute_batch("COMMIT").expect("peer commit");
    let (store, report) = sweeper.join().expect("sweeper");
    assert!(report.retired.is_empty(), "{report:?}");
    assert!(report.over_limit.is_empty(), "{report:?}");
    assert_eq!(
        store.audio_at_rest(&older).expect("state"),
        AudioAtRest::Journal
    );
    assert_eq!(
        store.audio_at_rest(&newer).expect("state"),
        AudioAtRest::Journal
    );
}

#[test]
fn a_take_a_peer_moved_away_is_never_removed_later_in_the_walk() {
    // Three takes under a limit that fits one. While the sweep waits to
    // retire the middle one, a peer moves the oldest into the archival
    // class: the sweep reaches it after its recount and must still not
    // remove it under the standard limit.
    let dir = TempDir::new().expect("tempdir");
    let mut store = store_in(&dir);
    let oldest = aged_take(&mut store, 150, 16_000);
    let middle = aged_take(&mut store, 120, 16_000);
    let newest = aged_take(&mut store, 90, 16_000);
    let limit = store.audio_bytes(&newest);
    let peer = Connection::open(store.root().join(DB_FILE)).expect("peer");
    peer.busy_timeout(std::time::Duration::from_secs(5))
        .expect("busy timeout");
    peer.execute_batch("BEGIN IMMEDIATE")
        .expect("peer write lock");
    peer.execute(
        "UPDATE captures SET retention_class = ?2 WHERE id = ?1",
        params![oldest, ARCHIVAL_CLASS],
    )
    .expect("peer class change");

    let sweeper = std::thread::spawn(move || {
        let report = store
            .apply_retention_policy(
                &policy(
                    STANDARD_CLASS,
                    ClassLimits {
                        max_age_days: None,
                        max_total_bytes: Some(limit),
                    },
                ),
                time::OffsetDateTime::now_utc(),
            )
            .expect("apply");
        (store, report)
    });
    std::thread::sleep(std::time::Duration::from_millis(300));
    peer.execute_batch("COMMIT").expect("peer commit");
    let (store, report) = sweeper.join().expect("sweeper");
    assert_eq!(retired_ids(&report), vec![middle]);
    assert_eq!(
        store.audio_at_rest(&oldest).expect("state"),
        AudioAtRest::Journal
    );
    assert_eq!(
        store.audio_at_rest(&newest).expect("state"),
        AudioAtRest::Journal
    );
}

#[test]
fn a_peer_compression_mid_sweep_is_counted_at_its_new_size() {
    // The sweep counted a newer take as its journal; a peer then
    // compresses it. With the FLAC's size the class fits, so the older
    // take keeps its audio.
    let dir = TempDir::new().expect("tempdir");
    let mut store = store_in(&dir);
    let older = aged_take(&mut store, 120, 16_000);
    let newer = aged_take(&mut store, 90, 16_000);
    let job = store
        .compression_candidates(10)
        .expect("candidates")
        .into_iter()
        .find(|job| job.id == newer)
        .expect("newer is a candidate");
    let prepared = prepare_compression(&job).expect("prepare");
    let limit = prepared.flac_bytes + store.audio_bytes(&older);
    assert!(store.audio_bytes(&newer) + store.audio_bytes(&older) > limit);
    let (journal, flac) = (store.audio_path(&newer), store.flac_path(&newer));
    // The peer compressor holds the write lock across its publish, as
    // `commit_compression` does.
    let peer = Connection::open(store.root().join(DB_FILE)).expect("peer");
    peer.busy_timeout(std::time::Duration::from_secs(5))
        .expect("busy timeout");
    peer.execute_batch("BEGIN IMMEDIATE")
        .expect("peer write lock");

    let sweeper = std::thread::spawn(move || {
        let report = store
            .apply_retention_policy(
                &policy(
                    STANDARD_CLASS,
                    ClassLimits {
                        max_age_days: None,
                        max_total_bytes: Some(limit),
                    },
                ),
                time::OffsetDateTime::now_utc(),
            )
            .expect("apply");
        (store, report)
    });
    std::thread::sleep(std::time::Duration::from_millis(300));
    std::fs::rename(&prepared.temp, &flac).expect("publish");
    std::fs::remove_file(&journal).expect("journal unlink");
    peer.execute(
        "INSERT INTO meta(key, value) VALUES ('audio_generation', '1')
         ON CONFLICT(key) DO UPDATE SET value = CAST(value AS INTEGER) + 1",
        [],
    )
    .expect("generation bump");
    peer.execute_batch("COMMIT").expect("peer commit");
    let (store, report) = sweeper.join().expect("sweeper");
    assert!(report.retired.is_empty(), "{report:?}");
    assert_eq!(
        store.audio_at_rest(&older).expect("state"),
        AudioAtRest::Journal
    );
    assert_eq!(
        store.audio_at_rest(&newer).expect("state"),
        AudioAtRest::Flac
    );
}

#[test]
fn a_compression_bumps_the_audio_generation_other_connections_see() {
    let dir = TempDir::new().expect("tempdir");
    let mut store = store_in(&dir);
    let id = take_at(&mut store, 16_000, &speechy(16_000, 3));
    let peer = Connection::open(store.root().join(DB_FILE)).expect("peer");
    let version = |conn: &Connection| -> i64 {
        conn.query_row("PRAGMA data_version", [], |row| row.get(0))
            .expect("version")
    };
    let before = version(&peer);
    compressed(store.compress_audio(&id).expect("compress"));
    assert_ne!(version(&peer), before);
    assert_eq!(files(&store, &id), (false, true));
}
