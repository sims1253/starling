//! The app's storage facade (E02 cutover phase 1): one handle that is
//! either the v1 file session store or the v2 SQLite+journals store,
//! selected once at startup. Every daily-use operation — list, load audio,
//! save, transcribe bookkeeping, delete — goes through it, speaking the
//! v1-shaped types the UI already consumes. There is deliberately no
//! fallback: a v2 operation that fails surfaces its error, and the
//! documented way back to v1 is a restart (flag cleared / choice rolled
//! back), never a silent mid-session switch.

use std::sync::{Arc, Mutex, MutexGuard};

use starling_dictation::{
    audio, recorder,
    settings::StorageBackend,
    storage::{
        self, DamagedRecord, FileSessionStore, ListedRecord, SessionStatus, SessionSummary,
        TranscriptionResult,
    },
    store_v2::{
        self, AttemptRecord, CaptureRecord, CaptureStatus, ListedCapture, RecognitionOutcome,
        StoreV2, StoreV2Error,
    },
};

/// Which backend a session is running on.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum StorageKind {
    V1,
    V2,
}

impl StorageKind {
    pub(crate) fn label(self) -> &'static str {
        match self {
            StorageKind::V1 => "v1 file store (sessions/)",
            StorageKind::V2 => "v2 (SQLite + audio journals)",
        }
    }
}

/// The startup selection (E02): the `STARLING_STORAGE_V2` environment flag
/// (existing testing semantics — `v2_enabled_from`) opts in explicitly and
/// wins while set; otherwise the persisted choice from settings applies;
/// otherwise v1. Pure, so the matrix is testable without racing the
/// process environment.
pub(crate) fn select_backend(
    flag: Option<&str>,
    persisted: StorageBackend,
) -> (StorageKind, &'static str) {
    if store_v2::v2_enabled_from(flag) {
        return (StorageKind::V2, "STARLING_STORAGE_V2 is set");
    }
    match persisted {
        StorageBackend::V2 => (StorageKind::V2, "your saved storage choice"),
        StorageBackend::V1 => (StorageKind::V1, "default"),
    }
}

/// Map a v2 failure onto the app's existing storage error plumbing, keeping
/// [`storage::StorageError::NotFound`] intact — the transcription job's
/// delete-race decision keys on exactly that variant (R05).
fn v2_err(err: StoreV2Error) -> storage::StorageError {
    match err {
        StoreV2Error::NotFound(id) => storage::StorageError::NotFound(id),
        StoreV2Error::Invalid(reason) => storage::StorageError::Invalid(reason),
        other => storage::StorageError::Invalid(other.to_string()),
    }
}

/// Lock the shared v2 handle (a poisoned lock is recovered: the SQLite
/// connection is still consistent after a panic between statements).
pub(crate) fn lock_v2(handle: &Arc<Mutex<StoreV2>>) -> MutexGuard<'_, StoreV2> {
    handle
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// The active store. Constructed once at startup; never switches flavor
/// mid-session.
#[derive(Clone)]
pub(crate) enum Store {
    V1(Arc<FileSessionStore>),
    V2(Arc<Mutex<StoreV2>>),
}

impl Store {
    /// The v2 handle when this is the active backend (the migration flow
    /// reuses it instead of opening a second connection).
    pub(crate) fn as_v2(&self) -> Option<Arc<Mutex<StoreV2>>> {
        match self {
            Store::V1(_) => None,
            Store::V2(handle) => Some(handle.clone()),
        }
    }

    /// Whether a record with this id exists (the R05 delete-race
    /// re-check; metadata-only).
    pub(crate) fn exists(&self, id: &str) -> Result<bool, storage::StorageError> {
        match self {
            Store::V1(store) => Ok(store.get(id)?.is_some()),
            Store::V2(handle) => {
                let store = lock_v2(handle);
                Ok(store.get_capture(id).map_err(v2_err)?.is_some())
            }
        }
    }

    /// Metadata-only listing (G02): readable records as summaries, damaged
    /// ones flagged with their reason. The v2 path pages through bounded
    /// reads and maps each capture + its recognition attempts onto the
    /// v1-shaped summary the history list consumes.
    pub(crate) fn list(&self) -> Result<Vec<ListedRecord>, storage::StorageError> {
        match self {
            Store::V1(store) => store.list_records().map_err(Into::into),
            Store::V2(handle) => {
                let store = lock_v2(handle);
                list_v2(&store)
            }
        }
    }

    /// One recording's WAV, loaded on demand (G02). `Ok(None)` when the id
    /// is unknown — the delete race is indistinguishable from a missing
    /// record and is not an error.
    pub(crate) fn audio_wav(&self, id: &str) -> Result<Option<Arc<Vec<u8>>>, storage::StorageError> {
        match self {
            Store::V1(store) => match store.get(id)? {
                Some(session) => Ok(Some(session.wav.clone())),
                None => Ok(None),
            },
            Store::V2(handle) => {
                let store = lock_v2(handle);
                match store.load_audio(id) {
                    Ok(journal) => {
                        let pcm = audio::PcmAudio {
                            samples: journal.samples,
                            sample_rate: journal.sample_rate,
                            channels: 1,
                        };
                        let wav = audio::encode_wav_16k(&pcm)
                            .map_err(|err| storage::StorageError::Invalid(err.to_string()))?;
                        Ok(Some(Arc::new(wav)))
                    }
                    Err(StoreV2Error::NotFound(_)) => Ok(None),
                    Err(err) => Err(v2_err(err)),
                }
            }
        }
    }

    /// Persists a finished take: the recorder's journal is adopted as the
    /// v2 audio evidence when it has verified samples (moved, never
    /// re-encoded); otherwise the take is written from its encoded WAV
    /// through the full §4 protocol. v1 keeps its manifest linkage.
    /// Returns the record id.
    pub(crate) fn save_capture(
        &self,
        wav: Arc<Vec<u8>>,
        duration_ms: Option<f64>,
        journal: Option<&recorder::JournalReport>,
    ) -> Result<String, storage::StorageError> {
        match self {
            Store::V1(store) => store
                .create_with_journal(
                    (*wav).clone(),
                    duration_ms,
                    journal.map(|report| report.id.as_str()),
                )
                .map(|session| session.id)
                .map_err(Into::into),
            Store::V2(handle) => {
                let mut store = lock_v2(handle);
                if let Some(record) = adopt_or_none(&mut store, journal, None)? {
                    return Ok(record);
                }
                let take = store
                    .save_wav_capture(wav.as_slice(), store_v2::TakeMeta::for_device(""))
                    .map_err(v2_err)?;
                Ok(take.record.id)
            }
        }
    }

    /// Persists a salvaged take as interrupted-but-usable with `note`
    /// stating exactly what survived (I1 phase 2).
    pub(crate) fn save_interrupted_capture(
        &self,
        wav: Arc<Vec<u8>>,
        duration_ms: f64,
        journal: Option<&recorder::JournalReport>,
        note: &str,
    ) -> Result<String, storage::StorageError> {
        match self {
            Store::V1(store) => {
                let session = store
                    .create_with_journal(
                        (*wav).clone(),
                        Some(duration_ms),
                        journal.map(|report| report.id.as_str()),
                    )
                    .map_err(storage::StorageError::from)?;
                store.mark_interrupted(&session.id, note)?;
                Ok(session.id)
            }
            Store::V2(handle) => {
                let mut store = lock_v2(handle);
                if let Some(id) = adopt_or_none(&mut store, journal, Some(note))? {
                    return Ok(id);
                }
                let take = store
                    .save_wav_capture(wav.as_slice(), store_v2::TakeMeta::for_device(""))
                    .map_err(v2_err)?;
                let id = take.record.id.clone();
                store
                    .update_capture_status(&id, CaptureStatus::Interrupted, Some(note))
                    .map_err(v2_err)?;
                Ok(id)
            }
        }
    }

    /// Marks a transcription attempt as started on the record. `backend`
    /// labels the attempt ("starling:parakeet", "openai:whisper-large-v3").
    pub(crate) fn mark_attempt(&self, id: &str, backend: &str) -> Result<(), storage::StorageError> {
        match self {
            Store::V1(store) => store.mark_attempt(id).map(|_| ()).map_err(Into::into),
            Store::V2(handle) => {
                let mut store = lock_v2(handle);
                store
                    .begin_recognition(id, backend, None)
                    .map(|_| ())
                    .map_err(v2_err)
            }
        }
    }

    /// Records a successful transcript (v1: status + history bump; v2: the
    /// in-flight attempt is completed with the transcript preserved
    /// verbatim in its row).
    pub(crate) fn save_transcript(
        &self,
        id: &str,
        transcript: TranscriptionResult,
    ) -> Result<(), storage::StorageError> {
        match self {
            Store::V1(store) => store
                .save_transcript(id, transcript)
                .map(|_| ())
                .map_err(Into::into),
            Store::V2(handle) => {
                let mut store = lock_v2(handle);
                store
                    .finish_recognition_transcript(id, &transcript)
                    .map_err(v2_err)
            }
        }
    }

    /// Records a failed transcription attempt.
    pub(crate) fn save_failure(&self, id: &str, message: &str) -> Result<(), storage::StorageError> {
        match self {
            Store::V1(store) => store
                .save_failure(id, message)
                .map(|_| ())
                .map_err(Into::into),
            Store::V2(handle) => {
                let mut store = lock_v2(handle);
                store
                    .finish_recognition(id, RecognitionOutcome::Failed { message })
                    .map_err(v2_err)
            }
        }
    }

    /// Confirmed deletion (R21): v1 quarantines the linked journal then
    /// removes the session; v2 quarantines the audio journal and
    /// tombstones the row — resurrection-proof in both.
    pub(crate) fn delete(&self, id: &str) -> Result<(), storage::StorageError> {
        match self {
            Store::V1(store) => {
                let journals_root = starling_dictation::journal::default_journals_root();
                starling_dictation::journal::delete_session_and_journal(
                    store.as_ref(),
                    &journals_root,
                    id,
                )
                .map_err(Into::into)
            }
            Store::V2(handle) => {
                let mut store = lock_v2(handle);
                store.delete_capture(id).map_err(v2_err)
            }
        }
    }

    /// The startup recovery pass, per backend: v1 runs its journal-only
    /// recovery (I1 phase 2); v2 reconciles journals and metadata rows
    /// (§4), then fails recognition attempts still marked started by a
    /// previous run. Returns a user-facing summary string, empty when
    /// there was nothing to report.
    pub(crate) fn startup_recovery(&self) -> Result<String, storage::StorageError> {
        /// The note a stale recognition attempt gets at startup — same
        /// wording as the v1 "stuck in Transcribing" fix.
        const STALE_ATTEMPT_NOTE: &str =
            "Interrupted before the server returned a transcript. Your audio is ready to retry.";
        match self {
            Store::V1(store) => {
                let journals_root = starling_dictation::journal::default_journals_root();
                starling_dictation::journal::recover_interrupted_takes(
                    store.as_ref(),
                    &journals_root,
                )
                .map(|report| {
                    if report.has_findings() {
                        report.summary()
                    } else {
                        String::new()
                    }
                })
                .map_err(Into::into)
            }
            Store::V2(handle) => {
                let mut store = lock_v2(handle);
                let report = store.reconcile().map_err(v2_err)?;
                let summary = if report.has_findings() {
                    report.summary()
                } else {
                    String::new()
                };
                store
                    .interrupt_stale_attempts(STALE_ATTEMPT_NOTE)
                    .map_err(v2_err)?;
                Ok(summary)
            }
        }
    }
}

/// Adopt the recorder's journal into v2 when it holds verified samples.
/// `Ok(Some(id))` — adopted, with `note` (the salvage note) riding along
/// when given. `Ok(None)` — no journal, or one without verified samples
/// (the writer faulted before its first boundary): not an error, the
/// caller stores the take from its encoded WAV instead. Any other failure
/// propagates.
fn adopt_or_none(
    store: &mut StoreV2,
    journal: Option<&recorder::JournalReport>,
    note: Option<&str>,
) -> Result<Option<String>, storage::StorageError> {
    let Some(report) = journal else {
        return Ok(None);
    };
    match store.adopt_journal(&report.path, note) {
        Ok(record) => Ok(Some(record.id)),
        Err(StoreV2Error::Invalid(reason)) if reason.contains("no verified samples") => Ok(None),
        Err(err) => Err(v2_err(err)),
    }
}

/// Page through the v2 listing with bounded reads (page size fixed for
/// phase 1; bounded streaming is later E02 work) and map every record onto
/// the v1-shaped listing. Damaged rows never abort the listing.
fn list_v2(store: &StoreV2) -> Result<Vec<ListedRecord>, storage::StorageError> {
    const PAGE: usize = 200;
    let mut records = Vec::new();
    let mut offset = 0usize;
    loop {
        let page = store.list_records(offset, PAGE).map_err(v2_err)?;
        let total = page.total;
        let listed = page.records.len();
        for record in page.records {
            records.push(match record {
                ListedCapture::Capture(listing) => {
                    let attempts = store.attempts_for(&listing.record.id).map_err(v2_err)?;
                    ListedRecord::Session(v2_summary(&listing.record, &listing.problems, &attempts))
                }
                ListedCapture::Damaged(damaged) => {
                    ListedRecord::Damaged(DamagedRecord {
                        id: damaged.id,
                        reason: damaged.reason,
                    })
                }
            });
        }
        offset += PAGE;
        if offset >= total || listed == 0 {
            break;
        }
    }
    Ok(records)
}

/// Map one v2 capture + its recognition attempts onto the session summary
/// the history list and drawer consume. Recognition state derives from the
/// attempts (v2 keeps lifecycle status out of `captures`): no attempt means
/// captured (or interrupted, when the take itself was), the latest started
/// attempt means transcribing, a failed latest attempt means failed, and a
/// completed final attempt is the current transcript with earlier ones as
/// history — the same reading order the v1 migration writes.
pub(crate) fn v2_summary(
    record: &CaptureRecord,
    problems: &[String],
    attempts: &[AttemptRecord],
) -> SessionSummary {
    let status = match attempts.last() {
        None if record.status == CaptureStatus::Interrupted => SessionStatus::Interrupted,
        None => SessionStatus::Captured,
        Some(attempt) if attempt.status == "started" => SessionStatus::Transcribing,
        // A failed retry keeps the earlier transcript but reports failed,
        // matching the v1 save_failure semantics.
        Some(attempt) if attempt.status == "failed" => SessionStatus::Failed,
        Some(_) => SessionStatus::Transcribed,
    };

    // The latest completed final is the current transcript (earlier ones
    // stay in the attempt rows as history, the same reading order the v1
    // migration writes).
    let transcript = attempts
        .iter()
        .rev()
        .find(|attempt| attempt.is_final_transcript())
        .map(attempt_transcript);

    let mut last_error = attempts
        .last()
        .filter(|attempt| attempt.status == "failed")
        .and_then(AttemptRecord::failure_message);
    if last_error.is_none() && record.status == CaptureStatus::Interrupted {
        last_error = record.recovery_note();
    }
    if !problems.is_empty() {
        let note = problems.join("; ");
        last_error = Some(match last_error {
            Some(existing) => format!("{existing} {note}"),
            None => note,
        });
    }

    let duration_ms = if record.actual_rate > 0 {
        Some(record.frame_count as f64 * 1000.0 / f64::from(record.actual_rate))
    } else {
        None
    };

    SessionSummary {
        id: record.id.clone(),
        created_at: record.created_utc.clone(),
        updated_at: record.created_utc.clone(),
        status,
        duration_ms,
        attempt_count: attempts.len() as u32,
        transcript,
        last_error,
        // The capture id *is* the journal linkage in v2; there is no
        // separate v1 journal to point at.
        journal_id: None,
    }
}

/// The v1-shaped transcript one attempt row carries: the full result when
/// its row preserved one (the shape both the app and the migration write),
/// the bare text otherwise — never a dropped attempt.
fn attempt_transcript(attempt: &AttemptRecord) -> TranscriptionResult {
    attempt.transcript().unwrap_or_else(|| TranscriptionResult {
        text: attempt.text.clone(),
        segments: Vec::new(),
        duration_seconds: None,
        request_id: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fresh scratch directory under the system temp dir, removed first
    /// so reruns start clean. Each test uses its own tag.
    fn scratch_dir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("starling-e02-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create scratch dir");
        dir
    }

    fn v2_store(tag: &str) -> Store {
        let root = scratch_dir(tag);
        let store = StoreV2::open(root.join("v2")).expect("open v2");
        Store::V2(Arc::new(Mutex::new(store)))
    }

    /// A small canonical 16 kHz WAV (the shape every capture path hands
    /// the store).
    fn tiny_wav(samples: usize) -> Arc<Vec<u8>> {
        let pcm = audio::PcmAudio {
            samples: (0..samples).map(|i| (i % 97) as f32 * 0.001).collect(),
            sample_rate: 16_000,
            channels: 1,
        };
        Arc::new(audio::encode_wav_16k(&pcm).expect("encode wav"))
    }

    fn transcript(text: &str) -> TranscriptionResult {
        TranscriptionResult {
            text: text.to_string(),
            segments: Vec::new(),
            duration_seconds: None,
            request_id: None,
        }
    }

    // ---- selection ------------------------------------------------------

    #[test]
    fn the_env_flag_selects_v2_and_wins_over_the_saved_choice() {
        for on in ["1", "true", "YES", " on "] {
            let (kind, reason) = select_backend(Some(on), StorageBackend::V1);
            assert_eq!(kind, StorageKind::V2, "{on:?}");
            assert_eq!(reason, "STARLING_STORAGE_V2 is set");
        }
        // Even when the saved choice already says v2, the flag is the
        // stated reason (it is what a testing override looks like).
        let (kind, _) = select_backend(Some("1"), StorageBackend::V2);
        assert_eq!(kind, StorageKind::V2);
    }

    #[test]
    fn unset_or_off_flags_fall_back_to_the_saved_choice() {
        for off in [None, Some(""), Some("0"), Some("false"), Some("no"), Some("off")] {
            let (kind, reason) = select_backend(off, StorageBackend::V1);
            assert_eq!(kind, StorageKind::V1, "{off:?}");
            assert_eq!(reason, "default");
            let (kind, reason) = select_backend(off, StorageBackend::V2);
            assert_eq!(kind, StorageKind::V2, "{off:?}");
            assert_eq!(reason, "your saved storage choice");
        }
    }

    // ---- the v2 summary mapping ----------------------------------------

    fn attempt(status: &str, final_text: Option<&str>, extra: Option<&str>) -> AttemptRecord {
        AttemptRecord {
            id: format!("a_{status}"),
            capture_id: "c_x".to_string(),
            backend: "starling:parakeet".to_string(),
            model_hash: None,
            language: None,
            options_json: None,
            text: final_text.unwrap_or_default().to_string(),
            partial_or_final: if final_text.is_some() { "final" } else { "partial" }.to_string(),
            status: status.to_string(),
            timing_json: None,
            extra_json: extra.map(str::to_string),
        }
    }

    fn record(status: CaptureStatus, extra: Option<&str>) -> CaptureRecord {
        CaptureRecord {
            id: "c_x".to_string(),
            created_utc: "2026-09-20T10:00:00.000Z".to_string(),
            tz: "UTC".to_string(),
            device: String::new(),
            actual_rate: 16_000,
            policy: "default".to_string(),
            frame_count: 1_600,
            ack_sample_index: 1_600,
            journal_hash: "00".to_string(),
            status,
            retention_class: "standard".to_string(),
            extra_json: extra.map(str::to_string),
        }
    }

    #[test]
    fn summaries_derive_recognition_state_from_attempts() {
        // Fresh take: captured.
        let summary = v2_summary(&record(CaptureStatus::Complete, None), &[], &[]);
        assert_eq!(summary.status, SessionStatus::Captured);
        assert_eq!(summary.attempt_count, 0);
        assert_eq!(summary.duration_ms, Some(100.0));

        // Interrupted take, never recognized: interrupted with its note.
        let summary = v2_summary(
            &record(CaptureStatus::Interrupted, Some(r#"{"recovery":"torn tail"}"#)),
            &[],
            &[],
        );
        assert_eq!(summary.status, SessionStatus::Interrupted);
        assert_eq!(summary.last_error.as_deref(), Some("torn tail"));

        // In flight: transcribing.
        let summary = v2_summary(
            &record(CaptureStatus::Complete, None),
            &[],
            &[attempt("started", None, None)],
        );
        assert_eq!(summary.status, SessionStatus::Transcribing);
        assert_eq!(summary.attempt_count, 1);

        // Completed: transcribed, transcript parsed verbatim (the extra
        // JSON is the camelCase shape the core preserves).
        let extra = r#"{"text":"hello v2","segments":[]}"#;
        let summary = v2_summary(
            &record(CaptureStatus::Complete, None),
            &[],
            &[attempt("completed", Some("hello v2"), Some(&extra))],
        );
        assert_eq!(summary.status, SessionStatus::Transcribed);
        assert_eq!(summary.transcript.as_ref().unwrap().text, "hello v2");

        // A failed retry: failed, but the earlier transcript survives and
        // the failure surfaces — the v1 save_failure semantics.
        let failed = attempt("failed", None, Some(r#"{"error":"offline"}"#));
        let summary = v2_summary(
            &record(CaptureStatus::Complete, None),
            &[],
            &[attempt("completed", Some("hello v2"), Some(&extra)), failed],
        );
        assert_eq!(summary.status, SessionStatus::Failed);
        assert_eq!(summary.transcript.as_ref().unwrap().text, "hello v2");
        assert_eq!(summary.last_error.as_deref(), Some("offline"));

        // A retry of an interrupted take reports the retry, not the
        // interruption (the note stays available via last_error ordering:
        // the attempt error wins while present).
        let summary = v2_summary(
            &record(CaptureStatus::Interrupted, Some(r#"{"recovery":"gap"}"#)),
            &[],
            &[attempt("started", None, None)],
        );
        assert_eq!(summary.status, SessionStatus::Transcribing);
    }

    #[test]
    fn audio_problems_surface_on_the_summary() {
        let summary = v2_summary(
            &record(CaptureStatus::Complete, None),
            &["audio journal is missing".to_string()],
            &[],
        );
        assert!(summary.last_error.as_deref().unwrap().contains("missing"));
    }

    // ---- the facade over a real v2 store (daily path) ------------------

    #[test]
    fn the_v2_facade_covers_the_daily_path_end_to_end() {
        let store = v2_store("facade");
        let wav = tiny_wav(300);

        // Save (no journal: the WAV path through the §4 protocol).
        let id = store
            .save_capture(wav.clone(), Some(300.0), None)
            .expect("save");
        assert!(store.exists(&id).expect("exists"));

        // List: metadata-only, mapped onto the v1 shape.
        let listed = store.list().expect("list");
        assert_eq!(listed.len(), 1);
        let ListedRecord::Session(summary) = &listed[0] else {
            panic!("expected a session summary, got {:?}", listed[0]);
        };
        assert_eq!(summary.id, id);
        assert_eq!(summary.status, SessionStatus::Captured);
        // v2 derives the duration from the stored samples (300 at 16 kHz),
        // not from the caller's wall-clock figure.
        assert_eq!(summary.duration_ms, Some(18.75));

        // Audio on demand: decodes back to the same length and content
        // domain (16 kHz in, 16 kHz out).
        let loaded = store.audio_wav(&id).expect("audio").expect("present");
        let decoded = audio::decode_pcm16_wav(&loaded).expect("decode");
        assert_eq!(decoded.samples.len(), 300);

        // Transcribe lifecycle through the same handles the app uses.
        store.mark_attempt(&id, "starling:parakeet").expect("begin");
        assert_eq!(
            session_status(&store, &id),
            SessionStatus::Transcribing
        );
        store
            .save_transcript(&id, transcript("round trip"))
            .expect("save transcript");
        assert_eq!(session_status(&store, &id), SessionStatus::Transcribed);
        assert_eq!(
            transcript_text(&store, &id).as_deref(),
            Some("round trip")
        );

        // A failed retry: failed status, transcript kept.
        store.mark_attempt(&id, "starling:parakeet").expect("retry");
        store.save_failure(&id, "server offline").expect("failure");
        assert_eq!(session_status(&store, &id), SessionStatus::Failed);
        assert_eq!(transcript_text(&store, &id).as_deref(), Some("round trip"));

        // Delete (R21): gone from the listing, audio unavailable, and a
        // repeat delete is a no-op, not an error.
        store.delete(&id).expect("delete");
        store.delete(&id).expect("delete again");
        assert!(!store.exists(&id).expect("exists"));
        assert!(store.list().expect("list").is_empty());
        assert!(store.audio_wav(&id).expect("missing id").is_none());
    }

    #[test]
    fn a_missing_v2_audio_surfaces_its_error_instead_of_falling_back() {
        let store = v2_store("degrade");
        let id = store
            .save_capture(tiny_wav(50), Some(50.0), None)
            .expect("save");

        // The journal vanishes under the store (disk trouble).
        {
            let Store::V2(handle) = &store else { unreachable!("v2 fixture") };
            let inner = lock_v2(handle);
            let audio = inner.root().join("audio").join(format!("{id}.sj"));
            std::fs::remove_file(&audio).expect("remove journal");
        }

        // The read fails loudly with the store's own reason — never a
        // silent fallback to v1 or a made-up empty recording.
        match store.audio_wav(&id) {
            Err(storage::StorageError::Invalid(reason)) => {
                assert!(reason.contains("no audio journal"), "{reason}")
            }
            other => panic!("expected a surfaced error, got {other:?}"),
        }
    }

    #[test]
    fn startup_recovery_reports_and_repairs_on_v2() {
        let store = v2_store("recovery");
        let id = store
            .save_capture(tiny_wav(80), Some(80.0), None)
            .expect("save");
        // A recognition attempt left "started" by a previous run.
        store.mark_attempt(&id, "starling:parakeet").expect("begin");

        let summary = store.startup_recovery().expect("recovery");
        assert!(summary.is_empty(), "a healthy store has nothing to say");

        // After the pass the stale attempt is failed, ready to retry.
        assert_eq!(session_status(&store, &id), SessionStatus::Failed);
        assert!(transcript_text(&store, &id).is_none());
    }

    fn session_status(store: &Store, id: &str) -> SessionStatus {
        let listed = store.list().expect("list");
        listed
            .iter()
            .find_map(|record| match record {
                ListedRecord::Session(summary) if &summary.id == id => Some(summary.status),
                _ => None,
            })
            .unwrap_or_else(|| panic!("record {id} missing from listing"))
    }

    fn transcript_text(store: &Store, id: &str) -> Option<String> {
        let listed = store.list().expect("list");
        listed
            .iter()
            .find_map(|record| match record {
                ListedRecord::Session(summary) if &summary.id == id => {
                    summary.transcript.as_ref().map(|t| t.text.clone())
                }
                _ => None,
            })
    }
}
