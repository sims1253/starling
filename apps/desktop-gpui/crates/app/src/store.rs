//! The app's storage facade (D14: storage v2 is THE store — there is no
//! backend choice, no flag, and no fallback). The wrapper owns the shared
//! [`StoreV2`] handle and maps its rows onto the v1-shaped types the UI
//! already consumes (list, load audio, save, transcribe bookkeeping,
//! delete). A v2 failure surfaces its error; the documented remedy is
//! fixing the cause and restarting, never a silent switch to anything
//! else.

use std::sync::{Arc, Mutex, MutexGuard};

use starling_dictation::{
    audio, recorder,
    storage::{
        self, DamagedRecord, ListedRecord, SessionStatus, SessionSummary, TranscriptionResult,
    },
    store_v2::{
        self, AttemptRecord, CaptureRecord, CaptureStatus, ListedCapture, RecognitionOutcome,
        StoreV2, StoreV2Error,
    },
};

/// Map a v2 failure onto the app's existing storage error plumbing.
/// [`storage::StorageError::NotFound`] stays intact — the transcription
/// job's delete-race decision keys on exactly that variant (R05) — and an
/// I/O failure keeps its class ([`storage::StorageError::Io`]): a disk
/// fault must not read as "the record was invalid" in diagnostics. The
/// remaining variants (database, schema-too-new, storage, audio) already
/// carry their class in their display text and land in `Invalid` with it
/// preserved.
fn v2_err(err: StoreV2Error) -> storage::StorageError {
    match err {
        StoreV2Error::NotFound(id) => storage::StorageError::NotFound(id),
        StoreV2Error::Invalid(reason) => storage::StorageError::Invalid(reason),
        StoreV2Error::Io(io) => storage::StorageError::Io(io),
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

/// The store: the shared v2 handle behind the daily-use operations.
/// Constructed once at startup; there is nothing to switch to mid-session.
#[derive(Clone)]
pub(crate) struct Store(Arc<Mutex<StoreV2>>);

impl Store {
    /// Open the store at its default data root. This is the app's only
    /// store-opening path (D14): a failure is a startup error for the
    /// caller to surface — there is no fallback backend.
    pub(crate) fn open() -> Result<Store, StoreV2Error> {
        let store = StoreV2::default_root().and_then(StoreV2::open)?;
        Ok(Store(Arc::new(Mutex::new(store))))
    }

    /// Whether a record with this id exists (the R05 delete-race
    /// re-check; metadata-only).
    pub(crate) fn exists(&self, id: &str) -> Result<bool, storage::StorageError> {
        let store = lock_v2(&self.0);
        Ok(store.get_capture(id).map_err(v2_err)?.is_some())
    }

    /// Metadata-only listing (G02): readable records as summaries, damaged
    /// ones flagged with their reason. The store pages through bounded
    /// reads and maps each capture + its recognition attempts onto the
    /// v1-shaped summary the history list consumes.
    pub(crate) fn list(&self) -> Result<Vec<ListedRecord>, storage::StorageError> {
        let store = lock_v2(&self.0);
        list_v2(&store)
    }

    /// One recording's WAV, loaded on demand (G02). `Ok(None)` when the id
    /// is unknown — the delete race is indistinguishable from a missing
    /// record and is not an error. The guard covers only the metadata
    /// lookup; the journal read + verify and the WAV encode — the heavy
    /// half, seconds for a long take — run without the shared handle, so
    /// listing, playback, deletes, and the transcription job's
    /// exists-checks are never pinned behind one load.
    pub(crate) fn audio_wav(&self, id: &str) -> Result<Option<Arc<Vec<u8>>>, storage::StorageError> {
        let path = {
            let store = lock_v2(&self.0);
            match store.audio_journal_path(id) {
                Ok(path) => path,
                Err(StoreV2Error::NotFound(_)) => return Ok(None),
                Err(err) => return Err(v2_err(err)),
            }
        };
        let journal = store_v2::read_audio_journal(&path).map_err(v2_err)?;
        let pcm = audio::PcmAudio {
            samples: journal.samples,
            sample_rate: journal.sample_rate,
            channels: 1,
        };
        let wav = audio::encode_wav_16k(&pcm)
            .map_err(|err| storage::StorageError::Invalid(err.to_string()))?;
        Ok(Some(Arc::new(wav)))
    }

    /// Persists a finished take: the recorder's journal is adopted as the
    /// v2 audio evidence when it has verified samples (moved, never
    /// re-encoded); otherwise the take is written from its encoded WAV
    /// through the full §4 protocol. The stored duration is derived from
    /// the samples themselves, never the caller's wall clock. Returns the
    /// record id and the WAV any transcription must run on: when the
    /// journal was adopted that is the *stored* audio (the evidence every
    /// retry loads), so the first transcript and its retries can never
    /// disagree — a faulted journal holds fewer samples than the
    /// in-memory take.
    pub(crate) fn save_capture(
        &self,
        wav: Arc<Vec<u8>>,
        journal: Option<&recorder::JournalReport>,
    ) -> Result<SavedTake, storage::StorageError> {
        let adopted = {
            let mut store = lock_v2(&self.0);
            try_adopt(&mut store, journal, None)
        };
        if let Ok(Some(id)) = adopted {
            // A failure to load what was just committed would be odd, but
            // the take is saved either way — fall back to the caller's WAV
            // rather than failing the save over the transcript source.
            let wav = self.audio_wav(&id).ok().flatten().unwrap_or(wav);
            return Ok(SavedTake { id, wav });
        }
        // No journal evidence — or a journal-level failure (the store
        // itself is typically healthy): the fully encoded WAV is in hand,
        // so the take is stored from it; a journal problem never costs the
        // audio.
        let pcm = decode_wav(&wav)?;
        let id = self
            .save_pcm_take(pcm, store_v2::CommitMark::Complete)
            .map_err(|err| join_adopt_failure(adopted, err))?;
        Ok(SavedTake { id, wav })
    }

    /// Persists a salvaged take as interrupted-but-usable with `note`
    /// stating exactly what survived (I1 phase 2). The interruption is
    /// written *with* the take: on the journal path adoption carries the
    /// note and the status is forced under the same guard; on the WAV path
    /// the row is committed already-interrupted in one transaction — there
    /// is no separate status update a crash could skip (R34).
    pub(crate) fn save_interrupted_capture(
        &self,
        wav: Arc<Vec<u8>>,
        journal: Option<&recorder::JournalReport>,
        note: &str,
    ) -> Result<String, storage::StorageError> {
        let adopted = {
            let mut store = lock_v2(&self.0);
            let adopted = try_adopt(&mut store, journal, Some(note));
            if let Ok(Some(id)) = &adopted {
                // A salvaged take is interrupted no matter what its journal
                // looked like (R34): the QuiesceTimeout path carries an
                // already-finalized journal — the writer's exit path closed
                // it — which adoption alone marks Complete. Interrupted-ness
                // derives from the salvage, not the journal's finalized-ness.
                // The note itself is already stored by the adoption (merged
                // with any torn-tail wording under `extra_json.recovery`), so
                // this only forces the status — passing no note keeps that
                // combined wording intact.
                store
                    .update_capture_status(id, CaptureStatus::Interrupted, None)
                    .map_err(v2_err)?;
                return Ok(id.clone());
            }
            adopted
        };
        // Fall-through WAV path (no journal, or an unusable one — see
        // [`try_adopt`]).
        let pcm = decode_wav(&wav)?;
        self.save_pcm_take(
            pcm,
            store_v2::CommitMark::Interrupted {
                note: note.to_string(),
            },
        )
        .map_err(|err| join_adopt_failure(adopted, err))
    }

    /// Marks a transcription attempt as started on the record. `backend`
    /// labels the attempt ("starling:parakeet", "openai:whisper-large-v3").
    pub(crate) fn mark_attempt(&self, id: &str, backend: &str) -> Result<(), storage::StorageError> {
        let mut store = lock_v2(&self.0);
        store
            .begin_recognition(id, backend, None)
            .map(|_| ())
            .map_err(v2_err)
    }

    /// Records a successful transcript: the in-flight attempt is completed
    /// with the transcript preserved verbatim in its row.
    pub(crate) fn save_transcript(
        &self,
        id: &str,
        transcript: TranscriptionResult,
    ) -> Result<(), storage::StorageError> {
        let mut store = lock_v2(&self.0);
        store
            .finish_recognition_transcript(id, &transcript)
            .map_err(v2_err)
    }

    /// Records a failed transcription attempt.
    pub(crate) fn save_failure(&self, id: &str, message: &str) -> Result<(), storage::StorageError> {
        let mut store = lock_v2(&self.0);
        store
            .finish_recognition(id, RecognitionOutcome::Failed { message })
            .map_err(v2_err)
    }

    /// Confirmed deletion (R21): the audio journal is quarantined and the
    /// row tombstoned — resurrection-proof.
    pub(crate) fn delete(&self, id: &str) -> Result<(), storage::StorageError> {
        let mut store = lock_v2(&self.0);
        store.delete_capture(id).map_err(v2_err)
    }

    /// The startup recovery pass: reconcile journals against the metadata
    /// rows (§4), then fail recognition attempts still marked started by a
    /// previous run. Returns a user-facing summary string, empty when
    /// there was nothing to report. The two repairs are independent
    /// (journal state vs. recognition-attempt rows), so both always run —
    /// a reconciliation failure must not strand records stuck in
    /// "Transcribing", and vice versa; when both fail the errors combine.
    pub(crate) fn startup_recovery(&self) -> Result<String, storage::StorageError> {
        /// The note a stale recognition attempt gets at startup — same
        /// wording as the v1 "stuck in Transcribing" fix.
        const STALE_ATTEMPT_NOTE: &str =
            "Interrupted before the server returned a transcript. Your audio is ready to retry.";
        let (reconciled, staled) = {
            let mut store = lock_v2(&self.0);
            (
                store.reconcile(),
                store.interrupt_stale_attempts(STALE_ATTEMPT_NOTE),
            )
        };
        match (reconciled.map_err(v2_err), staled.map_err(v2_err)) {
            (Ok(report), Ok(_)) => {
                let summary = if report.has_findings() {
                    report.summary()
                } else {
                    String::new()
                };
                Ok(summary)
            }
            (Err(reconcile_err), Err(stale_err)) => Err(storage::StorageError::Invalid(format!(
                "{reconcile_err}; {stale_err}"
            ))),
            (Err(err), Ok(_)) | (Ok(_), Err(err)) => Err(err),
        }
    }

    /// Write a take from decoded PCM through the §4 protocol with the
    /// guard held only for the cheap steps — minting the staging journal
    /// and the metadata commit. The bulk journal writes and fsyncs run
    /// through the take's own writer, off the shared handle, so one long
    /// take's save cannot pin every other store call behind it.
    fn save_pcm_take(
        &self,
        pcm: audio::PcmAudio,
        mark: store_v2::CommitMark,
    ) -> Result<String, storage::StorageError> {
        let rate = pcm.sample_rate;
        let mut take = {
            let store = lock_v2(&self.0);
            store.begin_take_at_rate(rate, store_v2::TakeMeta::for_device(""))
        }
        .map_err(v2_err)?;
        take.append_and_seal(&pcm.samples).map_err(v2_err)?;
        let finalized = take.finalize().map_err(v2_err)?;
        let mut store = lock_v2(&self.0);
        let committed = finalized.commit_marked(&mut store, mark).map_err(v2_err)?;
        Ok(committed.record.id)
    }
}

/// What a persisted take hands back to the pipeline: the record id, and
/// the WAV any transcription of it must run on. When the recorder's
/// journal was adopted, that is the stored audio — the evidence every
/// retry loads — so the first transcript and later retries transcribe the
/// same bytes ([`Store::save_capture`]).
pub(crate) struct SavedTake {
    pub(crate) id: String,
    pub(crate) wav: Arc<Vec<u8>>,
}

/// Try to adopt the recorder's journal into v2. `Ok(Some(id))` — adopted,
/// with `note` (the salvage note) riding along when given. `Ok(None)` — no
/// journal to adopt. `Err(reason)` — adoption failed (an unreadable
/// source, a destination conflict, I/O or database trouble), and that is
/// **never fatal to the save**: the caller still holds the fully encoded
/// WAV and stores the take from it instead, so a journal-level problem
/// cannot cost the audio. `reason` resurfaces only if the WAV write also
/// fails (see [`join_adopt_failure`]). This includes
/// [`StoreV2Error::NoVerifiedSamples`] — a header-only journal from a
/// writer that faulted before its first boundary — which is the common
/// no-evidence case, not a fault.
fn try_adopt(
    store: &mut StoreV2,
    journal: Option<&recorder::JournalReport>,
    note: Option<&str>,
) -> Result<Option<String>, String> {
    let Some(report) = journal else {
        return Ok(None);
    };
    match store.adopt_journal(&report.path, note) {
        Ok(record) => Ok(Some(record.id)),
        Err(err) => Err(err.to_string()),
    }
}

/// Fold a journal-adoption failure into a later WAV-path failure: the take
/// was stored from neither, and the surfaced error must say both — the WAV
/// error alone would hide the adoption trouble that forced the fall-back.
fn join_adopt_failure(
    adopted: Result<Option<String>, String>,
    wav_err: storage::StorageError,
) -> storage::StorageError {
    match adopted {
        Ok(_) => wav_err,
        Err(reason) => storage::StorageError::Invalid(format!(
            "journal adoption failed ({reason}); storing the take from its encoded WAV \
             failed too: {wav_err}"
        )),
    }
}

/// Decode the caller's canonical WAV — pure CPU work done *outside* the
/// store lock; the shared handle must not sit behind it.
fn decode_wav(wav: &[u8]) -> Result<audio::PcmAudio, storage::StorageError> {
    let pcm = audio::decode_pcm16_wav(wav)
        .map_err(|err| storage::StorageError::Invalid(err.to_string()))?;
    if pcm.samples.is_empty() {
        return Err(storage::StorageError::Invalid(
            "wav has no samples; nothing to store".to_string(),
        ));
    }
    Ok(pcm)
}

/// Page through the v2 listing with bounded reads (page size fixed;
/// bounded streaming is later work) and map every record onto the
/// v1-shaped listing. Damaged rows never abort the listing.
fn list_v2(store: &StoreV2) -> Result<Vec<ListedRecord>, storage::StorageError> {
    const PAGE: usize = 200;
    let mut records = Vec::new();
    let mut offset = 0usize;
    loop {
        let page = store.list_records(offset, PAGE).map_err(v2_err)?;
        let total = page.total;
        let listed = page.records.len();
        // One attempts query per page, not per record: this listing runs
        // after every save, transcript, failure, and delete, and the
        // per-record round-trips added up.
        let ids: Vec<String> = page
            .records
            .iter()
            .filter_map(|record| match record {
                ListedCapture::Capture(listing) => Some(listing.record.id.clone()),
                ListedCapture::Damaged(_) => None,
            })
            .collect();
        let attempts = store
            .attempts_grouped_by_capture(&ids)
            .map_err(v2_err)?;
        for record in page.records {
            records.push(match record {
                ListedCapture::Capture(listing) => {
                    let attempts = attempts
                        .get(&listing.record.id)
                        .cloned()
                        .unwrap_or_default();
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
        // The whole listing runs under one guard, so the pages see a
        // single snapshot in-process; the short-page bound is belt and
        // braces for if that ever changes — and it avoids the one extra
        // empty query an exact-multiple history would otherwise cost.
        if listed < PAGE || offset >= total {
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
/// history.
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
    // stay in the attempt rows as history).
    let transcript = attempts
        .iter()
        .rev()
        .find(|attempt| attempt.is_final_transcript())
        .map(attempt_transcript);

    let mut last_error = attempts
        .last()
        .filter(|attempt| attempt.status == "failed")
        .and_then(AttemptRecord::failure_message);
    if record.status == CaptureStatus::Interrupted {
        // The gap note survives alongside any attempt outcome: the user
        // should always know part of the audio was discarded, whether or
        // not a later recognition attempt also failed or is in flight.
        if let Some(note) = record.recovery_note() {
            last_error = Some(match last_error {
                Some(existing) => format!("{existing} {note}"),
                None => note,
            });
        }
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
        // A real updated-at: the latest attempt's creation time, falling
        // back to the capture's creation time when no attempt (or a
        // pre-schema-v2 attempt row without a timestamp) exists.
        updated_at: attempts
            .last()
            .and_then(|attempt| attempt.created_utc.clone())
            .unwrap_or_else(|| record.created_utc.clone()),
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
/// its row preserved one (the shape the app writes), the bare text
/// otherwise — never a dropped attempt.
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
    /// so reruns start clean. Unique per call, not just per tag + process:
    /// two helper invocations sharing a tag inside one test would otherwise
    /// wipe each other's state (the remove-first is the hazard), so a
    /// counter makes collisions impossible.
    fn scratch_dir(tag: &str) -> std::path::PathBuf {
        static CALL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "starling-e02-{tag}-{}-{}",
            std::process::id(),
            CALL.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create scratch dir");
        dir
    }

    fn v2_store(tag: &str) -> Store {
        let root = scratch_dir(tag);
        let store = StoreV2::open(root.join("v2")).expect("open v2");
        Store(Arc::new(Mutex::new(store)))
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

    // ---- the v2 summary mapping ----------------------------------------

    fn attempt(
        status: &str,
        final_text: Option<&str>,
        extra: Option<&str>,
        created_utc: Option<&str>,
    ) -> AttemptRecord {
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
            created_utc: created_utc.map(str::to_string),
        }
    }

    /// All attempt fixtures below default to no recorded timestamp
    /// (pre-schema-v2 rows); tests that care pass one.
    fn plain_attempt(status: &str, final_text: Option<&str>, extra: Option<&str>) -> AttemptRecord {
        attempt(status, final_text, extra, None)
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
            &[plain_attempt("started", None, None)],
        );
        assert_eq!(summary.status, SessionStatus::Transcribing);
        assert_eq!(summary.attempt_count, 1);

        // Completed: transcribed, transcript parsed verbatim (the extra
        // JSON is the camelCase shape the core preserves).
        let extra = r#"{"text":"hello v2","segments":[]}"#;
        let summary = v2_summary(
            &record(CaptureStatus::Complete, None),
            &[],
            &[plain_attempt("completed", Some("hello v2"), Some(&extra))],
        );
        assert_eq!(summary.status, SessionStatus::Transcribed);
        assert_eq!(summary.transcript.as_ref().unwrap().text, "hello v2");

        // A failed retry: failed, but the earlier transcript survives and
        // the failure surfaces — the v1 save_failure semantics.
        let failed = plain_attempt("failed", None, Some(r#"{"error":"offline"}"#));
        let summary = v2_summary(
            &record(CaptureStatus::Complete, None),
            &[],
            &[plain_attempt("completed", Some("hello v2"), Some(&extra)), failed],
        );
        assert_eq!(summary.status, SessionStatus::Failed);
        assert_eq!(summary.transcript.as_ref().unwrap().text, "hello v2");
        assert_eq!(summary.last_error.as_deref(), Some("offline"));

        // A retry of an interrupted take reports the retry as its status,
        // and the recovery note rides alongside the attempt outcome — the
        // user never loses sight of the gap (R34).
        let summary = v2_summary(
            &record(CaptureStatus::Interrupted, Some(r#"{"recovery":"gap"}"#)),
            &[],
            &[plain_attempt("started", None, None)],
        );
        assert_eq!(summary.status, SessionStatus::Transcribing);
        assert_eq!(summary.last_error.as_deref(), Some("gap"));

        // A failed retry of an interrupted take surfaces both sentences.
        let failed = plain_attempt("failed", None, Some(r#"{"error":"offline"}"#));
        let summary = v2_summary(
            &record(CaptureStatus::Interrupted, Some(r#"{"recovery":"gap"}"#)),
            &[],
            &[failed],
        );
        assert_eq!(summary.status, SessionStatus::Failed);
        let last_error = summary.last_error.expect("both messages");
        assert!(last_error.starts_with("offline"), "{last_error}");
        assert!(last_error.ends_with("gap"), "{last_error}");
    }

    #[test]
    fn updated_at_is_the_latest_attempt_time_not_the_creation_time() {
        // No attempts (or none with a timestamp — pre-schema-v2 rows):
        // falls back to the capture's creation time.
        let summary = v2_summary(
            &record(CaptureStatus::Complete, None),
            &[],
            &[plain_attempt("failed", None, Some(r#"{"error":"x"}"#))],
        );
        assert_eq!(summary.created_at, "2026-09-20T10:00:00.000Z");
        assert_eq!(summary.updated_at, "2026-09-20T10:00:00.000Z");

        // A timestamped attempt moves the updated-at with it.
        let summary = v2_summary(
            &record(CaptureStatus::Complete, None),
            &[],
            &[
                attempt(
                    "completed",
                    Some("first"),
                    None,
                    Some("2026-09-20T10:00:05.000Z"),
                ),
                attempt("started", None, None, Some("2026-09-20T10:42:00.000Z")),
            ],
        );
        assert_eq!(summary.created_at, "2026-09-20T10:00:00.000Z");
        assert_eq!(summary.updated_at, "2026-09-20T10:42:00.000Z");
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
        let saved = store.save_capture(wav.clone(), None).expect("save");
        let id = saved.id;
        assert!(store.exists(&id).expect("exists"));
        // The WAV path hands the caller's own bytes back as the transcript
        // source — the store wrote exactly those samples.
        assert!(Arc::ptr_eq(&saved.wav, &wav));

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

    /// A journal the recorder finalized before handing it over — the
    /// QuiesceTimeout salvage shape (the writer's exit path closes the
    /// journal even when the device never quiesced). Built through a
    /// scratch store's public take protocol: `begin_take` + `finalize`
    /// leaves the finished journal in the scratch root's `staging/`, the
    /// same on-disk shape the recorder produces.
    fn finalized_journal(tag: &str, samples: &[f32]) -> recorder::JournalReport {
        let root = scratch_dir(tag);
        let scratch = StoreV2::open(root.join("scratch")).expect("scratch store");
        let mut take = scratch
            .begin_take(store_v2::TakeMeta::for_device("test"))
            .expect("begin take");
        take.append_frames(samples).expect("append");
        take.write_boundary().expect("boundary");
        let finalized = take.finalize().expect("finalize");
        recorder::JournalReport {
            path: root
                .join("scratch")
                .join("staging")
                .join(format!("{}.sj", finalized.id)),
            id: finalized.id,
            sample_rate: finalized.sample_rate,
            acknowledged_samples: finalized.total_samples,
            finalized: true,
            fault: None,
        }
    }

    #[test]
    fn a_quiesce_salvaged_take_with_a_finalized_journal_is_still_interrupted() {
        // R34 regression: adoption alone marks a finalized, intact journal
        // Complete — but a take saved through the salvage path is
        // interrupted regardless, exactly as the v1 mark did, and its
        // salvage note must surface.
        let store = v2_store("quiesce-salvage");
        let samples: Vec<f32> = (0..120).map(|i| (i % 31) as f32 * 0.002).collect();
        let report = finalized_journal("quiesce-salvage-src", &samples);

        let id = store
            .save_interrupted_capture(
                tiny_wav(120),
                Some(&report),
                "all captured samples were salvaged and kept as this interrupted recording",
            )
            .expect("salvaged save");

        // The journal moved in and became the stored audio.
        assert!(!report.path.exists());
        assert!(
            lock_v2(&store.0)
                .load_audio(&id)
                .expect("audio")
                .finalized
        );

        // The listing shows the take as interrupted with its note — not
        // "Captured" with the note invisible.
        let summary = summary_of(&store, &id);
        assert_eq!(summary.status, SessionStatus::Interrupted);
        let last_error = summary.last_error.expect("the salvage note surfaces");
        assert!(last_error.contains("salvaged and kept"), "{last_error}");
    }

    #[test]
    fn a_cleanly_stopped_take_with_a_finalized_journal_stays_captured() {
        // The mirror of the R34 fix: forcing interrupted-ness belongs to
        // the salvage path only — a normal completion that adopts a
        // finalized journal stays a complete take.
        let store = v2_store("clean-adopt");
        let samples: Vec<f32> = (0..90).map(|i| (i % 17) as f32 * 0.003).collect();
        let report = finalized_journal("clean-adopt-src", &samples);

        let id = store
            .save_capture(tiny_wav(90), Some(&report))
            .expect("clean save")
            .id;

        let summary = summary_of(&store, &id);
        assert_eq!(summary.status, SessionStatus::Captured);
        assert_eq!(summary.last_error, None);
    }

    #[test]
    fn a_missing_v2_audio_surfaces_its_error_instead_of_falling_back() {
        let store = v2_store("degrade");
        let id = store
            .save_capture(tiny_wav(50), None)
            .expect("save")
            .id;

        // The journal vanishes under the store (disk trouble).
        {
            let inner = lock_v2(&store.0);
            let audio = inner.root().join("audio").join(format!("{id}.sj"));
            std::fs::remove_file(&audio).expect("remove journal");
        }

        // The read fails loudly with the store's own reason — never a
        // silent fallback to another store or a made-up empty recording.
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
            .save_capture(tiny_wav(80), None)
            .expect("save")
            .id;
        // A recognition attempt left "started" by a previous run.
        store.mark_attempt(&id, "starling:parakeet").expect("begin");

        let summary = store.startup_recovery().expect("recovery");
        assert!(summary.is_empty(), "a healthy store has nothing to say");

        // After the pass the stale attempt is failed, ready to retry.
        assert_eq!(session_status(&store, &id), SessionStatus::Failed);
        assert!(transcript_text(&store, &id).is_none());
    }

    #[test]
    fn an_io_failure_keeps_its_class_at_the_boundary() {
        // A disk fault must surface as an I/O error, not "invalid record"
        // — the class is what tells the user (and any caller branching on
        // it) what actually went wrong.
        let mapped = v2_err(StoreV2Error::Io(std::io::Error::other("disk on fire")));
        match mapped {
            storage::StorageError::Io(err) => {
                assert!(err.to_string().contains("disk on fire"), "{err}")
            }
            other => panic!("expected an Io error, got {other:?}"),
        }
        // Variants without a StorageError counterpart keep their class in
        // the message: a schema refusal must read as a schema problem.
        let mapped = v2_err(StoreV2Error::SchemaTooNew {
            found: 99,
            supported: 2,
        });
        match mapped {
            storage::StorageError::Invalid(reason) => {
                assert!(reason.contains("schema version 99"), "{reason}")
            }
            other => panic!("expected an Invalid error, got {other:?}"),
        }
    }

    #[test]
    fn an_adopted_take_transcribes_the_stored_audio_not_the_caller_wav() {
        // The adopted journal is the stored evidence every retry loads. A
        // faulted writer can leave the journal shorter than the in-memory
        // take, so the first transcript must run on the stored bytes too —
        // otherwise attempt one and its retry transcribe different audio.
        let store = v2_store("adopt-consistent");
        let samples: Vec<f32> = (0..120).map(|i| (i % 23) as f32 * 0.004).collect();
        let report = finalized_journal("adopt-consistent-src", &samples);

        // The caller's WAV carries more samples than the journal does.
        let saved = store
            .save_capture(tiny_wav(300), Some(&report))
            .expect("adopting save");

        let decoded = audio::decode_pcm16_wav(&saved.wav).expect("decode");
        assert_eq!(decoded.samples.len(), 120, "the stored journal's samples");
        // And exactly what a retry loads, so the two can never diverge.
        let reloaded = store.audio_wav(&saved.id).expect("reload").expect("present");
        let reloaded = audio::decode_pcm16_wav(&reloaded).expect("decode");
        assert_eq!(reloaded.samples, decoded.samples);
    }

    #[test]
    fn an_unusable_journal_still_saves_the_take_from_its_wav() {
        // An adoption failure is a journal-level problem; the store is
        // healthy and the fully encoded WAV is in hand. The save must not
        // abort (that would drop the take into the memory-only stash) — it
        // falls through to the WAV path.
        let store = v2_store("adopt-fallback");
        let junk_root = scratch_dir("adopt-fallback-junk");
        let junk_path = junk_root.join("j_unreadable.sj");
        std::fs::write(&junk_path, b"not a journal at all").expect("write junk journal");
        let report = recorder::JournalReport {
            path: junk_path.clone(),
            id: "j_unreadable".to_string(),
            sample_rate: 16_000,
            acknowledged_samples: 0,
            finalized: true,
            fault: None,
        };

        let saved = store
            .save_capture(tiny_wav(90), Some(&report))
            .expect("the take is saved from its WAV");

        assert_ne!(saved.id, "j_unreadable", "a fresh take, not the journal id");
        let decoded = audio::decode_pcm16_wav(&saved.wav).expect("decode");
        assert_eq!(decoded.samples.len(), 90);
        let summary = summary_of(&store, &saved.id);
        assert_eq!(summary.status, SessionStatus::Captured);
        // The unusable journal was left where it was.
        assert!(junk_path.exists());
    }

    #[test]
    fn a_wav_path_salvage_lands_interrupted_with_its_note() {
        // The no-journal salvage path: the interruption and its note are
        // committed with the take (one transaction — R34 leaves no crash
        // window between "saved" and "marked").
        let store = v2_store("wav-salvage");
        let id = store
            .save_interrupted_capture(tiny_wav(60), None, "kept from memory after the fault")
            .expect("salvage save");

        let summary = summary_of(&store, &id);
        assert_eq!(summary.status, SessionStatus::Interrupted);
        let last_error = summary.last_error.expect("the salvage note surfaces");
        assert!(last_error.contains("kept from memory"), "{last_error}");
        // The audio is playable: the stored take round-trips.
        let loaded = store.audio_wav(&id).expect("audio").expect("present");
        let decoded = audio::decode_pcm16_wav(&loaded).expect("decode");
        assert_eq!(decoded.samples.len(), 60);
    }

    #[test]
    fn startup_recovery_repairs_stale_attempts_even_when_reconciliation_fails() {
        // The two startup repairs are independent: a reconciliation failure
        // must not leave records stuck in "Transcribing" from a previous
        // run — the stale-attempt repair runs anyway and the reconcile
        // error still surfaces.
        let store = v2_store("reconcile-fail");
        let stuck = store
            .save_capture(tiny_wav(70), None)
            .expect("save")
            .id;
        store.mark_attempt(&stuck, "starling:parakeet").expect("begin");

        // Sabotage reconciliation only: a tombstoned capture whose journal
        // was resurrected under audio/ while its quarantine destination is
        // a directory — the tombstone-completion rename cannot succeed, so
        // reconcile errors while the rest of the store stays healthy.
        let doomed = store
            .save_capture(tiny_wav(30), None)
            .expect("save")
            .id;
        store.delete(&doomed).expect("delete");
        {
            let inner = lock_v2(&store.0);
            let audio = inner.root().join("audio").join(format!("{doomed}.sj"));
            let quarantine = inner.root().join("quarantine").join(format!("{doomed}.sj"));
            std::fs::copy(&quarantine, &audio).expect("resurrect the journal");
            std::fs::remove_file(&quarantine).expect("clear the destination");
            std::fs::create_dir(&quarantine).expect("the destination is now a directory");
        }

        assert!(
            store.startup_recovery().is_err(),
            "the sabotaged reconciliation must surface its error"
        );
        // …but the stale attempt was still repaired: the stuck take is
        // retryable (failed with its note), not still "Transcribing".
        assert_eq!(session_status(&store, &stuck), SessionStatus::Failed);
        let summary = summary_of(&store, &stuck);
        let last_error = summary.last_error.expect("the stale-attempt note");
        assert!(last_error.contains("ready to retry"), "{last_error}");
    }

    fn session_status(store: &Store, id: &str) -> SessionStatus {
        summary_of(store, id).status
    }

    fn summary_of(store: &Store, id: &str) -> SessionSummary {
        let listed = store.list().expect("list");
        listed
            .iter()
            .find_map(|record| match record {
                ListedRecord::Session(summary) if &summary.id == id => Some(summary.clone()),
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
