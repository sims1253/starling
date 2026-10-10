//! The app's storage facade (D14: storage v2 is THE store — there is no
//! backend choice, no flag, and no fallback). The wrapper owns the shared
//! [`StoreV2`] handle and maps its rows onto the v1-shaped types the UI
//! already consumes (list, load audio, save, transcribe bookkeeping,
//! delete). A v2 failure surfaces its error; the documented remedy is
//! fixing the cause and restarting, never a silent switch to anything
//! else.

use std::sync::{Arc, Mutex, MutexGuard};

use starling_dictation::{
    audio,
    storage::{
        self, DamagedRecord, ListedRecord, SessionStatus, SessionSummary, TranscriptionResult,
    },
    store_v2::{
        self, AttemptRecord, AudioAtRest, CaptureRecord, CaptureStatus, CompressionOutcome,
        HoldReason, ListedCapture, RecognitionOutcome, RetentionPolicy, RevisionRow, StoreV2,
        StoreV2Error,
    },
};

/// The `documents.name` of a take's processing document.
const PROCESSING_DOC: &str = "processing";

/// A take's durable processing state (#295).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ProcessingDoc {
    pub head_revision: u64,
    pub head_text: String,
    pub head_is_raw: bool,
    pub raw_attempt_id: String,
    pub raw_text: String,
    pub proposals: Vec<ProposalRow>,
    /// The accepted proposal the head derives from, as the head records it.
    pub accepted_request: Option<String>,
}

/// What produced a proposal, pinned at request time for the correction
/// records: the mode, the provider, and the pipeline's stage timings.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub(crate) struct ProposalOrigin {
    pub mode_id: String,
    pub mode_version: u32,
    pub provider_id: String,
    pub provider_kind: String,
    pub provider_model: String,
    pub locality: String,
    pub transform_kinds: Vec<String>,
    pub language: Option<String>,
    pub queued_ms: f64,
    pub processing_ms: f64,
}

/// One processing result as stored: a proposal pinned to the head
/// revision its request read, or the failure it ended in.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub(crate) struct ProposalRow {
    pub request_id: String,
    pub base_revision: u64,
    pub text: String,
    pub status: RowStatus,
    /// Where the text went, for the drawer ("S1-mini · this computer").
    pub label: String,
    pub failure: Option<String>,
    pub stop_to_result_ms: Option<f64>,
    /// `None` on rows written before provenance was recorded.
    pub origin: Option<ProposalOrigin>,
}

/// The disposition of a stored processing result.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum RowStatus {
    Proposed,
    Accepted,
    Rejected,
    Superseded,
    Failed,
}

impl RowStatus {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            RowStatus::Proposed => "proposed",
            RowStatus::Accepted => "accepted",
            RowStatus::Rejected => "rejected",
            RowStatus::Superseded => "superseded",
            RowStatus::Failed => "failed",
        }
    }

    /// `None` for a value this build does not know; such a row is skipped.
    pub(crate) fn parse(value: &str) -> Option<RowStatus> {
        [
            RowStatus::Proposed,
            RowStatus::Accepted,
            RowStatus::Rejected,
            RowStatus::Superseded,
            RowStatus::Failed,
        ]
        .into_iter()
        .find(|status| status.as_str() == value)
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
struct ProposalProvenance {
    label: String,
    failure: Option<String>,
    stop_to_result_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    origin: Option<ProposalOrigin>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct HeadSources {
    attempt_id: String,
    /// The accepted proposal this head derives from.
    request_id: Option<String>,
}

fn head_rev_id(id: &str, revision: u64) -> String {
    format!("{id}#h{revision}")
}

fn head_row(id: &str, revision: u64, text: &str, is_raw: bool, attempt_id: &str, request_id: Option<&str>) -> RevisionRow {
    RevisionRow {
        rev_id: head_rev_id(id, revision),
        doc_id: id.to_string(),
        base_revision: revision.checked_sub(1).filter(|base| *base > 0),
        sources_json: serde_json::to_string(&HeadSources {
            attempt_id: attempt_id.to_string(),
            request_id: request_id.map(str::to_string),
        })
        .ok(),
        text: text.to_string(),
        status: if is_raw { "raw" } else { "processed" }.to_string(),
        provenance: None,
        disposition: Some("committed".to_string()),
    }
}

impl ProposalRow {
    fn to_row(&self, id: &str) -> RevisionRow {
        RevisionRow {
            rev_id: format!("{id}#p:{}", self.request_id),
            doc_id: id.to_string(),
            base_revision: Some(self.base_revision),
            sources_json: serde_json::to_string(&serde_json::json!({ "request_id": self.request_id })).ok(),
            text: self.text.clone(),
            status: self.status.as_str().to_string(),
            provenance: serde_json::to_string(&ProposalProvenance {
                label: self.label.clone(),
                failure: self.failure.clone(),
                stop_to_result_ms: self.stop_to_result_ms,
                origin: self.origin.clone(),
            })
            .ok(),
            disposition: Some("proposal".to_string()),
        }
    }
}

impl ProcessingDoc {
    fn from_row(id: &str, document: store_v2::DocumentRow) -> Option<ProcessingDoc> {
        let head = document
            .revisions
            .iter()
            .find(|row| row.rev_id == head_rev_id(id, document.head_revision))?;
        let raw = document.revisions.iter().find(|row| row.rev_id == head_rev_id(id, 1))?;
        let sources: HeadSources = serde_json::from_str(raw.sources_json.as_deref()?).ok()?;
        let accepted_request = head
            .sources_json
            .as_deref()
            .and_then(|json| serde_json::from_str::<HeadSources>(json).ok())
            .and_then(|sources| sources.request_id)
            .filter(|_| head.status != "raw");
        let proposals = document
            .revisions
            .iter()
            .filter(|row| row.disposition.as_deref() == Some("proposal"))
            .filter_map(|row| {
                let provenance: ProposalProvenance =
                    serde_json::from_str(row.provenance.as_deref()?).ok()?;
                let request_id = row.rev_id.split_once("#p:")?.1.to_string();
                Some(ProposalRow {
                    request_id,
                    base_revision: row.base_revision?,
                    text: row.text.clone(),
                    status: RowStatus::parse(&row.status)?,
                    label: provenance.label,
                    failure: provenance.failure,
                    stop_to_result_ms: provenance.stop_to_result_ms,
                    origin: provenance.origin,
                })
            })
            .collect();
        Some(ProcessingDoc {
            head_revision: document.head_revision,
            head_text: head.text.clone(),
            head_is_raw: head.status == "raw",
            raw_attempt_id: sources.attempt_id,
            raw_text: raw.text.clone(),
            proposals,
            accepted_request,
        })
    }
}

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

/// See [`Store::recognition`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Recognition {
    /// Nothing transcribed it, and nothing alive is transcribing it.
    Due,
    /// A live process (another app, or this one) is transcribing it.
    InFlight,
    /// It has a transcript.
    Done,
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
    #[cfg(test)]
    pub(crate) fn at_test_root(root: &std::path::Path) -> Self {
        Self(Arc::new(Mutex::new(StoreV2::open(root).unwrap())))
    }

    /// A short silent take marked as captured against a secure field
    /// (the desktop app has no such capture path).
    #[cfg(test)]
    pub(crate) fn save_secure_take(&self) -> String {
        let mut meta = store_v2::TakeMeta::for_device("");
        meta.secure_field = true;
        let mut v2 = lock_v2(&self.0);
        let mut take = v2.begin_take_at_rate(16_000, meta).unwrap();
        take.append_and_seal(&[0.0; 160]).unwrap();
        let finalized = take.finalize().unwrap();
        finalized
            .commit_marked(&mut v2, store_v2::CommitMark::Complete)
            .unwrap()
            .record
            .id
    }

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

    /// Whether recording `id` still waits for its transcript (history
    /// shows it as being sent): no attempt yet, or one still running.
    /// `false` once it is gone.
    pub(crate) fn pending(&self, id: &str) -> Result<bool, storage::StorageError> {
        let store = lock_v2(&self.0);
        let Some(record) = store.get_capture(id).map_err(v2_err)? else {
            return Ok(false);
        };
        let attempts = store.attempts_for(id).map_err(v2_err)?;
        let summary = v2_summary(&record, &[], &attempts);
        Ok(matches!(
            summary.status,
            SessionStatus::Captured | SessionStatus::Transcribing
        ))
    }

    /// Where stored take `id`'s transcription stands, across processes:
    /// what a window the host offers the take decides by (#220), since an
    /// app that transcribed it may have gone before telling the host.
    pub(crate) fn recognition(&self, id: &str) -> Result<Recognition, storage::StorageError> {
        let store = lock_v2(&self.0);
        let attempts = store.attempts_for(id).map_err(v2_err)?;
        if attempts.iter().any(|attempt| attempt.is_final_transcript()) {
            return Ok(Recognition::Done);
        }
        if attempts
            .iter()
            .any(|attempt| attempt.status == "started" && store.attempt_owned(&attempt.id))
        {
            return Ok(Recognition::InFlight);
        }
        Ok(Recognition::Due)
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
        // #342: upkeep neither compresses nor retires the take mid-read.
        let _pin = self.pin_audio(id);
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

    /// One recording as lossless FLAC (#356 export): the stored FLAC
    /// itself when upkeep already compressed the take (#342), else the
    /// same 16 kHz PCM16 every transcription receives, encoded now — so
    /// the export and the WAV export hold identical samples. `Ok(None)`
    /// when the id is unknown, like [`Self::audio_wav`].
    pub(crate) fn audio_flac(&self, id: &str) -> Result<Option<Arc<Vec<u8>>>, storage::StorageError> {
        let _pin = self.pin_audio(id);
        let path = {
            let store = lock_v2(&self.0);
            match store.audio_journal_path(id) {
                Ok(path) => path,
                Err(StoreV2Error::NotFound(_)) => return Ok(None),
                Err(err) => return Err(v2_err(err)),
            }
        };
        if path.extension().and_then(|ext| ext.to_str()) == Some(starling_dictation::flac::FLAC_EXT) {
            return Ok(Some(Arc::new(std::fs::read(&path)?)));
        }
        let journal = store_v2::read_audio_journal(&path).map_err(v2_err)?;
        let pcm = audio::request_pcm16(&journal.samples, journal.sample_rate)
            .map_err(|err| storage::StorageError::Invalid(err.to_string()))?;
        let flac = starling_dictation::flac::encode(&pcm)
            .map_err(|err| storage::StorageError::Invalid(err.to_string()))?;
        Ok(Some(Arc::new(flac)))
    }

    /// Persists an imported take from its encoded WAV through the full §4
    /// protocol (#220: recorded takes are stored by the recording service,
    /// never here). The stored duration is derived from the samples
    /// themselves. Returns the record id and the WAV to transcribe.
    pub(crate) fn save_capture(&self, wav: Arc<Vec<u8>>) -> Result<SavedTake, storage::StorageError> {
        let pcm = decode_wav(&wav)?;
        let id = self.save_pcm_take(pcm, store_v2::CommitMark::Complete)?;
        Ok(SavedTake { id, wav })
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

    /// Move a take into the archival retention class or back (#342).
    pub(crate) fn set_archival(&self, id: &str, archival: bool) -> Result<(), storage::StorageError> {
        let class = if archival {
            store_v2::ARCHIVAL_CLASS
        } else {
            store_v2::STANDARD_CLASS
        };
        lock_v2(&self.0)
            .set_retention_class(id, class)
            .map_err(v2_err)
    }

    /// Confirmed deletion (R21): the audio journal is quarantined and the
    /// row tombstoned — resurrection-proof. The take's processing
    /// document (#295) goes with it; its insight events cascade with the
    /// capture row.
    pub(crate) fn delete(&self, id: &str) -> Result<(), storage::StorageError> {
        let mut store = lock_v2(&self.0);
        // The processing document first: it holds transcript-derived text
        // and has no foreign key to the capture, so a failure between the
        // two steps must leave a take without its document, never an
        // orphaned document without its take.
        store.delete_document(id).map_err(v2_err)?;
        store.delete_capture(id).map_err(v2_err)
    }

    // ------------------------------------------------------------------
    // Processing (#295). One `documents` row per take, keyed by the
    // capture id: head revision 1 is the raw transcript, later heads are
    // accepted processed text or a return to raw, and every processing
    // result is a `proposal` revision pinned to the head it read. Raw
    // recognition stays in `recognition_attempts`, untouched.
    // ------------------------------------------------------------------

    /// The take's latest final transcript and its attempt id.
    pub(crate) fn latest_raw(&self, id: &str) -> Result<Option<(String, String)>, storage::StorageError> {
        let store = lock_v2(&self.0);
        let attempts = store.attempts_for(id).map_err(v2_err)?;
        Ok(current_transcript(&attempts).map(|attempt| (attempt.id.clone(), attempt.text.clone())))
    }

    /// The take's processing document, if processing ran on its current
    /// raw transcript. A document built on an earlier transcript (the take
    /// was re-transcribed since) is stale and reads as none: what was used
    /// or proposed for the old text does not apply to the new one.
    pub(crate) fn processing_doc(&self, id: &str) -> Result<Option<ProcessingDoc>, storage::StorageError> {
        let store = lock_v2(&self.0);
        let attempts = store.attempts_for(id).map_err(v2_err)?;
        let latest = current_transcript(&attempts).map(|attempt| attempt.id.clone());
        let document = store.get_document(id).map_err(v2_err)?;
        Ok(document
            .and_then(|document| ProcessingDoc::from_row(id, document))
            .filter(|doc| latest.as_deref() == Some(doc.raw_attempt_id.as_str())))
    }

    /// The processing document for the take's current raw transcript:
    /// the stored one when it was built on this attempt, otherwise a
    /// fresh one (head 1 = raw). A re-transcription therefore starts
    /// processing over; results for an older raw text do not carry over.
    pub(crate) fn start_processing_doc(
        &self,
        id: &str,
        attempt_id: &str,
        raw: &str,
    ) -> Result<ProcessingDoc, storage::StorageError> {
        let store = lock_v2(&self.0);
        if store.get_capture(id).map_err(v2_err)?.is_none() {
            return Err(storage::StorageError::NotFound(id.to_string()));
        }
        if let Some(existing) = store
            .get_document(id)
            .map_err(v2_err)?
            .and_then(|document| ProcessingDoc::from_row(id, document))
        {
            if existing.raw_attempt_id == attempt_id {
                return Ok(existing);
            }
            store.delete_document(id).map_err(v2_err)?;
        }
        store
            .commit_document_head(PROCESSING_DOC, 1, 0, &head_row(id, 1, raw, true, attempt_id, None))
            .map_err(v2_err)?;
        Ok(ProcessingDoc {
            head_revision: 1,
            head_text: raw.to_string(),
            head_is_raw: true,
            raw_attempt_id: attempt_id.to_string(),
            raw_text: raw.to_string(),
            proposals: Vec::new(),
            accepted_request: None,
        })
    }

    /// Stores (or updates) one proposal row. A take deleted meanwhile is
    /// `NotFound`: the result lands nowhere.
    pub(crate) fn save_proposal(&self, id: &str, proposal: &ProposalRow) -> Result<(), storage::StorageError> {
        let store = lock_v2(&self.0);
        if store.get_capture(id).map_err(v2_err)?.is_none() {
            return Err(storage::StorageError::NotFound(id.to_string()));
        }
        let Some(document) = store.get_document(id).map_err(v2_err)? else {
            return Err(storage::StorageError::NotFound(id.to_string()));
        };
        // A settled row (accepted or rejected) is final: a job's late
        // write of its proposal, or a Dismiss racing an accept, lands
        // nowhere. (An accept settles its row through
        // commit_processing_head, not here.)
        let row = proposal.to_row(id);
        let settled = document.revisions.iter().any(|stored| {
            stored.rev_id == row.rev_id
                && matches!(RowStatus::parse(&stored.status), Some(RowStatus::Accepted | RowStatus::Rejected))
        });
        if settled {
            return Ok(());
        }
        store.store_document_revision(&row).map_err(v2_err)
    }

    /// Commits a new head revision (accepted processed text, or raw
    /// again), marking the accepted proposal in the same lock.
    pub(crate) fn commit_processing_head(
        &self,
        id: &str,
        revision: u64,
        text: &str,
        is_raw: bool,
        attempt_id: &str,
        accepted: Option<&ProposalRow>,
        derived_from: Option<&str>,
    ) -> Result<(), storage::StorageError> {
        let store = lock_v2(&self.0);
        if store.get_capture(id).map_err(v2_err)?.is_none() {
            return Err(storage::StorageError::NotFound(id.to_string()));
        }
        // A head chosen for an earlier transcript lands nowhere: a
        // re-transcription rebuilt the document on a new raw attempt.
        let current = store
            .get_document(id)
            .map_err(v2_err)?
            .and_then(|document| ProcessingDoc::from_row(id, document));
        if current.is_none_or(|doc| doc.raw_attempt_id != attempt_id) {
            return Err(storage::StorageError::NotFound(id.to_string()));
        }
        // The head and the proposal it accepted land together or not at
        // all, so a crash cannot leave an accepted head next to a live
        // proposal.
        let also: Vec<RevisionRow> = accepted.map(|proposal| proposal.to_row(id)).into_iter().collect();
        store
            .commit_document_head_with(PROCESSING_DOC, revision, 0, &head_row(id, revision, text, is_raw, attempt_id, derived_from), &also)
            .map_err(v2_err)
    }

    /// Records one insight event for the take (idempotent on its id). A
    /// take deleted meanwhile is `NotFound`, like the other processing
    /// writes: the event lands nowhere.
    pub(crate) fn record_insight(
        &self,
        id: &str,
        event_id: &str,
        kind: &str,
        occurred_at: &str,
        payload_json: &str,
    ) -> Result<(), storage::StorageError> {
        let store = lock_v2(&self.0);
        store
            .record_insight_event(event_id, id, kind, occurred_at, payload_json)
            .map_err(v2_err)
    }

    /// Writes or revises one correction record; `Ok(false)` when the take
    /// is excluded (secure field).
    pub(crate) fn record_correction(
        &self,
        record: &store_v2::CorrectionRecord,
    ) -> Result<bool, storage::StorageError> {
        let store = lock_v2(&self.0);
        store.upsert_correction_record(record).map_err(v2_err)
    }

    /// Revises the decision of an existing correction record; `Ok(false)`
    /// when there is none.
    pub(crate) fn revise_correction(
        &self,
        id: &str,
        request_id: &str,
        decision: store_v2::CorrectionDecision,
        decision_utc: &str,
        final_text: &str,
    ) -> Result<bool, storage::StorageError> {
        let store = lock_v2(&self.0);
        store
            .revise_correction_record(id, request_id, decision, decision_utc, final_text)
            .map_err(v2_err)
    }

    #[cfg(test)]
    pub(crate) fn correction_records(
        &self,
        id: &str,
    ) -> Result<Vec<store_v2::CorrectionRecord>, storage::StorageError> {
        let store = lock_v2(&self.0);
        store.correction_records_for(id).map_err(v2_err)
    }

    /// Pin `id`'s audio until the returned guard drops (#342): upkeep
    /// neither compresses nor retires it meanwhile.
    pub(crate) fn pin_audio(&self, id: &str) -> AudioPin {
        lock_v2(&self.0).pin_audio(id);
        AudioPin {
            store: Arc::clone(&self.0),
            id: id.to_string(),
        }
    }

    /// History audio upkeep (#342): replace every settled journal with
    /// its verified FLAC, apply the retention policy (nothing while it is
    /// off), then run the retention sweep. The guard covers only the
    /// cheap steps — listing the candidates, each publish, the policy
    /// run, the sweep — never an encode, so a long take's compression
    /// does not pin every other store call.
    /// `policy` is read under the guard when the policy run starts and
    /// again before each removal, so a limit the user lifted meanwhile is
    /// not applied (the run stops; the app runs again). `paused` is asked
    /// before each compression and before each removal: when it says yes
    /// (a take started recording), the pass ends there; the sweep asks it
    /// too, before each file it removes.
    ///
    /// The sweep runs on every pass, retention limits or not: it removes
    /// for good the audio of takes the user deleted (there is no undo for
    /// a delete, so no grace period — the next pass after the delete) and
    /// the recorder journals proven to be copies of a stored take (#356).
    /// Audio a reader still pins and journals no longer proven copies
    /// stay, with the reason in the report.
    pub(crate) fn audio_upkeep<P: std::borrow::Borrow<RetentionPolicy>>(
        &self,
        policy: impl Fn() -> P,
        paused: impl Fn() -> bool,
    ) -> Result<UpkeepReport, storage::StorageError> {
        let mut report = UpkeepReport::default();
        let jobs = lock_v2(&self.0)
            .compression_candidates(usize::MAX)
            .map_err(v2_err)?;
        for job in jobs {
            if paused() {
                report.paused = true;
                return Ok(report);
            }
            let prepared = match store_v2::prepare_compression(&job) {
                Ok(prepared) => prepared,
                Err(err) => {
                    lock_v2(&self.0).note_compression_failure(&job.id);
                    report.failures.push((job.id.clone(), err.to_string()));
                    continue;
                }
            };
            // The guard must drop before the arms: a failure locks again.
            let committed = lock_v2(&self.0).commit_compression(prepared);
            match committed {
                Ok(CompressionOutcome::Compressed {
                    journal_bytes,
                    flac_bytes,
                }) => {
                    report.compressed += 1;
                    report.saved_bytes += journal_bytes.saturating_sub(flac_bytes);
                }
                Ok(CompressionOutcome::Skipped(_)) => {}
                Err(err) => {
                    lock_v2(&self.0).note_compression_failure(&job.id);
                    report.failures.push((job.id.clone(), err.to_string()));
                }
            }
        }
        if paused() {
            report.paused = true;
            return Ok(report);
        }
        report.retention = lock_v2(&self.0)
            .apply_retention_policy_now(policy, &paused)
            .map_err(v2_err)?;
        report.paused = report.retention.stopped;
        if report.paused || paused() {
            report.paused = true;
            return Ok(report);
        }
        report.sweep = lock_v2(&self.0)
            .sweep_retention_until(&paused)
            .map_err(v2_err)?;
        report.paused = report.sweep.stopped;
        Ok(report)
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
        let meta = store_v2::TakeMeta::for_device("");
        let mut take = {
            let store = lock_v2(&self.0);
            store.begin_take_at_rate(rate, meta)
        }
        .map_err(v2_err)?;
        take.append_and_seal(&pcm.samples).map_err(v2_err)?;
        let finalized = take.finalize().map_err(v2_err)?;
        let mut store = lock_v2(&self.0);
        let committed = finalized.commit_marked(&mut store, mark).map_err(v2_err)?;
        Ok(committed.record.id)
    }
}

/// A [`Store::pin_audio`] guard. Dropping it takes the store lock: never
/// drop one while holding that lock.
pub(crate) struct AudioPin {
    store: Arc<Mutex<StoreV2>>,
    id: String,
}

impl Drop for AudioPin {
    fn drop(&mut self) {
        lock_v2(&self.store).unpin_audio(&self.id);
    }
}

/// What one [`Store::audio_upkeep`] pass did.
#[derive(Debug, Default)]
pub(crate) struct UpkeepReport {
    pub compressed: usize,
    pub saved_bytes: u64,
    /// `(id, reason)` for journals that could not be compressed; they
    /// stay journals and are tried again next pass.
    pub failures: Vec<(String, String)>,
    pub retention: store_v2::RetentionReport,
    /// Deleted and superseded audio removed for good, and what stayed.
    pub sweep: store_v2::SweepReport,
    /// A take started recording and the pass stopped early.
    pub paused: bool,
}

impl UpkeepReport {
    /// One line for the settings dialog; `None` when nothing happened.
    pub(crate) fn summary(&self) -> Option<String> {
        let mb = |bytes: u64| bytes.div_ceil(1024 * 1024);
        let plural = |count: usize| if count == 1 { "" } else { "s" };
        let mut parts = Vec::new();
        if self.compressed > 0 {
            parts.push(format!(
                "compressed {} recording{} losslessly (saved {} MB)",
                self.compressed,
                plural(self.compressed),
                mb(self.saved_bytes)
            ));
        }
        let retired = self.retention.retired.len();
        if retired > 0 {
            parts.push(format!(
                "removed the audio of {retired} recording{} ({} MB); transcripts are kept",
                plural(retired),
                mb(self.retention.retired_bytes)
            ));
        }
        let count = |wanted: fn(&HoldReason) -> bool| {
            self.retention
                .held
                .iter()
                .filter(|held| wanted(&held.reason))
                .count()
        };
        let referenced = count(|reason| matches!(reason, HoldReason::Referenced { .. }));
        if referenced > 0 {
            parts.push(format!(
                "kept {referenced} due recording{} that documents or corrections use",
                plural(referenced)
            ));
        }
        let untranscribed = count(|reason| matches!(reason, HoldReason::Untranscribed));
        if untranscribed > 0 {
            parts.push(format!(
                "kept {untranscribed} due recording{} that never got a transcript",
                plural(untranscribed)
            ));
        }
        for (class, bytes) in &self.retention.over_limit {
            parts.push(format!(
                "the {class} size limit is still exceeded by {} MB",
                mb(*bytes)
            ));
        }
        let swept = self.sweep.swept.len();
        if swept > 0 {
            parts.push(format!(
                "permanently removed {swept} deleted or duplicate audio file{} ({} MB)",
                plural(swept),
                mb(self.sweep.swept_bytes)
            ));
        }
        let mut retained: Vec<(&str, usize)> = Vec::new();
        for (_, reason) in &self.sweep.retained {
            match retained.iter_mut().find(|(seen, _)| *seen == reason.as_str()) {
                Some((_, count)) => *count += 1,
                None => retained.push((reason, 1)),
            }
        }
        for (reason, count) in retained {
            parts.push(format!(
                "left {count} entr{} in the deleted-audio folders: {reason}",
                if count == 1 { "y" } else { "ies" }
            ));
        }
        if !self.failures.is_empty() {
            parts.push(format!(
                "{} recording{} could not be compressed and stay as they are",
                self.failures.len(),
                plural(self.failures.len())
            ));
        }
        if self.paused && !parts.is_empty() {
            parts.push("paused while a take records".to_string());
        }
        (!parts.is_empty()).then(|| {
            let mut text = parts.join("; ");
            if let Some(first) = text.get_mut(..1) {
                first.make_ascii_uppercase();
            }
            format!("{text}.")
        })
    }
}

/// What a persisted import hands back to the pipeline: the record id,
/// and the WAV its transcription runs on.
pub(crate) struct SavedTake {
    pub(crate) id: String,
    pub(crate) wav: Arc<Vec<u8>>,
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
                    let mut problems = listing.problems;
                    if let AudioAtRest::Retired { utc } = &listing.audio {
                        problems.push(format!(
                            "Audio removed by your retention policy on {}; the transcript is kept.",
                            utc.get(..10).unwrap_or(utc)
                        ));
                    }
                    ListedRecord::Session(v2_summary(&listing.record, &problems, &attempts))
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

    // The current transcript (earlier ones stay in the attempt rows as
    // history, listed in `results`).
    let current = current_transcript(attempts);
    let transcript = current.map(attempt_transcript);
    // #363: the label names the attempt that produced that transcript —
    // not the latest attempt. A later retry that failed (or is in
    // flight) on another backend must not rewrite the label of the
    // transcript the take still shows.
    let model_label = current.map(|attempt| attempt.backend.clone());

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
        model_label,
        // The capture id *is* the journal linkage in v2; there is no
        // separate v1 journal to point at.
        journal_id: None,
        archival: record.retention_class == store_v2::ARCHIVAL_CLASS,
        interrupted: record.status == CaptureStatus::Interrupted,
        confirmed_ms: (record.actual_rate > 0)
            .then(|| record.ack_sample_index as f64 * 1000.0 / f64::from(record.actual_rate)),
        results: attempts
            .iter()
            .filter(|attempt| attempt.is_final_transcript())
            .map(|attempt| storage::TakeResult {
                attempt_id: attempt.id.clone(),
                backend: attempt.backend.clone(),
                text: attempt_transcript(attempt).text,
                created_at: attempt.created_utc.clone(),
                shown: current.is_some_and(|current| current.id == attempt.id),
            })
            .collect(),
    }
}

/// The transcript a take shows (#356): its latest completed transcript
/// with words in it — a retry that came back empty is kept as a result
/// but never replaces real text — else the latest completed one.
fn current_transcript(attempts: &[AttemptRecord]) -> Option<&AttemptRecord> {
    let mut finals = attempts.iter().rev().filter(|attempt| attempt.is_final_transcript());
    let latest = finals.clone().next();
    finals
        .find(|attempt| !attempt.text.trim().is_empty())
        .or(latest)
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
        reopen_v2(&root)
    }

    /// The facade over an existing scratch root — the "previous run
    /// crashed" shape (#213): a fresh open of the same directory holds no
    /// in-memory attempt ownership, exactly like a new process.
    fn reopen_v2(root: &std::path::Path) -> Store {
        let store = StoreV2::open(root.join("v2")).expect("reopen v2");
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

    /// A recorder journal a killed take left: `samples` under a fsynced
    /// boundary, placed in `root`'s journal tree as `id`.
    fn killed_journal(root: &std::path::Path, id: &str, samples: &[f32]) -> std::path::PathBuf {
        let scratch = scratch_dir("killed-src");
        let writer = StoreV2::open(&scratch).expect("scratch store");
        let mut take = writer
            .begin_take(store_v2::TakeMeta::for_device("test"))
            .expect("begin");
        take.append_frames(samples).expect("append");
        take.write_boundary().expect("boundary");
        let staged = scratch.join("staging").join(format!("{}.sj", take.id()));
        drop(take);
        let tree = root.join("journals");
        std::fs::create_dir_all(&tree).expect("journals tree");
        let path = tree.join(format!("{id}.sj"));
        std::fs::rename(&staged, &path).expect("place journal");
        path
    }

    fn wav_of(samples: &[f32]) -> Arc<Vec<u8>> {
        let pcm = audio::PcmAudio {
            samples: samples.to_vec(),
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
            secure_field: false,
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
            &[plain_attempt("completed", Some("hello v2"), Some(extra))],
        );
        assert_eq!(summary.status, SessionStatus::Transcribed);
        assert_eq!(summary.transcript.as_ref().unwrap().text, "hello v2");

        // A failed retry: failed, but the earlier transcript survives and
        // the failure surfaces — the v1 save_failure semantics.
        let failed = plain_attempt("failed", None, Some(r#"{"error":"offline"}"#));
        let summary = v2_summary(
            &record(CaptureStatus::Complete, None),
            &[],
            &[plain_attempt("completed", Some("hello v2"), Some(extra)), failed],
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

    #[test]
    fn the_label_names_the_attempt_that_produced_the_transcript() {
        // #363: provenance is per take. A transcript produced by the
        // built-in engine keeps its `engine:` label even when a later
        // retry — here on the manual server — fails over it: the label
        // describes the transcript the take shows, not the last try.
        let extra = r#"{"text":"hello v2","segments":[]}"#;
        let mut on_engine = plain_attempt("completed", Some("hello v2"), Some(extra));
        on_engine.backend = "engine:parakeet-v3-q4km-s16".to_string();
        let mut failed_retry = plain_attempt("failed", None, Some(r#"{"error":"offline"}"#));
        failed_retry.backend = "openai:whisper-large-v3".to_string();
        let summary = v2_summary(
            &record(CaptureStatus::Complete, None),
            &[],
            &[on_engine, failed_retry.clone()],
        );
        assert_eq!(summary.model_label.as_deref(), Some("engine:parakeet-v3-q4km-s16"));

        // No transcript yet: no label, never the failed attempt's backend.
        let summary = v2_summary(
            &record(CaptureStatus::Complete, None),
            &[],
            &[failed_retry],
        );
        assert_eq!(summary.model_label, None);
    }

    #[test]
    fn each_take_shows_its_own_transcription_label() {
        // Two takes, two backends: the labels never bleed across takes
        // (the summary query is per capture).
        let extra = r#"{"text":"hi","segments":[]}"#;
        let mut on_engine = plain_attempt("completed", Some("hi"), Some(extra));
        on_engine.backend = "engine:parakeet-v3-q8".to_string();
        let first = v2_summary(&record(CaptureStatus::Complete, None), &[], &[on_engine]);
        let mut on_server = plain_attempt("completed", Some("hi"), Some(extra));
        on_server.backend = "openai:whisper-large-v3".to_string();
        let second = v2_summary(&record(CaptureStatus::Complete, None), &[], &[on_server]);
        assert_eq!(
            first.model_label.as_deref(),
            Some("engine:parakeet-v3-q8")
        );
        assert_eq!(
            second.model_label.as_deref(),
            Some("openai:whisper-large-v3")
        );
    }

    // ---- the facade over a real v2 store (daily path) ------------------

    fn transcribed(store: &Store, text: &str) -> String {
        let id = store.save_capture(tiny_wav(160)).expect("save").id;
        store.mark_attempt(&id, "starling:parakeet").expect("begin");
        store.save_transcript(&id, transcript(text)).expect("transcript");
        id
    }

    /// `transcribed` with an explicit backend label (#363): the label the
    /// summary must hand back unchanged.
    fn transcribed_on(store: &Store, text: &str, backend: &str) -> String {
        let id = store.save_capture(tiny_wav(160)).expect("save").id;
        store.mark_attempt(&id, backend).expect("begin");
        store.save_transcript(&id, transcript(text)).expect("transcript");
        id
    }

    #[test]
    fn summaries_carry_the_backend_that_produced_each_transcript() {
        // End to end over the real store: one take on the built-in
        // engine, one on a manual server — and a failed retry on the
        // first take with a third backend, which must not rewrite the
        // label of the transcript that take still shows.
        let store = v2_store("labels");
        let on_engine = transcribed_on(&store, "engine take", "engine:parakeet-v3-q4km-s16");
        let on_server = transcribed_on(&store, "server take", "openai:whisper-large-v3");
        store.mark_attempt(&on_engine, "engine:moss-2b-q4e8").expect("retry");
        store
            .save_failure(&on_engine, "The built-in engine stopped while transcribing.")
            .expect("fail");

        assert_eq!(
            summary_of(&store, &on_engine).model_label.as_deref(),
            Some("engine:parakeet-v3-q4km-s16")
        );
        assert_eq!(
            summary_of(&store, &on_server).model_label.as_deref(),
            Some("openai:whisper-large-v3")
        );
    }

    fn proposal(request_id: &str, base: u64, text: &str) -> ProposalRow {
        ProposalRow {
            request_id: request_id.to_string(),
            base_revision: base,
            text: text.to_string(),
            status: RowStatus::Proposed,
            label: "S1-mini · this computer".to_string(),
            failure: None,
            stop_to_result_ms: Some(1234.5),
            origin: None,
        }
    }

    /// A re-transcribed take's old processing document is stale: it
    /// reads as none, so an earlier processed head never comes back over
    /// the new transcript (not even after a restart).
    #[test]
    fn a_retranscribed_take_has_no_processing_document() {
        let root = scratch_dir("processing-retranscribed");
        let store = reopen_v2(&root);
        let id = transcribed(&store, "um first take");
        let (attempt, raw) = store.latest_raw(&id).expect("raw").expect("final");
        store.start_processing_doc(&id, &attempt, &raw).expect("start");
        store
            .commit_processing_head(&id, 2, "First take.", false, &attempt, None, None)
            .expect("accept");
        assert!(store.processing_doc(&id).expect("load").is_some());
        store.mark_attempt(&id, "starling:parakeet").expect("retry");
        store.save_transcript(&id, transcript("second take")).expect("transcript");
        assert_eq!(store.processing_doc(&id).expect("load"), None);
        assert_eq!(reopen_v2(&root).processing_doc(&id).expect("load"), None);
    }

    /// #295: a take's processing document survives a restart with its
    /// head and proposals, and never touches the raw attempt.
    #[test]
    fn processing_documents_round_trip_and_keep_raw_apart() {
        let root = scratch_dir("processing-doc");
        let store = reopen_v2(&root);
        let id = transcribed(&store, "um so hello there");
        let (attempt, raw) = store.latest_raw(&id).expect("raw").expect("final");
        assert_eq!(raw, "um so hello there");
        let doc = store.start_processing_doc(&id, &attempt, &raw).expect("start");
        assert_eq!((doc.head_revision, doc.head_is_raw), (1, true));
        // Starting again on the same attempt keeps the document.
        store.save_proposal(&id, &proposal("p1", 1, "So, hello there.")).expect("proposal");
        let again = store.start_processing_doc(&id, &attempt, &raw).expect("again");
        assert_eq!(again.proposals.len(), 1);

        let accepted = ProposalRow { status: RowStatus::Accepted, ..proposal("p1", 1, "So, hello there.") };
        store
            .commit_processing_head(&id, 2, "So, hello there.", false, &attempt, Some(&accepted), Some("p1"))
            .expect("accept");
        // A job's late write of the same proposal never demotes it.
        store.save_proposal(&id, &proposal("p1", 1, "So, hello there.")).expect("late write");
        let reopened = reopen_v2(&root);
        let doc = reopened.processing_doc(&id).expect("load").expect("doc");
        assert_eq!(doc.head_revision, 2);
        assert_eq!(doc.head_text, "So, hello there.");
        assert!(!doc.head_is_raw);
        assert_eq!(doc.raw_text, "um so hello there");
        assert_eq!(doc.proposals, vec![accepted]);
        // The recognition attempt is exactly what the recognizer returned.
        assert_eq!(transcript_text(&reopened, &id).as_deref(), Some("um so hello there"));
    }

    #[test]
    fn a_new_transcription_starts_processing_over() {
        let store = v2_store("processing-retranscribe");
        let id = transcribed(&store, "first raw");
        let (attempt, raw) = store.latest_raw(&id).expect("raw").expect("final");
        store.start_processing_doc(&id, &attempt, &raw).expect("start");
        store.save_proposal(&id, &proposal("p1", 1, "First.")).expect("proposal");
        store.mark_attempt(&id, "starling:parakeet").expect("retry");
        store.save_transcript(&id, transcript("second raw")).expect("transcript");
        let (attempt, raw) = store.latest_raw(&id).expect("raw").expect("final");
        let doc = store.start_processing_doc(&id, &attempt, &raw).expect("restart");
        assert_eq!(doc.raw_text, "second raw");
        assert!(doc.proposals.is_empty(), "results for the old raw text do not carry over");
    }

    #[test]
    fn deleting_a_take_takes_its_processing_and_insight_rows() {
        let store = v2_store("processing-delete");
        let id = transcribed(&store, "to be deleted");
        let (attempt, raw) = store.latest_raw(&id).expect("raw").expect("final");
        store.start_processing_doc(&id, &attempt, &raw).expect("start");
        store
            .record_insight(&id, "proc-p1", "processing_recorded", "2026-09-24T10:00:00Z", "{}")
            .expect("insight");
        store.delete(&id).expect("delete");
        assert!(store.processing_doc(&id).expect("load").is_none());
        // A late result for the deleted take lands nowhere.
        assert!(matches!(
            store.save_proposal(&id, &proposal("p2", 1, "Late.")),
            Err(storage::StorageError::NotFound(_))
        ));
        assert!(lock_v2(&store.0).insight_events_for(&id).expect("events").is_empty());
    }

    #[test]
    fn deleting_a_take_takes_its_correction_records_too() {
        let store = v2_store("correction-delete");
        let id = transcribed(&store, "um to be deleted");
        let (attempt, raw) = store.latest_raw(&id).expect("raw").expect("final");
        store
            .start_processing_doc(&id, &attempt, &raw)
            .expect("start");
        let record = store_v2::CorrectionRecord {
            capture_id: id.clone(),
            request_id: "r1".to_string(),
            raw_attempt_id: attempt.clone(),
            raw_text: raw.clone(),
            processed_text: "Deleted.".to_string(),
            final_text: Some("Deleted.".to_string()),
            decision: store_v2::CorrectionDecision::Accepted,
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
        };
        assert!(store.record_correction(&record).expect("record"));
        assert_eq!(store.correction_records(&id).expect("read").len(), 1);
        store.delete(&id).expect("delete");
        assert!(store.correction_records(&id).expect("read").is_empty());
        assert!(matches!(
            store.record_correction(&record),
            Err(storage::StorageError::NotFound(_))
        ));
    }

    #[test]
    fn the_v2_facade_covers_the_daily_path_end_to_end() {
        let store = v2_store("facade");
        let wav = tiny_wav(300);

        // Save (no journal: the WAV path through the §4 protocol).
        let saved = store.save_capture(wav.clone()).expect("save");
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

    #[test]
    fn a_missing_v2_audio_surfaces_its_error_instead_of_falling_back() {
        let store = v2_store("degrade");
        let id = store
            .save_capture(tiny_wav(50))
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
    fn upkeep_compresses_and_a_retry_sends_the_same_request_wav() {
        // #342 acceptance: a retried take from FLAC produces the same
        // request audio as from the journal.
        let store = v2_store("upkeep-flac");
        let samples: Vec<f32> = (0..32_000)
            .map(|i| ((i as f32 * 0.05).sin() * 0.3) + ((i % 7) as f32 * 0.001))
            .collect();
        let saved = store.save_capture(wav_of(&samples)).expect("save");
        let before = store.audio_wav(&saved.id).expect("load").expect("present");
        assert_eq!(*before, *saved.wav, "the first transcription's audio");

        let upkeep = store
            .audio_upkeep(store_v2::RetentionPolicy::default, || false)
            .expect("upkeep");
        assert_eq!(upkeep.compressed, 1);
        assert!(upkeep.saved_bytes > 0);
        assert!(upkeep.retention.retired.is_empty(), "retention is off");
        assert!(upkeep.summary().expect("summary").starts_with("Compressed 1 recording"));
        let after = store.audio_wav(&saved.id).expect("load").expect("present");
        assert_eq!(*after, *before);
        // A second pass finds nothing to do.
        let again = store
            .audio_upkeep(store_v2::RetentionPolicy::default, || false)
            .expect("upkeep");
        assert_eq!(again.compressed, 0);
        assert!(again.summary().is_none());
    }

    #[test]
    fn upkeep_stops_when_a_take_starts_recording() {
        let store = v2_store("upkeep-paused");
        let saved = store.save_capture(wav_of(&[0.1f32; 32_000])).expect("save");
        let mut policy = store_v2::RetentionPolicy::default();
        policy.grace = std::time::Duration::ZERO;
        policy.include_referenced = true;
        policy.limits.insert(
            store_v2::STANDARD_CLASS.to_string(),
            store_v2::ClassLimits {
                max_age_days: Some(0),
                max_total_bytes: None,
            },
        );
        let upkeep = store
            .audio_upkeep(|| policy.clone(), || true)
            .expect("upkeep");
        assert!(upkeep.paused);
        assert_eq!(upkeep.compressed, 0);
        assert!(upkeep.retention.retired.is_empty());
        assert!(upkeep.summary().is_none());
        let wav = store.audio_wav(&saved.id).expect("load").expect("present");
        assert_eq!(*wav, *saved.wav, "still the journal, untouched");
    }

    #[test]
    fn a_failed_compression_commit_leaves_the_store_usable() {
        let store = v2_store("upkeep-commit-fails");
        let saved = store.save_capture(wav_of(&[0.1f32; 32_000])).expect("save");
        // A directory where the FLAC goes: the publish's rename fails.
        let flac = lock_v2(&store.0)
            .root()
            .join("audio")
            .join(format!("{}.flac", saved.id));
        std::fs::create_dir_all(flac.join("blocker")).expect("blocking directory");
        let (done, finished) = std::sync::mpsc::channel();
        let worker = store.clone();
        std::thread::spawn(move || {
            let upkeep = worker
                .audio_upkeep(store_v2::RetentionPolicy::default, || false)
                .expect("upkeep");
            let candidates = lock_v2(&worker.0)
                .compression_candidates(10)
                .expect("candidates")
                .len();
            done.send((upkeep, candidates)).expect("send");
        });
        let (upkeep, candidates) = finished
            .recv_timeout(std::time::Duration::from_secs(30))
            .expect("upkeep returned and the store is still usable");
        assert_eq!(upkeep.compressed, 0);
        assert_eq!(upkeep.failures.len(), 1, "{upkeep:?}");
        assert_eq!(candidates, 1, "one failure does not give up yet");
    }

    #[test]
    fn upkeep_stops_removing_once_a_take_starts_recording() {
        let store = v2_store("upkeep-paused-mid-run");
        let id = transcribed(&store, "a take past its limit");
        let mut policy = store_v2::RetentionPolicy::default();
        policy.grace = std::time::Duration::ZERO;
        policy.limits.insert(
            store_v2::STANDARD_CLASS.to_string(),
            store_v2::ClassLimits {
                max_age_days: Some(0),
                max_total_bytes: None,
            },
        );
        // A take starts recording right after the policy run began.
        let recording = std::sync::atomic::AtomicBool::new(false);
        let upkeep = store
            .audio_upkeep(
                || {
                    recording.store(true, std::sync::atomic::Ordering::SeqCst);
                    policy.clone()
                },
                || recording.load(std::sync::atomic::Ordering::SeqCst),
            )
            .expect("upkeep");
        assert!(upkeep.paused);
        assert!(upkeep.retention.retired.is_empty(), "{upkeep:?}");
        assert!(store.audio_wav(&id).expect("load").is_some());
    }

    #[test]
    fn a_summary_never_slices_inside_a_character() {
        let report = UpkeepReport {
            failures: vec![("c_x".to_string(), "unreadable".to_string())],
            ..UpkeepReport::default()
        };
        assert!(report.summary().expect("summary").starts_with("1 recording"));
    }

    #[test]
    fn archiving_a_take_moves_it_under_the_archival_limits() {
        let store = v2_store("upkeep-archive");
        let id = transcribed(&store, "an archived take");
        assert!(!summary_of(&store, &id).archival);
        store.set_archival(&id, true).expect("archive");
        assert!(summary_of(&store, &id).archival);
        // A standard-only limit no longer reaches it; the archival one does.
        let mut policy = store_v2::RetentionPolicy::default();
        policy.grace = std::time::Duration::ZERO;
        let expired = store_v2::ClassLimits {
            max_age_days: Some(0),
            max_total_bytes: None,
        };
        policy
            .limits
            .insert(store_v2::STANDARD_CLASS.to_string(), expired);
        let upkeep = store.audio_upkeep(|| policy.clone(), || false).expect("upkeep");
        assert!(upkeep.retention.retired.is_empty());
        policy
            .limits
            .insert(store_v2::ARCHIVAL_CLASS.to_string(), expired);
        let upkeep = store.audio_upkeep(|| policy.clone(), || false).expect("upkeep");
        assert_eq!(upkeep.retention.retired.len(), 1);
        store.set_archival(&id, false).expect("unarchive");
        assert!(!summary_of(&store, &id).archival);
    }

    #[test]
    fn a_pinned_take_keeps_its_audio_until_the_pin_drops() {
        let store = v2_store("upkeep-pin");
        let id = transcribed(&store, "pinned");
        let mut policy = store_v2::RetentionPolicy::default();
        policy.grace = std::time::Duration::ZERO;
        policy.limits.insert(
            store_v2::STANDARD_CLASS.to_string(),
            store_v2::ClassLimits {
                max_age_days: Some(0),
                max_total_bytes: None,
            },
        );
        let pin = store.pin_audio(&id);
        let upkeep = store.audio_upkeep(|| policy.clone(), || false).expect("upkeep");
        assert!(upkeep.retention.retired.is_empty());
        assert!(store.audio_wav(&id).expect("load").is_some());
        drop(pin);
        let upkeep = store.audio_upkeep(|| policy.clone(), || false).expect("upkeep");
        assert_eq!(upkeep.retention.retired.len(), 1);
    }

    #[test]
    fn upkeep_sweeps_deleted_audio_and_proven_duplicate_journals() {
        let store = v2_store("upkeep-sweep");
        let root = lock_v2(&store.0).root().to_path_buf();
        // A take the user deleted: its audio waits in quarantine.
        let deleted = store.save_capture(tiny_wav(200)).expect("save").id;
        store.delete(&deleted).expect("delete");
        let quarantined = root.join("quarantine").join(format!("{deleted}.sj"));
        assert!(quarantined.exists());
        // A faulted recorder journal a stored take provably holds, moved
        // aside the way the recording service does when it stores a take
        // from memory in its place (#220).
        let samples: Vec<f32> = (0..120).map(|i| (i % 97) as f32 * 0.001).collect();
        let path = killed_journal(&root, "j_proven", &samples);
        let replacement = {
            let mut meta = store_v2::TakeMeta::for_device("");
            meta.supersedes_journal = Some("j_proven".to_string());
            let pcm = decode_wav(&tiny_wav(300)).expect("decode");
            let mut v2 = lock_v2(&store.0);
            let mut take = v2.begin_take_at_rate(16_000, meta).expect("begin");
            take.append_and_seal(&pcm.samples).expect("append");
            let committed = take
                .finalize()
                .expect("finalize")
                .commit_marked(&mut v2, store_v2::CommitMark::Complete)
                .expect("commit");
            v2.audio_journal_path(&committed.record.id).expect("stored path")
        };
        assert!(store_v2::supersede_journal_held_by(&path, &replacement).expect("move aside"));
        let superseded = root.join("journals").join(store_v2::SUPERSEDED_SUBDIR);
        let proven = superseded.join("j_proven.sj");
        assert!(proven.exists());
        // A journal under superseded/ that no stored take holds.
        let path = killed_journal(&root, "j_unproven", &samples);
        let unproven = superseded.join("j_unproven.sj");
        std::fs::rename(&path, &unproven).expect("place journal");

        // Nothing is swept while a take records.
        let upkeep = store
            .audio_upkeep(store_v2::RetentionPolicy::default, || true)
            .expect("upkeep");
        assert!(upkeep.paused);
        assert!(upkeep.sweep.swept.is_empty() && upkeep.sweep.retained.is_empty());
        assert!(quarantined.exists() && proven.exists() && unproven.exists());

        // The next pass removes both copies; the unproven journal stays.
        let upkeep = store
            .audio_upkeep(store_v2::RetentionPolicy::default, || false)
            .expect("upkeep");
        let mut swept: Vec<&str> = upkeep.sweep.swept.iter().map(|file| file.id.as_str()).collect();
        swept.sort();
        let mut expected = vec![deleted.as_str(), "j_proven"];
        expected.sort();
        assert_eq!(swept, expected, "{upkeep:?}");
        assert!(upkeep.sweep.swept_bytes > 0);
        assert!(!quarantined.exists() && !proven.exists());
        assert!(unproven.exists());
        assert_eq!(upkeep.sweep.retained.len(), 1, "{upkeep:?}");
        assert_eq!(upkeep.sweep.retained[0].0, "j_unproven.sj");
        assert!(upkeep.sweep.retained[0].1.contains("proven"));
        let summary = upkeep.summary().expect("summary");
        assert!(
            summary.contains("permanently removed 2 deleted or duplicate audio files"),
            "{summary}"
        );
        assert!(
            summary.contains("left 1 entry in the deleted-audio folders: no stored take"),
            "{summary}"
        );
    }

    #[test]
    fn a_deleted_take_read_before_the_delete_is_swept_once_the_read_ends() {
        let store = v2_store("upkeep-sweep-pin");
        let id = store.save_capture(tiny_wav(32_000)).expect("save").id;
        let upkeep = store
            .audio_upkeep(store_v2::RetentionPolicy::default, || false)
            .expect("upkeep");
        assert_eq!(upkeep.compressed, 1, "{upkeep:?}");
        // Two reads overlap the delete: the FLAC stays until both end.
        let first = store.pin_audio(&id);
        let second = store.pin_audio(&id);
        store.delete(&id).expect("delete");
        let quarantined = lock_v2(&store.0)
            .root()
            .join("quarantine")
            .join(format!("{id}.flac"));
        assert!(quarantined.exists());
        for pin in [first, second] {
            let upkeep = store
                .audio_upkeep(store_v2::RetentionPolicy::default, || false)
                .expect("upkeep");
            assert!(upkeep.sweep.swept.is_empty(), "{upkeep:?}");
            assert_eq!(upkeep.sweep.retained.len(), 1, "{upkeep:?}");
            assert!(upkeep.sweep.retained[0].1.contains("still being read"));
            assert!(quarantined.exists());
            drop(pin);
        }
        let upkeep = store
            .audio_upkeep(store_v2::RetentionPolicy::default, || false)
            .expect("upkeep");
        assert_eq!(upkeep.sweep.swept.len(), 1, "{upkeep:?}");
        assert_eq!(upkeep.sweep.swept[0].id, id);
        assert!(!quarantined.exists());
    }

    #[test]
    fn a_take_starting_to_record_stops_the_sweep_before_its_next_removal() {
        let store = v2_store("upkeep-sweep-paused");
        let root = lock_v2(&store.0).root().to_path_buf();
        let quarantined: Vec<std::path::PathBuf> = (0..2)
            .map(|_| {
                let id = store.save_capture(tiny_wav(200)).expect("save").id;
                store.delete(&id).expect("delete");
                root.join("quarantine").join(format!("{id}.sj"))
            })
            .collect();
        // Recording starts as soon as the sweep removed its first file.
        let upkeep = store
            .audio_upkeep(store_v2::RetentionPolicy::default, || {
                quarantined.iter().any(|path| !path.exists())
            })
            .expect("upkeep");
        assert!(upkeep.paused && upkeep.sweep.stopped, "{upkeep:?}");
        assert_eq!(upkeep.sweep.swept.len(), 1, "{upkeep:?}");
        assert_eq!(quarantined.iter().filter(|path| path.exists()).count(), 1);
        // The next pass finishes the job.
        let upkeep = store
            .audio_upkeep(store_v2::RetentionPolicy::default, || false)
            .expect("upkeep");
        assert_eq!(upkeep.sweep.swept.len(), 1, "{upkeep:?}");
        assert!(quarantined.iter().all(|path| !path.exists()));
    }

    #[test]
    fn retired_audio_is_explained_and_the_transcript_stays() {
        let store = v2_store("upkeep-retire");
        let id = transcribed(&store, "keep this text");
        let mut policy = store_v2::RetentionPolicy::default();
        policy.grace = std::time::Duration::ZERO;
        policy.limits.insert(
            store_v2::STANDARD_CLASS.to_string(),
            store_v2::ClassLimits {
                max_age_days: Some(0),
                max_total_bytes: None,
            },
        );
        let upkeep = store.audio_upkeep(|| policy.clone(), || false).expect("upkeep");
        assert_eq!(upkeep.retention.retired.len(), 1);
        assert!(
            upkeep.summary().expect("summary").contains("removed the audio of 1 recording"),
            "{:?}",
            upkeep.summary()
        );
        let summary = summary_of(&store, &id);
        assert_eq!(summary.status, SessionStatus::Transcribed);
        assert_eq!(transcript_text(&store, &id).as_deref(), Some("keep this text"));
        let note = summary.last_error.expect("note");
        assert!(note.contains("retention policy"), "{note}");
        let err = store.audio_wav(&id).expect_err("no audio left");
        assert!(err.to_string().contains("retention policy"), "{err}");
    }

    // ---- #356: durable audio, retries, recovery --------------------------

    #[test]
    fn a_retry_adds_a_result_and_never_replaces_an_earlier_one() {
        let store = v2_store("retry-adds");
        let id = store.save_capture(tiny_wav(160)).expect("save").id;
        store.mark_attempt(&id, "engine:model-a").expect("begin");
        store.save_transcript(&id, transcript("first words")).expect("first");
        store.mark_attempt(&id, "openai:whisper").expect("retry");
        store.save_transcript(&id, transcript("second words")).expect("second");

        let take = summary_of(&store, &id);
        assert_eq!(take.transcript.expect("shown").text, "second words");
        assert_eq!(take.model_label.as_deref(), Some("openai:whisper"));
        let results: Vec<_> = take
            .results
            .iter()
            .map(|result| (result.backend.as_str(), result.text.as_str(), result.shown))
            .collect();
        assert_eq!(
            results,
            vec![
                ("engine:model-a", "first words", false),
                ("openai:whisper", "second words", true),
            ]
        );

        // A failed retry adds no result and keeps what the take shows.
        store.mark_attempt(&id, "engine:model-b").expect("third");
        store.save_failure(&id, "The built-in engine stopped").expect("fail");
        let take = summary_of(&store, &id);
        assert_eq!(take.status, SessionStatus::Failed);
        assert_eq!(take.transcript.expect("kept").text, "second words");
        assert_eq!(take.results.len(), 2);
        // The audio is untouched by any of it.
        let audio = store.audio_wav(&id).expect("load").expect("present");
        assert_eq!(audio::decode_pcm16_wav(&audio).expect("decode").samples.len(), 160);
    }

    #[test]
    fn an_empty_retry_never_replaces_a_real_transcript() {
        let store = v2_store("empty-retry");
        let id = store.save_capture(tiny_wav(160)).expect("save").id;
        store.mark_attempt(&id, "engine:model-a").expect("begin");
        store.save_transcript(&id, transcript("real words")).expect("first");
        store.mark_attempt(&id, "engine:model-b").expect("retry");
        store.save_transcript(&id, transcript("  ")).expect("empty");

        let take = summary_of(&store, &id);
        assert_eq!(take.transcript.expect("shown").text, "real words");
        assert_eq!(take.model_label.as_deref(), Some("engine:model-a"));
        assert_eq!(take.results.len(), 2, "the empty result is kept as history");
        assert!(take.results[0].shown && !take.results[1].shown);
        // Processing reads the same transcript the take shows.
        assert_eq!(
            store.latest_raw(&id).expect("raw").map(|(_, text)| text).as_deref(),
            Some("real words")
        );

        // A take whose only result is empty still shows it as such.
        let only = store.save_capture(tiny_wav(160)).expect("save").id;
        store.mark_attempt(&only, "engine:model-b").expect("begin");
        store.save_transcript(&only, transcript("")).expect("empty");
        assert_eq!(summary_of(&store, &only).transcript.expect("shown").text, "");
    }

    #[test]
    fn a_pinned_retry_keeps_its_audio_through_upkeep() {
        // #342 pins + #356: while a retry holds its pin (loading, waiting
        // for a model switch) upkeep neither compresses nor retires the
        // take; afterwards the retry's request audio is byte-identical
        // whether it reads the journal or the FLAC.
        let store = v2_store("retry-pin");
        let id = store.save_capture(tiny_wav(32_000)).expect("save").id;
        let before = store.audio_wav(&id).expect("load").expect("present");
        let pin = store.pin_audio(&id);
        let report = store
            .audio_upkeep(store_v2::RetentionPolicy::default, || false)
            .expect("upkeep");
        assert_eq!(report.compressed, 0, "a pinned take is not compressed");
        drop(pin);
        let report = store
            .audio_upkeep(store_v2::RetentionPolicy::default, || false)
            .expect("upkeep");
        assert_eq!(report.compressed, 1);
        let after = store.audio_wav(&id).expect("load").expect("present");
        assert_eq!(*before, *after, "a retry from FLAC sends the same bytes");
    }

    fn session_status(store: &Store, id: &str) -> SessionStatus {
        summary_of(store, id).status
    }

    fn summary_of(store: &Store, id: &str) -> SessionSummary {
        let listed = store.list().expect("list");
        listed
            .iter()
            .find_map(|record| match record {
                ListedRecord::Session(summary) if summary.id == id => Some(summary.clone()),
                _ => None,
            })
            .unwrap_or_else(|| panic!("record {id} missing from listing"))
    }

    fn transcript_text(store: &Store, id: &str) -> Option<String> {
        let listed = store.list().expect("list");
        listed
            .iter()
            .find_map(|record| match record {
                ListedRecord::Session(summary) if summary.id == id => {
                    summary.transcript.as_ref().map(|t| t.text.clone())
                }
                _ => None,
            })
    }
}
