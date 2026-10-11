//! The store behind the app's history (#220): the operations the app's
//! history, processing and imports run, on the host's own handle. Moved
//! from the app, which reaches them through [`super::StoreRequest`]s and
//! holds no store handle of its own. Rows map onto the v1-shaped types
//! the UI consumes (list, audio, imports, processing documents,
//! corrections, delete); a v2 failure surfaces its error.

use std::sync::{Arc, Mutex, MutexGuard};

use starling_dictation::{
    audio,
    storage::{
        self, DamagedRecord, ListedRecord, SessionStatus, SessionSummary, TranscriptionResult,
    },
    store_v2::{
        self, AttemptRecord, AudioAtRest, CaptureRecord, CaptureStatus, CompressionOutcome,
        HoldReason, ListedCapture, RetentionPolicy, RevisionRow, StoreV2,
        StoreV2Error,
    },
};

/// The `documents.name` of a take's processing document.
const PROCESSING_DOC: &str = "processing";

/// A take's durable processing state (#295).
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ProcessingDoc {
    pub head_revision: u64,
    pub head_text: String,
    pub head_is_raw: bool,
    pub raw_attempt_id: String,
    pub raw_text: String,
    pub proposals: Vec<ProposalRow>,
    /// The accepted proposal the head derives from, as the head records it.
    pub accepted_request: Option<String>,
    /// Deliveries the insertion-boundary rules adjusted (#341), oldest
    /// first. They never move the head.
    #[serde(default)]
    pub boundary: Vec<BoundaryRevision>,
}

/// One delivery the insertion-boundary rules adjusted (#341): the text
/// typed, a revision derived from the take's text as it was delivered.
/// It holds no text of the field it went into.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct BoundaryRevision {
    /// The text that was typed.
    pub text: String,
    /// The text it was derived from: the transcript, or a staged draft.
    pub source_text: String,
    /// The revision id of the head whose text that was, when it was one.
    pub derived_from: Option<String>,
    /// The contract kinds of the rules that fired, in rule order.
    pub changes: Vec<String>,
}

/// The provenance of a [`BoundaryRevision`]'s row.
const BOUNDARY_PROVENANCE: &str = "insertion-boundary";

/// Whether `row` is a boundary adjustment: a derived revision the
/// boundary rules stored, not one another feature derived.
fn is_boundary_row(row: &RevisionRow) -> bool {
    row.disposition.as_deref() == Some("derived")
        && row.provenance.as_deref() == Some(BOUNDARY_PROVENANCE)
}

/// A boundary row's `sources_json`; `derivedFrom` is the key the
/// runtime's document store reads a derived revision's source from.
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct BoundarySources {
    derived_from: Option<String>,
    source_text: String,
    changes: Vec<String>,
}

/// What produced a proposal, pinned at request time for the correction
/// records: the mode, the provider, and the pipeline's stage timings.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ProposalOrigin {
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
pub struct ProposalRow {
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
pub enum RowStatus {
    Proposed,
    Accepted,
    Rejected,
    Superseded,
    Failed,
}

impl RowStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            RowStatus::Proposed => "proposed",
            RowStatus::Accepted => "accepted",
            RowStatus::Rejected => "rejected",
            RowStatus::Superseded => "superseded",
            RowStatus::Failed => "failed",
        }
    }

    /// `None` for a value this build does not know; such a row is skipped.
    pub fn parse(value: &str) -> Option<RowStatus> {
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
        let boundary = document
            .revisions
            .iter()
            .filter(|row| is_boundary_row(row))
            .filter_map(|row| {
                let sources: BoundarySources =
                    serde_json::from_str(row.sources_json.as_deref()?).ok()?;
                Some(BoundaryRevision {
                    text: row.text.clone(),
                    source_text: sources.source_text,
                    derived_from: sources.derived_from,
                    changes: sources.changes,
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
            boundary,
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

/// Lock the shared v2 handle (a poisoned lock is recovered: the SQLite
/// connection is still consistent after a panic between statements).
fn lock_v2(handle: &Arc<Mutex<StoreV2>>) -> MutexGuard<'_, StoreV2> {
    handle
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// The host's history store: the host's v2 handle behind the app's
/// daily-use operations and the audio upkeep. The in-memory read pins
/// ([`Facade::pin_audio`]) keep audio from this handle's own upkeep.
#[derive(Clone)]
pub struct Facade(Arc<Mutex<StoreV2>>);

impl Facade {
    /// Over a handle the host shares.
    pub fn shared(store: Arc<Mutex<StoreV2>>) -> Facade {
        Facade(store)
    }

    /// The store's root directory.
    pub fn root(&self) -> std::path::PathBuf {
        lock_v2(&self.0).root().to_path_buf()
    }

    /// Metadata-only listing (G02): readable records as summaries, damaged
    /// ones flagged with their reason. The store pages through bounded
    /// reads and maps each capture + its recognition attempts onto the
    /// v1-shaped summary the history list consumes.
    pub fn list(&self) -> Result<Vec<ListedRecord>, storage::StorageError> {
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
    pub fn audio_wav(&self, id: &str) -> Result<Option<Arc<Vec<u8>>>, storage::StorageError> {
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
    pub fn audio_flac(&self, id: &str) -> Result<Option<Arc<Vec<u8>>>, storage::StorageError> {
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
    /// protocol. The stored duration is derived from the samples
    /// themselves. `transcribe`: the intent is stored with the take
    /// (#220), so the host transcribes it even if the window that
    /// imported it goes away before it asks.
    pub fn save_import(&self, wav: &[u8], transcribe: bool) -> Result<SavedTake, storage::StorageError> {
        let pcm = decode_wav(wav)?;
        let id = self.save_pcm_take(pcm, store_v2::CommitMark::Complete, transcribe)?;
        Ok(SavedTake { id })
    }

    /// [`Self::save_import`] without the intent (tests' fixtures).
    #[cfg(test)]
    pub(crate) fn save_capture(&self, wav: Arc<Vec<u8>>) -> Result<SavedTake, storage::StorageError> {
        self.save_import(&wav, false)
    }

    /// Marks a transcription attempt as started on the record (tests; the
    /// recording service writes attempts). `backend` labels the attempt
    /// ("starling:parakeet", "openai:whisper-large-v3").
    #[cfg(test)]
    pub(crate) fn mark_attempt(&self, id: &str, backend: &str) -> Result<(), storage::StorageError> {
        let mut store = lock_v2(&self.0);
        store
            .begin_recognition(id, backend, None)
            .map(|_| ())
            .map_err(v2_err)
    }

    /// Records a successful transcript: the in-flight attempt is completed
    /// with the transcript preserved verbatim in its row.
    #[cfg(test)]
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

    /// Move a take into the archival retention class or back (#342).
    pub fn set_archival(&self, id: &str, archival: bool) -> Result<(), storage::StorageError> {
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
    pub fn delete(&self, id: &str) -> Result<(), storage::StorageError> {
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
    pub fn latest_raw(&self, id: &str) -> Result<Option<(String, String)>, storage::StorageError> {
        let store = lock_v2(&self.0);
        let attempts = store.attempts_for(id).map_err(v2_err)?;
        Ok(current_transcript(&attempts).map(|attempt| (attempt.id.clone(), attempt.text.clone())))
    }

    /// The take's processing document, if processing ran on its current
    /// raw transcript. A document built on an earlier transcript (the take
    /// was re-transcribed since) is stale and reads as none: what was used
    /// or proposed for the old text does not apply to the new one.
    pub fn processing_doc(&self, id: &str) -> Result<Option<ProcessingDoc>, storage::StorageError> {
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
    pub fn start_processing_doc(
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
            boundary: Vec::new(),
        })
    }

    /// Records a delivery the insertion-boundary rules adjusted (#341):
    /// `text` was typed for the take's `source_text`, with the rules
    /// `changes`. A derived revision in the take's processing document
    /// as it stands (started on the current transcript only when there is
    /// none; one built on an earlier transcript is kept, never started
    /// over here): the head, the raw transcript and every other revision
    /// stay as they are, so the unadjusted text is still what Copy and
    /// history give.
    pub fn record_boundary_revision(
        &self,
        id: &str,
        source_text: &str,
        text: &str,
        changes: &[String],
    ) -> Result<(), storage::StorageError> {
        let store = lock_v2(&self.0);
        if store.get_capture(id).map_err(v2_err)?.is_none() {
            return Err(storage::StorageError::NotFound(id.to_string()));
        }
        let document = match store.get_document(id).map_err(v2_err)? {
            Some(document) => document,
            None => {
                let attempts = store.attempts_for(id).map_err(v2_err)?;
                let Some(attempt) = current_transcript(&attempts) else {
                    return Err(storage::StorageError::NotFound(id.to_string()));
                };
                let raw = head_row(id, 1, &attempt.text, true, &attempt.id, None);
                store.commit_document_head(PROCESSING_DOC, 1, 0, &raw).map_err(v2_err)?;
                store
                    .get_document(id)
                    .map_err(v2_err)?
                    .ok_or_else(|| storage::StorageError::NotFound(id.to_string()))?
            }
        };
        let count = document
            .revisions
            .iter()
            .filter(|row| is_boundary_row(row))
            .count();
        // The latest committed head holding exactly the delivered text.
        let derived_from = (1..=document.head_revision)
            .rev()
            .map(|revision| head_rev_id(id, revision))
            .find(|rev_id| {
                document
                    .revisions
                    .iter()
                    .any(|row| &row.rev_id == rev_id && row.text == source_text)
            });
        let row = RevisionRow {
            rev_id: format!("{id}#b{}", count + 1),
            doc_id: id.to_string(),
            base_revision: Some(document.head_revision),
            // The only record of the source and the rules: never stored
            // without it.
            sources_json: Some(
                serde_json::to_string(&BoundarySources {
                    derived_from,
                    source_text: source_text.to_string(),
                    changes: changes.to_vec(),
                })
                .map_err(|error| {
                    storage::StorageError::Invalid(format!("boundary sources: {error}"))
                })?,
            ),
            text: text.to_string(),
            status: "derived".to_string(),
            provenance: Some(BOUNDARY_PROVENANCE.to_string()),
            disposition: Some("derived".to_string()),
        };
        store.store_document_revision(&row).map_err(v2_err)
    }

    /// Stores (or updates) one proposal row. A take deleted meanwhile is
    /// `NotFound`: the result lands nowhere.
    pub fn save_proposal(&self, id: &str, proposal: &ProposalRow) -> Result<(), storage::StorageError> {
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
    pub fn commit_processing_head(
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
    pub fn record_insight(
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
    pub fn record_correction(
        &self,
        record: &store_v2::CorrectionRecord,
    ) -> Result<bool, storage::StorageError> {
        let store = lock_v2(&self.0);
        store.upsert_correction_record(record).map_err(v2_err)
    }

    /// Revises the decision of an existing correction record; `Ok(false)`
    /// when there is none.
    pub fn revise_correction(
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

    /// Hold `id`'s audio against every process's upkeep until the guard
    /// drops (#220: a retry the host has not started yet).
    pub fn hold_audio(&self, id: &str) -> Result<AudioHold, storage::StorageError> {
        let hold = lock_v2(&self.0).hold_audio(id).map_err(v2_err)?;
        Ok(AudioHold {
            store: Arc::clone(&self.0),
            id: hold,
        })
    }

    /// Pin `id`'s audio until the returned guard drops (#342): upkeep
    /// neither compresses nor retires it meanwhile.
    pub fn pin_audio(&self, id: &str) -> AudioPin {
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
    ///
    /// `policy` and `paused` run while the store lock is held: neither may
    /// call back into this `Facade` (or anything that takes its lock), or
    /// the pass deadlocks.
    pub fn audio_upkeep<P: std::borrow::Borrow<RetentionPolicy>>(
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
            // A take that started recording while this one was encoded, or
            // while its publish waited for the store: its FLAC is not
            // published now (the next pass encodes it again).
            // The guard must drop before the arms: a failure locks again.
            let committed = lock_v2(&self.0).commit_compression_unless(prepared, &paused);
            if matches!(committed, Ok(CompressionOutcome::Skipped(_))) && paused() {
                report.paused = true;
                return Ok(report);
            }
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
        transcribe: bool,
    ) -> Result<String, storage::StorageError> {
        let rate = pcm.sample_rate;
        let mut meta = store_v2::TakeMeta::for_device("");
        meta.transcribe = transcribe;
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

/// A [`Facade::hold_audio`] guard: the take's audio is kept from every
/// process's upkeep until it drops. Dropping it takes the store lock:
/// never drop one while holding that lock.
pub struct AudioHold {
    store: Arc<Mutex<StoreV2>>,
    id: String,
}

impl Drop for AudioHold {
    fn drop(&mut self) {
        if let Err(err) = lock_v2(&self.store).release_audio_hold(&self.id) {
            // Left for its holder's exit to void (a hold whose holder is
            // gone counts for nothing).
            eprintln!("Starling: releasing an audio hold failed: {err}");
        }
    }
}

/// A [`Facade::pin_audio`] guard. Dropping it takes the store lock: never
/// drop one while holding that lock.
pub struct AudioPin {
    store: Arc<Mutex<StoreV2>>,
    id: String,
}

impl Drop for AudioPin {
    fn drop(&mut self) {
        lock_v2(&self.store).unpin_audio(&self.id);
    }
}

/// What one [`Facade::audio_upkeep`] pass did.
#[derive(Debug, Default)]
pub struct UpkeepReport {
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
    pub fn summary(&self) -> Option<String> {
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

/// What a persisted import hands back: the record id.
pub struct SavedTake {
    pub id: String,
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
pub fn v2_summary(
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

    fn v2_store(tag: &str) -> Facade {
        let root = scratch_dir(tag);
        reopen_v2(&root)
    }

    /// The facade over an existing scratch root — the "previous run
    /// crashed" shape (#213): a fresh open of the same directory holds no
    /// in-memory attempt ownership, exactly like a new process.
    fn reopen_v2(root: &std::path::Path) -> Facade {
        let store = StoreV2::open(root.join("v2")).expect("reopen v2");
        Facade(Arc::new(Mutex::new(store)))
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

    // ---- the facade over a real v2 store (daily path) ------------------

    fn transcribed(store: &Facade, text: &str) -> String {
        let id = store.save_capture(tiny_wav(160)).expect("save").id;
        store.mark_attempt(&id, "starling:parakeet").expect("begin");
        store.save_transcript(&id, transcript(text)).expect("transcript");
        id
    }

    /// A take that starts recording while another is being encoded: the
    /// encode is not published, its temporary is gone, and the next pass
    /// compresses it.
    #[test]
    fn a_take_starting_mid_encode_keeps_that_compression_unpublished() {
        let store = v2_store("upkeep-paused-mid-encode");
        let saved = store.save_capture(wav_of(&[0.1f32; 32_000])).expect("save");
        let asked = std::sync::atomic::AtomicUsize::new(0);
        // Asked before the encode (not yet recording), then before the
        // publish (recording).
        let upkeep = store
            .audio_upkeep(store_v2::RetentionPolicy::default, || {
                asked.fetch_add(1, std::sync::atomic::Ordering::SeqCst) >= 1
            })
            .expect("upkeep");
        assert!(upkeep.paused);
        assert_eq!(upkeep.compressed, 0);
        let audio = lock_v2(&store.0).root().join("audio");
        let names: Vec<String> = std::fs::read_dir(&audio)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names, vec![format!("{}.sj", saved.id)], "only the journal: {names:?}");
        let upkeep = store
            .audio_upkeep(store_v2::RetentionPolicy::default, || false)
            .expect("upkeep");
        assert_eq!(upkeep.compressed, 1);
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

}
