//! The app's history through the host (#220): every read and write of
//! the store the app makes — the history list, a take's audio for
//! playback and export, imports, deletes, retention classes, retry holds,
//! processing documents, insights and correction records — is a
//! [`StoreRequest`] the host answers on its own store handle. The app
//! holds no store handle; the host owns the store's lease and is its only
//! writer besides the processes it starts. History audio upkeep
//! (compression, retention, the sweep) runs here too, on the host's
//! schedule, and pauses while a take records ([`History::upkeep_loop`]).
//!
//! Wire: [`crate::frame::Frame::Store`] carries a request,
//! [`crate::frame::Frame::Stored`] its answer under the same `req`. An
//! answer too large for one frame — a long history, a take's audio — is
//! kept by the host for that connection and fetched in chunks
//! ([`StoreRequest::Fetch`]); an import's audio goes up the same way
//! ([`StoreRequest::Upload`]). What a connection was given — kept
//! answers, uploads, audio holds — goes with the connection.
//!
//! [`HistoryClient`] is the typed side the app calls, over any
//! [`StoreCall`]: a [`crate::client::HostClient`], or [`LocalHistory`] —
//! the same service run in-process (the app's tests' fake host).

mod facade;
mod service;

use std::sync::Arc;

use base64::Engine;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use starling_dictation::storage::{ListedRecord, StorageError};
use starling_dictation::store_v2::{CorrectionDecision, CorrectionRecord};

pub use facade::{
    v2_summary, AudioHold, AudioPin, Facade, ProcessingDoc, ProposalOrigin, ProposalRow,
    RowStatus, SavedTake, UpkeepReport,
};
pub use service::{History, LocalHistory, UPKEEP_FIRST, UPKEEP_INTERVAL};
pub(crate) use service::{start_workers, store_queue, submit, StoreJob};

/// Which encoding of a take's audio [`StoreRequest::Audio`] reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AudioFormat {
    /// The 16 kHz PCM16 WAV every transcription receives.
    Wav,
    /// Lossless FLAC of the same samples (#356).
    Flac,
}

/// One request to the host's store. What each answers is on its variant.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum StoreRequest {
    /// Every take, as the history list shows it: `Vec<ListedRecord>`.
    List,
    /// One take's audio: raw bytes, or `null` for an unknown take.
    Audio { id: String, format: AudioFormat },
    /// The take's current transcript and its attempt:
    /// `Option<(attempt id, text)>`.
    LatestRaw { id: String },
    /// `Option<ProcessingDoc>` for the take's current transcript.
    ProcessingDoc { id: String },
    /// The processing document for raw attempt `attempt_id`, started
    /// when there is none: `ProcessingDoc`.
    StartProcessingDoc {
        id: String,
        attempt_id: String,
        raw: String,
    },
    SaveProposal { id: String, proposal: ProposalRow },
    CommitProcessingHead {
        id: String,
        revision: u64,
        text: String,
        is_raw: bool,
        attempt_id: String,
        accepted: Option<ProposalRow>,
        derived_from: Option<String>,
    },
    RecordInsight {
        id: String,
        event_id: String,
        kind: String,
        occurred_at: String,
        payload_json: String,
    },
    /// `bool`: `false` when the take is excluded (secure field).
    RecordCorrection { record: CorrectionRecord },
    /// `bool`: `false` when there is no record to revise.
    ReviseCorrection {
        id: String,
        request_id: String,
        decision: CorrectionDecision,
        decision_utc: String,
        final_text: String,
    },
    Delete { id: String },
    SetArchival { id: String, archival: bool },
    /// Keeps the take's audio from every process's upkeep until
    /// [`StoreRequest::ReleaseHold`] or the connection ends: the hold's
    /// token.
    HoldAudio { id: String },
    ReleaseHold { hold: String },
    /// Appends `data` (base64) to upload `upload` at byte `offset`; an
    /// upload starts at offset 0.
    Upload {
        upload: String,
        offset: u64,
        data: String,
    },
    /// Stores the WAV uploaded as `upload` as a new take (`transcribe`:
    /// with the intent to transcribe it): the take's id.
    Import { upload: String, transcribe: bool },
    /// Runs the request uploaded as `upload` (its JSON): one too large
    /// for a frame, such as a proposal on a very long transcript.
    Uploaded { upload: String },
    /// The next chunk of a kept answer, from byte `offset`.
    Fetch { blob: String, offset: u64 },
    /// Drops a kept answer or an upload this connection gave up on.
    Discard { id: String },
    /// The latest upkeep report (`Option<String>`); `run` asks for a pass
    /// now (after the running one, if one runs).
    Upkeep { run: bool },
}

/// The host's answer to a [`StoreRequest`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum StoreReply {
    Done { value: Value },
    /// The answer's JSON is too large for one frame: `bytes` of it wait
    /// in `blob` ([`StoreRequest::Fetch`]).
    Large { blob: String, bytes: u64 },
    /// Raw bytes (audio) waiting in `blob`.
    Bytes { blob: String, bytes: u64 },
    /// One [`StoreRequest::Fetch`]: base64.
    Chunk { data: String },
    Failed { failure: StoreFailure },
}

/// Why a store request failed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StoreFailure {
    pub kind: StoreFailureKind,
    /// The take's id for [`StoreFailureKind::NotFound`]; otherwise what
    /// to show.
    pub message: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StoreFailureKind {
    /// The take is gone (or was never stored).
    NotFound,
    /// The store refused it as invalid.
    Invalid,
    /// A disk or database fault, or the host could not answer.
    Io,
}

impl From<&StorageError> for StoreFailure {
    fn from(err: &StorageError) -> StoreFailure {
        match err {
            StorageError::NotFound(id) => StoreFailure {
                kind: StoreFailureKind::NotFound,
                message: id.clone(),
            },
            StorageError::Invalid(reason) => StoreFailure {
                kind: StoreFailureKind::Invalid,
                message: reason.clone(),
            },
            other => StoreFailure {
                kind: StoreFailureKind::Io,
                message: other.to_string(),
            },
        }
    }
}

impl From<StoreFailure> for StorageError {
    fn from(failure: StoreFailure) -> StorageError {
        match failure.kind {
            StoreFailureKind::NotFound => StorageError::NotFound(failure.message),
            StoreFailureKind::Invalid => StorageError::Invalid(failure.message),
            StoreFailureKind::Io => StorageError::Io(std::io::Error::other(failure.message)),
        }
    }
}

/// How many raw bytes one chunk carries under a `max_frame_bytes` frame
/// cap: base64 grows them by a third, and the frame around them needs a
/// little room.
pub fn chunk_bytes(max_frame_bytes: usize) -> usize {
    (max_frame_bytes.saturating_sub(4096) / 4 * 3).max(1)
}

/// The most an answer fetched in chunks may claim to be: what any honest
/// answer stays far below (an hour of WAV is 115 MB).
const MAX_FETCH_BYTES: u64 = 4 * 1024 * 1024 * 1024;

/// One connection to a store service: what [`HistoryClient`] calls.
pub trait StoreCall {
    fn call(&self, request: StoreRequest) -> Result<StoreReply, StorageError>;
    /// How many raw bytes one upload chunk may carry.
    fn chunk_bytes(&self) -> usize;
}

impl<T: StoreCall + ?Sized> StoreCall for &T {
    fn call(&self, request: StoreRequest) -> Result<StoreReply, StorageError> {
        (**self).call(request)
    }

    fn chunk_bytes(&self) -> usize {
        (**self).chunk_bytes()
    }
}

impl<T: StoreCall + ?Sized> StoreCall for Arc<T> {
    fn call(&self, request: StoreRequest) -> Result<StoreReply, StorageError> {
        (**self).call(request)
    }

    fn chunk_bytes(&self) -> usize {
        (**self).chunk_bytes()
    }
}

/// The typed store operations over one connection.
pub struct HistoryClient<C>(pub C);

fn invalid(what: impl std::fmt::Display) -> StorageError {
    StorageError::Invalid(what.to_string())
}

impl<C: StoreCall> HistoryClient<C> {
    /// Sends `request`, through an upload when it is too large for a
    /// frame.
    fn call(&self, request: StoreRequest) -> Result<StoreReply, StorageError> {
        let plumbing = matches!(
            request,
            StoreRequest::Upload { .. }
                | StoreRequest::Fetch { .. }
                | StoreRequest::Discard { .. }
                | StoreRequest::ReleaseHold { .. }
        );
        if !plumbing {
            let json = serde_json::to_vec(&request)
                .map_err(|err| invalid(format!("the request does not serialize: {err}")))?;
            if json.len() > self.0.chunk_bytes() {
                let upload = self.upload(&json)?;
                return self.consume(upload, |upload| StoreRequest::Uploaded { upload });
            }
        }
        self.0.call(request)
    }

    /// Sends the request that consumes `upload`; when it fails (the host
    /// may not have taken it), the upload is dropped rather than left
    /// holding one of the connection's upload slots.
    fn consume(
        &self,
        upload: String,
        request: impl FnOnce(String) -> StoreRequest,
    ) -> Result<StoreReply, StorageError> {
        let reply = self.0.call(request(upload.clone()));
        if !matches!(reply, Ok(ref reply) if !matches!(reply, StoreReply::Failed { .. })) {
            let _ = self.0.call(StoreRequest::Discard { id: upload });
        }
        reply
    }

    /// Uploads `bytes` in chunks; the upload's id.
    fn upload(&self, bytes: &[u8]) -> Result<String, StorageError> {
        let upload = starling_runtime::bus::new_id("up");
        let mut offset = 0;
        for part in bytes.chunks(self.0.chunk_bytes()) {
            let sent = self.done(StoreRequest::Upload {
                upload: upload.clone(),
                offset: offset as u64,
                data: base64::engine::general_purpose::STANDARD.encode(part),
            });
            if let Err(err) = sent {
                let _ = self.0.call(StoreRequest::Discard { id: upload });
                return Err(err);
            }
            offset += part.len();
        }
        Ok(upload)
    }

    fn value<T: DeserializeOwned>(&self, request: StoreRequest) -> Result<T, StorageError> {
        let reply = self.call(request)?;
        self.value_of(reply)
    }

    fn value_of<T: DeserializeOwned>(&self, reply: StoreReply) -> Result<T, StorageError> {
        match reply {
            StoreReply::Done { value } => serde_json::from_value(value)
                .map_err(|err| invalid(format!("the recording service answered oddly: {err}"))),
            StoreReply::Large { blob, bytes } => {
                let json = self.fetch(&blob, bytes)?;
                serde_json::from_slice(&json)
                    .map_err(|err| invalid(format!("the recording service answered oddly: {err}")))
            }
            StoreReply::Failed { failure } => Err(failure.into()),
            other => Err(invalid(format!("the recording service answered {other:?}"))),
        }
    }

    fn done(&self, request: StoreRequest) -> Result<(), StorageError> {
        self.value::<Value>(request).map(|_| ())
    }

    fn bytes(&self, request: StoreRequest) -> Result<Option<Vec<u8>>, StorageError> {
        match self.call(request)? {
            StoreReply::Done { value: Value::Null } => Ok(None),
            StoreReply::Bytes { blob, bytes } => self.fetch(&blob, bytes).map(Some),
            StoreReply::Failed { failure } => Err(failure.into()),
            other => Err(invalid(format!("the recording service answered {other:?}"))),
        }
    }

    /// Reads kept answer `blob` (`bytes` long) chunk by chunk.
    fn fetch(&self, blob: &str, bytes: u64) -> Result<Vec<u8>, StorageError> {
        if bytes > MAX_FETCH_BYTES {
            let _ = self.0.call(StoreRequest::Discard { id: blob.to_string() });
            return Err(invalid(format!("an answer of {bytes} bytes is too large to read")));
        }
        let mut data = Vec::with_capacity(bytes.min(64 * 1024 * 1024) as usize);
        while (data.len() as u64) < bytes {
            let request = StoreRequest::Fetch {
                blob: blob.to_string(),
                offset: data.len() as u64,
            };
            let chunk = match self.0.call(request) {
                Ok(StoreReply::Chunk { data }) => base64::engine::general_purpose::STANDARD
                    .decode(data)
                    .map_err(|err| invalid(format!("a chunk was not base64: {err}"))),
                Ok(StoreReply::Failed { failure }) => Err(failure.into()),
                Ok(other) => Err(invalid(format!("the recording service answered {other:?}"))),
                Err(err) => Err(err),
            };
            match chunk {
                Ok(chunk) if !chunk.is_empty() => data.extend_from_slice(&chunk),
                Ok(_) => {
                    let _ = self.0.call(StoreRequest::Discard { id: blob.to_string() });
                    return Err(invalid("the recording service sent an empty chunk"));
                }
                Err(err) => {
                    let _ = self.0.call(StoreRequest::Discard { id: blob.to_string() });
                    return Err(err);
                }
            }
        }
        if data.len() as u64 != bytes {
            return Err(invalid("the recording service sent more than it announced"));
        }
        Ok(data)
    }

    /// Every take, as the history list shows it.
    pub fn list(&self) -> Result<Vec<ListedRecord>, StorageError> {
        self.value(StoreRequest::List)
    }

    /// One take's audio; `None` when the take is unknown.
    pub fn audio(&self, id: &str, format: AudioFormat) -> Result<Option<Vec<u8>>, StorageError> {
        self.bytes(StoreRequest::Audio {
            id: id.to_string(),
            format,
        })
    }

    pub fn latest_raw(&self, id: &str) -> Result<Option<(String, String)>, StorageError> {
        self.value(StoreRequest::LatestRaw { id: id.to_string() })
    }

    pub fn processing_doc(&self, id: &str) -> Result<Option<ProcessingDoc>, StorageError> {
        self.value(StoreRequest::ProcessingDoc { id: id.to_string() })
    }

    pub fn start_processing_doc(
        &self,
        id: &str,
        attempt_id: &str,
        raw: &str,
    ) -> Result<ProcessingDoc, StorageError> {
        self.value(StoreRequest::StartProcessingDoc {
            id: id.to_string(),
            attempt_id: attempt_id.to_string(),
            raw: raw.to_string(),
        })
    }

    pub fn save_proposal(&self, id: &str, proposal: &ProposalRow) -> Result<(), StorageError> {
        self.done(StoreRequest::SaveProposal {
            id: id.to_string(),
            proposal: proposal.clone(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn commit_processing_head(
        &self,
        id: &str,
        revision: u64,
        text: &str,
        is_raw: bool,
        attempt_id: &str,
        accepted: Option<&ProposalRow>,
        derived_from: Option<&str>,
    ) -> Result<(), StorageError> {
        self.done(StoreRequest::CommitProcessingHead {
            id: id.to_string(),
            revision,
            text: text.to_string(),
            is_raw,
            attempt_id: attempt_id.to_string(),
            accepted: accepted.cloned(),
            derived_from: derived_from.map(str::to_string),
        })
    }

    pub fn record_insight(
        &self,
        id: &str,
        event_id: &str,
        kind: &str,
        occurred_at: &str,
        payload_json: &str,
    ) -> Result<(), StorageError> {
        self.done(StoreRequest::RecordInsight {
            id: id.to_string(),
            event_id: event_id.to_string(),
            kind: kind.to_string(),
            occurred_at: occurred_at.to_string(),
            payload_json: payload_json.to_string(),
        })
    }

    pub fn record_correction(&self, record: &CorrectionRecord) -> Result<bool, StorageError> {
        self.value(StoreRequest::RecordCorrection {
            record: record.clone(),
        })
    }

    pub fn revise_correction(
        &self,
        id: &str,
        request_id: &str,
        decision: CorrectionDecision,
        decision_utc: &str,
        final_text: &str,
    ) -> Result<bool, StorageError> {
        self.value(StoreRequest::ReviseCorrection {
            id: id.to_string(),
            request_id: request_id.to_string(),
            decision,
            decision_utc: decision_utc.to_string(),
            final_text: final_text.to_string(),
        })
    }

    pub fn delete(&self, id: &str) -> Result<(), StorageError> {
        self.done(StoreRequest::Delete { id: id.to_string() })
    }

    pub fn set_archival(&self, id: &str, archival: bool) -> Result<(), StorageError> {
        self.done(StoreRequest::SetArchival {
            id: id.to_string(),
            archival,
        })
    }

    /// The hold's token, for [`Self::release_hold`].
    pub fn hold_audio(&self, id: &str) -> Result<String, StorageError> {
        self.value(StoreRequest::HoldAudio { id: id.to_string() })
    }

    pub fn release_hold(&self, hold: &str) -> Result<(), StorageError> {
        self.done(StoreRequest::ReleaseHold {
            hold: hold.to_string(),
        })
    }

    /// Stores `wav` as a new take (`transcribe`: with the intent to
    /// transcribe it); the take's id.
    pub fn import(&self, wav: &[u8], transcribe: bool) -> Result<String, StorageError> {
        let upload = self.upload(wav)?;
        let reply = self.consume(upload, |upload| StoreRequest::Import { upload, transcribe })?;
        self.value_of(reply)
    }

    /// The latest upkeep report; `run`: a pass runs now.
    pub fn upkeep(&self, run: bool) -> Result<Option<String>, StorageError> {
        self.value(StoreRequest::Upkeep { run })
    }
}
