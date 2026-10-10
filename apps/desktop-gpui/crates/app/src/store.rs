//! The app's store is the runtime host's (#220). History reads, a take's
//! audio, imports, deletes, retention classes, retry holds, processing
//! documents, insights and correction records are requests the host
//! answers on its own store handle ([`starling_runtime_host::history`]);
//! the app opens no store and writes nothing itself. History audio
//! upkeep runs in the host too.
//!
//! [`Store`] keeps the facade the UI always called — the same operations,
//! blocking, run off the UI thread — over the link's current connection.
//! While there is none it waits a little for the link to reconnect, then
//! fails with an error that says so.

use std::sync::{Arc, Mutex, Weak};
use std::time::{Duration, Instant};

use starling_dictation::storage::{self, ListedRecord};
use starling_dictation::store_v2;
use starling_runtime_host::client::HostClient;
use starling_runtime_host::history::{AudioFormat, HistoryClient, StoreRequest};

pub(crate) use starling_runtime_host::history::{
    ProcessingDoc, ProposalOrigin, ProposalRow, RowStatus,
};

/// How long a request waits for the link to (re)connect.
#[cfg(not(test))]
const CONNECT_WAIT: Duration = Duration::from_secs(10);
#[cfg(test)]
const CONNECT_WAIT: Duration = Duration::from_secs(2);

/// What a persisted import hands back: the record id.
pub(crate) struct SavedTake {
    pub(crate) id: String,
}

/// The store, through the host.
#[derive(Clone)]
pub(crate) struct Store(Arc<Backend>);

enum Backend {
    /// The connection the host link holds (none while it has none).
    Host(Arc<Mutex<Option<Arc<HostClient>>>>),
    /// The host's store service run in this process: the tests' fake
    /// host, with the store under it for their fixtures.
    #[cfg(test)]
    Local {
        service: starling_runtime_host::history::LocalHistory,
        fixtures: Mutex<store_v2::StoreV2>,
    },
}

fn unreachable() -> storage::StorageError {
    storage::StorageError::Io(std::io::Error::other(
        "Starling's recording service is not reachable right now; try again once it is",
    ))
}

impl Store {
    /// The store of the host `connection` reaches (the link's current
    /// connection).
    pub(crate) fn through(connection: Arc<Mutex<Option<Arc<HostClient>>>>) -> Store {
        Store(Arc::new(Backend::Host(connection)))
    }

    /// Runs `operation` on the host's store over one connection.
    fn with<T>(
        &self,
        operation: impl FnOnce(&HistoryClient<&dyn starling_runtime_host::history::StoreCall>) -> Result<T, storage::StorageError>,
    ) -> Result<T, storage::StorageError> {
        match &*self.0 {
            Backend::Host(connection) => {
                let client = connected(connection)?;
                let call: &dyn starling_runtime_host::history::StoreCall = &*client;
                operation(&HistoryClient(call))
            }
            #[cfg(test)]
            Backend::Local { service, .. } => {
                let call: &dyn starling_runtime_host::history::StoreCall = service;
                operation(&HistoryClient(call))
            }
        }
    }

    /// Every take, metadata only (G02): readable records as summaries,
    /// damaged ones flagged with their reason.
    pub(crate) fn list(&self) -> Result<Vec<ListedRecord>, storage::StorageError> {
        self.with(|history| history.list())
    }

    /// One recording's WAV, loaded on demand (G02). `Ok(None)` when the id
    /// is unknown — the delete race is indistinguishable from a missing
    /// record and is not an error.
    pub(crate) fn audio_wav(&self, id: &str) -> Result<Option<Arc<Vec<u8>>>, storage::StorageError> {
        self.with(|history| history.audio(id, AudioFormat::Wav))
            .map(|audio| audio.map(Arc::new))
    }

    /// One recording as lossless FLAC (#356 export): the same samples as
    /// [`Self::audio_wav`]. `Ok(None)` when the id is unknown.
    pub(crate) fn audio_flac(&self, id: &str) -> Result<Option<Arc<Vec<u8>>>, storage::StorageError> {
        self.with(|history| history.audio(id, AudioFormat::Flac))
            .map(|audio| audio.map(Arc::new))
    }

    /// Stores an imported take with the intent to transcribe it (#220):
    /// the host transcribes it even if this window goes away before it
    /// asks.
    pub(crate) fn save_import(&self, wav: Arc<Vec<u8>>) -> Result<SavedTake, storage::StorageError> {
        self.with(|history| history.import(&wav, true))
            .map(|id| SavedTake { id })
    }

    /// Move a take into the archival retention class or back (#342).
    pub(crate) fn set_archival(&self, id: &str, archival: bool) -> Result<(), storage::StorageError> {
        self.with(|history| history.set_archival(id, archival))
    }

    /// Confirmed deletion (R21): the audio is quarantined, the row
    /// tombstoned, and the take's processing document, insight events and
    /// correction records go with it.
    pub(crate) fn delete(&self, id: &str) -> Result<(), storage::StorageError> {
        self.with(|history| history.delete(id))
    }

    /// The take's latest final transcript and its attempt id.
    pub(crate) fn latest_raw(&self, id: &str) -> Result<Option<(String, String)>, storage::StorageError> {
        self.with(|history| history.latest_raw(id))
    }

    /// The take's processing document, if processing ran on its current
    /// raw transcript (#295).
    pub(crate) fn processing_doc(&self, id: &str) -> Result<Option<ProcessingDoc>, storage::StorageError> {
        self.with(|history| history.processing_doc(id))
    }

    /// The processing document for the take's current raw transcript,
    /// started over when it was built on an earlier one.
    pub(crate) fn start_processing_doc(
        &self,
        id: &str,
        attempt_id: &str,
        raw: &str,
    ) -> Result<ProcessingDoc, storage::StorageError> {
        self.with(|history| history.start_processing_doc(id, attempt_id, raw))
    }

    /// Stores (or updates) one proposal row. A take deleted meanwhile is
    /// `NotFound`: the result lands nowhere.
    pub(crate) fn save_proposal(&self, id: &str, proposal: &ProposalRow) -> Result<(), storage::StorageError> {
        self.with(|history| history.save_proposal(id, proposal))
    }

    /// Commits a new head revision (accepted processed text, or raw
    /// again), marking the accepted proposal with it.
    #[allow(clippy::too_many_arguments)]
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
        self.with(|history| {
            history.commit_processing_head(id, revision, text, is_raw, attempt_id, accepted, derived_from)
        })
    }

    /// Records one insight event for the take (idempotent on its id).
    pub(crate) fn record_insight(
        &self,
        id: &str,
        event_id: &str,
        kind: &str,
        occurred_at: &str,
        payload_json: &str,
    ) -> Result<(), storage::StorageError> {
        self.with(|history| history.record_insight(id, event_id, kind, occurred_at, payload_json))
    }

    /// Writes or revises one correction record; `Ok(false)` when the take
    /// is excluded (secure field).
    pub(crate) fn record_correction(
        &self,
        record: &store_v2::CorrectionRecord,
    ) -> Result<bool, storage::StorageError> {
        self.with(|history| history.record_correction(record))
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
        self.with(|history| history.revise_correction(id, request_id, decision, decision_utc, final_text))
    }

    /// Hold `id`'s audio against every process's upkeep until the guard
    /// drops (#220: a retry the host has not started yet). The host keeps
    /// it for the connection that asked: a connection that ends releases
    /// it.
    pub(crate) fn hold_audio(&self, id: &str) -> Result<AudioHold, storage::StorageError> {
        match &*self.0 {
            Backend::Host(connection) => {
                let client = connected(connection)?;
                let token = HistoryClient(&*client).hold_audio(id)?;
                Ok(AudioHold {
                    token,
                    on: HoldOn::Host(Arc::downgrade(&client)),
                })
            }
            #[cfg(test)]
            Backend::Local { service, .. } => {
                let token = HistoryClient(service).hold_audio(id)?;
                Ok(AudioHold {
                    token,
                    on: HoldOn::Local(self.clone()),
                })
            }
        }
    }

    /// The host's latest history audio upkeep report; `run`: a pass runs
    /// now (the storage settings changed).
    pub(crate) fn upkeep(&self, run: bool) -> Result<Option<String>, storage::StorageError> {
        self.with(|history| history.upkeep(run))
    }
}

/// The connection `connection` holds, waiting up to [`CONNECT_WAIT`] for
/// one.
fn connected(
    connection: &Mutex<Option<Arc<HostClient>>>,
) -> Result<Arc<HostClient>, storage::StorageError> {
    let until = Instant::now() + CONNECT_WAIT;
    loop {
        let current = connection
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();
        match current {
            Some(client) if !client.is_closed() => return Ok(client),
            _ if Instant::now() >= until => return Err(unreachable()),
            _ => std::thread::sleep(Duration::from_millis(50)),
        }
    }
}

/// A [`Store::hold_audio`] guard: the take's audio is kept from every
/// process's upkeep until it drops. Dropping it never waits: the release
/// is sent without waiting for its answer.
pub(crate) struct AudioHold {
    token: String,
    on: HoldOn,
}

enum HoldOn {
    /// The connection the host keeps the hold for; gone with it.
    Host(Weak<HostClient>),
    #[cfg(test)]
    Local(Store),
}

impl Drop for AudioHold {
    fn drop(&mut self) {
        let release = StoreRequest::ReleaseHold {
            hold: std::mem::take(&mut self.token),
        };
        match &self.on {
            HoldOn::Host(client) => {
                // A connection that is gone took the hold with it. The
                // release is written off this thread: a guard may drop on
                // the UI thread, and the connection's writer may be busy
                // (an import's upload) or stuck behind a host that stopped
                // reading.
                if let Some(client) = client.upgrade() {
                    let spawned = std::thread::Builder::new()
                        .name("starling-hold-release".to_string())
                        .spawn(move || {
                            let _ = client.store_unanswered(release);
                        });
                    if let Err(err) = spawned {
                        eprintln!("Starling: an audio hold is kept until reconnect: {err}");
                    }
                }
            }
            #[cfg(test)]
            HoldOn::Local(store) => {
                if let Backend::Local { service, .. } = &*store.0 {
                    use starling_runtime_host::history::StoreCall;
                    let _ = service.call(release);
                }
            }
        }
    }
}

#[cfg(test)]
impl Store {
    /// The host's store service over `root`, in this process (the tests'
    /// fake host), with the store under it for fixtures.
    pub(crate) fn at_test_root(root: &std::path::Path) -> Self {
        Store(Arc::new(Backend::Local {
            service: starling_runtime_host::history::LocalHistory::open(root).unwrap(),
            fixtures: Mutex::new(store_v2::StoreV2::open(root).unwrap()),
        }))
    }

    fn fixtures(&self) -> std::sync::MutexGuard<'_, store_v2::StoreV2> {
        match &*self.0 {
            Backend::Local { fixtures, .. } => fixtures
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()),
            Backend::Host(_) => panic!("fixtures need a test store"),
        }
    }

    /// A take stored as another process would (the host, recording).
    pub(crate) fn save_capture(&self, wav: Arc<Vec<u8>>) -> Result<SavedTake, storage::StorageError> {
        let id = self
            .fixtures()
            .save_wav_capture(&wav, store_v2::TakeMeta::for_device(""))
            .map_err(|err| storage::StorageError::Invalid(err.to_string()))?
            .record
            .id;
        Ok(SavedTake { id })
    }

    /// A short silent take marked as captured against a secure field
    /// (the desktop app has no such capture path).
    pub(crate) fn save_secure_take(&self) -> String {
        let mut meta = store_v2::TakeMeta::for_device("");
        meta.secure_field = true;
        let mut v2 = self.fixtures();
        let mut take = v2.begin_take_at_rate(16_000, meta).unwrap();
        take.append_and_seal(&[0.0; 160]).unwrap();
        take.finalize()
            .unwrap()
            .commit_marked(&mut v2, store_v2::CommitMark::Complete)
            .unwrap()
            .record
            .id
    }

    /// Marks a transcription attempt as started on the record (the host
    /// writes attempts; tests stand in for it).
    pub(crate) fn mark_attempt(&self, id: &str, backend: &str) -> Result<(), storage::StorageError> {
        self.fixtures()
            .begin_recognition(id, backend, None)
            .map(|_| ())
            .map_err(|err| storage::StorageError::Invalid(err.to_string()))
    }

    /// Completes the in-flight attempt with `transcript`.
    pub(crate) fn save_transcript(
        &self,
        id: &str,
        transcript: storage::TranscriptionResult,
    ) -> Result<(), storage::StorageError> {
        self.fixtures()
            .finish_recognition_transcript(id, &transcript)
            .map_err(|err| storage::StorageError::Invalid(err.to_string()))
    }

    /// Fails the in-flight attempt.
    pub(crate) fn save_failure(&self, id: &str, message: &str) -> Result<(), storage::StorageError> {
        self.fixtures()
            .finish_recognition(id, store_v2::RecognitionOutcome::Failed { message })
            .map_err(|err| storage::StorageError::Invalid(err.to_string()))
    }

    pub(crate) fn correction_records(
        &self,
        id: &str,
    ) -> Result<Vec<store_v2::CorrectionRecord>, storage::StorageError> {
        self.fixtures()
            .correction_records_for(id)
            .map_err(|err| storage::StorageError::Invalid(err.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_wav(samples: usize) -> Arc<Vec<u8>> {
        let pcm = starling_dictation::audio::PcmAudio {
            samples: (0..samples).map(|i| (i % 97) as f32 * 0.001).collect(),
            sample_rate: 16_000,
            channels: 1,
        };
        Arc::new(starling_dictation::audio::encode_wav_16k(&pcm).expect("encode wav"))
    }

    /// The app's daily path goes through the host's store service: an
    /// import, the list, its audio, a delete — and a hold released when
    /// its guard drops.
    #[test]
    fn the_daily_path_runs_through_the_store_service() {
        let root = tempfile::tempdir().unwrap();
        let store = Store::at_test_root(root.path());
        let wav = tiny_wav(300);
        let id = store.save_import(wav.clone()).expect("import").id;
        let listed = store.list().expect("list");
        assert!(matches!(&listed[..], [ListedRecord::Session(summary)] if summary.id == id));
        assert_eq!(*store.audio_wav(&id).unwrap().unwrap(), *wav);
        assert!(store.audio_flac(&id).unwrap().unwrap().starts_with(b"fLaC"));
        let hold = store.hold_audio(&id).expect("held");
        drop(hold);
        store.set_archival(&id, true).expect("archive");
        store.delete(&id).expect("delete");
        assert!(store.list().unwrap().is_empty());
        assert!(store.audio_wav(&id).unwrap().is_none());
    }

    /// With no connection the store says so, after a short wait.
    #[test]
    fn without_a_connection_the_store_says_the_service_is_unreachable() {
        let store = Store::through(Arc::new(Mutex::new(None)));
        let started = Instant::now();
        let err = store.list().expect_err("no connection");
        assert!(started.elapsed() >= CONNECT_WAIT);
        assert!(err.to_string().contains("not reachable"), "{err}");
    }
}
