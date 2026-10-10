//! The store service: [`History`] answers [`StoreRequest`]s on the host's
//! [`Facade`], keeping per connection what a request handed out (answers
//! fetched in chunks, uploads, audio holds) until it is used up, dropped,
//! or the connection ends. [`start_workers`] runs requests off the
//! connection readers; [`History::upkeep_loop`] runs the audio upkeep.

use std::collections::{HashMap, VecDeque};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, RecvTimeoutError, SyncSender, TrySendError};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use base64::Engine;
use serde::Serialize;
use serde_json::Value;
use starling_dictation::storage::StorageError;
use starling_dictation::store_v2::RetentionPolicy;

use super::{
    chunk_bytes, AudioFormat, AudioHold, Facade, StoreCall, StoreFailure, StoreReply,
    StoreRequest,
};
use crate::frame::Frame;
use crate::server::ConnState;

/// How many answers one connection may have waiting to be fetched; the
/// oldest goes first.
const MAX_BLOBS: usize = 8;
/// How many bytes of waiting answers one connection may hold in memory;
/// an answer that does not fit beside them waits in a spool file instead.
const MAX_BLOB_BYTES: usize = 1024 * 1024 * 1024;
/// Where spooled answers wait, under the store's root. The files are
/// unlinked (deleted on close on Windows): nothing outlives the host.
const SPOOL_DIR: &str = "spool";
/// How many uploads one connection may have open, and their bytes.
const MAX_UPLOADS: usize = 4;
const MAX_UPLOAD_BYTES: usize = 1024 * 1024 * 1024;

/// How many requests run at once, and how many may wait.
const WORKERS: usize = 3;
const QUEUE: usize = 256;

/// When the host's first upkeep pass runs (after startup recovery and the
/// transcriber's first claims), how often it runs after that, and how
/// soon a pass a recording take paused is tried again.
pub const UPKEEP_FIRST: Duration = Duration::from_secs(10);
pub const UPKEEP_INTERVAL: Duration = Duration::from_secs(15 * 60);
const UPKEEP_PAUSED_RETRY: Duration = Duration::from_secs(60);

/// Who a request comes from: what the service keeps for it is keyed by
/// `key`, and nothing is kept for a caller that is `gone`.
pub(crate) trait Caller {
    fn key(&self) -> usize;
    fn gone(&self) -> bool;
}

impl Caller for ConnState {
    fn key(&self) -> usize {
        self as *const ConnState as usize
    }

    fn gone(&self) -> bool {
        self.closed.load(std::sync::atomic::Ordering::SeqCst)
    }
}

/// An answer waiting to be fetched: in memory, or in a spool file when it
/// did not fit under the connection's memory bound.
#[derive(Clone)]
enum Kept {
    Memory(Arc<Vec<u8>>),
    Spooled { file: Arc<Mutex<File>>, len: usize },
}

impl Kept {
    fn len(&self) -> usize {
        match self {
            Kept::Memory(bytes) => bytes.len(),
            Kept::Spooled { len, .. } => *len,
        }
    }

    /// The bytes it holds in memory.
    fn in_memory(&self) -> usize {
        match self {
            Kept::Memory(bytes) => bytes.len(),
            Kept::Spooled { .. } => 0,
        }
    }

    fn read(&self, start: usize, end: usize) -> Result<Vec<u8>, StorageError> {
        match self {
            Kept::Memory(bytes) => Ok(bytes[start..end].to_vec()),
            Kept::Spooled { file, .. } => {
                let mut file = lock(file);
                let mut part = vec![0; end - start];
                file.seek(SeekFrom::Start(start as u64))?;
                file.read_exact(&mut part)?;
                Ok(part)
            }
        }
    }
}

/// What one connection was handed and has not used up.
#[derive(Default)]
struct Stash {
    blobs: VecDeque<(String, Kept)>,
    uploads: HashMap<String, Vec<u8>>,
    holds: HashMap<String, AudioHold>,
}

impl Stash {
    fn blob_bytes(&self) -> usize {
        self.blobs.iter().map(|(_, kept)| kept.in_memory()).sum()
    }

    /// Keeps `kept` as answer `id`; past [`MAX_BLOBS`] the oldest goes.
    fn push_blob(&mut self, id: String, kept: Kept) {
        self.blobs.push_back((id, kept));
        while self.blobs.len() > MAX_BLOBS {
            self.blobs.pop_front();
        }
    }
}

/// The store service.
pub struct History {
    facade: Facade,
    /// Raw bytes per fetched chunk, and the largest answer sent whole.
    chunk: usize,
    inline: usize,
    /// [`MAX_BLOB_BYTES`]; smaller in tests.
    memory_limit: usize,
    spool: PathBuf,
    stashes: Mutex<HashMap<usize, Stash>>,
    /// Releases run on threads of their own ([`Self::release_later`]):
    /// joined at shutdown, before the lease goes.
    releases: Mutex<Vec<JoinHandle<()>>>,
    upkeep: UpkeepState,
    /// Told when a request changed the history list (see
    /// [`History::on_change`]).
    changed: std::sync::OnceLock<Box<dyn Fn() + Send + Sync>>,
}

#[derive(Default)]
struct UpkeepState {
    last: Mutex<Option<String>>,
    /// A pass was asked for.
    wanted: Mutex<bool>,
    wake: Condvar,
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn json(value: impl Serialize) -> Result<Value, StorageError> {
    serde_json::to_value(value)
        .map_err(|err| StorageError::Invalid(format!("the answer does not serialize: {err}")))
}

fn new_id(prefix: &str) -> String {
    starling_runtime::bus::new_id(prefix)
}

impl History {
    /// The service over `facade`, answering connections whose frames are
    /// capped at `max_frame_bytes`.
    pub fn new(facade: Facade, max_frame_bytes: usize) -> History {
        History {
            spool: facade.root().join(SPOOL_DIR),
            facade,
            chunk: chunk_bytes(max_frame_bytes),
            inline: max_frame_bytes.saturating_sub(4096).max(1),
            memory_limit: MAX_BLOB_BYTES,
            stashes: Mutex::new(HashMap::new()),
            releases: Mutex::new(Vec::new()),
            upkeep: UpkeepState::default(),
            changed: std::sync::OnceLock::new(),
        }
    }

    /// `changed` runs whenever a request changed the history list (a
    /// delete, an import, a retention class): other windows show it.
    pub fn on_change(&self, changed: impl Fn() + Send + Sync + 'static) {
        let _ = self.changed.set(Box::new(changed));
    }

    fn changed(&self) {
        if let Some(changed) = self.changed.get() {
            changed();
        }
    }

    pub fn facade(&self) -> &Facade {
        &self.facade
    }

    /// Answers `request` from `caller`.
    pub(crate) fn handle(&self, caller: &dyn Caller, request: StoreRequest) -> StoreReply {
        match self.run(caller, request) {
            Ok(reply) => reply,
            Err(err) => StoreReply::Failed {
                failure: StoreFailure::from(&err),
            },
        }
    }

    fn run(&self, caller: &dyn Caller, request: StoreRequest) -> Result<StoreReply, StorageError> {
        let facade = &self.facade;
        let value = match request {
            StoreRequest::List => json(facade.list()?)?,
            StoreRequest::Audio { id, format } => {
                let audio = match format {
                    AudioFormat::Wav => facade.audio_wav(&id)?,
                    AudioFormat::Flac => facade.audio_flac(&id)?,
                };
                return Ok(match audio {
                    None => StoreReply::Done { value: Value::Null },
                    Some(bytes) => {
                        let len = bytes.len() as u64;
                        let blob = self.keep(caller, bytes)?;
                        StoreReply::Bytes { blob, bytes: len }
                    }
                });
            }
            StoreRequest::LatestRaw { id } => json(facade.latest_raw(&id)?)?,
            StoreRequest::ProcessingDoc { id } => json(facade.processing_doc(&id)?)?,
            StoreRequest::StartProcessingDoc {
                id,
                attempt_id,
                raw,
            } => json(facade.start_processing_doc(&id, &attempt_id, &raw)?)?,
            StoreRequest::SaveProposal { id, proposal } => {
                facade.save_proposal(&id, &proposal)?;
                Value::Null
            }
            StoreRequest::CommitProcessingHead {
                id,
                revision,
                text,
                is_raw,
                attempt_id,
                accepted,
                derived_from,
            } => {
                facade.commit_processing_head(
                    &id,
                    revision,
                    &text,
                    is_raw,
                    &attempt_id,
                    accepted.as_ref(),
                    derived_from.as_deref(),
                )?;
                Value::Null
            }
            StoreRequest::RecordInsight {
                id,
                event_id,
                kind,
                occurred_at,
                payload_json,
            } => {
                facade.record_insight(&id, &event_id, &kind, &occurred_at, &payload_json)?;
                Value::Null
            }
            StoreRequest::RecordBoundaryRevision {
                id,
                source_text,
                text,
                changes,
            } => {
                facade.record_boundary_revision(&id, &source_text, &text, &changes)?;
                Value::Null
            }
            StoreRequest::RecordCorrection { record } => json(facade.record_correction(&record)?)?,
            StoreRequest::ReviseCorrection {
                id,
                request_id,
                decision,
                decision_utc,
                final_text,
            } => json(facade.revise_correction(
                &id,
                &request_id,
                decision,
                &decision_utc,
                &final_text,
            )?)?,
            StoreRequest::Delete { id } => {
                facade.delete(&id)?;
                self.changed();
                Value::Null
            }
            StoreRequest::SetArchival { id, archival } => {
                facade.set_archival(&id, archival)?;
                self.changed();
                Value::Null
            }
            StoreRequest::HoldAudio { id } => {
                let hold = facade.hold_audio(&id)?;
                let token = new_id("hold");
                let refused = {
                    let mut stashes = lock(&self.stashes);
                    if caller.gone() {
                        Some(hold)
                    } else {
                        stashes
                            .entry(caller.key())
                            .or_default()
                            .holds
                            .insert(token.clone(), hold);
                        None
                    }
                };
                // Released off the stash lock: a release takes the store's.
                drop(refused);
                Value::String(token)
            }
            StoreRequest::ReleaseHold { hold } => {
                let released = lock(&self.stashes)
                    .get_mut(&caller.key())
                    .and_then(|stash| stash.holds.remove(&hold));
                drop(released);
                Value::Null
            }
            StoreRequest::Upload {
                upload,
                offset,
                data,
            } => {
                let data = base64::engine::general_purpose::STANDARD
                    .decode(data)
                    .map_err(|err| StorageError::Invalid(format!("upload data is not base64: {err}")))?;
                self.append_upload(caller, upload, offset, &data)?;
                Value::Null
            }
            StoreRequest::Import { upload, transcribe } => {
                let wav = lock(&self.stashes)
                    .get_mut(&caller.key())
                    .and_then(|stash| stash.uploads.remove(&upload))
                    .ok_or_else(|| StorageError::Invalid("nothing was uploaded to import".to_string()))?;
                let id = facade.save_import(&wav, transcribe)?.id;
                self.changed();
                Value::String(id)
            }
            StoreRequest::Uploaded { upload } => {
                let json = lock(&self.stashes)
                    .get_mut(&caller.key())
                    .and_then(|stash| stash.uploads.remove(&upload))
                    .ok_or_else(|| StorageError::Invalid("nothing was uploaded to run".to_string()))?;
                let inner: StoreRequest = serde_json::from_slice(&json).map_err(|err| {
                    StorageError::Invalid(format!("the uploaded request is not one: {err}"))
                })?;
                if matches!(
                    inner,
                    StoreRequest::Uploaded { .. }
                        | StoreRequest::Upload { .. }
                        | StoreRequest::Fetch { .. }
                ) {
                    return Err(StorageError::Invalid(
                        "an uploaded request cannot itself move uploads or answers".to_string(),
                    ));
                }
                return self.run(caller, inner);
            }
            StoreRequest::Fetch { blob, offset } => return self.fetch(caller, &blob, offset),
            StoreRequest::Discard { id } => {
                if let Some(stash) = lock(&self.stashes).get_mut(&caller.key()) {
                    stash.blobs.retain(|(kept, _)| *kept != id);
                    stash.uploads.remove(&id);
                }
                Value::Null
            }
            StoreRequest::Upkeep { run } => {
                if run {
                    self.upkeep_now();
                }
                json(lock(&self.upkeep.last).clone())?
            }
        };
        self.answer(caller, value)
    }

    /// `value` whole, or kept to be fetched when it is too large.
    fn answer(&self, caller: &dyn Caller, value: Value) -> Result<StoreReply, StorageError> {
        let encoded = serde_json::to_vec(&value)
            .map_err(|err| StorageError::Invalid(format!("the answer does not serialize: {err}")))?;
        if encoded.len() <= self.inline {
            return Ok(StoreReply::Done { value });
        }
        drop(value);
        let bytes = encoded.len() as u64;
        let blob = self.keep(caller, Arc::new(encoded))?;
        Ok(StoreReply::Large { blob, bytes })
    }

    /// Keeps `bytes` for `caller` to fetch; its id. The answers a
    /// connection holds in memory stay under the memory bound, this one
    /// included: one that does not fit beside them is written to a spool
    /// file and dropped from memory.
    fn keep(&self, caller: &dyn Caller, bytes: Arc<Vec<u8>>) -> Result<String, StorageError> {
        let id = new_id("blob");
        {
            let mut stashes = lock(&self.stashes);
            if caller.gone() {
                return Ok(id);
            }
            let stash = stashes.entry(caller.key()).or_default();
            if stash.blob_bytes().saturating_add(bytes.len()) <= self.memory_limit {
                stash.push_blob(id.clone(), Kept::Memory(bytes));
                return Ok(id);
            }
        }
        // Off the stash lock: the write may take a while.
        let spooled = self.spool(&bytes)?;
        drop(bytes);
        let mut stashes = lock(&self.stashes);
        if !caller.gone() {
            stashes.entry(caller.key()).or_default().push_blob(id.clone(), spooled);
        }
        Ok(id)
    }

    /// `bytes` in a new unlinked spool file.
    fn spool(&self, bytes: &[u8]) -> Result<Kept, StorageError> {
        let spooled = (|| {
            std::fs::create_dir_all(&self.spool)?;
            let mut file = tempfile::tempfile_in(&self.spool)?;
            file.write_all(bytes)?;
            Ok::<_, std::io::Error>(file)
        })()
        .map_err(|err| {
            StorageError::Invalid(format!(
                "the answer ({} MB) is too large to keep in memory and could not be set aside \
                 on disk: {err}",
                bytes.len() / (1024 * 1024)
            ))
        })?;
        Ok(Kept::Spooled {
            file: Arc::new(Mutex::new(spooled)),
            len: bytes.len(),
        })
    }

    fn fetch(&self, caller: &dyn Caller, blob: &str, offset: u64) -> Result<StoreReply, StorageError> {
        let kept = {
            let mut stashes = lock(&self.stashes);
            let stash = stashes.get_mut(&caller.key());
            let Some((index, kept)) = stash.as_ref().and_then(|stash| {
                stash
                    .blobs
                    .iter()
                    .position(|(id, _)| id == blob)
                    .map(|index| (index, stash.blobs[index].1.clone()))
            }) else {
                return Err(StorageError::Invalid(
                    "that answer is no longer kept; ask again".to_string(),
                ));
            };
            let start = usize::try_from(offset).unwrap_or(usize::MAX).min(kept.len());
            if start + self.chunk >= kept.len() {
                // The last chunk: the answer is used up.
                if let Some(stash) = stash {
                    stash.blobs.remove(index);
                }
            }
            kept
        };
        let start = usize::try_from(offset).unwrap_or(usize::MAX).min(kept.len());
        let end = (start + self.chunk).min(kept.len());
        Ok(StoreReply::Chunk {
            data: base64::engine::general_purpose::STANDARD.encode(kept.read(start, end)?),
        })
    }

    fn append_upload(
        &self,
        caller: &dyn Caller,
        upload: String,
        offset: u64,
        data: &[u8],
    ) -> Result<(), StorageError> {
        let mut stashes = lock(&self.stashes);
        if caller.gone() {
            return Ok(());
        }
        let stash = stashes.entry(caller.key()).or_default();
        let held: usize = stash.uploads.values().map(Vec::len).sum();
        if held + data.len() > MAX_UPLOAD_BYTES {
            stash.uploads.remove(&upload);
            return Err(StorageError::Invalid(
                "the recording is too large to import".to_string(),
            ));
        }
        if offset == 0 && !stash.uploads.contains_key(&upload) {
            if stash.uploads.len() >= MAX_UPLOADS {
                return Err(StorageError::Invalid(
                    "too many imports at once; try again when one finished".to_string(),
                ));
            }
            stash.uploads.insert(upload.clone(), Vec::new());
        }
        match stash.uploads.get_mut(&upload) {
            Some(buffer) if buffer.len() as u64 == offset => {
                buffer.extend_from_slice(data);
                Ok(())
            }
            _ => {
                stash.uploads.remove(&upload);
                Err(StorageError::Invalid(
                    "the import's upload arrived out of order".to_string(),
                ))
            }
        }
    }

    /// Releases `caller`'s hold `hold` without waiting for the store: what
    /// a connection reader does when the workers cannot take the release.
    pub(crate) fn release_later(&self, caller: &dyn Caller, hold: &str) {
        let released = lock(&self.stashes)
            .get_mut(&caller.key())
            .and_then(|stash| stash.holds.remove(hold));
        if let Some(released) = released {
            match std::thread::Builder::new()
                .name("starling-host-release".to_string())
                .spawn(move || drop(released))
            {
                Ok(thread) => {
                    let mut releases = lock(&self.releases);
                    releases.retain(|thread| !thread.is_finished());
                    releases.push(thread);
                }
                Err(err) => {
                    eprintln!("starling-runtime-host: releasing an audio hold inline: {err}");
                }
            }
        }
    }

    /// Waits for the releases [`Self::release_later`] started: the host
    /// joins them with its other store threads, so none touches a store
    /// a successor owns.
    pub(crate) fn join_releases(&self) {
        let releases = std::mem::take(&mut *lock(&self.releases));
        for thread in releases {
            let _ = thread.join();
        }
    }

    /// Drops what `caller` was handed: kept answers, uploads, and its
    /// audio holds (released).
    pub(crate) fn caller_gone(&self, caller: &dyn Caller) {
        let stash = lock(&self.stashes).remove(&caller.key());
        // Holds release off the stash lock: a release takes the store's.
        drop(stash);
    }

    /// Asks for an upkeep pass now (after the running one).
    pub fn upkeep_now(&self) {
        *lock(&self.upkeep.wanted) = true;
        self.upkeep.wake.notify_all();
    }

    /// The latest upkeep report.
    pub fn last_upkeep(&self) -> Option<String> {
        lock(&self.upkeep.last).clone()
    }

    /// History audio upkeep (#342) on the host's schedule: a pass
    /// [`UPKEEP_FIRST`] after start, then every `interval` and whenever
    /// one is asked for ([`Self::upkeep_now`]: the app saved new storage
    /// settings). `policy` is the user's retention policy as the settings
    /// say now — read again before every removal, so a limit lifted
    /// mid-pass is not applied. `paused` says a take records: no pass
    /// starts then, a running one stops at its next step, and the pass is
    /// tried again soon. Each pass's report goes to `report` (with
    /// whether audio was retired, which changes the history list). Runs
    /// until `stopping`.
    pub fn upkeep_loop(
        &self,
        first: Duration,
        interval: Duration,
        policy: impl Fn() -> RetentionPolicy,
        paused: impl Fn() -> bool,
        report: impl Fn(&str, bool),
        stopping: impl Fn() -> bool,
    ) {
        let mut next = Instant::now() + first;
        loop {
            {
                let mut wanted = lock(&self.upkeep.wanted);
                while !*wanted && Instant::now() < next {
                    if stopping() {
                        return;
                    }
                    let wait = next
                        .saturating_duration_since(Instant::now())
                        .min(Duration::from_millis(250));
                    wanted = self
                        .upkeep
                        .wake
                        .wait_timeout(wanted, wait)
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .0;
                }
                *wanted = false;
            }
            if stopping() {
                return;
            }
            if paused() {
                next = Instant::now() + UPKEEP_PAUSED_RETRY;
                continue;
            }
            let outcome = self.facade.audio_upkeep(&policy, || paused() || stopping());
            let (summary, retired, stopped) = match &outcome {
                Ok(pass) => {
                    for (id, reason) in &pass.failures {
                        eprintln!("starling-runtime-host: could not compress the audio of {id}: {reason}");
                    }
                    for file in &pass.sweep.swept {
                        eprintln!(
                            "starling-runtime-host: swept {} {} ({} bytes)",
                            file.kind, file.id, file.bytes
                        );
                    }
                    for (name, reason) in &pass.sweep.retained {
                        eprintln!("starling-runtime-host: kept {name} at the retention sweep: {reason}");
                    }
                    (pass.summary(), !pass.retention.retired.is_empty(), pass.paused)
                }
                Err(err) => (Some(format!("History audio upkeep failed: {err}")), false, false),
            };
            if let Some(summary) = summary {
                *lock(&self.upkeep.last) = Some(summary.clone());
                report(&summary, retired);
            }
            next = Instant::now() + if stopped { UPKEEP_PAUSED_RETRY } else { interval };
        }
    }
}

/// One request a connection reader hands the workers.
pub(crate) struct StoreJob {
    pub conn: Arc<ConnState>,
    pub req: String,
    pub request: StoreRequest,
}

/// The queue connection readers hand store requests to
/// ([`start_workers`] answers them).
pub(crate) fn store_queue() -> (SyncSender<StoreJob>, Receiver<StoreJob>) {
    std::sync::mpsc::sync_channel(QUEUE)
}

/// Starts the workers that answer store requests off the connection
/// readers (a long history or a take's encoding never stalls a
/// connection's other frames). A full queue answers at once that the
/// host is busy. The workers end once `stopping` says so.
pub(crate) fn start_workers(
    history: Arc<History>,
    jobs: Receiver<StoreJob>,
    stopping: impl Fn() -> bool + Clone + Send + 'static,
) -> Vec<JoinHandle<()>> {
    let rx = Arc::new(Mutex::new(jobs));
    let mut threads = Vec::new();
    for index in 0..WORKERS {
        let history = Arc::clone(&history);
        let rx = Arc::clone(&rx);
        let stopping = stopping.clone();
        let spawned = std::thread::Builder::new()
            .name(format!("starling-host-store-{index}"))
            .spawn(move || work(&history, &rx, &stopping));
        match spawned {
            Ok(thread) => threads.push(thread),
            Err(err) => eprintln!("starling-runtime-host: cannot start a store worker: {err}"),
        }
    }
    threads
}

fn work(history: &History, jobs: &Mutex<Receiver<StoreJob>>, stopping: &dyn Fn() -> bool) {
    loop {
        let next = lock(jobs).recv_timeout(Duration::from_millis(250));
        match next {
            Ok(job) => {
                if job.conn.gone() {
                    continue;
                }
                let reply = history.handle(&*job.conn, job.request);
                if job
                    .conn
                    .try_deliver(Frame::Stored { req: job.req, reply })
                    .is_err()
                {
                    // A reader that stopped reading: the same posture as
                    // an undeliverable receipt.
                    job.conn.close();
                }
            }
            Err(RecvTimeoutError::Timeout) => {
                if stopping() {
                    return;
                }
            }
            Err(RecvTimeoutError::Disconnected) => return,
        }
    }
}

/// Hands `job` to the workers; `false` when they cannot take it now (the
/// caller answers that the host is busy).
pub(crate) fn submit(jobs: &SyncSender<StoreJob>, job: StoreJob) -> Result<(), StoreJob> {
    match jobs.try_send(job) {
        Ok(()) => Ok(()),
        Err(TrySendError::Full(job)) | Err(TrySendError::Disconnected(job)) => Err(job),
    }
}

/// The store service run in-process, as one caller: the same requests and
/// answers as over the host's socket, without the socket.
pub struct LocalHistory {
    history: History,
}

impl LocalHistory {
    /// The service over the store at `root`.
    pub fn open(root: &std::path::Path) -> Result<LocalHistory, StorageError> {
        let facade = Facade::open(root)
            .map_err(|err| StorageError::Invalid(format!("the store would not open: {err}")))?;
        Ok(LocalHistory {
            history: History::new(facade, crate::frame::DEFAULT_MAX_FRAME_BYTES),
        })
    }

    pub fn history(&self) -> &History {
        &self.history
    }
}

struct Local;

impl Caller for Local {
    fn key(&self) -> usize {
        0
    }

    fn gone(&self) -> bool {
        false
    }
}

impl StoreCall for LocalHistory {
    fn call(&self, request: StoreRequest) -> Result<StoreReply, StorageError> {
        // Through the wire form, as the socket would carry it.
        let request: StoreRequest = serde_json::from_value(json(request)?)
            .map_err(|err| StorageError::Invalid(err.to_string()))?;
        let reply = self.history.handle(&Local, request);
        serde_json::from_value(json(reply)?).map_err(|err| StorageError::Invalid(err.to_string()))
    }

    fn chunk_bytes(&self) -> usize {
        self.history.chunk
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::history::{HistoryClient, StoreFailureKind};

    fn wav(samples: usize) -> Vec<u8> {
        starling_dictation::audio::encode_wav_16k(&starling_dictation::audio::PcmAudio {
            samples: (0..samples).map(|i| (i % 50) as f32 * 0.01).collect(),
            sample_rate: 16_000,
            channels: 1,
        })
        .expect("wav")
    }

    /// A hold the workers could not take is released anyway, off the
    /// caller's thread: the take compresses again.
    #[test]
    fn a_hold_released_later_lets_upkeep_compress_the_take() {
        let root = tempfile::tempdir().unwrap();
        let local = LocalHistory::open(root.path()).unwrap();
        let history = HistoryClient(&local);
        let id = history.import(&wav(32_000), false).unwrap();
        let hold = history.hold_audio(&id).unwrap();
        let pass = || {
            local
                .history()
                .facade()
                .audio_upkeep(RetentionPolicy::default, || false)
                .unwrap()
                .compressed
        };
        assert_eq!(pass(), 0, "held");
        local.history().release_later(&Local, &hold);
        let deadline = Instant::now() + Duration::from_secs(5);
        while pass() == 0 {
            assert!(Instant::now() < deadline, "the hold was never released");
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    /// #220: the answers a connection holds in memory never pass the
    /// bound, the newest included: an answer that does not fit beside the
    /// others — or alone — waits in a spool file and is read back whole.
    #[test]
    fn answers_past_the_memory_bound_wait_on_disk() {
        let root = tempfile::tempdir().unwrap();
        let mut local = LocalHistory::open(root.path()).unwrap();
        let first = HistoryClient(&local).import(&wav(48_000), false).unwrap();
        let second = HistoryClient(&local).import(&wav(16_000), false).unwrap();
        let wav_of = |id: &str| HistoryClient(&local).audio(id, AudioFormat::Wav).unwrap().unwrap();
        let (first_wav, second_wav) = (wav_of(&first), wav_of(&second));
        // Room for the second take's answer, not the first's.
        local.history.memory_limit = second_wav.len() + 1024;
        let history = &local.history;
        let ask = |id: &str| match history.handle(
            &Local,
            StoreRequest::Audio {
                id: id.to_string(),
                format: AudioFormat::Wav,
            },
        ) {
            StoreReply::Bytes { blob, .. } => blob,
            other => panic!("{other:?}"),
        };
        let kept = |blob: &str| {
            lock(&history.stashes)[&0]
                .blobs
                .iter()
                .find(|(id, _)| id == blob)
                .map(|(_, kept)| matches!(kept, Kept::Spooled { .. }))
                .expect("kept")
        };
        let in_memory = || lock(&history.stashes)[&0].blob_bytes();

        let alone_too_large = ask(&first);
        assert!(kept(&alone_too_large), "larger than the bound alone: spooled");
        assert_eq!(in_memory(), 0);
        let fits = ask(&second);
        assert!(!kept(&fits), "fits: in memory");
        let beside = ask(&second);
        assert!(kept(&beside), "does not fit beside the other: spooled");
        assert!(in_memory() <= history.memory_limit);
        #[cfg(unix)]
        assert_eq!(std::fs::read_dir(root.path().join(SPOOL_DIR)).unwrap().count(), 0, "unlinked");

        let fetch = |blob: &str, len: usize| {
            let mut data = Vec::new();
            while data.len() < len {
                match history.handle(
                    &Local,
                    StoreRequest::Fetch {
                        blob: blob.to_string(),
                        offset: data.len() as u64,
                    },
                ) {
                    StoreReply::Chunk { data: part } => data.extend(
                        base64::engine::general_purpose::STANDARD.decode(part).unwrap(),
                    ),
                    other => panic!("{other:?}"),
                }
            }
            data
        };
        assert_eq!(fetch(&alone_too_large, first_wav.len()), first_wav);
        assert_eq!(fetch(&beside, second_wav.len()), second_wav);
        assert_eq!(fetch(&fits, second_wav.len()), second_wav);
        assert!(lock(&history.stashes)[&0].blobs.is_empty(), "each was used up");
    }

    /// A store connection that refuses whatever consumes an upload, as a
    /// host with a full queue does.
    struct Refusing<'a>(&'a LocalHistory);

    impl StoreCall for Refusing<'_> {
        fn call(&self, request: StoreRequest) -> Result<StoreReply, StorageError> {
            match request {
                StoreRequest::Import { .. } | StoreRequest::Uploaded { .. } => {
                    Ok(StoreReply::Failed {
                        failure: StoreFailure {
                            kind: StoreFailureKind::Io,
                            message: "busy".to_string(),
                        },
                    })
                }
                other => self.0.call(other),
            }
        }

        fn chunk_bytes(&self) -> usize {
            4096
        }
    }

    /// An upload whose import or request was refused does not keep one of
    /// the connection's upload slots.
    #[test]
    fn a_refused_consumer_drops_its_upload() {
        let root = tempfile::tempdir().unwrap();
        let local = LocalHistory::open(root.path()).unwrap();
        let refusing = HistoryClient(Refusing(&local));
        for _ in 0..MAX_UPLOADS + 2 {
            assert!(refusing.import(&wav(8_000), false).is_err());
            assert!(refusing.record_insight("c_x", "e", "k", "t", &"x".repeat(8192)).is_err());
        }
        assert!(local.history().stashes.lock().unwrap().values().all(|stash| stash.uploads.is_empty()));
        HistoryClient(&local).import(&wav(8_000), false).expect("slots are free");
    }
}
