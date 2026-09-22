//! The capture machine (§2.1): the actor that owns a take end to end —
//! device acquisition, recording, the stop handshake, and the salvage
//! paths — behind the I0 command/event set.
//!
//! Production wiring drives the **existing hardened recorder**
//! (`starling-dictation` `recorder.rs`, ring + journal, I1) through
//! [`DeviceCaptureSource`]: `capture.start` maps to
//! `start_recording_with_journal`, `capture.stop`/`capture.abort` to the
//! `stop()` handshake. Gaps surface as `capture.gap`; a quiesce-timeout
//! stop degrades to `capture.error{quiesce_timeout}` with the salvaged
//! samples preserved as an interrupted take — the landed `upload.rs`
//! semantics (audio kept, take marked interrupted), here as I0 events.
//!
//! The recorder is a concrete type over a real CPAL device, so the actor
//! talks to it through the [`CaptureSource`] seam;
//! [`FakeCaptureSource`] (test twin, `testing` module) scripts devices,
//! gaps, faults and quiesce timeouts without hardware.
//!
//! Two recovery rules complete the picture: a failed device open is fatal
//! to the take *attempt*, not the machine — after `capture.error{
//! device_open_failed}` the actor takes the runtime-internal settle edge
//! `Interrupted → Idle` so the next `capture.start` can retry (issue
//! #211); and a stop handshake that returns `Err` never drops the take —
//! the quiesce timeout's preserved samples, or a metadata-only record
//! when a device-side error salvages nothing in-process, are still
//! registered as interrupted with their gap evidence (issue #212). Every
//! outcome that ends a take also releases its audio-route freeze (the
//! context cycle must never stay wedged behind a finished take), and a
//! persist failure on a salvage path surfaces as a non-fatal
//! `capture.error{persist_interrupted_failed}` while the machine is
//! still in a state that admits the event.
//!
//! **The persist runs off the actor loop** (issue #249, the capture-side
//! twin of #216's jobs fix). A store commit does whole-audio work — the
//! v2 store's adoption verifies and moves the journal and its samples
//! path writes the take's every sample through a fsync'd staging journal,
//! tens of MB of CPU and I/O for a multi-minute take — and running that
//! inline on the stop handshake stalled every `capture.*` command behind
//! it. The actor therefore hands each take's persist to a dedicated
//! worker thread
//! (one per take, like the jobs scheduler's encode workers) and defers
//! only the emissions that the durable commit gates: `capture.stopped`
//! still follows the successful commit (§4, the watermark-agreement ack
//! of #204 — the recorder-side handshake itself never leaves the actor),
//! a failed commit still degrades to `storage_commit_failed` before the
//! take's close-out, and the registry still receives the take only once
//! its store outcome is known. While the worker runs, the machine
//! honestly sits in the state the stop path left it in (`Draining` for a
//! stop, `Recording` for a fatal mid-take salvage, `Idle` for an abort),
//! answering commands the whole time. Reports are scoped to their take
//! (an epoch set at `capture.start`): a report landing after a newer take
//! began is stale and lands registry-only — never against the new take's
//! machine state, where a stale `capture.stopped` would swallow the new
//! take's own announcement and a stale fatal `capture.error` would kill
//! it. The one new interleaving the table always allowed but the inline
//! stall hid — `capture.abort` landing during a stop's persist — takes
//! the machine to `Idle` and releases the route at the abort decision
//! point (the persist must not extend the freeze); the persist then lands
//! silently (registry-only, no wire event; no `capture.*` event is legal
//! from `Idle`). Shutdown drains in-flight persists on its way out, but
//! bounded: a worker wedged in a hung store write is abandoned after
//! [`CaptureConfig::persist_drain_timeout`] with its takes named on
//! stderr — their durable journals survive for startup recovery, and an
//! unbounded shutdown hang is a worse failure than an abandoned store
//! row.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use starling_dictation::audio::encode_wav_16k_parts;
use starling_dictation::recorder::{
    CaptureGap, CapturedTake, JournalReport, RecorderError, RecorderFault, RecorderHandle,
};
use starling_dictation::store_v2::{CaptureStatus, CommitMark, StoreV2, TakeMeta};

use crate::bus::EventBus;
use crate::machine::{Inbound, MachineCore, Receipt, Rejection};
use crate::protocol::tables::CAPTURE;
use crate::protocol::{Command, Event, SampleGap};

use super::context::RouteFreezer;

/// How a take ended — the persistence-facing counterpart of the machine's
/// `Persisted` state (a `Persisted` take may reference a `complete` or an
/// `interrupted` storage row; an interrupted row is still a committed row).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TakeStatus {
    Complete,
    Interrupted,
}

/// Everything the runtime knows about a finished (or salvaged) take. The
/// jobs executor's `Loading` step pulls audio from here by `captureRef`.
#[derive(Debug, Clone)]
pub struct TakeRecord {
    /// The take's correlation id (`capture.start`'s `corr`).
    pub id: String,
    pub device: String,
    pub policy: String,
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub gaps: Vec<SampleGap>,
    pub acknowledged_samples: u64,
    pub final_sample_index: u64,
    pub journal: Option<JournalReport>,
    pub status: TakeStatus,
    pub sample_duration_ms: f64,
    pub wall_clock_ms: f64,
    /// The storage-facing id (the journal's id when the take journaled).
    pub capture_id: String,
}

impl TakeRecord {
    /// The FNV-1a 64 content hash of the take's sample payloads — the same
    /// value `journal.rs` seals into a journal trailer (hash over the frame
    /// payload bytes, i.e. the samples' little-endian f32 bits in order).
    /// With a healthy journal and no incremental drains this equals the
    /// journal's sealed hash.
    pub fn journal_hash(&self) -> String {
        const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
        const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
        let mut hash = FNV_OFFSET;
        for &sample in &self.samples {
            for byte in sample.to_bits().to_le_bytes() {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(FNV_PRIME);
            }
        }
        format!("{hash:016x}")
    }

    /// The upload WAV for transcription jobs (16 kHz mono), encoded
    /// straight off the borrowed samples (a multi-minute take must not be
    /// cloned just to build the encoder's owned-struct parameter; issue
    /// #216).
    pub fn to_wav(&self) -> Result<Vec<u8>, String> {
        encode_wav_16k_parts(&self.samples, self.sample_rate, 1).map_err(|err| err.to_string())
    }
}

/// Runtime-owned registry of finished takes (write: capture actor; read:
/// jobs executor `Loading`, docs turn appends).
pub type TakeRegistry = Arc<Mutex<HashMap<String, Arc<TakeRecord>>>>;

// ---------------------------------------------------------------------------
// The recorder seam
// ---------------------------------------------------------------------------

/// A live capture session — the recorder-facing slice of `RecorderHandle`
/// the actor needs. `stop` consumes the session exactly like
/// `RecorderHandle::stop`.
///
/// `Send` so the actor type can cross its spawn boundary; see the
/// `SAFETY` note on `RecorderSession` for why that is sound despite the
/// platform `cpal::Stream` not being `Send` on every host.
pub trait CaptureSession: Send {
    fn sample_rate(&self) -> u32;
    /// Total samples the audio callback has pushed (the sequence frontier;
    /// the `finalSampleIndex` the stop handshake declares).
    fn captured_sample_count(&self) -> u64;
    /// Samples covered by the last fsynced journal boundary — the honest
    /// "survives a process kill right now" count.
    fn acknowledged_samples(&self) -> u64;
    /// Fraction of raw (pre-attenuation) samples at full scale.
    fn source_clip_ratio(&self) -> f64;
    /// Gap spans surfaced so far, in capture order.
    fn gaps(&self) -> Vec<CaptureGap>;
    /// The first capture-path fault, typed by origin (the recorder's
    /// `capture_fault`): the device-side error wins over the journal
    /// fault. Fatality is decided from the variant — never from the
    /// message text, which carries no reliable marker (issue #216).
    fn capture_fault(&self) -> Option<RecorderFault>;
    /// A recent window for level metering.
    fn latest_window(&self, n: usize) -> Vec<f32>;
    /// The §3 R09 stop handshake.
    fn stop(self: Box<Self>) -> Result<CapturedTake, RecorderError>;
}

/// How the actor acquires devices. Production wraps
/// `recorder::start_recording_with_journal`.
pub trait CaptureSource: Send + Sync {
    fn start(&self, journals_dir: &Path, policy: &str) -> Result<Box<dyn CaptureSession>, String>;
}

struct RecorderSession {
    handle: RecorderHandle,
}

// SAFETY: a recorder session is created, used and consumed exclusively on
// the capture actor's thread — `start`, every poll and `stop` all run
// there, and no session value is ever moved across threads. The `Send`
// impl exists so the actor *type* (whose session slot is empty at the
// spawn boundary, before any take starts) can cross into the one thread
// that will own it. The platform `cpal::Stream` inside the handle is not
// `Send` on every host, which is precisely why the ownership protocol
// above is single-threaded.
unsafe impl Send for RecorderSession {}

impl CaptureSession for RecorderSession {
    fn sample_rate(&self) -> u32 {
        self.handle.sample_rate()
    }
    fn captured_sample_count(&self) -> u64 {
        self.handle.captured_sample_count()
    }
    fn acknowledged_samples(&self) -> u64 {
        self.handle.acknowledged_samples()
    }
    fn source_clip_ratio(&self) -> f64 {
        self.handle.source_clip_ratio()
    }
    fn gaps(&self) -> Vec<CaptureGap> {
        self.handle.gaps()
    }
    fn capture_fault(&self) -> Option<RecorderFault> {
        self.handle.capture_fault()
    }
    fn latest_window(&self, n: usize) -> Vec<f32> {
        self.handle.latest_window(n)
    }
    fn stop(self: Box<Self>) -> Result<CapturedTake, RecorderError> {
        self.handle.stop()
    }
}

/// The production source: the hardened recorder with its per-take durable
/// journal (I1). `capture.start{policy}` maps here; the recorder does not
/// surface the picked device's name, so `capture.started` reports the
/// `"default-input"` label (documented interpretation).
pub struct DeviceCaptureSource;

impl CaptureSource for DeviceCaptureSource {
    fn start(&self, journals_dir: &Path, _policy: &str) -> Result<Box<dyn CaptureSession>, String> {
        starling_dictation::recorder::start_recording_with_journal(journals_dir)
            .map(|handle| Box::new(RecorderSession { handle }) as Box<dyn CaptureSession>)
            .map_err(|err| err.to_string())
    }
}

/// Classifies a surfaced capture fault: journal/storage faults are
/// non-fatal (the recorder keeps capturing in memory and honestly freezes
/// acknowledgment), device-side errors are fatal (→ `Interrupted`).
///
/// The decision comes from [`RecorderFault`]'s variant — the fault's
/// *origin* — not from its message text. The old substring test (any
/// message containing "journal" counts as a journal fault) misclassified
/// device errors whose text happened to mention the journal, letting a
/// take continue from a dead device until the stop handshake failed into
/// the audio-dropping path (issue #216).
fn error_is_fatal(fault: &RecorderFault) -> bool {
    fault.is_fatal()
}

// ---------------------------------------------------------------------------
// Persistence seam
// ---------------------------------------------------------------------------

/// Where finished/salvaged takes are persisted. The `capture.stopped`
/// event is emitted only after [`CaptureStore::commit_take`] returns `Ok`
/// (§4: the durable ack follows the metadata commit).
///
/// Implementations run on the **persist worker**, not the capture actor
/// loop (issue #249): the actor hands each take off to a dedicated thread
/// and resumes the take's close-out when the commit's result comes back,
/// so whole-audio work inside an implementation (the v2 store's journal
/// verification/move on adoption, or its staging-journal write of every
/// sample on the samples path) cannot stall `capture.*` commands.
pub trait CaptureStore: Send + Sync {
    /// Persists a cleanly stopped take.
    fn commit_take(&self, take: &TakeRecord) -> Result<(), String>;
    /// Persists a salvaged take as interrupted-but-usable (quiesce
    /// timeout, device fault, abort) with a note stating exactly what was
    /// kept — `upload.rs`'s `save_interrupted_take` semantics.
    fn mark_interrupted(&self, take: &TakeRecord, note: &str) -> Result<(), String>;
    /// A label for snapshots and diagnostics.
    fn describe(&self) -> String;
}

/// In-memory store (tests, and runtimes started without a data root).
#[derive(Default)]
pub struct InMemoryCaptureStore {
    pub takes: Mutex<Vec<(TakeStatus, String, String)>>,
}

impl InMemoryCaptureStore {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }
}

impl CaptureStore for InMemoryCaptureStore {
    fn commit_take(&self, take: &TakeRecord) -> Result<(), String> {
        self.takes
            .lock()
            .expect("capture store lock")
            .push((TakeStatus::Complete, take.id.clone(), take.capture_id.clone()));
        Ok(())
    }
    fn mark_interrupted(&self, take: &TakeRecord, note: &str) -> Result<(), String> {
        self.takes.lock().expect("capture store lock").push((
            TakeStatus::Interrupted,
            take.id.clone(),
            format!("{}: {}", take.capture_id, note),
        ));
        Ok(())
    }
    fn describe(&self) -> String {
        "in-memory".to_string()
    }
}

/// The storage v2 store (D14: THE store — there is no v1 seam anymore).
/// A take with journal evidence is **adopted**: the recorder's finalized
/// `.sj` journal is read, verified, sealed to its last valid boundary when
/// torn, moved into `<root>/audio/`, and the `captures` row committed in
/// one SQLite transaction — the §4 metadata step, on the store's own
/// protocol rather than a hand-built row. A take without usable journal
/// evidence (no journal, a writer that faulted before its first boundary,
/// an unreadable file) is written from its samples through the same crash
/// protocol the app's WAV path uses (staging journal → finalize → promote
/// → commit), so a journal-level problem can never cost the audio.
///
/// **Row keying contract (the samples path).** The samples-fallback row
/// is keyed by the staging journal's minted id (`c_<uuid>`), NOT by
/// `take.capture_id` — deliberately, not as an oversight. The capture id
/// equals the journal's id on the journaled paths, and the one shape that
/// reaches the samples path *with* journal evidence is an adoption that
/// failed leaving nothing behind; on the occupied-id shape (a row
/// already holds that id) keying the fallback row by the same id would
/// collide with the existing row and lose the audio. So the take's real
/// id rides in `extra_json["captureId"]` (beside `takeCorr`) for any
/// consumer that needs to find the row from the machine's ids, and a
/// re-commit of the same take stores a second row instead of colliding
/// idempotently the way the adoption path does — preventing retries is
/// the capture actor's job (each take is handed to exactly one persist
/// worker, issue #249).
///
/// An adoption that fails *after* its durable effects is never silently
/// re-stored from samples: if the row landed anyway (a retry against an
/// already-adopted take, or a destination that already holds the
/// capture) the commit is satisfied by that row — no duplicate. Only a
/// failure that left nothing behind falls back to the samples path, and
/// the reason rides along: recorded in the stored row's `extra_json`
/// (`journalAdoptionError`) and chained into the error should the samples
/// write fail too. A failed post-adoption status flip is non-fatal for
/// the same reason — the take is durably persisted; the caller must not
/// read "not persisted" and retry into the adoption's row.
///
/// The evidence the runtime layers on top — `takeCorr`, `captureId`,
/// `gaps`, the sample / wall-clock split — rides in `extra_json` on the
/// samples path, plus `journalHash`: the take's own FNV-1a content hash,
/// the forensic bridge to journal evidence the store could not adopt.
/// Do not join it against the row's `journal_hash` column by name: the
/// column holds the staged journal's sealed hash — the same function
/// over the same bytes on this path, but a different claim in general
/// (on torn or drained evidence the take's hash and the journal's sealed
/// hash diverge). On the adoption path the journal itself is the durable
/// evidence and the row carries the store's own adoption semantics
/// (plus the salvage note on the interrupted paths). Nothing reads those
/// keys back today; the registry is the runtime's session-scoped source
/// of take detail.
pub struct V2CaptureStore {
    store: Mutex<StoreV2>,
}

/// Report a post-commit (or rollback) divergence that must not change
/// the answer the caller gets: a durably committed take whose follow-up
/// failed, or a staging rollback that leaked, is diagnosable state
/// divergence — surfacing it as `Err` would invite a retry that
/// collides with the committed row, or mask the write's own root cause.
/// One channel for all of them — stderr, the same channel the capture
/// actor's drain notices use — because this crate has no logging
/// facade; the day the workspace adopts one, this is the single site to
/// swap.
fn report_divergence(message: String) {
    eprintln!("v2 capture store: {message}");
}

/// The post-commit divergence for a committed row whose interrupted
/// status flip failed — the row reads complete while the take was
/// salvaged. Non-fatal on purpose (see [`report_divergence`]); the
/// adoption carried the salvage note, so the row keeps its wording.
fn report_status_flip_failure(id: &str, err: impl std::fmt::Display) {
    report_divergence(format!(
        "adopted take {id} committed; the interrupted status flip failed ({err}) — \
         the row keeps the salvage note"
    ));
}

impl V2CaptureStore {
    pub fn open(root: impl Into<PathBuf>) -> Result<Self, String> {
        let store = StoreV2::open(root).map_err(|err| err.to_string())?;
        Ok(V2CaptureStore {
            store: Mutex::new(store),
        })
    }

    fn commit(
        &self,
        take: &TakeRecord,
        status: CaptureStatus,
        note: Option<&str>,
    ) -> Result<(), String> {
        // Journal evidence first, when it exists on disk. The lock is held
        // only for the adoption (one SQLite transaction); the samples path
        // below releases it for the staging-journal writes — tens of MB of
        // fsync'd I/O for a long take — and retakes it for the commit, the
        // same split the app facade's save path uses, so concurrent
        // persists do not serialize behind each other's writes.
        let mut adoption_error = None;
        {
            let mut store = self.store.lock().expect("v2 store lock");
            let adoption = match &take.journal {
                Some(report) if report.path.exists() => {
                    Some((report, store.adopt_journal(&report.path, note)))
                }
                _ => None,
            };
            match adoption {
                Some((_, Ok(record))) => {
                    // The salvage paths force interrupted-ness regardless of
                    // the journal's own verdict (the app facade's R34 rule):
                    // the interruption derives from how the take ended, not
                    // from the journal's finalized-ness. The note itself
                    // already rode along with the adoption; passing no note
                    // keeps its combined wording intact. A flip failure is
                    // non-fatal via [`report_status_flip_failure`].
                    if status == CaptureStatus::Interrupted {
                        if let Err(err) = store.update_capture_status(
                            &record.id,
                            CaptureStatus::Interrupted,
                            None,
                        ) {
                            report_status_flip_failure(&record.id, err);
                        }
                    }
                    return Ok(());
                }
                Some((report, Err(err))) => {
                    // The adoption may have failed *after* its durable
                    // effects: a row for the journal's id means the take is
                    // already stored (a retry, or a partial prior adoption)
                    // and must not be stored a second time from samples —
                    // but only if the row holds THIS journal's audio. The
                    // id is the file stem and proves nothing about content:
                    // a stale, renamed or reused journal (or an id
                    // collision) must not silently satisfy the commit while
                    // this take's audio is discarded, so the row's stored
                    // hash is checked against the journal's verified
                    // payload hash before the retry counts as satisfied.
                    match store.get_capture(&report.id) {
                        Ok(Some(existing)) => {
                            let same_audio = starling_dictation::journal::verified_journal_hash(
                                &report.path,
                            )
                            .map(|hash| hash == existing.journal_hash)
                            .unwrap_or(false);

                            if !same_audio {
                                report_divergence(format!(
                                    "journal id {} is occupied by a different capture (stored \
                                     journal hash {} does not match this journal) — storing \
                                     this take from samples",
                                    report.id, existing.journal_hash
                                ));
                                adoption_error = Some(format!(
                                    "journal id {} already holds different audio \
                                     (journal hash mismatch)",
                                    report.id
                                ));
                            } else {
                                if status == CaptureStatus::Interrupted {
                                    if let Err(flip_err) = store.update_capture_status(
                                        &existing.id,
                                        CaptureStatus::Interrupted,
                                        None,
                                    ) {
                                        // The row being flipped is the
                                        // pre-existing occupant of the
                                        // journal id, not a take this call
                                        // adopted — say so.
                                        report_divergence(format!(
                                            "existing capture {} committed; the interrupted \
                                             status flip failed ({flip_err})",
                                            existing.id
                                        ));
                                    }
                                }
                                return Ok(());
                            }
                        }
                        _ => {
                            // Nothing landed: a journal-readability failure
                            // (unreadable source, no verified samples) is
                            // the legitimate fallback; anything else
                            // (destination conflict without a row, SQLite
                            // trouble) falls back too — the audio must not
                            // be lost — but the reason is kept and recorded.
                            adoption_error = Some(err.to_string());
                        }
                    }
                }
                None => {}
            }
        }

        // No journal evidence (or an adoption that left nothing behind):
        // the take's samples through the §4 protocol.
        let mut meta = TakeMeta::for_device(take.device.clone());
        meta.policy = take.policy.clone();
        let mut extra = serde_json::json!({
            "takeCorr": take.id,
            "captureId": take.capture_id,
            "gaps": take.gaps,
            "acknowledgedSamples": take.acknowledged_samples,
            "journalFinalized": take.journal.as_ref().map(|r| r.finalized),
            "journalFault": take.journal.as_ref().and_then(|r| r.fault.clone()),
            // The take's own content hash — the forensic bridge to journal
            // evidence even when the journal could not be adopted (the
            // row's journal_hash column is the staged journal's hash).
            "journalHash": take.journal_hash(),
            "wallClockMs": take.wall_clock_ms,
        });
        if let Some(reason) = &adoption_error {
            // The fallback's provenance stays diagnosable in the stored
            // row instead of dying with the process.
            extra["journalAdoptionError"] = serde_json::Value::String(reason.clone());
        }
        meta.extra_json = Some(extra.to_string());

        // Cheap step under the guard: mint the staging journal. The bulk
        // writes and fsyncs run on the take's own writer, off the lock.
        let mut v2_take = {
            let store = self.store.lock().expect("v2 store lock");
            store
                .begin_take_at_rate(take.sample_rate, meta)
                .map_err(|err| chain_adoption_failure(&adoption_error, err.to_string()))?
        };
        let staged_id = v2_take.id().to_string();
        // The staging journal of a write that failed is not evidence to
        // salvage — it is a partial (or, past finalize, complete) duplicate
        // of whatever a retry stores instead — so every failure arm below
        // rolls it back explicitly. The write's own error is always the one
        // returned (the root cause); a rollback that fails too is reported
        // beside it — never swallowed, never allowed to replace the root
        // cause.
        if let Err(err) = v2_take.append_and_seal(&take.samples) {
            if let Err(discard_err) = self
                .store
                .lock()
                .expect("v2 store lock")
                .discard_staging(&staged_id)
            {
                report_divergence(format!(
                    "staging journal {staged_id} leaked after the append failure ({discard_err})"
                ));
            }
            return Err(chain_adoption_failure(&adoption_error, err.to_string()));
        }
        let finalized = match v2_take.finalize() {
            Ok(finalized) => finalized,
            Err(err) => {
                if let Err(discard_err) = self
                    .store
                    .lock()
                    .expect("v2 store lock")
                    .discard_staging(&staged_id)
                {
                    report_divergence(format!(
                        "staging journal {staged_id} leaked after the finalize failure \
                         ({discard_err})"
                    ));
                }
                return Err(chain_adoption_failure(&adoption_error, err.to_string()));
            }
        };
        let mark = match (status, note) {
            (CaptureStatus::Interrupted, Some(note)) => CommitMark::Interrupted {
                note: note.to_string(),
            },
            (CaptureStatus::Interrupted, None) => CommitMark::Interrupted {
                note: "The take ended without a clean stop; its captured samples were kept."
                    .to_string(),
            },
            (CaptureStatus::Complete, _) => CommitMark::Complete,
        };
        let mut store = self.store.lock().expect("v2 store lock");
        if let Err(err) = finalized.commit_marked(&mut store, mark) {
            // commit_marked is promote → commit → gc, so the failure may
            // sit before OR after the promoting rename, and each shape
            // gets its own honest answer:
            //
            // - the row may have landed anyway (an error after the
            //   transaction committed — a failed WAL checkpoint or gc
            //   pass): the take IS durably stored, and an Err would
            //   invite a retry that collides with the row — answer Ok,
            //   with the failure named so the persisted take keeps a
            //   trace of it;
            // - the row read itself may fail: the commit's outcome is
            //   unknown, and Err carries the same retry-collision risk —
            //   the sealed journal survives on disk either way (staging
            //   or audio/, for reconcile to surface), so the honest
            //   answer is Ok with the unknown state reported;
            // - a provably rowless failure before the rename leaves the
            //   sealed journal in staging — discard_staging rolls it
            //   back, else reconcile would salvage it as an interrupted
            //   duplicate of a retry;
            // - a provably rowless failure after the rename (commit or
            //   gc) leaves the journal in audio/ with no row: the discard
            //   is a no-op (staging is gone), so the orphan is named in a
            //   divergence report — reconcile heals it as an orphaned
            //   session, never silently.
            match store.get_capture(&staged_id) {
                Ok(Some(_)) => {
                    report_divergence(format!(
                        "commit_marked errored after the row for {staged_id} landed ({err}) — \
                         the take is durably persisted"
                    ));
                    return Ok(());
                }
                Err(read_err) => {
                    report_divergence(format!(
                        "commit_marked failed ({err}) and the row for {staged_id} could not be \
                         read back ({read_err}) — the commit's outcome is unknown; reconcile \
                         will surface whatever landed"
                    ));
                    return Ok(());
                }
                Ok(None) => {}
            }
            let err = chain_adoption_failure(&adoption_error, err.to_string());
            // Metadata-only probe (one stat; load_audio would read and
            // verify the whole journal under the lock): which side of the
            // promoting rename did the failure leave the bytes on?
            let promoted = store.audio_journal_exists(&staged_id).unwrap_or(false);
            drop(store); // the filesystem work below runs off the lock
            if promoted {
                report_divergence(format!(
                    "commit failed after the audio for {staged_id} was promoted — the \
                     journal sits in audio/ with no row; reconcile will surface it as an \
                     orphaned session"
                ));
            } else if let Err(discard_err) = self
                .store
                .lock()
                .expect("v2 store lock")
                .discard_staging(&staged_id)
            {
                report_divergence(format!(
                    "staging journal {staged_id} leaked after the commit failure ({discard_err})"
                ));
            }
            return Err(err);
        }
        Ok(())
    }
}

/// Fold the adoption failure into a samples-path failure: the take was
/// stored from neither path, and the surfaced error must say both — the
/// samples error alone would hide the journal trouble that forced the
/// fall-back.
fn chain_adoption_failure(adoption_error: &Option<String>, samples_error: String) -> String {
    match adoption_error {
        None => samples_error,
        Some(reason) => format!(
            "journal adoption failed ({reason}); storing the take from its samples \
             failed too: {samples_error}"
        ),
    }
}

impl CaptureStore for V2CaptureStore {
    fn commit_take(&self, take: &TakeRecord) -> Result<(), String> {
        self.commit(take, CaptureStatus::Complete, None)
    }
    fn mark_interrupted(&self, take: &TakeRecord, note: &str) -> Result<(), String> {
        self.commit(take, CaptureStatus::Interrupted, Some(note))
    }
    fn describe(&self) -> String {
        "storage-v2".to_string()
    }
}

// ---------------------------------------------------------------------------
// The actor
// ---------------------------------------------------------------------------

/// Messages the capture actor receives.
pub enum CaptureMsg {
    Command(Inbound),
    /// Runtime-internal: replay an interrupted take to its durable boundary
    /// (`Interrupted → Recovering → Persisted`); used by recovery wiring
    /// and tests. No v1 command exists for this edge.
    Recover(String),
    /// A persist worker's report: a take's store commit finished off the
    /// actor loop (issue #249) and the close-out deferred to it resumes.
    Persist(PersistReport),
    Shutdown,
}

/// What a persist worker was asked to do with the take. `Clone` because
/// the spawn-failure fallback keeps a copy when the worker takes one (the
/// interrupted note is a short string — the take's audio is never
/// duplicated).
#[derive(Clone)]
enum PersistIntent {
    /// [`CaptureStore::commit_take`] — a cleanly stopped take.
    Commit,
    /// [`CaptureStore::mark_interrupted`] — a salvaged take, with its
    /// persisted note.
    Interrupted(String),
}

/// Which take-ending path handed the persist off — and therefore which
/// emissions are gated on the commit's result when the report lands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PersistFollow {
    /// `capture.stop`'s clean arm: `capture.stopped` on success, the fatal
    /// `storage_commit_failed` degradation on failure.
    CleanStop,
    /// The quiesce-timeout arm: the non-fatal `capture.error` and the tail
    /// gap already went out at stop time; `capture.stopped` follows the
    /// persist regardless of its result (the salvage is the outcome).
    QuiesceStop,
    /// A device error on the stop handshake: an optional
    /// `persist_interrupted_failed`, then the fatal
    /// `device_error_on_stop` that enters `Interrupted`.
    DeviceStopFatal,
    /// `capture.abort`: the machine is already `Idle`, where no `capture.*`
    /// event is legal — the report only registers the take and releases
    /// the route (the abort's documented no-wire-surface limit).
    Abort,
    /// The fatal mid-take salvage: an optional
    /// `persist_interrupted_failed`, then the fatal
    /// `device_stream_lost` that enters `Interrupted`.
    FatalSalvage,
}

/// A persist worker's report back to the actor: the take (back by handle,
/// for the registry), the store commit's result, the close-out to resume,
/// and the take epoch at handoff — if the epoch no longer matches the
/// actor's, a newer take has started since and the report is stale (it
/// lands registry-only, never against the new take's machine state).
pub struct PersistReport {
    corr: String,
    record: Arc<TakeRecord>,
    result: Result<(), String>,
    follow: PersistFollow,
    epoch: u64,
}

/// One live take: the session plus everything the stop path needs once the
/// session is consumed.
struct LiveTake {
    corr: String,
    policy: String,
    session: Box<dyn CaptureSession>,
    started_at: Instant,
    /// Number of gap spans already surfaced as `capture.gap`.
    surfaced_gaps: usize,
    last_progress_ack: u64,
    /// Whether the non-fatal journal-fault event was already emitted.
    journal_fault_surfaced: bool,
    device: String,
}

impl LiveTake {
    fn gaps_so_far(&self) -> Vec<SampleGap> {
        self.session
            .gaps()
            .into_iter()
            .map(|gap| SampleGap {
                start_sample: gap.start_sample,
                end_sample: gap.end_sample,
            })
            .collect()
    }
}

/// A salvage outcome: the interrupted record to persist, whether any
/// audio came back through the stop handshake (drives the persisted note),
/// and the unacknowledged-tail span when the salvaged audio stops short
/// of the sequence frontier.
struct SalvagedTake {
    record: TakeRecord,
    /// Whether the recorder handed back audio (a clean stop, or the
    /// quiesce timeout's preserved samples) — `false` for a device-side
    /// stop error, where the record is metadata-only.
    audio_preserved: bool,
    /// The span past the acknowledged boundary that no salvaged audio
    /// covers. `capture.stop`'s quiesce arm surfaces it as a `capture.gap`
    /// event while the machine is still in `Draining`; the abort and
    /// fatal-salvage paths run after the machine has left the states
    /// where that event is legal (`Idle` after `capture.abort`,
    /// `Interrupted` after the fatal error) — emitting there would only
    /// record a table violation, never reach the wire — so on those paths
    /// the persisted record is the evidence carrier.
    tail_gap: Option<SampleGap>,
}

/// Everything a salvage needs that must be read before the §3 stop
/// handshake consumes the session: afterwards only the outcome — not the
/// session — is left to ask.
struct SalvageFacts {
    corr: String,
    device: String,
    policy: String,
    rate: u32,
    /// The sequence frontier when the handshake began (the
    /// `finalSampleIndex` the record declares).
    final_sample_index: u64,
    /// Gap spans surfaced before the handshake.
    gaps: Vec<SampleGap>,
    /// The last acknowledged boundary the actor polled — the honest
    /// fallback when a failing handshake salvages no journal to ask.
    acknowledged_hint: u64,
    wall_clock_ms: f64,
}

impl LiveTake {
    fn salvage_facts(&self) -> SalvageFacts {
        SalvageFacts {
            corr: self.corr.clone(),
            device: self.device.clone(),
            policy: self.policy.clone(),
            rate: self.session.sample_rate(),
            final_sample_index: self.session.captured_sample_count(),
            gaps: self.gaps_so_far(),
            acknowledged_hint: self.last_progress_ack,
            wall_clock_ms: self.started_at.elapsed().as_secs_f64() * 1000.0,
        }
    }
}

/// The interrupted record for a stop outcome that carries no audio (a
/// device-side stop error, `RecorderError::Device`): metadata only. The
/// handshake consumed the session and the error salvages nothing
/// in-process, but the take must not vanish with it (issue #212): the
/// registry keeps the take visible for the jobs loader, the gap evidence
/// survives (the surfaced gap spans plus the unacknowledged tail), and
/// the acknowledged samples remain in the on-disk journal for the app's
/// orphan-recovery scan.
#[allow(clippy::too_many_arguments)]
fn metadata_only_record(
    corr: String,
    device: String,
    policy: String,
    rate: u32,
    final_sample_index: u64,
    mut gaps: Vec<SampleGap>,
    acknowledged: u64,
    wall_clock_ms: f64,
) -> TakeRecord {
    if final_sample_index > acknowledged {
        // The record holds no samples, so everything past the last
        // acknowledged boundary is missing audio — record it as a gap,
        // mirroring the quiesce-timeout salvage.
        gaps.push(SampleGap {
            start_sample: acknowledged,
            end_sample: final_sample_index,
        });
    }
    TakeRecord {
        id: corr,
        device,
        policy,
        samples: Vec::new(),
        sample_rate: rate,
        gaps,
        acknowledged_samples: acknowledged,
        final_sample_index,
        journal: None,
        status: TakeStatus::Interrupted,
        sample_duration_ms: acknowledged as f64 * 1000.0 / rate.max(1) as f64,
        wall_clock_ms,
        capture_id: crate::bus::new_id("cap"),
    }
}

/// The persisted note for a device-side stop error: the handshake returned
/// no audio, so the note states exactly what was kept (`upload.rs`'s
/// "never silently dropped" rule).
fn device_stop_note(cause: &str, acknowledged: u64) -> String {
    format!(
        "{cause}; the stop handshake failed on a device error, so this interrupted \
         recording keeps the take's metadata and gap evidence while {acknowledged} \
         acknowledged samples remain in the durable journal."
    )
}

/// Records the span past `acknowledged` that no salvaged audio covers
/// (the quiesce-timeout salvage's rule, now shared by every audio-carrying
/// outcome so the arms cannot drift): pushed into the record's `gaps` and
/// returned for the one path that can still surface it as an event.
fn unacknowledged_tail(
    acknowledged: u64,
    final_sample_index: u64,
    gaps: &mut Vec<SampleGap>,
) -> Option<SampleGap> {
    (final_sample_index > acknowledged).then(|| {
        let gap = SampleGap {
            start_sample: acknowledged,
            end_sample: final_sample_index,
        };
        gaps.push(gap.clone());
        gap
    })
}

/// Computes the interrupted record for a finished stop handshake — the
/// shared back half of `capture.stop`'s degraded outcomes, `capture.abort`,
/// and the fatal mid-take salvage (issue #212; one computation, so the
/// span/acknowledgement arithmetic cannot drift between the paths). A
/// clean stop and a quiesce timeout both hand back audio (the timeout's
/// preserved samples ride in the error, R09/I1 phase 2); a device-side
/// stop error carries none, so the record is metadata-only;
/// `RecorderError::Empty` has nothing to keep.
fn salvage_outcome(
    facts: SalvageFacts,
    outcome: Result<CapturedTake, RecorderError>,
) -> Option<SalvagedTake> {
    let SalvageFacts {
        corr,
        device,
        policy,
        rate,
        final_sample_index,
        mut gaps,
        acknowledged_hint,
        wall_clock_ms,
    } = facts;
    let (samples, journal, acknowledged, audio_preserved, tail_gap) = match outcome {
        Ok(captured) => {
            // The journal's fsynced boundary is the honest acknowledged
            // count — `stop_take`'s Ok arm computes it the same way. It
            // can exceed the handed-back sample count when earlier chunks
            // were already drained out of the recorder, and (the `.max`)
            // it never under-reports the audio this record itself holds.
            // Whatever lies past that boundary is tail evidence: a gap.
            let salvaged = captured.audio.samples.len() as u64;
            let acknowledged = captured
                .journal
                .as_ref()
                .map(|report| report.acknowledged_samples)
                .unwrap_or(salvaged)
                .max(salvaged);
            let tail_gap = unacknowledged_tail(acknowledged, final_sample_index, &mut gaps);
            (
                captured.audio.samples,
                captured.journal,
                acknowledged,
                true,
                tail_gap,
            )
        }
        Err(RecorderError::QuiesceTimeout { audio, journal, .. }) => {
            let salvaged = audio.samples.len() as u64;
            let tail_gap = unacknowledged_tail(salvaged, final_sample_index, &mut gaps);
            (audio.samples, journal, salvaged, true, tail_gap)
        }
        Err(RecorderError::Device(_)) => {
            return Some(SalvagedTake {
                record: metadata_only_record(
                    corr,
                    device,
                    policy,
                    rate,
                    final_sample_index,
                    gaps,
                    acknowledged_hint,
                    wall_clock_ms,
                ),
                audio_preserved: false,
                tail_gap: None,
            });
        }
        Err(RecorderError::Empty) => return None,
    };
    let capture_id = journal
        .as_ref()
        .map(|report| report.id.clone())
        .unwrap_or_else(|| crate::bus::new_id("cap"));
    Some(SalvagedTake {
        record: TakeRecord {
            id: corr,
            device,
            policy,
            samples,
            sample_rate: rate,
            gaps,
            acknowledged_samples: acknowledged,
            final_sample_index: final_sample_index.max(acknowledged),
            journal,
            status: TakeStatus::Interrupted,
            sample_duration_ms: acknowledged as f64 * 1000.0 / rate.max(1) as f64,
            wall_clock_ms,
            capture_id,
        },
        audio_preserved,
        tail_gap,
    })
}

/// Consumes a live take's session through the stop handshake and salvages
/// whatever comes back as an interrupted record — the entry point for the
/// paths that own the whole take (`capture.abort`, the fatal mid-take
/// salvage). `capture.stop`'s degraded arms share the same computation
/// through [`salvage_outcome`] but keep their own emissions (they are the
/// only ones still in a state where `capture.gap`/`capture.stopped` are
/// legal).
fn salvage_take(live: LiveTake) -> Option<SalvagedTake> {
    let facts = live.salvage_facts();
    let outcome = live.session.stop();
    salvage_outcome(facts, outcome)
}

/// The capture actor's configuration.
#[derive(Clone)]
pub struct CaptureConfig {
    /// Where per-take durable journals are created (recorder seam).
    pub journals_dir: PathBuf,
    /// Progress/gap polling cadence while `Recording` (`capture.progress`
    /// is throttled to this).
    pub poll_interval: Duration,
    /// How long a graceful shutdown waits for in-flight persist workers
    /// before abandoning them (issue #249's bounded drain). A normal
    /// persist — even a multi-minute take's WAV encode — fits comfortably
    /// inside the default; a worker wedged in a hung store write (fsync on
    /// a full disk) must not hold `Runtime::shutdown` hostage forever, so
    /// the drain is bounded: past the timeout the actor logs what it
    /// abandoned and exits. The takes' durable journals still survive on
    /// disk for startup recovery; only the store rows are lost.
    pub persist_drain_timeout: Duration,
}

impl Default for CaptureConfig {
    fn default() -> Self {
        CaptureConfig {
            journals_dir: starling_dictation::journal::default_journals_root(),
            poll_interval: Duration::from_millis(250),
            persist_drain_timeout: Duration::from_secs(30),
        }
    }
}

/// The capture actor. Spawned by [`crate::Runtime`]; single-owned.
pub struct CaptureActor {
    inbox: crate::channel::Receiver<CaptureMsg>,
    /// Persist workers post their reports here (a sender clone of the
    /// inbox — the same self-addressed-report shape the jobs scheduler
    /// uses for its workers).
    persist_inbox: crate::channel::Sender<CaptureMsg>,
    bus: Arc<EventBus>,
    view: super::ViewSlot,
    core: MachineCore,
    source: Arc<dyn CaptureSource>,
    store: Arc<dyn CaptureStore>,
    registry: TakeRegistry,
    config: CaptureConfig,
    freezer: RouteFreezer,
    take: Option<LiveTake>,
    /// The corr of the take the machine is currently working (set at
    /// `capture.start`, advanced by every new take). A persist report
    /// whose take is not current is stale — a newer take has started since
    /// the handoff — and lands registry-only, never against the machine
    /// state of the take that replaced it (review on #252).
    take_epoch: u64,
    /// Corrs of persist workers in flight. Empty is the actor's
    /// idle-and-exitable condition; `Shutdown` waits for these to report,
    /// bounded by [`CaptureConfig::persist_drain_timeout`].
    /// Persists in flight, keyed by `(take_epoch, corr)` — the corr alone
    /// is client-supplied and not unique across takes (two takes sharing
    /// a reused corr, or both defaulting to `take-anon`, must not share
    /// one pending entry; the epoch disambiguates) (review on #252).
    pending_persists: Vec<(u64, String)>,
    /// When `Shutdown` was first seen. The exit condition of the run loop
    /// is this flag (plus the drain state), never `RecvError::Closed`:
    /// the actor owns a sender clone of its own inbox for persist reports,
    /// so the channel cannot close while the actor lives and the `Closed`
    /// arm is unreachable by construction (review on #252).
    shutdown_at: Option<Instant>,
}

impl CaptureActor {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        inbox: crate::channel::Receiver<CaptureMsg>,
        persist_inbox: crate::channel::Sender<CaptureMsg>,
        bus: Arc<EventBus>,
        view: super::ViewSlot,
        source: Arc<dyn CaptureSource>,
        store: Arc<dyn CaptureStore>,
        registry: TakeRegistry,
        config: CaptureConfig,
        freezer: RouteFreezer,
    ) -> CaptureActor {
        CaptureActor {
            inbox,
            persist_inbox,
            bus,
            view,
            core: MachineCore::new(&CAPTURE),
            source,
            store,
            registry,
            config,
            freezer,
            take: None,
            take_epoch: 0,
            pending_persists: Vec::new(),
            shutdown_at: None,
        }
    }

    pub fn run(mut self) {
        loop {
            // The exit condition, checked up front: shutdown was seen AND
            // (every persist reported OR the bounded drain gave up on the
            // rest). A worker stuck in a hung store write must not hold
            // `Runtime::shutdown` — which joins this thread — hostage
            // forever: past the drain timeout the actor logs the takes it
            // abandoned and exits. Their durable journals survive on disk
            // for startup recovery; only the store rows are lost.
            if let Some(shutdown_at) = self.shutdown_at {
                let drained = self.pending_persists.is_empty();
                let timed_out = shutdown_at.elapsed() >= self.config.persist_drain_timeout;
                if drained || timed_out {
                    if !self.pending_persists.is_empty() {
                        eprintln!(
                            "starling-runtime: capture actor shutdown abandoned {} in-flight \
                             persist(s) after {:?} — the store write did not finish; takes {:?} \
                             keep their durable journals for startup recovery",
                            self.pending_persists.len(),
                            self.config.persist_drain_timeout,
                            self.pending_persists,
                        );
                    }
                    break;
                }
            }
            let timeout = if self.core.state() == "Recording" {
                self.config.poll_interval
            } else {
                Duration::from_millis(50)
            };
            match self.inbox.recv_timeout(timeout) {
                Ok(CaptureMsg::Command(inbound)) => self.handle_command(inbound),
                Ok(CaptureMsg::Recover(take_id)) => self.handle_recover(&take_id),
                Ok(CaptureMsg::Persist(report)) => self.handle_persist(report),
                Ok(CaptureMsg::Shutdown) | Err(crate::channel::RecvError::Closed) => {
                    // Start (or continue) the bounded drain. `Closed` is
                    // unreachable while the actor lives — it holds the
                    // persist-report sender clone of its own inbox — so
                    // the loop's exit is the flag checked above, never
                    // the channel closing. The loop keeps receiving
                    // throughout the drain, so a worker parked on a full
                    // inbox always makes progress — no join, no deadlock;
                    // each worker posts exactly one report, so a healthy
                    // drain terminates.
                    self.shutdown_at.get_or_insert_with(Instant::now);
                }
                Err(crate::channel::RecvError::Timeout) => {
                    // Once shutdown started, a live take is this runtime's
                    // past — do not let its polling open new persists that
                    // would extend the drain.
                    if self.shutdown_at.is_none() && self.core.state() == "Recording" {
                        self.poll_recording();
                    }
                }
            }
            *self.view.lock().expect("capture view lock") = self.core.view();
        }
    }

    fn publish_view(&self) {
        *self.view.lock().expect("capture view lock") = self.core.view();
    }

    /// Every machine emission goes through the core first: an illegal
    /// event is never sent (the stream stays oracle-legal), and the
    /// violation is recorded in the snapshot instead of being absorbed.
    fn emit(&mut self, event: Event, corr: &str) {
        let fatal = matches!(&event, Event::CaptureError { fatal, .. } if *fatal);
        match self.core.emit_event(event.type_name(), Some(fatal)) {
            Ok(_) => {
                let _ = self.bus.emit(event, Some(corr));
            }
            Err(violation) => {
                self.core.record_violation(violation);
            }
        }
    }

    fn handle_command(&mut self, inbound: Inbound) {
        let super::Inbound {
            corr, command, reply, ..
        } = inbound;
        let corr = corr.unwrap_or_else(|| "take-anon".to_string());
        match command {
            Command::CaptureStart { policy } => {
                match self.core.commit_command("capture.start", Some(corr.clone())) {
                    Ok(_) => {
                        let _ = reply.try_send(Ok(Receipt::Accepted));
                        self.start_take(corr, policy);
                    }
                    Err(violation) => {
                        self.core.record_violation(violation.clone());
                        let _ = reply.try_send(Err(rejection_for(
                            "capture.start",
                            self.core.state(),
                            violation,
                        )));
                    }
                }
            }
            Command::CaptureStop { drain: _ } => {
                match self.core.commit_command("capture.stop", Some(corr.clone())) {
                    Ok(_) => {
                        let _ = reply.try_send(Ok(Receipt::Accepted));
                        self.stop_take();
                    }
                    Err(violation) => {
                        self.core.record_violation(violation.clone());
                        let _ = reply.try_send(Err(rejection_for(
                            "capture.stop",
                            self.core.state(),
                            violation,
                        )));
                    }
                }
            }
            Command::CaptureAbort => {
                match self.core.commit_command("capture.abort", Some(corr.clone())) {
                    Ok(_) => {
                        let _ = reply.try_send(Ok(Receipt::Accepted));
                        self.abort_take(&corr);
                    }
                    Err(violation) => {
                        self.core.record_violation(violation.clone());
                        let _ = reply.try_send(Err(rejection_for(
                            "capture.abort",
                            self.core.state(),
                            violation,
                        )));
                    }
                }
            }
            other => {
                let _ = reply.try_send(Err(Rejection::UnknownMessageType(
                    other.type_name().to_string(),
                )));
            }
        }
        self.publish_view();
    }

    fn start_take(&mut self, corr: String, policy: String) {
        // The audio route freezes before the first frame can leave the
        // runtime: ask the context service to freeze now (it emits
        // mode.routeFrozen only from ModeDecided). A context that cannot
        // freeze does not block the take; submits on that route will then
        // be rejected at the audio-leave proxy.
        let _route = self.freezer.freeze(&corr);

        match self.source.start(&self.config.journals_dir, &policy) {
            Ok(session) => {
                // A successfully opened take invalidates every persist
                // still in flight for older ones: their reports will land
                // stale (registry-only), never against this take's
                // machine state (review on #252). The bump sits in this
                // arm only — a failed start (device_open_failed, back to
                // Idle) leaves no new take running, and bumping there
                // would strand the previous take's in-flight persist as
                // silently stale, losing its durable ack for nobody's
                // protection (review round on #252).
                self.take_epoch += 1;
                let device = "default-input".to_string();
                let rate = session.sample_rate();
                self.emit(
                    Event::CaptureStarted {
                        device: device.clone(),
                        actual_rate: rate,
                        channels: 1,
                    },
                    &corr,
                );
                self.take = Some(LiveTake {
                    corr,
                    policy,
                    session,
                    started_at: Instant::now(),
                    surfaced_gaps: 0,
                    last_progress_ack: 0,
                    journal_fault_surfaced: false,
                    device,
                });
            }
            Err(_message) => {
                // Fatal open failure from Acquiring -> Interrupted (fixture
                // take_9's device_open_failed), then the runtime-internal
                // settle edge back to Idle: the failure killed the take
                // *attempt*, not the machine — no session was created, so
                // there is nothing to salvage and no journal to replay, and
                // `Interrupted`'s only other exit (the `Recovering` replay)
                // requires a registered take. Staying there would reject
                // every later `capture.start` for the process lifetime
                // (issue #211). The fatal error event above is what the UI
                // sees; the next `capture.start` retries the device.
                self.emit(
                    Event::CaptureError {
                        code: "device_open_failed".into(),
                        fatal: true,
                    },
                    &corr,
                );
                if let Err(violation) = self.core.advance_internal("Idle") {
                    self.core.record_violation(violation);
                }
                // The freeze taken above for this corr is no longer backed
                // by any take; release the audio route (RouteFrozen ->
                // Released, runtime-internal) so the context cycle can run
                // again for the retry.
                let _ = self.freezer.take_completed(&corr);
            }
        }
        self.publish_view();
    }

    /// Recording-phase polling: surface new gaps, surface the first fault,
    /// and emit throttled progress with the honest acknowledged count.
    fn poll_recording(&mut self) {
        enum Action {
            Gap(SampleGap),
            Progress { ack: u64, clip: f64, level: f64 },
            JournalFault,
            FatalFault(String),
        }
        let mut actions: Vec<Action> = Vec::new();
        let corr = if let Some(take) = &mut self.take {
            let gaps = take.session.gaps();
            while take.surfaced_gaps < gaps.len() {
                let gap = gaps[take.surfaced_gaps];
                take.surfaced_gaps += 1;
                actions.push(Action::Gap(SampleGap {
                    start_sample: gap.start_sample,
                    end_sample: gap.end_sample,
                }));
            }
            if let Some(fault) = take.session.capture_fault() {
                if error_is_fatal(&fault) {
                    actions.push(Action::FatalFault(fault.message().to_string()));
                } else if !take.journal_fault_surfaced {
                    // Journal fault: surfaced once as a non-fatal error
                    // (capture continues in memory, acknowledgment frozen —
                    // the recorder's honest degraded state).
                    take.journal_fault_surfaced = true;
                    actions.push(Action::JournalFault);
                }
            }
            let ack = take.session.acknowledged_samples();
            if ack > take.last_progress_ack {
                take.last_progress_ack = ack;
                let clip = take.session.source_clip_ratio();
                let window = take.session.latest_window(2048);
                let level = if window.is_empty() {
                    0.0
                } else {
                    let mean_square =
                        window.iter().map(|s| s * s).sum::<f32>() / window.len() as f32;
                    (mean_square.sqrt()) as f64
                };
                actions.push(Action::Progress { ack, clip, level });
            }
            take.corr.clone()
        } else {
            return;
        };
        let mut fatal: Option<String> = None;
        for action in actions {
            match action {
                Action::Gap(gap) => self.emit(
                    Event::CaptureGap {
                        start_sample: gap.start_sample,
                        end_sample: gap.end_sample,
                    },
                    &corr,
                ),
                Action::JournalFault => self.emit(
                    Event::CaptureError {
                        code: "journal_fault".into(),
                        fatal: false,
                    },
                    &corr,
                ),
                Action::Progress { ack, clip, level } => {
                    self.emit(
                        Event::CaptureProgress {
                            ack_samples: ack,
                            clip_ratio: clip,
                            level,
                        },
                        &corr,
                    );
                }
                Action::FatalFault(message) => fatal = Some(message),
            }
        }
        if let Some(message) = fatal {
            // The fatal `capture.error` (the transition into Interrupted)
            // is emitted inside, *after* the salvage and its persist
            // attempt — a persist failure has to surface while the machine
            // is still in Recording, and the fatal event stays the last
            // thing on the wire.
            self.salvage_interrupted(format!("device error mid-take: {message}"));
        }
    }

    fn stop_take(&mut self) {
        let Some(live) = self.take.take() else {
            return;
        };
        let facts = live.salvage_facts();
        let corr = facts.corr.clone();
        let rate = facts.rate;
        let final_sample_index = facts.final_sample_index;
        let clip = live.session.source_clip_ratio();
        let outcome = live.session.stop();
        match outcome {
            Ok(captured) => {
                let journal = captured.journal.clone();
                let capture_id = journal
                    .as_ref()
                    .map(|report| report.id.clone())
                    .unwrap_or_else(|| crate::bus::new_id("cap"));
                let acknowledged = journal
                    .as_ref()
                    .map(|report| report.acknowledged_samples)
                    .unwrap_or(captured.audio.samples.len() as u64);
                // Drain-phase progress (fixture evt_006's shape).
                self.emit(
                    Event::CaptureProgress {
                        ack_samples: acknowledged,
                        clip_ratio: clip,
                        level: 0.0,
                    },
                    &corr,
                );
                let record = TakeRecord {
                    id: corr.clone(),
                    device: facts.device.clone(),
                    policy: facts.policy.clone(),
                    samples: captured.audio.samples,
                    sample_rate: rate,
                    gaps: facts.gaps.clone(),
                    acknowledged_samples: acknowledged,
                    final_sample_index: final_sample_index.max(acknowledged),
                    journal,
                    status: TakeStatus::Complete,
                    sample_duration_ms: acknowledged as f64 * 1000.0 / rate.max(1) as f64,
                    wall_clock_ms: facts.wall_clock_ms,
                    capture_id,
                };
                // The durable commit runs on a persist worker (issue #249):
                // the v1-file store encodes the whole take here, and that
                // must not stall the actor loop. `capture.stopped` (the
                // §4 durable ack) and the storage-fault degradation are
                // gated on the report in `handle_persist`.
                self.hand_off_persist(record, corr, PersistIntent::Commit, PersistFollow::CleanStop);
            }
            Err(err @ RecorderError::QuiesceTimeout { .. }) => {
                // R09/I1 phase 2 semantics: never a silent empty result —
                // the salvaged samples are kept, the take is marked
                // interrupted, and the timeout surfaces as an I0 event.
                self.emit(
                    Event::CaptureError {
                        code: "quiesce_timeout".into(),
                        fatal: false,
                    },
                    &corr,
                );
                let salvaged = salvage_outcome(facts, Err(err))
                    .expect("a quiesce timeout always salvages its preserved samples");
                // The unacknowledged tail goes on the wire while the
                // machine is still in Draining — the same computation the
                // record carries (salvage_outcome), so the two cannot
                // disagree.
                if let Some(gap) = salvaged.tail_gap {
                    self.emit(
                        Event::CaptureGap {
                            start_sample: gap.start_sample,
                            end_sample: gap.end_sample,
                        },
                        &corr,
                    );
                }
                let record = salvaged.record;
                let salvaged_count = record.acknowledged_samples;
                // The interrupted persist runs on a worker (issue #249);
                // `capture.stopped` follows its report regardless of the
                // commit's result (the salvage is the outcome — the
                // pre-persist events above already told the story).
                self.hand_off_persist(
                    record,
                    corr,
                    PersistIntent::Interrupted(format!(
                        "The microphone did not stop cleanly within the quiesce timeout; \
                         {salvaged_count} captured samples were salvaged and kept as this \
                         interrupted recording."
                    )),
                    PersistFollow::QuiesceStop,
                );
            }
            Err(RecorderError::Empty) => {
                self.emit(
                    Event::CaptureError {
                        code: "empty_capture".into(),
                        fatal: false,
                    },
                    &corr,
                );
                self.emit(
                    Event::CaptureStopped {
                        final_sample_index: 0,
                        acknowledged_samples: 0,
                        gaps: vec![],
                        journal_id: "unjournaled".into(),
                        sample_duration_ms: 0.0,
                        wall_clock_ms: facts.wall_clock_ms,
                    },
                    &corr,
                );
                // The take is resolved (nothing to keep); its route freeze
                // must not outlive it.
                let _ = self.freezer.take_completed(&corr);
            }
            Err(err @ RecorderError::Device(..)) => {
                // Salvaged through the shared computation (issue #212) and
                // persisted *before* the fatal error: while still in
                // Draining a persist failure can surface on the wire —
                // after the fatal error, Interrupted admits no capture.*
                // event at all. The persist itself runs on a worker
                // (issue #249); the fatal close-out is gated on its
                // report, preserving that ordering.
                let salvaged = salvage_outcome(facts, Err(err))
                    .expect("a device stop error always keeps the metadata-only record");
                let acknowledged = salvaged.record.acknowledged_samples;
                self.hand_off_persist(
                    salvaged.record,
                    corr,
                    PersistIntent::Interrupted(device_stop_note(
                        "The microphone failed during the stop handshake",
                        acknowledged,
                    )),
                    PersistFollow::DeviceStopFatal,
                );
            }
        }
        self.publish_view();
    }

    /// Hands a take's store persist to a dedicated worker thread and
    /// returns immediately — the encode-and-fsync a v1-file commit performs
    /// over a multi-minute take's whole audio must never run on the actor
    /// loop, where it stalled every `capture.*` command behind the stop
    /// handshake (issue #249; the same move #216 made for the jobs
    /// scheduler's workers). The worker shares the record by handle (the
    /// registry's own `Arc<TakeRecord>` shape — no sample clone; the store
    /// borrows it), posts exactly one report into the actor's inbox, and
    /// the close-out that the durable commit gates — `capture.stopped` and
    /// friends, the registry registration, the route release — resumes in
    /// [`Self::handle_persist`].
    ///
    /// A worker that panics is demoted to a persist failure (the same
    /// `Err` the store would have returned; the old inline shape would
    /// have taken the whole actor down with it). A thread that cannot
    /// spawn at all falls back to the inline persist: a stall is the
    /// lesser failure next to dropping the durability contract.
    fn hand_off_persist(
        &mut self,
        record: TakeRecord,
        corr: String,
        intent: PersistIntent,
        follow: PersistFollow,
    ) {
        let epoch = self.take_epoch;
        let record = Arc::new(record);
        let worker_record = Arc::clone(&record);
        let worker_intent = intent.clone();
        let store = Arc::clone(&self.store);
        let inbox = self.persist_inbox.clone();
        let worker_corr = corr.clone();
        let spawned = std::thread::Builder::new()
            .name(format!("starling-capture-persist-{worker_corr}"))
            .spawn(move || {
                let run = || match &worker_intent {
                    PersistIntent::Commit => store.commit_take(&worker_record),
                    PersistIntent::Interrupted(note) => {
                        store.mark_interrupted(&worker_record, note)
                    }
                };
                let result = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(run)) {
                    Ok(result) => result,
                    Err(payload) => Err(format!(
                        "persist worker panicked during {} for take {worker_corr}: {}",
                        match &worker_intent {
                            PersistIntent::Commit => "commit_take",
                            PersistIntent::Interrupted(_) => "mark_interrupted",
                        },
                        panic_message(&payload),
                    )),
                };
                // The Result is already surfaced inside (stderr on a
                // closed inbox); the worker can do nothing further.
                let _ = deliver_persist_report(
                    &inbox,
                    PersistReport {
                        corr: worker_corr,
                        record: worker_record,
                        result,
                        follow,
                        epoch,
                    },
                );
            });
        match spawned {
            Ok(_) => self.pending_persists.push((epoch, corr)),
            Err(_) => {
                let result = match &intent {
                    PersistIntent::Commit => self.store.commit_take(&record),
                    PersistIntent::Interrupted(note) => self.store.mark_interrupted(&record, note),
                };
                // The worker never existed, so this handle is the only one.
                self.handle_persist(PersistReport {
                    corr,
                    record,
                    result,
                    follow,
                    epoch,
                });
            }
        }
    }

    /// Resumes a take's close-out from its persist worker's report. Every
    /// emission here was, before issue #249, emitted inline after the
    /// store call returned — the ordering contracts are unchanged:
    /// `capture.stopped` still follows the successful commit (§4's durable
    /// ack), a failed commit still degrades before the take's fatal
    /// close-out, the registry still receives the take only once its store
    /// outcome is known (and before the `capture.stopped` a jobs submit
    /// could be gated on), and the route release still brings up the rear
    /// (the abort path excepted: its freeze releases at the decision
    /// point, so the slow persist cannot wedge `context.snapshot`).
    ///
    /// Reports are scoped to their take by the epoch captured at handoff
    /// (review on #252). A report whose epoch no longer matches is stale
    /// — a newer take started while the persist ran — and lands
    /// registry-only: its emissions must not be judged legal from the
    /// *new* take's state, where a stale `capture.stopped` would consume
    /// the new take's `Draining` (swallowing its own announcement) and a
    /// stale fatal `capture.error` would kill the new take outright. The
    /// stale take is still never dropped: it registers, and its route
    /// releases if its path had not already.
    ///
    /// The one genuinely new interleaving is `capture.abort` landing while
    /// a stop's persist runs: the table has always allowed abort from
    /// `Draining`, but the inline stall meant no abort could ever arrive
    /// there. It takes the machine to `Idle`, where no `capture.*` event
    /// is legal — so the persist lands silently (registry only, no wire
    /// event) instead of being forced through `emit`, which would record
    /// a table violation for a sequence the table itself permits. The
    /// store write still completes; only its announcement is skipped,
    /// because the abort already announced the take's end.
    fn handle_persist(&mut self, report: PersistReport) {
        let PersistReport {
            corr,
            record,
            result,
            follow,
            epoch,
        } = report;
        self.pending_persists
            .retain(|pending| pending != &(epoch, corr.clone()));
        // A failed clean-stop commit is not persisted: the record the
        // registry keeps says Interrupted (source preserved), stale or
        // not — the flip is record data, not an emission, so it happens
        // before any branching.
        let record = match (&result, follow) {
            (Err(_), PersistFollow::CleanStop) => demote_to_interrupted(record),
            _ => record,
        };
        // Stale scoping (review on #252): a newer take has started since
        // this persist was handed off, so the machine's current state —
        // `Acquiring`/`Recording`/`Draining`, whatever the new take is in
        // — belongs to that take, not this report. Landing this report's
        // emissions there would poison the new take (a stale
        // `capture.stopped` consumes the new take's `Draining`, so its
        // own announcement is then refused; a stale fatal
        // `capture.error` kills it outright). The take itself is never
        // dropped: it lands in the registry, its route releases if the
        // abort decision point has not already released it, and nothing
        // is emitted — the same rule the jobs scheduler applies to a
        // report arriving after its job was cancelled.
        if epoch != self.take_epoch {
            self.register(record);
            // No route release here (pullfrog on #252): with the abort's
            // decision-point release covering the no-live-take arm, every
            // stale take's route already released at its abort — and
            // `ContextActor`'s `TakeCompleted` is corr-blind, so this call
            // could only release a *newer* take's freeze (reachable when a
            // fresh context mode-cycle re-froze `RouteFrozen` between the
            // abort and the new take).
            self.publish_view();
            return;
        }
        // The states each follow's emissions are still legal from. A state
        // outside them means a command moved the machine while the persist
        // ran (only abort can, on the stop paths; the abort and
        // fatal-salvage paths spell out their own corners below).
        let stopped_legal = matches!(self.core.state(), "Draining" | "Recovering");
        let error_legal = matches!(
            self.core.state(),
            "Acquiring" | "Recording" | "Draining" | "Recovering"
        );
        // TODO(#249, review): when `capture.abort` lands while a stop's
        // persist is in flight (epoch unchanged, machine taken to Idle),
        // both gates are false and every emission for the STOP's corr is
        // suppressed — the client that issued capture.stop never gets its
        // durable ack (capture.stopped or the storage_commit_failed
        // degradation) and can only discover the outcome by timeout. The
        // store write itself succeeds and the take is durable; only its
        // announcement is lost. A stop-correlated terminal event for this
        // interleaving is a protocol-level decision to make separately.
        match follow {
            PersistFollow::CleanStop => match result {
                Ok(()) => {
                    let stopped = stopped_event(&record);
                    self.register(record);
                    if stopped_legal {
                        self.emit(stopped, &corr);
                    }
                    let _ = self.freezer.take_completed(&corr);
                }
                Err(_message) => {
                    // Storage fault: the take is not persisted —
                    // Interrupted, source preserved in the registry.
                    if error_legal {
                        self.emit(
                            Event::CaptureError {
                                code: "storage_commit_failed".into(),
                                fatal: true,
                            },
                            &corr,
                        );
                    }
                    self.register(record);
                    // The take is over even though its store write
                    // failed; release the audio route so the context
                    // cycle is not wedged behind it.
                    let _ = self.freezer.take_completed(&corr);
                }
            },
            PersistFollow::QuiesceStop => {
                // The quiesce arm emits `capture.stopped` regardless of
                // the persist's result (the salvage is the outcome).
                let stopped = stopped_event(&record);
                self.register(record);
                if stopped_legal {
                    self.emit(stopped, &corr);
                }
                let _ = self.freezer.take_completed(&corr);
            }
            PersistFollow::DeviceStopFatal => {
                if error_legal {
                    if result.is_err() {
                        // Surface, don't swallow: the take must not vanish
                        // without a trace (issue #212's rule).
                        self.emit(
                            Event::CaptureError {
                                code: "persist_interrupted_failed".into(),
                                fatal: false,
                            },
                            &corr,
                        );
                    }
                    // The handshake consumed the session and the error
                    // carries no audio — the take still must not vanish
                    // with it. The metadata-only record is registered
                    // (gap evidence included); the acknowledged samples
                    // remain in the on-disk journal for the
                    // orphan-recovery scan. The fatal error below enters
                    // `Interrupted`, the contract's terminal state for a
                    // lost device (the fixture capture-interrupted.json),
                    // so no `capture.stopped` follows.
                    self.emit(
                        Event::CaptureError {
                            code: "device_error_on_stop".into(),
                            fatal: true,
                        },
                        &corr,
                    );
                }
                self.register(record);
                // The machine is Interrupted, but the audio route is not
                // the device's to keep: release the freeze so the context
                // cycle can run again once recovery replays this take
                // (the same wedge class issue #211 fixed, one layer down).
                let _ = self.freezer.take_completed(&corr);
            }
            PersistFollow::Abort => {
                // No `capture.*` event is legal from Idle after the abort
                // (and a whole new take may already be running): land the
                // take, wire-silent. The route already released at the
                // abort decision point — the persist must not extend the
                // freeze (review on #252).
                self.register(record);
            }
            PersistFollow::FatalSalvage => {
                if error_legal {
                    if result.is_err() {
                        // Surface, don't swallow: the take must not vanish
                        // without a trace (legal from Recording/Draining;
                        // after the fatal error below it would not be).
                        self.emit(
                            Event::CaptureError {
                                code: "persist_interrupted_failed".into(),
                                fatal: false,
                            },
                            &corr,
                        );
                    }
                    self.emit(
                        Event::CaptureError {
                            code: "device_stream_lost".into(),
                            fatal: true,
                        },
                        &corr,
                    );
                }
                self.register(record);
                // The machine is Interrupted, but the audio route is not
                // the device's to keep: release the freeze so the context
                // cycle can run again for whatever follows (recovery or a
                // restart).
                let _ = self.freezer.take_completed(&corr);
            }
        }
        self.publish_view();
    }

    /// `capture.abort` — v1 defines no event; the machine returns to Idle
    /// and whatever the recorder acknowledged is salvaged as an
    /// interrupted take (source preserved, never deleted). A stop
    /// handshake that itself fails is no exception (issue #212): the
    /// quiesce timeout's preserved samples ride in the error, and a
    /// device-side stop error still registers the metadata-only record.
    fn abort_take(&mut self, corr: &str) {
        // The route releases at the abort DECISION point, not with the
        // persist report (review on #252): the store commit is precisely
        // the slow encode, and letting the freeze outlive the take by its
        // full duration wedges `context.snapshot` out of `RouteFrozen`
        // for the whole window. This runs BEFORE the live-take guard so
        // the no-live-take arm — an abort arriving while a stop's persist
        // is in flight (`stop_take` consumes the take before the
        // handoff) — releases too; otherwise the freeze would hold for
        // the whole persist, exactly the window this module makes
        // seconds long (pullfrog, reproduced). Idempotent on the context
        // side (RouteFrozen → Released is the only edge it takes).
        let _ = self.freezer.take_completed(corr);
        let Some(live) = self.take.take() else {
            self.publish_view();
            return;
        };
        let corr = live.corr.clone();
        if let Some(salvaged) = salvage_take(live) {
            let note = if salvaged.audio_preserved {
                "Take aborted by user; captured samples kept as an interrupted recording."
                    .to_string()
            } else {
                device_stop_note(
                    "Take aborted by user",
                    salvaged.record.acknowledged_samples,
                )
            };
            // The interrupted persist runs on a worker (issue #249); the
            // report's [`PersistFollow::Abort`] close-out registers the
            // take. A persist failure here has no legal wire surface (v1
            // defines no event for abort and no capture.* event is legal
            // from Idle — the machine committed capture.abort before the
            // salvage runs), so this failure is a documented limit, not an
            // oversight; the registry registration keeps the take itself
            // visible to the jobs loader, and only the store's
            // interrupted row is lost.
            self.hand_off_persist(
                salvaged.record,
                corr.clone(),
                PersistIntent::Interrupted(note),
                PersistFollow::Abort,
            );
        }
        self.publish_view();
    }

    /// Fatal mid-take error: salvage and persist *first* — while the
    /// machine is still `Recording`, a persist failure can surface on the
    /// wire — then the fatal `capture.error`, which is the transition into
    /// `Interrupted`. The wire keeps its contract shape: the fatal event is
    /// the last emission, and nothing follows it. The persist runs on a
    /// worker (issue #249), so the machine honestly sits in `Recording`
    /// with no live take while it runs; the report's
    /// [`PersistFollow::FatalSalvage`] close-out performs the gated
    /// emissions in the same order.
    fn salvage_interrupted(&mut self, note: String) {
        let Some(live) = self.take.take() else {
            return;
        };
        let corr = live.corr.clone();
        let facts = live.salvage_facts();
        let outcome = live.session.stop();
        let salvaged = salvage_outcome(facts, outcome);
        match salvaged {
            Some(salvaged) => {
                // Whatever the recorder acknowledged is salvaged, and the
                // gap spans already surfaced as `capture.gap` events travel
                // with the persisted record (issue #212: they used to be
                // dropped on this path).
                let persisted_note = if salvaged.audio_preserved {
                    note
                } else {
                    device_stop_note(&note, salvaged.record.acknowledged_samples)
                };
                self.hand_off_persist(
                    salvaged.record,
                    corr,
                    PersistIntent::Interrupted(persisted_note),
                    PersistFollow::FatalSalvage,
                );
            }
            None => {
                // Nothing salvageable: the fatal error is the whole story,
                // and the take's route freeze must not outlive it.
                self.emit(
                    Event::CaptureError {
                        code: "device_stream_lost".into(),
                        fatal: true,
                    },
                    &corr,
                );
                let _ = self.freezer.take_completed(&corr);
            }
        }
        self.publish_view();
    }

    /// `Interrupted → Recovering → Persisted` (journal replay): re-publish
    /// the salvaged boundary of a take this runtime holds. On-disk journal
    /// recovery of *previous* processes remains the app's startup scan
    /// (I1/I2); this path serves same-process interrupted takes.
    fn handle_recover(&mut self, take_id: &str) {
        if self.core.state() != "Interrupted" {
            return;
        }
        let record = self
            .registry
            .lock()
            .expect("take registry lock")
            .get(take_id)
            .cloned();
        let Some(record) = record else {
            return;
        };
        if let Err(violation) = self.core.advance_internal("Recovering") {
            self.core.record_violation(violation);
            return;
        }
        self.publish_view();
        // Recovery surfaces the tail-truncation gap (fixture shape), then
        // publishes the recovered boundary.
        if record.final_sample_index > record.acknowledged_samples {
            self.emit(
                Event::CaptureGap {
                    start_sample: record.acknowledged_samples,
                    end_sample: record.final_sample_index,
                },
                take_id,
            );
        }
        self.emit(
            Event::CaptureProgress {
                ack_samples: record.acknowledged_samples,
                clip_ratio: 0.0,
                level: 0.0,
            },
            take_id,
        );
        self.emit(
            Event::CaptureStopped {
                final_sample_index: record.final_sample_index,
                acknowledged_samples: record.acknowledged_samples,
                gaps: record.gaps.clone(),
                journal_id: record.capture_id.clone(),
                sample_duration_ms: record.sample_duration_ms,
                wall_clock_ms: record.wall_clock_ms,
            },
            take_id,
        );
    }

    fn register(&self, record: Arc<TakeRecord>) {
        let id = record.id.clone();
        self.registry
            .lock()
            .expect("take registry lock")
            .insert(id, record);
    }
}

/// The `capture.stopped` event for a finished take — one shape shared by
/// every close-out that emits it, so the persisted record and the wire
/// announcement cannot drift.
fn stopped_event(record: &TakeRecord) -> Event {
    Event::CaptureStopped {
        final_sample_index: record.final_sample_index,
        acknowledged_samples: record.acknowledged_samples,
        gaps: record.gaps.clone(),
        journal_id: record.capture_id.clone(),
        sample_duration_ms: record.sample_duration_ms,
        wall_clock_ms: record.wall_clock_ms,
    }
}

/// Marks a registry-bound record as interrupted (a failed clean-stop
/// commit is not persisted; the registry keeps the source). The report's
/// handle is the record's only reference by construction — the worker
/// moved its clone into the report and dropped it — so the unwrap
/// succeeds without copying the samples; if that invariant ever breaks,
/// the fallback clones rather than taking the actor down with a panic
/// (the samples are the one thing worth being gentle about).
fn demote_to_interrupted(record: Arc<TakeRecord>) -> Arc<TakeRecord> {
    match Arc::try_unwrap(record) {
        Ok(mut owned) => {
            owned.status = TakeStatus::Interrupted;
            Arc::new(owned)
        }
        Err(shared) => {
            let mut copy = (*shared).clone();
            copy.status = TakeStatus::Interrupted;
            Arc::new(copy)
        }
    }
}

/// A panic payload rendered for the demoted persist-failure message:
/// string payloads (the common `panic!("...")` and `assert!` shapes) are
/// kept verbatim; anything else is named as such instead of being lost.
fn panic_message(payload: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(text) = payload.downcast_ref::<&str>() {
        (*text).to_string()
    } else if let Some(text) = payload.downcast_ref::<String>() {
        text.clone()
    } else {
        "non-string panic payload".to_string()
    }
}

/// Posts a persist worker's report into the capture actor's inbox with a
/// delivery guarantee instead of a best-effort `try_send` — the same
/// contract the jobs scheduler's `deliver_worker_report` holds for its
/// workers. The inbox is the bounded queue that carries every `capture.*`
/// command, and a report dropped on a full queue would silently strand
/// the take's close-out (no `capture.stopped`, no registry entry, no
/// route release) while its worker exits; delivery therefore parks on
/// [`Sender::send_blocking`] while the actor drains, which applies
/// backpressure to the worker's own thread and never the actor loop.
///
/// `Closed` means the actor is gone (runtime shutdown that did not wait
/// — by construction it does, but a crashed actor cannot be unwedged by
/// a worker): the report is surfaced on stderr and dropped rather than
/// vanishing silently.
fn deliver_persist_report(
    inbox: &crate::channel::Sender<CaptureMsg>,
    report: PersistReport,
) -> Result<(), crate::channel::RecvError> {
    let corr = report.corr.clone();
    let sent = match inbox.try_send(CaptureMsg::Persist(report)) {
        Ok(()) => Ok(()),
        Err(crate::channel::TrySendError::Full(message)) => inbox.send_blocking(message),
        Err(crate::channel::TrySendError::Closed(_)) => Err(crate::channel::RecvError::Closed),
    };
    if sent.is_err() {
        eprintln!(
            "starling-runtime: capture actor gone before take {corr} persist reported; report dropped"
        );
    }
    sent
}

fn rejection_for(command: &str, state: &str, violation: crate::protocol::replay::Violation) -> Rejection {
    Rejection::IllegalInState {
        command: command.to_string(),
        state: state.to_string(),
        detail: violation.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn journal_hash_matches_the_journal_writer_algorithm() {
        let samples = vec![0.0f32, 1.0, -0.5, 0.25];
        let take = TakeRecord {
            id: "t".into(),
            device: "d".into(),
            policy: "p".into(),
            samples: samples.clone(),
            sample_rate: 16000,
            gaps: vec![],
            acknowledged_samples: samples.len() as u64,
            final_sample_index: samples.len() as u64,
            journal: None,
            status: TakeStatus::Complete,
            sample_duration_ms: 0.0,
            wall_clock_ms: 0.0,
            capture_id: "c".into(),
        };
        const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
        const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
        let mut expected = FNV_OFFSET;
        for &sample in &samples {
            for byte in sample.to_bits().to_le_bytes() {
                expected ^= u64::from(byte);
                expected = expected.wrapping_mul(FNV_PRIME);
            }
        }
        assert_eq!(take.journal_hash(), format!("{expected:016x}"));
    }

    #[test]
    fn error_classification_separates_journal_faults_from_device_faults() {
        use starling_dictation::recorder::RecorderFault;
        // Fatality is decided by variant (origin), never by message text:
        // a device error whose text happens to mention the journal is
        // still fatal, and a journal fault is not (issue #216).
        assert!(!error_is_fatal(&RecorderFault::Journal(
            "The capture journal failed: disk full. Recording continues.".into()
        )));
        assert!(error_is_fatal(&RecorderFault::Device(
            "DeviceUnavailable".into()
        )));
        assert!(error_is_fatal(&RecorderFault::Device(
            "stream error while flushing the journal buffer".into()
        )));
    }
}
