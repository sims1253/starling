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

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use starling_dictation::audio::encode_wav_16k_parts;
use starling_dictation::recorder::{
    CaptureGap, CapturedTake, JournalReport, RecorderError, RecorderFault, RecorderHandle,
};
use starling_dictation::storage::FileSessionStore;
use starling_dictation::store_v2::{CaptureRecord, CaptureStatus, StoreV2, TakeMeta};

use crate::bus::EventBus;
use crate::machine::{Inbound, MachineCore, Receipt, Rejection};
use crate::protocol::tables::CAPTURE;
use crate::protocol::{Command, Event, SampleGap};

use super::context::RouteFreezer;

/// The environment flag that opts into the storage v2 capture store
/// (`store_v2::STORAGE_V2_FLAG_ENV`, I2: v2 ships alongside v1 with no
/// automatic switchover).
pub const STORE_FLAG: &str = starling_dictation::store_v2::STORAGE_V2_FLAG_ENV;

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

/// The landed v1 store (`FileSessionStore`): audio as WAV sessions with
/// the additive journal linkage, interrupted takes marked on the session.
pub struct V1FileCaptureStore {
    store: FileSessionStore,
}

impl V1FileCaptureStore {
    pub fn open(root: impl Into<PathBuf>) -> Result<Self, String> {
        FileSessionStore::open(root)
            .map(|store| V1FileCaptureStore { store })
            .map_err(|err| err.to_string())
    }
}

impl CaptureStore for V1FileCaptureStore {
    fn commit_take(&self, take: &TakeRecord) -> Result<(), String> {
        let wav = take.to_wav()?;
        self.store
            .create_with_journal(
                wav,
                Some(take.sample_duration_ms),
                take.journal.as_ref().map(|report| report.id.as_str()),
            )
            .map(|_| ())
            .map_err(|err| err.to_string())
    }
    fn mark_interrupted(&self, take: &TakeRecord, note: &str) -> Result<(), String> {
        let wav = take.to_wav()?;
        let session = self
            .store
            .create_with_journal(
                wav,
                Some(take.sample_duration_ms),
                take.journal.as_ref().map(|report| report.id.as_str()),
            )
            .map_err(|err| err.to_string())?;
        self.store
            .mark_interrupted(&session.id, note)
            .map(|_| ())
            .map_err(|err| err.to_string())
    }
    fn describe(&self) -> String {
        "v1-file".to_string()
    }
}

/// The storage v2 store (behind `STARLING_STORAGE_V2`, I2): the recorder's
/// finalized `.sj` journal is adopted into `<root>/audio/` (the journal
/// format is the one `store_v2` reads) and the `captures` row committed in
/// one SQLite transaction — the §4 metadata step.
pub struct V2CaptureStore {
    store: Mutex<StoreV2>,
    root: PathBuf,
}

impl V2CaptureStore {
    pub fn open(root: impl Into<PathBuf>) -> Result<Self, String> {
        let root = root.into();
        let store = StoreV2::open(&root).map_err(|err| err.to_string())?;
        Ok(V2CaptureStore {
            store: Mutex::new(store),
            root,
        })
    }

    fn adopt_journal(&self, take: &TakeRecord) -> Result<(), String> {
        let audio_dir = self.root.join("audio");
        std::fs::create_dir_all(&audio_dir).map_err(|err| err.to_string())?;
        let Some(report) = &take.journal else {
            return Ok(()); // unjournaled take: row only, noted in extra_json
        };
        let destination = audio_dir.join(format!("{}.sj", take.capture_id));
        if !destination.exists() {
            std::fs::copy(&report.path, &destination).map_err(|err| err.to_string())?;
        }
        Ok(())
    }

    fn commit(&self, take: &TakeRecord, status: CaptureStatus) -> Result<(), String> {
        self.adopt_journal(take)?;
        let mut meta = TakeMeta::for_device(take.device.clone());
        meta.policy = take.policy.clone();
        meta.extra_json = Some(
            serde_json::json!({
                "takeCorr": take.id,
                "gaps": take.gaps,
                "acknowledgedSamples": take.acknowledged_samples,
                "journalFinalized": take.journal.as_ref().map(|r| r.finalized),
                "journalFault": take.journal.as_ref().and_then(|r| r.fault.clone()),
                "wallClockMs": take.wall_clock_ms,
            })
            .to_string(),
        );
        let record = CaptureRecord {
            id: take.capture_id.clone(),
            created_utc: crate::bus::now_ts(),
            tz: meta.tz.clone(),
            device: meta.device.clone(),
            actual_rate: take.sample_rate,
            policy: meta.policy.clone(),
            frame_count: take.samples.len() as u64,
            ack_sample_index: take.acknowledged_samples,
            journal_hash: take.journal_hash(),
            status,
            retention_class: meta.retention_class.clone(),
            extra_json: meta.extra_json.clone(),
        };
        let mut store = self.store.lock().expect("v2 store lock");
        store.commit_capture(&record).map_err(|err| err.to_string())
    }
}

impl CaptureStore for V2CaptureStore {
    fn commit_take(&self, take: &TakeRecord) -> Result<(), String> {
        self.commit(take, CaptureStatus::Complete)
    }
    fn mark_interrupted(&self, take: &TakeRecord, _note: &str) -> Result<(), String> {
        self.commit(take, CaptureStatus::Interrupted)
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
    Shutdown,
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
}

impl Default for CaptureConfig {
    fn default() -> Self {
        CaptureConfig {
            journals_dir: starling_dictation::journal::default_journals_root(),
            poll_interval: Duration::from_millis(250),
        }
    }
}

/// The capture actor. Spawned by [`crate::Runtime`]; single-owned.
pub struct CaptureActor {
    inbox: crate::channel::Receiver<CaptureMsg>,
    bus: Arc<EventBus>,
    view: super::ViewSlot,
    core: MachineCore,
    source: Arc<dyn CaptureSource>,
    store: Arc<dyn CaptureStore>,
    registry: TakeRegistry,
    config: CaptureConfig,
    freezer: RouteFreezer,
    take: Option<LiveTake>,
}

impl CaptureActor {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        inbox: crate::channel::Receiver<CaptureMsg>,
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
            bus,
            view,
            core: MachineCore::new(&CAPTURE),
            source,
            store,
            registry,
            config,
            freezer,
            take: None,
        }
    }

    pub fn run(mut self) {
        loop {
            let timeout = if self.core.state() == "Recording" {
                self.config.poll_interval
            } else {
                Duration::from_millis(50)
            };
            match self.inbox.recv_timeout(timeout) {
                Ok(CaptureMsg::Command(inbound)) => self.handle_command(inbound),
                Ok(CaptureMsg::Recover(take_id)) => self.handle_recover(&take_id),
                Ok(CaptureMsg::Shutdown) | Err(crate::channel::RecvError::Closed) => break,
                Err(crate::channel::RecvError::Timeout) => {
                    if self.core.state() == "Recording" {
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
                match self.core.commit_command("capture.abort", Some(corr)) {
                    Ok(_) => {
                        let _ = reply.try_send(Ok(Receipt::Accepted));
                        self.abort_take();
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
                match self.store.commit_take(&record) {
                    Ok(()) => {
                        self.emit(
                            Event::CaptureStopped {
                                final_sample_index: record.final_sample_index,
                                acknowledged_samples: record.acknowledged_samples,
                                gaps: record.gaps.clone(),
                                journal_id: record.capture_id.clone(),
                                sample_duration_ms: record.sample_duration_ms,
                                wall_clock_ms: record.wall_clock_ms,
                            },
                            &corr,
                        );
                        self.register(record);
                        let _ = self.freezer.take_completed(&corr);
                    }
                    Err(_message) => {
                        // Storage fault: the take is not persisted —
                        // Interrupted, source preserved in the registry.
                        self.emit(
                            Event::CaptureError {
                                code: "storage_commit_failed".into(),
                                fatal: true,
                            },
                            &corr,
                        );
                        let mut record = record;
                        record.status = TakeStatus::Interrupted;
                        self.register(record);
                        // The take is over even though its store write
                        // failed; release the audio route so the context
                        // cycle is not wedged behind it.
                        let _ = self.freezer.take_completed(&corr);
                    }
                }
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
                let _ = self.store.mark_interrupted(
                    &record,
                    &format!(
                        "The microphone did not stop cleanly within the quiesce timeout; \
                         {salvaged_count} captured samples were salvaged and kept as this \
                         interrupted recording."
                    ),
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
                    &corr,
                );
                self.register(record);
                let _ = self.freezer.take_completed(&corr);
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
                // event at all.
                let salvaged = salvage_outcome(facts, Err(err))
                    .expect("a device stop error always keeps the metadata-only record");
                let acknowledged = salvaged.record.acknowledged_samples;
                if let Err(_e) = self.store.mark_interrupted(
                    &salvaged.record,
                    &device_stop_note(
                        "The microphone failed during the stop handshake",
                        acknowledged,
                    ),
                ) {
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
                // The handshake consumed the session and the error carries
                // no audio — the take still must not vanish with it. The
                // metadata-only record is registered (gap evidence
                // included); the acknowledged samples remain in the
                // on-disk journal for the orphan-recovery scan. The fatal
                // error below already enters `Interrupted`, the
                // contract's terminal state for a lost device (the fixture
                // capture-interrupted.json), so no `capture.stopped`
                // follows.
                self.emit(
                    Event::CaptureError {
                        code: "device_error_on_stop".into(),
                        fatal: true,
                    },
                    &corr,
                );
                self.register(salvaged.record);
                // The machine is Interrupted, but the audio route is not
                // the device's to keep: release the freeze so the context
                // cycle can run again once recovery replays this take
                // (the same wedge class issue #211 fixed, one layer down).
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
    fn abort_take(&mut self) {
        let Some(live) = self.take.take() else {
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
            if let Err(_e) = self.store.mark_interrupted(&salvaged.record, &note) {
                // v1 defines no event for abort and no capture.* event is
                // legal from Idle — the machine committed capture.abort
                // before the salvage runs — so this failure has no legal
                // wire surface here (a documented limit, not an
                // oversight; the paths still in Draining/Recording emit
                // `persist_interrupted_failed`). The registry
                // registration below keeps the take itself visible to the
                // jobs loader; only the store's interrupted row is lost.
            }
            self.register(salvaged.record);
        }
        // The take is over whatever the handshake said — release the
        // audio route so the context cycle can run again (an aborted take
        // whose stop failed device-side used to leak its freeze, wedging
        // `context.snapshot` out of `RouteFrozen`).
        let _ = self.freezer.take_completed(&corr);
        self.publish_view();
    }

    /// Fatal mid-take error: salvage and persist *first* — while the
    /// machine is still `Recording`, a persist failure can surface on the
    /// wire — then the fatal `capture.error`, which is the transition into
    /// `Interrupted`. The wire keeps its contract shape: the fatal event is
    /// the last emission, and nothing follows it.
    fn salvage_interrupted(&mut self, note: String) {
        let Some(live) = self.take.take() else {
            return;
        };
        let corr = live.corr.clone();
        let facts = live.salvage_facts();
        let outcome = live.session.stop();
        let salvaged = salvage_outcome(facts, outcome);
        if let Some(salvaged) = &salvaged {
            // Whatever the recorder acknowledged is salvaged, and the gap
            // spans already surfaced as `capture.gap` events travel with
            // the persisted record (issue #212: they used to be dropped on
            // this path).
            let persisted_note = if salvaged.audio_preserved {
                note
            } else {
                device_stop_note(&note, salvaged.record.acknowledged_samples)
            };
            if let Err(_e) = self.store.mark_interrupted(&salvaged.record, &persisted_note) {
                // Surface, don't swallow: the take must not vanish without
                // a trace (still legal from Recording; after the fatal
                // error below it would not be).
                self.emit(
                    Event::CaptureError {
                        code: "persist_interrupted_failed".into(),
                        fatal: false,
                    },
                    &corr,
                );
            }
        }
        self.emit(
            Event::CaptureError {
                code: "device_stream_lost".into(),
                fatal: true,
            },
            &corr,
        );
        if let Some(salvaged) = salvaged {
            self.register(salvaged.record);
        }
        // The machine is Interrupted, but the audio route is not the
        // device's to keep: release the freeze so the context cycle can
        // run again for whatever follows (recovery or a restart).
        let _ = self.freezer.take_completed(&corr);
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

    fn register(&self, record: TakeRecord) {
        self.registry
            .lock()
            .expect("take registry lock")
            .insert(record.id.clone(), Arc::new(record));
    }
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
