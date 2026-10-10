//! The engine manager: the thread-safe, UI-free public API over the
//! whole engine subsystem (#362 step 3/4, #363).
//!
//! Everything that can block — backend probing, sidecar startup, model
//! downloads, switch orchestration — runs on the supervisor thread and
//! worker threads it spawns; the public methods only send commands and
//! read the shared snapshot. The gpui app (wave C) consumes
//! [`EngineManager`], renders [`EngineSnapshot`], and holds
//! [`EngineLease`] values for takes, so a switch mid-take finishes the
//! take on the engine it started with.
//!
//! Model switches (#363) preserve the one-model-per-process server
//! contract app-side: the incoming model is loaded on a second loopback
//! sidecar, the endpoint cuts over atomically, and the outgoing engine
//! drains (stops when its last lease drops, 15 min hard cap). The memory
//! policy (`memory::swap_plan`) decides rolling vs. drain vs. refuse
//! before anything is spawned.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crate::engine::catalog::CatalogEntry;
use crate::engine::download::{
    delete_model_files, download_model, scan_install, verify_placed_file, DownloadError,
};
use crate::engine::memory::{available_memory, estimate_resident, process_rss, swap_plan};
use crate::engine::probe::{try_select_backend, BackendSelection};
use crate::engine::registry::{
    acquire_lease_marker, clear_engine_leases, foreign_leases, is_retired, remove_registration,
    retire_engine, try_spawn_lock, unretire_engine, write_registration, EngineKey, LeaseMarker,
    LockOutcome, MarkerOutcome, SidecarRegistration, SpawnLock,
};
use crate::engine::sidecar::{HealthSnapshot, LoopbackHttp, ReadyError, ReadyStage, Sidecar};
use crate::engine::{Backend, EngineFailure, SwapPlan, SWAP_MARGIN_BYTES};

/// Crash-loop policy: this many crashes within this window stop the
/// engine until the user retries (#362 step 4).
pub const CRASH_LOOP_THRESHOLD: usize = 5;
/// The crash-loop counting window.
pub const CRASH_LOOP_WINDOW: Duration = Duration::from_secs(5 * 60);
/// Default restart backoff schedule (seconds), capped at the last entry.
pub const DEFAULT_BACKOFF: [Duration; 6] = [
    Duration::from_secs(1),
    Duration::from_secs(2),
    Duration::from_secs(4),
    Duration::from_secs(8),
    Duration::from_secs(16),
    Duration::from_secs(30),
];
/// A draining engine stops when its last lease drops — or after this.
pub const DRAIN_HARD_CAP: Duration = Duration::from_secs(15 * 60);
/// Attached (not owned) sidecars are polled on this cadence; two
/// consecutive failures mean the sidecar is gone and we take over.
const ATTACH_POLL: Duration = Duration::from_secs(2);
const ATTACH_FAILURES_BEFORE_TAKEOVER: u32 = 2;
/// The peak-RSS sampler cadence during a switch (#363 step 4).
const RSS_SAMPLE: Duration = Duration::from_millis(200);
/// An unanswered "finish the current take, then switch?" question
/// cancels the switch after this long.
pub const DRAIN_DECISION_CAP: Duration = Duration::from_secs(5 * 60);

/// Configuration for one [`EngineManager`]. Paths are injectable so
/// tests (and alternative hosts) never touch the real user directories.
#[derive(Clone, Debug)]
pub struct EngineConfig {
    /// `None` = discover (env, exe dir, macOS Resources).
    pub engine_dir: Option<PathBuf>,
    /// Where verified model files live.
    pub models_dir: PathBuf,
    /// Registry, spawn lock, and `logs/engine.log` live here.
    pub state_dir: PathBuf,
    /// The selectable models.
    pub catalog: Vec<CatalogEntry>,
    /// `Some(Cpu)` is the app's "Use CPU engine" action: Vulkan is
    /// skipped during selection.
    pub backend_override: Option<Backend>,
    /// Vulkan ICD directories for the pre-check (tests).
    pub icd_dirs: Option<Vec<PathBuf>>,
    /// Memory reading override (tests): `Some(reading)` replaces
    /// `memory::available_memory`; `Some(None)` forces the Unknown path.
    pub available_memory_override: Option<Option<u64>>,
    /// Restart backoff schedule (tests use short values); `None` uses
    /// [`DEFAULT_BACKOFF`].
    pub backoff_schedule: Option<Vec<Duration>>,
}

impl EngineConfig {
    /// The default layout under the platform data dir:
    /// `<data>/starling-gpui/models` and `<data>/starling-gpui/engine`.
    pub fn default_paths() -> Result<EngineConfig, EngineError> {
        let base = dirs::data_dir().ok_or(EngineError::DataDirUnavailable)?;
        Ok(EngineConfig {
            engine_dir: None,
            models_dir: base.join("starling-gpui/models"),
            state_dir: base.join("starling-gpui/engine"),
            catalog: crate::engine::catalog::default_catalog(),
            backend_override: None,
            icd_dirs: None,
            available_memory_override: None,
            backoff_schedule: None,
        })
    }

    fn backoff(&self, crashes_so_far: usize) -> Duration {
        let schedule = self.backoff_schedule.as_deref().unwrap_or(&DEFAULT_BACKOFF);
        backoff_delay(schedule, crashes_so_far)
    }
}

/// Errors for direct manager operations (not engine lifecycle — those
/// live in the snapshot as [`EngineFailure`]).
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum EngineError {
    #[error("could not resolve the user data directory; set XDG_DATA_HOME or HOME")]
    DataDirUnavailable,
    #[error("unknown model {id}")]
    UnknownModel { id: String },
    #[error("model {id} is currently serving; switch to another model before deleting it")]
    ModelActive { id: String },
    #[error("a switch to {id} is running; cancel it before deleting the model")]
    ModelSwitching { id: String },
    #[error("model {id} is downloading; cancel the download first")]
    ModelDownloading { id: String },
    #[error("{0}")]
    Io(String),
}

/// Installation state of one catalog entry, shown per model.
#[derive(Clone, Debug, PartialEq)]
pub enum InstallState {
    NotInstalled,
    Downloading {
        done: u64,
        total: u64,
    },
    Verifying,
    /// Present but not verified yet (e.g. hand-placed); activation
    /// verifies first and never deletes the user's file on mismatch.
    NeedsVerification,
    Installed,
    Failed(String),
}

/// The manager's lifecycle phase.
#[derive(Clone, Debug, PartialEq)]
pub enum EnginePhase {
    /// Probing/selecting the bundled backend.
    SelectingBackend,
    /// No model selected (or the persisted one is not installed).
    NoModel,
    Starting,
    Loading,
    Warming,
    Ready,
    /// The active engine crashed; restarting with backoff.
    Restarting {
        attempt: u32,
        retry_in: Duration,
    },
    Failed(EngineFailure),
}

/// Stages of an in-flight model switch.
#[derive(Clone, Debug, PartialEq)]
pub enum SwitchStage {
    Downloading {
        done: u64,
        total: u64,
    },
    Verifying,
    Loading,
    Warming,
    /// Making the incoming engine active; new leases point at it.
    CuttingOver,
    /// Stopping the old engine after its leases dropped (drain swap).
    Draining,
    /// Waiting for the open take to finish before stopping the old
    /// engine (drain swap, confirmed by the user).
    WaitingForTake,
}

/// A pending memory decision surfaced to the user.
#[derive(Clone, Debug, PartialEq)]
pub enum SwapDecision {
    /// Both models fit only after the outgoing one is unloaded; the app
    /// asks "finish the current take, then switch?".
    NeedsDrain { needed: u64, available: u64 },
    /// Not enough memory even after draining; nothing changed.
    Refused { needed: u64, available: u64 },
}

/// How a switch was carried out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SwapMode {
    /// Both models were briefly resident.
    Rolling,
    /// The old engine was stopped before the new one loaded.
    Drain,
}

/// Measurements recorded when a switch completes (#363 step 4).
#[derive(Clone, Debug)]
pub struct SwitchReport {
    pub from: Option<String>,
    pub to: String,
    /// From `activate` to cutover.
    pub duration: Duration,
    /// Max over the switch of the summed RSS of all owned sidecars
    /// (sampled every 200 ms); `None` when no reading was possible.
    pub peak_rss_bytes: Option<u64>,
    pub mode: SwapMode,
}

/// The chosen backend, its version, and (after load) the device the
/// engine actually runs on.
#[derive(Clone, Debug)]
pub struct BackendSelectionView {
    pub backend: Backend,
    pub version: String,
    /// From `/health.backend`: the compile-time family before load, the
    /// device (`Vulkan0`, `CPU`, ...) after.
    pub device: Option<String>,
}

/// The active engine as seen by the app.
#[derive(Clone, Debug)]
pub struct ActiveEngineView {
    pub model_id: String,
    pub endpoint: String,
    pub pid: u32,
    /// `false` when another app instance owns the sidecar and we
    /// attached to it.
    pub owned: bool,
    pub device: Option<String>,
}

/// One catalog entry as seen by the app.
#[derive(Clone, Debug)]
pub struct ModelView {
    pub id: String,
    pub label: String,
    pub slug: String,
    pub note: String,
    pub size_bytes: u64,
    pub recommended: bool,
    pub install: InstallState,
    pub active: bool,
}

/// An in-flight switch as seen by the app.
#[derive(Clone, Debug, PartialEq)]
pub struct SwitchView {
    pub target_model_id: String,
    pub stage: SwitchStage,
    pub started: Instant,
}

/// The complete observable state; cheap to clone, refreshed by the
/// supervisor and its workers, versioned by
/// [`EngineManager::generation`].
#[derive(Clone, Debug)]
pub struct EngineSnapshot {
    pub backend: Option<BackendSelectionView>,
    pub phase: EnginePhase,
    pub active: Option<ActiveEngineView>,
    pub switch: Option<SwitchView>,
    pub pending_decision: Option<SwapDecision>,
    pub last_switch: Option<SwitchReport>,
    pub models: Vec<ModelView>,
    pub notices: Vec<String>,
    pub last_error: Option<String>,
    /// The newest [`EngineManager::activate`] request the supervisor has
    /// taken up. From then on `switch`, `pending_decision` and
    /// `last_error` answer that request or a later intent; before, they
    /// are left over from earlier ones.
    pub activations_handled: u64,
}

/// A take's hold on the engine that serves it (#363 step 3): requests
/// use `endpoint` + `slug`; dropping the lease lets a draining engine
/// stop.
pub struct EngineLease {
    engine: Arc<ActiveEngine>,
    endpoint: String,
    slug: String,
    model_id: String,
    /// On an attached engine: makes this take visible to the owning
    /// instance, which will not stop the engine under it.
    _marker: Option<LeaseMarker>,
}

impl EngineLease {
    /// The base URL to send transcription requests to.
    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    /// The `model` form field to send.
    pub fn slug(&self) -> &str {
        &self.slug
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// The pid of the engine process this lease holds — with
    /// [`Self::endpoint`], the engine's identity (a restarted or
    /// taken-over engine differs in pid even on a reused port).
    pub fn pid(&self) -> u32 {
        self.engine.pid
    }

    /// Provenance label stored with the recognition attempt.
    pub fn provenance(&self) -> String {
        format!("engine:{}", self.model_id)
    }
}

impl Drop for EngineLease {
    fn drop(&mut self) {
        self.engine.leases.fetch_sub(1, Ordering::Release);
    }
}

/// A running engine the manager knows about: the active one, a draining
/// one, or an incoming one mid-switch.
pub(crate) struct ActiveEngine {
    pub model_id: String,
    pub slug: String,
    pub endpoint: String,
    pub pid: u32,
    pub owned: bool,
    pub sidecar: Option<Sidecar>,
    pub device: Mutex<Option<String>>,
    pub leases: AtomicUsize,
}

impl ActiveEngine {
    fn owned_engine(
        model_id: &str,
        slug: &str,
        sidecar: Sidecar,
        endpoint: String,
        device: Option<String>,
    ) -> Arc<ActiveEngine> {
        Arc::new(ActiveEngine {
            model_id: model_id.to_string(),
            slug: slug.to_string(),
            pid: sidecar.pid(),
            endpoint,
            owned: true,
            sidecar: Some(sidecar),
            device: Mutex::new(device),
            leases: AtomicUsize::new(0),
        })
    }

    fn attached_engine(
        model_id: &str,
        slug: &str,
        endpoint: String,
        pid: u32,
        device: Option<String>,
    ) -> Arc<ActiveEngine> {
        Arc::new(ActiveEngine {
            model_id: model_id.to_string(),
            slug: slug.to_string(),
            endpoint,
            pid,
            owned: false,
            sidecar: None,
            device: Mutex::new(device),
            leases: AtomicUsize::new(0),
        })
    }

    /// Stop is only meaningful for owned engines; an attached sidecar
    /// belongs to another instance and is never stopped by us.
    fn stop(&self) {
        if self.owned {
            if let Some(sidecar) = &self.sidecar {
                sidecar.stop();
            }
        }
    }

    fn set_device(&self, device: String) {
        if let Ok(mut slot) = self.device.lock() {
            *slot = Some(device);
        }
    }

    fn port(&self) -> u16 {
        self.endpoint
            .rsplit(':')
            .next()
            .and_then(|port| port.parse::<u16>().ok())
            .unwrap_or(0)
    }

    /// The cross-instance identity used by lease markers.
    fn key(&self) -> EngineKey {
        EngineKey {
            pid: self.pid,
            port: self.port(),
        }
    }
}

/// Internal mutable state, always accessed through one mutex; mutations
/// bump the generation.
pub(crate) struct SharedState {
    pub generation: u64,
    pub phase: EnginePhase,
    pub selection: Option<BackendSelection>,
    pub backend_view: Option<BackendSelectionView>,
    pub active: Option<Arc<ActiveEngine>>,
    pub draining: Vec<Arc<ActiveEngine>>,
    /// The incoming engine of a running switch (sampled for RSS, stopped
    /// at shutdown).
    pub switch_incoming: Option<Arc<ActiveEngine>>,
    /// The pid of an incoming sidecar that is still loading or warming
    /// (not yet an `ActiveEngine`): loading is where a switch peaks, so the
    /// RSS sampler must see it from spawn on (#363 step 4).
    pub incoming_pid: Option<u32>,
    pub switch: Option<SwitchState>,
    pub pending_decision: Option<SwapDecision>,
    pub last_switch: Option<SwitchReport>,
    pub install: HashMap<String, InstallState>,
    /// Cancel flags of running downloads (background or switch-owned).
    pub downloads: HashMap<String, Arc<AtomicBool>>,
    /// Models whose files `delete_model` is removing; downloads and
    /// activations of them wait until it is done.
    pub deleting: HashSet<String>,
    pub notices: Vec<String>,
    pub last_error: Option<String>,
    /// The id the next `activate` request gets (ids start at 1).
    pub next_activation: u64,
    /// See [`EngineSnapshot::activations_handled`].
    pub activations_handled: u64,
    pub backend_override: Option<Backend>,
    /// Recent crashes (time, stderr tail) for the loop policy.
    pub crashes: Vec<(Instant, String)>,
    /// The last model that was (or was being made) active — the retry
    /// target after a crash loop.
    pub last_active_model: Option<String>,
    pub stopped: bool,
    /// Set when the current user intent (activate, backend change,
    /// retry) is replaced by a newer one, or at shutdown. Every worker
    /// holds the flag of the intent it serves; a set flag means it must
    /// not install its engine.
    pub superseded: Arc<AtomicBool>,
}

impl SharedState {
    fn new(backend_override: Option<Backend>) -> SharedState {
        SharedState {
            generation: 0,
            phase: EnginePhase::SelectingBackend,
            selection: None,
            backend_view: None,
            active: None,
            draining: Vec::new(),
            switch_incoming: None,
            incoming_pid: None,
            switch: None,
            pending_decision: None,
            last_switch: None,
            install: HashMap::new(),
            downloads: HashMap::new(),
            deleting: HashSet::new(),
            notices: Vec::new(),
            last_error: None,
            next_activation: 1,
            activations_handled: 0,
            backend_override,
            crashes: Vec::new(),
            last_active_model: None,
            stopped: false,
            superseded: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Starts a new intent: workers of the previous one become stale.
    fn supersede(&mut self) -> Arc<AtomicBool> {
        self.superseded.store(true, Ordering::Release);
        self.superseded = Arc::new(AtomicBool::new(false));
        Arc::clone(&self.superseded)
    }

    fn bump(&mut self) {
        self.generation += 1;
    }

    fn set_phase(&mut self, phase: EnginePhase) {
        self.phase = phase;
        self.bump();
    }

    fn add_notice(&mut self, notice: String) {
        if !self.notices.contains(&notice) {
            self.notices.push(notice);
        }
        self.bump();
    }

    fn set_install(&mut self, id: &str, install: InstallState) {
        self.install.insert(id.to_string(), install);
        self.bump();
    }

    fn record_crash(&mut self, stderr_tail: String) {
        let now = Instant::now();
        self.crashes
            .retain(|(at, _)| now.duration_since(*at) < CRASH_LOOP_WINDOW);
        self.crashes.push((now, stderr_tail));
        self.bump();
    }

    fn recent_crashes(&self) -> usize {
        self.crashes.len()
    }

    fn set_last_error(&mut self, message: String) {
        self.last_error = Some(message);
        self.bump();
    }
}

/// The manager-side view of a running switch (supervisor-owned; switch
/// workers read/mutate it under the shared lock with token guards).
pub(crate) struct SwitchState {
    pub token: u64,
    pub target: String,
    pub stage: SwitchStage,
    pub started: Instant,
    pub cancel: Arc<AtomicBool>,
    pub confirm: Arc<AtomicBool>,
}

enum Command {
    Activate { model_id: String, request: u64 },
    Download(String),
    CancelDownload(String),
    ConfirmDrainSwap,
    CancelSwitch,
    Retry,
    SetBackendOverride(Option<Backend>),
    Shutdown,
}

struct Inner {
    cmd_tx: mpsc::Sender<Command>,
    state: Arc<Mutex<SharedState>>,
    catalog: Arc<Vec<CatalogEntry>>,
    shutdown: Arc<AtomicBool>,
    supervisor: Mutex<Option<JoinHandle<()>>>,
    config: EngineConfig,
}

impl Inner {
    fn lock(&self) -> MutexGuard<'_, SharedState> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn entry(&self, id: &str) -> Option<CatalogEntry> {
        self.catalog.iter().find(|entry| entry.id == id).cloned()
    }
}

/// The public handle. Clone freely (UI, worker threads); call
/// [`EngineManager::shutdown`] once when the app quits.
#[derive(Clone)]
pub struct EngineManager {
    inner: Arc<Inner>,
}

impl EngineManager {
    /// Starts the supervisor thread and returns immediately. Backend
    /// selection runs on that thread — never on the caller's.
    /// `active_model` is the persisted model id (the app owns persisting
    /// it; it reads it back from `snapshot().active`).
    pub fn start(config: EngineConfig, active_model: Option<String>) -> EngineManager {
        let (cmd_tx, cmd_rx) = mpsc::channel::<Command>();
        let inner = Arc::new(Inner {
            cmd_tx,
            state: Arc::new(Mutex::new(SharedState::new(config.backend_override))),
            catalog: Arc::new(config.catalog.clone()),
            shutdown: Arc::new(AtomicBool::new(false)),
            supervisor: Mutex::new(None),
            config: config.clone(),
        });
        let supervisor = Supervisor {
            rx: cmd_rx,
            inner: Arc::clone(&inner),
            config,
            http: LoopbackHttp::new(),
            switch: None,
            downloads: HashMap::new(),
            pending_restart: None,
            attach_failures: 0,
            next_attach_poll: Instant::now(),
            switch_token: 0,
        };
        match std::thread::Builder::new()
            .name("starling-engine-supervisor".to_string())
            .spawn(move || supervisor.run(active_model))
        {
            Ok(thread) => {
                *inner.supervisor.lock().unwrap_or_else(|p| p.into_inner()) = Some(thread);
            }
            Err(error) => {
                // Without its supervisor the manager can do nothing; the
                // snapshot says why instead of the app crashing.
                inner
                    .lock()
                    .set_phase(EnginePhase::Failed(EngineFailure::LoadFailed(format!(
                        "could not start the engine supervisor: {error}"
                    ))));
            }
        }
        EngineManager { inner }
    }

    /// The current snapshot (cheap clone).
    pub fn snapshot(&self) -> EngineSnapshot {
        let state = self.inner.lock();
        build_snapshot(&state, &self.inner.catalog)
    }

    /// Bumps on every observable change; poll-compare friendly.
    pub fn generation(&self) -> u64 {
        self.inner.lock().generation
    }

    /// Starts (or joins) a background download of `model_id`. Progress
    /// and outcome land in `snapshot().models`.
    pub fn download(&self, model_id: &str) {
        let _ = self
            .inner
            .cmd_tx
            .send(Command::Download(model_id.to_string()));
    }

    /// Cancels a running download of `model_id` (the partial file stays
    /// for resume; nothing installed is touched).
    pub fn cancel_download(&self, model_id: &str) {
        let _ = self
            .inner
            .cmd_tx
            .send(Command::CancelDownload(model_id.to_string()));
    }

    /// Removes a model's files. Refuses while the model is active,
    /// switching, or downloading.
    pub fn delete_model(&self, model_id: &str) -> Result<(), EngineError> {
        let entry = self
            .inner
            .entry(model_id)
            .ok_or_else(|| EngineError::UnknownModel {
                id: model_id.to_string(),
            })?;
        // The checks and the `deleting` claim happen under one lock; the
        // file work then runs without it (the UI thread reads snapshots
        // through this lock). Downloads and activations of the model see
        // the claim and wait, so nothing can reinstall or load it while
        // its files disappear.
        let mut state = self.inner.lock();
        if let Some(active) = &state.active {
            if active.model_id == model_id {
                return Err(EngineError::ModelActive {
                    id: model_id.to_string(),
                });
            }
        }
        // A crashed model waiting for its restart is still the active
        // model as far as the user is concerned.
        if matches!(state.phase, EnginePhase::Restarting { .. })
            && state.last_active_model.as_deref() == Some(model_id)
        {
            return Err(EngineError::ModelActive {
                id: model_id.to_string(),
            });
        }
        if state
            .switch
            .as_ref()
            .is_some_and(|switch| switch.target == model_id)
        {
            return Err(EngineError::ModelSwitching {
                id: model_id.to_string(),
            });
        }
        if state.downloads.contains_key(model_id) {
            return Err(EngineError::ModelDownloading {
                id: model_id.to_string(),
            });
        }
        if !state.deleting.insert(model_id.to_string()) {
            // Another delete of the same model is already running.
            return Ok(());
        }
        drop(state);
        let models_dir = &self.inner.config.models_dir;
        let result = delete_model_files(models_dir, &entry).map_err(EngineError::Io);
        // Disk truth either way: a partial failure may have removed some
        // files, and the settings row must not keep offering Activate.
        let install = scan_install(models_dir, &entry);
        let mut state = self.inner.lock();
        state.deleting.remove(model_id);
        state.set_install(model_id, install);
        result
    }

    /// Download-if-needed, verify, then switch to `model_id` (see the
    /// module docs for the switch protocol). A running switch is
    /// cancelled first; rapid repeated switches leave exactly one owned
    /// sidecar. Returns the request's id, which
    /// [`EngineSnapshot::activations_handled`] reaches once the switch is
    /// under way.
    pub fn activate(&self, model_id: &str) -> u64 {
        // Ids are handed out and sent under the state lock, so the
        // supervisor takes requests up in id order.
        let mut state = self.inner.lock();
        let request = state.next_activation;
        state.next_activation += 1;
        let _ = self.inner.cmd_tx.send(Command::Activate {
            model_id: model_id.to_string(),
            request,
        });
        request
    }

    /// Answers a pending [`SwapDecision::NeedsDrain`].
    pub fn confirm_drain_swap(&self) {
        let _ = self.inner.cmd_tx.send(Command::ConfirmDrainSwap);
    }

    /// Cancels a running switch (the old model keeps serving).
    pub fn cancel_switch(&self) {
        let _ = self.inner.cmd_tx.send(Command::CancelSwitch);
    }

    /// Clears `Failed`/`CrashLoop` and retries the last model.
    pub fn retry(&self) {
        let _ = self.inner.cmd_tx.send(Command::Retry);
    }

    /// Re-selects the backend and restarts the active model on the new
    /// engine (the app's CPU/auto toggle). In-flight leases drain.
    pub fn set_backend_override(&self, backend: Option<Backend>) {
        let _ = self.inner.cmd_tx.send(Command::SetBackendOverride(backend));
    }

    /// A take's hold on the active engine; `None` when no engine is
    /// ready.
    pub fn lease(&self) -> Option<EngineLease> {
        let state = self.inner.lock();
        if state.phase != EnginePhase::Ready {
            return None;
        }
        let engine = state.active.as_ref()?;
        // A take on another instance's engine is announced to that
        // owner; one that is retiring the engine gets no new takes.
        let marker = if engine.owned {
            None
        } else {
            match acquire_lease_marker(&self.inner.config.state_dir, engine.key()) {
                MarkerOutcome::Held(marker) => Some(marker),
                MarkerOutcome::Retired => return None,
                MarkerOutcome::Unavailable => None,
            }
        };
        engine.leases.fetch_add(1, Ordering::Acquire);
        Some(EngineLease {
            engine: Arc::clone(engine),
            endpoint: engine.endpoint.clone(),
            slug: engine.slug.clone(),
            model_id: engine.model_id.clone(),
            _marker: marker,
        })
    }

    /// Stops every owned sidecar (blocking, at most a few seconds) and
    /// releases the registry if it names us. Idempotent; the
    /// `--parent-pid` backstop covers a crashing app.
    pub fn shutdown(&self) {
        self.inner.shutdown.store(true, Ordering::Release);
        // Cancel whatever is in flight directly — the supervisor may be
        // mid-probe and unreachable through the command channel.
        {
            let mut state = self.inner.lock();
            state.stopped = true;
            state.superseded.store(true, Ordering::Release);
            if let Some(switch) = &state.switch {
                switch.cancel.store(true, Ordering::Release);
            }
            for (_, cancel) in state.downloads.drain() {
                cancel.store(true, Ordering::Release);
            }
            state.bump();
        }
        let _ = self.inner.cmd_tx.send(Command::Shutdown);
        // Give the supervisor a moment to wind down by itself.
        if let Some(handle) = self
            .inner
            .supervisor
            .lock()
            .ok()
            .and_then(|mut slot| slot.take())
        {
            let deadline = Instant::now() + Duration::from_millis(1500);
            while !handle.is_finished() && Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(25));
            }
            if handle.is_finished() {
                let _ = handle.join();
            }
        }
        // Belt and braces: stop every engine we own, whatever the
        // supervisor was doing.
        let engines = collect_engines(&self.inner.lock());
        for engine in engines {
            engine.stop();
        }
        remove_registration(&self.inner.config.state_dir, std::process::id());
    }
}

/// Collects every engine handle (active, draining, incoming) for
/// shutdown stops and RSS sampling.
fn collect_engines(state: &SharedState) -> Vec<Arc<ActiveEngine>> {
    let mut engines = Vec::new();
    if let Some(active) = &state.active {
        engines.push(Arc::clone(active));
    }
    for engine in &state.draining {
        engines.push(Arc::clone(engine));
    }
    if let Some(incoming) = &state.switch_incoming {
        engines.push(Arc::clone(incoming));
    }
    engines
}

fn build_snapshot(state: &SharedState, catalog: &[CatalogEntry]) -> EngineSnapshot {
    EngineSnapshot {
        backend: state.backend_view.clone(),
        phase: state.phase.clone(),
        active: state.active.as_ref().map(|engine| ActiveEngineView {
            model_id: engine.model_id.clone(),
            endpoint: engine.endpoint.clone(),
            pid: engine.pid,
            owned: engine.owned,
            device: engine.device.lock().ok().and_then(|device| device.clone()),
        }),
        switch: state.switch.as_ref().map(|switch| SwitchView {
            target_model_id: switch.target.clone(),
            stage: switch.stage.clone(),
            started: switch.started,
        }),
        pending_decision: state.pending_decision.clone(),
        last_switch: state.last_switch.clone(),
        models: catalog
            .iter()
            .map(|entry| ModelView {
                id: entry.id.clone(),
                label: entry.label.clone(),
                slug: entry.slug.clone(),
                note: entry.note.clone(),
                size_bytes: entry.size_bytes,
                recommended: entry.recommended,
                install: state
                    .install
                    .get(&entry.id)
                    .cloned()
                    .unwrap_or(InstallState::NotInstalled),
                active: state
                    .active
                    .as_ref()
                    .is_some_and(|engine| engine.model_id == entry.id),
            })
            .collect(),
        notices: state.notices.clone(),
        last_error: state.last_error.clone(),
        activations_handled: state.activations_handled,
    }
}

// ---------------------------------------------------------------------------
// Supervisor
// ---------------------------------------------------------------------------

struct SwitchHandle {
    token: u64,
    cancel: Arc<AtomicBool>,
    thread: JoinHandle<()>,
}

struct DownloadHandle {
    #[allow(dead_code)]
    cancel: Arc<AtomicBool>,
    thread: JoinHandle<()>,
}

struct RestartState {
    model_id: String,
    at: Instant,
}

/// Attach information returned by a successful registry attach.
struct Attached {
    endpoint: String,
    pid: u32,
    device: Option<String>,
}

struct Supervisor {
    rx: mpsc::Receiver<Command>,
    inner: Arc<Inner>,
    config: EngineConfig,
    http: LoopbackHttp,
    switch: Option<SwitchHandle>,
    downloads: HashMap<String, DownloadHandle>,
    pending_restart: Option<RestartState>,
    attach_failures: u32,
    next_attach_poll: Instant,
    switch_token: u64,
}

impl Supervisor {
    fn run(mut self, active_model: Option<String>) {
        self.startup(active_model);
        loop {
            if self.inner.shutdown.load(Ordering::Acquire) {
                break;
            }
            match self.rx.recv_timeout(Duration::from_millis(100)) {
                Ok(Command::Shutdown) => break,
                Ok(command) => self.handle(command),
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            }
            if self.inner.shutdown.load(Ordering::Acquire) {
                break;
            }
            self.tick();
        }
        self.cleanup();
    }

    /// Initial selection + startup of the persisted model. Runs on the
    /// supervisor thread; commands queue up meanwhile.
    fn startup(&mut self, active_model: Option<String>) {
        {
            let mut state = self.inner.lock();
            for entry in self.inner.catalog.iter() {
                let install = scan_install(&self.config.models_dir, entry);
                state.install.insert(entry.id.clone(), install);
            }
            state.bump();
        }
        let override_backend = self.inner.lock().backend_override;
        match try_select_backend(
            self.config.engine_dir.as_deref(),
            override_backend,
            self.config.icd_dirs.as_deref(),
        ) {
            Ok(selection) => apply_selection(&self.inner, selection),
            Err(failure) => {
                self.inner.lock().set_phase(EnginePhase::Failed(failure));
                return;
            }
        }

        let Some(model_id) = active_model else {
            self.inner.lock().set_phase(EnginePhase::NoModel);
            return;
        };
        let installed = matches!(
            self.inner.lock().install.get(&model_id),
            Some(InstallState::Installed) | Some(InstallState::NeedsVerification)
        );
        if installed {
            self.launch_model(&model_id);
        } else {
            // The persisted model is gone (or was never finished): the
            // app shows the picker rather than failing silently.
            let mut state = self.inner.lock();
            state.last_active_model = Some(model_id);
            state.set_phase(EnginePhase::NoModel);
        }
    }

    fn handle(&mut self, command: Command) {
        match command {
            Command::Shutdown => {}
            Command::Activate { model_id, request } => {
                if self.inner.shutdown.load(Ordering::Acquire) {
                    return;
                }
                let superseded = self.begin_intent();
                self.start_switch(model_id, SwitchKind::Activate, superseded, Some(request));
            }
            Command::Download(model_id) => self.start_download(model_id),
            Command::CancelDownload(model_id) => {
                let state = self.inner.lock();
                if let Some(cancel) = state.downloads.get(&model_id) {
                    cancel.store(true, Ordering::Release);
                }
                drop(state);
                if let Some(handle) = self.downloads.get(&model_id) {
                    handle.cancel.store(true, Ordering::Release);
                }
            }
            Command::ConfirmDrainSwap => {
                let confirm = self
                    .inner
                    .lock()
                    .switch
                    .as_ref()
                    .map(|switch| Arc::clone(&switch.confirm));
                if let Some(confirm) = confirm {
                    confirm.store(true, Ordering::Release);
                }
            }
            Command::CancelSwitch => {
                self.cancel_current_switch();
                let mut state = self.inner.lock();
                state.pending_decision = None;
                state.bump();
            }
            Command::Retry => {
                self.begin_intent();
                let model_id = {
                    let mut state = self.inner.lock();
                    state.crashes.clear();
                    state.pending_decision = None;
                    state.bump();
                    state.last_active_model.clone()
                };
                let installed = matches!(
                    self.inner
                        .lock()
                        .install
                        .get(&model_id.clone().unwrap_or_default()),
                    Some(InstallState::Installed) | Some(InstallState::NeedsVerification)
                );
                match model_id.filter(|_| installed) {
                    Some(model_id) => {
                        self.pending_restart = None;
                        self.launch_model(&model_id);
                    }
                    None => {
                        self.inner.lock().set_phase(EnginePhase::NoModel);
                    }
                }
            }
            Command::SetBackendOverride(backend) => {
                let superseded = self.begin_intent();
                self.inner.lock().backend_override = backend;
                // Selection runs here, not on a worker: commands queue
                // behind it, so a later activation always sees (and
                // spawns from) the new backend, and no second worker can
                // race the reload below.
                match try_select_backend(
                    self.config.engine_dir.as_deref(),
                    backend,
                    self.config.icd_dirs.as_deref(),
                ) {
                    Err(failure) => {
                        // The old engine keeps serving; say why the toggle
                        // did not take effect.
                        self.inner.lock().set_last_error(failure.to_string());
                    }
                    Ok(selection) => {
                        apply_selection(&self.inner, selection);
                        // Restart the active model on the new engine; the
                        // old one drains (leases are honored).
                        let active = self
                            .inner
                            .lock()
                            .active
                            .as_ref()
                            .map(|engine| engine.model_id.clone());
                        if let Some(model_id) = active {
                            self.start_switch(model_id, SwitchKind::Reload, superseded, None);
                        }
                    }
                }
            }
        }
    }

    /// Starts a new user intent (activate, backend change, retry): the
    /// previous intent's workers become stale and the running switch is
    /// cancelled. Superseding comes first: a worker restoring the previous
    /// engine only listens to that flag, and must stop before the new
    /// intent acts. Returns the new intent's flag.
    fn begin_intent(&mut self) -> Arc<AtomicBool> {
        let superseded = self.inner.lock().supersede();
        self.cancel_current_switch();
        superseded
    }

    /// Registers and spawns a switch worker for the intent begun with
    /// [`Supervisor::begin_intent`]. `activation` is the id of the
    /// `activate` request it serves, recorded in the same update that
    /// clears the previous switch's decision and error.
    fn start_switch(
        &mut self,
        model_id: String,
        kind: SwitchKind,
        superseded: Arc<AtomicBool>,
        activation: Option<u64>,
    ) {
        self.switch_token += 1;
        let token = self.switch_token;
        let cancel = Arc::new(AtomicBool::new(false));
        let confirm = Arc::new(AtomicBool::new(false));
        {
            let mut state = self.inner.lock();
            state.pending_decision = None;
            state.last_error = None;
            if let Some(request) = activation {
                state.activations_handled = request;
            }
            state.switch = Some(SwitchState {
                token,
                target: model_id.clone(),
                stage: SwitchStage::Loading,
                started: Instant::now(),
                cancel: Arc::clone(&cancel),
                confirm: Arc::clone(&confirm),
            });
            state.bump();
        }
        self.spawn_switch_worker(token, model_id, kind, cancel, confirm, superseded);
    }

    /// Cancels the running switch (if any) and waits briefly for its
    /// worker to kill its incoming engine. A worker that outlives the
    /// wait is harmless: `cut_over` checks its cancel flag under the
    /// state lock, so it can no longer install its engine.
    fn cancel_current_switch(&mut self) {
        if let Some(handle) = self.switch.take() {
            handle.cancel.store(true, Ordering::Release);
            let deadline = Instant::now() + Duration::from_secs(5);
            while !handle.thread.is_finished() && Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(25));
            }
            if !handle.thread.is_finished() {
                // Still winding down, e.g. restoring the previous engine
                // after a cancelled drain swap: it clears its own entry
                // when done, and stays reachable for a later cancel.
                self.switch = Some(handle);
                return;
            }
            let _ = handle.thread.join();
            let mut state = self.inner.lock();
            if state
                .switch
                .as_ref()
                .is_some_and(|switch| switch.token == handle.token)
            {
                state.switch = None;
                state.bump();
            }
        }
    }

    fn tick(&mut self) {
        // Active engine supervision.
        let active = self.inner.lock().active.as_ref().map(Arc::clone);
        if let Some(active) = active {
            if active.owned {
                if let Some(sidecar) = &active.sidecar {
                    if let Some(status) = sidecar.exited_status() {
                        self.on_active_crash(&active, status);
                    }
                }
            } else if Instant::now() >= self.next_attach_poll {
                self.next_attach_poll = Instant::now() + ATTACH_POLL;
                self.poll_attached(&active);
            }
        }

        // Due restarts. A running switch decides what serves next: the
        // restart waits for it, and is dropped once the switch (or anyone
        // else) installed an engine — it must never revert a completed
        // switch.
        let due = self
            .pending_restart
            .as_ref()
            .filter(|restart| restart.at <= Instant::now())
            .map(|restart| restart.model_id.clone());
        if let Some(model_id) = due {
            let (switching, has_active) = {
                let state = self.inner.lock();
                (state.switch.is_some(), state.active.is_some())
            };
            if has_active {
                self.pending_restart = None;
            } else if !switching {
                self.pending_restart = None;
                self.launch_model(&model_id);
            }
        }

        // Join finished download threads (their state cleanup is done by
        // the workers themselves).
        let finished: Vec<String> = self
            .downloads
            .iter()
            .filter(|(_, handle)| handle.thread.is_finished())
            .map(|(id, _)| id.clone())
            .collect();
        for id in finished {
            if let Some(handle) = self.downloads.remove(&id) {
                let _ = handle.thread.join();
            }
        }
    }

    fn on_active_crash(&mut self, active: &Arc<ActiveEngine>, status: String) {
        let tail = active
            .sidecar
            .as_ref()
            .map(|sidecar| sidecar.stderr_tail())
            .unwrap_or(status);
        let model_id = active.model_id.clone();
        let mut state = self.inner.lock();
        if !state
            .active
            .as_ref()
            .is_some_and(|current| Arc::ptr_eq(current, active))
        {
            // A cutover replaced this engine after `tick` captured it: its
            // exit is a stale report, and the engine now serving (and its
            // phase) must stay untouched.
            return;
        }
        state.active = None;
        state.record_crash(tail.clone());
        if crash_loop_reached(&state.crashes, CRASH_LOOP_WINDOW, CRASH_LOOP_THRESHOLD) {
            state.set_phase(EnginePhase::Failed(EngineFailure::CrashLoop {
                last_stderr: tail,
            }));
            return;
        }
        let attempt = state.recent_crashes();
        let backoff = self.config.backoff(attempt);
        state.set_phase(EnginePhase::Restarting {
            attempt: attempt as u32,
            retry_in: backoff,
        });
        drop(state);
        self.pending_restart = Some(RestartState {
            model_id,
            at: Instant::now() + backoff,
        });
    }

    fn poll_attached(&mut self, active: &Arc<ActiveEngine>) {
        // The owner retiring the engine (it switched away) means no new
        // take may start on it: move on right away. Takes already open
        // hold their own lease and finish there.
        let retired = is_retired(&self.config.state_dir, active.key());
        let healthy = !retired
            && self
                .http
                .get_text(&format!("{}/health", active.endpoint))
                .ok()
                .and_then(|body| HealthSnapshot::parse(&body).ok())
                .is_some_and(|health| health.model == active.slug);
        if healthy {
            self.attach_failures = 0;
            return;
        }
        self.attach_failures += 1;
        if !retired && self.attach_failures < ATTACH_FAILURES_BEFORE_TAKEOVER {
            return;
        }
        self.attach_failures = 0;
        // The sidecar we attached to is gone: take over (spawn our own)
        // through the restart path, which defers to a running switch.
        let model_id = active.model_id.clone();
        {
            let mut state = self.inner.lock();
            if state
                .active
                .as_ref()
                .is_some_and(|current| Arc::ptr_eq(current, active))
            {
                state.active = None;
            }
            state.set_phase(EnginePhase::Starting);
        }
        self.pending_restart = Some(RestartState {
            model_id,
            at: Instant::now(),
        });
    }

    /// Brings `model_id` up: attach to a matching shared sidecar when
    /// possible, else spawn (through the spawn lock), driving readiness
    /// with crash-restart backoff. Blocks the supervisor thread.
    fn launch_model(&mut self, model_id: &str) {
        let Some(entry) = self.inner.entry(model_id) else {
            self.inner
                .lock()
                .set_phase(EnginePhase::Failed(EngineFailure::LoadFailed(format!(
                    "unknown model {model_id}"
                ))));
            return;
        };
        {
            let mut state = self.inner.lock();
            state.last_active_model = Some(model_id.to_string());
            state.bump();
        }
        // A hand-placed file is verified (and marked) before first use.
        if matches!(
            self.inner.lock().install.get(model_id),
            Some(InstallState::NeedsVerification)
        ) {
            self.inner
                .lock()
                .set_install(model_id, InstallState::Verifying);
            match verify_placed_file(&self.config.models_dir, &entry) {
                Ok(()) => self
                    .inner
                    .lock()
                    .set_install(model_id, InstallState::Installed),
                Err(message) => {
                    let mut state = self.inner.lock();
                    state.set_install(model_id, InstallState::Failed(message.clone()));
                    state.set_phase(EnginePhase::Failed(EngineFailure::LoadFailed(message)));
                    return;
                }
            }
        }

        loop {
            if self.inner.shutdown.load(Ordering::Acquire) {
                return;
            }
            // The shared launch protocol: attach to a live, warm,
            // model-matching sidecar, or win the model's spawn lock.
            let shutdown = Arc::clone(&self.inner.shutdown);
            let lock = match claim_spawn_slot(&self.inner, &self.http, &entry, &|| {
                shutdown.load(Ordering::Acquire)
            }) {
                SpawnSlot::Cancelled => return,
                SpawnSlot::Attach(attached) => {
                    self.set_active_attached(model_id, &entry.slug, attached);
                    return;
                }
                SpawnSlot::Spawn(lock) => lock,
            };

            let gguf = self.config.models_dir.join(&entry.file_name);
            let engine_path = self
                .inner
                .lock()
                .selection
                .as_ref()
                .map(|selection| selection.path.clone());
            let Some(engine_path) = engine_path else {
                release_slot(lock);
                self.inner
                    .lock()
                    .set_phase(EnginePhase::Failed(EngineFailure::NoUsableEngine {
                        rejected: vec![],
                    }));
                return;
            };
            self.inner.lock().set_phase(EnginePhase::Starting);
            let log_path = self.config.state_dir.join("logs/engine.log");
            match Sidecar::spawn(&engine_path, &entry.slug, &gguf, &log_path) {
                Err(message) => {
                    release_slot(lock);
                    self.inner
                        .lock()
                        .set_phase(EnginePhase::Failed(EngineFailure::LoadFailed(message)));
                    return;
                }
                Ok(sidecar) => {
                    let shutdown = Arc::clone(&self.inner.shutdown);
                    let inner = Arc::clone(&self.inner);
                    let phase_stage = move |stage: ReadyStage| {
                        inner.lock().set_phase(stage_phase(stage));
                    };
                    let outcome = sidecar.wait_ready(&entry.slug, Some(&shutdown), &phase_stage);
                    match outcome {
                        Ok(health) => {
                            let endpoint = sidecar
                                .endpoint()
                                .unwrap_or_else(|| "http://127.0.0.1:0".to_string());
                            let device = health.backend.clone();
                            let engine = ActiveEngine::owned_engine(
                                model_id,
                                &entry.slug,
                                sidecar,
                                endpoint,
                                Some(device.clone()),
                            );
                            {
                                let mut state = self.inner.lock();
                                if state.active.is_some() {
                                    // A switch installed its engine while
                                    // this one started: it wins.
                                    drop(state);
                                    engine.stop();
                                    release_slot(lock);
                                    return;
                                }
                                state.active = Some(Arc::clone(&engine));
                                state.set_phase(EnginePhase::Ready);
                            }
                            register_engine(&self.inner, &engine, &engine_path, &gguf);
                            release_slot(lock);
                            update_device_views(&self.inner, device.clone());
                            check_runtime_device_truth(&self.inner, &device);
                            return;
                        }
                        Err(ReadyError::Cancelled) => {
                            sidecar.stop();
                            release_slot(lock);
                            return;
                        }
                        Err(ReadyError::Crashed {
                            status: _,
                            stderr_tail,
                        }) => {
                            sidecar.stop();
                            release_slot(lock);
                            let mut state = self.inner.lock();
                            state.record_crash(stderr_tail.clone());
                            if crash_loop_reached(
                                &state.crashes,
                                CRASH_LOOP_WINDOW,
                                CRASH_LOOP_THRESHOLD,
                            ) {
                                state.set_phase(EnginePhase::Failed(EngineFailure::CrashLoop {
                                    last_stderr: stderr_tail,
                                }));
                                return;
                            }
                            let attempt = state.recent_crashes();
                            let backoff = self.config.backoff(attempt);
                            state.set_phase(EnginePhase::Restarting {
                                attempt: attempt as u32,
                                retry_in: backoff,
                            });
                            drop(state);
                            if !sleep_cancelable(backoff, &self.inner.shutdown) {
                                return;
                            }
                            continue;
                        }
                        Err(error) => {
                            sidecar.stop();
                            release_slot(lock);
                            self.inner
                                .lock()
                                .set_phase(EnginePhase::Failed(map_ready_error(error)));
                            return;
                        }
                    }
                }
            }
        }
    }

    fn set_active_attached(&mut self, model_id: &str, slug: &str, attached: Attached) {
        let engine = ActiveEngine::attached_engine(
            model_id,
            slug,
            attached.endpoint,
            attached.pid,
            attached.device,
        );
        let mut state = self.inner.lock();
        if state.active.is_some() {
            return;
        }
        state.active = Some(engine);
        state.set_phase(EnginePhase::Ready);
    }

    fn start_download(&mut self, model_id: String) {
        let Some(entry) = self.inner.entry(&model_id) else {
            self.inner
                .lock()
                .set_last_error(format!("unknown model {model_id}"));
            return;
        };
        let cancel = Arc::new(AtomicBool::new(false));
        {
            // Claim the slot under the lock, before the worker exists: a
            // second command, a switch's `ensure_installed`, or a delete
            // must see the download from this moment on.
            let mut state = self.inner.lock();
            if matches!(state.install.get(&model_id), Some(InstallState::Installed)) {
                return;
            }
            if state.downloads.contains_key(&model_id)
                || self.downloads.contains_key(&model_id)
                || state.deleting.contains(&model_id)
            {
                return;
            }
            state.downloads.insert(model_id.clone(), Arc::clone(&cancel));
            state.set_install(
                &model_id,
                InstallState::Downloading {
                    done: 0,
                    total: entry.size_bytes,
                },
            );
        }
        let Some(handle) =
            spawn_download_worker(Arc::clone(&self.inner), entry, Arc::clone(&cancel))
        else {
            return;
        };
        self.downloads.insert(
            model_id,
            DownloadHandle {
                cancel,
                thread: handle,
            },
        );
    }

    fn spawn_switch_worker(
        &mut self,
        token: u64,
        target: String,
        kind: SwitchKind,
        cancel: Arc<AtomicBool>,
        confirm: Arc<AtomicBool>,
        superseded: Arc<AtomicBool>,
    ) {
        let ctx = SwitchCtx {
            token,
            target,
            kind,
            cancel: Arc::clone(&cancel),
            confirm,
            superseded,
            spawn_lock: Mutex::new(None),
            inner: Arc::clone(&self.inner),
        };
        let inner = Arc::clone(&self.inner);
        let thread = match std::thread::Builder::new()
            .name("starling-engine-switch".to_string())
            .spawn(move || run_switch(ctx))
        {
            Ok(thread) => thread,
            Err(error) => {
                // Out of threads: nothing changed, say so.
                let mut state = inner.lock();
                if state
                    .switch
                    .as_ref()
                    .is_some_and(|switch| switch.token == token)
                {
                    state.switch = None;
                }
                state.set_last_error(format!("could not start the switch: {error}"));
                return;
            }
        };
        self.switch = Some(SwitchHandle {
            token,
            cancel,
            thread,
        });
    }

    fn cleanup(&mut self) {
        let engines = collect_engines(&self.inner.lock());
        for engine in engines {
            engine.stop();
        }
        remove_registration(&self.config.state_dir, std::process::id());
        let mut state = self.inner.lock();
        state.active = None;
        state.draining.clear();
        state.switch_incoming = None;
        state.bump();
    }
}

/// Registry-first attach: an entry answers `/health` with the right
/// slug, is loaded and warm, serves the model we want, and is not being
/// retired by its owner.
fn try_attach(inner: &Inner, http: &LoopbackHttp, entry: &CatalogEntry) -> Option<Attached> {
    let state_dir = &inner.config.state_dir;
    let registration = crate::engine::registry::read_registration(state_dir)?;
    if registration.model_id != entry.id || registration.slug != entry.slug {
        return None;
    }
    let key = EngineKey {
        pid: registration.pid,
        port: registration.port,
    };
    if is_retired(state_dir, key) {
        // Its owner is draining it; it takes no new work.
        return None;
    }
    let endpoint = format!("http://127.0.0.1:{}", registration.port);
    let body = http.get_text(&format!("{endpoint}/health")).ok()?;
    let health = HealthSnapshot::parse(&body).ok()?;
    if health.model != entry.slug || !health.loaded || !health.warm {
        return None;
    }
    Some(Attached {
        endpoint,
        pid: registration.pid,
        device: Some(health.backend),
    })
}

/// What the shared launch protocol decided.
enum SpawnSlot {
    /// Another instance serves the model: use its sidecar.
    Attach(Attached),
    /// Spawn our own. The lock (when the lock file could be created)
    /// must be held until the registry names the new sidecar.
    Spawn(Option<SpawnLock>),
    Cancelled,
}

/// The shared launch protocol (#362 step 3), used by every path that
/// brings a model up: attach to a live sidecar serving it, else take the
/// model's spawn lock. While another instance holds the lock this waits
/// for its sidecar to become attachable, or for the lock to be released
/// or go stale (holder gone, or past the stale age).
fn claim_spawn_slot(
    inner: &Inner,
    http: &LoopbackHttp,
    entry: &CatalogEntry,
    cancelled: &dyn Fn() -> bool,
) -> SpawnSlot {
    loop {
        if cancelled() {
            return SpawnSlot::Cancelled;
        }
        if let Some(attached) = try_attach(inner, http, entry) {
            return SpawnSlot::Attach(attached);
        }
        match try_spawn_lock(&inner.config.state_dir, &entry.id) {
            Ok(LockOutcome::Acquired(lock)) => {
                // The holder we waited on may have registered right
                // before releasing: attach to it rather than duplicate.
                if let Some(attached) = try_attach(inner, http, entry) {
                    lock.release();
                    return SpawnSlot::Attach(attached);
                }
                return SpawnSlot::Spawn(Some(lock));
            }
            Ok(LockOutcome::HeldElsewhere) => std::thread::sleep(Duration::from_millis(250)),
            // No lock file possible (unwritable state dir): sharing is
            // unavailable, but the model still starts.
            Err(_) => return SpawnSlot::Spawn(None),
        }
    }
}

fn release_slot(lock: Option<SpawnLock>) {
    if let Some(lock) = lock {
        lock.release();
    }
}

fn stage_phase(stage: ReadyStage) -> EnginePhase {
    match stage {
        ReadyStage::Starting => EnginePhase::Starting,
        ReadyStage::Loading => EnginePhase::Loading,
        ReadyStage::Warming => EnginePhase::Warming,
    }
}

fn map_ready_error(error: ReadyError) -> EngineFailure {
    match error {
        ReadyError::Crashed {
            status,
            stderr_tail,
        } => EngineFailure::CrashLoop {
            last_stderr: format!("exited {status}: {stderr_tail}"),
        },
        ReadyError::AnnounceTimeout => EngineFailure::AnnounceTimeout,
        ReadyError::BindFailed(message) => EngineFailure::BindFailed(message),
        ReadyError::LoadFailed(message) => EngineFailure::LoadFailed(message),
        ReadyError::Cancelled => EngineFailure::LoadFailed("start cancelled".to_string()),
        ReadyError::TimedOut(message) => EngineFailure::LoadFailed(message),
    }
}

/// Names `engine` in the shared registry. A failed write only costs
/// sharing (a second window starts its own engine), so it is a notice.
fn register_engine(
    inner: &Arc<Inner>,
    engine: &ActiveEngine,
    engine_path: &std::path::Path,
    gguf: &std::path::Path,
) {
    if let Err(error) = write_registration(
        &inner.config.state_dir,
        &registration_for(engine, engine_path, gguf),
    ) {
        inner.lock().add_notice(format!(
            "Could not record the running engine for other Starling windows ({error}); \
             each window will start its own engine."
        ));
    }
}

fn registration_for(
    engine: &ActiveEngine,
    engine_path: &std::path::Path,
    gguf: &std::path::Path,
) -> SidecarRegistration {
    SidecarRegistration {
        pid: engine.pid,
        port: engine.port(),
        slug: engine.slug.clone(),
        model_id: engine.model_id.clone(),
        gguf: gguf.display().to_string(),
        engine_path: engine_path.display().to_string(),
        owner_pid: std::process::id(),
    }
}

fn sleep_cancelable(duration: Duration, shutdown: &AtomicBool) -> bool {
    let deadline = Instant::now() + duration;
    while Instant::now() < deadline {
        if shutdown.load(Ordering::Acquire) {
            return false;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    true
}

fn apply_selection(inner: &Arc<Inner>, selection: BackendSelection) {
    let mut state = inner.lock();
    if let Some(notice) = &selection.fallback_notice {
        state.add_notice(notice.clone());
    }
    state.backend_view = Some(BackendSelectionView {
        backend: selection.chosen,
        version: selection.version.clone(),
        device: None,
    });
    state.selection = Some(selection);
    state.bump();
}

/// Mirrors the actual device into the backend and active-engine views.
fn update_device_views(inner: &Arc<Inner>, device: String) {
    let mut state = inner.lock();
    if let Some(active) = &state.active {
        active.set_device(device.clone());
    }
    if let Some(view) = &mut state.backend_view {
        view.device = Some(device);
    }
    state.bump();
}

/// After load, the health `backend` is the actual device: a Vulkan
/// engine that ended up on the CPU must surface as a notice, never
/// silently (#362).
fn check_runtime_device_truth(inner: &Arc<Inner>, device: &str) {
    let chosen = inner.lock().selection.as_ref().map(|s| s.chosen);
    if !matches!(chosen, Some(Backend::Vulkan)) {
        return;
    }
    if !device.to_ascii_lowercase().starts_with("vulkan") {
        inner.lock().add_notice(format!(
            "The Vulkan engine found no usable GPU and runs on the CPU ({device})."
        ));
    }
}

// ---------------------------------------------------------------------------
// Download worker (background downloads)
// ---------------------------------------------------------------------------

/// Runs a download the caller already claimed in `state.downloads`.
/// `None` (with the claim undone and the reason shown) when no thread
/// could be started.
fn spawn_download_worker(
    inner: Arc<Inner>,
    entry: CatalogEntry,
    cancel: Arc<AtomicBool>,
) -> Option<JoinHandle<()>> {
    let models_dir = inner.config.models_dir.clone();
    let worker_inner = Arc::clone(&inner);
    let claimed = entry.clone();
    let spawned = std::thread::Builder::new()
        .name("starling-engine-download".to_string())
        .spawn(move || {
            let inner = worker_inner;
            let progress_inner = Arc::clone(&inner);
            let progress_id = entry.id.clone();
            let result = download_model(
                &models_dir,
                &entry,
                &move |done, total| {
                    report_download_progress(&progress_inner, &progress_id, done, total)
                },
                &cancel,
            );
            let mut state = inner.lock();
            state.downloads.remove(&entry.id);
            match result {
                Ok(()) => state.set_install(&entry.id, InstallState::Installed),
                Err(DownloadError::Cancelled) => {
                    state.set_install(&entry.id, InstallState::NotInstalled)
                }
                Err(error) => {
                    let message = error.to_string();
                    state.set_install(&entry.id, InstallState::Failed(message.clone()));
                    state.set_last_error(message);
                }
            }
        });
    match spawned {
        Ok(handle) => Some(handle),
        Err(error) => {
            let mut state = inner.lock();
            state.downloads.remove(&claimed.id);
            state.set_install(
                &claimed.id,
                scan_install(&inner.config.models_dir, &claimed),
            );
            state.set_last_error(format!("could not start the download: {error}"));
            None
        }
    }
}


/// Download progress reaches both the model's install state and, when a
/// switch targets the same model, its stage.
fn report_download_progress(inner: &Arc<Inner>, id: &str, done: u64, total: u64) {
    let mut state = inner.lock();
    state.set_install(id, InstallState::Downloading { done, total });
    if let Some(switch) = state.switch.as_mut() {
        if switch.target == id {
            switch.stage = SwitchStage::Downloading { done, total };
        }
    }
    state.bump();
}

// ---------------------------------------------------------------------------
// Switch worker
// ---------------------------------------------------------------------------

/// What a switch worker was started for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SwitchKind {
    /// The user picked a model.
    Activate,
    /// The backend changed: restart the active model on the new engine.
    Reload,
}

struct SwitchCtx {
    token: u64,
    target: String,
    kind: SwitchKind,
    /// Set by the Cancel action, by a newer intent, and at shutdown.
    cancel: Arc<AtomicBool>,
    confirm: Arc<AtomicBool>,
    /// Set only when a newer intent replaced this one (or at shutdown):
    /// unlike `cancel`, a user's Cancel does not set it, so recovery
    /// after a cancel (restoring the previous engine) keys off it.
    superseded: Arc<AtomicBool>,
    /// The spawn lock of this worker's own sidecar, held from spawn until
    /// the cutover registers it (or the attempt fails).
    spawn_lock: Mutex<Option<SpawnLock>>,
    inner: Arc<Inner>,
}

impl SwitchCtx {
    fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::Acquire) || self.superseded()
    }

    fn superseded(&self) -> bool {
        self.superseded.load(Ordering::Acquire) || self.inner.shutdown.load(Ordering::Acquire)
    }

    /// The context for bringing the previous engine back: the user's
    /// Cancel is what triggered it, so only a newer intent or shutdown
    /// may stop it.
    fn recovery(&self) -> SwitchCtx {
        SwitchCtx {
            token: self.token,
            target: self.target.clone(),
            kind: self.kind,
            cancel: Arc::clone(&self.superseded),
            confirm: Arc::clone(&self.confirm),
            superseded: Arc::clone(&self.superseded),
            spawn_lock: Mutex::new(None),
            inner: Arc::clone(&self.inner),
        }
    }

    fn hold_spawn_lock(&self, lock: Option<SpawnLock>) {
        let previous = std::mem::replace(
            &mut *self
                .spawn_lock
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()),
            lock,
        );
        release_slot(previous);
    }

    fn release_spawn_lock(&self) {
        self.hold_spawn_lock(None);
    }

    fn lock(&self) -> MutexGuard<'_, SharedState> {
        self.inner.lock()
    }

    fn set_stage(&self, stage: SwitchStage) {
        let mut state = self.lock();
        if let Some(switch) = state.switch.as_mut() {
            if switch.token == self.token {
                switch.stage = stage;
            }
        }
        state.bump();
    }

    fn phase_when_no_active(&self, phase: EnginePhase) {
        let mut state = self.lock();
        if state.active.is_none() {
            state.set_phase(phase);
        }
    }

    fn set_last_error(&self, message: String) {
        self.lock().set_last_error(message);
    }

    /// Clears the switch entry if it is still ours. A pending
    /// `NeedsDrain` decision dies with its switch (there is nothing to
    /// answer anymore), but a `Refused` decision stays: it is the
    /// outcome the app must show, not a question awaiting an answer.
    fn end_switch(&self) {
        let mut state = self.lock();
        if state
            .switch
            .as_ref()
            .is_some_and(|switch| switch.token == self.token)
        {
            state.switch = None;
        }
        if matches!(
            state.pending_decision,
            Some(SwapDecision::NeedsDrain { .. })
        ) {
            state.pending_decision = None;
        }
        state.bump();
    }

    fn switch_started(&self) -> Instant {
        self.lock()
            .switch
            .as_ref()
            .map(|switch| switch.started)
            .unwrap_or_else(Instant::now)
    }
}

fn run_switch(ctx: SwitchCtx) {
    let Some(entry) = ctx.inner.entry(&ctx.target) else {
        ctx.set_last_error(format!("unknown model {}", ctx.target));
        ctx.end_switch();
        return;
    };
    if ctx.kind == SwitchKind::Reload {
        run_reload(&ctx, &entry);
        return;
    }

    // Already serving the target: nothing to do.
    {
        let state = ctx.lock();
        let trivial = state.phase == EnginePhase::Ready
            && state
                .active
                .as_ref()
                .is_some_and(|engine| engine.model_id == ctx.target);
        drop(state);
        if trivial {
            ctx.end_switch();
            return;
        }
    }

    // 1. Download (if needed) and verify the target. A failure here ends
    //    the switch; whatever is active keeps serving.
    if !ensure_installed(&ctx, &entry) {
        return;
    }

    // 2. Another instance already serving the target costs no memory and
    //    no startup: use its sidecar.
    if let Some(attached) = try_attach(&ctx.inner, &LoopbackHttp::new(), &entry) {
        let started = ctx.switch_started();
        ctx.set_stage(SwitchStage::CuttingOver);
        let engine = ActiveEngine::attached_engine(
            &entry.id,
            &entry.slug,
            attached.endpoint,
            attached.pid,
            attached.device,
        );
        let report = CutoverReport {
            mode: SwapMode::Rolling,
            started,
            from: None,
            peak_rss_bytes: None,
        };
        if !cut_over(&ctx, engine, &entry, Some(report)) {
            recover_if_idle(&ctx, None);
        }
        ctx.end_switch();
        return;
    }

    // 3. Memory policy.
    let incoming_est = estimate_resident(entry.size_bytes);
    // Only an engine this instance owns can be unloaded to make room: an
    // attached engine belongs to another instance and keeps running (and
    // using its memory) whatever this switch does.
    let outgoing_est = {
        let state = ctx.lock();
        state
            .active
            .as_ref()
            .filter(|engine| engine.owned)
            .and_then(|engine| ctx.inner.entry(&engine.model_id))
            .map(|outgoing| estimate_resident(outgoing.size_bytes))
    };
    let available = ctx
        .inner
        .config
        .available_memory_override
        .unwrap_or_else(available_memory);
    let plan = swap_plan(available, incoming_est, outgoing_est, SWAP_MARGIN_BYTES);

    match plan {
        SwapPlan::Refuse { needed, available } => {
            let mut state = ctx.lock();
            state.pending_decision = Some(SwapDecision::Refused { needed, available });
            state.bump();
            drop(state);
            ctx.end_switch();
            return;
        }
        SwapPlan::Unknown => {
            ctx.lock().add_notice(
                "No memory reading was available; the switch runs without a memory check."
                    .to_string(),
            );
        }
        SwapPlan::Rolling | SwapPlan::NeedsDrain { .. } => {}
    }

    // 4. Execute. Drain swaps stop the old engine first (with the
    //    user's confirmation); rolling swaps cut over atomically.
    if matches!(plan, SwapPlan::NeedsDrain { .. }) {
        run_drain_swap(&ctx, &entry, incoming_est, available);
    } else {
        run_rolling_swap(&ctx, &entry);
    }
}

/// Downloads/verifies the target. Returns false when the switch must
/// end (failure reported or cancelled).
fn ensure_installed(ctx: &SwitchCtx, entry: &CatalogEntry) -> bool {
    loop {
        if ctx.cancelled() {
            ctx.end_switch();
            return false;
        }
        if ctx.lock().deleting.contains(&entry.id) {
            // A delete is removing the files: wait, then see what is left.
            std::thread::sleep(Duration::from_millis(50));
            continue;
        }
        let current = {
            let state = ctx.lock();
            state
                .install
                .get(&entry.id)
                .cloned()
                .unwrap_or_else(|| scan_install(&ctx.inner.config.models_dir, entry))
        };
        match current {
            InstallState::Installed => return true,
            InstallState::NeedsVerification => {
                ctx.set_stage(SwitchStage::Verifying);
                ctx.lock().set_install(&entry.id, InstallState::Verifying);
                match verify_placed_file(&ctx.inner.config.models_dir, entry) {
                    Ok(()) => {
                        ctx.lock().set_install(&entry.id, InstallState::Installed);
                        return true;
                    }
                    Err(message) => {
                        let mut state = ctx.lock();
                        state.set_install(&entry.id, InstallState::Failed(message.clone()));
                        state.set_last_error(message);
                        drop(state);
                        ctx.end_switch();
                        return false;
                    }
                }
            }
            InstallState::Downloading { .. } | InstallState::Verifying => {
                // A background download is working on it; wait it out.
                std::thread::sleep(Duration::from_millis(100));
            }
            InstallState::NotInstalled | InstallState::Failed(_) => {
                // Claim the download slot (or wait for a background one).
                let claimed = {
                    let mut state = ctx.lock();
                    if state.downloads.contains_key(&entry.id) || state.deleting.contains(&entry.id)
                    {
                        false
                    } else {
                        state
                            .downloads
                            .insert(entry.id.clone(), Arc::clone(&ctx.cancel));
                        true
                    }
                };
                if !claimed {
                    std::thread::sleep(Duration::from_millis(100));
                    continue;
                }
                ctx.set_stage(SwitchStage::Downloading {
                    done: 0,
                    total: entry.size_bytes,
                });
                ctx.lock().set_install(
                    &entry.id,
                    InstallState::Downloading {
                        done: 0,
                        total: entry.size_bytes,
                    },
                );
                let progress_inner = Arc::clone(&ctx.inner);
                let progress_id = entry.id.clone();
                let result = download_model(
                    &ctx.inner.config.models_dir,
                    entry,
                    &move |done, total| {
                        report_download_progress(&progress_inner, &progress_id, done, total)
                    },
                    &ctx.cancel,
                );
                {
                    let mut state = ctx.lock();
                    if state
                        .downloads
                        .get(&entry.id)
                        .is_some_and(|registered| Arc::ptr_eq(registered, &ctx.cancel))
                    {
                        state.downloads.remove(&entry.id);
                    }
                    match result {
                        Ok(()) => {
                            state.set_install(&entry.id, InstallState::Installed);
                            drop(state);
                            return true;
                        }
                        Err(DownloadError::Cancelled) => {
                            state.set_install(&entry.id, InstallState::NotInstalled);
                            drop(state);
                            ctx.end_switch();
                            return false;
                        }
                        Err(error) => {
                            let message = error.to_string();
                            state.set_install(&entry.id, InstallState::Failed(message.clone()));
                            state.set_last_error(message);
                            drop(state);
                            ctx.end_switch();
                            return false;
                        }
                    }
                }
            }
        }
    }
}

enum SwitchSpawnError {
    Cancelled,
    CrashLooped,
    Failed(String),
}

/// Spawns an engine for `entry`, drives it to warm, registers it as the
/// switch's incoming engine. `retry_crashes` restarts with backoff when
/// there is no old engine to fall back on (first activation) — that is
/// the path where a crash loop must trip (#362 step 4).
///
/// `share` applies the shared launch protocol first: when another
/// instance already serves the model, its sidecar is used (an attached,
/// not owned, engine) instead of starting a duplicate. A backend reload
/// must not share — the registry may name the very engine it replaces.
fn spawn_incoming(
    ctx: &SwitchCtx,
    entry: &CatalogEntry,
    retry_crashes: bool,
    share: bool,
) -> Result<Arc<ActiveEngine>, SwitchSpawnError> {
    let http = LoopbackHttp::new();
    loop {
        if ctx.cancelled() {
            return Err(SwitchSpawnError::Cancelled);
        }
        if share {
            match claim_spawn_slot(&ctx.inner, &http, entry, &|| ctx.cancelled()) {
                SpawnSlot::Cancelled => return Err(SwitchSpawnError::Cancelled),
                SpawnSlot::Attach(attached) => {
                    return Ok(ActiveEngine::attached_engine(
                        &entry.id,
                        &entry.slug,
                        attached.endpoint,
                        attached.pid,
                        attached.device,
                    ));
                }
                SpawnSlot::Spawn(lock) => ctx.hold_spawn_lock(lock),
            }
        }
        let engine_path = ctx.lock().selection.as_ref().map(|s| s.path.clone());
        let Some(engine_path) = engine_path else {
            ctx.release_spawn_lock();
            return Err(SwitchSpawnError::Failed(
                "no usable engine was selected".to_string(),
            ));
        };
        let gguf = ctx.inner.config.models_dir.join(&entry.file_name);
        let log_path = ctx.inner.config.state_dir.join("logs/engine.log");
        let sidecar = match Sidecar::spawn(&engine_path, &entry.slug, &gguf, &log_path) {
            Ok(sidecar) => sidecar,
            Err(message) => {
                ctx.release_spawn_lock();
                return Err(SwitchSpawnError::Failed(message));
            }
        };
        ctx.lock().incoming_pid = Some(sidecar.pid());
        ctx.phase_when_no_active(EnginePhase::Starting);
        let stage_token = ctx.token;
        let stage_inner = Arc::clone(&ctx.inner);
        let stage = move |stage: ReadyStage| {
            let mut state = stage_inner.lock();
            if state.active.is_none() {
                state.set_phase(stage_phase(stage));
            }
            if let Some(switch) = state.switch.as_mut() {
                if switch.token == stage_token {
                    match stage {
                        ReadyStage::Starting => {}
                        ReadyStage::Loading => switch.stage = SwitchStage::Loading,
                        ReadyStage::Warming => switch.stage = SwitchStage::Warming,
                    }
                }
            }
        };
        let ready = sidecar.wait_ready(&entry.slug, Some(&ctx.cancel), &stage);
        {
            // Another (newer) switch may already sample its own sidecar.
            let mut state = ctx.lock();
            if state.incoming_pid == Some(sidecar.pid()) {
                state.incoming_pid = None;
            }
        }
        match ready {
            Ok(health) => {
                let endpoint = sidecar
                    .endpoint()
                    .unwrap_or_else(|| "http://127.0.0.1:0".to_string());
                let engine = ActiveEngine::owned_engine(
                    &entry.id,
                    &entry.slug,
                    sidecar,
                    endpoint,
                    Some(health.backend.clone()),
                );
                ctx.lock().switch_incoming = Some(Arc::clone(&engine));
                return Ok(engine);
            }
            Err(ReadyError::Cancelled) => {
                sidecar.stop();
                ctx.release_spawn_lock();
                return Err(SwitchSpawnError::Cancelled);
            }
            Err(error) if error.is_crash() && retry_crashes => {
                sidecar.stop();
                // The next attempt claims the slot again (another instance
                // may have brought the model up meanwhile).
                ctx.release_spawn_lock();
                let tail = crash_tail(&error);
                let mut state = ctx.lock();
                state.record_crash(tail.clone());
                if crash_loop_reached(&state.crashes, CRASH_LOOP_WINDOW, CRASH_LOOP_THRESHOLD) {
                    state.set_phase(EnginePhase::Failed(EngineFailure::CrashLoop {
                        last_stderr: tail,
                    }));
                    drop(state);
                    return Err(SwitchSpawnError::CrashLooped);
                }
                let attempt = state.recent_crashes();
                let backoff = ctx.inner.config.backoff(attempt);
                state.set_phase(EnginePhase::Restarting {
                    attempt: attempt as u32,
                    retry_in: backoff,
                });
                drop(state);
                if !sleep_cancelable_ctx(backoff, ctx) {
                    return Err(SwitchSpawnError::Cancelled);
                }
                continue;
            }
            Err(error) => {
                sidecar.stop();
                ctx.release_spawn_lock();
                return Err(SwitchSpawnError::Failed(map_ready_error(error).to_string()));
            }
        }
    }
}

fn crash_tail(error: &ReadyError) -> String {
    match error {
        ReadyError::Crashed { stderr_tail, .. } => stderr_tail.clone(),
        other => other.to_string(),
    }
}

fn sleep_cancelable_ctx(duration: Duration, ctx: &SwitchCtx) -> bool {
    let deadline = Instant::now() + duration;
    while Instant::now() < deadline {
        if ctx.cancelled() {
            return false;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    true
}

/// What a completed switch records in `last_switch`.
struct CutoverReport {
    mode: SwapMode,
    started: Instant,
    /// Overrides the report's origin (the drain path stopped the old
    /// engine before the cutover, so there is nothing left to replace).
    from: Option<String>,
    peak_rss_bytes: Option<u64>,
}

impl CutoverReport {
    fn new(mode: SwapMode, started: Instant, from: Option<String>, sampler: &RssSampler) -> Self {
        stop_sampler(sampler);
        CutoverReport {
            mode,
            started,
            from,
            peak_rss_bytes: sampler_peak(sampler),
        }
    }
}

/// The rolling path: spawn incoming while the old engine serves, cut
/// over atomically, then drain the old one (#363). Without an old
/// engine (first activation) incoming crashes are retried — that is the
/// crash-loop path.
fn run_rolling_swap(ctx: &SwitchCtx, entry: &CatalogEntry) {
    let started = ctx.switch_started();
    let has_old = ctx.lock().active.is_some();
    let sampler = start_rss_sampler(ctx);
    ctx.set_stage(SwitchStage::Loading);
    match spawn_incoming(ctx, entry, !has_old, true) {
        Ok(engine) => {
            ctx.set_stage(SwitchStage::CuttingOver);
            let report = CutoverReport::new(SwapMode::Rolling, started, None, &sampler);
            if !cut_over(ctx, engine, entry, Some(report)) {
                // Cancelled at the last moment: the old engine (if any)
                // never stopped serving.
                recover_if_idle(ctx, None);
            }
        }
        Err(SwitchSpawnError::Cancelled) => {
            stop_sampler(&sampler);
            recover_if_idle(ctx, None);
        }
        Err(SwitchSpawnError::CrashLooped) => {
            stop_sampler(&sampler);
            // The phase is already Failed(CrashLoop) from the retry loop.
        }
        Err(SwitchSpawnError::Failed(message)) => {
            stop_sampler(&sampler);
            if ctx.lock().active.is_some() {
                // The old engine keeps serving; say why the switch stopped.
                ctx.set_last_error(format!(
                    "the new engine did not start ({message}); the previous model keeps serving"
                ));
            } else {
                recover_if_idle(ctx, Some(&message));
            }
        }
    }
    ctx.end_switch();
}

/// After a rolling switch ended without installing its engine, nothing
/// may be serving: a first activation, a crashed old engine, or an
/// earlier drain swap that unloaded its engine before this switch
/// superseded it. Bring the last model back, or say plainly why nothing
/// is loaded. A superseded switch leaves this to the newer intent.
fn recover_if_idle(ctx: &SwitchCtx, failure: Option<&str>) {
    if ctx.superseded() {
        return;
    }
    let last = {
        let state = ctx.lock();
        if state.active.is_some() {
            return;
        }
        state.last_active_model.clone()
    };
    let previous = last.filter(|id| *id != ctx.target && model_installed(ctx, id));
    match previous {
        Some(previous) => restore_previous(ctx, &previous, failure),
        None => {
            let mut state = ctx.lock();
            if state.active.is_none() {
                match failure {
                    Some(message) => {
                        state.set_last_error(format!("the new engine did not start ({message})"));
                        state.set_phase(EnginePhase::Failed(EngineFailure::LoadFailed(
                            message.to_string(),
                        )));
                    }
                    None => state.set_phase(EnginePhase::NoModel),
                }
            }
        }
    }
}

fn model_installed(ctx: &SwitchCtx, id: &str) -> bool {
    matches!(
        ctx.lock().install.get(id),
        Some(InstallState::Installed) | Some(InstallState::NeedsVerification)
    )
}

fn run_drain_swap(
    ctx: &SwitchCtx,
    entry: &CatalogEntry,
    incoming_est: u64,
    available: Option<u64>,
) {
    let started = ctx.switch_started();
    // Surface the decision.
    {
        let mut state = ctx.lock();
        state.pending_decision = Some(SwapDecision::NeedsDrain {
            needed: incoming_est.saturating_add(SWAP_MARGIN_BYTES),
            available: available.unwrap_or(0),
        });
        state.bump();
    }
    // Wait for the user's answer (or cancellation). An unanswered
    // question does not pin the switch forever.
    let asked = Instant::now();
    loop {
        if ctx.cancelled() {
            ctx.end_switch();
            return;
        }
        if ctx.confirm.load(Ordering::Acquire) {
            break;
        }
        if asked.elapsed() >= DRAIN_DECISION_CAP {
            ctx.set_last_error(format!(
                "the switch to {} was not confirmed and was cancelled; the current model keeps serving",
                entry.label
            ));
            ctx.end_switch();
            return;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    {
        let mut state = ctx.lock();
        state.pending_decision = None;
        if let Some(switch) = state.switch.as_mut() {
            if switch.token == ctx.token {
                switch.stage = SwitchStage::WaitingForTake;
            }
        }
        state.bump();
    }
    // Other app instances attached to this engine may hold takes too:
    // retire it first (no new foreign take starts on it), then wait for
    // both our leases and theirs.
    let state_dir = ctx.inner.config.state_dir.clone();
    let old_key = ctx
        .lock()
        .active
        .as_ref()
        .filter(|engine| engine.owned)
        .map(|engine| engine.key());
    if let Some(key) = old_key {
        retire_engine(&state_dir, key);
    }
    // Wait for the old engine's leases to drop (hard cap). The zero check
    // and taking the engine out of `active` happen under one lock: `lease()`
    // takes the same lock, so a take cannot start on the old engine between
    // the check and the stop and then lose its engine mid-take.
    let deadline = Instant::now() + DRAIN_HARD_CAP;
    let old = loop {
        if ctx.cancelled() {
            if let Some(key) = old_key {
                unretire_engine(&state_dir, key);
            }
            ctx.end_switch();
            return;
        }
        let foreign = old_key
            .map(|key| foreign_leases(&state_dir, key))
            .unwrap_or(0);
        {
            let mut state = ctx.lock();
            let leases = state
                .active
                .as_ref()
                .map(|engine| engine.leases.load(Ordering::Acquire))
                .unwrap_or(0);
            if (leases == 0 && foreign == 0) || Instant::now() >= deadline {
                break state.active.take();
            }
        }
        std::thread::sleep(Duration::from_millis(100));
    };
    // Stop the old engine (its leases are gone).
    ctx.set_stage(SwitchStage::Draining);
    if let Some(old) = &old {
        old.stop();
    }
    if let Some(key) = old_key {
        clear_engine_leases(&state_dir, key);
    }
    let previous_id = old.as_ref().map(|engine| engine.model_id.clone());
    drop(old);
    ctx.phase_when_no_active(EnginePhase::Starting);

    let sampler = start_rss_sampler(ctx);
    ctx.set_stage(SwitchStage::Loading);
    let failure = match spawn_incoming(ctx, entry, false, true) {
        Ok(engine) => {
            ctx.set_stage(SwitchStage::CuttingOver);
            let report =
                CutoverReport::new(SwapMode::Drain, started, previous_id.clone(), &sampler);
            if cut_over(ctx, engine, entry, Some(report)) {
                return;
            }
            // Cancelled at the cutover: same as cancelled while loading.
            None
        }
        Err(error) => {
            stop_sampler(&sampler);
            match error {
                SwitchSpawnError::Cancelled => None,
                SwitchSpawnError::CrashLooped => Some("the new engine kept crashing".to_string()),
                SwitchSpawnError::Failed(message) => Some(message),
            }
        }
    };
    // The app is never left without an engine: whether the new one
    // failed or the user cancelled after the old one stopped, restart the
    // previous model. Only a newer intent (or shutdown) skips this — it
    // decides what serves next.
    if !ctx.superseded() {
        match previous_id {
            Some(previous_id) => restore_previous(ctx, &previous_id, failure.as_deref()),
            None => recover_if_idle(ctx, failure.as_deref()),
        }
    }
    ctx.end_switch();
}

/// Brings `previous_id` back up after a switch unloaded it (or it was
/// lost) and the switch did not complete. Runs under the recovery
/// context: the Cancel that led here must not cancel the recovery too.
fn restore_previous(ctx: &SwitchCtx, previous_id: &str, failure: Option<&str>) {
    let Some(entry) = ctx.inner.entry(previous_id) else {
        return;
    };
    let recovery = ctx.recovery();
    recovery.set_stage(SwitchStage::Loading);
    match spawn_incoming(&recovery, &entry, false, true) {
        Ok(engine) => {
            // Not a switch: the report keeps describing the last real one.
            if cut_over(&recovery, engine, &entry, None) {
                if let Some(failure) = failure {
                    ctx.set_last_error(format!(
                        "the new engine failed ({failure}); the previous model was restarted"
                    ));
                }
            }
        }
        Err(_) => {
            if recovery.superseded() {
                return;
            }
            let mut state = ctx.lock();
            if state.active.is_none() {
                let reason = match failure {
                    Some(failure) => format!(
                        "could not restart the previous model ({previous_id}) after the new engine failed ({failure})"
                    ),
                    None => format!(
                        "could not restart the previous model ({previous_id}) after the switch was cancelled"
                    ),
                };
                state.set_last_error(reason.clone());
                state.set_phase(EnginePhase::Failed(EngineFailure::LoadFailed(reason)));
            }
        }
    }
}

/// Restarts the active model on the newly selected backend (the CPU
/// toggle). The old engine keeps serving until the cutover and then
/// drains. A newer activation cancels this like any switch.
fn run_reload(ctx: &SwitchCtx, entry: &CatalogEntry) {
    ctx.set_stage(SwitchStage::Loading);
    match spawn_incoming(ctx, entry, false, false) {
        Ok(engine) => {
            ctx.set_stage(SwitchStage::CuttingOver);
            // Not a model switch: `last_switch` keeps reporting the user's
            // last one, not this backend move.
            cut_over(ctx, engine, entry, None);
        }
        Err(SwitchSpawnError::Cancelled) => {}
        Err(_) => {
            ctx.set_last_error(
                "the engine could not restart on the new backend; the previous engine keeps serving"
                    .to_string(),
            );
        }
    }
    ctx.end_switch();
}

/// Makes `engine` the active one: new leases point at it, the registry
/// names it, the previous engine (if any) drains. Returns `false`, with
/// `engine` stopped, when the switch was cancelled or superseded: the
/// check runs under the state lock, and a newer intent marks this one
/// superseded under the same lock, so a stale worker can never replace
/// a newer engine.
fn cut_over(
    ctx: &SwitchCtx,
    engine: Arc<ActiveEngine>,
    entry: &CatalogEntry,
    report: Option<CutoverReport>,
) -> bool {
    let engine_path = ctx.lock().selection.as_ref().map(|s| s.path.clone());
    let gguf = ctx.inner.config.models_dir.join(&entry.file_name);
    let old = {
        let mut state = ctx.lock();
        if state
            .switch_incoming
            .as_ref()
            .is_some_and(|incoming| Arc::ptr_eq(incoming, &engine))
        {
            state.switch_incoming = None;
        }
        if ctx.cancelled() {
            state.bump();
            drop(state);
            engine.stop();
            ctx.release_spawn_lock();
            return false;
        }
        let old = state.active.replace(Arc::clone(&engine));
        if let Some(old) = &old {
            state.draining.push(Arc::clone(old));
        }
        state.last_active_model = Some(entry.id.clone());
        state.set_phase(EnginePhase::Ready);
        if let Some(report) = report {
            let from = report
                .from
                .or_else(|| old.as_ref().map(|old| old.model_id.clone()));
            state.last_switch = Some(SwitchReport {
                from,
                to: entry.id.clone(),
                duration: report.started.elapsed(),
                peak_rss_bytes: report.peak_rss_bytes,
                mode: report.mode,
            });
        }
        if state
            .switch
            .as_ref()
            .is_some_and(|switch| switch.token == ctx.token)
        {
            state.switch = None;
        }
        state.bump();
        old
    };
    // Only an engine we own is ours to register; an attached one is
    // already named by its owner.
    if engine.owned {
        if let Some(path) = engine_path {
            register_engine(&ctx.inner, &engine, &path, &gguf);
        }
    }
    ctx.release_spawn_lock();
    if let Some(old) = old {
        spawn_drain_watcher(Arc::clone(&ctx.inner), old);
    }
    let device = match &engine.sidecar {
        Some(sidecar) => sidecar.health().map(|health| health.backend),
        None => engine.device.lock().ok().and_then(|device| device.clone()),
    };
    if let Some(device) = device {
        update_device_views(&ctx.inner, device.clone());
        check_runtime_device_truth(&ctx.inner, &device);
    }
    true
}

/// Watches a draining engine: stop it once its last lease drops (ours
/// and those of other instances attached to it), or at the hard cap.
/// Runs on its own thread so the supervisor never blocks a drain on it.
fn spawn_drain_watcher(inner: Arc<Inner>, engine: Arc<ActiveEngine>) {
    let fallback_inner = Arc::clone(&inner);
    let spawned = std::thread::Builder::new()
        .name("starling-engine-drain".to_string())
        .spawn(move || {
            let state_dir = inner.config.state_dir.clone();
            // Only the owner stops the engine, so only the owner retires
            // it and waits for other instances' takes.
            let key = engine.owned.then(|| engine.key());
            if let Some(key) = key {
                retire_engine(&state_dir, key);
            }
            let started = Instant::now();
            while started.elapsed() < DRAIN_HARD_CAP {
                let local = engine.leases.load(Ordering::Acquire);
                let foreign = key
                    .map(|key| foreign_leases(&state_dir, key))
                    .unwrap_or(0);
                if local == 0 && foreign == 0 {
                    break;
                }
                std::thread::sleep(Duration::from_millis(200));
            }
            engine.stop();
            if let Some(key) = key {
                clear_engine_leases(&state_dir, key);
            }
            let mut state = inner.lock();
            state
                .draining
                .retain(|draining| !Arc::ptr_eq(draining, &engine));
            state.bump();
        });
    if let Err(error) = spawned {
        // The engine stays in `draining` (its takes finish there) and is
        // stopped with everything else at shutdown.
        fallback_inner.lock().add_notice(format!(
            "The previous engine could not be scheduled to stop ({error}); it stops when Starling quits."
        ));
    }
}

// ---------------------------------------------------------------------------
// RSS sampling
// ---------------------------------------------------------------------------

struct RssSampler {
    max: Arc<AtomicU64>,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

fn start_rss_sampler(ctx: &SwitchCtx) -> RssSampler {
    let max = Arc::new(AtomicU64::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let inner = Arc::clone(&ctx.inner);
    let max_thread = Arc::clone(&max);
    let stop_thread = Arc::clone(&stop);
    // No thread, no measurement: the report then carries no peak.
    let thread = std::thread::Builder::new()
        .name("starling-engine-rss".to_string())
        .spawn(move || {
            while !stop_thread.load(Ordering::Acquire) {
                let pids: Vec<u32> = {
                    let state = inner.lock();
                    let mut pids: Vec<u32> = collect_engines(&state)
                        .iter()
                        .filter(|engine| engine.owned)
                        .map(|engine| engine.pid)
                        .collect();
                    if let Some(pid) = state.incoming_pid {
                        if !pids.contains(&pid) {
                            pids.push(pid);
                        }
                    }
                    pids
                };
                let total: u64 = pids.iter().filter_map(|pid| process_rss(*pid)).sum();
                if total > 0 {
                    max_thread.fetch_max(total, Ordering::Relaxed);
                }
                std::thread::sleep(RSS_SAMPLE);
            }
        })
        .ok();
    RssSampler { max, stop, thread }
}

fn stop_sampler(sampler: &RssSampler) {
    sampler.stop.store(true, Ordering::Release);
    if let Some(thread) = sampler.thread.as_ref() {
        // Sampling stops within one cadence; do not block the switch on
        // it longer than that.
        let deadline = Instant::now() + RSS_SAMPLE + Duration::from_millis(200);
        while !thread.is_finished() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(20));
        }
    }
}

fn sampler_peak(sampler: &RssSampler) -> Option<u64> {
    let peak = sampler.max.load(Ordering::Relaxed);
    (peak > 0).then_some(peak)
}

// ---------------------------------------------------------------------------
// Pure policy helpers
// ---------------------------------------------------------------------------

/// Backoff for the nth crash (1-based), capped at the last entry.
pub(crate) fn backoff_delay(schedule: &[Duration], crashes_so_far: usize) -> Duration {
    if schedule.is_empty() {
        return DEFAULT_BACKOFF
            .last()
            .copied()
            .unwrap_or(Duration::from_secs(30));
    }
    let index = crashes_so_far.saturating_sub(1).min(schedule.len() - 1);
    schedule[index]
}

/// Whether the recorded crashes trip the crash-loop policy.
pub(crate) fn crash_loop_reached(
    crashes: &[(Instant, String)],
    window: Duration,
    threshold: usize,
) -> bool {
    let Some(&(newest, _)) = crashes.last() else {
        return false;
    };
    crashes
        .iter()
        .filter(|(at, _)| newest.duration_since(*at) <= window)
        .count()
        >= threshold
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backoff_schedule_progresses_and_caps() {
        let schedule = [
            Duration::from_secs(1),
            Duration::from_secs(2),
            Duration::from_secs(4),
            Duration::from_secs(8),
            Duration::from_secs(16),
            Duration::from_secs(30),
        ];
        assert_eq!(backoff_delay(&schedule, 1), Duration::from_secs(1));
        assert_eq!(backoff_delay(&schedule, 2), Duration::from_secs(2));
        assert_eq!(backoff_delay(&schedule, 3), Duration::from_secs(4));
        assert_eq!(backoff_delay(&schedule, 6), Duration::from_secs(30));
        // Beyond the schedule: capped at the last entry.
        assert_eq!(backoff_delay(&schedule, 40), Duration::from_secs(30));
        // Degenerate schedule falls back to the default cap.
        assert_eq!(backoff_delay(&[], 1), Duration::from_secs(30));
    }

    #[test]
    fn crash_loop_trips_at_threshold_within_window() {
        let now = Instant::now();
        let tail = "boom".to_string();
        let window = Duration::from_secs(300);
        // Four crashes: not yet.
        let four: Vec<(Instant, String)> = (0..4)
            .map(|i| (now - Duration::from_secs(i), tail.clone()))
            .collect();
        assert!(!crash_loop_reached(&four, window, 5));
        // Five recent: trip.
        let five: Vec<(Instant, String)> = (0..5)
            .map(|i| (now - Duration::from_secs(i), tail.clone()))
            .collect();
        assert!(crash_loop_reached(&five, window, 5));
        // Five spread over more than the window (oldest first, the order
        // `record_crash` pushes): no trip.
        let spread: Vec<(Instant, String)> = [720, 600, 240, 120, 0]
            .iter()
            .map(|i| (now - Duration::from_secs(*i), tail.clone()))
            .collect();
        assert!(!crash_loop_reached(&spread, window, 5));
        // Empty: no trip.
        assert!(!crash_loop_reached(&[], window, 5));
    }

    #[test]
    fn snapshot_lists_catalog_with_install_states() {
        let catalog = crate::engine::catalog::default_catalog();
        let mut state = SharedState::new(None);
        state
            .install
            .insert(catalog[0].id.clone(), InstallState::Installed);
        state.phase = EnginePhase::NoModel;
        let snapshot = build_snapshot(&state, &catalog);
        assert_eq!(snapshot.models.len(), 3);
        assert_eq!(snapshot.models[0].install, InstallState::Installed);
        assert_eq!(snapshot.models[1].install, InstallState::NotInstalled);
        assert!(snapshot.models.iter().all(|model| !model.active));
        assert_eq!(snapshot.phase, EnginePhase::NoModel);
    }

    /// A supervisor over a fresh state, never started (no thread).
    fn idle_supervisor(dir: &std::path::Path) -> Supervisor {
        let config = EngineConfig {
            engine_dir: Some(dir.join("engines")),
            models_dir: dir.join("models"),
            state_dir: dir.join("state"),
            catalog: crate::engine::catalog::default_catalog(),
            backend_override: None,
            icd_dirs: None,
            available_memory_override: None,
            backoff_schedule: None,
        };
        let (cmd_tx, rx) = mpsc::channel::<Command>();
        let inner = Arc::new(Inner {
            cmd_tx,
            state: Arc::new(Mutex::new(SharedState::new(None))),
            catalog: Arc::new(config.catalog.clone()),
            shutdown: Arc::new(AtomicBool::new(false)),
            supervisor: Mutex::new(None),
            config: config.clone(),
        });
        Supervisor {
            rx,
            inner,
            config,
            http: LoopbackHttp::new(),
            switch: None,
            downloads: HashMap::new(),
            pending_restart: None,
            attach_failures: 0,
            next_attach_poll: Instant::now(),
            switch_token: 0,
        }
    }

    /// An activation is taken up in the same update that clears the
    /// previous switch's outcome: a reader never sees its id beside a
    /// refusal or error left over from before it (#356).
    #[test]
    fn taking_up_an_activation_clears_the_previous_outcome_with_its_id() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut supervisor = idle_supervisor(dir.path());
        {
            let mut state = supervisor.inner.lock();
            state.pending_decision = Some(SwapDecision::Refused {
                needed: 2,
                available: 1,
            });
            state.last_error = Some("an earlier switch failed".to_string());
        }
        let before = build_snapshot(&supervisor.inner.lock(), &supervisor.inner.catalog);
        assert_eq!(before.activations_handled, 0);
        supervisor.handle(Command::Activate {
            model_id: "no-such-model".to_string(),
            request: 7,
        });
        let after = build_snapshot(&supervisor.inner.lock(), &supervisor.inner.catalog);
        assert_eq!(after.activations_handled, 7);
        assert_eq!(after.pending_decision, None);
        assert_ne!(after.last_error.as_deref(), Some("an earlier switch failed"));
        // A reload is no activation: the id stays.
        supervisor.cancel_current_switch();
        let superseded = supervisor.inner.lock().supersede();
        supervisor.start_switch("no-such-model".to_string(), SwitchKind::Reload, superseded, None);
        assert_eq!(supervisor.inner.lock().activations_handled, 7);
        supervisor.cancel_current_switch();
    }

    /// `tick` captured engine A, then a cutover installed B before A's
    /// exit was handled: that exit is stale and must leave B serving.
    #[test]
    fn a_stale_crash_report_leaves_the_replacement_serving() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut supervisor = idle_supervisor(dir.path());
        let a = ActiveEngine::attached_engine("model-a", "parakeet", "http://127.0.0.1:1".into(), 1, None);
        let b = ActiveEngine::attached_engine("model-b", "parakeet", "http://127.0.0.1:2".into(), 2, None);
        {
            let mut state = supervisor.inner.lock();
            state.active = Some(Arc::clone(&b));
            state.phase = EnginePhase::Ready;
        }
        supervisor.on_active_crash(&a, "exited 9".to_string());
        let state = supervisor.inner.lock();
        assert_eq!(state.phase, EnginePhase::Ready);
        assert!(state.active.as_ref().is_some_and(|active| Arc::ptr_eq(active, &b)));
        assert!(state.crashes.is_empty(), "a stale exit is not a crash");
        drop(state);
        assert!(supervisor.pending_restart.is_none());

        // The engine actually serving crashing is still handled.
        supervisor.on_active_crash(&b, "exited 9".to_string());
        let state = supervisor.inner.lock();
        assert!(state.active.is_none());
        assert!(matches!(state.phase, EnginePhase::Restarting { .. }));
    }

    #[test]
    fn lease_provenance_names_the_model() {
        let engine = ActiveEngine::attached_engine(
            "parakeet-v3-q8",
            "parakeet",
            "http://127.0.0.1:1234".to_string(),
            42,
            None,
        );
        assert_eq!(engine.leases.load(Ordering::Acquire), 0);
        engine.leases.fetch_add(1, Ordering::Acquire);
        let lease = EngineLease {
            engine: Arc::clone(&engine),
            endpoint: engine.endpoint.clone(),
            slug: engine.slug.clone(),
            model_id: engine.model_id.clone(),
            _marker: None,
        };
        assert_eq!(lease.endpoint(), "http://127.0.0.1:1234");
        assert_eq!(lease.slug(), "parakeet");
        assert_eq!(lease.model_id(), "parakeet-v3-q8");
        assert_eq!(lease.provenance(), "engine:parakeet-v3-q8");
        assert_eq!(engine.leases.load(Ordering::Acquire), 1);
        drop(lease);
        assert_eq!(engine.leases.load(Ordering::Acquire), 0);
    }
}
