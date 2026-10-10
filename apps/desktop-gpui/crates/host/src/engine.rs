//! The supervised engine belongs to the **host**, not the renderer
//! (E17 §1 Mode B, #220): the host owns the bundled-engine
//! [`EngineManager`] (the `starling-serve` sidecar, its crash restarts
//! and model switches, #362/#363, model downloads and deletes), and the
//! runtime's jobs machine reaches it through [`EngineProvider`]. Engine
//! lifetime is host lifetime — the sidecar is spawned with the host's
//! pid as its `--parent-pid`, a renderer that dies mid-job costs
//! nothing, and the host's shutdown stops the engine after the machines
//! have joined. The app runs no engine of its own: its Settings drive
//! this one through [`EngineRequest`]s and render the [`EngineStatus`]
//! the host pushes to every watching window.
//!
//! Which engine serves is the user's engine choice
//! ([`EngineChoice::from_settings`] at startup, [`EngineIntent`] after):
//! the bundled engine with the persisted model and backend, the user's
//! own server in manual mode, or none. The host reads the same settings
//! file and the same model/state directories the desktop app always
//! used.
//!
//! # Following the user's choice while the host runs
//!
//! The host installs [`SettingsProvider`] — a switchable provider — as
//! the runtime's provider and owns an [`EngineHost`] that tracks the live
//! engine state. Two things move it, one transition at a time:
//!
//! - **Requests** ([`EngineHost::handle`]): the app applies an engine
//!   setting the moment the user makes it (`Configure`) and sends
//!   Settings → Engine's actions (activate, download, delete, retry, …).
//!   The app persists the settings itself; the host never writes the
//!   settings file.
//! - **The settings file** ([`watch_settings`]): a hand edit, or a
//!   window that changed the file. Only the fields the file *changed*
//!   since its last read are carried over ([`EngineHost::follow_file`]):
//!   a request changes the engine before the app has written the file,
//!   and a write of unrelated settings (or by a window that has not seen
//!   the change yet) must not drag the engine back.
//!
//! [`EngineHost::transition`] carries a choice over: `activate` for a
//! new model, a backend reload for a changed CPU/automatic override (the
//! engine manager declines it for an engine another process owns rather
//! than start a second sidecar beside it), a fresh manual provider for an
//! endpoint/model change, a started supervisor for manual→builtin, and a
//! stopped engine for builtin→manual. An in-flight recognition runs on
//! the provider (and, through it, the engine lease) it started with: the
//! provider and its in-flight count are captured together under the slot
//! lock, and a mode switch drains that count before it stops the engine.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use starling_dictation::client::{ClientError, StarlingClient};
use starling_dictation::engine::{
    Backend, EngineConfig, EngineLease, EngineManager, EnginePhase, EngineSnapshot,
};
use starling_dictation::settings::{EngineMode, Settings};
use starling_runtime::provider::{
    failure_from_client_error, CancelToken, Partial, ProviderOutcome, StarlingProvider,
    TranscriptionProvider, UnconfiguredProvider,
};

/// How long a job waits for the engine to become ready before failing
/// `engine_not_ready` (retryable). Covers a host that just booted and is
/// still loading or warming its model, and a crash restart's backoff.
pub const DEFAULT_READY_WAIT: Duration = Duration::from_secs(120);

/// How often a waiting job re-checks the engine (and its cancel token).
const READY_POLL: Duration = Duration::from_millis(50);

/// How long the avoid-wait backs off between probes: while the only
/// ready engine is the one a request just failed against, re-leasing
/// every `READY_POLL` would churn lease-marker file I/O on an engine
/// this job will not use — poll it gently instead while the
/// supervisor's own detection (its ~2 s attach poll and restart
/// backoff) replaces or reaps it.
const AVOID_POLL: Duration = Duration::from_millis(250);

/// How often the settings watcher polls the file: one small read, so
/// a hand edit (or another window's write) is visible within a poll or
/// two.
pub const DEFAULT_SETTINGS_POLL: Duration = Duration::from_millis(1500);

/// How long a builtin→manual switch waits for in-flight recognitions
/// (each holds its provider — and through it its engine lease — for its
/// whole request) before stopping the engine anyway: a stuck job must
/// not keep the engine (and its memory) alive indefinitely.
const ENGINE_DRAIN_GRACE: Duration = Duration::from_secs(10);

/// The sleep slice shared by the drain wait and the watcher's poll loop
/// (what keeps both promptly stoppable).
const FOLLOW_SLICE: Duration = Duration::from_millis(50);

/// Locks one of the engine host's mutexes, tolerating poison: none of
/// this state is ownership-critical, and a panic in one apply must not
/// cascade into every later status read.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// What the app says when the user's engine is their own server.
const BUILTIN_OFF: &str = "The built-in engine is off (Settings → Engine uses your own server).";

/// The user's engine choice, resolved for the host at startup.
#[derive(Debug, Clone)]
pub enum EngineChoice {
    /// No engine: jobs fail `no_provider_configured` (the honest
    /// default; `--engine none` and tests).
    None,
    /// The bundled engine, supervised by this host.
    Builtin {
        config: EngineConfig,
        active_model: Option<String>,
    },
    /// The user's own server (`engine.mode = manual`).
    Manual { endpoint: String, model: String },
    /// The bundled engine was chosen but cannot run here: its data
    /// directory does not resolve. Nothing serves (jobs fail
    /// `no_provider_configured`); `reason` is what the app shows, and the
    /// user can still switch to their own server.
    Unavailable { reason: String },
}

impl EngineChoice {
    /// The choice the desktop settings file states, with the app's own
    /// defaults: builtin on the default engine paths with the persisted
    /// model and backend override, or the manual endpoint/model.
    pub fn from_settings(settings: &Settings) -> EngineChoice {
        match settings.engine.mode {
            EngineMode::Builtin => match EngineConfig::default_paths() {
                Ok(mut config) => {
                    config.backend_override = settings
                        .engine
                        .backend_override
                        .as_deref()
                        .and_then(Backend::parse);
                    EngineChoice::Builtin {
                        config,
                        active_model: settings.engine.active_model.clone(),
                    }
                }
                Err(err) => EngineChoice::Unavailable {
                    reason: err.to_string(),
                },
            },
            EngineMode::Manual => EngineChoice::Manual {
                endpoint: settings.endpoint.clone(),
                model: settings.model.clone(),
            },
        }
    }

    /// The label the host's status line reports.
    pub fn label(&self) -> &'static str {
        match self {
            EngineChoice::None => "none",
            EngineChoice::Builtin { .. } => "builtin",
            EngineChoice::Manual { .. } => "manual",
            EngineChoice::Unavailable { .. } => "unavailable",
        }
    }
}

/// The engine settings as the user states them — Settings → Engine and
/// the settings file's engine fields (#220). `active_model` is what a
/// newly started engine loads (and, from the file, the model to switch
/// to); `endpoint`/`model` are the user's own server.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EngineIntent {
    pub mode: EngineMode,
    pub active_model: Option<String>,
    pub backend_override: Option<Backend>,
    pub endpoint: String,
    pub model: String,
}

impl EngineIntent {
    pub fn from_settings(settings: &Settings) -> EngineIntent {
        EngineIntent {
            mode: settings.engine.mode,
            active_model: settings.engine.active_model.clone(),
            backend_override: settings
                .engine
                .backend_override
                .as_deref()
                .and_then(Backend::parse),
            endpoint: settings.endpoint.clone(),
            model: settings.model.clone(),
        }
    }
}

/// What the app asks of the host's engine ([`crate::frame::Frame::Engine`]).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EngineRequest {
    /// The user changed the engine settings: mode, CPU/automatic
    /// backend, their own server. Applied now. `intent.active_model` only
    /// names the model an engine started by this change loads; switching
    /// models is [`EngineRequest::Activate`].
    Configure { intent: EngineIntent },
    /// Download if needed, verify, then switch to the model (#363).
    Activate { model_id: String },
    /// Download a model without activating it.
    Download { model_id: String },
    CancelDownload { model_id: String },
    /// Remove a model's files (refused while it serves, switches in, or
    /// downloads).
    Delete { model_id: String },
    /// Clear a failed engine and start the last model again.
    Retry,
    /// Answer a pending "finish the current take, then switch?".
    ConfirmDrainSwap,
    /// Cancel a running switch; the current model keeps serving.
    CancelSwitch,
}

/// The host's answer to an [`EngineRequest`]
/// ([`crate::frame::Frame::EngineReply`]). What a request set in motion
/// (a download's progress, a switch's stages, its failure) arrives as
/// [`EngineStatus`] pushes, not here.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EngineReply {
    /// Carried out, or under way. `revision` is the engine settings'
    /// revision once this request applied ([`EngineStatus::revision`]).
    Done { revision: u64 },
    /// The activation is under way as the engine manager's request
    /// `request` (see [`EngineSnapshot::activations_handled`]).
    Activating { request: u64, revision: u64 },
    /// Not carried out; `message` says why, for the user.
    Refused { message: String },
}

/// The host's engine as the app renders it
/// ([`crate::frame::Frame::EngineState`]).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EngineStatus {
    /// Which engine the host runs: the built-in one, or the user's
    /// server.
    pub mode: EngineMode,
    /// The built-in engine's CPU/automatic choice (builtin mode).
    pub backend_override: Option<Backend>,
    /// The user's server and model the host transcribes with (manual
    /// mode): what every window's Settings show.
    pub server: Option<(String, String)>,
    /// What serves (`builtin`, `manual:<endpoint>` redacted,
    /// `unconfigured`, `unavailable`, `none`).
    pub label: String,
    /// The built-in engine's manager state (builtin mode, when it runs).
    pub snapshot: Option<EngineSnapshot>,
    /// Why no built-in engine runs although it was chosen.
    pub unavailable: Option<String>,
    /// Bumped whenever the engine settings this host runs change (mode,
    /// backend, server): an app ignores the settings of a status older
    /// than the change it just made.
    pub revision: u64,
}

impl EngineStatus {
    /// What a host running no engine at all reports (`--engine none`).
    pub fn without_engine() -> EngineStatus {
        EngineStatus {
            mode: EngineMode::Builtin,
            backend_override: None,
            server: None,
            label: "none".to_string(),
            snapshot: None,
            unavailable: Some(
                "This Starling recording service runs without a transcription engine.".to_string(),
            ),
            revision: 0,
        }
    }
}

/// Attaches `choice` to the runtime about to start: builtin starts the
/// engine supervisor behind [`SettingsProvider`] (the switchable
/// provider the host can re-aim while it serves), manual installs the
/// plain server provider behind the same switch, none leaves
/// `runtime.provider` untouched. A manual endpoint that does not
/// validate is reported and left unconfigured — the host still owns
/// capture and storage, jobs fail `no_provider_configured`, and the
/// user's settings are not guessed at. The returned [`EngineHost`] is
/// the host's live handle on the engine (current manager, settings
/// changes, requests, shutdown).
pub fn attach(
    choice: EngineChoice,
    runtime: &mut starling_runtime::RuntimeConfig,
) -> Option<Arc<EngineHost>> {
    attach_with_paths(choice, None, runtime)
}

/// [`attach`], with the paths and catalog the built-in engine runs on
/// whenever this host starts one (`None`: the startup choice's, else the
/// default data paths).
pub fn attach_with_paths(
    choice: EngineChoice,
    paths: Option<EngineConfig>,
    runtime: &mut starling_runtime::RuntimeConfig,
) -> Option<Arc<EngineHost>> {
    let host = match choice {
        EngineChoice::None => return None,
        EngineChoice::Builtin {
            config,
            active_model,
        } => EngineHost::start_builtin(paths.unwrap_or_else(|| config.clone()), config, active_model),
        EngineChoice::Manual { endpoint, model } => {
            let (provider, label) = manual_slot(&endpoint, &model);
            EngineHost::new(provider, label, None, EngineState::Manual { endpoint, model }, paths)
        }
        EngineChoice::Unavailable { reason } => EngineHost::new(
            Arc::new(UnconfiguredProvider),
            "unavailable".to_string(),
            None,
            EngineState::Unavailable {
                reason,
                model: None,
                backend_override: None,
            },
            paths,
        ),
    };
    let host = Arc::new(host);
    runtime.provider = host.provider_slot();
    Some(host)
}

/// The runtime's switchable provider: whichever inner provider the
/// current engine settings resolve to, behind one stable
/// [`TranscriptionProvider`] the jobs machine holds for the host's
/// whole lifetime. An in-flight call runs on the inner provider it
/// captured at its start — a settings change mid-request moves the next
/// job, never the running one. The slot also carries its provider's
/// in-flight counter: [`SettingsProvider::recognize`] raises it under
/// the slot lock, so a mode switch that swaps the slot after that point
/// sees the call in its drain instead of stopping the engine under it.
pub struct SettingsProvider {
    current: Mutex<Slot>,
}

struct Slot {
    provider: Arc<dyn TranscriptionProvider>,
    label: String,
    /// The counter of live recognitions on `provider`: `Some` for the
    /// engine provider (the same counter [`EngineState::Builtin`]
    /// holds, so the mode-switch drain in [`EngineHost::transition`]
    /// waits on exactly these calls), `None` for manual/unconfigured
    /// slots (nothing drains behind them).
    in_flight: Option<Arc<AtomicUsize>>,
}

impl SettingsProvider {
    fn new(
        provider: Arc<dyn TranscriptionProvider>,
        label: String,
        in_flight: Option<Arc<AtomicUsize>>,
    ) -> SettingsProvider {
        SettingsProvider {
            current: Mutex::new(Slot {
                provider,
                label,
                in_flight,
            }),
        }
    }

    /// What serves right now — `builtin`, `manual:<endpoint>`,
    /// `unconfigured` or `unavailable` (status lines and tests).
    pub fn label(&self) -> String {
        lock(&self.current).label.clone()
    }

    /// Swaps the inner provider (with its in-flight counter — `None`
    /// when no drain waits behind it). The next recognition starts on
    /// `provider`; calls already running finish on the one they
    /// captured — and stay counted on the counter they captured, which
    /// is what a mode switch drains on before it stops the engine.
    fn install(
        &self,
        provider: Arc<dyn TranscriptionProvider>,
        label: String,
        in_flight: Option<Arc<AtomicUsize>>,
    ) {
        *lock(&self.current) = Slot {
            provider,
            label,
            in_flight,
        };
    }
}

impl TranscriptionProvider for SettingsProvider {
    fn recognize(
        &self,
        wav: Vec<u8>,
        request_id: &str,
        on_partial: &mut dyn FnMut(Partial),
        cancel: &CancelToken,
    ) -> ProviderOutcome {
        // Capture and count under one lock hold: the count rises before
        // the slot can be swapped, so a mode switch that installs after
        // this point finds this call already in the drain's counter
        // (raising it inside the inner provider instead would leave a
        // window where the switch sees zero and stops the engine under
        // this worker). The guard lives in this frame, released only
        // when the call returns; the swap below moves the *next* job,
        // never this one.
        let (provider, _busy) = {
            let slot = lock(&self.current);
            let busy = slot.in_flight.as_ref().map(InFlight::on);
            (Arc::clone(&slot.provider), busy)
        };
        provider.recognize(wav, request_id, on_partial, cancel)
    }
}

/// The host's live engine: the switchable provider the runtime calls
/// through, the engine state it tracks, and the transitions that follow
/// the user's choice. Shared between the host handle (status, shutdown),
/// the settings watcher and the request worker.
pub struct EngineHost {
    provider: Arc<SettingsProvider>,
    state: Mutex<EngineState>,
    /// Set when the host begins shutting down: a builtin→manual drain
    /// in progress stops waiting, so the host's shutdown (which joins
    /// the watcher) is not held for the drain's grace. The runtime's own
    /// shutdown cancels the drained jobs.
    closing: AtomicBool,
    /// The paths and catalog the built-in engine runs on (the startup
    /// choice's, or the host config's); `None`: the default data paths,
    /// resolved when an engine starts.
    paths: Option<EngineConfig>,
    /// One transition at a time (the watcher and requests both move the
    /// engine; a builtin→manual drain included).
    transitions: Mutex<()>,
    /// What the settings file stated at the watcher's last read: the
    /// base the next read is compared with.
    file: Mutex<Option<EngineIntent>>,
    /// See [`EngineStatus::revision`].
    revision: AtomicU64,
}

/// What the host runs right now.
enum EngineState {
    /// Builtin mode: this host supervises an engine. `in_flight` counts
    /// live recognitions on the slot fronting its provider — what a
    /// mode switch drains before it stops the engine.
    Builtin {
        manager: EngineManager,
        in_flight: Arc<AtomicUsize>,
        /// Set just before a mode switch stops `manager`: a recognition
        /// still waiting for a lease then fails at once instead of
        /// polling a stopped engine for its whole ready wait.
        stopped: Arc<AtomicBool>,
        /// The model last asked for (at start, or by an activation).
        model: Option<String>,
        backend_override: Option<Backend>,
    },
    /// Manual mode: the endpoint/model the provider was last built from.
    Manual { endpoint: String, model: String },
    /// Builtin mode, but no engine can run: `reason` says why. `model`
    /// and `backend_override` are what a retry starts with.
    Unavailable {
        reason: String,
        model: Option<String>,
        backend_override: Option<Backend>,
    },
}

/// Which fields of an [`EngineIntent`] a transition carries over.
#[derive(Debug, Clone, Copy)]
struct Fields {
    mode: bool,
    backend: bool,
    server: bool,
    model: bool,
}

impl Fields {
    const ALL: Fields = Fields {
        mode: true,
        backend: true,
        server: true,
        model: true,
    };
    /// A `Configure` request: everything but the model (activating one
    /// is its own request).
    const CONFIGURE: Fields = Fields {
        mode: true,
        backend: true,
        server: true,
        model: false,
    };
}

/// A started engine: its state and the slot that fronts it.
fn builtin_engine(
    config: EngineConfig,
    model: Option<String>,
) -> (EngineState, Arc<dyn TranscriptionProvider>, Arc<AtomicUsize>) {
    let backend_override = config.backend_override;
    let manager = EngineManager::start(config, model.clone());
    // One counter, shared by the slot (which raises it under its lock per
    // call) and the state (whose mode-switch drain waits on it): the
    // engine provider itself stays counter-free.
    let in_flight = Arc::new(AtomicUsize::new(0));
    let provider = EngineProvider::new(manager.clone());
    let stopped = provider.stopped_flag();
    (
        EngineState::Builtin {
            manager,
            in_flight: Arc::clone(&in_flight),
            stopped,
            model,
            backend_override,
        },
        Arc::new(provider),
        in_flight,
    )
}

impl EngineHost {
    fn new(
        provider: Arc<dyn TranscriptionProvider>,
        label: String,
        in_flight: Option<Arc<AtomicUsize>>,
        state: EngineState,
        paths: Option<EngineConfig>,
    ) -> EngineHost {
        EngineHost {
            provider: Arc::new(SettingsProvider::new(provider, label, in_flight)),
            state: Mutex::new(state),
            closing: AtomicBool::new(false),
            paths,
            transitions: Mutex::new(()),
            file: Mutex::new(None),
            revision: AtomicU64::new(0),
        }
    }

    fn start_builtin(
        paths: EngineConfig,
        config: EngineConfig,
        active_model: Option<String>,
    ) -> EngineHost {
        let (state, provider, in_flight) = builtin_engine(config, active_model);
        EngineHost::new(
            provider,
            "builtin".to_string(),
            Some(in_flight),
            state,
            Some(paths),
        )
    }

    /// The runtime's provider slot this host fills (`attach` installs
    /// it; nothing else constructs an `EngineHost`).
    fn provider_slot(&self) -> Arc<SettingsProvider> {
        Arc::clone(&self.provider)
    }

    /// What serves right now (`builtin`, `manual:<endpoint>`,
    /// `unconfigured` or `unavailable`) — the live analogue of
    /// [`EngineChoice::label`].
    pub fn label(&self) -> String {
        self.provider.label()
    }

    /// The engine manager this host supervises right now (builtin
    /// mode), for status reporting and tests. `None` otherwise —
    /// including after a live builtin→manual switch.
    pub fn manager(&self) -> Option<EngineManager> {
        match &*lock(&self.state) {
            EngineState::Builtin { manager, .. } => Some(manager.clone()),
            EngineState::Manual { .. } | EngineState::Unavailable { .. } => None,
        }
    }

    /// The engine as the app renders it.
    pub fn status(&self) -> EngineStatus {
        let state = lock(&self.state);
        let revision = self.revision.load(Ordering::SeqCst);
        let label = self.label();
        match &*state {
            EngineState::Builtin {
                manager,
                backend_override,
                ..
            } => EngineStatus {
                mode: EngineMode::Builtin,
                backend_override: *backend_override,
                server: None,
                label,
                snapshot: Some(manager.snapshot()),
                unavailable: None,
                revision,
            },
            EngineState::Manual { endpoint, model } => EngineStatus {
                mode: EngineMode::Manual,
                backend_override: None,
                server: Some((endpoint.clone(), model.clone())),
                label,
                snapshot: None,
                unavailable: None,
                revision,
            },
            EngineState::Unavailable {
                reason,
                backend_override,
                ..
            } => EngineStatus {
                mode: EngineMode::Builtin,
                backend_override: *backend_override,
                server: None,
                label,
                snapshot: None,
                unavailable: Some(reason.clone()),
                revision,
            },
        }
    }

    /// Changes whenever [`Self::status`] would read differently: the
    /// settings revision, and the running manager's generation.
    pub fn generation(&self) -> (u64, u64) {
        let state = lock(&self.state);
        let manager = match &*state {
            EngineState::Builtin { manager, .. } => manager.generation(),
            EngineState::Manual { .. } | EngineState::Unavailable { .. } => 0,
        };
        (self.revision.load(Ordering::SeqCst), manager)
    }

    fn bump(&self) -> u64 {
        self.revision.fetch_add(1, Ordering::SeqCst) + 1
    }

    /// The built-in engine's config for `backend_override`: this host's
    /// paths, or the default data paths.
    fn builtin_config(&self, backend_override: Option<Backend>) -> Result<EngineConfig, String> {
        let mut config = match &self.paths {
            Some(paths) => paths.clone(),
            None => EngineConfig::default_paths().map_err(|err| err.to_string())?,
        };
        config.backend_override = backend_override;
        Ok(config)
    }

    /// Carries out an app's engine request (one at a time, on the host's
    /// engine worker — a `Configure` may drain in-flight recognitions).
    pub fn handle(&self, request: EngineRequest) -> EngineReply {
        let done = |host: &EngineHost| EngineReply::Done {
            revision: host.revision.load(Ordering::SeqCst),
        };
        let refused = |message: String| EngineReply::Refused { message };
        match request {
            EngineRequest::Configure { intent } => match self.transition(&intent, Fields::CONFIGURE)
            {
                Ok(()) => done(self),
                Err(message) => refused(message),
            },
            EngineRequest::Activate { model_id } => {
                let mut state = lock(&self.state);
                match &mut *state {
                    EngineState::Builtin { manager, model, .. } => {
                        let request = manager.activate(&model_id);
                        *model = Some(model_id);
                        EngineReply::Activating {
                            request,
                            revision: self.revision.load(Ordering::SeqCst),
                        }
                    }
                    other => refused(not_running(other)),
                }
            }
            EngineRequest::Retry => {
                // Held from the look at the state to the restart: a mode
                // switch the settings watcher makes in between must not
                // be overwritten by a restart meant for an engine that
                // could not run.
                let _one = lock(&self.transitions);
                let retry_unavailable = {
                    let state = lock(&self.state);
                    match &*state {
                        EngineState::Builtin { manager, .. } => {
                            manager.retry();
                            return done(self);
                        }
                        EngineState::Manual { .. } => return refused(BUILTIN_OFF.to_string()),
                        EngineState::Unavailable {
                            model,
                            backend_override,
                            ..
                        } => (model.clone(), *backend_override),
                    }
                };
                // No engine could start: try again from scratch (the data
                // directory may resolve now).
                let (model, backend_override) = retry_unavailable;
                let intent = EngineIntent {
                    mode: EngineMode::Builtin,
                    active_model: model,
                    backend_override,
                    endpoint: String::new(),
                    model: String::new(),
                };
                match self.transition_held(&intent, Fields::CONFIGURE) {
                    Ok(()) => done(self),
                    Err(message) => refused(message),
                }
            }
            other => {
                // The rest only need the manager; it is cloned out so a
                // delete's file work never holds the state lock. A
                // transition waits for them (and they for it): none acts
                // on a manager a mode switch is shutting down.
                let _one = lock(&self.transitions);
                let manager = match &*lock(&self.state) {
                    EngineState::Builtin { manager, .. } => manager.clone(),
                    other_state => return refused(not_running(other_state)),
                };
                match other {
                    EngineRequest::Download { model_id } => manager.download(&model_id),
                    EngineRequest::CancelDownload { model_id } => {
                        manager.cancel_download(&model_id)
                    }
                    EngineRequest::Delete { model_id } => {
                        if let Err(err) = manager.delete_model(&model_id) {
                            return refused(err.to_string());
                        }
                    }
                    EngineRequest::ConfirmDrainSwap => manager.confirm_drain_swap(),
                    EngineRequest::CancelSwitch => manager.cancel_switch(),
                    EngineRequest::Configure { .. }
                    | EngineRequest::Activate { .. }
                    | EngineRequest::Retry => unreachable!("handled above"),
                }
                done(self)
            }
        }
    }

    /// Follows the settings file: `intent` is what it says now. The first
    /// read carries everything over (a change between the host's startup
    /// load and the watcher's start must not be frozen out — and a file
    /// that still agrees costs nothing); after that, only the fields the
    /// file changed since its last read (see the module docs).
    pub fn follow_file(&self, intent: EngineIntent) {
        let previous = lock(&self.file).replace(intent.clone());
        let fields = match previous {
            None => Fields::ALL,
            Some(previous) => Fields {
                mode: previous.mode != intent.mode,
                backend: previous.backend_override != intent.backend_override,
                server: previous.endpoint != intent.endpoint || previous.model != intent.model,
                model: previous.active_model != intent.active_model,
            },
        };
        if let Err(reason) = self.transition(&intent, fields) {
            eprintln!("starling-runtime-host: the engine settings cannot apply: {reason}");
        }
    }

    /// Brings the live engine to `intent` for the `fields` given — the
    /// host-side twin of the app's engine actions: a new model activates,
    /// a changed backend reloads the engine, manual endpoint/model
    /// changes rebuild the server provider, manual→builtin starts a
    /// supervisor, builtin→manual stops routing to the engine and (once
    /// in-flight recognitions finished, bounded by
    /// [`ENGINE_DRAIN_GRACE`]) shuts it down. Only the deltas run: an
    /// unchanged choice costs nothing. An error is why the built-in
    /// engine cannot run (its data directory does not resolve).
    fn transition(&self, intent: &EngineIntent, fields: Fields) -> Result<(), String> {
        let _one = lock(&self.transitions);
        self.transition_held(intent, fields)
    }

    /// [`Self::transition`], for a caller already holding `transitions`.
    fn transition_held(&self, intent: &EngineIntent, fields: Fields) -> Result<(), String> {
        let mut state = lock(&self.state);
        match intent.mode {
            EngineMode::Builtin => match &mut *state {
                EngineState::Builtin {
                    manager,
                    model,
                    backend_override,
                    ..
                } => {
                    if fields.backend && intent.backend_override != *backend_override {
                        // Live: the app runs no engine of its own, so the
                        // reload replaces the engine this host owns (the
                        // old one drains). One another process owns keeps
                        // its backend — the manager will not start a
                        // second sidecar beside it, and says so.
                        manager.set_backend_override(intent.backend_override);
                        *backend_override = intent.backend_override;
                        self.bump();
                    }
                    if fields.model {
                        match intent.active_model.as_deref() {
                            Some(model_id) => {
                                if !serves_or_loads(manager, model.as_deref(), model_id) {
                                    manager.activate(model_id);
                                    *model = Some(model_id.to_string());
                                }
                            }
                            // The settings stopped naming a model. The
                            // app always persists its active choice, so
                            // this is a hand-edited or foreign file; keep
                            // the engine the running jobs know (the
                            // supervisor has no "serve nothing" command,
                            // and stopping under the user's takes is the
                            // worse failure).
                            None => eprintln!(
                                "starling-runtime-host: the settings name no engine model; \
                                 keeping the running one"
                            ),
                        }
                    }
                    Ok(())
                }
                EngineState::Manual { .. } | EngineState::Unavailable { .. } if fields.mode => {
                    // → builtin: start a supervisor, then aim the runtime
                    // at it (with the counter its future mode switch will
                    // drain on).
                    let config = match self.builtin_config(intent.backend_override) {
                        Ok(config) => config,
                        Err(reason) => {
                            self.provider.install(
                                Arc::new(UnconfiguredProvider),
                                "unavailable".to_string(),
                                None,
                            );
                            *state = EngineState::Unavailable {
                                reason: reason.clone(),
                                model: intent.active_model.clone(),
                                backend_override: intent.backend_override,
                            };
                            self.bump();
                            return Err(reason);
                        }
                    };
                    let (started, provider, in_flight) =
                        builtin_engine(config, intent.active_model.clone());
                    self.provider
                        .install(provider, "builtin".to_string(), Some(in_flight));
                    *state = started;
                    self.bump();
                    Ok(())
                }
                // The file left the mode alone: nothing of the built-in
                // engine's applies to the engine that runs.
                EngineState::Manual { .. } | EngineState::Unavailable { .. } => Ok(()),
            },
            EngineMode::Manual => {
                let changed = match &*state {
                    EngineState::Manual {
                        endpoint: current,
                        model: current_model,
                    } => fields.server && (current != &intent.endpoint || current_model != &intent.model),
                    EngineState::Builtin { .. } | EngineState::Unavailable { .. } => fields.mode,
                };
                if !changed {
                    return Ok(());
                }
                // Swap first: every job submitted after this line routes
                // to the manual provider, never to the engine a mode
                // switch is about to stop. The label is the effective
                // one: an endpoint that does not validate reads as
                // `unconfigured`, not as a manual engine that serves.
                // A recognition that entered the old slot before this
                // swap already raised the counter the drain below waits
                // on (see `SettingsProvider::recognize`), so it cannot
                // be missed.
                let (provider, label) = manual_slot(&intent.endpoint, &intent.model);
                self.provider.install(provider, label, None);
                let previous = std::mem::replace(
                    &mut *state,
                    EngineState::Manual {
                        endpoint: intent.endpoint.clone(),
                        model: intent.model.clone(),
                    },
                );
                self.bump();
                // The state already says manual: release it before the
                // drain below, so status reads (`manager()`) and the
                // host's shutdown never wait out the grace behind it.
                drop(state);
                if let EngineState::Builtin {
                    manager,
                    in_flight,
                    stopped,
                    ..
                } = previous
                {
                    // In-flight recognitions hold their provider (and its
                    // engine lease) for their whole request: give them
                    // their grace, then stop the engine — the host must
                    // not keep a user-rejected engine (and its memory)
                    // alive behind a stuck job.
                    let deadline = Instant::now() + ENGINE_DRAIN_GRACE;
                    while in_flight.load(Ordering::SeqCst) > 0
                        && Instant::now() < deadline
                        && !self.closing.load(Ordering::SeqCst)
                    {
                        std::thread::sleep(FOLLOW_SLICE);
                    }
                    // A recognition still past the grace is waiting for a
                    // lease (or a crash retry): tell it the engine is
                    // going away so it fails now, not after its ready wait.
                    stopped.store(true, Ordering::SeqCst);
                    manager.shutdown();
                }
                Ok(())
            }
        }
    }

    /// The target a take starting now binds to (#363), when something
    /// serves now: a lease on the ready engine, or the manual server.
    /// When nothing does (the take records anyway; its transcription
    /// waits for an engine, [`Self::wait_target`]) the error is the
    /// recording window's notice: what serves no live text, and why.
    pub fn bind_now(&self) -> Result<Target, String> {
        match &*lock(&self.state) {
            EngineState::Builtin {
                manager, in_flight, ..
            } => manager
                .lease()
                .map(|lease| Target::from_lease(lease, in_flight))
                .ok_or_else(|| {
                    "The built-in engine is not ready, so this recording shows no live text. It \
                     is saved either way, and transcribed after you stop once the engine is \
                     ready."
                        .to_string()
                }),
            EngineState::Manual { endpoint, model } => {
                Target::manual(endpoint, model).map_err(|err| {
                    format!(
                        "Your server's endpoint in Settings is not usable ({err}), so this \
                         recording shows no live text. It is saved either way; fix the endpoint \
                         and retry it."
                    )
                })
            }
            EngineState::Unavailable { reason, .. } => Err(format!(
                "The built-in engine cannot run ({reason}), so this recording shows no live \
                 text. It is saved either way."
            )),
        }
    }

    /// Waits up to `wait` (cancel-aware) for a target serving `want`
    /// other than the engine `avoid` names (one a request just failed
    /// against). The error is why there is none: a sentence for the
    /// take's history.
    pub fn wait_target(
        &self,
        want: &Want,
        wait: Duration,
        cancel: &CancelToken,
        avoid: Option<&(String, u32)>,
    ) -> Result<Target, String> {
        let deadline = Instant::now() + wait;
        loop {
            if cancel.is_cancelled() {
                return Err("The transcription was cancelled.".to_string());
            }
            // The lease is taken, and counted, under the state lock: a
            // switch away from the built-in engine either happened first
            // (and this sees the manual state) or sees this request in the
            // drain it waits on before stopping the engine.
            let (manager, found, avoiding) = match &*lock(&self.state) {
                EngineState::Builtin {
                    manager,
                    stopped,
                    in_flight,
                    ..
                } => {
                    if stopped.load(Ordering::SeqCst) {
                        return Err(not_ready_sentence());
                    }
                    let mut avoiding = false;
                    let mut found = None;
                    if let Some(lease) = manager.lease() {
                        let wanted = match want {
                            Want::Current => true,
                            Want::Model(model_id) => lease.model_id() == model_id,
                        };
                        let identity = (lease.endpoint().to_string(), lease.pid());
                        if wanted && avoid != Some(&identity) {
                            found = Some(Target::from_lease(lease, in_flight));
                        }
                        avoiding = wanted;
                    }
                    (manager.clone(), found, avoiding)
                }
                EngineState::Manual { endpoint, model } => {
                    return match want {
                        Want::Current => Target::manual(endpoint, model).map_err(|err| {
                            format!("Your server's endpoint in Settings is not usable: {err}")
                        }),
                        Want::Model(_) => Err(BUILTIN_OFF.to_string()),
                    };
                }
                EngineState::Unavailable { reason, .. } => {
                    return Err(format!("The built-in engine cannot run: {reason}"));
                }
            };
            if let Some(target) = found {
                return Ok(target);
            }
            let phase = manager.snapshot().phase;
            // A failed engine or a missing model will not fix itself while
            // this waits (both need the user).
            if matches!(phase, EnginePhase::NoModel | EnginePhase::Failed(_))
                || Instant::now() >= deadline
            {
                return Err(match want {
                    Want::Model(model_id) if !matches!(phase, EnginePhase::NoModel) => format!(
                        "The built-in engine did not start serving {model_id} in time."
                    ),
                    _ => not_ready_sentence(),
                });
            }
            std::thread::sleep(if avoiding { AVOID_POLL } else { READY_POLL });
        }
    }

    /// Marks the host as shutting down (see `closing`). The host calls
    /// this before it joins the settings watcher.
    pub fn begin_shutdown(&self) {
        self.closing.store(true, Ordering::SeqCst);
    }

    /// Stops the engine this host supervises (a no-op in manual mode,
    /// or for an engine another process owns — the attached case leaves
    /// that owner's sidecar running). The host calls this on its
    /// shutdown path, after the settings watcher stopped and the
    /// runtime's machines joined.
    pub fn shutdown(&self) {
        // Clone out and release the state lock before the blocking stop,
        // so status reads (`label()`, `manager()`) never wait behind it.
        let manager = self.manager();
        if let Some(manager) = manager {
            manager.shutdown();
        }
    }
}

/// Why a request for the built-in engine finds none.
fn not_running(state: &EngineState) -> String {
    match state {
        EngineState::Unavailable { reason, .. } => {
            format!("The built-in engine cannot run: {reason}")
        }
        EngineState::Builtin { .. } | EngineState::Manual { .. } => BUILTIN_OFF.to_string(),
    }
}

/// Whether `manager` already serves (or is bringing up) `model_id`: a
/// switch to it runs, it is active with no switch away from it, or —
/// before anything is active — it is the model the engine started with
/// (its first launch is no switch).
fn serves_or_loads(manager: &EngineManager, started_with: Option<&str>, model_id: &str) -> bool {
    let snapshot = manager.snapshot();
    match (&snapshot.switch, &snapshot.active) {
        (Some(switch), _) => switch.target_model_id == model_id,
        (None, Some(active)) => active.model_id == model_id,
        (None, None) => started_with == Some(model_id),
    }
}

/// What a transcription asks the engine for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Want {
    /// Whatever serves now.
    Current,
    /// The built-in engine serving this model.
    Model(String),
}

/// Where one transcription goes (#220, #363): the endpoint and model it
/// is sent to, the label its attempt row carries, and — on the built-in
/// engine — the lease that keeps that engine serving until the target is
/// dropped (a model switch drains it instead of cutting it off), counted
/// on the engine's in-flight counter for as long (a switch away from the
/// built-in engine waits, bounded, before stopping it under the take).
pub struct Target {
    pub endpoint: String,
    pub model: String,
    /// The attempt row's backend label (`engine:<model id>`,
    /// `openai:<model>`).
    pub backend: String,
    pub builtin: bool,
    lease: Option<EngineLease>,
    _counted: Option<InFlight>,
}

impl Target {
    /// The user's own server; an error when the endpoint is not usable.
    pub fn manual(endpoint: &str, model: &str) -> Result<Target, ClientError> {
        StarlingClient::new(endpoint, model)?;
        Ok(Target {
            endpoint: endpoint.to_string(),
            model: model.to_string(),
            backend: starling_dictation::storage::BackendLabel::OpenAi {
                model: model.to_string(),
            }
            .to_string(),
            builtin: false,
            lease: None,
            _counted: None,
        })
    }

    fn from_lease(lease: EngineLease, in_flight: &Arc<AtomicUsize>) -> Target {
        Target {
            endpoint: lease.endpoint().to_string(),
            model: lease.slug().to_string(),
            backend: starling_dictation::storage::BackendLabel::Engine {
                model_id: lease.model_id().to_string(),
            }
            .to_string(),
            builtin: true,
            lease: Some(lease),
            _counted: Some(InFlight::on(in_flight)),
        }
    }

    /// The built-in engine this target holds (`(endpoint, pid)`), for a
    /// retry that must not land on it again.
    pub fn engine(&self) -> Option<(String, u32)> {
        self.lease
            .as_ref()
            .map(|lease| (lease.endpoint().to_string(), lease.pid()))
    }
}

/// Why a transcription found no engine.
fn not_ready_sentence() -> String {
    "The built-in engine is not ready; the recording is saved — pick a model in Settings, or \
     retry once the engine is ready."
        .to_string()
}

/// The manual provider for `endpoint`/`model` and the label that goes
/// with it: the honest `manual:<endpoint>` when the endpoint validates,
/// or `unconfigured` beside the honest [`UnconfiguredProvider`] when it
/// does not (reported, not guessed at — the startup and the follow
/// paths share this posture). The status line reads this label, so a
/// manual endpoint that cannot serve never claims an engine is
/// attached while jobs fail `no_provider_configured`.
fn manual_slot(endpoint: &str, model: &str) -> (Arc<dyn TranscriptionProvider>, String) {
    match StarlingProvider::new(endpoint, model) {
        Ok(provider) => (Arc::new(provider), manual_label(endpoint)),
        Err(err) => {
            eprintln!(
                "starling-runtime-host: manual engine endpoint {:?} is unusable \
                 ({err}); transcription stays unconfigured (reporting engine: unconfigured)",
                redact_endpoint(endpoint)
            );
            (Arc::new(UnconfiguredProvider), "unconfigured".to_string())
        }
    }
}

/// The label goes to stdout (the launcher's `owner` line) and to every
/// client's status, so it carries the redacted endpoint too.
fn manual_label(endpoint: &str) -> String {
    format!("manual:{}", redact_endpoint(endpoint))
}

/// Query keys whose values read as credentials, matched as substrings
/// of the lower-cased key (`access_token`, `X-Api-Key`, `sig`, …).
/// Over-matching only costs a `***` in a log line.
const SECRET_QUERY_KEYS: &[&str] = &[
    "token",
    "key",
    "secret",
    "password",
    "passwd",
    "pwd",
    "auth",
    "sig",
    "credential",
    "session",
];

/// `endpoint` as it may be echoed (stderr, the status label): userinfo
/// (`user:pass@`) and the values of token-like query parameters become
/// `***`. Plain string surgery rather than a URL parse — the text that
/// most needs redacting is the endpoint that did not validate (the
/// client rejects userinfo outright), and it may not parse at all.
///
/// The authority is found the way a lenient URL parser finds it: after
/// the scheme any run of `/` or `\` is skipped (`http:///u:p@h` and
/// `http:/u:p@h` still carry userinfo), and without a scheme followed
/// by a slash the authority starts at the very beginning (`http:u:p@h`
/// is userinfo too, so the conservative reading covers it). The text is
/// first normalized as that parser does — leading and trailing control
/// characters and spaces trimmed, embedded tabs and newlines removed —
/// so ` http://u:p@h` or `?to\tken=` cannot hide a credential from it.
fn redact_endpoint(endpoint: &str) -> String {
    let normalized: String = endpoint
        .trim_matches(|c: char| c <= ' ')
        .chars()
        .filter(|c| !matches!(c, '\t' | '\n' | '\r'))
        .collect();
    let endpoint = normalized.as_str();
    // Authority slashes are skipped with or without a scheme, so a
    // scheme-relative `//user:secret@host` is redacted too.
    let after_scheme = scheme_end(endpoint)
        .filter(|&at| endpoint[at..].starts_with(['/', '\\']))
        .unwrap_or(0);
    let authority_start = after_scheme
        + endpoint[after_scheme..]
            .find(|c| c != '/' && c != '\\')
            .unwrap_or(endpoint.len() - after_scheme);
    let authority_end = endpoint[authority_start..]
        .find(['/', '\\', '?', '#'])
        .map_or(endpoint.len(), |at| authority_start + at);
    let mut out = String::with_capacity(endpoint.len());
    out.push_str(&endpoint[..authority_start]);
    let authority = &endpoint[authority_start..authority_end];
    match authority.rfind('@') {
        Some(at) => {
            out.push_str("***");
            out.push_str(&authority[at..]);
        }
        None => out.push_str(authority),
    }
    let rest = &endpoint[authority_end..];
    let Some(query_start) = rest.find('?') else {
        out.push_str(rest);
        return out;
    };
    // A `?` after a `#` is fragment text, not a query.
    if rest.find('#').is_some_and(|hash| hash < query_start) {
        out.push_str(rest);
        return out;
    }
    out.push_str(&rest[..=query_start]);
    let query_and_fragment = &rest[query_start + 1..];
    let query_end = query_and_fragment
        .find('#')
        .unwrap_or(query_and_fragment.len());
    let pairs: Vec<String> = query_and_fragment[..query_end]
        .split('&')
        .map(|pair| match pair.split_once('=') {
            Some((key, _)) if is_secret_key(key) => format!("{key}=***"),
            _ => pair.to_string(),
        })
        .collect();
    out.push_str(&pairs.join("&"));
    out.push_str(&query_and_fragment[query_end..]);
    out
}

/// Just past `scheme:` when `endpoint` starts with one
/// (`ALPHA *( ALPHA / DIGIT / "+" / "-" / "." ) ":"`).
fn scheme_end(endpoint: &str) -> Option<usize> {
    let colon = endpoint.find(':')?;
    let scheme = &endpoint[..colon];
    let mut chars = scheme.chars();
    let valid = chars.next().is_some_and(|c| c.is_ascii_alphabetic())
        && chars.all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.'));
    valid.then_some(colon + 1)
}

/// Classifies the percent-decoded key: `%74oken` is `token` to the
/// server, so it is to the redactor too.
fn is_secret_key(key: &str) -> bool {
    let bytes = key.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut at = 0;
    while at < bytes.len() {
        let hex = bytes
            .get(at + 1..at + 3)
            // Exactly two hex digits: `from_str_radix` alone would take
            // a sign (`%+f`).
            .filter(|hex| hex.iter().all(u8::is_ascii_hexdigit))
            .and_then(|hex| std::str::from_utf8(hex).ok())
            .and_then(|hex| u8::from_str_radix(hex, 16).ok());
        match (bytes[at], hex) {
            (b'%', Some(byte)) => {
                decoded.push(byte);
                at += 3;
            }
            (byte, _) => {
                decoded.push(byte);
                at += 1;
            }
        }
    }
    let key = String::from_utf8_lossy(&decoded).to_ascii_lowercase();
    SECRET_QUERY_KEYS.iter().any(|secret| key.contains(secret))
}

/// Spawns the host's settings watcher (#220): polls `path`'s bytes
/// every `poll` and, when they change, parses the **captured** bytes
/// ([`Settings::from_json_bytes`], never a second read of the file —
/// the bytes between two reads could differ) and has `host` follow the
/// engine settings they state ([`EngineHost::follow_file`]).
///
/// Bytes that are not valid JSON — an empty or truncated file, what a
/// non-atomic writer looks like mid-write — state no choice: the host
/// keeps its last-applied one (reported once per distinct bad content)
/// until the file says something parseable again. A missing file is
/// the same non-event, and a deletion racing the stability re-read
/// must not read as "the user chose the defaults".
///
/// The watcher starts with no `last`: whatever the file says at its
/// first poll is followed, so a change between the startup load (which
/// resolved the host's initial [`EngineChoice`]) and the watcher's
/// start is not silently frozen out. That first read is a no-op when
/// the file still agrees with the running engine — transitions only run
/// deltas. Stops when `stop` is set (checked every slice), so the
/// host's shutdown joins it before it stops the engine.
pub fn watch_settings(
    host: Arc<EngineHost>,
    path: PathBuf,
    poll: Duration,
    stop: Arc<AtomicBool>,
) -> std::io::Result<std::thread::JoinHandle<()>> {
    std::thread::Builder::new()
        .name("starling-host-settings".to_string())
        .spawn(move || {
            // No seed: the first poll applies what the file says now (a
            // change since the startup load must not wait for a second
            // one).
            let mut last: Option<Vec<u8>> = None;
            let mut reported_bad: Option<Vec<u8>> = None;
            loop {
                if stop.load(Ordering::SeqCst) {
                    return;
                }
                let deadline = Instant::now() + poll;
                while Instant::now() < deadline {
                    if stop.load(Ordering::SeqCst) {
                        return;
                    }
                    std::thread::sleep(
                        FOLLOW_SLICE.min(deadline.saturating_duration_since(Instant::now())),
                    );
                }
                let current = std::fs::read(&path).ok();
                if current != last {
                    let Some(bytes) = current.as_deref() else {
                        last = None;
                        continue;
                    };
                    // Apply only what the file stably says: a deletion
                    // racing between the two reads must not read as a
                    // choice (the missing file is not "the user chose
                    // the defaults"). An unstable read leaves `last`
                    // untouched, so the next poll retries these bytes
                    // instead of treating them as already seen.
                    if std::fs::read(&path).ok().as_deref() != Some(bytes) {
                        continue;
                    }
                    last = current.clone();
                    match Settings::from_json_bytes(bytes) {
                        Some(settings) => {
                            // Valid again: a later recurrence of a bad
                            // content is reported anew.
                            reported_bad = None;
                            host.follow_file(EngineIntent::from_settings(&settings));
                        }
                        // Not JSON — an empty or truncated file. The host
                        // keeps its last-applied choice; report each
                        // distinct bad content once (the same truncated
                        // bytes on every poll must not spam).
                        None => {
                            if reported_bad.as_deref() != Some(bytes) {
                                eprintln!(
                                    "starling-runtime-host: engine settings at {} are not \
                                     valid JSON; the engine stays as it is",
                                    path.display()
                                );
                                reported_bad = current.clone();
                            }
                        }
                    }
                }
            }
        })
}

/// The jobs machine's provider over the host-owned engine: each
/// recognition leases the active engine for its whole request (so a
/// model switch drains it rather than cutting it off), sends the take
/// through `starling-dictation`'s client, and reports the model it ran
/// on as the completion's `backend` (`engine:<model id>`).
///
/// In-flight counting is **not** here: the [`SettingsProvider`] slot
/// that fronts this provider raises (and drains on) the one counter
/// under its own lock — one counter, no window between "captured the
/// provider" and "counted the call" for a mode switch to slip through
/// (see [`SettingsProvider::recognize`]).
pub struct EngineProvider {
    manager: EngineManager,
    ready_wait: Duration,
    /// Set when the host stops `manager` for good (a switch away from
    /// the builtin engine): lease waits end at once.
    stopped: Arc<AtomicBool>,
}

impl EngineProvider {
    pub fn new(manager: EngineManager) -> EngineProvider {
        EngineProvider {
            manager,
            ready_wait: DEFAULT_READY_WAIT,
            stopped: Arc::new(AtomicBool::new(false)),
        }
    }

    /// The flag that tells this provider its engine was stopped for
    /// good.
    fn stopped_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.stopped)
    }

    /// Overrides how long a job waits for a ready engine.
    pub fn with_ready_wait(mut self, wait: Duration) -> EngineProvider {
        self.ready_wait = wait;
        self
    }
}

/// One live recognition's count on its slot's counter, released on
/// every exit (including panics). It owns a handle to the counter
/// rather than borrowing it: the count must outlive the slot lock it
/// was raised under (the slot swaps mid-call; the count drains only
/// when the call returns).
struct InFlight(Arc<AtomicUsize>);

impl InFlight {
    fn on(counter: &Arc<AtomicUsize>) -> InFlight {
        counter.fetch_add(1, Ordering::SeqCst);
        InFlight(Arc::clone(counter))
    }
}

impl Drop for InFlight {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

/// Why no engine could take a job, as a v1 `jobs.failed` reason. Every
/// case is retryable: the take is durable, and the same job succeeds
/// once the engine serves (a model installed, a crash restart done).
fn unready_reason(phase: &EnginePhase) -> &'static str {
    match phase {
        EnginePhase::NoModel => "engine_no_model",
        EnginePhase::Failed(_) => "engine_unavailable",
        _ => "engine_not_ready",
    }
}

/// `engine:<model id>` as a contract `safeToken`
/// (`[A-Za-z0-9_.:+-]{1,128}`): catalog ids already fit; anything else
/// is replaced rather than letting the completion event fail
/// validation. An id past the bound keeps its distinctness: the tail is
/// replaced by a fold of the whole id, so two long ids sharing a prefix
/// must not collapse into one token (that would silently corrupt
/// completion-event analytics).
fn backend_token(model_id: &str) -> String {
    /// The contract `safeToken`'s length bound.
    const SAFE_TOKEN_MAX: usize = 128;
    let token: String = format!("engine:{model_id}")
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || "_.:+-".contains(c) {
                c
            } else {
                '_'
            }
        })
        .collect();
    if token.len() > SAFE_TOKEN_MAX {
        // The token is pure ASCII by construction, so byte and char
        // indices agree. The fold is deterministic within a process —
        // all distinctness needs (two ids, one comparison).
        let digest = tail_fold(&token);
        let mut folded = token[..SAFE_TOKEN_MAX - digest.len()].to_string();
        folded.push_str(&digest);
        return folded;
    }
    token
}

/// A short hex fold of the full `token`, spliced into the tail when a
/// model id outgrows the contract's `safeToken` bound: enough bits
/// that distinct ids stay distinct in practice, short enough to keep
/// most of the readable prefix.
fn tail_fold(token: &str) -> String {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    token.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

impl EngineProvider {
    /// Waits (bounded by `deadline`, cancel-aware) for a lease on a ready
    /// engine other than `avoid` — the `(endpoint, pid)` of an engine a
    /// request just failed against, which the manager may still report
    /// Ready until its supervisor notices it is gone. The identity comes
    /// from the lease itself (the engine it holds), never from a
    /// separately-locked snapshot that could already have moved on.
    fn wait_for_lease(
        &self,
        deadline: Instant,
        cancel: &CancelToken,
        avoid: Option<&(String, u32)>,
    ) -> Result<EngineLease, ProviderOutcome> {
        loop {
            if cancel.is_cancelled() {
                return Err(ProviderOutcome::Failed {
                    reason: "cancelled".to_string(),
                    retryable: false,
                });
            }
            if self.stopped.load(Ordering::SeqCst) {
                // The host switched away from this engine: the take is
                // durable and a retry routes to the new choice.
                return Err(ProviderOutcome::Failed {
                    reason: "engine_unavailable".to_string(),
                    retryable: true,
                });
            }
            let mut avoiding = false;
            if let Some(lease) = self.manager.lease() {
                let identity = (lease.endpoint().to_string(), lease.pid());
                if avoid != Some(&identity) {
                    return Ok(lease);
                }
                avoiding = true;
            }
            let phase = self.manager.snapshot().phase;
            // A failed engine or a missing model will not fix itself
            // while this job waits (both need the user): fail now.
            let settled = matches!(phase, EnginePhase::NoModel | EnginePhase::Failed(_));
            if settled || Instant::now() >= deadline {
                return Err(ProviderOutcome::Failed {
                    reason: unready_reason(&phase).to_string(),
                    retryable: true,
                });
            }
            // While the only ready engine is the one being avoided,
            // back off: every `lease()` is lease-marker file I/O on an
            // engine this job will not use again, so poll it gently
            // until the supervisor replaces it.
            std::thread::sleep(if avoiding { AVOID_POLL } else { READY_POLL });
        }
    }
}

impl TranscriptionProvider for EngineProvider {
    fn recognize(
        &self,
        wav: Vec<u8>,
        request_id: &str,
        _on_partial: &mut dyn FnMut(Partial),
        cancel: &CancelToken,
    ) -> ProviderOutcome {
        // Counted at the slot that fronts this provider (see the struct
        // doc): a mode switch that swaps the slot out waits (bounded) on
        // that count before it stops the engine under this call.
        let wav = Arc::new(wav);
        let mut failed_engine: Option<(String, u32)> = None;
        // One ready-wait deadline for the whole call: the initial wait
        // and the crash-retry wait share it, so a job never waits more
        // than `ready_wait` in total — the documented bound — and the
        // retry waits only for what is left of it.
        let deadline = Instant::now() + self.ready_wait;
        loop {
            let lease = match self.wait_for_lease(deadline, cancel, failed_engine.as_ref()) {
                Ok(found) => found,
                Err(outcome) => return outcome,
            };
            let client = match StarlingClient::new(lease.endpoint(), lease.slug()) {
                Ok(client) => client,
                Err(error) => {
                    let (reason, retryable) = failure_from_client_error(&error);
                    return ProviderOutcome::Failed { reason, retryable };
                }
            };
            // The clock starts when the request is issued on a held
            // lease, not when the job arrived: `timing_ms` measures the
            // recognition, not the engine's warm-up or a crash-retry
            // wait.
            let started = Instant::now();
            // The lease is held for the whole request: a model switch that
            // starts mid-request drains this engine instead of stopping it
            // under the take (#363). It drops at the end of this iteration.
            match client.transcribe_with_cancel(Arc::clone(&wav), request_id, Some(cancel)) {
                Ok(result) => {
                    return ProviderOutcome::Completed {
                        text: result.text,
                        backend: backend_token(lease.model_id()),
                        timing_ms: started.elapsed().as_secs_f64() * 1000.0,
                        completion_evidence: "final_decode".to_string(),
                    }
                }
                // The engine went away under the request: its process
                // died (a crash, or — when this host attached to a sidecar
                // the desktop app started — the app being killed, which
                // takes that sidecar with it). The supervisor restarts or
                // takes the engine over; this job retries **once** on the
                // replacement, so the engine's death costs the job a
                // delay, not its result. Recognition is idempotent (same
                // audio, a new request), and one retry bounds the cost of
                // an engine that keeps dying.
                Err(ClientError::Transport(_))
                    if failed_engine.is_none() && !cancel.is_cancelled() =>
                {
                    failed_engine = Some((lease.endpoint().to_string(), lease.pid()));
                }
                Err(error) => {
                    let (reason, retryable) = failure_from_client_error(&error);
                    return ProviderOutcome::Failed { reason, retryable };
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_tokens_fit_the_contract_pattern() {
        assert_eq!(backend_token("parakeet-v3-q8"), "engine:parakeet-v3-q8");
        assert_eq!(backend_token("we ird/id"), "engine:we_ird_id");
        // Long ids fold their tail instead of silently truncating: the
        // token still fits the 128-char bound, two ids sharing the
        // whole readable prefix stay distinct, and every character
        // stays inside the safeToken alphabet.
        let long = backend_token(&"x".repeat(300));
        let sibling = backend_token(&format!("{}y", "x".repeat(300)));
        assert_eq!(long.len(), 128);
        assert_eq!(sibling.len(), 128);
        assert_ne!(long, sibling, "distinct ids must not collapse to one token");
        assert!(long.starts_with("engine:"));
        for token in [long, sibling] {
            assert!(token
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || "_.:+-".contains(c)));
        }
    }

    #[test]
    fn echoed_endpoints_redact_userinfo_and_token_params() {
        for (endpoint, echoed) in [
            ("http://127.0.0.1:8181", "http://127.0.0.1:8181"),
            (
                "https://user:hunter2@example.com:8443/v1",
                "https://***@example.com:8443/v1",
            ),
            ("http://token@example.com", "http://***@example.com"),
            // The last `@` ends the userinfo; an `@` in the path is not one.
            ("http://a@b:c@example.com/x@y", "http://***@example.com/x@y"),
            ("user:pw@example.com:8181", "***@example.com:8181"),
            // Authorities a lenient URL parser still finds.
            (
                "http:///user:secret@example.com/v1",
                "http:///***@example.com/v1",
            ),
            (
                "http:/user:secret@example.com/v1",
                "http:/***@example.com/v1",
            ),
            (
                "http:\\\\u:p@example.com\\v1",
                "http:\\\\***@example.com\\v1",
            ),
            ("http:user:secret@example.com", "***@example.com"),
            // Normalized as the client's URL parser normalizes.
            (
                " http://user:secret@example.com\n",
                "http://***@example.com",
            ),
            ("ht\ttp://user:secret@example.com", "http://***@example.com"),
            (
                "http://example.com/?to\tken=secret",
                "http://example.com/?token=***",
            ),
            // Percent-encoded keys are classified decoded.
            (
                "http://example.com/?%74oken=secret&api_%6bey=k&%zz=1",
                "http://example.com/?%74oken=***&api_%6bey=***&%zz=1",
            ),
            (
                "http://example.com/v1?model=small&access_token=abc&X-Api-Key=k&sig=s",
                "http://example.com/v1?model=small&access_token=***&X-Api-Key=***&sig=***",
            ),
            (
                "http://u:p@example.com?token=abc#frag",
                "http://***@example.com?token=***#frag",
            ),
            (
                "http://example.com/#a?token=x",
                "http://example.com/#a?token=x",
            ),
            (
                "http://example.com/?flag&password",
                "http://example.com/?flag&password",
            ),
            // Scheme-relative endpoints still have an authority.
            ("//user:secret@example.com/v1", "//***@example.com/v1"),
            ("\\\\user:secret@example.com", "\\\\***@example.com"),
            ("not a url", "not a url"),
            ("", ""),
        ] {
            assert_eq!(redact_endpoint(endpoint), echoed, "{endpoint}");
        }
        assert_eq!(
            manual_label("http://example.com/?api_key=abc"),
            "manual:http://example.com/?api_key=***"
        );
        assert_eq!(
            manual_label("http://example.com/?to\tken=abc"),
            "manual:http://example.com/?token=***"
        );
    }

    #[test]
    fn unready_reasons_name_what_the_user_must_fix() {
        assert_eq!(unready_reason(&EnginePhase::NoModel), "engine_no_model");
        assert_eq!(unready_reason(&EnginePhase::Loading), "engine_not_ready");
        assert_eq!(
            unready_reason(&EnginePhase::Failed(
                starling_dictation::engine::EngineFailure::NoBundledEngine
            )),
            "engine_unavailable"
        );
    }

    #[test]
    fn manual_and_builtin_settings_resolve_to_their_engines() {
        let mut settings = Settings::default_settings();
        settings.engine.mode = EngineMode::Manual;
        settings.endpoint = "http://127.0.0.1:9999".into();
        match EngineChoice::from_settings(&settings) {
            EngineChoice::Manual { endpoint, model } => {
                assert_eq!(endpoint, "http://127.0.0.1:9999");
                assert_eq!(model, settings.model);
            }
            other => panic!("expected manual, got {other:?}"),
        }
        settings.engine.mode = EngineMode::Builtin;
        settings.engine.active_model = Some("parakeet-v3-q8".into());
        settings.engine.backend_override = Some("cpu".into());
        // default_paths needs a data dir; every CI runner has one.
        match EngineChoice::from_settings(&settings) {
            EngineChoice::Builtin {
                config,
                active_model,
            } => {
                assert_eq!(active_model.as_deref(), Some("parakeet-v3-q8"));
                assert_eq!(config.backend_override, Some(Backend::Cpu));
            }
            other => panic!("expected builtin, got {other:?}"),
        }
    }

    /// A provider that parks in `recognize` until the test releases it,
    /// reporting which slot served the call.
    struct GatedProvider {
        started: std::sync::mpsc::Sender<()>,
        gate: Mutex<std::sync::mpsc::Receiver<()>>,
        tag: &'static str,
    }

    impl TranscriptionProvider for GatedProvider {
        fn recognize(
            &self,
            _wav: Vec<u8>,
            _request_id: &str,
            _on_partial: &mut dyn FnMut(Partial),
            _cancel: &CancelToken,
        ) -> ProviderOutcome {
            let _ = self.started.send(());
            let _ = self.gate.lock().unwrap().recv();
            ProviderOutcome::Completed {
                text: self.tag.to_string(),
                backend: self.tag.to_string(),
                timing_ms: 0.0,
                completion_evidence: "final_decode".to_string(),
            }
        }
    }

    /// The in-flight rule (#220): a settings change mid-recognition
    /// moves the next job, never the running one — the call finishes on
    /// (and reports) the provider it captured at its start.
    #[test]
    fn a_recognition_finishes_on_the_provider_it_started_with() {
        let (first_started, first_started_rx) = std::sync::mpsc::channel();
        let (release_first, first_gate) = std::sync::mpsc::channel();
        let first = Arc::new(GatedProvider {
            started: first_started,
            gate: Mutex::new(first_gate),
            tag: "first",
        });
        let (second_started, second_started_rx) = std::sync::mpsc::channel();
        let (release_second, second_gate) = std::sync::mpsc::channel();
        let second = Arc::new(GatedProvider {
            started: second_started,
            gate: Mutex::new(second_gate),
            tag: "second",
        });

        let settings = Arc::new(SettingsProvider::new(
            Arc::clone(&first) as Arc<dyn TranscriptionProvider>,
            "first".to_string(),
            None,
        ));
        let runner = {
            let settings = Arc::clone(&settings);
            std::thread::spawn(move || {
                settings.recognize(
                    vec![0u8; 64],
                    "job-x",
                    &mut |_partial| {},
                    &CancelToken::new(),
                )
            })
        };
        first_started_rx
            .recv()
            .expect("the first provider is running the job");

        // The settings move on while the job runs.
        settings.install(
            Arc::clone(&second) as Arc<dyn TranscriptionProvider>,
            "second".to_string(),
            None,
        );
        assert_eq!(settings.label(), "second");

        release_first.send(()).expect("release the first provider");
        match runner.join().expect("the recognition finished") {
            ProviderOutcome::Completed { backend, .. } => assert_eq!(backend, "first"),
            ProviderOutcome::Failed { .. } => {
                panic!("expected a completion on the first provider")
            }
        }
        // The replacement never saw the running job.
        assert!(
            second_started_rx.try_recv().is_err(),
            "the second provider must not serve the job that started on the first"
        );
        drop(release_second);
    }

    /// An unusable manual slot fails jobs honestly instead of guessing
    /// at the user's settings.
    #[test]
    fn an_unconfigured_slot_fails_no_provider_configured() {
        let settings = SettingsProvider::new(
            Arc::new(UnconfiguredProvider) as Arc<dyn TranscriptionProvider>,
            "unconfigured".to_string(),
            None,
        );
        match settings.recognize(
            vec![0u8; 64],
            "job-y",
            &mut |_partial| {},
            &CancelToken::new(),
        ) {
            ProviderOutcome::Failed { reason, retryable } => {
                assert_eq!(reason, "no_provider_configured");
                assert!(!retryable);
            }
            ProviderOutcome::Completed { .. } => {
                panic!("expected the honest failure, got a completion")
            }
        }
        assert_eq!(settings.label(), "unconfigured");
    }

    /// The effective engine label (review on #220): a manual endpoint
    /// that does not validate reads as `unconfigured` — at startup and
    /// after a settings change — so the status line never claims a
    /// manual engine serves while jobs fail `no_provider_configured`.
    /// A later change to a usable endpoint installs it.
    #[test]
    fn an_unusable_manual_endpoint_reports_unconfigured() {
        let mut runtime = starling_runtime::RuntimeConfig::default();
        let host = attach(
            EngineChoice::Manual {
                endpoint: "not a valid endpoint".into(),
                model: "parakeet".into(),
            },
            &mut runtime,
        )
        .expect("manual mode attaches");
        assert_eq!(host.label(), "unconfigured");
        // A take started now hears that its server is the problem, not a
        // built-in engine the user did not choose.
        let notice = host.bind_now().err().expect("nothing serves");
        assert!(notice.contains("Your server's endpoint"), "{notice}");
        assert!(!notice.contains("built-in"), "{notice}");

        // The follow path keeps the same honesty: another unusable
        // endpoint (a scheme the client rejects) stays unconfigured.
        apply(&host, manual("ftp://127.0.0.1:1"));
        assert_eq!(host.label(), "unconfigured");

        // And a usable endpoint takes over.
        apply(&host, manual("http://127.0.0.1:8181"));
        assert_eq!(host.label(), "manual:http://127.0.0.1:8181");
        host.shutdown();
    }

    /// The user's own server at `endpoint`.
    fn manual(endpoint: &str) -> EngineIntent {
        EngineIntent {
            mode: EngineMode::Manual,
            active_model: None,
            backend_override: None,
            endpoint: endpoint.into(),
            model: "parakeet".into(),
        }
    }

    /// The built-in engine with no model.
    fn builtin() -> EngineIntent {
        EngineIntent {
            mode: EngineMode::Builtin,
            ..manual("")
        }
    }

    /// Brings `host` to `intent` as a first read of the settings file
    /// would (every field).
    fn apply(host: &EngineHost, intent: EngineIntent) {
        host.transition(&intent, Fields::ALL).expect("the transition applies");
    }

    /// An engine-less config over a temp dir (no engine staged, no
    /// models): the supervisor sits at `NoModel`, so mode transitions
    /// can be driven without spawning anything.
    fn detached_config(root: &std::path::Path) -> EngineConfig {
        EngineConfig {
            engine_dir: Some(root.join("engines")),
            models_dir: root.join("models"),
            state_dir: root.join("engine-state"),
            catalog: Vec::new(),
            backend_override: None,
            icd_dirs: Some(Vec::new()),
            available_memory_override: Some(None),
            backoff_schedule: Some(vec![Duration::from_millis(100)]),
        }
    }

    /// The mode transitions while the host runs (#220): manual→builtin
    /// starts a supervisor, builtin→manual swaps the provider back and
    /// stops the engine.
    #[test]
    fn apply_moves_the_live_host_between_manual_and_builtin() {
        let root = tempfile::tempdir().unwrap();
        let mut runtime = starling_runtime::RuntimeConfig::default();
        let host = attach_with_paths(
            EngineChoice::Manual {
                endpoint: "http://127.0.0.1:8181".into(),
                model: "parakeet".into(),
            },
            Some(detached_config(root.path())),
            &mut runtime,
        )
        .expect("manual mode attaches");
        assert_eq!(host.label(), "manual:http://127.0.0.1:8181");
        assert!(host.manager().is_none(), "manual mode runs no engine");

        // manual → builtin: a supervisor starts and the runtime routes
        // to it.
        apply(&host, builtin());
        assert_eq!(host.label(), "builtin");
        assert!(
            host.manager().is_some(),
            "builtin mode supervises an engine"
        );

        // builtin → manual: the provider swaps (the runtime would route
        // manual from here) and the engine stops.
        apply(&host, manual("http://127.0.0.1:9192"));
        assert_eq!(host.label(), "manual:http://127.0.0.1:9192");
        assert!(
            host.manager().is_none(),
            "the mode switch stopped the engine"
        );

        // A manual endpoint change rebuilds the provider in place.
        apply(&host, manual("http://127.0.0.1:9193"));
        assert_eq!(host.label(), "manual:http://127.0.0.1:9193");

        // An unchanged choice is a no-op, and shutdown (manual mode)
        // stops nothing.
        apply(&host, manual("http://127.0.0.1:9193"));
        assert_eq!(host.label(), "manual:http://127.0.0.1:9193");
        host.shutdown();
    }

    /// The file-follow rule (#220): a request moves the engine before
    /// the app has written the file, so a later write that leaves the
    /// engine fields as they were (other settings, or a window that has
    /// not seen the change) must not drag it back — only what the file
    /// changed is carried over.
    #[test]
    fn a_file_write_that_leaves_the_engine_alone_does_not_undo_a_request() {
        let root = tempfile::tempdir().unwrap();
        let mut runtime = starling_runtime::RuntimeConfig::default();
        let host = attach_with_paths(
            EngineChoice::Manual {
                endpoint: "http://127.0.0.1:8181".into(),
                model: "parakeet".into(),
            },
            Some(detached_config(root.path())),
            &mut runtime,
        )
        .expect("manual mode attaches");
        host.follow_file(manual("http://127.0.0.1:8181"));
        let before = host.status().revision;

        // The app's request: another server.
        let reply = host.handle(EngineRequest::Configure {
            intent: manual("http://127.0.0.1:9201"),
        });
        assert!(
            matches!(reply, EngineReply::Done { revision } if revision > before),
            "{reply:?}"
        );
        assert_eq!(host.label(), "manual:http://127.0.0.1:9201");

        // A write that still names the old server (unchanged since the
        // last read) leaves the engine where the request put it.
        host.follow_file(manual("http://127.0.0.1:8181"));
        assert_eq!(host.label(), "manual:http://127.0.0.1:9201");

        // A write that changes it is followed.
        host.follow_file(manual("http://127.0.0.1:9202"));
        assert_eq!(host.label(), "manual:http://127.0.0.1:9202");

        // A changed mode is followed too, and the status says so.
        host.follow_file(builtin());
        let status = host.status();
        assert_eq!(status.mode, EngineMode::Builtin);
        assert!(status.snapshot.is_some());
        // Requests the built-in engine cannot serve in manual mode are
        // refused with a sentence, not dropped.
        host.follow_file(manual("http://127.0.0.1:9202"));
        match host.handle(EngineRequest::Activate {
            model_id: "parakeet".into(),
        }) {
            EngineReply::Refused { message } => assert!(message.contains("built-in engine is off")),
            other => panic!("{other:?}"),
        }
        host.shutdown();
    }

    /// Re-applying the same builtin choice runs no deltas — the
    /// property the watcher's unseeded first poll relies on: applying
    /// what the file already said must not start a switch (same model
    /// and backend ⇒ no activate, no backend reload).
    #[test]
    fn an_unchanged_builtin_choice_is_a_no_op() {
        let root = tempfile::tempdir().unwrap();
        let config = detached_config(root.path());
        let mut runtime = starling_runtime::RuntimeConfig::default();
        let host = attach(
            EngineChoice::Builtin {
                config: config.clone(),
                active_model: None,
            },
            &mut runtime,
        )
        .expect("builtin mode attaches");
        let manager = host.manager().expect("builtin mode supervises an engine");
        assert!(manager.snapshot().switch.is_none());

        apply(&host, builtin());

        assert!(
            manager.snapshot().switch.is_none(),
            "an unchanged choice must not start a switch (the watcher's first poll relies on it)"
        );
        assert_eq!(host.label(), "builtin");
        host.shutdown();
    }

    /// The drain rule (review on #220): a recognition that entered the
    /// slot before a builtin→manual switch keeps that switch's drain
    /// waiting until the call returns — the count is raised under the
    /// slot lock, before the switch can swap the slot out, so the
    /// switch never sees zero and stops the engine under the worker.
    #[test]
    fn a_switch_drains_a_recognition_captured_before_it() {
        let root = tempfile::tempdir().unwrap();
        let mut runtime = starling_runtime::RuntimeConfig::default();
        let host = attach(
            EngineChoice::Builtin {
                config: detached_config(root.path()),
                active_model: None,
            },
            &mut runtime,
        )
        .expect("builtin mode attaches");
        // The counter the switch will drain on — the one the builtin
        // slot carries (the real engine provider is swapped for a
        // gated one on the SAME counter, so the drain observes this
        // call instead of a real engine request).
        let in_flight = match &*lock(&host.state) {
            EngineState::Builtin { in_flight, .. } => Arc::clone(in_flight),
            EngineState::Manual { .. } | EngineState::Unavailable { .. } => {
                panic!("the host started builtin")
            }
        };
        let (started, started_rx) = std::sync::mpsc::channel();
        let (release, gate) = std::sync::mpsc::channel();
        host.provider.install(
            Arc::new(GatedProvider {
                started,
                gate: Mutex::new(gate),
                tag: "gated",
            }),
            "builtin".to_string(),
            Some(Arc::clone(&in_flight)),
        );

        let runner = {
            let provider = Arc::clone(&host.provider);
            std::thread::spawn(move || {
                provider.recognize(
                    vec![0u8; 64],
                    "job-d",
                    &mut |_partial| {},
                    &CancelToken::new(),
                )
            })
        };
        started_rx
            .recv()
            .expect("the gated provider is running the job");
        // Raised before the swap, held for the whole call.
        assert_eq!(in_flight.load(Ordering::SeqCst), 1);

        // The switch to manual: it swaps the slot, then drains.
        let applier = {
            let host = Arc::clone(&host);
            std::thread::spawn(move || {
                apply(&host, manual("http://127.0.0.1:9194"))
            })
        };
        let deadline = Instant::now() + Duration::from_secs(5);
        while host.label() != "manual:http://127.0.0.1:9194" {
            assert!(
                Instant::now() < deadline,
                "the slot never swapped (still {})",
                host.label()
            );
            std::thread::sleep(FOLLOW_SLICE);
        }
        // The slot has moved on, but the switch has not finished: it is
        // draining the captured recognition (well inside the 10 s
        // grace, which a finished switch would not wait out).
        std::thread::sleep(Duration::from_millis(300));
        assert!(
            !applier.is_finished(),
            "the switch must still be waiting for the in-flight recognition"
        );

        release.send(()).expect("release the gated provider");
        match runner.join().expect("the recognition finished") {
            ProviderOutcome::Completed { backend, .. } => assert_eq!(backend, "gated"),
            ProviderOutcome::Failed { .. } => {
                panic!("expected a completion on the gated provider")
            }
        }
        assert_eq!(in_flight.load(Ordering::SeqCst), 0);
        applier.join().expect("the switch finishes once drained");
        assert!(
            host.manager().is_none(),
            "the switch stopped the engine after the drain"
        );
        host.shutdown();
    }
}
