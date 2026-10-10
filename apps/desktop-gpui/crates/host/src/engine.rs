//! The supervised engine attaches to the **host**, not the renderer
//! (E17 §1 Mode B, #220): the host owns the bundled-engine
//! [`EngineManager`] (the `starling-serve` sidecar, its crash restarts
//! and model switches, #362/#363), and the runtime's jobs machine
//! reaches it through [`EngineProvider`]. Engine lifetime is host
//! lifetime — the sidecar is spawned with the host's pid as its
//! `--parent-pid`, a renderer that dies mid-job costs nothing, and the
//! host's shutdown stops the engine after the machines have joined.
//!
//! Which engine serves is the user's existing engine choice
//! ([`EngineChoice::from_settings`]): the bundled engine with the
//! persisted model, the user's own server in manual mode, or none. The
//! host reads the same settings file and the same model/state
//! directories the desktop app uses, so the app and the host share one
//! sidecar through the engine registry (one owns it, the other
//! attaches) instead of loading the model twice.
//!
//! # Following the settings while the host runs (#220)
//!
//! The desktop app applies engine changes immediately (its
//! `apply_engine_mode_change` / activate flows), so a host that froze
//! its startup choice would drift from what the user just chose. The
//! host therefore installs [`SettingsProvider`] — a switchable provider
//! — as the runtime's provider and owns an [`EngineHost`] that tracks
//! the live engine state. [`watch_settings`] polls the settings file
//! (path and interval injectable; `Settings::default_path` in
//! production) and [`EngineHost::apply`] carries each change over:
//! `activate` for a new model, a fresh manual provider for an endpoint/model
//! change, a started supervisor for manual→builtin, and a stopped
//! engine for builtin→manual. A changed CPU/automatic backend override
//! is **not** applied live — it takes effect at the host's next start
//! (see [`EngineHost::apply`] for why). An in-flight recognition runs on the
//! provider (and, through it, the engine lease) it started with: the
//! provider and its in-flight count are captured together under the
//! slot lock, and a mode switch drains that count before it stops the
//! engine.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use starling_dictation::client::{ClientError, StarlingClient};
use starling_dictation::engine::{Backend, EngineConfig, EngineLease, EngineManager, EnginePhase};
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
/// the desktop app's engine changes are visible within a poll or two.
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

/// The user's engine choice, resolved for the host.
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
}

impl EngineChoice {
    /// The choice the desktop settings file states, with the app's own
    /// defaults: builtin on the default engine paths with the persisted
    /// model and backend override, or the manual endpoint/model. Errors
    /// only when the builtin engine's data directory cannot resolve.
    pub fn from_settings(settings: &Settings) -> Result<EngineChoice, String> {
        match settings.engine.mode {
            EngineMode::Builtin => {
                let mut config = EngineConfig::default_paths().map_err(|err| err.to_string())?;
                config.backend_override = settings
                    .engine
                    .backend_override
                    .as_deref()
                    .and_then(Backend::parse);
                Ok(EngineChoice::Builtin {
                    config,
                    active_model: settings.engine.active_model.clone(),
                })
            }
            EngineMode::Manual => Ok(EngineChoice::Manual {
                endpoint: settings.endpoint.clone(),
                model: settings.model.clone(),
            }),
        }
    }

    /// The label the host's status line reports.
    pub fn label(&self) -> &'static str {
        match self {
            EngineChoice::None => "none",
            EngineChoice::Builtin { .. } => "builtin",
            EngineChoice::Manual { .. } => "manual",
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
/// changes, shutdown).
pub fn attach(
    choice: EngineChoice,
    runtime: &mut starling_runtime::RuntimeConfig,
) -> Option<Arc<EngineHost>> {
    match choice {
        EngineChoice::None => None,
        EngineChoice::Builtin {
            config,
            active_model,
        } => {
            let host = Arc::new(EngineHost::start_builtin(config, active_model));
            runtime.provider = host.provider_slot();
            Some(host)
        }
        EngineChoice::Manual { endpoint, model } => {
            let host = Arc::new(EngineHost::manual(endpoint, model));
            runtime.provider = host.provider_slot();
            Some(host)
        }
    }
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
    /// holds, so the mode-switch drain in [`EngineHost::apply`] waits
    /// on exactly these calls), `None` for manual/unconfigured slots
    /// (nothing drains behind them).
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

    /// What serves right now — `builtin`, `manual:<endpoint>`, or
    /// `unconfigured` (status lines and tests).
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

/// The host's live engine attachment: the switchable provider the
/// runtime calls through, the engine state it tracks, and the
/// transitions ([`EngineHost::apply`]) that follow the settings file.
/// Shared between the host handle (status, shutdown) and the settings
/// watcher, which is the only writer while the host serves.
pub struct EngineHost {
    provider: Arc<SettingsProvider>,
    state: Mutex<EngineState>,
    /// Set when the host begins shutting down: a builtin→manual drain
    /// in progress on the watcher thread stops waiting, so the host's
    /// shutdown (which joins the watcher) is not held for the drain's
    /// grace. The runtime's own shutdown cancels the drained jobs.
    closing: AtomicBool,
}

/// What the host runs right now; the diff base for the next
/// [`EngineHost::apply`].
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
        active_model: Option<String>,
        backend_override: Option<Backend>,
    },
    /// Manual mode: the endpoint/model the provider was last built from.
    Manual { endpoint: String, model: String },
}

impl EngineHost {
    fn start_builtin(config: EngineConfig, active_model: Option<String>) -> EngineHost {
        let manager = EngineManager::start(config.clone(), active_model.clone());
        // One counter, shared by the slot (which raises it under its
        // lock per call) and the state (whose mode-switch drain waits
        // on it): the engine provider itself stays counter-free.
        let in_flight = Arc::new(AtomicUsize::new(0));
        let provider = EngineProvider::new(manager.clone());
        let stopped = provider.stopped_flag();
        EngineHost {
            provider: Arc::new(SettingsProvider::new(
                Arc::new(provider),
                "builtin".to_string(),
                Some(Arc::clone(&in_flight)),
            )),
            state: Mutex::new(EngineState::Builtin {
                manager,
                in_flight,
                stopped,
                active_model,
                backend_override: config.backend_override,
            }),
            closing: AtomicBool::new(false),
        }
    }

    fn manual(endpoint: String, model: String) -> EngineHost {
        let (provider, label) = manual_slot(&endpoint, &model);
        EngineHost {
            provider: Arc::new(SettingsProvider::new(provider, label, None)),
            state: Mutex::new(EngineState::Manual { endpoint, model }),
            closing: AtomicBool::new(false),
        }
    }

    /// The runtime's provider slot this host fills (`attach` installs
    /// it; nothing else constructs an `EngineHost`).
    fn provider_slot(&self) -> Arc<SettingsProvider> {
        Arc::clone(&self.provider)
    }

    /// What serves right now (`builtin`, `manual:<endpoint>`, or
    /// `unconfigured`) — the live analogue of [`EngineChoice::label`].
    pub fn label(&self) -> String {
        self.provider.label()
    }

    /// The engine manager this host supervises right now (builtin
    /// mode), for status reporting and tests. `None` in manual mode —
    /// including after a live builtin→manual switch.
    pub fn manager(&self) -> Option<EngineManager> {
        match &*lock(&self.state) {
            EngineState::Builtin { manager, .. } => Some(manager.clone()),
            EngineState::Manual { .. } => None,
        }
    }

    /// Carries a settings-resolved engine choice over to the running
    /// host, the host-side twin of the app's immediate engine actions:
    /// a new `activeModel` activates the model, manual endpoint/model changes
    /// rebuild the server provider, manual→builtin starts a
    /// supervisor, builtin→manual stops routing to the engine and (once
    /// in-flight recognitions finished, bounded by
    /// [`ENGINE_DRAIN_GRACE`]) shuts it down. Only the deltas run: an
    /// unchanged choice costs nothing.
    ///
    /// A changed `backendOverride` (the app's "Use CPU engine" toggle)
    /// is recorded and reported, not applied live: it takes effect at
    /// the host's next start. Applying it means
    /// `EngineManager::set_backend_override`, whose reload spawns the
    /// replacement sidecar unshared — and whether this host owns its
    /// engine or is attached to the desktop app's is only known once the
    /// supervisor's startup resolves, *after* a queued command was
    /// accepted. Forwarding from here would therefore let host and app
    /// each end up owning a separate sidecar (two models resident); the
    /// app's own toggle reloads the engine it owns, and an attached host
    /// follows that replacement through its attach poll. A live
    /// host-side reload needs ownership-aware support in the engine
    /// manager itself — recorded as a follow-up, not guessed at here.
    pub fn apply(&self, choice: EngineChoice) {
        let mut state = lock(&self.state);
        match choice {
            EngineChoice::None => {}
            EngineChoice::Builtin {
                config,
                active_model,
            } => match &mut *state {
                EngineState::Builtin {
                    manager,
                    active_model: current_model,
                    backend_override,
                    ..
                } => {
                    if config.backend_override != *backend_override {
                        eprintln!(
                            "starling-runtime-host: the engine backend setting changed; \
                             it applies at the host's next start"
                        );
                        *backend_override = config.backend_override;
                    }
                    if active_model != *current_model {
                        match active_model.as_deref() {
                            Some(model_id) => {
                                manager.activate(model_id);
                            }
                            // The settings stopped naming a model. The
                            // app always persists its active choice, so
                            // this is a hand-edited or foreign file; keep
                            // the engine the running jobs know (the
                            // supervisor has no "serve nothing"
                            // command, and stopping under the user's
                            // takes is the worse failure).
                            None => eprintln!(
                                "starling-runtime-host: the settings name no engine \
                                 model; keeping the running one"
                            ),
                        }
                        *current_model = active_model;
                    }
                }
                EngineState::Manual { .. } => {
                    // manual → builtin: start a supervisor, then aim the
                    // runtime at it (with the counter its future mode
                    // switch will drain on).
                    let manager = EngineManager::start(config.clone(), active_model.clone());
                    let in_flight = Arc::new(AtomicUsize::new(0));
                    let provider = EngineProvider::new(manager.clone());
                    let stopped = provider.stopped_flag();
                    self.provider.install(
                        Arc::new(provider),
                        "builtin".to_string(),
                        Some(Arc::clone(&in_flight)),
                    );
                    *state = EngineState::Builtin {
                        manager,
                        in_flight,
                        stopped,
                        active_model,
                        backend_override: config.backend_override,
                    };
                }
            },
            EngineChoice::Manual { endpoint, model } => {
                let changed = match &*state {
                    EngineState::Manual {
                        endpoint: current,
                        model: current_model,
                    } => current != &endpoint || current_model != &model,
                    EngineState::Builtin { .. } => true,
                };
                if !changed {
                    return;
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
                let (provider, label) = manual_slot(&endpoint, &model);
                self.provider.install(provider, label, None);
                let previous = std::mem::replace(
                    &mut *state,
                    EngineState::Manual {
                        endpoint: endpoint.clone(),
                        model: model.clone(),
                    },
                );
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
            }
        }
    }

    /// The target a take starting now binds to (#363), when something
    /// serves now: a lease on the ready engine, or the manual server.
    /// `None` when nothing does yet (the take records anyway; its
    /// transcription waits for an engine, [`Self::wait_target`]).
    pub fn bind_now(&self) -> Option<Target> {
        match &*lock(&self.state) {
            EngineState::Builtin {
                manager, in_flight, ..
            } => manager
                .lease()
                .map(|lease| Target::from_lease(lease, in_flight)),
            EngineState::Manual { endpoint, model } => Target::manual(endpoint, model).ok(),
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
            let (manager, stopped, in_flight) = match &*lock(&self.state) {
                EngineState::Builtin {
                    manager,
                    stopped,
                    in_flight,
                    ..
                } => (manager.clone(), Arc::clone(stopped), Arc::clone(in_flight)),
                EngineState::Manual { endpoint, model } => {
                    return match want {
                        Want::Current => Target::manual(endpoint, model).map_err(|err| {
                            format!("Your server's endpoint in Settings is not usable: {err}")
                        }),
                        Want::Model(_) => Err("The built-in engine is off (Settings → Engine uses your own server)."
                            .to_string()),
                    };
                }
            };
            if stopped.load(Ordering::SeqCst) {
                return Err(not_ready_sentence());
            }
            let mut avoiding = false;
            if let Some(lease) = manager.lease() {
                let wanted = match want {
                    Want::Current => true,
                    Want::Model(model_id) => lease.model_id() == model_id,
                };
                let identity = (lease.endpoint().to_string(), lease.pid());
                if wanted && avoid != Some(&identity) {
                    return Ok(Target::from_lease(lease, &in_flight));
                }
                avoiding = wanted;
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
        let manager = match &*lock(&self.state) {
            EngineState::Builtin { manager, .. } => Some(manager.clone()),
            EngineState::Manual { .. } => None,
        };
        if let Some(manager) = manager {
            manager.shutdown();
        }
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
/// the bytes between two reads could differ) and applies the engine
/// choice they resolve to ([`EngineChoice::from_settings`], the same
/// load the host started from) to `host`.
///
/// Bytes that are not valid JSON — an empty or truncated file, what a
/// non-atomic writer looks like mid-write — state no choice: the host
/// keeps its last-applied one (reported once per distinct bad content)
/// until the file says something parseable again. A missing file is
/// the same non-event, and a deletion racing the stability re-read
/// must not read as "the user chose the defaults".
///
/// The watcher starts with no `last`: whatever the file says at its
/// first poll is applied, so a change between the startup load (which
/// resolved the host's initial [`EngineChoice`]) and the watcher's
/// start is not silently frozen out. That first apply is a no-op when
/// the file still agrees with the startup choice — `apply` only runs
/// deltas (same model and backend for builtin, same endpoint and model
/// for manual). Stops when `stop` is set (checked every slice), so the
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
                        Some(settings) => match EngineChoice::from_settings(&settings) {
                            Ok(choice) => {
                                // Valid again: a later recurrence of a
                                // bad content is reported anew.
                                reported_bad = None;
                                host.apply(choice);
                            }
                            Err(err) => eprintln!(
                                "starling-runtime-host: engine settings at {}: {err}; \
                                 the engine stays as it is",
                                path.display()
                            ),
                        },
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
        match EngineChoice::from_settings(&settings).unwrap() {
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
        match EngineChoice::from_settings(&settings).unwrap() {
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

        // The follow path keeps the same honesty: another unusable
        // endpoint (a scheme the client rejects) stays unconfigured.
        host.apply(EngineChoice::Manual {
            endpoint: "ftp://127.0.0.1:1".into(),
            model: "parakeet".into(),
        });
        assert_eq!(host.label(), "unconfigured");

        // And a usable endpoint takes over.
        host.apply(EngineChoice::Manual {
            endpoint: "http://127.0.0.1:8181".into(),
            model: "parakeet".into(),
        });
        assert_eq!(host.label(), "manual:http://127.0.0.1:8181");
        host.shutdown();
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
        let host = attach(
            EngineChoice::Manual {
                endpoint: "http://127.0.0.1:8181".into(),
                model: "parakeet".into(),
            },
            &mut runtime,
        )
        .expect("manual mode attaches");
        assert_eq!(host.label(), "manual:http://127.0.0.1:8181");
        assert!(host.manager().is_none(), "manual mode runs no engine");

        // manual → builtin: a supervisor starts and the runtime routes
        // to it.
        host.apply(EngineChoice::Builtin {
            config: detached_config(root.path()),
            active_model: None,
        });
        assert_eq!(host.label(), "builtin");
        assert!(
            host.manager().is_some(),
            "builtin mode supervises an engine"
        );

        // builtin → manual: the provider swaps (the runtime would route
        // manual from here) and the engine stops.
        host.apply(EngineChoice::Manual {
            endpoint: "http://127.0.0.1:9192".into(),
            model: "parakeet".into(),
        });
        assert_eq!(host.label(), "manual:http://127.0.0.1:9192");
        assert!(
            host.manager().is_none(),
            "the mode switch stopped the engine"
        );

        // A manual endpoint change rebuilds the provider in place.
        host.apply(EngineChoice::Manual {
            endpoint: "http://127.0.0.1:9193".into(),
            model: "parakeet".into(),
        });
        assert_eq!(host.label(), "manual:http://127.0.0.1:9193");

        // An unchanged choice is a no-op, and shutdown (manual mode)
        // stops nothing.
        host.apply(EngineChoice::Manual {
            endpoint: "http://127.0.0.1:9193".into(),
            model: "parakeet".into(),
        });
        assert_eq!(host.label(), "manual:http://127.0.0.1:9193");
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

        host.apply(EngineChoice::Builtin {
            config,
            active_model: None,
        });

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
            EngineState::Manual { .. } => panic!("the host started builtin"),
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
                host.apply(EngineChoice::Manual {
                    endpoint: "http://127.0.0.1:9194".into(),
                    model: "parakeet".into(),
                })
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
