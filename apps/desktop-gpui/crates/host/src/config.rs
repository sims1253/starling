//! The service host's configuration: where it lives (data root, endpoint
//! directory), the transport's per-connection limits, the peer-auth
//! policy, and the [`RuntimeConfig`] it boots the runtime with.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use starling_runtime::machine::capture::V2CaptureStore;
use starling_runtime::RuntimeConfig;

use crate::auth::PeerPolicy;
use crate::engine::EngineChoice;
use crate::limits::RateLimit;
use crate::platform;

/// Everything the host needs to launch. Field-by-field:
///
/// - [`HostConfig::data_root`] is the storage v2 root the host **must**
///   hold the lease on. A host without a data root has no ownership
///   semantics, so unlike [`RuntimeConfig::default`] (whose in-memory
///   store keeps test construction side-effect-free) the host refuses to
///   degrade: opening or leasing the root is what makes it the owner.
/// - [`HostConfig::runtime_dir`] holds the IPC endpoint; the endpoint
///   name itself derives from the data root (see [`platform`]).
pub struct HostConfig {
    /// The storage v2 root. The host acquires this root's runtime lease
    /// (§4 ownership) before binding anything.
    pub data_root: PathBuf,
    /// Directory the IPC endpoint lives in (0700).
    pub runtime_dir: PathBuf,
    /// Per-connection frame cap (both directions).
    pub max_frame_bytes: usize,
    /// Per-connection client→host frame budget.
    pub command_rate: RateLimit,
    /// Maximum simultaneous connections; past this the host answers
    /// `too_many_connections` and closes.
    pub max_connections: usize,
    /// Per-connection outbound event queue before `slow_consumer`.
    pub outbound_capacity: usize,
    /// How long a freshly-admitted connection may hold its slot without
    /// sending a frame — the pre-greeting idle bound (see
    /// `server::connection_reader` for the posture). A field so tests
    /// tighten it instead of waiting out the production deadline.
    pub first_frame_idle: std::time::Duration,
    /// Who may connect.
    pub peer_policy: Arc<dyn PeerPolicy>,
    /// The settings file the host follows while it runs (see
    /// [`crate::engine`]): when set, engine-setting changes apply to the
    /// serving host without a restart. `None` — the default, tests and
    /// `--engine none` — fixes the engine at [`HostConfig::engine`].
    pub settings_path: Option<PathBuf>,
    /// How often the settings watcher polls [`HostConfig::settings_path`]
    /// ([`crate::engine::DEFAULT_SETTINGS_POLL`] in production; tests
    /// tighten it).
    pub settings_poll: Duration,
    /// Which transcription engine the host attaches to the runtime once
    /// it owns the root (see [`crate::engine`]). [`EngineChoice::None`]
    /// leaves `runtime.provider` as configured — tests inject doubles
    /// there.
    pub engine: EngineChoice,
    /// The paths and catalog the built-in engine runs on whenever this
    /// host starts one after startup (a switch to builtin mode). `None`:
    /// the startup choice's, else the default data paths. Tests point it
    /// at temp dirs.
    pub engine_paths: Option<starling_dictation::engine::EngineConfig>,
    /// This host's build, as the version handshake compares it
    /// ([`crate::version`]); [`BuildStamp::current`] unless a test
    /// plays another build.
    ///
    /// [`BuildStamp::current`]: crate::version::BuildStamp::current
    pub build: crate::version::BuildStamp,
    /// How long a recording take may go without any app following it
    /// before the host stops and stores it itself (see [`crate::takes`]).
    pub orphan_grace: Duration,
    /// How long a transcription waits for an engine to serve.
    pub engine_wait: Duration,
    /// When the history audio upkeep first runs after startup, and how
    /// often after that ([`crate::history::History::upkeep_loop`]).
    pub upkeep_first: Duration,
    pub upkeep_interval: Duration,
    /// The agent allowlist file (see [`crate::agent::Allowlist`]).
    /// `None` or a missing file denies every agent client; a malformed
    /// file refuses startup.
    pub agent_allowlist: Option<PathBuf>,
    /// Treats every non-agent connection as the Starling app, which may
    /// show and answer prompts. No app-role credential exists yet, so
    /// builds without the `test-support` feature have no app and every
    /// ask fails with `no_app`.
    #[cfg(feature = "test-support")]
    pub insecure_test_app_role: bool,
    /// The runtime this host owns. Production builds pass
    /// [`HostConfig::production`]; tests inject doubles through the same
    /// builders [`RuntimeConfig`] offers.
    pub runtime: RuntimeConfig,
}

impl HostConfig {
    /// A test/development host at `data_root`: transport defaults, the
    /// default peer policy, and a [`RuntimeConfig::default`] whose
    /// capture store is **in-memory** — constructing a config never
    /// touches user data (the same philosophy as
    /// [`RuntimeConfig::default`]). Tests that want persistence pass
    /// `.runtime.with_capture_store(Arc::new(V2CaptureStore::open(root)?))`.
    pub fn new(data_root: impl Into<PathBuf>, runtime_dir: impl Into<PathBuf>) -> HostConfig {
        HostConfig {
            data_root: data_root.into(),
            runtime_dir: runtime_dir.into(),
            max_frame_bytes: crate::frame::DEFAULT_MAX_FRAME_BYTES,
            command_rate: RateLimit::default(),
            max_connections: crate::limits::DEFAULT_MAX_CONNECTIONS,
            outbound_capacity: crate::limits::DEFAULT_OUTBOUND_CAPACITY,
            first_frame_idle: Duration::from_secs(10),
            peer_policy: crate::auth::default_policy(),
            settings_path: None,
            settings_poll: crate::engine::DEFAULT_SETTINGS_POLL,
            engine: EngineChoice::None,
            engine_paths: None,
            build: crate::version::BuildStamp::current(),
            orphan_grace: crate::takes::DEFAULT_ORPHAN_GRACE,
            engine_wait: crate::transcribe::DEFAULT_ENGINE_WAIT,
            upkeep_first: crate::history::UPKEEP_FIRST,
            upkeep_interval: crate::history::UPKEEP_INTERVAL,
            agent_allowlist: None,
            #[cfg(feature = "test-support")]
            insecure_test_app_role: false,
            runtime: RuntimeConfig::default(),
        }
    }

    /// The production host at `data_root`: storage v2 is THE capture
    /// persistence (D14 — no v1 seam, no second backend) **and** the
    /// documents machine's persistence (I5 wiring, issue #220:
    /// [`V2DocumentStore`] over the same root's `documents`/`revisions`
    /// tables, its own SQLite connection like the capture store's). The
    /// endpoint directory is `runtime_dir` when given (else the per-user
    /// default), and only that final directory is created here — so a
    /// missing `$XDG_RUNTIME_DIR` surfaces at launch, not at first bind,
    /// and an overridden dir never leaves the default behind as stray
    /// residue. The inference engine is chosen separately
    /// ([`HostConfig::with_engine`]; the binary passes the user's
    /// settings) — left alone, jobs fail `no_provider_configured` rather
    /// than inventing an endpoint; the context and
    /// delivery adapters likewise stay the honest stubs until E03's
    /// platform adapters (issue #221) plug into the `RuntimeConfig`
    /// seams.
    ///
    /// Errors when the root cannot open (the host is the designed lease
    /// acquirer; a root it cannot open is a host it must not be).
    pub fn production(
        data_root: impl Into<PathBuf>,
        runtime_dir: Option<PathBuf>,
    ) -> Result<HostConfig, String> {
        let data_root: PathBuf = data_root.into();
        let runtime_dir = runtime_dir.unwrap_or_else(platform::default_runtime_dir);
        platform::ensure_runtime_dir(&runtime_dir).map_err(|err| {
            format!(
                "cannot create the runtime endpoint directory {}: {err}",
                runtime_dir.display()
            )
        })?;
        let store = V2CaptureStore::open(&data_root)
            .map_err(|err| format!("capture store at {} will not open: {err}", data_root.display()))?;
        let documents = starling_runtime::machine::docs::V2DocumentStore::open(&data_root)
            .map_err(|err| format!("documents store at {} will not open: {err}", data_root.display()))?;
        let mut config = HostConfig::new(&data_root, &runtime_dir);
        // The recorder journals into the tree beside the store the host
        // recovers from (#220: the app no longer opens takes itself), on
        // the microphone the desktop settings choose.
        let settings_path = starling_dictation::settings::Settings::default_path().ok();
        config.runtime = config
            .runtime
            .with_capture_store(Arc::new(store))
            .with_document_store(Arc::new(documents))
            .with_capture_source(Arc::new(crate::capture::SettingsCaptureSource::new(
                settings_path,
            )))
            .with_capture_config(starling_runtime::machine::capture::CaptureConfig {
                journals_dir: crate::recovery::journals_dir(&data_root),
                ..Default::default()
            });
        Ok(config)
    }

    /// Overrides how long a transcription waits for an engine.
    pub fn with_engine_wait(mut self, wait: Duration) -> Self {
        self.engine_wait = wait;
        self
    }

    /// Overrides how long an unfollowed take records before the host
    /// stops it.
    /// Overrides when the history audio upkeep runs.
    pub fn with_upkeep(mut self, first: Duration, interval: Duration) -> Self {
        self.upkeep_first = first;
        self.upkeep_interval = interval;
        self
    }

    pub fn with_orphan_grace(mut self, grace: Duration) -> Self {
        self.orphan_grace = grace;
        self
    }

    /// Overrides the peer-auth policy (tests inject stand-ins for a
    /// foreign user).
    pub fn with_peer_policy(mut self, policy: Arc<dyn PeerPolicy>) -> Self {
        self.peer_policy = policy;
        self
    }

    /// Sets the engine the host attaches once it owns the root.
    pub fn with_engine(mut self, engine: EngineChoice) -> Self {
        self.engine = engine;
        self
    }

    /// Sets the paths and catalog the built-in engine runs on (see
    /// [`HostConfig::engine_paths`]).
    pub fn with_engine_paths(mut self, paths: starling_dictation::engine::EngineConfig) -> Self {
        self.engine_paths = Some(paths);
        self
    }

    /// Plays another build in the version handshake (tests).
    pub fn with_build(mut self, build: crate::version::BuildStamp) -> Self {
        self.build = build;
        self
    }

    /// Sets the settings file the host follows while it runs (the
    /// `--engine settings` path; the binary passes
    /// `Settings::default_path()`). Without it the startup engine choice
    /// is fixed — the posture every test that injects an
    /// [`EngineChoice`] wants.
    pub fn with_settings_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.settings_path = Some(path.into());
        self
    }

    /// Overrides how often the settings watcher polls the file.
    pub fn with_settings_poll(mut self, poll: Duration) -> Self {
        self.settings_poll = poll;
        self
    }

    /// Sets the agent allowlist file.
    pub fn with_agent_allowlist(mut self, path: Option<PathBuf>) -> Self {
        self.agent_allowlist = path;
        self
    }

    /// See [`HostConfig::insecure_test_app_role`].
    #[cfg(feature = "test-support")]
    pub fn with_insecure_test_app_role(mut self) -> Self {
        self.insecure_test_app_role = true;
        self
    }

    /// Overrides the frame cap.
    pub fn with_max_frame_bytes(mut self, cap: usize) -> Self {
        self.max_frame_bytes = cap;
        self
    }

    /// Overrides the per-connection command rate limit.
    pub fn with_command_rate(mut self, limit: RateLimit) -> Self {
        self.command_rate = limit;
        self
    }

    /// Overrides the outbound event queue depth.
    pub fn with_outbound_capacity(mut self, capacity: usize) -> Self {
        self.outbound_capacity = capacity;
        self
    }

    /// The IPC endpoint path for this host's data root (unix: socket
    /// file under `runtime_dir`; windows: the `\\.\pipe\…` name).
    pub fn socket_path(&self) -> PathBuf {
        platform::socket_path(&self.runtime_dir, &self.data_root)
    }
}

/// A path helper shared by the binary's CLI: the storage v2 default root
/// or a clear error — the host does not silently invent one.
pub fn default_data_root() -> Result<PathBuf, String> {
    starling_dictation::store_v2::StoreV2::default_root()
        .map_err(|err| format!("no storage v2 default data root: {err}"))
}
