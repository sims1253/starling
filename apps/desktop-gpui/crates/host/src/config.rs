//! The service host's configuration: where it lives (data root, endpoint
//! directory), the transport's per-connection limits, the peer-auth
//! policy, and the [`RuntimeConfig`] it boots the runtime with.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use starling_runtime::machine::capture::V2CaptureStore;
use starling_runtime::RuntimeConfig;

use crate::auth::PeerPolicy;
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
    /// residue. The inference provider stays unconfigured — an honest
    /// default that fails jobs with `no_provider_configured` rather than
    /// inventing an endpoint — until a configuration surface (app
    /// settings or the E03 engine attach) supplies one; the context and
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
        config.runtime = config
            .runtime
            .with_capture_store(Arc::new(store))
            .with_document_store(Arc::new(documents));
        Ok(config)
    }

    /// Overrides the peer-auth policy (tests inject stand-ins for a
    /// foreign user).
    pub fn with_peer_policy(mut self, policy: Arc<dyn PeerPolicy>) -> Self {
        self.peer_policy = policy;
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
