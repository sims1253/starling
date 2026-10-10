//! The service host (E17 I4, §1 Mode B): the one process per user
//! session that owns the runtime, the storage-v2 lease, and the IPC
//! endpoint.
//!
//! Ownership ladder, in the order [`serve`] walks it — "never two
//! owners" is the invariant every step protects:
//!
//! 1. **Lease** (§4): `StoreV2::acquire_lease` on the data root. A live
//!    foreign owner answers [`HostError::OwnerLive`] and this process is
//!    a client — it must not serve (the binary then reports the owner's
//!    endpoint and exits 0). A dead owner's lease breaks via the lease
//!    machinery (flock released at process death, heartbeat TTL
//!    otherwise). With the lease held, the host runs
//!    `StoreV2::reconcile` in owner mode: it is the recovering owner the
//!    client-mode deferral names, so a crashed predecessor's staging
//!    journals and orphan sessions are salvaged at startup (the report is
//!    logged and kept on the handle).
//! 2. **Endpoint**: with the lease held, probe the endpoint. Live →
//!    [`HostError::ForeignServer`] (a server answering without holding
//!    the lease is a contradiction; refuse rather than fight). Dead →
//!    remove the stale residue (a killed host's socket file) and bind.
//! 3. **Runtime**: `Runtime::start` with the host's config — the
//!    capture machine's store is the host's storage-v2 store, workers
//!    are the runtime's supervised provider workers, and nothing in the
//!    runtime knows any renderer exists.
//!
//! Per connection: authenticate (peer credentials → [`crate::auth`]),
//! greet ([`Frame::Hello`] carrying the limits), then a reader thread
//! (frames in, receipts/snapshot replies out) and a writer thread
//! (serialized writes from a bounded outbound queue). One event pump
//! fans every runtime event to every live connection — bounded per
//! connection, because the runtime must never stall on one renderer: a
//! client that stops reading is closed with `slow_consumer` and
//! reconnects via snapshot + fresh events, while acknowledged audio sits
//! in storage v2 untouched (renderer-kill acceptance).

use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use starling_dictation::store_v2::{
    LeaseAcquisition, ReconciliationReport, StoreV2, LEASE_HEARTBEAT_TTL,
};
use starling_runtime::bus::EventSub;
use starling_runtime::channel::{bounded, Receiver, Sender, TrySendError};
use starling_runtime::machine::Rejection;
use starling_runtime::{Runtime, RuntimeClient};

use crate::agent::{broker_loop, Allowlist, AllowlistError, ASK_PREFIX};
use crate::auth::PeerPolicy;
use crate::config::HostConfig;
use crate::frame::{encode, Frame, FrameError, FrameReader, TransportErrorCode};
use crate::limits::{RateLimit, SlidingWindow};
use crate::platform::{self, Probe, TransportConn, TransportListener};

/// How often the accept loop and event pump re-check the shutdown flag.
const POLL: Duration = Duration::from_millis(50);

/// The agent broker's inbox depth. A full inbox refuses new asks with
/// `host_busy`; the other broker messages wait for room.
const BROKER_INBOX: usize = 64;

/// How long shutdown waits for every connection thread to end itself
/// before it kills the sockets outright. A well-behaved peer drains its
/// bye near-instantly; a peer that stopped reading (a writer parked in
/// `write_all` against a full kernel buffer) must not hold the whole
/// host hostage — the runtime shutdown, the lease release, the endpoint
/// removal and the `stopped` line all wait behind this bound, never
/// behind a peer.
const SHUTDOWN_DRAIN: Duration = Duration::from_secs(2);

/// Why a host could not start serving.
#[derive(Debug, thiserror::Error)]
pub enum HostError {
    /// A live foreign owner holds the data-root lease: this process is a
    /// client. `socket_path` is where that owner is (or should be)
    /// serving — the launched-binary path reports this and exits 0.
    #[error(
        "a live owner (id {owner_id}, pid {owner_pid}) holds the data root; \
         this process is a client — use the host at {socket_path}"
    )]
    OwnerLive {
        owner_id: String,
        owner_pid: u32,
        socket_path: PathBuf,
    },
    /// A lease file exists that can be neither probed nor broken (it
    /// will not open or parse, and no evidence proves its owner dead —
    /// never break what cannot be proven dead): no ownership was taken,
    /// rather than a second owner writing beside an unanswerable one
    /// whose reconciles would defer forever. `unreadable` names each
    /// lease file with its reason — repair (remove or fix the named
    /// files) or investigate that owner, then retry.
    #[error(
        "a lease file that cannot be probed blocks ownership of {root:?} — \
         never break what cannot be proven dead: {unreadable:?}; repair the \
         named file(s) or investigate that owner, then retry"
    )]
    LeaseUnanswerable {
        root: PathBuf,
        unreadable: Vec<(String, String)>,
    },
    /// The endpoint probe answered while this host held the lease: a
    /// server that serves without owning. Refuse.
    #[error("a live server is already bound at {0:?} while this host holds the lease")]
    ForeignServer(PathBuf),
    #[error("storage v2 at {root:?} will not open: {source}")]
    StoreOpen {
        root: PathBuf,
        source: starling_dictation::store_v2::StoreV2Error,
    },
    #[error("acquiring the runtime lease failed: {0}")]
    Lease(String),
    #[error("probing the endpoint {0:?} failed: {1}")]
    Probe(PathBuf, String),
    #[error("binding the endpoint {path:?} failed: {source}")]
    Bind {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("preparing the endpoint directory {0:?}: {1}")]
    RuntimeDir(PathBuf, String),
    /// The agent allowlist exists but will not load (fail closed).
    #[error("the agent allowlist {path:?} will not load: {source}")]
    Allowlist {
        path: PathBuf,
        source: AllowlistError,
    },
}

/// The running host. [`HostHandle::shutdown`] stops everything in order
/// (accept loop → connections → runtime → lease release → endpoint
/// removal); dropping the handle does the same, so a test or launcher
/// that lets it fall out of scope still leaves no lease or socket
/// behind.
pub struct HostHandle {
    socket_path: PathBuf,
    owner_id: String,
    shared: Arc<HostShared>,
    threads: Mutex<Vec<JoinHandle<()>>>,
    lease: Arc<Mutex<StoreV2>>,
    runtime: Option<Runtime>,
    /// The host's live engine attachment (builtin or manual mode): the
    /// switchable provider the jobs machine calls through and the
    /// engine it supervises, stopped after the machines join. `None`
    /// for `--engine none` and tests that inject their own provider.
    engine: Option<Arc<crate::engine::EngineHost>>,
    /// Set at shutdown so the settings watcher stops with the host's
    /// threads (joined before the engine stops, never after it).
    watch_stop: Arc<AtomicBool>,
    startup_reconciliation: ReconciliationReport,
    takes: Arc<crate::takes::TakeHub>,
    done: AtomicBool,
}

impl HostHandle {
    pub fn socket_path(&self) -> &std::path::Path {
        &self.socket_path
    }

    pub fn owner_id(&self) -> &str {
        &self.owner_id
    }

    /// The §4 reconciliation this host ran when it became the owner (see
    /// [`serve`]): what the previous owner's crash left behind and what
    /// startup did about it. Surfaced for status reporting — an empty
    /// report is the normal case.
    pub fn startup_reconciliation(&self) -> &ReconciliationReport {
        &self.startup_reconciliation
    }

    /// Whether the host has nothing to do (#220's idle exit): no client
    /// connected, no take recording or being stored, no job queued or
    /// running.
    pub fn idle(&self) -> bool {
        if self.shared.live_connections.load(Ordering::SeqCst) > 0 || self.takes.busy() {
            return false;
        }
        let snapshot = self.shared.client.snapshot();
        let capture_settled = matches!(
            snapshot.capture.state.as_str(),
            "Idle" | "Persisted" | "Interrupted"
        );
        capture_settled && snapshot.jobs.active == 0 && snapshot.jobs.waiting == 0
    }

    /// The engine manager this host supervises right now, when it runs
    /// one (builtin mode) — for status reporting and tests. A host that
    /// follows the settings file keeps this current while it serves: a
    /// model switch keeps the manager, a switch to manual mode takes it
    /// away.
    pub fn engine(&self) -> Option<starling_dictation::engine::EngineManager> {
        self.engine.as_ref().and_then(|engine| engine.manager())
    }

    /// The *effective* engine label right now (`builtin`,
    /// `manual:<endpoint>`, `unconfigured` for an engine attachment
    /// whose manual endpoint does not validate, `none` for a host
    /// without one): what the status line reports. Unlike the startup
    /// choice's label this one follows what actually serves — a manual
    /// endpoint that failed validation never shows up as a serving
    /// engine while jobs fail `no_provider_configured`.
    pub fn engine_label(&self) -> String {
        self.engine
            .as_ref()
            .map(|engine| engine.label())
            .unwrap_or_else(|| "none".to_string())
    }

    /// Graceful shutdown: no client is served past its `bye`, machines
    /// join, the lease is released, the endpoint is removed. Idempotent.
    ///
    /// The connection drain is **bounded**: after [`SHUTDOWN_DRAIN`] the
    /// host kills any connection whose writer has not ended itself (a
    /// peer that stopped reading parks `write_all` against a full kernel
    /// buffer; the socket shutdown unblocks it on unix). The host process
    /// therefore always finishes shutdown — a wedged renderer costs its
    /// own bye, never the lease release.
    pub fn shutdown(&mut self) {
        if self.done.swap(true, Ordering::SeqCst) {
            return;
        }
        self.shared.shutdown.store(true, Ordering::SeqCst);
        // The settings watcher stops with the host's threads below
        // (joined before the runtime and the engine) — flag it now so it
        // exits even while the connection drain above still waits, and
        // so no engine change is applied to a host that is going away.
        self.watch_stop.store(true, Ordering::SeqCst);
        if let Some(engine) = &self.engine {
            engine.begin_shutdown();
        }

        // Say goodbye and close every live connection first: writers
        // drain their queues (Bye included) before the senders drop.
        let conns = lock_registry(&self.shared.conns).clone();
        for conn in &conns {
            let _ = conn.try_deliver(Frame::Bye {
                reason: "host shutdown".to_string(),
            });
            // Marked, not killed: each writer drains its queue (the
            // bye included) and ends the stream itself.
            conn.mark_closed();
        }

        // Bounded drain: wait for every connection thread to end itself;
        // once the deadline passes, force the stragglers' sockets closed
        // (which unblocks a writer parked in write_all) and join. The
        // join after a forced close is prompt by construction — every
        // connection loop treats a socket error as its exit condition.
        // The force-close walks the (handle, state) pairs, not the
        // registry snapshot above: a connection whose reader already
        // exited (and unregistered itself) is still owed its kill — its
        // writer may be parked — and a connection accepted between the
        // two steps must not escape the bound either.
        let conn_threads = lock_registry(&self.shared.conn_threads)
            .drain(..)
            .collect::<Vec<_>>();
        let deadline = Instant::now() + SHUTDOWN_DRAIN;
        for (thread, _) in &conn_threads {
            while !thread.is_finished() && Instant::now() < deadline {
                std::thread::sleep(POLL);
            }
        }
        for (_, state) in &conn_threads {
            state.close();
        }
        for (thread, _) in conn_threads {
            let _ = thread.join();
        }

        let threads = self
            .threads
            .lock()
            .expect("host threads lock")
            .drain(..)
            .collect::<Vec<_>>();
        for thread in threads {
            let _ = thread.join();
        }

        // Connections are closed and the accept thread is down. Now stop
        // the machines, then release the lease and the endpoint.
        if let Some(runtime) = self.runtime.take() {
            runtime.shutdown();
        }
        // Close work the transport still has in flight (Windows'
        // lingering pipe disconnects hold handles that keep the endpoint
        // name bound) finishes before ownership is released, so a
        // successor never finds this host's instances still there.
        platform::finish_pending_closes();
        // The engine outlives every job the machines ran (the settings
        // watcher is down with the threads above); stop it only now (a
        // no-op for an engine another process owns — the attached case
        // leaves that owner's sidecar running).
        if let Some(engine) = self.engine.take() {
            engine.shutdown();
        }
        if let Ok(mut lease) = self.lease.lock() {
            let _ = lease.release_lease();
        }
        #[cfg(unix)]
        {
            let _ = std::fs::remove_file(&self.socket_path);
        }
    }
}

impl Drop for HostHandle {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// State shared by the accept loop, the event pump, the agent broker
/// and every connection.
///
/// Every atomic here runs at `SeqCst` as a deliberate blanket choice:
/// these are low-frequency flags and counters (shutdown polls, one
/// admission per connection) where the strongest ordering costs nothing
/// measurable, and picking per-field orderings would document
/// synchronization intent the host does not actually depend on beyond
/// "this store is visible to the next load".
pub struct HostShared {
    pub(crate) shutdown: AtomicBool,
    pub(crate) client: RuntimeClient,
    owner_id: String,
    max_frame_bytes: usize,
    command_rate: RateLimit,
    /// How long a freshly-admitted connection may hold its slot without
    /// sending a frame (the pre-greeting idle bound; the production
    /// default and the posture's rationale live on
    /// [`crate::config::HostConfig`]'s field).
    first_frame_idle: Duration,
    /// Live, **authenticated and greeted** connections — the event
    /// pump's fan-out set (see `connection_reader` for why registration
    /// waits until the hello is queued).
    pub(crate) conns: Mutex<Vec<Arc<ConnState>>>,
    /// Every connection's threads, paired with the connection's state
    /// so shutdown's force-close can reach a connection whose reader
    /// already exited and unregistered (a parked writer must never
    /// escape the drain bound by leaving the registry first).
    conn_threads: Mutex<Vec<(JoinHandle<()>, Arc<ConnState>)>>,
    live_connections: AtomicUsize,
    /// The agent broker's inbox (see [`crate::agent`]).
    pub(crate) broker: Sender<BrokerMsg>,
    /// The app's take feed (see [`crate::takes`]).
    pub(crate) takes: Arc<crate::takes::TakeHub>,
    /// The lease-holding store: owner-side repairs run on it.
    lease: Arc<Mutex<StoreV2>>,
    agent_allowlist: Allowlist,
    #[cfg(feature = "test-support")]
    insecure_test_app_role: bool,
}

impl HostShared {
    /// Whether a new connection holds the app role. Nothing grants it
    /// outside the `test-support` feature until the app-role credential
    /// exists.
    fn grants_app_role(&self) -> bool {
        #[cfg(feature = "test-support")]
        return self.insecure_test_app_role;
        #[cfg(not(feature = "test-support"))]
        false
    }
}

pub(crate) struct ConnState {
    outbound: Sender<Frame>,
    pub(crate) closed: AtomicBool,
    /// Set exactly once by `unregister` (the reader's exit path, also
    /// armed as a panic guard) so the live-connection count is
    /// decremented once per admission even if the reader panics.
    unregistered: AtomicBool,
    /// Set once an allowlisted `Frame::AgentHello` is accepted: only
    /// agent connections may ask, and they never see prompts.
    agent: AtomicBool,
    /// Whether this connection is the Starling app, the only role that
    /// sees and answers prompts. Nothing grants it in production yet,
    /// so asks fail closed with `no_app`.
    app: AtomicBool,
    /// A handle to the connection for immediate shutdown of both
    /// directions (the reader owns the original; the writer a clone).
    closer: Box<dyn TransportConn>,
}

/// What connection readers hand to the agent broker, each with the
/// sending connection.
pub(crate) enum BrokerMsg {
    Ask {
        conn: Arc<ConnState>,
        client_req: String,
        questions: Vec<String>,
        timeout_ms: u64,
    },
    Cancel {
        conn: Arc<ConnState>,
        client_req: String,
        reason: String,
    },
    Ack {
        conn: Arc<ConnState>,
        ask_id: String,
        visible: bool,
    },
    Done { conn: Arc<ConnState>, ask_id: String },
    ConnGone { conn: Arc<ConnState> },
}

impl ConnState {
    pub(crate) fn is_agent(&self) -> bool {
        self.agent.load(Ordering::SeqCst)
    }

    pub(crate) fn is_app(&self) -> bool {
        self.app.load(Ordering::SeqCst) && !self.is_agent()
    }

    /// Stops accepting frames for this connection. Does **not** touch
    /// the socket: queued frames (a final transport error, a goodbye)
    /// must still reach the peer — the writer thread performs the
    /// socket shutdown after it drains the queue.
    fn mark_closed(&self) {
        self.closed.store(true, Ordering::SeqCst);
    }

    /// Ends both directions immediately. For paths with nothing left to
    /// say (a slow consumer, a peer that is already gone).
    fn close(&self) {
        self.mark_closed();
        let _ = self.closer.shutdown_both();
    }

    /// How full this connection's outbound queue is: `(queued,
    /// capacity)`.
    pub(crate) fn outbound_depth(&self) -> (usize, usize) {
        self.outbound.depth()
    }

    /// Offers a frame to this connection's writer. `Err` when the
    /// connection's queue is full (slow consumer) or it is already
    /// closed.
    pub(crate) fn try_deliver(&self, frame: Frame) -> Result<(), ()> {
        if self.closed.load(Ordering::SeqCst) {
            return Err(());
        }
        match self.outbound.try_send(frame) {
            Ok(()) => Ok(()),
            Err(TrySendError::Full(_)) | Err(TrySendError::Closed(_)) => Err(()),
        }
    }
}

/// Boots the host per the ownership ladder (see the module docs).
pub fn serve(config: HostConfig) -> Result<HostHandle, HostError> {
    // 0. The agent allowlist, before any ownership step: a file that
    //    will not load leaves no lease or endpoint behind.
    let agent_allowlist = match &config.agent_allowlist {
        None => Allowlist::default(),
        Some(path) => Allowlist::load(path).map_err(|source| HostError::Allowlist {
            path: path.clone(),
            source,
        })?,
    };
    // 1. The lease. The store instance holding it stays alive (and the
    //    flock with it) for the host's lifetime; the capture machine's
    //    own StoreV2 handle over the same root is a second SQLite
    //    connection, which is the store's documented multi-process
    //    shape — ownership lives here, persistence there.
    let mut lease_store =
        StoreV2::open(&config.data_root).map_err(|source| HostError::StoreOpen {
            root: config.data_root.clone(),
            source,
        })?;
    let owner_id = match lease_store.acquire_lease() {
        Ok(LeaseAcquisition::Owner { owner_id, broke }) => {
            if !broke.is_empty() {
                eprintln!(
                    "starling-runtime-host: broke stale leases {broke:?} on {}",
                    config.data_root.display()
                );
            }
            owner_id
        }
        Ok(LeaseAcquisition::Client { owner }) => {
            return Err(HostError::OwnerLive {
                owner_id: owner.owner_id,
                owner_pid: owner.pid,
                socket_path: config.socket_path(),
            })
        }
        Ok(LeaseAcquisition::UnanswerableLeases { unreadable }) => {
            // An unanswerable lease reads as an owner we must defer to
            // (reconcile would, forever) — surface it instead of serving
            // beside it: this host would be a second writer on a root
            // whose recovery is already disabled.
            return Err(HostError::LeaseUnanswerable {
                root: config.data_root.clone(),
                unreadable,
            });
        }
        Err(source) => return Err(HostError::Lease(source.to_string())),
    };
    let lease = Arc::new(Mutex::new(lease_store));
    let release_lease_now = |lease: &Arc<Mutex<StoreV2>>| {
        if let Ok(mut store) = lease.lock() {
            let _ = store.release_lease();
        }
    };

    // 1b. §4 recovery: this host is now the **recovering** owner. Client
    //     mode defers staging salvage and orphan adoption to "the owner"
    //     (store_v2's contract) — in Mode B that owner is exactly this
    //     process, so reconcile runs here, before the first client can
    //     connect, on the lease-holding store (the only handle whose
    //     reconcile runs owner-mode: the runtime's own V2CaptureStore
    //     sees our lease as a live foreign owner and correctly defers).
    //     A crashed predecessor's interrupted takes are salvaged, and the
    //     report is surfaced (logged here, held on the handle for
    //     status). A reconcile that fails does not keep the host from
    //     serving (#220: the app records through it): the failure is
    //     surfaced to the app like the app surfaced it when it ran
    //     recovery itself, and the rest of recovery still runs.
    let mut reconcile_error = None;
    let startup_reconciliation = match lease
        .lock()
        // Poison-tolerant like the registries: the only other locker
        // (the heartbeat thread) panicking must not turn startup into a
        // panic cascade; reconcile on intact-but-poisoned data reports
        // its own errors through the Result.
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .reconcile()
    {
        Ok(report) => report,
        Err(source) => {
            eprintln!(
                "starling-runtime-host: reconciling storage v2 at {} failed: {source}",
                config.data_root.display()
            );
            reconcile_error = Some(format!("Could not recover interrupted recordings: {source}"));
            ReconciliationReport::default()
        }
    };
    if startup_reconciliation.has_findings() {
        eprintln!(
            "starling-runtime-host: storage v2 reconciliation on {}: \
             recovered {} interrupted take(s), promoted {} finalized journal(s), \
             adopted {} orphan session(s), marked {} row(s) interrupted, \
             completed {} interrupted delete(s); {} unreadable",
            config.data_root.display(),
            startup_reconciliation.recovered_torn.len(),
            startup_reconciliation.promoted_finalized.len(),
            startup_reconciliation.orphan_sessions.len(),
            startup_reconciliation.marked_interrupted.len(),
            startup_reconciliation.completed_deletes.len(),
            startup_reconciliation.unreadable.len(),
        );
    }

    // 1c. The rest of startup recovery (#220, from the app): attempts a
    //     previous run left started, takes a killed process left in the
    //     recorder's journal tree (#356). The app hears the findings
    //     through the take feed.
    let journals = config.runtime.capture.journals_dir.clone();
    let startup_recovery = crate::recovery::recover(
        &mut lease.lock().unwrap_or_else(|poisoned| poisoned.into_inner()),
        &journals,
        &startup_reconciliation,
        reconcile_error,
    );
    let takes = crate::takes::TakeHub::new(
        config.orphan_grace,
        Some(config.data_root.join(crate::takes::UNCLAIMED_FILE)),
    );
    takes.set_recovery(startup_recovery.recovery);

    // 2. The endpoint.
    if let Err(err) = platform::ensure_runtime_dir(&config.runtime_dir) {
        // Every post-acquisition failure releases the lease before
        // returning; this one is no exception.
        release_lease_now(&lease);
        return Err(HostError::RuntimeDir(
            config.runtime_dir.clone(),
            err.to_string(),
        ));
    }
    let socket_path = config.socket_path();
    match platform::probe(&socket_path) {
        Probe::Live => {
            // We hold the lease but a server answers: refuse — and leave
            // no lease behind on the way out.
            release_lease_now(&lease);
            return Err(HostError::ForeignServer(socket_path));
        }
        Probe::Dead => {
            #[cfg(unix)]
            platform::unix::remove_stale(&socket_path).map_err(|err| {
                release_lease_now(&lease);
                HostError::Bind {
                    path: socket_path.clone(),
                    source: err,
                }
            })?;
        }
        Probe::Unknown(detail) => {
            release_lease_now(&lease);
            return Err(HostError::Probe(socket_path, detail));
        }
    }
    let listener = platform::listen(&socket_path).map_err(|source| {
        release_lease_now(&lease);
        HostError::Bind {
            path: socket_path.clone(),
            source,
        }
    })?;
    listener.set_nonblocking(true).map_err(|source| {
        release_lease_now(&lease);
        HostError::Bind {
            path: socket_path.clone(),
            source,
        }
    })?;

    // 3. The engine, then the runtime (the host owns worker lifetime
    //    from here on). The engine starts only now — after the lease and
    //    the endpoint are this process's — so a host that turns out to
    //    be a client never spawns a sidecar.
    let mut runtime_config = config
        .runtime
        .with_capture_observer(Arc::clone(&takes) as Arc<dyn starling_runtime::machine::capture::CaptureObserver>);
    let engine = crate::engine::attach(config.engine, &mut runtime_config);
    let (runtime, client) = starling_runtime::Runtime::start(runtime_config);
    takes.attach(client.clone());
    let events = client.subscribe();

    let (broker_tx, broker_rx) = bounded(BROKER_INBOX);

    let shared = Arc::new(HostShared {
        shutdown: AtomicBool::new(false),
        client,
        owner_id: owner_id.clone(),
        max_frame_bytes: config.max_frame_bytes,
        command_rate: config.command_rate,
        first_frame_idle: config.first_frame_idle,
        conns: Mutex::new(Vec::new()),
        conn_threads: Mutex::new(Vec::new()),
        live_connections: AtomicUsize::new(0),
        broker: broker_tx,
        takes: Arc::clone(&takes),
        lease: Arc::clone(&lease),
        agent_allowlist,
        #[cfg(feature = "test-support")]
        insecure_test_app_role: config.insecure_test_app_role,
    });

    let watch_stop = Arc::new(AtomicBool::new(false));

    let mut threads = Vec::new();
    threads.push(spawn("starling-host-agent", {
        let shared = Arc::clone(&shared);
        move || broker_loop(shared, broker_rx)
    }));
    threads.push(spawn("starling-host-accept", {
        let shared = Arc::clone(&shared);
        let listener = listener;
        let policy = Arc::clone(&config.peer_policy);
        let max_connections = config.max_connections;
        let outbound_capacity = config.outbound_capacity;
        move || accept_loop(shared, listener, policy, max_connections, outbound_capacity)
    }));
    threads.push(spawn("starling-host-events", {
        let shared = Arc::clone(&shared);
        move || event_pump(shared, events)
    }));
    threads.push(spawn("starling-host-lease", {
        let lease = Arc::clone(&lease);
        let shared = Arc::clone(&shared);
        move || lease_heartbeat(lease, shared)
    }));
    threads.push(spawn("starling-host-takes", {
        let takes = Arc::clone(&takes);
        let shared = Arc::clone(&shared);
        move || crate::takes::tick_loop(takes, shared)
    }));
    // The recheck rides with the host's threads too: shutdown joins it
    // before the lease is released, so it never touches a store a
    // successor owns.
    threads.extend(crate::recovery::recheck_later(
        Arc::clone(&lease),
        journals,
        startup_recovery.recheck,
        {
            let shared = Arc::clone(&shared);
            move || shared.shutdown.load(Ordering::SeqCst)
        },
        {
            let takes = Arc::clone(&takes);
            move |found| takes.notice(found)
        },
    ));
    // The settings follower (#220): while the host serves, engine
    // changes in the settings file apply to it. The watcher rides with
    // the host's threads, so shutdown joins it before the runtime and
    // the engine stop — it never applies a change to a host that is
    // going away.
    if let (Some(engine), Some(settings_path)) = (&engine, config.settings_path.as_ref()) {
        match crate::engine::watch_settings(
            Arc::clone(engine),
            settings_path.clone(),
            config.settings_poll,
            Arc::clone(&watch_stop),
        ) {
            Ok(watcher) => threads.push(watcher),
            // No thread for the watcher: degrade to the startup choice
            // (the `--engine` posture without a settings path) rather
            // than abort a host that otherwise serves.
            Err(err) => eprintln!(
                "starling-runtime-host: cannot start the settings watcher ({err}); \
                 the engine stays as configured at startup"
            ),
        }
    }

    Ok(HostHandle {
        socket_path,
        owner_id,
        shared,
        threads: Mutex::new(threads),
        lease,
        runtime: Some(runtime),
        engine,
        watch_stop,
        startup_reconciliation,
        takes,
        done: AtomicBool::new(false),
    })
}

/// Locks one of the host's registry mutexes, tolerating poison: a
/// connection thread that panicked leaves the data intact (worst case
/// stale), and aborting every remaining client's session over bookkeeping
/// is the wrong trade. The lease mutex keeps `expect` — its invariants
/// are ownership-critical, not bookkeeping.
pub(crate) fn lock_registry<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn spawn(name: &str, run: impl FnOnce() + Send + 'static) -> JoinHandle<()> {
    std::thread::Builder::new()
        .name(name.to_string())
        .spawn(run)
        .expect("host thread spawn")
}

/// Accepts, enforces the connection cap, and hands each connection to
/// its reader/writer pair.
fn accept_loop(
    shared: Arc<HostShared>,
    listener: Box<dyn TransportListener>,
    policy: Arc<dyn PeerPolicy>,
    max_connections: usize,
    outbound_capacity: usize,
) {
    while !shared.shutdown.load(Ordering::SeqCst) {
        match listener.accept() {
            Ok(conn) => {
                let mut conn = conn;
                // Reserve the slot atomically: load-then-act would let
                // concurrent accepts all observe the same count below the
                // cap and overshoot it (the cap is a documented bound).
                let live = shared.live_connections.fetch_add(1, Ordering::SeqCst);
                if live >= max_connections {
                    shared.live_connections.fetch_sub(1, Ordering::SeqCst);
                    // Answer honestly, then close: the client learns why.
                    // The write is bounded: a peer that connected with a
                    // full receive buffer must not park the one accept
                    // thread — the error is dropped after the timeout,
                    // the connection dies either way.
                    let error = Frame::TransportError {
                        code: TransportErrorCode::TooManyConnections,
                        detail: format!("{live} connections already live (cap {max_connections})"),
                    };
                    if let Ok(wire) = encode(&error, shared.max_frame_bytes) {
                        let _ = conn.set_write_timeout(Some(POLL * 4));
                        let sink = conn.as_mut();
                        let _ = sink.write_all(&wire);
                        let _ = sink.flush();
                        let _ = conn.set_write_timeout(None);
                    }
                    let _ = conn.shutdown_both();
                    continue;
                }
                let writer_conn = match conn.try_clone() {
                    Ok(clone) => clone,
                    Err(err) => {
                        eprintln!("starling-runtime-host: connection clone failed: {err}");
                        shared.live_connections.fetch_sub(1, Ordering::SeqCst);
                        let _ = conn.shutdown_both();
                        continue;
                    }
                };
                let closer = match conn.try_clone() {
                    Ok(clone) => clone,
                    Err(err) => {
                        eprintln!("starling-runtime-host: connection clone failed: {err}");
                        shared.live_connections.fetch_sub(1, Ordering::SeqCst);
                        let _ = conn.shutdown_both();
                        continue;
                    }
                };
                let (outbound_tx, outbound_rx) = bounded(outbound_capacity);
                let state = Arc::new(ConnState {
                    outbound: outbound_tx,
                    closed: AtomicBool::new(false),
                    unregistered: AtomicBool::new(false),
                    agent: AtomicBool::new(false),
                    app: AtomicBool::new(shared.grants_app_role()),
                    closer,
                });

                // Spawn failures are survived, not fatal: a transient
                // thread exhaustion must cost this one connection, never
                // the accept loop (whose panic would silently stop all
                // future serving).
                let reader = match spawn_conn_thread("starling-host-conn-read", {
                    let shared = Arc::clone(&shared);
                    let state = Arc::clone(&state);
                    let policy = Arc::clone(&policy);
                    let conn = conn;
                    move || connection_reader(shared, state, policy, conn)
                }) {
                    Ok(reader) => reader,
                    Err(err) => {
                        eprintln!("starling-runtime-host: reader spawn failed: {err}");
                        // The reader never runs, so its unregister never
                        // will: give the slot back here.
                        shared.live_connections.fetch_sub(1, Ordering::SeqCst);
                        let _ = state.close();
                        continue;
                    }
                };
                let writer = match spawn_conn_thread("starling-host-conn-write", {
                    let shared = Arc::clone(&shared);
                    let state = Arc::clone(&state);
                    move || connection_writer(shared, state, writer_conn, outbound_rx)
                }) {
                    Ok(writer) => writer,
                    Err(err) => {
                        eprintln!("starling-runtime-host: writer spawn failed: {err}");
                        // Kill the socket; the (running) reader observes
                        // the error and unregisters — the slot is
                        // returned exactly once, by the reader.
                        state.close();
                        lock_registry(&shared.conn_threads)
                            .push((reader, Arc::clone(&state)));
                        continue;
                    }
                };
                // Register, then reap: every accept sweeps the pairs
                // whose threads already exited, so the registry stays
                // bounded on a long-lived host serving many short-lived
                // clients (a crashlooping renderer reconnecting once a
                // second must not grow it forever).
                let mut threads = lock_registry(&shared.conn_threads);
                threads.retain(|(thread, _)| !thread.is_finished());
                threads.push((reader, Arc::clone(&state)));
                threads.push((writer, Arc::clone(&state)));
            }
            Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                std::thread::sleep(POLL);
            }
            Err(err) => {
                eprintln!("starling-runtime-host: accept failed: {err}");
                std::thread::sleep(POLL);
            }
        }
    }
}

fn spawn_conn_thread(
    name: &str,
    run: impl FnOnce() + Send + 'static,
) -> std::io::Result<JoinHandle<()>> {
    std::thread::Builder::new()
        .name(name.to_string())
        .spawn(run)
}

/// One connection's inbound side: auth, hello, then frames.
///
/// The connection joins the event fan-out only **after** authentication
/// succeeded and the hello is queued: registering earlier would let the
/// pump enqueue events ahead of the greeting (and, on an auth failure,
/// leak runtime events to a peer that was never admitted).
fn connection_reader(
    shared: Arc<HostShared>,
    state: Arc<ConnState>,
    policy: Arc<dyn PeerPolicy>,
    mut conn: Box<dyn TransportConn>,
) {
    // Panic-proof bookkeeping: exactly once per admitted connection,
    // whether the reader returns or panics, the registry entry goes and
    // the reserved slot is returned.
    let guard = UnregisterOnDrop {
        shared: &shared,
        state: &state,
    };

    // Fail closed: a credential read failure is an auth failure.
    let credentials = conn.peer_credentials().unwrap_or_default();
    if let Err(refused) = policy.authenticate(credentials) {
        let error = Frame::TransportError {
            code: TransportErrorCode::AuthFailed,
            detail: refused.to_string(),
        };
        // Best-effort direct write: the writer thread has nothing queued
        // (no hello was sent), so the two never race. Bounded, so a
        // peer that connected with a full buffer cannot park this
        // reader before it even starts.
        if let Ok(wire) = encode(&error, shared.max_frame_bytes) {
            let _ = conn.set_write_timeout(Some(POLL * 4));
            let sink = conn.as_mut();
            let _ = sink.write_all(&wire);
            let _ = sink.flush();
            let _ = conn.set_write_timeout(None);
        }
        // The frame went out on our own handle; the writer (with nothing
        // queued — no hello was ever sent) shuts the socket down after
        // the queue closes.
        state.mark_closed();
        guard.run();
        return;
    }

    // The greeting carries the limits this connection runs under.
    if state
        .try_deliver(Frame::Hello {
            protocol: 1,
            owner_id: shared.owner_id.clone(),
            pid: std::process::id(),
            max_frame_bytes: shared.max_frame_bytes as u64,
            rate_max: shared.command_rate.max,
            rate_window_ms: shared.command_rate.per.as_millis() as u64,
        })
        .is_err()
    {
        state.close();
        guard.run();
        return;
    }
    // Authenticated and greeted: the connection may now receive events.
    // The hello is already queued, so nothing the pump enqueues from
    // here on can overtake it.
    lock_registry(&shared.conns).push(Arc::clone(&state));

    let mut reader = FrameReader::new(conn, shared.max_frame_bytes);
    let mut rate = SlidingWindow::new(shared.command_rate);
    let admitted_at = Instant::now();
    let mut first_frame = true;
    while !shared.shutdown.load(Ordering::SeqCst) && !state.closed.load(Ordering::SeqCst) {
        match reader.read_frame() {
            Ok(frame) => {
                first_frame = false;
                if !rate.allow(Instant::now()) {
                    terminate(
                        &state,
                        TransportErrorCode::RateLimited,
                        format!(
                            "more than {} frames in {:?}; connection closed",
                            shared.command_rate.max, shared.command_rate.per
                        ),
                    );
                    break;
                }
                match frame {
                    // Agents reach the microphone only through asks.
                    Frame::Command { .. } if state.is_agent() => {
                        terminate(
                            &state,
                            TransportErrorCode::ProtocolViolation,
                            "agent connections cannot send runtime commands; use ask_user"
                                .to_string(),
                        );
                        break;
                    }
                    Frame::Command { mut envelope } => {
                        // The connection that starts a take owns it in
                        // the take feed, and only it ends the take while
                        // it is connected.
                        let kind = envelope
                            .get("type")
                            .and_then(serde_json::Value::as_str)
                            .unwrap_or_default()
                            .to_string();
                        let corr = envelope
                            .get("corr")
                            .and_then(serde_json::Value::as_str)
                            .map(str::to_string);
                        // A start without a corr records under the
                        // runtime's anonymous take id: it is owned too.
                        let owned_as = (kind == "capture.start").then(|| {
                            corr.clone().unwrap_or_else(|| {
                                starling_runtime::machine::capture::ANON_TAKE.to_string()
                            })
                        });
                        let registered = owned_as
                            .as_deref()
                            .is_some_and(|take| shared.takes.starting(take, &state));
                        if (kind == "capture.stop" || kind == "capture.abort")
                            && !shared.takes.may_end(corr.as_deref(), &state)
                        {
                            let id = envelope
                                .get("id")
                                .and_then(serde_json::Value::as_str)
                                .unwrap_or_default()
                                .to_string();
                            let refusal = Frame::Receipt {
                                req: id,
                                seq: None,
                                result: Err(Rejection::IllegalInState {
                                    command: kind,
                                    state: "Recording".to_string(),
                                    detail: "the current take belongs to another connection"
                                        .to_string(),
                                }),
                            };
                            if state.try_deliver(refusal).is_err() {
                                state.close();
                                break;
                            }
                            continue;
                        }
                        match handle_command(&shared, &state, &mut envelope) {
                            Err(()) => break,
                            Ok(false) if registered => {
                                // Refused: no take opens, so the
                                // connection owns nothing by it.
                                if let Some(take) = owned_as.as_deref() {
                                    shared.takes.start_refused(take, &state);
                                }
                            }
                            Ok(_) => {}
                        }
                    }
                    // The agent ask surface. A full broker refuses new
                    // asks with `host_busy`; every other broker message
                    // waits for room, since dropping a cancel or a
                    // dismissal could leave the microphone open.
                    Frame::AgentHello { req, client, token } => {
                        if state.is_agent() {
                            terminate(
                                &state,
                                TransportErrorCode::ProtocolViolation,
                                "a second agent hello on one connection".to_string(),
                            );
                            break;
                        }
                        if let Err(refused) = shared.agent_allowlist.authenticate(&client, &token) {
                            terminate(&state, TransportErrorCode::AuthFailed, refused);
                            break;
                        }
                        state.agent.store(true, Ordering::SeqCst);
                        if state
                            .try_deliver(Frame::AgentWelcome { req, client })
                            .is_err()
                        {
                            state.close();
                            break;
                        }
                    }
                    Frame::AskUser {
                        req,
                        questions,
                        timeout_ms,
                    } => {
                        if !state.is_agent() {
                            terminate(
                                &state,
                                TransportErrorCode::ProtocolViolation,
                                "ask_user from a connection that is not an allowlisted agent"
                                    .to_string(),
                            );
                            break;
                        }
                        let ask = BrokerMsg::Ask {
                            conn: Arc::clone(&state),
                            client_req: req.clone(),
                            questions,
                            timeout_ms,
                        };
                        if shared.broker.try_send(ask).is_err() {
                            let _ = state.try_deliver(Frame::AskResult {
                                req,
                                outcome: crate::frame::AskOutcome::Error {
                                    code: "host_busy".to_string(),
                                    message: "the host cannot take another ask right now; retry"
                                        .to_string(),
                                },
                            });
                        }
                    }
                    Frame::AskCancel { req, reason } => {
                        let _ = shared.broker.send_blocking(BrokerMsg::Cancel {
                            conn: Arc::clone(&state),
                            client_req: req,
                            reason,
                        });
                    }
                    // Only the app sees prompts, so prompt frames from
                    // anyone else can only be an attempt to open the gate.
                    Frame::PromptAck { .. } | Frame::PromptDone { .. } if !state.is_app() => {
                        terminate(
                            &state,
                            TransportErrorCode::ProtocolViolation,
                            "prompt frames are only accepted from the Starling app".to_string(),
                        );
                        break;
                    }
                    Frame::PromptAck { req, visible } => {
                        let _ = shared.broker.send_blocking(BrokerMsg::Ack {
                            conn: Arc::clone(&state),
                            ask_id: req,
                            visible,
                        });
                    }
                    Frame::PromptDone { req } => {
                        let _ = shared.broker.send_blocking(BrokerMsg::Done {
                            conn: Arc::clone(&state),
                            ask_id: req,
                        });
                    }
                    // The take feed is the app's; agents reach the
                    // microphone only through asks.
                    Frame::TakeWatch { .. } | Frame::TakeTap { .. } | Frame::TakeHandled { .. }
                        if state.is_agent() =>
                    {
                        terminate(
                            &state,
                            TransportErrorCode::ProtocolViolation,
                            "agent connections cannot follow the take feed".to_string(),
                        );
                        break;
                    }
                    Frame::TakeWatch { req } => {
                        // An app (re)connected: attempts a dead app left
                        // "started" are failed now, ready to retry — a
                        // live app's attempts are protected by their
                        // markers (#213). Startup ran the same repair.
                        if let Err(err) = shared
                            .lease
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner())
                            .interrupt_stale_attempts(crate::recovery::STALE_ATTEMPT_NOTE)
                        {
                            eprintln!(
                                "starling-runtime-host: failing stale transcription attempts: {err}"
                            );
                        }
                        if shared.takes.watch(&state, req).is_err() {
                            state.close();
                            break;
                        }
                    }
                    Frame::TakeTap { take, from } => {
                        shared.takes.tap(&state, take, from);
                    }
                    Frame::TakeHandled {
                        stored_id,
                        handed_back,
                    } => {
                        if handed_back {
                            shared.takes.handed_back(&state, &stored_id);
                        } else {
                            shared.takes.handled(&stored_id);
                        }
                    }
                    Frame::GetSnapshot { req } => {
                        let snapshot = shared.client.snapshot();
                        let value = serde_json::to_value(&snapshot)
                            .unwrap_or_else(|_| serde_json::Value::Null);
                        if state
                            .try_deliver(Frame::Snapshot {
                                req,
                                snapshot: value,
                            })
                            .is_err()
                        {
                            // Same posture as an overflowing receipt
                            // above: a full queue means a peer that is
                            // not reading; kill the socket, never park
                            // the writer behind an undeliverable
                            // explanation.
                            state.close();
                            break;
                        }
                    }
                    Frame::Hello { .. }
                    | Frame::Receipt { .. }
                    | Frame::Event { .. }
                    | Frame::Snapshot { .. }
                    | Frame::TransportError { .. }
                    | Frame::Bye { .. }
                    | Frame::AgentWelcome { .. }
                    | Frame::AskResult { .. }
                    | Frame::ShowPrompt { .. }
                    | Frame::HidePrompt { .. }
                    | Frame::TakeWatching { .. }
                    | Frame::LiveTake { .. }
                    | Frame::TakeStartFailed { .. }
                    | Frame::TakePersisted { .. }
                    | Frame::HostNotice { .. } => {
                        terminate(
                            &state,
                            TransportErrorCode::ProtocolViolation,
                            "host-to-client frame sent by a client".to_string(),
                        );
                        break;
                    }
                }
            }
            Err(FrameError::TooLarge { declared, cap }) => {
                terminate(
                    &state,
                    TransportErrorCode::MessageTooLarge,
                    format!("frame declared {declared} bytes; cap is {cap}"),
                );
                break;
            }
            Err(FrameError::Malformed(detail)) => {
                terminate(&state, TransportErrorCode::MalformedFrame, detail);
                break;
            }
            // Peer went away (or timed out past our 250ms read poll and
            // the flags did not say stop): a read poll timeout surfaces
            // as WouldBlock/TimedOut errors from the underlying stream.
            Err(FrameError::Eof) => break,
            Err(FrameError::Io(err))
                if err.kind() == std::io::ErrorKind::WouldBlock
                    || err.kind() == std::io::ErrorKind::TimedOut =>
            {
                // The pre-greeting idle bound: a connection that has
                // sent nothing since admit may not hold its slot past
                // the configured deadline (see HostConfig's field docs
                // for the posture — the credential gate, not this
                // deadline, is the security boundary).
                let idle_bound = shared.first_frame_idle;
                if first_frame && admitted_at.elapsed() > idle_bound {
                    terminate(
                        &state,
                        TransportErrorCode::ProtocolViolation,
                        format!(
                            "no frame within {idle_bound:?} of connect; connection closed"
                        ),
                    );
                    break;
                }
                continue;
            }
            Err(FrameError::Io(_)) => break,
        }
    }
    state.mark_closed();
    guard.run();
}

/// Returns a connection's registry entry and reserved slot exactly
/// once, from the reader's exit — including a panic exit (a panicking
/// reader must not leak the cap slot its admission consumed).
struct UnregisterOnDrop<'a> {
    shared: &'a HostShared,
    state: &'a Arc<ConnState>,
}

impl UnregisterOnDrop<'_> {
    /// The normal exit path.
    fn run(self) {
        self.once();
    }

    /// Idempotent on the connection's own flag, so run() followed by
    /// drop (and a panic followed by drop) both count once.
    fn once(&self) {
        if !self.state.unregistered.swap(true, Ordering::SeqCst) {
            lock_registry(&self.shared.conns)
                .retain(|registered| !Arc::ptr_eq(registered, self.state));
            self.shared.live_connections.fetch_sub(1, Ordering::SeqCst);
            // The broker cancels any ask this connection asked or
            // acked; it must not be dropped on a full inbox.
            let _ = self.shared.broker.send_blocking(BrokerMsg::ConnGone {
                conn: Arc::clone(self.state),
            });
        }
    }
}

impl Drop for UnregisterOnDrop<'_> {
    fn drop(&mut self) {
        // Panic path: run() was never reached; do its work now.
        self.once();
    }
}

/// Bridges one command envelope into the runtime and returns the receipt
/// frame. The envelope is the I3 wire form; when the client left `seq`
/// out, the host assigns it from the runtime's own frontier (see
/// `RuntimeClient::assign_seq`) so a reconnecting client cannot collide
/// with stream positions a dead connection consumed. `Ok(true)` when the
/// runtime accepted the command.
fn handle_command(
    shared: &HostShared,
    state: &ConnState,
    envelope: &mut serde_json::Value,
) -> Result<bool, ()> {
    // Shape first: the receipt is keyed by the envelope's string `id`,
    // so an envelope without one could never be answered — the command
    // would execute while the client sat out its reply timeout. Refuse
    // the frame instead of routing an unanswerable command.
    let id = match envelope
        .as_object()
        .and_then(|object| object.get("id"))
        .and_then(serde_json::Value::as_str)
    {
        Some(id) => id.to_string(),
        None => {
            terminate(
                state,
                TransportErrorCode::MalformedFrame,
                "command envelope must be an object carrying a string id".to_string(),
            );
            return Err(());
        }
    };
    let corr = envelope
        .as_object()
        .and_then(|object| object.get("corr"))
        .and_then(serde_json::Value::as_str)
        .map(str::to_string);
    // The broker's corrs are reserved, so no client can address an ask's
    // take or context by name.
    if corr.as_deref().is_some_and(|corr| corr.starts_with(ASK_PREFIX)) {
        let refusal = Rejection::InvalidEnvelope(format!(
            "corr prefix {ASK_PREFIX:?} is reserved for the host's agent asks"
        ));
        return state
            .try_deliver(Frame::Receipt {
                req: id,
                seq: None,
                result: Err(refusal),
            })
            .map(|()| false)
            .map_err(|()| state.close());
    }
    let client_supplied_seq = envelope
        .as_object()
        .is_some_and(|object| object.contains_key("seq"));
    if client_supplied_seq {
        // A present-but-non-numeric seq (null, string, …) is the same
        // unanswerable/mis-sequenced shape as a missing id: refuse it
        // rather than route a malformed stream position.
        if !envelope
            .as_object()
            .and_then(|object| object.get("seq"))
            .is_some_and(serde_json::Value::is_u64)
        {
            terminate(
                state,
                TransportErrorCode::MalformedFrame,
                "seq, when present, must be an unsigned number".to_string(),
            );
            return Err(());
        }
    } else {
        let seq = shared.client.assign_seq(corr.as_deref());
        if let Some(object) = envelope.as_object_mut() {
            object.insert("seq".to_string(), serde_json::Value::from(seq));
        }
    }
    let seq = envelope
        .as_object()
        .and_then(|object| object.get("seq"))
        .and_then(serde_json::Value::as_u64);
    // The envelope is consumed here (nothing reads it after); moving it
    // avoids a full JSON deep-clone on the per-command hot path.
    let result = shared.client.send_raw(std::mem::take(envelope));
    let accepted = result.is_ok();
    if state
        .try_deliver(Frame::Receipt {
            req: id,
            seq,
            result,
        })
        .is_err()
    {
        // The command already ran; the client must never sit out a bare
        // reply timeout for an executed command (a timeout reads as
        // "safe to resend" and a resend would duplicate it). The queue
        // being full means the peer is not reading — an explanation
        // frame would sit behind frames it has not read either — so the
        // connection is killed outright (unblocking the writer parked
        // in write_all against the full kernel buffer); the client
        // reconnects and resynchronizes from the snapshot.
        state.close();
        return Err(());
    }
    Ok(accepted)
}

/// One connection's outbound side: serialized writes from the bounded
/// queue.
fn connection_writer(
    shared: Arc<HostShared>,
    state: Arc<ConnState>,
    conn: Box<dyn TransportConn>,
    outbound: Receiver<Frame>,
) {
    let mut sink = conn;
    // Whoever exits this loop is the last writer this connection has:
    // end the stream so the peer observes the close deterministically
    // (after every queued frame — including final errors and byes — was
    // written above).
    let finish = |sink: &mut Box<dyn TransportConn>, state: &ConnState| {
        state.mark_closed();
        let _ = sink.shutdown_both();
    };
    loop {
        match outbound.recv_timeout(POLL) {
            Ok(frame) => {
                let wire = match encode(&frame, shared.max_frame_bytes) {
                    Ok(wire) => wire,
                    Err(FrameError::TooLarge { declared, cap }) => {
                        // Never truncate: an event too big for this
                        // connection's cap is a real fault on it.
                        let error = Frame::TransportError {
                            code: TransportErrorCode::MessageTooLarge,
                            detail: format!("outbound frame {declared} bytes exceeds cap {cap}"),
                        };
                        if let Ok(wire) = encode(&error, shared.max_frame_bytes) {
                            let _ = sink.write_all(&wire);
                            let _ = sink.flush();
                        }
                        state.close();
                        break;
                    }
                    Err(_) => {
                        state.close();
                        break;
                    }
                };
                if sink.write_all(&wire).is_err() || sink.flush().is_err() {
                    finish(&mut sink, &state);
                    break;
                }
            }
            Err(starling_runtime::channel::RecvError::Timeout) => {
                if shared.shutdown.load(Ordering::SeqCst) || state.closed.load(Ordering::SeqCst) {
                    // Graceful: the shutdown path queued Bye frames
                    // before marking connections closed; the drain above
                    // wrote them in order.
                    finish(&mut sink, &state);
                    break;
                }
            }
            Err(starling_runtime::channel::RecvError::Closed) => {
                // Every sender dropped: reader gone, registry gone (or
                // both closing after a Bye). The queue drained first —
                // recv only reports Closed when empty.
                finish(&mut sink, &state);
                break;
            }
        }
    }
}

/// Ends a connection with a final transport-error frame: the frame is
/// queued, the connection is marked closed, and the writer thread
/// delivers the frame and then ends the stream. For **reader-side
/// violations only** — the peer just sent a frame, so it is
/// demonstrably still reading and the explanation will reach it.
/// Queue-overflow paths (receipt/snapshot delivery in
/// [`handle_command`], the event pump's eviction) must instead use
/// [`ConnState::close`]: the queue being full means the peer is not
/// reading, so a queued explanation would never arrive and the socket
/// must be killed to unblock the writer parked against it.
///
/// Ordering caveat, by construction: the error frame shares the
/// outbound queue with event deliveries from the pump, so a concurrent
/// event may be written after it — the terminal frame is
/// best-effort-last, not guaranteed-last. The client treats any close
/// after a transport error as terminal either way (it reconnects and
/// resynchronizes from the snapshot).
fn terminate(state: &ConnState, code: TransportErrorCode, detail: String) {
    let _ = state.try_deliver(Frame::TransportError { code, detail });
    state.mark_closed();
}

/// Fans runtime events out to every live connection. This decouples the
/// runtime's EventBus (whose backpressure semantics are in-process:
/// machines stall on a full subscriber) from IPC clients (whose policy
/// is the opposite: a stopped reader is dropped, machines never stall —
/// §1 Mode B's renderer-kill posture).
///
/// Cost note: the registry is a Mutex<Vec<..>> and each event clones
/// the Vec (Arc bumps, one allocation). At the documented connection
/// cap and event rates this is noise; if the cap ever grows by orders
/// of magnitude, switch to a map keyed by connection id or reuse a
/// scratch Vec here.
fn event_pump(shared: Arc<HostShared>, events: EventSub) {
    loop {
        if shared.shutdown.load(Ordering::SeqCst) {
            break;
        }
        match events.recv_timeout(Duration::from_millis(100)) {
            Ok(message) => {
                let frame = Frame::Event {
                    envelope: message.to_value(),
                };
                let conns = lock_registry(&shared.conns).clone();
                for conn in conns {
                    if conn.try_deliver(frame.clone()).is_err()
                        && !conn.closed.swap(true, Ordering::SeqCst)
                    {
                        // A peer whose queue overflowed is not reading;
                        // the queued explanation could not reach it
                        // anyway. Kill the socket so the writer's parked
                        // write unblocks; a reconnecting client resyncs
                        // from the snapshot.
                        conn.close();
                    }
                }
            }
            Err(starling_runtime::channel::RecvError::Timeout) => continue,
            Err(starling_runtime::channel::RecvError::Closed) => break,
        }
    }
}

/// Renews the storage-v2 lease heartbeat at TTL/3 — the designed cadence
/// (a 30 s [`LEASE_HEARTBEAT_TTL`] renews every ~10 s, so a successor
/// needs three missed beats before it may break the lease). The flock on
/// the identity file is the ownership signal the OS maintains by itself;
/// the heartbeat is the fallback for flock-less hosts, so a renewal
/// failure is logged and outlived, never fatal here. The cadence is
/// sliced into poll-sized sleeps so shutdown stays prompt.
fn lease_heartbeat(lease: Arc<Mutex<StoreV2>>, shared: Arc<HostShared>) {
    loop {
        let next_beat = Instant::now() + LEASE_HEARTBEAT_TTL / 3;
        while Instant::now() < next_beat {
            std::thread::sleep(POLL);
            if shared.shutdown.load(Ordering::SeqCst) {
                return;
            }
        }
        if let Ok(mut store) = lease.lock() {
            if let Err(err) = store.heartbeat_lease() {
                eprintln!("starling-runtime-host: lease heartbeat failed: {err}");
            }
        }
    }
}
