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
//!    otherwise).
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

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use starling_dictation::store_v2::{LeaseAcquisition, StoreV2};
use starling_runtime::bus::EventSub;
use starling_runtime::channel::{bounded, Receiver, Sender, TrySendError};
use starling_runtime::{Runtime, RuntimeClient};

use crate::auth::PeerPolicy;
use crate::config::HostConfig;
use crate::frame::{encode, Frame, FrameError, FrameReader, TransportErrorCode};
use crate::limits::{RateLimit, SlidingWindow};
use crate::platform::{self, Probe, TransportConn, TransportListener};

/// How often the accept loop and event pump re-check the shutdown flag.
const POLL: Duration = Duration::from_millis(50);

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
    done: AtomicBool,
}

impl HostHandle {
    pub fn socket_path(&self) -> &std::path::Path {
        &self.socket_path
    }

    pub fn owner_id(&self) -> &str {
        &self.owner_id
    }

    /// Graceful shutdown: no client is served past its `bye`, machines
    /// join, the lease is released, the endpoint is removed. Idempotent.
    pub fn shutdown(&mut self) {
        if self.done.swap(true, Ordering::SeqCst) {
            return;
        }
        self.shared.shutdown.store(true, Ordering::SeqCst);

        // Say goodbye and close every live connection first: writers
        // drain their queues (Bye included) before the senders drop.
        {
            let conns = self.shared.conns.lock().expect("conn registry").clone();
            for conn in conns {
                let _ = conn.try_deliver(Frame::Bye {
                    reason: "host shutdown".to_string(),
                });
                // Marked, not killed: each writer drains its queue (the
                // bye included) and ends the stream itself.
                conn.mark_closed();
            }
        }
        let conn_threads = self
            .shared
            .conn_threads
            .lock()
            .expect("conn threads")
            .drain(..)
            .collect::<Vec<_>>();
        for thread in conn_threads {
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

/// State shared by the accept loop, the event pump and every connection.
pub struct HostShared {
    shutdown: AtomicBool,
    client: RuntimeClient,
    owner_id: String,
    max_frame_bytes: usize,
    command_rate: RateLimit,
    /// Live connections (readers unregister themselves on exit).
    conns: Mutex<Vec<Arc<ConnState>>>,
    /// Connection threads to join at shutdown.
    conn_threads: Mutex<Vec<JoinHandle<()>>>,
    live_connections: AtomicUsize,
}

struct ConnState {
    outbound: Sender<Frame>,
    closed: AtomicBool,
    /// A handle to the connection for immediate shutdown of both
    /// directions (the reader owns the original; the writer a clone).
    closer: Box<dyn TransportConn>,
}

impl ConnState {
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

    /// Offers a frame to this connection's writer. `Err` when the
    /// connection's queue is full (slow consumer) or it is already
    /// closed.
    fn try_deliver(&self, frame: Frame) -> Result<(), ()> {
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
        Err(source) => return Err(HostError::Lease(source.to_string())),
    };
    let lease = Arc::new(Mutex::new(lease_store));
    let release_lease_now = |lease: &Arc<Mutex<StoreV2>>| {
        if let Ok(mut store) = lease.lock() {
            let _ = store.release_lease();
        }
    };

    // 2. The endpoint.
    platform::ensure_runtime_dir(&config.runtime_dir)
        .map_err(|err| HostError::RuntimeDir(config.runtime_dir.clone(), err.to_string()))?;
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

    // 3. The runtime (the host owns worker lifetime from here on).
    let (runtime, client) = starling_runtime::Runtime::start(config.runtime);
    let events = client.subscribe();

    let shared = Arc::new(HostShared {
        shutdown: AtomicBool::new(false),
        client,
        owner_id: owner_id.clone(),
        max_frame_bytes: config.max_frame_bytes,
        command_rate: config.command_rate,
        conns: Mutex::new(Vec::new()),
        conn_threads: Mutex::new(Vec::new()),
        live_connections: AtomicUsize::new(0),
    });

    let mut threads = Vec::new();
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

    Ok(HostHandle {
        socket_path,
        owner_id,
        shared,
        threads: Mutex::new(threads),
        lease,
        runtime: Some(runtime),
        done: AtomicBool::new(false),
    })
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
                let live = shared.live_connections.load(Ordering::SeqCst);
                if live >= max_connections {
                    // Answer honestly, then close: the client learns why.
                    let error = Frame::TransportError {
                        code: TransportErrorCode::TooManyConnections,
                        detail: format!("{live} connections already live (cap {max_connections})"),
                    };
                    if let Ok(wire) = encode(&error, shared.max_frame_bytes) {
                        let sink = conn.as_mut();
                        let _ = sink.write_all(&wire);
                        let _ = sink.flush();
                    }
                    let _ = conn.shutdown_both();
                    continue;
                }
                let writer_conn = match conn.try_clone() {
                    Ok(clone) => clone,
                    Err(err) => {
                        eprintln!("starling-runtime-host: connection clone failed: {err}");
                        let _ = conn.shutdown_both();
                        continue;
                    }
                };
                let closer = match conn.try_clone() {
                    Ok(clone) => clone,
                    Err(err) => {
                        eprintln!("starling-runtime-host: connection clone failed: {err}");
                        let _ = conn.shutdown_both();
                        continue;
                    }
                };
                let (outbound_tx, outbound_rx) = bounded(outbound_capacity);
                let state = Arc::new(ConnState {
                    outbound: outbound_tx,
                    closed: AtomicBool::new(false),
                    closer,
                });
                shared
                    .conns
                    .lock()
                    .expect("conn registry")
                    .push(Arc::clone(&state));
                shared.live_connections.fetch_add(1, Ordering::SeqCst);

                let reader = spawn_conn_thread("starling-host-conn-read", {
                    let shared = Arc::clone(&shared);
                    let state = Arc::clone(&state);
                    let policy = Arc::clone(&policy);
                    let conn = conn;
                    move || connection_reader(shared, state, policy, conn)
                });
                let writer = spawn_conn_thread("starling-host-conn-write", {
                    let shared = Arc::clone(&shared);
                    let state = Arc::clone(&state);
                    move || connection_writer(shared, state, writer_conn, outbound_rx)
                });
                let mut threads = shared.conn_threads.lock().expect("conn threads");
                threads.push(reader);
                threads.push(writer);
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

fn spawn_conn_thread(name: &str, run: impl FnOnce() + Send + 'static) -> JoinHandle<()> {
    std::thread::Builder::new()
        .name(name.to_string())
        .spawn(run)
        .expect("host connection thread spawn")
}

/// One connection's inbound side: auth, hello, then frames.
fn connection_reader(
    shared: Arc<HostShared>,
    state: Arc<ConnState>,
    policy: Arc<dyn PeerPolicy>,
    mut conn: Box<dyn TransportConn>,
) {
    // Fail closed: a credential read failure is an auth failure.
    let credentials = conn.peer_credentials().unwrap_or_default();
    if let Err(refused) = policy.authenticate(credentials) {
        let error = Frame::TransportError {
            code: TransportErrorCode::AuthFailed,
            detail: refused.to_string(),
        };
        // Best-effort direct write: the writer thread has nothing queued
        // (no hello was sent), so the two never race.
        if let Ok(wire) = encode(&error, shared.max_frame_bytes) {
            let sink = conn.as_mut();
            let _ = sink.write_all(&wire);
            let _ = sink.flush();
        }
        // The frame went out on our own handle; the writer (with nothing
        // queued — no hello was ever sent) shuts the socket down after
        // the queue closes.
        state.mark_closed();
        unregister(&shared, &state);
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
        unregister(&shared, &state);
        return;
    }

    let mut reader = FrameReader::new(conn, shared.max_frame_bytes);
    let mut rate = SlidingWindow::new(shared.command_rate);
    while !shared.shutdown.load(Ordering::SeqCst) && !state.closed.load(Ordering::SeqCst) {
        match reader.read_frame() {
            Ok(frame) => {
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
                    Frame::Command { mut envelope } => {
                        if handle_command(&shared, &state, &mut envelope).is_err() {
                            break;
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
                            terminate(
                                &state,
                                TransportErrorCode::SlowConsumer,
                                "snapshot reply would overflow the outbound queue".to_string(),
                            );
                            break;
                        }
                    }
                    Frame::Hello { .. }
                    | Frame::Receipt { .. }
                    | Frame::Event { .. }
                    | Frame::Snapshot { .. }
                    | Frame::TransportError { .. }
                    | Frame::Bye { .. } => {
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
                continue;
            }
            Err(FrameError::Io(_)) => break,
        }
    }
    state.mark_closed();
    unregister(&shared, &state);
}

/// Bridges one command envelope into the runtime and returns the receipt
/// frame. The envelope is the I3 wire form; when the client left `seq`
/// out, the host assigns it from the runtime's own frontier (see
/// `RuntimeClient::assign_seq`) so a reconnecting client cannot collide
/// with stream positions a dead connection consumed.
fn handle_command(
    shared: &HostShared,
    state: &ConnState,
    envelope: &mut serde_json::Value,
) -> Result<(), ()> {
    let corr = envelope
        .as_object()
        .and_then(|object| object.get("corr"))
        .and_then(serde_json::Value::as_str)
        .map(str::to_string);
    if envelope
        .as_object()
        .is_some_and(|object| !object.contains_key("seq"))
    {
        let seq = shared.client.assign_seq(corr.as_deref());
        if let Some(object) = envelope.as_object_mut() {
            object.insert("seq".to_string(), serde_json::Value::from(seq));
        }
    }
    let seq = envelope
        .as_object()
        .and_then(|object| object.get("seq"))
        .and_then(serde_json::Value::as_u64);
    let id = envelope
        .as_object()
        .and_then(|object| object.get("id"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or("?")
        .to_string();
    let result = shared.client.send_raw(envelope.clone());
    state.try_deliver(Frame::Receipt {
        req: id,
        seq,
        result,
    })
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
                            use std::io::Write;
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
                use std::io::Write;
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
/// delivers the frame and then ends the stream (the reader-side
/// violations all have a peer that is demonstrably still reading — it
/// just sent us a frame).
fn terminate(state: &ConnState, code: TransportErrorCode, detail: String) {
    let _ = state.try_deliver(Frame::TransportError { code, detail });
    state.mark_closed();
}

fn unregister(shared: &HostShared, state: &Arc<ConnState>) {
    shared
        .conns
        .lock()
        .expect("conn registry")
        .retain(|registered| !Arc::ptr_eq(registered, state));
    shared.live_connections.fetch_sub(1, Ordering::SeqCst);
}

/// Fans runtime events out to every live connection. This decouples the
/// runtime's EventBus (whose backpressure semantics are in-process:
/// machines stall on a full subscriber) from IPC clients (whose policy
/// is the opposite: a stopped reader is dropped, machines never stall —
/// §1 Mode B's renderer-kill posture).
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
                let conns = shared.conns.lock().expect("conn registry").clone();
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

/// Renews the storage-v2 lease heartbeat. The flock on the identity file
/// is the ownership signal the OS maintains by itself; the heartbeat is
/// the fallback for flock-less hosts, so a renewal failure is logged and
/// outlived, never fatal here.
fn lease_heartbeat(lease: Arc<Mutex<StoreV2>>, shared: Arc<HostShared>) {
    loop {
        for _ in 0..20 {
            // ~10s at the default 30s TTL (TTL/3 cadence), in poll-sized
            // slices so shutdown is prompt.
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
