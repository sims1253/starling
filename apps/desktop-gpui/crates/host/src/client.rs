//! The client side of the host's IPC: what the GPUI app and the Electron
//! comparison adapter will hold (E17 §1 Mode B — "the UI is a
//! projection: commands in, events + snapshots out").
//!
//! The wire is [`crate::frame`] — every application payload is the I3
//! envelope; receipts carry the runtime's own
//! `Result<Receipt, Rejection>` serialization, so an IPC client observes
//! the exact rejection an embedded client would (the conformance suite
//! exploits this: the invalid-fixture corpus replays over the socket and
//! must produce the same rejections as in-process).
//!
//! `seq`: [`HostClient::send`] sends envelopes **without** `seq` and lets
//! the host assign it from the runtime's frontier — a reconnecting
//! client therefore continues each stream monotonically without knowing
//! the dead connection's positions. [`HostClient::send_raw`] passes a
//! client-supplied envelope through untouched (own `seq`, own
//! monotonicity exposure) — the envelope-level path the NACK contract
//! tests use.
//!
//! # Draining contract (receipts never wait on events)
//!
//! The reader thread never parks delivering an event: when the event
//! channel is full (the application stopped draining), events accumulate
//! in a bounded local backlog while receipts and snapshots keep flowing —
//! they have their own reply channels and never queue behind events.
//! Past [`EVENT_BACKLOG_CAP`] undelivered events the connection is failed
//! outright (the same slow-consumer posture the host applies: the client
//! reconnects and resynchronizes from the snapshot, which is always the
//! recovery story for a lost event tail).
//!
//! Delivery semantics worth stating for the Mode B switchover: a
//! [`ClientError::Timeout`] on a send does **not** mean the command was
//! not executed — receipts issue at acceptance, and a reply lost to a
//! closed connection is indistinguishable from a slow one. A retry gets
//! a fresh host-assigned `seq` (it passes the router's monotonicity
//! check) and executes again: sends are at-least-once, and a caller that
//! cannot tolerate a duplicate resolves it through the snapshot, the same
//! way a reconnect does.

use std::collections::{HashMap, VecDeque};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::Value;
use starling_runtime::bus::{new_id, now_ts};
use starling_runtime::channel::{bounded, Receiver, RecvError, Sender, TrySendError};
use starling_runtime::machine::{Receipt, Rejection};
use starling_runtime::protocol::Command;

use crate::frame::{encode, Frame, FrameError, FrameReader};
use crate::platform::{self, TransportConn};

/// How long a send waits for its receipt. Receipts are issued at command
/// *acceptance* (not outcome completion), so this is deliberately far
/// above any healthy machine's answer time.
const REPLY_TIMEOUT: Duration = Duration::from_secs(10);

/// The reader's read-poll slice: an idle connection wakes this often so
/// the event backlog can flush the moment the application makes room
/// (platforms whose transport cannot poll — see
/// [`TransportConn::set_read_timeout`] — flush on the next inbound frame
/// instead).
const READ_POLL: Duration = Duration::from_millis(250);

/// How many events accumulate locally while the application is not
/// draining before the connection is failed. Mirrors the host's own
/// outbound posture ([`crate::limits::DEFAULT_OUTBOUND_CAPACITY`]): by
/// the time both sides' bounds are exhausted, the reconnect-and-snapshot
/// resynchronization is the designed answer, not a degraded one.
const EVENT_BACKLOG_CAP: usize = 1024;

/// What the host told us at connect time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostInfo {
    pub protocol: u32,
    pub owner_id: String,
    pub pid: u32,
    pub max_frame_bytes: u64,
    pub rate_max: u32,
    pub rate_window_ms: u64,
}

/// One event envelope off the wire (the I3 wire form; typed event
/// parsing stays with the envelope's owner — see the module docs).
#[derive(Debug, Clone, PartialEq)]
pub struct EventWire(pub Value);

impl EventWire {
    pub fn type_name(&self) -> &str {
        self.0.get("type").and_then(Value::as_str).unwrap_or("")
    }

    pub fn payload(&self) -> &Value {
        self.0.get("payload").unwrap_or(&Value::Null)
    }

    pub fn corr(&self) -> Option<&str> {
        self.0.get("corr").and_then(Value::as_str)
    }

    pub fn seq(&self) -> Option<u64> {
        self.0.get("seq").and_then(Value::as_u64)
    }
}

/// What came back for a registered request.
enum Reply {
    /// The receipt plus the `seq` the host routed the command under.
    Receipt(Result<Receipt, Rejection>, Option<u64>),
    Snapshot(Value),
}

/// Why a client call failed.
#[derive(Debug, thiserror::Error)]
pub enum ClientError {
    #[error("connect to {path:?} failed: {source}")]
    Connect {
        path: std::path::PathBuf,
        source: std::io::Error,
    },
    #[error("the connection closed: {0}")]
    Closed(String),
    #[error("no reply within {REPLY_TIMEOUT:?}")]
    Timeout,
    #[error("protocol violation from the host: {0}")]
    Protocol(String),
    /// The host answered with the runtime's typed rejection — the exact
    /// `Rejection` an embedded client would have received (the
    /// conformance suite compares these for equality across the
    /// transport).
    #[error("the runtime refused the command: {0}")]
    Rejected(Rejection),
}

/// A live connection to the host. Clone-free single connection; the
/// events stream is read through [`HostClient::recv_event_timeout`] on
/// the caller's thread (GPUI's update loop, a test loop).
pub struct HostClient {
    writer: Mutex<Box<dyn TransportConn>>,
    pending: Arc<Mutex<HashMap<String, Sender<Reply>>>>,
    events: Receiver<EventWire>,
    closed: Arc<AtomicBool>,
    close_reason: Arc<Mutex<Option<String>>>,
    pub info: HostInfo,
}

impl HostClient {
    /// Connects and completes the `hello` handshake. Fails when nothing
    /// serves the endpoint (the caller decides whether to launch the
    /// host binary and retry).
    pub fn connect(path: &Path) -> Result<HostClient, ClientError> {
        let conn = platform::connect(path).map_err(|source| ClientError::Connect {
            path: path.to_path_buf(),
            source,
        })?;
        let writer = conn.try_clone().map_err(|source| ClientError::Connect {
            path: path.to_path_buf(),
            source,
        })?;
        // Arm the read poll so the reader thread's idle wakeups can flush
        // the event backlog; where the platform cannot poll this is a
        // no-op and the backlog flushes on the next inbound frame instead.
        let _ = conn.set_read_timeout(Some(READ_POLL));

        let pending: Arc<Mutex<HashMap<String, Sender<Reply>>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let closed = Arc::new(AtomicBool::new(false));
        let close_reason: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
        let (event_tx, events) = bounded(4096);
        let (hello_tx, hello_rx) = bounded::<Result<HostInfo, String>>(1);

        let reader = std::thread::Builder::new()
            .name("starling-host-client-read".to_string())
            .spawn({
                let pending = Arc::clone(&pending);
                let closed = Arc::clone(&closed);
                let close_reason = Arc::clone(&close_reason);
                move || client_reader(conn, pending, event_tx, hello_tx, closed, close_reason)
            })
            .map_err(|err| {
                // No reader will close the connection: do it here so
                // nothing leaks behind a failed handshake.
                let _ = writer.shutdown_both();
                ClientError::Protocol(format!("reader spawn: {err}"))
            })?;
        reader.detach();

        let info = match hello_rx.recv_timeout(REPLY_TIMEOUT) {
            Ok(Ok(info)) => info,
            Ok(Err(detail)) => {
                let _ = writer.shutdown_both();
                return Err(ClientError::Protocol(format!("handshake failed: {detail}")));
            }
            Err(RecvError::Timeout) => {
                // No HostClient object is constructed on this path, so its
                // Drop (which ends both directions) never runs: shut the
                // connection down explicitly so the detached reader's
                // blocking read returns and its thread exits instead of
                // leaking per failed connect attempt.
                let _ = writer.shutdown_both();
                return Err(ClientError::Timeout);
            }
            Err(RecvError::Closed) => {
                let _ = writer.shutdown_both();
                return Err(ClientError::Closed("no hello from the host".to_string()));
            }
        };

        Ok(HostClient {
            writer: Mutex::new(writer),
            pending,
            events,
            closed,
            close_reason,
            info,
        })
    }

    /// Sends a typed command; the host assigns `seq`. `Ok` means the
    /// owning machine accepted it — outcome events follow on the event
    /// stream.
    pub fn send(&self, corr: Option<&str>, command: Command) -> Result<Receipt, ClientError> {
        self.send_and_seq(corr, command).map(|(receipt, _)| receipt)
    }

    /// [`Self::send`], also returning the `seq` the host assigned — what
    /// a client needs to reconstruct the full envelope it sent (the
    /// conformance suite replays those through the oracle).
    pub fn send_and_seq(
        &self,
        corr: Option<&str>,
        command: Command,
    ) -> Result<(Receipt, Option<u64>), ClientError> {
        let mut envelope = serde_json::Map::new();
        envelope.insert("v".into(), Value::from(1u64));
        envelope.insert("id".into(), Value::from(new_id("cmd")));
        envelope.insert("ts".into(), Value::from(now_ts()));
        if let Some(corr) = corr {
            envelope.insert("corr".into(), Value::from(corr));
        }
        envelope.insert("type".into(), Value::from(command.type_name()));
        envelope.insert("payload".into(), command.payload_value());
        match self.exchange(Frame::Command {
            envelope: Value::Object(envelope),
        })? {
            (receipt, seq) => Ok((receipt, seq)),
        }
    }

    /// Sends a raw envelope — the envelope-level path (client-owned `id`,
    /// `ts`, `seq`): the NACK contract, the monotonicity check, and any
    /// conformance-fixture command replay.
    pub fn send_raw(&self, envelope: Value) -> Result<Receipt, ClientError> {
        if envelope.get("id").and_then(Value::as_str).is_none() {
            return Err(ClientError::Protocol(
                "a raw envelope must carry its own id".to_string(),
            ));
        }
        self.exchange(Frame::Command { envelope })
            .map(|(receipt, _)| receipt)
    }

    /// Requests the runtime projection snapshot.
    pub fn snapshot(&self) -> Result<Value, ClientError> {
        let req = new_id("snap");
        self.exchange_reply(
            Frame::GetSnapshot { req: req.clone() },
            req,
            |reply| match reply {
                Reply::Snapshot(value) => Ok(value),
                Reply::Receipt(..) => Err(ClientError::Protocol(
                    "snapshot request answered by a receipt".to_string(),
                )),
            },
        )
    }

    fn exchange(&self, frame: Frame) -> Result<(Receipt, Option<u64>), ClientError> {
        let id = match &frame {
            Frame::Command { envelope } => envelope
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or("?")
                .to_string(),
            _ => {
                return Err(ClientError::Protocol(
                    "only command envelopes exchange for receipts".to_string(),
                ))
            }
        };
        self.exchange_reply(frame, id, |reply| match reply {
            Reply::Receipt(result, seq) => result
                .map_err(ClientError::Rejected)
                .map(|receipt| (receipt, seq)),
            Reply::Snapshot(_) => Err(ClientError::Protocol(
                "command answered by a snapshot".to_string(),
            )),
        })
    }

    fn exchange_reply<T>(
        &self,
        frame: impl Into<Frame>,
        id: String,
        interpret: impl FnOnce(Reply) -> Result<T, ClientError>,
    ) -> Result<T, ClientError> {
        let frame = frame.into();
        if self.closed.load(Ordering::SeqCst) {
            return Err(ClientError::Closed(self.close_reason()));
        }
        let (tx, rx) = bounded(1);
        {
            let mut pending = self.pending.lock().expect("pending map");
            // Ids must be unique among in-flight requests: a duplicate
            // would overwrite the first caller's reply channel and turn
            // its outcome into a full reply-timeout. Refuse the second
            // caller instead (the check and the insert share one lock, so
            // the guard is exact).
            if pending.contains_key(&id) {
                return Err(ClientError::Protocol(format!(
                    "request id {id:?} is already in flight; ids must be \
                     unique among concurrent requests"
                )));
            }
            pending.insert(id.clone(), tx);
        }
        let result = (|| {
            // Encode against the host's advertised cap so an oversized
            // send is refused here, before any bytes hit the wire — a
            // server-side refusal costs the whole connection.
            let cap = usize::try_from(self.info.max_frame_bytes).unwrap_or(usize::MAX);
            let wire = encode(&frame, cap)
                .map_err(|err| ClientError::Protocol(format!("frame does not encode: {err:?}")))?;
            {
                use std::io::Write;
                let mut writer = self.writer.lock().expect("writer lock");
                writer
                    .write_all(&wire)
                    .and_then(|()| writer.flush())
                    .map_err(|err| ClientError::Closed(format!("write failed: {err}")))?;
            }
            match rx.recv_timeout(REPLY_TIMEOUT) {
                Ok(reply) => interpret(reply),
                Err(RecvError::Timeout) => Err(ClientError::Timeout),
                // The reader delivered nothing and the reply channel is
                // gone: the connection failed the call.
                Err(RecvError::Closed) => Err(ClientError::Closed(self.close_reason())),
            }
        })();
        self.pending.lock().expect("pending map").remove(&id);
        result
    }

    /// The next event envelope, if one arrived within `timeout`
    /// ([`RecvError::Timeout`] on idle — the normal poll result).
    pub fn recv_event_timeout(&self, timeout: Duration) -> Result<EventWire, RecvError> {
        self.events.recv_timeout(timeout)
    }

    /// Non-blocking event poll.
    pub fn try_recv_event(&self) -> Result<EventWire, RecvError> {
        self.events.try_recv()
    }

    pub fn is_closed(&self) -> bool {
        self.closed.load(Ordering::SeqCst)
    }

    pub fn close_reason(&self) -> String {
        self.close_reason
            .lock()
            .expect("close reason")
            .clone()
            .unwrap_or_else(|| "connection ended".to_string())
    }
}

impl Drop for HostClient {
    fn drop(&mut self) {
        // End both directions so the detached reader thread's blocking
        // read returns; it then drops the original handle.
        if let Ok(writer) = self.writer.lock() {
            let _ = writer.shutdown_both();
        }
    }
}

/// The detached reader thread's body.
fn client_reader(
    conn: Box<dyn TransportConn>,
    pending: Arc<Mutex<HashMap<String, Sender<Reply>>>>,
    events: Sender<EventWire>,
    hello: Sender<Result<HostInfo, String>>,
    closed: Arc<AtomicBool>,
    close_reason: Arc<Mutex<Option<String>>>,
) {
    let fail = |reason: String| {
        closed.store(true, Ordering::SeqCst);
        *close_reason.lock().expect("close reason") = Some(reason.clone());
        // Wake every waiter with the closed flag set.
        pending.lock().expect("pending map").clear();
    };
    let mut reader = FrameReader::new(conn, usize::MAX);
    let mut hello_done = false;
    // Events the application has not made room for yet. Delivering from
    // this backlog instead of parking on a full channel is what keeps
    // receipts flowing while events back up: the reader stays free to
    // pull the next frame off the socket, and receipts/snapshots answer
    // through their own per-request channels.
    let mut backlog: VecDeque<EventWire> = VecDeque::new();
    loop {
        // An idle read-poll timeout is the flush tick: events must move
        // the moment the application drains, not only when the host says
        // something else.
        flush_backlog(&events, &mut backlog);
        match reader.read_frame() {
            Ok(Frame::Hello {
                protocol,
                owner_id,
                pid,
                max_frame_bytes,
                rate_max,
                rate_window_ms,
            }) => {
                if hello_done {
                    fail("second hello".to_string());
                    break;
                }
                hello_done = true;
                let _ = hello.try_send(Ok(HostInfo {
                    protocol,
                    owner_id,
                    pid,
                    max_frame_bytes,
                    rate_max,
                    rate_window_ms,
                }));
            }
            Ok(Frame::Receipt { req, seq, result }) => {
                deliver(&pending, &req, Reply::Receipt(result, seq));
            }
            Ok(Frame::Snapshot { req, snapshot }) => {
                deliver(&pending, &req, Reply::Snapshot(snapshot));
            }
            Ok(Frame::Event { envelope }) => {
                backlog.push_back(EventWire(envelope));
                flush_backlog(&events, &mut backlog);
                if backlog.len() > EVENT_BACKLOG_CAP {
                    // The application is not draining at all. The
                    // connection is failed on purpose — the same
                    // slow-consumer posture the host applies to a client
                    // that stopped reading — rather than letting the
                    // backlog (and with it the reader's memory) grow
                    // without bound. A reconnect resynchronizes from the
                    // snapshot; nothing acknowledged is lost.
                    fail(format!(
                        "event stream undrained past the {}-event backlog cap; \
                         reconnect and resynchronize from the snapshot",
                        EVENT_BACKLOG_CAP
                    ));
                    break;
                }
            }
            Ok(Frame::TransportError { code, detail }) => {
                let reason = format!("transport {}: {detail}", code.as_str());
                if !hello_done {
                    let _ = hello.try_send(Err(reason.clone()));
                }
                fail(reason);
                break;
            }
            Ok(Frame::Bye { reason }) => {
                fail(format!("host said goodbye: {reason}"));
                break;
            }
            Ok(Frame::Command { .. } | Frame::GetSnapshot { .. }) => {
                fail("host sent a client frame".to_string());
                break;
            }
            Err(FrameError::Eof) => {
                fail("connection closed by the host".to_string());
                break;
            }
            Err(FrameError::Io(err))
                if err.kind() == std::io::ErrorKind::WouldBlock
                    || err.kind() == std::io::ErrorKind::TimedOut =>
            {
                // Read-poll timeout: idle, not an error. The loop head
                // flushed the backlog; keep waiting.
                continue;
            }
            Err(FrameError::Io(err)) => {
                fail(format!("connection error: {err}"));
                break;
            }
            Err(FrameError::TooLarge { declared, .. }) => {
                fail(format!("host sent a {declared}-byte frame over the cap"));
                break;
            }
            Err(FrameError::Malformed(detail)) => {
                fail(format!("host sent a malformed frame: {detail}"));
                break;
            }
        }
    }
}

/// Moves backlog into the event channel while there is room; never
/// parks. A dropped receiver clears the backlog (the client is gone).
fn flush_backlog(events: &Sender<EventWire>, backlog: &mut VecDeque<EventWire>) {
    while let Some(event) = backlog.pop_front() {
        match events.try_send(event) {
            Ok(()) => {}
            Err(TrySendError::Full(event)) => {
                backlog.push_front(event);
                break;
            }
            Err(TrySendError::Closed(_)) => {
                backlog.clear();
                break;
            }
        }
    }
}

fn deliver(pending: &Mutex<HashMap<String, Sender<Reply>>>, req: &str, reply: Reply) {
    if let Some(sender) = pending.lock().expect("pending map").remove(req) {
        let _ = sender.try_send(reply);
    }
    // An unknown req is a late reply to a timed-out request: dropped,
    // and the map entry was already removed at timeout.
}

/// JoinHandle helper: the reader thread outlives the client object by
/// design (see [`HostClient::drop`]); this keeps the spawn site honest
/// about never joining it.
trait Detach {
    fn detach(self);
}

impl Detach for std::thread::JoinHandle<()> {
    fn detach(self) {}
}
