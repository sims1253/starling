//! The agent-dictation ask surface: what a coding agent's MCP server
//! sees on the other side of the IPC transport.
//!
//! - [`Allowlist`] gates every `Frame::AgentHello`. Default deny: no
//!   file, or an empty one, admits no agent client.
//! - [`broker_loop`] owns every `Frame::AskUser`. It serializes asks
//!   (one visible prompt, one capture at a time), opens the microphone
//!   only after the Starling app acks `PromptAck { visible: true }`,
//!   and then drives the existing capture path — `context.snapshot` →
//!   `mode.set` → `capture.start` → `capture.stop` → `jobs.submit` —
//!   like any other client. There is no second recording stack.
//!
//! # Lifecycle rules
//!
//! - **Timeout**: the `timeout_ms` budget starts at admission (a queued
//!   ask shows no prompt) and ends when the microphone closes; expiry
//!   before that aborts the take with `NoAnswer { timeout }`. A
//!   captured take is never discarded on the clock: persistence and
//!   transcription run to their own outcome.
//! - **Prompt ack bound**: no visible ack within
//!   `min(ACK_BOUND, timeout)` fails with `Error { no_prompt_ack }` and
//!   the microphone is never touched.
//! - **Agent cancel / disconnect**: `Frame::AskCancel`, or the asking
//!   connection ending for any reason, aborts the take
//!   (`NoAnswer { agent_cancelled }`). The host enforces this, so it
//!   holds even when the MCP process is killed.
//! - **User dismissal**: `PromptAck { visible: false }` declines the
//!   ask before capture (`declined`) and stops it after
//!   (`user_cancelled`); the acking app disconnecting before or during
//!   the take counts as a dismissal.
//! - **Settling**: an ask that ends leaves cleanup behind (aborting
//!   its take, expiring its provisional context). The broker retries it
//!   until the runtime accepts it or reports nothing left to undo, and
//!   admits no other ask until then. It only ever aborts the take it
//!   started: the capture machine refuses a stop or abort naming
//!   another take.
//!
//! # Trust boundary
//!
//! Prompts go only to connections holding the app role, and only they
//! may answer them. No app-role credential exists yet, so the surface
//! fails closed: every ask is refused with `Error { no_app }`. Agent
//! connections are admitted by the allowlist (a token the user
//! provisions in `mcp-clients.json` and in the agent's MCP config),
//! never see prompts, and cannot send runtime commands; they reach the
//! microphone only through an ask the app shows.

use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use starling_runtime::bus::{new_id, EventMessage};
use starling_runtime::channel::{bounded, Receiver, RecvError, TrySendError};
use starling_runtime::machine::Rejection;
use starling_runtime::protocol::{Command, Event as RtEvent, Manual};

use crate::frame::{AskOutcome, Frame, NoAnswerReason};
use crate::server::{lock_registry, BrokerMsg, ConnState, HostShared};

pub const MAX_QUESTIONS: usize = 8;
pub const MAX_QUESTION_CHARS: usize = 2_000;
pub const MIN_TIMEOUT_MS: u64 = 1_000;
pub const MAX_TIMEOUT_MS: u64 = 600_000;

/// How many asks may wait behind the live one before `queue_full`.
pub const MAX_WAITING_ASKS: usize = 4;

/// How long an admitted ask waits for a visible ack (clamped by its
/// timeout) before failing with `no_prompt_ack`.
pub const ACK_BOUND: Duration = Duration::from_secs(10);

/// The allowlist's file name inside the host's data root.
pub const ALLOWLIST_FILE: &str = "mcp-clients.json";

/// Prefix of every ask id, and so of every runtime corr the broker uses.
/// Reserved: the host refuses client commands whose corr carries it.
pub(crate) const ASK_PREFIX: &str = "ask_";

/// Runtime events buffered for the broker. An ask produces a handful
/// plus capture progress, so overflow means the broker is stuck.
const EVENT_BUFFER: usize = 1024;

/// How often the broker re-checks deadlines and drains its event
/// subscription (the bus backpressures slow subscribers).
const POLL: Duration = Duration::from_millis(20);

/// Checks an ask's shape: `Err((code, message))` on violation. Both the
/// MCP layer and the broker apply it, the broker because raw IPC clients
/// bypass the MCP layer.
pub fn validate_ask(questions: &[String], timeout_ms: u64) -> Result<(), (&'static str, String)> {
    if questions.is_empty() || questions.len() > MAX_QUESTIONS {
        return Err((
            "invalid_questions",
            format!("questions must carry 1..={MAX_QUESTIONS} entries"),
        ));
    }
    if questions
        .iter()
        .any(|q| q.trim().is_empty() || q.chars().count() > MAX_QUESTION_CHARS)
    {
        return Err((
            "invalid_questions",
            format!("every question needs 1..={MAX_QUESTION_CHARS} non-whitespace characters"),
        ));
    }
    if !(MIN_TIMEOUT_MS..=MAX_TIMEOUT_MS).contains(&timeout_ms) {
        return Err((
            "invalid_timeout",
            format!("timeout_ms must be {MIN_TIMEOUT_MS}..={MAX_TIMEOUT_MS}"),
        ));
    }
    Ok(())
}

// ------------------------------------------------------------------ //
// The allowlist
// ------------------------------------------------------------------ //

/// The agent clients allowed to use the ask surface, by name and token.
///
/// ```json
/// { "version": 1, "clients": [ { "name": "claude-code", "token": "…" } ] }
/// ```
///
/// Unknown versions and duplicate names are refused rather than guessed
/// at. Loaded once at host startup.
#[derive(Debug, Clone, Default)]
pub struct Allowlist {
    clients: HashMap<String, String>,
}

/// Why an existing allowlist file could not be loaded (a missing file
/// is an empty list, not an error).
#[derive(Debug, thiserror::Error)]
pub enum AllowlistError {
    #[error("the allowlist {path:?} cannot be read: {source}")]
    Read {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("the allowlist {path:?} is not valid: {detail}")]
    Malformed { path: PathBuf, detail: String },
}

impl Allowlist {
    pub fn load(path: &Path) -> Result<Allowlist, AllowlistError> {
        match std::fs::read(path) {
            Ok(bytes) => Allowlist::from_bytes(path, &bytes),
            Err(source) if source.kind() == std::io::ErrorKind::NotFound => {
                Ok(Allowlist::default())
            }
            Err(source) => Err(AllowlistError::Read {
                path: path.to_path_buf(),
                source,
            }),
        }
    }

    pub fn from_bytes(path: &Path, bytes: &[u8]) -> Result<Allowlist, AllowlistError> {
        #[derive(serde::Deserialize)]
        struct Entry {
            name: String,
            token: String,
        }
        #[derive(serde::Deserialize)]
        struct File {
            version: u32,
            clients: Vec<Entry>,
        }
        let malformed = |detail: String| AllowlistError::Malformed {
            path: path.to_path_buf(),
            detail,
        };
        let file: File = serde_json::from_slice(bytes)
            .map_err(|err| malformed(format!("not valid allowlist JSON: {err}")))?;
        if file.version != 1 {
            return Err(malformed(format!(
                "unsupported allowlist version {} (this host reads version 1)",
                file.version
            )));
        }
        let mut clients = HashMap::new();
        for Entry { name, token } in file.clients {
            if name.is_empty() || token.is_empty() {
                return Err(malformed(
                    "every client needs a non-empty name and token".to_string(),
                ));
            }
            if clients.contains_key(&name) {
                return Err(malformed(format!("duplicate client name {name:?}")));
            }
            clients.insert(name, token);
        }
        Ok(Allowlist { clients })
    }

    pub fn authenticate(&self, client: &str, token: &str) -> Result<(), String> {
        // Plain equality: a timing side channel over a same-user
        // transport only helps someone who already is the user.
        match self.clients.get(client) {
            Some(expected) if expected == token => Ok(()),
            Some(_) => Err(format!(
                "agent client {client:?} is allowlisted but the token is wrong"
            )),
            None => Err(format!("agent client {client:?} is not allowlisted")),
        }
    }
}

// ------------------------------------------------------------------ //
// The broker
// ------------------------------------------------------------------ //

struct Ask {
    /// The asking agent connection; the result and liveness ride it.
    conn: Arc<ConnState>,
    /// The agent's correlation token, echoed on `AskResult`.
    client_req: String,
    /// The broker-scoped id used on prompt frames and as the runtime
    /// corr. Agent tokens are only unique per connection.
    ask_id: String,
    questions: Vec<String>,
    /// Budget start; reset at admission.
    started: Instant,
    timeout: Duration,
    /// The app connections the prompt was shown to.
    shown_to: Vec<Arc<ConnState>>,
    /// The connection whose visible ack opened the gate.
    acker: Option<Arc<ConnState>>,
}

impl Ask {
    fn is(&self, conn: &Arc<ConnState>, client_req: &str) -> bool {
        self.client_req == client_req && Arc::ptr_eq(&self.conn, conn)
    }

    fn acked_by(&self, conn: &Arc<ConnState>) -> bool {
        self.acker
            .as_ref()
            .is_some_and(|acker| Arc::ptr_eq(acker, conn))
    }

    /// Answers the agent and hides the prompt everywhere.
    fn resolve(self, shared: &HostShared, outcome: AskOutcome, hide_reason: &str) {
        let _ = self.conn.try_deliver(Frame::AskResult {
            req: self.client_req,
            outcome,
        });
        for conn in app_connections(shared) {
            let _ = conn.try_deliver(Frame::HidePrompt {
                req: self.ask_id.clone(),
                reason: hide_reason.to_string(),
            });
        }
    }
}

/// Where the live ask is in the capture flow. The route is frozen
/// (snapshot, mode) only after the visibility ack.
enum Phase {
    /// Prompt shown, waiting for the first ack.
    Prompting,
    /// Waiting `context.targetSnapshot` (corr `<ask>-ctx`).
    Snapshotting,
    /// Waiting `mode.decision` (corr `<ask>-mode`).
    SettingMode,
    /// `capture.start` accepted; the microphone is open.
    Recording,
    /// `capture.stop` accepted; waiting `capture.stopped`.
    Persisting,
    /// `jobs.submit` accepted; waiting the job's outcome.
    Transcribing { job: String },
}

/// The broker thread. Exits when the host shuts down, resolving
/// everything live or queued with `Error { shutting_down }`.
pub(crate) fn broker_loop(shared: Arc<HostShared>, inbox: Receiver<BrokerMsg>) {
    // The broker waits for receipts from runtime actors, and an actor
    // publishing into a full subscription waits for its reader, so a
    // separate thread keeps the subscription drained. It forwards only
    // the broker's own events, into a bounded buffer; anything that
    // does not fit is dropped and flagged, and the broker then gives up
    // the live ask rather than miss its outcome.
    let (event_tx, events) = bounded(EVENT_BUFFER);
    let overflowed = Arc::new(AtomicBool::new(false));
    let forwarder = {
        let shared = Arc::clone(&shared);
        let overflowed = Arc::clone(&overflowed);
        let subscription = shared.client.subscribe();
        std::thread::Builder::new()
            .name("starling-host-agent-events".to_string())
            .spawn(move || {
                while !shared.shutdown.load(Ordering::SeqCst) {
                    match subscription.recv_timeout(POLL) {
                        Ok(message) => {
                            let ours = message
                                .corr
                                .as_deref()
                                .is_some_and(|corr| corr.starts_with(ASK_PREFIX));
                            if !ours {
                                continue;
                            }
                            match event_tx.try_send(message) {
                                Ok(()) => {}
                                Err(TrySendError::Full(_)) => {
                                    overflowed.store(true, Ordering::SeqCst)
                                }
                                Err(TrySendError::Closed(_)) => return,
                            }
                        }
                        Err(RecvError::Timeout) => {}
                        Err(RecvError::Closed) => return,
                    }
                }
            })
            .expect("agent event thread spawns")
    };
    let mut broker = Broker {
        shared,
        live: None,
        waiting: VecDeque::new(),
        settling: Vec::new(),
    };
    loop {
        if broker.shared.shutdown.load(Ordering::SeqCst) {
            break;
        }
        // Cancellations and disconnects first, so the gate never
        // advances on behalf of an ask that is already dead.
        while let Ok(msg) = inbox.try_recv() {
            broker.on_msg(msg);
        }
        if overflowed.swap(false, Ordering::SeqCst) {
            broker.fail_live(
                "capture_failed",
                "the host fell behind the runtime's events for this ask".to_string(),
            );
        }
        while let Ok(message) = events.try_recv() {
            broker.on_event(message);
        }
        broker.tick();
        broker.publish_capturing();
        match inbox.recv_timeout(POLL) {
            Ok(msg) => broker.on_msg(msg),
            Err(RecvError::Timeout) => {}
            Err(RecvError::Closed) => break,
        }
        broker.publish_capturing();
    }
    broker.finish_all("the host is shutting down");
    drop(events);
    let _ = forwarder.join();
}

struct Broker {
    shared: Arc<HostShared>,
    live: Option<(Ask, Phase)>,
    waiting: VecDeque<Ask>,
    /// Cleanup an ended ask still owes the runtime. No ask is admitted
    /// until every entry has settled.
    settling: Vec<Cleanup>,
}

/// A command that undoes what an ended ask left behind, retried until
/// the runtime accepts it or says there is nothing left to undo.
struct Cleanup {
    corr: String,
    command: fn() -> Command,
}

impl Broker {
    /// Mirrors whether the live ask has the microphone or its take in
    /// hand into [`crate::server::HostShared`]'s `ask_capturing`, for a
    /// retire's idle check. Only the broker writes it.
    fn publish_capturing(&self) {
        let capturing = matches!(
            self.live,
            Some((
                _,
                Phase::Recording | Phase::Persisting | Phase::Transcribing { .. }
            ))
        );
        self.shared.ask_capturing.store(capturing, Ordering::SeqCst);
    }

    fn on_msg(&mut self, msg: BrokerMsg) {
        match msg {
            BrokerMsg::Ask {
                conn,
                client_req,
                questions,
                timeout_ms,
            } => {
                self.on_ask(conn, client_req, questions, timeout_ms);
                self.admit_next();
            }
            BrokerMsg::Cancel {
                conn,
                client_req,
                reason,
            } => {
                if let Some(index) = self
                    .waiting
                    .iter()
                    .position(|ask| ask.is(&conn, &client_req))
                {
                    let ask = self.waiting.remove(index).expect("index is in range");
                    let _ = ask.conn.try_deliver(Frame::AskResult {
                        req: ask.client_req,
                        outcome: AskOutcome::NoAnswer {
                            reason: NoAnswerReason::AgentCancelled,
                        },
                    });
                }
                if self
                    .live
                    .as_ref()
                    .is_some_and(|(ask, _)| ask.is(&conn, &client_req))
                {
                    self.cancel_live(NoAnswerReason::AgentCancelled, &reason);
                }
            }
            BrokerMsg::Ack {
                conn,
                ask_id,
                visible,
            } => self.on_ack(&conn, &ask_id, visible),
            BrokerMsg::Done { conn, ask_id } => self.on_done(&conn, &ask_id),
            BrokerMsg::ConnGone { conn } => self.on_conn_gone(&conn),
        }
    }

    fn on_ask(
        &mut self,
        conn: Arc<ConnState>,
        client_req: String,
        questions: Vec<String>,
        timeout_ms: u64,
    ) {
        let in_flight = self.waiting.iter().any(|ask| ask.is(&conn, &client_req))
            || self
                .live
                .as_ref()
                .is_some_and(|(ask, _)| ask.is(&conn, &client_req));
        let refusal = validate_ask(&questions, timeout_ms).err().or_else(|| {
            if in_flight {
                Some((
                    "duplicate_req",
                    format!("ask token {client_req:?} is already in flight on this connection"),
                ))
            } else if self.waiting.len() >= MAX_WAITING_ASKS {
                Some((
                    "queue_full",
                    format!("{MAX_WAITING_ASKS} asks are already waiting; retry later"),
                ))
            } else {
                None
            }
        });
        if let Some((code, message)) = refusal {
            let _ = conn.try_deliver(Frame::AskResult {
                req: client_req,
                outcome: AskOutcome::Error {
                    code: code.to_string(),
                    message,
                },
            });
            return;
        }
        self.waiting.push_back(Ask {
            conn,
            client_req,
            // `ask_<uuid>`: unguessable, so no client can aim at the take.
            ask_id: new_id("ask"),
            questions,
            started: Instant::now(),
            timeout: Duration::from_millis(timeout_ms),
            shown_to: Vec::new(),
            acker: None,
        });
    }

    /// Fills the live slot from the queue. Loops because an ask can
    /// fail at admission (no app attached) and must not wedge the rest.
    fn admit_next(&mut self) {
        while self.live.is_none() && self.settling.is_empty() {
            let Some(mut ask) = self.waiting.pop_front() else {
                return;
            };
            ask.started = Instant::now();
            let apps = app_connections(&self.shared);
            if apps.is_empty() {
                ask.resolve(
                    &self.shared,
                    error("no_app", "no Starling app can show the prompt"),
                    "no app",
                );
                continue;
            }
            for conn in &apps {
                let _ = conn.try_deliver(Frame::ShowPrompt {
                    req: ask.ask_id.clone(),
                    questions: ask.questions.clone(),
                    timeout_ms: ask.timeout.as_millis() as u64,
                });
            }
            ask.shown_to = apps;
            self.live = Some((ask, Phase::Prompting));
        }
    }

    fn on_ack(&mut self, conn: &Arc<ConnState>, ask_id: &str, visible: bool) {
        let Some((ask, phase)) = self.live.as_mut() else {
            return;
        };
        if ask.ask_id != ask_id || !ask.shown_to.iter().any(|c| Arc::ptr_eq(c, conn)) {
            return;
        }
        match (&*phase, visible) {
            (Phase::Prompting, true) => {
                ask.acker = Some(Arc::clone(conn));
                let corr = format!("{ask_id}-ctx");
                let command = Command::ContextSnapshot {
                    source: "agent-dictation".to_string(),
                };
                self.advance(&corr, command, Phase::Snapshotting, "capture_failed");
            }
            (Phase::Prompting, false) => {
                self.cancel_live(NoAnswerReason::Declined, "the app declined the prompt")
            }
            (_, false) if ask.acked_by(conn) => self.cancel_live(
                NoAnswerReason::UserCancelled,
                "the user dismissed the prompt",
            ),
            _ => {}
        }
    }

    fn on_done(&mut self, conn: &Arc<ConnState>, ask_id: &str) {
        let Some((ask, phase)) = self.live.as_ref() else {
            return;
        };
        // Before a visible ack there is no acker, so an early done is
        // ignored and the ack bound still applies.
        if ask.ask_id != ask_id || !ask.acked_by(conn) {
            return;
        }
        match phase {
            Phase::Snapshotting | Phase::SettingMode => self.cancel_live(
                NoAnswerReason::UserCancelled,
                "the user finished before the microphone opened",
            ),
            Phase::Recording => {
                let corr = ask.ask_id.clone();
                let command = Command::CaptureStop { drain: Some(true) };
                self.advance(&corr, command, Phase::Persisting, "capture_failed");
            }
            Phase::Prompting | Phase::Persisting | Phase::Transcribing { .. } => {}
        }
    }

    fn on_conn_gone(&mut self, conn: &Arc<ConnState>) {
        self.waiting.retain(|ask| !Arc::ptr_eq(&ask.conn, conn));
        let Some((ask, phase)) = self.live.as_ref() else {
            return;
        };
        let mic_open_or_pending = !matches!(phase, Phase::Persisting | Phase::Transcribing { .. });
        if Arc::ptr_eq(&ask.conn, conn) {
            self.cancel_live(NoAnswerReason::AgentCancelled, "the agent connection ended");
        } else if ask.acked_by(conn) && mic_open_or_pending {
            // No visible prompt is left to stop the take.
            self.cancel_live(
                NoAnswerReason::UserCancelled,
                "the app showing the prompt disconnected",
            );
        }
    }

    /// Sends `command` for the live ask: on acceptance the ask moves to
    /// `next`, on refusal it fails with `code`.
    fn advance(&mut self, corr: &str, command: Command, next: Phase, code: &str) {
        let name = command.type_name();
        match self.shared.client.send(Some(corr), command) {
            Ok(_) => {
                if let Some((_, phase)) = self.live.as_mut() {
                    *phase = next;
                }
            }
            Err(rejection) => self.fail_live(code, format!("{name} was refused: {rejection}")),
        }
    }

    fn cancel_live(&mut self, reason: NoAnswerReason, detail: &str) {
        self.end_live(AskOutcome::NoAnswer { reason }, detail);
    }

    fn fail_live(&mut self, code: &str, message: String) {
        self.end_live(error(code, &message), &message);
    }

    /// Stops whatever the live ask has running, resolves it, and admits
    /// the next one.
    fn end_live(&mut self, outcome: AskOutcome, hide_reason: &str) {
        let Some((ask, phase)) = self.live.take() else {
            return;
        };
        self.abort_phase(&ask, &phase);
        ask.resolve(&self.shared, outcome, hide_reason);
        self.admit_next();
    }

    /// Queues the cleanup the phase leaves behind and tries it once.
    fn abort_phase(&mut self, ask: &Ask, phase: &Phase) {
        match phase {
            Phase::Prompting => {}
            // The provisional context must not reach the next snapshot,
            // or someone else's capture.
            Phase::Snapshotting | Phase::SettingMode => self.settling.push(Cleanup {
                corr: format!("{}-ctx", ask.ask_id),
                command: || Command::ContextExpire,
            }),
            Phase::Recording | Phase::Persisting => self.settling.push(Cleanup {
                corr: ask.ask_id.clone(),
                command: || Command::CaptureAbort,
            }),
            // The microphone is already closed; the job only costs time.
            Phase::Transcribing { job } => {
                let job_id = job.clone();
                let _ = self
                    .shared
                    .client
                    .send(None, Command::JobsCancel { job_id });
            }
        }
        self.settle();
    }

    /// Retries every pending cleanup. One is done when the runtime
    /// accepts it or refuses it as illegal in the machine's state (the
    /// take or context already ended, or the current one belongs to
    /// someone else). A closed actor never comes back: its cleanup is
    /// dropped and queued asks fail with `runtime_unavailable`. Anything
    /// else (a full queue, a sequencing race, a pending command) is
    /// retried on the next tick.
    fn settle(&mut self) {
        let client = &self.shared.client;
        let mut runtime_gone = false;
        self.settling.retain(|cleanup| {
            match client.send(Some(&cleanup.corr), (cleanup.command)()) {
                Ok(_) | Err(Rejection::IllegalInState { .. }) => false,
                // The actor is gone for good: nothing is left to undo,
                // and nothing can be served either.
                Err(Rejection::Closed) => {
                    runtime_gone = true;
                    false
                }
                Err(_) => true,
            }
        });
        if runtime_gone {
            for ask in self.waiting.drain(..) {
                ask.resolve(
                    &self.shared,
                    error("runtime_unavailable", "the runtime is no longer running"),
                    "runtime unavailable",
                );
            }
        }
    }

    /// Who left, if the asking agent or the acking app is gone.
    fn departed(&self, ask: &Ask) -> Option<(NoAnswerReason, &'static str)> {
        let registered = lock_registry(&self.shared.conns);
        let alive = |conn: &Arc<ConnState>| registered.iter().any(|c| Arc::ptr_eq(c, conn));
        if !alive(&ask.conn) {
            Some((NoAnswerReason::AgentCancelled, "the agent connection ended"))
        } else if ask.acker.as_ref().is_some_and(|acker| !alive(acker)) {
            Some((
                NoAnswerReason::UserCancelled,
                "the app showing the prompt disconnected",
            ))
        } else {
            None
        }
    }

    fn tick(&mut self) {
        if !self.settling.is_empty() {
            self.settle();
            self.admit_next();
        }
        let Some((ask, phase)) = self.live.as_ref() else {
            return;
        };
        let now = Instant::now();
        match phase {
            Phase::Prompting => {
                let bound = ACK_BOUND.min(ask.timeout);
                if now >= ask.started + bound {
                    self.fail_live(
                        "no_prompt_ack",
                        format!(
                            "no app connection acknowledged the prompt within {} ms; \
                             the microphone was never started",
                            bound.as_millis()
                        ),
                    );
                }
            }
            // The microphone is closed: the take is captured.
            Phase::Persisting | Phase::Transcribing { .. } => {}
            _ if now >= ask.started + ask.timeout => {
                self.cancel_live(NoAnswerReason::Timeout, "the ask timed out")
            }
            _ => {}
        }
    }

    /// Advances the live ask on runtime events carrying its corr; every
    /// other event is ignored.
    fn on_event(&mut self, message: EventMessage) {
        let Some((ask, phase)) = self.live.as_ref() else {
            return;
        };
        let id = ask.ask_id.clone();
        let corr = message.corr.as_deref();
        let name = message.event.type_name();
        match phase {
            Phase::Snapshotting
                if name == "context.targetSnapshot" && corr == Some(&format!("{id}-ctx")) =>
            {
                let command = Command::ModeSet {
                    mode: "code-guidance".to_string(),
                    source: Manual,
                };
                self.advance(
                    &format!("{id}-mode"),
                    command,
                    Phase::SettingMode,
                    "capture_failed",
                );
            }
            Phase::SettingMode
                if name == "mode.decision" && corr == Some(&format!("{id}-mode")) =>
            {
                if let Some((reason, detail)) = self.departed(ask) {
                    self.cancel_live(reason, detail);
                    return;
                }
                // Opening the microphone is work a retiring host must
                // not take on: checked and flagged under admission, so a
                // retire either sees the recording or is seen here.
                let shared = Arc::clone(&self.shared);
                let _admitted = shared
                    .admission
                    .read()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                if shared.retire.load(Ordering::SeqCst) {
                    self.fail_live(
                        "shutting_down",
                        "the recording service is stepping aside for a newer version".to_string(),
                    );
                    return;
                }
                shared.ask_capturing.store(true, Ordering::SeqCst);
                let command = Command::CaptureStart {
                    policy: "push-to-talk".to_string(),
                };
                self.advance(&id, command, Phase::Recording, "capture_busy");
            }
            Phase::Recording | Phase::Persisting if corr == Some(&id) => {
                let persisting = matches!(phase, Phase::Persisting);
                if let RtEvent::CaptureError { fatal: true, code } = &message.event {
                    self.fail_live(
                        "capture_failed",
                        format!(
                            "the capture failed fatally ({code}); any salvaged audio stays in \
                             storage for manual recovery"
                        ),
                    );
                } else if persisting && name == "capture.stopped" {
                    let job = format!("{id}_job");
                    let command = Command::JobsSubmit {
                        capture_ref: id,
                        route: "local-default".to_string(),
                        budget: "standard".to_string(),
                    };
                    self.advance(
                        &job.clone(),
                        command,
                        Phase::Transcribing { job },
                        "transcription_failed",
                    );
                }
            }
            Phase::Transcribing { job } if corr == Some(job.as_str()) => match message.event {
                RtEvent::JobsCompleted(data) => {
                    if let Some((ask, _)) = self.live.take() {
                        let outcome = AskOutcome::Answered {
                            text: data.text,
                            backend: data.backend,
                        };
                        ask.resolve(&self.shared, outcome, "answered");
                        self.admit_next();
                    }
                }
                RtEvent::JobsFailed { reason, .. } => self.fail_live(
                    "transcription_failed",
                    format!(
                        "transcription failed ({reason}); the take is saved in storage and \
                         can be retried manually"
                    ),
                ),
                _ => {}
            },
            _ => {}
        }
    }

    fn finish_all(&mut self, detail: &str) {
        if let Some((ask, phase)) = self.live.take() {
            self.abort_phase(&ask, &phase);
            self.waiting.push_front(ask);
        }
        for ask in self.waiting.drain(..) {
            ask.resolve(&self.shared, error("shutting_down", detail), detail);
        }
    }
}

fn error(code: &str, message: &str) -> AskOutcome {
    AskOutcome::Error {
        code: code.to_string(),
        message: message.to_string(),
    }
}

/// Every live app-role connection: the only ones that see prompts.
fn app_connections(shared: &HostShared) -> Vec<Arc<ConnState>> {
    lock_registry(&shared.conns)
        .iter()
        .filter(|conn| conn.is_app())
        .cloned()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_allowlist_file_admits_nobody() {
        let list = Allowlist::load(Path::new("/nonexistent/dir/mcp-clients.json")).unwrap();
        assert!(list.authenticate("claude-code", "any-token").is_err());
    }

    #[test]
    fn malformed_allowlists_are_refused() {
        let parse = |body: &str| Allowlist::from_bytes(Path::new("f"), body.as_bytes());
        assert!(matches!(
            parse("{ nope"),
            Err(AllowlistError::Malformed { .. })
        ));
        let err = parse(r#"{"version": 2, "clients": []}"#).unwrap_err();
        assert!(err.to_string().contains("version 2"), "{err}");
        let err = parse(
            r#"{"version":1,"clients":[
                {"name":"claude-code","token":"tok-1"},
                {"name":"claude-code","token":"tok-2"}
            ]}"#,
        )
        .unwrap_err();
        assert!(err.to_string().contains("duplicate"), "{err}");
    }

    #[test]
    fn authentication_requires_name_and_token() {
        let list = Allowlist::from_bytes(
            Path::new("f"),
            br#"{"version":1,"clients":[{"name":"claude-code","token":"tok-1"}]}"#,
        )
        .unwrap();
        assert!(list.authenticate("claude-code", "tok-1").is_ok());
        assert!(list.authenticate("claude-code", "wrong").is_err());
        assert!(list.authenticate("codex", "tok-1").is_err());
        assert!(list.authenticate("", "").is_err());
    }
}
