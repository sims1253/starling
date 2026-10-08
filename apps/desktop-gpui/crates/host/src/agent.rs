//! The agent-dictation ask surface (issue #309): what a coding agent's
//! MCP server sees on the other side of the IPC transport.
//!
//! Two halves live here:
//!
//! - [`Allowlist`] — the per-client gate every agent connection passes
//!   (`Frame::AgentHello`, checked in `server::connection_reader`).
//!   Default **deny**: a host without an allowlist file (or with an
//!   empty one) admits no agent client at all.
//! - [`broker_loop`] — the one actor that owns every `Frame::AskUser`.
//!   It serializes asks (queued while one is live — never two visible
//!   prompts, never overlapping captures), enforces the
//!   **prompt-visibility gate** (capture starts only after an app
//!   connection acks `Frame::PromptAck { visible: true }`), then drives
//!   the *existing* capture path — the same `context.snapshot` →
//!   `mode.set` → `capture.start` → `capture.stop` → `jobs.submit`
//!   sequence any renderer client runs (see the engine suite) — and
//!   resolves the ask with the transcript. There is deliberately no
//!   second recording stack: the capture machine owns the microphone,
//!   the store owns persistence, the jobs machine owns transcription.
//!
//! # Lifecycle rules (deliberate, each tested)
//!
//! - **Timeout**: each ask carries a `timeout_ms` budget that starts at
//!   admission (when the ask leaves the queue — a queued ask shows no
//!   prompt, so the user-facing clock starts when the prompt could).
//!   Expiry aborts a live take and resolves `NoAnswer { timeout }`.
//! - **Agent cancel** (`Frame::AskCancel`, or the MCP layer's
//!   `notifications/cancelled`): aborts a live take
//!   (`NoAnswer { agent_cancelled }`); a queued ask is dropped without
//!   ever prompting.
//! - **Disconnect**: an ask is bound to the connection that sent it.
//!   That connection dying — EOF, crash, kill — cancels its asks the
//!   same way an explicit cancel would (`capture.abort`, hide the
//!   prompt). The microphone never outlives the agent that asked
//!   because the *host* enforces this, not the (possibly dead) MCP
//!   server process.
//! - **User cancel / decline**: the app that acked the prompt can
//!   dismiss it (`PromptAck { visible: false }`) — before capture that
//!   declines the ask, during capture it stops the take
//!   (`NoAnswer { user_cancelled }`).
//! - **Prompt ack bound**: an ask whose prompt no app acks within
//!   [`ACK_BOUND`] (clamped by the ask's remaining budget) fails with
//!   `Error { no_prompt_ack }` — no capture ever starts. A host with no
//!   app connection attached fails the same way, immediately.
//!
//! # Trust boundary (documented honestly)
//!
//! MCP over stdio has no strong client identity: whoever can spawn a
//! process can speak the protocol. The allowlist therefore gates on a
//! shared secret the **user** provisions: a `mcp-clients.json` beside
//! the host's data (written by hand, `0600`), naming each agent client
//! and its token; the user pastes the same token into the agent's MCP
//! config when registering the command. The token rides the
//! same-user-authenticated IPC transport (UDS `SO_PEERCRED` / a DACL'd
//! named pipe), so it is never exposed cross-user, and any local
//! process running as the same user can already do anything the user
//! can — the allowlist is a *user intent* boundary (which agents may
//! summon the microphone), not a defense against malware on the same
//! account. It is still checked per connection, per hello, because the
//! alternative (an unmarked ask surface open to any connected renderer)
//! would make "which agent started the mic" unanswerable.

use std::collections::VecDeque;
use std::path::Path;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

use starling_runtime::bus::{EventMessage, EventSub};
use starling_runtime::channel::{Receiver, RecvError};
use starling_runtime::protocol::{Command, Event as RtEvent, Manual};
use starling_runtime::RuntimeClient;

use crate::frame::{AskOutcome, Frame, NoAnswerReason};
use crate::server::{BrokerMsg, ConnState, HostShared};

/// How many questions one ask may carry. More than a handful is a
/// prompt nobody reads aloud; the bound also keeps `ShowPrompt` frames
/// tiny.
pub const MAX_QUESTIONS: usize = 8;

/// Per-question character bound (the prompt is read, not studied).
pub const MAX_QUESTION_CHARS: usize = 2_000;

/// Ask timeout floor. Below this the ask cannot cover prompt display,
/// speaking, and transcription; it would be a misconfiguration.
pub const MIN_TIMEOUT_MS: u64 = 1_000;

/// Ask timeout ceiling: no ask may park the queue for unbounded time.
pub const MAX_TIMEOUT_MS: u64 = 600_000;

/// How many asks may wait behind the live one. Past this the host
/// answers `Error { queue_full }` immediately (bounded queues, explicit
/// rejection — the machines' own posture).
pub const MAX_WAITING_ASKS: usize = 4;

/// How long an admitted ask waits for a prompt-visibility ack before
/// failing with `Error { no_prompt_ack }`. Clamped by the ask's
/// remaining timeout budget.
pub const ACK_BOUND: Duration = Duration::from_secs(10);

/// The allowlist's file name inside the host's data root.
pub const ALLOWLIST_FILE: &str = "mcp-clients.json";

/// The broker's poll slice: how often it re-checks deadlines while
/// parked, and how promptly it drains its event subscription (which it
/// must — the runtime's bus backpressures slow subscribers).
const POLL: Duration = Duration::from_millis(20);

// ------------------------------------------------------------------ //
// The allowlist
// ------------------------------------------------------------------ //

/// The named agent clients allowed to use the ask surface, with their
/// tokens. Default deny: an empty list (or no file at all) admits
/// nobody.
///
/// File shape (`mcp-clients.json` at the host's data root):
///
/// ```json
/// { "version": 1, "clients": [
///     { "name": "claude-code", "token": "…hex…" }
/// ] }
/// ```
///
/// Unknown `version` values are refused (the file is user-writable
/// configuration; silently guessing at a future shape could turn a
/// refuse-list into an allowlist). Loaded once at host startup — see
/// the module docs for the trust boundary this draws.
#[derive(Debug, Clone, Default)]
pub struct Allowlist {
    clients: Vec<(String, String)>,
}

/// Why an allowlist could not be loaded. A *missing* file is not here —
/// it is the unconfigured host, an empty list (deny all); only a file
/// that exists but cannot be read or parsed refuses to serve.
#[derive(Debug, thiserror::Error)]
pub enum AllowlistError {
    #[error("the allowlist {path:?} cannot be read: {source}")]
    Read {
        path: std::path::PathBuf,
        source: std::io::Error,
    },
    #[error("the allowlist {path:?} is not valid: {detail}")]
    Malformed {
        path: std::path::PathBuf,
        detail: String,
    },
}

impl Allowlist {
    /// The list that admits nobody (the default-deny default).
    pub fn empty() -> Allowlist {
        Allowlist::default()
    }

    /// Loads the allowlist (missing file → empty, see the type docs).
    pub fn load(path: &Path) -> Result<Allowlist, AllowlistError> {
        let bytes = match std::fs::read(path) {
            Ok(bytes) => bytes,
            Err(source) if source.kind() == std::io::ErrorKind::NotFound => {
                return Ok(Allowlist::empty())
            }
            Err(source) => {
                return Err(AllowlistError::Read {
                    path: path.to_path_buf(),
                    source,
                })
            }
        };
        Allowlist::from_bytes(path, &bytes)
    }

    /// Parses the file body (see [`Allowlist::load`] for the missing
    /// file posture; this is the unreadable/malformed arm).
    pub fn from_bytes(path: &Path, bytes: &[u8]) -> Result<Allowlist, AllowlistError> {
        let malformed = |detail: String| AllowlistError::Malformed {
            path: path.to_path_buf(),
            detail,
        };
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
        let file: File = serde_json::from_slice(bytes)
            .map_err(|err| malformed(format!("not valid allowlist JSON: {err}")))?;
        if file.version != 1 {
            return Err(malformed(format!(
                "unsupported allowlist version {} (this host reads version 1)",
                file.version
            )));
        }
        let mut clients = Vec::new();
        for entry in file.clients {
            if entry.name.is_empty() || entry.token.is_empty() {
                return Err(malformed(
                    "every client needs a non-empty name and token".to_string(),
                ));
            }
            clients.push((entry.name, entry.token));
        }
        Ok(Allowlist { clients })
    }

    /// Whether the named client with this token may use the ask
    /// surface. Default deny: unknown name, wrong token, empty list —
    /// all refuse.
    pub fn authenticate(&self, client: &str, token: &str) -> Result<(), String> {
        // Plain equality: the tokens ride a same-user-authenticated
        // transport (see the module docs) — a timing side channel would
        // only help an attacker who is already the user.
        self.clients
            .iter()
            .find(|(name, _)| name == client)
            .map(|(_, expected)| {
                if expected == token {
                    Ok(())
                } else {
                    Err(format!(
                        "agent client {client:?} is allowlisted but the token is wrong"
                    ))
                }
            })
            .unwrap_or_else(|| Err(format!("agent client {client:?} is not allowlisted")))
    }
}

// ------------------------------------------------------------------ //
// The broker
// ------------------------------------------------------------------ //

/// One ask, in the broker's terms: the owning connection (answers and
/// liveness ride it), the agent's own correlation token (echoed on
/// `AskResult`), and the broker-scoped ask id (what app-facing frames
/// and runtime `corr` streams use — agent tokens from different
/// connections would otherwise collide, MCP request ids being `1, 2,
/// 3…` on every client).
struct Ask {
    conn: Arc<ConnState>,
    client_req: String,
    ask_id: String,
    questions: Vec<String>,
    /// Budget start (admission) and total; every phase draws on it.
    started: Instant,
    timeout: Duration,
}

impl Ask {
    fn deadline(&self) -> Instant {
        self.started + self.timeout
    }

    /// Resolves the ask on its owning connection and hides the prompt
    /// on every app connection. The one exit path for a finished ask.
    fn resolve(self, shared: &HostShared, outcome: AskOutcome, hide_reason: &str) {
        // Delivery failure here means the agent connection is gone (its
        // cancel/disconnect is usually what brought us here); nothing
        // to deliver.
        let _ = self.conn.try_deliver(Frame::AskResult {
            req: self.client_req,
            outcome,
        });
        fan_out_hide(shared, &self.ask_id, hide_reason);
    }
}

/// Where the live ask is in the capture flow. The context dance
/// (`context.snapshot` → `mode.set`) runs after the visibility ack and
/// before the mic opens — never before: the gate is the first thing,
/// the route freeze the last.
enum Phase {
    /// Prompt shown, waiting for the first `PromptAck` (until the ack
    /// bound or the ask budget).
    Prompting,
    /// Acked; running the route-freeze dance. `step` names the event
    /// awaited next.
    Preparing { step: PrepStep },
    /// `capture.start` accepted; the mic is open until the app reports
    /// the user done (or a cancel path stops it).
    Recording,
    /// `capture.stop` accepted; waiting `capture.stopped` (the journal
    /// is draining/persisting) before `jobs.submit`.
    Persisting,
    /// `jobs.submit` accepted; waiting the job's outcome.
    Transcribing { job: String },
}

enum PrepStep {
    /// Waiting `context.targetSnapshot` (corr `<ask>-ctx`).
    TargetSnapshot,
    /// Waiting `mode.decision` (corr `<ask>-mode`).
    ModeDecision,
}

/// The broker thread: drains its inbox and the runtime event stream on
/// a poll slice, advancing the live ask. Spawned by `server::serve`;
/// exits when the host's shutdown flag rises (resolving anything live
/// with `Error { shutting_down }` first).
pub(crate) fn broker_loop(shared: Arc<HostShared>, inbox: Receiver<BrokerMsg>) {
    let client: RuntimeClient = shared.client.clone();
    let events: EventSub = client.subscribe();
    let mut broker = Broker {
        shared,
        inbox,
        events,
        next_ask: 0,
        live: None,
        waiting: VecDeque::new(),
    };
    loop {
        if broker.shared.shutdown.load(Ordering::SeqCst) {
            broker.finish_all("the host is shutting down");
            return;
        }
        // Inbox first (cancels and prompts outrank progress events),
        // then every event that piled up — the bus backpressures a
        // subscriber that stops draining, and this host must never be
        // that subscriber.
        loop {
            match broker.inbox.try_recv() {
                Ok(msg) => broker.on_msg(msg),
                Err(RecvError::Timeout) => break,
                Err(RecvError::Closed) => {
                    // Every sender lives in the host's connection
                    // threads or HostShared itself; closed means the
                    // host is going away. The flag sweep above is the
                    // authoritative exit; treat this as it.
                    broker.finish_all("the host is shutting down");
                    return;
                }
            }
        }
        loop {
            match broker.events.try_recv() {
                Ok(message) => broker.on_event(message),
                // Empty (and, if the runtime is gone, closed — nothing
                // more can arrive either way).
                Err(RecvError::Timeout) | Err(RecvError::Closed) => break,
            }
        }
        broker.tick_deadlines();
        // Park until the next message or the poll wake (the shutdown
        // flag and every deadline stay within one POLL of current).
        match broker.inbox.recv_timeout(POLL) {
            Ok(msg) => broker.on_msg(msg),
            Err(RecvError::Timeout) => {}
            Err(RecvError::Closed) => {
                broker.finish_all("the host is shutting down");
                return;
            }
        }
    }
}

struct Broker {
    shared: Arc<HostShared>,
    inbox: Receiver<BrokerMsg>,
    events: EventSub,
    next_ask: u64,
    live: Option<(Ask, Phase)>,
    waiting: VecDeque<Ask>,
}

impl Broker {
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
            } => self.on_cancel(&conn, &client_req, &reason),
            BrokerMsg::Ack { ask_id, visible } => self.on_ack(ask_id, visible),
            BrokerMsg::Done { ask_id } => self.on_done(ask_id),
            BrokerMsg::ConnGone { conn } => self.on_conn_gone(&conn),
        }
    }

    /// Validates and (if the queue has room) enqueues a fresh ask.
    fn on_ask(
        &mut self,
        conn: Arc<ConnState>,
        client_req: String,
        questions: Vec<String>,
        timeout_ms: u64,
    ) {
        let reject = |code: &str, message: String| {
            let _ = conn.try_deliver(Frame::AskResult {
                req: client_req.clone(),
                outcome: AskOutcome::Error {
                    code: code.to_string(),
                    message,
                },
            });
        };
        if questions.is_empty() || questions.len() > MAX_QUESTIONS {
            reject(
                "invalid_questions",
                format!("questions must carry 1..={MAX_QUESTIONS} entries"),
            );
            return;
        }
        if questions
            .iter()
            .any(|q| q.trim().is_empty() || q.chars().count() > MAX_QUESTION_CHARS)
        {
            reject(
                "invalid_questions",
                format!(
                    "every question needs 1..={MAX_QUESTION_CHARS} non-whitespace characters"
                ),
            );
            return;
        }
        if !(MIN_TIMEOUT_MS..=MAX_TIMEOUT_MS).contains(&timeout_ms) {
            reject(
                "invalid_timeout",
                format!("timeout_ms must be {MIN_TIMEOUT_MS}..={MAX_TIMEOUT_MS}"),
            );
            return;
        }
        // A duplicate token on the same connection would alias the
        // agent's own correlation; refuse rather than answer one of the
        // two callers with the other's result.
        let duplicate = self.waiting.iter().any(|ask| {
            ask.client_req == client_req && Arc::ptr_eq(&ask.conn, &conn)
        }) || self
            .live
            .as_ref()
            .is_some_and(|(ask, _)| ask.client_req == client_req && Arc::ptr_eq(&ask.conn, &conn));
        if duplicate {
            reject(
                "invalid_questions",
                format!("ask token {client_req:?} is already in flight on this connection"),
            );
            return;
        }
        if self.waiting.len() >= MAX_WAITING_ASKS {
            reject(
                "queue_full",
                format!(
                    "{} asks are already waiting; retry when the current ask resolves",
                    self.waiting.len()
                ),
            );
            return;
        }
        self.next_ask += 1;
        let ask_id = format!("ask_{}", self.next_ask);
        self.waiting.push_back(Ask {
            conn,
            client_req,
            ask_id,
            questions,
            started: Instant::now(),
            timeout: Duration::from_millis(timeout_ms),
        });
    }

    /// Moves the head of the queue into the live slot while it is free
    /// (a loop: an ask that fails at admission — no app connection —
    /// must not wedge the queue behind it).
    fn admit_next(&mut self) {
        while self.live.is_none() {
            let Some(mut ask) = self.waiting.pop_front() else {
                return;
            };
            // Admission resets the budget clock: while queued, no
            // prompt could be shown, so the user-facing budget starts
            // here (see the module docs).
            ask.started = Instant::now();
            // The gate's first arm: with no app connection attached
            // there is nobody who could ever ack visibility — fail now
            // instead of burning the ack bound on a prompt nobody
            // displays.
            if !self.any_app_connection() {
                ask.resolve(
                    &self.shared,
                    AskOutcome::Error {
                        code: "no_prompt_ack".to_string(),
                        message: "no app connection is attached to show the prompt".to_string(),
                    },
                    "no app connection",
                );
                continue;
            }
            fan_out_show(
                &self.shared,
                &ask.ask_id,
                ask.questions.clone(),
                ask.timeout.as_millis() as u64,
            );
            self.live = Some((ask, Phase::Prompting));
        }
    }

    fn any_app_connection(&self) -> bool {
        self.shared
            .conns
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .iter()
            .any(|conn| !conn.is_agent())
    }

    fn on_ack(&mut self, ask_id: String, visible: bool) {
        enum AckEffect {
            None,
            Decline,
            Dismiss,
            OpenGate(String),
        }
        let effect = match self.live.as_ref() {
            Some((ask, phase)) if ask.ask_id == ask_id => match (phase, visible) {
                // The gate opens: capture may start (after the
                // route-freeze dance; see `Preparing`).
                (Phase::Prompting, true) => AckEffect::OpenGate(ask.ask_id.clone()),
                (Phase::Prompting, false) => AckEffect::Decline,
                // A dismissal after the gate opened: the mic (or the
                // job) must stop — the no-mic-without-a-visible-prompt
                // rule is held for the whole take, not just its start.
                (_, false) => AckEffect::Dismiss,
                // A second visible:true restates what the gate already
                // recorded; nothing to do.
                _ => AckEffect::None,
            },
            // A late ack for a resolved ask, or an id from an earlier
            // one: already answered.
            _ => AckEffect::None,
        };
        match effect {
            AckEffect::None => {}
            AckEffect::Decline => {
                self.cancel_live(NoAnswerReason::Declined, "the app declined the prompt")
            }
            AckEffect::Dismiss => {
                self.cancel_live(NoAnswerReason::UserCancelled, "the user dismissed the prompt")
            }
            AckEffect::OpenGate(ask_id) => {
                let shared = Arc::clone(&self.shared);
                let corr = format!("{ask_id}-ctx");
                match shared.client.send(
                    Some(&corr),
                    Command::ContextSnapshot {
                        source: "agent-dictation".to_string(),
                    },
                ) {
                    Ok(_) => self.set_phase(Phase::Preparing {
                        step: PrepStep::TargetSnapshot,
                    }),
                    Err(rejection) => self.fail_live(
                        "capture_failed",
                        format!("context.snapshot was refused: {rejection}"),
                    ),
                }
            }
        }
    }

    fn on_done(&mut self, ask_id: String) {
        let applies = match self.live.as_ref() {
            // Done before the gate opened: the app is misbehaving (its
            // own protocol says Done follows a visible ack). Ignore —
            // the ack bound still applies.
            Some((ask, Phase::Prompting | Phase::Preparing { .. })) => ask.ask_id == ask_id,
            // A second Done (or one racing the stop handshake): the
            // stop already covers it.
            Some((_, Phase::Persisting | Phase::Transcribing { .. })) => false,
            Some((ask, Phase::Recording)) => ask.ask_id == ask_id,
            None => false,
        };
        if !applies {
            return;
        }
        let ask_id = self
            .live
            .as_ref()
            .map(|(ask, _)| ask.ask_id.clone())
            .unwrap_or_default();
        let shared = Arc::clone(&self.shared);
        if let Err(rejection) =
            shared.client.send(Some(&ask_id), Command::CaptureStop { drain: Some(true) })
        {
            self.fail_live(
                "capture_failed",
                format!("capture.stop was refused: {rejection}"),
            );
            return;
        }
        self.set_phase(Phase::Persisting);
    }

    /// An explicit agent cancel: the owner's queued ask is dropped
    /// silently (nothing was shown, nothing captured); the owner's live
    /// ask is cancelled the deliberate way.
    fn on_cancel(&mut self, conn: &Arc<ConnState>, client_req: &str, reason: &str) {
        let live_owned = self.live.as_ref().is_some_and(|(ask, _)| {
            ask.client_req == client_req && Arc::ptr_eq(&ask.conn, conn)
        });
        self.waiting.retain(|ask| {
            !(ask.client_req == client_req && Arc::ptr_eq(&ask.conn, conn))
        });
        if live_owned {
            self.cancel_live(NoAnswerReason::AgentCancelled, reason);
        }
    }

    /// The connection died: drop its queued asks (there is nobody to
    /// answer) and cancel its live one — the disconnect rule (see the
    /// module docs).
    fn on_conn_gone(&mut self, conn: &Arc<ConnState>) {
        self.waiting.retain(|ask| !Arc::ptr_eq(&ask.conn, conn));
        if self
            .live
            .as_ref()
            .is_some_and(|(ask, _)| Arc::ptr_eq(&ask.conn, conn))
        {
            self.cancel_live(
                NoAnswerReason::AgentCancelled,
                "the agent connection ended",
            );
        }
    }

    /// Cancels the live ask: abort the take if the mic is open (or
    /// draining), cancel the job if transcription is underway, resolve,
    /// admit the next.
    fn cancel_live(&mut self, reason: NoAnswerReason, detail: &str) {
        let Some((ask, phase)) = self.live.take() else {
            return;
        };
        self.abort_phase(&ask, &phase);
        ask.resolve(&self.shared, AskOutcome::NoAnswer { reason }, detail);
        self.admit_next();
    }

    /// Fails the live ask with an `Error` outcome (refusals and
    /// failures; never a transcript claim).
    fn fail_live(&mut self, code: &str, message: String) {
        let Some((ask, phase)) = self.live.take() else {
            return;
        };
        self.abort_phase(&ask, &phase);
        ask.resolve(
            &self.shared,
            AskOutcome::Error {
                code: code.to_string(),
                message: message.clone(),
            },
            &message,
        );
        self.admit_next();
    }

    /// Stops whatever the phase has running: `capture.abort` from
    /// Recording/Persisting (the microphone stops; the take is
    /// discarded — nothing acknowledged is lost), `jobs.cancel` from
    /// Transcribing.
    fn abort_phase(&self, ask: &Ask, phase: &Phase) {
        match phase {
            // No mic opened yet.
            Phase::Prompting | Phase::Preparing { .. } => {}
            Phase::Recording | Phase::Persisting => {
                let _ = self
                    .shared
                    .client
                    .send(Some(&ask.ask_id), Command::CaptureAbort);
            }
            Phase::Transcribing { job } => {
                let _ = self
                    .shared
                    .client
                    .send(None, Command::JobsCancel { job_id: job.clone() });
            }
        }
    }

    /// Answers the live ask with the transcript and admits the next.
    fn answer_live(&mut self, text: String, backend: String) {
        let Some((ask, _)) = self.live.take() else {
            return;
        };
        ask.resolve(
            &self.shared,
            AskOutcome::Answered { text, backend },
            "answered",
        );
        self.admit_next();
    }

    fn set_phase(&mut self, phase: Phase) {
        if let Some((_, current)) = self.live.as_mut() {
            *current = phase;
        }
    }

    /// Deadline sweep: the ack bound and the overall budget.
    fn tick_deadlines(&mut self) {
        let (budget_expired, ack_expired, ack_bound) = match self.live.as_ref() {
            None => (false, false, Duration::ZERO),
            Some((ask, phase)) => {
                let now = Instant::now();
                if now >= ask.deadline() {
                    (true, false, Duration::ZERO)
                } else if matches!(phase, Phase::Prompting) {
                    // The visibility gate's own bound:
                    // min(ACK_BOUND, the remaining budget).
                    let bound = ACK_BOUND.min(ask.deadline() - now);
                    let expired = ask.started + bound <= now;
                    (false, expired, bound)
                } else {
                    (false, false, Duration::ZERO)
                }
            }
        };
        if budget_expired {
            self.cancel_live(NoAnswerReason::Timeout, "the ask timed out");
        } else if ack_expired {
            self.fail_live(
                "no_prompt_ack",
                format!(
                    "no app connection acknowledged the prompt within {} ms; \
                     the microphone was never started",
                    ack_bound.as_millis()
                ),
            );
        }
    }

    /// Every runtime event, matched against the live ask's corr
    /// streams. Events for other corr (other clients' takes, jobs) pass
    /// through untouched — the broker is a client like any other.
    fn on_event(&mut self, message: EventMessage) {
        // What the event asks the broker to do, computed under one
        // borrow and applied after it ends.
        enum Step {
            None,
            SendModeSet,
            SendCaptureStart,
            CaptureFatal(String),
            SubmitJob,
            JobCompleted { text: String, backend: String },
            JobFailed(String),
        }
        let corr = message.corr.as_deref();
        let name = message.event.type_name();
        let step = match self.live.as_ref() {
            None => Step::None,
            Some((_, Phase::Prompting)) => Step::None,
            Some((ask, Phase::Preparing { step })) => {
                let ctx_corr = format!("{}-ctx", ask.ask_id);
                let mode_corr = format!("{}-mode", ask.ask_id);
                match (step, corr, name) {
                    (
                        PrepStep::TargetSnapshot,
                        Some(c),
                        "context.targetSnapshot",
                    ) if c == ctx_corr => Step::SendModeSet,
                    (PrepStep::ModeDecision, Some(c), "mode.decision") if c == mode_corr => {
                        Step::SendCaptureStart
                    }
                    _ => Step::None,
                }
            }
            Some((ask, Phase::Recording)) => {
                if corr == Some(ask.ask_id.as_str()) {
                    match &message.event {
                        RtEvent::CaptureError { fatal: true, code } => {
                            Step::CaptureFatal(code.clone())
                        }
                        _ => Step::None,
                    }
                } else {
                    Step::None
                }
            }
            Some((ask, Phase::Persisting)) => {
                if corr == Some(ask.ask_id.as_str()) && name == "capture.stopped" {
                    Step::SubmitJob
                } else {
                    Step::None
                }
            }
            Some((_, Phase::Transcribing { job })) => {
                if corr != Some(job.as_str()) {
                    return;
                }
                match &message.event {
                    RtEvent::JobsCompleted(data) => Step::JobCompleted {
                        text: data.text.clone(),
                        backend: data.backend.clone(),
                    },
                    RtEvent::JobsFailed { reason, .. } => Step::JobFailed(reason.clone()),
                    _ => Step::None,
                }
            }
        };
        let ask_id = self
            .live
            .as_ref()
            .map(|(ask, _)| ask.ask_id.clone())
            .unwrap_or_default();
        let shared = Arc::clone(&self.shared);
        match step {
            Step::None => {}
            Step::SendModeSet => {
                let corr = format!("{ask_id}-mode");
                match shared.client.send(
                    Some(&corr),
                    Command::ModeSet {
                        mode: "code-guidance".to_string(),
                        source: Manual,
                    },
                ) {
                    Ok(_) => self.set_phase(Phase::Preparing {
                        step: PrepStep::ModeDecision,
                    }),
                    Err(rejection) => self.fail_live(
                        "capture_failed",
                        format!("mode.set was refused: {rejection}"),
                    ),
                }
            }
            Step::SendCaptureStart => {
                // The route froze at the capture actor's start; the mic
                // opens here — the earliest point the gate allows.
                match shared.client.send(
                    Some(&ask_id),
                    Command::CaptureStart {
                        policy: "push-to-talk".to_string(),
                    },
                ) {
                    Ok(_) => self.set_phase(Phase::Recording),
                    Err(rejection) => self.fail_live(
                        "capture_busy",
                        format!(
                            "capture.start was refused (another take or the device is busy): \
                             {rejection}"
                        ),
                    ),
                }
            }
            Step::CaptureFatal(code) => self.fail_live(
                "capture_failed",
                format!(
                    "the capture failed fatally ({code}); any salvaged audio stays in \
                     storage for manual recovery"
                ),
            ),
            Step::SubmitJob => {
                let job = format!("{ask_id}_job");
                match shared.client.send(
                    Some(&job),
                    Command::JobsSubmit {
                        capture_ref: ask_id.clone(),
                        route: "local-default".to_string(),
                        budget: "standard".to_string(),
                    },
                ) {
                    Ok(_) => self.set_phase(Phase::Transcribing { job }),
                    Err(rejection) => self.fail_live(
                        "transcription_failed",
                        format!("jobs.submit was refused: {rejection}"),
                    ),
                }
            }
            Step::JobCompleted { text, backend } => self.answer_live(text, backend),
            Step::JobFailed(reason) => self.fail_live(
                "transcription_failed",
                format!(
                    "transcription failed ({reason}); the take is saved in storage and \
                     can be retried manually"
                ),
            ),
        }
    }

    /// Host shutdown: everything live or queued resolves with
    /// `Error { shutting_down }`; prompts hide.
    fn finish_all(&mut self, detail: &str) {
        while let Some((ask, phase)) = self.live.take() {
            self.abort_phase(&ask, &phase);
            ask.resolve(
                &self.shared,
                AskOutcome::Error {
                    code: "shutting_down".to_string(),
                    message: detail.to_string(),
                },
                detail,
            );
        }
        for ask in self.waiting.drain(..) {
            ask.resolve(
                &self.shared,
                AskOutcome::Error {
                    code: "shutting_down".to_string(),
                    message: detail.to_string(),
                },
                detail,
            );
        }
    }
}

/// Sends a frame to every live non-agent connection (the app side).
/// Delivery is best-effort per connection — a slow app is closed by its
/// own queue policy (see `ConnState::try_deliver`), never allowed to
/// hold the broker.
fn fan_out(shared: &HostShared, frame: Frame) {
    let conns = shared
        .conns
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone();
    for conn in conns {
        if !conn.is_agent() {
            let _ = conn.try_deliver(frame.clone());
        }
    }
}

fn fan_out_show(shared: &HostShared, ask_id: &str, questions: Vec<String>, timeout_ms: u64) {
    fan_out(
        shared,
        Frame::ShowPrompt {
            req: ask_id.to_string(),
            questions,
            timeout_ms,
        },
    );
}

fn fan_out_hide(shared: &HostShared, ask_id: &str, reason: &str) {
    fan_out(
        shared,
        Frame::HidePrompt {
            req: ask_id.to_string(),
            reason: reason.to_string(),
        },
    );
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
    fn malformed_allowlist_is_an_error_not_a_silent_deny() {
        let err = Allowlist::from_bytes(Path::new("f"), b"{ nope").unwrap_err();
        assert!(matches!(err, AllowlistError::Malformed { .. }));
        let err =
            Allowlist::from_bytes(Path::new("f"), br#"{"version": 2, "clients": []}"#.as_slice())
                .unwrap_err();
        assert!(err.to_string().contains("version 2"));
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
