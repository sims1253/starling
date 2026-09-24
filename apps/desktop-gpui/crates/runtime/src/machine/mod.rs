//! The live, single-owned machine actors (E17 §2: "each machine is a single
//! owned actor/task; the UI is a projection").
//!
//! [`MachineCore`] is the enforcement layer every actor drives: it holds
//! one machine's state, applies the frozen I0 transition tables, and
//! guarantees **the event stream an actor emits is oracle-legal by
//! construction** — a stream collected from the live runtime replays green
//! through [`crate::protocol::replay`]. Actors perform their real side
//! effects (device I/O, provider calls, persistence) between the core's
//! transitions.

pub mod capture;
pub mod context;
pub mod delivery;
pub mod docs;
pub mod jobs;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::protocol::replay::{TransitionKind, TransitionRecord, Violation};
use crate::protocol::tables::MachineSpec;

/// What a committed command left the machine waiting for.
#[derive(Debug, Clone, PartialEq)]
pub enum CommandAction {
    /// The command's `to` state was entered; nothing is pending.
    Entered,
    /// The command now awaits exactly one of `outcomes`, correlated by
    /// `corr` — the actor must resolve it with one of those events.
    Awaiting {
        command: &'static str,
        corr: Option<String>,
        outcomes: Vec<(&'static str, Option<&'static str>)>,
    },
}

/// The pending command, as [`CommandAction::Awaiting`] hands it to the
/// actor's emitter side.
#[derive(Debug, Clone)]
struct Pending {
    command: &'static str,
    corr: Option<String>,
    outcomes: Vec<(&'static str, Option<&'static str>)>,
}

/// One machine's live state + transition + violation history. Owned by
/// exactly one actor thread; never shared mutably.
pub struct MachineCore {
    spec: &'static MachineSpec,
    state: &'static str,
    visited: Vec<&'static str>,
    transitions: Vec<TransitionRecord>,
    violations: Vec<Violation>,
    pending: Option<Pending>,
}

impl MachineCore {
    pub fn new(spec: &'static MachineSpec) -> MachineCore {
        MachineCore {
            spec,
            state: spec.initial,
            visited: vec![spec.initial],
            transitions: Vec::new(),
            violations: Vec::new(),
            pending: None,
        }
    }

    pub fn spec(&self) -> &'static MachineSpec {
        self.spec
    }

    pub fn state(&self) -> &'static str {
        self.state
    }

    /// Whether an outcome-pending command is unresolved (actors must not
    /// accept new commands into this machine while one is).
    pub fn is_awaiting(&self) -> bool {
        self.pending.is_some()
    }

    fn enter(
        &mut self,
        target: Option<&'static str>,
        kind: TransitionKind,
        msg_type: &'static str,
    ) {
        match target.filter(|to| *to != self.state) {
            Some(to) => {
                self.transitions.push(TransitionRecord {
                    kind,
                    type_: Some(msg_type),
                    from: self.state.to_string(),
                    to: to.to_string(),
                    pending: Vec::new(),
                });
                self.state = to;
                self.visited.push(to);
            }
            None => {
                let from = self.state.to_string();
                let to = self.state.to_string();
                self.transitions.push(TransitionRecord {
                    kind,
                    type_: Some(msg_type),
                    from,
                    to,
                    pending: Vec::new(),
                });
            }
        }
    }

    /// Applies a command's transition table entry. `Ok` means the command
    /// was legal and is now committed (the actor proceeds with its side
    /// effects); the `Violation` is what the client receipt will carry.
    pub fn commit_command(
        &mut self,
        command: &'static str,
        corr: Option<String>,
    ) -> Result<CommandAction, Violation> {
        if let Some(pending) = &self.pending {
            let mut awaited: Vec<&str> = pending.outcomes.iter().map(|(name, _)| *name).collect();
            awaited.sort_unstable();
            return Err(Violation::PendingUnresolved {
                detail: format!(
                    "{command} arrived while {} still awaits one of {:?}",
                    pending.command, awaited
                ),
            });
        }
        let Some(rule) = self.spec.command_rule(command) else {
            return Err(Violation::ForeignMessage {
                detail: format!(
                    "{command:?} does not belong to machine {:?}",
                    self.spec.name
                ),
            });
        };
        if !rule.from.allows(self.state) {
            return Err(Violation::IllegalCommand {
                detail: format!(
                    "{command} is not legal in state {:?} (allowed from {})",
                    self.state,
                    rule.from.describe()
                ),
            });
        }
        if rule.outcomes.is_empty() {
            self.enter(rule.to, TransitionKind::Command, command);
            Ok(CommandAction::Entered)
        } else {
            self.pending = Some(Pending {
                command,
                corr,
                outcomes: rule.outcomes.to_vec(),
            });
            let mut awaited: Vec<String> = rule
                .outcomes
                .iter()
                .map(|(name, _)| name.to_string())
                .collect();
            awaited.sort();
            let from = self.state.to_string();
            let to = self.state.to_string();
            self.transitions.push(TransitionRecord {
                kind: TransitionKind::Command,
                type_: Some(command),
                from,
                to,
                pending: awaited,
            });
            Ok(CommandAction::Awaiting {
                command,
                corr: self.pending.as_ref().map(|p| p.corr.clone()).flatten(),
                outcomes: rule.outcomes.to_vec(),
            })
        }
    }

    /// Resolves the pending command with `event` (which must be one of its
    /// outcomes and carry the pending `corr`). Returns the entered state.
    pub fn resolve_outcome(
        &mut self,
        event: &'static str,
        corr: Option<&str>,
    ) -> Result<Option<&'static str>, Violation> {
        let Some(pending) = &self.pending else {
            return Err(Violation::IllegalEvent {
                detail: format!(
                    "{event} arrives with no command awaiting it on machine {:?}",
                    self.spec.name
                ),
            });
        };
        let Some((_, target)) = pending.outcomes.iter().find(|(name, _)| *name == event) else {
            let mut awaited: Vec<&str> = pending.outcomes.iter().map(|(name, _)| *name).collect();
            awaited.sort_unstable();
            return Err(Violation::IllegalEvent {
                detail: format!(
                    "{event} arrived while {} awaits one of {:?}",
                    pending.command, awaited
                ),
            });
        };
        if corr != pending.corr.as_deref() {
            return Err(Violation::CorrMismatch {
                detail: format!(
                    "{event} resolves {} but corr {corr:?} != {:?}",
                    pending.command, pending.corr
                ),
            });
        }
        let target = *target;
        self.pending = None;
        self.enter(target, TransitionKind::Event, event);
        Ok(target)
    }

    /// Emits a non-pending event (progress, gaps, errors, completions the
    /// machine tracks itself). `fatal` feeds the `capture.error` payload
    /// condition; `None` for events without the field. Returns the entered
    /// state.
    pub fn emit_event(
        &mut self,
        event: &'static str,
        fatal: Option<bool>,
    ) -> Result<Option<&'static str>, Violation> {
        let Some(rule) = self.spec.event_rule(event) else {
            return Err(Violation::ForeignMessage {
                detail: format!("{event:?} does not belong to machine {:?}", self.spec.name),
            });
        };
        if !rule.from.contains(&self.state) {
            return Err(Violation::IllegalEvent {
                detail: format!(
                    "{event} is not legal in state {:?} (legal from {:?})",
                    self.state, rule.from
                ),
            });
        }
        let mut target = rule.to;
        if let Some(fatal_to) = rule.to_when_fatal_to {
            if fatal == Some(true) {
                target = Some(fatal_to);
            }
        }
        self.enter(target, TransitionKind::Event, event);
        Ok(target)
    }

    /// Takes a runtime-internal edge (no wire message).
    pub fn advance_internal(&mut self, target: &str) -> Result<(), Violation> {
        if !self.spec.internal_targets(self.state).contains(&target) {
            return Err(Violation::InternalEdgeViolation {
                detail: format!(
                    "no runtime-internal edge {:?} -> {target:?} in machine {:?}",
                    self.state, self.spec.name
                ),
            });
        }
        self.transitions.push(TransitionRecord {
            kind: TransitionKind::Internal,
            type_: None,
            from: self.state.to_string(),
            to: target.to_string(),
            pending: Vec::new(),
        });
        self.state = self
            .spec
            .states
            .iter()
            .copied()
            .position(|state| state == target)
            .map(|index| self.spec.states[index])
            .expect("internal edges stay inside declared states");
        self.visited.push(self.state);
        Ok(())
    }

    /// Records a violation that could not surface on the wire (v1 defines
    /// no event for it); it stays visible in the machine snapshot so
    /// nothing is silently absorbed.
    pub fn record_violation(&mut self, violation: Violation) {
        self.violations.push(violation);
    }

    /// A projection copy for [`crate::RuntimeSnapshot`].
    pub fn view(&self) -> MachineView {
        MachineView {
            machine: self.spec.name.to_string(),
            state: self.state.to_string(),
            pending: self.pending.as_ref().map(|pending| PendingView {
                command: pending.command.to_string(),
                corr: pending.corr.clone(),
                outcomes: pending
                    .outcomes
                    .iter()
                    .map(|(name, _)| name.to_string())
                    .collect(),
            }),
            transitions: self.transitions.len(),
            violations: self.violations.iter().map(|v| v.to_string()).collect(),
        }
    }
}

/// The projection snapshot of one machine (updated by its actor after each
/// transition; read by the UI through [`crate::RuntimeClient::snapshot`]).
#[derive(Debug, Clone, serde::Serialize)]
pub struct MachineView {
    pub machine: String,
    pub state: String,
    pub pending: Option<PendingView>,
    pub transitions: usize,
    pub violations: Vec<String>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct PendingView {
    pub command: String,
    pub corr: Option<String>,
    pub outcomes: Vec<String>,
}

/// A machine actor's shared, mutex-guarded view slot — written only by the
/// owning actor, read by the projection.
pub type ViewSlot = std::sync::Arc<std::sync::Mutex<MachineView>>;

pub fn view_slot(spec: &'static MachineSpec) -> ViewSlot {
    std::sync::Arc::new(std::sync::Mutex::new(MachineCore::new(spec).view()))
}

/// The receipt a client receives for a command: `Ok` once the owning
/// machine accepted it (any outcome event follows asynchronously on the
/// event stream), `Err` with the machine- or envelope-level rejection.
///
/// `Serialize`/`Deserialize` (external tagging — `"Accepted"` /
/// `{"Served": …}` / `{"IllegalInState": {…}}`) are the I4 service host's
/// receipt wire form: the host bridges an IPC command to
/// [`crate::RuntimeClient::send_raw`] and returns the resulting
/// `Result<Receipt, Rejection>` verbatim over the transport, so the IPC
/// client observes exactly the in-process rejection, not a stringly
/// re-telling of it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Receipt {
    /// The command was accepted; state transitions are visible via events
    /// and the snapshot.
    Accepted,
    /// The command was accepted and answered synchronously with a view
    /// (`docs.get`, served via snapshot — v1 defines no event for it).
    Served(Value),
}

/// Why a command was rejected at the runtime boundary. v1 defines
/// `runtime.nack` only for unsupported versions and `jobs.rejected` only
/// for the three admission reasons, so every other rejection surfaces here
/// — synchronously, never silently.
///
/// Like [`Receipt`], the serde derives are the I4 IPC receipt's wire form
/// (external tagging; every variant round-trips exactly, so a client
/// across the transport can pattern-match the same rejection an embedded
/// client would).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Rejection {
    /// The envelope version is unsupported; the runtime additionally
    /// emitted `runtime.nack{unsupported_version}` on the event stream with
    /// `corr` = the rejected message's id.
    UnsupportedVersion { id: Option<String> },
    /// Structural envelope failure (missing/unknown fields, bad tokens).
    InvalidEnvelope(String),
    /// The command type is not a v1 command.
    UnknownMessageType(String),
    /// The payload does not match the v1 grammar for this command.
    InvalidPayload(String),
    /// The command is not legal in the machine's current state.
    IllegalInState {
        command: String,
        state: String,
        detail: String,
    },
    /// An outcome-pending command is unresolved in this machine.
    PendingUnresolved { detail: String },
    /// `seq` did not strictly increase on its stream.
    SeqNotMonotonic { detail: String },
    /// `jobs.submit` referenced a route no earlier `mode.routeFrozen` froze
    /// (the audio-leave proxy; held as a synchronous rejection because v1's
    /// `jobs.rejected` reason vocabulary does not include it).
    RouteNotFrozen { route: String },
    /// The submit's `captureRef` names no take this runtime captured.
    UnknownCaptureRef { capture_ref: String },
    /// The named job is not active (already terminal or unknown).
    UnknownJob { job_id: String },
    /// The named revision does not exist (delivery.prepare).
    UnknownRevision { revision_id: String },
    /// The named delivery does not exist or is not in the required state.
    UnknownDelivery { delivery_id: String },
    /// The machine's bounded inbox is full; retry.
    InboxFull,
    /// The runtime is shutting down or the machine actor is gone.
    Closed,
}

impl std::fmt::Display for Rejection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Rejection::UnsupportedVersion { id } => {
                write!(
                    f,
                    "unsupported envelope version (id {id:?}); runtime.nack emitted"
                )
            }
            Rejection::InvalidEnvelope(detail) => write!(f, "invalid envelope: {detail}"),
            Rejection::UnknownMessageType(t) => write!(f, "{t:?} is not a v1 command type"),
            Rejection::InvalidPayload(detail) => write!(f, "invalid payload: {detail}"),
            Rejection::IllegalInState {
                command,
                state,
                detail,
            } => {
                write!(f, "{command} illegal in {state}: {detail}")
            }
            Rejection::PendingUnresolved { detail } => write!(f, "{detail}"),
            Rejection::SeqNotMonotonic { detail } => write!(f, "{detail}"),
            Rejection::RouteNotFrozen { route } => {
                write!(
                    f,
                    "route {route:?} was not frozen by an earlier mode.routeFrozen"
                )
            }
            Rejection::UnknownCaptureRef { capture_ref } => {
                write!(f, "unknown captureRef {capture_ref:?}")
            }
            Rejection::UnknownJob { job_id } => write!(f, "unknown or terminal job {job_id:?}"),
            Rejection::UnknownRevision { revision_id } => {
                write!(f, "unknown revision {revision_id:?}")
            }
            Rejection::UnknownDelivery { delivery_id } => {
                write!(f, "unknown delivery {delivery_id:?}")
            }
            Rejection::InboxFull => write!(f, "machine inbox full; retry"),
            Rejection::Closed => write!(f, "runtime closed"),
        }
    }
}

impl std::error::Error for Rejection {}

/// The bounded reply channel for one command's receipt.
pub type ReceiptTx = crate::channel::Sender<Result<Receipt, Rejection>>;
pub type ReceiptRx = crate::channel::Receiver<Result<Receipt, Rejection>>;

/// A command envelope en route to a machine actor.
pub struct Inbound {
    pub id: String,
    pub ts: String,
    pub corr: Option<String>,
    pub seq: Option<u64>,
    pub command: crate::protocol::Command,
    pub reply: ReceiptTx,
}

impl Inbound {
    /// Replies and consumes the reply channel.
    pub fn reply(self, result: Result<Receipt, Rejection>) {
        let _ = self.reply.try_send(result);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::tables::CAPTURE;

    #[test]
    fn receipt_and_rejection_round_trip_the_ipc_wire_form() {
        // The I4 host serializes these across its transport; every variant
        // must survive the external-tagged JSON round trip exactly.
        let pairs = vec![
            (Receipt::Accepted, serde_json::json!("Accepted")),
            (
                Receipt::Served(serde_json::json!({ "docId": "notes" })),
                serde_json::json!({ "Served": { "docId": "notes" } }),
            ),
        ];
        for (value, wire) in pairs {
            assert_eq!(serde_json::to_value(&value).unwrap(), wire);
            let back: Receipt = serde_json::from_value(wire).unwrap();
            assert_eq!(back, value);
        }
        // Pin the wire form explicitly for the externally-observed
        // rejections too: the derives ARE the IPC protocol (the host
        // serializes Frame::Receipt verbatim), so an accidental
        // representation change — a newtype here, a field rename there —
        // must fail here, not at the first IPC client. A bare round trip
        // would only prove serde's self-consistency. Every variant is
        // pinned (all fourteen), including the Option-absence shape of
        // UnsupportedVersion and every unit/newtype variant.
        let rejection_pairs = vec![
            (
                Rejection::UnsupportedVersion {
                    id: Some("cmd_9".into()),
                },
                serde_json::json!({ "UnsupportedVersion": { "id": "cmd_9" } }),
            ),
            (
                Rejection::UnsupportedVersion { id: None },
                serde_json::json!({ "UnsupportedVersion": { "id": null } }),
            ),
            (
                Rejection::InvalidEnvelope("missing 'ts'".into()),
                serde_json::json!({ "InvalidEnvelope": "missing 'ts'" }),
            ),
            (
                Rejection::UnknownMessageType("bogus.type".into()),
                serde_json::json!({ "UnknownMessageType": "bogus.type" }),
            ),
            (
                Rejection::InvalidPayload("payload is invalid".into()),
                serde_json::json!({ "InvalidPayload": "payload is invalid" }),
            ),
            (
                Rejection::IllegalInState {
                    command: "capture.stop".into(),
                    state: "Idle".into(),
                    detail: "not legal".into(),
                },
                serde_json::json!({
                    "IllegalInState": {
                        "command": "capture.stop",
                        "state": "Idle",
                        "detail": "not legal"
                    }
                }),
            ),
            (
                Rejection::PendingUnresolved {
                    detail: "waiting".into(),
                },
                serde_json::json!({ "PendingUnresolved": { "detail": "waiting" } }),
            ),
            (
                Rejection::SeqNotMonotonic {
                    detail: "seq 4 follows 4".into(),
                },
                serde_json::json!({ "SeqNotMonotonic": { "detail": "seq 4 follows 4" } }),
            ),
            (
                Rejection::RouteNotFrozen {
                    route: "local-default".into(),
                },
                serde_json::json!({ "RouteNotFrozen": { "route": "local-default" } }),
            ),
            (
                Rejection::UnknownCaptureRef {
                    capture_ref: "take_x".into(),
                },
                serde_json::json!({ "UnknownCaptureRef": { "capture_ref": "take_x" } }),
            ),
            (
                Rejection::UnknownJob {
                    job_id: "job-1".into(),
                },
                serde_json::json!({ "UnknownJob": { "job_id": "job-1" } }),
            ),
            (
                Rejection::UnknownRevision {
                    revision_id: "rev-1".into(),
                },
                serde_json::json!({ "UnknownRevision": { "revision_id": "rev-1" } }),
            ),
            (
                Rejection::UnknownDelivery {
                    delivery_id: "d-1".into(),
                },
                serde_json::json!({ "UnknownDelivery": { "delivery_id": "d-1" } }),
            ),
            (Rejection::InboxFull, serde_json::json!("InboxFull")),
            (Rejection::Closed, serde_json::json!("Closed")),
        ];
        for (value, wire) in rejection_pairs {
            assert_eq!(
                serde_json::to_value(&value).unwrap(),
                wire,
                "the rejection wire form changed; every IPC client's \
                 pattern match rides on it"
            );
            let back: Rejection = serde_json::from_value(wire).unwrap();
            assert_eq!(back, value);
        }
    }

    #[test]
    fn core_reproduces_the_capture_happy_path() {
        let mut core = MachineCore::new(&CAPTURE);
        assert_eq!(core.state(), "Idle");

        assert!(matches!(
            core.commit_command("capture.start", Some("take_77".into())),
            Ok(CommandAction::Entered)
        ));
        assert_eq!(core.state(), "Acquiring");
        assert_eq!(
            core.emit_event("capture.started", None).unwrap(),
            Some("Recording")
        );
        assert!(core
            .commit_command("capture.stop", Some("take_77".into()))
            .is_ok());
        assert_eq!(core.state(), "Draining");
        // A fatal error from Draining enters Interrupted.
        assert_eq!(
            core.emit_event("capture.error", Some(true)).unwrap(),
            Some("Interrupted")
        );
        // Internal recovery edge, then stopped from Recovering.
        core.advance_internal("Recovering").unwrap();
        assert_eq!(
            core.emit_event("capture.stopped", None).unwrap(),
            Some("Persisted")
        );
    }

    #[test]
    fn core_rejects_illegal_commands_like_the_oracle() {
        let mut core = MachineCore::new(&CAPTURE);
        let violation = core
            .commit_command("capture.stop", None)
            .expect_err("stop from Idle is illegal");
        assert_eq!(violation.code(), "illegal_command");
        let view = core.view();
        assert_eq!(view.state, "Idle");
    }

    /// Issue #211: a fatal device-open failure (`Acquiring → Interrupted`,
    /// fixture take_9's `device_open_failed`) must not wedge the machine —
    /// the runtime-internal settle edge returns it to Idle so the next
    /// `capture.start` is legal again.
    #[test]
    fn failed_device_open_settles_back_to_idle() {
        let mut core = MachineCore::new(&CAPTURE);
        assert!(matches!(
            core.commit_command("capture.start", Some("take_9".into())),
            Ok(CommandAction::Entered)
        ));
        assert_eq!(
            core.emit_event("capture.error", Some(true)).unwrap(),
            Some("Interrupted")
        );
        core.advance_internal("Idle")
            .expect("Interrupted -> Idle edge");
        assert_eq!(core.state(), "Idle");
        // The retry is legal — the machine is not wedged in Interrupted.
        assert!(matches!(
            core.commit_command("capture.start", Some("take_10".into())),
            Ok(CommandAction::Entered)
        ));
    }

    use crate::protocol::tables::JOBS;

    #[test]
    fn core_enforces_outcome_correlation() {
        let mut core = MachineCore::new(&JOBS);
        assert!(matches!(
            core.commit_command("jobs.submit", Some("job-1".into())),
            Ok(CommandAction::Awaiting { .. })
        ));
        let mismatch = core
            .resolve_outcome("jobs.queued", Some("wrong"))
            .expect_err("corr mismatch");
        assert_eq!(mismatch.code(), "corr_mismatch");
        assert_eq!(
            core.resolve_outcome("jobs.queued", Some("job-1")).unwrap(),
            Some("Queued")
        );
    }
}
