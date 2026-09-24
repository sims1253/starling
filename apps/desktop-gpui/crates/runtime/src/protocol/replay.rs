//! Trace replay — the executable oracle port (`tests/runtime_protocol.py`),
//! crate-visible so the conformance suite and the live actors share one set
//! of transition semantics.
//!
//! [`MachineReplay`] applies a fixture trace message by message, asserting
//! every event is legal for the current machine state, every command is
//! legal in the state it arrives in, and `seq` is strictly monotonic per
//! stream — exactly the oracle's `MachineReplay`, including the
//! pending-outcome / correlation discipline and the `$advance` directives
//! that step over runtime-internal edges.

use serde_json::Value;
use std::collections::HashMap;

use super::tables::{spec_for, MachineSpec};
use super::{envelope_errors, Envelope, Kind, SUPPORTED_VERSIONS};

/// A fixture trace broke the contract. The variants are the oracle's
/// violation classes verbatim — invalid fixtures are rejected with the
/// reason class named in their `replay` field.
#[derive(Debug, Clone, PartialEq)]
pub enum Violation {
    /// `seq` did not strictly increase on its stream.
    SeqNotMonotonic { detail: String },
    /// The command is not legal in the current state.
    IllegalCommand { detail: String },
    /// The event is not legal in the current state (or does not resolve the
    /// pending command).
    IllegalEvent { detail: String },
    /// Envelope version is unsupported; the receiver answers
    /// `runtime.nack{unsupported_version}` and never applies the payload.
    NackUnsupportedVersion { detail: String },
    /// Structural envelope failure.
    InvalidEnvelope(String),
    /// The `type` is not a v1 message type.
    UnknownMessageType(String),
    /// The message belongs to another machine.
    ForeignMessage { detail: String },
    /// A new command or directive arrived while an outcome-pending command
    /// was unresolved, or the trace ended with one pending.
    PendingUnresolved { detail: String },
    /// An outcome event resolved the pending command but carried the wrong
    /// `corr`.
    CorrMismatch { detail: String },
    /// An unsupported trace directive.
    UnknownDirective(String),
    /// `$advance` named an edge that is not runtime-internal.
    InternalEdgeViolation { detail: String },
    /// The initial state is not a state of this machine.
    UnknownState { detail: String },
}

impl Violation {
    /// The oracle's `code` string (invalid fixtures name this in `replay`).
    pub fn code(&self) -> &'static str {
        match self {
            Violation::SeqNotMonotonic { .. } => "seq_not_monotonic",
            Violation::IllegalCommand { .. } => "illegal_command",
            Violation::IllegalEvent { .. } => "illegal_event",
            Violation::NackUnsupportedVersion { .. } => "nack_unsupported_version",
            Violation::InvalidEnvelope(_) => "invalid_envelope",
            Violation::UnknownMessageType(_) => "unknown_message_type",
            Violation::ForeignMessage { .. } => "foreign_message",
            Violation::PendingUnresolved { .. } => "pending_unresolved",
            Violation::CorrMismatch { .. } => "corr_mismatch",
            Violation::UnknownDirective(_) => "unknown_directive",
            Violation::InternalEdgeViolation { .. } => "internal_edge_violation",
            Violation::UnknownState { .. } => "unknown_state",
        }
    }
}

impl std::fmt::Display for Violation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Violation::SeqNotMonotonic { detail }
            | Violation::IllegalCommand { detail }
            | Violation::IllegalEvent { detail }
            | Violation::NackUnsupportedVersion { detail }
            | Violation::ForeignMessage { detail }
            | Violation::PendingUnresolved { detail }
            | Violation::CorrMismatch { detail }
            | Violation::InternalEdgeViolation { detail }
            | Violation::UnknownState { detail } => write!(f, "{}: {}", self.code(), detail),
            Violation::InvalidEnvelope(detail) => write!(f, "invalid_envelope: {detail}"),
            Violation::UnknownMessageType(detail) => write!(f, "unknown_message_type: {detail}"),
            Violation::UnknownDirective(detail) => write!(f, "unknown_directive: {detail}"),
        }
    }
}

impl std::error::Error for Violation {}

/// Whether a trace record is a `$`-directive rather than an envelope.
pub fn is_directive(record: &Value) -> bool {
    record
        .as_object()
        .is_some_and(|object| object.keys().any(|key| key.starts_with('$')))
}

/// Which v1 set a wire type belongs to: a command, an event
/// (`runtime.nack` included), or neither.
pub fn message_kind(type_: &str) -> Option<Kind> {
    if is_command_type(type_) {
        Some(Kind::Command)
    } else if is_event_type(type_) {
        Some(Kind::Event)
    } else {
        None
    }
}

fn is_command_type(type_: &str) -> bool {
    crate::protocol::tables::all_specs()
        .iter()
        .any(|spec| spec.command_rule(type_).is_some())
}

fn is_event_type(type_: &str) -> bool {
    crate::protocol::tables::all_specs()
        .iter()
        .any(|spec| spec.event_rule(type_).is_some())
        || type_ == "runtime.nack"
}

/// The stream a message belongs to (the oracle's `_stream_key`): the `corr`
/// channel, or the direction stream (`__commands__` / `__events__`).
fn stream_key(envelope: &Envelope, kind: Kind) -> String {
    match &envelope.corr {
        Some(corr) => corr.clone(),
        None => format!(
            "__{}s__",
            match kind {
                Kind::Command => "command",
                Kind::Event => "event",
            }
        ),
    }
}

/// One applied transition, as the oracle records them for the conformance
/// assertions.
#[derive(Debug, Clone, PartialEq)]
pub struct TransitionRecord {
    pub kind: TransitionKind,
    pub type_: Option<&'static str>,
    pub from: String,
    pub to: String,
    /// Sorted names of the events an outcome-pending command awaits.
    pub pending: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransitionKind {
    Command,
    Event,
    Internal,
}

/// The pending outcome discipline: an outcome-pending command and the events
/// that may resolve it.
#[derive(Debug, Clone)]
struct Pending {
    command_type: &'static str,
    corr: Option<String>,
    #[allow(dead_code)] // mirrors the oracle's pending record shape
    id: String,
    outcomes: Vec<(&'static str, Option<&'static str>)>,
}

/// Replays one fixture trace against one machine's transition table — the
/// oracle's `MachineReplay`, ported line for line.
pub struct MachineReplay {
    spec: &'static MachineSpec,
    pub state: String,
    pub visited: Vec<String>,
    pub transitions: Vec<TransitionRecord>,
    last_seq: HashMap<String, u64>,
    pending: Option<Pending>,
}

impl MachineReplay {
    pub fn new(machine: &str, initial: Option<&str>) -> Result<MachineReplay, Violation> {
        let spec = spec_for(machine).ok_or_else(|| Violation::UnknownState {
            detail: format!("unknown machine {machine:?}"),
        })?;
        let state = initial.unwrap_or(spec.initial);
        if !spec.states.contains(&state) {
            return Err(Violation::UnknownState {
                detail: format!("initial state {state:?} not in machine {machine}"),
            });
        }
        Ok(MachineReplay {
            spec,
            state: state.to_string(),
            visited: vec![state.to_string()],
            transitions: Vec::new(),
            last_seq: HashMap::new(),
            pending: None,
        })
    }

    fn enter(&mut self, state: Option<&'static str>, kind: TransitionKind, msg_type: &'static str) {
        let target = state.filter(|to| *to != self.state);
        match target {
            Some(to) => {
                self.transitions.push(TransitionRecord {
                    kind,
                    type_: Some(msg_type),
                    from: self.state.clone(),
                    to: to.to_string(),
                    pending: Vec::new(),
                });
                self.state = to.to_string();
                self.visited.push(to.to_string());
            }
            None => {
                let to = self.state.clone();
                self.transitions.push(TransitionRecord {
                    kind,
                    type_: Some(msg_type),
                    from: self.state.clone(),
                    to,
                    pending: Vec::new(),
                });
            }
        }
    }

    fn check_seq(&mut self, envelope: &Envelope, kind: Kind) -> Result<(), Violation> {
        let Some(seq) = envelope.seq else {
            return Ok(());
        };
        let stream = stream_key(envelope, kind);
        if let Some(last) = self.last_seq.get(&stream) {
            if seq <= *last {
                return Err(Violation::SeqNotMonotonic {
                    detail: format!(
                        "{} seq {seq} on stream {stream:?} follows seq {last}",
                        envelope.type_
                    ),
                });
            }
        }
        self.last_seq.insert(stream, seq);
        Ok(())
    }

    fn check_envelope(&self, envelope: &Envelope) -> Result<Kind, Violation> {
        if let Some(v) = envelope_v(envelope) {
            if !SUPPORTED_VERSIONS.contains(&v) {
                return Err(Violation::NackUnsupportedVersion {
                    detail: format!(
                        "envelope version {v:?} is not supported (supported: {SUPPORTED_VERSIONS:?}); \
                         receiver answers runtime.nack{{unsupported_version}}"
                    ),
                });
            }
        }
        let value = envelope.to_value();
        let errors = envelope_errors(&value);
        if !errors.is_empty() {
            return Err(Violation::InvalidEnvelope(errors.join("; ")));
        }
        let msg_type = envelope.type_.as_str();
        let kind = if is_command_type(msg_type) {
            Kind::Command
        } else if is_event_type(msg_type) {
            Kind::Event
        } else {
            return Err(Violation::UnknownMessageType(format!(
                "{msg_type:?} is not a v1 message type"
            )));
        };
        let own = self
            .spec
            .commands
            .iter()
            .map(|(name, _)| *name)
            .chain(self.spec.events.iter().map(|(name, _)| *name))
            .collect::<Vec<_>>();
        if !own.contains(&msg_type) {
            return Err(Violation::ForeignMessage {
                detail: format!(
                    "{msg_type:?} does not belong to machine {:?}",
                    self.spec.name
                ),
            });
        }
        Ok(kind)
    }

    fn apply_command(&mut self, envelope: &Envelope) -> Result<(), Violation> {
        if let Some(pending) = &self.pending {
            let mut awaited: Vec<&str> = pending.outcomes.iter().map(|(name, _)| *name).collect();
            awaited.sort_unstable();
            return Err(Violation::PendingUnresolved {
                detail: format!(
                    "{} arrived while {} still awaits one of {:?}",
                    envelope.type_, pending.command_type, awaited
                ),
            });
        }
        let rule = self
            .spec
            .command_rule(&envelope.type_)
            .expect("check_envelope proved the rule exists");
        if !rule.from.allows(&self.state) {
            return Err(Violation::IllegalCommand {
                detail: format!(
                    "{} is not legal in state {:?} (allowed from {})",
                    envelope.type_,
                    self.state,
                    rule.from.describe()
                ),
            });
        }
        if !rule.outcomes.is_empty() {
            let mut awaited: Vec<String> = rule
                .outcomes
                .iter()
                .map(|(name, _)| name.to_string())
                .collect();
            awaited.sort();
            self.pending = Some(Pending {
                command_type: static_type(&envelope.type_),
                corr: envelope.corr.clone(),
                id: envelope.id.clone(),
                outcomes: rule.outcomes.to_vec(),
            });
            let from = self.state.clone();
            let to = self.state.clone();
            self.transitions.push(TransitionRecord {
                kind: TransitionKind::Command,
                type_: Some(static_type(&envelope.type_)),
                from,
                to,
                pending: awaited,
            });
        } else {
            self.enter(
                rule.to,
                TransitionKind::Command,
                static_type(&envelope.type_),
            );
        }
        Ok(())
    }

    fn apply_event(&mut self, envelope: &Envelope) -> Result<(), Violation> {
        let msg_type = static_type(&envelope.type_);
        if let Some(pending) = &self.pending {
            if let Some((_, target)) = pending.outcomes.iter().find(|(name, _)| *name == msg_type) {
                if envelope.corr != pending.corr {
                    return Err(Violation::CorrMismatch {
                        detail: format!(
                            "{} resolves {} but corr {:?} != {:?}",
                            msg_type, pending.command_type, envelope.corr, pending.corr
                        ),
                    });
                }
                let target = *target;
                self.pending = None;
                self.enter(target, TransitionKind::Event, msg_type);
                return Ok(());
            }
            let mut awaited: Vec<&str> = pending.outcomes.iter().map(|(name, _)| *name).collect();
            awaited.sort_unstable();
            return Err(Violation::IllegalEvent {
                detail: format!(
                    "{} arrived while {} awaits one of {:?}",
                    msg_type, pending.command_type, awaited
                ),
            });
        }
        let rule =
            self.spec
                .event_rule(&envelope.type_)
                .ok_or_else(|| Violation::IllegalEvent {
                    detail: format!("{msg_type} is only reachable as a pending outcome"),
                })?;
        if !rule.from.contains(&self.state.as_str()) {
            return Err(Violation::IllegalEvent {
                detail: format!(
                    "{} is not legal in state {:?} (legal from {:?})",
                    msg_type, self.state, rule.from
                ),
            });
        }
        let mut target = rule.to;
        if let Some(fatal_to) = rule.to_when_fatal_to {
            let fatal = envelope
                .payload
                .get("fatal")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            if fatal {
                target = Some(fatal_to);
            }
        }
        self.enter(target, TransitionKind::Event, msg_type);
        Ok(())
    }

    fn apply_directive(&mut self, record: &Value) -> Result<(), Violation> {
        let Some(object) = record.as_object() else {
            return Err(Violation::UnknownDirective(record.to_string()));
        };
        if object.len() != 1 || !object.contains_key("$advance") {
            return Err(Violation::UnknownDirective(record.to_string()));
        }
        let target = object.get("$advance").and_then(Value::as_str).unwrap_or("");
        if let Some(pending) = &self.pending {
            let mut awaited: Vec<&str> = pending.outcomes.iter().map(|(name, _)| *name).collect();
            awaited.sort_unstable();
            return Err(Violation::PendingUnresolved {
                detail: format!(
                    "$advance arrived while {} still awaits one of {:?}",
                    pending.command_type, awaited
                ),
            });
        }
        if !self.spec.internal_targets(&self.state).contains(&target) {
            return Err(Violation::InternalEdgeViolation {
                detail: format!(
                    "no runtime-internal edge {:?} -> {target:?} in machine {:?}",
                    self.state, self.spec.name
                ),
            });
        }
        let from = self.state.clone();
        self.transitions.push(TransitionRecord {
            kind: TransitionKind::Internal,
            type_: None,
            from,
            to: target.to_string(),
            pending: Vec::new(),
        });
        self.state = target.to_string();
        self.visited.push(target.to_string());
        Ok(())
    }

    /// Applies one trace record (an envelope or a `$`-directive).
    pub fn feed(&mut self, record: &Value) -> Result<(), Violation> {
        if is_directive(record) {
            return self.apply_directive(record);
        }
        let Some(envelope) = Envelope::from_value(record) else {
            // The envelope did not even parse (missing/ill-typed required
            // fields). The oracle still checks the version first when it
            // can, so mirror that before falling back to structure.
            return Err(structural_violation(record));
        };
        let kind = self.check_envelope(&envelope)?;
        self.check_seq(&envelope, kind)?;
        match kind {
            Kind::Command => self.apply_command(&envelope),
            Kind::Event => self.apply_event(&envelope),
        }
    }

    /// Ends the trace; a still-pending command is unresolved.
    pub fn finish(self) -> Result<MachineReplay, Violation> {
        if let Some(pending) = &self.pending {
            let mut awaited: Vec<&str> = pending.outcomes.iter().map(|(name, _)| *name).collect();
            awaited.sort_unstable();
            return Err(Violation::PendingUnresolved {
                detail: format!(
                    "trace ends while {} still awaits one of {:?}",
                    pending.command_type, awaited
                ),
            });
        }
        Ok(self)
    }
}

fn envelope_v(envelope: &Envelope) -> Option<u64> {
    // Envelope::from_value only parses integer v; a non-integer v never
    // reaches this far (envelope_errors flags it first) — but the oracle
    // checks the version before structure, so mirror that: a v that failed
    // to parse as u64 (e.g. a bool or string) is structurally invalid, not
    // unsupported.
    Some(envelope.v)
}

/// The violation for a record whose envelope did not parse: version check
/// first (a parseable-but-unsupported integer `v` is a NACK), structure
/// otherwise — the oracle's `_check_envelope` ordering.
fn structural_violation(record: &Value) -> Violation {
    if let Some(v) = record.get("v") {
        if let Some(version) = v.as_u64() {
            if !SUPPORTED_VERSIONS.contains(&version) {
                return Violation::NackUnsupportedVersion {
                    detail: format!(
                        "envelope version {version:?} is not supported (supported: \
                         {SUPPORTED_VERSIONS:?}); receiver answers \
                         runtime.nack{{unsupported_version}}"
                    ),
                };
            }
        }
    }
    let errors = envelope_errors(record);
    let detail = if errors.is_empty() {
        "message is not an object".to_string()
    } else {
        errors.join("; ")
    };
    Violation::InvalidEnvelope(detail)
}

fn static_type(type_: &str) -> &'static str {
    // All message types in the tables are 'static; for validated unknowns
    // this is never called with a non-table type. Leak-free: look up in the
    // tables instead of leaking.
    for spec in crate::protocol::tables::all_specs() {
        for (name, _) in spec.commands {
            if *name == type_ {
                return name;
            }
        }
        for (name, _) in spec.events {
            if *name == type_ {
                return name;
            }
        }
    }
    "runtime.nack"
}

/// Replays a fixture trace; errors on the first break — the oracle's
/// `replay_trace`.
pub fn replay_trace(trace: &Value) -> Result<MachineReplay, Violation> {
    let machine = trace.get("machine").and_then(Value::as_str).unwrap_or("");
    let initial = trace.get("initial").and_then(Value::as_str);
    let mut replay = MachineReplay::new(machine, initial)?;
    for record in trace
        .get("messages")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[])
    {
        replay.feed(record)?;
    }
    replay.finish()
}

/// Standalone per-stream monotonicity check — the oracle's `seq_violations`.
pub fn seq_violations(messages: &[Value]) -> Vec<String> {
    let mut last: HashMap<String, u64> = HashMap::new();
    let mut found = Vec::new();
    for msg in messages {
        if is_directive(msg) {
            continue;
        }
        let (Some(seq), Some(type_)) = (
            msg.get("seq").and_then(Value::as_u64),
            msg.get("type").and_then(Value::as_str),
        ) else {
            continue;
        };
        let kind = if is_command_type(type_) {
            Kind::Command
        } else {
            Kind::Event
        };
        let stream = msg
            .get("corr")
            .and_then(Value::as_str)
            .map(str::to_string)
            .unwrap_or_else(|| {
                format!(
                    "__{}s__",
                    match kind {
                        Kind::Command => "command",
                        Kind::Event => "event",
                    }
                )
            });
        if let Some(previous) = last.get(&stream) {
            if seq <= *previous {
                found.push(format!(
                    "{type_} seq {seq} on stream {stream:?} follows seq {previous}"
                ));
            }
        }
        last.insert(stream, seq);
    }
    found
}

/// One route-freeze violation (the oracle's record shape).
#[derive(Debug, Clone, PartialEq)]
pub struct RouteFreezeViolation {
    pub id: Option<String>,
    pub route: String,
    pub detail: String,
}

/// Every `jobs.submit` must reference a route frozen by an earlier
/// `mode.routeFrozen` — the oracle's `route_freeze_violations` (messages in
/// global `ts` order; `jobs.submit` is the audio-leave proxy).
pub fn route_freeze_violations(messages: &[Value]) -> Vec<RouteFreezeViolation> {
    let mut frozen: HashMap<String, String> = HashMap::new();
    let mut violations = Vec::new();
    for msg in messages {
        if is_directive(msg) {
            continue;
        }
        let type_ = msg.get("type").and_then(Value::as_str).unwrap_or("");
        let ts = msg.get("ts").and_then(Value::as_str).unwrap_or("");
        if type_ == "mode.routeFrozen" {
            if let Some(route) = msg
                .pointer("/payload/route")
                .and_then(Value::as_str)
                .map(str::to_string)
            {
                let better = !frozen.contains_key(&route) || ts < frozen[&route].as_str();
                if better {
                    frozen.insert(route, ts.to_string());
                }
            }
        } else if type_ == "jobs.submit" {
            if let Some(route) = msg
                .pointer("/payload/route")
                .and_then(Value::as_str)
                .map(str::to_string)
            {
                let ok = frozen
                    .get(&route)
                    .is_some_and(|frozen_ts| frozen_ts.as_str() < ts);
                if !ok {
                    violations.push(RouteFreezeViolation {
                        id: msg.get("id").and_then(Value::as_str).map(str::to_string),
                        route: route.clone(),
                        detail: format!(
                            "jobs.submit {} references route {route:?} before any \
                             mode.routeFrozen froze it",
                            msg.get("id").and_then(Value::as_str).unwrap_or("?")
                        ),
                    });
                }
            }
        }
    }
    violations
}

/// Loads a fixture trace from a JSON string.
pub fn load_fixture(text: &str) -> Value {
    serde_json::from_str(text).expect("fixture is valid JSON")
}

/// Splits fixture files into (valid, invalid) by path — the oracle's
/// `load_fixtures` convention: a trace under a directory named `invalid/`
/// is invalid.
pub fn split_fixtures(
    paths: &[std::path::PathBuf],
) -> (
    Vec<(std::path::PathBuf, Value)>,
    Vec<(std::path::PathBuf, Value)>,
) {
    let mut valid = Vec::new();
    let mut invalid = Vec::new();
    for path in paths {
        let text = std::fs::read_to_string(path)
            .unwrap_or_else(|err| panic!("fixture {} unreadable: {err}", path.display()));
        let trace = load_fixture(&text);
        let is_invalid = path
            .parent()
            .map(|parent| parent.file_name().is_some_and(|name| name == "invalid"))
            .unwrap_or(false);
        (if is_invalid { &mut invalid } else { &mut valid }).push((path.clone(), trace));
    }
    (valid, invalid)
}

/// All envelopes of all valid fixtures in global timestamp order — the
/// oracle's `valid_corpus_messages` (sort key `(ts, id)`).
pub fn valid_corpus_messages(valid: &[(std::path::PathBuf, Value)]) -> Vec<Value> {
    let mut messages: Vec<(String, String, Value)> = Vec::new();
    for (_, trace) in valid {
        if let Some(records) = trace.get("messages").and_then(Value::as_array) {
            for record in records {
                if is_directive(record) {
                    continue;
                }
                let ts = record
                    .get("ts")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                let id = record
                    .get("id")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                messages.push((ts, id, record.clone()));
            }
        }
    }
    messages.sort_by(|a, b| (&a.0, &a.1).cmp(&(&b.0, &b.1)));
    messages.into_iter().map(|(_, _, msg)| msg).collect()
}
