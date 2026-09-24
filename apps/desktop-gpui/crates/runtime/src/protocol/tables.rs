//! The five state machines as data (port of the I0 oracle's `MACHINES`,
//! `tests/runtime_protocol.py`) — E17 design note §2 frozen by increment I0.
//!
//! Command spec keys (oracle semantics, kept verbatim):
//! - `from` — states the command is legal in (`Any` = the oracle's `"*"`);
//! - `to` — state entered immediately (no outcome event expected);
//! - `outcomes` — the command awaits exactly one of these events, correlated
//!   by `corr` (`None` target = state kept).
//!
//! Event spec keys:
//! - `from` — states the event is legal in (empty = only reachable as the
//!   pending outcome of its command);
//! - `to` — state entered (`None` = unchanged);
//! - `to_when_fatal_to` — the oracle's `to_when` payload-conditioned target
//!   (`capture.error` with `fatal: true` → `Interrupted`).
//!
//! `internal` edges carry no wire message (scheduler dispatch, expiry
//! timers, journal recovery, interrupted-take settlement); fixtures step
//! over them with `{"$advance": "<state>"}` directives, and the live
//! actors take them themselves at the equivalent points.

use std::collections::HashMap;
use std::sync::OnceLock;

/// Which states a command is legal from — the oracle's list or `"*"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FromStates {
    Any,
    Only(&'static [&'static str]),
}

impl FromStates {
    pub fn allows(&self, state: &str) -> bool {
        match self {
            FromStates::Any => true,
            FromStates::Only(states) => states.contains(&state),
        }
    }

    /// The oracle's `rule["from"]` rendering, used in violation details.
    pub fn describe(&self) -> String {
        match self {
            FromStates::Any => "*".to_string(),
            FromStates::Only(states) => format!("[{}]", states.join(", ")),
        }
    }
}

/// One command's transition rule.
#[derive(Debug, Clone, Copy)]
pub struct CommandRule {
    pub from: FromStates,
    /// State entered immediately, when the command defines one.
    pub to: Option<&'static str>,
    /// `{event type: target state}` the command awaits (exactly one
    /// arrives, on the same `corr`).
    pub outcomes: &'static [(&'static str, Option<&'static str>)],
}

/// One event's transition rule.
#[derive(Debug, Clone, Copy)]
pub struct EventRule {
    /// States the event is legal from (empty = pending-outcome-only).
    pub from: &'static [&'static str],
    /// State entered (`None` = unchanged).
    pub to: Option<&'static str>,
    /// The oracle's `to_when`: this event's payload carries a `fatal` flag,
    /// and `fatal == true` redirects to this state instead of `to`.
    pub to_when_fatal_to: Option<&'static str>,
}

/// One machine's complete specification.
#[derive(Debug, Clone, Copy)]
pub struct MachineSpec {
    pub name: &'static str,
    pub initial: &'static str,
    pub states: &'static [&'static str],
    pub commands: &'static [(&'static str, CommandRule)],
    pub events: &'static [(&'static str, EventRule)],
    pub internal: &'static [(&'static str, &'static str)],
}

impl MachineSpec {
    pub fn command_rule(&self, command: &str) -> Option<CommandRule> {
        self.commands
            .iter()
            .find(|(name, _)| *name == command)
            .map(|(_, rule)| *rule)
    }

    pub fn event_rule(&self, event: &str) -> Option<EventRule> {
        self.events
            .iter()
            .find(|(name, _)| *name == event)
            .map(|(_, rule)| *rule)
    }

    /// Message types that can move the machine out of `state` — port of the
    /// oracle's `exits()` (used by the "Draining never waits on jobs"
    /// invariant test).
    pub fn exits(&self, state: &str) -> Vec<&'static str> {
        let mut leaving: Vec<&'static str> = Vec::new();
        for (command, rule) in self.commands {
            if rule.from.allows(state) {
                if let Some(to) = rule.to {
                    if to != state {
                        leaving.push(command);
                    }
                }
                for (_, target) in rule.outcomes {
                    if target.is_some_and(|to| to != state) {
                        leaving.push(command);
                    }
                }
            }
        }
        for (event, rule) in self.events {
            if rule.from.contains(&state) {
                let mut target = rule.to;
                if let Some(fatal_to) = rule.to_when_fatal_to {
                    if fatal_to != state {
                        target = Some(fatal_to);
                    }
                }
                if target.is_some_and(|to| to != state) {
                    leaving.push(event);
                }
            }
        }
        leaving.sort_unstable();
        leaving.dedup();
        leaving
    }

    /// Runtime-internal targets out of `state` — the oracle's
    /// `internal_targets()`.
    pub fn internal_targets(&self, state: &str) -> Vec<&'static str> {
        self.internal
            .iter()
            .filter(|(from, _)| *from == state)
            .map(|(_, to)| *to)
            .collect()
    }
}

macro_rules! only {
    ($($state:literal),* $(,)?) => {
        FromStates::Only(&[$($state),*])
    };
}

macro_rules! event {
    (from [$($from:literal),*] $(, to $to:expr)? $(, when_fatal_to $fatal:expr)? $(,)?) => {
        EventRule {
            from: &[$($from),*],
            to: { #[allow(unused_mut, unused_assignments)] let mut target: Option<&'static str> = None; $( target = $to; )? target },
            to_when_fatal_to: { #[allow(unused_mut, unused_assignments)] let mut fatal: Option<&'static str> = None; $( fatal = $fatal; )? fatal },
        }
    };
}

macro_rules! command {
    (from $from:expr $(, to $to:expr)? $(, outcomes $outcomes:expr)? $(,)?) => {
        CommandRule {
            from: $from,
            to: { #[allow(unused_mut, unused_assignments)] let mut target: Option<&'static str> = None; $( target = $to; )? target },
            outcomes: { #[allow(unused_mut, unused_assignments)] let mut outcomes: &'static [(&'static str, Option<&'static str>)] = &[]; $( outcomes = $outcomes; )? outcomes },
        }
    };
}

/// capture (writer task + actor) — §2.1 / the oracle's `MACHINES["capture"]`.
pub static CAPTURE: MachineSpec = MachineSpec {
    name: "capture",
    initial: "Idle",
    states: &[
        "Idle",
        "Acquiring",
        "Recording",
        "Draining",
        "Persisted",
        "Interrupted",
        "Recovering",
    ],
    commands: &[
        (
            "capture.start",
            command!(from only!("Idle", "Persisted"), to Some("Acquiring")),
        ),
        (
            "capture.stop",
            command!(from only!("Recording"), to Some("Draining")),
        ),
        (
            "capture.abort",
            command!(from only!("Acquiring", "Recording", "Draining"), to Some("Idle")),
        ),
    ],
    events: &[
        (
            "capture.started",
            event!(from ["Acquiring"], to Some("Recording")),
        ),
        (
            "capture.progress",
            event!(from ["Recording", "Draining", "Recovering"]),
        ),
        (
            "capture.gap",
            event!(from ["Recording", "Draining", "Recovering"]),
        ),
        (
            "capture.error",
            event!(
                from ["Acquiring", "Recording", "Draining", "Recovering"],
                when_fatal_to Some("Interrupted"),
            ),
        ),
        (
            "capture.stopped",
            event!(from ["Draining", "Recovering"], to Some("Persisted")),
        ),
    ],
    // Runtime-internal edges out of `Interrupted` (no wire message): a
    // salvaged take replays its durable boundary through `Recovering`,
    // while a failure that killed only the *take attempt* — the device
    // never opened, so there is nothing to salvage or replay — settles
    // straight back to `Idle` so the machine can accept the next
    // `capture.start` (issue #211: without this edge, `Interrupted` has no
    // reachable exit in the live actor).
    internal: &[("Interrupted", "Recovering"), ("Interrupted", "Idle")],
};

/// jobs (scheduler; supervised workers) — §2.2 / `MACHINES["jobs"]`. The
/// table models one job's lifecycle; see the jobs actor docs for how the
/// live scheduler maps concurrent jobs onto it.
pub static JOBS: MachineSpec = MachineSpec {
    name: "jobs",
    initial: "Idle",
    states: &[
        "Idle",
        "Queued",
        "Dispatched",
        "Loading",
        "Recognizing",
        "Transforming",
        "Completed",
        "Failed",
        "Cancelled",
        "Rejected",
    ],
    commands: &[
        (
            "jobs.submit",
            command!(
                from only!("Idle", "Completed", "Failed", "Cancelled", "Rejected"),
                outcomes &[("jobs.queued", Some("Queued")), ("jobs.rejected", Some("Rejected"))],
            ),
        ),
        (
            "jobs.cancel",
            command!(
                from only!("Queued", "Dispatched", "Loading", "Recognizing", "Transforming"),
                to Some("Cancelled"),
            ),
        ),
        ("jobs.setLimits", command!(from FromStates::Any)),
    ],
    events: &[
        ("jobs.queued", event!(from [], to Some("Queued"))),
        ("jobs.rejected", event!(from [], to Some("Rejected"))),
        (
            "jobs.progress",
            event!(from ["Loading", "Recognizing", "Transforming"]),
        ),
        (
            "jobs.completed",
            event!(from ["Recognizing", "Transforming"], to Some("Completed")),
        ),
        (
            "jobs.failed",
            event!(
                from ["Queued", "Dispatched", "Loading", "Recognizing", "Transforming"],
                to Some("Failed"),
            ),
        ),
    ],
    internal: &[
        ("Queued", "Dispatched"),
        ("Dispatched", "Loading"),
        ("Loading", "Recognizing"),
        ("Recognizing", "Transforming"),
    ],
};

/// context / mode (context service) — §2.3 / `MACHINES["context"]`.
pub static CONTEXT: MachineSpec = MachineSpec {
    name: "context",
    initial: "Observing",
    states: &[
        "Observing",
        "SnapshotTaken",
        "ModeDecided",
        "RouteFrozen",
        "Expired",
        "Released",
    ],
    commands: &[
        (
            "context.snapshot",
            command!(
                from only!("Observing"),
                outcomes &[("context.targetSnapshot", Some("SnapshotTaken"))],
            ),
        ),
        (
            "mode.set",
            command!(
                from only!("SnapshotTaken", "ModeDecided"),
                outcomes &[("mode.decision", Some("ModeDecided"))],
            ),
        ),
        (
            "context.expire",
            command!(
                from only!("SnapshotTaken", "ModeDecided", "RouteFrozen"),
                to Some("Expired"),
            ),
        ),
    ],
    events: &[
        (
            "context.targetSnapshot",
            event!(from [], to Some("SnapshotTaken")),
        ),
        ("mode.decision", event!(from [], to Some("ModeDecided"))),
        // Emitted when the audio route freezes at capture start; only legal
        // once a mode has been decided, never after (a later spoken phrase
        // cannot un-send audio).
        (
            "mode.routeFrozen",
            event!(from ["ModeDecided"], to Some("RouteFrozen")),
        ),
    ],
    internal: &[
        ("SnapshotTaken", "Expired"),
        ("ModeDecided", "Expired"),
        ("RouteFrozen", "Released"),
        ("Expired", "Observing"),
        ("Released", "Observing"),
    ],
};

/// docs / revisions (document service) — §2.4 / `MACHINES["docs"]`.
pub static DOCS: MachineSpec = MachineSpec {
    name: "docs",
    initial: "Steady",
    states: &["Steady", "Validating", "Committed", "Conflicted"],
    commands: &[
        (
            "docs.updateHead",
            command!(
                from only!("Steady", "Committed", "Conflicted"),
                to Some("Validating"),
            ),
        ),
        (
            "docs.appendTurn",
            command!(
                from only!("Steady", "Committed"),
                outcomes &[("docs.turnAppended", None)],
            ),
        ),
        (
            "docs.get",
            command!(from only!("Steady", "Committed", "Conflicted")),
        ),
    ],
    events: &[
        (
            "docs.headUpdated",
            event!(from ["Validating"], to Some("Committed")),
        ),
        (
            "docs.headConflict",
            event!(from ["Validating"], to Some("Conflicted")),
        ),
        ("docs.turnAppended", event!(from [])),
    ],
    internal: &[("Committed", "Steady")],
};

/// delivery (delivery service) — §2.5 / `MACHINES["delivery"]`.
pub static DELIVERY: MachineSpec = MachineSpec {
    name: "delivery",
    initial: "Idle",
    states: &[
        "Idle",
        "Prepared",
        "Revalidating",
        "SubmittedUnconfirmed",
        "Confirmed",
        "Failed",
        "Conflict",
        "Cancelled",
    ],
    commands: &[
        (
            "delivery.prepare",
            command!(
                from only!("Idle", "Confirmed", "Failed", "Conflict", "Cancelled"),
                outcomes &[("delivery.prepared", Some("Prepared"))],
            ),
        ),
        (
            "delivery.apply",
            command!(from only!("Prepared"), to Some("Revalidating")),
        ),
        (
            "delivery.cancel",
            command!(
                from only!("Prepared", "Revalidating", "SubmittedUnconfirmed"),
                to Some("Cancelled"),
            ),
        ),
        (
            "delivery.copyFallback",
            command!(from only!("Failed", "Conflict")),
        ),
    ],
    events: &[
        ("delivery.prepared", event!(from [], to Some("Prepared"))),
        (
            "delivery.submittedUnconfirmed",
            event!(from ["Revalidating"], to Some("SubmittedUnconfirmed")),
        ),
        (
            "delivery.confirmed",
            event!(from ["SubmittedUnconfirmed"], to Some("Confirmed")),
        ),
        (
            "delivery.failed",
            event!(from ["Revalidating", "SubmittedUnconfirmed"], to Some("Failed")),
        ),
        (
            "delivery.conflict",
            event!(from ["Revalidating", "SubmittedUnconfirmed"], to Some("Conflict")),
        ),
    ],
    internal: &[],
};

/// All five specs by machine name.
pub fn spec_for(machine: &str) -> Option<&'static MachineSpec> {
    match machine {
        "capture" => Some(&CAPTURE),
        "jobs" => Some(&JOBS),
        "context" => Some(&CONTEXT),
        "docs" => Some(&DOCS),
        "delivery" => Some(&DELIVERY),
        _ => None,
    }
}

/// All five specs.
pub fn all_specs() -> [&'static MachineSpec; 5] {
    [&CAPTURE, &JOBS, &CONTEXT, &DOCS, &DELIVERY]
}

/// The v1 command type set across all machines (the oracle's
/// `machine_commands` union — used by the tables-match-schema test).
pub fn all_command_types() -> Vec<&'static str> {
    all_specs()
        .iter()
        .flat_map(|spec| spec.commands.iter().map(|(name, _)| *name))
        .collect()
}

/// The v1 event type set across all machines (the oracle's
/// `machine_events` union).
pub fn all_event_types() -> Vec<&'static str> {
    all_specs()
        .iter()
        .flat_map(|spec| spec.events.iter().map(|(name, _)| *name))
        .collect()
}

/// Maps a wire type to its owning machine, or `None` for unknown types /
/// `runtime.*` events.
pub fn machine_of_type(type_: &str) -> Option<&'static str> {
    static INDEX: OnceLock<HashMap<&'static str, &'static str>> = OnceLock::new();
    let index = INDEX.get_or_init(|| {
        let mut map = HashMap::new();
        for spec in all_specs() {
            for (command, _) in spec.commands {
                map.insert(*command, spec.name);
            }
            for (event, _) in spec.events {
                map.insert(*event, spec.name);
            }
        }
        map
    });
    index.get(type_).copied()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The oracle's `test_internal_edges_stay_inside_declared_states`.
    #[test]
    fn internal_edges_stay_inside_declared_states() {
        for spec in all_specs() {
            for (from, to) in spec.internal {
                assert!(spec.states.contains(from), "{}: {from}", spec.name);
                assert!(spec.states.contains(to), "{}: {to}", spec.name);
            }
        }
    }

    /// The oracle's `test_machine_tables_match_schema_kinds`: the union of
    /// the tables equals the typed v1 sets (here: the `Command`/`Event`
    /// enums, which are the schemas' Rust spelling).
    #[test]
    fn machine_tables_match_typed_sets() {
        use crate::protocol::{Command, Event, JobLimits};

        let schema_commands: std::collections::HashSet<&str> =
            all_command_types().into_iter().collect();
        let mut typed_commands = std::collections::HashSet::new();
        // Every enum variant reports its wire type; enumerate via samples.
        let samples = [
            Command::CaptureStart { policy: "p".into() },
            Command::CaptureStop { drain: None },
            Command::CaptureAbort,
            Command::JobsSubmit {
                capture_ref: "t".into(),
                route: "r".into(),
                budget: "b".into(),
            },
            Command::JobsCancel { job_id: "j".into() },
            Command::JobsSetLimits(JobLimits {
                max_queued: 1,
                max_concurrent: 1,
                per_route: vec![],
            }),
            Command::ContextSnapshot { source: "s".into() },
            Command::ModeSet {
                mode: "m".into(),
                source: crate::protocol::Manual,
            },
            Command::ContextExpire,
            Command::DocsUpdateHead {
                doc_id: "d".into(),
                expected_base: 0,
                new_revision: crate::protocol::Revision {
                    rev_id: "r".into(),
                    base_revision: 0,
                    source_attempt_ids: vec![],
                    instruction_template_id: "t".into(),
                    text: String::new(),
                    status: "candidate".into(),
                    provenance: "recognition".into(),
                },
            },
            Command::DocsAppendTurn {
                doc_id: "d".into(),
                take_ref: "t".into(),
            },
            Command::DocsGet {
                doc_id: "d".into(),
                page: 0,
            },
            Command::DeliveryPrepare {
                revision_id: "r".into(),
                target_ref: "t".into(),
            },
            Command::DeliveryApply {
                delivery_id: "d".into(),
            },
            Command::DeliveryCancel { delivery_id: None },
            Command::DeliveryCopyFallback {
                delivery_id: "d".into(),
            },
        ];
        assert_eq!(samples.len(), 16, "all 16 v1 commands enumerated");
        for command in &samples {
            assert!(
                typed_commands.insert(command.type_name()),
                "duplicate type {}",
                command.type_name()
            );
        }
        assert_eq!(typed_commands, schema_commands);

        let schema_events: std::collections::HashSet<&str> =
            all_event_types().into_iter().collect();
        assert_eq!(schema_events.len(), 21, "21 machine event types");
        // Every Event variant's type name must appear in the tables (the
        // reverse direction plus the count is checked in conformance.rs via
        // the fixtures).
        let event_samples = [
            Event::CaptureStarted {
                device: "d".into(),
                actual_rate: 1,
                channels: 1,
            },
            Event::CaptureProgress {
                ack_samples: 0,
                clip_ratio: 0.0,
                level: 0.0,
            },
            Event::CaptureGap {
                start_sample: 0,
                end_sample: 0,
            },
            Event::CaptureError {
                code: "c".into(),
                fatal: false,
            },
            Event::CaptureStopped {
                final_sample_index: 0,
                acknowledged_samples: 0,
                gaps: vec![],
                journal_id: "j".into(),
                sample_duration_ms: 0.0,
                wall_clock_ms: 0.0,
            },
            Event::JobsQueued,
            Event::JobsRejected {
                reason: crate::protocol::RejectReason::QueueFull,
            },
            Event::JobsProgress {
                partial: String::new(),
                stability_hint: "growing".into(),
            },
            Event::JobsCompleted(crate::protocol::CompletionData {
                attempt_id: "a".into(),
                text: String::new(),
                backend: "b".into(),
                timing: 0.0,
                completion_evidence: "final_decode".into(),
            }),
            Event::JobsFailed {
                reason: "r".into(),
                retryable: false,
            },
            Event::ContextTargetSnapshot(crate::protocol::TargetSnapshotData {
                descriptor: "d".into(),
                digest: "g".into(),
                capabilities: vec![],
                selection_range: crate::protocol::Span {
                    start_offset: 0,
                    end_offset: 0,
                },
                offset_encoding: "utf-16".into(),
                expiry: "2026-09-20T10:00:00Z".into(),
            }),
            Event::ModeDecision(crate::protocol::DecisionData {
                mode_id: "m".into(),
                source: crate::protocol::DecisionSource::Manual,
                matched_prefix_span: None,
                explanation: String::new(),
                payload_view: "raw".into(),
            }),
            Event::ModeRouteFrozen {
                route: "r".into(),
                decided_at: "2026-09-20T10:00:00Z".into(),
            },
            Event::DocsHeadUpdated {
                doc_id: "d".into(),
                head_revision: 1,
            },
            Event::DocsHeadConflict {
                expected: 1,
                actual: 2,
                candidate_preserved: true,
            },
            Event::DocsTurnAppended { turn_seq: 1 },
            Event::DeliveryPrepared {
                delivery_id: "d".into(),
                compare_token: "t".into(),
            },
            Event::DeliverySubmittedUnconfirmed,
            Event::DeliveryConfirmed {
                evidence_level: "target_ack".into(),
            },
            Event::DeliveryFailed {
                reason: "r".into(),
                fallback_suggested: true,
            },
            Event::DeliveryConflict {
                expected_target: "a".into(),
                actual_target: "b".into(),
            },
            Event::RuntimeNack {
                reason: crate::protocol::NackReason,
            },
        ];
        assert_eq!(event_samples.len(), 22, "all 22 v1 events enumerated");
        for event in &event_samples {
            let name = event.type_name();
            if name == "runtime.nack" {
                assert!(!schema_events.contains(name), "nack is runtime-level");
            } else {
                assert!(schema_events.contains(name), "{name} missing from tables");
            }
        }
    }

    /// The oracle's `test_capture_never_blocks_draining_on_jobs`.
    #[test]
    fn capture_never_blocks_draining_on_jobs() {
        for state in CAPTURE.states {
            let leaving = CAPTURE.exits(state);
            assert!(
                !leaving.iter().any(|t| t.starts_with("jobs.")),
                "{state}: {leaving:?}"
            );
        }
        let draining_exits = CAPTURE.exits("Draining");
        assert!(draining_exits.contains(&"capture.stopped"));
    }
}
