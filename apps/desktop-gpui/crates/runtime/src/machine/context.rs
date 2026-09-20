//! The context / mode machine (§2.3): the context service actor.
//!
//! `Observing → SnapshotTaken → ModeDecided → RouteFrozen → Expired |
//! Released`, with the runtime-internal cycle edges taken by the actor at
//! the points the fixtures mark with `$advance` (expiry → `Observing` on
//! the next cycle, `RouteFrozen → Released` on take completion).
//!
//! [`RouteFreezer`] is the capture actor's synchronous probe: `capture.start`
//! asks the context service to freeze the audio route **before the first
//! frame can leave the runtime** — `mode.routeFrozen` is emitted only from
//! `ModeDecided`, and a later spoken phrase cannot un-send audio. The
//! frozen-route registry the freezer maintains is what the jobs scheduler
//! checks at `jobs.submit` (the audio-leave proxy).

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::bus::EventBus;
use crate::machine::{Inbound, MachineCore, Receipt, Rejection};
use crate::protocol::tables::CONTEXT;
use crate::protocol::{Command, DecisionData, DecisionSource, Event, TargetSnapshotData};

/// Where target snapshots come from (E03 wiring is I5; the stub is
/// clearly labeled and never fabricates authority).
pub trait ContextProvider: Send + Sync {
    /// Resolves `context.snapshot{source}` into
    /// `context.targetSnapshot` data.
    fn snapshot(&self, source: &str) -> Result<TargetSnapshotData, String>;
}

/// The Mode A stub: a synthetic, clearly-labeled snapshot for the named
/// source. Real editor integration lands with I5.
pub struct StubContextProvider {
    ttl: Duration,
}

impl StubContextProvider {
    pub fn new() -> Arc<Self> {
        Arc::new(StubContextProvider {
            ttl: Duration::from_secs(30),
        })
    }
}

impl Default for StubContextProvider {
    fn default() -> Self {
        StubContextProvider {
            ttl: Duration::from_secs(30),
        }
    }
}

impl ContextProvider for StubContextProvider {
    fn snapshot(&self, source: &str) -> Result<TargetSnapshotData, String> {
        let expiry = time::OffsetDateTime::from_unix_timestamp(
            time::OffsetDateTime::now_utc().unix_timestamp() + self.ttl.as_secs() as i64,
        )
        .unwrap_or_else(|_| time::OffsetDateTime::now_utc());
        Ok(TargetSnapshotData {
            descriptor: format!("{source}:stub"),
            digest: format!("stub:{source}"),
            capabilities: vec!["text-insert".to_string()],
            selection_range: crate::protocol::Span {
                start_offset: 0,
                end_offset: 0,
            },
            offset_encoding: "utf-16".to_string(),
            expiry: expiry
                .format(&time::format_description::well_known::Rfc3339)
                .unwrap_or_else(|_| "1970-01-01T00:00:00Z".to_string()),
        })
    }
}

/// How a decided mode maps to the audio route token it freezes. Default:
/// `verbatim` → `local-authoring-default`, everything else →
/// `local-default` (the fixture corpus's two routes; a real policy catalog
/// is I5).
pub trait RoutePolicy: Send + Sync {
    fn route_for_mode(&self, mode: &str) -> String;
}

/// The default route policy (see [`RoutePolicy`]).
pub struct DefaultRoutePolicy;

impl RoutePolicy for DefaultRoutePolicy {
    fn route_for_mode(&self, mode: &str) -> String {
        if mode == "verbatim" {
            "local-authoring-default".to_string()
        } else {
            "local-default".to_string()
        }
    }
}

/// Routes frozen this session, with their `mode.routeFrozen` timestamps —
/// consulted by the jobs scheduler's audio-leave proxy. Once frozen, a
/// route stays frozen for the session (freezing is one-way: audio that
/// left cannot be un-sent, and the corpus expires contexts without
/// unfreezing their routes).
pub type FrozenRoutes = Arc<Mutex<HashMap<String, String>>>;

/// Messages the context actor receives.
pub enum ContextMsg {
    Command(Inbound),
    /// The capture actor asking to freeze the route for `take` before the
    /// first frame leaves. Replies with the frozen route, or `None` when
    /// no mode has been decided (the freeze invariant then falls to the
    /// jobs.submit check).
    FreezeRoute {
        take: String,
        reply: crate::channel::Sender<Option<String>>,
    },
    /// Take completion: `RouteFrozen → Released` (runtime-internal).
    TakeCompleted { take: String },
    Shutdown,
}

/// The capture actor's handle onto the context service.
#[derive(Clone)]
pub struct RouteFreezer {
    inbox: crate::channel::Sender<ContextMsg>,
}

impl RouteFreezer {
    pub fn new(inbox: crate::channel::Sender<ContextMsg>) -> RouteFreezer {
        RouteFreezer { inbox }
    }

    /// Asks the context service to freeze the audio route for `take`.
    /// Bounded wait; `None` means "could not freeze now" and never blocks
    /// capture.
    pub fn freeze(&self, take: &str) -> Option<String> {
        let (tx, rx) = crate::channel::bounded(1);
        self.inbox
            .try_send(ContextMsg::FreezeRoute {
                take: take.to_string(),
                reply: tx,
            })
            .ok()?;
        match rx.recv_timeout(Duration::from_millis(1_000)) {
            Ok(route) => route,
            Err(_) => None,
        }
    }

    /// Notifies take completion (best effort).
    pub fn take_completed(&self, take: &str) -> bool {
        self.inbox
            .try_send(ContextMsg::TakeCompleted { take: take.to_string() })
            .is_ok()
    }
}

/// The context service actor.
pub struct ContextActor {
    inbox: crate::channel::Receiver<ContextMsg>,
    bus: Arc<EventBus>,
    view: super::ViewSlot,
    core: MachineCore,
    provider: Arc<dyn ContextProvider>,
    route_policy: Arc<dyn RoutePolicy>,
    frozen_routes: FrozenRoutes,
    /// The currently decided mode (drives the frozen route).
    mode: Option<String>,
    /// The current snapshot's explicit expiry (short-lived by contract).
    snapshot_expiry: Option<String>,
}

impl ContextActor {
    pub fn new(
        inbox: crate::channel::Receiver<ContextMsg>,
        bus: Arc<EventBus>,
        view: super::ViewSlot,
        provider: Arc<dyn ContextProvider>,
        route_policy: Arc<dyn RoutePolicy>,
        frozen_routes: FrozenRoutes,
    ) -> ContextActor {
        ContextActor {
            inbox,
            bus,
            view,
            core: MachineCore::new(&CONTEXT),
            provider,
            route_policy,
            frozen_routes,
            mode: None,
            snapshot_expiry: None,
        }
    }

    pub fn run(mut self) {
        loop {
            match self.inbox.recv() {
                Ok(ContextMsg::Command(inbound)) => self.handle_command(inbound),
                Ok(ContextMsg::FreezeRoute { take, reply }) => {
                    let _ = reply.try_send(self.handle_freeze(&take));
                }
                Ok(ContextMsg::TakeCompleted { take: _ }) => {
                    // RouteFrozen -> Released (runtime-internal).
                    if self.core.state() == "RouteFrozen" {
                        if let Err(violation) = self.core.advance_internal("Released") {
                            self.core.record_violation(violation);
                        }
                    }
                }
                Ok(ContextMsg::Shutdown) | Err(crate::channel::RecvError::Closed) => break,
                Err(crate::channel::RecvError::Timeout) => {
                    unreachable!("recv has no timeout")
                }
            }
            *self.view.lock().expect("context view lock") = self.core.view();
        }
    }

    /// Snapshot/mode expiry, checked on activity: the runtime-internal
    /// `SnapshotTaken → Expired` / `ModeDecided → Expired` edges when the
    /// snapshot's explicit expiry has passed. (No background timer in
    /// Mode A — `context.expire` remains the explicit command form.)
    fn auto_expire(&mut self) {
        if self.core.state() == "SnapshotTaken" || self.core.state() == "ModeDecided" {
            if let Some(expiry) = &self.snapshot_expiry {
                if let Ok(expiry_at) = time::OffsetDateTime::parse(
                    expiry,
                    &time::format_description::well_known::Rfc3339,
                ) {
                    if time::OffsetDateTime::now_utc() >= expiry_at {
                        if let Err(violation) = self.core.advance_internal("Expired") {
                            self.core.record_violation(violation);
                        }
                    }
                }
            }
        }
    }

    /// The runtime-internal next-cycle edge: a snapshot arriving in
    /// `Expired`/`Released` first returns to `Observing` (the fixtures'
    /// `{"$advance": "Observing"}` — the live actor takes it itself).
    fn cycle_to_observing(&mut self) {
        for state in ["Expired", "Released"] {
            if self.core.state() == state {
                if let Err(violation) = self.core.advance_internal("Observing") {
                    self.core.record_violation(violation);
                }
            }
        }
    }

    fn handle_command(&mut self, inbound: Inbound) {
        self.auto_expire();
        let super::Inbound { corr, command, reply, .. } = inbound;
        let corr = corr.unwrap_or_else(|| "ctx-anon".to_string());
        match command {
            Command::ContextSnapshot { source } => {
                self.cycle_to_observing();
                match self.core.commit_command("context.snapshot", Some(corr.clone())) {
                    Ok(_) => match self.provider.snapshot(&source) {
                        Ok(data) => {
                            let _ = reply.try_send(Ok(Receipt::Accepted));
                            self.snapshot_expiry = Some(data.expiry.clone());
                            // The snapshot resolves the pending
                            // context.snapshot on this corr.
                            match self.core.resolve_outcome("context.targetSnapshot", Some(&corr)) {
                                Ok(_) => {
                                    let _ = self.bus.emit(
                                        Event::ContextTargetSnapshot(data),
                                        Some(&corr),
                                    );
                                }
                                Err(violation) => self.core.record_violation(violation),
                            }
                        }
                        Err(message) => {
                            // The provider failed; v1 defines no failure
                            // event for context.snapshot, so the command is
                            // answered by receipt only and nothing is
                            // emitted (never absorbed silently).
                            self.core.record_violation(
                                crate::protocol::replay::Violation::InvalidEnvelope(
                                    format!("context provider failed: {message}"),
                                ),
                            );
                            let _ = reply.try_send(Err(Rejection::InvalidPayload(message)));
                        }
                    },
                    Err(violation) => {
                        self.core.record_violation(violation.clone());
                        let _ = reply.try_send(Err(illegal(
                            "context.snapshot",
                            self.core.state(),
                            violation,
                        )));
                    }
                }
            }
            Command::ModeSet { mode, source: _ } => {
                match self.core.commit_command("mode.set", Some(corr.clone())) {
                    Ok(_) => {
                        let _ = reply.try_send(Ok(Receipt::Accepted));
                        let decision = DecisionData {
                            mode_id: mode.clone(),
                            source: DecisionSource::Manual,
                            matched_prefix_span: None,
                            explanation: format!(
                                "Manual mode selection for {mode:?} (runtime Mode A; mode \
                                 catalog wiring is I5)."
                            ),
                            payload_view: "raw-text-only".to_string(),
                        };
                        self.mode = Some(mode);
                        // The decision resolves the pending mode.set on this corr.
                        match self.core.resolve_outcome("mode.decision", Some(&corr)) {
                            Ok(_) => {
                                let _ = self.bus.emit(Event::ModeDecision(decision), Some(&corr));
                            }
                            Err(violation) => self.core.record_violation(violation),
                        }
                    }
                    Err(violation) => {
                        self.core.record_violation(violation.clone());
                        let _ = reply.try_send(Err(illegal("mode.set", self.core.state(), violation)));
                    }
                }
            }
            Command::ContextExpire => {
                match self.core.commit_command("context.expire", Some(corr)) {
                    Ok(_) => {
                        let _ = reply.try_send(Ok(Receipt::Accepted));
                    }
                    Err(violation) => {
                        self.core.record_violation(violation.clone());
                        let _ = reply.try_send(Err(illegal(
                            "context.expire",
                            self.core.state(),
                            violation,
                        )));
                    }
                }
            }
            other => {
                let _ = reply.try_send(Err(Rejection::UnknownMessageType(
                    other.type_name().to_string(),
                )));
            }
        }
    }

    /// `mode.routeFrozen` — legal only from `ModeDecided`.
    fn handle_freeze(&mut self, take: &str) -> Option<String> {
        if self.core.state() != "ModeDecided" {
            return None;
        }
        let mode = self.mode.clone()?;
        let route = self.route_policy.route_for_mode(&mode);
        let decided_at = crate::bus::now_ts();
        match self.core.emit_event("mode.routeFrozen", None) {
            Ok(_) => {
                let _ = self.bus.emit(
                    Event::ModeRouteFrozen {
                        route: route.clone(),
                        decided_at: decided_at.clone(),
                    },
                    Some(take),
                );
                self.frozen_routes
                    .lock()
                    .expect("frozen routes lock")
                    .insert(route.clone(), decided_at);
                Some(route)
            }
            Err(violation) => {
                self.core.record_violation(violation);
                None
            }
        }
    }
}

fn illegal(command: &str, state: &str, violation: crate::protocol::replay::Violation) -> Rejection {
    Rejection::IllegalInState {
        command: command.to_string(),
        state: state.to_string(),
        detail: violation.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_route_policy_matches_the_corpus_routes() {
        let policy = DefaultRoutePolicy;
        assert_eq!(policy.route_for_mode("verbatim"), "local-authoring-default");
        assert_eq!(policy.route_for_mode("code-guidance"), "local-default");
    }
}
