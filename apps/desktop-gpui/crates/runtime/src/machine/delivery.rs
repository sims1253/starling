//! The external delivery machine (§2.5): the delivery service actor.
//!
//! `Prepared → Revalidating → SubmittedUnconfirmed → Confirmed | Failed |
//! Conflict | Cancelled`, plus the pre-delivery `Idle`. The table models
//! one delivery's lifecycle, so the actor keeps one [`MachineCore`] per
//! delivery (keyed by `deliveryId`) exactly the way the jobs scheduler
//! keeps one per job — the fixtures never interleave two live deliveries,
//! and neither does a serialized-by-correlation replay of the live
//! stream.
//!
//! Invariants held here: target identity is revalidated immediately
//! before apply (`delivery.apply` → `Revalidating`); **no Enter injection**
//! — there is no code path that sends, executes, auto-confirms or presses
//! anything, `apply` is user-initiated and names its delivery; the stub
//! adapter below never fabricates a confirmation.
//!
//! Real delivery adapters are E03's platform work (issue #221) on the
//! [`DeliveryAdapter`] seam — a seam the host crate's adapter suite
//! proves end-to-end over the IPC transport with a real target;
//! [`StubDeliveryAdapter`] stays the unwired default and records what it
//! did, failing apply honestly (`no_delivery_adapter`) instead of
//! pretending text landed.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::bus::EventBus;
use crate::machine::{Inbound, MachineCore, Receipt, Rejection};
use crate::protocol::tables::DELIVERY;
use crate::protocol::{Command, Event};

use super::docs::RevisionRegistry;

/// What an adapter learned revalidating a target just before apply.
#[derive(Debug, Clone, PartialEq)]
pub enum Revalidation {
    /// The target still matches the frozen compare token.
    Unchanged,
    /// The target changed since freeze: `{expectedTarget, actualTarget}`.
    Changed { expected: String, actual: String },
}

/// Why an insertion failed. `fallback_suggested` feeds
/// `delivery.failed{fallbackSuggested}`.
#[derive(Debug, Clone)]
pub struct InsertionFailure {
    pub reason: String,
    pub fallback_suggested: bool,
}

/// The evidence an adapter could honestly establish for a completed
/// insertion — `delivery.confirmed{evidenceLevel}` states it (synthetic
/// key acceptance is not proof text landed).
#[derive(Debug, Clone)]
pub struct InsertEvidence {
    pub level: String,
}

/// The external-target seam (the adapters are E03's platform work,
/// issue #221). `insert` is only ever called from the user-initiated
/// `delivery.apply` path.
pub trait DeliveryAdapter: Send + Sync {
    /// Freeze-time target validation; returns the compare token.
    fn prepare(&self, target_ref: &str) -> Result<String, String>;
    /// Immediate-before-apply revalidation of identity/range/version.
    fn revalidate(&self, target_ref: &str, compare_token: &str) -> Revalidation;
    /// The insertion itself. Err means the text did not land.
    fn insert(&self, delivery_id: &str, target_ref: &str, text: &str)
        -> Result<InsertEvidence, InsertionFailure>;
    /// An honest description for snapshots.
    fn describe(&self) -> String;
}

/// The Mode A stub adapter. It prepares (bookkeeping it can honestly do)
/// and revalidates, but **never confirms an insertion**: `insert` always
/// fails with `no_delivery_adapter` and suggests the copy fallback. Every
/// action is recorded in a public log for inspection and tests.
pub struct StubDeliveryAdapter {
    log: Mutex<Vec<String>>,
}

impl StubDeliveryAdapter {
    pub fn new() -> Arc<Self> {
        Arc::new(StubDeliveryAdapter {
            log: Mutex::new(Vec::new()),
        })
    }

    fn record(&self, entry: String) {
        self.log.lock().expect("stub adapter log lock").push(entry);
    }

    /// What the stub actually did, in order.
    pub fn log(&self) -> Vec<String> {
        self.log.lock().expect("stub adapter log lock").clone()
    }
}

impl Default for StubDeliveryAdapter {
    fn default() -> Self {
        StubDeliveryAdapter {
            log: Mutex::new(Vec::new()),
        }
    }
}

impl DeliveryAdapter for StubDeliveryAdapter {
    fn prepare(&self, target_ref: &str) -> Result<String, String> {
        let token = format!("stub:{target_ref}:{}", crate::bus::new_id("tok"));
        self.record(format!("prepared target={target_ref} token={token} (stub: no real target)"));
        Ok(token)
    }

    fn revalidate(&self, _target_ref: &str, _compare_token: &str) -> Revalidation {
        // The stub holds no real target, so there is nothing that could
        // have changed; a real adapter compares live identity here.
        Revalidation::Unchanged
    }

    fn insert(
        &self,
        delivery_id: &str,
        target_ref: &str,
        _text: &str,
    ) -> Result<InsertEvidence, InsertionFailure> {
        self.record(format!(
            "apply attempted delivery={delivery_id} target={target_ref}: NOT delivered — no \
             real delivery adapter is wired (E03); refusing to fake a confirmation"
        ));
        Err(InsertionFailure {
            reason: "no_delivery_adapter".to_string(),
            fallback_suggested: true,
        })
    }

    fn describe(&self) -> String {
        "stub (never confirms)".to_string()
    }
}

struct DeliveryState {
    core: MachineCore,
    #[allow(dead_code)] // identifies the prepared revision; no v1 wire
                        // field carries it (the E21 documents product
                        // surface decides how it surfaces)
    revision_id: String,
    target_ref: String,
    compare_token: String,
    /// The revision's text, captured at prepare for the apply path.
    text: String,
    /// Set by `delivery.copyFallback` (records intent; closes nothing).
    fallback_requested: bool,
}

/// Messages the delivery actor receives.
pub enum DeliveryMsg {
    Command(Inbound),
    Shutdown,
}

/// The delivery service actor.
pub struct DeliveryActor {
    inbox: crate::channel::Receiver<DeliveryMsg>,
    bus: Arc<EventBus>,
    view: super::ViewSlot,
    deliveries: HashMap<String, DeliveryState>,
    order: Vec<String>,
    revisions: RevisionRegistry,
    adapter: Arc<dyn DeliveryAdapter>,
}

impl DeliveryActor {
    pub fn new(
        inbox: crate::channel::Receiver<DeliveryMsg>,
        bus: Arc<EventBus>,
        view: super::ViewSlot,
        revisions: RevisionRegistry,
        adapter: Arc<dyn DeliveryAdapter>,
    ) -> DeliveryActor {
        DeliveryActor {
            inbox,
            bus,
            view,
            deliveries: HashMap::new(),
            order: Vec::new(),
            revisions,
            adapter,
        }
    }

    pub fn run(mut self) {
        loop {
            match self.inbox.recv() {
                Ok(DeliveryMsg::Command(inbound)) => self.handle_command(inbound),
                Ok(DeliveryMsg::Shutdown) | Err(crate::channel::RecvError::Closed) => break,
                Err(crate::channel::RecvError::Timeout) => {
                    unreachable!("recv has no timeout")
                }
            }
            *self.view.lock().expect("delivery view lock") = self.snapshot_view();
        }
    }

    /// The aggregate view: the most recent delivery's machine state (v1
    /// models one delivery per wire stream; the actor holds the rest).
    fn snapshot_view(&self) -> super::MachineView {
        let state = self
            .order
            .last()
            .and_then(|id| self.deliveries.get(id))
            .map(|delivery| delivery.core.view());
        match state {
            Some(mut view) => {
                view.machine = "delivery".to_string();
                view
            }
            None => MachineCore::new(&DELIVERY).view(),
        }
    }


    /// Emits through the delivery's own machine core: an illegal event is
    /// never sent (the stream stays oracle-legal) and the violation lands
    /// in the core's history instead of being absorbed.
    fn emit(&mut self, delivery_id: &str, event: Event, corr: &str) {
        let Some(state) = self.deliveries.get_mut(delivery_id) else {
            return;
        };
        match state.core.emit_event(event.type_name(), None) {
            Ok(_) => {
                let _ = self.bus.emit(event, Some(corr));
            }
            Err(violation) => state.core.record_violation(violation),
        }
    }

    fn handle_command(&mut self, inbound: Inbound) {
        let super::Inbound { corr, command, reply, .. } = inbound;
        let corr = corr.unwrap_or_else(|| "dlv-anon".to_string());
        match command {
            Command::DeliveryPrepare {
                revision_id,
                target_ref,
            } => self.handle_prepare(reply, corr, revision_id, target_ref),
            Command::DeliveryApply { delivery_id } => self.handle_apply(reply, corr, delivery_id),
            Command::DeliveryCancel { delivery_id } => {
                let id = delivery_id.unwrap_or_else(|| self.latest_active().unwrap_or_default());
                if id.is_empty() {
                    let _ = reply.try_send(Err(Rejection::UnknownDelivery { delivery_id: id }));
                    return;
                }
                self.handle_cancel(reply, corr, id);
            }
            Command::DeliveryCopyFallback { delivery_id } => {
                let Some(state) = self.deliveries.get_mut(&delivery_id) else {
                    let _ = reply.try_send(Err(Rejection::UnknownDelivery { delivery_id }));
                    return;
                };
                match state.core.commit_command("delivery.copyFallback", Some(corr)) {
                    Ok(_) => {
                        state.fallback_requested = true;
                        let _ = reply.try_send(Ok(Receipt::Accepted));
                    }
                    Err(violation) => {
                        state.core.record_violation(violation.clone());
                        let state_now = state.core.state().to_string();
                        let _ = reply.try_send(Err(illegal(
                            "delivery.copyFallback",
                            &state_now,
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

    fn handle_prepare(
        &mut self,
        reply: super::ReceiptTx,
        corr: String,
        revision_id: String,
        target_ref: String,
    ) {
        // A delivery may only be prepared against a revision this runtime
        // knows (committed heads published by the docs service).
        let text = {
            let revisions = self.revisions.lock().expect("revision registry lock");
            revisions
                .get(&revision_id)
                .map(|(_, revision)| revision.text.clone())
        };
        let Some(text) = text else {
            let _ = reply.try_send(Err(Rejection::UnknownRevision { revision_id }));
            return;
        };
        // The adapter must be able to prepare before the command commits
        // (v1 defines no prepare-failure event, so a failure answers here).
        let token = match self.adapter.prepare(&target_ref) {
            Ok(token) => token,
            Err(message) => {
                let _ = reply.try_send(Err(Rejection::InvalidPayload(
                    format!("delivery adapter could not prepare: {message}"),
                )));
                return;
            }
        };
        let mut core = MachineCore::new(&DELIVERY);
        match core.commit_command("delivery.prepare", Some(corr.clone())) {
            Ok(_) => {
                let delivery_id = crate::bus::new_id("dlv");
                let _ = reply.try_send(Ok(Receipt::Accepted));
                // `delivery.prepare` is outcome-pending: resolve_outcome
                // performs the `delivery.prepared` transition itself, and
                // the event rule for `delivery.prepared` is `from []` (an
                // outcome only, never a free event) — so the bus emit
                // follows the resolve directly. The extra emit_event the
                // actor used to attempt here always violated the table
                // and was swallowed, which meant the live stream never
                // carried `delivery.prepared` at all (found by the I5
                // adapter wiring's live delivery walk, issue #220; the
                // conformance corpus replays this trace through the table
                // and so never saw the actor's side).
                match core.resolve_outcome("delivery.prepared", Some(&corr)) {
                    Ok(_) => {
                        let event = Event::DeliveryPrepared {
                            delivery_id: delivery_id.clone(),
                            compare_token: token.clone(),
                        };
                        let _ = self.bus.emit(event, Some(&corr));
                        self.deliveries.insert(
                            delivery_id.clone(),
                            DeliveryState {
                                core,
                                revision_id,
                                target_ref,
                                compare_token: token,
                                text,
                                fallback_requested: false,
                            },
                        );
                        self.order.push(delivery_id);
                    }
                    Err(violation) => core.record_violation(violation),
                }
            }
            Err(violation) => {
                core.record_violation(violation.clone());
                let _ = reply.try_send(Err(illegal("delivery.prepare", core.state(), violation)));
            }
        }
    }

    fn handle_apply(&mut self, reply: super::ReceiptTx, corr: String, delivery_id: String) {
        let Some(state) = self.deliveries.get_mut(&delivery_id) else {
            let _ = reply.try_send(Err(Rejection::UnknownDelivery { delivery_id }));
            return;
        };
        let text = state.text.clone();
        let target_ref = state.target_ref.clone();
        let compare_token = state.compare_token.clone();
        match state.core.commit_command("delivery.apply", Some(corr.clone())) {
            Ok(_) => {
                let _ = reply.try_send(Ok(Receipt::Accepted));
                // Revalidate immediately before apply.
                match self.adapter.revalidate(&target_ref, &compare_token) {
                    Revalidation::Changed { expected, actual } => {
                        self.emit(
                            &delivery_id,
                            Event::DeliveryConflict {
                                expected_target: expected,
                                actual_target: actual,
                            },
                            &corr,
                        );
                    }
                    Revalidation::Unchanged => {
                        match self.adapter.insert(&delivery_id, &target_ref, &text) {
                            Ok(evidence) => {
                                // submittedUnconfirmed then confirmed —
                                // stating the evidence level honestly.
                                self.emit(
                                    &delivery_id,
                                    Event::DeliverySubmittedUnconfirmed,
                                    &corr,
                                );
                                self.emit(
                                    &delivery_id,
                                    Event::DeliveryConfirmed {
                                        evidence_level: evidence.level,
                                    },
                                    &corr,
                                );
                            }
                            Err(failure) => {
                                self.emit(
                                    &delivery_id,
                                    Event::DeliveryFailed {
                                        reason: failure.reason,
                                        fallback_suggested: failure.fallback_suggested,
                                    },
                                    &corr,
                                );
                            }
                        }
                    }
                }
            }
            Err(violation) => {
                state.core.record_violation(violation.clone());
                let state_now = state.core.state().to_string();
                let _ = reply.try_send(Err(illegal("delivery.apply", &state_now, violation)));
            }
        }
    }

    fn handle_cancel(&mut self, reply: super::ReceiptTx, corr: String, delivery_id: String) {
        let Some(state) = self.deliveries.get_mut(&delivery_id) else {
            let _ = reply.try_send(Err(Rejection::UnknownDelivery { delivery_id }));
            return;
        };
        match state.core.commit_command("delivery.cancel", Some(corr)) {
            Ok(_) => {
                let _ = reply.try_send(Ok(Receipt::Accepted));
            }
            Err(violation) => {
                state.core.record_violation(violation.clone());
                let state_now = state.core.state().to_string();
                let _ = reply.try_send(Err(illegal("delivery.cancel", &state_now, violation)));
            }
        }
    }

    fn latest_active(&self) -> Option<String> {
        self.order
            .iter()
            .rev()
            .find(|id| {
                self.deliveries
                    .get(*id)
                    .map(|state| {
                        matches!(
                            state.core.state(),
                            "Prepared" | "Revalidating" | "SubmittedUnconfirmed"
                        )
                    })
                    .unwrap_or(false)
            })
            .cloned()
    }
}

fn illegal(command: &str, state: &str, violation: crate::protocol::replay::Violation) -> Rejection {
    Rejection::IllegalInState {
        command: command.to_string(),
        state: state.to_string(),
        detail: violation.to_string(),
    }
}
