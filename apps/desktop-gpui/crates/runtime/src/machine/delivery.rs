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
//!
//! When an adapter reports the text around the insertion point, prepare
//! applies the insertion-boundary rules (`starling-processing`'s
//! `boundary` module) and delivers the result as a revision derived from
//! the requested one, registered through the document service
//! ([`DerivedRevisions`]); the requested revision itself is never edited.
//! Apply reads the surrounding text again and re-derives when it changed,
//! so a stale adjustment is never delivered. `delivery.prepare{boundary}`
//! turns the rules off: `raw` (the user's bypass) and `verbatim` (a
//! verbatim mode) deliver the requested text unchanged without reading
//! anything around the insertion point.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::bus::EventBus;
use crate::machine::{Inbound, MachineCore, Receipt, Rejection};
use crate::protocol::tables::DELIVERY;
use crate::protocol::{BoundaryPolicy, Command, Event, Revision};

use starling_processing::boundary::{self, BoundaryChange, BoundaryContext, BoundaryOptions};

use super::docs::{DerivedRevisions, RevisionRegistry};

/// The provenance of a revision produced by the insertion-boundary rules.
const BOUNDARY_PROVENANCE: &str = "insertion-boundary";

/// How often apply derives again when the boundary keeps moving while a
/// derivation is being registered, before it gives up.
const APPLY_DERIVE_ATTEMPTS: usize = 3;

/// The protocol's `msgId` length limit: a derived id stays nameable by a
/// later `delivery.prepare`.
const MSG_ID_MAX: usize = 128;

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

/// The text immediately around an insertion point.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SurroundingText {
    pub before: String,
    pub after: String,
    /// The field shows only its placeholder (Android's
    /// `isShowingHintText`): the text is hint text, and the field counts
    /// as empty.
    pub showing_hint: bool,
}

/// Why an adapter refused to read the text around an insertion point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProtectedField {
    /// A secure/password field.
    Secure,
    /// A field marked incognito (`IME_FLAG_NO_PERSONALIZED_LEARNING`).
    Incognito,
}

/// What an adapter could learn about the text around an insertion point.
/// Anything but [`SurroundingRead::Text`] delivers the text unchanged.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SurroundingRead {
    Text(SurroundingText),
    /// The platform does not report it for this target.
    Unsupported,
    /// A secure or incognito field: the adapter refused without reading.
    Protected(ProtectedField),
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
    /// The text around the insertion point, where the platform exposes it
    /// without extra permissions; `Unsupported` (the default) delivers the
    /// dictated text unchanged. Adapters must answer `Protected` for
    /// secure and incognito targets before reading anything. Called at
    /// prepare and again at apply (never for `raw`/`verbatim`
    /// deliveries). The text is used for the boundary decision only: never
    /// stored, never sent to a provider.
    fn surrounding_text(&self, _target_ref: &str) -> SurroundingRead {
        SurroundingRead::Unsupported
    }
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
    #[allow(dead_code)] // identifies the delivered revision; no v1 wire
                        // field carries it (the E21 documents product
                        // surface decides how it surfaces)
    revision_id: String,
    target_ref: String,
    compare_token: String,
    /// The revision's text, captured at prepare for the apply path.
    text: String,
    /// The requested revision and its document when the boundary rules
    /// apply to this delivery: apply re-derives from it against the
    /// boundary as it is then. `None` for raw and verbatim deliveries and
    /// for a revision that is itself derived.
    boundary_source: Option<(String, Revision)>,
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
    derived: DerivedRevisions,
    adapter: Arc<dyn DeliveryAdapter>,
}

impl DeliveryActor {
    pub fn new(
        inbox: crate::channel::Receiver<DeliveryMsg>,
        bus: Arc<EventBus>,
        view: super::ViewSlot,
        revisions: RevisionRegistry,
        derived: DerivedRevisions,
        adapter: Arc<dyn DeliveryAdapter>,
    ) -> DeliveryActor {
        DeliveryActor {
            inbox,
            bus,
            view,
            deliveries: HashMap::new(),
            order: Vec::new(),
            revisions,
            derived,
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
            self.publish_view();
        }
    }

    /// Called before every bus emit too, so a client reacting to an
    /// event never reads a snapshot behind it.
    fn publish_view(&self) {
        *self.view.lock().expect("delivery view lock") = self.snapshot_view();
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
                self.publish_view();
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
                boundary,
            } => self.handle_prepare(reply, corr, revision_id, target_ref, boundary),
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
        boundary: BoundaryPolicy,
    ) {
        // A delivery may only be prepared against a revision this runtime
        // knows (committed heads and derived revisions published by the
        // docs service).
        let source = {
            let revisions = self.revisions.lock().expect("revision registry lock");
            revisions.get(&revision_id).cloned()
        };
        let Some((doc_id, source)) = source else {
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
        // A derived revision is delivered exactly as recorded.
        let boundary_source = (boundary == BoundaryPolicy::Adjust
            && source.provenance != BOUNDARY_PROVENANCE)
            .then(|| (doc_id, source.clone()));
        let derived = match &boundary_source {
            Some((doc_id, source)) => {
                let surrounding = self.read_surrounding(&target_ref);
                self.boundary_revision(doc_id, source, surrounding.as_ref())
            }
            None => Ok(None),
        };
        let (revision_id, text) = match derived {
            Ok(Some(derived)) => (derived.rev_id, derived.text),
            Ok(None) => (revision_id, source.text),
            Err(rejection) => {
                let _ = reply.try_send(Err(rejection));
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
                        // Registered (and published) before the event, so
                        // the snapshot already shows the new delivery.
                        self.deliveries.insert(
                            delivery_id.clone(),
                            DeliveryState {
                                core,
                                revision_id,
                                target_ref,
                                compare_token: token,
                                text,
                                boundary_source,
                                fallback_requested: false,
                            },
                        );
                        self.order.push(delivery_id);
                        self.publish_view();
                        let _ = self.bus.emit(event, Some(&corr));
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

    /// The text around the insertion point; `None` when the target
    /// reports none or is a protected field. The caller holds it only for
    /// the boundary decision.
    fn read_surrounding(&self, target_ref: &str) -> Option<SurroundingText> {
        match self.adapter.surrounding_text(target_ref) {
            SurroundingRead::Text(surrounding) => Some(surrounding),
            SurroundingRead::Unsupported | SurroundingRead::Protected(_) => None,
        }
    }

    /// When the insertion-boundary rules change `source` at `surrounding`,
    /// registers the adjusted text as a revision derived from it through
    /// the document service. `None` delivers `source` unchanged: no text
    /// reported, a protected field, or no rule fired.
    fn boundary_revision(
        &self,
        doc_id: &str,
        source: &Revision,
        surrounding: Option<&SurroundingText>,
    ) -> Result<Option<Revision>, Rejection> {
        let Some(derived) = derive(source, surrounding) else {
            return Ok(None);
        };
        self.derived
            .record(doc_id, &source.rev_id, derived.clone())?;
        Ok(Some(derived))
    }

    /// What apply delivers under the boundary rules: `source` derived
    /// against the boundary as it is now — `source` itself when no rule
    /// fires any more (it is already recorded, so nothing waits and the
    /// revalidation just made still covers the target). A derivation
    /// prepare already recorded (`recorded`) is delivered as is;
    /// registering a new one waits on the document service, so the
    /// boundary is read again afterwards and the derivation repeated when
    /// it moved in the meantime (a busy document service is retried the
    /// same way). The flag says whether a registration waited (the target
    /// is then revalidated again); the error is a `delivery.failed` reason.
    fn derive_at_apply(
        &self,
        doc_id: &str,
        source: &Revision,
        target_ref: &str,
        recorded: &str,
    ) -> Result<(Revision, bool), &'static str> {
        let mut surrounding = self.read_surrounding(target_ref);
        let mut waited = false;
        let mut failure = "boundary_unstable";
        for _ in 0..APPLY_DERIVE_ATTEMPTS {
            let Some(derived) = derive(source, surrounding.as_ref()) else {
                return Ok((source.clone(), waited));
            };
            if derived.rev_id == recorded {
                return Ok((derived, waited));
            }
            waited = true;
            match self
                .derived
                .record(doc_id, &source.rev_id, derived.clone())
            {
                Ok(()) => {}
                Err(rejection @ Rejection::InboxFull) => {
                    failure = failure_reason(&rejection);
                    surrounding = self.read_surrounding(target_ref);
                    continue;
                }
                Err(rejection) => return Err(failure_reason(&rejection)),
            }
            let now = self.read_surrounding(target_ref);
            if now == surrounding {
                return Ok((derived, waited));
            }
            failure = "boundary_unstable";
            surrounding = now;
        }
        Err(failure)
    }

    fn handle_apply(&mut self, reply: super::ReceiptTx, corr: String, delivery_id: String) {
        let Some(state) = self.deliveries.get_mut(&delivery_id) else {
            let _ = reply.try_send(Err(Rejection::UnknownDelivery { delivery_id }));
            return;
        };
        let mut text = state.text.clone();
        let target_ref = state.target_ref.clone();
        let compare_token = state.compare_token.clone();
        let boundary_source = state.boundary_source.clone();
        let recorded = state.revision_id.clone();
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
                        // The boundary may have changed since prepare (the
                        // user kept typing): derive again from the
                        // requested revision, never deliver the stale text.
                        if let Some((doc_id, source)) = &boundary_source {
                            let (delivered, waited) =
                                match self.derive_at_apply(doc_id, source, &target_ref, &recorded) {
                                    Ok(derived) => derived,
                                    Err(reason) => {
                                        self.emit(
                                            &delivery_id,
                                            Event::DeliveryFailed {
                                                reason: reason.to_string(),
                                                fallback_suggested: true,
                                            },
                                            &corr,
                                        );
                                        return;
                                    }
                                };
                            // The target may have moved while the
                            // registration waited.
                            if waited {
                                if let Revalidation::Changed { expected, actual } =
                                    self.adapter.revalidate(&target_ref, &compare_token)
                                {
                                    self.emit(
                                        &delivery_id,
                                        Event::DeliveryConflict {
                                            expected_target: expected,
                                            actual_target: actual,
                                        },
                                        &corr,
                                    );
                                    return;
                                }
                            }
                            text = delivered.text;
                            if let Some(state) = self.deliveries.get_mut(&delivery_id) {
                                state.revision_id = delivered.rev_id;
                                state.text = text.clone();
                            }
                        }
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

/// The revision the insertion-boundary rules make of `source` at
/// `surrounding`; `None` when no text was reported or no rule fired.
fn derive(source: &Revision, surrounding: Option<&SurroundingText>) -> Option<Revision> {
    let surrounding = surrounding?;
    let context = BoundaryContext {
        before: &surrounding.before,
        after: &surrounding.after,
        showing_hint: surrounding.showing_hint,
    };
    let adjustment = boundary::adjust(&source.text, &context, &BoundaryOptions::default());
    if adjustment.is_unchanged() {
        return None;
    }
    // The fired rules determine the adjusted text, so naming the id after
    // them never maps one id to two different texts.
    let rules: Vec<&str> = adjustment
        .changes
        .iter()
        .map(|change| match change {
            BoundaryChange::LeadingSpace => "space",
            BoundaryChange::FirstLetterCase => "case",
        })
        .collect();
    Some(Revision {
        rev_id: derived_id(&source.rev_id, &rules.join("-")),
        text: adjustment.text,
        provenance: BOUNDARY_PROVENANCE.to_string(),
        ..source.clone()
    })
}

/// `{source}:boundary-{rules}` (`:` keeps it a wire-legal `msgId`). An
/// id over [`MSG_ID_MAX`] is shortened and tagged with a digest of the
/// full id, so it always stays within the limit and one source and rule
/// list still map to one id. The suffix is kept whole when it fits next
/// to the tag (any realistic rule list, pinned by the tests below).
fn derived_id(source_id: &str, rules: &str) -> String {
    let id = format!("{source_id}:boundary-{rules}");
    if id.len() <= MSG_ID_MAX {
        return id;
    }
    // 64-bit FNV-1a: an id tag, not a security boundary (a collision is
    // still refused as `RevisionIdTaken`).
    let digest = id.bytes().fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
    });
    let tag = format!(".{digest:016x}");
    let prefix = |text: &str, mut keep: usize| {
        while !text.is_char_boundary(keep) {
            keep -= 1;
        }
        text[..keep].to_string()
    };
    let suffix = &id[source_id.len()..];
    match MSG_ID_MAX.checked_sub(suffix.len() + tag.len()) {
        Some(keep) => format!("{}{tag}{suffix}", prefix(source_id, keep)),
        None => format!("{}{tag}", prefix(&id, MSG_ID_MAX.saturating_sub(tag.len()))),
    }
}

/// `delivery.failed.reason` for a derived revision apply could not record.
fn failure_reason(rejection: &Rejection) -> &'static str {
    match rejection {
        Rejection::RevisionIdTaken { .. } => "revision_id_taken",
        Rejection::InboxFull => "document_service_busy",
        _ => "derived_revision_unrecorded",
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
    use crate::protocol::is_msg_id;

    #[test]
    fn derived_ids_stay_wire_legal() {
        assert_eq!(
            derived_id("rev-1", "space-case"),
            "rev-1:boundary-space-case"
        );
        let long = "r".repeat(MSG_ID_MAX);
        let near = format!("{}x", "r".repeat(MSG_ID_MAX - 1));
        let (a, b) = (
            derived_id(&long, "space-case"),
            derived_id(&near, "space-case"),
        );
        for id in [&a, &b] {
            assert!(id.len() <= MSG_ID_MAX && is_msg_id(id), "{id}");
            assert!(id.ends_with(":boundary-space-case"), "{id}");
        }
        // Same shortened prefix, different sources: different ids.
        assert_ne!(a, b);
        assert_eq!(a, derived_id(&long, "space-case"));
        // A rule list longer than the limit itself still yields a legal,
        // distinct id.
        let (rules, other) = ("case-".repeat(MSG_ID_MAX), "space-".repeat(MSG_ID_MAX));
        let (c, d) = (derived_id("rev-1", &rules), derived_id("rev-1", &other));
        for id in [&c, &d] {
            assert!(id.len() <= MSG_ID_MAX && is_msg_id(id), "{id}");
        }
        assert_ne!(c, d);
    }
}
