//! The bridge from this crate's backends to the runtime's delivery
//! seam (feature `runtime`, issue #221 slice 2): one
//! [`DeliveryAdapter`] implementation over an [`Inserter`].
//!
//! Wiring is #220's call, deliberately not done here: the host
//! constructs its session's `Inserter`, wraps it, and passes the
//! adapter through
//! `RuntimeConfig::with_delivery_adapter(adapter)`. Mode B keeps the
//! process split honest: the app captures a `target_ref` when a take
//! starts, the host's delivery actor prepares/applies against it, and
//! refs are process-independent strings exactly so that works.
//!
//! Mapping choices (each keeps `delivery.*` events truthful):
//!
//! - `prepare` parses the ref and requires the owning backend to be
//!   present in this host's inserter — that is freeze-time
//!   *bookkeeping*, not a live probe; the binding target check is the
//!   contract's "revalidate immediately before apply", which the
//!   backends perform inside `insert` itself. The compare token is a
//!   digest of the ref (`ins-v1:<hex>`), so a token/ref mismatch is
//!   detectable at apply even before the live check runs.
//! - `Revalidation::Changed` carries the live ref as `actualTarget`;
//!   a destroyed window reports `actual = "gone"` — v1's conflict
//!   payload has no separate "closed" shape, and `"gone"` says what
//!   happened in one safeToken.
//! - A backend that *errors* while revalidating (dead connection at
//!   apply time) reports `Unchanged` rather than a fake conflict, so
//!   the failure surfaces from `insert` as `delivery.failed{reason}`
//!   with the honest code (`insertion_unavailable`), not as a
//!   target-change story the user would misread.
//! - A conflict detected *inside* `insert` maps to
//!   `delivery.failed{reason: target_changed}`, **not** to
//!   `delivery.conflict`: the `DeliveryAdapter` seam has no conflict
//!   channel out of `insert` (its result is `InsertEvidence` or
//!   `InsertionFailure`), and inventing one is deliberately left to a
//!   seam extension tracked by #220. The same applies to a target
//!   that changes *part-way through* typing — the backend stops
//!   immediately and the bridge reports
//!   `delivery.failed{reason: partial_delivery}` (the counts live in
//!   the error's human message, which the wire does not carry; only
//!   the safeToken code crosses the seam).
//! - `insert` failures map `InsertError::code()` straight to
//!   `delivery.failed{reason}` (safeToken-shaped by construction) and
//!   `fallback_suggested()` to `fallbackSuggested`; success passes
//!   the receipt's evidence level through unchanged — a typing backend
//!   can therefore only ever confirm with `synthetic_keys_sent`.

use std::sync::Arc;

use starling_runtime::machine::delivery::{
    DeliveryAdapter, InsertEvidence, InsertionFailure, Revalidation,
};

use crate::{fnv1a64, Inserter, TargetCheck};

/// The `DeliveryAdapter` over an [`Inserter`]. Construct through
/// [`Self::new`] (which returns the `Arc` the runtime config wants).
pub struct InsertionDeliveryAdapter {
    inserter: Arc<Inserter>,
}

impl InsertionDeliveryAdapter {
    pub fn new(inserter: Arc<Inserter>) -> Arc<Self> {
        Arc::new(InsertionDeliveryAdapter { inserter })
    }
}

/// The compare token for a prepared ref: `ins-v1:` plus a 64-bit
/// digest of the ref. Version-tagged so a future token scheme never
/// collides with one of these, and safeToken-clean end to end.
fn compare_token(target_ref: &str) -> String {
    format!("ins-v1:{:016x}", fnv1a64(target_ref.as_bytes()))
}

impl DeliveryAdapter for InsertionDeliveryAdapter {
    fn prepare(&self, target_ref: &str) -> Result<String, String> {
        // Self-describing refs parse without a server round trip; a
        // ref that does not parse here never will, and a ref whose
        // backend is absent from this session cannot be inserted by
        // this host — both refuse at prepare, before any delivery
        // exists to strand.
        let snapshot = self
            .inserter
            .parse_target_ref(target_ref)
            .ok_or_else(|| format!("unknown or malformed target ref: {target_ref}"))?;
        if self.inserter.backend_for(&snapshot).is_none() {
            return Err(format!(
                "this session has no insertion backend for the {} ref scheme",
                snapshot.backend.scheme()
            ));
        }
        Ok(compare_token(target_ref))
    }

    fn revalidate(&self, target_ref: &str, token: &str) -> Revalidation {
        // Integrity first: a token that does not digest this ref means
        // the delivery was prepared against a different target —
        // conflict, and nothing gets typed. The actual side can only
        // say what happened (the token carries no reversible ref).
        if token != compare_token(target_ref) {
            return Revalidation::Changed {
                expected: target_ref.to_string(),
                actual: "compare_token_mismatch".to_string(),
            };
        }
        let Some(snapshot) = self.inserter.parse_target_ref(target_ref) else {
            return Revalidation::Changed {
                expected: target_ref.to_string(),
                actual: "unparsable_target_ref".to_string(),
            };
        };
        let Some(backend) = self.inserter.backend_for(&snapshot) else {
            return Revalidation::Changed {
                expected: target_ref.to_string(),
                actual: "no_backend_for_ref".to_string(),
            };
        };
        match backend.revalidate(&snapshot) {
            Ok(TargetCheck::Same) => Revalidation::Unchanged,
            Ok(TargetCheck::Changed { expected, actual }) => {
                Revalidation::Changed { expected, actual }
            }
            Ok(TargetCheck::Gone) => Revalidation::Changed {
                expected: target_ref.to_string(),
                actual: "gone".to_string(),
            },
            // A backend that cannot even ask (connection died) has no
            // target-change story to tell; `insert` fails honestly a
            // moment later with the real reason (module docs).
            Err(_) => Revalidation::Unchanged,
        }
    }

    fn insert(
        &self,
        _delivery_id: &str,
        target_ref: &str,
        text: &str,
    ) -> Result<InsertEvidence, InsertionFailure> {
        let Some(snapshot) = self.inserter.parse_target_ref(target_ref) else {
            return Err(InsertionFailure {
                reason: "unparsable_target_ref".to_string(),
                fallback_suggested: true,
            });
        };
        let Some(backend) = self.inserter.backend_for(&snapshot) else {
            return Err(InsertionFailure {
                reason: "no_backend_for_ref".to_string(),
                fallback_suggested: true,
            });
        };
        match backend.insert(&snapshot, text) {
            Ok(receipt) => Ok(InsertEvidence {
                level: receipt.evidence.to_string(),
            }),
            Err(error) => Err(InsertionFailure {
                reason: error.code().to_string(),
                fallback_suggested: error.fallback_suggested(),
            }),
        }
    }

    fn describe(&self) -> String {
        format!("starling-insertion ({})", self.inserter.describe())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{FakeBackend, FakeTarget, InsertBehavior};
    use crate::{BackendKind, Inserter, InsertionBackend, TargetSnapshot};

    fn session() -> (Arc<FakeBackend>, Arc<InsertionDeliveryAdapter>) {
        let fake = Arc::new(FakeBackend::new());
        fake.focus(FakeTarget::named("Notes", "Meeting notes"));
        let adapter =
            InsertionDeliveryAdapter::new(Arc::new(Inserter::with_backends(vec![Box::new(
                fake.clone(),
            )])));
        (fake, adapter)
    }

    #[test]
    fn prepare_freezes_a_digest_token_and_rejects_foreign_refs() {
        let (_fake, adapter) = session();
        // A fake-scheme ref the inserter can serve.
        let target = adapter.inserter.capture().expect("focus is set");
        let token = adapter.prepare(&target.target_ref).expect("prepares");
        assert!(
            token.starts_with("ins-v1:"),
            "token is version-tagged: {token}"
        );
        // ... but a ref from a backend this session does not run, and
        // plain garbage, both refuse — before any delivery exists.
        assert!(adapter.prepare("x11:1:2:3").is_err());
        assert!(adapter.prepare("not-a-ref").is_err());
        assert!(adapter.prepare("").is_err());
    }

    #[test]
    fn revalidate_reports_unchanged_while_the_same_target_stays_focused() {
        let (fake, adapter) = session();
        let target = fake.capture().expect("focus is set");
        let token = adapter.prepare(&target.target_ref).expect("prepares");
        assert_eq!(
            adapter.revalidate(&target.target_ref, &token),
            Revalidation::Unchanged
        );
    }

    #[test]
    fn revalidate_conflicts_when_focus_moved_to_another_target() {
        let (fake, adapter) = session();
        let target = fake.capture().expect("focus is set");
        let token = adapter.prepare(&target.target_ref).expect("prepares");
        // The user tabs to a browser mid-take.
        fake.focus(FakeTarget::named("Browser", "A tab"));
        match adapter.revalidate(&target.target_ref, &token) {
            Revalidation::Changed { expected, actual } => {
                assert_eq!(expected, target.target_ref);
                assert_ne!(actual, target.target_ref);
                assert!(actual.starts_with("fake:"), "actual names the new target");
            }
            other => panic!("focus moved, so the check must conflict: {other:?}"),
        }
    }

    #[test]
    fn revalidate_conflicts_with_gone_when_the_window_closed() {
        let (fake, adapter) = session();
        let target = fake.capture().expect("focus is set");
        let token = adapter.prepare(&target.target_ref).expect("prepares");
        fake.destroy_target();
        assert_eq!(
            adapter.revalidate(&target.target_ref, &token),
            Revalidation::Changed {
                expected: target.target_ref.clone(),
                actual: "gone".to_string(),
            }
        );
    }

    #[test]
    fn revalidate_flags_a_token_prepared_against_a_different_ref() {
        let (fake, adapter) = session();
        let first = fake.capture().expect("focus is set");
        let token = adapter.prepare(&first.target_ref).expect("prepares");
        // A stale token aimed at a different ref: conflict, nothing
        // typed (the integrity check precedes the live probe).
        assert_eq!(
            adapter.revalidate("fake:9a:9a:7", &token),
            Revalidation::Changed {
                expected: "fake:9a:9a:7".to_string(),
                actual: "compare_token_mismatch".to_string(),
            }
        );
    }

    #[test]
    fn insert_passes_the_synthetic_keys_evidence_level_through() {
        let (fake, adapter) = session();
        let target = fake.capture().expect("focus is set");
        let evidence = adapter
            .insert("dlv-1", &target.target_ref, "Café, bitte.")
            .expect("the fake types");
        assert_eq!(evidence.level, "synthetic_keys_sent");
        assert_eq!(
            fake.insertions(),
            vec![(target.target_ref.clone(), "Café, bitte.".to_string())]
        );
    }

    #[test]
    fn insert_maps_errors_to_safe_token_reasons_and_the_fallback_flag() {
        let (fake, adapter) = session();
        let target = fake.capture().expect("focus is set");

        // Scripted target-level refusal.
        fake.set_insert_behavior(InsertBehavior::FailWith(
            crate::InsertError::PermissionDenied {
                reason: "elevated target".to_string(),
                settings_hint: "same integrity level".to_string(),
            },
        ));
        match adapter.insert("dlv-1", &target.target_ref, "hello") {
            Err(InsertionFailure {
                reason,
                fallback_suggested,
            }) => {
                assert_eq!(reason, "insertion_permission_denied");
                assert!(fallback_suggested);
            }
            other => panic!("scripted failure must surface: {other:?}"),
        }

        // And the no-enter rule: control characters never reach the
        // backend (the guards refuse first, so nothing is recorded).
        fake.set_insert_behavior(InsertBehavior::Type);
        for forbidden in ["line\nbreak", "tab\there", "carriage\rreturn", "nul\0byte"] {
            match adapter.insert("dlv-1", &target.target_ref, forbidden) {
                Err(InsertionFailure { reason, .. }) => {
                    assert_eq!(reason, "multiline_unsupported", "text: {forbidden:?}");
                }
                other => panic!("{forbidden:?} must be refused: {other:?}"),
            }
        }
        assert!(
            fake.insertions().is_empty(),
            "no refused text may reach a target"
        );
    }

    #[test]
    fn insert_conflicts_instead_of_typing_when_the_target_changed() {
        let (fake, adapter) = session();
        let target = fake.capture().expect("focus is set");
        fake.focus(FakeTarget::named("Browser", "A tab"));
        // The bridge's insert delegates to the backend, whose own
        // guards revalidate first — the changed target refuses there.
        // Note *what* crosses the seam: a delivery.failure with reason
        // `target_changed`, not a delivery.conflict — the
        // DeliveryAdapter seam cannot carry a conflict out of `insert`
        // (see the module docs; the seam extension is #220's to make).
        match adapter.insert("dlv-1", &target.target_ref, "hello") {
            Err(InsertionFailure { reason, .. }) => assert_eq!(reason, "target_changed"),
            other => panic!("a changed target must refuse: {other:?}"),
        }
        assert!(fake.insertions().is_empty());
    }

    #[test]
    fn insert_never_types_when_revalidation_persists_erroring() {
        let (fake, adapter) = session();
        let target = fake.capture().expect("focus is set");
        // A backend whose revalidate keeps erroring (a dying
        // connection, say): the bridge reports Unchanged from
        // `revalidate` (no fake conflict), `insert` fails with the
        // honest reason, and nothing is ever typed — the guard cannot
        // be waited out.
        fake.fail_revalidate(
            crate::InsertError::Unavailable {
                reason: "connection reset".to_string(),
                setup_hint: None,
            },
            None,
        );
        let token = adapter.prepare(&target.target_ref).expect("prepares");
        assert_eq!(
            adapter.revalidate(&target.target_ref, &token),
            Revalidation::Unchanged,
            "an erroring check is not a conflict"
        );
        for attempt in 1..=3 {
            match adapter.insert("dlv-1", &target.target_ref, "hello") {
                Err(InsertionFailure {
                    reason,
                    fallback_suggested,
                }) => {
                    assert_eq!(reason, "insertion_unavailable", "attempt {attempt}");
                    assert!(fallback_suggested);
                }
                other => panic!("a persistently erroring check must refuse: {other:?}"),
            }
        }
        assert!(
            fake.insertions().is_empty(),
            "nothing may be typed while the check cannot answer"
        );
    }

    #[test]
    fn insert_recovers_once_a_transient_revalidate_error_clears() {
        let (fake, adapter) = session();
        let target = fake.capture().expect("focus is set");
        // The check errors exactly once (a blip), then answers again:
        // the first insert refuses, the next one types in full.
        fake.fail_revalidate(
            crate::InsertError::Unavailable {
                reason: "connection blip".to_string(),
                setup_hint: None,
            },
            Some(1),
        );
        match adapter.insert("dlv-1", &target.target_ref, "hello") {
            Err(InsertionFailure { reason, .. }) => {
                assert_eq!(reason, "insertion_unavailable")
            }
            other => panic!("the blip must surface: {other:?}"),
        }
        assert!(fake.insertions().is_empty());
        let evidence = adapter
            .insert("dlv-2", &target.target_ref, "hello")
            .expect("the transient error cleared");
        assert_eq!(evidence.level, "synthetic_keys_sent");
        assert_eq!(
            fake.insertions(),
            vec![(target.target_ref.clone(), "hello".to_string())]
        );
    }

    #[test]
    fn insert_refuses_a_starling_owned_target_with_its_own_code() {
        let fake = Arc::new(FakeBackend::new());
        fake.focus(FakeTarget::owned_by_this_process());
        let adapter =
            InsertionDeliveryAdapter::new(Arc::new(Inserter::with_backends(vec![Box::new(
                fake.clone(),
            )])));
        // Capture itself refuses, so drive insert with a hand-built
        // snapshot carrying our pid — the mode-B shape (a ref captured
        // by the app, typed by the host, both being Starling here).
        let target = TargetSnapshot {
            backend: BackendKind::Fake,
            target_ref: format!("fake:1:1:{}", std::process::id()),
            app: None,
            title: None,
            pid: Some(std::process::id()),
            capabilities: BackendKind::Fake.capabilities(),
        };
        match adapter.insert("dlv-1", &target.target_ref, "hello") {
            Err(InsertionFailure { reason, .. }) => assert_eq!(reason, "target_is_starling"),
            other => panic!("Starling must not type into itself: {other:?}"),
        }
        assert!(fake.insertions().is_empty());
    }

    /// End to end through the real runtime: the delivery actor, the
    /// event stream, and this adapter — the wiring #220 owns, proven
    /// here once so phase A ships a bridge known to work under the
    /// actual machine. Mirrors `delivery_lifecycle.rs` in
    /// starling-runtime.
    #[test]
    fn the_runtime_emits_the_honest_delivery_events_over_this_adapter() {
        use starling_runtime::bus::EventSub;
        use starling_runtime::channel::RecvError;
        use starling_runtime::protocol::{Command, Event, Revision};
        use starling_runtime::{Runtime, RuntimeConfig};

        fn until(events: &EventSub, wanted: &str) -> Event {
            loop {
                match events.recv_timeout(std::time::Duration::from_millis(50)) {
                    Ok(message) if message.event.type_name() == wanted => return message.event,
                    Ok(_) => continue,
                    Err(RecvError::Timeout) => {
                        panic!("timed out waiting for {wanted} (the live actor must emit it)")
                    }
                    Err(other) => panic!("event stream error: {other:?}"),
                }
            }
        }

        fn revision(rev_id: &str, text: &str) -> Revision {
            Revision {
                rev_id: rev_id.to_string(),
                base_revision: 0,
                source_attempt_ids: vec![],
                instruction_template_id: "tpl-none".to_string(),
                text: text.to_string(),
                status: "candidate".to_string(),
                provenance: "recognition".to_string(),
            }
        }

        let (fake, adapter) = session();
        let config = RuntimeConfig::default().with_delivery_adapter(adapter);
        let (_runtime, client) = Runtime::start(config);
        let events = client.subscribe();

        client
            .send(
                Some("doc-1"),
                Command::DocsUpdateHead {
                    doc_id: "notes".into(),
                    expected_base: 0,
                    new_revision: revision("rev-1", "Café, bitte."),
                },
            )
            .expect("updateHead accepted");
        until(&events, "docs.headUpdated");

        let target = fake.capture().expect("focus is set");
        client
            .send(
                Some("dlv-1"),
                Command::DeliveryPrepare {
                    revision_id: "rev-1".into(),
                    target_ref: target.target_ref.clone(),
                },
            )
            .expect("prepare accepted");
        let delivery_id = match until(&events, "delivery.prepared") {
            Event::DeliveryPrepared { delivery_id, .. } => delivery_id,
            other => panic!("prepared carries the delivery id: {other:?}"),
        };

        client
            .send(Some("dlv-2"), Command::DeliveryApply { delivery_id })
            .expect("apply accepted");
        until(&events, "delivery.submittedUnconfirmed");
        match until(&events, "delivery.confirmed") {
            Event::DeliveryConfirmed { evidence_level } => {
                assert_eq!(evidence_level, "synthetic_keys_sent");
            }
            other => panic!("confirmed carries the evidence level: {other:?}"),
        }
        assert_eq!(
            fake.insertions(),
            vec![(target.target_ref.clone(), "Café, bitte.".to_string())]
        );
    }
}
