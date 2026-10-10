//! The runtime's [`DeliveryAdapter`] over an [`Inserter`] (feature
//! `runtime`). The host wires it with
//! `RuntimeConfig::with_delivery_adapter`.
//!
//! - `prepare` only parses the ref and requires its backend; the binding
//!   target check happens inside `insert`, immediately before typing. The
//!   compare token is a digest of the ref, so a token/ref mismatch
//!   conflicts at apply without a live check.
//! - A closed window revalidates as a conflict with `actual = "gone"`.
//! - A backend that errors while revalidating reports `Unchanged`, so the
//!   real error surfaces from `insert` as `delivery.failed` instead of as a
//!   made-up target change.
//! - The seam has no conflict channel out of `insert`: a target change
//!   detected there (including part-way through typing) becomes
//!   `delivery.failed{reason: target_changed | partial_delivery}`.

use std::sync::Arc;

use starling_runtime::machine::delivery::{
    DeliveryAdapter, InsertEvidence, InsertionFailure, Revalidation,
};

use crate::{Inserter, InsertionBackend, TargetCheck, TargetSnapshot};

pub struct InsertionDeliveryAdapter {
    inserter: Arc<Inserter>,
}

impl InsertionDeliveryAdapter {
    pub fn new(inserter: Arc<Inserter>) -> Arc<Self> {
        Arc::new(InsertionDeliveryAdapter { inserter })
    }

    /// The snapshot and backend for a ref, or the safeToken saying why
    /// there is none.
    fn resolve(
        &self,
        target_ref: &str,
    ) -> Result<(TargetSnapshot, &dyn InsertionBackend), &'static str> {
        let snapshot = self
            .inserter
            .parse_target_ref(target_ref)
            .ok_or("unparsable_target_ref")?;
        let backend = self
            .inserter
            .backend_for(&snapshot)
            .ok_or("no_backend_for_ref")?;
        Ok((snapshot, backend))
    }
}

/// `ins-v1:` plus a 64-bit FNV-1a digest of the ref. An integrity check
/// of which ref a delivery was prepared against, not a security boundary.
fn compare_token(target_ref: &str) -> String {
    let digest = target_ref
        .bytes()
        .fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
        });
    format!("ins-v1:{digest:016x}")
}

impl DeliveryAdapter for InsertionDeliveryAdapter {
    fn prepare(&self, target_ref: &str) -> Result<String, String> {
        self.resolve(target_ref)
            .map_err(|reason| format!("cannot deliver to target ref {target_ref:?}: {reason}"))?;
        Ok(compare_token(target_ref))
    }

    fn revalidate(&self, target_ref: &str, token: &str) -> Revalidation {
        let changed = |actual: &str| Revalidation::Changed {
            expected: target_ref.to_string(),
            actual: actual.to_string(),
        };
        if token != compare_token(target_ref) {
            return changed("compare_token_mismatch");
        }
        let (snapshot, backend) = match self.resolve(target_ref) {
            Ok(resolved) => resolved,
            Err(reason) => return changed(reason),
        };
        match backend.revalidate(&snapshot) {
            Ok(TargetCheck::Same) | Err(_) => Revalidation::Unchanged,
            Ok(TargetCheck::Changed { expected, actual }) => {
                Revalidation::Changed { expected, actual }
            }
            Ok(TargetCheck::Gone) => changed("gone"),
        }
    }

    fn insert(
        &self,
        _delivery_id: &str,
        target_ref: &str,
        text: &str,
    ) -> Result<InsertEvidence, InsertionFailure> {
        // Every failure leaves the transcript undelivered, so the copy
        // fallback is always offered.
        let failure = |reason: &str| InsertionFailure {
            reason: reason.to_string(),
            fallback_suggested: true,
        };
        let (snapshot, backend) = self.resolve(target_ref).map_err(failure)?;
        let receipt = backend
            .insert(&snapshot, text)
            .map_err(|error| failure(error.code()))?;
        Ok(InsertEvidence {
            level: receipt.evidence.to_string(),
        })
    }

    fn describe(&self) -> String {
        format!("starling-insertion ({})", self.inserter.describe())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{FakeBackend, FakeTarget, InsertBehavior};
    use crate::InsertError;

    fn session() -> (Arc<FakeBackend>, Arc<InsertionDeliveryAdapter>) {
        let fake = Arc::new(FakeBackend::new());
        fake.focus(FakeTarget::named("Notes", "Meeting notes"));
        let inserter = Inserter::with_backends(vec![Box::new(fake.clone())]);
        (fake, InsertionDeliveryAdapter::new(Arc::new(inserter)))
    }

    fn failure_reason(result: Result<InsertEvidence, InsertionFailure>) -> String {
        match result {
            Err(failure) => {
                assert!(failure.fallback_suggested);
                failure.reason
            }
            Ok(evidence) => panic!("expected a failure, got {evidence:?}"),
        }
    }

    #[test]
    fn prepare_mints_a_versioned_token_and_rejects_foreign_refs() {
        let (fake, adapter) = session();
        let target = fake.capture().unwrap();
        let token = adapter.prepare(&target.target_ref).unwrap();
        assert!(token.starts_with("ins-v1:"), "{token}");
        assert_ne!(token, compare_token("fake:9a:9a:7"));
        // No X11 backend in this session, and garbage.
        assert!(adapter.prepare("x11:1:2:3").is_err());
        assert!(adapter.prepare("not-a-ref").is_err());
    }

    #[test]
    fn revalidate_maps_every_target_check() {
        let (fake, adapter) = session();
        let target = fake.capture().unwrap();
        let token = adapter.prepare(&target.target_ref).unwrap();
        assert_eq!(
            adapter.revalidate(&target.target_ref, &token),
            Revalidation::Unchanged
        );

        // A token prepared against a different ref.
        assert_eq!(
            adapter.revalidate("fake:9a:9a:7", &token),
            Revalidation::Changed {
                expected: "fake:9a:9a:7".to_string(),
                actual: "compare_token_mismatch".to_string(),
            }
        );

        fake.focus(FakeTarget::named("Browser", "A tab"));
        let browser = fake.capture().unwrap();
        assert_eq!(
            adapter.revalidate(&target.target_ref, &token),
            Revalidation::Changed {
                expected: target.target_ref.clone(),
                actual: browser.target_ref.clone(),
            }
        );

        let browser_token = adapter.prepare(&browser.target_ref).unwrap();
        fake.destroy_target();
        assert_eq!(
            adapter.revalidate(&browser.target_ref, &browser_token),
            Revalidation::Changed {
                expected: browser.target_ref.clone(),
                actual: "gone".to_string(),
            }
        );
    }

    #[test]
    fn insert_maps_errors_to_their_codes_and_types_nothing() {
        let (fake, adapter) = session();
        let target = fake.capture().unwrap();

        fake.set_insert_behavior(InsertBehavior::FailWith(InsertError::Rejected {
            reason: "blocked".to_string(),
        }));
        assert_eq!(
            failure_reason(adapter.insert("dlv-1", &target.target_ref, "hello")),
            "insertion_rejected"
        );

        fake.set_insert_behavior(InsertBehavior::Type);
        for forbidden in ["line\nbreak", "tab\there", "carriage\rreturn", "nul\0byte"] {
            assert_eq!(
                failure_reason(adapter.insert("dlv-1", &target.target_ref, forbidden)),
                "multiline_unsupported",
                "{forbidden:?}"
            );
        }

        fake.focus(FakeTarget::named("Browser", "A tab"));
        assert_eq!(
            failure_reason(adapter.insert("dlv-1", &target.target_ref, "hello")),
            "target_changed"
        );

        let own = format!("fake:1:1:{}", std::process::id());
        assert_eq!(
            failure_reason(adapter.insert("dlv-1", &own, "hello")),
            "target_is_starling"
        );
        assert!(fake.insertions().is_empty());
    }

    #[test]
    fn a_revalidation_error_is_not_a_conflict_and_still_blocks_insert() {
        let (fake, adapter) = session();
        let target = fake.capture().unwrap();
        let token = adapter.prepare(&target.target_ref).unwrap();
        fake.fail_revalidate(Some(InsertError::Unavailable {
            reason: "connection reset".to_string(),
        }));
        assert_eq!(
            adapter.revalidate(&target.target_ref, &token),
            Revalidation::Unchanged
        );
        assert_eq!(
            failure_reason(adapter.insert("dlv-1", &target.target_ref, "hello")),
            "insertion_unavailable"
        );
        assert!(fake.insertions().is_empty());
    }

    /// End to end through the real runtime's delivery actor.
    #[test]
    fn the_runtime_confirms_a_delivery_over_this_adapter() {
        use starling_runtime::bus::EventSub;
        use starling_runtime::channel::RecvError;
        use starling_runtime::protocol::{Command, Event, Revision};
        use starling_runtime::{Runtime, RuntimeConfig};

        fn until(events: &EventSub, wanted: &str) -> Event {
            loop {
                match events.recv_timeout(std::time::Duration::from_millis(50)) {
                    Ok(message) if message.event.type_name() == wanted => return message.event,
                    Ok(_) => continue,
                    Err(RecvError::Timeout) => panic!("timed out waiting for {wanted}"),
                    Err(other) => panic!("event stream error: {other:?}"),
                }
            }
        }

        let (fake, adapter) = session();
        let (_runtime, client) =
            Runtime::start(RuntimeConfig::default().with_delivery_adapter(adapter));
        let events = client.subscribe();

        client
            .send(
                Some("doc-1"),
                Command::DocsUpdateHead {
                    doc_id: "notes".into(),
                    expected_base: 0,
                    new_revision: Revision {
                        rev_id: "rev-1".to_string(),
                        base_revision: 0,
                        source_attempt_ids: vec![],
                        instruction_template_id: "tpl-none".to_string(),
                        text: "Café, bitte.".to_string(),
                        status: "candidate".to_string(),
                        provenance: "recognition".to_string(),
                    },
                },
            )
            .unwrap();
        until(&events, "docs.headUpdated");

        let target = fake.capture().unwrap();
        client
            .send(
                Some("dlv-1"),
                Command::DeliveryPrepare {
                    revision_id: "rev-1".into(),
                    target_ref: target.target_ref.clone(),
                    boundary: Default::default(),
                },
            )
            .unwrap();
        let Event::DeliveryPrepared { delivery_id, .. } = until(&events, "delivery.prepared")
        else {
            unreachable!()
        };

        client
            .send(Some("dlv-2"), Command::DeliveryApply { delivery_id })
            .unwrap();
        until(&events, "delivery.submittedUnconfirmed");
        let Event::DeliveryConfirmed { evidence_level } = until(&events, "delivery.confirmed")
        else {
            unreachable!()
        };
        assert_eq!(evidence_level, "synthetic_keys_sent");
        assert_eq!(
            fake.insertions(),
            vec![(target.target_ref, "Café, bitte.".to_string())]
        );
    }
}
