//! The insertion-boundary wiring in the delivery machine, pinned through
//! the runtime protocol: an adapter that reports surrounding text gets the
//! boundary rules applied at prepare, the adjusted text is delivered as a
//! derived revision, and the requested revision is never edited.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use starling_runtime::bus::EventSub;
use starling_runtime::channel::RecvError;
use starling_runtime::machine::delivery::{
    DeliveryAdapter, InsertEvidence, InsertionFailure, Revalidation, SurroundingText,
};
use starling_runtime::protocol::{Command, Event, Revision};
use starling_runtime::{Runtime, RuntimeClient, RuntimeConfig};

/// Reports surrounding text per target: `mid-sentence`, `after-period` and
/// `field-start` have it, every other target does not.
#[derive(Default)]
struct RecordingAdapter {
    inserted: Mutex<Vec<String>>,
}

impl DeliveryAdapter for RecordingAdapter {
    fn prepare(&self, _target_ref: &str) -> Result<String, String> {
        Ok("token".to_string())
    }

    fn revalidate(&self, _target_ref: &str, _compare_token: &str) -> Revalidation {
        Revalidation::Unchanged
    }

    fn insert(
        &self,
        _delivery_id: &str,
        _target_ref: &str,
        text: &str,
    ) -> Result<InsertEvidence, InsertionFailure> {
        self.inserted.lock().unwrap().push(text.to_string());
        Ok(InsertEvidence {
            level: "recorded".to_string(),
        })
    }

    fn describe(&self) -> String {
        "recording adapter (tests)".to_string()
    }

    fn surrounding_text(&self, target_ref: &str) -> Option<SurroundingText> {
        let before = match target_ref {
            "mid-sentence" => "The quick brown",
            "after-period" => "Done.",
            "field-start" => "",
            _ => return None,
        };
        Some(SurroundingText {
            before: before.to_string(),
            after: String::new(),
        })
    }
}

fn until(events: &EventSub, wanted: &str) -> Event {
    let started = Instant::now();
    loop {
        match events.recv_timeout(Duration::from_millis(20)) {
            Ok(message) if message.event.type_name() == wanted => return message.event,
            Ok(_) => {}
            Err(RecvError::Timeout) if started.elapsed() > Duration::from_secs(5) => {
                panic!("timed out waiting for {wanted}")
            }
            Err(RecvError::Timeout) => {}
            Err(other) => panic!("event stream error: {other:?}"),
        }
    }
}

/// Prepares and applies `rev` against `target`; returns the inserted text.
fn prepare_and_apply(
    client: &RuntimeClient,
    events: &EventSub,
    adapter: &RecordingAdapter,
    rev: &str,
    target: &str,
) -> String {
    client
        .send(
            Some("dlv"),
            Command::DeliveryPrepare {
                revision_id: rev.to_string(),
                target_ref: target.to_string(),
            },
        )
        .unwrap_or_else(|err| panic!("prepare {rev}: {err:?}"));
    let Event::DeliveryPrepared { delivery_id, .. } = until(events, "delivery.prepared") else {
        unreachable!()
    };
    client
        .send(Some("dlv"), Command::DeliveryApply { delivery_id })
        .expect("apply accepted");
    until(events, "delivery.confirmed");
    adapter.inserted.lock().unwrap().last().cloned().unwrap()
}

#[test]
fn boundary_adjustments_are_delivered_as_derived_revisions() {
    let adapter = Arc::new(RecordingAdapter::default());
    let (runtime, client) =
        Runtime::start(RuntimeConfig::default().with_delivery_adapter(adapter.clone()));
    let events = client.subscribe();
    client
        .send(
            Some("doc"),
            Command::DocsUpdateHead {
                doc_id: "notes".into(),
                expected_base: 0,
                new_revision: Revision {
                    rev_id: "rev-1".to_string(),
                    base_revision: 0,
                    source_attempt_ids: vec![],
                    instruction_template_id: "tpl-none".to_string(),
                    text: "Fox jumps".to_string(),
                    status: "candidate".to_string(),
                    provenance: "recognition".to_string(),
                },
            },
        )
        .expect("updateHead accepted");
    until(&events, "docs.headUpdated");
    let deliver = |rev, target| prepare_and_apply(&client, &events, &adapter, rev, target);

    assert_eq!(deliver("rev-1", "mid-sentence"), " fox jumps");
    assert_eq!(deliver("rev-1", "after-period"), " Fox jumps");
    // Each derived revision keeps its own text, and is delivered exactly as
    // recorded even where the rules would change it.
    assert_eq!(
        deliver("rev-1#boundary-space", "mid-sentence"),
        " Fox jumps"
    );
    assert_eq!(deliver("rev-1#boundary-space-case", "plain"), " fox jumps");
    // The requested revision is untouched.
    assert_eq!(deliver("rev-1", "plain"), "Fox jumps");
    assert_eq!(deliver("rev-1", "field-start"), "Fox jumps");

    runtime.shutdown();
}
