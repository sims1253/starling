//! The #341 wiring in the delivery machine, pinned black-box through the
//! runtime protocol: an adapter that reports surrounding text gets the
//! frozen boundary rules applied at prepare, the adjustment is recorded as
//! a derived revision (raw round-trips byte-for-byte), and an adapter
//! without the capability delivers the dictated text unchanged.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use starling_runtime::bus::EventSub;
use starling_runtime::channel::RecvError;
use starling_runtime::machine::delivery::{
    DeliveryAdapter, InsertEvidence, InsertionFailure, Revalidation, SurroundingText,
};
use starling_runtime::protocol::{Command, Event, Revision};
use starling_runtime::{Runtime, RuntimeConfig};

/// A recording adapter: `with-context` reports mid-sentence surrounding
/// text, everything else reports the (default) unavailable capability —
/// the two honest adapter states of #341's capability model.
struct RecordingAdapter {
    inserted: Mutex<Vec<(String, String)>>,
}

impl RecordingAdapter {
    fn new() -> Arc<Self> {
        Arc::new(RecordingAdapter {
            inserted: Mutex::new(Vec::new()),
        })
    }

    fn insertions(&self) -> Vec<(String, String)> {
        self.inserted.lock().expect("inserted lock").clone()
    }
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
        target_ref: &str,
        text: &str,
    ) -> Result<InsertEvidence, InsertionFailure> {
        self.inserted
            .lock()
            .expect("inserted lock")
            .push((target_ref.to_string(), text.to_string()));
        Ok(InsertEvidence {
            level: "recorded".to_string(),
        })
    }

    fn describe(&self) -> String {
        "recording adapter (tests)".to_string()
    }

    fn surrounding_text(&self, target_ref: &str) -> SurroundingText {
        if target_ref == "with-context" {
            SurroundingText::Available {
                before: "The quick brown".to_string(),
                after: String::new(),
            }
        } else {
            SurroundingText::Unavailable {
                reason: "no surrounding-text capability on this target".to_string(),
            }
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

fn until(events: &EventSub, wanted: &str, deadline: Duration) -> Event {
    let started = Instant::now();
    loop {
        match events.recv_timeout(Duration::from_millis(20)) {
            Ok(message) => {
                if message.event.type_name() == wanted {
                    return message.event;
                }
            }
            Err(RecvError::Timeout) => {
                if started.elapsed() > deadline {
                    panic!("timed out waiting for {wanted}");
                }
            }
            Err(other) => panic!("event stream error: {other:?}"),
        }
    }
}

fn prepare_and_apply(
    client: &starling_runtime::RuntimeClient,
    events: &EventSub,
    corr: &str,
    rev: &str,
    target: &str,
) {
    client
        .send(
            Some(corr),
            Command::DeliveryPrepare {
                revision_id: rev.to_string(),
                target_ref: target.to_string(),
            },
        )
        .expect("prepare accepted");
    let prepared = until(events, "delivery.prepared", Duration::from_secs(5));
    let delivery_id = match prepared {
        Event::DeliveryPrepared { delivery_id, .. } => delivery_id,
        other => panic!("expected DeliveryPrepared, got {other:?}"),
    };
    client
        .send(Some(corr), Command::DeliveryApply { delivery_id })
        .expect("apply accepted");
    until(events, "delivery.confirmed", Duration::from_secs(5));
}

/// The full #341 walk: mid-sentence context adjusts the boundary, the
/// adjusted text is what lands, the derived revision exists in the
/// registry (a prepare against it succeeds) and delivers without a second
/// adjustment, and the raw revision still round-trips byte-for-byte.
#[test]
fn boundary_adjustment_lands_is_recorded_and_never_touches_the_raw() {
    let adapter = RecordingAdapter::new();
    let (runtime, client) =
        Runtime::start(RuntimeConfig::default().with_delivery_adapter(adapter.clone()));
    let events = client.subscribe();

    client
        .send(
            Some("doc-1"),
            Command::DocsUpdateHead {
                doc_id: "notes".into(),
                expected_base: 0,
                new_revision: revision("rev-1", "Fox jumps"),
            },
        )
        .expect("updateHead accepted");
    until(&events, "docs.headUpdated", Duration::from_secs(5));

    // 1. With surrounding text: the boundary adjusts to " fox jumps".
    prepare_and_apply(&client, &events, "dlv-1", "rev-1", "with-context");
    assert_eq!(
        adapter.insertions(),
        vec![("with-context".to_string(), " fox jumps".to_string())],
        "the adjusted text is what lands"
    );

    // 2. The derived revision exists and delivers exactly its text (no
    //    double adjustment, even against a context-bearing target).
    prepare_and_apply(&client, &events, "dlv-2", "rev-1#boundary", "with-context");
    assert_eq!(
        adapter.insertions()[1],
        ("with-context".to_string(), " fox jumps".to_string()),
        "the derived revision delivers without a second adjustment"
    );

    // 3. The raw revision is unchanged and one action away: preparing it
    //    against a target without the capability delivers it byte-for-byte.
    prepare_and_apply(&client, &events, "dlv-3", "rev-1", "without-context");
    assert_eq!(
        adapter.insertions()[2],
        ("without-context".to_string(), "Fox jumps".to_string()),
        "the raw recognition text round-trips unchanged"
    );

    // 4. A capability-less target from the start inserts unchanged.
    client
        .send(
            Some("doc-2"),
            Command::DocsUpdateHead {
                doc_id: "notes2".into(),
                expected_base: 0,
                new_revision: revision("rev-2", "Next one"),
            },
        )
        .expect("updateHead accepted");
    until(&events, "docs.headUpdated", Duration::from_secs(5));
    prepare_and_apply(&client, &events, "dlv-4", "rev-2", "without-context");
    assert_eq!(
        adapter.insertions()[3],
        ("without-context".to_string(), "Next one".to_string()),
        "no capability, no adjustment"
    );

    runtime.shutdown();
}

/// Field-start context (empty before-text) never adjusts anything, and a
/// derived revision is never created for it.
#[test]
fn field_start_context_leaves_the_text_untouched() {
    let adapter = Arc::new(FieldStartAdapter::default());
    let (runtime, client) =
        Runtime::start(RuntimeConfig::default().with_delivery_adapter(adapter.clone()));
    let events = client.subscribe();

    client
        .send(
            Some("doc-1"),
            Command::DocsUpdateHead {
                doc_id: "notes".into(),
                expected_base: 0,
                new_revision: revision("rev-1", "Hello"),
            },
        )
        .expect("updateHead accepted");
    until(&events, "docs.headUpdated", Duration::from_secs(5));

    prepare_and_apply(&client, &events, "dlv-1", "rev-1", "field-start");
    assert_eq!(
        adapter.inserted.lock().unwrap().as_slice(),
        [("field-start".to_string(), "Hello".to_string())],
    );

    // No derived revision was registered: preparing the would-be id is
    // refused exactly like any other unknown revision.
    match client.send(
        Some("dlv-2"),
        Command::DeliveryPrepare {
            revision_id: "rev-1#boundary".into(),
            target_ref: "field-start".into(),
        },
    ) {
        Err(starling_runtime::machine::Rejection::UnknownRevision { revision_id }) => {
            assert_eq!(revision_id, "rev-1#boundary");
        }
        other => panic!("expected an UnknownRevision refusal, got {other:?}"),
    }

    runtime.shutdown();
}

#[derive(Default)]
struct FieldStartAdapter {
    inserted: Mutex<Vec<(String, String)>>,
}

impl DeliveryAdapter for FieldStartAdapter {
    fn prepare(&self, _target_ref: &str) -> Result<String, String> {
        Ok("token".to_string())
    }
    fn revalidate(&self, _target_ref: &str, _compare_token: &str) -> Revalidation {
        Revalidation::Unchanged
    }
    fn insert(
        &self,
        _delivery_id: &str,
        target_ref: &str,
        text: &str,
    ) -> Result<InsertEvidence, InsertionFailure> {
        self.inserted
            .lock()
            .unwrap()
            .push((target_ref.to_string(), text.to_string()));
        Ok(InsertEvidence {
            level: "recorded".to_string(),
        })
    }
    fn describe(&self) -> String {
        "field-start adapter (tests)".to_string()
    }
    fn surrounding_text(&self, _target_ref: &str) -> SurroundingText {
        SurroundingText::Available {
            before: String::new(),
            after: " world".to_string(),
        }
    }
}
