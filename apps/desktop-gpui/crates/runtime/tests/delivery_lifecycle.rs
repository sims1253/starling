//! The live delivery actor's first lifecycle event, pinned in-process:
//! `delivery.prepare` is outcome-pending, so `delivery.prepared` reaches
//! the event stream through the resolve itself — the actor used to
//! follow the resolve with an extra `emit_event` for the same type,
//! which the table always refuses (`delivery.prepared` is `from []` as a
//! free event), and the swallowed violation meant the live stream never
//! carried the event at all. Found by the I5 adapter wiring (issue
//! #220); the conformance corpus replays this trace through the oracle
//! table and so never exercised the actor's side. The host crate's
//! `adapters` suite pins the same walk over IPC with a real adapter.

use std::time::{Duration, Instant};

use starling_runtime::bus::EventSub;
use starling_runtime::channel::RecvError;
use starling_runtime::protocol::{Command, Event, Revision};
use starling_runtime::{Runtime, RuntimeConfig};

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
                    panic!("timed out waiting for {wanted} (the live actor must emit it)");
                }
            }
            Err(other) => panic!("event stream error: {other:?}"),
        }
    }
}

/// The stub adapter's honest walk: `delivery.prepared` reaches the wire
/// (the regression), and the stub keeps its word downstream — `apply`
/// fails with `no_delivery_adapter` and suggests the copy fallback
/// rather than faking a confirmation.
#[test]
fn delivery_prepared_reaches_the_live_event_stream_and_the_stub_refuses_to_confirm() {
    let (runtime, client) = Runtime::start(RuntimeConfig::default());
    let events = client.subscribe();

    client
        .send(
            Some("doc-1"),
            Command::DocsUpdateHead {
                doc_id: "notes".into(),
                expected_base: 0,
                new_revision: revision("rev-1", "A revision worth delivering."),
            },
        )
        .expect("updateHead accepted");
    until(&events, "docs.headUpdated", Duration::from_secs(5));

    client
        .send(
            Some("dlv-1"),
            Command::DeliveryPrepare {
                revision_id: "rev-1".into(),
                target_ref: "some-editor-target".into(),
            },
        )
        .expect("prepare accepted");
    let prepared = until(&events, "delivery.prepared", Duration::from_secs(5));
    let delivery_id = match prepared {
        Event::DeliveryPrepared { delivery_id, .. } => delivery_id,
        other => panic!("expected DeliveryPrepared, got {other:?}"),
    };
    assert!(!delivery_id.is_empty(), "the event carries the delivery id");

    client
        .send(Some("dlv-1"), Command::DeliveryApply { delivery_id })
        .expect("apply accepted");
    let failed = until(&events, "delivery.failed", Duration::from_secs(5));
    match failed {
        Event::DeliveryFailed {
            reason,
            fallback_suggested,
        } => {
            assert_eq!(reason, "no_delivery_adapter");
            assert!(fallback_suggested, "the stub suggests the copy fallback");
        }
        other => panic!("expected DeliveryFailed, got {other:?}"),
    }

    runtime.shutdown();
}

/// The refusal side stays honest too: a prepare against a revision no
/// docs machine ever committed answers the typed rejection, and nothing
/// is emitted for it.
#[test]
fn a_prepare_against_an_unknown_revision_is_refused_without_events() {
    let (runtime, client) = Runtime::start(RuntimeConfig::default());
    match client.send(
        Some("dlv-1"),
        Command::DeliveryPrepare {
            revision_id: "rev-never".into(),
            target_ref: "some-editor-target".into(),
        },
    ) {
        Err(starling_runtime::machine::Rejection::UnknownRevision { revision_id }) => {
            assert_eq!(revision_id, "rev-never");
        }
        other => panic!("expected an UnknownRevision refusal, got {other:?}"),
    }
    // No delivery event for the refused prepare: the stream is idle.
    match client.subscribe().recv_timeout(Duration::from_millis(200)) {
        Err(RecvError::Timeout) => {}
        Ok(message) => panic!("unexpected event: {}", message.event.type_name()),
        Err(other) => panic!("event stream error: {other:?}"),
    }
    runtime.shutdown();
}
