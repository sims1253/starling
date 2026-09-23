//! The I5 documents-machine wiring (issue #220): the documents machine
//! persists through storage v2's `documents`/`revisions` tables
//! ([`V2DocumentStore`]) and **hydrates from them on first touch**, so a
//! restarted runtime answers `docs.get` from the durable rows and CASes
//! against the durable head — the property a write-only seam could not
//! claim, and the one that makes `docs.updateHead` safe across host
//! restarts: a forgotten head would let `expected_base: 0` "succeed"
//! against a document whose durable head is 2 and silently rewind it.

use std::time::Duration;

use starling_runtime::bus::EventSub;
use starling_runtime::channel::RecvError;
use starling_runtime::machine::docs::V2DocumentStore;
use starling_runtime::machine::Receipt;
use starling_runtime::protocol::{Command, Event, Revision};
use starling_runtime::{Runtime, RuntimeConfig};

fn revision(rev_id: &str, base: u64, text: &str) -> Revision {
    Revision {
        rev_id: rev_id.to_string(),
        base_revision: base,
        source_attempt_ids: vec!["att-1".to_string()],
        instruction_template_id: "tpl-none".to_string(),
        text: text.to_string(),
        status: "candidate".to_string(),
        provenance: "recognition".to_string(),
    }
}

/// Collects events until `type` arrives (bounded; panics with the types
/// seen so far on timeout — the same shape the sibling suites use).
fn until(events: &EventSub, wanted: &str, deadline: Duration) -> Event {
    let started = std::time::Instant::now();
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

fn boot(root: &std::path::Path) -> (Runtime, starling_runtime::RuntimeClient) {
    let store = V2DocumentStore::open(root).expect("v2 document store opens");
    Runtime::start(RuntimeConfig::default().with_document_store(std::sync::Arc::new(store)))
}

/// A fresh client drives the durable claim end to end: two committed
/// heads, one preserved conflict candidate, a turn; a restart serves the
/// document from the rows, refuses a CAS against a forgotten head, and
/// accepts one against the durable head.
#[test]
fn documents_survive_a_runtime_restart_through_storage_v2() {
    let root = tempfile::tempdir().expect("tempdir");

    {
        let (runtime, client) = boot(root.path());
        let events = client.subscribe();

        client
            .send(
                Some("doc-1"),
                Command::DocsUpdateHead {
                    doc_id: "notes".into(),
                    expected_base: 0,
                    new_revision: revision("rev-1", 0, "First head."),
                },
            )
            .expect("first update accepted");
        let event = until(&events, "docs.headUpdated", Duration::from_secs(5));
        match event {
            Event::DocsHeadUpdated { head_revision, .. } => assert_eq!(head_revision, 1),
            other => panic!("expected headUpdated, got {other:?}"),
        }

        // A stale-base conflict BEFORE the second commit: the candidate is
        // preserved alongside the committed history.
        client
            .send(
                Some("doc-1"),
                Command::DocsUpdateHead {
                    doc_id: "notes".into(),
                    expected_base: 0,
                    new_revision: revision("rev-raced", 0, "Stale writer's candidate."),
                },
            )
            .expect("stale update is answered (the event decides)");
        let event = until(&events, "docs.headConflict", Duration::from_secs(5));
        match event {
            Event::DocsHeadConflict {
                expected,
                actual,
                candidate_preserved,
            } => {
                assert_eq!((expected, actual), (0, 1));
                assert!(candidate_preserved);
            }
            other => panic!("expected headConflict, got {other:?}"),
        }

        client
            .send(
                Some("doc-1"),
                Command::DocsUpdateHead {
                    doc_id: "notes".into(),
                    expected_base: 1,
                    new_revision: revision("rev-2", 1, "Second head."),
                },
            )
            .expect("second update accepted");
        until(&events, "docs.headUpdated", Duration::from_secs(5));

        client
            .send(
                Some("doc-1"),
                Command::DocsAppendTurn {
                    doc_id: "notes".into(),
                    take_ref: "take_1".into(),
                },
            )
            .expect("turn accepted");
        let event = until(&events, "docs.turnAppended", Duration::from_secs(5));
        match event {
            Event::DocsTurnAppended { turn_seq } => assert_eq!(turn_seq, 1),
            other => panic!("expected turnAppended, got {other:?}"),
        }

        runtime.shutdown();
    }

    // A fresh runtime over the same root: the machine hydrates on first
    // touch and the document is there — head 2, one turn, three
    // revisions in insertion order with the candidate preserved.
    {
        let (runtime, client) = boot(root.path());
        match client
            .send(
                Some("doc-2"),
                Command::DocsGet {
                    doc_id: "notes".into(),
                    page: 0,
                },
            )
            .expect("get accepted")
        {
            Receipt::Served(view) => {
                assert_eq!(view["found"], true, "{view}");
                assert_eq!(view["headRevision"], 2, "{view}");
                assert_eq!(view["turnSeq"], 1, "{view}");
                let revisions = view["revisions"].as_array().expect("revisions");
                assert_eq!(revisions.len(), 3, "{view}");
                assert_eq!(revisions[0]["revId"], "rev-1");
                assert_eq!(revisions[0]["slot"], "committed");
                assert_eq!(revisions[1]["revId"], "rev-raced");
                assert_eq!(revisions[1]["slot"], "preserved");
                assert_eq!(revisions[2]["revId"], "rev-2");
                assert_eq!(revisions[2]["slot"], "committed");
                assert_eq!(revisions[2]["text"], "Second head.");
            }
            other => panic!("expected a served view, got {other:?}"),
        }

        // The durable head is the CAS truth: base 0 — which the first
        // session accepted — must conflict now.
        client
            .send(
                Some("doc-2"),
                Command::DocsUpdateHead {
                    doc_id: "notes".into(),
                    expected_base: 0,
                    new_revision: revision("rev-rewind", 0, "Would rewind the head."),
                },
            )
            .expect("stale update is answered (the event decides)");
        let events = client.subscribe();
        let event = until(&events, "docs.headConflict", Duration::from_secs(5));
        match event {
            Event::DocsHeadConflict { actual, .. } => assert_eq!(actual, 2),
            other => panic!("expected headConflict, got {other:?}"),
        }

        // And the correct base still commits through the hydrated state.
        client
            .send(
                Some("doc-2"),
                Command::DocsUpdateHead {
                    doc_id: "notes".into(),
                    expected_base: 2,
                    new_revision: revision("rev-3", 2, "Third head, post-restart."),
                },
            )
            .expect("post-restart update accepted");
        let event = until(&events, "docs.headUpdated", Duration::from_secs(5));
        match event {
            Event::DocsHeadUpdated { head_revision, .. } => assert_eq!(head_revision, 3),
            other => panic!("expected headUpdated, got {other:?}"),
        }

        runtime.shutdown();
    }
}

/// The store-level durable witness: the rows a live session wrote are in
/// SQLite (not just in the machine), read through `StoreV2`'s own public
/// API — the same evidence shape the capture machine's kill tests use.
#[test]
fn the_durable_rows_are_readable_through_store_v2_itself() {
    use starling_dictation::store_v2::StoreV2;

    let root = tempfile::tempdir().expect("tempdir");
    {
        let (runtime, client) = boot(root.path());
        let events = client.subscribe();
        client
            .send(
                Some("doc-1"),
                Command::DocsUpdateHead {
                    doc_id: "notes".into(),
                    expected_base: 0,
                    new_revision: revision("rev-1", 0, "Row-level witness."),
                },
            )
            .expect("update accepted");
        until(&events, "docs.headUpdated", Duration::from_secs(5));
        runtime.shutdown();
    }

    let store = StoreV2::open(root.path()).expect("reopen through the store API");
    let doc = store
        .get_document("notes")
        .expect("get")
        .expect("the rows are durable");
    assert_eq!(doc.head_revision, 1);
    assert_eq!(doc.revisions.len(), 1);
    assert_eq!(doc.revisions[0].rev_id, "rev-1");
    assert_eq!(doc.revisions[0].disposition.as_deref(), Some("committed"));
    // The I3 provenance rode through `sources_json`.
    let sources: serde_json::Value =
        serde_json::from_str(doc.revisions[0].sources_json.as_deref().expect("sources"))
            .expect("sources parse");
    assert_eq!(sources["attempts"][0], "att-1");
    assert_eq!(sources["instructionTemplateId"], "tpl-none");
}
