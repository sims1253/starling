//! The insertion-boundary wiring in the delivery machine, pinned through
//! the runtime protocol: an adapter that reports surrounding text gets the
//! boundary rules applied, the adjusted text is delivered as a derived
//! revision registered through the document service, and the requested
//! revision is never edited. Apply reads the boundary again; `raw` and
//! `verbatim` deliveries read nothing and deliver the text unchanged.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::Value;
use starling_dictation::store_v2::StoreV2;
use starling_runtime::bus::EventSub;
use starling_runtime::channel::RecvError;
use starling_runtime::machine::delivery::{
    DeliveryAdapter, InsertEvidence, InsertionFailure, ProtectedField, Revalidation,
    SurroundingRead, SurroundingText,
};
use starling_runtime::machine::docs::V2DocumentStore;
use starling_runtime::machine::{Receipt, Rejection};
use starling_runtime::protocol::{BoundaryPolicy, Command, Event, Revision};
use starling_runtime::{Runtime, RuntimeClient, RuntimeConfig};

/// Reports scripted surrounding text per target (`Unsupported` for any
/// target never scripted) and records every read and insertion. Queued
/// reads and revalidations are answered first, one per call.
#[derive(Default)]
struct ScriptedAdapter {
    surroundings: Mutex<HashMap<String, SurroundingRead>>,
    queued_reads: Mutex<VecDeque<SurroundingRead>>,
    queued_revalidations: Mutex<VecDeque<Revalidation>>,
    reads: Mutex<Vec<String>>,
    inserted: Mutex<Vec<String>>,
}

impl ScriptedAdapter {
    fn new() -> Arc<Self> {
        let adapter = Arc::new(ScriptedAdapter::default());
        adapter.before("mid-sentence", "The quick brown");
        adapter.before("after-period", "Done.");
        adapter.before("field-start", "");
        adapter
    }

    fn set(&self, target: &str, read: SurroundingRead) {
        self.surroundings
            .lock()
            .unwrap()
            .insert(target.to_string(), read);
    }

    fn before(&self, target: &str, before: &str) {
        self.set(
            target,
            SurroundingRead::Text(SurroundingText {
                before: before.to_string(),
                ..SurroundingText::default()
            }),
        );
    }

    /// The next reads, whatever the target, before the scripted ones.
    fn queue_befores(&self, befores: &[&str]) {
        self.queued_reads
            .lock()
            .unwrap()
            .extend(befores.iter().map(|before| {
                SurroundingRead::Text(SurroundingText {
                    before: before.to_string(),
                    ..SurroundingText::default()
                })
            }));
    }

    fn reads(&self) -> usize {
        self.reads.lock().unwrap().len()
    }

    fn inserted(&self) -> Vec<String> {
        self.inserted.lock().unwrap().clone()
    }
}

impl DeliveryAdapter for ScriptedAdapter {
    fn prepare(&self, _target_ref: &str) -> Result<String, String> {
        Ok("token".to_string())
    }

    fn revalidate(&self, _target_ref: &str, _compare_token: &str) -> Revalidation {
        self.queued_revalidations
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or(Revalidation::Unchanged)
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
        "scripted adapter (tests)".to_string()
    }

    fn surrounding_text(&self, target_ref: &str) -> SurroundingRead {
        self.reads.lock().unwrap().push(target_ref.to_string());
        if let Some(read) = self.queued_reads.lock().unwrap().pop_front() {
            return read;
        }
        self.surroundings
            .lock()
            .unwrap()
            .get(target_ref)
            .cloned()
            .unwrap_or(SurroundingRead::Unsupported)
    }
}

/// Waits for the first event of one of `wanted`; every event seen on the
/// way is appended to `seen`.
fn until_any(events: &EventSub, wanted: &[&str], seen: &mut Vec<Event>) -> Event {
    let started = Instant::now();
    loop {
        match events.recv_timeout(Duration::from_millis(20)) {
            Ok(message) => {
                seen.push(message.event.clone());
                if wanted.contains(&message.event.type_name()) {
                    return message.event;
                }
            }
            Err(RecvError::Timeout) if started.elapsed() > Duration::from_secs(5) => {
                panic!("timed out waiting for {wanted:?}")
            }
            Err(RecvError::Timeout) => {}
            Err(other) => panic!("event stream error: {other:?}"),
        }
    }
}

struct Session {
    runtime: Runtime,
    client: RuntimeClient,
    events: EventSub,
    adapter: Arc<ScriptedAdapter>,
    /// Every event this session's subscription saw.
    seen: Vec<Event>,
}

impl Session {
    fn start(adapter: Arc<ScriptedAdapter>) -> Session {
        Self::start_with(RuntimeConfig::default(), adapter)
    }

    fn start_with(config: RuntimeConfig, adapter: Arc<ScriptedAdapter>) -> Session {
        let (runtime, client) = Runtime::start(config.with_delivery_adapter(adapter.clone()));
        let events = client.subscribe();
        Session {
            runtime,
            client,
            events,
            adapter,
            seen: Vec::new(),
        }
    }

    fn until(&mut self, wanted: &str) -> Event {
        until_any(&self.events, &[wanted], &mut self.seen)
    }

    /// Commits `text` as the head of a fresh document `doc_id`.
    fn commit(&mut self, doc_id: &str, rev_id: &str, text: &str) {
        self.client
            .send(
                Some("doc"),
                Command::DocsUpdateHead {
                    doc_id: doc_id.into(),
                    expected_base: 0,
                    new_revision: Revision {
                        rev_id: rev_id.to_string(),
                        base_revision: 0,
                        source_attempt_ids: vec!["att-1".to_string()],
                        instruction_template_id: "tpl-none".to_string(),
                        text: text.to_string(),
                        status: "candidate".to_string(),
                        provenance: "recognition".to_string(),
                    },
                },
            )
            .expect("updateHead accepted");
        self.until("docs.headUpdated");
    }

    fn prepare_with(
        &mut self,
        rev: &str,
        target: &str,
        boundary: BoundaryPolicy,
    ) -> Result<String, Rejection> {
        self.client.send(
            Some("dlv"),
            Command::DeliveryPrepare {
                revision_id: rev.to_string(),
                target_ref: target.to_string(),
                boundary,
            },
        )?;
        let Event::DeliveryPrepared { delivery_id, .. } = self.until("delivery.prepared") else {
            unreachable!()
        };
        Ok(delivery_id)
    }

    fn prepare(&mut self, rev: &str, target: &str) -> String {
        self.prepare_with(rev, target, BoundaryPolicy::Adjust)
            .unwrap_or_else(|err| panic!("prepare {rev}: {err:?}"))
    }

    /// Applies a prepared delivery; the terminal event.
    fn apply(&mut self, delivery_id: String) -> Event {
        self.client
            .send(Some("dlv"), Command::DeliveryApply { delivery_id })
            .expect("apply accepted");
        until_any(
            &self.events,
            &["delivery.confirmed", "delivery.failed"],
            &mut self.seen,
        )
    }

    /// Prepares and applies `rev` against `target`; the inserted text.
    fn deliver_with(&mut self, rev: &str, target: &str, boundary: BoundaryPolicy) -> String {
        let delivery_id = self
            .prepare_with(rev, target, boundary)
            .unwrap_or_else(|err| panic!("prepare {rev}: {err:?}"));
        let terminal = self.apply(delivery_id);
        assert_eq!(terminal.type_name(), "delivery.confirmed", "{terminal:?}");
        self.adapter.inserted().last().cloned().unwrap()
    }

    fn deliver(&mut self, rev: &str, target: &str) -> String {
        self.deliver_with(rev, target, BoundaryPolicy::Adjust)
    }

    fn docs_get(&self, doc_id: &str) -> Value {
        match self
            .client
            .send(
                Some("doc"),
                Command::DocsGet {
                    doc_id: doc_id.into(),
                    page: 0,
                },
            )
            .expect("docs.get served")
        {
            Receipt::Served(view) => view,
            other => panic!("docs.get answered {other:?}"),
        }
    }
}

/// `(revId, slot, text, derivedFrom)` of every revision `docs.get` serves.
fn revisions(view: &Value) -> Vec<(String, String, String, Option<String>)> {
    view["revisions"]
        .as_array()
        .unwrap_or_else(|| panic!("no revisions in {view}"))
        .iter()
        .map(|revision| {
            (
                revision["revId"].as_str().unwrap().to_string(),
                revision["slot"].as_str().unwrap().to_string(),
                revision["text"].as_str().unwrap().to_string(),
                revision["derivedFrom"].as_str().map(str::to_string),
            )
        })
        .collect()
}

fn derived(rev: &str, text: &str, from: &str) -> (String, String, String, Option<String>) {
    (
        rev.to_string(),
        "derived".to_string(),
        text.to_string(),
        Some(from.to_string()),
    )
}

#[test]
fn boundary_adjustments_are_delivered_as_derived_revisions() {
    let mut session = Session::start(ScriptedAdapter::new());
    session.commit("notes", "rev-1", "Fox jumps");

    assert_eq!(session.deliver("rev-1", "mid-sentence"), " fox jumps");
    assert_eq!(session.deliver("rev-1", "after-period"), " Fox jumps");
    // Each derived revision keeps its own text, and is delivered exactly as
    // recorded even where the rules would change it.
    assert_eq!(
        session.deliver("rev-1:boundary-space", "mid-sentence"),
        " Fox jumps"
    );
    assert_eq!(
        session.deliver("rev-1:boundary-space-case", "plain"),
        " fox jumps"
    );
    // Re-deriving the same revision reuses it.
    assert_eq!(session.deliver("rev-1", "mid-sentence"), " fox jumps");
    // The requested revision is untouched.
    assert_eq!(session.deliver("rev-1", "plain"), "Fox jumps");
    assert_eq!(session.deliver("rev-1", "field-start"), "Fox jumps");

    session.runtime.shutdown();
}

#[test]
fn derived_revisions_are_served_by_docs_get_without_moving_the_head() {
    let mut session = Session::start(ScriptedAdapter::new());
    session.commit("notes", "rev-1", "Fox jumps");
    session.deliver("rev-1", "mid-sentence");
    session.deliver("rev-1", "after-period");
    session.deliver("rev-1", "mid-sentence");

    let view = session.docs_get("notes");
    assert_eq!(view["headRevision"], 1, "{view}");
    assert_eq!(
        revisions(&view),
        vec![
            (
                "rev-1".to_string(),
                "committed".to_string(),
                "Fox jumps".to_string(),
                None
            ),
            derived("rev-1:boundary-space-case", " fox jumps", "rev-1"),
            derived("rev-1:boundary-space", " Fox jumps", "rev-1"),
        ]
    );
    assert_eq!(view["revisions"][1]["provenance"], "insertion-boundary");
    // The head still moves by compare-and-swap against the committed head.
    session
        .client
        .send(
            Some("doc"),
            Command::DocsUpdateHead {
                doc_id: "notes".into(),
                expected_base: 1,
                new_revision: Revision {
                    rev_id: "rev-2".into(),
                    base_revision: 1,
                    source_attempt_ids: vec![],
                    instruction_template_id: "tpl-none".into(),
                    text: "Fox jumps high".into(),
                    status: "candidate".into(),
                    provenance: "recognition".into(),
                },
            },
        )
        .expect("updateHead accepted");
    session.until("docs.headUpdated");

    session.runtime.shutdown();
}

#[test]
fn raw_and_verbatim_deliver_the_source_unchanged_without_reading() {
    let mut session = Session::start(ScriptedAdapter::new());
    session.commit("notes", "rev-1", "Fox jumps");

    assert_eq!(session.deliver("rev-1", "mid-sentence"), " fox jumps");
    let reads = session.adapter.reads();
    // The raw text stays one action away at the same target.
    assert_eq!(
        session.deliver_with("rev-1", "mid-sentence", BoundaryPolicy::Raw),
        "Fox jumps"
    );
    assert_eq!(
        session.deliver_with("rev-1", "mid-sentence", BoundaryPolicy::Verbatim),
        "Fox jumps"
    );
    assert_eq!(session.adapter.reads(), reads, "raw/verbatim read nothing");
    // A derived revision is delivered as recorded without a read either.
    assert_eq!(
        session.deliver("rev-1:boundary-space-case", "after-period"),
        " fox jumps"
    );
    assert_eq!(session.adapter.reads(), reads);

    let view = session.docs_get("notes");
    assert_eq!(view["revisions"].as_array().unwrap().len(), 2, "{view}");
    session.runtime.shutdown();
}

#[test]
fn apply_rederives_when_the_boundary_changed_since_prepare() {
    let mut session = Session::start(ScriptedAdapter::new());
    session.commit("notes", "rev-1", "Fox jumps");

    // Unchanged boundary: prepare and apply both read, same text lands.
    let delivery = session.prepare("rev-1", "editor-a");
    assert_eq!(session.adapter.reads(), 1);
    session.apply(delivery);
    assert_eq!(session.adapter.reads(), 2);
    assert_eq!(session.adapter.inserted().last().unwrap(), "Fox jumps");

    // The user typed between prepare and apply.
    session.adapter.before("editor-b", "Done.");
    let delivery = session.prepare("rev-1", "editor-b");
    session.adapter.before("editor-b", "Done. The quick brown");
    session.apply(delivery);
    assert_eq!(session.adapter.inserted().last().unwrap(), " fox jumps");

    // The field was cleared: the requested text lands unchanged.
    session.adapter.before("editor-c", "The quick brown");
    let delivery = session.prepare("rev-1", "editor-c");
    session.adapter.before("editor-c", "");
    session.apply(delivery);
    assert_eq!(session.adapter.inserted().last().unwrap(), "Fox jumps");

    // The field became secure: nothing derived from it is delivered.
    session.adapter.before("editor-d", "The quick brown");
    let delivery = session.prepare("rev-1", "editor-d");
    session.adapter.set(
        "editor-d",
        SurroundingRead::Protected(ProtectedField::Secure),
    );
    session.apply(delivery);
    assert_eq!(session.adapter.inserted().last().unwrap(), "Fox jumps");

    // Both derivations, the stale one included, stay recorded; the source
    // is untouched.
    let view = session.docs_get("notes");
    let revisions = revisions(&view);
    assert_eq!(revisions[0].2, "Fox jumps");
    assert!(revisions.contains(&derived("rev-1:boundary-space", " Fox jumps", "rev-1")));
    assert!(revisions.contains(&derived("rev-1:boundary-space-case", " fox jumps", "rev-1")));
    session.runtime.shutdown();
}

/// Registering a derivation waits on the document service: apply reads the
/// boundary again afterwards, derives again when it moved, gives up when
/// it keeps moving, and revalidates the target once more.
#[test]
fn apply_rechecks_the_boundary_and_target_after_registering() {
    let mut session = Session::start(ScriptedAdapter::new());
    session.commit("notes", "rev-1", "Fox jumps");

    // Cleared while the derivation was being registered.
    let delivery = session.prepare("rev-1", "field-start");
    session.adapter.queue_befores(&["The quick brown", ""]);
    session.apply(delivery);
    assert_eq!(session.adapter.inserted().last().unwrap(), "Fox jumps");

    // Moving on every read: nothing is inserted.
    let inserted = session.adapter.inserted().len();
    let delivery = session.prepare("rev-1", "field-start");
    session
        .adapter
        .queue_befores(&["The quick brown", "Done.", "The quick brown", "Done."]);
    assert_eq!(
        session.apply(delivery),
        Event::DeliveryFailed {
            reason: "boundary_unstable".into(),
            fallback_suggested: true,
        }
    );
    assert_eq!(session.adapter.inserted().len(), inserted);

    // The target moved while the registration waited.
    let delivery = session.prepare("rev-1", "field-start");
    session
        .adapter
        .queue_befores(&["The quick brown", "The quick brown"]);
    session
        .adapter
        .queued_revalidations
        .lock()
        .unwrap()
        .extend([
            Revalidation::Unchanged,
            Revalidation::Changed {
                expected: "field-start".into(),
                actual: "elsewhere".into(),
            },
        ]);
    session
        .client
        .send(
            Some("dlv"),
            Command::DeliveryApply {
                delivery_id: delivery,
            },
        )
        .expect("apply accepted");
    assert_eq!(
        session.until("delivery.conflict"),
        Event::DeliveryConflict {
            expected_target: "field-start".into(),
            actual_target: "elsewhere".into(),
        }
    );
    assert_eq!(session.adapter.inserted().len(), inserted);

    session.runtime.shutdown();
}

/// A derivation prepare already recorded is delivered as is: apply reads
/// the boundary once and does not wait on the document service, so the
/// target is revalidated only once.
#[test]
fn apply_delivers_the_derivation_prepare_recorded_without_waiting() {
    let mut session = Session::start(ScriptedAdapter::new());
    session.commit("notes", "rev-1", "Fox jumps");
    let delivery = session.prepare("rev-1", "mid-sentence");
    let reads = session.adapter.reads();
    // A second revalidation would report a conflict.
    session
        .adapter
        .queued_revalidations
        .lock()
        .unwrap()
        .extend([
            Revalidation::Unchanged,
            Revalidation::Changed {
                expected: "mid-sentence".into(),
                actual: "elsewhere".into(),
            },
        ]);
    assert_eq!(session.apply(delivery).type_name(), "delivery.confirmed");
    assert_eq!(session.adapter.reads(), reads + 1);
    assert_eq!(session.adapter.inserted().last().unwrap(), " fox jumps");
    session.runtime.shutdown();
}

#[test]
fn protected_and_hint_only_fields_are_never_adjusted() {
    let adapter = ScriptedAdapter::new();
    adapter.set(
        "password",
        SurroundingRead::Protected(ProtectedField::Secure),
    );
    adapter.set(
        "incognito",
        SurroundingRead::Protected(ProtectedField::Incognito),
    );
    adapter.set(
        "hint-only",
        SurroundingRead::Text(SurroundingText {
            before: "Search notes".to_string(),
            after: String::new(),
            showing_hint: true,
        }),
    );
    let mut session = Session::start(adapter);
    session.commit("notes", "rev-1", "Fox jumps");

    for target in ["password", "incognito", "hint-only", "unsupported"] {
        assert_eq!(session.deliver("rev-1", target), "Fox jumps", "{target}");
    }
    let view = session.docs_get("notes");
    assert_eq!(view["revisions"].as_array().unwrap().len(), 1, "{view}");
    session.runtime.shutdown();
}

#[test]
fn a_derived_id_never_replaces_an_unrelated_revision() {
    let mut session = Session::start(ScriptedAdapter::new());
    session.commit("notes", "r", "Next one");
    session.commit("other", "r:boundary-space", "Unrelated");

    assert_eq!(
        session.prepare_with("r", "after-period", BoundaryPolicy::Adjust),
        Err(Rejection::RevisionIdTaken {
            revision_id: "r:boundary-space".into()
        })
    );
    assert_eq!(session.deliver("r:boundary-space", "plain"), "Unrelated");

    // The same refusal at apply, when re-deriving lands on a taken id:
    // nothing is inserted and the copy fallback is offered.
    session.commit("other-2", "r:boundary-space-case", "Also unrelated");
    session.adapter.before("editor", "");
    let delivery = session.prepare("r", "editor");
    session.adapter.before("editor", "The quick brown");
    let inserted = session.adapter.inserted().len();
    assert_eq!(
        session.apply(delivery),
        Event::DeliveryFailed {
            reason: "revision_id_taken".into(),
            fallback_suggested: true,
        }
    );
    assert_eq!(session.adapter.inserted().len(), inserted);
    assert_eq!(
        revisions(&session.docs_get("notes")),
        vec![(
            "r".to_string(),
            "committed".to_string(),
            "Next one".to_string(),
            None
        )]
    );

    session.runtime.shutdown();
}

/// A derived id durably held by a document this session never loaded is
/// refused like one held in memory.
#[test]
fn a_derived_id_held_by_an_unloaded_document_is_refused() {
    let root = tempfile::tempdir().expect("tempdir");
    let config = || {
        let store = V2DocumentStore::open(root.path()).expect("v2 document store opens");
        RuntimeConfig::default().with_document_store(Arc::new(store))
    };
    let mut session = Session::start_with(config(), ScriptedAdapter::new());
    session.commit("notes", "r", "Next one");
    session.commit("other", "r:boundary-space", "Unrelated");
    session.runtime.shutdown();

    let mut session = Session::start_with(config(), ScriptedAdapter::new());
    session.docs_get("notes");
    assert_eq!(
        session.prepare_with("r", "after-period", BoundaryPolicy::Adjust),
        Err(Rejection::RevisionIdTaken {
            revision_id: "r:boundary-space".into()
        })
    );
    assert_eq!(revisions(&session.docs_get("notes")).len(), 1);
    assert_eq!(revisions(&session.docs_get("other"))[0].2, "Unrelated");
    session.runtime.shutdown();
}

/// Derived revisions persist through storage v2 and survive a restart;
/// the surrounding text reaches neither the stored rows nor the event
/// stream.
#[test]
fn derived_revisions_persist_in_storage_v2_and_the_context_does_not() {
    const MARKER: &str = "Zanzibar marker";
    let root = tempfile::tempdir().expect("tempdir");
    let config = || {
        let store = V2DocumentStore::open(root.path()).expect("v2 document store opens");
        RuntimeConfig::default().with_document_store(Arc::new(store))
    };

    let adapter = ScriptedAdapter::new();
    adapter.before("editor", &format!("{MARKER} and the quick brown"));
    let mut session = Session::start_with(config(), adapter);
    session.commit("notes", "rev-1", "Fox jumps");
    assert_eq!(session.deliver("rev-1", "editor"), " fox jumps");
    let seen = std::mem::take(&mut session.seen);
    session.runtime.shutdown();
    for event in &seen {
        assert!(
            !event.payload_value().to_string().contains("Zanzibar"),
            "{event:?}"
        );
    }

    let mut session = Session::start_with(config(), ScriptedAdapter::new());
    let view = session.docs_get("notes");
    assert_eq!(view["headRevision"], 1, "{view}");
    assert_eq!(
        revisions(&view),
        vec![
            (
                "rev-1".to_string(),
                "committed".to_string(),
                "Fox jumps".to_string(),
                None
            ),
            derived("rev-1:boundary-space-case", " fox jumps", "rev-1"),
        ]
    );
    // Hydration republishes the derived revision for delivery.
    assert_eq!(
        session.deliver("rev-1:boundary-space-case", "plain"),
        " fox jumps"
    );
    session.runtime.shutdown();

    let store = StoreV2::open(root.path()).expect("store reopens");
    let document = store.get_document("notes").unwrap().expect("document row");
    assert_eq!(document.head_revision, 1);
    let row = &document.revisions[1];
    assert_eq!(row.disposition.as_deref(), Some("derived"));
    assert_eq!(row.provenance.as_deref(), Some("insertion-boundary"));
    let sources: Value = serde_json::from_str(row.sources_json.as_deref().unwrap()).unwrap();
    assert_eq!(sources["derivedFrom"], "rev-1");
    assert_eq!(sources["attempts"][0], "att-1");
    for row in &document.revisions {
        assert!(!format!("{row:?}").contains("Zanzibar"), "{row:?}");
    }
}

/// A recorded derived revision's id is never written again through
/// `docs.updateHead` — neither as a new head nor as a conflict candidate,
/// in its own document or another — so the delivered text stays the
/// recorded one in memory, in storage and after a restart.
#[test]
fn update_head_never_rewrites_a_derived_revision() {
    let root = tempfile::tempdir().expect("tempdir");
    let config = || {
        let store = V2DocumentStore::open(root.path()).expect("v2 document store opens");
        RuntimeConfig::default().with_document_store(Arc::new(store))
    };
    let update = |doc_id: &str, expected_base: u64| Command::DocsUpdateHead {
        doc_id: doc_id.into(),
        expected_base,
        new_revision: Revision {
            rev_id: "r:boundary-space".to_string(),
            base_revision: expected_base,
            source_attempt_ids: vec!["att-9".to_string()],
            instruction_template_id: "tpl-none".to_string(),
            text: "Replaced".to_string(),
            status: "candidate".to_string(),
            provenance: "recognition".to_string(),
        },
    };
    let taken = Err(Rejection::RevisionIdTaken {
        revision_id: "r:boundary-space".into(),
    });
    let recorded = vec![
        (
            "r".to_string(),
            "committed".to_string(),
            "Next one".to_string(),
            None,
        ),
        derived("r:boundary-space", " Next one", "r"),
    ];

    let mut session = Session::start_with(config(), ScriptedAdapter::new());
    session.commit("notes", "r", "Next one");
    assert_eq!(session.deliver("r", "after-period"), " Next one");
    session.seen.clear();
    // As the next head, as a stale-base conflict candidate, and in
    // another document.
    assert_eq!(session.client.send(Some("doc"), update("notes", 1)), taken);
    assert_eq!(session.client.send(Some("doc"), update("notes", 0)), taken);
    assert_eq!(session.client.send(Some("doc"), update("other", 0)), taken);
    // The machine was left untouched: the next update commits normally.
    let mut next = match update("notes", 1) {
        Command::DocsUpdateHead { new_revision, .. } => new_revision,
        _ => unreachable!(),
    };
    next.rev_id = "r-2".to_string();
    session
        .client
        .send(
            Some("doc"),
            Command::DocsUpdateHead {
                doc_id: "notes".into(),
                expected_base: 1,
                new_revision: next,
            },
        )
        .expect("updateHead accepted");
    session.until("docs.headUpdated");
    assert!(
        !session
            .seen
            .iter()
            .any(|event| event.type_name() == "docs.headConflict"),
        "{:?}",
        session.seen
    );
    let view = session.docs_get("notes");
    assert_eq!(view["headRevision"], 2, "{view}");
    assert_eq!(revisions(&view)[..2], recorded[..]);
    assert_eq!(session.docs_get("other")["found"], false);
    assert_eq!(session.deliver("r:boundary-space", "plain"), " Next one");
    session.runtime.shutdown();

    // After a restart, also while the holding document is not loaded.
    let session = Session::start_with(config(), ScriptedAdapter::new());
    assert_eq!(session.client.send(Some("doc"), update("other", 0)), taken);
    assert_eq!(session.client.send(Some("doc"), update("notes", 2)), taken);
    assert_eq!(revisions(&session.docs_get("notes"))[..2], recorded[..]);
    session.runtime.shutdown();

    let store = StoreV2::open(root.path()).expect("store reopens");
    let document = store.get_document("notes").unwrap().expect("document row");
    let row = &document.revisions[1];
    assert_eq!(row.rev_id, "r:boundary-space");
    assert_eq!(row.text, " Next one");
    assert_eq!(row.disposition.as_deref(), Some("derived"));
    assert!(store.get_document("other").unwrap().is_none());
}
