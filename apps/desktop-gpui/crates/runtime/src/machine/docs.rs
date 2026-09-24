//! The documents / revisions machine (§2.4): the document service actor.
//!
//! Per head update `Validating → Committed | Conflicted`, between updates
//! `Steady` (the `Committed → Steady` edge the fixtures mark with
//! `$advance` is taken by the actor right after `docs.headUpdated`).
//! `docs.updateHead` is a compare-and-swap on `expectedBase`; a conflict
//! retains the candidate revision for explicit user choice (nothing is
//! overwritten), and turn order is the explicit `turnSeq`.
//!
//! Persistence seam: [`DocumentStore`]. The I5 wiring (issue #220) lands
//! [`V2DocumentStore`] over storage v2's `documents`/`revisions` tables —
//! the host's production config opens it at the data root — while
//! [`crate::RuntimeConfig::default`] keeps the in-memory store so test
//! construction stays side-effect-free (the same philosophy as the
//! capture store's default). Writes go through on every transition and
//! state is hydrated lazily on first touch per document, so a restarted
//! machine answers `docs.get` from the durable rows and CASes against
//! the durable head — a forgotten head would let `expectedBase: 0`
//! "succeed" against a document whose durable head is 5 and silently
//! rewind it. A persistence failure is reported on stderr (this crate's
//! divergence channel — same posture as the capture store's
//! `report_divergence`) and outrun by the session state: v1 defines no
//! docs failure event, so the machine keeps its in-memory truth and the
//! divergence surfaces at the next hydration.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use serde_json::{json, Value};

use starling_dictation::store_v2::{DocumentRow, RevisionRow, StoreV2};

use crate::bus::EventBus;
use crate::machine::{Inbound, MachineCore, Receipt, Rejection};
use crate::protocol::tables::DOCS;
use crate::protocol::{Command, Event, Revision};

/// Revisions per document page served by `docs.get`.
pub const DOCS_PAGE_SIZE: usize = 16;

/// Where a revision stands in its document's history.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RevisionSlot {
    /// The document's committed head (or a past head).
    Committed,
    /// A conflict candidate retained for explicit user choice.
    Preserved,
}

impl RevisionSlot {
    /// The storage-v2 `disposition` encoding (§4 `revisions`).
    fn as_disposition(self) -> &'static str {
        match self {
            RevisionSlot::Committed => "committed",
            RevisionSlot::Preserved => "preserved",
        }
    }

    /// A disposition this build does not know (a newer writer's
    /// vocabulary) reads as committed: the row belongs to the head side
    /// of some future history, and hiding it would be the lossy choice.
    fn from_disposition(text: &str) -> RevisionSlot {
        match text {
            "preserved" => RevisionSlot::Preserved,
            "committed" => RevisionSlot::Committed,
            other => {
                eprintln!("documents machine: unknown disposition {other:?}, read as committed");
                RevisionSlot::Committed
            }
        }
    }
}

#[derive(Debug, Clone)]
struct StoredRevision {
    revision: Revision,
    slot: RevisionSlot,
}

#[derive(Debug, Clone)]
struct DocumentRecord {
    name: String,
    head_revision: u64,
    turn_seq: u32,
    revisions: Vec<StoredRevision>,
}

/// Committed heads the delivery service may prepare against
/// (`{docId → revision}`), runtime-owned.
pub type RevisionRegistry = Arc<Mutex<HashMap<String, (String, Revision)>>>;

/// A document's durable state as [`DocumentStore::load_document`] returns
/// it — the hydration shape a restarted documents machine rebuilds from.
#[derive(Debug, Clone)]
pub struct StoredDocument {
    pub name: String,
    pub head_revision: u64,
    pub turn_seq: u32,
    pub revisions: Vec<(Revision, RevisionSlot)>,
}

/// The persistence seam for documents and revisions: the write-through
/// the actor performs on every transition, plus the lazy load a
/// restarted machine hydrates from. The default `load_document` answers
/// "no durable state", so the in-memory store needs no bookkeeping.
pub trait DocumentStore: Send + Sync {
    fn upsert_document(&self, doc_id: &str, name: &str, head_revision: u64, turn_seq: u32)
        -> Result<(), String>;
    fn store_revision(&self, doc_id: &str, revision: &Revision, slot: RevisionSlot)
        -> Result<(), String>;
    /// Advances the head and stores its committed revision. Stores that
    /// can should do this atomically (see [`V2DocumentStore`]).
    fn commit_head(
        &self,
        doc_id: &str,
        name: &str,
        head_revision: u64,
        turn_seq: u32,
        revision: &Revision,
    ) -> Result<(), String> {
        self.upsert_document(doc_id, name, head_revision, turn_seq)?;
        self.store_revision(doc_id, revision, RevisionSlot::Committed)
    }
    fn bump_turn(&self, doc_id: &str, turn_seq: u32) -> Result<(), String>;
    /// One document's durable rows; `Ok(None)` when this store has never
    /// held the document. Hydration consults this exactly once per
    /// document per session (the actor caches the answer).
    fn load_document(&self, doc_id: &str) -> Result<Option<StoredDocument>, String> {
        let _ = doc_id;
        Ok(None)
    }
    fn describe(&self) -> String;
}

/// The default in-memory document store: session-scoped by design — a
/// document it never saw answers "no durable state" on load, so a
/// restarted runtime starts empty exactly as an ephemeral store should.
#[derive(Default)]
pub struct MemoryDocumentStore {
    state: Mutex<Vec<String>>,
}

impl MemoryDocumentStore {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }
}

impl DocumentStore for MemoryDocumentStore {
    fn upsert_document(
        &self,
        doc_id: &str,
        name: &str,
        head_revision: u64,
        turn_seq: u32,
    ) -> Result<(), String> {
        self.state
            .lock()
            .expect("document store lock")
            .push(format!("doc {doc_id} name={name} head={head_revision} turn={turn_seq}"));
        Ok(())
    }
    fn store_revision(
        &self,
        doc_id: &str,
        revision: &Revision,
        slot: RevisionSlot,
    ) -> Result<(), String> {
        self.state.lock().expect("document store lock").push(format!(
            "rev {doc_id}/{} slot={:?}",
            revision.rev_id, slot
        ));
        Ok(())
    }
    fn bump_turn(&self, doc_id: &str, turn_seq: u32) -> Result<(), String> {
        self.state
            .lock()
            .expect("document store lock")
            .push(format!("turn {doc_id} -> {turn_seq}"));
        Ok(())
    }
    fn describe(&self) -> String {
        "in-memory".to_string()
    }
}

/// The storage-v2 documents store (I5 wiring, issue #220): the I3 seam
/// over `StoreV2`'s `documents`/`revisions` tables. `sources_json`
/// carries the I3 [`Revision`] provenance the §4 schema has no column
/// for — `sourceAttemptIds` + `instructionTemplateId` as one JSON
/// object, parsed back defensively on load (a missing or malformed
/// object reads as empty provenance, never as a row that cannot be
/// served).
pub struct V2DocumentStore {
    store: Mutex<StoreV2>,
}

impl V2DocumentStore {
    pub fn open(root: impl Into<std::path::PathBuf>) -> Result<Self, String> {
        let store = StoreV2::open(root).map_err(|err| err.to_string())?;
        Ok(V2DocumentStore {
            store: Mutex::new(store),
        })
    }

    /// The §4 provenance encoding of an I3 revision.
    fn sources_json(revision: &Revision) -> String {
        json!({
            "attempts": revision.source_attempt_ids,
            "instructionTemplateId": revision.instruction_template_id,
        })
        .to_string()
    }

    /// Parses [`Self::sources_json`] back; unknown shapes (a newer
    /// writer's vocabulary) degrade to empty provenance rather than
    /// failing the document.
    fn parse_sources(text: Option<&str>) -> (Vec<String>, String) {
        let Some(text) = text else {
            return (Vec::new(), String::new());
        };
        let Ok(value) = serde_json::from_str::<Value>(text) else {
            eprintln!("documents machine: unparsable sources_json, provenance lost: {text:?}");
            return (Vec::new(), String::new());
        };
        let attempts = value
            .get("attempts")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default();
        let template = value
            .get("instructionTemplateId")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        (attempts, template)
    }

    fn revision_row(doc_id: &str, revision: &Revision, slot: RevisionSlot) -> RevisionRow {
        RevisionRow {
            rev_id: revision.rev_id.clone(),
            doc_id: doc_id.to_string(),
            base_revision: Some(revision.base_revision),
            sources_json: Some(Self::sources_json(revision)),
            text: revision.text.clone(),
            status: revision.status.clone(),
            provenance: Some(revision.provenance.clone()),
            disposition: Some(slot.as_disposition().to_string()),
        }
    }

    fn row_to_stored(row: &RevisionRow) -> (Revision, RevisionSlot) {
        let (source_attempt_ids, instruction_template_id) =
            Self::parse_sources(row.sources_json.as_deref());
        (
            Revision {
                rev_id: row.rev_id.clone(),
                base_revision: row.base_revision.unwrap_or(0),
                source_attempt_ids,
                instruction_template_id,
                text: row.text.clone(),
                status: row.status.clone(),
                provenance: row.provenance.clone().unwrap_or_default(),
            },
            RevisionSlot::from_disposition(row.disposition.as_deref().unwrap_or("committed")),
        )
    }
}

impl DocumentStore for V2DocumentStore {
    fn upsert_document(
        &self,
        doc_id: &str,
        name: &str,
        head_revision: u64,
        turn_seq: u32,
    ) -> Result<(), String> {
        self.store
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .upsert_document(doc_id, name, head_revision, turn_seq)
            .map_err(|err| err.to_string())
    }
    fn store_revision(
        &self,
        doc_id: &str,
        revision: &Revision,
        slot: RevisionSlot,
    ) -> Result<(), String> {
        let row = Self::revision_row(doc_id, revision, slot);
        self.store
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .store_document_revision(&row)
            .map_err(|err| err.to_string())
    }
    fn commit_head(
        &self,
        doc_id: &str,
        name: &str,
        head_revision: u64,
        turn_seq: u32,
        revision: &Revision,
    ) -> Result<(), String> {
        let row = Self::revision_row(doc_id, revision, RevisionSlot::Committed);
        self.store
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .commit_document_head(name, head_revision, turn_seq, &row)
            .map_err(|err| err.to_string())
    }
    fn bump_turn(&self, doc_id: &str, turn_seq: u32) -> Result<(), String> {
        self.store
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .bump_document_turn(doc_id, turn_seq)
            .map_err(|err| err.to_string())
    }
    fn load_document(&self, doc_id: &str) -> Result<Option<StoredDocument>, String> {
        let document: Option<DocumentRow> = self
            .store
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get_document(doc_id)
            .map_err(|err| err.to_string())?;
        Ok(document.map(|row| StoredDocument {
            name: row.name,
            head_revision: row.head_revision,
            turn_seq: row.turn_seq,
            revisions: row.revisions.iter().map(Self::row_to_stored).collect(),
        }))
    }
    fn describe(&self) -> String {
        "storage-v2".to_string()
    }
}

/// The documents machine's divergence channel, same posture as the
/// capture store's: a persistence failure the session already moved past
/// is reported on stderr (this crate has no logging facade) and the
/// in-memory state stays the session's truth; hydration after a restart
/// surfaces what actually landed.
fn report_store_failure(operation: &str, doc: &str, detail: String) {
    eprintln!("documents machine: {operation} for {doc:?} failed durably: {detail}");
}

/// Messages the document actor receives.
pub enum DocsMsg {
    Command(Inbound),
    Shutdown,
}

/// The document service actor.
pub struct DocsActor {
    inbox: crate::channel::Receiver<DocsMsg>,
    bus: Arc<EventBus>,
    view: super::ViewSlot,
    core: MachineCore,
    documents: HashMap<String, DocumentRecord>,
    revisions: RevisionRegistry,
    store: Arc<dyn DocumentStore>,
}

impl DocsActor {
    pub fn new(
        inbox: crate::channel::Receiver<DocsMsg>,
        bus: Arc<EventBus>,
        view: super::ViewSlot,
        revisions: RevisionRegistry,
        store: Arc<dyn DocumentStore>,
    ) -> DocsActor {
        DocsActor {
            inbox,
            bus,
            view,
            core: MachineCore::new(&DOCS),
            documents: HashMap::new(),
            revisions,
            store,
        }
    }

    pub fn run(mut self) {
        loop {
            match self.inbox.recv() {
                Ok(DocsMsg::Command(inbound)) => self.handle_command(inbound),
                Ok(DocsMsg::Shutdown) | Err(crate::channel::RecvError::Closed) => break,
                Err(crate::channel::RecvError::Timeout) => {
                    unreachable!("recv has no timeout")
                }
            }
            *self.view.lock().expect("docs view lock") = self.core.view();
        }
    }

    fn emit(&mut self, event: Event, corr: &str) {
        match self.core.emit_event(event.type_name(), None) {
            Ok(_) => {
                let _ = self.bus.emit(event, Some(corr));
            }
            Err(violation) => self.core.record_violation(violation),
        }
    }

    /// Hydrates `doc_id` from the durable store on its first touch this
    /// session. A restarted machine must not answer `docs.get` from an
    /// empty map, and — the sharper invariant — must not CAS against a
    /// forgotten head: `expected_base: 0` would "succeed" against a
    /// document whose durable head is 5 and silently rewind it. At most
    /// one load per document per session; a load failure is reported and
    /// treated as no durable state (the session's own writes still go
    /// through; the divergence is diagnosable on stderr).
    fn hydrate(&mut self, doc_id: &str) {
        if self.documents.contains_key(doc_id) {
            return;
        }
        match self.store.load_document(doc_id) {
            Ok(Some(stored)) => {
                // Re-publish the durable committed revisions so
                // delivery.prepare resolves them after a restart (the
                // registry is the delivery actor's only lookup source).
                self.revisions
                    .lock()
                    .expect("revision registry lock")
                    .extend(
                        stored
                            .revisions
                            .iter()
                            .filter(|(_, slot)| *slot == RevisionSlot::Committed)
                            .map(|(revision, _)| {
                                (revision.rev_id.clone(), (doc_id.to_string(), revision.clone()))
                            }),
                    );
                self.documents.insert(
                    doc_id.to_string(),
                    DocumentRecord {
                        name: stored.name,
                        head_revision: stored.head_revision,
                        turn_seq: stored.turn_seq,
                        revisions: stored
                            .revisions
                            .into_iter()
                            .map(|(revision, slot)| StoredRevision { revision, slot })
                            .collect(),
                    },
                );
            }
            Ok(None) => {}
            Err(detail) => report_store_failure("load", doc_id, detail),
        }
    }

    fn handle_command(&mut self, inbound: Inbound) {
        let super::Inbound { corr, command, reply, .. } = inbound;
        let corr = corr.unwrap_or_else(|| "docs-anon".to_string());
        match command {
            Command::DocsGet { doc_id, page } => {
                match self.core.commit_command("docs.get", Some(corr.clone())) {
                    Ok(_) => {
                        self.hydrate(&doc_id);
                        let view = self.get_view(&doc_id, page);
                        let _ = reply.try_send(Ok(Receipt::Served(view)));
                    }
                    Err(violation) => {
                        self.core.record_violation(violation.clone());
                        let _ = reply.try_send(Err(illegal("docs.get", self.core.state(), violation)));
                    }
                }
            }
            Command::DocsUpdateHead {
                doc_id,
                expected_base,
                new_revision,
            } => {
                match self.core.commit_command("docs.updateHead", Some(corr.clone())) {
                    Ok(_) => {
                        let _ = reply.try_send(Ok(Receipt::Accepted));
                        self.apply_cas(corr, doc_id, expected_base, new_revision);
                    }
                    Err(violation) => {
                        self.core.record_violation(violation.clone());
                        let _ = reply.try_send(Err(illegal(
                            "docs.updateHead",
                            self.core.state(),
                            violation,
                        )));
                    }
                }
            }
            Command::DocsAppendTurn { doc_id, take_ref: _ } => {
                match self.core.commit_command("docs.appendTurn", Some(corr.clone())) {
                    Ok(_) => {
                        let _ = reply.try_send(Ok(Receipt::Accepted));
                        self.hydrate(&doc_id);
                        let record = self
                            .documents
                            .entry(doc_id.clone())
                            .or_insert_with(|| DocumentRecord {
                                name: doc_id.clone(),
                                head_revision: 0,
                                turn_seq: 0,
                                revisions: Vec::new(),
                            });
                        record.turn_seq += 1;
                        let turn_seq = record.turn_seq;
                        if let Err(detail) = self.store.bump_turn(&doc_id, turn_seq) {
                            report_store_failure("bump_turn", &doc_id, detail);
                        }
                        // docs.turnAppended keeps the state (outcome target
                        // None): resolve the pending outcome (corr-checked,
                        // transition recorded by the core), then deliver.
                        match self.core.resolve_outcome("docs.turnAppended", Some(&corr)) {
                            Ok(_) => {
                                let _ = self.bus.emit(
                                    Event::DocsTurnAppended { turn_seq },
                                    Some(&corr),
                                );
                            }
                            Err(violation) => self.core.record_violation(violation),
                        }
                    }
                    Err(violation) => {
                        self.core.record_violation(violation.clone());
                        let _ = reply.try_send(Err(illegal(
                            "docs.appendTurn",
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

    /// The compare-and-swap: base match commits (`docs.headUpdated`),
    /// mismatch retains the candidate (`docs.headConflict`,
    /// `candidatePreserved: true`).
    fn apply_cas(&mut self, corr: String, doc_id: String, expected_base: u64, revision: Revision) {
        // The durable head is the CAS's truth for a document this session
        // has not touched yet — hydrate before reading `actual`.
        self.hydrate(&doc_id);
        let actual = self
            .documents
            .get(&doc_id)
            .map(|record| record.head_revision)
            .unwrap_or(0);
        if expected_base == actual {
            let new_head = actual + 1;
            let record = self
                .documents
                .entry(doc_id.clone())
                .or_insert_with(|| DocumentRecord {
                    name: doc_id.clone(),
                    head_revision: 0,
                    turn_seq: 0,
                    revisions: Vec::new(),
                });
            record.head_revision = new_head;
            let mut committed = revision.clone();
            committed.base_revision = actual;
            record.revisions.push(StoredRevision {
                revision: committed.clone(),
                slot: RevisionSlot::Committed,
            });
            if let Err(detail) =
                self.store
                    .commit_head(&doc_id, &record.name, new_head, record.turn_seq, &committed)
            {
                report_store_failure("commit_head", &doc_id, detail);
            }
            self.revisions
                .lock()
                .expect("revision registry lock")
                .insert(committed.rev_id.clone(), (doc_id.clone(), committed));
            // headUpdated is a free event from Validating (updateHead is a
            // to-state command, not an outcome-pending one).
            self.emit(
                Event::DocsHeadUpdated {
                    doc_id,
                    head_revision: new_head,
                },
                &corr,
            );
            // Committed -> Steady (runtime-internal; the fixtures' $advance).
            if self.core.state() == "Committed" {
                if let Err(violation) = self.core.advance_internal("Steady") {
                    self.core.record_violation(violation);
                }
            }
        } else {
            let record = self
                .documents
                .entry(doc_id.clone())
                .or_insert_with(|| DocumentRecord {
                    name: doc_id.clone(),
                    head_revision: actual,
                    turn_seq: 0,
                    revisions: Vec::new(),
                });
            let mut candidate = revision.clone();
            candidate.base_revision = expected_base;
            // The candidate is retained for explicit user choice — the
            // conflict persists until the user acts.
            record.revisions.push(StoredRevision {
                revision: candidate,
                slot: RevisionSlot::Preserved,
            });
            if let Err(detail) = self
                .store
                .store_revision(&doc_id, &revision, RevisionSlot::Preserved)
            {
                report_store_failure("store_revision(preserved)", &doc_id, detail);
            }
            // headConflict is a free event from Validating.
            self.emit(
                Event::DocsHeadConflict {
                    expected: expected_base,
                    actual,
                    candidate_preserved: true,
                },
                &corr,
            );
        }
    }

    /// The `docs.get` view: document head + one page of revisions (page 0
    /// is the first page, newest last).
    fn get_view(&self, doc_id: &str, page: u32) -> Value {
        match self.documents.get(doc_id) {
            None => json!({ "docId": doc_id, "found": false }),
            Some(record) => {
                let start = page as usize * DOCS_PAGE_SIZE;
                let revisions: Vec<Value> = record
                    .revisions
                    .iter()
                    .skip(start)
                    .take(DOCS_PAGE_SIZE)
                    .map(|stored| {
                        json!({
                            "revId": stored.revision.rev_id,
                            "baseRevision": stored.revision.base_revision,
                            "slot": match stored.slot {
                                RevisionSlot::Committed => "committed",
                                RevisionSlot::Preserved => "preserved",
                            },
                            "text": stored.revision.text,
                            "provenance": stored.revision.provenance,
                        })
                    })
                    .collect();
                json!({
                    "docId": doc_id,
                    "found": true,
                    "name": record.name,
                    "headRevision": record.head_revision,
                    "turnSeq": record.turn_seq,
                    "revisions": revisions,
                    "page": page,
                })
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
