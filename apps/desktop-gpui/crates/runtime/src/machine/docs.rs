//! The documents / revisions machine (§2.4): the document service actor.
//!
//! Per head update `Validating → Committed | Conflicted`, between updates
//! `Steady` (the `Committed → Steady` edge the fixtures mark with
//! `$advance` is taken by the actor right after `docs.headUpdated`).
//! `docs.updateHead` is a compare-and-swap on `expectedBase`; a conflict
//! retains the candidate revision for explicit user choice (nothing is
//! overwritten), and turn order is the explicit `turnSeq`.
//!
//! Persistence seam: [`DocumentStore`]. `store_v2` carries the
//! `documents`/`revisions` tables but exposes no public API for them yet
//! (I5 wiring), so the default store is in-memory and the trait is the
//! seam a v2 implementation will plug into.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use serde_json::{json, Value};

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

/// The persistence seam for documents and revisions. `store_v2`'s
/// document tables are not yet surfaced as a public API (I5); the
/// in-memory default keeps the machine honest about that gap instead of
/// faking rows.
pub trait DocumentStore: Send + Sync {
    fn upsert_document(&self, doc_id: &str, name: &str, head_revision: u64, turn_seq: u32)
        -> Result<(), String>;
    fn store_revision(&self, doc_id: &str, revision: &Revision, slot: RevisionSlot)
        -> Result<(), String>;
    fn bump_turn(&self, doc_id: &str, turn_seq: u32) -> Result<(), String>;
    fn describe(&self) -> String;
}

/// The default in-memory document store (see [`DocumentStore`]).
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

    fn handle_command(&mut self, inbound: Inbound) {
        let super::Inbound { corr, command, reply, .. } = inbound;
        let corr = corr.unwrap_or_else(|| "docs-anon".to_string());
        match command {
            Command::DocsGet { doc_id, page } => {
                match self.core.commit_command("docs.get", Some(corr.clone())) {
                    Ok(_) => {
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
                        let _ = self.store.bump_turn(&doc_id, turn_seq);
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
            let _ = self.store.upsert_document(&doc_id, &record.name, new_head, record.turn_seq);
            let _ = self.store.store_revision(&doc_id, &committed, RevisionSlot::Committed);
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
            let _ = self
                .store
                .store_revision(&doc_id, &revision, RevisionSlot::Preserved);
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
