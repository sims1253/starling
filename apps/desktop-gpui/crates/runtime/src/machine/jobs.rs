//! The inference jobs machine (§2.2): the job scheduler actor with
//! supervised workers.
//!
//! **Machine mapping.** The frozen v1 table models *one job's* lifecycle
//! (`Queued → Dispatched → Loading → Recognizing → [Transforming] →
//! Completed | Failed | Cancelled`, `Rejected` at admission, pre-job
//! `Idle`): the fixtures never interleave two live jobs on one trace, and
//! the oracle's `jobs.submit` is legal only from `Idle`/terminal states.
//! The live scheduler therefore keeps **one [`MachineCore`] per job**
//! (created in `Idle` at submit, so every job's own stream replays green)
//! while the real work — the bounded waiting queue, `maxConcurrent`
//! workers, per-route caps — is scheduler state around those cores.
//!
//! **Admission (I0):** `jobs.submit` is answered by exactly one of
//! `jobs.queued` or `jobs.rejected{reason}` on the same `corr`, with the
//! enumerated reasons `queue_full` (waiting queue at `maxQueued`),
//! `resource_limits` (executor disabled: `maxConcurrent == 0`) and
//! `duplicate_submission` (an active job already holds this `captureRef`).
//!
//! **Capture independence (E17 AC):** the scheduler is its own actor and
//! its workers are its own threads — no `jobs.*` message participates in
//! any capture exit; a crashed worker demotes its job to
//! `Failed{worker_crash, retryable}` without touching capture or history.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use crate::bus::EventBus;
use crate::machine::capture::{TakeRecord, TakeRegistry};
use crate::machine::{Inbound, MachineCore, Receipt, Rejection};
use crate::protocol::tables::JOBS;
use crate::protocol::{Command, CompletionData, Event, JobLimits, RejectReason};

use super::context::FrozenRoutes;
use crate::provider::{Partial, ProviderOutcome, TranscriptionProvider};

/// Messages the scheduler receives.
pub enum JobsMsg {
    Command(Inbound),
    /// A worker's report about its job.
    Worker { job: String, report: WorkerReport },
    Shutdown,
}

/// What a worker posts back.
pub enum WorkerReport {
    Partial(Partial),
    Done(ProviderOutcome),
    /// The take's audio could not be prepared for recognition: the WAV
    /// encode failed on the worker (the `Loading` failure, which used to
    /// run synchronously on the scheduler loop — issue #216 moved it onto
    /// the worker so a multi-minute encode cannot stall submits, cancels
    /// and limit changes). A non-retryable job failure.
    LoadFailed(String),
}

struct Job {
    capture_ref: String,
    route: String,
    #[allow(dead_code)]
    budget: String,
    core: MachineCore,
    cancel: Arc<AtomicBool>,
}

/// The scheduler's projection for [`crate::RuntimeSnapshot`].
#[derive(Debug, Clone, serde::Serialize)]
pub struct JobsSnapshot {
    /// The most recent job's machine state (v1's wire view).
    pub state: String,
    pub limits: JobLimits,
    pub waiting: usize,
    pub active: usize,
    pub jobs: Vec<JobSummary>,
    pub violations: Vec<String>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct JobSummary {
    pub job: String,
    pub capture_ref: String,
    pub route: String,
    pub state: String,
}

/// The job scheduler actor.
pub struct JobsActor {
    inbox: crate::channel::Receiver<JobsMsg>,
    /// Workers post their reports here (a sender clone of the inbox).
    worker_inbox: crate::channel::Sender<JobsMsg>,
    bus: Arc<EventBus>,
    view: Arc<Mutex<JobsSnapshot>>,
    provider: Arc<dyn TranscriptionProvider>,
    registry: TakeRegistry,
    frozen_routes: FrozenRoutes,
    limits: JobLimits,
    waiting: VecDeque<String>,
    /// Live and in-flight jobs only: an entry is removed when its job
    /// reaches a terminal outcome (issue #216 — the map used to grow one
    /// `MachineCore` per submit for the process lifetime, and
    /// `duplicate_submission`'s scan paid for all of them). The guard
    /// only ever compares *active* states, so terminal removal is safe;
    /// `jobs.cancel` on a retired job answers `UnknownJob` instead of
    /// `IllegalInState` (a documented consequence: the job is gone, not
    /// merely finished).
    jobs: HashMap<String, Job>,
    active: HashSet<String>,
    /// Most recent job id (drives the snapshot's wire view).
    latest: Option<String>,
    /// The wire view of the most recent job, retained after its entry is
    /// retired on a terminal outcome: the projection keeps reporting
    /// `Completed`/`Failed`/... instead of snapping back to `Idle`.
    retired_state: String,
    retired_violations: Vec<String>,
}

impl JobsActor {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        inbox: crate::channel::Receiver<JobsMsg>,
        worker_inbox: crate::channel::Sender<JobsMsg>,
        bus: Arc<EventBus>,
        view: Arc<Mutex<JobsSnapshot>>,
        provider: Arc<dyn TranscriptionProvider>,
        registry: TakeRegistry,
        frozen_routes: FrozenRoutes,
        limits: JobLimits,
    ) -> JobsActor {
        JobsActor {
            inbox,
            worker_inbox,
            bus,
            view,
            provider,
            registry,
            frozen_routes,
            limits,
            waiting: VecDeque::new(),
            jobs: HashMap::new(),
            active: HashSet::new(),
            latest: None,
            retired_state: "Idle".to_string(),
            retired_violations: Vec::new(),
        }
    }

    pub fn run(mut self) {
        loop {
            match self.inbox.recv() {
                Ok(JobsMsg::Command(inbound)) => self.handle_command(inbound),
                Ok(JobsMsg::Worker { job, report }) => self.handle_worker(job, report),
                Ok(JobsMsg::Shutdown) | Err(crate::channel::RecvError::Closed) => break,
                Err(crate::channel::RecvError::Timeout) => {
                    unreachable!("recv has no timeout")
                }
            }
            self.publish_view();
        }
    }

    fn publish_view(&self) {
        let latest_job = self.latest.as_ref().and_then(|id| self.jobs.get(id));
        // A retired job keeps its wire view in the snapshot: the entry is
        // gone from the map (bounded, issue #216), but the projection
        // still reports the terminal state it reached.
        let (state, violations) = match latest_job {
            Some(job) => (job.core.state().to_string(), job.core.view().violations),
            None => (self.retired_state.clone(), self.retired_violations.clone()),
        };
        let jobs = self
            .jobs
            .iter()
            .map(|(id, job)| JobSummary {
                job: id.clone(),
                capture_ref: job.capture_ref.clone(),
                route: job.route.clone(),
                state: job.core.state().to_string(),
            })
            .collect();
        *self.view.lock().expect("jobs view lock") = JobsSnapshot {
            state,
            limits: self.limits.clone(),
            waiting: self.waiting.len(),
            active: self.active.len(),
            jobs,
            violations,
        };
    }

    fn emit(&mut self, job_id: &str, event: Event) {
        let Some(job) = self.jobs.get_mut(job_id) else {
            return;
        };
        match job.core.emit_event(event.type_name(), None) {
            Ok(_) => {
                let _ = self.bus.emit(event, Some(job_id));
            }
            Err(violation) => job.core.record_violation(violation),
        }
    }

    fn handle_command(&mut self, inbound: Inbound) {
        let super::Inbound { corr, command, reply, .. } = inbound;
        match command {
            Command::JobsSubmit {
                capture_ref,
                route,
                budget,
            } => self.handle_submit(reply, corr, capture_ref, route, budget),
            Command::JobsCancel { job_id } => self.handle_cancel(reply, job_id),
            Command::JobsSetLimits(limits) => {
                // Legal in every state (from "*").
                self.limits = limits;
                let _ = reply.try_send(Ok(Receipt::Accepted));
                self.try_dispatch();
            }
            other => {
                let _ = reply.try_send(Err(Rejection::UnknownMessageType(
                    other.type_name().to_string(),
                )));
            }
        }
    }

    fn handle_submit(
        &mut self,
        reply: super::ReceiptTx,
        corr: Option<String>,
        capture_ref: String,
        route: String,
        budget: String,
    ) {
        let job_id = corr.unwrap_or_else(|| crate::bus::new_id("job"));

        // Pre-admission checks that have no v1 event (receipt-level, so
        // nothing is silently absorbed):
        // 1. The audio-leave proxy: the route must have been frozen by an
        //    earlier mode.routeFrozen.
        if !self
            .frozen_routes
            .lock()
            .expect("frozen routes lock")
            .contains_key(&route)
        {
            let _ = reply.try_send(Err(Rejection::RouteNotFrozen { route }));
            return;
        }
        // 2. The submit must reference a take this runtime captured.
        if !self
            .registry
            .lock()
            .expect("take registry lock")
            .contains_key(&capture_ref)
        {
            let _ = reply.try_send(Err(Rejection::UnknownCaptureRef { capture_ref }));
            return;
        }

        // Commit the command on a fresh per-job core (from Idle — always
        // legal), which parks it awaiting exactly one of
        // jobs.queued | jobs.rejected on this corr.
        let mut core = MachineCore::new(&JOBS);
        if let Err(violation) = core.commit_command("jobs.submit", Some(job_id.clone())) {
            core.record_violation(violation.clone());
            let _ = reply.try_send(Err(Rejection::IllegalInState {
                command: "jobs.submit".to_string(),
                state: core.state().to_string(),
                detail: violation.to_string(),
            }));
            return;
        }

        // Admission control (I0): the enumerated rejection reasons.
        let rejection = if self.limits.max_concurrent == 0 {
            Some(RejectReason::ResourceLimits)
        } else if self
            .jobs
            .values()
            .any(|job| job.capture_ref == capture_ref && is_active(job.core.state()))
        {
            Some(RejectReason::DuplicateSubmission)
        } else if self.waiting.len() >= self.limits.max_queued as usize {
            Some(RejectReason::QueueFull)
        } else {
            None
        };

        let job = Job {
            capture_ref,
            route,
            budget,
            core,
            cancel: Arc::new(AtomicBool::new(false)),
        };
        self.jobs.insert(job_id.clone(), job);
        self.latest = Some(job_id.clone());

        let outcome_event = match rejection {
            Some(reason) => Event::JobsRejected { reason },
            None => Event::JobsQueued,
        };
        let resolved = self
            .jobs
            .get_mut(&job_id)
            .and_then(|job| {
                job.core
                    .resolve_outcome(outcome_event.type_name(), Some(&job_id))
                    .ok()
            })
            .is_some();
        // Reply *before* emitting the outcome — the ordering every other
        // actor already follows (issue #216). `bus.emit` applies bounded
        // backpressure: it parks while any subscriber queue is full, so
        // emitting first can park the actor with the submit's receipt
        // still unsent. A caller that drains events and sends commands on
        // the same thread (the natural GPUI pattern) then deadlocks on
        // its own submit: the emit waits for a drain only that caller
        // will ever perform, while the caller waits for the receipt.
        // Receipt first breaks the cycle — the submit resolves, and any
        // parking that follows is ordinary backpressure the caller
        // unwinds by draining.
        let _ = reply.try_send(Ok(Receipt::Accepted));
        if resolved {
            let _ = self.bus.emit(outcome_event, Some(&job_id));
        }
        if rejection.is_none() {
            self.waiting.push_back(job_id);
            self.try_dispatch();
        } else {
            // Rejected at admission: `Rejected` is terminal, so the
            // entry (created so the outcome could resolve on its own
            // core) retires immediately instead of lingering in the map.
            self.retire(&job_id);
        }
    }

    fn handle_cancel(&mut self, reply: super::ReceiptTx, job_id: String) {
        let result = {
            let Some(job) = self.jobs.get_mut(&job_id) else {
                let _ = reply.try_send(Err(Rejection::UnknownJob { job_id }));
                return;
            };
            // v1 defines no cancel event — the state snapshot is the
            // acknowledgement.
            job.core
                .commit_command("jobs.cancel", Some(job_id.clone()))
                .map_err(|violation| {
                    job.core.record_violation(violation.clone());
                    Rejection::IllegalInState {
                        command: "jobs.cancel".to_string(),
                        state: job.core.state().to_string(),
                        detail: violation.to_string(),
                    }
                })
        };
        match result {
            Ok(_) => {
                if let Some(job) = self.jobs.get_mut(&job_id) {
                    job.cancel.store(true, Ordering::Release);
                }
                self.waiting.retain(|id| id != &job_id);
                self.active.remove(&job_id);
                let _ = reply.try_send(Ok(Receipt::Accepted));
                // `jobs.cancel` entered `Cancelled` at commit — a
                // terminal state, so the entry retires (issue #216). A
                // worker still running for it reports into the void: its
                // job is gone, so the report lands nowhere by design.
                self.retire(&job_id);
                self.try_dispatch();
            }
            Err(rejection) => {
                let _ = reply.try_send(Err(rejection));
            }
        }
    }

    /// Dispatches waiting jobs onto free worker slots (bounded by
    /// `maxConcurrent` and the per-route caps).
    fn try_dispatch(&mut self) {
        loop {
            if self.active.len() >= self.limits.max_concurrent as usize {
                return;
            }
            let per_route_active = |jobs: &HashMap<String, Job>, active: &HashSet<String>, route: &str| {
                active
                    .iter()
                    .filter(|id| jobs.get(*id).map(|job| job.route == route).unwrap_or(false))
                    .count() as u32
            };
            let route_cap = |limits: &JobLimits, route: &str| -> u32 {
                limits
                    .per_route
                    .iter()
                    .find(|entry| entry.route == route)
                    .map(|entry| entry.max_concurrent)
                    .unwrap_or(u32::MAX)
            };
            let candidates: Vec<String> = self
                .waiting
                .iter()
                .filter(|id| {
                    self.jobs.get(*id).is_some_and(|job| {
                        per_route_active(&self.jobs, &self.active, &job.route)
                            < route_cap(&self.limits, &job.route)
                            && job.core.state() == "Queued"
                    })
                })
                .cloned()
                .collect();
            let Some(job_id) = candidates.first().cloned() else {
                return;
            };
            self.waiting.retain(|id| id != &job_id);
            // Internal chain: Queued -> Dispatched -> Loading.
            let advanced = self.jobs.get_mut(&job_id).and_then(|job| {
                job.core.advance_internal("Dispatched").ok()?;
                job.core.advance_internal("Loading").ok()
            });
            if advanced.is_none() {
                continue;
            }
            // The take's audio is fetched by handle (a registry lock and
            // an `Arc` clone — microseconds); the WAV encode itself runs
            // on the worker (issue #216): encoding a multi-minute take is
            // whole-take CPU work producing tens of MB of WAV, and doing
            // it on this loop stalled every `jobs.*` command behind it
            // and deferred queued worker reports. The per-job core
            // advances to `Recognizing` now, at dispatch (the projection
            // observably sits there while the worker runs — the shape the
            // #210 storm test gates on), and a load failure comes back as
            // `WorkerReport::LoadFailed`.
            let record = {
                let registry = self.registry.lock().expect("take registry lock");
                self.jobs
                    .get(&job_id)
                    .map(|job| job.capture_ref.clone())
                    .and_then(|capture_ref| registry.get(&capture_ref).cloned())
            };
            // Per-job core advances to `Recognizing` at dispatch, BEFORE
            // the worker spawns (the projection observably sits there while
            // the worker runs — the shape the #210 storm test gates on). If
            // the internal edge were ever refused, the job must NOT run
            // with a core left behind in `Loading` — that snapshot would
            // read as in-flight forever once retired (review on #248).
            let recognized = self
                .jobs
                .get_mut(&job_id)
                .and_then(|job| job.core.advance_internal("Recognizing").ok());
            if recognized.is_none() {
                // A refused internal edge is a machine invariant break, not
                // a transient state: the candidate filter only re-selects
                // `Queued` jobs, so re-queueing could never pick this one
                // again — surface the failure and retire the entry instead
                // of leaking a forever-in-flight snapshot (review on #248).
                let state = self
                    .jobs
                    .get(&job_id)
                    .map(|job| job.core.state())
                    .unwrap_or("?")
                    .to_string();
                self.emit(
                    &job_id,
                    Event::JobsFailed {
                        reason: format!("internal dispatch refused in state {state}"),
                        retryable: false,
                    },
                );
                self.retire(&job_id);
                continue;
            }
            self.active.insert(job_id.clone());
            self.spawn_worker(job_id, record);
        }
    }

    fn spawn_worker(&mut self, job_id: String, record: Option<Arc<TakeRecord>>) {
        let provider = Arc::clone(&self.provider);
        let inbox = self.worker_inbox.clone();
        let cancel = self
            .jobs
            .get(&job_id)
            .map(|job| Arc::clone(&job.cancel))
            .unwrap_or_default();
        let worker_job = job_id.clone();
        // Supervised worker: a panic is caught here and demoted to
        // Failed{worker_crash, retryable} — it never touches capture or
        // history, and never takes the scheduler down with it.
        let spawned = std::thread::Builder::new()
            .name(format!("starling-job-{worker_job}"))
            .spawn(move || {
                let report = |report: WorkerReport| {
                    // The Result is already surfaced inside (stderr on a
                    // closed inbox); the worker can do nothing further.
                    let _ = deliver_worker_report(&inbox, &worker_job, report);
                };
                let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    // The upload WAV is encoded here, on the worker
                    // (issue #216): the encode is the take's whole audio
                    // — tens of MB of WAV for a multi-minute take — and
                    // running it on the scheduler loop stalled submits,
                    // cancels and setLimits behind it. Encoding borrows
                    // the shared record's samples (no clone to build the
                    // encoder's owned-struct parameter) and the bytes
                    // move straight into `recognize`; a failure at either
                    // the lookup or the encode is the `Loading` failure,
                    // shipped back as a report.
                    let wav = match record
                        .as_deref()
                        .map(TakeRecord::to_wav)
                        .unwrap_or_else(|| Err("capture audio unavailable".to_string()))
                    {
                        Ok(wav) => wav,
                        Err(reason) => return Err(reason),
                    };
                    let mut on_partial = |partial: Partial| {
                        if !cancel.load(Ordering::Acquire) {
                            report(WorkerReport::Partial(partial));
                        }
                    };
                    Ok(provider.recognize(wav, &worker_job, &mut on_partial))
                }));
                match outcome {
                    Ok(Ok(result)) => report(WorkerReport::Done(result)),
                    Ok(Err(reason)) => report(WorkerReport::LoadFailed(reason)),
                    Err(_) => report(WorkerReport::Done(ProviderOutcome::Failed {
                        reason: "worker_crash".to_string(),
                        retryable: true,
                    })),
                }
            });
        if spawned.is_err() {
            self.emit(
                &job_id,
                Event::JobsFailed {
                    reason: "worker_spawn_failed".to_string(),
                    retryable: true,
                },
            );
            self.active.remove(&job_id);
            self.retire(&job_id);
        }
    }

    fn handle_worker(&mut self, job_id: String, report: WorkerReport) {
        let cancelled = self
            .jobs
            .get(&job_id)
            .map(|job| job.cancel.load(Ordering::Acquire))
            .unwrap_or(true);
        match report {
            WorkerReport::Partial(partial) => {
                if !cancelled {
                    // jobs.progress is legal from Loading|Recognizing|Transforming
                    // (stability reported independently of recording
                    // completeness); the per-job core enforces it.
                    self.emit(
                        &job_id,
                        Event::JobsProgress {
                            partial: partial.text,
                            stability_hint: partial.stability_hint,
                        },
                    );
                }
            }
            WorkerReport::Done(ProviderOutcome::Completed {
                text,
                backend,
                timing_ms,
                completion_evidence,
                transformed,
            }) => {
                if cancelled {
                    // The job was cancelled while its worker ran: the
                    // result lands nowhere (state is already Cancelled and
                    // must stay so — emitting completed now would be an
                    // illegal transition, which the core would refuse).
                    self.active.remove(&job_id);
                    return;
                }
                if transformed {
                    // The provider passed through its post-recognition
                    // transform stage (internal edge).
                    if let Some(job) = self.jobs.get_mut(&job_id) {
                        let _ = job.core.advance_internal("Transforming");
                    }
                }
                self.emit(
                    &job_id,
                    Event::JobsCompleted(CompletionData {
                        attempt_id: crate::bus::new_id("att"),
                        text,
                        backend,
                        timing: timing_ms,
                        completion_evidence,
                    }),
                );
                self.active.remove(&job_id);
                self.retire(&job_id);
                self.try_dispatch();
            }
            WorkerReport::Done(ProviderOutcome::Failed { reason, retryable }) => {
                if cancelled {
                    self.active.remove(&job_id);
                    return;
                }
                self.emit(&job_id, Event::JobsFailed { reason, retryable });
                self.active.remove(&job_id);
                self.retire(&job_id);
                self.try_dispatch();
            }
            WorkerReport::LoadFailed(reason) => {
                if cancelled {
                    self.active.remove(&job_id);
                    return;
                }
                // The `Loading` failure, reported from the worker: the
                // take's audio could not be prepared for recognition —
                // a non-retryable job failure.
                self.emit(
                    &job_id,
                    Event::JobsFailed {
                        reason,
                        retryable: false,
                    },
                );
                self.active.remove(&job_id);
                self.retire(&job_id);
                self.try_dispatch();
            }
        }
    }

    /// Removes a job that reached a terminal state (Completed, Failed,
    /// Cancelled, Rejected), retaining its wire view for the snapshot.
    ///
    /// The map holds only in-flight jobs: without this, a long session
    /// leaked one `MachineCore` per submit and every admission check
    /// scanned all of them (issue #216). Terminal removal is safe for the
    /// `duplicate_submission` guard — it compares only active states —
    /// so no recent-id memory is needed; the one observable change is
    /// that `jobs.cancel` for an already-finished job answers
    /// `UnknownJob` rather than `IllegalInState` (the job is gone, not
    /// merely finished).
    fn retire(&mut self, job_id: &str) {
        if let Some(job) = self.jobs.remove(job_id) {
            let state = job.core.state();
            // The retained view is permanent — the entry is gone from the
            // map, so nothing can ever update it again. Only a TERMINAL
            // state may be frozen into it (review on #248): a retire from
            // a non-terminal state (a refused emit path) keeps the
            // previous retained view rather than freezing "in-flight"
            // forever; the debug_assert fires in testing on the invariant
            // break itself.
            debug_assert!(
                matches!(
                    state,
                    "Completed" | "Failed" | "Cancelled" | "Rejected"
                ),
                "retiring job {job_id} in non-terminal state {state}"
            );
            if matches!(state, "Completed" | "Failed" | "Cancelled" | "Rejected")
                && self.latest.as_deref() == Some(job_id)
            {
                self.retired_state = state.to_string();
                self.retired_violations = job.core.view().violations;
            }
        }
    }
}

fn is_active(state: &str) -> bool {
    matches!(
        state,
        "Queued" | "Dispatched" | "Loading" | "Recognizing" | "Transforming"
    )
}

/// Posts a worker's report into the scheduler's inbox with a delivery
/// guarantee instead of a best-effort `try_send`.
///
/// The inbox is the same bounded queue that carries every `jobs.*`
/// command, so a burst of submits can fill it while the scheduler is
/// busy — and a report dropped there would silently corrupt the
/// projection (a lost `Done` wedges the job's core in `Recognizing`
/// forever and leaks its worker slot, the failure mode of issue #210).
/// Delivery therefore mirrors [`crate::bus::EventBus::emit`]:
/// `try_send` first (the common case), then [`Sender::send_blocking`]
/// for bounded backpressure while the scheduler drains. This runs on
/// the worker's own thread, so parking applies backpressure to the
/// provider — it never blocks the scheduler loop, which is the only
/// consumer that could free the queue (no cycle, no lost report while
/// the actor lives).
///
/// `Closed` means the scheduler actor itself is gone — runtime
/// shutdown, where in-flight workers legitimately outlive the actor
/// (see [`crate::Runtime`]). There is no scheduler left that could
/// wedge or recover, so the report is surfaced on stderr and dropped
/// rather than vanishing silently.
fn deliver_worker_report(
    inbox: &crate::channel::Sender<JobsMsg>,
    job: &str,
    report: WorkerReport,
) -> Result<(), crate::channel::RecvError> {
    let kind = match &report {
        WorkerReport::Partial(_) => "partial",
        WorkerReport::Done(ProviderOutcome::Completed { .. }) => "done(completed)",
        WorkerReport::Done(ProviderOutcome::Failed { .. }) => "done(failed)",
        WorkerReport::LoadFailed(_) => "load_failed",
    };
    let sent = match inbox.try_send(JobsMsg::Worker {
        job: job.to_string(),
        report,
    }) {
        Ok(()) => Ok(()),
        // Full is recoverable: park until the scheduler makes room.
        Err(crate::channel::TrySendError::Full(message)) => inbox.send_blocking(message),
        Err(crate::channel::TrySendError::Closed(_)) => Err(crate::channel::RecvError::Closed),
    };
    if sent.is_err() {
        eprintln!(
            "starling-runtime: jobs scheduler gone before job {job} reported {kind}; report dropped"
        );
    }
    sent
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    use crate::bus::EventBus;
    use crate::machine::capture::TakeStatus;
    use crate::machine::context::FrozenRoutes;
    use crate::protocol::JobLimits;

    fn completed(text: &str) -> ProviderOutcome {
        ProviderOutcome::Completed {
            text: text.to_string(),
            backend: "fake-provider".to_string(),
            timing_ms: 12.0,
            completion_evidence: "final_decode".to_string(),
            transformed: false,
        }
    }

    #[test]
    fn report_on_a_full_inbox_parks_then_lands() {
        let (tx, rx) = crate::channel::bounded::<JobsMsg>(1);
        // Occupy the only slot so the next send faces Full.
        assert!(tx.try_send(JobsMsg::Shutdown).is_ok());
        let sender = tx.clone();
        let worker = std::thread::spawn(move || {
            deliver_worker_report(&sender, "job-1", WorkerReport::Done(completed("kept")))
        });
        // The delivery must park (backpressure), not drop the report.
        std::thread::sleep(Duration::from_millis(30));
        assert!(
            !worker.is_finished(),
            "report delivery dropped instead of parking"
        );
        assert!(matches!(rx.recv(), Ok(JobsMsg::Shutdown)));
        assert_eq!(worker.join().expect("worker thread"), Ok(()));
        match rx.recv() {
            Ok(JobsMsg::Worker {
                job,
                report: WorkerReport::Done(ProviderOutcome::Completed { text, .. }),
            }) => {
                assert_eq!(job, "job-1");
                assert_eq!(text, "kept");
            }
            _ => panic!("the parked report must land once the inbox drains"),
        }
    }

    #[test]
    fn report_on_a_closed_inbox_is_surfaced_not_silent() {
        let (tx, rx) = crate::channel::bounded::<JobsMsg>(2);
        drop(rx); // the scheduler actor is gone
        assert_eq!(
            deliver_worker_report(&tx, "job-2", WorkerReport::Done(completed("lost"))),
            Err(crate::channel::RecvError::Closed)
        );
    }

    /// Issue #216 part 1: the submit's receipt is sent *before* the
    /// `jobs.queued`/`jobs.rejected` emit. `EventBus::emit` applies
    /// bounded backpressure — it parks while any subscriber queue is
    /// full — so emitting first could park the actor with the receipt
    /// still unsent, deadlocking a caller that drains events and sends
    /// commands on the same thread (the emit waits for a drain only that
    /// caller will perform, while the caller waits on the receipt).
    ///
    /// The proof: with the only subscriber's queue already full, the
    /// receipt still arrives while the emit is parked behind it, and the
    /// outcome event only lands once the queue is drained.
    #[test]
    fn submit_receipt_precedes_the_outcome_emit_even_under_backpressure() {
        let bus = Arc::new(EventBus::new(1));
        let subscription = bus.subscribe();
        // Occupy the only subscriber slot: the next emit will park.
        bus.emit(
            Event::JobsProgress {
                partial: "prefill".into(),
                stability_hint: "stable".into(),
            },
            None,
        )
        .expect("prefill emit");

        let (inbox_tx, inbox) = crate::channel::bounded::<JobsMsg>(4);
        let worker_inbox = inbox_tx.clone();
        drop(inbox_tx);
        let registry: TakeRegistry = Arc::default();
        registry.lock().unwrap().insert(
            "take_p".into(),
            Arc::new(TakeRecord {
                id: "take_p".into(),
                device: "default-input".into(),
                policy: "push-to-talk".into(),
                samples: vec![0.0, 0.25, -0.25, 0.5],
                sample_rate: 16_000,
                gaps: vec![],
                acknowledged_samples: 4,
                final_sample_index: 4,
                journal: None,
                status: TakeStatus::Complete,
                sample_duration_ms: 0.25,
                wall_clock_ms: 1.0,
                capture_id: "cap_p".into(),
            }),
        );
        let frozen_routes: FrozenRoutes = Arc::default();
        frozen_routes
            .lock()
            .unwrap()
            .insert("local-default".into(), "code-guidance".into());
        let limits = JobLimits {
            max_queued: 2,
            max_concurrent: 1,
            per_route: vec![],
        };
        let mut actor = JobsActor::new(
            inbox,
            worker_inbox,
            Arc::clone(&bus),
            Arc::new(Mutex::new(JobsSnapshot {
                state: "Idle".into(),
                limits: limits.clone(),
                waiting: 0,
                active: 0,
                jobs: vec![],
                violations: vec![],
            })),
            crate::provider::FakeProvider::new(vec![crate::provider::FakeJob::completes_with(
                "done",
            )]),
            registry,
            frozen_routes,
            limits,
        );
        let (reply_tx, reply_rx) = crate::channel::bounded(1);
        let submitter = std::thread::spawn(move || {
            actor.handle_submit(
                reply_tx,
                Some("job-p".into()),
                "take_p".into(),
                "local-default".into(),
                "standard".into(),
            );
        });

        // The receipt must arrive while the outcome emit is parked on the
        // full subscriber queue — this is exactly the ordering the fix
        // pins, and the deadlock the old emit-first shape could produce.
        let receipt = reply_rx
            .recv_timeout(Duration::from_millis(500))
            .expect("the submit receipt must not queue behind the outcome emit");
        assert!(receipt.is_ok(), "expected Accepted, got {receipt:?}");
        std::thread::sleep(Duration::from_millis(30));
        assert!(
            !submitter.is_finished(),
            "the outcome emit should still be parked on the full subscriber queue"
        );

        // Draining unparks the actor; the queued outcome then lands.
        assert_eq!(
            subscription
                .recv_timeout(Duration::from_secs(1))
                .expect("prefill event")
                .type_name(),
            "jobs.progress"
        );
        assert_eq!(
            subscription
                .recv_timeout(Duration::from_secs(1))
                .expect("queued outcome after the drain")
                .type_name(),
            "jobs.queued"
        );
        submitter.join().expect("submitter thread");
    }

    #[test]
    fn report_parked_on_full_surfaces_closed_when_the_actor_dies() {
        let (tx, rx) = crate::channel::bounded::<JobsMsg>(1);
        assert!(tx.try_send(JobsMsg::Shutdown).is_ok());
        let sender = tx.clone();
        let worker = std::thread::spawn(move || {
            deliver_worker_report(
                &sender,
                "job-3",
                WorkerReport::Partial(Partial {
                    text: "mid-flight".to_string(),
                    stability_hint: "unstable".to_string(),
                }),
            )
        });
        std::thread::sleep(Duration::from_millis(30));
        assert!(!worker.is_finished(), "delivery should be parked on the full inbox");
        drop(rx); // shutdown while the worker waits for room
        assert_eq!(
            worker.join().expect("worker thread"),
            Err(crate::channel::RecvError::Closed)
        );
    }
}
