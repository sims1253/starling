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
use crate::machine::capture::TakeRegistry;
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
    jobs: HashMap<String, Job>,
    active: HashSet<String>,
    /// Most recent job id (drives the snapshot's wire view).
    latest: Option<String>,
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
        let state = latest_job
            .map(|job| job.core.state().to_string())
            .unwrap_or_else(|| "Idle".to_string());
        let violations = latest_job
            .map(|job| job.core.view().violations)
            .unwrap_or_default();
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
        if resolved {
            let _ = self.bus.emit(outcome_event, Some(&job_id));
        }
        if rejection.is_none() {
            self.waiting.push_back(job_id);
            let _ = reply.try_send(Ok(Receipt::Accepted));
            self.try_dispatch();
        } else {
            let _ = reply.try_send(Ok(Receipt::Accepted));
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
            match self.load_audio(&job_id) {
                Ok(wav) => {
                    if let Some(job) = self.jobs.get_mut(&job_id) {
                        let _ = job.core.advance_internal("Recognizing");
                    }
                    self.active.insert(job_id.clone());
                    self.spawn_worker(job_id, wav);
                }
                Err(reason) => {
                    // Loading failed: the take's audio could not be
                    // prepared — a non-retryable job failure.
                    self.emit(
                        &job_id,
                        Event::JobsFailed {
                            reason,
                            retryable: false,
                        },
                    );
                }
            }
        }
    }

    fn load_audio(&self, job_id: &str) -> Result<Vec<u8>, String> {
        let capture_ref = self
            .jobs
            .get(job_id)
            .map(|job| job.capture_ref.clone())
            .ok_or_else(|| "job vanished".to_string())?;
        let take = {
            let registry = self.registry.lock().expect("take registry lock");
            registry.get(&capture_ref).cloned()
        };
        let take = take.ok_or_else(|| "capture audio unavailable".to_string())?;
        take.to_wav()
    }

    fn spawn_worker(&mut self, job_id: String, wav: Vec<u8>) {
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
                    let _ = inbox.try_send(JobsMsg::Worker {
                        job: worker_job.clone(),
                        report,
                    });
                };
                let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let mut on_partial = |partial: Partial| {
                        if !cancel.load(Ordering::Acquire) {
                            report(WorkerReport::Partial(partial));
                        }
                    };
                    provider.recognize(wav, &worker_job, &mut on_partial)
                }));
                match outcome {
                    Ok(result) => report(WorkerReport::Done(result)),
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
                self.try_dispatch();
            }
            WorkerReport::Done(ProviderOutcome::Failed { reason, retryable }) => {
                if cancelled {
                    self.active.remove(&job_id);
                    return;
                }
                self.emit(&job_id, Event::JobsFailed { reason, retryable });
                self.active.remove(&job_id);
                self.try_dispatch();
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
