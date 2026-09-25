//! The inference provider seam. The production adapter **reuses
//! `starling-dictation`'s `client.rs`** (the design's code disposition:
//! endpoint validation, bounded timeouts, error taxonomy and tested
//! parsers) as [`StarlingProvider`]; [`FakeProvider`] is the scripted test
//! double the integration tests drive. The processing seam (#294) sits
//! next to it: [`TransformProcessor`], with [`PipelineProcessor`] over the
//! `starling-processing` pipeline in production and [`FakeProcessor`] in
//! tests.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use starling_dictation::client::{ClientError, StarlingClient};
use starling_processing::contract::{
    Failure, FailureReason, ProviderKind, ResultStatus, Timing, TransformRequest, TransformResult,
};
use starling_processing::pipeline::{self, failure_result, Clock, Registry};

/// The cancellation signal the jobs scheduler hands a provider (issue
/// #251): [`crate::machine::jobs`] creates one per job, `jobs.cancel`
/// trips it, and the production client aborts its in-flight HTTP request
/// on it. Re-exported from the dictation client, which owns the
/// request-abort machinery.
pub use starling_dictation::client::CancelToken;

/// One partial-transcript observation streamed by a provider
/// (`jobs.progress{partial, stabilityHint}` — stability is reported
/// independently of recording completeness).
#[derive(Clone)]
pub struct Partial {
    pub text: String,
    pub stability_hint: String,
}

/// A provider's final outcome.
#[derive(Clone)]
pub enum ProviderOutcome {
    Completed {
        text: String,
        backend: String,
        timing_ms: f64,
        completion_evidence: String,
    },
    Failed {
        reason: String,
        retryable: bool,
    },
}

/// Why a provider call failed, mapped to v1 `jobs.failed{reason,
/// retryable}` tokens.
pub fn failure_from_client_error(error: &ClientError) -> (String, bool) {
    match error {
        ClientError::Input(_) => ("invalid_input".to_string(), false),
        ClientError::Transport(_) => ("transport_error".to_string(), true),
        ClientError::Timeout(_) => ("timeout".to_string(), true),
        ClientError::Redirect(_) => ("redirect_blocked".to_string(), false),
        ClientError::Http { status, .. } => (format!("http_{status}"), *status >= 500),
        ClientError::Protocol(_) => ("protocol_error".to_string(), false),
        // An oversized body is deterministic server behavior: the same
        // request would produce the same wall of bytes, so retrying
        // cannot help (issue #235).
        ClientError::ResponseTooLarge(_) => ("response_too_large".to_string(), false),
        // A cancelled call is not a server condition: non-retryable, and
        // the scheduler drops it anyway (the job is already Cancelled).
        ClientError::Cancelled => ("cancelled".to_string(), false),
    }
}

/// What the jobs executor's workers call. Blocking by design (the
/// production client is blocking); workers run on their own threads and
/// are supervised by the scheduler.
pub trait TranscriptionProvider: Send + Sync {
    /// Streams partials (when the provider can) and returns the final
    /// outcome. `request_id` is the job's correlation id.
    ///
    /// `cancel` is the job's cancellation flag (issue #251): the
    /// scheduler trips it on `jobs.cancel`, and a well-behaved provider
    /// watches it, aborts its in-flight work and returns early instead of
    /// running to completion. [`ProviderOutcome::Failed`] with the
    /// `cancelled` reason is the conventional early return; the scheduler
    /// never delivers it (a cancelled job is already retired), so the
    /// value only matters to direct callers.
    fn recognize(
        &self,
        wav: Vec<u8>,
        request_id: &str,
        on_partial: &mut dyn FnMut(Partial),
        cancel: &CancelToken,
    ) -> ProviderOutcome;
}

/// The production provider: `starling-dictation`'s `StarlingClient`
/// (endpoint validation, bounded timeouts, error
/// taxonomy) behind the runtime's adapter trait.
pub struct StarlingProvider {
    client: StarlingClient,
}

impl StarlingProvider {
    pub fn new(endpoint: &str, model: &str) -> Result<Self, ClientError> {
        Ok(StarlingProvider {
            client: StarlingClient::new(endpoint, model)?,
        })
    }
}

impl TranscriptionProvider for StarlingProvider {
    fn recognize(
        &self,
        wav: Vec<u8>,
        request_id: &str,
        _on_partial: &mut dyn FnMut(Partial),
        cancel: &CancelToken,
    ) -> ProviderOutcome {
        let started = Instant::now();
        // The owned WAV moves into the shared buffer the client uploads
        // zero-copy (issue #235) — no second copy of the recording.
        let wav = Arc::new(wav);
        match self
            .client
            .transcribe_with_cancel(wav, request_id, Some(cancel))
        {
            Ok(result) => ProviderOutcome::Completed {
                text: result.text,
                backend: "starling-server".to_string(),
                timing_ms: started.elapsed().as_secs_f64() * 1000.0,
                completion_evidence: "final_decode".to_string(),
            },
            Err(error) => {
                let (reason, retryable) = failure_from_client_error(&error);
                ProviderOutcome::Failed { reason, retryable }
            }
        }
        // The blocking HTTP client cannot stream partials; the adapter is
        // honest about that and emits none.
    }
}

/// The default provider for [`crate::RuntimeConfig`]: honest absence. It
/// fails every job with `no_provider_configured` (not retryable) rather
/// than pretending an endpoint exists.
pub struct UnconfiguredProvider;

impl TranscriptionProvider for UnconfiguredProvider {
    fn recognize(
        &self,
        _wav: Vec<u8>,
        _request_id: &str,
        _on_partial: &mut dyn FnMut(Partial),
        _cancel: &CancelToken,
    ) -> ProviderOutcome {
        ProviderOutcome::Failed {
            reason: "no_provider_configured".to_string(),
            retryable: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Processing (#294)
// ---------------------------------------------------------------------------

/// The processing seam: what a transform job's worker calls. Blocking by
/// design, like [`TranscriptionProvider`]; `cancel` is the job's token
/// and a well-behaved processor aborts its in-flight request on it.
/// `on_partial` receives the output streamed so far (the whole text, not
/// a delta), surfaced as `jobs.progress`.
pub trait TransformProcessor: Send + Sync {
    fn transform(
        &self,
        request: &TransformRequest,
        queued_ms: f64,
        on_partial: &mut dyn FnMut(Partial),
        cancel: &CancelToken,
    ) -> TransformResult;
}

/// The production processor: the `starling-processing` pipeline over the
/// providers an embedder configured. A request naming a provider that is
/// not configured fails `provider_unavailable`; it never runs elsewhere.
pub struct PipelineProcessor {
    registry: Registry,
}

impl PipelineProcessor {
    pub fn new(registry: Registry) -> PipelineProcessor {
        PipelineProcessor { registry }
    }
}

impl TransformProcessor for PipelineProcessor {
    fn transform(
        &self,
        request: &TransformRequest,
        queued_ms: f64,
        on_partial: &mut dyn FnMut(Partial),
        cancel: &CancelToken,
    ) -> TransformResult {
        let clock = Clock {
            queued_ms,
            stopped_at: None,
        };
        let provider = if request.provider.kind == ProviderKind::Builtin {
            None
        } else {
            match self.registry.get(&request.provider.id) {
                Some(provider) => Some(provider),
                None => {
                    return failure_result(
                        request,
                        Failure::new(
                            FailureReason::ProviderUnavailable,
                            false,
                            format!("provider {} is not configured", request.provider.id),
                        ),
                        Timing {
                            queued_ms,
                            processing_ms: 0.0,
                            stop_to_result_ms: None,
                        },
                    )
                }
            }
        };
        // `jobs.progress` carries the whole text so far. The copy per delta
        // is bounded by the request's output cap, which the pipeline
        // enforces.
        let mut so_far = String::new();
        let mut on_delta = |delta: &str| {
            so_far.push_str(delta);
            on_partial(Partial {
                text: so_far.clone(),
                stability_hint: "growing".to_string(),
            });
        };
        pipeline::run(request, provider.as_deref(), clock, &mut on_delta, cancel)
    }
}

/// The default processor for [`crate::RuntimeConfig`]: honest absence.
/// Every transform fails `provider_unavailable` (not retryable).
pub struct UnconfiguredProcessor;

impl TransformProcessor for UnconfiguredProcessor {
    fn transform(
        &self,
        request: &TransformRequest,
        queued_ms: f64,
        _on_partial: &mut dyn FnMut(Partial),
        _cancel: &CancelToken,
    ) -> TransformResult {
        failure_result(
            request,
            Failure::new(
                FailureReason::ProviderUnavailable,
                false,
                "no processing configured",
            ),
            Timing {
                queued_ms,
                processing_ms: 0.0,
                stop_to_result_ms: None,
            },
        )
    }
}

/// One scripted transform for [`FakeProcessor`].
#[derive(Clone)]
pub struct FakeTransform {
    /// Output streamed before the outcome (each entry is a delta).
    pub deltas: Vec<String>,
    /// The output text, or the typed failure.
    pub outcome: Result<String, Failure>,
    /// Simulated work, polled against the cancel token.
    pub work_ms: u64,
}

impl FakeTransform {
    pub fn returns(text: &str) -> FakeTransform {
        FakeTransform {
            deltas: vec![text.to_string()],
            outcome: Ok(text.to_string()),
            work_ms: 5,
        }
    }

    pub fn fails(reason: FailureReason, retryable: bool) -> FakeTransform {
        FakeTransform {
            deltas: Vec::new(),
            outcome: Err(Failure::new(reason, retryable, "scripted")),
            work_ms: 5,
        }
    }
}

/// The scripted processing double: pops the next [`FakeTransform`] (in
/// script order), waits out its work while polling the cancel token, and
/// records every request and every return.
pub struct FakeProcessor {
    script: Mutex<std::collections::VecDeque<FakeTransform>>,
    requests: Mutex<Vec<TransformRequest>>,
    returned: Mutex<Vec<String>>,
}

impl FakeProcessor {
    pub fn new(script: Vec<FakeTransform>) -> Arc<FakeProcessor> {
        Arc::new(FakeProcessor {
            script: Mutex::new(script.into()),
            requests: Mutex::new(Vec::new()),
            returned: Mutex::new(Vec::new()),
        })
    }

    pub fn requests(&self) -> Vec<TransformRequest> {
        self.requests.lock().expect("fake processor lock").clone()
    }

    /// Whether the `transform` call for `request_id` returned.
    pub fn returned(&self, request_id: &str) -> bool {
        self.returned
            .lock()
            .expect("fake processor lock")
            .iter()
            .any(|id| id == request_id)
    }
}

impl TransformProcessor for FakeProcessor {
    fn transform(
        &self,
        request: &TransformRequest,
        queued_ms: f64,
        on_partial: &mut dyn FnMut(Partial),
        cancel: &CancelToken,
    ) -> TransformResult {
        self.requests
            .lock()
            .expect("fake processor lock")
            .push(request.clone());
        let step = self
            .script
            .lock()
            .expect("fake processor lock")
            .pop_front()
            .unwrap_or_else(|| FakeTransform::fails(FailureReason::ProviderUnavailable, false));
        let started = Instant::now();
        let timing = |started: Instant| Timing {
            queued_ms,
            processing_ms: started.elapsed().as_secs_f64() * 1000.0,
            stop_to_result_ms: None,
        };
        let deadline = started + Duration::from_millis(step.work_ms);
        let mut so_far = String::new();
        let mut deltas = step.deltas.into_iter();
        while Instant::now() < deadline {
            if cancel.is_cancelled() {
                self.returned
                    .lock()
                    .expect("fake processor lock")
                    .push(request.request_id.clone());
                return failure_result(
                    request,
                    Failure::new(FailureReason::Cancelled, false, ""),
                    timing(started),
                );
            }
            if let Some(delta) = deltas.next() {
                so_far.push_str(&delta);
                on_partial(Partial {
                    text: so_far.clone(),
                    stability_hint: "growing".to_string(),
                });
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        // Deltas the pacing had no tick left for still stream before the
        // outcome, so the script is honored whatever `work_ms` is.
        for delta in deltas {
            so_far.push_str(&delta);
            on_partial(Partial {
                text: so_far.clone(),
                stability_hint: "growing".to_string(),
            });
        }
        self.returned
            .lock()
            .expect("fake processor lock")
            .push(request.request_id.clone());
        match step.outcome {
            Ok(text) => TransformResult {
                schema_version: 1,
                request_id: request.request_id.clone(),
                base_revision: request.base_revision,
                status: ResultStatus::Completed,
                text: Some(text),
                failure: None,
                provider: request.provider.clone(),
                timing: timing(started),
            },
            Err(failure) => failure_result(request, failure, timing(started)),
        }
    }
}

/// One scripted job for [`FakeProvider`].
#[derive(Clone)]
pub struct FakeJob {
    pub partials: Vec<Partial>,
    pub outcome: ProviderOutcome,
    /// Simulated work; keeps the scheduler's dispatch machinery real.
    pub work_ms: u64,
}

impl FakeJob {
    /// A clean completion with the given text and no partials.
    pub fn completes_with(text: &str) -> FakeJob {
        FakeJob {
            partials: Vec::new(),
            outcome: ProviderOutcome::Completed {
                text: text.to_string(),
                backend: "fake-provider".to_string(),
                timing_ms: 12.0,
                completion_evidence: "final_decode".to_string(),
            },
            work_ms: 5,
        }
    }

    pub fn fails(reason: &str, retryable: bool) -> FakeJob {
        FakeJob {
            partials: Vec::new(),
            outcome: ProviderOutcome::Failed {
                reason: reason.to_string(),
                retryable,
            },
            work_ms: 5,
        }
    }
}

/// The scripted test double: `recognize` pops the next [`FakeJob`],
/// waits out its `work_ms` **while polling the cancel token** (issue
/// #251 — a cancelled job's provider call returns promptly instead of
/// sleeping out the whole simulated work), streams its partials through
/// `on_partial`, and returns its outcome. It also records every request
/// for assertions, and every `recognize` *return* — the observable the
/// cancellation tests gate on when proving a worker stopped early.
pub struct FakeProvider {
    script: Mutex<Vec<FakeJob>>,
    requests: Mutex<Vec<(String, usize)>>,
    /// Request ids whose `recognize` call returned (entry recorded just
    /// before the return).
    returned: Mutex<Vec<String>>,
}

impl FakeProvider {
    pub fn new(script: Vec<FakeJob>) -> Arc<Self> {
        Arc::new(FakeProvider {
            script: Mutex::new(script),
            requests: Mutex::new(Vec::new()),
            returned: Mutex::new(Vec::new()),
        })
    }

    /// The (request_id, wav size) pairs the provider saw.
    pub fn requests(&self) -> Vec<(String, usize)> {
        self.requests.lock().expect("fake provider lock").clone()
    }

    /// Whether the `recognize` call for `request_id` has returned — the
    /// provider-side proof that a worker stopped (or finished) rather
    /// than still being parked inside the provider.
    pub fn recognize_returned(&self, request_id: &str) -> bool {
        self.returned
            .lock()
            .expect("fake provider lock")
            .iter()
            .any(|id| id == request_id)
    }

    /// Records that the `recognize` call for `request_id` is returning
    /// (the last thing a path does before its `return`).
    fn note_return(&self, request_id: &str) {
        self.returned
            .lock()
            .expect("fake provider lock")
            .push(request_id.to_string());
    }
}

impl TranscriptionProvider for FakeProvider {
    fn recognize(
        &self,
        wav: Vec<u8>,
        request_id: &str,
        on_partial: &mut dyn FnMut(Partial),
        cancel: &CancelToken,
    ) -> ProviderOutcome {
        self.requests
            .lock()
            .expect("fake provider lock")
            .push((request_id.to_string(), wav.len()));
        let job = self
            .script
            .lock()
            .expect("fake provider lock")
            .pop()
            .unwrap_or(FakeJob::fails("script_exhausted", false));
        // Simulated work that honors cancellation: the wait polls the
        // token in small slices instead of sleeping blind, so cancelling
        // mid-recognition unwinds the provider promptly — exactly what
        // the scheduler's workers need to stop early (issue #251).
        let deadline = Instant::now() + Duration::from_millis(job.work_ms);
        while Instant::now() < deadline {
            if cancel.is_cancelled() {
                self.note_return(request_id);
                return ProviderOutcome::Failed {
                    reason: "cancelled".to_string(),
                    retryable: false,
                };
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        for partial in job.partials {
            on_partial(partial);
        }
        self.note_return(request_id);
        job.outcome
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Issue #251, fake half: a recognition cancelled mid-work returns
    /// promptly with the conventional `cancelled` failure — not after
    /// the job's full simulated work.
    #[test]
    fn fake_provider_unwinds_promptly_when_cancelled_mid_work() {
        let provider = FakeProvider::new(vec![FakeJob {
            work_ms: 30_000,
            ..FakeJob::completes_with("too late")
        }]);
        let token = CancelToken::new();
        let canceller = {
            let token = token.clone();
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(50));
                token.cancel();
            })
        };
        let started = Instant::now();
        let outcome =
            provider.recognize(vec![0u8; 64], "job-x", &mut |_partial: Partial| {}, &token);
        let elapsed = started.elapsed();
        canceller.join().expect("canceller thread");
        assert!(
            elapsed < Duration::from_secs(5),
            "cancelled recognition took {elapsed:?}; it must unwind long before its 30 s work"
        );
        match outcome {
            ProviderOutcome::Failed { reason, retryable } => {
                assert_eq!(reason, "cancelled");
                assert!(!retryable);
            }
            ProviderOutcome::Completed { .. } => {
                panic!("expected the cancelled failure, got a completion")
            }
        }
        assert!(provider.recognize_returned("job-x"));
    }

    /// The already-cancelled case returns before any simulated work.
    #[test]
    fn fake_provider_returns_immediately_when_precancelled() {
        let provider = FakeProvider::new(vec![FakeJob {
            work_ms: 30_000,
            ..FakeJob::completes_with("never")
        }]);
        let token = CancelToken::new();
        token.cancel();
        let started = Instant::now();
        let outcome =
            provider.recognize(vec![0u8; 64], "job-y", &mut |_partial: Partial| {}, &token);
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "a precancelled recognition must not run its 30 s work"
        );
        assert!(matches!(
            outcome,
            ProviderOutcome::Failed {
                ref reason,
                retryable: false
            } if reason == "cancelled"
        ));
    }
}
