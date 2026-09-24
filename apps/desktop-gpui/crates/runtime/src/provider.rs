//! The inference provider seam. The production adapter **reuses
//! `starling-dictation`'s `client.rs`** (the design's code disposition:
//! endpoint validation, bounded timeouts, error taxonomy and tested
//! parsers) as [`StarlingProvider`]; [`FakeProvider`] is the scripted test
//! double the integration tests drive.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use starling_dictation::client::{ClientError, Protocol, StarlingClient};

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
        /// The provider passed through its post-recognition transform
        /// stage (mode-dependent; the scheduler surfaces the internal
        /// `Transforming` edge when set).
        transformed: bool,
    },
    Failed { reason: String, retryable: bool },
}

/// Why a provider call failed, mapped to v1 `jobs.failed{reason,
/// retryable}` tokens.
pub fn failure_from_client_error(error: &ClientError) -> (String, bool) {
    match error {
        ClientError::Input(_) => ("invalid_input".to_string(), false),
        ClientError::Transport(_) => ("transport_error".to_string(), true),
        ClientError::Timeout(_) => ("timeout".to_string(), true),
        ClientError::Redirect(_) => ("redirect_blocked".to_string(), false),
        ClientError::Http { status, .. } => {
            (format!("http_{status}"), *status >= 500)
        }
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
/// (PR #193's protocol enum, endpoint validation, bounded timeouts, error
/// taxonomy) behind the runtime's adapter trait.
pub struct StarlingProvider {
    client: StarlingClient,
}

impl StarlingProvider {
    pub fn new(endpoint: &str, protocol: Protocol, model: &str) -> Result<Self, ClientError> {
        Ok(StarlingProvider {
            client: StarlingClient::new(endpoint, protocol, model)?,
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
                transformed: false,
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
                transformed: false,
            },
            work_ms: 5,
        }
    }

    /// A completion that passes through the `Transforming` stage.
    pub fn transforms_to(text: &str) -> FakeJob {
        FakeJob {
            partials: vec![Partial {
                text: text.to_string(),
                stability_hint: "stable".to_string(),
            }],
            outcome: ProviderOutcome::Completed {
                text: text.to_string(),
                backend: "fake-provider".to_string(),
                timing_ms: 12.0,
                completion_evidence: "final_decode".to_string(),
                transformed: true,
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
        let outcome = provider.recognize(
            vec![0u8; 64],
            "job-x",
            &mut |_partial: Partial| {},
            &token,
        );
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
        let outcome = provider.recognize(
            vec![0u8; 64],
            "job-y",
            &mut |_partial: Partial| {},
            &token,
        );
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
