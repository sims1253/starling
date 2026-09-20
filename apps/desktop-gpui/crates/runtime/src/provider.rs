//! The inference provider seam. The production adapter **reuses
//! `starling-dictation`'s `client.rs`** (the design's code disposition:
//! endpoint validation, bounded timeouts, error taxonomy and tested
//! parsers) as [`StarlingProvider`]; [`FakeProvider`] is the scripted test
//! double the integration tests drive.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use starling_dictation::client::{ClientError, Protocol, StarlingClient};

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
    }
}

/// What the jobs executor's workers call. Blocking by design (the
/// production client is blocking); workers run on their own threads and
/// are supervised by the scheduler.
pub trait TranscriptionProvider: Send + Sync {
    /// Streams partials (when the provider can) and returns the final
    /// outcome. `request_id` is the job's correlation id.
    fn recognize(
        &self,
        wav: Vec<u8>,
        request_id: &str,
        on_partial: &mut dyn FnMut(Partial),
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
    ) -> ProviderOutcome {
        let started = Instant::now();
        match self.client.transcribe(&wav, request_id) {
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

/// The scripted test double: `recognize` pops the next [`FakeJob`]
/// (blocking briefly for `work_ms`), streams its partials through
/// `on_partial`, and returns its outcome. It also records every request
/// for assertions.
pub struct FakeProvider {
    script: Mutex<Vec<FakeJob>>,
    requests: Mutex<Vec<(String, usize)>>,
}

impl FakeProvider {
    pub fn new(script: Vec<FakeJob>) -> Arc<Self> {
        Arc::new(FakeProvider {
            script: Mutex::new(script),
            requests: Mutex::new(Vec::new()),
        })
    }

    /// The (request_id, wav size) pairs the provider saw.
    pub fn requests(&self) -> Vec<(String, usize)> {
        self.requests.lock().expect("fake provider lock").clone()
    }
}

impl TranscriptionProvider for FakeProvider {
    fn recognize(
        &self,
        wav: Vec<u8>,
        request_id: &str,
        on_partial: &mut dyn FnMut(Partial),
    ) -> ProviderOutcome {
        self.requests
            .lock()
            .expect("fake provider lock")
            .push((request_id.to_string(), wav.len()));
        std::thread::sleep(std::time::Duration::from_millis(5));
        let job = self
            .script
            .lock()
            .expect("fake provider lock")
            .pop()
            .unwrap_or(FakeJob::fails("script_exhausted", false));
        for partial in job.partials {
            on_partial(partial);
        }
        job.outcome
    }
}
