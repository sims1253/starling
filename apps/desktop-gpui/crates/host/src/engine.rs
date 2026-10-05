//! The supervised engine attaches to the **host**, not the renderer
//! (E17 §1 Mode B, #220): the host owns the bundled-engine
//! [`EngineManager`] (the `starling-serve` sidecar, its crash restarts
//! and model switches, #362/#363), and the runtime's jobs machine
//! reaches it through [`EngineProvider`]. Engine lifetime is host
//! lifetime — the sidecar is spawned with the host's pid as its
//! `--parent-pid`, a renderer that dies mid-job costs nothing, and the
//! host's shutdown stops the engine after the machines have joined.
//!
//! Which engine serves is the user's existing engine choice
//! ([`EngineChoice::from_settings`]): the bundled engine with the
//! persisted model, the user's own server in manual mode, or none. The
//! host reads the same settings file and the same model/state
//! directories the desktop app uses, so the app and the host share one
//! sidecar through the engine registry (one owns it, the other
//! attaches) instead of loading the model twice.

use std::time::{Duration, Instant};

use starling_dictation::client::{ClientError, StarlingClient};
use starling_dictation::engine::{EngineConfig, EngineLease, EngineManager, EnginePhase};
use starling_dictation::settings::{EngineMode, Settings};
use starling_runtime::provider::{
    failure_from_client_error, CancelToken, Partial, ProviderOutcome, TranscriptionProvider,
};

/// How long a job waits for the engine to become ready before failing
/// `engine_not_ready` (retryable). Covers a host that just booted and is
/// still loading or warming its model, and a crash restart's backoff.
pub const DEFAULT_READY_WAIT: Duration = Duration::from_secs(120);

/// How often a waiting job re-checks the engine (and its cancel token).
const READY_POLL: Duration = Duration::from_millis(50);

/// The user's engine choice, resolved for the host.
#[derive(Debug, Clone)]
pub enum EngineChoice {
    /// No engine: jobs fail `no_provider_configured` (the honest
    /// default; `--engine none` and tests).
    None,
    /// The bundled engine, supervised by this host.
    Builtin {
        config: EngineConfig,
        active_model: Option<String>,
    },
    /// The user's own server (`engine.mode = manual`).
    Manual { endpoint: String, model: String },
}

impl EngineChoice {
    /// The choice the desktop settings file states, with the app's own
    /// defaults: builtin on the default engine paths with the persisted
    /// model and backend override, or the manual endpoint/model. Errors
    /// only when the builtin engine's data directory cannot resolve.
    pub fn from_settings(settings: &Settings) -> Result<EngineChoice, String> {
        match settings.engine.mode {
            EngineMode::Builtin => {
                let mut config = EngineConfig::default_paths().map_err(|err| err.to_string())?;
                config.backend_override = settings
                    .engine
                    .backend_override
                    .as_deref()
                    .and_then(starling_dictation::engine::Backend::parse);
                Ok(EngineChoice::Builtin {
                    config,
                    active_model: settings.engine.active_model.clone(),
                })
            }
            EngineMode::Manual => Ok(EngineChoice::Manual {
                endpoint: settings.endpoint.clone(),
                model: settings.model.clone(),
            }),
        }
    }

    /// The label the host's status line reports.
    pub fn label(&self) -> &'static str {
        match self {
            EngineChoice::None => "none",
            EngineChoice::Builtin { .. } => "builtin",
            EngineChoice::Manual { .. } => "manual",
        }
    }
}

/// Attaches `choice` to the runtime about to start: builtin starts the
/// engine supervisor (returned, for the host to stop at shutdown) and
/// installs [`EngineProvider`]; manual installs the plain server
/// provider; none leaves `runtime.provider` untouched. A manual
/// endpoint that does not validate is reported and left unconfigured —
/// the host still owns capture and storage, jobs fail
/// `no_provider_configured`, and the user's settings are not guessed at.
pub fn attach(
    choice: EngineChoice,
    runtime: &mut starling_runtime::RuntimeConfig,
) -> Option<EngineManager> {
    match choice {
        EngineChoice::None => None,
        EngineChoice::Builtin {
            config,
            active_model,
        } => {
            let manager = EngineManager::start(config, active_model);
            runtime.provider = std::sync::Arc::new(EngineProvider::new(manager.clone()));
            Some(manager)
        }
        EngineChoice::Manual { endpoint, model } => {
            match starling_runtime::provider::StarlingProvider::new(&endpoint, &model) {
                Ok(provider) => runtime.provider = std::sync::Arc::new(provider),
                Err(err) => eprintln!(
                    "starling-runtime-host: manual engine endpoint {endpoint:?} is unusable \
                     ({err}); transcription stays unconfigured"
                ),
            }
            None
        }
    }
}

/// The jobs machine's provider over the host-owned engine: each
/// recognition leases the active engine for its whole request (so a
/// model switch drains it rather than cutting it off), sends the take
/// through `starling-dictation`'s client, and reports the model it ran
/// on as the completion's `backend` (`engine:<model id>`).
pub struct EngineProvider {
    manager: EngineManager,
    ready_wait: Duration,
}

impl EngineProvider {
    pub fn new(manager: EngineManager) -> EngineProvider {
        EngineProvider {
            manager,
            ready_wait: DEFAULT_READY_WAIT,
        }
    }

    /// Overrides how long a job waits for a ready engine.
    pub fn with_ready_wait(mut self, wait: Duration) -> EngineProvider {
        self.ready_wait = wait;
        self
    }
}

/// Why no engine could take a job, as a v1 `jobs.failed` reason. Every
/// case is retryable: the take is durable, and the same job succeeds
/// once the engine serves (a model installed, a crash restart done).
fn unready_reason(phase: &EnginePhase) -> &'static str {
    match phase {
        EnginePhase::NoModel => "engine_no_model",
        EnginePhase::Failed(_) => "engine_unavailable",
        _ => "engine_not_ready",
    }
}

/// `engine:<model id>` as a contract `safeToken`
/// (`[A-Za-z0-9_.:+-]{1,128}`): catalog ids already fit; anything else
/// is replaced rather than letting the completion event fail
/// validation.
fn backend_token(model_id: &str) -> String {
    let mut token: String = format!("engine:{model_id}")
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || "_.:+-".contains(c) {
                c
            } else {
                '_'
            }
        })
        .collect();
    token.truncate(128);
    token
}

impl EngineProvider {
    /// Waits (bounded by `deadline`, cancel-aware) for a lease on a ready
    /// engine other than `avoid` — the `(endpoint, pid)` of an engine a
    /// request just failed against, which the manager may still report
    /// Ready until its supervisor notices it is gone. The identity comes
    /// from the lease itself (the engine it holds), never from a
    /// separately-locked snapshot that could already have moved on.
    fn wait_for_lease(
        &self,
        deadline: Instant,
        cancel: &CancelToken,
        avoid: Option<&(String, u32)>,
    ) -> Result<EngineLease, ProviderOutcome> {
        loop {
            if cancel.is_cancelled() {
                return Err(ProviderOutcome::Failed {
                    reason: "cancelled".to_string(),
                    retryable: false,
                });
            }
            if let Some(lease) = self.manager.lease() {
                let identity = (lease.endpoint().to_string(), lease.pid());
                if avoid != Some(&identity) {
                    return Ok(lease);
                }
            }
            let phase = self.manager.snapshot().phase;
            // A failed engine or a missing model will not fix itself
            // while this job waits (both need the user): fail now.
            let settled = matches!(phase, EnginePhase::NoModel | EnginePhase::Failed(_));
            if settled || Instant::now() >= deadline {
                return Err(ProviderOutcome::Failed {
                    reason: unready_reason(&phase).to_string(),
                    retryable: true,
                });
            }
            std::thread::sleep(READY_POLL);
        }
    }
}

impl TranscriptionProvider for EngineProvider {
    fn recognize(
        &self,
        wav: Vec<u8>,
        request_id: &str,
        _on_partial: &mut dyn FnMut(Partial),
        cancel: &CancelToken,
    ) -> ProviderOutcome {
        let started = Instant::now();
        let wav = std::sync::Arc::new(wav);
        let mut failed_engine: Option<(String, u32)> = None;
        loop {
            let deadline = Instant::now() + self.ready_wait;
            let lease = match self.wait_for_lease(deadline, cancel, failed_engine.as_ref()) {
                Ok(found) => found,
                Err(outcome) => return outcome,
            };
            let client = match StarlingClient::new(lease.endpoint(), lease.slug()) {
                Ok(client) => client,
                Err(error) => {
                    let (reason, retryable) = failure_from_client_error(&error);
                    return ProviderOutcome::Failed { reason, retryable };
                }
            };
            // The lease is held for the whole request: a model switch that
            // starts mid-request drains this engine instead of stopping it
            // under the take (#363). It drops at the end of this iteration.
            match client.transcribe_with_cancel(
                std::sync::Arc::clone(&wav),
                request_id,
                Some(cancel),
            ) {
                Ok(result) => {
                    return ProviderOutcome::Completed {
                        text: result.text,
                        backend: backend_token(lease.model_id()),
                        timing_ms: started.elapsed().as_secs_f64() * 1000.0,
                        completion_evidence: "final_decode".to_string(),
                    }
                }
                // The engine went away under the request: its process
                // died (a crash, or — when this host attached to a sidecar
                // the desktop app started — the app being killed, which
                // takes that sidecar with it). The supervisor restarts or
                // takes the engine over; this job retries **once** on the
                // replacement, so the engine's death costs the job a
                // delay, not its result. Recognition is idempotent (same
                // audio, a new request), and one retry bounds the cost of
                // an engine that keeps dying.
                Err(ClientError::Transport(_))
                    if failed_engine.is_none() && !cancel.is_cancelled() =>
                {
                    failed_engine = Some((lease.endpoint().to_string(), lease.pid()));
                }
                Err(error) => {
                    let (reason, retryable) = failure_from_client_error(&error);
                    return ProviderOutcome::Failed { reason, retryable };
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_tokens_fit_the_contract_pattern() {
        assert_eq!(backend_token("parakeet-v3-q8"), "engine:parakeet-v3-q8");
        assert_eq!(backend_token("we ird/id"), "engine:we_ird_id");
        assert_eq!(backend_token(&"x".repeat(300)).len(), 128);
    }

    #[test]
    fn unready_reasons_name_what_the_user_must_fix() {
        assert_eq!(unready_reason(&EnginePhase::NoModel), "engine_no_model");
        assert_eq!(unready_reason(&EnginePhase::Loading), "engine_not_ready");
        assert_eq!(
            unready_reason(&EnginePhase::Failed(
                starling_dictation::engine::EngineFailure::NoBundledEngine
            )),
            "engine_unavailable"
        );
    }

    #[test]
    fn manual_and_builtin_settings_resolve_to_their_engines() {
        let mut settings = Settings::default_settings();
        settings.engine.mode = EngineMode::Manual;
        settings.endpoint = "http://127.0.0.1:9999".into();
        match EngineChoice::from_settings(&settings).unwrap() {
            EngineChoice::Manual { endpoint, model } => {
                assert_eq!(endpoint, "http://127.0.0.1:9999");
                assert_eq!(model, settings.model);
            }
            other => panic!("expected manual, got {other:?}"),
        }
        settings.engine.mode = EngineMode::Builtin;
        settings.engine.active_model = Some("parakeet-v3-q8".into());
        settings.engine.backend_override = Some("cpu".into());
        // default_paths needs a data dir; every CI runner has one.
        match EngineChoice::from_settings(&settings).unwrap() {
            EngineChoice::Builtin {
                config,
                active_model,
            } => {
                assert_eq!(active_model.as_deref(), Some("parakeet-v3-q8"));
                assert_eq!(
                    config.backend_override,
                    Some(starling_dictation::engine::Backend::Cpu)
                );
            }
            other => panic!("expected builtin, got {other:?}"),
        }
    }
}
