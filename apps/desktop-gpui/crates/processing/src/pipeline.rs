//! The processing pipeline (#294): the deterministic step, at most one
//! model step, and a [`TransformResult`] with per-job timing.
//!
//! [`plan`] turns a mode into what will run (nothing, the builtin step,
//! one provider, or a block with its reason) through the contract's
//! `processing_route`, so there is no second place that picks providers
//! and no fallback: a blocked plan runs nothing and the raw text stands.
//! [`build_request`] reads a [`Draft`] (never its live partials) and
//! pins the request to the draft's revision; [`run`] executes it. A
//! result is only ever a proposal for the draft to judge.

use std::sync::Arc;
use std::time::Instant;

use starling_dictation::client::CancelToken;

use crate::contract::{
    processing_route, ContextField, Failure, FailureReason, Locality, ModeEntry, ProcessingRoute,
    ProviderDecl, ProviderKind, RequestContext, ResultStatus, RouteBlock, Timing, TransformRequest,
    TransformResult,
};
use crate::http::failure;
use crate::providers::Provider;
use crate::staging::Draft;
use crate::transforms;

/// The deterministic step's identity on requests and results.
pub fn builtin_declaration() -> ProviderDecl {
    ProviderDecl {
        schema_version: 1,
        id: "builtin".to_string(),
        route: "local-builtin".to_string(),
        kind: ProviderKind::Builtin,
        locality: Locality::Local,
        transform_kinds: Vec::new(),
        languages: vec!["*".to_string()],
        instructions: false,
        context_fields: Vec::new(),
        model: "spoken-commands.v1".to_string(),
        artifact: None,
        max_input_chars: 1_000_000,
    }
}

/// The providers an embedder configured.
#[derive(Clone, Default)]
pub struct Registry {
    providers: Vec<Arc<dyn Provider>>,
}

impl Registry {
    pub fn new(providers: Vec<Arc<dyn Provider>>) -> Registry {
        Registry { providers }
    }

    pub fn declarations(&self) -> Vec<ProviderDecl> {
        self.providers
            .iter()
            .map(|p| p.declaration().clone())
            .collect()
    }

    pub fn get(&self, id: &str) -> Option<Arc<dyn Provider>> {
        self.providers
            .iter()
            .find(|p| p.declaration().id == id)
            .cloned()
    }
}

/// What will run for a mode.
#[derive(Clone)]
pub enum Plan {
    /// Transcribe only: the raw text is the output.
    Nothing,
    /// Spoken commands / snippets only, no model call.
    Builtin,
    /// The deterministic step, then this provider, sending these context
    /// fields.
    Model {
        provider: Arc<dyn Provider>,
        context_fields: Vec<ContextField>,
    },
    /// Nothing runs; the raw text stands.
    Blocked(RouteBlock),
}

impl Plan {
    /// Where text goes and which context fields are sent, for the UI to
    /// show before a mode is used. `None` when nothing leaves the
    /// transcript's own process.
    pub fn destination(&self) -> Option<(ProviderDecl, Vec<ContextField>)> {
        match self {
            Plan::Model {
                provider,
                context_fields,
            } => Some((provider.declaration().clone(), context_fields.clone())),
            _ => None,
        }
    }
}

pub fn plan(mode: &ModeEntry, registry: &Registry) -> Plan {
    let declarations = registry.declarations();
    match processing_route(mode, &declarations) {
        ProcessingRoute::None => {
            if mode.spoken_commands || !mode.snippets.is_empty() {
                Plan::Builtin
            } else {
                Plan::Nothing
            }
        }
        ProcessingRoute::Ready {
            provider,
            context_fields,
        } => match registry.get(&provider.id) {
            Some(provider) => Plan::Model {
                provider,
                context_fields,
            },
            None => Plan::Blocked(RouteBlock::ProviderUnavailable),
        },
        ProcessingRoute::Blocked(block) => Plan::Blocked(block),
    }
}

/// Context values an embedder holds; only the fields the plan allows are
/// copied into a request.
#[derive(Clone, Debug, Default)]
pub struct ContextValues {
    pub personal_context: Option<String>,
}

#[derive(Clone, Debug)]
pub struct RequestOptions {
    pub request_id: String,
    pub retry_of: Option<String>,
    pub deadline_ms: u64,
    pub max_output_chars: u64,
}

/// Builds the request for the draft's current revision. The input is the
/// draft's payload (command regions excluded) after the deterministic
/// step; the trailing instruction, if any, travels separately.
pub fn build_request(
    draft: &Draft,
    mode: &ModeEntry,
    provider: &ProviderDecl,
    context_fields: &[ContextField],
    values: &ContextValues,
    options: &RequestOptions,
) -> TransformRequest {
    let input = transforms::apply(
        &draft.payload_text(),
        mode.language.as_deref(),
        mode.spoken_commands,
        &mode.snippets,
    );
    let model_step = provider.kind != ProviderKind::Builtin;
    let mut context = RequestContext::default();
    if model_step {
        if context_fields.contains(&ContextField::Vocabulary) && !mode.vocabulary.is_empty() {
            context.vocabulary = Some(mode.vocabulary.clone());
        }
        if context_fields.contains(&ContextField::PersonalContext) {
            context.personal_context = values
                .personal_context
                .as_ref()
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty());
        }
    }
    let instruction = draft.instruction().filter(|_| {
        mode.transform_kinds
            .iter()
            .any(|kind| kind.needs_instructions())
    });
    TransformRequest {
        schema_version: 1,
        request_id: options.request_id.clone(),
        retry_of: options.retry_of.clone(),
        draft_id: draft.draft_id().to_string(),
        capture_id: draft.capture_id().to_string(),
        base_revision: draft.revision(),
        source_attempt_ids: draft.source_attempt_ids(),
        mode_id: mode.id.clone(),
        mode_version: mode.version,
        prompt_version: (model_step && provider.kind != ProviderKind::S1)
            .then(|| crate::prompt::PROMPT_VERSION.to_string()),
        // Kinds name the model step only (validate_provider): a builtin
        // request carries none, and the insight event records the same.
        kinds: if model_step {
            mode.transform_kinds.clone()
        } else {
            Vec::new()
        },
        language: mode.language.clone(),
        input,
        instruction,
        style: mode.style,
        context,
        provider: provider.reference(),
        local_only: mode.local_only,
        deadline_ms: options.deadline_ms,
        max_output_chars: options.max_output_chars,
    }
}

/// When the job was queued and when the take stopped, for timing.
#[derive(Clone, Copy, Debug, Default)]
pub struct Clock {
    pub queued_ms: f64,
    pub stopped_at: Option<Instant>,
}

fn ms(since: Instant) -> f64 {
    since.elapsed().as_secs_f64() * 1000.0
}

/// Runs one request. `provider` is `None` for the builtin step. The
/// request is checked against the provider before anything is sent: a
/// `local_only` request never reaches a remote provider, even if a caller
/// built it by hand.
pub fn run(
    request: &TransformRequest,
    provider: Option<&dyn Provider>,
    clock: Clock,
    on_delta: &mut dyn FnMut(&str),
    cancel: &CancelToken,
) -> TransformResult {
    let started = Instant::now();
    let outcome = execute(request, provider, on_delta, cancel);
    let timing = Timing {
        queued_ms: clock.queued_ms.max(0.0),
        processing_ms: ms(started),
        stop_to_result_ms: clock.stopped_at.map(ms),
    };
    let provider_ref = request.provider.clone();
    match outcome {
        Ok(text) => TransformResult {
            schema_version: 1,
            request_id: request.request_id.clone(),
            base_revision: request.base_revision,
            status: ResultStatus::Completed,
            text: Some(text),
            failure: None,
            provider: provider_ref,
            timing,
        },
        Err(failure) => failure_result(request, failure, timing),
    }
}

/// A result for a request that never reached a provider (or whose
/// provider could not be built): the typed failure, with timing.
pub fn failure_result(
    request: &TransformRequest,
    failure: Failure,
    timing: Timing,
) -> TransformResult {
    TransformResult {
        schema_version: 1,
        request_id: request.request_id.clone(),
        base_revision: request.base_revision,
        status: if failure.reason == FailureReason::Cancelled {
            ResultStatus::Cancelled
        } else {
            ResultStatus::Failed
        },
        text: None,
        failure: Some(failure),
        provider: request.provider.clone(),
        timing,
    }
}

fn execute(
    request: &TransformRequest,
    provider: Option<&dyn Provider>,
    on_delta: &mut dyn FnMut(&str),
    cancel: &CancelToken,
) -> Result<String, Failure> {
    if cancel.is_cancelled() {
        return Err(failure(FailureReason::Cancelled, false, ""));
    }
    let Some(provider) = provider else {
        if request.provider.kind != ProviderKind::Builtin || !request.kinds.is_empty() {
            return Err(failure(
                FailureReason::ProviderUnavailable,
                false,
                "no provider for a model step",
            ));
        }
        return Ok(request.input.clone());
    };
    let decl = provider.declaration();
    if request.local_only && decl.locality == Locality::Remote {
        return Err(failure(
            FailureReason::RemoteForbidden,
            false,
            "this mode is local-only; nothing was sent",
        ));
    }
    if decl.reference() != request.provider {
        return Err(failure(
            FailureReason::InvalidInput,
            false,
            "the request names a different provider",
        ));
    }
    // S1 splits its input into prompts of at most `max_input_chars` (see
    // `providers::s1`), so for it the cap is per prompt, not per request.
    if request.input.chars().count() as u64 > u64::from(decl.max_input_chars)
        && decl.kind != ProviderKind::S1
    {
        return Err(failure(
            FailureReason::InvalidInput,
            false,
            format!(
                "the text is longer than {} characters",
                decl.max_input_chars
            ),
        ));
    }
    if request.input.trim().is_empty() {
        // Nothing to send; an empty take stays empty.
        return Ok(String::new());
    }
    let text = provider.run(request, on_delta, cancel)?;
    if cancel.is_cancelled() {
        return Err(failure(FailureReason::Cancelled, false, ""));
    }
    if text.chars().count() as u64 > request.max_output_chars {
        return Err(crate::providers::over_cap(request.max_output_chars));
    }
    Ok(text)
}
