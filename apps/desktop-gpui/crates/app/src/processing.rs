//! Text processing in the desktop app (#295, first half; the pipeline is
//! #294's `starling-processing`).
//!
//! The mode comes from the app's built-in modes
//! (`modes/desktop-profiles.json`, validated by the contract tests like any
//! profiles document); the providers come from settings: S1-mini on a
//! second, loopback starling-serve (one model is resident per server, so
//! it cannot share the transcription server) and an OpenAI-compatible API.
//! `pipeline::plan` picks at most one of them; a blocked plan runs nothing.
//! A finalized take is routed first: a leading spoken phrase (or the
//! literal escape) overrides the active mode for that one take.
//!
//! Every result is a proposal. The take's staged draft (`staging::Draft`,
//! resumed from the take's processing document in storage v2) decides
//! whether a result is current or stale; only "Use processed" (accept) or
//! "Back to raw" moves the head, so a late result can never replace text
//! the user changed since the job started. Raw recognition stays in its
//! attempt row, whatever happens here.
//!
//! This app does not embed the runtime (see the host crate's notes), so
//! jobs run here directly on the pipeline: one per take, a new one
//! superseding (cancelling) the old, each bounded by its deadline and
//! output cap. The runtime's `jobs.transform` is the same pipeline behind
//! the scheduler.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use gpui::{AppContext, Context};
use starling_dictation::settings::ProcessingSettings;
use starling_dictation::storage::{self, now_iso};
use starling_processing::CancelToken;
use starling_processing::contract::{
    ContextField, FailureReason, Locality, ModeEntry, ProcessingRoute, ProfilesDocument,
    ProviderDecl, ProviderKind, ResultStatus, RouteBlock, Timing, TransformKind, TransformRequest,
    TransformResult, processing_route,
};
use starling_processing::http::validate_endpoint;
use starling_processing::insight::{Arrival, ProcessingRecorded};
use starling_processing::pipeline::{self, Clock, ContextValues, Plan, Registry, RequestOptions};
use starling_processing::providers::openai::OpenAiProvider;
use starling_processing::providers::s1::S1Provider;
use starling_processing::providers::{ChatConfig, Provider};
use starling_processing::routing::{self, RouteRequest, Source};
use starling_processing::staging::{
    Attempt, CommandKind, Draft, Outcome, ProposalStatus, RegionKind, ResultKind, StoredProposal,
};

use crate::app::StarlingApp;
use crate::store::{ProcessingDoc, ProposalRow, RowStatus};

const MODES_JSON: &str = include_str!("../modes/desktop-profiles.json");

/// Processing deadline per job. S1-mini on a CPU needs seconds per
/// chunk; an API answer normally far less.
const DEADLINE_MS: u64 = 120_000;

pub(crate) const S1_ROUTE: &str = "local-authoring-s1";
pub(crate) const API_ROUTE: &str = "remote-authoring-api";

pub(crate) fn modes() -> &'static ProfilesDocument {
    static MODES: OnceLock<ProfilesDocument> = OnceLock::new();
    MODES.get_or_init(|| serde_json::from_str(MODES_JSON).expect("desktop-profiles.json parses"))
}

/// The mode with this id, or the default when the id is unknown (a
/// settings file naming a mode this build no longer ships).
pub(crate) fn mode(id: &str) -> &'static ModeEntry {
    let modes = modes();
    modes
        .profile(id)
        .or_else(|| modes.profile(&modes.default_profile))
        .expect("the default mode exists")
}

/// The notice for a saved mode id no built-in mode has (it runs as the
/// default mode instead).
pub(crate) fn unknown_mode_note(id: &str) -> Option<String> {
    let modes = modes();
    modes.profile(id).is_none().then(|| {
        format!(
            "The processing mode \"{id}\" no longer exists; using \"{}\". Pick a mode in Settings.",
            modes.default_profile
        )
    })
}

/// Where a finalized take's leading text routes it, per [`routing::resolve`].
#[derive(Debug)]
enum TakeRoute<'a> {
    /// No leading phrase: the active mode processes the take.
    Active,
    /// A leading phrase (or the literal escape) selected a mode for this
    /// take; `span` is the consumed prefix in code points.
    Override {
        mode: &'a ModeEntry,
        span: (usize, usize),
    },
    /// Equal-rank phrases matched: nothing runs, the message names them.
    Conflict(String),
}

/// Routes a finalized take. The active mode is an unlocked manual
/// choice, so only a mode with `allow_spoken_overrides` has its text
/// parsed for phrases.
fn route_take<'a>(doc: &'a ProfilesDocument, text: &str, active: &'a ModeEntry) -> TakeRoute<'a> {
    let request = RouteRequest {
        raw_text: text.to_string(),
        manual_mode: Some(active.id.clone()),
        manual_locked: false,
        session_allows_aliases: true,
        ..RouteRequest::default()
    };
    let resolved =
        routing::resolve(doc, &request).expect("the built-in modes pass validate_config");
    match resolved.source {
        Source::Phrase(_) | Source::EscapeLiteral => TakeRoute::Override {
            mode: resolved
                .mode
                .as_deref()
                .and_then(|id| doc.profile(id))
                .expect("resolve routes to a mode of the document"),
            span: resolved
                .prefix_span_codepoints
                .expect("a phrase match carries its prefix span"),
        },
        Source::PhraseConflict => {
            let names = routing::conflicts_for(doc, &request, &resolved)
                .iter()
                .map(|candidate| format!("\"{}\" ({})", candidate.via, candidate.mode_id))
                .collect::<Vec<_>>()
                .join(", ");
            TakeRoute::Conflict(format!(
                "The leading phrase matches several modes — {names}. Say which you mean; \
                 the take was not processed. The raw transcript is unchanged."
            ))
        }
        _ => TakeRoute::Active,
    }
}

pub(crate) fn s1_declaration() -> ProviderDecl {
    ProviderDecl {
        schema_version: 1,
        id: "local-s1".to_string(),
        route: S1_ROUTE.to_string(),
        kind: ProviderKind::S1,
        locality: Locality::Local,
        transform_kinds: vec![TransformKind::Clean, TransformKind::Format],
        languages: vec!["en".to_string()],
        instructions: false,
        context_fields: Vec::new(),
        model: "s1-mini".to_string(),
        // The app talks to a server; it has no GGUF of its own to hash.
        // The identity it can state is the model and its terms.
        artifact: None,
        max_input_chars: 1_000_000,
    }
}

/// The S1-mini naming clause (Apache 2.0 with attribution).
pub(crate) const S1_ATTRIBUTION: &str = "S1-mini by Superwhisper (Apache-2.0)";

pub(crate) fn api_declaration(model: &str) -> ProviderDecl {
    ProviderDecl {
        schema_version: 1,
        id: "remote-api".to_string(),
        route: API_ROUTE.to_string(),
        kind: ProviderKind::OpenaiCompatible,
        locality: Locality::Remote,
        transform_kinds: vec![
            TransformKind::Clean,
            TransformKind::Format,
            TransformKind::Rewrite,
            TransformKind::Translate,
        ],
        languages: vec!["*".to_string()],
        instructions: true,
        context_fields: vec![ContextField::PersonalContext, ContextField::Vocabulary],
        model: model.trim().to_string(),
        artifact: None,
        max_input_chars: 64_000,
    }
}

/// The providers built from settings, plus why a provider could not be
/// built (shown instead of a silent absence).
#[derive(Clone, Default)]
pub(crate) struct Providers {
    pub registry: Registry,
    pub problems: HashMap<&'static str, String>,
}

pub(crate) fn build_providers(settings: &ProcessingSettings) -> Providers {
    let mut built: Vec<Arc<dyn Provider>> = Vec::new();
    let mut problems = HashMap::new();
    match S1Provider::new(s1_declaration(), &settings.s1_endpoint) {
        Ok(provider) => built.push(Arc::new(provider)),
        Err(err) => {
            problems.insert(S1_ROUTE, format!("S1-mini endpoint: {err}"));
        }
    }
    if settings.api_model.trim().is_empty() {
        problems.insert(API_ROUTE, "Set the API model in settings.".to_string());
    } else {
        let mut config = ChatConfig::new(settings.api_endpoint.trim());
        match api_key(settings.api_key_env.trim()) {
            Ok(key) => {
                config.api_key = key;
                match OpenAiProvider::new(api_declaration(&settings.api_model), config) {
                    Ok(provider) => built.push(Arc::new(provider)),
                    Err(err) => {
                        problems.insert(API_ROUTE, format!("API endpoint: {err}"));
                    }
                }
            }
            Err(problem) => {
                problems.insert(API_ROUTE, problem);
            }
        }
    }
    Providers {
        registry: Registry::new(built),
        problems,
    }
}

/// The API key from the environment variable the settings name. An unset
/// or empty variable is no key (keyless local gateways need none); a name
/// that cannot be a variable, or a value that is not text, is a problem to
/// show rather than a request that fails later with a vague auth error.
fn api_key(name: &str) -> Result<Option<String>, String> {
    if name.is_empty() {
        return Ok(None);
    }
    let valid = name
        .chars()
        .enumerate()
        .all(|(i, c)| c == '_' || c.is_ascii_alphabetic() || (i > 0 && c.is_ascii_digit()));
    if !valid {
        return Err(format!(
            "API key: \"{name}\" is not an environment variable name."
        ));
    }
    match std::env::var(name) {
        Ok(key) => Ok(Some(key).filter(|key| !key.is_empty())),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => {
            Err(format!("API key: {name} does not hold valid text."))
        }
    }
}

fn host_of(endpoint: &str, locality: Locality) -> String {
    validate_endpoint(endpoint, locality)
        .ok()
        .and_then(|url| url.host_str().map(str::to_string))
        .unwrap_or_else(|| endpoint.to_string())
}

/// A short "where did this run" label for the drawer.
pub(crate) fn provider_label(decl: &ProviderDecl, settings: &ProcessingSettings) -> String {
    match decl.kind {
        ProviderKind::Builtin => "spoken commands · this computer".to_string(),
        ProviderKind::S1 => "S1-mini · this computer".to_string(),
        _ => format!(
            "{} · {}",
            host_of(&settings.api_endpoint, Locality::Remote),
            decl.model
        ),
    }
}

fn fields_text(fields: &[ContextField]) -> String {
    if fields.is_empty() {
        return "none".to_string();
    }
    fields
        .iter()
        .map(|field| match field {
            ContextField::PersonalContext => "about me",
            ContextField::Vocabulary => "vocabulary",
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn blocked_text(
    block: RouteBlock,
    mode: &ModeEntry,
    problems: &HashMap<&'static str, String>,
) -> String {
    if let Some(problem) = mode
        .authoring_route
        .as_deref()
        .and_then(|route| problems.get(route))
    {
        return format!("Not set up: {problem}");
    }
    match block {
        RouteBlock::RemoteForbidden => {
            "This mode is local-only; nothing will be sent off this computer.".to_string()
        }
        RouteBlock::UnsupportedLanguage => {
            "The provider does not support this mode's language.".to_string()
        }
        RouteBlock::UnsupportedKind => "The provider cannot do what this mode asks.".to_string(),
        RouteBlock::ProviderUnavailable | RouteBlock::ProviderConflict => {
            "No provider is set up for this mode.".to_string()
        }
    }
}

/// The provider declarations the settings would produce, and why any
/// could not be set up: the same checks `build_providers` makes, without
/// building HTTP clients (the settings dialog calls this every render).
fn declared(settings: &ProcessingSettings) -> (Vec<ProviderDecl>, HashMap<&'static str, String>) {
    let mut declarations = Vec::new();
    let mut problems = HashMap::new();
    match validate_endpoint(&settings.s1_endpoint, Locality::Local) {
        Ok(_) => declarations.push(s1_declaration()),
        Err(err) => {
            problems.insert(S1_ROUTE, format!("S1-mini endpoint: {err}"));
        }
    }
    if settings.api_model.trim().is_empty() {
        problems.insert(API_ROUTE, "Set the API model in settings.".to_string());
    } else {
        // The same checks in the same order as build_providers (key,
        // then endpoint), so the preview reports the same first problem.
        match api_key(settings.api_key_env.trim()).and_then(|_| {
            validate_endpoint(&settings.api_endpoint, Locality::Remote)
                .map_err(|err| format!("API endpoint: {err}"))
        }) {
            Ok(_) => declarations.push(api_declaration(&settings.api_model)),
            Err(problem) => {
                problems.insert(API_ROUTE, problem);
            }
        }
    }
    (declarations, problems)
}

/// What a mode does with a transcript, shown in settings before the mode
/// is saved: where the text goes and which context fields go with it.
pub(crate) fn disclosure(mode_id: &str, settings: &ProcessingSettings) -> String {
    let mode = mode(mode_id);
    let (declarations, problems) = declared(settings);
    match processing_route(mode, &declarations) {
        ProcessingRoute::None if mode.spoken_commands || !mode.snippets.is_empty() => {
            "Spoken commands only, on this computer. Nothing is sent.".to_string()
        }
        ProcessingRoute::None => {
            "Transcripts are not processed. Nothing is sent anywhere after transcription."
                .to_string()
        }
        ProcessingRoute::Ready {
            provider: decl,
            context_fields,
        } => match decl.locality {
            Locality::Local => format!(
                "Sends the transcript text to {} at {} on this computer. Context sent: {}. \
                     English dictation only. {}.",
                decl.model,
                settings.s1_endpoint.trim(),
                fields_text(&context_fields),
                S1_ATTRIBUTION,
            ),
            Locality::Remote => {
                let key = match settings.api_key_env.trim() {
                    "" => "no key".to_string(),
                    name => format!("key from ${name}"),
                };
                format!(
                    "Sends the transcript text to {} (model {}, {key}). Context sent: {}. \
                     Nothing else leaves this computer for processing.",
                    settings.api_endpoint.trim(),
                    decl.model,
                    fields_text(&context_fields),
                )
            }
        },
        ProcessingRoute::Blocked(block) => blocked_text(block, mode, &problems),
    }
}

/// One take's processing, as the drawer shows it.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum ProcessingState {
    /// Nothing to show (never processed, or a proposal was used/dismissed).
    Idle,
    Running,
    Proposal {
        row: ProposalRow,
        current: bool,
    },
    Failed {
        message: String,
    },
    Cancelled,
    Blocked {
        message: String,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct TakeProcessing {
    pub label: String,
    pub state: ProcessingState,
    /// The used processed text, when the take's head is not raw.
    pub processed_head: Option<String>,
}

impl TakeProcessing {
    pub(crate) fn from_doc(doc: &ProcessingDoc) -> TakeProcessing {
        let latest = doc
            .proposals
            .iter()
            .rev()
            .find(|row| row.status != RowStatus::Superseded);
        let state = match latest {
            Some(row) if row.status == RowStatus::Proposed => ProcessingState::Proposal {
                row: row.clone(),
                current: row.base_revision == doc.head_revision,
            },
            Some(row) if row.status == RowStatus::Failed => ProcessingState::Failed {
                message: row.failure.clone().unwrap_or_default(),
            },
            _ => ProcessingState::Idle,
        };
        TakeProcessing {
            label: latest.map(|row| row.label.clone()).unwrap_or_default(),
            state,
            processed_head: (!doc.head_is_raw).then(|| doc.head_text.clone()),
        }
    }
}

/// Resumes the take's draft from its processing document.
pub(crate) fn draft_from_doc(id: &str, doc: &ProcessingDoc) -> Draft {
    let proposals = doc
        .proposals
        .iter()
        .filter_map(|row| {
            let status = match row.status {
                RowStatus::Proposed => ProposalStatus::Current,
                RowStatus::Accepted => ProposalStatus::Accepted,
                RowStatus::Rejected => ProposalStatus::Rejected,
                RowStatus::Superseded => ProposalStatus::Superseded,
                RowStatus::Failed => return None,
            };
            Some(StoredProposal {
                request_id: row.request_id.clone(),
                base_revision: row.base_revision,
                text: row.text.clone(),
                status,
            })
        })
        .collect();
    Draft::resume(
        id,
        id,
        doc.head_revision,
        &doc.head_text,
        if doc.head_is_raw {
            RegionKind::Raw
        } else {
            RegionKind::Processed
        },
        vec![Attempt {
            attempt_id: doc.raw_attempt_id.clone(),
            segment: 0,
            text: doc.raw_text.clone(),
        }],
        proposals,
    )
}

fn new_request_id() -> String {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|elapsed| elapsed.as_nanos())
        .unwrap_or_default();
    format!("p{nanos:x}-{}", NEXT.fetch_add(1, Ordering::Relaxed))
}

/// The user-facing sentence for a failed result. Raw text always stays.
pub(crate) fn failure_message(
    result: &TransformResult,
    label: &str,
    settings: &ProcessingSettings,
) -> String {
    let Some(failure) = &result.failure else {
        return String::new();
    };
    let what = match failure.reason {
        FailureReason::UnsupportedKind
            if failure.detail == pipeline::INSTRUCTION_CAPABLE_DETAIL =>
        {
            format!(
                "{label} cannot follow spoken instructions: an instruction \
                 requires an instruction-capable model."
            )
        }
        FailureReason::InvalidInput if failure.detail == pipeline::INSTRUCTION_EMPTY_DETAIL => {
            "A spoken instruction needs text to work on.".to_string()
        }
        FailureReason::InvalidInput if failure.detail == pipeline::INSTRUCTION_BLANK_DETAIL => {
            "The spoken instruction was empty.".to_string()
        }
        FailureReason::ProviderUnavailable if result.provider.kind == ProviderKind::S1 => format!(
            "S1-mini is not reachable at {}. Start a starling-serve with the S1-mini GGUF there. ({})",
            settings.s1_endpoint.trim(),
            failure.detail
        ),
        FailureReason::UnsupportedLanguage if result.provider.kind == ProviderKind::S1 => {
            "S1-mini only processes English.".to_string()
        }
        FailureReason::UnsupportedLanguage => {
            format!("{label} does not process this language.")
        }
        FailureReason::RateLimited => format!(
            "{label} is rate-limiting requests; try again later. ({})",
            failure.detail
        ),
        FailureReason::Timeout => {
            format!("{label} did not answer within {} s.", DEADLINE_MS / 1000)
        }
        FailureReason::UnknownModel => {
            format!("{label} does not know this model. ({})", failure.detail)
        }
        FailureReason::TruncatedOutput => {
            format!("{label} returned a cut-off answer. ({})", failure.detail)
        }
        FailureReason::EmptyOutput => format!("{label} returned no text."),
        FailureReason::RemoteForbidden => "This mode is local-only; nothing was sent.".to_string(),
        _ => format!("{}: {}", failure.reason.as_str(), failure.detail),
    };
    format!("{what} The raw transcript is unchanged.")
}

/// A spoken phrase whose command region could not be marked would reach
/// the model as text, so the take is refused instead.
fn marking_refused(kind: &str) -> String {
    format!(
        "The spoken {kind} could not be kept out of the text; nothing was \
         sent. The raw transcript is unchanged."
    )
}

/// Whether a phrase-routed plan would send the take to a remote provider,
/// or remote context, the active mode's plan does not already use. A
/// leading phrase recognized after capture cannot authorize additional
/// providers or context (decision.schema.json's permission boundary).
fn widens_scope(routed: &Plan, active: &Plan) -> bool {
    let Plan::Model {
        provider,
        context_fields,
    } = routed
    else {
        return false;
    };
    let decl = provider.declaration();
    if decl.locality != Locality::Remote {
        return false;
    }
    match active {
        Plan::Model {
            provider: active_provider,
            context_fields: active_fields,
        } => {
            active_provider.declaration().reference() != decl.reference()
                || !context_fields
                    .iter()
                    .all(|field| active_fields.contains(field))
        }
        _ => true,
    }
}

/// Marks a trailing instruction as the draft's command region, so it
/// travels as the request's instruction and never as input. Only a mode
/// that takes instructions parses one; elsewhere "Starling, …" or
/// "pounds sterling" is ordinary text. `prefix` is the routed phrase's
/// length, which the payload's offsets skip. False when the region
/// cannot be marked.
fn mark_instruction(draft: &mut Draft, mode: &ModeEntry, prefix: usize) -> bool {
    if !mode
        .transform_kinds
        .iter()
        .any(|kind| kind.needs_instructions())
    {
        return true;
    }
    let split =
        starling_processing::instructions::split(&draft.payload_text(), mode.language.as_deref());
    match (split.delimiter_span, split.instruction_span) {
        (Some((start, _)), Some((_, end))) => {
            draft.mark_command(
                start + prefix,
                end + prefix,
                CommandKind::TrailingInstruction,
            ) == Outcome::Applied
        }
        _ => true,
    }
}

fn result_kind(status: ResultStatus) -> ResultKind {
    match status {
        ResultStatus::Completed => ResultKind::Completed,
        ResultStatus::Failed => ResultKind::Failed,
        ResultStatus::Cancelled => ResultKind::Cancelled,
    }
}

/// The row a finished job stores, if any: proposals (current, stale or
/// superseded) and failures. A cancelled or discarded job stores none.
fn stored_row(
    request: &TransformRequest,
    result: &TransformResult,
    outcome: Outcome,
    label: &str,
    message: &str,
) -> Option<ProposalRow> {
    let status = match (result.status, outcome) {
        (ResultStatus::Completed, Outcome::Current | Outcome::Stale) => RowStatus::Proposed,
        (ResultStatus::Completed, Outcome::Superseded) => RowStatus::Superseded,
        (ResultStatus::Failed, Outcome::Failed) => RowStatus::Failed,
        _ => return None,
    };
    Some(ProposalRow {
        request_id: request.request_id.clone(),
        base_revision: request.base_revision,
        text: result.text.clone().unwrap_or_default(),
        status,
        label: label.to_string(),
        failure: result.failure.as_ref().map(|_| message.to_string()),
        stop_to_result_ms: result.timing.stop_to_result_ms,
    })
}

impl StarlingApp {
    pub(crate) fn active_mode(&self) -> &'static ModeEntry {
        mode(&self.processing_settings.mode)
    }

    /// Whether the active mode does anything after transcription.
    pub(crate) fn mode_processes(&self) -> bool {
        !matches!(
            pipeline::plan(self.active_mode(), &self.providers.registry),
            Plan::Nothing
        )
    }

    fn job_is(&self, id: &str, request_id: &str) -> bool {
        self.processing_jobs
            .get(id)
            .is_some_and(|(running, _)| running == request_id)
    }

    fn set_processing(&mut self, id: &str, label: String, state: ProcessingState) {
        let processed_head = self
            .processing
            .get(id)
            .and_then(|take| take.processed_head.clone());
        self.processing.insert(
            id.to_string(),
            TakeProcessing {
                label,
                state,
                processed_head,
            },
        );
    }

    /// A take just got its transcript: the active mode processes it.
    /// Whatever was used or proposed for an earlier transcript of the take
    /// (a re-transcription) no longer applies, whether or not the mode
    /// processes the new one.
    pub(crate) fn after_transcription(&mut self, id: String, cx: &mut Context<Self>) {
        // A staged take rebases its draft instead (#297).
        if self.staged_transcript(&id, cx) {
            return;
        }
        self.processing.remove(&id);
        self.drafts.remove(&id);
        self.processing_loading.remove(&id);
        if self.mode_processes() {
            self.process_take(id, cx);
        } else {
            if let Some((_, cancel)) = self.processing_jobs.remove(&id) {
                cancel.cancel();
            }
            self.stop_instants.remove(&id);
            cx.notify();
        }
    }

    /// Loads a take's stored processing state (on selection). While it
    /// loads, [`Self::head_text`] does not guess.
    pub(crate) fn load_processing(&mut self, id: String, cx: &mut Context<Self>) {
        if self.processing.contains_key(&id) || !self.processing_loading.insert(id.clone()) {
            return;
        }
        let Some(store) = self.store.clone() else {
            self.processing_loading.remove(&id);
            return;
        };
        cx.spawn(async move |this, cx| {
            let loaded = {
                let id = id.clone();
                cx.background_spawn(async move { store.processing_doc(&id) })
                    .await
            };
            this.update(cx, |app, cx| {
                // A load a newer transcript overtook was withdrawn: its
                // document may be for the old text.
                let wanted = app.processing_loading.remove(&id);
                if let (true, Err(err)) = (wanted, &loaded) {
                    // Copy/Export fall back to raw without the document;
                    // say it could not be read rather than guess silently.
                    app.error = Some(format!(
                        "Could not load this take's processing history ({err}); Copy and Export use the raw transcript."
                    ));
                }
                if let (true, Ok(Some(doc))) = (wanted, loaded) {
                    if !app.processing.contains_key(&id) {
                        app.drafts
                            .entry(id.clone())
                            .or_insert_with(|| draft_from_doc(&id, &doc));
                        app.processing.insert(id, TakeProcessing::from_doc(&doc));
                    }
                }
                cx.notify();
            })
            .ok();
        })
        .detach();
    }

    /// Ends a job before anything runs: the draft is kept, the superseded
    /// request (if any) is cancelled, and the take shows `state`.
    fn end_job(
        &mut self,
        id: &str,
        mut draft: Draft,
        retry_of: Option<&str>,
        label: String,
        state: ProcessingState,
        cx: &mut Context<Self>,
    ) {
        if let Some(earlier) = retry_of {
            draft.cancel(earlier);
        }
        self.drafts.insert(id.to_string(), draft);
        self.processing_jobs.remove(id);
        self.set_processing(id, label, state);
        cx.notify();
    }

    /// Runs the take's routed mode on its latest transcript: a leading
    /// spoken phrase overrides the active mode for this take only. A job
    /// already running for the take is superseded and cancelled.
    pub(crate) fn process_take(&mut self, id: String, cx: &mut Context<Self>) {
        let Some(store) = self.store.clone() else {
            return;
        };
        let active = self.active_mode();
        let settings = self.processing_settings.clone();
        // Taken whatever the plan: a take that is not processed must not
        // keep its stop instant.
        let stopped_at: Option<Instant> = self.stop_instants.remove(&id);
        // The plan is only known once the take's text is loaded and routed.
        let retry_of = self
            .processing_jobs
            .remove(&id)
            .map(|(request_id, cancel)| {
                cancel.cancel();
                request_id
            });
        let request_id = new_request_id();
        let cancel = CancelToken::new();
        self.processing_jobs
            .insert(id.clone(), (request_id.clone(), cancel.clone()));
        self.set_processing(&id, String::new(), ProcessingState::Running);
        cx.notify();

        cx.spawn(async move |this, cx| {
            // 1. The durable state for the take's current raw transcript.
            let prepared = {
                let store = store.clone();
                let id = id.clone();
                cx.background_spawn(async move {
                    let (attempt_id, raw) = store
                        .latest_raw(&id)?
                        .ok_or_else(|| storage::StorageError::NotFound(id.clone()))?;
                    store.start_processing_doc(&id, &attempt_id, &raw)
                })
                .await
            };
            let doc = match prepared {
                Ok(doc) => doc,
                Err(err) => {
                    this.update(cx, |app, cx| {
                        if app.job_is(&id, &request_id) {
                            app.processing_jobs.remove(&id);
                            app.set_processing(
                                &id,
                                String::new(),
                                ProcessingState::Failed {
                                    message: format!("Could not start processing: {err}"),
                                },
                            );
                            cx.notify();
                        }
                    })
                    .ok();
                    return;
                }
            };

            // 2. Route the take, mark its spoken phrases, and build the
            //    request pinned to the draft's revision.
            let built = this
                .update(cx, |app, cx| {
                    if !app.job_is(&id, &request_id) {
                        return None;
                    }
                    // Keep the live draft when it is still this document
                    // (it knows the requests in flight); otherwise resume.
                    // A draft ahead of the stored head is the staging
                    // editor's, with edits not yet written (#297).
                    let mut draft = match app.drafts.remove(&id) {
                        Some(draft)
                            if draft.revision() >= doc.head_revision
                                && draft.attempts().first().map(|a| a.attempt_id.as_str())
                                    == Some(doc.raw_attempt_id.as_str()) =>
                        {
                            draft
                        }
                        _ => draft_from_doc(&id, &doc),
                    };
                    // Re-mark from the current text each run, so a region an
                    // edit moved or removed cannot survive.
                    draft.clear_commands();
                    let retired = retry_of.as_deref();
                    let mut prefix = 0;
                    let mode = match route_take(modes(), &draft.text(), active) {
                        TakeRoute::Active => active,
                        TakeRoute::Override { mode, span } => {
                            if draft.mark_command(span.0, span.1, CommandKind::ModePhrase)
                                != Outcome::Applied
                            {
                                let message = marking_refused("phrase");
                                let state = ProcessingState::Failed { message };
                                app.end_job(&id, draft, retired, String::new(), state, cx);
                                return None;
                            }
                            prefix = span.1;
                            mode
                        }
                        TakeRoute::Conflict(message) => {
                            let state = ProcessingState::Failed { message };
                            app.end_job(&id, draft, retired, String::new(), state, cx);
                            return None;
                        }
                    };
                    let plan = pipeline::plan(mode, &app.providers.registry);
                    if prefix > 0
                        && widens_scope(&plan, &pipeline::plan(active, &app.providers.registry))
                    {
                        let message = format!(
                            "The leading phrase selects {}, which would send this take \
                             where {} does not; a spoken phrase cannot add a provider. \
                             Nothing was sent. The raw transcript is unchanged.",
                            mode.name, active.name
                        );
                        let state = ProcessingState::Failed { message };
                        app.end_job(&id, draft, retired, String::new(), state, cx);
                        return None;
                    }
                    let (decl, provider, fields): (
                        ProviderDecl,
                        Option<Arc<dyn Provider>>,
                        Vec<ContextField>,
                    ) = match plan {
                        // A phrase routed the take to a mode that processes
                        // nothing (the literal escape): the builtin step
                        // offers the payload without the phrase as a proposal.
                        Plan::Nothing if prefix > 0 => {
                            (pipeline::builtin_declaration(), None, Vec::new())
                        }
                        Plan::Nothing => {
                            app.end_job(
                                &id,
                                draft,
                                retired,
                                String::new(),
                                ProcessingState::Idle,
                                cx,
                            );
                            return None;
                        }
                        Plan::Blocked(block) => {
                            let message = blocked_text(block, mode, &app.providers.problems);
                            let state = ProcessingState::Blocked { message };
                            app.end_job(&id, draft, retired, String::new(), state, cx);
                            return None;
                        }
                        Plan::Builtin => (pipeline::builtin_declaration(), None, Vec::new()),
                        Plan::Model {
                            provider,
                            context_fields,
                        } => (
                            provider.declaration().clone(),
                            Some(provider),
                            context_fields,
                        ),
                    };
                    let label = provider_label(&decl, &settings);
                    app.set_processing(&id, label.clone(), ProcessingState::Running);
                    if !mark_instruction(&mut draft, mode, prefix) {
                        let message = marking_refused("instruction");
                        let state = ProcessingState::Failed { message };
                        app.end_job(&id, draft, retired, label, state, cx);
                        return None;
                    }
                    let retry = retry_of.filter(|earlier| draft.request(earlier).is_some());
                    if draft.request_transform(&request_id, retry.as_deref()) != Outcome::Pending {
                        app.drafts.insert(id.clone(), draft);
                        app.processing_jobs.remove(&id);
                        app.set_processing(
                            &id,
                            label.clone(),
                            ProcessingState::Failed {
                                message: "This take cannot be processed right now.".to_string(),
                            },
                        );
                        cx.notify();
                        return None;
                    }
                    let input_chars = draft.payload_text().chars().count() as u64;
                    let (request, refusal) = match pipeline::build_request(
                        &draft,
                        mode,
                        &decl,
                        &fields,
                        &ContextValues::default(),
                        &RequestOptions {
                            request_id: request_id.clone(),
                            retry_of: retry,
                            deadline_ms: DEADLINE_MS,
                            // Cleanup shortens text; four times the input is
                            // far past any honest answer.
                            max_output_chars: input_chars.saturating_mul(4).max(2_000),
                        },
                    ) {
                        Ok(request) => (request, None),
                        Err((request, failure)) => (request, Some(failure)),
                    };
                    app.drafts.insert(id.clone(), draft);
                    cx.notify();
                    Some((request, refusal, provider, label))
                })
                .ok()
                .flatten();
            let Some((request, refusal, provider, label)) = built else {
                return;
            };

            // 3. The job itself, off the UI thread. A refused request
            //    never runs; its failure is recorded like any other.
            let result = match refusal {
                Some(failure) => pipeline::failure_result(
                    &request,
                    failure,
                    Timing {
                        queued_ms: 0.0,
                        processing_ms: 0.0,
                        stop_to_result_ms: stopped_at.map(|at| at.elapsed().as_secs_f64() * 1000.0),
                    },
                ),
                None => {
                    let request = request.clone();
                    let cancel = cancel.clone();
                    cx.background_spawn(async move {
                        let clock = Clock {
                            queued_ms: 0.0,
                            stopped_at,
                        };
                        pipeline::run(&request, provider.as_deref(), clock, &mut |_| {}, &cancel)
                    })
                    .await
                }
            };

            // 4. The draft judges the result; only the view changes here.
            let judged = this
                .update(cx, |app, cx| {
                    let outcome = match app.drafts.get_mut(&id) {
                        Some(draft) => draft.result(
                            &request.request_id,
                            result_kind(result.status),
                            result.text.as_deref(),
                        ),
                        None => Outcome::Discarded,
                    };
                    let owned = app.job_is(&id, &request.request_id);
                    if owned {
                        app.processing_jobs.remove(&id);
                    }
                    let message = failure_message(&result, &label, &settings);
                    // A failure only matters for the job that still owns
                    // the take; a superseded job's failure is not stored,
                    // or a restart could resurrect it over a newer result.
                    let row = stored_row(&request, &result, outcome, &label, &message)
                        .filter(|row| owned || row.status != RowStatus::Failed);
                    match (&row, outcome) {
                        (Some(row), Outcome::Current | Outcome::Stale) => {
                            app.set_processing(
                                &id,
                                label.clone(),
                                ProcessingState::Proposal {
                                    row: row.clone(),
                                    current: outcome == Outcome::Current,
                                },
                            );
                        }
                        (_, Outcome::Failed) if owned => {
                            app.set_processing(
                                &id,
                                label.clone(),
                                ProcessingState::Failed { message },
                            );
                        }
                        // The job ended with nothing to show (discarded,
                        // cancelled, superseded): the drawer must not keep
                        // spinning for a job that no longer exists.
                        _ if owned => {
                            app.set_processing(&id, label.clone(), ProcessingState::Idle);
                        }
                        _ => {}
                    }
                    cx.notify();
                    (outcome, row)
                })
                .ok();
            let Some((outcome, row)) = judged else {
                return;
            };

            // 5. Persist the proposal and the latency record. A take
            // deleted meanwhile answers NotFound: nothing lands.
            let event = ProcessingRecorded::new(
                &request,
                &result,
                Arrival::from_outcome(outcome),
                now_iso(),
            );
            let persisted = cx
                .background_spawn(async move {
                    if let Some(row) = row {
                        store.save_proposal(&id, &row)?;
                    }
                    let payload = serde_json::to_string(&event)
                        .map_err(|err| storage::StorageError::Invalid(err.to_string()))?;
                    store.record_insight(
                        &id,
                        &event.event_id,
                        "processing_recorded",
                        &event.occurred_at,
                        &payload,
                    )
                })
                .await;
            if let Err(err) = persisted {
                if !matches!(err, storage::StorageError::NotFound(_)) {
                    this.update(cx, |app, cx| {
                        app.error = Some(format!("Could not store the processing result: {err}"));
                        cx.notify();
                    })
                    .ok();
                }
            }
        })
        .detach();
    }

    /// Stops a running job; its result, if it still arrives, lands
    /// nowhere. The raw transcript is untouched.
    pub(crate) fn cancel_processing(&mut self, id: &str, cx: &mut Context<Self>) {
        if let Some((request_id, cancel)) = self.processing_jobs.remove(id) {
            cancel.cancel();
            if let Some(draft) = self.drafts.get_mut(id) {
                draft.cancel(&request_id);
            }
            let label = self
                .processing
                .get(id)
                .map(|take| take.label.clone())
                .unwrap_or_default();
            self.set_processing(id, label, ProcessingState::Cancelled);
            cx.notify();
        }
    }

    /// Forgets a take's processing (it is being deleted): the job is
    /// cancelled and its draft deleted, so a late result is discarded.
    pub(crate) fn drop_processing(&mut self, id: &str) {
        self.drop_staging_for(id);
        if let Some((_, cancel)) = self.processing_jobs.remove(id) {
            cancel.cancel();
        }
        if let Some(draft) = self.drafts.get_mut(id) {
            draft.delete_draft();
        }
        self.processing.remove(id);
        self.processing_loading.remove(id);
        self.stop_instants.remove(id);
    }

    /// "Use processed": the proposal becomes the take's head, if it is
    /// still current (or the user forces a stale one).
    pub(crate) fn accept_processed(&mut self, id: &str, force: bool, cx: &mut Context<Self>) {
        let Some(store) = self.store.clone() else {
            return;
        };
        let Some(TakeProcessing {
            label,
            state: ProcessingState::Proposal { row, .. },
            ..
        }) = self.processing.get(id).cloned()
        else {
            return;
        };
        let Some(draft) = self.drafts.get_mut(id) else {
            // Proposals are shown only once the draft is loaded, so this
            // is not expected; say so instead of dropping the click.
            self.error = Some(self.missing_draft_message(id));
            cx.notify();
            return;
        };
        match draft.accept(&row.request_id, force) {
            Outcome::Applied => {}
            Outcome::StaleRejected => {
                self.set_processing(
                    id,
                    label,
                    ProcessingState::Proposal {
                        row,
                        current: false,
                    },
                );
                cx.notify();
                return;
            }
            _ => return,
        }
        let revision = draft.revision();
        let text = draft.text();
        let attempt_id = draft
            .attempts()
            .first()
            .map(|attempt| attempt.attempt_id.clone())
            .unwrap_or_default();
        let accepted = ProposalRow {
            status: RowStatus::Accepted,
            ..row
        };
        self.processing.insert(
            id.to_string(),
            TakeProcessing {
                label,
                state: ProcessingState::Idle,
                processed_head: Some(text.clone()),
            },
        );
        cx.notify();
        self.persist_head(
            store,
            id.to_string(),
            revision,
            text,
            false,
            attempt_id,
            Some(accepted),
            cx,
        );
    }

    /// "Back to raw": the raw transcript becomes the head again.
    pub(crate) fn revert_to_raw(&mut self, id: &str, cx: &mut Context<Self>) {
        let Some(store) = self.store.clone() else {
            return;
        };
        let Some(draft) = self.drafts.get_mut(id) else {
            self.error = Some(self.missing_draft_message(id));
            cx.notify();
            return;
        };
        if draft.revert_raw() != Outcome::Applied {
            return;
        }
        let revision = draft.revision();
        let text = draft.text();
        let attempt_id = draft
            .attempts()
            .first()
            .map(|attempt| attempt.attempt_id.clone())
            .unwrap_or_default();
        if let Some(take) = self.processing.get_mut(id) {
            take.processed_head = None;
            if let ProcessingState::Proposal { current, .. } = &mut take.state {
                *current = false;
            }
        }
        cx.notify();
        self.persist_head(
            store,
            id.to_string(),
            revision,
            text,
            true,
            attempt_id,
            None,
            cx,
        );
    }

    /// "Dismiss": the proposal is rejected and stays in history.
    pub(crate) fn dismiss_processed(&mut self, id: &str, cx: &mut Context<Self>) {
        let Some(store) = self.store.clone() else {
            return;
        };
        let Some(TakeProcessing {
            label,
            state: ProcessingState::Proposal { row, .. },
            ..
        }) = self.processing.get(id).cloned()
        else {
            return;
        };
        if let Some(draft) = self.drafts.get_mut(id) {
            draft.reject(&row.request_id);
        }
        self.set_processing(id, label, ProcessingState::Idle);
        cx.notify();
        let id = id.to_string();
        let rejected = ProposalRow {
            status: RowStatus::Rejected,
            ..row
        };
        cx.spawn(async move |this, cx| {
            let saved = cx
                .background_spawn(async move { store.save_proposal(&id, &rejected) })
                .await;
            if let Err(err) = saved {
                if !matches!(err, storage::StorageError::NotFound(_)) {
                    this.update(cx, |app, cx| {
                        app.error = Some(format!("Could not store the change: {err}"));
                        cx.notify();
                    })
                    .ok();
                }
            }
        })
        .detach();
    }

    /// Why a proposal's buttons found no draft: still loading, or the
    /// take changed under them (a new transcript replaced it).
    fn missing_draft_message(&self, id: &str) -> String {
        if self.processing_loading.contains(id) {
            "This take's history is still loading; try again.".to_string()
        } else {
            "This take changed since the result was shown; run the mode again.".to_string()
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn persist_head(
        &mut self,
        store: crate::store::Store,
        id: String,
        revision: u64,
        text: String,
        is_raw: bool,
        attempt_id: String,
        accepted: Option<ProposalRow>,
        cx: &mut Context<Self>,
    ) {
        let accepted = self.staging_head_started(&id, revision, accepted);
        cx.spawn(async move |this, cx| {
            let write_id = id.clone();
            let write_attempt = attempt_id.clone();
            let saved = cx
                .background_spawn(async move {
                    store.commit_processing_head(
                        &write_id,
                        revision,
                        &text,
                        is_raw,
                        &write_attempt,
                        accepted.as_ref(),
                    )
                })
                .await;
            this.update(cx, |app, cx| {
                if app.staging_head_finished(&id, &attempt_id, revision, &saved, cx) {
                    return;
                }
                if let Err(err) = saved {
                    if !matches!(err, storage::StorageError::NotFound(_)) {
                        app.error = Some(format!("Could not store the change: {err}"));
                        cx.notify();
                    }
                }
            })
            .ok();
        })
        .detach();
    }

    /// Copy/Export found no head because the take is still loading: say
    /// so rather than leave a dead click.
    pub(crate) fn note_head_loading(&mut self, id: &str, cx: &mut Context<Self>) {
        if self.processing_loading.contains(id) {
            self.error = Some("This take's history is still loading; try again.".to_string());
            cx.notify();
        }
    }

    /// The text Copy/Export use: the used processed text when the take's
    /// head is processed, the raw transcript otherwise. `None` while the
    /// take's processing state is still loading: the head is not known
    /// yet, and the raw text may not be it.
    pub(crate) fn head_text(&self, id: &str) -> Option<String> {
        if self.processing_loading.contains(id) {
            return None;
        }
        self.processing
            .get(id)
            .and_then(|take| take.processed_head.clone())
            .or_else(|| {
                self.sessions
                    .iter()
                    .find(|session| session.id == id)
                    .and_then(|session| session.transcript.as_ref())
                    .map(|transcript| transcript.text.clone())
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use starling_processing::contract::validate_processing;

    fn settings() -> ProcessingSettings {
        ProcessingSettings {
            api_model: "gpt-4.1-mini".to_string(),
            ..ProcessingSettings::default()
        }
    }

    #[test]
    fn an_unknown_saved_mode_is_announced() {
        assert!(unknown_mode_note("verbatim").is_none());
        let note = unknown_mode_note("retired-mode").expect("a note");
        assert!(
            note.contains("retired-mode") && note.contains("verbatim"),
            "{note}"
        );
    }

    #[test]
    fn a_bad_key_variable_name_is_a_problem_not_a_missing_key() {
        assert_eq!(api_key(""), Ok(None));
        assert_eq!(api_key("STARLING_TEST_SURELY_UNSET_KEY"), Ok(None));
        assert!(
            api_key("sk-live-abc").is_err(),
            "a pasted key is not a name"
        );
        assert!(api_key("1KEY").is_err());
        let problems = build_providers(&ProcessingSettings {
            api_key_env: "not a name".to_string(),
            ..settings()
        })
        .problems;
        assert!(
            problems[API_ROUTE].contains("environment variable"),
            "{problems:?}"
        );
    }

    #[test]
    fn the_built_in_modes_parse_and_pass_the_processing_rules() {
        validate_processing(modes()).expect("desktop modes are valid");
        assert_eq!(modes().default_profile, "verbatim");
        assert_eq!(mode("no-such-mode").id, "verbatim");
    }

    /// A capture with a finalized transcript, like the tests in
    /// `staging.rs` build one.
    fn saved_take_with_transcript(store: &crate::store::Store, text: &str) -> (String, String) {
        use starling_dictation::{audio, storage::TranscriptionResult};
        let wav = audio::encode_wav_16k(&audio::PcmAudio {
            samples: vec![0.; 160],
            sample_rate: 16_000,
            channels: 1,
        })
        .unwrap();
        let id = store
            .save_capture(std::sync::Arc::new(wav), None)
            .unwrap()
            .id;
        store.mark_attempt(&id, "test").unwrap();
        store
            .save_transcript(
                &id,
                TranscriptionResult {
                    text: text.into(),
                    segments: vec![],
                    duration_seconds: None,
                    request_id: None,
                },
            )
            .unwrap();
        let (attempt, _) = store.latest_raw(&id).unwrap().unwrap();
        (id, attempt)
    }

    #[gpui::test]
    fn a_leading_alias_routes_the_take_and_never_reaches_the_payload(
        cx: &mut gpui::TestAppContext,
    ) {
        // The longest alias ("clean this online") routes the take to
        // clean-api. The test settings have no API model, so the routed
        // plan is blocked, which is how the test sees the routed mode.
        let root = tempfile::tempdir().unwrap();
        let store = crate::store::Store::at_test_root(root.path());
        let raw = "Clean this online: send the report";
        let (id, _attempt) = saved_take_with_transcript(&store, raw);
        let app = cx.new(|cx| StarlingApp::for_test(Some(store.clone()), cx));
        app.update(cx, |app, cx| {
            app.processing_settings.mode = "clean-local".to_string();
            app.after_transcription(id.clone(), cx);
        });
        cx.run_until_parked();
        app.update(cx, |app, _| {
            let processing = app.processing.get(&id).expect("the take was routed");
            let ProcessingState::Blocked { message } = &processing.state else {
                panic!(
                    "the routed mode (clean-api) is not set up: {:?}",
                    processing.state
                );
            };
            assert!(message.contains("Set the API model"), "{message}");
            // The phrase is the take's ModePhrase command region: it never
            // reaches the payload, and the active mode is untouched.
            let draft = app.drafts.get(&id).expect("the draft survives");
            assert_eq!(draft.text(), raw);
            assert_eq!(draft.payload_text(), "send the report");
            assert_eq!(app.active_mode().id, "clean-local");
            let snapshot = draft.snapshot();
            let commands: Vec<_> = snapshot
                .regions
                .iter()
                .filter(|region| region.kind == RegionKind::Command)
                .collect();
            assert_eq!(commands.len(), 1, "{:?}", snapshot.regions);
            assert_eq!(commands[0].span, [0, 19]);
            assert_eq!(commands[0].command, Some(CommandKind::ModePhrase));
        });
    }

    /// Processes a saved take with `active` as the active mode.
    fn process_with(
        cx: &mut gpui::TestAppContext,
        store: &crate::store::Store,
        raw: &str,
        active: &str,
        with_api: bool,
    ) -> (gpui::Entity<StarlingApp>, String) {
        let (id, _attempt) = saved_take_with_transcript(store, raw);
        let app = cx.new(|cx| StarlingApp::for_test(Some(store.clone()), cx));
        app.update(cx, |app, cx| {
            if with_api {
                app.processing_settings = settings();
                app.providers = build_providers(&app.processing_settings);
            }
            app.processing_settings.mode = active.to_string();
            app.after_transcription(id.clone(), cx);
        });
        cx.run_until_parked();
        (app, id)
    }

    fn command_regions(app: &StarlingApp, id: &str) -> Vec<(Vec<usize>, Option<CommandKind>)> {
        app.drafts
            .get(id)
            .expect("the draft survives")
            .snapshot()
            .regions
            .iter()
            .filter(|region| region.kind == RegionKind::Command)
            .map(|region| (region.span.to_vec(), region.command))
            .collect()
    }

    #[gpui::test]
    fn a_leading_alias_cannot_add_a_remote_provider(cx: &mut gpui::TestAppContext) {
        // With the API set up, "clean this online" from the local-only
        // mode would send the take to the API: refused, nothing sent.
        let root = tempfile::tempdir().unwrap();
        let store = crate::store::Store::at_test_root(root.path());
        let raw = "clean this online: confidential text";
        let (app, id) = process_with(cx, &store, raw, "clean-local", true);
        app.update(cx, |app, _| {
            let processing = app.processing.get(&id).expect("the take was routed");
            let ProcessingState::Failed { message } = &processing.state else {
                panic!("the widening route is refused: {:?}", processing.state);
            };
            assert!(message.contains("cannot add a provider"), "{message}");
            assert_eq!(app.drafts.get(&id).unwrap().text(), raw);
            assert!(
                store
                    .processing_doc(&id)
                    .unwrap()
                    .unwrap()
                    .proposals
                    .is_empty()
            );
        });
    }

    #[gpui::test]
    fn ordinary_dictation_is_never_an_instruction_in_a_cleanup_mode(cx: &mut gpui::TestAppContext) {
        let root = tempfile::tempdir().unwrap();
        let store = crate::store::Store::at_test_root(root.path());
        let raw = "I paid five pounds sterling yesterday";
        let (app, id) = process_with(cx, &store, raw, "clean-local", false);
        app.update(cx, |app, _| {
            assert!(command_regions(app, &id).is_empty());
            assert_eq!(app.drafts.get(&id).unwrap().payload_text(), raw);
            // The take ran on S1-mini (not running in tests) as plain text.
            let ProcessingState::Failed { message } = &app.processing[&id].state else {
                panic!("{:?}", app.processing[&id].state);
            };
            assert!(message.contains("S1-mini is not reachable"), "{message}");
        });
    }

    #[gpui::test]
    fn the_literal_escape_proposes_the_payload_without_the_escape(cx: &mut gpui::TestAppContext) {
        let root = tempfile::tempdir().unwrap();
        let store = crate::store::Store::at_test_root(root.path());
        let raw = "literal send the report";
        let (app, id) = process_with(cx, &store, raw, "clean-local", false);
        app.update(cx, |app, _| {
            let ProcessingState::Proposal { row, current } = &app.processing[&id].state else {
                panic!("{:?}", app.processing[&id].state);
            };
            assert!(current);
            assert_eq!(row.text, "send the report");
            assert_eq!(app.drafts.get(&id).unwrap().raw_text(), raw);
        });
    }

    #[gpui::test]
    fn command_regions_are_recomputed_from_the_edited_text(cx: &mut gpui::TestAppContext) {
        // A phrase region from an earlier run never survives the edit
        // that made the phrase stop matching.
        let root = tempfile::tempdir().unwrap();
        let store = crate::store::Store::at_test_root(root.path());
        let (app, id) = process_with(
            cx,
            &store,
            "clean this send the report",
            "clean-local",
            false,
        );
        app.update(cx, |app, _| {
            assert_eq!(
                command_regions(app, &id),
                vec![(vec![0, 11], Some(CommandKind::ModePhrase))]
            );
        });
        app.update(cx, |app, cx| {
            app.drafts.get_mut(&id).unwrap().insert(0, "now ");
            app.process_take(id.clone(), cx);
        });
        cx.run_until_parked();
        app.update(cx, |app, _| {
            assert!(command_regions(app, &id).is_empty());
            let draft = app.drafts.get(&id).unwrap();
            assert_eq!(draft.payload_text(), draft.text());
        });
    }

    #[test]
    fn trailing_instructions_are_parsed_only_by_instruction_modes() {
        let text = "send the report Starling, make it formal";
        let draft = || {
            let mut draft = Draft::new("d", "c");
            draft.final_attempt(0, "a", text);
            draft
        };
        let mut cleanup = draft();
        assert!(mark_instruction(&mut cleanup, mode("clean-local"), 0));
        assert_eq!(cleanup.instruction(), None);

        let mut rewrite_mode = mode("clean-api").clone();
        rewrite_mode.transform_kinds = vec![TransformKind::Rewrite];
        let mut rewrite = draft();
        assert!(mark_instruction(&mut rewrite, &rewrite_mode, 0));
        assert_eq!(
            rewrite.instruction().as_deref(),
            Some("Starling, make it formal")
        );
        // A live partial over the delimiter cannot be marked.
        let mut live = draft();
        live.partial(1, " now Starling, formalize");
        assert!(!mark_instruction(&mut live, &rewrite_mode, 0));
    }

    #[test]
    fn route_take_uses_the_frozen_oracle() {
        // The longest alias wins over the active mode's own shorter one.
        assert!(matches!(
            route_take(modes(), "Clean this online: send the report", mode("clean-local")),
            TakeRoute::Override { mode, span: (0, 19) } if mode.id == "clean-api"
        ));
        // The literal escape routes this one take to verbatim.
        assert!(matches!(
            route_take(modes(), "literal clean this up please", mode("clean-local")),
            TakeRoute::Override { mode, span: (0, 8) } if mode.id == "verbatim"
        ));
        // Mid-sentence mentions never match, and a mode that disallows
        // spoken overrides (verbatim) never has its text parsed.
        assert!(matches!(
            route_take(modes(), "please clean this up", mode("clean-local")),
            TakeRoute::Active
        ));
        assert!(matches!(
            route_take(modes(), "clean this: hello", mode("verbatim")),
            TakeRoute::Active
        ));
    }

    #[test]
    fn a_leading_alias_conflict_surfaces_the_candidates() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(
            "../../../../packages/contracts/mode-routing/fixtures/profiles-alias-collision.json",
        );
        let doc: ProfilesDocument = serde_json::from_str(
            &std::fs::read_to_string(&path)
                .unwrap_or_else(|err| panic!("{}: {err}", path.display())),
        )
        .unwrap();
        let active = doc.profile("capture").unwrap();
        let TakeRoute::Conflict(message) = route_take(&doc, "code this: hello", active) else {
            panic!("equal-rank phrases conflict");
        };
        assert!(
            message.contains("faithful") && message.contains("code-guidance"),
            "{message}"
        );
        assert!(message.contains("raw transcript is unchanged"), "{message}");
    }

    #[test]
    fn modes_route_to_the_desktop_providers_without_fallback() {
        let providers = vec![s1_declaration(), api_declaration("m")];
        assert!(matches!(
            processing_route(mode("clean-local"), &providers),
            ProcessingRoute::Ready { provider, .. } if provider.id == "local-s1"
        ));
        assert!(matches!(
            processing_route(mode("clean-api"), &providers),
            ProcessingRoute::Ready { provider, .. } if provider.id == "remote-api"
        ));
        assert!(matches!(
            processing_route(mode("clean-local"), &providers[1..]),
            ProcessingRoute::Blocked(RouteBlock::ProviderUnavailable)
        ));
        assert_eq!(
            processing_route(mode("verbatim"), &providers),
            ProcessingRoute::None
        );
    }

    #[test]
    fn the_disclosure_names_the_destination_and_the_context_sent() {
        let local = disclosure("clean-local", &settings());
        assert!(local.contains("http://127.0.0.1:8182"), "{local}");
        assert!(local.contains("Context sent: none"), "{local}");
        assert!(local.contains("S1-mini by Superwhisper"), "{local}");
        let api = disclosure("clean-api", &settings());
        assert!(api.contains("https://api.openai.com/v1"), "{api}");
        assert!(api.contains("gpt-4.1-mini"), "{api}");
        assert!(api.contains("$OPENAI_API_KEY"), "{api}");
        assert!(api.contains("Context sent: none"), "{api}");
        let raw = disclosure("verbatim", &settings());
        assert!(raw.contains("Nothing is sent"), "{raw}");
    }

    #[test]
    fn a_mode_that_is_not_set_up_says_why_instead_of_falling_back() {
        let unset = ProcessingSettings::default();
        assert!(disclosure("clean-api", &unset).contains("Set the API model"));
        let remote_s1 = ProcessingSettings {
            s1_endpoint: "http://192.168.1.20:8182".to_string(),
            ..settings()
        };
        let text = disclosure("clean-local", &remote_s1);
        assert!(text.starts_with("Not set up: S1-mini endpoint"), "{text}");
        let insecure = ProcessingSettings {
            api_endpoint: "http://api.example.com/v1".to_string(),
            ..settings()
        };
        assert!(disclosure("clean-api", &insecure).contains("https"));
    }

    #[test]
    fn a_stored_document_resumes_into_the_same_view() {
        let row = ProposalRow {
            request_id: "p1".to_string(),
            base_revision: 1,
            text: "Clean.".to_string(),
            status: RowStatus::Proposed,
            label: "S1-mini · this computer".to_string(),
            failure: None,
            stop_to_result_ms: Some(900.0),
        };
        let doc = ProcessingDoc {
            head_revision: 1,
            head_text: "um clean".to_string(),
            head_is_raw: true,
            raw_attempt_id: "a1".to_string(),
            raw_text: "um clean".to_string(),
            proposals: vec![row.clone()],
        };
        let view = TakeProcessing::from_doc(&doc);
        assert_eq!(view.state, ProcessingState::Proposal { row, current: true });
        let mut draft = draft_from_doc("take", &doc);
        assert_eq!(draft.proposal_status("p1"), Some(ProposalStatus::Current));
        assert_eq!(draft.accept("p1", false), Outcome::Applied);
        assert_eq!(draft.text(), "Clean.");
        assert_eq!(draft.raw_text(), "um clean");
    }

    #[test]
    fn a_proposal_read_before_the_head_moved_resumes_stale() {
        let doc = ProcessingDoc {
            head_revision: 2,
            head_text: "Used earlier.".to_string(),
            head_is_raw: false,
            raw_attempt_id: "a1".to_string(),
            raw_text: "um used earlier".to_string(),
            proposals: vec![ProposalRow {
                request_id: "late".to_string(),
                base_revision: 1,
                text: "Late.".to_string(),
                status: RowStatus::Proposed,
                label: String::new(),
                failure: None,
                stop_to_result_ms: None,
            }],
        };
        assert!(matches!(
            TakeProcessing::from_doc(&doc).state,
            ProcessingState::Proposal { current: false, .. }
        ));
        let mut draft = draft_from_doc("take", &doc);
        assert_eq!(draft.accept("late", false), Outcome::StaleRejected);
        assert_eq!(draft.text(), "Used earlier.");
    }

    #[test]
    fn only_proposals_and_failures_are_stored() {
        let decl = s1_declaration();
        let mut draft = Draft::new("t", "t");
        draft.final_attempt(0, "a", "hello");
        draft.request_transform("r", None);
        let request = pipeline::build_request(
            &draft,
            mode("clean-local"),
            &decl,
            &[],
            &ContextValues::default(),
            &RequestOptions {
                request_id: "r".to_string(),
                retry_of: None,
                deadline_ms: 1,
                max_output_chars: 10,
            },
        )
        .unwrap();
        let cancelled = pipeline::run(&request, None, Clock::default(), &mut |_| {}, &{
            let token = CancelToken::new();
            token.cancel();
            token
        });
        assert_eq!(cancelled.status, ResultStatus::Cancelled);
        assert!(stored_row(&request, &cancelled, Outcome::Discarded, "l", "m").is_none());
    }
}
