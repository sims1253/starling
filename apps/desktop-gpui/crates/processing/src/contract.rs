//! The #293 records as Rust types: the mode entry
//! (`mode.schema.json`), provider declarations (`provider.schema.json`),
//! transform requests and results (`transform-request.schema.json`,
//! `transform-result.schema.json`), plus the ports of the oracle's
//! cross-field rules in `tests/mode_routing.py` (`validate_processing`,
//! `validate_provider`, `processing_route`).
//!
//! Every struct denies unknown fields, the serde counterpart of the
//! schemas' `additionalProperties: false`, so a record this build does not
//! understand fails to parse instead of silently losing a field.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransformKind {
    Clean,
    Format,
    Rewrite,
    Translate,
}

impl TransformKind {
    /// Kinds that follow a free-form instruction; only providers with
    /// `instructions: true` may take them.
    pub fn needs_instructions(self) -> bool {
        matches!(self, TransformKind::Rewrite | TransformKind::Translate)
    }

    pub fn as_str(self) -> &'static str {
        match self {
            TransformKind::Clean => "clean",
            TransformKind::Format => "format",
            TransformKind::Rewrite => "rewrite",
            TransformKind::Translate => "translate",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Formality {
    Casual,
    SemiCasual,
    SemiFormal,
    Formal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Structure {
    Prose,
    Lists,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StyleContext {
    General,
    Email,
}

/// The closed style axes. They map one to one onto S1-mini's control
/// line; instruction providers render them into their prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Style {
    pub formality: Formality,
    pub structure: Structure,
    pub context: StyleContext,
}

impl Style {
    /// The S1-mini control values (`cpp/s1/config.hpp` value sets).
    pub fn s1_controls(&self) -> (&'static str, &'static str, &'static str) {
        let styling = match self.formality {
            Formality::Casual => "casual",
            Formality::SemiCasual => "semi-casual",
            Formality::SemiFormal => "semi-formal",
            Formality::Formal => "formal",
        };
        let structure = match self.structure {
            Structure::Prose => "prose",
            Structure::Lists => "lists",
        };
        let context = match self.context {
            StyleContext::General => "general",
            StyleContext::Email => "email",
        };
        (styling, structure, context)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContextField {
    PersonalContext,
    Vocabulary,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Intent {
    Dictate,
    Edit,
    Ask,
    Capture,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Behavior {
    Faithful,
    Verbatim,
    CodeGuidance,
    EditSelection,
    AskSelection,
    Capture,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SelectedText {
    Off,
    Reference,
    EditTarget,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Delivery {
    Insert,
    ReplaceSelection,
    Preview,
    /// History only: the take is kept, nothing is inserted anywhere.
    SaveNote,
    Copy,
    /// Insert, then press Enter. Explicit per-mode opt-in only.
    InsertEnter,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProcessingDelivery {
    /// Raw text is delivered at once; a processed revision may replace it
    /// only while the target is unchanged.
    Direct,
    /// Output waits in the draft until the user accepts a revision.
    Staged,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Snippet {
    pub spoken: String,
    pub expansion: String,
}

/// One mode entry (`mode.schema.json`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModeEntry {
    pub id: String,
    pub version: u32,
    pub name: String,
    pub description: String,
    pub intent: Intent,
    pub behavior: Behavior,
    pub prompt_file: Option<String>,
    pub aliases: Vec<String>,
    pub allow_spoken_overrides: bool,
    pub vocabulary: Vec<String>,
    pub snippets: Vec<Snippet>,
    pub asr_route: String,
    pub authoring_route: Option<String>,
    pub local_only: bool,
    pub selected_text: SelectedText,
    pub selection_required: bool,
    pub clipboard: bool,
    pub max_context_characters: u32,
    pub transform_kinds: Vec<TransformKind>,
    pub language: Option<String>,
    pub style: Option<Style>,
    pub spoken_commands: bool,
    pub context_fields: Vec<ContextField>,
    pub processing_delivery: ProcessingDelivery,
    pub delivery: Delivery,
    pub automatic_delivery: bool,
}

/// A scoped rule of a profiles document. Routing itself is the frozen
/// oracle's job (`tests/mode_routing.py resolve`); this build only carries
/// the rules so a profiles document round-trips.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Rule {
    pub id: String,
    pub profile_id: String,
    pub priority: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub site: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub app_id: Option<String>,
}

/// A profiles document (`profiles.schema.json`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProfilesDocument {
    pub schema_version: u32,
    pub default_profile: String,
    pub profiles: Vec<ModeEntry>,
    pub rules: Vec<Rule>,
}

impl ProfilesDocument {
    pub fn profile(&self, id: &str) -> Option<&ModeEntry> {
        self.profiles.iter().find(|profile| profile.id == id)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderKind {
    /// The deterministic step alone: no model call.
    Builtin,
    S1,
    OpenaiCompatible,
    Anthropic,
    Gemini,
}

impl ProviderKind {
    pub fn as_str(self) -> &'static str {
        match self {
            ProviderKind::Builtin => "builtin",
            ProviderKind::S1 => "s1",
            ProviderKind::OpenaiCompatible => "openai_compatible",
            ProviderKind::Anthropic => "anthropic",
            ProviderKind::Gemini => "gemini",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Locality {
    Local,
    Remote,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Artifact {
    pub sha256: String,
    pub tokenizer: String,
    pub template: String,
    pub license: String,
    pub attribution: String,
}

/// A provider declaration (`provider.schema.json`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderDecl {
    pub schema_version: u32,
    pub id: String,
    pub route: String,
    pub kind: ProviderKind,
    pub locality: Locality,
    pub transform_kinds: Vec<TransformKind>,
    pub languages: Vec<String>,
    pub instructions: bool,
    pub context_fields: Vec<ContextField>,
    pub model: String,
    pub artifact: Option<Artifact>,
    pub max_input_chars: u32,
}

impl ProviderDecl {
    /// The identity a request and its result carry.
    pub fn reference(&self) -> ProviderRef {
        ProviderRef {
            id: self.id.clone(),
            kind: self.kind,
            locality: self.locality,
            route: self.route.clone(),
            model: self.model.clone(),
            artifact_sha256: self
                .artifact
                .as_ref()
                .map(|artifact| artifact.sha256.clone()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderRef {
    pub id: String,
    pub kind: ProviderKind,
    pub locality: Locality,
    pub route: String,
    pub model: String,
    pub artifact_sha256: Option<String>,
}

/// The context fields a request actually sent, and nothing else.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RequestContext {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub personal_context: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vocabulary: Option<Vec<String>>,
}

/// One transform request (`transform-request.schema.json`). `input` is
/// data: the text to transform, never instructions to follow.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TransformRequest {
    pub schema_version: u32,
    pub request_id: String,
    pub retry_of: Option<String>,
    pub draft_id: String,
    pub capture_id: String,
    pub base_revision: u64,
    pub source_attempt_ids: Vec<String>,
    pub mode_id: String,
    pub mode_version: u32,
    pub prompt_version: Option<String>,
    pub kinds: Vec<TransformKind>,
    pub language: Option<String>,
    pub input: String,
    pub instruction: Option<String>,
    pub style: Option<Style>,
    pub context: RequestContext,
    pub provider: ProviderRef,
    pub local_only: bool,
    pub deadline_ms: u64,
    pub max_output_chars: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResultStatus {
    Completed,
    Failed,
    Cancelled,
}

/// The typed failure vocabulary (`transform-result.schema.json`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FailureReason {
    Timeout,
    Cancelled,
    TransportError,
    RedirectBlocked,
    RateLimited,
    HttpError,
    UnknownModel,
    MalformedResponse,
    EmptyOutput,
    TruncatedOutput,
    UnsupportedLanguage,
    UnsupportedKind,
    RemoteForbidden,
    ProviderUnavailable,
    InvalidInput,
}

impl FailureReason {
    pub fn as_str(self) -> &'static str {
        match self {
            FailureReason::Timeout => "timeout",
            FailureReason::Cancelled => "cancelled",
            FailureReason::TransportError => "transport_error",
            FailureReason::RedirectBlocked => "redirect_blocked",
            FailureReason::RateLimited => "rate_limited",
            FailureReason::HttpError => "http_error",
            FailureReason::UnknownModel => "unknown_model",
            FailureReason::MalformedResponse => "malformed_response",
            FailureReason::EmptyOutput => "empty_output",
            FailureReason::TruncatedOutput => "truncated_output",
            FailureReason::UnsupportedLanguage => "unsupported_language",
            FailureReason::UnsupportedKind => "unsupported_kind",
            FailureReason::RemoteForbidden => "remote_forbidden",
            FailureReason::ProviderUnavailable => "provider_unavailable",
            FailureReason::InvalidInput => "invalid_input",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Failure {
    pub reason: FailureReason,
    pub retryable: bool,
    pub detail: String,
}

impl Failure {
    pub fn new(reason: FailureReason, retryable: bool, detail: impl Into<String>) -> Failure {
        Failure {
            reason,
            retryable,
            detail: detail.into(),
        }
    }
}

/// Per-job latency; no text ever goes in here.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Timing {
    pub queued_ms: f64,
    pub processing_ms: f64,
    pub stop_to_result_ms: Option<f64>,
}

/// One transform result (`transform-result.schema.json`). It never
/// changes a draft by itself; the draft judges it as a proposal.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TransformResult {
    pub schema_version: u32,
    pub request_id: String,
    pub base_revision: u64,
    pub status: ResultStatus,
    pub text: Option<String>,
    pub failure: Option<Failure>,
    pub provider: ProviderRef,
    pub timing: Timing,
}

// --------------------------------------------------------------------------
// Cross-field rules (ports of tests/mode_routing.py)
// --------------------------------------------------------------------------

/// `validate_processing`: the processing rules a schema cannot express.
pub fn validate_processing(doc: &ProfilesDocument) -> Result<(), String> {
    for profile in &doc.profiles {
        let kinds = &profile.transform_kinds;
        if has_duplicates(kinds) || has_duplicates(&profile.context_fields) {
            return Err(format!(
                "{}: duplicate transform kind or context field",
                profile.id
            ));
        }
        if !kinds.is_empty() && profile.authoring_route.is_none() {
            return Err(format!(
                "{}: processing needs an authoring route",
                profile.id
            ));
        }
        if profile.behavior == Behavior::Verbatim && (!kinds.is_empty() || profile.spoken_commands)
        {
            return Err(format!(
                "{}: verbatim returns raw text untouched",
                profile.id
            ));
        }
        if profile.style.is_some()
            && !kinds
                .iter()
                .any(|kind| matches!(kind, TransformKind::Clean | TransformKind::Format))
        {
            return Err(format!(
                "{}: style applies to clean/format only",
                profile.id
            ));
        }
        if profile.delivery == Delivery::InsertEnter {
            if profile.id == doc.default_profile {
                return Err("insert_enter cannot be the default delivery".to_string());
            }
            if profile.selected_text == SelectedText::EditTarget {
                return Err("insert_enter cannot replace a selection".to_string());
            }
        }
    }
    Ok(())
}

/// `validate_provider`.
pub fn validate_provider(provider: &ProviderDecl) -> Result<(), String> {
    let remote_route = provider.route.starts_with("remote-");
    if remote_route != (provider.locality == Locality::Remote) {
        return Err(format!("{}: locality must match the route", provider.id));
    }
    if provider
        .transform_kinds
        .iter()
        .any(|kind| kind.needs_instructions())
        && !provider.instructions
    {
        return Err(format!(
            "{}: rewrite/translate need instructions",
            provider.id
        ));
    }
    if (provider.kind == ProviderKind::Builtin) != provider.transform_kinds.is_empty() {
        return Err(format!(
            "{}: only the builtin provider runs no model kinds",
            provider.id
        ));
    }
    if provider.kind == ProviderKind::Builtin && provider.locality != Locality::Local {
        return Err(format!("{}: the builtin step is local", provider.id));
    }
    if provider.locality == Locality::Remote && provider.artifact.is_some() {
        return Err(format!(
            "{}: a remote provider has no local artifact",
            provider.id
        ));
    }
    Ok(())
}

fn has_duplicates<T: PartialEq>(items: &[T]) -> bool {
    items
        .iter()
        .enumerate()
        .any(|(i, item)| items[..i].contains(item))
}

/// Whether a provider accepting `accepted` languages may take text the
/// mode declares as `language` (`None` = undeclared).
pub fn language_ok(language: Option<&str>, accepted: &[String]) -> bool {
    if accepted.iter().any(|value| value == "*") {
        return true;
    }
    let Some(language) = language else {
        return false;
    };
    let primary = language.split('-').next().unwrap_or(language);
    accepted
        .iter()
        .any(|value| value == language || value == primary)
}

/// Why a mode's processing cannot run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouteBlock {
    RemoteForbidden,
    ProviderUnavailable,
    ProviderConflict,
    UnsupportedKind,
    UnsupportedLanguage,
}

impl RouteBlock {
    pub fn as_str(self) -> &'static str {
        match self {
            RouteBlock::RemoteForbidden => "remote_forbidden",
            RouteBlock::ProviderUnavailable => "provider_unavailable",
            RouteBlock::ProviderConflict => "provider_conflict",
            RouteBlock::UnsupportedKind => "unsupported_kind",
            RouteBlock::UnsupportedLanguage => "unsupported_language",
        }
    }

    /// The result failure a blocked route surfaces as.
    pub fn failure_reason(self) -> FailureReason {
        match self {
            RouteBlock::RemoteForbidden => FailureReason::RemoteForbidden,
            RouteBlock::ProviderUnavailable | RouteBlock::ProviderConflict => {
                FailureReason::ProviderUnavailable
            }
            RouteBlock::UnsupportedKind => FailureReason::UnsupportedKind,
            RouteBlock::UnsupportedLanguage => FailureReason::UnsupportedLanguage,
        }
    }
}

/// `processing_route`'s answer.
#[derive(Debug, Clone, PartialEq)]
pub enum ProcessingRoute<'a> {
    /// Transcribe only.
    None,
    /// Exactly this provider, with the context fields that will be sent.
    Ready {
        provider: &'a ProviderDecl,
        context_fields: Vec<ContextField>,
    },
    /// Nothing runs and the raw text stands; there is no fallback.
    Blocked(RouteBlock),
}

/// `processing_route`: the one provider for this mode's processing, or why
/// there is none. Never falls back to another provider, and a `local_only`
/// mode never reaches a remote one.
pub fn processing_route<'a>(
    profile: &ModeEntry,
    providers: &'a [ProviderDecl],
) -> ProcessingRoute<'a> {
    if profile.transform_kinds.is_empty() {
        return ProcessingRoute::None;
    }
    let route = profile.authoring_route.as_deref();
    let candidates: Vec<&ProviderDecl> = providers
        .iter()
        .filter(|provider| Some(provider.route.as_str()) == route)
        .collect();
    if profile.local_only
        && (route.is_some_and(|route| route.starts_with("remote-"))
            || candidates
                .iter()
                .any(|provider| provider.locality == Locality::Remote))
    {
        return ProcessingRoute::Blocked(RouteBlock::RemoteForbidden);
    }
    let provider = match candidates.as_slice() {
        [] => return ProcessingRoute::Blocked(RouteBlock::ProviderUnavailable),
        [provider] => *provider,
        _ => return ProcessingRoute::Blocked(RouteBlock::ProviderConflict),
    };
    if !profile
        .transform_kinds
        .iter()
        .all(|kind| provider.transform_kinds.contains(kind))
    {
        return ProcessingRoute::Blocked(RouteBlock::UnsupportedKind);
    }
    if !language_ok(profile.language.as_deref(), &provider.languages) {
        return ProcessingRoute::Blocked(RouteBlock::UnsupportedLanguage);
    }
    let context_fields = profile
        .context_fields
        .iter()
        .copied()
        .filter(|field| provider.context_fields.contains(field))
        .collect();
    ProcessingRoute::Ready {
        provider,
        context_fields,
    }
}
