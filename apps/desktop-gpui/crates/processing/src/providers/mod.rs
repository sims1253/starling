//! Processing providers (#294): one contract for local and remote models.
//!
//! - [`s1::S1Provider`]: S1-mini on a local starling-serve (`POST
//!   /normalize`), English clean/format only, no instructions.
//! - [`openai::OpenAiProvider`]: any OpenAI-compatible chat endpoint
//!   (OpenAI, OpenRouter, llama-server, vLLM, Groq, ...).
//! - [`anthropic::AnthropicProvider`], [`gemini::GeminiProvider`].
//!
//! Every provider validates its endpoint against the locality it declares
//! ([`crate::http::validate_endpoint`]), refuses redirects, honors the
//! job's cancel token and deadline, and reports typed failures. Chat
//! providers treat an empty answer as `empty_output` and a length stop as
//! `truncated_output`; S1-mini's empty answer on filler-only input is a
//! valid result.

pub mod anthropic;
pub mod gemini;
pub mod openai;
pub mod s1;

use std::time::Duration;

use starling_dictation::client::CancelToken;

use crate::contract::{Failure, FailureReason, ProviderDecl, TransformRequest};
use crate::http::{failure, Http};

/// The model step of the pipeline.
pub trait Provider: Send + Sync {
    fn declaration(&self) -> &ProviderDecl;

    /// Runs the model on `request.input`. Streams text deltas through
    /// `on_delta` when the provider can; returns the complete output.
    fn run(
        &self,
        request: &TransformRequest,
        on_delta: &mut dyn FnMut(&str),
        cancel: &CancelToken,
    ) -> Result<String, Failure>;
}

/// Connection settings every chat provider shares.
#[derive(Clone, Debug)]
pub struct ChatConfig {
    /// Base URL (`https://api.openai.com/v1`, `http://127.0.0.1:8080/v1`).
    pub endpoint: String,
    /// The API key, read from wherever the embedder keeps it (the desktop
    /// app reads a named environment variable); never logged.
    pub api_key: Option<String>,
    /// Stream the answer (server-sent events) instead of one response.
    pub stream: bool,
    /// OpenAI `reasoning_effort` for reasoning models ("minimal", "low").
    /// Left unset, nothing is sent: plain chat models reject the field.
    pub reasoning_effort: Option<String>,
    /// Ask for a non-thinking answer where the API has a switch
    /// (llama-server/vLLM `chat_template_kwargs.enable_thinking=false`,
    /// Gemini `thinkingBudget: 0`).
    pub disable_thinking: bool,
    /// Response size cap in bytes.
    pub max_response_bytes: usize,
}

impl ChatConfig {
    pub fn new(endpoint: impl Into<String>) -> ChatConfig {
        ChatConfig {
            endpoint: endpoint.into(),
            api_key: None,
            stream: true,
            reasoning_effort: None,
            disable_thinking: false,
            max_response_bytes: crate::http::DEFAULT_MAX_RESPONSE_BYTES,
        }
    }
}

/// The output token budget for a request: generous against its character
/// cap (one token is at least ~2 characters in every script these models
/// write), so a length stop means the answer really was too long.
pub(crate) fn max_tokens(request: &TransformRequest) -> u64 {
    (request.max_output_chars / 2).clamp(64, 32_768)
}

pub(crate) fn deadline(request: &TransformRequest) -> Duration {
    Duration::from_millis(request.deadline_ms.max(1))
}

/// Streaming output bookkeeping shared by the chat providers: collects
/// deltas, forwards them, and fails as soon as the output passes the
/// request's cap (the rest is never read).
pub(crate) struct Collected<'a> {
    pub text: String,
    chars: u64,
    cap: u64,
    on_delta: &'a mut dyn FnMut(&str),
}

impl<'a> Collected<'a> {
    pub(crate) fn new(cap: u64, on_delta: &'a mut dyn FnMut(&str)) -> Collected<'a> {
        Collected {
            text: String::new(),
            chars: 0,
            cap,
            on_delta,
        }
    }

    pub(crate) fn push(&mut self, delta: &str) -> Result<(), Failure> {
        if delta.is_empty() {
            return Ok(());
        }
        self.chars += delta.chars().count() as u64;
        if self.chars > self.cap {
            return Err(over_cap(self.cap));
        }
        self.text.push_str(delta);
        (self.on_delta)(delta);
        Ok(())
    }
}

pub(crate) fn over_cap(cap: u64) -> Failure {
    failure(
        FailureReason::TruncatedOutput,
        false,
        format!("the output passed the {cap} character cap"),
    )
}

pub(crate) fn malformed(detail: impl Into<String>) -> Failure {
    failure(FailureReason::MalformedResponse, false, detail)
}

/// A chat answer must say something; whitespace only is `empty_output`.
pub(crate) fn non_empty(text: String) -> Result<String, Failure> {
    if text.trim().is_empty() {
        Err(failure(
            FailureReason::EmptyOutput,
            false,
            "the model returned no text",
        ))
    } else {
        Ok(text)
    }
}

pub(crate) fn build_http(max_response_bytes: usize) -> Result<Http, String> {
    Http::new(max_response_bytes)
}

/// Parses one server-sent-events `data:` line; `None` for comments, event
/// names, blank separators and other fields.
pub(crate) fn sse_data(line: &str) -> Option<&str> {
    let data = line.strip_prefix("data:")?;
    Some(data.strip_prefix(' ').unwrap_or(data))
}
