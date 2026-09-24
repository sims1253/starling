//! S1-mini on a local starling-serve (`POST /normalize`, see
//! `cpp/serve/main.cpp` and `docs/ggml-engine.md` "s1 engine notes").
//!
//! starling-serve keeps one model resident, so S1-mini runs in its own
//! instance, on loopback, next to the transcription server. The guardrails
//! S1-mini needs are enforced here as well as in the route choice: English
//! only, clean/format only, no free-form instruction, and the mode's style
//! mapped onto the control line. Its empty answer on filler-only input is
//! a valid result.
//!
//! S1-mini was trained on prompts of at most 1000 tokens. Longer
//! transcripts are split at sentence ends into chunks of at most
//! [`CHUNK_CHARS`] and normalized one after another; the outputs are
//! joined with a space. A cancel aborts the connection and also asks the
//! server to stop the chunk it is decoding (`DELETE
//! /v1/audio/transcriptions/<request id>`, which cancels any registered
//! request id).

use std::time::{Duration, Instant};

use serde_json::{json, Value};
use starling_dictation::client::CancelToken;

use super::{deadline, malformed, over_cap, Provider};
use crate::contract::{
    self, Failure, FailureReason, Formality, ProviderDecl, Structure, Style, StyleContext,
    TransformKind, TransformRequest,
};
use crate::http::{failure, join, validate_endpoint, Http};

/// Chunk size for long transcripts, in characters. At ~4 characters per
/// token this keeps each prompt near half of S1-mini's 1000-token limit,
/// so dense text (digits tokenize one per token) still fits.
pub const CHUNK_CHARS: usize = 2000;

/// The style S1-mini applies when a mode declares none (its trained
/// defaults).
const DEFAULT_STYLE: Style = Style {
    formality: Formality::SemiFormal,
    structure: Structure::Prose,
    context: StyleContext::General,
};

pub struct S1Provider {
    decl: ProviderDecl,
    endpoint: url::Url,
    http: Http,
}

impl S1Provider {
    /// `endpoint` is the S1 starling-serve base URL; it must be loopback
    /// because the declaration says local.
    pub fn new(decl: ProviderDecl, endpoint: &str) -> Result<S1Provider, String> {
        let endpoint = validate_endpoint(endpoint, decl.locality)?;
        Ok(S1Provider {
            http: super::build_http(crate::http::DEFAULT_MAX_RESPONSE_BYTES)?,
            decl,
            endpoint,
        })
    }

    fn guard(&self, request: &TransformRequest) -> Result<(), Failure> {
        if !contract::language_ok(request.language.as_deref(), &self.decl.languages) {
            return Err(failure(
                FailureReason::UnsupportedLanguage,
                false,
                "S1-mini only processes English; declare the mode's language as English",
            ));
        }
        if request.instruction.is_some()
            || request
                .kinds
                .iter()
                .any(|kind| !matches!(kind, TransformKind::Clean | TransformKind::Format))
        {
            return Err(failure(
                FailureReason::UnsupportedKind,
                false,
                "S1-mini only cleans and formats; it cannot follow instructions",
            ));
        }
        Ok(())
    }

    fn normalize(
        &self,
        chunk: &str,
        id: &str,
        style: &Style,
        deadline: Duration,
        cancel: &CancelToken,
    ) -> Result<String, Failure> {
        let (styling, structure, context) = style.s1_controls();
        let body = json!({
            "transcript": chunk,
            "styling": styling,
            "structure": structure,
            "context": context,
        });
        let url = join(&self.endpoint, "normalize");
        let headers = [("x-request-id", id.to_string())];
        let answer = match self
            .http
            .post_json(&url, &headers, &body, deadline, cancel, None)
        {
            Ok(answer) => answer,
            Err(error) => {
                if error.reason == FailureReason::Cancelled {
                    self.http.delete_best_effort(
                        &join(&self.endpoint, &format!("v1/audio/transcriptions/{id}")),
                        Duration::from_millis(500),
                    );
                }
                return Err(s1_failure(error));
            }
        };
        let value: Value =
            serde_json::from_str(&answer).map_err(|_| malformed("the response is not JSON"))?;
        value
            .get("text")
            .and_then(Value::as_str)
            .map(str::to_string)
            .ok_or_else(|| malformed("the response has no text"))
    }
}

/// Refines the generic status mapping with what starling-serve's
/// `/normalize` says (see its handler).
fn s1_failure(error: Failure) -> Failure {
    let detail = error.detail.clone();
    if detail.contains("model has no text path") {
        return failure(
            FailureReason::ProviderUnavailable,
            false,
            "the server at this endpoint runs an audio model; start a second starling-serve with the S1-mini GGUF",
        );
    }
    if detail.contains("server busy") || detail.contains("model not loaded") {
        return failure(FailureReason::ProviderUnavailable, true, detail);
    }
    if detail.contains("HTTP 499") {
        return failure(FailureReason::Cancelled, false, detail);
    }
    if detail.contains("HTTP 504") {
        return failure(FailureReason::Timeout, true, detail);
    }
    if detail.contains("HTTP 400") {
        // Prompt too long, unknown control value, malformed body: the
        // same request would fail the same way.
        return failure(FailureReason::InvalidInput, false, detail);
    }
    error
}

/// Splits `text` into chunks of at most `limit` characters, preferring
/// sentence ends, then whitespace, then a hard cut inside a single
/// oversized word. Every character lands in exactly one chunk, in order.
pub fn chunks(text: &str, limit: usize) -> Vec<&str> {
    let limit = limit.max(1);
    let mut out = Vec::new();
    let mut rest = text;
    while rest.chars().count() > limit {
        let window_end = rest
            .char_indices()
            .nth(limit)
            .map_or(rest.len(), |(index, _)| index);
        let window = &rest[..window_end];
        let sentence = window
            .char_indices()
            .filter(|&(index, c)| {
                matches!(c, '.' | '?' | '!' | '\n')
                    && window[index + c.len_utf8()..].starts_with(char::is_whitespace)
            })
            .map(|(index, c)| index + c.len_utf8())
            .last();
        let space = window.rfind(char::is_whitespace).filter(|&index| index > 0);
        let cut = sentence.or(space).unwrap_or(window_end);
        out.push(&rest[..cut]);
        rest = &rest[cut..];
    }
    if !rest.is_empty() {
        out.push(rest);
    }
    out
}

impl Provider for S1Provider {
    fn declaration(&self) -> &ProviderDecl {
        &self.decl
    }

    fn run(
        &self,
        request: &TransformRequest,
        on_delta: &mut dyn FnMut(&str),
        cancel: &CancelToken,
    ) -> Result<String, Failure> {
        self.guard(request)?;
        let style = request.style.unwrap_or(DEFAULT_STYLE);
        let started = Instant::now();
        let budget = deadline(request);
        let mut output = String::new();
        for (index, chunk) in chunks(&request.input, CHUNK_CHARS).into_iter().enumerate() {
            if chunk.trim().is_empty() {
                continue;
            }
            let remaining = budget
                .checked_sub(started.elapsed())
                .filter(|left| !left.is_zero())
                .ok_or_else(|| {
                    failure(
                        FailureReason::Timeout,
                        true,
                        "the deadline passed between chunks",
                    )
                })?;
            let id = format!("{}.{index}", request.request_id);
            let text = self.normalize(chunk.trim(), &id, &style, remaining, cancel)?;
            if text.is_empty() {
                continue;
            }
            let piece = if output.is_empty() {
                text
            } else {
                format!(" {text}")
            };
            if (output.chars().count() + piece.chars().count()) as u64 > request.max_output_chars {
                return Err(over_cap(request.max_output_chars));
            }
            on_delta(&piece);
            output.push_str(&piece);
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_prefer_sentence_ends_and_keep_every_character() {
        let text = "One two. Three four five. Six seven eight nine ten.";
        let parts = chunks(text, 20);
        assert_eq!(parts.concat(), text);
        assert!(parts.iter().all(|part| part.chars().count() <= 20));
        assert_eq!(parts[0], "One two.");
    }

    #[test]
    fn chunks_fall_back_to_spaces_and_hard_cuts() {
        let text = "aaaa bbbb cccc";
        assert_eq!(chunks(text, 9), vec!["aaaa", " bbbb", " cccc"]);
        let word = "é".repeat(7);
        let parts = chunks(&word, 3);
        assert_eq!(parts.concat(), word);
        assert!(parts.iter().all(|part| part.chars().count() <= 3));
    }

    #[test]
    fn short_text_is_one_chunk() {
        assert_eq!(chunks("hello", CHUNK_CHARS), vec!["hello"]);
        assert!(chunks("", CHUNK_CHARS).is_empty());
    }
}
