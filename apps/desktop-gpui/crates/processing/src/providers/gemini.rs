//! Gemini `generateContent` / `streamGenerateContent?alt=sse`
//! (`{endpoint}/v1beta/models/{model}:...`).
//!
//! Thought parts (`thought: true`) are not the answer and are skipped;
//! with `disable_thinking` the request asks for a zero thinking budget. A
//! `MAX_TOKENS` finish is `truncated_output`; a stream that ends without
//! any finish reason is too.

use serde_json::{json, Value};
use starling_dictation::client::CancelToken;

use super::{deadline, malformed, max_tokens, non_empty, ChatConfig, Collected, Provider};
use crate::contract::{Failure, FailureReason, ProviderDecl, TransformRequest};
use crate::http::{failure, join, validate_endpoint, Http, LineControl};
use crate::prompt;

pub struct GeminiProvider {
    decl: ProviderDecl,
    url: String,
    config: ChatConfig,
    http: Http,
}

impl GeminiProvider {
    pub fn new(decl: ProviderDecl, config: ChatConfig) -> Result<GeminiProvider, String> {
        let endpoint = validate_endpoint(&config.endpoint, decl.locality)?;
        // The model id goes into the URL path as is.
        if decl.model.is_empty()
            || !decl
                .model
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
        {
            return Err(format!(
                "\"{}\" is not a Gemini model id (letters, digits, '-', '_', '.')",
                decl.model
            ));
        }
        let method = if config.stream {
            "streamGenerateContent?alt=sse"
        } else {
            "generateContent"
        };
        Ok(GeminiProvider {
            url: join(&endpoint, &format!("v1beta/models/{}:{method}", decl.model)),
            http: super::build_http(config.max_response_bytes)?,
            decl,
            config,
        })
    }

    fn body(&self, request: &TransformRequest) -> Value {
        let prompt = prompt::render(request);
        let mut generation = json!({"temperature": 0, "maxOutputTokens": max_tokens(request)});
        if self.config.disable_thinking {
            generation["thinkingConfig"] = json!({"thinkingBudget": 0});
        }
        json!({
            "systemInstruction": {"parts": [{"text": prompt.system}]},
            "contents": [{"role": "user", "parts": [{"text": prompt.user}]}],
            "generationConfig": generation,
        })
    }

    fn headers(&self) -> Vec<(&'static str, String)> {
        match &self.config.api_key {
            Some(key) => vec![("x-goog-api-key", key.clone())],
            None => Vec::new(),
        }
    }
}

/// Folds one response object (a whole body or one stream chunk) into the
/// output; returns whether it carried a finish reason.
pub(crate) fn fold(value: &Value, out: &mut Collected) -> Result<bool, Failure> {
    if let Some(reason) = value
        .pointer("/promptFeedback/blockReason")
        .and_then(Value::as_str)
    {
        return Err(failure(
            FailureReason::HttpError,
            false,
            format!("blocked: {reason}"),
        ));
    }
    let Some(candidate) = value.get("candidates").and_then(|c| c.get(0)) else {
        return Ok(false);
    };
    if let Some(parts) = candidate
        .pointer("/content/parts")
        .and_then(Value::as_array)
    {
        for part in parts {
            if part.get("thought").and_then(Value::as_bool) == Some(true) {
                continue;
            }
            if let Some(text) = part.get("text") {
                out.push(
                    text.as_str()
                        .ok_or_else(|| malformed("a part's text is not a string"))?,
                )?;
            }
        }
    }
    match candidate.get("finishReason").and_then(Value::as_str) {
        None | Some("FINISH_REASON_UNSPECIFIED") => Ok(false),
        Some("STOP") => Ok(true),
        Some("MAX_TOKENS") => Err(failure(
            FailureReason::TruncatedOutput,
            false,
            "the model stopped at its token limit",
        )),
        Some(other) => Err(failure(
            FailureReason::HttpError,
            false,
            format!("finish reason {other}"),
        )),
    }
}

impl Provider for GeminiProvider {
    fn declaration(&self) -> &ProviderDecl {
        &self.decl
    }

    fn run(
        &self,
        request: &TransformRequest,
        on_delta: &mut dyn FnMut(&str),
        cancel: &CancelToken,
    ) -> Result<String, Failure> {
        let body = self.body(request);
        let headers = self.headers();
        let mut out = Collected::new(request.max_output_chars, on_delta);
        if self.config.stream {
            let mut finished = false;
            let mut on_event = |data: &str| -> Result<LineControl, Failure> {
                if data.trim().is_empty() {
                    return Ok(LineControl::Continue);
                }
                let value: Value = serde_json::from_str(data)
                    .map_err(|_| malformed("a stream chunk is not JSON"))?;
                if fold(&value, &mut out)? {
                    // Done: nothing after the finish reason becomes part
                    // of the answer.
                    finished = true;
                    return Ok(LineControl::Done);
                }
                Ok(LineControl::Continue)
            };
            self.http.post_json(
                &self.url,
                &headers,
                &body,
                deadline(request),
                cancel,
                Some(&mut on_event),
            )?;
            if !finished {
                return Err(failure(
                    FailureReason::TruncatedOutput,
                    true,
                    "the stream ended before the answer was complete",
                ));
            }
        } else {
            let body =
                self.http
                    .post_json(&self.url, &headers, &body, deadline(request), cancel, None)?;
            let value: Value =
                serde_json::from_str(&body).map_err(|_| malformed("the response is not JSON"))?;
            if !fold(&value, &mut out)? {
                return Err(malformed("the response has no finished candidate"));
            }
        }
        non_empty(out.text)
    }
}
