//! Anthropic Messages API (`POST {endpoint}/v1/messages`).
//!
//! Extended thinking is never enabled, so the answer is the text blocks
//! alone. Streaming reads `content_block_delta` text deltas until
//! `message_stop`; a `max_tokens` stop is `truncated_output`, and a stream
//! that ends without `message_stop` is too.

use serde_json::{json, Value};
use starling_dictation::client::CancelToken;

use super::{deadline, malformed, max_tokens, non_empty, ChatConfig, Collected, Provider};
use crate::contract::{Failure, FailureReason, ProviderDecl, TransformRequest};
use crate::http::{failure, join, validate_endpoint, Http, LineControl};
use crate::prompt;

const API_VERSION: &str = "2023-06-01";

pub struct AnthropicProvider {
    decl: ProviderDecl,
    url: String,
    config: ChatConfig,
    http: Http,
}

impl AnthropicProvider {
    pub fn new(decl: ProviderDecl, config: ChatConfig) -> Result<AnthropicProvider, String> {
        let endpoint = validate_endpoint(&config.endpoint, decl.locality)?;
        Ok(AnthropicProvider {
            url: join(&endpoint, "v1/messages"),
            http: super::build_http(config.max_response_bytes)?,
            decl,
            config,
        })
    }

    fn body(&self, request: &TransformRequest) -> Value {
        let prompt = prompt::render(request);
        json!({
            "model": self.decl.model,
            "system": prompt.system,
            "messages": [{"role": "user", "content": prompt.user}],
            "max_tokens": max_tokens(request),
            // No `temperature`: models after Claude Opus 4.6 reject any
            // value but the default.
            "stream": self.config.stream,
        })
    }

    fn headers(&self) -> Vec<(&'static str, String)> {
        let mut headers = vec![("anthropic-version", API_VERSION.to_string())];
        if let Some(key) = &self.config.api_key {
            headers.push(("x-api-key", key.clone()));
        }
        headers
    }
}

fn stop(reason: Option<&str>) -> Result<(), Failure> {
    match reason {
        Some("max_tokens") => Err(failure(
            FailureReason::TruncatedOutput,
            false,
            "the model stopped at its token limit",
        )),
        Some("refusal") => Err(failure(
            FailureReason::HttpError,
            false,
            "the model refused",
        )),
        _ => Ok(()),
    }
}

pub(crate) fn stream_event(
    data: &str,
    finished: &mut bool,
    out: &mut Collected,
) -> Result<LineControl, Failure> {
    if data.trim().is_empty() {
        return Ok(LineControl::Continue);
    }
    let event: Value =
        serde_json::from_str(data).map_err(|_| malformed("a stream event is not JSON"))?;
    match event.get("type").and_then(Value::as_str) {
        Some("content_block_delta") => {
            if event.pointer("/delta/type").and_then(Value::as_str) == Some("text_delta") {
                let text = event
                    .pointer("/delta/text")
                    .and_then(Value::as_str)
                    .ok_or_else(|| malformed("text_delta without text"))?;
                out.push(text)?;
            }
            Ok(LineControl::Continue)
        }
        Some("message_delta") => {
            stop(event.pointer("/delta/stop_reason").and_then(Value::as_str))?;
            Ok(LineControl::Continue)
        }
        Some("message_stop") => {
            *finished = true;
            Ok(LineControl::Done)
        }
        Some("error") => {
            let message = event
                .pointer("/error/message")
                .and_then(Value::as_str)
                .unwrap_or("stream error");
            let overloaded =
                event.pointer("/error/type").and_then(Value::as_str) == Some("overloaded_error");
            Err(failure(
                FailureReason::HttpError,
                overloaded,
                message.to_string(),
            ))
        }
        _ => Ok(LineControl::Continue),
    }
}

pub(crate) fn parse_body(body: &str, out: &mut Collected) -> Result<(), Failure> {
    let value: Value =
        serde_json::from_str(body).map_err(|_| malformed("the response is not JSON"))?;
    let content = value
        .get("content")
        .and_then(Value::as_array)
        .ok_or_else(|| malformed("the response has no content"))?;
    for block in content {
        if block.get("type").and_then(Value::as_str) == Some("text") {
            let text = block
                .get("text")
                .and_then(Value::as_str)
                .ok_or_else(|| malformed("a text block without text"))?;
            out.push(text)?;
        }
    }
    stop(value.get("stop_reason").and_then(Value::as_str))
}

impl Provider for AnthropicProvider {
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
            let mut on_event = |data: &str| stream_event(data, &mut finished, &mut out);
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
            parse_body(&body, &mut out)?;
        }
        non_empty(out.text)
    }
}
