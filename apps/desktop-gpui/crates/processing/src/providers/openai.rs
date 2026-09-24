//! OpenAI-compatible chat completions (`POST {endpoint}/chat/completions`):
//! OpenAI, OpenRouter, llama-server, vLLM, Groq and anything else that
//! speaks the same shape.
//!
//! Streaming reads `data:` lines until `[DONE]` or a `finish_reason`. A
//! `length` stop, or a stream that ends without either, is
//! `truncated_output`: a cut answer is never passed off as complete.
//! Separate reasoning channels (`reasoning_content`) are ignored, and the
//! answer's own text is never edited.

use serde_json::{json, Value};
use starling_dictation::client::CancelToken;

use super::{
    deadline, malformed, max_tokens, non_empty, sse_data, ChatConfig, Collected, Provider,
};
use crate::contract::{Failure, FailureReason, ProviderDecl, TransformRequest};
use crate::http::{failure, join, validate_endpoint, Http, LineControl};
use crate::prompt;

pub struct OpenAiProvider {
    decl: ProviderDecl,
    url: String,
    config: ChatConfig,
    http: Http,
}

impl OpenAiProvider {
    /// Fails when the endpoint contradicts the declared locality.
    pub fn new(decl: ProviderDecl, config: ChatConfig) -> Result<OpenAiProvider, String> {
        let endpoint = validate_endpoint(&config.endpoint, decl.locality)?;
        Ok(OpenAiProvider {
            url: join(&endpoint, "chat/completions"),
            http: super::build_http(config.max_response_bytes)?,
            decl,
            config,
        })
    }

    fn body(&self, request: &TransformRequest) -> Value {
        let prompt = prompt::render(request);
        let mut body = json!({
            "model": self.decl.model,
            "messages": [
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user},
            ],
            "temperature": 0,
            "max_tokens": max_tokens(request),
            "stream": self.config.stream,
        });
        if let Some(effort) = &self.config.reasoning_effort {
            body["reasoning_effort"] = json!(effort);
        }
        if self.config.disable_thinking {
            body["chat_template_kwargs"] = json!({"enable_thinking": false});
        }
        body
    }

    fn headers(&self) -> Vec<(&'static str, String)> {
        match &self.config.api_key {
            Some(key) => vec![("authorization", format!("Bearer {key}"))],
            None => Vec::new(),
        }
    }
}

/// Stream state for one answer.
#[derive(Default)]
pub(crate) struct StreamState {
    pub finished: bool,
}

/// Handles one SSE line of a chat-completions stream.
pub(crate) fn stream_line(
    line: &str,
    state: &mut StreamState,
    out: &mut Collected,
) -> Result<LineControl, Failure> {
    let Some(data) = sse_data(line) else {
        return Ok(LineControl::Continue);
    };
    let data = data.trim();
    if data == "[DONE]" {
        state.finished = true;
        return Ok(LineControl::Done);
    }
    if data.is_empty() {
        return Ok(LineControl::Continue);
    }
    let chunk: Value =
        serde_json::from_str(data).map_err(|_| malformed("a stream chunk is not JSON"))?;
    if let Some(error) = chunk.get("error") {
        let message = error
            .get("message")
            .and_then(Value::as_str)
            .unwrap_or("stream error");
        return Err(failure(
            FailureReason::HttpError,
            false,
            message.to_string(),
        ));
    }
    let Some(choice) = chunk.get("choices").and_then(|choices| choices.get(0)) else {
        // Usage-only or keep-alive chunks carry no choice.
        return Ok(LineControl::Continue);
    };
    if let Some(content) = choice.pointer("/delta/content") {
        match content {
            Value::String(text) => out.push(text)?,
            Value::Null => {}
            _ => return Err(malformed("delta.content is not a string")),
        }
    }
    finish(choice.get("finish_reason"), state)
}

fn finish(reason: Option<&Value>, state: &mut StreamState) -> Result<LineControl, Failure> {
    match reason.and_then(Value::as_str) {
        None => Ok(LineControl::Continue),
        Some("length") => Err(failure(
            FailureReason::TruncatedOutput,
            false,
            "the model stopped at its token limit",
        )),
        Some("content_filter") => Err(failure(
            FailureReason::HttpError,
            false,
            "the provider's content filter stopped the answer",
        )),
        Some(_) => {
            state.finished = true;
            Ok(LineControl::Continue)
        }
    }
}

/// Parses a non-streaming chat-completions body.
pub(crate) fn parse_body(body: &str, out: &mut Collected) -> Result<(), Failure> {
    let value: Value =
        serde_json::from_str(body).map_err(|_| malformed("the response is not JSON"))?;
    let choice = value
        .get("choices")
        .and_then(|choices| choices.get(0))
        .ok_or_else(|| malformed("the response has no choices"))?;
    match choice.pointer("/message/content") {
        Some(Value::String(text)) => out.push(text)?,
        Some(Value::Null) | None => {}
        Some(_) => return Err(malformed("message.content is not a string")),
    }
    let mut state = StreamState::default();
    finish(choice.get("finish_reason"), &mut state)?;
    Ok(())
}

impl Provider for OpenAiProvider {
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
            let mut state = StreamState::default();
            let mut on_line = |line: &str| stream_line(line, &mut state, &mut out);
            self.http.post_json(
                &self.url,
                &headers,
                &body,
                deadline(request),
                cancel,
                Some(&mut on_line),
            )?;
            if !state.finished {
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
