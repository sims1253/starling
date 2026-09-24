//! The HTTP core every processing provider shares (#294).
//!
//! Same posture as `starling-dictation`'s transcription client: a
//! blocking facade over async reqwest on a private current-thread runtime,
//! redirects refused (a 3xx is `redirect_blocked`, never followed), bodies
//! read under a hard size cap, and every request raced against the job's
//! [`CancelToken`] and its deadline, so a cancel or an expired deadline
//! closes the connection instead of waiting it out.
//!
//! Failures come back as the contract's typed [`Failure`]. Details carry
//! status codes and provider error messages, never request text, and are
//! cut to [`MAX_DETAIL_CHARS`].

use std::time::Duration;

use reqwest::redirect::Policy;
use reqwest::{Client, Url};
use serde_json::Value;
use starling_dictation::client::CancelToken;

use crate::contract::{Failure, FailureReason, Locality};

/// Response bodies are refused past this many bytes. A processed
/// transcript is text; 4 MiB is far above any real answer while still
/// bounding what a broken endpoint can make the client buffer.
pub const DEFAULT_MAX_RESPONSE_BYTES: usize = 4 * 1024 * 1024;

const MAX_DETAIL_CHARS: usize = 300;

pub(crate) fn failure(
    reason: FailureReason,
    retryable: bool,
    detail: impl Into<String>,
) -> Failure {
    let detail: String = detail.into();
    Failure {
        reason,
        retryable,
        detail: detail.chars().take(MAX_DETAIL_CHARS).collect(),
    }
}

/// Whether `url`'s host is this machine.
pub fn is_loopback(url: &Url) -> bool {
    match url.host() {
        Some(url::Host::Domain(domain)) => domain.eq_ignore_ascii_case("localhost"),
        Some(url::Host::Ipv4(ip)) => ip.is_loopback(),
        Some(url::Host::Ipv6(ip)) => ip.is_loopback(),
        None => false,
    }
}

/// Validates a provider endpoint against the locality it declares, so
/// the declaration is true: a `local` provider must listen on this
/// machine, and a `remote` one must use https unless it is on loopback
/// (an API key never crosses the network in the clear). Credentials in
/// the URL are refused, like the transcription client does.
pub fn validate_endpoint(value: &str, locality: Locality) -> Result<Url, String> {
    let url = Url::parse(value.trim()).map_err(|_| "Invalid endpoint URL.".to_string())?;
    if url.scheme() != "http" && url.scheme() != "https" {
        return Err("The endpoint must use http or https.".to_string());
    }
    if !url.username().is_empty() || url.password().is_some() {
        return Err("Put credentials in the key setting, not the endpoint URL.".to_string());
    }
    if url.host().is_none() {
        return Err("The endpoint has no host.".to_string());
    }
    match locality {
        Locality::Local if !is_loopback(&url) => Err(
            "A local provider must listen on this machine (localhost or 127.0.0.1).".to_string(),
        ),
        Locality::Remote if url.scheme() != "https" && !is_loopback(&url) => {
            Err("A remote provider must use https.".to_string())
        }
        _ => Ok(url),
    }
}

/// Joins `path` onto the endpoint's path (`https://h/v1` + `chat/completions`).
pub(crate) fn join(base: &Url, path: &str) -> String {
    let base = base.as_str().trim_end_matches('/');
    format!("{base}/{}", path.trim_start_matches('/'))
}

/// What a streaming reader does after one line.
pub(crate) enum LineControl {
    Continue,
    /// The stream said it is done; stop reading.
    Done,
}

pub(crate) struct Http {
    client: Client,
    runtime: tokio::runtime::Runtime,
    max_response_bytes: usize,
}

impl Http {
    pub(crate) fn new(max_response_bytes: usize) -> Result<Http, String> {
        let client = Client::builder()
            .redirect(Policy::none())
            .build()
            .map_err(|error| error.to_string())?;
        // One worker thread, not a current-thread runtime: hyper's
        // connection task must keep running after `block_on` returns, so
        // that dropping a cancelled or timed-out stream actually closes
        // the socket instead of leaving it open until the next request.
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .enable_all()
            .build()
            .map_err(|error| error.to_string())?;
        Ok(Http {
            client,
            runtime,
            max_response_bytes: max_response_bytes.max(1),
        })
    }

    /// POSTs `body` as JSON and returns the response body. With
    /// `on_line`, the body is handed over line by line as it arrives
    /// (server-sent events) and the returned string is empty; the reader
    /// may stop early with [`LineControl::Done`] or abort with a failure.
    pub(crate) fn post_json(
        &self,
        url: &str,
        headers: &[(&str, String)],
        body: &Value,
        deadline: Duration,
        cancel: &CancelToken,
        mut on_line: Option<&mut dyn FnMut(&str) -> Result<LineControl, Failure>>,
    ) -> Result<String, Failure> {
        if cancel.is_cancelled() {
            return Err(failure(FailureReason::Cancelled, false, ""));
        }
        let mut request = self.client.post(url).json(body);
        for (name, value) in headers {
            request = request.header(*name, value);
        }
        let limit = self.max_response_bytes;
        let perform = async {
            let mut response = request.send().await.map_err(transport_failure)?;
            let status = response.status().as_u16();
            if (300..400).contains(&status) {
                return Err(failure(
                    FailureReason::RedirectBlocked,
                    false,
                    format!("the endpoint answered {status}; set the final URL explicitly"),
                ));
            }
            if !(200..300).contains(&status) {
                let retry_after = response
                    .headers()
                    .get("retry-after")
                    .and_then(|value| value.to_str().ok())
                    .map(str::to_string);
                let body = read_capped(&mut response, limit).await.unwrap_or_default();
                return Err(status_failure(status, retry_after.as_deref(), &body));
            }
            match on_line.as_mut() {
                None => read_capped(&mut response, limit).await,
                Some(on_line) => {
                    let mut pending: Vec<u8> = Vec::new();
                    let mut total = 0usize;
                    while let Some(chunk) = response.chunk().await.map_err(body_failure)? {
                        total += chunk.len();
                        if total > limit {
                            return Err(too_large(limit));
                        }
                        pending.extend_from_slice(&chunk);
                        while let Some(newline) = pending.iter().position(|byte| *byte == b'\n') {
                            let line: Vec<u8> = pending.drain(..=newline).collect();
                            let line = String::from_utf8_lossy(&line);
                            if let LineControl::Done = on_line(line.trim_end_matches(['\n', '\r']))?
                            {
                                return Ok(String::new());
                            }
                        }
                    }
                    if !pending.is_empty() {
                        let line = String::from_utf8_lossy(&pending).into_owned();
                        on_line(line.trim_end_matches('\r'))?;
                    }
                    Ok(String::new())
                }
            }
        };
        self.runtime.block_on(async {
            tokio::select! {
                result = tokio::time::timeout(deadline, perform) => match result {
                    Ok(result) => result,
                    Err(_) => Err(failure(
                        FailureReason::Timeout,
                        true,
                        format!("no complete answer within {} ms", deadline.as_millis()),
                    )),
                },
                _ = cancel.cancelled() => Err(failure(FailureReason::Cancelled, false, "")),
            }
        })
    }

    /// Fire-and-forget DELETE with a short timeout (server-side cancel);
    /// its outcome does not matter to the caller.
    pub(crate) fn delete_best_effort(&self, url: &str, timeout: Duration) {
        let request = self.client.delete(url).timeout(timeout);
        let _ = self.runtime.block_on(async { request.send().await });
    }
}

fn too_large(limit: usize) -> Failure {
    failure(
        FailureReason::TruncatedOutput,
        false,
        format!("the response exceeded the {limit} byte limit"),
    )
}

async fn read_capped(response: &mut reqwest::Response, limit: usize) -> Result<String, Failure> {
    if response
        .content_length()
        .is_some_and(|length| length > limit as u64)
    {
        return Err(too_large(limit));
    }
    let mut bytes = Vec::new();
    while let Some(chunk) = response.chunk().await.map_err(body_failure)? {
        if bytes.len() + chunk.len() > limit {
            return Err(too_large(limit));
        }
        bytes.extend_from_slice(&chunk);
    }
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

/// A failure while reading an answer that had already started: the
/// answer is cut, whatever the transport said.
fn body_failure(error: reqwest::Error) -> Failure {
    if error.is_timeout() {
        return failure(FailureReason::Timeout, true, "the request timed out");
    }
    failure(
        FailureReason::TruncatedOutput,
        true,
        format!(
            "the connection closed before the answer was complete: {}",
            error_chain(&error)
        ),
    )
}

fn transport_failure(error: reqwest::Error) -> Failure {
    if error.is_timeout() {
        return failure(FailureReason::Timeout, true, "the request timed out");
    }
    if error.is_connect() {
        return failure(
            FailureReason::ProviderUnavailable,
            true,
            format!("could not connect: {}", error_chain(&error)),
        );
    }
    failure(FailureReason::TransportError, true, error_chain(&error))
}

fn error_chain(error: &(dyn std::error::Error + 'static)) -> String {
    let mut message = error.to_string();
    let mut source = error.source();
    while let Some(cause) = source {
        message.push_str(": ");
        message.push_str(&cause.to_string());
        source = cause.source();
    }
    message
}

/// The provider's own error message, when its body carries one
/// (`{error: {message}}`, `{error: "..."}`, `{message}`, `{detail}`).
pub(crate) fn error_message(body: &str) -> Option<String> {
    let value: Value = serde_json::from_str(body).ok()?;
    let value = match &value {
        Value::Array(items) => items.first()?.clone(),
        _ => value,
    };
    match value.get("error") {
        Some(Value::String(message)) => return Some(message.clone()),
        Some(Value::Object(error)) => {
            if let Some(Value::String(message)) = error.get("message") {
                return Some(message.clone());
            }
        }
        _ => {}
    }
    ["message", "detail"]
        .iter()
        .find_map(|key| value.get(key).and_then(Value::as_str).map(str::to_string))
}

fn error_code(body: &str) -> Option<String> {
    let value: Value = serde_json::from_str(body).ok()?;
    let error = value.get("error")?;
    ["code", "type", "status"]
        .iter()
        .find_map(|key| error.get(key).and_then(Value::as_str).map(str::to_string))
}

/// Maps a non-2xx status to the typed failure vocabulary.
pub(crate) fn status_failure(status: u16, retry_after: Option<&str>, body: &str) -> Failure {
    let message = error_message(body).unwrap_or_default();
    let detail = if message.is_empty() {
        format!("HTTP {status}")
    } else {
        format!("HTTP {status}: {message}")
    };
    let code = error_code(body).unwrap_or_default();
    if code == "model_not_found" || (status == 404 && message.to_lowercase().contains("model")) {
        return failure(FailureReason::UnknownModel, false, detail);
    }
    match status {
        429 => {
            let detail = match retry_after {
                Some(after) => format!("{detail} (retry after {after})"),
                None => detail,
            };
            failure(FailureReason::RateLimited, true, detail)
        }
        408 => failure(FailureReason::Timeout, true, detail),
        500..=599 => failure(FailureReason::HttpError, true, detail),
        _ => failure(FailureReason::HttpError, false, detail),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn locality_is_enforced_on_the_endpoint() {
        assert!(validate_endpoint("http://127.0.0.1:8182", Locality::Local).is_ok());
        assert!(validate_endpoint("http://localhost:8182/", Locality::Local).is_ok());
        assert!(validate_endpoint("http://[::1]:8182", Locality::Local).is_ok());
        assert!(validate_endpoint("http://192.168.1.4:8182", Locality::Local).is_err());
        assert!(validate_endpoint("https://api.example.com/v1", Locality::Local).is_err());
        assert!(validate_endpoint("https://api.example.com/v1", Locality::Remote).is_ok());
        assert!(validate_endpoint("http://api.example.com/v1", Locality::Remote).is_err());
        assert!(validate_endpoint("http://127.0.0.1:8080/v1", Locality::Remote).is_ok());
        assert!(validate_endpoint("https://user:pw@api.example.com", Locality::Remote).is_err());
        assert!(validate_endpoint("ftp://api.example.com", Locality::Remote).is_err());
    }

    #[test]
    fn statuses_map_to_typed_failures() {
        let rate = status_failure(429, Some("7"), r#"{"error":{"message":"slow down"}}"#);
        assert_eq!(rate.reason, FailureReason::RateLimited);
        assert!(rate.retryable);
        assert!(rate.detail.contains("retry after 7"));
        let model = status_failure(
            404,
            None,
            r#"{"error":{"message":"The model `nope` does not exist","code":"model_not_found"}}"#,
        );
        assert_eq!(model.reason, FailureReason::UnknownModel);
        assert_eq!(
            status_failure(401, None, "").reason,
            FailureReason::HttpError
        );
        assert!(!status_failure(401, None, "").retryable);
        assert!(status_failure(502, None, "").retryable);
    }

    #[test]
    fn join_keeps_the_base_path() {
        let base = Url::parse("https://h.example/v1/").unwrap();
        assert_eq!(
            join(&base, "chat/completions"),
            "https://h.example/v1/chat/completions"
        );
    }
}
