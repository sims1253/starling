// Ported from `apps/desktop/electron/ipc.ts` + `electron/main.ts` — the native
// Electron request path (`healthProgram` / `transcribeProgram`) — together with
// the response normalization of `packages/dictation/src/client.ts`.
// See `apps/desktop-gpui/PORT.md` ("Transcription client") for the contract.

use std::error::Error as StdError;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use reqwest::multipart;
use reqwest::redirect::Policy;
use reqwest::{Client, RequestBuilder, Url};
use serde_json::{Map, Value};
use tokio::runtime::Runtime;
use tokio::sync::Notify;

use crate::storage::{TranscriptionResult, TranscriptionSegment};

const DEFAULT_TIMEOUT_MS: u64 = 180_000;
const MIN_TIMEOUT_MS: u64 = 1;
const MAX_TIMEOUT_MS: u64 = 600_000;
const MIN_AUDIO_BYTES: usize = 44;
const MAX_AUDIO_BYTES: usize = 256 * 1024 * 1024;

/// Transcription backend wire protocol.
#[derive(Clone, Copy, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Protocol {
    Starling,
    OpenAi,
}

/// Health snapshot mirroring the desktop bridge's `ServerHealth`.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ServerHealth {
    pub status: String,
    pub phase: Option<String>,
    pub model: Option<String>,
    pub loaded: Option<bool>,
    pub busy: Option<bool>,
    pub queue_depth: Option<f64>,
}

#[derive(Debug, thiserror::Error)]
pub enum ClientError {
    #[error("{0}")]
    Input(String),
    #[error("{0}")]
    Transport(String),
    #[error("Request timed out after {0} ms.")]
    Timeout(u64),
    #[error("Server redirect blocked ({0}). Set the final endpoint explicitly.")]
    Redirect(u16),
    #[error("{message}")]
    Http { status: u16, message: String },
    #[error("The server returned invalid {0} JSON.")]
    Protocol(&'static str),
    /// The caller cancelled the request while it was in flight (issue
    /// #251): the connection was aborted, not timed out — distinct from
    /// [`ClientError::Timeout`], which is the server being slow.
    #[error("The request was cancelled before it completed.")]
    Cancelled,
}

/// A cooperative cancellation signal for in-flight requests (issue #251).
///
/// One side calls [`CancelToken::cancel`] — from any thread, once — and
/// every request carrying this token aborts: the flag covers the
/// before-send fast path, and the [`Notify`] permit wakes a request that
/// is already selecting on the token (`notify_one` stores its permit, so
/// a cancel that lands between the flag check and the select is still
/// observed). Clone shares the signal.
#[derive(Clone, Default)]
pub struct CancelToken {
    state: Arc<CancelState>,
}

#[derive(Default)]
struct CancelState {
    flag: AtomicBool,
    notify: Notify,
}

impl CancelToken {
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the signal and wakes every request waiting on this token.
    pub fn cancel(&self) {
        self.state.flag.store(true, Ordering::Release);
        self.state.notify.notify_one();
    }

    /// Whether [`CancelToken::cancel`] has been called.
    pub fn is_cancelled(&self) -> bool {
        self.state.flag.load(Ordering::Acquire)
    }

    /// Resolves once the token is cancelled. The `notify_one` permit is
    /// stored when no one is waiting, so this completes immediately for a
    /// token cancelled before the future is first polled.
    async fn notified(&self) {
        self.state.notify.notified().await
    }
}

/// Blocking HTTP client mirroring the Electron native request programs.
///
/// The façade stays blocking (callers are worker threads); inside, the
/// async reqwest client runs on a dedicated current-thread runtime — the
/// async future is what makes an in-flight request abortable at all
/// (issue #251): dropping it in [`tokio::select!`] closes the connection,
/// something the blocking API cannot do.
pub struct StarlingClient {
    base_url: String,
    protocol: Protocol,
    model: String,
    timeout_ms: u64,
    http: Client,
    runtime: Runtime,
}

struct ApiResponse {
    body: String,
    request_id: Option<String>,
}

impl StarlingClient {
    /// `new StarlingClient` for the desktop path: endpoint validated like
    /// `cleanEndpoint`, default timeout 180 s, redirects refused, model
    /// trimmed (`client.ts`) with the OpenAI fallback to `parakeet`
    /// (`main.ts`).
    pub fn new(endpoint: &str, protocol: Protocol, model: &str) -> Result<Self, ClientError> {
        Self::build(
            clean_endpoint(endpoint)?,
            protocol,
            model,
            DEFAULT_TIMEOUT_MS,
        )
    }

    /// Overrides the total per-request timeout (1 ms ..= 10 min), mirroring
    /// `requestTimeout` in `main.ts`.
    pub fn with_timeout_ms(mut self, timeout_ms: u64) -> Result<Self, ClientError> {
        self.timeout_ms = request_timeout(timeout_ms)?;
        self.http = build_http_client(self.timeout_ms)?;
        Ok(self)
    }

    fn build(
        base_url: String,
        protocol: Protocol,
        model: &str,
        timeout_ms: u64,
    ) -> Result<Self, ClientError> {
        Ok(Self {
            base_url,
            protocol,
            model: model.trim().to_string(),
            timeout_ms,
            http: build_http_client(timeout_ms)?,
            runtime: build_runtime()?,
        })
    }

    /// `healthProgram`: `GET {base}/health` (starling) or `GET {base}/v1/models`
    /// (openai, mapped to a ready health snapshot).
    pub fn health(&self) -> Result<ServerHealth, ClientError> {
        let route = match self.protocol {
            Protocol::OpenAi => "/v1/models",
            Protocol::Starling => "/health",
        };
        let response = self.execute(self.http.get(format!("{}{route}", self.base_url)), None)?;
        match self.protocol {
            Protocol::OpenAi => parse_models_health(&response.body),
            Protocol::Starling => parse_starling_health(&response.body),
        }
    }

    /// `transcribeProgram` without cancellation: runs to the timeout.
    pub fn transcribe(
        &self,
        wav: &[u8],
        request_id: &str,
    ) -> Result<TranscriptionResult, ClientError> {
        self.transcribe_with_cancel(wav, request_id, None)
    }

    /// `transcribeProgram` with an abort signal (issue #251): when the
    /// token fires, the in-flight HTTP request is aborted (the connection
    /// closed) and the call returns [`ClientError::Cancelled`] promptly —
    /// instead of holding the worker and the connection for the rest of
    /// the whole-recognition timeout.
    pub fn transcribe_with_cancel(
        &self,
        wav: &[u8],
        request_id: &str,
        cancel: Option<&CancelToken>,
    ) -> Result<TranscriptionResult, ClientError> {
        let sent_request_id = validate_request_id(request_id)?;
        if wav.len() < MIN_AUDIO_BYTES || wav.len() > MAX_AUDIO_BYTES {
            return Err(ClientError::Input(
                "Audio payload is empty or too large.".to_string(),
            ));
        }

        let response = match self.protocol {
            Protocol::Starling => {
                let url = format!("{}/transcribe", self.base_url);
                let request = self
                    .http
                    .post(&url)
                    .header("x-request-id", sent_request_id.as_str())
                    .header("content-type", "audio/wav")
                    .body(wav.to_vec());
                self.execute(request, cancel)?
            }
            Protocol::OpenAi => {
                let url = format!("{}/v1/audio/transcriptions", self.base_url);
                let model = if self.model.is_empty() {
                    "parakeet"
                } else {
                    self.model.as_str()
                };
                let file = multipart::Part::bytes(wav.to_vec())
                    .file_name("recording.wav")
                    .mime_str("audio/wav")
                    .map_err(|error| ClientError::Transport(error.to_string()))?;
                let form = multipart::Form::new()
                    .part("file", file)
                    .text("model", model.to_string())
                    .text("response_format", "json");
                let request = self
                    .http
                    .post(&url)
                    .header("x-request-id", sent_request_id.as_str())
                    .multipart(form);
                self.execute(request, cancel)?
            }
        };

        parse_transcription(
            &response.body,
            response.request_id.as_deref(),
            &sent_request_id,
        )
    }

    /// Sends one request and buffers the response, mirroring `requestBody` in
    /// `main.ts`: 3xx -> Redirect, non-2xx -> Http with the best detail.
    ///
    /// With `cancel`, the request future races the token in
    /// [`tokio::select!`]: a cancellation aborts the in-flight request —
    /// dropping the future closes the socket — and reports
    /// [`ClientError::Cancelled`]. A token already cancelled returns
    /// before any connection is attempted.
    fn execute(
        &self,
        request: RequestBuilder,
        cancel: Option<&CancelToken>,
    ) -> Result<ApiResponse, ClientError> {
        let perform = async {
            let response = request
                .send()
                .await
                .map_err(|error| transport_error(&error, self.timeout_ms))?;
            let status = response.status().as_u16();
            let request_id = response
                .headers()
                .get("x-request-id")
                .and_then(|value| value.to_str().ok())
                .map(str::to_string);
            let body = response
                .text()
                .await
                .map_err(|error| transport_error(&error, self.timeout_ms))?;

            if (300..400).contains(&status) {
                return Err(ClientError::Redirect(status));
            }
            if !(200..300).contains(&status) {
                return Err(ClientError::Http {
                    status,
                    message: http_error_message(status, &body),
                });
            }
            Ok(ApiResponse { body, request_id })
        };
        match cancel {
            None => self.runtime.block_on(perform),
            Some(cancel) if !cancel.is_cancelled() => self.runtime.block_on(async {
                tokio::select! {
                    result = perform => result,
                    _ = cancel.notified() => Err(ClientError::Cancelled),
                }
            }),
            Some(_) => Err(ClientError::Cancelled),
        }
    }
}

fn build_http_client(timeout_ms: u64) -> Result<Client, ClientError> {
    Client::builder()
        .redirect(Policy::none())
        .timeout(Duration::from_millis(timeout_ms))
        .build()
        .map_err(|error| ClientError::Transport(error_chain(&error)))
}

/// The runtime behind the blocking façade: current-thread (no extra
/// threads — requests run on the caller's thread inside `block_on`) with
/// the I/O and time drivers reqwest needs.
fn build_runtime() -> Result<Runtime, ClientError> {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|error| ClientError::Transport(error_chain(&error)))
}

/// `cleanEndpoint`: http(s) only, no embedded credentials, one trailing `/`
/// trimmed from the WHATWG `href` serialization.
///
/// Public (#213) so the settings save path can validate a draft with
/// exactly the rules the client itself applies — whatever passes here is
/// an endpoint a saved configuration can honor, and whatever fails carries
/// the message the settings dialog should refuse the save with.
pub fn clean_endpoint(value: &str) -> Result<String, ClientError> {
    let url = Url::parse(value)
        .map_err(|_| ClientError::Input("Invalid server endpoint.".to_string()))?;
    if url.scheme() != "http" && url.scheme() != "https" {
        return Err(ClientError::Input(
            "Server endpoint must use http or https.".to_string(),
        ));
    }
    if !url.username().is_empty() || url.password().is_some() {
        return Err(ClientError::Input(
            "Put credentials in a trusted proxy, not the endpoint URL.".to_string(),
        ));
    }
    let mut href = url.to_string();
    if href.ends_with('/') {
        href.pop();
    }
    Ok(href)
}

/// `requestTimeout`: default 180 s, must be 1 ms ..= 10 minutes.
fn request_timeout(timeout_ms: u64) -> Result<u64, ClientError> {
    if (MIN_TIMEOUT_MS..=MAX_TIMEOUT_MS).contains(&timeout_ms) {
        Ok(timeout_ms)
    } else {
        Err(ClientError::Input(
            "Request timeout must be between 1 ms and 10 minutes.".to_string(),
        ))
    }
}

/// `validateRequestId`: non-empty, no CR/LF, not starting with `#`.
fn validate_request_id(value: &str) -> Result<String, ClientError> {
    if !value.is_empty() && !value.starts_with('#') && !value.contains(['\r', '\n']) {
        Ok(value.to_string())
    } else {
        Err(ClientError::Input(
            "Invalid transcription request id.".to_string(),
        ))
    }
}

fn transport_error(error: &reqwest::Error, timeout_ms: u64) -> ClientError {
    if error.is_timeout() {
        ClientError::Timeout(timeout_ms)
    } else {
        ClientError::Transport(error_chain(error))
    }
}

fn error_chain(error: &(dyn StdError + 'static)) -> String {
    let mut message = error.to_string();
    let mut source = error.source();
    while let Some(cause) = source {
        message.push_str(": ");
        message.push_str(&cause.to_string());
        source = cause.source();
    }
    message
}

/// `errorMessage`: first present of `{detail}` / `{error: string}` /
/// `{error: {message}}` / `{message}`, else the raw-body fallback.
fn http_error_message(status: u16, body: &str) -> String {
    if let Ok(Value::Object(map)) = serde_json::from_str::<Value>(body) {
        if let Some(Value::String(detail)) = map.get("detail") {
            return detail.clone();
        }
        match map.get("error") {
            Some(Value::String(error)) => return error.clone(),
            Some(Value::Object(nested)) => {
                if let Some(Value::String(message)) = nested.get("message") {
                    return message.clone();
                }
            }
            _ => {}
        }
        if let Some(Value::String(message)) = map.get("message") {
            return message.clone();
        }
    }
    let truncated: String = body.chars().take(500).collect();
    format!("Server returned {status}: {truncated}")
}

fn decode_object(body: &str, label: &'static str) -> Result<Map<String, Value>, ClientError> {
    match serde_json::from_str::<Value>(body) {
        Ok(Value::Object(map)) => Ok(map),
        _ => Err(ClientError::Protocol(label)),
    }
}

fn required_string(
    map: &Map<String, Value>,
    key: &str,
    label: &'static str,
) -> Result<String, ClientError> {
    match map.get(key) {
        Some(Value::String(text)) => Ok(text.clone()),
        _ => Err(ClientError::Protocol(label)),
    }
}

fn optional_string(
    map: &Map<String, Value>,
    key: &str,
    label: &'static str,
) -> Result<Option<String>, ClientError> {
    match map.get(key) {
        None => Ok(None),
        Some(Value::String(text)) => Ok(Some(text.clone())),
        Some(_) => Err(ClientError::Protocol(label)),
    }
}

fn optional_bool(
    map: &Map<String, Value>,
    key: &str,
    label: &'static str,
) -> Result<Option<bool>, ClientError> {
    match map.get(key) {
        None => Ok(None),
        Some(Value::Bool(flag)) => Ok(Some(*flag)),
        Some(_) => Err(ClientError::Protocol(label)),
    }
}

fn optional_number(
    map: &Map<String, Value>,
    key: &str,
    label: &'static str,
) -> Result<Option<f64>, ClientError> {
    match map.get(key) {
        None => Ok(None),
        Some(Value::Number(number)) => Ok(number.as_f64()),
        Some(_) => Err(ClientError::Protocol(label)),
    }
}

/// `WireHealthSchema` + `normalizeHealth`: strict field types (present fields
/// must be strings/bools/finite numbers as declared).
fn parse_starling_health(body: &str) -> Result<ServerHealth, ClientError> {
    const LABEL: &str = "health";
    let map = decode_object(body, LABEL)?;
    Ok(ServerHealth {
        status: required_string(&map, "status", LABEL)?,
        phase: optional_string(&map, "phase", LABEL)?,
        model: optional_string(&map, "model", LABEL)?,
        loaded: optional_bool(&map, "loaded", LABEL)?,
        busy: optional_bool(&map, "busy", LABEL)?,
        queue_depth: optional_number(&map, "queue_depth", LABEL)?,
    })
}

/// `OpenAiModelsSchema` + `normalizeModels`: `{data: [{id}, …]}` (the `object`
/// key, when present, must be `"list"`) mapped to a ready health snapshot.
fn parse_models_health(body: &str) -> Result<ServerHealth, ClientError> {
    const LABEL: &str = "models";
    let map = decode_object(body, LABEL)?;
    if let Some(object) = map.get("object") {
        if object != &Value::String("list".to_string()) {
            return Err(ClientError::Protocol(LABEL));
        }
    }
    let data = match map.get("data") {
        Some(Value::Array(items)) => items,
        _ => return Err(ClientError::Protocol(LABEL)),
    };
    for item in data {
        let Value::Object(entry) = item else {
            return Err(ClientError::Protocol(LABEL));
        };
        if !matches!(entry.get("id"), Some(Value::String(_))) {
            return Err(ClientError::Protocol(LABEL));
        }
    }
    let model = data
        .first()
        .and_then(|item| item.get("id"))
        .and_then(Value::as_str)
        .map(str::to_string);
    Ok(ServerHealth {
        status: "ok".to_string(),
        phase: Some("ready".to_string()),
        model,
        loaded: None,
        busy: Some(false),
        queue_depth: None,
    })
}

/// `WireTranscriptionSchema` + `normalizeTranscription`: strict field types;
/// `start_s`/`end_s` win over `start`/`end`; both timestamps required with
/// `0 <= start <= end`; `duration >= 0`. Request id preference:
/// body `request_id` -> `x-request-id` header -> the id we sent.
fn parse_transcription(
    body: &str,
    header_request_id: Option<&str>,
    sent_request_id: &str,
) -> Result<TranscriptionResult, ClientError> {
    const LABEL: &str = "transcription";
    let map = decode_object(body, LABEL)?;
    let text = required_string(&map, "text", LABEL)?;
    let wire_segments: &[Value] = match map.get("segments") {
        Some(Value::Array(items)) => items,
        Some(_) => return Err(ClientError::Protocol(LABEL)),
        None => &[],
    };
    let mut segments = Vec::with_capacity(wire_segments.len());
    for item in wire_segments {
        segments.push(parse_segment(item, LABEL)?);
    }

    let duration =
        optional_number(&map, "duration_s", LABEL)?.or(optional_number(&map, "duration", LABEL)?);
    if duration.is_some_and(|value| value < 0.0) {
        return Err(ClientError::Protocol(LABEL));
    }

    let request_id = optional_string(&map, "request_id", LABEL)?
        .or_else(|| header_request_id.map(str::to_string))
        .unwrap_or_else(|| sent_request_id.to_string());

    Ok(TranscriptionResult {
        text,
        segments,
        duration_seconds: duration,
        request_id: Some(request_id),
    })
}

fn parse_segment(item: &Value, label: &'static str) -> Result<TranscriptionSegment, ClientError> {
    let map = match item {
        Value::Object(map) => map,
        _ => return Err(ClientError::Protocol(label)),
    };
    let text = required_string(map, "text", label)?;
    let start = optional_number(map, "start_s", label)?.or(optional_number(map, "start", label)?);
    let end = optional_number(map, "end_s", label)?.or(optional_number(map, "end", label)?);
    let (Some(start), Some(end)) = (start, end) else {
        return Err(ClientError::Protocol(label));
    };
    if start < 0.0 || end < start {
        return Err(ClientError::Protocol(label));
    }
    Ok(TranscriptionSegment {
        text,
        start_seconds: start,
        end_seconds: end,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;
    use std::io::{Read, Write};
    use std::net::{SocketAddr, TcpListener, TcpStream};
    use std::sync::{Arc, Mutex};

    struct RecordedRequest {
        method: String,
        path: String,
        headers: Vec<(String, String)>,
        body: Vec<u8>,
    }

    impl RecordedRequest {
        fn header(&self, name: &str) -> Option<&str> {
            self.headers
                .iter()
                .find(|(key, _)| key == name)
                .map(|(_, value)| value.as_str())
        }
    }

    /// Minimal HTTP/1.1 test double: accepts requests on an ephemeral port,
    /// records them, and answers with one canned response per connection.
    fn spawn_server(responses: Vec<Vec<u8>>) -> (SocketAddr, Arc<Mutex<Vec<RecordedRequest>>>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind test server");
        let addr = listener.local_addr().expect("test server address");
        let recorded = Arc::new(Mutex::new(Vec::new()));
        let mut responses = VecDeque::from(responses);
        let log = Arc::clone(&recorded);
        std::thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                let mut stream = stream;
                let request = read_request(&mut stream);
                log.lock().expect("recorded lock").push(request);
                if let Some(response) = responses.pop_front() {
                    let _ = stream.write_all(&response);
                    let _ = stream.flush();
                }
            }
        });
        (addr, recorded)
    }

    fn read_request(stream: &mut TcpStream) -> RecordedRequest {
        let mut buffer: Vec<u8> = Vec::new();
        let mut chunk = [0u8; 4096];
        let header_end = loop {
            let read = stream.read(&mut chunk).expect("read request head");
            assert!(read > 0, "client hung up before sending a request");
            buffer.extend_from_slice(&chunk[..read]);
            if let Some(position) = locate(&buffer, b"\r\n\r\n") {
                break position;
            }
        };
        let head = String::from_utf8_lossy(&buffer[..header_end]).to_string();
        let mut lines = head.split("\r\n");
        let mut parts = lines.next().unwrap_or_default().split_whitespace();
        let method = parts.next().unwrap_or_default().to_string();
        let path = parts.next().unwrap_or_default().to_string();
        let mut headers = Vec::new();
        for line in lines {
            if let Some((name, value)) = line.split_once(':') {
                headers.push((name.trim().to_ascii_lowercase(), value.trim().to_string()));
            }
        }
        let content_length = headers
            .iter()
            .find(|(name, _)| name == "content-length")
            .and_then(|(_, value)| value.parse::<usize>().ok())
            .unwrap_or(0);
        let mut body = buffer[header_end + 4..].to_vec();
        while body.len() < content_length {
            let read = stream.read(&mut chunk).expect("read request body");
            assert!(read > 0, "client hung up mid-body");
            body.extend_from_slice(&chunk[..read]);
        }
        body.truncate(content_length);
        RecordedRequest {
            method,
            path,
            headers,
            body,
        }
    }

    fn locate(haystack: &[u8], needle: &[u8]) -> Option<usize> {
        haystack
            .windows(needle.len())
            .position(|window| window == needle)
    }

    fn canned_response(status_line: &str, extra_headers: &[(&str, &str)], body: &str) -> Vec<u8> {
        let mut head = format!(
            "HTTP/1.1 {status_line}\r\ncontent-length: {}\r\n",
            body.len()
        );
        for (name, value) in extra_headers {
            head.push_str(name);
            head.push_str(": ");
            head.push_str(value);
            head.push_str("\r\n");
        }
        head.push_str("connection: close\r\n\r\n");
        let mut response = head.into_bytes();
        response.extend_from_slice(body.as_bytes());
        response
    }

    fn ok_json(body: &str) -> Vec<u8> {
        canned_response("200 OK", &[("content-type", "application/json")], body)
    }

    fn fake_wav() -> Vec<u8> {
        let mut wav = Vec::new();
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&36u32.to_le_bytes());
        wav.extend_from_slice(b"WAVEfmt ");
        wav.extend_from_slice(&[0u8; 20]);
        wav.extend_from_slice(b"starling-test-audio-payload");
        assert!(wav.len() >= MIN_AUDIO_BYTES);
        wav
    }

    fn client_at(addr: SocketAddr, protocol: Protocol) -> StarlingClient {
        StarlingClient::new(&format!("http://{addr}"), protocol, "").expect("valid client")
    }

    fn expect_input(error: ClientError, message: &str) {
        match error {
            ClientError::Input(text) => assert_eq!(text, message),
            other => panic!("expected Input error, got {other:?}"),
        }
    }

    #[test]
    fn starling_health_happy_path() {
        let body = r#"{"status":"ok","phase":"ready","model":"parakeet","loaded":true,"busy":false,"queue_depth":2}"#;
        let (addr, recorded) = spawn_server(vec![ok_json(body)]);
        let health = client_at(addr, Protocol::Starling)
            .health()
            .expect("health");

        let request = recorded.lock().expect("recorded lock");
        assert_eq!(request.len(), 1);
        assert_eq!(request[0].method, "GET");
        assert_eq!(request[0].path, "/health");
        drop(request);

        assert_eq!(health.status, "ok");
        assert_eq!(health.phase.as_deref(), Some("ready"));
        assert_eq!(health.model.as_deref(), Some("parakeet"));
        assert_eq!(health.loaded, Some(true));
        assert_eq!(health.busy, Some(false));
        assert_eq!(health.queue_depth, Some(2.0));
    }

    #[test]
    fn openai_health_maps_models_to_ready() {
        let (addr, recorded) = spawn_server(vec![ok_json(
            r#"{"object":"list","data":[{"id":"parakeet"},{"id":"whisper"}]}"#,
        )]);
        let health = client_at(addr, Protocol::OpenAi).health().expect("health");

        let request = recorded.lock().expect("recorded lock");
        assert_eq!(request[0].method, "GET");
        assert_eq!(request[0].path, "/v1/models");
        drop(request);

        assert_eq!(health.status, "ok");
        assert_eq!(health.phase.as_deref(), Some("ready"));
        assert_eq!(health.busy, Some(false));
        assert_eq!(health.model.as_deref(), Some("parakeet"));
        assert_eq!(health.loaded, None);
        assert_eq!(health.queue_depth, None);
    }

    #[test]
    fn starling_transcribe_sends_raw_wav_and_normalizes() {
        let body = r#"{"text":"hello world","segments":[{"text":"hello","start_s":0,"end_s":0.5},{"text":"world","start":0.5,"end":1}],"duration_s":1,"request_id":"from-body"}"#;
        let (addr, recorded) = spawn_server(vec![ok_json(body)]);
        let wav = fake_wav();
        let result = client_at(addr, Protocol::Starling)
            .transcribe(&wav, "req-1")
            .expect("transcription");

        let request = recorded.lock().expect("recorded lock");
        assert_eq!(request.len(), 1);
        assert_eq!(request[0].method, "POST");
        assert_eq!(request[0].path, "/transcribe");
        assert_eq!(request[0].header("x-request-id"), Some("req-1"));
        assert_eq!(request[0].header("content-type"), Some("audio/wav"));
        assert_eq!(request[0].body, wav);
        drop(request);

        assert_eq!(result.text, "hello world");
        assert_eq!(result.segments.len(), 2);
        assert_eq!(result.segments[0].text, "hello");
        assert_eq!(result.segments[0].start_seconds, 0.0);
        assert_eq!(result.segments[0].end_seconds, 0.5);
        assert_eq!(result.segments[1].text, "world");
        assert_eq!(result.segments[1].start_seconds, 0.5);
        assert_eq!(result.segments[1].end_seconds, 1.0);
        assert_eq!(result.duration_seconds, Some(1.0));
        assert_eq!(result.request_id.as_deref(), Some("from-body"));
    }

    #[test]
    fn openai_transcribe_sends_multipart_form() {
        let (addr, recorded) = spawn_server(vec![ok_json(r#"{"text":"hi"}"#)]);
        let wav = fake_wav();
        let result = client_at(addr, Protocol::OpenAi)
            .transcribe(&wav, "req-2")
            .expect("transcription");

        let request = recorded.lock().expect("recorded lock");
        assert_eq!(request.len(), 1);
        assert_eq!(request[0].method, "POST");
        assert_eq!(request[0].path, "/v1/audio/transcriptions");
        assert_eq!(request[0].header("x-request-id"), Some("req-2"));
        let content_type = request[0]
            .header("content-type")
            .expect("multipart content type");
        assert!(
            content_type.starts_with("multipart/form-data; boundary="),
            "got content-type {content_type}"
        );
        let body = String::from_utf8_lossy(&request[0].body).to_string();
        assert!(
            body.contains("name=\"file\"; filename=\"recording.wav\""),
            "multipart body missing file part: {body}"
        );
        assert!(
            body.contains("name=\"model\""),
            "multipart body missing model part"
        );
        assert!(
            body.contains("name=\"response_format\""),
            "multipart body missing response_format part"
        );
        assert!(
            locate(&request[0].body, &wav).is_some(),
            "multipart body must embed the raw wav bytes"
        );
        drop(request);

        assert_eq!(result.text, "hi");
        assert!(result.segments.is_empty());
        assert_eq!(result.duration_seconds, None);
        // No body id, no header: the id we sent wins.
        assert_eq!(result.request_id.as_deref(), Some("req-2"));
    }

    #[test]
    fn http_error_extracts_detail_then_falls_back_to_body() {
        let (addr, _) = spawn_server(vec![canned_response(
            "500 Internal Server Error",
            &[],
            r#"{"detail":"model is loading"}"#,
        )]);
        match client_at(addr, Protocol::Starling).health() {
            Err(ClientError::Http { status, message }) => {
                assert_eq!(status, 500);
                assert_eq!(message, "model is loading");
            }
            other => panic!("expected Http error, got {other:?}"),
        }

        let (addr, _) = spawn_server(vec![canned_response(
            "503 Service Unavailable",
            &[],
            "boom",
        )]);
        match client_at(addr, Protocol::Starling).health() {
            Err(ClientError::Http { status, message }) => {
                assert_eq!(status, 503);
                assert_eq!(message, "Server returned 503: boom");
            }
            other => panic!("expected Http error, got {other:?}"),
        }
    }

    #[test]
    fn redirect_statuses_are_blocked() {
        let (addr, _) = spawn_server(vec![canned_response(
            "302 Found",
            &[("location", "http://127.0.0.1:9/elsewhere")],
            "",
        )]);
        assert!(matches!(
            client_at(addr, Protocol::Starling).health(),
            Err(ClientError::Redirect(302))
        ));
    }

    #[test]
    fn malformed_json_is_a_protocol_error() {
        let (addr, _) = spawn_server(vec![ok_json("this is not json")]);
        assert!(matches!(
            client_at(addr, Protocol::Starling).health(),
            Err(ClientError::Protocol("health"))
        ));

        let (addr, _) = spawn_server(vec![ok_json("{")]);
        assert!(matches!(
            client_at(addr, Protocol::Starling).transcribe(&fake_wav(), "req"),
            Err(ClientError::Protocol("transcription"))
        ));
    }

    #[test]
    fn wrong_typed_health_fields_are_rejected() {
        let (addr, _) = spawn_server(vec![ok_json(r#"{"status":"ok","busy":"nope"}"#)]);
        assert!(matches!(
            client_at(addr, Protocol::Starling).health(),
            Err(ClientError::Protocol("health"))
        ));
    }

    #[test]
    fn invalid_segment_timestamps_are_protocol_errors() {
        let cases = [
            r#"{"text":"x","segments":[{"text":"a","start_s":2,"end_s":1}]}"#,
            r#"{"text":"x","segments":[{"text":"a","start":-0.5,"end":1}]}"#,
            r#"{"text":"x","segments":[{"text":"a","start_s":1}]}"#,
            r#"{"text":"x","duration_s":-1}"#,
        ];
        for body in cases {
            let (addr, _) = spawn_server(vec![ok_json(body)]);
            let error = client_at(addr, Protocol::Starling)
                .transcribe(&fake_wav(), "req")
                .unwrap_err();
            assert!(
                matches!(&error, ClientError::Protocol("transcription")),
                "expected Protocol(transcription) for {body}, got {error:?}"
            );
        }
    }

    fn expect_client_input_error(result: Result<StarlingClient, ClientError>, message: &str) {
        match result {
            Err(ClientError::Input(text)) => assert_eq!(text, message),
            Err(other) => panic!("expected Input error, got {other:?}"),
            Ok(_) => panic!("expected Input error, got Ok(client)"),
        }
    }

    #[test]
    fn endpoint_validation_mirrors_clean_endpoint() {
        expect_client_input_error(
            StarlingClient::new("not a url", Protocol::Starling, ""),
            "Invalid server endpoint.",
        );
        expect_client_input_error(
            StarlingClient::new("ftp://example.com/models", Protocol::OpenAi, ""),
            "Server endpoint must use http or https.",
        );
        expect_client_input_error(
            StarlingClient::new("http://user:pass@example.com:9000", Protocol::Starling, ""),
            "Put credentials in a trusted proxy, not the endpoint URL.",
        );
    }

    #[test]
    fn trailing_slash_is_trimmed_from_endpoint() {
        let (addr, recorded) = spawn_server(vec![ok_json(r#"{"status":"ok"}"#)]);
        let endpoint = format!("http://{addr}/");
        StarlingClient::new(&endpoint, Protocol::Starling, "")
            .expect("valid client")
            .health()
            .expect("health");
        let request = recorded.lock().expect("recorded lock");
        assert_eq!(request[0].path, "/health");
    }

    #[test]
    fn transcribe_validates_request_id_and_payload() {
        let client = StarlingClient::new("http://127.0.0.1:9", Protocol::Starling, "")
            .expect("valid client");
        expect_input(
            client.transcribe(&fake_wav(), "").unwrap_err(),
            "Invalid transcription request id.",
        );
        expect_input(
            client.transcribe(&fake_wav(), "#hidden").unwrap_err(),
            "Invalid transcription request id.",
        );
        expect_input(
            client.transcribe(&fake_wav(), "a\r\nb").unwrap_err(),
            "Invalid transcription request id.",
        );
        expect_input(
            client.transcribe(&[0u8; 43], "req").unwrap_err(),
            "Audio payload is empty or too large.",
        );
    }

    #[test]
    fn slow_response_maps_to_timeout_error() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind test server");
        let addr = listener.local_addr().expect("test server address");
        std::thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                let mut stream = stream;
                let _ = read_request(&mut stream);
                std::thread::sleep(Duration::from_millis(300));
                let _ = stream.write_all(&ok_json(r#"{"status":"ok"}"#));
            }
        });
        let client = StarlingClient::new(&format!("http://{addr}"), Protocol::Starling, "")
            .expect("valid client")
            .with_timeout_ms(50)
            .expect("valid timeout");
        match client.health() {
            Err(ClientError::Timeout(50)) => {}
            other => panic!("expected Timeout(50), got {other:?}"),
        }
    }

    /// Issue #251, client half: a cancelled request must abort the
    /// in-flight HTTP call — the blocking `transcribe_with_cancel` returns
    /// promptly with `Cancelled` while the server sits on its response,
    /// instead of holding the caller for the rest of the request timeout.
    #[test]
    fn cancelled_request_aborts_the_in_flight_transcription() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind test server");
        let addr = listener.local_addr().expect("test server address");
        let seen = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counter = Arc::clone(&seen);
        std::thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                let mut stream = stream;
                let _ = read_request(&mut stream);
                counter.fetch_add(1, Ordering::SeqCst);
                // Never answer within the test's horizon: the only way
                // the client can return is cancellation (or its 30 s
                // timeout, which the assertions below rule out).
                std::thread::sleep(Duration::from_secs(600));
            }
        });
        let client = StarlingClient::new(&format!("http://{addr}"), Protocol::Starling, "")
            .expect("valid client")
            .with_timeout_ms(30_000)
            .expect("valid timeout");
        let token = CancelToken::new();
        let wav = fake_wav();
        let (done_tx, done_rx) = std::sync::mpsc::channel::<()>();
        let caller_token = token.clone();
        let caller = std::thread::spawn(move || {
            let result = client.transcribe_with_cancel(&wav, "req-cancel", Some(&caller_token));
            let _ = done_tx.send(());
            result
        });

        // Gate the cancellation on the request provably being in flight:
        // the server has read it. A fixed sleep could cancel before the
        // connection was even established (#217).
        let start = std::time::Instant::now();
        while seen.load(Ordering::SeqCst) == 0 {
            assert!(
                start.elapsed() < Duration::from_secs(5),
                "test server never saw the transcription request"
            );
            std::thread::sleep(Duration::from_millis(5));
        }
        token.cancel();
        done_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("transcribe must return promptly after cancellation");
        match caller.join().expect("caller thread") {
            Err(ClientError::Cancelled) => {}
            other => panic!("expected Cancelled, got {other:?}"),
        }
    }

    /// The already-cancelled case needs no server cooperation: the early
    /// flag check returns before any connection is attempted, so nothing
    /// is ever sent.
    #[test]
    fn precancelled_transcribe_returns_without_sending() {
        let (addr, recorded) = spawn_server(vec![ok_json(r#"{"text":"too late"}"#)]);
        let token = CancelToken::new();
        token.cancel();
        let client = client_at(addr, Protocol::Starling);
        match client.transcribe_with_cancel(&fake_wav(), "req-precancel", Some(&token)) {
            Err(ClientError::Cancelled) => {}
            other => panic!("expected Cancelled, got {other:?}"),
        }
        let recorded = recorded.lock().expect("recorded lock");
        assert_eq!(
            recorded.len(),
            0,
            "a precancelled request must not reach the server"
        );
    }
}
