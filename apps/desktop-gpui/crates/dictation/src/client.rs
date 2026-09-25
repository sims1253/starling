// Ported from `apps/desktop/electron/ipc.ts` + `electron/main.ts` — the native
// Electron request path (`healthProgram` / `transcribeProgram`) — together with
// the response normalization of `packages/dictation/src/client.ts`.
// See `apps/desktop-gpui/PORT.md` ("Transcription client") for the contract.

use std::error::Error as StdError;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use reqwest::multipart;
use reqwest::redirect::Policy;
use reqwest::{Client, RequestBuilder, Url};
use serde_json::{Map, Value};
use tokio::runtime::Runtime;

use crate::storage::TranscriptionResult;

const DEFAULT_TIMEOUT_MS: u64 = 180_000;
const MIN_TIMEOUT_MS: u64 = 1;
const MAX_TIMEOUT_MS: u64 = 600_000;
const MIN_AUDIO_BYTES: usize = 44;
const MAX_AUDIO_BYTES: usize = 256 * 1024 * 1024;
/// Response bodies are refused past this size (issue #235). Health and
/// transcription payloads are small JSON documents (a transcript is text),
/// so 10 MiB is orders of magnitude above anything a real backend returns
/// while still bounding what a broken or hostile server can make the
/// client buffer. Override per client with
/// [`StarlingClient::with_max_response_bytes`].
const DEFAULT_MAX_RESPONSE_BYTES: usize = 10 * 1024 * 1024;

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
    /// The server's response body exceeded the client's size limit (issue
    /// #235): a malformed or hostile endpoint cannot make the client buffer
    /// unbounded output. Exceeding the limit fails the request outright —
    /// never a silent truncation.
    #[error("Server response body exceeded the {0} byte limit.")]
    ResponseTooLarge(usize),
    /// The caller cancelled the request while it was in flight (issue
    /// #251): the connection was aborted, not timed out — distinct from
    /// [`ClientError::Timeout`], which is the server being slow.
    #[error("The request was cancelled before it completed.")]
    Cancelled,
}

/// A cooperative cancellation signal for in-flight requests (issue #251).
///
/// One side calls [`CancelToken::cancel`] — from any thread; the signal
/// is monotonic, so extra trips are harmless — and every request
/// carrying this token aborts. The wakeup is a *broadcast*: any number
/// of concurrent waiters each observe the change, so a token shared by
/// several in-flight requests cancels all of them (the durable flag
/// covers the before-send fast path; a `watch` channel reaches every
/// waiter). Clone shares the signal.
#[derive(Clone, Default)]
pub struct CancelToken {
    state: Arc<CancelState>,
}

struct CancelState {
    flag: AtomicBool,
    /// The broadcast side of the signal. Every waiter subscribes its
    /// own receiver — unlike `Notify`'s single stored permit, which
    /// would wake exactly one of several concurrent waiters.
    cancelled: tokio::sync::watch::Sender<bool>,
}

impl Default for CancelState {
    fn default() -> Self {
        let (cancelled, _) = tokio::sync::watch::channel(false);
        CancelState {
            flag: AtomicBool::new(false),
            cancelled,
        }
    }
}

impl CancelToken {
    pub fn new() -> Self {
        Self::default()
    }

    /// Trips the signal (idempotently) and wakes every request waiting
    /// on this token.
    pub fn cancel(&self) {
        self.state.flag.store(true, Ordering::Release);
        // `send` fails only when no receiver is alive — then there is
        // nothing to wake; the flag above is the durable half of the
        // signal for anyone who checks later.
        let _ = self.state.cancelled.send(true);
    }

    /// Whether [`CancelToken::cancel`] has been called.
    pub fn is_cancelled(&self) -> bool {
        self.state.flag.load(Ordering::Acquire)
    }

    /// Resolves once the token is cancelled: the public form of the
    /// broadcast wait, so other request clients (the processing
    /// providers, #294) can race their own futures against the same
    /// signal.
    pub async fn cancelled(&self) {
        self.notified().await
    }

    /// Resolves once the token is cancelled — for *every* concurrent
    /// caller: each waiter watches its own receiver, and `wait_for`
    /// consults the current value first, so a cancel that landed before
    /// the subscription is seen immediately (no registration race, no
    /// permit to win).
    async fn notified(&self) {
        let mut receiver = self.state.cancelled.subscribe();
        let _ = receiver.wait_for(|cancelled| *cancelled).await;
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
    model: String,
    timeout_ms: u64,
    max_response_bytes: usize,
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
    pub fn new(endpoint: &str, model: &str) -> Result<Self, ClientError> {
        Self::build(
            clean_endpoint(endpoint)?,
            model,
            DEFAULT_TIMEOUT_MS,
            DEFAULT_MAX_RESPONSE_BYTES,
        )
    }

    /// Overrides the total per-request timeout (1 ms ..= 10 min), mirroring
    /// `requestTimeout` in `main.ts`.
    pub fn with_timeout_ms(mut self, timeout_ms: u64) -> Result<Self, ClientError> {
        self.timeout_ms = request_timeout(timeout_ms)?;
        self.http = build_http_client(self.timeout_ms)?;
        Ok(self)
    }

    /// Overrides the response-body size limit (default 10 MiB, issue
    /// #235). A body declared or delivered past the limit fails with
    /// [`ClientError::ResponseTooLarge`] instead of being buffered.
    pub fn with_max_response_bytes(mut self, max_bytes: usize) -> Result<Self, ClientError> {
        self.max_response_bytes = response_limit(max_bytes)?;
        Ok(self)
    }

    fn build(
        base_url: String,
        model: &str,
        timeout_ms: u64,
        max_response_bytes: usize,
    ) -> Result<Self, ClientError> {
        Ok(Self {
            base_url,
            model: model.trim().to_string(),
            timeout_ms,
            max_response_bytes,
            http: build_http_client(timeout_ms)?,
            runtime: build_runtime()?,
        })
    }

    /// Probe the configured server's model list.
    pub fn health(&self) -> Result<ServerHealth, ClientError> {
        let response = self.execute(self.http.get(format!("{}/v1/models", self.base_url)), None)?;
        parse_models_health(&response.body)
    }

    /// `transcribeProgram` without cancellation: runs to the timeout.
    ///
    /// The WAV travels as the caller's shared buffer: the request body is
    /// built from it without copying (issue #235 — a 256 MB take must not
    /// double peak memory per request), so both sides hold one recording,
    /// not two.
    pub fn transcribe(
        &self,
        wav: Arc<Vec<u8>>,
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
        wav: Arc<Vec<u8>>,
        request_id: &str,
        cancel: Option<&CancelToken>,
    ) -> Result<TranscriptionResult, ClientError> {
        let sent_request_id = validate_request_id(request_id)?;
        if wav.len() < MIN_AUDIO_BYTES || wav.len() > MAX_AUDIO_BYTES {
            return Err(ClientError::Input(
                "Audio payload is empty or too large.".to_string(),
            ));
        }

        let url = format!("{}/v1/audio/transcriptions", self.base_url);
        let model = if self.model.is_empty() {
            "parakeet"
        } else {
            self.model.as_str()
        };
        let file = multipart::Part::stream_with_length(wav_bytes(&wav), wav.len() as u64)
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
        let response = self.execute(request, cancel)?;

        parse_transcription(
            &response.body,
            response.request_id.as_deref(),
            &sent_request_id,
        )
    }

    /// Sends one request and reads the response under the client's size
    /// limit, mirroring `requestBody` in `main.ts`: 3xx -> Redirect,
    /// non-2xx -> Http with the best detail.
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
            let body = read_body_capped(response, self.max_response_bytes, self.timeout_ms).await?;

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
///
/// A build failure here is environmental (thread/resource limits on
/// this machine), not a request condition. `ClientError` has no
/// config/setup variant — the taxonomy is the request lifecycle — so it
/// maps to `Transport`, the local-unavailability bucket; either way the
/// client could not be constructed, which every caller treats as a
/// hard setup failure rather than something to classify finer.
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

/// The response-body limit must leave room for at least one byte; there
/// is no upper bound (a caller may legitimately want a bigger ceiling,
/// not a smaller floor).
fn response_limit(max_bytes: usize) -> Result<usize, ClientError> {
    if max_bytes >= 1 {
        Ok(max_bytes)
    } else {
        Err(ClientError::Input(
            "Response size limit must be at least 1 byte.".to_string(),
        ))
    }
}

/// A borrowed view of the caller's shared recording buffer. `Bytes::
/// from_owner` keeps the `Arc` alive while the request body points into
/// it, so the upload streams the WAV hyper already holds — no second
/// copy of a 256 MB take exists anywhere (issue #235). `Arc<Vec<u8>>`
/// has no `AsRef<[u8]>` of its own, hence the newtype.
struct SharedWav(Arc<Vec<u8>>);

impl AsRef<[u8]> for SharedWav {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

fn wav_bytes(wav: &Arc<Vec<u8>>) -> Bytes {
    Bytes::from_owner(SharedWav(Arc::clone(wav)))
}

/// Reads a response body under a hard size cap (issue #235): the declared
/// `content-length` is checked before a byte is read, and a chunked or
/// lying body is policed chunk by chunk — a malformed server cannot make
/// the client buffer unbounded output. Exceeding the limit is
/// [`ClientError::ResponseTooLarge`], distinct from truncation and from
/// transport failures; the response is dropped (closing the connection)
/// as soon as the cap is crossed. Decoding is lossy UTF-8, matching
/// `Response::text` for the charset-less JSON these endpoints return.
async fn read_body_capped(
    mut response: reqwest::Response,
    limit: usize,
    timeout_ms: u64,
) -> Result<String, ClientError> {
    // The cap bounds what the client holds in memory, so it counts
    // DECOMPRESSED bytes: with transparent gzip/br decompression,
    // `content_length()` is the compressed wire size while the chunk
    // loop below sees the decoded stream — the loop is what enforces the
    // real bound; this pre-check is the fast path for uncompressed
    // bodies.
    if response
        .content_length()
        .is_some_and(|length| length > limit as u64)
    {
        return Err(ClientError::ResponseTooLarge(limit));
    }
    // The declared length — already pre-checked against the limit —
    // reserves capacity up front so a content-length-framed body is read
    // without reallocation, but the reservation is capped small: a lying
    // content-length near the limit must not force a full-limit
    // allocation before any body arrives. The Vec grows as chunks
    // actually arrive; an undeclared (chunked) body starts empty.
    const MAX_INITIAL_RESERVATION: usize = 64 * 1024;
    let capacity = response.content_length().map_or(0, |length| {
        length.min(MAX_INITIAL_RESERVATION as u64).min(limit as u64) as usize
    });
    let mut bytes = Vec::with_capacity(capacity);
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|error| transport_error(&error, timeout_ms))?
    {
        if bytes.len() + chunk.len() > limit {
            return Err(ClientError::ResponseTooLarge(limit));
        }
        bytes.extend_from_slice(&chunk);
    }
    Ok(String::from_utf8_lossy(&bytes).into_owned())
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

/// The batch API returns raw text; the request id is a response header.
fn parse_transcription(
    body: &str,
    header_request_id: Option<&str>,
    sent_request_id: &str,
) -> Result<TranscriptionResult, ClientError> {
    const LABEL: &str = "transcription";
    let map = decode_object(body, LABEL)?;
    let text = required_string(&map, "text", LABEL)?;
    let request_id = header_request_id
        .map(str::to_string)
        .unwrap_or_else(|| sent_request_id.to_string());

    Ok(TranscriptionResult {
        text,
        segments: Vec::new(),
        duration_seconds: None,
        request_id: Some(request_id),
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

    /// A 200 with `transfer-encoding: chunked` framing: no declared
    /// length, so the client's cap can only bite while streaming (issue
    /// #235).
    fn chunked_response(chunks: &[&str]) -> Vec<u8> {
        let mut response =
            b"HTTP/1.1 200 OK\r\ntransfer-encoding: chunked\r\nconnection: close\r\n\r\n".to_vec();
        for chunk in chunks {
            response.extend_from_slice(format!("{:x}\r\n", chunk.len()).as_bytes());
            response.extend_from_slice(chunk.as_bytes());
            response.extend_from_slice(b"\r\n");
        }
        response.extend_from_slice(b"0\r\n\r\n");
        response
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

    fn client_at(addr: SocketAddr) -> StarlingClient {
        StarlingClient::new(&format!("http://{addr}"), "").expect("valid client")
    }

    fn expect_input(error: ClientError, message: &str) {
        match error {
            ClientError::Input(text) => assert_eq!(text, message),
            other => panic!("expected Input error, got {other:?}"),
        }
    }

    #[test]
    fn openai_health_maps_models_to_ready() {
        let (addr, recorded) = spawn_server(vec![ok_json(
            r#"{"object":"list","data":[{"id":"parakeet"},{"id":"whisper"}]}"#,
        )]);
        let health = client_at(addr).health().expect("health");

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
    fn openai_transcribe_sends_multipart_form() {
        let (addr, recorded) = spawn_server(vec![ok_json(r#"{"text":"hi"}"#)]);
        let wav = fake_wav();
        let result = client_at(addr)
            .transcribe(Arc::new(wav.clone()), "req-2")
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
        // `stream_with_length` keeps the whole form length-declared, so
        // the streamed file part does not switch the request to chunked
        // transfer encoding (issue #235).
        assert_eq!(
            request[0].header("content-length"),
            Some(request[0].body.len().to_string().as_str())
        );
        assert_eq!(request[0].header("transfer-encoding"), None);
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
        match client_at(addr).health() {
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
        match client_at(addr).health() {
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
            client_at(addr).health(),
            Err(ClientError::Redirect(302))
        ));
    }

    #[test]
    fn malformed_json_is_a_protocol_error() {
        let (addr, _) = spawn_server(vec![ok_json("this is not json")]);
        assert!(matches!(
            client_at(addr).health(),
            Err(ClientError::Protocol("models"))
        ));

        let (addr, _) = spawn_server(vec![ok_json("{")]);
        assert!(matches!(
            client_at(addr).transcribe(Arc::new(fake_wav()), "req"),
            Err(ClientError::Protocol("transcription"))
        ));
    }

    #[test]
    fn wrong_typed_health_fields_are_rejected() {
        let (addr, _) = spawn_server(vec![ok_json(r#"{"object":"list","data":[{"id":42}]}"#)]);
        assert!(matches!(
            client_at(addr).health(),
            Err(ClientError::Protocol("models"))
        ));
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
            StarlingClient::new("not a url", ""),
            "Invalid server endpoint.",
        );
        expect_client_input_error(
            StarlingClient::new("ftp://example.com/models", ""),
            "Server endpoint must use http or https.",
        );
        expect_client_input_error(
            StarlingClient::new("http://user:pass@example.com:9000", ""),
            "Put credentials in a trusted proxy, not the endpoint URL.",
        );
    }

    #[test]
    fn trailing_slash_is_trimmed_from_endpoint() {
        let (addr, recorded) = spawn_server(vec![ok_json(r#"{"object":"list","data":[]}"#)]);
        let endpoint = format!("http://{addr}/");
        StarlingClient::new(&endpoint, "")
            .expect("valid client")
            .health()
            .expect("health");
        let request = recorded.lock().expect("recorded lock");
        assert_eq!(request[0].path, "/v1/models");
    }

    #[test]
    fn transcribe_validates_request_id_and_payload() {
        let client = StarlingClient::new("http://127.0.0.1:9", "").expect("valid client");
        expect_input(
            client.transcribe(Arc::new(fake_wav()), "").unwrap_err(),
            "Invalid transcription request id.",
        );
        expect_input(
            client
                .transcribe(Arc::new(fake_wav()), "#hidden")
                .unwrap_err(),
            "Invalid transcription request id.",
        );
        expect_input(
            client
                .transcribe(Arc::new(fake_wav()), "a\r\nb")
                .unwrap_err(),
            "Invalid transcription request id.",
        );
        expect_input(
            client
                .transcribe(Arc::new(vec![0u8; 43]), "req")
                .unwrap_err(),
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
                let _ = stream.write_all(&ok_json(r#"{"object":"list","data":[]}"#));
            }
        });
        let client = StarlingClient::new(&format!("http://{addr}"), "")
            .expect("valid client")
            .with_timeout_ms(50)
            .expect("valid timeout");
        match client.health() {
            Err(ClientError::Timeout(50)) => {}
            other => panic!("expected Timeout(50), got {other:?}"),
        }
    }

    /// Issue #235: a response whose declared content-length already
    /// exceeds the client's limit is refused before a byte is read — a
    /// malformed or hostile server cannot make the client buffer it.
    #[test]
    fn oversized_declared_response_is_refused() {
        let (addr, _) = spawn_server(vec![canned_response("200 OK", &[], &"x".repeat(65))]);
        let client = client_at(addr)
            .with_max_response_bytes(64)
            .expect("valid limit");
        match client.health() {
            Err(ClientError::ResponseTooLarge(64)) => {}
            other => panic!("expected ResponseTooLarge(64), got {other:?}"),
        }
    }

    /// A body with no declared length (chunked framing) is policed chunk
    /// by chunk: the refusal lands once the stream crosses the limit, not
    /// after buffering everything the server cared to send.
    #[test]
    fn oversized_chunked_response_is_refused_while_streaming() {
        let (addr, _) = spawn_server(vec![chunked_response(&[&"a".repeat(40), &"b".repeat(40)])]);
        let client = client_at(addr)
            .with_max_response_bytes(64)
            .expect("valid limit");
        match client.health() {
            Err(ClientError::ResponseTooLarge(64)) => {}
            other => panic!("expected ResponseTooLarge(64), got {other:?}"),
        }
    }

    /// The cap must not bite honest payloads: a body exactly at the limit
    /// parses normally (every smaller one — the rest of this suite —
    /// already exercises the under-the-cap path).
    #[test]
    fn response_exactly_at_the_limit_is_accepted() {
        let padded = format!("{:<64}", r#"{"object":"list","data":[]}"#);
        assert_eq!(padded.len(), 64);
        let (addr, _) = spawn_server(vec![ok_json(&padded)]);
        let client = client_at(addr)
            .with_max_response_bytes(64)
            .expect("valid limit");
        assert_eq!(client.health().expect("health").status, "ok");
    }

    #[test]
    fn response_limit_must_be_positive() {
        let client = StarlingClient::new("http://127.0.0.1:9", "").expect("valid client");
        expect_client_input_error(
            client.with_max_response_bytes(0),
            "Response size limit must be at least 1 byte.",
        );
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
            // Exactly one connection is expected; accepting it and
            // dropping the listener lets this thread exit instead of
            // parking on `incoming()` for the rest of the suite.
            let Ok((mut stream, _)) = listener.accept() else {
                return;
            };
            let _ = read_request(&mut stream);
            counter.fetch_add(1, Ordering::SeqCst);
            // Hold the connection open (never answer) until the client's
            // abort closes it — read EOF here. The read timeout is only
            // an escape hatch so a misbehaving client cannot pin this
            // thread; the abort path is what ends it, promptly.
            let _ = stream.set_read_timeout(Some(Duration::from_secs(10)));
            let mut sink = [0u8; 1024];
            loop {
                match stream.read(&mut sink) {
                    Ok(0) | Err(_) => return,
                    Ok(_) => {}
                }
            }
        });
        let client = StarlingClient::new(&format!("http://{addr}"), "")
            .expect("valid client")
            .with_timeout_ms(30_000)
            .expect("valid timeout");
        let token = CancelToken::new();
        let wav = Arc::new(fake_wav());
        let (done_tx, done_rx) = std::sync::mpsc::channel::<()>();
        let caller_token = token.clone();
        let caller = std::thread::spawn(move || {
            let result = client.transcribe_with_cancel(wav, "req-cancel", Some(&caller_token));
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

    /// The precancelled case needs no server cooperation: the early
    /// flag check returns before any connection is attempted, so nothing
    /// is ever sent.
    #[test]
    fn precancelled_transcribe_returns_without_sending() {
        let (addr, recorded) = spawn_server(vec![ok_json(r#"{"text":"too late"}"#)]);
        let token = CancelToken::new();
        token.cancel();
        let client = client_at(addr);
        match client.transcribe_with_cancel(Arc::new(fake_wav()), "req-precancel", Some(&token)) {
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

    /// Review on #253: the token's wakeup is a broadcast, not a single
    /// `Notify` permit — two waiters registered *before* the cancel must
    /// both wake on one trip (the shape that would leave exactly one of
    /// them parked forever under `notify_one`).
    #[tokio::test]
    async fn one_cancel_wakes_every_registered_waiter() {
        let token = CancelToken::new();
        let canceller = {
            let token = token.clone();
            tokio::spawn(async move {
                // A trigger, not a gate: both waiters below are polled
                // (registered) before this fires.
                tokio::time::sleep(Duration::from_millis(50)).await;
                token.cancel();
            })
        };
        tokio::time::timeout(Duration::from_secs(1), async {
            let ((), ()) = tokio::join!(token.notified(), token.notified());
        })
        .await
        .expect("both concurrent waiters must wake on a single cancel");
        canceller.await.expect("canceller task");
    }
}
