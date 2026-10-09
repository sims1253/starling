//! The MCP stdio server for `ask_user_dictation`: newline-delimited
//! JSON-RPC 2.0 on stdin/stdout, bridged onto the host's ask surface
//! ([`crate::agent`]). Hand-rolled over `serde_json` to avoid a new
//! dependency; it implements `initialize`, `ping`, `tools/list`,
//! `tools/call` and `notifications/cancelled`.
//!
//! # Result mapping
//!
//! - **Answered** → the transcript as plain text content.
//! - **No answer** (timeout, cancel, decline) → `isError: true` with
//!   `"No answer: <reason>."`, so a model cannot mistake it for words
//!   the user spoke.
//! - **Error** → `isError: true` with `"Error [<code>]: <message>."`.
//! - Malformed arguments → JSON-RPC `-32602`.
//!
//! `tools/call` requests may overlap. Replies are written whenever the
//! host resolves each ask, so the stdin loop never waits on an answer
//! and stays free to read cancellations. Replies leave through a writer
//! thread with a bounded queue: a stdout write failure or a reply-queue
//! overflow ends the session rather than stalling the stdin loop or
//! losing replies.

use std::collections::HashMap;
use std::io::{BufRead, Read, Write};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use serde_json::{json, Value};
use starling_runtime::channel::{bounded, Receiver, Sender};

use crate::agent::{validate_ask, MAX_QUESTIONS, MAX_TIMEOUT_MS, MIN_TIMEOUT_MS};
use crate::frame::{AskOutcome, NoAnswerReason};

/// The protocol revision advertised when the client asks for one this
/// server does not know.
pub const PROTOCOL_VERSION: &str = "2025-06-18";

const KNOWN_VERSIONS: [&str; 4] = ["2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25"];

pub const DEFAULT_TIMEOUT_MS: u64 = 120_000;

/// The longest accepted stdin line; a legitimate message is kilobytes.
const MAX_LINE_BYTES: usize = 1024 * 1024;

const SERVER_NOT_INITIALIZED: i64 = -32002;

/// Replies waiting for stdout. A full queue means the agent stopped
/// reading.
const OUT_QUEUE: usize = 256;

pub const TOOL_NAME: &str = "ask_user_dictation";

/// Where asks go: the host connection in the binary, a fake in tests.
pub trait AskSink: Send + Sync {
    fn ask(&self, req: &str, questions: Vec<String>, timeout_ms: u64) -> Result<(), String>;
    fn cancel(&self, req: &str, reason: &str);
}

/// One agent session. Run [`McpServer::serve_read`] on the stdin thread
/// and feed host results to [`McpServer::complete`] from a clone.
#[derive(Clone)]
pub struct McpServer {
    inner: Arc<Shared>,
}

struct Shared {
    next_req: AtomicU64,
    initialized: AtomicBool,
    /// Host-side ask token → the JSON-RPC id waiting on it.
    pending: Mutex<HashMap<String, Value>>,
    out: Sender<String>,
    /// Replies queued or being written.
    unwritten: Arc<AtomicUsize>,
    broken: Arc<dyn Fn() + Send + Sync>,
}

impl McpServer {
    /// Writes replies to `out` on a dedicated thread. `on_broken` runs
    /// when the agent stops draining replies or the write fails; the
    /// caller ends the session there.
    pub fn new(
        out: impl Write + Send + 'static,
        on_broken: impl Fn() + Send + Sync + 'static,
    ) -> McpServer {
        let (tx, rx) = bounded(OUT_QUEUE);
        let broken: Arc<dyn Fn() + Send + Sync> = Arc::new(on_broken);
        let unwritten = Arc::new(AtomicUsize::new(0));
        {
            let broken = Arc::clone(&broken);
            let unwritten = Arc::clone(&unwritten);
            std::thread::Builder::new()
                .name("starling-mcp-write".to_string())
                .spawn(move || write_loop(out, rx, &unwritten, &*broken))
                .expect("mcp writer thread spawns");
        }
        McpServer {
            inner: Arc::new(Shared {
                next_req: AtomicU64::new(0),
                initialized: AtomicBool::new(false),
                pending: Mutex::new(HashMap::new()),
                out: tx,
                unwritten,
                broken,
            }),
        }
    }

    /// Serves stdin until EOF or a read error.
    pub fn serve_read(&self, mut input: impl BufRead, sink: &dyn AskSink) {
        while let Some(line) = read_line(&mut input) {
            let message = line.and_then(|line| {
                serde_json::from_str(&line).map_err(|err| format!("parse error: {err}"))
            });
            match message {
                Ok(message) => self.handle(message, sink),
                Err(err) => self.error(Value::Null, -32700, &err),
            }
        }
    }

    /// Answers the `tools/call` waiting on `req`, if any.
    pub fn complete(&self, req: &str, outcome: AskOutcome) {
        let Some(id) = self.pending().remove(req) else {
            return;
        };
        let result = match outcome {
            AskOutcome::Answered { text, .. } => tool_text(text, false),
            AskOutcome::NoAnswer { reason } => {
                tool_text(format!("No answer: {}.", no_answer_text(reason)), true)
            }
            AskOutcome::Error { code, message } => {
                tool_text(format!("Error [{code}]: {message}."), true)
            }
        };
        self.reply(id, result);
    }

    /// Fails every pending `tools/call` after the host connection died.
    pub fn fail_pending(&self, message: &str) {
        let pending: Vec<Value> = self.pending().drain().map(|(_, id)| id).collect();
        for id in pending {
            let text = format!("Error [host_disconnected]: {message}.");
            self.reply(id, tool_text(text, true));
        }
    }

    fn pending(&self) -> MutexGuard<'_, HashMap<String, Value>> {
        self.inner.pending.lock().expect("mcp pending lock")
    }

    fn handle(&self, message: Value, sink: &dyn AskSink) {
        // MCP has no batching, so an array is as malformed as a scalar.
        let Some(object) = message.as_object() else {
            return self.error(Value::Null, -32600, "request must be a JSON object");
        };
        if object.get("jsonrpc").is_some_and(|v| v != "2.0") {
            return self.error(Value::Null, -32600, "jsonrpc must be exactly \"2.0\"");
        }
        let id = object.get("id").filter(|id| !id.is_null()).cloned();
        let Some(method) = object.get("method").and_then(Value::as_str) else {
            let id = id.unwrap_or(Value::Null);
            return self.error(id, -32600, "a request must carry a string method");
        };
        let params = object.get("params").unwrap_or(&Value::Null);
        // Notifications are never answered.
        let Some(id) = id else {
            if method == "notifications/cancelled" {
                self.handle_cancelled(params, sink);
            }
            return;
        };
        if method != "initialize" && !self.inner.initialized.load(Ordering::SeqCst) {
            return self.error(
                id,
                SERVER_NOT_INITIALIZED,
                "server not initialized: send initialize first",
            );
        }
        match method {
            "initialize" => self.handle_initialize(id, params),
            "ping" => self.reply(id, json!({})),
            "tools/list" => self.reply(id, json!({"tools": [tool_spec()]})),
            "tools/call" => self.handle_tools_call(id, params, sink),
            _ => self.error(id, -32601, &format!("method not found: {method}")),
        }
    }

    fn handle_initialize(&self, id: Value, params: &Value) {
        let requested = params.get("protocolVersion").and_then(Value::as_str);
        let version = requested
            .filter(|version| KNOWN_VERSIONS.contains(version))
            .unwrap_or(PROTOCOL_VERSION);
        self.inner.initialized.store(true, Ordering::SeqCst);
        self.reply(
            id,
            json!({
                "protocolVersion": version,
                "capabilities": {"tools": {}},
                "serverInfo": {
                    "name": "starling",
                    "version": env!("CARGO_PKG_VERSION"),
                    "title": "Starling dictation",
                },
                "instructions": format!(
                    "Ask the user questions by voice with `{TOOL_NAME}`. The desktop app \
                     shows the prompt and records the spoken answer; the tool returns the \
                     transcript. Cancelled, declined, and timed-out asks return \
                     `isError: true` with a \"No answer\" message — never treat those as \
                     spoken words."
                ),
            }),
        );
    }

    fn handle_tools_call(&self, id: Value, params: &Value, sink: &dyn AskSink) {
        let (questions, timeout_ms) = match parse_call(params) {
            Ok(call) => call,
            Err(detail) => return self.error(id, -32602, &detail),
        };
        let req = format!("ask{}", self.inner.next_req.fetch_add(1, Ordering::SeqCst));
        // Registered before sending: a fast refusal can arrive before
        // `ask` returns.
        self.pending().insert(req.clone(), id);
        if let Err(message) = sink.ask(&req, questions, timeout_ms) {
            if let Some(id) = self.pending().remove(&req) {
                self.error(id, -32603, &format!("the ask could not be sent: {message}"));
            }
        }
    }

    /// The completion still arrives (as a no-answer) and answers the
    /// request; MCP permits responding to a cancelled request.
    fn handle_cancelled(&self, params: &Value, sink: &dyn AskSink) {
        let Some(request_id) = params.get("requestId") else {
            return;
        };
        let reason = params
            .get("reason")
            .and_then(Value::as_str)
            .unwrap_or("cancelled by the agent");
        let req = self
            .pending()
            .iter()
            .find(|(_, id)| *id == request_id)
            .map(|(req, _)| req.clone());
        if let Some(req) = req {
            sink.cancel(&req, reason);
        }
    }

    fn reply(&self, id: Value, result: Value) {
        self.write(json!({"jsonrpc": "2.0", "id": id, "result": result}));
    }

    fn error(&self, id: Value, code: i64, message: &str) {
        self.write(json!({
            "jsonrpc": "2.0",
            "id": id,
            "error": {"code": code, "message": message},
        }));
    }

    /// One JSON document per line; the lock keeps concurrent replies
    /// from interleaving.
    fn write(&self, message: Value) {
        self.inner.unwritten.fetch_add(1, Ordering::SeqCst);
        if self.inner.out.try_send(message.to_string()).is_err() {
            self.inner.unwritten.fetch_sub(1, Ordering::SeqCst);
            (self.inner.broken)();
        }
    }

    /// Waits, up to `within`, for every queued reply to be written.
    pub fn drain(&self, within: Duration) {
        let deadline = Instant::now() + within;
        while self.inner.unwritten.load(Ordering::SeqCst) > 0 && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(5));
        }
    }
}

fn write_loop(
    mut out: impl Write,
    replies: Receiver<String>,
    unwritten: &AtomicUsize,
    broken: &dyn Fn(),
) {
    while let Ok(line) = replies.recv() {
        if writeln!(out, "{line}").and_then(|()| out.flush()).is_err() {
            broken();
            return;
        }
        unwritten.fetch_sub(1, Ordering::SeqCst);
    }
}

pub fn tool_spec() -> Value {
    json!({
        "name": TOOL_NAME,
        "description": "Ask the user one or more questions by voice. The Starling desktop \
                        app shows the questions and records the user's spoken answer; the \
                        finalized transcript is returned as plain text. A cancelled, \
                        declined, or timed-out ask returns isError with a \"No answer\" \
                        message. The microphone starts only after the prompt is visible \
                        to the user.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": MAX_QUESTIONS,
                    "description": "The questions to show the user.",
                },
                "timeout_ms": {
                    "type": "integer",
                    "minimum": MIN_TIMEOUT_MS,
                    "maximum": MAX_TIMEOUT_MS,
                    "default": DEFAULT_TIMEOUT_MS,
                    "description": "The ask's budget, from the prompt appearing until \
                                    the microphone closes; a captured answer is never \
                                    discarded on the clock.",
                },
            },
            "required": ["questions"],
        },
    })
}

fn parse_call(params: &Value) -> Result<(Vec<String>, u64), String> {
    #[derive(serde::Deserialize)]
    struct Arguments {
        questions: Vec<String>,
        timeout_ms: Option<u64>,
    }
    let name = params.get("name").and_then(Value::as_str).unwrap_or("");
    if name != TOOL_NAME {
        return Err(format!(
            "unknown tool {name:?}; this server exposes only {TOOL_NAME:?}"
        ));
    }
    let arguments = params.get("arguments").cloned().unwrap_or(json!({}));
    let Arguments {
        questions,
        timeout_ms,
    } = serde_json::from_value(arguments).map_err(|err| format!("invalid arguments: {err}"))?;
    let timeout_ms = timeout_ms.unwrap_or(DEFAULT_TIMEOUT_MS);
    validate_ask(&questions, timeout_ms).map_err(|(_, message)| message)?;
    Ok((questions, timeout_ms))
}

fn tool_text(text: String, is_error: bool) -> Value {
    let mut result = json!({"content": [{"type": "text", "text": text}]});
    if is_error {
        result["isError"] = Value::Bool(true);
    }
    result
}

fn no_answer_text(reason: NoAnswerReason) -> &'static str {
    match reason {
        NoAnswerReason::Timeout => "the ask timed out",
        NoAnswerReason::AgentCancelled => "the agent cancelled the ask",
        NoAnswerReason::UserCancelled => "the user cancelled the ask",
        NoAnswerReason::Declined => "the user declined the prompt",
    }
}

/// The next line, read with at most [`MAX_LINE_BYTES`] buffered. `None`
/// at EOF or on a read error (stdin is gone either way); an over-long
/// line is skipped and reported.
fn read_line(input: &mut impl BufRead) -> Option<Result<String, String>> {
    let mut bytes = Vec::new();
    let limit = MAX_LINE_BYTES as u64 + 1;
    match Read::take(&mut *input, limit).read_until(b'\n', &mut bytes) {
        Ok(0) | Err(_) => None,
        Ok(_) if bytes.len() > MAX_LINE_BYTES && bytes.last() != Some(&b'\n') => {
            skip_rest_of_line(input);
            Some(Err(format!("line exceeds {MAX_LINE_BYTES} bytes")))
        }
        Ok(_) => {
            let line = String::from_utf8_lossy(&bytes);
            Some(Ok(line.trim_end_matches(['\n', '\r']).to_string()))
        }
    }
}

fn skip_rest_of_line(input: &mut impl BufRead) {
    loop {
        let (used, done) = match input.fill_buf() {
            Ok([]) | Err(_) => return,
            Ok(buf) => match buf.iter().position(|&b| b == b'\n') {
                Some(newline) => (newline + 1, true),
                None => (buf.len(), false),
            },
        };
        input.consume(used);
        if done {
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Default)]
    struct Output(Arc<Mutex<Vec<u8>>>);

    impl Write for Output {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl Output {
        /// Every reply written since the last call, once the writer
        /// thread has written at least `expected` and then gone quiet.
        fn replies(&self, expected: usize) -> Vec<Value> {
            let lines = || {
                self.0
                    .lock()
                    .unwrap()
                    .iter()
                    .filter(|&&b| b == b'\n')
                    .count()
            };
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
            while lines() < expected && std::time::Instant::now() < deadline {
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
            let bytes = std::mem::take(&mut *self.0.lock().unwrap());
            String::from_utf8(bytes)
                .unwrap()
                .lines()
                .map(|line| serde_json::from_str(line).expect("one JSON reply per line"))
                .collect()
        }
    }

    #[derive(Default)]
    struct FakeSink {
        asks: Mutex<Vec<(String, Vec<String>, u64)>>,
        cancels: Mutex<Vec<String>>,
    }

    impl AskSink for FakeSink {
        fn ask(&self, req: &str, questions: Vec<String>, timeout_ms: u64) -> Result<(), String> {
            self.asks
                .lock()
                .unwrap()
                .push((req.to_string(), questions, timeout_ms));
            Ok(())
        }

        fn cancel(&self, req: &str, _reason: &str) {
            self.cancels.lock().unwrap().push(req.to_string());
        }
    }

    fn server() -> (McpServer, Output) {
        let output = Output::default();
        (McpServer::new(output.clone(), || {}), output)
    }

    fn drive(server: &McpServer, sink: &dyn AskSink, lines: &[String]) {
        server.serve_read(lines.join("\n").as_bytes(), sink);
    }

    fn request(id: impl Into<Value>, method: &str, params: Value) -> String {
        json!({"jsonrpc": "2.0", "id": id.into(), "method": method, "params": params}).to_string()
    }

    fn notification(method: &str, params: Value) -> String {
        json!({"jsonrpc": "2.0", "method": method, "params": params}).to_string()
    }

    fn initialize() -> String {
        request(0, "initialize", json!({"protocolVersion": "2025-06-18"}))
    }

    fn call(id: impl Into<Value>, arguments: Value) -> String {
        request(
            id,
            "tools/call",
            json!({"name": TOOL_NAME, "arguments": arguments}),
        )
    }

    #[test]
    fn initialize_handshake_and_tools_list_shape() {
        let (server, output) = server();
        drive(
            &server,
            &FakeSink::default(),
            &[
                initialize(),
                notification("notifications/initialized", json!({})),
                request(2, "tools/list", json!({})),
            ],
        );
        let replies = output.replies(2);
        assert_eq!(
            replies.len(),
            2,
            "notifications are never answered: {replies:?}"
        );
        assert_eq!(replies[0]["result"]["protocolVersion"], "2025-06-18");
        assert!(replies[0]["result"]["capabilities"]["tools"].is_object());

        let tools = replies[1]["result"]["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["name"], TOOL_NAME);
        assert_eq!(tools[0]["inputSchema"]["required"], json!(["questions"]));
    }

    #[test]
    fn initialize_negotiates_known_versions_and_falls_back() {
        for (requested, expected) in [
            ("2024-11-05", "2024-11-05"),
            ("1999-01-01", PROTOCOL_VERSION),
        ] {
            let (server, output) = server();
            let init = request(7, "initialize", json!({"protocolVersion": requested}));
            drive(&server, &FakeSink::default(), &[init]);
            assert_eq!(output.replies(1)[0]["result"]["protocolVersion"], expected);
        }
    }

    #[test]
    fn unknown_method_and_pre_init_requests_are_refused() {
        let (server, output) = server();
        drive(
            &server,
            &FakeSink::default(),
            &[
                request(1, "tools/list", json!({})),
                initialize(),
                request(2, "resources/list", json!({})),
            ],
        );
        let codes: Vec<Value> = output
            .replies(3)
            .iter()
            .map(|r| r["error"]["code"].clone())
            .collect();
        assert_eq!(
            codes,
            [json!(SERVER_NOT_INITIALIZED), Value::Null, json!(-32601)]
        );
    }

    #[test]
    fn parse_errors_batches_and_overlong_lines_are_refused() {
        let (server, output) = server();
        let overlong = "x".repeat(MAX_LINE_BYTES + 10);
        drive(
            &server,
            &FakeSink::default(),
            &["not json".into(), "[[1,2]]".into(), overlong, initialize()],
        );
        let replies = output.replies(4);
        assert_eq!(replies.len(), 4, "{replies:?}");
        assert_eq!(replies[0]["error"]["code"], -32700);
        assert_eq!(replies[1]["error"]["code"], -32600);
        assert_eq!(replies[2]["error"]["code"], -32700);
        assert!(
            replies[3]["result"].is_object(),
            "the line after an over-long one is still served"
        );
    }

    #[test]
    fn tools_call_validates_arguments() {
        let (server, output) = server();
        let sink = FakeSink::default();
        let wrong_tool = request(1, "tools/call", json!({"name": "other", "arguments": {}}));
        let mut lines = vec![initialize(), wrong_tool];
        for arguments in [
            json!({}),
            json!({"questions": []}),
            json!({"questions": [" "]}),
            json!({"questions": [1]}),
            json!({"questions": ["ok"], "timeout_ms": 10}),
            json!({"questions": ["ok"], "timeout_ms": 9_000_000}),
        ] {
            lines.push(call(2, arguments));
        }
        drive(&server, &sink, &lines);
        let replies = output.replies(lines.len());
        assert_eq!(replies.len(), lines.len());
        for reply in &replies[1..] {
            assert_eq!(reply["error"]["code"], -32602, "{reply}");
        }
        assert!(sink.asks.lock().unwrap().is_empty());
    }

    #[test]
    fn outcomes_answer_the_pending_call() {
        let (server, output) = server();
        let sink = FakeSink::default();
        drive(
            &server,
            &sink,
            &[
                initialize(),
                call("a", json!({"questions": ["Which fix?"]})),
                call("b", json!({"questions": ["Ready?"], "timeout_ms": 1_000})),
            ],
        );
        assert_eq!(output.replies(1).len(), 1, "calls wait for the host");
        let asks = sink.asks.lock().unwrap().clone();
        assert_eq!(asks[0].1, ["Which fix?"]);
        assert_eq!(asks[0].2, DEFAULT_TIMEOUT_MS);
        assert_eq!(asks[1].2, 1_000);

        server.complete(
            &asks[0].0,
            AskOutcome::Answered {
                text: "Take the left branch.".to_string(),
                backend: "engine:fixture".to_string(),
            },
        );
        server.complete(
            &asks[1].0,
            AskOutcome::NoAnswer {
                reason: NoAnswerReason::Timeout,
            },
        );
        // A second outcome for the same req (already answered) drops.
        server.complete(
            &asks[1].0,
            AskOutcome::NoAnswer {
                reason: NoAnswerReason::Timeout,
            },
        );
        let replies = output.replies(2);
        assert_eq!(replies.len(), 2, "a req is answered once");
        assert_eq!(replies[0]["id"], "a");
        assert_eq!(
            replies[0]["result"]["content"][0]["text"],
            "Take the left branch."
        );
        assert!(replies[0]["result"].get("isError").is_none());
        assert_eq!(replies[1]["id"], "b");
        assert_eq!(replies[1]["result"]["isError"], true);
        let text = replies[1]["result"]["content"][0]["text"].as_str().unwrap();
        assert!(text.starts_with("No answer:"), "{text}");
    }

    #[test]
    fn an_outcome_arriving_before_ask_returns_is_not_lost() {
        // The host can refuse an ask (no app attached) before the
        // sending thread gets to record it.
        struct InstantRefusal(Mutex<Option<McpServer>>);
        impl AskSink for InstantRefusal {
            fn ask(&self, req: &str, _: Vec<String>, _: u64) -> Result<(), String> {
                let server = self.0.lock().unwrap().clone().unwrap();
                let outcome = AskOutcome::Error {
                    code: "no_prompt_ack".to_string(),
                    message: "no app".to_string(),
                };
                server.complete(req, outcome);
                Ok(())
            }
            fn cancel(&self, _: &str, _: &str) {}
        }
        let (server, output) = server();
        let sink = InstantRefusal(Mutex::new(Some(server.clone())));
        drive(
            &server,
            &sink,
            &[initialize(), call(1, json!({"questions": ["Q?"]}))],
        );
        let replies = output.replies(2);
        assert_eq!(replies.len(), 2, "{replies:?}");
        assert_eq!(replies[1]["result"]["isError"], true);
    }

    #[test]
    fn a_stalled_stdout_ends_the_session_instead_of_blocking_stdin() {
        struct Stalled;
        impl Write for Stalled {
            fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
                loop {
                    std::thread::park();
                }
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let broken = Arc::new(AtomicBool::new(false));
        let server = {
            let broken = Arc::clone(&broken);
            McpServer::new(Stalled, move || broken.store(true, Ordering::SeqCst))
        };
        let mut lines = vec![initialize()];
        lines.extend((1..=OUT_QUEUE as u64 + 10).map(|id| request(id, "ping", json!({}))));
        // Returning at all shows stdin was never blocked on stdout.
        drive(&server, &FakeSink::default(), &lines);
        assert!(broken.load(Ordering::SeqCst));
    }

    #[test]
    fn cancelled_notification_cancels_the_ask() {
        let (server, output) = server();
        let sink = FakeSink::default();
        drive(
            &server,
            &sink,
            &[
                initialize(),
                call(42, json!({"questions": ["Ready?"]})),
                notification("notifications/cancelled", json!({"requestId": 41})),
                notification("notifications/cancelled", json!({"requestId": 42})),
            ],
        );
        assert_eq!(output.replies(1).len(), 1, "only initialize is answered");
        let asked = sink.asks.lock().unwrap()[0].0.clone();
        assert_eq!(*sink.cancels.lock().unwrap(), [asked]);
    }
}
