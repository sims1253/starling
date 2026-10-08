//! The MCP stdio server for `ask_user_dictation` (issue #309) — the
//! process a coding agent launches and speaks JSON-RPC 2.0 to over
//! stdin/stdout, bridging onto the host's ask surface ([`crate::agent`]).
//!
//! Hand-rolled on purpose (the repo's dependency policy: zero new
//! crates): the MCP stdio transport is newline-delimited JSON-RPC 2.0,
//! which `serde_json` already covers. This module implements exactly
//! the minimal surface a coding agent needs:
//!
//! - `initialize` → protocol version + capabilities (`tools`) +
//!   serverInfo; `notifications/initialized` ignored;
//! - `tools/list` → the one tool this server exposes;
//! - `tools/call` → one ask (validated here, executed by the host
//!   broker; the answer arrives when the user has spoken);
//! - `notifications/cancelled` → the ask is cancelled (the host stops
//!   the microphone);
//! - unknown methods → `-32601`; anything before `initialize` →
//!   `-32002`; batches are refused (MCP has no batching);
//! - clean shutdown on stdin EOF: pending asks are cancelled — the
//!   deliberate disconnect rule (the host cancels them anyway when
//!   this process's connection dies; cancelling first keeps the
//!   take-abort prompt even on a graceful agent exit).
//!
//! HTTP transport is deliberately absent: the issue names Claude
//! Code's HTTP tool timeout as too short for a human speaking — stdio
//! only.
//!
//! # Result mapping (documented choice)
//!
//! An MCP tool result has two honest shapes: a result with content, or
//! a result with `isError: true` (a *tool* failure the model sees —
//! distinct from a JSON-RPC protocol error). This server maps:
//!
//! - **Answered** → `content[0].text` is exactly the transcript, no
//!   wrapper, `isError` absent. An agent can use the text as the
//!   answer verbatim.
//! - **No answer** (cancel/timeout/decline/user-cancel — a *completed*
//!   ask with deliberately no transcript) → `isError: true` with text
//!   `"No answer: <reason>."`. isError is the unambiguous choice: a
//!   plain content string reading "no answer" could be mistaken for a
//!   spoken answer by a model that does not read carefully; an error
//!   result cannot.
//! - **Error** (refused/failed asks) → `isError: true` with text
//!   `"Error [<code>]: <message>."`.
//! - Malformed `tools/call` arguments (wrong tool name, empty or
//!   oversized questions, out-of-bounds timeout) → JSON-RPC `-32602`:
//!   parameter misuse is a protocol-level failure before the host ever
//!   sees an ask.
//!
//! Concurrency: `tools/call` requests may overlap (JSON-RPC allows it;
//! an agent can ask two things at once). The replies come whenever the
//! host resolves each ask — serialized prompts, queued by the host —
//! so this server keeps a pending map, never blocks its stdin loop on
//! an answer (it must stay free to read `notifications/cancelled`),
//! and writes every reply from one writer thread so stdout stays
//! strictly line-atomic.

use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::{json, Value};

use starling_runtime::channel::{bounded, Receiver, RecvError, Sender};

use crate::agent::{MAX_QUESTION_CHARS, MAX_QUESTIONS, MAX_TIMEOUT_MS, MIN_TIMEOUT_MS};
use crate::frame::{AskOutcome, NoAnswerReason};

/// The MCP protocol revision this server speaks and advertises when
/// the client does not name one it knows.
pub const PROTOCOL_VERSION: &str = "2025-06-18";

/// Protocol revisions this server can honor as requested. A client
/// asking for something else gets [`PROTOCOL_VERSION`] back (the
/// spec's fallback).
const KNOWN_VERSIONS: [&str; 4] = ["2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25"];

/// The default `timeout_ms` when a caller omits it: two minutes covers
/// a prompt appearing, a human reading it, and a spoken answer.
pub const DEFAULT_TIMEOUT_MS: u64 = 120_000;

/// The longest stdin line accepted (the largest legitimate message is
/// a `tools/call` with a few bounded questions — kilobytes). Longer
/// lines are refused as parse errors, never buffered unboundedly.
const MAX_LINE_BYTES: usize = 1024 * 1024;

/// JSON-RPC error codes this server emits (besides the standard
/// -32700/-32600/-32601/-32602/-32603).
const SERVER_NOT_INITIALIZED: i64 = -32002;

/// The one tool this server exposes. The name is the issue's contract;
/// the schema is the host broker's bounds (see [`crate::agent`]).
pub const TOOL_NAME: &str = "ask_user_dictation";

/// What the agent-side caller provides: hand the ask to the host (the
/// binary's implementation rides a [`crate::client::HostClient`];
/// tests ride an in-memory double).
pub trait AskSink: Send + Sync {
    /// Queues the ask under `req`. `Err` answers the caller with a
    /// JSON-RPC internal error carrying the message.
    fn ask(&self, req: &str, questions: Vec<String>, timeout_ms: u64) -> Result<(), String>;
    /// Cancels a queued or in-flight ask (best-effort).
    fn cancel(&self, req: &str, reason: &str);
}

/// A running MCP server over one agent session. Clone the handle for
/// the thread that receives host-side ask results; run
/// [`McpServer::serve_read`] on the thread that owns stdin.
#[derive(Clone)]
pub struct McpServer {
    inner: Arc<Shared>,
}

struct Shared {
    next_req: AtomicU64,
    initialized: AtomicBool,
    /// req → the JSON-RPC id waiting on it.
    pending: Mutex<HashMap<String, Value>>,
    out: Mutex<Option<Sender<String>>>,
}

impl McpServer {
    pub fn new() -> McpServer {
        McpServer {
            inner: Arc::new(Shared {
                next_req: AtomicU64::new(0),
                initialized: AtomicBool::new(false),
                pending: Mutex::new(HashMap::new()),
                out: Mutex::new(None),
            }),
        }
    }

    /// Spawns the single writer thread: every reply leaves through it,
    /// one JSON document per line, so concurrent completions can never
    /// interleave bytes on stdout.
    pub fn spawn_writer(&self, sink: impl Write + Send + 'static) -> std::thread::JoinHandle<()> {
        let (tx, rx) = bounded::<String>(256);
        *self.inner.out.lock().expect("mcp writer lock") = Some(tx);
        std::thread::Builder::new()
            .name("starling-mcp-write".to_string())
            .spawn(move || writer_loop(sink, rx))
            .expect("mcp writer spawn")
    }

    /// Drops the writer's sender so the writer thread drains what is
    /// queued and exits — the shutdown path (every reply already
    /// queued still reaches stdout; later writes drop with nobody to
    /// serve them).
    pub fn shutdown_writer(&self) {
        self.inner.out.lock().expect("mcp writer lock").take();
    }

    /// Reads stdin until EOF, answering everything. Returns on EOF (or
    /// a stdin error — the agent is gone either way); the caller then
    /// cancels what is pending ([`McpServer::cancel_pending`]).
    pub fn serve_read(&self, input: impl BufRead, sink: &dyn AskSink) {
        for line in LineReader::new(input) {
            let line = match line {
                Ok(line) => line,
                Err(err) => {
                    // Over-long or unreadable line: one honest parse
                    // error, then keep reading (a notification cannot
                    // be answered; the stream stays usable).
                    self.write_message(json!({
                        "jsonrpc": "2.0",
                        "id": Value::Null,
                        "error": {"code": -32700, "message": err},
                    }));
                    continue;
                }
            };
            let message: Value = match serde_json::from_str(&line) {
                Ok(value) => value,
                Err(err) => {
                    self.write_message(json!({
                        "jsonrpc": "2.0",
                        "id": Value::Null,
                        "error": {
                            "code": -32700,
                            "message": format!("parse error: {err}"),
                        },
                    }));
                    continue;
                }
            };
            self.handle(message, sink);
        }
    }

    /// Delivers one ask outcome to the JSON-RPC caller waiting on it
    /// (the drain loop of the host connection calls this). Unknown reqs
    /// (already cancelled and answered, or a late duplicate) drop.
    pub fn complete(&self, req: &str, outcome: AskOutcome) {
        let Some(id) = self.take_pending(req) else {
            return;
        };
        let result = match outcome {
            AskOutcome::Answered { text, .. } => json!({
                "content": [{"type": "text", "text": text}],
            }),
            AskOutcome::NoAnswer { reason } => json!({
                "content": [{
                    "type": "text",
                    "text": format!("No answer: {}.", no_answer_text(reason)),
                }],
                "isError": true,
            }),
            AskOutcome::Error { code, message } => json!({
                "content": [{
                    "type": "text",
                    "text": format!("Error [{code}]: {message}."),
                }],
                "isError": true,
            }),
        };
        self.write_message(json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": result,
        }));
    }

    /// Fails every pending `tools/call` with an `isError` tool result
    /// (the host connection died mid-session; the tool ran, it just
    /// cannot finish).
    pub fn fail_pending(&self, message: &str) {
        let pending: Vec<(String, Value)> = self
            .inner
            .pending
            .lock()
            .expect("mcp pending lock")
            .drain()
            .collect();
        for (_, id) in pending {
            self.write_message(json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": {
                    "content": [{
                        "type": "text",
                        "text": format!("Error [host_disconnected]: {message}."),
                    }],
                    "isError": true,
                },
            }));
        }
    }

    /// Cancels everything pending through the sink (the stdin-EOF
    /// path; the answers still arrive and are written — nobody reads
    /// them, but stdout stays honest).
    pub fn cancel_pending(&self, sink: &dyn AskSink, reason: &str) {
        let pending: Vec<String> = self
            .inner
            .pending
            .lock()
            .expect("mcp pending lock")
            .keys()
            .cloned()
            .collect();
        for req in pending {
            sink.cancel(&req, reason);
        }
    }

    fn take_pending(&self, req: &str) -> Option<Value> {
        self.inner
            .pending
            .lock()
            .expect("mcp pending lock")
            .remove(req)
    }

    fn handle(&self, message: Value, sink: &dyn AskSink) {
        // Batching is refused (MCP has no batching; a JSON-RPC batch
        // is therefore a client that is not speaking MCP).
        let Some(object) = message.as_object() else {
            self.error_response(Value::Null, -32600, "request must be a JSON object");
            return;
        };
        if object.contains_key("jsonrpc") && object["jsonrpc"] != "2.0" {
            self.error_response(Value::Null, -32600, "jsonrpc must be exactly \"2.0\"");
            return;
        }
        let Some(method) = object.get("method").and_then(Value::as_str) else {
            self.error_response(
                object.get("id").cloned().unwrap_or(Value::Null),
                -32600,
                "a request must carry a string method",
            );
            return;
        };
        let id = object.get("id").cloned();
        let params = object.get("params").cloned().unwrap_or(Value::Null);
        // A message with an id (any non-null JSON-RPC id) is a request;
        // without one it is a notification and is never answered.
        let is_request = id.as_ref().is_some_and(|v| !v.is_null());
        // Per the MCP spec, nothing but `initialize` is served before
        // the handshake completed.
        if is_request
            && method != "initialize"
            && !self.inner.initialized.load(Ordering::SeqCst)
        {
            self.error_response(
                id.unwrap_or(Value::Null),
                SERVER_NOT_INITIALIZED,
                "server not initialized: send initialize first",
            );
            return;
        }
        match (method, is_request) {
            ("initialize", true) => self.handle_initialize(id.unwrap_or(Value::Null), params),
            // The one notification this server acts on (it is still
            // never answered).
            ("notifications/cancelled", false) => self.handle_cancelled(params, sink),
            ("tools/list", true) => {
                let id = id.unwrap_or(Value::Null);
                self.write_message(json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": {"tools": [tool_spec()]},
                }));
            }
            ("tools/call", true) => {
                self.handle_tools_call(id.unwrap_or(Value::Null), params, sink);
            }
            ("ping", true) => {
                let id = id.unwrap_or(Value::Null);
                self.write_message(json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": {},
                }));
            }
            (_, true) => self.error_response(
                id.unwrap_or(Value::Null),
                -32601,
                &format!("method not found: {method}"),
            ),
            // Notifications are never answered — not even unknown ones
            // (notifications/initialized included).
            (_, false) => {}
        }
    }

    fn handle_initialize(&self, id: Value, params: Value) {
        let requested = params
            .get("protocolVersion")
            .and_then(Value::as_str)
            .unwrap_or(PROTOCOL_VERSION);
        let negotiated = if KNOWN_VERSIONS.contains(&requested) {
            requested
        } else {
            PROTOCOL_VERSION
        };
        self.inner.initialized.store(true, Ordering::SeqCst);
        self.write_message(json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": {
                "protocolVersion": negotiated,
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
            },
        }));
    }

    fn handle_tools_call(&self, id: Value, params: Value, sink: &dyn AskSink) {
        let name = params.get("name").and_then(Value::as_str).unwrap_or("");
        if name != TOOL_NAME {
            self.error_response(
                id,
                -32602,
                &format!("unknown tool {name:?}; this server exposes only {TOOL_NAME:?}"),
            );
            return;
        }
        let arguments = params.get("arguments").cloned().unwrap_or(json!({}));
        let questions = match parse_questions(&arguments) {
            Ok(questions) => questions,
            Err(detail) => {
                self.error_response(id, -32602, &detail);
                return;
            }
        };
        let timeout_ms = match parse_timeout(&arguments) {
            Ok(timeout_ms) => timeout_ms,
            Err(detail) => {
                self.error_response(id, -32602, &detail);
                return;
            }
        };
        let req = format!("ask{}", self.inner.next_req.fetch_add(1, Ordering::SeqCst));
        match sink.ask(&req, questions, timeout_ms) {
            Ok(()) => {
                self.inner
                    .pending
                    .lock()
                    .expect("mcp pending lock")
                    .insert(req, id);
            }
            Err(message) => {
                self.error_response(
                    id,
                    -32603,
                    &format!("the ask could not be sent: {message}"),
                );
            }
        }
    }

    /// Handles a `notifications/cancelled`: the agent gave up on a
    /// request; cancel the matching ask. The completion still arrives
    /// (as a no-answer) and answers the pending request — MCP permits
    /// responding to a cancelled request.
    fn handle_cancelled(&self, params: Value, sink: &dyn AskSink) {
        let Some(request_id) = params.get("requestId") else {
            return;
        };
        let reason = params
            .get("reason")
            .and_then(Value::as_str)
            .unwrap_or("cancelled by the agent");
        let pending = self.inner.pending.lock().expect("mcp pending lock");
        let found = pending.iter().find(|(_, id)| **id == *request_id);
        if let Some((req, _)) = found {
            let req = req.clone();
            drop(pending);
            sink.cancel(&req, reason);
        }
    }

    fn error_response(&self, id: Value, code: i64, message: &str) {
        self.write_message(json!({
            "jsonrpc": "2.0",
            "id": id,
            "error": {"code": code, "message": message},
        }));
    }

    fn write_message(&self, message: Value) {
        // Never block the protocol loops on a wedged consumer: a full
        // writer queue means the agent stopped reading stdout —
        // nothing this server can say will reach it, and dropping the
        // line is the honest move.
        if let Some(out) = self.inner.out.lock().expect("mcp writer lock").as_ref() {
            let _ = out.try_send(serde_json::to_string(&message).unwrap_or_default());
        }
    }
}

fn writer_loop(mut sink: impl Write, outbound: Receiver<String>) {
    loop {
        match outbound.recv_timeout(Duration::from_millis(250)) {
            Ok(line) => {
                // One JSON document per line, atomic by construction:
                // the write happens under the writer's own buffering,
                // flushed per line.
                if sink.write_all(line.as_bytes()).is_err()
                    || sink.write_all(b"\n").is_err()
                    || sink.flush().is_err()
                {
                    return;
                }
            }
            Err(RecvError::Timeout) => continue,
            // Every handle dropped: shutdown.
            Err(RecvError::Closed) => {
                let _ = sink.flush();
                return;
            }
        }
    }
}

/// The `ask_user_dictation` tool spec (its bounds mirror the host
/// broker's, which re-validates — defense on both sides of the IPC).
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

fn parse_questions(arguments: &Value) -> Result<Vec<String>, String> {
    let questions = arguments
        .get("questions")
        .ok_or_else(|| "arguments.questions is required".to_string())?;
    let Some(items) = questions.as_array() else {
        return Err("arguments.questions must be an array of strings".to_string());
    };
    if items.is_empty() || items.len() > MAX_QUESTIONS {
        return Err(format!(
            "arguments.questions must carry 1..={MAX_QUESTIONS} entries"
        ));
    }
    let mut parsed = Vec::with_capacity(items.len());
    for item in items {
        let Some(text) = item.as_str() else {
            return Err("arguments.questions must be an array of strings".to_string());
        };
        if text.trim().is_empty() {
            return Err("arguments.questions must not contain empty questions".to_string());
        }
        if text.chars().count() > MAX_QUESTION_CHARS {
            return Err(format!(
                "each question is limited to {MAX_QUESTION_CHARS} characters"
            ));
        }
        parsed.push(text.to_string());
    }
    Ok(parsed)
}

fn parse_timeout(arguments: &Value) -> Result<u64, String> {
    match arguments.get("timeout_ms") {
        None | Some(Value::Null) => Ok(DEFAULT_TIMEOUT_MS),
        Some(value) => {
            let timeout = value
                .as_u64()
                .ok_or_else(|| "arguments.timeout_ms must be an integer".to_string())?;
            if !(MIN_TIMEOUT_MS..=MAX_TIMEOUT_MS).contains(&timeout) {
                return Err(format!(
                    "arguments.timeout_ms must be {MIN_TIMEOUT_MS}..={MAX_TIMEOUT_MS}"
                ));
            }
            Ok(timeout)
        }
    }
}

fn no_answer_text(reason: NoAnswerReason) -> &'static str {
    match reason {
        NoAnswerReason::Timeout => "the ask timed out",
        NoAnswerReason::AgentCancelled => "the agent cancelled the ask",
        NoAnswerReason::UserCancelled => "the user cancelled the ask",
        NoAnswerReason::Declined => "the user declined the prompt",
    }
}

/// Line iterator with a hard length bound (stdin lines can be as long
/// as the peer cares to write; the protocol's are not).
struct LineReader<R: BufRead> {
    inner: R,
}

impl<R: BufRead> LineReader<R> {
    fn new(inner: R) -> LineReader<R> {
        LineReader { inner }
    }

    fn drain_rest_of_line(&mut self) {
        loop {
            let mut more = Vec::new();
            match self.inner.read_until(b'\n', &mut more) {
                Ok(0) => return,
                Ok(_) if more.last() == Some(&b'\n') => return,
                Ok(_) => continue,
                Err(_) => return,
            }
        }
    }
}

impl<R: BufRead> Iterator for LineReader<R> {
    type Item = Result<String, String>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut bytes = Vec::new();
        match self.inner.read_until(b'\n', &mut bytes) {
            Ok(0) => None, // EOF
            Ok(_) => {
                if bytes.len() > MAX_LINE_BYTES {
                    self.drain_rest_of_line();
                    return Some(Err(format!(
                        "stdin line exceeded {MAX_LINE_BYTES} bytes"
                    )));
                }
                let line = String::from_utf8_lossy(&bytes);
                Some(Ok(line.trim_end_matches(['\n', '\r']).to_string()))
            }
            Err(err) => Some(Err(format!("stdin read failed: {err}"))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::BufReader;

    /// An in-memory ask sink: records asks, lets the test complete
    /// them on its own schedule.
    struct FakeSink {
        asks: Mutex<Vec<(String, Vec<String>, u64)>>,
        cancels: Mutex<Vec<String>>,
    }

    impl FakeSink {
        fn new() -> Arc<FakeSink> {
            Arc::new(FakeSink {
                asks: Mutex::new(Vec::new()),
                cancels: Mutex::new(Vec::new()),
            })
        }
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

    fn drive(server: &McpServer, sink: &dyn AskSink, input: &str) -> Vec<Value> {
        let (tx, rx) = bounded::<String>(256);
        *server.inner.out.lock().unwrap() = Some(tx);
        server.serve_read(BufReader::new(input.as_bytes()), sink);
        drop(server.inner.out.lock().unwrap().take());
        let mut replies = Vec::new();
        while let Ok(line) = rx.try_recv() {
            replies.push(serde_json::from_str(&line).expect("one JSON reply per line"));
        }
        replies
    }

    fn request(id: impl Into<Value>, method: &str, params: Value) -> String {
        serde_json::to_string(&json!({
            "jsonrpc": "2.0",
            "id": id.into(),
            "method": method,
            "params": params,
        }))
        .unwrap()
    }

    fn notification(method: &str, params: Value) -> String {
        serde_json::to_string(&json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }))
        .unwrap()
    }

    #[test]
    fn initialize_handshake_and_tools_list_shape() {
        let server = McpServer::new();
        let sink = FakeSink::new();
        let input = [
            request(1, "initialize", json!({
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "claude-code", "version": "x"},
            })),
            notification("notifications/initialized", json!({})),
            request(2, "tools/list", json!({})),
        ]
        .join("\n");
        let replies = drive(&server, sink.as_ref(), &input);
        assert_eq!(replies.len(), 2, "initialized is never answered: {replies:?}");

        let init = &replies[0];
        assert_eq!(init["id"], 1);
        assert_eq!(init["result"]["protocolVersion"], "2025-06-18");
        assert!(init["result"]["capabilities"]["tools"].is_object());
        assert_eq!(init["result"]["serverInfo"]["name"], "starling");

        let list = &replies[1];
        let tools = list["result"]["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1, "exactly one tool is exposed");
        assert_eq!(tools[0]["name"], "ask_user_dictation");
        let schema = &tools[0]["inputSchema"];
        assert_eq!(schema["properties"]["questions"]["maxItems"], MAX_QUESTIONS);
        assert_eq!(
            schema["properties"]["questions"]["minItems"],
            1
        );
        assert_eq!(
            schema["required"],
            json!(["questions"])
        );
    }

    #[test]
    fn initialize_negotiates_known_versions_and_falls_back() {
        let server = McpServer::new();
        let sink = FakeSink::new();
        let input = format!(
            "{}\n",
            request(
                7,
                "initialize",
                json!({"protocolVersion": "2024-11-05", "capabilities": {}}),
            )
        );
        let replies = drive(&server, sink.as_ref(), &input);
        assert_eq!(replies[0]["result"]["protocolVersion"], "2024-11-05");

        let server = McpServer::new();
        let input = format!(
            "{}\n",
            request(
                7,
                "initialize",
                json!({"protocolVersion": "1999-01-01", "capabilities": {}}),
            )
        );
        let replies = drive(&server, sink.as_ref(), &input);
        assert_eq!(replies[0]["result"]["protocolVersion"], PROTOCOL_VERSION);
    }

    #[test]
    fn unknown_method_and_pre_init_requests_are_refused() {
        let server = McpServer::new();
        let sink = FakeSink::new();
        let input = [
            request(1, "tools/list", json!({})), // before initialize
            request(2, "resources/list", json!({})),
            request(
                3,
                "initialize",
                json!({"protocolVersion": "2025-06-18", "capabilities": {}}),
            ),
            request(4, "resources/list", json!({})),
        ]
        .join("\n");
        let replies = drive(&server, sink.as_ref(), &input);
        assert_eq!(replies.len(), 4);
        assert_eq!(replies[0]["error"]["code"], SERVER_NOT_INITIALIZED);
        assert_eq!(replies[1]["error"]["code"], SERVER_NOT_INITIALIZED);
        assert!(replies[2]["result"].is_object());
        // Unknown methods after init are method-not-found, not the
        // pre-init code.
        assert_eq!(replies[3]["error"]["code"], -32601);
    }

    #[test]
    fn parse_errors_and_batches_are_refused() {
        let server = McpServer::new();
        let sink = FakeSink::new();
        let replies = drive(&server, sink.as_ref(), "not json\n[[1,2]]\n");
        assert_eq!(replies.len(), 2);
        assert_eq!(replies[0]["error"]["code"], -32700);
        assert_eq!(replies[1]["error"]["code"], -32600);
    }

    #[test]
    fn tools_call_validates_arguments() {
        let server = McpServer::new();
        let sink = FakeSink::new();
        let init = request(
            0,
            "initialize",
            json!({"protocolVersion": "2025-06-18", "capabilities": {}}),
        );
        let cases = [
            (json!({"name": "other_tool", "arguments": {}}), "wrong tool"),
            (json!({"name": TOOL_NAME, "arguments": {}}), "missing questions"),
            (
                json!({"name": TOOL_NAME, "arguments": {"questions": []}}),
                "empty questions",
            ),
            (
                json!({"name": TOOL_NAME, "arguments": {"questions": ["".to_string()]}}),
                "blank question",
            ),
            (
                json!({"name": TOOL_NAME, "arguments": {"questions": ["ok"], "timeout_ms": 10}}),
                "timeout below floor",
            ),
            (
                json!({"name": TOOL_NAME, "arguments": {"questions": ["ok"], "timeout_ms": 9_000_000}}),
                "timeout above ceiling",
            ),
        ];
        for (index, (arguments, label)) in cases.iter().enumerate() {
            let input = format!(
                "{init}\n{}\n",
                request(index + 1, "tools/call", json!({"name": arguments["name"], "arguments": arguments["arguments"]}))
            );
            let replies = drive(&server, sink.as_ref(), &input);
            assert_eq!(
                replies.last().unwrap()["error"]["code"],
                -32602,
                "case {label}"
            );
        }
        assert!(
            sink.asks.lock().unwrap().is_empty(),
            "no invalid ask reached the sink"
        );
    }

    #[test]
    fn tools_call_answers_through_the_pending_map() {
        let server = McpServer::new();
        let sink = FakeSink::new();
        let init = request(
            0,
            "initialize",
            json!({"protocolVersion": "2025-06-18", "capabilities": {}}),
        );
        let call = request(
            "call-1",
            "tools/call",
            json!({"name": TOOL_NAME, "arguments": {"questions": ["Which fix?"]}}),
        );
        let replies = drive(&server, sink.as_ref(), &format!("{init}\n{call}\n"));
        // Only the initialize reply: the call stays pending until the
        // host answers (the human has not spoken yet).
        assert_eq!(replies.len(), 1);

        let asks = sink.asks.lock().unwrap().clone();
        assert_eq!(asks.len(), 1);
        assert_eq!(asks[0].1, ["Which fix?"]);
        assert_eq!(asks[0].2, DEFAULT_TIMEOUT_MS);

        // Install the writer channel first: `complete` writes through
        // it, and a completion arriving with no channel installed is
        // dropped (the production writer is always up by then).
        let (tx, rx) = bounded::<String>(4);
        *server.inner.out.lock().unwrap() = Some(tx);
        server.complete(
            &asks[0].0,
            AskOutcome::Answered {
                text: "Take the left branch.".to_string(),
                backend: "engine:fixture".to_string(),
            },
        );
        let line: Value =
            serde_json::from_str(&rx.recv_timeout(Duration::from_secs(2)).expect("the answer was written")).unwrap();
        assert_eq!(line["id"], "call-1");
        assert_eq!(line["result"]["content"][0]["text"], "Take the left branch.");
        assert!(line["result"].get("isError").is_none());
    }

    #[test]
    fn no_answer_and_error_outcomes_are_is_error() {
        let server = McpServer::new();
        let sink = FakeSink::new();
        let init = request(
            0,
            "initialize",
            json!({"protocolVersion": "2025-06-18", "capabilities": {}}),
        );
        let call = request(
            1,
            "tools/call",
            json!({"name": TOOL_NAME, "arguments": {"questions": ["Ready?"], "timeout_ms": 1_000}}),
        );
        drive(&server, sink.as_ref(), &format!("{init}\n{call}\n"));
        let req = sink.asks.lock().unwrap()[0].0.clone();

        let (tx, rx) = bounded::<String>(4);
        *server.inner.out.lock().unwrap() = Some(tx);
        server.complete(
            &req,
            AskOutcome::NoAnswer {
                reason: NoAnswerReason::Timeout,
            },
        );
        let line: Value = serde_json::from_str(
            &rx.recv_timeout(Duration::from_secs(2))
                .expect("the no-answer was written"),
        )
        .unwrap();
        assert_eq!(line["result"]["isError"], true);
        let text = line["result"]["content"][0]["text"].as_str().unwrap();
        assert!(text.starts_with("No answer:"), "{text}");
    }

    #[test]
    fn cancelled_notification_cancels_the_ask() {
        let server = McpServer::new();
        let sink = FakeSink::new();
        let init = request(
            0,
            "initialize",
            json!({"protocolVersion": "2025-06-18", "capabilities": {}}),
        );
        let call = request(
            42,
            "tools/call",
            json!({"name": TOOL_NAME, "arguments": {"questions": ["Ready?"]}}),
        );
        let cancelled = notification(
            "notifications/cancelled",
            json!({"requestId": 42, "reason": "user bailed"}),
        );
        let input = format!("{init}\n{call}\n{cancelled}\n");
        let replies = drive(&server, sink.as_ref(), &input);
        assert_eq!(replies.len(), 1, "only initialize is answered");
        let cancels = sink.cancels.lock().unwrap().clone();
        assert_eq!(cancels.len(), 1, "the ask was cancelled host-side");
    }

    #[test]
    fn eof_cancels_pending_asks() {
        let server = McpServer::new();
        let sink = FakeSink::new();
        let init = request(
            0,
            "initialize",
            json!({"protocolVersion": "2025-06-18", "capabilities": {}}),
        );
        let call = request(
            1,
            "tools/call",
            json!({"name": TOOL_NAME, "arguments": {"questions": ["Ready?"]}}),
        );
        drive(&server, sink.as_ref(), &format!("{init}\n{call}\n"));
        server.cancel_pending(sink.as_ref(), "stdin closed");
        assert_eq!(sink.cancels.lock().unwrap().len(), 1);
    }
}
