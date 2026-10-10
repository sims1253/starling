//! A scripted transcription server for the host's suites: OpenAI-style
//! batch uploads (`POST /v1/audio/transcriptions`) and the native
//! `/stream` WebSocket, each answered from a script. Anything else (a
//! health probe) answers `{"status":"ok"}`.

use std::collections::VecDeque;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use tokio_tungstenite::tungstenite::{self, Message};

/// One batch request's answer.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub enum Reply {
    Text(String),
    Status(&'static str),
    /// Close the connection without an answer.
    HangUp,
    /// Answer `text` only once [`FakeEngine::release`] was called.
    Held(String),
}

/// How `/stream` sessions behave.
#[derive(Clone, Debug)]
pub enum StreamMode {
    /// No WebSocket: the upgrade answers 404.
    Refuse,
    /// One preview (`partial`) after the first audio frame, `final` at
    /// the commit.
    Echo { partial: String, final_text: String },
}

#[derive(Default)]
struct Script {
    replies: VecDeque<Reply>,
    stream: Option<StreamMode>,
    batch_requests: usize,
    stream_sessions: usize,
    stream_audio_frames: usize,
    released: bool,
}

#[derive(Clone)]
pub struct FakeEngine {
    pub addr: SocketAddr,
    script: Arc<(Mutex<Script>, Condvar)>,
}

impl FakeEngine {
    pub fn start(replies: Vec<Reply>, stream: StreamMode) -> FakeEngine {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");
        let script = Arc::new((
            Mutex::new(Script {
                replies: replies.into(),
                stream: Some(stream),
                ..Script::default()
            }),
            Condvar::new(),
        ));
        let engine = FakeEngine { addr, script };
        let serving = engine.clone();
        std::thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                let engine = serving.clone();
                std::thread::spawn(move || engine.serve(stream));
            }
        });
        engine
    }

    pub fn endpoint(&self) -> String {
        format!("http://{}", self.addr)
    }

    pub fn batch_requests(&self) -> usize {
        self.script.0.lock().unwrap().batch_requests
    }

    pub fn stream_sessions(&self) -> usize {
        self.script.0.lock().unwrap().stream_sessions
    }

    pub fn stream_audio_frames(&self) -> usize {
        self.script.0.lock().unwrap().stream_audio_frames
    }

    /// Lets every [`Reply::Held`] answer.
    pub fn release(&self) {
        self.script.0.lock().unwrap().released = true;
        self.script.1.notify_all();
    }

    fn serve(&self, mut stream: TcpStream) {
        let mut head = [0u8; 2048];
        let peeked = stream.peek(&mut head).unwrap_or(0);
        let request = String::from_utf8_lossy(&head[..peeked]).to_string();
        if request.starts_with("GET /stream") {
            let mode = self.script.0.lock().unwrap().stream.clone();
            match mode {
                Some(StreamMode::Echo {
                    partial,
                    final_text,
                }) => self.stream_session(stream, &partial, &final_text),
                _ => {
                    let _ = read_request(&mut stream);
                    respond(&mut stream, "404 Not Found", r#"{"error":"no stream"}"#);
                }
            }
            return;
        }
        let path = read_request(&mut stream);
        if path != "/v1/audio/transcriptions" {
            respond(&mut stream, "200 OK", r#"{"status":"ok"}"#);
            return;
        }
        let reply = {
            let mut script = self.script.0.lock().unwrap();
            script.batch_requests += 1;
            script.replies.pop_front()
        };
        match reply {
            Some(Reply::Text(text)) => respond(&mut stream, "200 OK", &text_body(&text)),
            Some(Reply::Status(status)) => respond(&mut stream, status, r#"{"error":"boom"}"#),
            Some(Reply::Held(text)) => {
                let (script, released) = &*self.script;
                let mut guard = script.lock().unwrap();
                while !guard.released {
                    guard = released.wait(guard).unwrap();
                }
                drop(guard);
                respond(&mut stream, "200 OK", &text_body(&text));
            }
            Some(Reply::HangUp) | None => drop(stream),
        }
    }

    fn stream_session(&self, stream: TcpStream, partial: &str, final_text: &str) {
        let _ = stream.set_read_timeout(Some(Duration::from_secs(30)));
        let Ok(mut socket) = tungstenite::accept(stream) else {
            return;
        };
        self.script.0.lock().unwrap().stream_sessions += 1;
        let mut previewed = false;
        loop {
            match socket.read() {
                Ok(Message::Binary(_)) => {
                    self.script.0.lock().unwrap().stream_audio_frames += 1;
                    if !previewed {
                        previewed = true;
                        let preview = serde_json::json!({
                            "type": "partial",
                            "text": partial,
                            "stable_words": 1,
                        });
                        let _ = socket.send(Message::Text(preview.to_string().into()));
                    }
                }
                Ok(Message::Text(text)) if text.contains("commit") => {
                    let done = serde_json::json!({
                        "type": "final",
                        "text": final_text,
                        "segments": [],
                    });
                    let _ = socket.send(Message::Text(done.to_string().into()));
                    let _ = socket.flush();
                    let _ = socket.close(None);
                    return;
                }
                Ok(_) => {}
                Err(_) => return,
            }
        }
    }
}

fn text_body(text: &str) -> String {
    serde_json::json!({ "text": text }).to_string()
}

/// Reads one request (head and body); its path.
fn read_request(stream: &mut TcpStream) -> String {
    let mut buffer = Vec::new();
    let mut chunk = [0u8; 8192];
    let head_end = loop {
        let read = stream.read(&mut chunk).unwrap_or(0);
        if read == 0 {
            return String::new();
        }
        buffer.extend_from_slice(&chunk[..read]);
        if let Some(at) = buffer.windows(4).position(|window| window == b"\r\n\r\n") {
            break at;
        }
    };
    let head = String::from_utf8_lossy(&buffer[..head_end]).to_string();
    let length = head
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse::<usize>().ok())
                .flatten()
        })
        .unwrap_or(0);
    let chunked = head.to_ascii_lowercase().contains("transfer-encoding: chunked");
    let mut body = buffer[head_end + 4..].to_vec();
    loop {
        let complete = if chunked {
            body.ends_with(b"0\r\n\r\n")
        } else {
            body.len() >= length
        };
        if complete {
            break;
        }
        let read = stream.read(&mut chunk).unwrap_or(0);
        if read == 0 {
            break;
        }
        body.extend_from_slice(&chunk[..read]);
    }
    head.split_whitespace().nth(1).unwrap_or_default().to_string()
}

fn respond(stream: &mut TcpStream, status: &str, body: &str) {
    let _ = write!(
        stream,
        "HTTP/1.1 {status}\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\
         connection: close\r\n\r\n{body}",
        body.len()
    );
    let _ = stream.flush();
}
