//! A scripted HTTP/1.1 server for the provider tests: every accepted
//! connection runs the next script step against the raw socket, so a test
//! can answer, stream slowly, stall, cut the connection or redirect, and
//! afterwards inspect every request the client really sent.

#![allow(dead_code)]

use std::collections::VecDeque;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::time::Duration;

#[derive(Clone, Debug)]
pub struct Recorded {
    pub method: String,
    pub path: String,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl Recorded {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, value)| value.as_str())
    }

    pub fn json(&self) -> serde_json::Value {
        serde_json::from_slice(&self.body).unwrap_or_else(|error| {
            // read_request only reads content-length bodies (no chunked
            // uploads, no 100-continue).
            panic!(
                "request body is not JSON ({error}): {:?}",
                String::from_utf8_lossy(&self.body)
            )
        })
    }
}

pub type Step = Box<dyn FnOnce(&Recorded, &mut TcpStream) + Send>;

pub struct FakeServer {
    pub addr: SocketAddr,
    pub requests: Arc<Mutex<Vec<Recorded>>>,
}

impl FakeServer {
    pub fn start(steps: Vec<Step>) -> FakeServer {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind fake server");
        let addr = listener.local_addr().unwrap();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let log = Arc::clone(&requests);
        let steps = Arc::new(Mutex::new(VecDeque::from(steps)));
        std::thread::spawn(move || {
            // Connections are served concurrently (a stalled answer must
            // not hold up the next request, e.g. a server-side cancel),
            // but each takes the next step in arrival order.
            for stream in listener.incoming().flatten() {
                let log = Arc::clone(&log);
                let steps = Arc::clone(&steps);
                std::thread::spawn(move || {
                    let mut stream = stream;
                    let Some(request) = read_request(&mut stream) else {
                        return;
                    };
                    // Logging and taking the step under one lock keeps
                    // request N paired with step N.
                    let step = {
                        let mut log = log.lock().unwrap();
                        log.push(request.clone());
                        steps.lock().unwrap().pop_front()
                    };
                    match step {
                        Some(step) => step(&request, &mut stream),
                        None => {
                            let _ =
                                stream.write_all(&response("500 Internal Server Error", &[], "{}"));
                        }
                    }
                });
            }
        });
        FakeServer { addr, requests }
    }

    pub fn url(&self) -> String {
        format!("http://{}", self.addr)
    }

    pub fn requests(&self) -> Vec<Recorded> {
        self.requests.lock().unwrap().clone()
    }
}

fn read_request(stream: &mut TcpStream) -> Option<Recorded> {
    let mut buffer = Vec::new();
    let mut chunk = [0u8; 4096];
    let header_end = loop {
        let read = stream.read(&mut chunk).ok()?;
        if read == 0 {
            return None;
        }
        buffer.extend_from_slice(&chunk[..read]);
        if let Some(position) = buffer.windows(4).position(|w| w == b"\r\n\r\n") {
            break position;
        }
    };
    let head = String::from_utf8_lossy(&buffer[..header_end]).to_string();
    let mut lines = head.split("\r\n");
    let mut parts = lines.next()?.split_whitespace();
    let method = parts.next()?.to_string();
    let path = parts.next()?.to_string();
    let headers: Vec<(String, String)> = lines
        .filter_map(|line| line.split_once(':'))
        .map(|(key, value)| (key.trim().to_ascii_lowercase(), value.trim().to_string()))
        .collect();
    let length = headers
        .iter()
        .find(|(key, _)| key == "content-length")
        .and_then(|(_, value)| value.parse::<usize>().ok())
        .unwrap_or(0);
    let mut body = buffer[header_end + 4..].to_vec();
    while body.len() < length {
        let read = stream.read(&mut chunk).ok()?;
        if read == 0 {
            break;
        }
        body.extend_from_slice(&chunk[..read]);
    }
    body.truncate(length);
    Some(Recorded {
        method,
        path,
        headers,
        body,
    })
}

pub fn response(status: &str, headers: &[(&str, &str)], body: &str) -> Vec<u8> {
    let mut head = format!("HTTP/1.1 {status}\r\ncontent-length: {}\r\n", body.len());
    for (name, value) in headers {
        head.push_str(&format!("{name}: {value}\r\n"));
    }
    head.push_str("connection: close\r\n\r\n");
    let mut bytes = head.into_bytes();
    bytes.extend_from_slice(body.as_bytes());
    bytes
}

/// Answers once with a complete response.
pub fn reply(
    status: &'static str,
    headers: Vec<(&'static str, &'static str)>,
    body: String,
) -> Step {
    Box::new(move |_, stream| {
        let _ = stream.write_all(&response(status, &headers, &body));
        let _ = stream.flush();
    })
}

pub fn ok_json(body: serde_json::Value) -> Step {
    reply(
        "200 OK",
        vec![("content-type", "application/json")],
        body.to_string(),
    )
}

/// Streams `events` as a chunked `text/event-stream`, pausing `gap`
/// between them; `finish` controls whether the terminating chunk is sent
/// (without it the connection just closes: a cut stream).
pub fn sse(events: Vec<String>, gap: Duration, finish: bool) -> Step {
    Box::new(move |_, stream| {
        let head = "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\nconnection: close\r\n\r\n";
        if stream.write_all(head.as_bytes()).is_err() {
            return;
        }
        for event in events {
            let payload = format!("{event}\n\n");
            let chunk = format!("{:x}\r\n{payload}\r\n", payload.len());
            if stream.write_all(chunk.as_bytes()).is_err() || stream.flush().is_err() {
                return;
            }
            std::thread::sleep(gap);
        }
        if finish {
            let _ = stream.write_all(b"0\r\n\r\n");
        }
        let _ = stream.flush();
    })
}

/// Accepts the request and never answers for `hold`.
pub fn stall(hold: Duration) -> Step {
    Box::new(move |_, _stream| std::thread::sleep(hold))
}

/// Streams one event every `gap` until the client hangs up (or `max`
/// events), recording whether the hang-up was observed. The hang-up is
/// seen by reading the socket (EOF or reset), not by waiting for a write
/// to fail, which the kernel's buffers can delay for many events.
pub fn slow_sse_until_closed(
    event: String,
    gap: Duration,
    max: usize,
    closed: Arc<Mutex<bool>>,
) -> Step {
    Box::new(move |_, stream| {
        if let Ok(mut reader) = stream.try_clone() {
            let closed = Arc::clone(&closed);
            std::thread::spawn(move || {
                let mut byte = [0u8; 1];
                while let Ok(read) = reader.read(&mut byte) {
                    if read == 0 {
                        break;
                    }
                }
                *closed.lock().unwrap() = true;
            });
        }
        let head = "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\nconnection: close\r\n\r\n";
        if stream.write_all(head.as_bytes()).is_err() {
            *closed.lock().unwrap() = true;
            return;
        }
        for _ in 0..max {
            if *closed.lock().unwrap() {
                return;
            }
            let payload = format!("{event}\n\n");
            let chunk = format!("{:x}\r\n{payload}\r\n", payload.len());
            if stream.write_all(chunk.as_bytes()).is_err() || stream.flush().is_err() {
                *closed.lock().unwrap() = true;
                return;
            }
            std::thread::sleep(gap);
        }
    })
}
