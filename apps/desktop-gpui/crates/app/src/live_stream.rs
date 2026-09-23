//! Recording-scoped WebSocket client for the native /stream protocol.
use std::sync::mpsc::{self, Receiver};
use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use serde_json::Value;
use starling_dictation::storage::{TranscriptionResult, TranscriptionSegment};
use tokio_tungstenite::{connect_async, tungstenite::Message};

enum Command {
    Audio(Vec<u8>),
    Commit,
}

pub(crate) enum Event {
    Partial(String),
    Final(TranscriptionResult),
    Error(String),
}

pub(crate) struct LiveStream {
    commands: tokio::sync::mpsc::Sender<Command>,
    events: Receiver<Event>,
}

pub(crate) fn stream_url(endpoint: &str) -> Result<String, String> {
    let endpoint = endpoint.trim().trim_end_matches('/');
    let (scheme, rest) = if let Some(rest) = endpoint.strip_prefix("https://") {
        ("wss://", rest)
    } else if let Some(rest) = endpoint.strip_prefix("http://") {
        ("ws://", rest)
    } else {
        return Err("Streaming requires an HTTP(S) server endpoint".into());
    };
    let rest = rest
        .strip_suffix("/v1/audio/transcriptions")
        .or_else(|| rest.strip_suffix("/v1"))
        .unwrap_or(rest);
    Ok(format!("{scheme}{rest}/stream"))
}

/// A whole input quantum maps to a whole count of 16 kHz output samples.
/// Cutting live WAV frames on this boundary avoids cumulative duration drift
/// when the device runs at 44.1 kHz.
pub(crate) fn exact_input_quantum(rate: u32) -> usize {
    if rate == 0 {
        return 1;
    }
    let (mut a, mut b) = (rate, 16_000);
    while b != 0 {
        (a, b) = (b, a % b);
    }
    (rate / a.max(1)) as usize
}

impl LiveStream {
    pub(crate) fn start(endpoint: &str) -> Result<Self, String> {
        let url = stream_url(endpoint)?;
        let (commands, mut command_rx) = tokio::sync::mpsc::channel(64);
        let (event_tx, events) = mpsc::channel();
        std::thread::spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build();
            match runtime {
                Ok(runtime) => runtime.block_on(async move {
                    let socket = tokio::time::timeout(Duration::from_secs(10), connect_async(url))
                        .await;
                    let (mut socket, _) = match socket {
                        Ok(Ok(connected)) => connected,
                        Ok(Err(err)) => {
                            let _ = event_tx.send(Event::Error(err.to_string()));
                            return;
                        }
                        Err(_) => {
                            let _ = event_tx.send(Event::Error("Stream connection timed out".into()));
                            return;
                        }
                    };
                    loop {
                        tokio::select! {
                            command = command_rx.recv() => {
                                let message = match command {
                                    Some(Command::Audio(bytes)) => Message::Binary(bytes.into()),
                                    Some(Command::Commit) => Message::Text(r#"{"type":"commit"}"#.into()),
                                    None => break,
                                };
                                if let Err(err) = socket.send(message).await {
                                    let _ = event_tx.send(Event::Error(err.to_string()));
                                    break;
                                }
                            }
                            message = socket.next() => {
                                let event = match message {
                                    Some(Ok(Message::Text(text))) => parse_message(text.as_ref())
                                        .or_else(|| Some(Event::Error("Invalid stream response".into()))),
                                    Some(Ok(Message::Close(_))) | None => Some(Event::Error("Stream closed before final transcript".into())),
                                    Some(Err(err)) => Some(Event::Error(err.to_string())),
                                    _ => None,
                                };
                                if let Some(event) = event {
                                    let done = matches!(event, Event::Final(_) | Event::Error(_));
                                    let _ = event_tx.send(event);
                                    if done { break; }
                                }
                            }
                        }
                    }
                }),
                Err(err) => { let _ = event_tx.send(Event::Error(err.to_string())); }
            }
        });
        Ok(Self { commands, events })
    }

    pub(crate) fn send_audio(&self, wav: Vec<u8>) -> bool {
        self.commands.try_send(Command::Audio(wav)).is_ok()
    }

    pub(crate) fn commit(&self) -> bool {
        self.commands.try_send(Command::Commit).is_ok()
    }

    pub(crate) fn poll_partial(&self) -> Result<Option<String>, String> {
        let mut latest = None;
        while let Ok(event) = self.events.try_recv() {
            match event {
                Event::Partial(text) => latest = Some(text),
                Event::Error(err) => return Err(err),
                Event::Final(_) => return Err("Stream finalized before recording stopped".into()),
            }
        }
        Ok(latest)
    }

    pub(crate) fn final_result(self) -> Result<TranscriptionResult, String> {
        loop {
            match self.events.recv_timeout(Duration::from_secs(120)) {
                Ok(Event::Final(result)) => return Ok(result),
                Ok(Event::Partial(_)) => {}
                Ok(Event::Error(err)) => return Err(err),
                Err(err) => return Err(err.to_string()),
            }
        }
    }
}

fn parse_message(text: &str) -> Option<Event> {
    let payload: Value = serde_json::from_str(text).ok()?;
    match payload.get("type")?.as_str()? {
        "partial" => Some(Event::Partial(payload.get("text")?.as_str()?.to_string())),
        "error" => Some(Event::Error(payload.get("message")?.as_str()?.to_string())),
        "final" => {
            let text = payload.get("text")?.as_str()?.to_string();
            let mut segments = Vec::new();
            if let Some(items) = payload.get("segments").and_then(Value::as_array) {
                for item in items {
                    segments.push(TranscriptionSegment {
                        text: item.get("text")?.as_str()?.to_string(),
                        start_seconds: item.get("start_s")?.as_f64()?,
                        end_seconds: item.get("end_s")?.as_f64()?,
                    });
                }
            }
            Some(Event::Final(TranscriptionResult {
                text,
                segments,
                duration_seconds: payload.get("duration_s").and_then(Value::as_f64),
                request_id: None,
            }))
        }
        "pong" | "reset_ack" => None,
        _ => Some(Event::Error("Unknown stream response".into())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_endpoint_and_final_contract() {
        assert_eq!(
            stream_url("https://starling.local:8181").unwrap(),
            "wss://starling.local:8181/stream"
        );
        assert_eq!(
            stream_url("http://localhost:8181/v1/audio/transcriptions").unwrap(),
            "ws://localhost:8181/stream"
        );
        match parse_message(r#"{"type":"final","text":"hello","segments":[],"duration_s":1.5}"#) {
            Some(Event::Final(result)) => assert_eq!(result.text, "hello"),
            _ => panic!("expected final"),
        }
        assert_eq!(exact_input_quantum(44_100), 441);
        assert_eq!(exact_input_quantum(48_000), 3);
    }
}
