//! The IPC transport frame (E17 I4): a length-prefixed JSON wrapper
//! around the I3 wire protocol.
//!
//! There is deliberately **no second message format** here. Every
//! application payload riding this transport is the v1 envelope exactly
//! as `packages/contracts/runtime-protocol/` defines it — commands the
//! client sends, events the host pushes. The [`Frame`] adds only what a
//! transport needs and the envelope does not carry: a kind
//! discriminator, the receipt for a command (the runtime's
//! `Result<Receipt, Rejection>`, serialized by the runtime crate
//! itself), the snapshot projection, and transport-level errors with
//! clear codes.
//!
//! Wire form: `u32` big-endian byte length, then that many bytes of
//! UTF-8 JSON. The length header is checked against the connection's
//! frame cap **before** the body is read, so an oversized frame never
//! buffers (an attacker or a wedged peer cannot make the host allocate
//! past the cap).

use std::io::{self, Read};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use starling_runtime::machine::{Receipt, Rejection};

/// Default per-connection frame cap (1 MiB). The largest legitimate v1
/// frame is a `docs.updateHead` carrying a revision's `text` — bounded by
/// this cap, not by trust.
pub const DEFAULT_MAX_FRAME_BYTES: usize = 1024 * 1024;

/// The frame header: 4 bytes, big-endian.
pub const FRAME_HEADER_BYTES: usize = 4;

/// Why the transport-level side of a connection failed or closed. These
/// are transport conditions, never machine outcomes — machine outcomes
/// ride the envelope (events) and the receipt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransportErrorCode {
    /// A frame declared a body larger than the connection's cap. The
    /// connection closes; the body is never read.
    MessageTooLarge,
    /// The per-connection command rate limit was exceeded. The
    /// connection closes — a client that floods the host is a client the
    /// host stops serving, not one it queues unbounded work for.
    RateLimited,
    /// The body was not a valid frame (zero length, invalid JSON, or a
    /// JSON shape that is not one of this protocol's frames).
    MalformedFrame,
    /// A frame was syntactically valid but in the wrong direction or at
    /// the wrong point in the exchange (a client sending `hello`).
    ProtocolViolation,
    /// Peer authentication failed (wrong user, or credentials the kernel
    /// could not supply on this platform). The connection closes
    /// immediately.
    AuthFailed,
    /// The connection's event queue filled: this client stopped reading
    /// while the runtime kept producing. The connection closes — the
    /// runtime must never stall on one renderer (§1 Mode B); the client
    /// reconnects and resynchronizes from the snapshot.
    SlowConsumer,
    /// The connection count is at the host's cap.
    TooManyConnections,
    /// The host is shutting down; the connection was closed gracefully.
    ShuttingDown,
}

impl TransportErrorCode {
    pub fn as_str(self) -> &'static str {
        match self {
            TransportErrorCode::MessageTooLarge => "message_too_large",
            TransportErrorCode::RateLimited => "rate_limited",
            TransportErrorCode::MalformedFrame => "malformed_frame",
            TransportErrorCode::ProtocolViolation => "protocol_violation",
            TransportErrorCode::AuthFailed => "auth_failed",
            TransportErrorCode::SlowConsumer => "slow_consumer",
            TransportErrorCode::TooManyConnections => "too_many_connections",
            TransportErrorCode::ShuttingDown => "shutting_down",
        }
    }
}

/// One transport frame. `kind` is the serde tag.
///
/// Direction rules (enforced by both sides):
/// - client → host: [`Frame::Command`], [`Frame::GetSnapshot`]
/// - host → client: [`Frame::Hello`], [`Frame::Receipt`], [`Frame::Event`],
///   [`Frame::Snapshot`], [`Frame::TransportError`], [`Frame::Bye`]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Frame {
    /// A command envelope, exactly the I3 wire form. A client that leaves
    /// `seq` absent asks the host to assign it (from the same frontier the
    /// embedded client uses); a client that carries `seq` owns its own
    /// numbering and faces the router's monotonicity check.
    Command { envelope: Value },
    /// Ask for the runtime projection snapshot. `req` is the caller's
    /// correlation token (any string; echoed on [`Frame::Snapshot`]).
    GetSnapshot { req: String },
    /// Sent first on every accepted connection: the host's identity and
    /// the limits this connection runs under.
    Hello {
        /// The transport protocol revision (1).
        protocol: u32,
        /// The storage-v2 lease owner id this host holds.
        owner_id: String,
        /// The host process id (diagnostics; also what a stale-owner probe
        /// checks for liveness).
        pid: u32,
        max_frame_bytes: u64,
        /// The per-connection command rate limit.
        rate_max: u32,
        /// The rate window in milliseconds.
        rate_window_ms: u64,
    },
    /// The receipt for the command envelope whose `id` was `req`. The
    /// result is the runtime's own `Result` serialization — an accepted
    /// command, a served view, or the exact rejection. `seq` echoes the
    /// sequence number the host routed the command under: the assigned
    /// one for typed sends (so a client can reconstruct full envelopes —
    /// the conformance suite replays them through the oracle), the
    /// client-supplied one on the raw path.
    Receipt {
        req: String,
        seq: Option<u64>,
        result: Result<Receipt, Rejection>,
    },
    /// An event envelope pushed to every live connection.
    Event { envelope: Value },
    /// The snapshot reply for [`Frame::GetSnapshot`].
    Snapshot { req: String, snapshot: Value },
    /// A fatal transport condition on this connection; it closes after
    /// this frame. `detail` is human-readable and logged verbatim by the
    /// client library.
    TransportError {
        code: TransportErrorCode,
        detail: String,
    },
    /// Graceful close (host shutdown). The host released nothing that
    /// survives it: acknowledged audio is in storage v2, and a new host
    /// can take over the socket via the lease.
    Bye { reason: String },
}

/// Why a frame could not be read.
#[derive(Debug)]
pub enum FrameError {
    /// The peer closed its side (normal end of stream).
    Eof,
    /// Underlying I/O failure.
    Io(io::Error),
    /// The declared body length exceeds the cap. The body was **not**
    /// read; the connection must be closed after reporting
    /// [`TransportErrorCode::MessageTooLarge`].
    TooLarge { declared: u32, cap: usize },
    /// The body is not a valid frame.
    Malformed(String),
}

/// Reads frames from a byte stream, enforcing the cap at the header.
pub struct FrameReader<R: Read> {
    inner: R,
    cap: usize,
}

impl<R: Read> FrameReader<R> {
    pub fn new(inner: R, cap: usize) -> FrameReader<R> {
        FrameReader { inner, cap }
    }

    /// Reads one frame. Blocking. On [`FrameError::TooLarge`] the body is
    /// left unread and the connection should be closed.
    pub fn read_frame(&mut self) -> Result<Frame, FrameError> {
        let mut header = [0u8; FRAME_HEADER_BYTES];
        read_exact_or_eof(&mut self.inner, &mut header)?;
        let declared = u32::from_be_bytes(header);
        if declared as usize > self.cap {
            return Err(FrameError::TooLarge {
                declared,
                cap: self.cap,
            });
        }
        if declared == 0 {
            return Err(FrameError::Malformed("zero-length frame".to_string()));
        }
        let mut body = vec![0u8; declared as usize];
        read_exact_or_eof(&mut self.inner, &mut body)?;
        let frame: Frame = serde_json::from_slice(&body)
            .map_err(|err| FrameError::Malformed(format!("frame body is not a frame: {err}")))?;
        Ok(frame)
    }
}

/// `read_exact` that reports a clean EOF (peer closed) instead of
/// lumping it in with I/O errors.
fn read_exact_or_eof<R: Read>(inner: &mut R, buf: &mut [u8]) -> Result<(), FrameError> {
    let mut filled = 0;
    while filled < buf.len() {
        match inner.read(&mut buf[filled..]) {
            Ok(0) => return Err(FrameError::Eof),
            Ok(n) => filled += n,
            Err(err) if err.kind() == io::ErrorKind::Interrupted => continue,
            Err(err) => return Err(FrameError::Io(err)),
        }
    }
    Ok(())
}

/// Encodes a frame to wire bytes. Errors when the encoded body exceeds
/// `cap` — the caller turns that into a `message_too_large` close for
/// that connection (the host never truncates a frame; a payload too big
/// for the cap is a real error, not something to silently shrink).
pub fn encode(frame: &Frame, cap: usize) -> Result<Vec<u8>, FrameError> {
    let body = serde_json::to_vec(frame)
        .map_err(|err| FrameError::Malformed(format!("frame does not serialize: {err}")))?;
    if body.len() > cap {
        return Err(FrameError::TooLarge {
            declared: body.len() as u32,
            cap,
        });
    }
    let mut wire = Vec::with_capacity(FRAME_HEADER_BYTES + body.len());
    wire.extend_from_slice(&(body.len() as u32).to_be_bytes());
    wire.extend_from_slice(&body);
    Ok(wire)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn command_frame(payload: &str) -> Frame {
        Frame::Command {
            envelope: serde_json::json!({
                "v": 1,
                "id": "cmd_1",
                "ts": "2026-09-20T10:00:00Z",
                "type": "capture.start",
                "payload": { "policy": payload }
            }),
        }
    }

    #[test]
    fn frames_round_trip_through_the_wire() {
        let frames = vec![
            command_frame("push-to-talk"),
            Frame::GetSnapshot { req: "s-1".into() },
            Frame::Hello {
                protocol: 1,
                owner_id: "l_abc".into(),
                pid: 42,
                max_frame_bytes: 1024,
                rate_max: 16,
                rate_window_ms: 1000,
            },
            Frame::Receipt {
                req: "cmd_1".into(),
                seq: Some(7),
                result: Ok(Receipt::Accepted),
            },
            Frame::Receipt {
                req: "cmd_2".into(),
                seq: None,
                result: Err(Rejection::IllegalInState {
                    command: "capture.stop".into(),
                    state: "Idle".into(),
                    detail: "not legal from Idle".into(),
                }),
            },
            Frame::Event {
                envelope: serde_json::json!({
                    "v": 1, "id": "evt_1", "ts": "2026-09-20T10:00:01Z",
                    "corr": "take_1", "seq": 4, "type": "capture.started",
                    "payload": { "device": "default-input", "actualRate": 16000, "channels": 1 }
                }),
            },
            Frame::Snapshot {
                req: "s-1".into(),
                snapshot: serde_json::json!({ "capture": { "state": "Idle" } }),
            },
            Frame::TransportError {
                code: TransportErrorCode::RateLimited,
                detail: "12 frames in 100ms window".into(),
            },
            Frame::Bye {
                reason: "shutdown".into(),
            },
        ];
        for frame in frames {
            let wire = encode(&frame, DEFAULT_MAX_FRAME_BYTES).expect("encodes");
            let mut reader = FrameReader::new(&wire[..], DEFAULT_MAX_FRAME_BYTES);
            assert_eq!(reader.read_frame().expect("decodes"), frame);
        }
    }

    #[test]
    fn oversized_header_is_rejected_without_reading_the_body() {
        // Declare a body past the cap and never send one: if the reader
        // tried to buffer the body this would hang or over-allocate,
        // not return promptly.
        let mut header = vec![0u8; FRAME_HEADER_BYTES];
        let huge = (DEFAULT_MAX_FRAME_BYTES + 1) as u32;
        header.copy_from_slice(&huge.to_be_bytes());
        let mut reader = FrameReader::new(&header[..], DEFAULT_MAX_FRAME_BYTES);
        match reader.read_frame() {
            Err(FrameError::TooLarge { declared, cap }) => {
                assert_eq!(declared as usize, DEFAULT_MAX_FRAME_BYTES + 1);
                assert_eq!(cap, DEFAULT_MAX_FRAME_BYTES);
            }
            other => panic!("expected TooLarge, got {other:?}"),
        }
    }

    #[test]
    fn zero_length_and_garbage_bodies_are_malformed() {
        let zero = 0u32.to_be_bytes();
        let mut reader = FrameReader::new(&zero[..], DEFAULT_MAX_FRAME_BYTES);
        assert!(matches!(reader.read_frame(), Err(FrameError::Malformed(_))));

        let garbage = b"not json at all".to_vec();
        let mut wire = (garbage.len() as u32).to_be_bytes().to_vec();
        wire.extend_from_slice(&garbage);
        let mut reader = FrameReader::new(&wire[..], DEFAULT_MAX_FRAME_BYTES);
        assert!(matches!(reader.read_frame(), Err(FrameError::Malformed(_))));

        // Valid JSON that is not a frame (no known "kind").
        let not_frame = br#"{"hello":true}"#;
        let mut wire = (not_frame.len() as u32).to_be_bytes().to_vec();
        wire.extend_from_slice(not_frame);
        let mut reader = FrameReader::new(&wire[..], DEFAULT_MAX_FRAME_BYTES);
        assert!(matches!(reader.read_frame(), Err(FrameError::Malformed(_))));
    }

    #[test]
    fn eof_is_distinguished_from_io_error() {
        let mut reader = FrameReader::new(&[][..], DEFAULT_MAX_FRAME_BYTES);
        assert!(matches!(reader.read_frame(), Err(FrameError::Eof)));
        // A truncated frame (header only) is also a clean EOF.
        let header = 8u32.to_be_bytes();
        let mut reader = FrameReader::new(&header[..], DEFAULT_MAX_FRAME_BYTES);
        assert!(matches!(reader.read_frame(), Err(FrameError::Eof)));
    }

    #[test]
    fn encode_enforces_the_cap() {
        let frame = command_frame(&"x".repeat(512));
        let wire = encode(&frame, DEFAULT_MAX_FRAME_BYTES).expect("fits the default cap");
        assert!(wire.len() > 512);
        match encode(&frame, 64) {
            Err(FrameError::TooLarge { cap, .. }) => assert_eq!(cap, 64),
            other => panic!("expected TooLarge, got {other:?}"),
        }
    }
}
