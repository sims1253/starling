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
///
/// The reader tolerates read-poll timeouts **mid-frame**: both the host
/// (`platform::unix`'s 250 ms accept-side poll) and the client
/// (`HostClient`'s idle poll) run their connections with a read timeout
/// so idle loops can wake, and a timeout that lands partway through a
/// header or body surfaces as [`FrameError::Io`] *without discarding the
/// bytes already read*. The partial frame is retained here; the next
/// [`FrameReader::read_frame`] resumes exactly where the timeout hit —
/// a poll can never desynchronize the stream.
pub struct FrameReader<R: Read> {
    inner: R,
    cap: usize,
    /// The frame in progress (header first, then body bytes as they
    /// arrive). Empty between frames; never abandoned half-read except
    /// across a poll timeout, which is the resume case above.
    partial: Vec<u8>,
}

impl<R: Read> FrameReader<R> {
    pub fn new(inner: R, cap: usize) -> FrameReader<R> {
        FrameReader {
            inner,
            cap,
            partial: Vec::new(),
        }
    }

    /// Reads one frame. Blocking (or, on a polled stream, waking at the
    /// stream's read timeout — see the type-level docs). On
    /// [`FrameError::TooLarge`] the body is left unread and the
    /// connection should be closed.
    pub fn read_frame(&mut self) -> Result<Frame, FrameError> {
        self.fill(FRAME_HEADER_BYTES)?;
        let declared = u32::from_be_bytes(
            self.partial[0..FRAME_HEADER_BYTES]
                .try_into()
                .expect("four header bytes"),
        );
        if declared as usize > self.cap {
            self.partial.clear();
            return Err(FrameError::TooLarge {
                declared,
                cap: self.cap,
            });
        }
        if declared == 0 {
            self.partial.clear();
            return Err(FrameError::Malformed("zero-length frame".to_string()));
        }
        self.fill(FRAME_HEADER_BYTES + declared as usize)?;
        let frame: Frame = serde_json::from_slice(&self.partial[FRAME_HEADER_BYTES..])
            .map_err(|err| FrameError::Malformed(format!("frame body is not a frame: {err}")))?;
        self.partial.clear();
        Ok(frame)
    }

    /// Grows [`FrameReader::partial`] to `want` bytes. A clean EOF or an
    /// I/O error mid-frame truncates `partial` back to what is really
    /// held and propagates — on the *timeout* path the caller's next
    /// `read_frame` continues from those retained bytes.
    fn fill(&mut self, want: usize) -> Result<(), FrameError> {
        while self.partial.len() < want {
            let start = self.partial.len();
            self.partial.resize(want, 0);
            match self.inner.read(&mut self.partial[start..]) {
                Ok(0) => {
                    self.partial.truncate(start);
                    return Err(FrameError::Eof);
                }
                Ok(read) => self.partial.truncate(start + read),
                Err(err) if err.kind() == io::ErrorKind::Interrupted => {
                    self.partial.truncate(start);
                    continue;
                }
                Err(err) => {
                    self.partial.truncate(start);
                    return Err(FrameError::Io(err));
                }
            }
        }
        Ok(())
    }
}

/// Encodes a frame to wire bytes. Errors when the encoded body exceeds
/// `cap` — the caller turns that into a `message_too_large` close for
/// that connection (the host never truncates a frame; a payload too big
/// for the cap is a real error, not something to silently shrink). The
/// wire length field is a `u32`, so a body over `u32::MAX` bytes is also
/// refused — casting it would truncate the prefix and desynchronize the
/// whole stream, whatever `cap` says.
pub fn encode(frame: &Frame, cap: usize) -> Result<Vec<u8>, FrameError> {
    let body = serde_json::to_vec(frame)
        .map_err(|err| FrameError::Malformed(format!("frame does not serialize: {err}")))?;
    if body.len() > cap || body.len() > u32::MAX as usize {
        return Err(FrameError::TooLarge {
            declared: body.len().min(u32::MAX as usize) as u32,
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

    /// A `Read` source over a byte slice that serves tiny chunks and
    /// answers one read-poll timeout once bytes have flowed — the
    /// timeout lands mid-frame by construction.
    struct TimeoutMidFrame<'a> {
        wire: &'a [u8],
        pos: usize,
        served: usize,
        timed_out: bool,
    }

    impl Read for TimeoutMidFrame<'_> {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            if !self.timed_out && self.served > 0 {
                self.timed_out = true;
                return Err(io::Error::new(io::ErrorKind::TimedOut, "poll"));
            }
            let take = buf.len().min(3).min(self.wire.len() - self.pos);
            if take == 0 {
                return Err(io::Error::new(io::ErrorKind::TimedOut, "poll"));
            }
            buf[..take].copy_from_slice(&self.wire[self.pos..self.pos + take]);
            self.pos += take;
            self.served += take;
            Ok(take)
        }
    }

    #[test]
    fn a_read_poll_timeout_mid_frame_does_not_desynchronize() {
        // First read: three bytes of the header land, then a timeout
        // hits with the frame half-read. Without byte retention the
        // next read_frame would re-read a header from the middle of
        // the stream and desynchronize the whole connection — this
        // pins the resume contract, then proves the stream stays
        // framed afterwards.
        let first = encode(&command_frame("push-to-talk"), DEFAULT_MAX_FRAME_BYTES).unwrap();
        let second =
            encode(&Frame::GetSnapshot { req: "s-2".into() }, DEFAULT_MAX_FRAME_BYTES).unwrap();
        let mut wire = first;
        wire.extend_from_slice(&second);

        let mut reader = FrameReader::new(
            TimeoutMidFrame {
                wire: &wire,
                pos: 0,
                served: 0,
                timed_out: false,
            },
            DEFAULT_MAX_FRAME_BYTES,
        );
        // The timeout surfaces as Io (the caller's idle loop ignores
        // it) — and only once.
        let first_try = reader.read_frame();
        assert!(
            matches!(
                &first_try,
                Err(FrameError::Io(err)) if err.kind() == io::ErrorKind::TimedOut
            ),
            "expected the mid-frame poll timeout to surface, got {first_try:?}"
        );
        // Resume: both frames decode, in order, after the timeout.
        assert_eq!(reader.read_frame().unwrap(), command_frame("push-to-talk"));
        assert_eq!(
            reader.read_frame().unwrap(),
            Frame::GetSnapshot { req: "s-2".into() }
        );
    }
}
