//! A renderer process double for `starling-runtime-host`'s kill-acceptance
//! suite (`tests/renderer_kill.rs`): a real separate process the tests
//! SIGKILL at a chosen connection phase, the way a renderer actually
//! dies. The modes name the phases:
//!
//! - `raw` — transport-level connect, then never read, never write: the
//!   harshest handshake kill (the hello is never even consumed).
//! - `hold` — full client handshake, then idle.
//! - `command` — transport-level connect, one valid command frame
//!   (`jobs.setLimits`), then never read the receipt.
//! - `take` — handshake, context snapshot + manual mode (route freeze),
//!   `capture.start`, wait for acknowledged progress, then hold the live
//!   take open.
//! - `fill` — transport-level connect, a flood of snapshot requests
//!   against a shrunk receive buffer (the wedged-peer shape), then never
//!   read.
//!
//! Each mode prints exactly one `ready` line on stdout when its phase is
//! reached (nothing else — a piped stdout nobody drains must not
//! deadlock the process), then parks until killed. Errors print to
//! stderr and exit 1.
//!
//! The binary is deliberately dependency-free beyond the host crate
//! itself: it builds on every platform the crate builds on (the `fill`
//! mode's receive-buffer shrink is unix-only — that mode exits with an
//! error elsewhere; the suite that spawns it is unix-only anyway).

use std::io::Write;
use std::time::Duration;

use starling_runtime_host::client::HostClient;
use starling_runtime_host::frame::{encode, Frame};
use starling_runtime_host::platform;

fn main() {
    let mut socket: Option<String> = None;
    let mut mode: Option<String> = None;
    let mut take_corr = "take_kill".to_string();
    let mut frames: usize = 512;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--socket" => socket = args.next(),
            "--mode" => mode = args.next(),
            "--take" => take_corr = args.next().unwrap_or_default(),
            "--frames" => {
                frames = args
                    .next()
                    .and_then(|value| value.parse().ok())
                    .unwrap_or(512)
            }
            other => {
                fail(&format!("unknown argument {other:?}"));
            }
        }
    }
    let Some(socket) = socket else {
        fail("missing --socket")
    };
    let Some(mode) = mode else {
        fail("missing --mode")
    };
    let socket = std::path::PathBuf::from(socket);

    match mode.as_str() {
        "raw" => raw_mode(&socket),
        "hold" => hold_mode(&socket),
        "command" => command_mode(&socket),
        "take" => take_mode(&socket, &take_corr),
        "fill" => fill_mode(&socket, frames),
        other => fail(&format!("unknown mode {other:?}")),
    }
    park();
}

fn fail(message: &str) -> ! {
    eprintln!("renderer-double: {message}");
    std::process::exit(1);
}

/// The one-line phase signal; nothing else may print here (see the module
/// docs on piped stdout).
fn ready() {
    let mut out = std::io::stdout().lock();
    let _ = writeln!(out, "ready");
    let _ = out.flush();
}

/// Blocks until SIGKILLed — the double never exits on its own.
fn park() -> ! {
    loop {
        std::thread::sleep(Duration::from_secs(3600));
    }
}

fn connect_transport(socket: &std::path::Path) -> Box<dyn platform::TransportConn> {
    match platform::connect(socket) {
        Ok(conn) => conn,
        Err(err) => fail(&format!("transport connect failed: {err}")),
    }
}

/// Transport-level connect; never read, never written. The connection
/// is leaked so it stays open until the SIGKILL — the kill, not a
/// graceful close, is what the host's reader sees.
fn raw_mode(socket: &std::path::Path) {
    std::mem::forget(connect_transport(socket));
    ready();
}

/// Full client handshake, then idle.
fn hold_mode(socket: &std::path::Path) {
    match HostClient::connect(socket) {
        Ok(client) => std::mem::forget(client),
        Err(err) => fail(&format!("client connect failed: {err}")),
    }
    ready();
}

/// One valid command frame, receipt never read.
fn command_mode(socket: &std::path::Path) {
    let mut conn = connect_transport(socket);
    let envelope = serde_json::json!({
        "v": 1,
        "id": "cmd-double",
        "ts": "2026-09-23T10:00:00Z",
        "type": "jobs.setLimits",
        "payload": { "maxQueued": 3, "maxConcurrent": 1 }
    });
    let wire = match encode(
        &Frame::Command { envelope },
        starling_runtime_host::frame::DEFAULT_MAX_FRAME_BYTES,
    ) {
        Ok(wire) => wire,
        Err(err) => fail(&format!("frame encode failed: {err:?}")),
    };
    if let Err(err) = conn.write_all(&wire) {
        fail(&format!("frame write failed: {err}"));
    }
    let _ = conn.flush();
    ready();
}

/// Route freeze + a live take with acknowledged audio.
fn take_mode(socket: &std::path::Path, take_corr: &str) {
    use starling_runtime::protocol::{Command, Manual};

    let client = match HostClient::connect(socket) {
        Ok(client) => client,
        Err(err) => fail(&format!("client connect failed: {err}")),
    };
    let send = |corr: &str, command: Command| {
        if let Err(err) = client.send(Some(corr), command) {
            fail(&format!("send failed: {err}"));
        }
    };
    fn wait_for(
        client: &HostClient,
        label: &str,
        predicate: impl Fn(&starling_runtime_host::client::EventWire) -> bool,
    ) {
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        loop {
            match client.recv_event_timeout(Duration::from_millis(50)) {
                Ok(event) => {
                    if predicate(&event) {
                        return;
                    }
                }
                Err(starling_runtime::channel::RecvError::Timeout) => {
                    if std::time::Instant::now() > deadline {
                        fail(&format!("timed out waiting for {label}"));
                    }
                }
                Err(other) => fail(&format!("event stream failed: {other:?}")),
            }
        }
    }

    send(
        "ctx-double",
        Command::ContextSnapshot {
            source: "vscode".into(),
        },
    );
    wait_for(
        &client,
        "context.targetSnapshot",
        |event: &starling_runtime_host::client::EventWire| {
            event.type_name() == "context.targetSnapshot"
        },
    );
    send(
        "ctx-double",
        Command::ModeSet {
            mode: "code-guidance".into(),
            source: Manual,
        },
    );
    wait_for(
        &client,
        "mode.decision",
        |event: &starling_runtime_host::client::EventWire| event.type_name() == "mode.decision",
    );

    send(
        take_corr,
        Command::CaptureStart {
            policy: "push-to-talk".into(),
        },
    );
    wait_for(
        &client,
        "acknowledged capture.progress",
        |event: &starling_runtime_host::client::EventWire| {
            event.type_name() == "capture.progress"
                && event.payload()["ackSamples"].as_u64().unwrap_or(0) > 0
        },
    );
    ready();
}

/// The wedged-peer shape: flood snapshot requests against a shrunk
/// receive buffer, never read a reply. Unix-only (the shrink is
/// `setsockopt`; elsewhere the mode fails fast — the suite that uses it
/// is unix-only too).
fn fill_mode(socket: &std::path::Path, frames: usize) {
    #[cfg(unix)]
    {
        use std::os::fd::AsRawFd;
        use std::os::unix::net::UnixStream;

        // A plain UDS, not the platform seam: this mode needs the raw fd
        // for the receive-buffer shrink, which the trait does not expose
        // (and should not — nothing production-side shrinks peer buffers).
        let mut conn = match UnixStream::connect(socket) {
            Ok(conn) => conn,
            Err(err) => fail(&format!("unix connect failed: {err}")),
        };
        // SAFETY: `size` is a valid, initialized `c_int` for the duration
        // of the call and the fd is open — the same established shape as
        // the host crate's own tests. Advisory like every SO_RCVBUF.
        let size: libc::c_int = 2048;
        let rc = unsafe {
            libc::setsockopt(
                conn.as_raw_fd(),
                libc::SOL_SOCKET,
                libc::SO_RCVBUF,
                &size as *const libc::c_int as *const libc::c_void,
                std::mem::size_of::<libc::c_int>() as libc::socklen_t,
            )
        };
        if rc != 0 {
            fail("failed to shrink the receive buffer");
        }
        let request = match serde_json::to_vec(&Frame::GetSnapshot {
            req: "double".into(),
        }) {
            Ok(bytes) => bytes,
            Err(err) => fail(&format!("request encode failed: {err}")),
        };
        for _ in 0..frames {
            // A failed write means the host already ended the connection
            // (eviction) — the wedged phase this mode builds is done; the
            // kill that follows is then merely prompt.
            let prefix = (request.len() as u32).to_be_bytes();
            if conn.write_all(&prefix).is_err() || conn.write_all(&request).is_err() {
                break;
            }
        }
        let _ = conn.flush();
        ready();
    }
    #[cfg(not(unix))]
    {
        let _ = (socket, frames);
        fail("fill mode is unix-only");
    }
}
