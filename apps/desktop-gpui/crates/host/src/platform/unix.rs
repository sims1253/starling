//! The unix transport: a `SOCK_STREAM` Unix domain socket with
//! kernel-supplied peer credentials.
//!
//! Security posture: the endpoint lives in a 0700 per-user directory and
//! the socket file itself is 0600 — but neither is load-bearing. The
//! authorization is the credential check at accept time:
//!
//! - Linux: `getsockopt(SOL_SOCKET, SO_PEERCRED)` on the accepted stream
//!   returns `struct ucred { pid, uid, gid }` **as filled in by the
//!   kernel at connect time** — the peer cannot lie about its uid.
//! - macOS: `getsockopt(SOL_SOCKET, LOCAL_PEERCRED)` returns
//!   `struct xucred { cr_uid, … }` (the peer's effective uid) and
//!   `LOCAL_PEERPID` the peer's pid.
//!
//! On any other unix the socket type alone works but credentials are
//! unavailable, and [`crate::auth::SameUser`] then fails closed.

use std::io::{self, Read, Write};
use std::os::fd::AsRawFd;
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::Path;

use super::{Probe, TransportConn, TransportListener};
use crate::auth::PeerCredentials;

/// Binds at `path` (an existing stale file is the caller's problem — see
/// [`super::probe`]).
///
/// The socket is bound under a unique temporary name in the same
/// directory, set to 0600, and then renamed onto `path`: the endpoint is
/// never reachable under its **final** name with anything looser than
/// 0600. The temp name itself exists briefly with umask-default
/// permissions between `bind` and the `set_permissions` that follows —
/// that window is closed by the caller's contract, not here: `serve`
/// only binds inside a runtime dir `ensure_runtime_dir` has already
/// tightened to 0700, and a caller of this function directly owes the
/// same containment. (A pre-bind `chmod` is not possible for filesystem
/// sockets: the file's mode is fixed at creation from the process
/// umask, which is process-global and not safely flippable here.)
pub fn listen(path: &Path) -> io::Result<Box<dyn TransportListener>> {
    let directory = path.parent().unwrap_or_else(|| Path::new("."));
    let file_name = path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| "starling-runtime.sock".to_string());
    static ATTEMPT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let mut listener = None;
    let mut temp = None;
    for _ in 0..8 {
        let attempt = ATTEMPT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let candidate = directory.join(format!(
            "{file_name}.{}.{}.tmp",
            std::process::id(),
            attempt
        ));
        match UnixListener::bind(&candidate) {
            Ok(bound) => {
                if let Err(err) =
                    std::fs::set_permissions(&candidate, std::fs::Permissions::from_mode(0o600))
                {
                    // Same cleanup discipline as the rename path below:
                    // drop the listener (closing the socket) and remove
                    // the temp file — no residue for the next boot.
                    drop(bound);
                    let _ = std::fs::remove_file(&candidate);
                    return Err(err);
                }
                listener = Some(bound);
                temp = Some(candidate);
                break;
            }
            // The unique name collided (practically impossible): try the
            // next. Any other error (missing directory, permissions) is
            // a real bind failure — surface it.
            Err(err) if err.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(err) => return Err(err),
        }
    }
    let (listener, temp) =
        match (listener, temp) {
            (Some(listener), Some(temp)) => (listener, temp),
            _ => return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                "could not find a free temp name for the endpoint socket",
            )),
        };
    // Rename publishes the 0600 socket under its final name atomically.
    // A server that bound `path` between the caller's probe and this
    // rename is silently displaced — the same check-then-act window
    // [`remove_stale`] documents; the data-root lease is what serializes
    // hosts against it. A failed rename drops the listener (closing the
    // temp socket) and removes the temp file — no residue for the next
    // boot to trip over.
    if let Err(err) = std::fs::rename(&temp, path) {
        drop(listener);
        let _ = std::fs::remove_file(&temp);
        return Err(err);
    }
    Ok(Box::new(UdsListener { listener }))
}

/// Connects to the endpoint at `path`.
pub fn connect(path: &Path) -> io::Result<Box<dyn TransportConn>> {
    let stream = UnixStream::connect(path)?;
    Ok(Box::new(UdsConn { stream }))
}

/// The dead-file-with-no-listener condition: the classic stale-owner
/// residue after a killed host.
pub fn probe(path: &Path) -> Probe {
    match UnixStream::connect(path) {
        // The probe connection is dropped immediately; a live host's
        // accept loop simply never hears from it again.
        Ok(_) => Probe::Live,
        Err(err) => match err.kind() {
            io::ErrorKind::ConnectionRefused | io::ErrorKind::NotFound => Probe::Dead,
            // EACCES means liveness could NOT be tested (a socket file
            // planted or re-permissioned by another local user, or one
            // whose mode was lost) — the caller must fail closed, not
            // treat the path as stale and unlink it (the Windows
            // transport maps ACCESS_DENIED the same way).
            _ => Probe::Unknown(err.to_string()),
        },
    }
}

/// Removes a stale socket file. Re-probes immediately before unlinking:
/// a server that bound between the caller's earlier `Dead` probe and now
/// must not lose its socket file (clients would silently miss it). A
/// `Live` answer aborts the takeover — the caller treats it as the
/// foreign-server refusal. The probe-to-unlink window itself cannot be
/// closed from user space (the lease serializes hosts; only exotic
/// non-host binders can race it).
pub fn remove_stale(path: &Path) -> io::Result<()> {
    if let Probe::Live = probe(path) {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            format!(
                "refusing to unlink {path:?}: a live server answered it \
                 between the stale probe and the removal"
            ),
        ));
    }
    match std::fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(err) => Err(err),
    }
}

pub struct UdsListener {
    listener: UnixListener,
}

impl TransportListener for UdsListener {
    fn accept(&self) -> io::Result<Box<dyn TransportConn>> {
        let (stream, _peer_addr) = self.listener.accept()?;
        // Readers run with a poll timeout so a connection the host has
        // decided to abandon (slow consumer, shutdown) never parks a
        // thread forever on a peer that will not speak or go away.
        stream.set_read_timeout(Some(std::time::Duration::from_millis(250)))?;
        Ok(Box::new(UdsConn { stream }))
    }

    fn set_nonblocking(&self, nonblocking: bool) -> io::Result<()> {
        self.listener.set_nonblocking(nonblocking)
    }
}

pub struct UdsConn {
    stream: UnixStream,
}

impl UdsConn {
    fn peer_credentials_of(stream: &UnixStream) -> io::Result<PeerCredentials> {
        #[cfg(target_os = "linux")]
        {
            // SO_PEERCRED (unix(7)): valid on connected socket-pair
            // sockets; the kernel filled the credentials at connect(2).
            let mut ucred = libc::ucred {
                pid: 0,
                uid: 0,
                gid: 0,
            };
            let mut len = std::mem::size_of::<libc::ucred>() as libc::socklen_t;
            // SAFETY: ucred is a plain repr(C) struct; getsockopt writes
            // exactly `len` bytes into it and takes no other pointers.
            let result = unsafe {
                libc::getsockopt(
                    stream.as_raw_fd(),
                    libc::SOL_SOCKET,
                    libc::SO_PEERCRED,
                    &mut ucred as *mut libc::ucred as *mut libc::c_void,
                    &mut len,
                )
            };
            if result != 0 {
                return Err(io::Error::last_os_error());
            }
            Ok(PeerCredentials {
                uid: Some(ucred.uid),
                pid: Some(ucred.pid as u32),
            })
        }
        #[cfg(target_os = "macos")]
        {
            // LOCAL_PEERCRED: struct xucred's cr_uid is the peer's
            // effective uid at connect time (the check is the uid; the
            // pid via LOCAL_PEERPID is diagnostics).
            let mut xucred: libc::xucred = unsafe { std::mem::zeroed() };
            let mut len = std::mem::size_of::<libc::xucred>() as libc::socklen_t;
            // SAFETY: as above; xucred is a plain repr(C) struct.
            let result = unsafe {
                libc::getsockopt(
                    stream.as_raw_fd(),
                    libc::SOL_SOCKET,
                    libc::LOCAL_PEERCRED,
                    &mut xucred as *mut libc::xucred as *mut libc::c_void,
                    &mut len,
                )
            };
            if result != 0 {
                return Err(io::Error::last_os_error());
            }
            let mut pid = None;
            let mut peer_pid: libc::pid_t = 0;
            let mut plen = std::mem::size_of::<libc::pid_t>() as libc::socklen_t;
            // SAFETY: peer_pid is a plain pid_t out-parameter.
            if unsafe {
                libc::getsockopt(
                    stream.as_raw_fd(),
                    libc::SOL_SOCKET,
                    libc::LOCAL_PEERPID,
                    &mut peer_pid as *mut libc::pid_t as *mut libc::c_void,
                    &mut plen,
                )
            } == 0
            {
                pid = Some(peer_pid as u32);
            }
            Ok(PeerCredentials {
                uid: Some(xucred.cr_uid),
                pid,
            })
        }
        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        {
            let _ = stream;
            // No supported credential mechanism on this unix: report
            // absence and let the auth policy fail closed (documented in
            // crate::auth). BSD's SOCKCREDSBUF exists but is not wired
            // here; adding it would be additive.
            Ok(PeerCredentials::absent())
        }
    }
}

impl TransportConn for UdsConn {
    fn peer_credentials(&self) -> io::Result<PeerCredentials> {
        Self::peer_credentials_of(&self.stream)
    }

    fn try_clone(&self) -> io::Result<Box<dyn TransportConn>> {
        Ok(Box::new(UdsConn {
            stream: self.stream.try_clone()?,
        }))
    }

    fn shutdown_both(&self) -> io::Result<()> {
        self.stream.shutdown(std::net::Shutdown::Both)
    }

    fn set_read_timeout(&self, timeout: Option<std::time::Duration>) -> io::Result<()> {
        self.stream.set_read_timeout(timeout)
    }

    fn set_write_timeout(&self, timeout: Option<std::time::Duration>) -> io::Result<()> {
        self.stream.set_write_timeout(timeout)
    }
}

impl Read for UdsConn {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.stream.read(buf)
    }
}

impl Write for UdsConn {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.stream.write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.stream.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_socket_file_without_permissions_probes_unknown_not_dead() {
        // EACCES means liveness could not be tested. Treating it as Dead
        // would let a host unlink (and rebind) a socket it could not
        // even connect to — an attacker-planted or re-permissioned file
        // must fail closed instead.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.sock");
        let listener = UnixListener::bind(&path).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o000)).unwrap();
        assert!(
            matches!(probe(&path), Probe::Unknown(_)),
            "EACCES must read Unknown (fail closed), not Dead"
        );
        drop(listener);
    }

    #[test]
    fn remove_stale_refuses_a_socket_that_went_live_again() {
        let dir = tempfile::tempdir().unwrap();
        // Live: a server is bound right now — removal must refuse.
        let live_path = dir.path().join("live.sock");
        let squatter = UnixListener::bind(&live_path).unwrap();
        let refused = remove_stale(&live_path);
        assert!(
            refused.is_err(),
            "a live server's socket file must not be unlinked"
        );
        assert_eq!(
            refused.unwrap_err().kind(),
            io::ErrorKind::PermissionDenied
        );
        assert!(live_path.exists(), "the squatter keeps its socket file");
        drop(squatter);

        // And the residue case still works: a leftover file with no
        // listener behind it is stale and removable.
        let dead_path = dir.path().join("dead.sock");
        let gone = UnixListener::bind(&dead_path).unwrap();
        drop(gone);
        assert!(matches!(probe(&dead_path), Probe::Dead));
        remove_stale(&dead_path).expect("stale residue is removed");
        assert!(!dead_path.exists());
    }
}
