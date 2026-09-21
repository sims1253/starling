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
/// [`super::probe`]); sets the socket file to 0600.
pub fn listen(path: &Path) -> io::Result<Box<dyn TransportListener>> {
    let listener = UnixListener::bind(path)?;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))?;
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
            io::ErrorKind::ConnectionRefused
            | io::ErrorKind::NotFound
            | io::ErrorKind::PermissionDenied => Probe::Dead,
            _ => Probe::Unknown(err.to_string()),
        },
    }
}

/// Removes a stale socket file. Refuses when a probe just said `Live` —
/// that is the caller's decision tree, not this helper's.
pub fn remove_stale(path: &Path) -> io::Result<()> {
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
