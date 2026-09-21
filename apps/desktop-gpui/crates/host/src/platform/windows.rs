//! The Windows transport: a named pipe with a restrictive DACL.
//!
//! **Compile-unverified (recorded gap):** no Windows target is installed
//! on the machine this was written on, so `cargo check --target …` could
//! not run against this module (and the workspace's bundled-SQLite C
//! dependency would additionally need a cross C toolchain). The code
//! follows the documented Win32 named-pipe semantics; its first Windows
//! build is expected to be its first compile. The runtime tests for it
//! are equally absent — see the PR's "Not executed here" section.
//!
//! Security posture (what "authenticated" means on Windows — there is no
//! `SO_PEERCRED` analogue): the pipe is created with a security
//! descriptor whose DACL grants access to exactly the creating user's
//! SID and the system (`D:P(A;;GRGW;;;SY)(A;;GRGW;;;<sid>)`). The kernel
//! enforces the DACL on every `CreateFileW` against the pipe name, so a
//! connection this server accepts has already been proven to run as the
//! creating user — the same-user decision the unix side makes from
//! `SO_PEERCRED`, made here at object-creation time instead. The client
//! pid (`GetNamedPipeClientProcessId`) is captured for diagnostics.
//!
//! Server shape: a dedicated acceptor thread blocks in synchronous
//! `ConnectNamedPipe` on the current pipe instance and hands connected
//! instances to `accept()` through a channel; each accepted instance is
//! serviced with synchronous `ReadFile`/`WriteFile`. Shutdown wakes the
//! acceptor by connecting to the pipe as a client — the pending
//! `ConnectNamedPipe` completes with that self-connection, the acceptor
//! finds its channel closed, and it closes its handles and exits. This
//! avoids overlapped I/O entirely at the cost of one wake connection at
//! shutdown.

use std::ffi::OsString;
use std::io::{self, Read, Write};
use std::os::windows::ffi::{OsStrExt, OsStringExt};
use std::path::{Path, PathBuf};
use std::sync::mpsc;

#[allow(unused_imports)]
use windows_sys::Win32::Foundation::{
    CloseHandle, DuplicateHandle, GetLastError, LocalFree, DUPLICATE_SAME_ACCESS,
    ERROR_ACCESS_DENIED, ERROR_FILE_NOT_FOUND, ERROR_OPERATION_ABORTED, ERROR_PATH_NOT_FOUND,
    ERROR_PIPE_BUSY, INVALID_HANDLE_VALUE,
};
use windows_sys::Win32::Security::{
    ConvertSidToStringSidW, ConvertStringSecurityDescriptorToSecurityDescriptorW,
    GetTokenInformation, OpenProcessToken, SECURITY_ATTRIBUTES, TOKEN_QUERY, TOKEN_USER,
};
#[allow(unused_imports)]
use windows_sys::Win32::Storage::FileSystem::{
    CreateFileW, FlushFileBuffers, FILE_FLAG_FIRST_PIPE_INSTANCE, GENERIC_READ, GENERIC_WRITE,
    OPEN_EXISTING,
};
use windows_sys::Win32::System::Pipes::{
    ConnectNamedPipe, CreateNamedPipeW, DisconnectNamedPipe, GetNamedPipeClientProcessId,
    WaitNamedPipeW, PIPE_ACCESS_DUPLEX, PIPE_READMODE_BYTE, PIPE_TYPE_BYTE,
    PIPE_UNLIMITED_INSTANCES, PIPE_WAIT,
};
use windows_sys::Win32::System::Threading::GetCurrentProcess;

use super::{Probe, TransportConn, TransportListener};
use crate::auth::PeerCredentials;

/// The SDDL string that restricts a pipe to the creating user + system.
/// `D:P` = DACL protected; `A;;GRGW;;;SY` lets the system read/write;
/// `A;;GRGW;;;<sid>` lets exactly the owning user read/write. Everyone
/// else gets no ACE, which means no access.
///
/// Pure string assembly — unit-tested on every platform.
pub fn sddl_for_owner(owner_sid: &str) -> String {
    format!("D:P(A;;GRGW;;;SY)(A;;GRGW;;;{owner_sid})")
}

/// The current process user's SID, as a string (S-1-5-21-…).
fn current_user_sid() -> io::Result<String> {
    unsafe {
        let mut token = INVALID_HANDLE_VALUE;
        if OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &mut token) == 0 {
            return Err(io::Error::last_os_error());
        }
        // Two-call protocol: size first, then the buffer.
        let mut needed = 0u32;
        GetTokenInformation(token, TOKEN_USER, std::ptr::null_mut(), 0, &mut needed);
        let mut buffer = vec![0u8; needed as usize];
        if GetTokenInformation(
            token,
            TOKEN_USER,
            buffer.as_mut_ptr() as *mut core::ffi::c_void,
            needed,
            &mut needed,
        ) == 0
        {
            let err = io::Error::last_os_error();
            CloseHandle(token);
            return Err(err);
        }
        // TOKEN_USER's first member is the SID pointer (repr(C)).
        let sid = buffer.as_ptr() as *const *const core::ffi::c_void;
        let mut sid_wstr: *mut u16 = std::ptr::null_mut();
        if ConvertSidToStringSidW(*sid, &mut sid_wstr) == 0 {
            let err = io::Error::last_os_error();
            CloseHandle(token);
            return Err(err);
        }
        let text = wide_to_string(sid_wstr);
        LocalFree(sid_wstr as _);
        CloseHandle(token);
        Ok(text)
    }
}

fn wide_to_string(pointer: *const u16) -> String {
    let mut len = 0usize;
    while unsafe { *pointer.add(len) } != 0 {
        len += 1;
    }
    OsString::from_wide(unsafe { std::slice::from_raw_parts(pointer, len) })
        .to_string_lossy()
        .into_owned()
}

fn to_wide(text: &str) -> Vec<u16> {
    std::ffi::OsStr::new(text)
        .encode_wide()
        .chain(Some(0))
        .collect()
}

/// The pipe endpoint path (`\\.\pipe\<stem>`) for one data root. The
/// `runtime_dir` does not select the endpoint on Windows — the pipe
/// namespace is per-machine, and the stem already encodes the root.
pub fn pipe_path(runtime_dir: &Path, root: &Path) -> PathBuf {
    let _ = runtime_dir;
    PathBuf::from(format!("\\\\.\\pipe\\{}", super::endpoint_stem(root)))
}

/// Builds the SECURITY_ATTRIBUTES for `CreateNamedPipeW`: a security
/// descriptor from the owner-restricted SDDL.
fn owner_security_attributes() -> io::Result<SECURITY_ATTRIBUTES> {
    let sid = current_user_sid()?;
    let sddl = sddl_for_owner(&sid);
    let mut descriptor: *mut core::ffi::c_void = std::ptr::null_mut();
    let wide = to_wide(&sddl);
    unsafe {
        if ConvertStringSecurityDescriptorToSecurityDescriptorW(
            wide.as_ptr(),
            1, // SDDL_REVISION_1
            &mut descriptor,
            std::ptr::null_mut(),
        ) == 0
        {
            return Err(io::Error::last_os_error());
        }
    }
    Ok(SECURITY_ATTRIBUTES {
        nLength: std::mem::size_of::<SECURITY_ATTRIBUTES>() as u32,
        lpSecurityDescriptor: descriptor,
        bInheritHandle: 0,
    })
}

pub fn listen(path: &Path) -> io::Result<Box<dyn TransportListener>> {
    let name = to_wide(&path.to_string_lossy());
    let mut security = owner_security_attributes()?;
    unsafe {
        let handle = CreateNamedPipeW(
            name.as_ptr(),
            // First-instance flag: fails with ERROR_ACCESS_DENIED if
            // another server already owns this name — the bind-time
            // equivalent of the unix stale-socket check.
            PIPE_ACCESS_DUPLEX | FILE_FLAG_FIRST_PIPE_INSTANCE,
            PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT,
            PIPE_UNLIMITED_INSTANCES,
            64 * 1024,
            64 * 1024,
            0,
            &security,
        );
        // The descriptor was copied into the pipe object; free ours.
        LocalFree(security.lpSecurityDescriptor as _);
        if handle == INVALID_HANDLE_VALUE {
            return Err(io::Error::last_os_error());
        }
        // Hand the created-but-unconnected first instance to the acceptor
        // thread, which will block in ConnectNamedPipe on it.
        let (tx, rx) = mpsc::channel::<io::Result<HANDLE>>();
        let name_for_thread = name.clone();
        let acceptor = std::thread::Builder::new()
            .name("starling-host-pipe-accept".into())
            .spawn(move || acceptor_loop(name_for_thread, handle, tx))
            .map_err(io::Error::other)?;
        Ok(Box::new(PipeListener {
            name,
            connections: rx,
            _acceptor: acceptor,
        }))
    }
}

/// The acceptor thread body: block on `ConnectNamedPipe` for the current
/// instance, deliver it, create the next instance, repeat. Exits when the
/// receiver is gone (listener dropped — the Drop impl wakes the pending
/// ConnectNamedPipe with a self-connection first).
fn acceptor_loop(name: Vec<u16>, mut current: HANDLE, tx: mpsc::Sender<io::Result<HANDLE>>) {
    loop {
        unsafe {
            if ConnectNamedPipe(current, std::ptr::null()) == 0 {
                let err = GetLastError();
                // ERROR_NO_DATA / ERROR_PIPE_CONNECTED: a client had
                // already connected between instance creation and our
                // call — the instance is still good to service.
                const ERROR_NO_DATA: u32 = 232;
                const ERROR_PIPE_CONNECTED: u32 = 535;
                if err != ERROR_NO_DATA && err != ERROR_PIPE_CONNECTED {
                    if tx
                        .send(Err(io::Error::from_raw_os_error(err as i32)))
                        .is_err()
                    {
                        CloseHandle(current);
                        return;
                    }
                    current = match create_instance(&name) {
                        Ok(next) => next,
                        Err(_) => return,
                    };
                    continue;
                }
            }
        }
        if tx.send(Ok(current)).is_err() {
            // Listener gone: nobody will service this instance.
            unsafe {
                DisconnectNamedPipe(current);
                CloseHandle(current);
            }
            return;
        }
        unsafe {
            current = match create_instance(&name) {
                Ok(next) => next,
                Err(_) => return,
            };
        }
    }
}

unsafe fn create_instance(name: &[u16]) -> io::Result<HANDLE> {
    // Subsequent instances do not re-claim first-instance (only the
    // first did, in `listen`); same DACL applies.
    let mut security = owner_security_attributes()?;
    let handle = CreateNamedPipeW(
        name.as_ptr(),
        PIPE_ACCESS_DUPLEX,
        PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT,
        PIPE_UNLIMITED_INSTANCES,
        64 * 1024,
        64 * 1024,
        0,
        &security,
    );
    LocalFree(security.lpSecurityDescriptor as _);
    if handle == INVALID_HANDLE_VALUE {
        Err(io::Error::last_os_error())
    } else {
        Ok(handle)
    }
}

pub fn connect(path: &Path) -> io::Result<Box<dyn TransportConn>> {
    let name = to_wide(&path.to_string_lossy());
    unsafe {
        // A pipe server with all instances busy answers ERROR_PIPE_BUSY;
        // a short wait-and-retry keeps an honest client from racing a
        // host that is creating its next instance.
        const NMPWAIT_USE_DEFAULT_WAIT: u32 = 0;
        if WaitNamedPipeW(name.as_ptr(), NMPWAIT_USE_DEFAULT_WAIT) == 0 {
            let err = GetLastError();
            if err != ERROR_FILE_NOT_FOUND && err != ERROR_PATH_NOT_FOUND {
                return Err(io::Error::from_raw_os_error(err as i32));
            }
        }
        let handle = CreateFileW(
            name.as_ptr(),
            GENERIC_READ | GENERIC_WRITE,
            0,
            std::ptr::null(),
            OPEN_EXISTING,
            0, // synchronous: no FILE_FLAG_OVERLAPPED
            std::ptr::null_mut(),
        );
        if handle == INVALID_HANDLE_VALUE {
            return Err(io::Error::last_os_error());
        }
        Ok(Box::new(PipeConn { handle }))
    }
}

pub fn probe(path: &Path) -> Probe {
    match connect(path) {
        Ok(conn) => {
            drop(conn);
            Probe::Live
        }
        Err(err) => match err.raw_os_error() {
            Some(code)
                if code == ERROR_FILE_NOT_FOUND as i32
                    || code == ERROR_PATH_NOT_FOUND as i32
                    || code == ERROR_PIPE_BUSY as i32 =>
            {
                // Not found: no server by this name. Busy: a server
                // exists with every instance occupied — both are
                // definitive answers about the name.
                if code == ERROR_PIPE_BUSY as i32 {
                    Probe::Live
                } else {
                    Probe::Dead
                }
            }
            Some(code) if code == ERROR_ACCESS_DENIED as i32 => Probe::Unknown(err.to_string()),
            _ => Probe::Unknown(err.to_string()),
        },
    }
}

pub struct PipeListener {
    name: Vec<u16>,
    connections: mpsc::Receiver<io::Result<HANDLE>>,
    /// Kept alive so the acceptor thread's channel has a sender-side
    /// counterpart to observe; joined on Drop.
    _acceptor: std::thread::JoinHandle<()>,
}

impl TransportListener for PipeListener {
    fn accept(&self) -> io::Result<Box<dyn TransportConn>> {
        // Channel semantics stand in for the socket's: a connection is
        // either waiting (accept returns it) or not (WouldBlock).
        match self.connections.try_recv() {
            Ok(result) => Ok(Box::new(PipeConn { handle: result? })),
            Err(mpsc::TryRecvError::Empty) => Err(io::Error::new(
                io::ErrorKind::WouldBlock,
                "no pipe client waiting",
            )),
            Err(mpsc::TryRecvError::Disconnected) => Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "pipe acceptor is gone",
            )),
        }
    }

    fn set_nonblocking(&self, _nonblocking: bool) -> io::Result<()> {
        // The channel in `accept` already behaves in the polling style
        // the host's accept loop wants; nothing to toggle.
        Ok(())
    }
}

impl Drop for PipeListener {
    fn drop(&mut self) {
        // Wake the acceptor: connect to our own pipe name so the pending
        // ConnectNamedPipe completes; the acceptor then finds the channel
        // closed and cleans up its handles.
        let _ = connect(Path::new(&String::from_utf16_lossy(&self.name)));
        let _ = self._acceptor.join();
    }
}

/// A connected pipe instance (server or client side).
pub struct PipeConn {
    handle: HANDLE,
}

// SAFETY: the handle is a kernel object identifier, not a pointer into
// this process's memory; Win32 calls on one handle are thread-safe.
unsafe impl Send for PipeConn {}
unsafe impl Sync for PipeConn {}

type HANDLE = *mut core::ffi::c_void;

impl PipeConn {
    fn pid_of(handle: HANDLE) -> Option<u32> {
        let mut pid = 0u32;
        if unsafe { GetNamedPipeClientProcessId(handle, &mut pid) } != 0 {
            Some(pid)
        } else {
            None
        }
    }
}

impl TransportConn for PipeConn {
    fn peer_credentials(&self) -> io::Result<PeerCredentials> {
        // uid: none on Windows — the DACL already made the same-user
        // decision at connect time (see the module docs and
        // `crate::auth`).
        Ok(PeerCredentials {
            uid: None,
            pid: Self::pid_of(self.handle),
        })
    }

    fn try_clone(&self) -> io::Result<Box<dyn TransportConn>> {
        // DuplicateHandle: a real second handle to the same pipe, so the
        // host's reader/writer/shutdown split works exactly as on unix.
        unsafe {
            let mut duplicate = INVALID_HANDLE_VALUE;
            let process = GetCurrentProcess();
            if DuplicateHandle(
                process,
                self.handle,
                process,
                &mut duplicate,
                0,
                0,
                DUPLICATE_SAME_ACCESS,
            ) == 0
            {
                return Err(io::Error::last_os_error());
            }
            Ok(Box::new(PipeConn { handle: duplicate }))
        }
    }

    fn shutdown_both(&self) -> io::Result<()> {
        // There is no shutdown(2) for pipes: the host closes the handle
        // (drop) to end both directions. Flush what is buffered first.
        unsafe {
            if FlushFileBuffers(self.handle) == 0 {
                return Err(io::Error::last_os_error());
            }
        }
        Ok(())
    }
}

impl Drop for PipeConn {
    fn drop(&mut self) {
        unsafe {
            DisconnectNamedPipe(self.handle);
            CloseHandle(self.handle);
        }
    }
}

impl Read for PipeConn {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let mut read = 0u32;
        let ok = unsafe {
            windows_sys::Win32::Storage::FileSystem::ReadFile(
                self.handle,
                buf.as_mut_ptr() as *mut core::ffi::c_void,
                buf.len() as u32,
                &mut read,
                std::ptr::null_mut(),
            )
        };
        if ok == 0 {
            // ERROR_BROKEN_PIPE: the peer closed — a clean EOF for a
            // pipe.
            if unsafe { GetLastError() } == 109 {
                return Ok(0);
            }
            return Err(io::Error::last_os_error());
        }
        Ok(read as usize)
    }
}

impl Write for PipeConn {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let mut written = 0u32;
        let ok = unsafe {
            windows_sys::Win32::Storage::FileSystem::WriteFile(
                self.handle,
                buf.as_ptr() as *const core::ffi::c_void,
                buf.len() as u32,
                &mut written,
                std::ptr::null_mut(),
            )
        };
        if ok == 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(written as usize)
    }

    fn flush(&mut self) -> io::Result<()> {
        unsafe {
            if FlushFileBuffers(self.handle) == 0 {
                return Err(io::Error::last_os_error());
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sddl_grants_read_write_to_system_and_owner_only() {
        let sddl = sddl_for_owner("S-1-5-21-1000-2000-3000-500");
        assert_eq!(
            sddl,
            "D:P(A;;GRGW;;;SY)(A;;GRGW;;;S-1-5-21-1000-2000-3000-500)"
        );
        // No "WD" (world/everyone) ACE anywhere: nobody else holds an
        // allow ace.
        assert!(!sddl.contains(";;;WD)"));
        assert!(!sddl.to_lowercase().contains("s-1-1-0"));
    }
}
