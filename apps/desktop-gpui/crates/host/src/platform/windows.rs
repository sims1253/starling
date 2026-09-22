//! The Windows transport: a named pipe with a restrictive DACL.
//!
//! **Compile-verified, runtime-untested (recorded gap):** this module
//! (with `auth` and the platform surface it sits on) is type-checked
//! against `x86_64-pc-windows-msvc` on Linux — the full crate cannot
//! cross-check there (the bundled-SQLite C dependency needs an MSVC C
//! toolchain), so the exact sources are checked in a dependency-free
//! scratch crate — and the whole crate is compiled on every push by the
//! `windows-check` CI job on a native Windows runner. But no Windows
//! machine has *executed* this code — the runtime tests for it are
//! equally absent. The code follows the documented Win32 named-pipe
//! semantics; its first Windows run is expected to be its first run,
//! not its first compile.
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
//! avoids overlapped I/O at the cost of one wake connection at shutdown.
//!
//! Recorded gaps (compile-verified shapes, unrun): a synchronous
//! `ReadFile` cannot poll (see [`PipeConn::set_read_timeout`]), and
//! canceling a parked synchronous write relies on the instance-wide
//! `DisconnectNamedPipe` in [`PipeConn::shutdown_both`].

use std::ffi::OsString;
use std::io::{self, Read, Write};
use std::os::windows::ffi::{OsStrExt, OsStringExt};
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use windows_sys::Win32::Foundation::{
    CloseHandle, DuplicateHandle, GetLastError, LocalFree, DUPLICATE_SAME_ACCESS, ERROR_BROKEN_PIPE,
    ERROR_FILE_NOT_FOUND, ERROR_NO_DATA, ERROR_PATH_NOT_FOUND, ERROR_PIPE_BUSY,
    ERROR_PIPE_CONNECTED, ERROR_SEM_TIMEOUT, GENERIC_READ, GENERIC_WRITE, HANDLE,
    INVALID_HANDLE_VALUE,
};
use windows_sys::Win32::Security::Authorization::{
    ConvertSidToStringSidW, ConvertStringSecurityDescriptorToSecurityDescriptorW,
};
use windows_sys::Win32::Security::{
    GetTokenInformation, TokenUser, PSID, PSECURITY_DESCRIPTOR, SECURITY_ATTRIBUTES, TOKEN_QUERY,
    TOKEN_USER,
};
use windows_sys::Win32::Storage::FileSystem::{
    CreateFileW, ReadFile, WriteFile, FILE_FLAG_FIRST_PIPE_INSTANCE, OPEN_EXISTING,
    PIPE_ACCESS_DUPLEX,
};
use windows_sys::Win32::System::IO::CancelIoEx;
use windows_sys::Win32::System::Pipes::{
    ConnectNamedPipe, CreateNamedPipeW, DisconnectNamedPipe, GetNamedPipeClientProcessId,
    NMPWAIT_NOWAIT, NMPWAIT_USE_DEFAULT_WAIT, PIPE_READMODE_BYTE, PIPE_TYPE_BYTE,
    PIPE_UNLIMITED_INSTANCES, PIPE_WAIT, WaitNamedPipeW,
};
use windows_sys::Win32::System::Threading::{GetCurrentProcess, OpenProcessToken};

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

/// A raw pipe handle wrapped for movement across threads. windows-sys
/// 0.59's `HANDLE` is a raw pointer (not `Send` by construction), but the
/// value is a kernel object identifier, not a pointer into this
/// process's memory — Win32 calls on one handle are thread-safe.
struct SendHandle(HANDLE);

// SAFETY: the handle is a kernel object identifier, not a pointer into
// this process's memory; Win32 calls on one handle are thread-safe.
unsafe impl Send for SendHandle {}

/// The current process user's SID, as a string (S-1-5-21-…).
fn current_user_sid() -> io::Result<String> {
    unsafe {
        let mut token: HANDLE = std::ptr::null_mut();
        if OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &mut token) == 0 {
            return Err(io::Error::last_os_error());
        }
        // Two-call protocol: the sizing call fails with
        // ERROR_INSUFFICIENT_BUFFER and sets `needed`. Any other outcome
        // (or a zero size) must be a real failure — an empty buffer
        // would make the second call fail confusingly.
        let mut needed = 0u32;
        if GetTokenInformation(token, TokenUser, std::ptr::null_mut(), 0, &mut needed) != 0
            || needed == 0
        {
            let err = io::Error::last_os_error();
            CloseHandle(token);
            return Err(err);
        }
        let mut buffer = vec![0u8; needed as usize];
        if GetTokenInformation(
            token,
            TokenUser,
            buffer.as_mut_ptr() as *mut core::ffi::c_void,
            needed,
            &mut needed,
        ) == 0
        {
            let err = io::Error::last_os_error();
            CloseHandle(token);
            return Err(err);
        }
        // The cast below dereferences a TOKEN_USER out of `buffer`; a
        // short buffer would make that an out-of-bounds read. The API
        // contract says the user's TOKEN_USER is at least
        // size_of::<TOKEN_USER>() — refuse anything smaller rather than
        // trust it.
        if (needed as usize) < std::mem::size_of::<TOKEN_USER>() {
            CloseHandle(token);
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "GetTokenInformation returned a truncated TOKEN_USER",
            ));
        }
        // TOKEN_USER is repr(C): { SID_AND_ATTRIBUTES { Sid: PSID, .. } }
        // — the SID pointer rides at the buffer's start.
        let user = &*(buffer.as_ptr() as *const TOKEN_USER);
        let sid: PSID = user.User.Sid;
        let mut sid_wstr: *mut u16 = std::ptr::null_mut();
        if ConvertSidToStringSidW(sid, &mut sid_wstr) == 0 {
            let err = io::Error::last_os_error();
            CloseHandle(token);
            return Err(err);
        }
        // SAFETY (the wide_to_string scan): ConvertSidToStringSidW's
        // contract is an allocated, NUL-terminated wide string — the
        // scan walks exactly up to that terminator and stops; the
        // allocation is released by the LocalFree below.
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

/// The pipe endpoint path (`\\.\pipe\<stem>-<user>`) for one data root.
/// The `runtime_dir` does not select the endpoint on Windows — the pipe
/// namespace is per-machine — so the stem encodes the root **and the
/// creating user's SID**: two local users whose data roots resolve to
/// the same path (a shared checkout) must never contend for one pipe
/// name (the DACL would already keep them off each other's pipes, but
/// the first-instance claim would make the second user's bind fail).
/// Both host and client derive the name in-process, so the SID is
/// stable for the pair that matters.
pub fn pipe_path(runtime_dir: &Path, root: &Path) -> PathBuf {
    let _ = runtime_dir;
    let user = current_user_sid().unwrap_or_default();
    PathBuf::from(format!(
        "\\\\.\\pipe\\{}-{}",
        super::endpoint_stem(root),
        super::endpoint_stem(Path::new(&user))
    ))
}

/// Builds the SECURITY_ATTRIBUTES for `CreateNamedPipeW`: a security
/// descriptor from the owner-restricted SDDL.
fn owner_security_attributes() -> io::Result<SECURITY_ATTRIBUTES> {
    let sid = current_user_sid()?;
    let sddl = sddl_for_owner(&sid);
    let mut descriptor: PSECURITY_DESCRIPTOR = std::ptr::null_mut();
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
    let security = owner_security_attributes()?;
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
        // thread, which will block in ConnectNamedPipe on it. The handle
        // crosses as a SendHandle — wrapped *before* the closure so the
        // closure captures the (Send) wrapper, not the raw pointer. The
        // handoff channel is bounded: a burst of clients completing
        // ConnectNamedPipe faster than the host's accept loop polls must
        // park the acceptor in `send` (backpressure at connection time)
        // rather than accumulate unbounded connected instances.
        let first = SendHandle(handle);
        let (tx, rx) = mpsc::sync_channel::<io::Result<SendHandle>>(4);
        let name_for_thread = name.clone();
        let acceptor = std::thread::Builder::new()
            .name("starling-host-pipe-accept".into())
            .spawn(move || acceptor_loop(name_for_thread, first, tx))
            .map_err(|err| {
                // The acceptor will never service (or close) the
                // instance: close it here or it leaks for the process
                // lifetime. (Inside this fn's outer unsafe block.)
                CloseHandle(handle);
                io::Error::other(err)
            })?;
        Ok(Box::new(PipeListener {
            name,
            connections: Some(rx),
            _acceptor: acceptor,
        }))
    }
}

/// The acceptor thread body: block on `ConnectNamedPipe` for the current
/// instance, deliver it, create the next instance, repeat. Exits when the
/// receiver is gone (listener dropped — the Drop impl wakes the pending
/// ConnectNamedPipe with a self-connection first).
fn acceptor_loop(
    name: Vec<u16>,
    mut current: SendHandle,
    tx: mpsc::SyncSender<io::Result<SendHandle>>,
) {
    loop {
        let connected = unsafe { ConnectNamedPipe(current.0, std::ptr::null_mut()) } != 0;
        if !connected {
            let err = unsafe { GetLastError() };
            // ERROR_PIPE_CONNECTED: a client completed the connection
            // between instance creation and this call — the instance is
            // good to service, fall through and deliver it.
            if err != ERROR_PIPE_CONNECTED {
                if err == ERROR_NO_DATA {
                    // The client connected and already went away: no
                    // usable connection exists on this instance. Discard
                    // it and listen on a fresh one rather than hand
                    // `accept()` a dead handle.
                    unsafe {
                        DisconnectNamedPipe(current.0);
                        CloseHandle(current.0);
                    }
                    current = match unsafe { create_instance(&name) } {
                        Ok(next) => next,
                        Err(err) => {
                            // Terminal: without a listening instance the
                            // pipe name stops existing. Log it — on a
                            // path that is compile-verified but has never
                            // run, the first failure must be diagnosable,
                            // not a silent BrokenPipe one accept later.
                            eprintln!("starling-host-pipe-accept: terminating: {err}");
                            return;
                        }
                    };
                    continue;
                }
                // A real acceptor failure: report it, close the failed
                // instance (its handle is ours alone now — the error we
                // sent never carried it), listen on the next. When even
                // the report cannot be delivered the listener is gone:
                // nobody else will close the instance.
                let reported = io::Error::from_raw_os_error(err as i32);
                let listener_gone = tx.send(Err(reported)).is_err();
                unsafe {
                    DisconnectNamedPipe(current.0);
                    CloseHandle(current.0);
                }
                if listener_gone {
                    return;
                }
                current = match unsafe { create_instance(&name) } {
                    Ok(next) => next,
                    Err(err) => {
                        eprintln!("starling-host-pipe-accept: terminating: {err}");
                        return;
                    }
                };
                continue;
            }
        }
        match tx.send(Ok(current)) {
            Ok(()) => {}
            Err(sent) => {
                // Listener gone: nobody will service this instance. The
                // value bounced back is the one we tried to send — the
                // connected instance by construction.
                if let Ok(SendHandle(dead)) = sent.0 {
                    unsafe {
                        DisconnectNamedPipe(dead);
                        CloseHandle(dead);
                    }
                }
                return;
            }
        }
        current = match unsafe { create_instance(&name) } {
            Ok(next) => next,
            Err(err) => {
                eprintln!("starling-host-pipe-accept: terminating: {err}");
                return;
            }
        };
    }
}

unsafe fn create_instance(name: &[u16]) -> io::Result<SendHandle> {
    // Subsequent instances do not re-claim first-instance (only the
    // first did, in `listen`); same DACL applies.
    let security = owner_security_attributes()?;
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
        Ok(SendHandle(handle))
    }
}

pub fn connect(path: &Path) -> io::Result<Box<dyn TransportConn>> {
    connect_with_wait(path, NMPWAIT_USE_DEFAULT_WAIT)
}

/// [`connect`] with an explicit `WaitNamedPipeW` mode. The probe path
/// passes [`NMPWAIT_NOWAIT`] so probing a busy server stays non-blocking
/// (a busy pipe is a `Live` answer, not something to wait out).
fn connect_with_wait(path: &Path, wait: u32) -> io::Result<Box<dyn TransportConn>> {
    let name = to_wide(&path.to_string_lossy());
    unsafe {
        // A pipe server with all instances busy answers ERROR_PIPE_BUSY;
        // a wait-and-retry keeps an honest client from racing a host
        // that is creating its next instance.
        if WaitNamedPipeW(name.as_ptr(), wait) == 0 {
            let err = GetLastError();
            // Not-found and busy-timeout fall through to CreateFileW,
            // which gives the canonical decisive answer for both (not
            // found; busy with ERROR_PIPE_BUSY). Returning early on
            // SEM_TIMEOUT would rob the probe of its Live answer for a
            // busy-but-healthy server.
            if err != ERROR_FILE_NOT_FOUND
                && err != ERROR_PATH_NOT_FOUND
                && err != ERROR_SEM_TIMEOUT
            {
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
        Ok(Box::new(PipeConn {
            handle: SendHandle(handle),
            server: false,
        }))
    }
}

pub fn probe(path: &Path) -> Probe {
    match connect_with_wait(path, NMPWAIT_NOWAIT) {
        Ok(conn) => {
            drop(conn);
            Probe::Live
        }
        Err(err) => match err.raw_os_error() {
            Some(code)
                if code == ERROR_FILE_NOT_FOUND as i32 || code == ERROR_PATH_NOT_FOUND as i32 =>
            {
                Probe::Dead
            }
            // Busy: a server exists with every instance occupied — a
            // definitive answer about the name (Live).
            Some(code) if code == ERROR_PIPE_BUSY as i32 => Probe::Live,
            // Everything else (ACCESS_DENIED included): liveness could
            // not be tested; the caller fails closed.
            _ => Probe::Unknown(err.to_string()),
        },
    }
}

pub struct PipeListener {
    name: Vec<u16>,
    /// `Option` so `Drop` can end the channel **before** joining the
    /// acceptor (see [`Drop for PipeListener`]).
    connections: Option<mpsc::Receiver<io::Result<SendHandle>>>,
    /// Kept alive so the acceptor thread's channel has a sender-side
    /// counterpart to observe; joined on Drop.
    _acceptor: std::thread::JoinHandle<()>,
}

impl TransportListener for PipeListener {
    fn accept(&self) -> io::Result<Box<dyn TransportConn>> {
        // Channel semantics stand in for the socket's: a connection is
        // either waiting (accept returns it) or not (WouldBlock).
        let connections = self
            .connections
            .as_ref()
            .expect("the receiver lives until Drop");
        match connections.try_recv() {
            Ok(result) => Ok(Box::new(PipeConn {
                handle: result?,
                server: true,
            })),
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
        // Documented divergence from the trait's toggle: this listener is
        // permanently non-blocking — the channel behind `accept` answers
        // in exactly the polling style the host's accept loop wants, and
        // there is no blocking mode to switch to. The only caller (the
        // host's loop) always polls.
        Ok(())
    }
}

impl Drop for PipeListener {
    fn drop(&mut self) {
        // Order matters. (1) End the channel first: once the wake
        // connection below completes the acceptor's pending
        // ConnectNamedPipe, its `send` must fail so it exits — with the
        // receiver still alive it would instead create the *next*
        // instance and park on it forever, and the join would hang even
        // on this happy path.
        drop(self.connections.take());
        // (2) Wake the acceptor: connect to our own pipe name so the
        // pending ConnectNamedPipe completes.
        let _ = connect(Path::new(&String::from_utf16_lossy(&self.name)));
        // (3) Bounded wait: a wake that cannot reach the acceptor
        // (create_instance failed, the name is gone) must not hang the
        // dropping thread — the acceptor is left detached instead.
        //
        // A detached acceptor may still hold a claimed pipe instance,
        // but only until this process exits: pipe handles are kernel
        // objects the OS closes at exit, and a listener is only ever
        // dropped on the host's shutdown path (there is no in-process
        // re-listen after this in the host's lifetime — the lease ladder
        // restarts serving in a new process).
        let deadline = Instant::now() + Duration::from_secs(2);
        while !self._acceptor.is_finished() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}

/// A connected pipe instance. `server` marks handles whose instance
/// this process created with `CreateNamedPipeW` — only those may be
/// passed to `DisconnectNamedPipe` (client-side handles from
/// `CreateFileW` are closed, never disconnected).
///
/// Duplicates keep the flag, on purpose: `DisconnectNamedPipe` is an
/// **instance-wide** operation reachable from any handle of the
/// instance, and the host's force-close paths (the eviction, the
/// bounded shutdown drain, overflow closes) run on the `closer`
/// duplicate precisely because that is the one call that unblocks a
/// writer parked in a synchronous `WriteFile` on this instance. The
/// truncation that disconnects mid-write is the designed bound, not an
/// accident: every such close runs only after the peer was given its
/// bounded chance to drain (see `SHUTDOWN_DRAIN` and the
/// close-on-overflow docs in `server.rs`).
pub struct PipeConn {
    handle: SendHandle,
    server: bool,
}

// SAFETY (Send and Sync): the handle is a kernel object identifier, not
// a pointer into this process's memory. Send: Win32 calls on one handle
// are thread-safe. Sync: `PipeConn` is shared across the host's
// reader/writer/closer threads through *duplicated* handle values, and
// concurrent synchronous ReadFile/WriteFile on distinct duplicates of
// one pipe instance is documented-safe; DisconnectNamedPipe/CancelIoEx
// racing in-flight I/O is likewise documented to fail those operations,
// not corrupt state.
unsafe impl Send for PipeConn {}
unsafe impl Sync for PipeConn {}

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
            pid: Self::pid_of(self.handle.0),
        })
    }

    fn try_clone(&self) -> io::Result<Box<dyn TransportConn>> {
        // DuplicateHandle: a real second handle to the same pipe, so the
        // host's reader/writer/shutdown split works exactly as on unix.
        unsafe {
            let mut duplicate: HANDLE = std::ptr::null_mut();
            let process = GetCurrentProcess();
            if DuplicateHandle(
                process,
                self.handle.0,
                process,
                &mut duplicate,
                0,
                0,
                DUPLICATE_SAME_ACCESS,
            ) == 0
            {
                return Err(io::Error::last_os_error());
            }
            Ok(Box::new(PipeConn {
                handle: SendHandle(duplicate),
                server: self.server,
            }))
        }
    }

    fn shutdown_both(&self) -> io::Result<()> {
        // There is no shutdown(2) for pipes, and the nearest flush
        // (FlushFileBuffers) BLOCKS on a full pipe — the exact stall the
        // callers (the event pump's eviction, bounded shutdown) must
        // never cause. Server instances: DisconnectNamedPipe forces the
        // disconnect; it is instance-wide, so pending operations on
        // every handle of the instance (a writer parked in a
        // synchronous WriteFile among them) complete with an error —
        // which is what unblocks them.
        //
        // Client handles: CancelIoEx cancels I/O issued on THIS handle
        // value only — it does not reach the reader's separately
        // duplicated handle, so a parked client-side ReadFile is NOT
        // unblocked here (recorded gap: converting the client read side
        // to overlapped I/O, or reading through one shared handle, is
        // the prerequisite). The client reader does end when the host
        // closes its end (broken pipe), which is the normal teardown
        // path; only the hostile-host wedge relies on this gap.
        unsafe {
            if self.server {
                if DisconnectNamedPipe(self.handle.0) == 0 {
                    return Err(io::Error::last_os_error());
                }
            } else if CancelIoEx(self.handle.0, std::ptr::null()) == 0 {
                return Err(io::Error::last_os_error());
            }
        }
        Ok(())
    }

    fn set_read_timeout(&self, _timeout: Option<Duration>) -> io::Result<()> {
        // Recorded gap: a synchronous (non-overlapped) ReadFile cannot
        // poll; converting the read side to overlapped I/O is the
        // prerequisite. Until then the client library's event backlog
        // flushes on inbound frames instead of on an idle tick (see
        // `client.rs`).
        Ok(())
    }

    fn set_write_timeout(&self, _timeout: Option<Duration>) -> io::Result<()> {
        // Recorded gap, same class as set_read_timeout: a synchronous
        // WriteFile cannot be bounded. The design compensates — the
        // host's writers live on dedicated connection threads (a parked
        // write costs its own connection, never the accept loop), and
        // the force-close paths unblock them via the instance-wide
        // DisconnectNamedPipe above.
        Ok(())
    }
}

impl Drop for PipeConn {
    fn drop(&mut self) {
        // Only server-side instances are disconnected; both sides'
        // handles are closed.
        unsafe {
            if self.server {
                DisconnectNamedPipe(self.handle.0);
            }
            CloseHandle(self.handle.0);
        }
    }
}

impl Read for PipeConn {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let mut read = 0u32;
        let ok = unsafe {
            ReadFile(
                self.handle.0,
                buf.as_mut_ptr(),
                buf.len() as u32,
                &mut read,
                std::ptr::null_mut(),
            )
        };
        if ok == 0 {
            // ERROR_BROKEN_PIPE: the peer closed — a clean EOF for a
            // pipe.
            if unsafe { GetLastError() } == ERROR_BROKEN_PIPE {
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
            WriteFile(
                self.handle.0,
                buf.as_ptr(),
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
        // Byte-mode synchronous writes are delivered by the time
        // WriteFile returns; FlushFileBuffers would add nothing but a
        // blocking wait against a peer that is not reading.
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
