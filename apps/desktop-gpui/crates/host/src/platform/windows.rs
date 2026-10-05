//! The Windows transport: a named pipe with a restrictive DACL.
//!
//! **Runtime-verified (#220):** the host's portable IPC, renderer-kill
//! and adapter suites run against this transport on a native Windows
//! runner (the `windows-test` CI job) as well as on unix. Their first
//! Windows run found two defects the compile-only check could not: the
//! DACL read-back freed an interior pointer (heap corruption), and
//! synchronous pipe I/O deadlocked — see the I/O model below.
//!
//! Security posture (what "authenticated" means on Windows — there is no
//! `SO_PEERCRED` analogue): the pipe is created with a security
//! descriptor whose DACL grants access to exactly the creating user's
//! SID and the system (`D:P(A;;GRGW;;;SY)(A;;GRGW;;;<sid>)`). The kernel
//! enforces the DACL on every `CreateFileW` against the pipe name, so a
//! connection this server accepts has already been proven to run as the
//! creating user — the same-user decision the unix side makes from
//! `SO_PEERCRED`, made here at object-creation time instead. As defense
//! in depth, `listen` reads the created instance's DACL back from the
//! kernel and refuses to serve unless it holds only allow-ACEs for
//! narrow principals (`verify_pipe_dacl`): a construction regression
//! fails the bind, not the first foreign connection. The client pid
//! (`GetNamedPipeClientProcessId`) is captured for diagnostics.
//!
//! I/O model: **overlapped**, on both ends. Every handle is opened with
//! `FILE_FLAG_OVERLAPPED`, because synchronous I/O is serialized per
//! *file object* — and `DuplicateHandle` shares the file object, so a
//! reader parked in a synchronous `ReadFile` blocks the writer's
//! `WriteFile` on its duplicate until a byte arrives (the host's
//! reader/writer split deadlocked on its first Windows run). Each
//! `read`/`write` issues one overlapped operation and waits for it in
//! bounded slices, which also gives the transport real read and write
//! timeouts (the trait's poll contract) and a cooperative close:
//! [`TransportConn::shutdown_both`] marks the connection closed (shared
//! by every duplicate) and cancels its I/O, and a waiter that sees the
//! mark cancels its own operation — no parked call outlives a close.
//!
//! Server shape: a dedicated acceptor thread waits on an overlapped
//! `ConnectNamedPipe` for the current pipe instance and hands connected
//! instances to `accept()` through a channel. Shutdown wakes the
//! acceptor by connecting to the pipe as a client — the pending connect
//! completes with that self-connection, the acceptor finds its channel
//! closed, and it closes its handles and exits.

use std::ffi::OsString;
use std::io::{self, Read, Write};
use std::os::windows::ffi::{OsStrExt, OsStringExt};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};

use windows_sys::Win32::Foundation::{
    CloseHandle, DuplicateHandle, GetLastError, LocalFree, BOOL, DUPLICATE_SAME_ACCESS,
    ERROR_BROKEN_PIPE, ERROR_FILE_NOT_FOUND, ERROR_IO_PENDING, ERROR_NO_DATA,
    ERROR_OPERATION_ABORTED, ERROR_PATH_NOT_FOUND, ERROR_PIPE_BUSY, ERROR_PIPE_CONNECTED,
    ERROR_PIPE_NOT_CONNECTED,
    ERROR_SEM_TIMEOUT, GENERIC_READ, GENERIC_WRITE, HANDLE, INVALID_HANDLE_VALUE, WAIT_OBJECT_0,
    WAIT_TIMEOUT,
};
use windows_sys::Win32::Security::Authorization::{
    ConvertSidToStringSidW, ConvertStringSecurityDescriptorToSecurityDescriptorW, GetSecurityInfo,
    SE_KERNEL_OBJECT,
};
use windows_sys::Win32::Security::{
    GetAce, GetTokenInformation, TokenUser, ACCESS_ALLOWED_ACE, ACE_HEADER, ACL, PSID,
    PSECURITY_DESCRIPTOR, DACL_SECURITY_INFORMATION, SECURITY_ATTRIBUTES, TOKEN_QUERY, TOKEN_USER,
};
use windows_sys::Win32::Storage::FileSystem::{
    CreateFileW, FlushFileBuffers, ReadFile, WriteFile, FILE_FLAG_FIRST_PIPE_INSTANCE, FILE_FLAG_OVERLAPPED,
    OPEN_EXISTING, PIPE_ACCESS_DUPLEX,
};
use windows_sys::Win32::System::IO::{CancelIoEx, GetOverlappedResult, OVERLAPPED};
use windows_sys::Win32::System::Pipes::{
    ConnectNamedPipe, CreateNamedPipeW, DisconnectNamedPipe, GetNamedPipeClientProcessId,
    NMPWAIT_NOWAIT, NMPWAIT_USE_DEFAULT_WAIT, PIPE_READMODE_BYTE, PIPE_TYPE_BYTE,
    PIPE_UNLIMITED_INSTANCES, PIPE_WAIT, WaitNamedPipeW,
};
use windows_sys::Win32::System::Threading::{
    CreateEventW, GetCurrentProcess, OpenProcessToken, WaitForSingleObject,
};

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

/// The allow-ACE type this DACL is built from (`ACCESS_ALLOWED_ACE_TYPE`
/// is not exported by windows-sys 0.59).
const ACCESS_ALLOWED_ACE_TYPE: u8 = 0;

/// Fail-closed self-check on the pipe's **actual** DACL, read back from
/// the kernel at bind time. The SDDL string says one thing, but the
/// entire admission guarantee on Windows rests on the descriptor the
/// API actually attached — a regression in the construction path (a
/// loosely-accepted malformed SDDL, a future edit) must fail the bind,
/// not silently admit every local user to a host that owns the user's
/// audio data. The check: the DACL exists, every ACE is an allow-ACE,
/// and no allow-ACE grants a broad principal — Everyone (`S-1-1-0`),
/// Anonymous (`S-1-5-7`), or BUILTIN\Users (`S-1-5-32-545`). The only
/// acceptable grants are the ones the SDDL names: SYSTEM and the owner.
fn verify_pipe_dacl(handle: HANDLE) -> io::Result<()> {
    unsafe {
        let mut dacl: *mut ACL = std::ptr::null_mut();
        let mut descriptor: PSECURITY_DESCRIPTOR = std::ptr::null_mut();
        // `ppSecurityDescriptor` is mandatory whenever `ppDacl` is asked
        // for: the DACL pointer handed back points *into* that one
        // allocated descriptor, which is the only thing LocalFree may
        // release. (Freeing the interior DACL pointer instead corrupted
        // the process heap on the first Windows run.)
        let result = GetSecurityInfo(
            handle,
            SE_KERNEL_OBJECT,
            DACL_SECURITY_INFORMATION,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            &mut dacl,
            std::ptr::null_mut(),
            &mut descriptor,
        );
        if result != 0 {
            return Err(io::Error::from_raw_os_error(result as i32));
        }
        let check = verify_dacl_aces(dacl);
        LocalFree(descriptor as _);
        check
    }
}

/// The ACE-walking half of [`verify_pipe_dacl`], split out so the
/// LocalFree above runs on every path.
fn verify_dacl_aces(dacl: *mut ACL) -> io::Result<()> {
    // A NULL DACL means "everyone, full access" — the worst possible
    // reading, refuse it outright.
    if dacl.is_null() {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "pipe carries no DACL; refusing to serve on it",
        ));
    }
    unsafe {
        let ace_count = (*dacl).AceCount;
        for index in 0..ace_count {
            let mut ace: *mut core::ffi::c_void = std::ptr::null_mut();
            if GetAce(dacl, index as u32, &mut ace) == 0 {
                return Err(io::Error::last_os_error());
            }
            // SAFETY: GetAce hands back a pointer into the ACL's ACE
            // array; the header is the common prefix of every ACE shape.
            let header = &*(ace as *const ACE_HEADER);
            if header.AceType != ACCESS_ALLOWED_ACE_TYPE {
                return Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    format!(
                        "unexpected ACE type {} in the pipe DACL; refusing to serve",
                        header.AceType
                    ),
                ));
            }
            // SAFETY: for an allow-ACE the shape is
            // { header, mask, sid… } — SidStart is the SID's first
            // dword, and the SID runs to the ACE's end.
            let allowed = &*(ace as *const ACCESS_ALLOWED_ACE);
            let sid: PSID = std::ptr::from_ref(&allowed.SidStart) as PSID;
            let mut sid_wstr: *mut u16 = std::ptr::null_mut();
            if ConvertSidToStringSidW(sid, &mut sid_wstr) == 0 {
                return Err(io::Error::last_os_error());
            }
            // SAFETY: ConvertSidToStringSidW's contract is an allocated,
            // NUL-terminated wide string; the scan stops at that
            // terminator and the LocalFree below releases it.
            let text = wide_to_string(sid_wstr);
            LocalFree(sid_wstr as _);
            if matches!(
                text.as_str(),
                "S-1-1-0" | "S-1-5-7" | "S-1-5-32-545"
            ) {
                return Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    format!(
                        "pipe DACL grants broad principal {text}; \
                         refusing to serve on it"
                    ),
                ));
            }
        }
    }
    Ok(())
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
            PIPE_ACCESS_DUPLEX | FILE_FLAG_FIRST_PIPE_INSTANCE | FILE_FLAG_OVERLAPPED,
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
        // Defense in depth (fail closed): the admission guarantee rests
        // on the DACL the kernel actually attached, so read it back and
        // verify it grants no broad principal before serving anyone. A
        // construction regression fails the bind here, not the first
        // foreign connection.
        if let Err(err) = verify_pipe_dacl(handle) {
            CloseHandle(handle);
            return Err(err);
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
        let connected = unsafe { connect_instance(current.0) };
        if let Err(err) = connected {
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

/// An owned manual-reset event for one overlapped operation.
struct Event(HANDLE);

impl Event {
    fn new() -> io::Result<Event> {
        let handle = unsafe { CreateEventW(std::ptr::null(), 1, 0, std::ptr::null()) };
        if handle.is_null() {
            Err(io::Error::last_os_error())
        } else {
            Ok(Event(handle))
        }
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        unsafe {
            CloseHandle(self.0);
        }
    }
}

/// Waits for a client on an overlapped pipe instance. `Ok` when one is
/// connected (including `ERROR_PIPE_CONNECTED`: it connected before the
/// call); `Err(code)` otherwise. Blocks for as long as no client comes —
/// the acceptor thread's job; the listener's Drop wakes it with a
/// self-connection.
unsafe fn connect_instance(handle: HANDLE) -> Result<(), u32> {
    let event = Event::new().map_err(|err| err.raw_os_error().unwrap_or(0) as u32)?;
    let mut overlapped: OVERLAPPED = std::mem::zeroed();
    overlapped.hEvent = event.0;
    if ConnectNamedPipe(handle, &mut overlapped) != 0 {
        return Ok(());
    }
    match GetLastError() {
        ERROR_PIPE_CONNECTED => Ok(()),
        ERROR_IO_PENDING => {
            // The kernel owns `overlapped` until completion: wait it out
            // (bWait = TRUE) before the frame (and the event) go away.
            let mut ignored = 0u32;
            if GetOverlappedResult(handle, &overlapped, &mut ignored, 1) != 0 {
                Ok(())
            } else {
                Err(GetLastError())
            }
        }
        err => Err(err),
    }
}

unsafe fn create_instance(name: &[u16]) -> io::Result<SendHandle> {
    // Subsequent instances do not re-claim first-instance (only the
    // first did, in `listen`); same DACL applies.
    let security = owner_security_attributes()?;
    let handle = CreateNamedPipeW(
        name.as_ptr(),
        PIPE_ACCESS_DUPLEX | FILE_FLAG_OVERLAPPED,
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
            FILE_FLAG_OVERLAPPED, // see the module docs' I/O model
            std::ptr::null_mut(),
        );
        if handle == INVALID_HANDLE_VALUE {
            return Err(io::Error::last_os_error());
        }
        Ok(Box::new(PipeConn {
            handle: SendHandle(handle),
            server: false,
            shared: Arc::default(),
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
            Ok(result) => {
                let conn = PipeConn {
                    handle: result?,
                    server: true,
                    shared: Arc::default(),
                };
                // Readers run with a poll timeout (the unix accept arms
                // the same 250 ms) so a connection the host abandoned —
                // or one that never speaks (the pre-greeting idle bound)
                // — never parks its reader forever.
                conn.set_read_timeout(Some(Duration::from_millis(250)))?;
                Ok(Box::new(conn))
            }
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
/// Duplicates keep the flag, on purpose: the host's force-close paths
/// (the eviction, the bounded shutdown drain, overflow closes) run on
/// the `closer` duplicate, and its `shutdown_both` must reach the whole
/// instance — the close mark is shared, and the lingering disconnect is
/// instance-wide. A writer parked in `WriteFile` is released by the mark
/// within `CLOSE_POLL`; the peer keeps `DISCONNECT_LINGER` to read what
/// was written before the instance is cut.
pub struct PipeConn {
    handle: SendHandle,
    server: bool,
    /// Timeouts and the close mark, shared by every duplicate — the
    /// socket-option semantics of unix (`SO_RCVTIMEO` and `shutdown(2)`
    /// act on the connection, not on one descriptor).
    shared: Arc<PipeShared>,
}

/// Per-connection state every duplicate of one pipe handle shares.
#[derive(Default)]
struct PipeShared {
    /// Read deadline in ms; 0 = block until data or close.
    read_timeout_ms: AtomicU64,
    /// Write deadline in ms; 0 = block until written or close.
    write_timeout_ms: AtomicU64,
    /// Set by `shutdown_both`: every pending and future operation ends.
    closed: AtomicBool,
    /// Set once a server-side close scheduled its linger + disconnect
    /// (the host may close one connection from two paths).
    disconnect_scheduled: AtomicBool,
}

/// How long a closing server instance waits for the client to read what
/// was already written (final error frames, a bye) before forcing the
/// disconnect. `DisconnectNamedPipe` discards unread data — unlike
/// unix `shutdown(2)`, which still delivers it — so disconnecting at
/// once would swallow exactly the frames that explain the close.
const DISCONNECT_LINGER: Duration = Duration::from_secs(2);

/// The server side of a close, off the caller's thread: flush (returns
/// once the client has read everything, or the pipe broke), bounded by
/// [`DISCONNECT_LINGER`], then the instance-wide disconnect — which also
/// ends a flush still parked against a client that stopped reading.
fn schedule_disconnect(handle: HANDLE) -> io::Result<()> {
    let duplicate = |handle: HANDLE| -> io::Result<SendHandle> {
        let mut duplicate: HANDLE = std::ptr::null_mut();
        let process = unsafe { GetCurrentProcess() };
        if unsafe {
            DuplicateHandle(process, handle, process, &mut duplicate, 0, 0, DUPLICATE_SAME_ACCESS)
        } == 0
        {
            return Err(io::Error::last_os_error());
        }
        Ok(SendHandle(duplicate))
    };
    let flusher = duplicate(handle)?;
    let disconnector = match duplicate(handle) {
        Ok(handle) => handle,
        Err(err) => {
            unsafe { CloseHandle(flusher.0) };
            return Err(err);
        }
    };
    let spawned = std::thread::Builder::new()
        .name("starling-host-pipe-linger".into())
        .spawn(move || {
            let disconnector = disconnector;
            let flush = std::thread::Builder::new()
                .name("starling-host-pipe-flush".into())
                .spawn(move || {
                    let flusher = flusher;
                    unsafe {
                        FlushFileBuffers(flusher.0);
                        CloseHandle(flusher.0);
                    }
                });
            let deadline = Instant::now() + DISCONNECT_LINGER;
            if let Ok(flush) = &flush {
                while !flush.is_finished() && Instant::now() < deadline {
                    std::thread::sleep(Duration::from_millis(10));
                }
            }
            unsafe {
                DisconnectNamedPipe(disconnector.0);
                CloseHandle(disconnector.0);
            }
            if let Ok(flush) = flush {
                // Bounded by the disconnect above, which completes a
                // flush parked against a non-reading client.
                let _ = flush.join();
            }
        });
    spawned.map(|_| ()).map_err(io::Error::other)
}

/// How often a waiting operation re-checks the close mark. Bounds the
/// one race `CancelIoEx` alone cannot: an operation issued just after
/// the close's cancel swept the handle.
const CLOSE_POLL: Duration = Duration::from_millis(100);

/// The error a closed connection's operations end with; `Read` maps it
/// to EOF (unix `shutdown(SHUT_RD)` reads 0).
fn closed_error() -> io::Error {
    io::Error::new(io::ErrorKind::ConnectionAborted, "pipe connection closed")
}

fn timeout_of(ms: &AtomicU64) -> Option<Duration> {
    match ms.load(Ordering::SeqCst) {
        0 => None,
        ms => Some(Duration::from_millis(ms)),
    }
}

fn store_timeout(slot: &AtomicU64, timeout: Option<Duration>) -> io::Result<()> {
    let ms = match timeout {
        None => 0,
        // Same contract as std's socket timeouts: zero is invalid.
        Some(duration) if duration.is_zero() => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "cannot set a zero duration timeout",
            ))
        }
        // Sub-millisecond rounds up rather than to "no timeout".
        Some(duration) => (duration.as_millis() as u64).max(1),
    };
    slot.store(ms, Ordering::SeqCst);
    Ok(())
}

/// Runs one overlapped `ReadFile`/`WriteFile` (issued by `start`) to
/// completion, bounded by `timeout` and by the close mark. Returns the
/// bytes transferred. The kernel owns the `OVERLAPPED` and the caller's
/// buffer until the operation completes, so every exit path that
/// cancels also *waits* for the cancellation (`GetOverlappedResult` with
/// bWait) before returning.
unsafe fn overlapped_io(
    handle: HANDLE,
    shared: &PipeShared,
    timeout: Option<Duration>,
    start: impl FnOnce(*mut OVERLAPPED) -> BOOL,
) -> io::Result<u32> {
    if shared.closed.load(Ordering::SeqCst) {
        return Err(closed_error());
    }
    let event = Event::new()?;
    let mut overlapped: OVERLAPPED = std::mem::zeroed();
    overlapped.hEvent = event.0;
    let mut transferred = 0u32;
    if start(&mut overlapped) == 0 {
        let err = GetLastError();
        if err != ERROR_IO_PENDING {
            return Err(io::Error::from_raw_os_error(err as i32));
        }
        let deadline = timeout.map(|timeout| Instant::now() + timeout);
        loop {
            let slice = match deadline {
                Some(deadline) => deadline
                    .saturating_duration_since(Instant::now())
                    .min(CLOSE_POLL),
                None => CLOSE_POLL,
            };
            let waited = WaitForSingleObject(event.0, slice.as_millis() as u32);
            if waited == WAIT_OBJECT_0 {
                break;
            }
            if waited != WAIT_TIMEOUT {
                let err = io::Error::last_os_error();
                CancelIoEx(handle, &overlapped);
                GetOverlappedResult(handle, &overlapped, &mut transferred, 1);
                return Err(err);
            }
            let closed = shared.closed.load(Ordering::SeqCst);
            let expired = deadline.is_some_and(|deadline| Instant::now() >= deadline);
            if closed || expired {
                CancelIoEx(handle, &overlapped);
                if GetOverlappedResult(handle, &overlapped, &mut transferred, 1) != 0 {
                    // Completed before the cancel landed: the bytes moved.
                    return Ok(transferred);
                }
                let err = GetLastError();
                if err == ERROR_OPERATION_ABORTED {
                    return Err(if closed {
                        closed_error()
                    } else {
                        io::Error::new(io::ErrorKind::TimedOut, "pipe operation timed out")
                    });
                }
                return Err(io::Error::from_raw_os_error(err as i32));
            }
        }
    }
    if GetOverlappedResult(handle, &overlapped, &mut transferred, 1) == 0 {
        let err = GetLastError();
        if err == ERROR_OPERATION_ABORTED && shared.closed.load(Ordering::SeqCst) {
            return Err(closed_error());
        }
        return Err(io::Error::from_raw_os_error(err as i32));
    }
    Ok(transferred)
}

// SAFETY (Send and Sync): the handle is a kernel object identifier, not
// a pointer into this process's memory. Send: Win32 calls on one handle
// are thread-safe. Sync: `PipeConn` is shared across the host's
// reader/writer/closer threads through *duplicated* handle values;
// each overlapped operation owns its own OVERLAPPED + event (on the
// issuing thread's stack, waited to completion before return), the
// shared state is atomics, and DisconnectNamedPipe/CancelIoEx racing
// in-flight I/O is documented to fail those operations, not corrupt
// state.
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
                shared: Arc::clone(&self.shared),
            }))
        }
    }

    fn shutdown_both(&self) -> io::Result<()> {
        // There is no shutdown(2) for pipes. Mark the connection closed
        // for every duplicate (a waiting operation sees the mark within
        // CLOSE_POLL and cancels itself; a later one fails at once) and
        // cancel what is in flight on this file object — this process's
        // own threads are released immediately. On the server side the
        // peer must observe the close too: a lingering disconnect (see
        // `schedule_disconnect`) lets it read the frames written before
        // the close, then ends the instance. Never a blocking flush on
        // the caller's thread (the event pump's eviction and the bounded
        // shutdown call this).
        self.shared.closed.store(true, Ordering::SeqCst);
        unsafe {
            CancelIoEx(self.handle.0, std::ptr::null());
        }
        if self.server && !self.shared.disconnect_scheduled.swap(true, Ordering::SeqCst) {
            if let Err(err) = schedule_disconnect(self.handle.0) {
                // No linger possible: disconnect now rather than leave
                // the peer connected to a closed host connection.
                unsafe { DisconnectNamedPipe(self.handle.0) };
                return Err(err);
            }
        }
        Ok(())
    }

    fn set_read_timeout(&self, timeout: Option<Duration>) -> io::Result<()> {
        store_timeout(&self.shared.read_timeout_ms, timeout)
    }

    fn set_write_timeout(&self, timeout: Option<Duration>) -> io::Result<()> {
        store_timeout(&self.shared.write_timeout_ms, timeout)
    }
}

impl Drop for PipeConn {
    fn drop(&mut self) {
        // Close this handle only. A disconnect here would be
        // instance-wide and discard data the peer has not read yet
        // (dropping the writer's duplicate right after a final error
        // frame would swallow it); the close path that must end the
        // instance is `shutdown_both`'s lingering disconnect, and when
        // the last handle closes the client reads what is buffered and
        // then EOF.
        unsafe {
            CloseHandle(self.handle.0);
        }
    }
}

impl Read for PipeConn {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        // One operation moves at most u32::MAX bytes; a shorter read is
        // a legal `Read` answer.
        let len = buf.len().min(u32::MAX as usize) as u32;
        let handle = self.handle.0;
        let timeout = timeout_of(&self.shared.read_timeout_ms);
        let result = unsafe {
            overlapped_io(handle, &self.shared, timeout, |overlapped| {
                ReadFile(handle, buf.as_mut_ptr(), len, std::ptr::null_mut(), overlapped)
            })
        };
        match result {
            Ok(read) => Ok(read as usize),
            // ERROR_BROKEN_PIPE: the peer closed its handles;
            // ERROR_PIPE_NOT_CONNECTED: the server disconnected the
            // instance (after its linger). Both are a clean EOF for a
            // pipe, and a locally closed connection reads EOF too.
            Err(err)
                if err.raw_os_error() == Some(ERROR_BROKEN_PIPE as i32)
                    || err.raw_os_error() == Some(ERROR_PIPE_NOT_CONNECTED as i32) =>
            {
                Ok(0)
            }
            Err(err) if err.kind() == io::ErrorKind::ConnectionAborted => Ok(0),
            Err(err) => Err(err),
        }
    }
}

impl Write for PipeConn {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let len = buf.len().min(u32::MAX as usize) as u32;
        let handle = self.handle.0;
        let timeout = timeout_of(&self.shared.write_timeout_ms);
        let written = unsafe {
            overlapped_io(handle, &self.shared, timeout, |overlapped| {
                WriteFile(handle, buf.as_ptr(), len, std::ptr::null_mut(), overlapped)
            })
        }?;
        Ok(written as usize)
    }

    fn flush(&mut self) -> io::Result<()> {
        // A byte-mode write is in the pipe once its operation completes;
        // FlushFileBuffers would add nothing but a blocking wait against
        // a peer that is not reading.
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
