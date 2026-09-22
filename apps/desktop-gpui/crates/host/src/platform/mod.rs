//! The OS-local transport (E17 §1 Mode B): Unix domain sockets on
//! Linux/macOS, named pipes with a restrictive DACL on Windows.
//!
//! The two implementations answer one surface:
//!
//! - [`TransportListener`] — bound, accepting server side;
//! - [`TransportConn`] — one connected peer, `Read`/`Write` plus the
//!   credential query the [`crate::auth`] layer needs;
//! - [`probe`] — is a live server already bound at this name? (stale-owner
//!   detection's first question);
//! - [`listen`]/[`connect`] — bind/serve and connect.
//!
//! Endpoint naming is deterministic from the data root: every process —
//! host, client, the Electron adapter — derives the same
//! `starling-runtime-<hash>.sock` / pipe name from the same root, so
//! "where is the host" never needs a discovery protocol. The hash is a
//! disambiguator, not a secret: it keeps two data roots (two checkouts'
//! test dirs, say) from sharing one endpoint.

#[cfg(unix)]
pub mod unix;
#[cfg(unix)]
pub use unix::{
    connect as platform_connect, listen as platform_listen, probe as platform_probe,
    UdsConn as PlatformConn, UdsListener as PlatformListener,
};

#[cfg(windows)]
pub mod windows;
#[cfg(windows)]
pub use windows::{
    connect as platform_connect, listen as platform_listen, probe as platform_probe,
    PipeConn as PlatformConn, PipeListener as PlatformListener,
};

use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use crate::auth::PeerCredentials;

/// A bound server endpoint.
pub trait TransportListener: Send {
    /// Blocks until a peer connects (or, in the host's polling loop,
    /// reports `WouldBlock` when non-blocking).
    fn accept(&self) -> io::Result<Box<dyn TransportConn>>;
    fn set_nonblocking(&self, nonblocking: bool) -> io::Result<()>;
}

/// One connected peer.
pub trait TransportConn: Read + Write + Send + Sync {
    /// The kernel's evidence about the peer (see [`crate::auth`] for what
    /// each platform can supply). Errors when the platform syscall fails;
    /// `PeerCredentials::absent()` when the platform has no mechanism
    /// at all.
    fn peer_credentials(&self) -> io::Result<PeerCredentials>;
    /// A second handle to the same connection (the host splits reader,
    /// writer and shutdown duties across threads).
    fn try_clone(&self) -> io::Result<Box<dyn TransportConn>>;
    /// Immediately ends both directions (unblocks a reader parked on this
    /// connection).
    fn shutdown_both(&self) -> io::Result<()>;
    /// Arms a read poll on this connection so a blocking read wakes up
    /// periodically ([io::ErrorKind::WouldBlock]/[io::ErrorKind::TimedOut])
    /// instead of parking forever — the mechanism behind both sides'
    /// "idle" loops. Errors where the platform cannot poll a synchronous
    /// read (Windows' synchronous `ReadFile`; recorded gap there — see
    /// `platform::windows`).
    fn set_read_timeout(&self, timeout: Option<std::time::Duration>) -> io::Result<()>;
    /// Bounds a blocking write the same way: past the deadline a write
    /// parked against a peer that stopped reading fails with
    /// [io::ErrorKind::WouldBlock]/[io::ErrorKind::TimedOut] instead of
    /// blocking the caller indefinitely. The accept path's direct
    /// writes (rejection and auth-failure frames) arm this so a stalled
    /// peer cannot park the one accept thread. A no-op where the
    /// platform cannot bound a synchronous write (recorded gap — see
    /// `platform::windows`).
    fn set_write_timeout(&self, timeout: Option<std::time::Duration>) -> io::Result<()>;
}

/// What a probe of an endpoint found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Probe {
    /// A server answered the connection: a live owner is bound here.
    Live,
    /// Nothing answered (ECONNREFUSED / ERROR_PIPE_BUSY-free absence): any
    /// filesystem residue at the endpoint is stale.
    Dead,
    /// The probe itself failed in a way that is neither of those
    /// (permission denied on the socket, an OS error we do not
    /// interpret). The caller decides, conservatively.
    Unknown(String),
}

/// The per-user directory endpoints live in.
///
/// unix: `$XDG_RUNTIME_DIR/starling` when the session provides one (it is
/// per-user by definition), else `/tmp/starling-runtime-<uid>` (the
/// fallback must encode the user: /tmp is shared). Created 0700.
///
/// Windows: named pipes live in a kernel namespace, not the filesystem —
/// this directory only matters for lock/coordination files and is
/// `%TEMP%\starling-runtime`.
pub fn default_runtime_dir() -> PathBuf {
    #[cfg(unix)]
    {
        if let Some(xdg) = std::env::var_os("XDG_RUNTIME_DIR") {
            if !xdg.is_empty() {
                return PathBuf::from(xdg).join("starling");
            }
        }
        let uid = crate::auth::current_uid();
        std::env::temp_dir().join(format!("starling-runtime-{uid}"))
    }
    #[cfg(windows)]
    {
        std::env::temp_dir().join("starling-runtime")
    }
}

/// The endpoint name for one data root: `starling-runtime-<fnv1a-64>.sock`
/// (unix) / the equivalent pipe name (windows prepends the pipe
/// namespace). Deterministic across processes; distinct roots never
/// collide in practice.
pub fn endpoint_stem(root: &Path) -> String {
    // FNV-1a 64 over the root's raw OS-encoded bytes. Not cryptographic —
    // it only has to not collide between a user's distinct roots — but
    // hashing the *bytes* (not a lossy UTF-8 rendering) keeps distinct
    // non-UTF-8 paths distinct: two roots that collapse under
    // `to_string_lossy` must never share one endpoint.
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in root.as_os_str().as_encoded_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("starling-runtime-{:016x}", hash)
}

/// The full socket path for one data root under `runtime_dir` (unix:
/// `<dir>/<stem>.sock`; windows: the `\\.\pipe\<stem>` name — the pipe
/// namespace is per-machine, so `runtime_dir` does not select it there).
#[cfg(unix)]
pub fn socket_path(runtime_dir: &Path, root: &Path) -> PathBuf {
    runtime_dir.join(format!("{}.sock", endpoint_stem(root)))
}

#[cfg(windows)]
pub fn socket_path(runtime_dir: &Path, root: &Path) -> PathBuf {
    windows::pipe_path(runtime_dir, root)
}

/// Ensure the runtime dir exists with user-only permissions. An existing
/// directory is **tightened** to 0700 (unix): the fallback
/// `/tmp/starling-runtime-<uid>` and an externally-supplied
/// `XDG_RUNTIME_DIR` are both paths this process did not create, and the
/// 0700 assumption the endpoint's security docs rely on must hold, not
/// be presumed. A directory this user cannot tighten (owned by someone
/// else) is an error — fail closed rather than serve from a shared dir.
/// A symlink at the path is rejected outright (`symlink_metadata`, no
/// following): the 0700 guarantee must hold on the directory itself,
/// not on whatever a swapped link points at.
///
/// Non-unix note: the tightening branch is unix-only. On Windows the
/// pipe namespace (not this directory) is the endpoint, so the mode is
/// moot there today — but any future non-unix, non-Windows target must
/// not inherit this function's unix-docs unchanged: the 0700 boundary
/// would be a silent no-op there.
pub fn ensure_runtime_dir(dir: &Path) -> io::Result<()> {
    use std::fs::DirBuilder;
    if dir.exists() {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            // symlink_metadata: do not follow — a symlinked runtime dir
            // is someone else's path wearing our name.
            let metadata = std::fs::symlink_metadata(dir)?;
            if !metadata.is_dir() {
                return Err(io::Error::new(
                    io::ErrorKind::NotADirectory,
                    format!(
                        "runtime dir {dir:?} exists and is not a directory \
                         (symlinks are refused)"
                    ),
                ));
            }
            let mode = metadata.permissions().mode();
            if mode & 0o077 != 0 {
                std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700))?;
            }
        }
        return Ok(());
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        DirBuilder::new().recursive(true).mode(0o700).create(dir)?;
    }
    #[cfg(not(unix))]
    {
        DirBuilder::new().recursive(true).create(dir)?;
    }
    Ok(())
}

/// Probes whether a live server is bound at `path`.
///
/// unix: `connect(2)`. Success = [`Probe::Live`] (the probe connection is
/// closed immediately — the host's accept loop sees a connect/disconnect
/// blip, which its auth+hello exchange simply never observes);
/// `ECONNREFUSED`/`ENOENT` = [`Probe::Dead`] (a leftover socket file with
/// no listener — the stale-owner case); anything else =
/// [`Probe::Unknown`].
///
/// windows: the pipe namespace has no "refused" signal — a connect to a
/// name no server created fails with file-not-found, which is this
/// platform's `Dead`; a busy pipe (`ERROR_PIPE_BUSY`) is `Live`.
pub fn probe(path: &Path) -> Probe {
    platform_probe(path)
}

/// Binds a listener at `path`. unix: fails if a live listener already
/// holds the socket (call [`probe`] first — the stale file was unlinked
/// by then). windows: creates the pipe's first instance with
/// `FILE_FLAG_FIRST_PIPE_INSTANCE`, failing if another server already
/// owns the name.
pub fn listen(path: &Path) -> io::Result<Box<dyn TransportListener>> {
    platform_listen(path)
}

/// Connects a client to the endpoint at `path`.
pub fn connect(path: &Path) -> io::Result<Box<dyn TransportConn>> {
    platform_connect(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoint_stem_is_deterministic_and_root_sensitive() {
        let a = endpoint_stem(Path::new("/home/alice/.local/share/starling"));
        let b = endpoint_stem(Path::new("/home/alice/.local/share/starling"));
        let c = endpoint_stem(Path::new("/tmp/other-root"));
        assert_eq!(a, b, "same root, same endpoint");
        assert_ne!(a, c, "different roots never share an endpoint");
        assert!(a.starts_with("starling-runtime-"), "{a}");
    }

    #[test]
    fn endpoint_stem_distinguishes_non_utf8_roots() {
        // Two roots that collapse under to_string_lossy (each invalid
        // byte becomes U+FFFD) must still hash differently: the hash is
        // over the raw OS bytes, not the lossy rendering.
        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStrExt;
            let a = std::ffi::OsStr::from_bytes(b"/tmp/root-\xff");
            let b = std::ffi::OsStr::from_bytes(b"/tmp/root-\xfe");
            assert_ne!(
                endpoint_stem(Path::new(a)),
                endpoint_stem(Path::new(b)),
                "distinct non-UTF-8 roots must not share an endpoint"
            );
            // And the lossy collision case itself: two *different*
            // invalid bytes render to the same lossy string.
            assert_eq!(a.to_string_lossy(), b.to_string_lossy());
        }
    }

    #[test]
    fn endpoint_stem_hex_is_stable() {
        // Pin the hash so a future refactor cannot silently move every
        // user's socket path (all live clients would miss the host).
        let stem = endpoint_stem(Path::new("/opt/starling-test-root"));
        let hex = stem.trim_start_matches("starling-runtime-");
        assert_eq!(hex.len(), 16);
        // FNV-1a 64 of that exact path's bytes, computed independently:
        assert_eq!(hex, fnv64(b"/opt/starling-test-root"));
    }

    fn fnv64(bytes: &[u8]) -> String {
        let mut hash: u64 = 0xcbf29ce484222325;
        for byte in bytes {
            hash ^= *byte as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        format!("{hash:016x}")
    }

    #[test]
    #[cfg(unix)]
    fn an_existing_runtime_dir_is_tightened_to_owner_only() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let runtime = dir.path().join("shared-runtime");
        std::fs::create_dir(&runtime).unwrap();
        std::fs::set_permissions(&runtime, std::fs::Permissions::from_mode(0o755)).unwrap();
        ensure_runtime_dir(&runtime).expect("existing dir is accepted");
        let mode = std::fs::metadata(&runtime).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o700, "group/world bits are gone");

        // Already-tight dirs pass through untouched.
        ensure_runtime_dir(&runtime).expect("tight dir stays accepted");
        let mode = std::fs::metadata(&runtime).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o700);

        // Fresh creation is 0700 from the start.
        let fresh = dir.path().join("fresh-runtime");
        ensure_runtime_dir(&fresh).expect("fresh dir is created");
        let mode = std::fs::metadata(&fresh).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o700);
    }

    #[test]
    #[cfg(unix)]
    fn a_symlinked_runtime_dir_is_refused() {
        // Even a symlink to a properly 0700 directory: the guarantee
        // must hold on the path itself, not on the link's target —
        // someone else's directory wearing our runtime dir's name is
        // exactly the substitution this function exists to refuse.
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let real = dir.path().join("real-runtime");
        std::fs::create_dir(&real).unwrap();
        std::fs::set_permissions(&real, std::fs::Permissions::from_mode(0o700)).unwrap();
        std::os::unix::fs::symlink(&real, dir.path().join("linked-runtime")).unwrap();
        let refused = ensure_runtime_dir(&dir.path().join("linked-runtime"));
        assert!(refused.is_err(), "a symlinked runtime dir must be refused");
    }
}
