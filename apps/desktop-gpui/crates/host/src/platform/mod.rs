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
    /// `PeerCredentials::absent()` when the platform has no mechanism at
    /// all.
    fn peer_credentials(&self) -> io::Result<PeerCredentials>;
    /// A second handle to the same connection (the host splits reader,
    /// writer and shutdown duties across threads).
    fn try_clone(&self) -> io::Result<Box<dyn TransportConn>>;
    /// Immediately ends both directions (unblocks a reader parked on this
    /// connection).
    fn shutdown_both(&self) -> io::Result<()>;
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
    // FNV-1a 64 over the canonical-ish root string. Not cryptographic —
    // it only has to not collide between a user's distinct roots.
    let text = root.to_string_lossy();
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in text.bytes() {
        hash ^= byte as u64;
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

/// Ensure the runtime dir exists with user-only permissions.
pub fn ensure_runtime_dir(dir: &Path) -> io::Result<()> {
    use std::fs::DirBuilder;
    if dir.exists() {
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
    fn endpoint_stem_hex_is_stable() {
        // Pin the hash so a future refactor cannot silently move every
        // user's socket path (all live clients would miss the host).
        let stem = endpoint_stem(Path::new("/opt/starling-test-root"));
        let hex = stem.trim_start_matches("starling-runtime-");
        assert_eq!(hex.len(), 16);
        // FNV-1a 64 of that exact path, computed independently:
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
}
