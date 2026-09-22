//! Peer authentication for the OS-local transport (E17 §1 Mode B).
//!
//! What "authenticated" means here, per platform — the kernel, not the
//! peer, supplies the evidence:
//!
//! - **Linux**: `SO_PEERCRED` on the accepted UDS connection yields the
//!   connecting process's `pid`, `uid`, `gid` as they were at `connect`
//!   time. The host admits the connection iff the peer's `uid` equals
//!   the host's own effective uid. Kernel-provided; unforgeable.
//! - **macOS**: `LOCAL_PEERCRED` yields the peer's effective `uid`
//!   (`xucred.cr_uid`) for the same same-user decision;
//!   `LOCAL_PEERPID` supplies the pid for diagnostics/liveness.
//! - **Windows**: named pipes carry no per-connection credential query
//!   comparable to `SO_PEERCRED`; the authentication is the pipe's DACL
//!   at creation — only the creating user (and the system) holds a pipe
//!   handle, so the kernel has already enforced "same user" before a
//!   connect can succeed. `GetNamedPipeClientProcessId` supplies the
//!   peer pid for diagnostics. See `platform::windows`. The DACL
//!   construction path is compile-verified by CI on a native Windows
//!   runner (the `windows-check` job in desktop-gpui-rust.yml), so a
//!   regression in the SDDL/security-descriptor code fails the build
//!   rather than shipping silently; an in-band re-check of the peer's
//!   user at accept time remains a recorded gap there.
//!
//! Fail-closed: on unix, a platform that cannot supply peer credentials
//! (an unsupported `cfg`) yields `PeerCredentials::absent()`, and the
//! [`PeerPolicy::SameUser`] policy rejects absent credentials — a host
//! that cannot verify the peer serves no one. The socket's directory
//! (0700) and the socket file itself (0600) are an outer ring on top of
//! the credential check, never a substitute for it.

/// What the platform could tell us about the peer, at accept time.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PeerCredentials {
    /// The peer's effective uid (unix). `None` where the platform cannot
    /// supply one.
    pub uid: Option<u32>,
    /// The peer's process id, where available (diagnostics and
    /// stale-owner probes; never an authorization input on its own — pids
    /// are recycled).
    pub pid: Option<u32>,
}

impl PeerCredentials {
    pub fn absent() -> PeerCredentials {
        PeerCredentials::default()
    }
}

/// Why a peer was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthError {
    /// The peer's uid is not the host's uid (includes the case where the
    /// platform could not read one: no credential, no admission).
    ForeignUser { peer: Option<u32>, expected: u32 },
}

impl std::fmt::Display for AuthError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AuthError::ForeignUser { peer, expected } => match peer {
                Some(uid) => write!(
                    f,
                    "peer uid {uid} is not the host's uid {expected}; connection refused"
                ),
                None => write!(
                    f,
                    "peer credentials unavailable on this platform; connection refused"
                ),
            },
        }
    }
}

/// Decides whether an accepted connection may be served.
pub trait PeerPolicy: Send + Sync {
    fn authenticate(&self, credentials: PeerCredentials) -> Result<(), AuthError>;
}

/// The production policy: same effective user as the host, credentials
/// required.
#[derive(Debug, Clone, Copy)]
pub struct SameUser {
    host_uid: u32,
}

impl SameUser {
    /// The policy for this process. On unix the host's uid is its
    /// effective uid; the Windows transport uses
    /// [`PeerPolicy`](this trait)'s DACL-backed variant instead (see
    /// `platform::windows` — there is no uid to compare there).
    #[cfg(unix)]
    pub fn for_this_process() -> SameUser {
        SameUser {
            host_uid: current_uid(),
        }
    }
}

impl PeerPolicy for SameUser {
    fn authenticate(&self, credentials: PeerCredentials) -> Result<(), AuthError> {
        match credentials.uid {
            Some(uid) if uid == self.host_uid => Ok(()),
            other => Err(AuthError::ForeignUser {
                peer: other,
                expected: self.host_uid,
            }),
        }
    }
}

/// The policy object the server runs by default, per platform: uid check
/// on unix, the DACL-guaranteed admission on Windows.
pub fn default_policy() -> std::sync::Arc<dyn PeerPolicy> {
    #[cfg(unix)]
    {
        std::sync::Arc::new(SameUser::for_this_process())
    }
    #[cfg(windows)]
    {
        // On Windows the DACL on the named pipe already restricted
        // connections to the creating user + system; a peer that completed
        // the connect passed that check. uid-style evidence does not
        // exist there, so "absent uid" is the expected credential shape,
        // not a failure.
        std::sync::Arc::new(AdmittedByAcl)
    }
}

/// Windows' default policy: admission was the pipe DACL's job (see
/// `platform::windows`); the policy layer accepts what the kernel
/// already verified.
#[cfg(windows)]
#[derive(Debug, Clone, Copy)]
pub struct AdmittedByAcl;

#[cfg(windows)]
impl PeerPolicy for AdmittedByAcl {
    fn authenticate(&self, _credentials: PeerCredentials) -> Result<(), AuthError> {
        Ok(())
    }
}

/// The host's effective uid (unix).
#[cfg(unix)]
pub fn current_uid() -> u32 {
    // SAFETY: `geteuid` is async-signal-safe, takes no pointers, and
    // cannot fail.
    unsafe { libc::geteuid() as u32 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_user_admits_the_matching_uid() {
        let policy = SameUser { host_uid: 1000 };
        assert_eq!(
            policy.authenticate(PeerCredentials {
                uid: Some(1000),
                pid: Some(4242),
            }),
            Ok(())
        );
    }

    #[test]
    fn same_user_rejects_a_foreign_uid() {
        let policy = SameUser { host_uid: 1000 };
        assert_eq!(
            policy.authenticate(PeerCredentials {
                uid: Some(0),
                pid: Some(1),
            }),
            Err(AuthError::ForeignUser {
                peer: Some(0),
                expected: 1000
            })
        );
    }

    #[test]
    fn same_user_fails_closed_on_absent_credentials() {
        // No kernel evidence, no admission — never "trust by default".
        let policy = SameUser { host_uid: 1000 };
        assert_eq!(
            policy.authenticate(PeerCredentials::absent()),
            Err(AuthError::ForeignUser {
                peer: None,
                expected: 1000
            })
        );
    }

    #[test]
    fn the_default_policy_is_real_and_satisfied_by_this_process() {
        // The production policy reads this process's own euid; run against
        // that uid it must admit. (The negative direction — a genuinely
        // foreign uid reaching a live socket — needs a second OS user, so
        // the enforcement path over a real socket is covered in the IPC
        // suite with an injected expecting-another-uid policy.)
        let policy: std::sync::Arc<dyn PeerPolicy> = default_policy();
        #[cfg(unix)]
        {
            let mine = PeerCredentials {
                uid: Some(current_uid()),
                pid: Some(std::process::id()),
            };
            assert_eq!(policy.authenticate(mine), Ok(()));
        }
    }
}

/// A policy that demands one specific uid — the IPC suite's stand-in for
/// a foreign user's connection (no second OS user needed: the policy
/// simply expects someone else, and the real kernel credential is what
/// fails the check).
#[derive(Debug, Clone, Copy)]
pub struct ExpectUid(pub u32);

impl PeerPolicy for ExpectUid {
    fn authenticate(&self, credentials: PeerCredentials) -> Result<(), AuthError> {
        SameUser { host_uid: self.0 }.authenticate(credentials)
    }
}

#[cfg(test)]
mod expect_uid_tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn expect_uid_policy_is_satisfiable_only_by_that_uid() {
        let policy: Arc<dyn PeerPolicy> = Arc::new(ExpectUid(12345));
        assert_eq!(
            policy.authenticate(PeerCredentials {
                uid: Some(12345),
                pid: None
            }),
            Ok(())
        );
        assert!(policy
            .authenticate(PeerCredentials {
                uid: Some(current_uid()),
                pid: None
            })
            .is_err());
    }
}
