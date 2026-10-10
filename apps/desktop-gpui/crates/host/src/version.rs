//! The host's build stamp and the version handshake (#220).
//!
//! The app binary runs the host (`starling-gpui --runtime-host`), so an
//! app and the host it starts share one build. A host started by an
//! earlier build can still be serving when a newer app starts (it idles
//! up to a minute after its last window closed, or the app was updated
//! under a running host): the newer app asks it to step aside
//! ([`crate::frame::Frame::Retire`]) and starts its own once it has. A
//! newer host serving an older app refuses it plainly instead of
//! speaking a protocol that app may not know
//! ([`crate::frame::TransportErrorCode::VersionMismatch`]).
//!
//! The stamp comes from this crate's build script: a content hash of the
//! code the host runs, and when that code last changed (the order of two
//! different builds).

use serde::{Deserialize, Serialize};

/// One build of the host, as the handshake compares it. Two stamps are
/// equal when their code is (`id`): the same code built twice — another
/// feature set, a rebuild — is the same build to the handshake.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildStamp {
    /// Equal for two builds of the same code.
    pub id: String,
    /// When that code was built (seconds since the epoch): of two
    /// different builds, the later one is the newer.
    pub built: u64,
}

impl BuildStamp {
    /// This binary's build.
    pub fn current() -> BuildStamp {
        BuildStamp {
            id: env!("STARLING_HOST_BUILD_ID").to_string(),
            built: env!("STARLING_HOST_BUILT")
                .parse()
                .expect("the build script stamps the build time as a number"),
        }
    }

    /// Whether `self` is an older build than `other` (a different build
    /// built earlier). Two different builds from the same second order by
    /// id, so of any two different builds exactly one is the older: the
    /// handshake always settles which one serves.
    pub fn older_than(&self, other: &BuildStamp) -> bool {
        self.id != other.id && (self.built, &self.id) < (other.built, &other.id)
    }
}

impl PartialEq for BuildStamp {
    fn eq(&self, other: &BuildStamp) -> bool {
        self.id == other.id
    }
}

impl Eq for BuildStamp {}

/// How a host answered an app asking it to step aside
/// ([`crate::frame::Frame::Retire`]).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RetireAnswer {
    /// The host stops now (it says goodbye to every connection); the app
    /// starts its own once it has gone.
    Retiring,
    /// The host has work in hand (a take, a transcription, another
    /// window); the app asks again later. `reason` says what.
    Busy { reason: String },
    /// The asking app is not newer than this host: it keeps serving.
    Refused { reason: String },
}

/// What a newer host tells an older app it will not serve (the detail
/// of its `version_mismatch` refusal).
pub fn older_app_refusal() -> String {
    "This Starling window is from an older version than the Starling recording service that is \
     running. Close this window and start Starling again."
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stamps_compare_by_code_and_order_by_build_time() {
        let a = BuildStamp {
            id: "same".into(),
            built: 1,
        };
        let rebuilt = BuildStamp {
            id: "same".into(),
            built: 2,
        };
        let newer = BuildStamp {
            id: "other".into(),
            built: 3,
        };
        assert_eq!(a, rebuilt, "the same code built twice is one build");
        assert!(!a.older_than(&rebuilt));
        assert!(a.older_than(&newer));
        assert!(!newer.older_than(&a));
        assert_ne!(a, newer);
        let same_second = BuildStamp {
            id: "another".into(),
            built: 3,
        };
        assert!(
            same_second.older_than(&newer) != newer.older_than(&same_second),
            "two builds of one second still settle which is older"
        );
    }
}
