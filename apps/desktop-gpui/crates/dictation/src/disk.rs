//! Free disk space for recording (#342).
//!
//! A take's journal grows by `rate × 4` bytes a second (f32 frames; about
//! 3.8 MB a minute at 16 kHz, 11.5 MB at 48 kHz). The app checks the
//! journal tree's free space before a take starts and the recorder's
//! writer task re-checks it while one runs ([`DiskWatch`]), so a filling
//! disk is warned about early and a take is stopped cleanly — final
//! boundary, trailer, adoption into the store — while there is still room
//! for that, instead of dying in a failed write.
//!
//! The probe is a trait so tests can drive every state without filling a
//! disk.

use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

/// Reports the bytes available to this process on the file system
/// holding a path.
pub trait FreeSpaceProbe: Send + Sync {
    fn available_bytes(&self, path: &Path) -> io::Result<u64>;
}

/// The operating system's answer (`statvfs` / `GetDiskFreeSpaceExW`),
/// asked of the nearest existing ancestor of the path — the journal tree
/// may not exist before the first take.
#[derive(Clone, Copy, Debug, Default)]
pub struct SystemProbe;

impl FreeSpaceProbe for SystemProbe {
    fn available_bytes(&self, path: &Path) -> io::Result<u64> {
        let existing = existing_ancestor(path)?;
        available_on(&existing)
    }
}

fn existing_ancestor(path: &Path) -> io::Result<PathBuf> {
    path.ancestors()
        .find(|candidate| !candidate.as_os_str().is_empty() && candidate.exists())
        .map(Path::to_path_buf)
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("no existing directory above {}", path.display()),
            )
        })
}

#[cfg(unix)]
fn available_on(path: &Path) -> io::Result<u64> {
    use std::os::unix::ffi::OsStrExt;
    let c_path = std::ffi::CString::new(path.as_os_str().as_bytes())
        .map_err(|err| io::Error::new(io::ErrorKind::InvalidInput, err))?;
    let mut stat: libc::statvfs = unsafe { std::mem::zeroed() };
    // SAFETY: `c_path` is a valid NUL-terminated string and `stat` is a
    // properly sized, writable statvfs.
    if unsafe { libc::statvfs(c_path.as_ptr(), &mut stat) } != 0 {
        return Err(io::Error::last_os_error());
    }
    // `f_bavail` (not `f_bfree`): what an unprivileged writer can use.
    Ok((stat.f_bavail as u64).saturating_mul(stat.f_frsize as u64))
}

#[cfg(windows)]
fn available_on(path: &Path) -> io::Result<u64> {
    use std::os::windows::ffi::OsStrExt;
    let wide: Vec<u16> = path.as_os_str().encode_wide().chain(Some(0)).collect();
    let mut available = 0u64;
    // SAFETY: `wide` is NUL-terminated; the out pointers are valid or null.
    let ok = unsafe {
        windows_sys::Win32::Storage::FileSystem::GetDiskFreeSpaceExW(
            wide.as_ptr(),
            &mut available,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
    };
    if ok == 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(available)
}

#[cfg(not(any(unix, windows)))]
fn available_on(_path: &Path) -> io::Result<u64> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "free-space checks are not supported on this platform",
    ))
}

/// Free-space thresholds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DiskPolicy {
    /// Below this the user is warned (before and during a take).
    pub warn_below: u64,
    /// Below this no take starts, and a running take is stopped cleanly.
    /// Leaves room for the stop itself: the final journal records, the
    /// SQLite commit and its WAL, and later housekeeping.
    pub stop_below: u64,
}

impl Default for DiskPolicy {
    fn default() -> Self {
        Self {
            warn_below: 1024 * 1024 * 1024,
            stop_below: 128 * 1024 * 1024,
        }
    }
}

/// How the free space compares with a [`DiskPolicy`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DiskLevel {
    Ok,
    Low,
    Critical,
}

/// One free-space reading.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DiskReading {
    pub available: u64,
    pub level: DiskLevel,
}

impl DiskPolicy {
    pub fn assess(&self, available: u64) -> DiskReading {
        let level = if available < self.stop_below {
            DiskLevel::Critical
        } else if available < self.warn_below {
            DiskLevel::Low
        } else {
            DiskLevel::Ok
        };
        DiskReading { available, level }
    }

    /// Probe `path` and assess the answer.
    pub fn check(&self, probe: &dyn FreeSpaceProbe, path: &Path) -> io::Result<DiskReading> {
        Ok(self.assess(probe.available_bytes(path)?))
    }

    /// Minutes of recording at `sample_rate` that fit before the stop
    /// threshold — what a low-space warning tells the user.
    pub fn minutes_left(&self, available: u64, sample_rate: u32) -> u64 {
        let per_minute = u64::from(sample_rate.max(1)) * 4 * 60;
        available.saturating_sub(self.stop_below) / per_minute
    }

    /// The warning text for a low reading at `sample_rate`; `None` while
    /// space is fine.
    pub fn warning(&self, reading: DiskReading, sample_rate: u32) -> Option<String> {
        match reading.level {
            DiskLevel::Ok => None,
            DiskLevel::Low => Some(format!(
                "Disk space is low ({} MB free): about {} minutes of recording left before \
                 Starling stops a take to keep it safe.",
                reading.available / (1024 * 1024),
                self.minutes_left(reading.available, sample_rate)
            )),
            DiskLevel::Critical => Some(format!(
                "The disk is almost full ({} MB free). Free up space to record; takes stop \
                 below {} MB so they can still be saved.",
                reading.available / (1024 * 1024),
                self.stop_below / (1024 * 1024)
            )),
        }
    }
}

/// The recorder's in-take free-space watch: the writer task probes the
/// journal's directory every `interval` (and once at start).
#[derive(Clone)]
pub struct DiskWatch {
    pub probe: Arc<dyn FreeSpaceProbe>,
    pub policy: DiskPolicy,
    pub interval: Duration,
}

/// How often a running take re-checks free space by default.
pub const DISK_WATCH_INTERVAL: Duration = Duration::from_secs(5);

impl DiskWatch {
    /// The system probe at the default policy and cadence.
    pub fn system() -> Self {
        Self {
            probe: Arc::new(SystemProbe),
            policy: DiskPolicy::default(),
            interval: DISK_WATCH_INTERVAL,
        }
    }
}

impl std::fmt::Debug for DiskWatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DiskWatch")
            .field("policy", &self.policy)
            .field("interval", &self.interval)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn readings_follow_the_thresholds() {
        let policy = DiskPolicy {
            warn_below: 1_000,
            stop_below: 100,
        };
        assert_eq!(policy.assess(5_000).level, DiskLevel::Ok);
        assert_eq!(policy.assess(1_000).level, DiskLevel::Ok);
        assert_eq!(policy.assess(999).level, DiskLevel::Low);
        assert_eq!(policy.assess(100).level, DiskLevel::Low);
        assert_eq!(policy.assess(99).level, DiskLevel::Critical);
        assert_eq!(policy.assess(0).level, DiskLevel::Critical);
        assert!(policy.warning(policy.assess(5_000), 16_000).is_none());
    }

    #[test]
    fn minutes_left_count_journal_bytes_above_the_stop_line() {
        let policy = DiskPolicy::default();
        let minute_16k = 16_000u64 * 4 * 60;
        assert_eq!(
            policy.minutes_left(policy.stop_below + 10 * minute_16k, 16_000),
            10
        );
        assert_eq!(
            policy.minutes_left(policy.stop_below + 10 * minute_16k, 48_000),
            3
        );
        assert_eq!(policy.minutes_left(policy.stop_below / 2, 16_000), 0);
        let text = policy
            .warning(policy.assess(policy.stop_below + 10 * minute_16k), 16_000)
            .expect("low");
        assert!(text.contains("about 10 minutes"), "{text}");
    }

    #[test]
    fn the_system_probe_answers_for_a_path_that_does_not_exist_yet() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let missing = dir.path().join("not/yet/created");
        let available = SystemProbe.available_bytes(&missing).expect("probe");
        assert!(available > 0);
    }
}
