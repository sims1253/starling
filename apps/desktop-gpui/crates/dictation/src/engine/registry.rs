//! The sidecar registry: how a second app instance reuses the first
//! instance's sidecar instead of spawning a duplicate (#362 step 3).
//!
//! `<state_dir>/sidecar.json` names the one shared sidecar (pid, port,
//! model). A manager that finds a live, warm, model-matching entry
//! ATTACHES to it — it never stops a process it does not own — while the
//! owner removes the file on shutdown only if the file still names it.
//! A `<state_dir>/spawn.lock` (created with `create_new`, so exactly one
//! winner) serializes two launching instances; the loser waits for the
//! winner's registry entry instead of racing it.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use serde::{Deserialize, Serialize};

/// The registry file name inside the state dir.
pub const REGISTRY_FILE: &str = "sidecar.json";
/// The spawn-serialization lock file name.
pub const SPAWN_LOCK_FILE: &str = "spawn.lock";
/// A spawn lock older than this is stale (its creator crashed mid-spawn)
/// and gets replaced.
pub const SPAWN_LOCK_STALE_AFTER: Duration = Duration::from_secs(60);

/// The shared-sidecar description on disk.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SidecarRegistration {
    /// The sidecar process id.
    pub pid: u32,
    /// The loopback port it serves on.
    pub port: u16,
    /// The slug it was started with (`--model`).
    pub slug: String,
    /// The app model id it serves (catalog id, not the slug).
    pub model_id: String,
    /// The GGUF path it was started with.
    pub gguf: String,
    /// The engine binary it was started from.
    pub engine_path: String,
    /// The manager process id that owns (and stops) this sidecar.
    pub owner_pid: u32,
}

/// `<state_dir>/sidecar.json`.
pub fn registry_path(state_dir: &Path) -> PathBuf {
    state_dir.join(REGISTRY_FILE)
}

/// Reads the registration; `None` when absent or unparseable (a corrupt
/// file is treated like no file: the reader spawns its own sidecar and
/// overwrites it).
pub fn read_registration(state_dir: &Path) -> Option<SidecarRegistration> {
    let text = fs::read_to_string(registry_path(state_dir)).ok()?;
    serde_json::from_str(&text).ok()
}

/// Writes the registration atomically (tmp + rename) so a concurrent
/// reader never sees a half-written file.
pub fn write_registration(state_dir: &Path, registration: &SidecarRegistration) -> io::Result<()> {
    fs::create_dir_all(state_dir)?;
    let path = registry_path(state_dir);
    let tmp = state_dir.join(format!("{REGISTRY_FILE}.tmp"));
    fs::write(
        &tmp,
        serde_json::to_string_pretty(registration).unwrap_or_default(),
    )?;
    fs::rename(&tmp, &path)
}

/// The owner removes the registry on shutdown — but only if the file
/// still names *it* as the owner: a newer entry written by another
/// instance (take-over, second instance switch) must survive.
pub fn remove_registration(state_dir: &Path, owner_pid: u32) {
    if let Some(registration) = read_registration(state_dir) {
        if registration.owner_pid == owner_pid {
            let _ = fs::remove_file(registry_path(state_dir));
        }
    }
}

/// An acquired spawn lock. Dropping it releases the on-disk file.
#[derive(Debug)]
pub struct SpawnLock {
    path: PathBuf,
    released: bool,
}

impl SpawnLock {
    /// Removes the lock file (called once the registry is written — the
    /// lock guards the spawn window, not the sidecar's life).
    pub fn release(mut self) {
        self.released = true;
        let _ = fs::remove_file(&self.path);
    }
}

impl Drop for SpawnLock {
    fn drop(&mut self) {
        if !self.released {
            let _ = fs::remove_file(&self.path);
        }
    }
}

/// The outcome of trying to take the spawn lock.
#[derive(Debug)]
pub enum LockOutcome {
    /// We hold it; proceed to spawn and write the registry.
    Acquired(SpawnLock),
    /// Another instance holds a fresh lock: wait for its registry entry
    /// (poll [`read_registration`]) instead of spawning.
    HeldElsewhere,
}

/// Tries to create the spawn lock with `create_new` — exactly one
/// concurrent instance wins. An existing lock older than
/// [`SPAWN_LOCK_STALE_AFTER`] is stale (its creator died mid-spawn) and
/// is replaced.
pub fn try_spawn_lock(state_dir: &Path) -> io::Result<LockOutcome> {
    let _ = fs::create_dir_all(state_dir);
    let path = state_dir.join(SPAWN_LOCK_FILE);
    match fs::File::options().write(true).create_new(true).open(&path) {
        Ok(_) => Ok(LockOutcome::Acquired(SpawnLock {
            path,
            released: false,
        })),
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
            if let Some(age) = lock_age(&path) {
                if age >= SPAWN_LOCK_STALE_AFTER {
                    // Stale: its creator cannot release it anymore.
                    let _ = fs::remove_file(&path);
                    return fs::File::options()
                        .write(true)
                        .create_new(true)
                        .open(&path)
                        .map(|_| {
                            LockOutcome::Acquired(SpawnLock {
                                path,
                                released: false,
                            })
                        });
                }
            }
            Ok(LockOutcome::HeldElsewhere)
        }
        Err(error) => Err(error),
    }
}

/// How old the lock file is (`None` when it vanished or has no mtime).
pub fn lock_age(path: &Path) -> Option<Duration> {
    let modified = fs::metadata(path).ok()?.modified().ok()?;
    SystemTime::now().duration_since(modified).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(port: u16) -> SidecarRegistration {
        SidecarRegistration {
            pid: 4242,
            port,
            slug: "parakeet".to_string(),
            model_id: "parakeet-v3-q4km-s16".to_string(),
            gguf: "/models/a.gguf".to_string(),
            engine_path: "/engines/starling-serve-cpu".to_string(),
            owner_pid: 111,
        }
    }

    #[test]
    fn registration_round_trips_atomically() {
        let dir = tempfile::tempdir().expect("tempdir");
        assert_eq!(read_registration(dir.path()), None);
        write_registration(dir.path(), &sample(8123)).expect("write");
        assert_eq!(read_registration(dir.path()), Some(sample(8123)));
        // No tmp file left behind.
        assert!(!dir.path().join("sidecar.json.tmp").exists());
    }

    #[test]
    fn corrupt_registration_reads_as_none() {
        let dir = tempfile::tempdir().expect("tempdir");
        fs::write(registry_path(dir.path()), "{ not json").expect("write junk");
        assert_eq!(read_registration(dir.path()), None);
    }

    #[test]
    fn removal_respects_ownership() {
        let dir = tempfile::tempdir().expect("tempdir");
        write_registration(dir.path(), &sample(9000)).expect("write");
        // A different owner's file survives.
        remove_registration(dir.path(), 999);
        assert!(read_registration(dir.path()).is_some());
        // The named owner removes it.
        remove_registration(dir.path(), 111);
        assert_eq!(read_registration(dir.path()), None);
    }

    #[test]
    fn spawn_lock_has_exactly_one_winner() {
        let dir = tempfile::tempdir().expect("tempdir");
        let first = match try_spawn_lock(dir.path()).expect("first lock") {
            LockOutcome::Acquired(lock) => lock,
            other => panic!("expected Acquired, got {other:?}"),
        };
        assert!(matches!(
            try_spawn_lock(dir.path()).expect("second lock"),
            LockOutcome::HeldElsewhere
        ));
        first.release();
        assert!(!dir.path().join(SPAWN_LOCK_FILE).exists());
        // Released: the next taker wins again.
        assert!(matches!(
            try_spawn_lock(dir.path()).expect("third lock"),
            LockOutcome::Acquired(_)
        ));
    }

    #[cfg(unix)]
    fn backdate(path: &Path, old: SystemTime) {
        // `File::set_times` is not stable; utimensat is.
        use std::os::unix::ffi::OsStrExt;
        let seconds = old
            .duration_since(SystemTime::UNIX_EPOCH)
            .expect("old is after the epoch")
            .as_secs() as i64;
        let times = [
            libc::timespec {
                tv_sec: seconds,
                tv_nsec: 0,
            },
            libc::timespec {
                tv_sec: seconds,
                tv_nsec: 0,
            },
        ];
        let cpath = std::ffi::CString::new(path.as_os_str().as_bytes()).expect("path");
        unsafe {
            assert_eq!(
                libc::utimensat(libc::AT_FDCWD, cpath.as_ptr(), times.as_ptr(), 0),
                0
            );
        }
    }

    #[test]
    fn stale_spawn_lock_is_replaced() {
        let dir = tempfile::tempdir().expect("tempdir");
        let lock = dir.path().join(SPAWN_LOCK_FILE);
        fs::write(&lock, "stale").expect("write stale lock");
        // Backdate it beyond the stale threshold.
        let old = SystemTime::now() - SPAWN_LOCK_STALE_AFTER - Duration::from_secs(5);
        #[cfg(unix)]
        backdate(&lock, old);
        assert!(matches!(
            try_spawn_lock(dir.path()).expect("lock"),
            LockOutcome::Acquired(_)
        ));
    }
}
