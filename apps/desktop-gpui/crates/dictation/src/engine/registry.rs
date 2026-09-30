//! The sidecar registry: how a second app instance reuses the first
//! instance's sidecar instead of spawning a duplicate (#362 step 3).
//!
//! `<state_dir>/sidecar.json` names the one shared sidecar (pid, port,
//! model). A manager that finds a live, warm, model-matching entry
//! ATTACHES to it — it never stops a process it does not own — while the
//! owner removes the file on shutdown only if the file still names it.
//! A per-model `<state_dir>/spawn-<model>.lock` (created with
//! `create_new`, so exactly one winner, holding the winner's pid)
//! serializes instances launching the same model; the loser waits for
//! the winner's registry entry instead of racing it.
//!
//! `<state_dir>/leases/` carries takes across instances (#363): an
//! attached instance's lease writes a marker naming the engine, and the
//! owner does not stop a draining engine while a live instance's marker
//! remains. The owner first writes a `retired` marker; an instance that
//! finds it after writing its lease marker backs out, so a take either
//! is counted by the owner or never starts on the retiring engine.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime};

use serde::{Deserialize, Serialize};

/// The registry file name inside the state dir.
pub const REGISTRY_FILE: &str = "sidecar.json";
/// A spawn lock whose holder process is gone is stale at once; one older
/// than this is stale even with a live holder (it wedged mid-spawn). Well
/// above the slowest model load, so a slow but healthy spawn is never
/// raced.
pub const SPAWN_LOCK_STALE_AFTER: Duration = Duration::from_secs(10 * 60);

/// `<state_dir>/spawn-<model>.lock`: one lock per model, so instances
/// starting different models never wait for each other.
pub fn spawn_lock_path(state_dir: &Path, model_id: &str) -> PathBuf {
    let safe: String = model_id
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' {
                c
            } else {
                '_'
            }
        })
        .collect();
    state_dir.join(format!("spawn-{safe}.lock"))
}

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

/// Distinguishes temp and lease files written by one process.
static FILE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

fn next_sequence() -> u64 {
    FILE_SEQUENCE.fetch_add(1, Ordering::Relaxed)
}

/// Writes the registration atomically (tmp + rename) so a concurrent
/// reader never sees a half-written file. The tmp name is unique per
/// writer, so two instances writing at once cannot rename each other's
/// half-written content into place.
pub fn write_registration(state_dir: &Path, registration: &SidecarRegistration) -> io::Result<()> {
    fs::create_dir_all(state_dir)?;
    let path = registry_path(state_dir);
    let tmp = state_dir.join(format!(
        "{REGISTRY_FILE}.{}.{}.tmp",
        std::process::id(),
        next_sequence()
    ));
    let json = serde_json::to_string_pretty(registration)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    if let Err(error) = fs::write(&tmp, json).and_then(|()| fs::rename(&tmp, &path)) {
        let _ = fs::remove_file(&tmp);
        return Err(error);
    }
    Ok(())
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

/// Tries to create `model_id`'s spawn lock with `create_new` — exactly
/// one concurrent instance wins; the file records the winner's pid. An
/// existing lock is stale, and replaced, when its holder process is gone
/// or it is older than [`SPAWN_LOCK_STALE_AFTER`].
pub fn try_spawn_lock(state_dir: &Path, model_id: &str) -> io::Result<LockOutcome> {
    let _ = fs::create_dir_all(state_dir);
    let path = spawn_lock_path(state_dir, model_id);
    match create_lock_file(&path) {
        Ok(()) => Ok(LockOutcome::Acquired(SpawnLock {
            path,
            released: false,
        })),
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
            if lock_is_stale(&path) {
                // Its creator cannot release it anymore. Another instance
                // may race us to the replacement; `create_new` still has
                // exactly one winner.
                let _ = fs::remove_file(&path);
                return match create_lock_file(&path) {
                    Ok(()) => Ok(LockOutcome::Acquired(SpawnLock {
                        path,
                        released: false,
                    })),
                    Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
                        Ok(LockOutcome::HeldElsewhere)
                    }
                    Err(error) => Err(error),
                };
            }
            Ok(LockOutcome::HeldElsewhere)
        }
        Err(error) => Err(error),
    }
}

fn create_lock_file(path: &Path) -> io::Result<()> {
    use std::io::Write;
    let mut file = fs::File::options()
        .write(true)
        .create_new(true)
        .open(path)?;
    // Best effort: a lock without a pid only goes stale by age.
    let _ = write!(file, "{}", std::process::id());
    Ok(())
}

/// Whether the lock at `path` can no longer be released by its holder.
/// A lock that is still being written (no pid yet) is judged by age.
fn lock_is_stale(path: &Path) -> bool {
    let holder = fs::read_to_string(path)
        .ok()
        .and_then(|text| text.trim().parse::<u32>().ok());
    if holder.is_some_and(|pid| !process_alive(pid)) {
        return true;
    }
    lock_age(path).is_some_and(|age| age >= SPAWN_LOCK_STALE_AFTER)
}

/// The lease directory name inside the state dir.
pub const LEASES_DIR: &str = "leases";

/// Identifies one engine process across instances. The port guards
/// against a recycled pid matching an old marker.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EngineKey {
    pub pid: u32,
    pub port: u16,
}

impl EngineKey {
    fn prefix(&self) -> String {
        format!("{}-{}", self.pid, self.port)
    }
}

fn leases_dir(state_dir: &Path) -> PathBuf {
    state_dir.join(LEASES_DIR)
}

fn retired_path(state_dir: &Path, key: EngineKey) -> PathBuf {
    leases_dir(state_dir).join(format!("{}.retired", key.prefix()))
}

/// An attached instance's take on a shared engine. Dropping it removes
/// the marker, which lets the owner stop a draining engine.
#[derive(Debug)]
pub struct LeaseMarker {
    path: PathBuf,
}

impl Drop for LeaseMarker {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

/// The outcome of [`acquire_lease_marker`].
#[derive(Debug)]
pub enum MarkerOutcome {
    /// The take is visible to the owner.
    Held(LeaseMarker),
    /// The owner is retiring this engine: start the take elsewhere.
    Retired,
    /// The marker could not be written; the take proceeds uncounted
    /// (the pre-marker behavior).
    Unavailable,
}

/// Writes a lease marker for `key`, then checks the retired marker. The
/// order matters: the owner writes `retired` before it counts markers,
/// so either the owner sees this marker or this call sees `retired`.
pub fn acquire_lease_marker(state_dir: &Path, key: EngineKey) -> MarkerOutcome {
    let dir = leases_dir(state_dir);
    if fs::create_dir_all(&dir).is_err() {
        return MarkerOutcome::Unavailable;
    }
    let path = dir.join(format!(
        "{}.{}.{}.lease",
        key.prefix(),
        std::process::id(),
        next_sequence()
    ));
    if fs::write(&path, b"").is_err() {
        return MarkerOutcome::Unavailable;
    }
    let marker = LeaseMarker { path };
    if retired_path(state_dir, key).exists() {
        drop(marker);
        return MarkerOutcome::Retired;
    }
    MarkerOutcome::Held(marker)
}

/// Whether the owner has started retiring `key`.
pub fn is_retired(state_dir: &Path, key: EngineKey) -> bool {
    retired_path(state_dir, key).exists()
}

/// The owner marks `key` as draining: no new take from another instance
/// starts on it from now on.
pub fn retire_engine(state_dir: &Path, key: EngineKey) {
    let dir = leases_dir(state_dir);
    let _ = fs::create_dir_all(&dir);
    let _ = fs::write(retired_path(state_dir, key), b"");
}

/// Undoes [`retire_engine`] (a cancelled drain swap keeps the engine).
pub fn unretire_engine(state_dir: &Path, key: EngineKey) {
    let _ = fs::remove_file(retired_path(state_dir, key));
}

/// Takes attached instances hold on `key`. Only attached instances
/// write markers (the owner counts its own takes in memory), so every
/// marker is foreign. Markers whose holder process is gone (a crashed
/// instance) are swept and not counted.
pub fn foreign_leases(state_dir: &Path, key: EngineKey) -> usize {
    let Ok(entries) = fs::read_dir(leases_dir(state_dir)) else {
        return 0;
    };
    let prefix = format!("{}.", key.prefix());
    let mut count = 0;
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        let Some(rest) = name
            .strip_prefix(&prefix)
            .and_then(|rest| rest.strip_suffix(".lease"))
        else {
            continue;
        };
        let Some(holder) = rest
            .split('.')
            .next()
            .and_then(|pid| pid.parse::<u32>().ok())
        else {
            continue;
        };
        if process_alive(holder) {
            count += 1;
        } else {
            let _ = fs::remove_file(entry.path());
        }
    }
    count
}

/// Removes every marker for a stopped engine.
pub fn clear_engine_leases(state_dir: &Path, key: EngineKey) {
    let Ok(entries) = fs::read_dir(leases_dir(state_dir)) else {
        return;
    };
    let prefix = format!("{}.", key.prefix());
    for entry in entries.flatten() {
        if entry.file_name().to_string_lossy().starts_with(&prefix) {
            let _ = fs::remove_file(entry.path());
        }
    }
}

/// Whether process `pid` is alive. When the OS cannot tell, the process
/// counts as alive: the drain hard cap still bounds the wait.
#[cfg(unix)]
pub fn process_alive(pid: u32) -> bool {
    // 0 and values past pid_t would address process groups, not a pid.
    let Ok(pid) = libc::pid_t::try_from(pid) else {
        return false;
    };
    if pid <= 0 {
        return false;
    }
    // SAFETY: kill(2) with signal 0 performs existence and permission
    // checks only; no signal is delivered.
    if unsafe { libc::kill(pid, 0) } == 0 {
        return true;
    }
    // EPERM: exists but belongs to another user. Only ESRCH means gone.
    io::Error::last_os_error().raw_os_error() != Some(libc::ESRCH)
}

/// Whether process `pid` is alive. When the OS cannot tell, the process
/// counts as alive: the drain hard cap still bounds the wait.
#[cfg(windows)]
pub fn process_alive(pid: u32) -> bool {
    use windows_sys::Win32::Foundation::{
        CloseHandle, GetLastError, ERROR_INVALID_PARAMETER, STILL_ACTIVE,
    };
    use windows_sys::Win32::System::Threading::{
        GetExitCodeProcess, OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION,
    };
    // SAFETY: plain Win32 calls on a handle we own and close.
    unsafe {
        let handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
        if handle.is_null() {
            return GetLastError() != ERROR_INVALID_PARAMETER;
        }
        let mut code: u32 = 0;
        let ok = GetExitCodeProcess(handle, &mut code);
        CloseHandle(handle);
        ok == 0 || code == STILL_ACTIVE as u32
    }
}

#[cfg(not(any(unix, windows)))]
pub fn process_alive(_pid: u32) -> bool {
    true
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
        let leftovers: Vec<_> = fs::read_dir(dir.path())
            .expect("read dir")
            .flatten()
            .filter(|entry| entry.file_name().to_string_lossy().ends_with(".tmp"))
            .collect();
        assert!(leftovers.is_empty(), "tmp files left: {leftovers:?}");
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
    fn spawn_lock_has_exactly_one_winner_per_model() {
        let dir = tempfile::tempdir().expect("tempdir");
        let first = match try_spawn_lock(dir.path(), "model-a").expect("first lock") {
            LockOutcome::Acquired(lock) => lock,
            other => panic!("expected Acquired, got {other:?}"),
        };
        // Our own live pid holds it: not stale.
        assert!(matches!(
            try_spawn_lock(dir.path(), "model-a").expect("second lock"),
            LockOutcome::HeldElsewhere
        ));
        // Another model does not wait for it.
        assert!(matches!(
            try_spawn_lock(dir.path(), "model-b").expect("other model"),
            LockOutcome::Acquired(_)
        ));
        first.release();
        assert!(!spawn_lock_path(dir.path(), "model-a").exists());
        // Released: the next taker wins again.
        assert!(matches!(
            try_spawn_lock(dir.path(), "model-a").expect("third lock"),
            LockOutcome::Acquired(_)
        ));
    }

    #[cfg(unix)]
    #[test]
    fn a_lock_whose_holder_died_is_replaced_at_once() {
        let dir = tempfile::tempdir().expect("tempdir");
        fs::write(
            spawn_lock_path(dir.path(), "model-a"),
            format!("{}", i32::MAX),
        )
        .expect("write dead holder's lock");
        assert!(matches!(
            try_spawn_lock(dir.path(), "model-a").expect("lock"),
            LockOutcome::Acquired(_)
        ));
    }

    #[test]
    fn lease_markers_count_foreign_live_holders_only() {
        let dir = tempfile::tempdir().expect("tempdir");
        let key = EngineKey {
            pid: 4242,
            port: 8123,
        };
        let leases = dir.path().join(LEASES_DIR);
        fs::create_dir_all(&leases).expect("leases dir");
        // A marker from this (live) process counts, and drops with it.
        let held = match acquire_lease_marker(dir.path(), key) {
            MarkerOutcome::Held(marker) => marker,
            other => panic!("expected Held, got {other:?}"),
        };
        assert_eq!(foreign_leases(dir.path(), key), 1);
        drop(held);
        assert_eq!(foreign_leases(dir.path(), key), 0);
        // A live holder (pid 1 always exists on unix) counts; a dead one
        // is swept. Another engine's marker is ignored.
        #[cfg(unix)]
        {
            fs::write(leases.join("4242-8123.1.0.lease"), b"").expect("live marker");
            fs::write(leases.join(format!("4242-8123.{}.0.lease", i32::MAX)), b"")
                .expect("dead marker");
            fs::write(leases.join("4242-9999.1.0.lease"), b"").expect("other engine");
            assert_eq!(foreign_leases(dir.path(), key), 1);
            assert!(!leases
                .join(format!("4242-8123.{}.0.lease", i32::MAX))
                .exists());
        }
        clear_engine_leases(dir.path(), key);
        assert_eq!(foreign_leases(dir.path(), key), 0);
        #[cfg(unix)]
        assert!(leases.join("4242-9999.1.0.lease").exists());
    }

    #[test]
    fn a_retired_engine_refuses_new_markers() {
        let dir = tempfile::tempdir().expect("tempdir");
        let key = EngineKey { pid: 7, port: 7000 };
        retire_engine(dir.path(), key);
        assert!(is_retired(dir.path(), key));
        assert!(matches!(
            acquire_lease_marker(dir.path(), key),
            MarkerOutcome::Retired
        ));
        // The backed-out marker is gone.
        assert_eq!(fs::read_dir(dir.path().join(LEASES_DIR)).unwrap().count(), 1);
        unretire_engine(dir.path(), key);
        assert!(matches!(
            acquire_lease_marker(dir.path(), key),
            MarkerOutcome::Held(_)
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
        let lock = spawn_lock_path(dir.path(), "model-a");
        fs::write(&lock, "stale").expect("write stale lock");
        // Backdate it beyond the stale threshold.
        let old = SystemTime::now() - SPAWN_LOCK_STALE_AFTER - Duration::from_secs(5);
        #[cfg(unix)]
        backdate(&lock, old);
        assert!(matches!(
            try_spawn_lock(dir.path(), "model-a").expect("lock"),
            LockOutcome::Acquired(_)
        ));
    }
}
