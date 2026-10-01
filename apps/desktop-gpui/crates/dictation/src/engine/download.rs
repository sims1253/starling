//! Verified model installs (#363 step 2, the runtime half of #222's
//! download machinery).
//!
//! A model is "installed" only when the final file exists with the exact
//! catalog size AND a `<file>.verified` marker naming the catalog sha256.
//! Downloads stream to `<file>.part` (incremental sha256, resumable via
//! HTTP `Range`), verify, fsync, rename, then write the marker. A user
//! who hand-placed a file into the models dir (Handy's offline
//! alternative) gets `NeedsVerification`: activation hashes it first and
//! writes the marker on success — a mismatch refuses activation without
//! deleting the user's file. A failed or cancelled download never
//! touches the active model or any other file. One `<file>.lock` per
//! model (holding the downloader's pid) keeps two app instances from
//! writing the same `.part`: the second waits, then finds the model
//! installed.

use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use sha2::{Digest, Sha256};

use crate::engine::catalog::CatalogEntry;
use crate::engine::manager::InstallState;

/// The suffix for in-flight downloads.
pub const PART_SUFFIX: &str = "part";
/// The suffix for the verification marker (content: the sha256 hex).
pub const VERIFIED_SUFFIX: &str = "verified";
/// The suffix for the per-model download lock (content: holder pid).
pub const LOCK_SUFFIX: &str = "lock";

/// The three file names a model resolves to inside the models dir.
#[derive(Clone, Debug)]
pub struct ModelFiles {
    /// The final, verified model file.
    pub final_path: PathBuf,
    /// The in-flight download target.
    pub part_path: PathBuf,
    /// `<final>.verified`, containing the sha hex on success.
    pub marker_path: PathBuf,
    /// `<final>.lock`, held by the one process downloading the model.
    pub lock_path: PathBuf,
}

/// `<models_dir>/<file_name>` plus its `.part`/`.verified` siblings.
pub fn model_files(models_dir: &Path, entry: &CatalogEntry) -> ModelFiles {
    // Suffixes are appended to the full file name: `with_extension`
    // would replace the last extension ("a.gguf" -> "a.part"), so names
    // differing only by extension would share sibling files.
    let sibling = |suffix: &str| models_dir.join(format!("{}.{suffix}", entry.file_name));
    ModelFiles {
        final_path: models_dir.join(&entry.file_name),
        part_path: sibling(PART_SUFFIX),
        marker_path: sibling(VERIFIED_SUFFIX),
        lock_path: sibling(LOCK_SUFFIX),
    }
}

/// Disk truth for one catalog entry. Installed requires file + size +
/// marker match; a right-size file without a matching marker is
/// [`InstallState::NeedsVerification`] (activation hashes it first). A
/// wrong-size file can never verify, so it is `Failed` with the reason
/// and the download action that replaces it.
pub fn scan_install(models_dir: &Path, entry: &CatalogEntry) -> InstallState {
    let files = model_files(models_dir, entry);
    let Some(meta) = fs::metadata(&files.final_path).ok() else {
        return InstallState::NotInstalled;
    };
    if meta.len() != entry.size_bytes {
        return InstallState::Failed(wrong_size_message(&files.final_path, meta.len(), entry));
    }
    match fs::read_to_string(&files.marker_path) {
        Ok(marker) if marker.trim().eq_ignore_ascii_case(&entry.sha256) => InstallState::Installed,
        _ => InstallState::NeedsVerification,
    }
}

/// Writes the verification marker for a file whose digest already
/// matched.
pub fn write_marker(marker_path: &Path, sha256_hex: &str) -> std::io::Result<()> {
    if let Some(parent) = marker_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(marker_path, sha256_hex)
}

/// Verifies a user-placed file (`NeedsVerification`): hash it, and on
/// success write the marker. On mismatch the file is NEVER deleted — it
/// is the user's; the error says it does not match.
pub fn verify_placed_file(models_dir: &Path, entry: &CatalogEntry) -> Result<(), String> {
    let files = model_files(models_dir, entry);
    // Size first: it is cheap, and a wrong-size file would otherwise be
    // reported as a checksum mismatch.
    let meta = fs::metadata(&files.final_path)
        .map_err(|error| format!("cannot stat {}: {error}", files.final_path.display()))?;
    if meta.len() != entry.size_bytes {
        return Err(wrong_size_message(&files.final_path, meta.len(), entry));
    }
    let digest = crate::engine::bundle::sha256_file(&files.final_path)
        .map_err(|error| format!("cannot read {}: {error}", files.final_path.display()))?;
    if !digest.eq_ignore_ascii_case(&entry.sha256) {
        return Err(format!(
            "the file {} does not match the expected checksum for {}; \
             it was not changed — replace it or delete it and download again",
            files.final_path.display(),
            entry.id
        ));
    }
    write_marker(&files.marker_path, &entry.sha256)
        .map_err(|error| format!("cannot write the verification marker: {error}"))?;
    Ok(())
}

fn wrong_size_message(path: &Path, size: u64, entry: &CatalogEntry) -> String {
    format!(
        "the file {} has {size} bytes, but the catalog expects {}; \
         it was not changed — replace it or download again",
        path.display(),
        entry.size_bytes
    )
}

/// How long connecting to the model host may take.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
/// How long the response may go without delivering a byte. A healthy
/// download of a large model takes as long as it takes; only a stalled
/// one fails.
const READ_INACTIVITY_TIMEOUT: Duration = Duration::from_secs(60);
/// How often a stalled read re-checks the cancel flag.
const CANCEL_POLL: Duration = Duration::from_millis(250);
/// How often a download waiting for another process's lock re-checks.
const LOCK_POLL: Duration = Duration::from_millis(200);
/// A lock file without a readable pid (its writer died between create
/// and write) is stale after this long.
const PIDLESS_LOCK_STALE_AFTER: Duration = Duration::from_secs(10);

/// The held per-model download lock; dropping it removes the file.
struct DownloadLock {
    path: PathBuf,
}

impl Drop for DownloadLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

/// Takes `files.lock_path`, waiting while a live process holds it. A lock
/// whose holder is gone (a crashed instance) is replaced; its `.part` is
/// then simply resumed.
fn acquire_download_lock(
    files: &ModelFiles,
    cancel: &AtomicBool,
) -> Result<DownloadLock, DownloadError> {
    loop {
        if cancel.load(Ordering::Relaxed) {
            return Err(DownloadError::Cancelled);
        }
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&files.lock_path)
        {
            Ok(mut file) => {
                let _ = write!(file, "{}", std::process::id());
                return Ok(DownloadLock {
                    path: files.lock_path.clone(),
                });
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                if download_lock_is_stale(&files.lock_path) {
                    let _ = fs::remove_file(&files.lock_path);
                    continue;
                }
                std::thread::sleep(LOCK_POLL);
            }
            Err(error) => return Err(io_error(error)),
        }
    }
}

fn download_lock_is_stale(path: &Path) -> bool {
    let holder = fs::read_to_string(path)
        .ok()
        .and_then(|text| text.trim().parse::<u32>().ok());
    match holder {
        Some(pid) => !crate::engine::registry::process_alive(pid),
        None => fs::metadata(path)
            .and_then(|meta| meta.modified())
            .ok()
            .and_then(|modified| std::time::SystemTime::now().duration_since(modified).ok())
            .is_some_and(|age| age >= PIDLESS_LOCK_STALE_AFTER),
    }
}

/// Why a download did not produce an installed model. Each carries a
/// sentence; none of them touch the active model.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum DownloadError {
    #[error("download failed: {0}")]
    Http(String),
    #[error("download I/O error: {0}")]
    Io(String),
    #[error(
        "the downloaded model does not match its expected checksum; \
         the partial file was removed. Download it again."
    )]
    ChecksumMismatch,
    #[error(
        "the download delivered {got} bytes, but the catalog expects {expected}; \
         the partial file was removed. Download it again."
    )]
    Size { expected: u64, got: u64 },
    #[error("the download was cancelled")]
    Cancelled,
}

/// Blocking download of one catalog entry to the models dir. Run on a
/// dedicated worker thread (the manager does); the async reqwest client
/// runs on a private current-thread runtime, the same pattern as
/// `client.rs`.
///
/// `progress(done, total)` fires as bytes land (also during a resumed
/// run, with `done` starting at the resumed offset), `cancel` is checked
/// per chunk. On success the final file is verified, fsynced, renamed
/// into place, and the marker written — the return is an installed
/// model.
pub fn download_model(
    models_dir: &Path,
    entry: &CatalogEntry,
    progress: &dyn Fn(u64, u64),
    cancel: &AtomicBool,
) -> Result<(), DownloadError> {
    let files = model_files(models_dir, entry);
    fs::create_dir_all(models_dir).map_err(io_error)?;
    // Held through verify, rename and marker: no other process may write
    // the `.part` while (or after) this one installs it.
    let _lock = acquire_download_lock(&files, cancel)?;
    if matches!(scan_install(models_dir, entry), InstallState::Installed) {
        // Another instance finished the download while we waited.
        progress(entry.size_bytes, entry.size_bytes);
        return Ok(());
    }
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|error| DownloadError::Io(error.to_string()))?;
    runtime.block_on(download_async(&files, entry, progress, cancel))
}

async fn download_async(
    files: &ModelFiles,
    entry: &CatalogEntry,
    progress: &dyn Fn(u64, u64),
    cancel: &AtomicBool,
) -> Result<(), DownloadError> {
    // No total-request timeout: it would cover the whole body, and a
    // 553 MB model on a slow link must not fail while bytes keep arriving.
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::limited(5))
        .connect_timeout(CONNECT_TIMEOUT)
        .read_timeout(READ_INACTIVITY_TIMEOUT)
        .build()
        .map_err(|error| DownloadError::Http(error.to_string()))?;

    // Resume bookkeeping: an existing smaller .part contributes its bytes
    // (and its hash) and the request carries a Range header. A .part at
    // or past the expected size is junk from a lying server — restart.
    let mut offset: u64 = 0;
    let mut hasher = Sha256::new();
    if files.part_path.exists() {
        let size = fs::metadata(&files.part_path).map_err(io_error)?.len();
        if size < entry.size_bytes {
            let part = File::open(&files.part_path).map_err(io_error)?;
            let mut reader = BufReader::new(part);
            let mut buffer = [0u8; 128 * 1024];
            loop {
                let read = reader.read(&mut buffer).map_err(io_error)?;
                if read == 0 {
                    break;
                }
                hasher.update(&buffer[..read]);
            }
            offset = size;
        } else {
            let _ = fs::remove_file(&files.part_path);
        }
    }

    let (response, resumed) = loop {
        let mut request = client.get(&entry.url);
        if offset > 0 {
            request = request.header("Range", format!("bytes={offset}-"));
        }
        let response = request.send().await.map_err(|error| {
            DownloadError::Http(format!("cannot reach {}: {error}", entry.url))
        })?;
        let status = response.status();
        if !status.is_success() {
            return Err(DownloadError::Http(format!(
                "the server answered {} for {}",
                status.as_u16(),
                entry.url
            )));
        }
        if status.as_u16() != 206 {
            // A full answer of the wrong size can never install: fail
            // before downloading it.
            if let Some(length) = response.content_length() {
                if length != entry.size_bytes {
                    return Err(DownloadError::Http(format!(
                        "the server's file {} has {length} bytes, but the catalog expects {}",
                        entry.url, entry.size_bytes
                    )));
                }
            }
            break (response, false);
        }
        // A 206 only resumes when it starts where the .part ends. One
        // that starts elsewhere (a proxy answering from byte 0) cannot be
        // appended: drop the .part and ask again for the whole file.
        let content_range = response
            .headers()
            .get(reqwest::header::CONTENT_RANGE)
            .and_then(|value| value.to_str().ok())
            .map(str::to_string);
        let start = content_range.as_deref().and_then(content_range_start);
        let total = content_range.as_deref().and_then(content_range_total);
        if let Some(total) = total.filter(|total| *total != entry.size_bytes) {
            // The remote file changed under the pinned URL: the resume
            // prefix belongs to a different file.
            let _ = fs::remove_file(&files.part_path);
            return Err(DownloadError::Http(format!(
                "the server's file {} now has {total} bytes, but the catalog expects {}; \
                 the partial download was removed",
                entry.url, entry.size_bytes
            )));
        }
        if offset > 0 && start == Some(offset) {
            break (response, true);
        }
        if offset == 0 {
            return Err(DownloadError::Http(format!(
                "the server answered a partial response for {} without being asked to resume",
                entry.url
            )));
        }
        drop(response);
        let _ = fs::remove_file(&files.part_path);
        offset = 0;
        hasher = Sha256::new();
    };

    let mut file = if resumed {
        OpenOptions::new()
            .append(true)
            .open(&files.part_path)
            .map_err(io_error)?
    } else {
        // 200 (the server ignored the Range header): restart clean.
        offset = 0;
        hasher = Sha256::new();
        File::create(&files.part_path).map_err(io_error)?
    };

    let mut done = offset;
    progress(done, entry.size_bytes);
    let mut response = response;
    loop {
        // Poll the cancel flag while a slow chunk is pending (the read
        // inactivity timeout can be much longer than a user waits).
        let chunk = loop {
            if cancel.load(Ordering::Relaxed) {
                // Keep .part for a later resume; drop the connection.
                return Err(DownloadError::Cancelled);
            }
            match tokio::time::timeout(CANCEL_POLL, response.chunk()).await {
                Ok(result) => break result,
                Err(_elapsed) => continue,
            }
        };
        let Some(chunk) = chunk.map_err(|error| DownloadError::Http(error.to_string()))? else {
            break;
        };
        if cancel.load(Ordering::Relaxed) {
            return Err(DownloadError::Cancelled);
        }
        if done + chunk.len() as u64 > entry.size_bytes {
            // A server that keeps sending past the catalog size is lying;
            // do not buffer or keep it.
            let _ = fs::remove_file(&files.part_path);
            return Err(DownloadError::Size {
                expected: entry.size_bytes,
                got: done + chunk.len() as u64,
            });
        }
        file.write_all(&chunk).map_err(io_error)?;
        hasher.update(&chunk);
        done += chunk.len() as u64;
        progress(done, entry.size_bytes);
    }
    file.flush().map_err(io_error)?;
    file.sync_all().map_err(io_error)?;
    drop(file);

    if done != entry.size_bytes {
        let _ = fs::remove_file(&files.part_path);
        return Err(DownloadError::Size {
            expected: entry.size_bytes,
            got: done,
        });
    }
    let digest = crate::engine::bundle::to_hex(&hasher.finalize());
    if !digest.eq_ignore_ascii_case(&entry.sha256) {
        let _ = fs::remove_file(&files.part_path);
        return Err(DownloadError::ChecksumMismatch);
    }
    fs::rename(&files.part_path, &files.final_path).map_err(io_error)?;
    write_marker(&files.marker_path, &entry.sha256).map_err(io_error)?;
    Ok(())
}

/// Removes a model's final file, `.part`, and marker. Refusing to touch
/// an active model is the *caller's* check (the manager knows what is
/// active); this only does the file work.
pub fn delete_model_files(models_dir: &Path, entry: &CatalogEntry) -> Result<(), String> {
    let files = model_files(models_dir, entry);
    for path in [&files.final_path, &files.part_path, &files.marker_path] {
        match fs::remove_file(path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(format!("cannot remove {}: {error}", path.display()));
            }
        }
    }
    Ok(())
}

/// The first byte of `Content-Range: bytes <start>-<end>/<total>`.
fn content_range_start(value: &str) -> Option<u64> {
    value
        .trim()
        .strip_prefix("bytes ")?
        .split('-')
        .next()?
        .trim()
        .parse()
        .ok()
}

/// The total of `Content-Range: bytes <start>-<end>/<total>`; `None`
/// when absent or unknown (`*`).
fn content_range_total(value: &str) -> Option<u64> {
    value.trim().rsplit('/').next()?.trim().parse().ok()
}

fn io_error(error: std::io::Error) -> DownloadError {
    DownloadError::Io(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;
    use std::sync::Arc;

    /// A minimal HTTP/1.1 file server honoring `Range` (one thread,
    /// `connection: close` per response) — the same double the
    /// integration tests use, kept here for unit-testing resume.
    fn spawn_file_server(bytes: Arc<Vec<u8>>) -> (String, std::thread::JoinHandle<()>) {
        spawn_file_server_with(bytes, false)
    }

    /// `partial_from_zero`: answer every Range request with a 206 that
    /// starts at byte 0 (a misbehaving proxy).
    fn spawn_file_server_with(
        bytes: Arc<Vec<u8>>,
        partial_from_zero: bool,
    ) -> (String, std::thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");
        let handle = std::thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                let mut stream = stream;
                let mut head = Vec::new();
                let mut byte = [0u8; 1];
                // Read until the end of the request head.
                while !head.ends_with(b"\r\n\r\n") {
                    match stream.read(&mut byte) {
                        Ok(0) | Err(_) => break,
                        Ok(_) => head.push(byte[0]),
                    }
                }
                let text = String::from_utf8_lossy(&head).to_string();
                let range = text.lines().find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    if !name.trim().eq_ignore_ascii_case("range") {
                        return None;
                    }
                    let start: u64 = value
                        .trim()
                        .trim_start_matches("bytes=")
                        .split('-')
                        .next()?
                        .parse()
                        .ok()?;
                    Some(start)
                });
                let (status, body_range) = match range {
                    Some(_) if partial_from_zero => ("206 Partial Content", 0..bytes.len()),
                    Some(start) if start < bytes.len() as u64 => {
                        ("206 Partial Content", start as usize..bytes.len())
                    }
                    Some(_) => ("416 Range Not Satisfiable", 0..0),
                    None => ("200 OK", 0..bytes.len()),
                };
                let content_range = if status.starts_with("206") {
                    format!(
                        "content-range: bytes {}-{}/{}\r\n",
                        body_range.start,
                        body_range.end - 1,
                        bytes.len()
                    )
                } else {
                    String::new()
                };
                let body = &bytes[body_range];
                let head = format!(
                    "HTTP/1.1 {status}\r\ncontent-length: {}\r\n{content_range}accept-ranges: bytes\r\nconnection: close\r\n\r\n",
                    body.len()
                );
                let _ = stream.write_all(head.as_bytes());
                let _ = stream.write_all(body);
                let _ = stream.flush();
            }
        });
        (format!("http://{addr}/models/model.gguf"), handle)
    }

    fn entry_for(url: &str, bytes: &[u8]) -> CatalogEntry {
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        CatalogEntry::new(
            "test-model",
            "Test model",
            "parakeet",
            url,
            bytes.len() as u64,
            &crate::engine::bundle::to_hex(&hasher.finalize()),
            false,
            "test",
        )
    }

    fn download_all(
        models_dir: &Path,
        entry: &CatalogEntry,
        cancel: &AtomicBool,
    ) -> Result<(), DownloadError> {
        download_model(models_dir, entry, &|_, _| {}, cancel)
    }

    #[test]
    fn downloads_verify_and_install() {
        let bytes: Vec<u8> = (0..100_000u32).map(|i| (i % 251) as u8).collect();
        let (url, server) = spawn_file_server(Arc::new(bytes.clone()));
        let entry = entry_for(&url, &bytes);
        let dir = tempfile::tempdir().expect("tempdir");

        let progresses = std::sync::Mutex::new(Vec::new());
        download_model(
            dir.path(),
            &entry,
            &|done, total| progresses.lock().unwrap().push((done, total)),
            &AtomicBool::new(false),
        )
        .expect("download succeeds");
        let files = model_files(dir.path(), &entry);
        assert!(files.final_path.exists());
        assert!(!files.part_path.exists());
        assert_eq!(
            fs::read(&files.final_path).expect("final file"),
            bytes,
            "installed bytes must be the served bytes"
        );
        assert_eq!(
            fs::read_to_string(&files.marker_path).expect("marker"),
            entry.sha256
        );
        assert!(matches!(
            scan_install(dir.path(), &entry),
            InstallState::Installed
        ));
        let progresses = progresses.into_inner().unwrap();
        assert!(progresses
            .last()
            .is_some_and(|&(done, total)| done == total));
        drop(server);
    }

    #[test]
    fn checksum_mismatch_deletes_the_part_and_reports() {
        let served: Vec<u8> = vec![7u8; 5000];
        let (url, server) = spawn_file_server(Arc::new(served));
        // The catalog expects different bytes.
        let entry = entry_for(&url, &vec![1u8; 5000]);
        let dir = tempfile::tempdir().expect("tempdir");
        match download_all(dir.path(), &entry, &AtomicBool::new(false)) {
            Err(DownloadError::ChecksumMismatch) => {}
            other => panic!("expected ChecksumMismatch, got {other:?}"),
        }
        let files = model_files(dir.path(), &entry);
        assert!(!files.part_path.exists());
        assert!(!files.final_path.exists());
        assert!(matches!(
            scan_install(dir.path(), &entry),
            InstallState::NotInstalled
        ));
        drop(server);
    }

    #[test]
    fn cancelled_download_keeps_the_part_for_resume() {
        let served: Vec<u8> = vec![3u8; 50_000];
        let (url, server) = spawn_file_server(Arc::new(served.clone()));
        let entry = entry_for(&url, &served);
        let dir = tempfile::tempdir().expect("tempdir");
        let files = model_files(dir.path(), &entry);

        // Pre-place a prefix as .part: the download resumes from it with
        // a Range request (the fake cancellation below then aborts).
        let prefix = &served[..10_000];
        fs::write(&files.part_path, prefix).expect("write part");
        let cancel = AtomicBool::new(true); // cancel before the first chunk
        match download_all(dir.path(), &entry, &cancel) {
            Err(DownloadError::Cancelled) => {}
            other => panic!("expected Cancelled, got {other:?}"),
        }
        assert!(files.part_path.exists(), "resume relies on the kept part");
        assert!(!files.final_path.exists());
        drop(server);
    }

    #[test]
    fn resumes_from_the_part_when_the_range_matches() {
        let served: Vec<u8> = (0..60_000u32).map(|i| (i % 241) as u8).collect();
        let (url, server) = spawn_file_server(Arc::new(served.clone()));
        let entry = entry_for(&url, &served);
        let dir = tempfile::tempdir().expect("tempdir");
        let files = model_files(dir.path(), &entry);
        fs::write(&files.part_path, &served[..25_000]).expect("write part");
        let first_progress = std::sync::Mutex::new(None);
        download_model(
            dir.path(),
            &entry,
            &|done, _| {
                first_progress.lock().unwrap().get_or_insert(done);
            },
            &AtomicBool::new(false),
        )
        .expect("resumed download succeeds");
        assert_eq!(*first_progress.lock().unwrap(), Some(25_000));
        assert_eq!(fs::read(&files.final_path).expect("final"), served);
        drop(server);
    }

    #[test]
    fn a_partial_answer_from_the_wrong_offset_restarts_clean() {
        let served: Vec<u8> = (0..60_000u32).map(|i| (i % 239) as u8).collect();
        let (url, server) = spawn_file_server_with(Arc::new(served.clone()), true);
        let entry = entry_for(&url, &served);
        let dir = tempfile::tempdir().expect("tempdir");
        let files = model_files(dir.path(), &entry);
        fs::write(&files.part_path, &served[..25_000]).expect("write part");
        // The retry without Range gets a 200 with the whole file.
        download_all(dir.path(), &entry, &AtomicBool::new(false))
            .expect("download restarts and succeeds");
        assert_eq!(fs::read(&files.final_path).expect("final"), served);
        drop(server);
    }

    #[test]
    fn content_range_start_parses_the_first_byte() {
        assert_eq!(content_range_start("bytes 100-199/200"), Some(100));
        assert_eq!(content_range_start("bytes 0-9/*"), Some(0));
        assert_eq!(content_range_start("items 1-2/3"), None);
        assert_eq!(content_range_start("bytes */200"), None);
        assert_eq!(content_range_total("bytes 100-199/200"), Some(200));
        assert_eq!(content_range_total("bytes 0-9/*"), None);
    }

    #[test]
    fn a_download_waits_for_another_processs_lock_then_finds_it_installed() {
        let served: Vec<u8> = vec![5u8; 20_000];
        let (url, server) = spawn_file_server(Arc::new(served.clone()));
        let entry = entry_for(&url, &served);
        let dir = tempfile::tempdir().expect("tempdir");
        let files = model_files(dir.path(), &entry);
        // A live holder (this process) has the lock, as another instance
        // mid-download would.
        let held = acquire_download_lock(&files, &AtomicBool::new(false)).expect("lock");
        let models_dir = dir.path().to_path_buf();
        let waiting_entry = entry.clone();
        let waiter = std::thread::spawn(move || {
            download_model(&models_dir, &waiting_entry, &|_, _| {}, &AtomicBool::new(false))
        });
        std::thread::sleep(Duration::from_millis(600));
        assert!(!waiter.is_finished(), "the second download waits for the lock");
        assert!(!files.part_path.exists(), "and writes nothing meanwhile");
        // The holder installs the model, then releases.
        fs::write(&files.final_path, &served).expect("install");
        write_marker(&files.marker_path, &entry.sha256).expect("marker");
        drop(held);
        waiter
            .join()
            .expect("waiter thread")
            .expect("the waiter succeeds without downloading");
        assert_eq!(fs::read(&files.final_path).expect("final"), served);
        assert!(!files.part_path.exists());
        assert!(!files.lock_path.exists());
        drop(server);
    }

    #[cfg(unix)]
    #[test]
    fn a_dead_holders_download_lock_is_replaced() {
        let served: Vec<u8> = vec![6u8; 8_000];
        let (url, server) = spawn_file_server(Arc::new(served.clone()));
        let entry = entry_for(&url, &served);
        let dir = tempfile::tempdir().expect("tempdir");
        let files = model_files(dir.path(), &entry);
        fs::write(&files.lock_path, format!("{}", i32::MAX)).expect("dead lock");
        download_all(dir.path(), &entry, &AtomicBool::new(false)).expect("download");
        assert_eq!(fs::read(&files.final_path).expect("final"), served);
        assert!(!files.lock_path.exists());
        drop(server);
    }

    #[test]
    fn a_full_answer_of_the_wrong_size_fails_before_downloading() {
        let served: Vec<u8> = vec![2u8; 3_000];
        let (url, server) = spawn_file_server(Arc::new(served));
        // The catalog expects a different size.
        let entry = entry_for(&url, &vec![2u8; 4_000]);
        let dir = tempfile::tempdir().expect("tempdir");
        match download_all(dir.path(), &entry, &AtomicBool::new(false)) {
            Err(DownloadError::Http(message)) => {
                assert!(message.contains("3000 bytes"), "got: {message}")
            }
            other => panic!("expected an Http size error, got {other:?}"),
        }
        assert!(!model_files(dir.path(), &entry).part_path.exists());
        drop(server);
    }

    #[test]
    fn placed_file_needs_verification_then_verifies_or_refuses() {
        let served: Vec<u8> = vec![9u8; 4096];
        let entry = entry_for("http://127.0.0.1:9/never-hit.gguf", &served);
        let dir = tempfile::tempdir().expect("tempdir");
        let files = model_files(dir.path(), &entry);

        // Nothing installed.
        assert!(matches!(
            scan_install(dir.path(), &entry),
            InstallState::NotInstalled
        ));
        // Right bytes, no marker: NeedsVerification.
        fs::write(&files.final_path, &served).expect("place file");
        assert!(matches!(
            scan_install(dir.path(), &entry),
            InstallState::NeedsVerification
        ));
        verify_placed_file(dir.path(), &entry).expect("placed file verifies");
        assert!(matches!(
            scan_install(dir.path(), &entry),
            InstallState::Installed
        ));
        assert_eq!(
            fs::read_to_string(&files.marker_path).expect("marker"),
            entry.sha256
        );

        // Wrong bytes: refused, and the user's file is left in place.
        fs::write(&files.final_path, vec![0u8; 4096]).expect("place wrong file");
        fs::remove_file(&files.marker_path).expect("drop marker");
        assert!(verify_placed_file(dir.path(), &entry).is_err());
        assert!(files.final_path.exists(), "never delete a user-placed file");
        // A wrong-size file is refused by its size (it can never verify),
        // shows as Failed with that reason, and is still never touched.
        fs::write(&files.final_path, vec![0u8; 10]).expect("place short file");
        let message = verify_placed_file(dir.path(), &entry).expect_err("short file refuses");
        assert!(message.contains("has 10 bytes"), "got: {message}");
        assert!(matches!(
            scan_install(dir.path(), &entry),
            InstallState::Failed(reason) if reason.contains("has 10 bytes")
        ));
        assert!(files.final_path.exists());
    }

    #[test]
    fn delete_removes_all_three_files() {
        let entry = entry_for("http://127.0.0.1:9/x.gguf", &[1u8; 100]);
        let dir = tempfile::tempdir().expect("tempdir");
        let files = model_files(dir.path(), &entry);
        fs::write(&files.final_path, b"gone").expect("final");
        fs::write(&files.part_path, b"gone").expect("part");
        fs::write(&files.marker_path, "gone").expect("marker");
        delete_model_files(dir.path(), &entry).expect("delete");
        assert!(!files.final_path.exists());
        assert!(!files.part_path.exists());
        assert!(!files.marker_path.exists());
        // Deleting an absent model is fine (idempotent).
        delete_model_files(dir.path(), &entry).expect("delete again");
    }

    #[test]
    fn file_names_stay_gguf_not_mangled() {
        let entry = CatalogEntry::new(
            "m",
            "m",
            "parakeet",
            "http://host/files/model.gguf",
            1,
            &"a".repeat(64),
            false,
            "",
        );
        let files = model_files(Path::new("/models"), &entry);
        assert_eq!(files.final_path, Path::new("/models/model.gguf"));
        assert_eq!(files.part_path, Path::new("/models/model.gguf.part"));
        assert_eq!(files.marker_path, Path::new("/models/model.gguf.verified"));
        assert_eq!(files.lock_path, Path::new("/models/model.gguf.lock"));
    }
}
