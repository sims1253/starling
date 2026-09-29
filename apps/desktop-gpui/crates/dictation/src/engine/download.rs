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
//! touches the active model or any other file.

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

/// The three file names a model resolves to inside the models dir.
#[derive(Clone, Debug)]
pub struct ModelFiles {
    /// The final, verified model file.
    pub final_path: PathBuf,
    /// The in-flight download target.
    pub part_path: PathBuf,
    /// `<final>.verified`, containing the sha hex on success.
    pub marker_path: PathBuf,
}

/// `<models_dir>/<file_name>` plus its `.part`/`.verified` siblings.
pub fn model_files(models_dir: &Path, entry: &CatalogEntry) -> ModelFiles {
    let final_path = models_dir.join(&entry.file_name);
    ModelFiles {
        part_path: final_path.with_extension(PART_SUFFIX),
        // with_extension would mangle "a.gguf" -> "a.verified"; build it
        // from the file name instead.
        marker_path: models_dir.join(format!("{}.{}", entry.file_name, VERIFIED_SUFFIX)),
        final_path,
    }
}

/// Disk truth for one catalog entry. Installed requires file + size +
/// marker match; a file without a matching marker is
/// [`InstallState::NeedsVerification`] (activation hashes it first).
pub fn scan_install(models_dir: &Path, entry: &CatalogEntry) -> InstallState {
    let files = model_files(models_dir, entry);
    let Some(meta) = fs::metadata(&files.final_path).ok() else {
        return InstallState::NotInstalled;
    };
    if meta.len() != entry.size_bytes {
        return InstallState::NeedsVerification;
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
    let meta = fs::metadata(&files.final_path)
        .map_err(|error| format!("cannot stat {}: {error}", files.final_path.display()))?;
    if meta.len() != entry.size_bytes {
        return Err(format!(
            "the file {} has {} bytes, but the catalog expects {}; \
             it was not changed — replace it or delete it and download again",
            files.final_path.display(),
            meta.len(),
            entry.size_bytes
        ));
    }
    write_marker(&files.marker_path, &entry.sha256)
        .map_err(|error| format!("cannot write the verification marker: {error}"))?;
    Ok(())
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
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::limited(5))
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

    let request_timeout = Duration::from_secs(60);
    let mut request = client.get(&entry.url).timeout(request_timeout);
    if offset > 0 {
        request = request.header("Range", format!("bytes={offset}-"));
    }
    let response = request
        .send()
        .await
        .map_err(|error| DownloadError::Http(format!("cannot reach {}: {error}", entry.url)))?;
    let status = response.status();
    if !status.is_success() {
        return Err(DownloadError::Http(format!(
            "the server answered {} for {}",
            status.as_u16(),
            entry.url
        )));
    }
    let resumed = status.as_u16() == 206;

    let mut file = if resumed && offset > 0 {
        OpenOptions::new()
            .append(true)
            .open(&files.part_path)
            .map_err(io_error)?
    } else {
        // 200 (or a resumed request answered from zero): restart clean.
        offset = 0;
        hasher = Sha256::new();
        File::create(&files.part_path).map_err(io_error)?
    };

    let mut done = offset;
    progress(done, entry.size_bytes);
    let mut response = response;
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|error| DownloadError::Http(error.to_string()))?
    {
        if cancel.load(Ordering::Relaxed) {
            // Keep .part for a later resume; drop the connection.
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
                    let value = line.strip_prefix("Range: ")?;
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
                    Some(start) if start < bytes.len() as u64 => {
                        ("206 Partial Content", start as usize..bytes.len())
                    }
                    Some(_) => ("416 Range Not Satisfiable", 0..0),
                    None => ("200 OK", 0..bytes.len()),
                };
                let body = &bytes[body_range];
                let head = format!(
                    "HTTP/1.1 {status}\r\ncontent-length: {}\r\naccept-ranges: bytes\r\nconnection: close\r\n\r\n",
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
        // A wrong-size file necessarily has a wrong digest too, so it is
        // refused the same way — and still never touched.
        fs::write(&files.final_path, vec![0u8; 10]).expect("place short file");
        let message = verify_placed_file(dir.path(), &entry).expect_err("short file refuses");
        assert!(message.contains("does not match"), "got: {message}");
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
        assert_eq!(files.part_path, Path::new("/models/model.part"));
        assert_eq!(files.marker_path, Path::new("/models/model.gguf.verified"));
    }
}
