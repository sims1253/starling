//! Shared helpers for the engine integration tests: a staged engine dir
//! built from the contract fixture, a local Range-capable model server,
//! temp catalogs, and poll helpers. The fixture binary comes from
//! `STARLING_CONTRACT_BIN` or the repo build dir; without it the tests
//! print a skip line and return (it is built on this machine).

#![allow(dead_code)]

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use starling_dictation::engine::bundle::sha256_file;
use starling_dictation::engine::{
    CatalogEntry, EngineConfig, EngineManager, EngineSnapshot, InstallState,
};

/// The contract-fixture server binary, or `None` (skip the test).
pub fn fixture() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("STARLING_CONTRACT_BIN") {
        let path = PathBuf::from(path);
        if path.is_file() {
            return Some(path);
        }
    }
    let default = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../../build/native-cpu/starling-serve-contract-fixture");
    if default.is_file() {
        Some(default)
    } else {
        println!(
            "skipping: contract fixture {} not found (build it with \
             cmake --build build/native-cpu --target starling-serve-contract-fixture)",
            default.display()
        );
        None
    }
}

/// Stages `<root>/engines/` with the fixture copied as
/// `starling-serve-cpu`, plus `engines.json` (version 0.1.0, abi 8) and
/// its `SHA256SUMS.txt`.
pub fn stage_engine_dir(root: &Path, fixture: &Path) -> PathBuf {
    let dir = root.join("engines");
    std::fs::create_dir_all(&dir).expect("create engines dir");
    let engine = dir.join("starling-serve-cpu");
    std::fs::copy(fixture, &engine).expect("copy fixture");
    make_executable(&engine);
    let sha = sha256_file(&engine).expect("hash staged engine");
    std::fs::write(
        dir.join("SHA256SUMS.txt"),
        format!("{sha}  starling-serve-cpu\n"),
    )
    .expect("write sums");
    std::fs::write(
        dir.join("engines.json"),
        r#"{"version":"0.1.0","abi":8,"engines":[{"backend":"cpu","file":"starling-serve-cpu"}]}"#,
    )
    .expect("write manifest");
    dir
}

#[cfg(unix)]
fn make_executable(path: &Path) {
    use std::os::unix::fs::PermissionsExt;
    let mut permissions = std::fs::metadata(path).expect("stat").permissions();
    permissions.set_mode(0o755);
    std::fs::set_permissions(path, permissions).expect("chmod");
}

#[cfg(not(unix))]
fn make_executable(_path: &Path) {}

/// A catalog entry served by [`spawn_model_server`]. The slug is always
/// `parakeet` (the fixture serves it); the id and file differ per model.
pub fn entry(id: &str, file: &str, addr: SocketAddr, bytes: &[u8]) -> CatalogEntry {
    let sha = sha256_bytes(bytes);
    CatalogEntry::new(
        id,
        id,
        "parakeet",
        &format!("http://{addr}/{file}"),
        bytes.len() as u64,
        &sha,
        false,
        "integration test model",
    )
}

/// A catalog entry whose digest is NOT the served bytes (for the
/// checksum-mismatch test); sizes match so the size gate passes.
pub fn entry_with_wrong_digest(
    id: &str,
    file: &str,
    addr: SocketAddr,
    bytes: &[u8],
) -> CatalogEntry {
    let mut entry = entry(id, file, addr, bytes);
    let mut flipped = sha256_bytes(bytes);
    flipped.replace_range(0..1, if flipped.starts_with('0') { "1" } else { "0" });
    entry.sha256 = flipped;
    entry
}

pub fn sha256_bytes(bytes: &[u8]) -> String {
    use sha2::Digest;
    let mut hasher = sha2::Sha256::new();
    hasher.update(bytes);
    starling_dictation::engine::bundle::to_hex(&hasher.finalize())
}

/// Writes a model file plus its verification marker: an Installed model
/// without any download round-trip.
pub fn install(models_dir: &Path, entry: &CatalogEntry, bytes: &[u8]) {
    std::fs::create_dir_all(models_dir).expect("create models dir");
    std::fs::write(models_dir.join(&entry.file_name), bytes).expect("write model");
    std::fs::write(
        models_dir.join(format!("{}.verified", entry.file_name)),
        &entry.sha256,
    )
    .expect("write marker");
}

/// Distinct model payloads (the fixture accepts any existing file).
pub fn model_bytes(seed: u8, size: usize) -> Vec<u8> {
    (0..size)
        .map(|i| (seed.wrapping_add(i as u8)).wrapping_mul(31))
        .collect()
}

/// A one-thread HTTP/1.1 server for `<name>` files, honoring `Range`
/// (resume); one response per connection, then close.
pub fn spawn_model_server(files: Vec<(String, Arc<Vec<u8>>)>) -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind model server");
    let addr = listener.local_addr().expect("model server addr");
    let files: HashMap<String, Arc<Vec<u8>>> = files.into_iter().collect();
    std::thread::Builder::new()
        .name("test-model-server".into())
        .spawn(move || {
            for stream in listener.incoming().flatten() {
                let mut stream = stream;
                let Some(request) = read_request(&mut stream) else {
                    continue;
                };
                let path = request
                    .path
                    .trim_start_matches('/')
                    .split('?')
                    .next()
                    .unwrap_or_default()
                    .to_string();
                let Some(bytes) = files.get(&path) else {
                    write_response(&mut stream, "404 Not Found", &[], None);
                    continue;
                };
                let range_start = request.range_start.filter(|start| *start > 0);
                match range_start {
                    Some(start) if (start as usize) < bytes.len() => {
                        let body = &bytes[start as usize..];
                        write_response(
                            &mut stream,
                            "206 Partial Content",
                            body,
                            Some(&format!(
                                "bytes {start}-{}/{}",
                                bytes.len() - 1,
                                bytes.len()
                            )),
                        );
                    }
                    Some(_) => write_response(&mut stream, "416 Range Not Satisfiable", &[], None),
                    None => {
                        write_response(&mut stream, "200 OK", bytes, None);
                    }
                }
            }
        })
        .expect("spawn model server");
    addr
}

struct HttpRequest {
    path: String,
    range_start: Option<u64>,
}

fn read_request(stream: &mut TcpStream) -> Option<HttpRequest> {
    let mut head = Vec::new();
    let mut byte = [0u8; 1];
    while !head.ends_with(b"\r\n\r\n") {
        match stream.read(&mut byte) {
            Ok(0) | Err(_) => return None,
            Ok(_) => head.push(byte[0]),
        }
    }
    let text = String::from_utf8_lossy(&head).to_string();
    let path = text
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .unwrap_or_default()
        .to_string();
    let range_start = text.lines().find_map(|line| {
        let value = line.strip_prefix("Range: ")?;
        value
            .trim()
            .trim_start_matches("bytes=")
            .split('-')
            .next()?
            .parse()
            .ok()
    });
    Some(HttpRequest { path, range_start })
}

fn write_response(stream: &mut TcpStream, status: &str, body: &[u8], content_range: Option<&str>) {
    let mut head = format!(
        "HTTP/1.1 {status}\r\ncontent-length: {}\r\naccept-ranges: bytes\r\n",
        body.len()
    );
    if let Some(range) = content_range {
        head.push_str(&format!("content-range: {range}\r\n"));
    }
    head.push_str("connection: close\r\n\r\n");
    let _ = stream.write_all(head.as_bytes());
    let _ = stream.write_all(body);
    let _ = stream.flush();
}

/// Polls the manager until `predicate` holds; returns the matching
/// snapshot or `None` on timeout (the caller's assert names the wait).
pub fn wait_until(
    manager: &EngineManager,
    timeout: Duration,
    predicate: impl Fn(&EngineSnapshot) -> bool,
) -> Option<EngineSnapshot> {
    let deadline = Instant::now() + timeout;
    loop {
        let snapshot = manager.snapshot();
        if predicate(&snapshot) {
            return Some(snapshot);
        }
        if Instant::now() >= deadline {
            return None;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

/// A minimal blocking GET against a loopback endpoint
/// (`http://127.0.0.1:<port><path>`), returning `(status, body)`.
pub fn http_get(port: u16, path: &str) -> Option<(u16, String)> {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).ok()?;
    stream.set_read_timeout(Some(Duration::from_secs(5))).ok()?;
    let request = format!("GET {path} HTTP/1.1\r\nhost: 127.0.0.1\r\nconnection: close\r\n\r\n");
    stream.write_all(request.as_bytes()).ok()?;
    let mut raw = Vec::new();
    stream.read_to_end(&mut raw).ok()?;
    let text = String::from_utf8_lossy(&raw);
    let status = text
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|code| code.parse().ok())?;
    let body = text
        .split_once("\r\n\r\n")
        .map(|(_, body)| body.to_string())
        .unwrap_or_default();
    Some((status, body))
}

/// The port of an `http://127.0.0.1:<port>` endpoint string.
pub fn endpoint_port(endpoint: &str) -> u16 {
    endpoint
        .rsplit(':')
        .next()
        .and_then(|port| port.parse().ok())
        .expect("endpoint port")
}

/// How many live processes have `needle` in their command line (their
/// engine path lives in this test's temp dir, so the count is isolated
/// from parallel tests). `None` off Linux.
pub fn count_engine_processes(needle: &str) -> Option<usize> {
    #[cfg(target_os = "linux")]
    {
        let mut count = 0;
        for entry in std::fs::read_dir("/proc").ok()?.flatten() {
            let name = entry.file_name();
            let Some(name) = name.to_str() else { continue };
            if !name.chars().all(|c| c.is_ascii_digit()) {
                continue;
            }
            if let Ok(mut cmdline) = std::fs::File::open(entry.path().join("cmdline")) {
                let mut raw = Vec::new();
                if cmdline.read_to_end(&mut raw).is_ok() {
                    let text = String::from_utf8_lossy(&raw).replace('\0', " ");
                    if text.contains(needle) {
                        count += 1;
                    }
                }
            }
        }
        Some(count)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = needle;
        None
    }
}

/// Whether `pid` is a live process (Linux; `true` elsewhere).
pub fn pid_alive(pid: u32) -> bool {
    #[cfg(target_os = "linux")]
    {
        Path::new(&format!("/proc/{pid}")).exists()
    }
    #[cfg(not(target_os = "linux"))]
    {
        true
    }
}

/// A manager config pointing at the test's temp dirs.
pub fn config(
    engine_dir: &Path,
    models_dir: &Path,
    state_dir: &Path,
    catalog: Vec<CatalogEntry>,
) -> EngineConfig {
    EngineConfig {
        engine_dir: Some(engine_dir.to_path_buf()),
        models_dir: models_dir.to_path_buf(),
        state_dir: state_dir.to_path_buf(),
        catalog,
        backend_override: None,
        icd_dirs: None,
        available_memory_override: None,
        backoff_schedule: None,
    }
}

/// The install state of a model id in the snapshot.
pub fn install_of(snapshot: &EngineSnapshot, id: &str) -> InstallState {
    snapshot
        .models
        .iter()
        .find(|model| model.id == id)
        .map(|model| model.install.clone())
        .unwrap_or(InstallState::NotInstalled)
}
