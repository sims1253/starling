//! The bundled `engines/` layout: discovery, manifest parsing, and
//! checksum verification (#362 step 1).
//!
//! Wave A's packaging places this directory next to the app executable:
//!
//! ```text
//! <exe dir>/engines/
//!   starling-serve-cpu[.exe]
//!   starling-serve-vulkan[.exe]
//!   SHA256SUMS.txt   "<sha256 hex>  <file name>" per line
//!   engines.json     {"version": "...", "abi": 8, "engines": [...]}
//!   RUNTIME.md
//! ```
//!
//! An engine with no checksum entry, or whose checksum does not match, is
//! never executed: a corrupted or tampered binary must fail loudly with a
//! reinstall hint, not crash at spawn time in a way the user cannot act
//! on.

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::engine::{Backend, EngineFailure};

/// File name of the manifest inside an engines directory.
pub const ENGINES_JSON: &str = "engines.json";
/// File name of the checksum list inside an engines directory.
pub const SHA256SUMS_TXT: &str = "SHA256SUMS.txt";
/// The directory name looked up next to the executable.
pub const ENGINE_DIR_NAME: &str = "engines";

/// A parsed engines directory. The manifest's preference order (vulkan
/// before cpu) drives [`crate::engine::probe::select_backend`].
#[derive(Clone, Debug)]
pub struct Bundle {
    /// The directory the manifest was loaded from.
    pub dir: PathBuf,
    /// The server version the manifest declares (must match `--version`).
    pub version: String,
    /// The ABI the manifest declares (must match
    /// [`crate::engine::EXPECTED_ENGINE_ABI`]).
    pub abi: u32,
    /// `(backend, file name)` in manifest order.
    pub engines: Vec<(Backend, String)>,
    /// `file name -> expected sha256 hex` from `SHA256SUMS.txt`.
    pub sums: HashMap<String, String>,
}

#[derive(Debug, Deserialize)]
struct EnginesJson {
    version: String,
    abi: u32,
    #[serde(default)]
    engines: Vec<EngineEntry>,
}

#[derive(Debug, Deserialize)]
struct EngineEntry {
    backend: String,
    file: String,
}

/// Discovery order for the engines directory (#362):
/// `STARLING_ENGINE_DIR`, then `<current_exe dir>/engines`, and on macOS
/// additionally `<exe dir>/../Resources/engines` (a `.app` bundle puts the
/// executable in `Contents/MacOS`). The first directory that actually
/// contains a manifest wins.
pub fn discover_engine_dir() -> Option<PathBuf> {
    let env_dir = std::env::var_os("STARLING_ENGINE_DIR").map(PathBuf::from);
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(Path::to_path_buf));
    discover_from(env_dir.as_deref(), exe_dir.as_deref())
}

/// The pure core of [`discover_engine_dir`], injectable for tests.
/// Returns the first candidate that holds an `engines.json`.
pub fn discover_from(env_dir: Option<&Path>, exe_dir: Option<&Path>) -> Option<PathBuf> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Some(dir) = env_dir {
        candidates.push(dir.to_path_buf());
    }
    if let Some(exe_dir) = exe_dir {
        candidates.push(exe_dir.join(ENGINE_DIR_NAME));
        if cfg!(target_os = "macos") {
            candidates.push(exe_dir.join("../Resources").join(ENGINE_DIR_NAME));
        }
    }
    candidates
        .into_iter()
        .find(|dir| dir.join(ENGINES_JSON).is_file())
}

/// Parses `engines.json` and `SHA256SUMS.txt` from `dir`. The error is a
/// human sentence naming what is broken; callers surface it instead of
/// guessing at a half-parsed bundle.
pub fn load_bundle(dir: &Path) -> Result<Bundle, String> {
    let manifest_path = dir.join(ENGINES_JSON);
    let manifest_text = fs::read_to_string(&manifest_path)
        .map_err(|error| format!("cannot read {}: {error}", manifest_path.display()))?;
    let manifest: EnginesJson = serde_json::from_str(&manifest_text)
        .map_err(|error| format!("invalid {ENGINES_JSON}: {error}"))?;

    let mut engines = Vec::new();
    for entry in &manifest.engines {
        let backend = Backend::parse(&entry.backend).ok_or_else(|| {
            format!(
                "unknown backend {:?} in {ENGINES_JSON}; this app knows only \"vulkan\" and \"cpu\"",
                entry.backend
            )
        })?;
        engines.push((backend, entry.file.clone()));
    }
    if engines.is_empty() {
        return Err(format!("{ENGINES_JSON} lists no engines"));
    }

    let sums_path = dir.join(SHA256SUMS_TXT);
    let sums_text = fs::read_to_string(&sums_path)
        .map_err(|error| format!("cannot read {}: {error}", sums_path.display()))?;
    let sums = parse_sums(&sums_text);
    if sums.is_empty() {
        return Err(format!("{SHA256SUMS_TXT} contains no checksums"));
    }

    Ok(Bundle {
        dir: dir.to_path_buf(),
        version: manifest.version,
        abi: manifest.abi,
        engines,
        sums,
    })
}

/// Parses `sha256sum`-style lines: `<sha256 hex>  <file name>` (any
/// run of whitespace accepted, comments and blank lines skipped).
pub fn parse_sums(text: &str) -> HashMap<String, String> {
    let mut sums = HashMap::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((hex, name)) = line.split_once(char::is_whitespace) else {
            continue;
        };
        let (hex, name) = (hex.trim(), name.trim());
        if hex.len() == 64 && hex.chars().all(|c| c.is_ascii_hexdigit()) && !name.is_empty() {
            // First entry wins, matching sha256sum's "first copy" rule.
            sums.entry(name.to_string())
                .or_insert_with(|| hex.to_string());
        }
    }
    sums
}

/// Streams `path` through sha256 and returns the lowercase hex digest.
/// Reading a 55 MiB engine must not allocate it in memory.
pub fn sha256_file(path: &Path) -> std::io::Result<String> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 128 * 1024];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(to_hex(&hasher.finalize()))
}

/// Lowercase hex of a digest (sha2 0.11's output type has no `LowerHex`,
/// so the encoding is explicit).
pub fn to_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0xf) as usize] as char);
    }
    out
}

/// Verifies an engine file against its expected hex digest. A missing or
/// unreadable file is a load failure carrying the io error text — it
/// says what is wrong with the disk, not that the bundle is corrupt;
/// only a genuine digest mismatch is a [`EngineFailure::ChecksumMismatch`]
/// — an unverifiable engine is never run (#362 acceptance: "Checksums
/// cover the bundled engines").
pub fn verify_engine(path: &Path, expected_hex: &str) -> Result<(), EngineFailure> {
    let file = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string());
    match sha256_file(path) {
        Ok(actual) if actual.eq_ignore_ascii_case(expected_hex) => Ok(()),
        Ok(_) => Err(EngineFailure::ChecksumMismatch { file }),
        Err(error) => Err(EngineFailure::LoadFailed(format!(
            "could not read {file} to verify its checksum: {error}"
        ))),
    }
}

/// Looks up the checksum entry for a bundled engine file. Missing entry
/// is the same failure as a mismatch: no entry, no run.
pub fn expected_sum<'a>(bundle: &'a Bundle, file: &str) -> Result<&'a str, EngineFailure> {
    bundle
        .sums
        .get(file)
        .map(String::as_str)
        .ok_or_else(|| EngineFailure::ChecksumMismatch {
            file: file.to_string(),
        })
}

/// Writes a `SHA256SUMS.txt`-style line (kept for the parsing tests).
#[cfg(test)]
pub(crate) fn sum_line(hex: &str, file: &str) -> String {
    format!("{hex}  {file}\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_bundle(dir: &Path, engines_json: &str, sums: &str) {
        fs::create_dir_all(dir).expect("create bundle dir");
        fs::write(dir.join(ENGINES_JSON), engines_json).expect("write manifest");
        fs::write(dir.join(SHA256SUMS_TXT), sums).expect("write sums");
    }

    #[test]
    fn parses_manifest_and_sums() {
        let dir = tempfile::tempdir().expect("tempdir");
        write_bundle(
            dir.path(),
            r#"{"version":"0.1.0","abi":8,"engines":[
                {"backend":"vulkan","file":"starling-serve-vulkan"},
                {"backend":"cpu","file":"starling-serve-cpu"}]}"#,
            &format!(
                "{}{}",
                sum_line(&"a".repeat(64), "starling-serve-vulkan"),
                sum_line(&"b".repeat(64), "starling-serve-cpu")
            ),
        );
        let bundle = load_bundle(dir.path()).expect("bundle parses");
        assert_eq!(bundle.version, "0.1.0");
        assert_eq!(bundle.abi, 8);
        assert_eq!(
            bundle.engines,
            vec![
                (Backend::Vulkan, "starling-serve-vulkan".to_string()),
                (Backend::Cpu, "starling-serve-cpu".to_string()),
            ]
        );
        assert_eq!(bundle.sums["starling-serve-cpu"], "b".repeat(64));
    }

    #[test]
    fn rejects_unknown_backend_and_empty_lists() {
        let dir = tempfile::tempdir().expect("tempdir");
        write_bundle(
            dir.path(),
            r#"{"version":"0.1.0","abi":8,"engines":[{"backend":"cuda","file":"x"}]}"#,
            &sum_line(&"a".repeat(64), "x"),
        );
        assert!(load_bundle(dir.path()).is_err());

        write_bundle(
            dir.path(),
            r#"{"version":"0.1.0","abi":8,"engines":[]}"#,
            &sum_line(&"a".repeat(64), "x"),
        );
        assert!(load_bundle(dir.path()).is_err());
    }

    #[test]
    fn missing_sums_file_is_an_error() {
        let dir = tempfile::tempdir().expect("tempdir");
        fs::create_dir_all(dir.path()).expect("create");
        fs::write(
            dir.path().join(ENGINES_JSON),
            r#"{"version":"0.1.0","abi":8,"engines":[{"backend":"cpu","file":"x"}]}"#,
        )
        .expect("write manifest");
        assert!(load_bundle(dir.path()).is_err());
    }

    #[test]
    fn sums_parser_accepts_sha256sum_format() {
        let sums = parse_sums(&format!(
            "# comment\n\n{}{}garbage-line\n{}",
            sum_line(&"A".repeat(64), "one"),
            sum_line(&"c".repeat(64), "two"),
            sum_line("short", "three")
        ));
        assert_eq!(sums.len(), 2);
        assert_eq!(sums["one"], "A".repeat(64));
        assert_eq!(sums["two"], "c".repeat(64));
    }

    #[test]
    fn verify_engine_detects_mismatch_and_missing_entry() {
        let dir = tempfile::tempdir().expect("tempdir");
        let file = dir.path().join("engine");
        fs::write(&file, b"engine bytes").expect("write engine");
        let good = sha256_file(&file).expect("hash");
        assert!(verify_engine(&file, &good).is_ok());
        let bad = "0".repeat(64);
        match verify_engine(&file, &bad) {
            Err(EngineFailure::ChecksumMismatch { file: name }) => {
                assert_eq!(name, "engine");
            }
            other => panic!("expected ChecksumMismatch, got {other:?}"),
        }
        // An unreadable/missing file is a load failure naming the io
        // error, not a checksum verdict about the bundle.
        match verify_engine(&dir.path().join("absent"), &good) {
            Err(EngineFailure::LoadFailed(message)) => {
                assert!(message.contains("absent"), "got: {message}");
                assert!(message.contains("could not read"), "got: {message}");
            }
            other => panic!("expected LoadFailed, got {other:?}"),
        }
    }

    #[test]
    fn expected_sum_missing_entry_is_checksum_mismatch() {
        let dir = tempfile::tempdir().expect("tempdir");
        write_bundle(
            dir.path(),
            r#"{"version":"0.1.0","abi":8,"engines":[{"backend":"cpu","file":"known"}]}"#,
            &sum_line(&"a".repeat(64), "known"),
        );
        let bundle = load_bundle(dir.path()).expect("bundle");
        assert!(expected_sum(&bundle, "known").is_ok());
        assert!(matches!(
            expected_sum(&bundle, "unknown"),
            Err(EngineFailure::ChecksumMismatch { file }) if file == "unknown"
        ));
    }

    #[test]
    fn discovery_prefers_env_then_exe_dir() {
        let root = tempfile::tempdir().expect("tempdir");
        let staged = root.path().join("staged");
        let bundled = root.path().join("bin");
        fs::create_dir_all(&staged).expect("create staged");
        fs::create_dir_all(bundled.join(ENGINE_DIR_NAME)).expect("create bundled");
        fs::write(
            staged.join(ENGINES_JSON),
            r#"{"version":"0.1.0","abi":8,"engines":[]}"#,
        )
        .expect("staged manifest");
        fs::write(
            bundled.join(ENGINE_DIR_NAME).join(ENGINES_JSON),
            r#"{"version":"0.1.0","abi":8,"engines":[]}"#,
        )
        .expect("bundled manifest");

        // Env wins over the exe dir.
        assert_eq!(
            discover_from(Some(&staged), Some(&bundled)).as_deref(),
            Some(staged.as_path())
        );
        // Without env, the exe dir's engines/ layout is used.
        assert_eq!(
            discover_from(None, Some(&bundled)).as_deref(),
            Some(bundled.join(ENGINE_DIR_NAME).as_path())
        );
        // A candidate without a manifest is skipped, later ones still win.
        assert_eq!(
            discover_from(Some(root.path()), Some(&bundled)).as_deref(),
            Some(bundled.join(ENGINE_DIR_NAME).as_path())
        );
        // Nothing found.
        assert_eq!(discover_from(None, None), None);
    }
}
