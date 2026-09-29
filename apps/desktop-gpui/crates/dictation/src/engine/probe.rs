//! Backend selection (#362 step 2): probe each bundled engine with
//! `--version`, classify failures into actionable reasons, and prefer
//! Vulkan with an honest CPU fallback.
//!
//! Probing happens before anything is spawned to serve: a binary missing
//! `libvulkan.so.1` or `vulkan-1.dll`, a wrong-architecture binary, or an
//! ABI mismatch must be rejected with a sentence the user can act on —
//! never discovered as an opaque crash after the app starts.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use crate::engine::bundle::{expected_sum, verify_engine, Bundle};
use crate::engine::{Backend, EngineFailure, ExitFamily, EXPECTED_ENGINE_ABI};

/// How long `--version` may take before the binary is killed. A healthy
/// engine answers in milliseconds; ten seconds covers cold-cache starts
/// on the slowest supported disk.
pub(crate) const PROBE_TIMEOUT: Duration = Duration::from_secs(10);

/// What `--version` told us about an engine binary.
#[derive(Clone, Debug, PartialEq)]
pub struct EngineVersion {
    /// The server version line (`starling-serve <version>`).
    pub version: String,
    /// The `abi-version:` line.
    pub abi: u32,
    /// The compile-time backend family (`backend:` line: `vulkan`,
    /// `cpu`, `contract-fixture`, ...).
    pub backend_family: String,
}

/// Why a candidate engine was rejected. These become the per-backend
/// reason strings in [`EngineFailure::NoUsableEngine`].
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum ProbeFailure {
    #[error("{reason}")]
    NotExecutable { reason: String },
    #[error(
        "a required library is missing ({library}); \
         install your GPU's Vulkan driver or the runtime library it names"
    )]
    MissingLibrary { library: String },
    #[error("it exited with code {code} during --version: {stderr_tail}")]
    Crashed { code: i32, stderr_tail: String },
    #[error("its --version output could not be parsed")]
    BadOutput,
    #[error("it speaks ABI {found}, but this app expects ABI {expected}")]
    AbiMismatch { found: u32, expected: u32 },
    #[error("it reports version {found}, but the bundle manifest says {expected}")]
    VersionMismatch { found: String, expected: String },
}

/// The outcome of backend selection.
#[derive(Clone, Debug)]
pub struct BackendSelection {
    pub chosen: Backend,
    pub path: PathBuf,
    pub version: String,
    /// Set when a preferred backend was rejected and a fallback was used,
    /// e.g. "Vulkan unavailable (...); using the CPU engine."
    pub fallback_notice: Option<String>,
    /// Every rejected candidate with its reason (honesty surface for the
    /// app's engine status).
    pub rejected: Vec<(Backend, String)>,
}

/// Runs `<path> --version` (10 s budget, killed on timeout) and checks
/// the result against the app's ABI expectation and, when given, the
/// manifest's version. A mismatched engine is never run (#362).
pub fn probe_engine(
    path: &Path,
    expected_version: Option<&str>,
) -> Result<EngineVersion, ProbeFailure> {
    let output = run_version(path)?;
    let version = parse_version_output(&output.stdout)?;
    if version.abi != EXPECTED_ENGINE_ABI {
        return Err(ProbeFailure::AbiMismatch {
            found: version.abi,
            expected: EXPECTED_ENGINE_ABI,
        });
    }
    if let Some(expected) = expected_version {
        if version.version != expected {
            return Err(ProbeFailure::VersionMismatch {
                found: version.version,
                expected: expected.to_string(),
            });
        }
    }
    Ok(version)
}

/// Spawns `--version`, enforces the timeout, and classifies the exit.
fn run_version(path: &Path) -> Result<VersionOutput, ProbeFailure> {
    let mut command = Command::new(path);
    command.arg("--version");
    command.stdin(Stdio::null());
    command.stdout(Stdio::piped());
    command.stderr(Stdio::piped());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        // No console flash for a background probe.
        command.creation_flags(0x0800_0000);
    }
    let mut child = command
        .spawn()
        .map_err(|error| ProbeFailure::NotExecutable {
            reason: spawn_error_reason(&error),
        })?;
    let deadline = Instant::now() + PROBE_TIMEOUT;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Some(status),
            Ok(None) => {
                if Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    break None;
                }
                std::thread::sleep(Duration::from_millis(20));
            }
            Err(error) => {
                return Err(ProbeFailure::NotExecutable {
                    reason: format!("cannot wait for --version: {error}"),
                })
            }
        }
    };
    let mut stdout = String::new();
    if let Some(mut pipe) = child.stdout.take() {
        use std::io::Read;
        let _ = pipe.read_to_string(&mut stdout);
    }
    let mut stderr = String::new();
    if let Some(mut pipe) = child.stderr.take() {
        use std::io::Read;
        let _ = pipe.read_to_string(&mut stderr);
    }
    let Some(status) = status else {
        return Err(ProbeFailure::Crashed {
            code: -1,
            stderr_tail: "did not answer --version within 10 s (killed)".to_string(),
        });
    };
    if !status.success() {
        let code = status.code().unwrap_or(-1);
        return Err(classify_exit(code, &stderr, ExitFamily::host()));
    }
    Ok(VersionOutput { stdout, stderr })
}

struct VersionOutput {
    stdout: String,
    #[allow(dead_code)]
    stderr: String,
}

/// A human reason for a spawn io error. `NotFound` and
/// `PermissionDenied` cover "not staged / not executable"; anything else
/// is reported verbatim.
fn spawn_error_reason(error: &std::io::Error) -> String {
    match error.kind() {
        std::io::ErrorKind::NotFound => {
            "the engine file does not exist at the expected path".to_string()
        }
        std::io::ErrorKind::PermissionDenied => "the engine file is not executable".to_string(),
        _ => error.to_string(),
    }
}

/// Classifies a nonzero `--version` exit into a probe failure. Pure so
/// the Windows loader statuses are unit-testable on any host.
pub(crate) fn classify_exit(code: i32, stderr: &str, family: ExitFamily) -> ProbeFailure {
    match family {
        ExitFamily::Windows => {
            if code == crate::engine::STATUS_DLL_NOT_FOUND {
                return ProbeFailure::MissingLibrary {
                    library: "a required DLL (for Vulkan: vulkan-1.dll)".to_string(),
                };
            }
            if code == crate::engine::STATUS_INVALID_IMAGE_FORMAT {
                return ProbeFailure::Crashed {
                    code,
                    stderr_tail: "the binary does not match this system's architecture (bad image)"
                        .to_string(),
                };
            }
        }
        ExitFamily::Unix => {
            // The dynamic loader's exit 127 with a named library is the
            // classic "installed the Vulkan build without the driver
            // package" failure.
            if code == 127 {
                if let Some(library) = missing_library_from_stderr(stderr) {
                    return ProbeFailure::MissingLibrary { library };
                }
            }
        }
    }
    ProbeFailure::Crashed {
        code,
        stderr_tail: stderr_tail(stderr),
    }
}

/// Extracts the library name from the glibc loader line
/// `error while loading shared libraries: <lib>: ...`.
pub(crate) fn missing_library_from_stderr(stderr: &str) -> Option<String> {
    let prefix = "error while loading shared libraries: ";
    let start = stderr.find(prefix)? + prefix.len();
    let rest = &stderr[start..];
    let end = rest.find(':')?;
    let library = &rest[..end];
    if library.is_empty() {
        None
    } else {
        Some(library.to_string())
    }
}

/// The last few non-empty stderr lines, joined — crash context for
/// reasons and crash-loop reports.
pub(crate) fn stderr_tail(stderr: &str) -> String {
    const MAX_LINES: usize = 8;
    let lines: Vec<&str> = stderr
        .lines()
        .filter(|line| !line.trim().is_empty())
        .collect();
    let tail: Vec<&str> = lines.iter().rev().take(MAX_LINES).rev().copied().collect();
    let joined = tail.join(" | ");
    if joined.trim().is_empty() {
        "(no output)".to_string()
    } else {
        joined
    }
}

/// Parses the `--version` output block:
///
/// ```text
/// starling-serve 0.1.0
/// abi-version: 8
/// backend: cpu
/// supported-models: parakeet ...
/// ```
pub(crate) fn parse_version_output(stdout: &str) -> Result<EngineVersion, ProbeFailure> {
    let mut version: Option<String> = None;
    let mut abi: Option<u32> = None;
    let mut backend_family: Option<String> = None;
    for line in stdout.lines() {
        if let Some(rest) = line.strip_prefix("starling-serve ") {
            version = Some(rest.trim().to_string());
        } else if let Some(rest) = line.strip_prefix("abi-version:") {
            abi = rest.trim().parse::<u32>().ok();
        } else if let Some(rest) = line.strip_prefix("backend:") {
            backend_family = Some(rest.trim().to_string());
        }
    }
    match (version, abi, backend_family) {
        (Some(version), Some(abi), Some(backend_family)) if !version.is_empty() => {
            Ok(EngineVersion {
                version,
                abi,
                backend_family,
            })
        }
        _ => Err(ProbeFailure::BadOutput),
    }
}

/// The directories Vulkan's loader scans for driver ICD manifests on
/// Linux. `$XDG_DATA_DIRS` defaults to `/usr/local/share:/usr/share` per
/// the basedir spec.
pub fn default_icd_dirs() -> Vec<PathBuf> {
    let mut dirs = vec![
        PathBuf::from("/etc/vulkan/icd.d"),
        PathBuf::from("/usr/share/vulkan/icd.d"),
        PathBuf::from("/usr/local/share/vulkan/icd.d"),
    ];
    let data_dirs = std::env::var("XDG_DATA_DIRS")
        .unwrap_or_else(|_| "/usr/local/share:/usr/share".to_string());
    for entry in data_dirs.split(':') {
        if !entry.is_empty() {
            dirs.push(PathBuf::from(entry).join("vulkan/icd.d"));
        }
    }
    let data_home = std::env::var("XDG_DATA_HOME")
        .ok()
        .filter(|value| !value.is_empty())
        .or_else(|| {
            std::env::var("HOME")
                .ok()
                .filter(|home| !home.is_empty())
                .map(|home| format!("{home}/.local/share"))
        });
    if let Some(home) = data_home {
        dirs.push(PathBuf::from(home).join("vulkan/icd.d"));
    }
    dirs
}

/// Whether any `*.json` ICD manifest exists under one of `dirs`.
pub fn icd_present_in(dirs: &[PathBuf]) -> bool {
    dirs.iter().any(|dir| {
        std::fs::read_dir(dir)
            .map(|entries| {
                entries.flatten().any(|entry| {
                    entry
                        .path()
                        .extension()
                        .is_some_and(|extension| extension.eq_ignore_ascii_case("json"))
                })
            })
            .unwrap_or(false)
    })
}

/// The Vulkan ICD pre-check (Linux only). `Some(reason)` means "skip the
/// Vulkan engine with this reason": the loader cannot enumerate a driver
/// without an ICD manifest, and starting the binary would only produce a
/// downstream device failure. An explicit `VK_ICD_FILENAMES` /
/// `VK_DRIVER_FILES` overrides the scan (the user knows about their
/// driver). `injected` replaces the directory list for tests.
pub fn vulkan_icd_reason(injected: Option<&[PathBuf]>) -> Option<String> {
    if !cfg!(target_os = "linux") {
        // Windows drivers register through the OS, macOS has no Vulkan
        // ICD mechanism worth pre-checking; probing decides there.
        return None;
    }
    let forced = ["VK_ICD_FILENAMES", "VK_DRIVER_FILES"].iter().any(|key| {
        std::env::var(key)
            .map(|value| !value.is_empty())
            .unwrap_or(false)
    });
    if forced {
        return None;
    }
    let dirs: Vec<PathBuf> = match injected {
        Some(dirs) => dirs.to_vec(),
        None => default_icd_dirs(),
    };
    if icd_present_in(&dirs) {
        None
    } else {
        Some("No Vulkan driver (ICD) is installed".to_string())
    }
}

/// Selects the backend to run (#362 step 2). Preference order comes from
/// the bundle manifest (vulkan, then cpu); each candidate is
/// checksum-verified, ICD pre-checked (Vulkan, Linux), and probed; the
/// first good one wins. `backend_override = Some(Cpu)` skips Vulkan —
/// the app's "Use CPU engine" action.
pub fn select_backend(
    bundle: &Bundle,
    backend_override: Option<Backend>,
    icd_dirs: Option<&[PathBuf]>,
) -> Result<BackendSelection, EngineFailure> {
    let candidates: Vec<(Backend, String)> = bundle
        .engines
        .iter()
        .filter(|(backend, _)| backend_override.is_none_or(|want| *backend == want))
        .cloned()
        .collect();

    let mut rejected: Vec<(Backend, String)> = Vec::new();
    let mut missing_library: Option<(Backend, String)> = None;
    let mut chosen: Option<(Backend, PathBuf, String)> = None;
    for (backend, file) in &candidates {
        let path = bundle.dir.join(file);
        match evaluate_candidate(backend, &path, file, bundle, icd_dirs) {
            Ok(version) => {
                chosen = Some((*backend, path, version.version));
                break;
            }
            Err(CandidateFailure::Terminal(failure)) => return Err(failure),
            Err(CandidateFailure::MissingLibrary(library)) => {
                missing_library = Some((*backend, library.clone()));
                rejected.push((
                    *backend,
                    ProbeFailure::MissingLibrary {
                        library: library.clone(),
                    }
                    .to_string(),
                ));
            }
            Err(CandidateFailure::Reject(reason)) => rejected.push((*backend, reason)),
        }
    }

    let Some((backend, path, version)) = chosen else {
        // No fallback engine was left: if the machine's blocker is a
        // missing driver library, say exactly that.
        if rejected.len() == 1 {
            if let Some((backend, library)) = missing_library {
                return Err(EngineFailure::MissingLibrary { backend, library });
            }
        }
        return Err(EngineFailure::NoUsableEngine { rejected });
    };

    // Honest fallback notice: something preferred was rejected and a CPU
    // engine was used instead.
    let vulkan_rejected = rejected
        .iter()
        .any(|(backend, _)| *backend == Backend::Vulkan);
    let fallback_notice =
        if vulkan_rejected && backend == Backend::Cpu && backend_override.is_none() {
            let reason = rejected
                .iter()
                .find(|(backend, _)| *backend == Backend::Vulkan)
                .map(|(_, reason)| reason.clone())
                .unwrap_or_default();
            Some(format!(
                "Vulkan unavailable ({reason}); using the CPU engine."
            ))
        } else {
            None
        };

    Ok(BackendSelection {
        chosen: backend,
        path,
        version,
        fallback_notice,
        rejected,
    })
}

/// Selects a backend starting from an optional explicit engine directory
/// (otherwise discovery runs). Distinguishes "nothing bundled" from
/// "bundled but unusable".
pub fn try_select_backend(
    engine_dir: Option<&Path>,
    backend_override: Option<Backend>,
    icd_dirs: Option<&[PathBuf]>,
) -> Result<BackendSelection, EngineFailure> {
    let dir = match engine_dir {
        Some(dir) => dir.to_path_buf(),
        None => {
            crate::engine::bundle::discover_engine_dir().ok_or(EngineFailure::NoBundledEngine)?
        }
    };
    let bundle = crate::engine::bundle::load_bundle(&dir).map_err(|message| {
        EngineFailure::NoUsableEngine {
            rejected: vec![(Backend::Cpu, message)],
        }
    })?;
    if bundle.abi != EXPECTED_ENGINE_ABI {
        return Err(EngineFailure::AbiMismatch {
            found: bundle.abi,
            expected: EXPECTED_ENGINE_ABI,
        });
    }
    select_backend(&bundle, backend_override, icd_dirs)
}

/// Why a candidate was not selected: a `Reject` reason is listed per
/// backend, while a `Terminal` failure ends selection outright — an ABI
/// or version mismatch means the bundle itself disagrees with this app
/// or with its own binaries, and falling through to another backend
/// would silently paper over that.
enum CandidateFailure {
    Reject(String),
    /// Kept typed so a bundle with no usable fallback can surface the
    /// precise "install your GPU's Vulkan driver" failure instead of a
    /// generic rejection list.
    MissingLibrary(String),
    Terminal(EngineFailure),
}

fn evaluate_candidate(
    backend: &Backend,
    path: &Path,
    file: &str,
    bundle: &Bundle,
    icd_dirs: Option<&[PathBuf]>,
) -> Result<EngineVersion, CandidateFailure> {
    let expected = expected_sum(bundle, file)
        .map_err(|failure| CandidateFailure::Reject(failure.to_string()))?;
    verify_engine(path, expected)
        .map_err(|failure| CandidateFailure::Reject(failure.to_string()))?;
    if *backend == Backend::Vulkan {
        if let Some(reason) = vulkan_icd_reason(icd_dirs) {
            return Err(CandidateFailure::Reject(reason));
        }
    }
    probe_engine(path, Some(&bundle.version)).map_err(|failure| match failure {
        ProbeFailure::AbiMismatch { found, expected } => {
            CandidateFailure::Terminal(EngineFailure::AbiMismatch { found, expected })
        }
        ProbeFailure::VersionMismatch { found, expected } => {
            CandidateFailure::Terminal(EngineFailure::VersionMismatch { found, expected })
        }
        ProbeFailure::MissingLibrary { library } => CandidateFailure::MissingLibrary(library),
        other => CandidateFailure::Reject(other.to_string()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unix_lib_stderr() -> &'static str {
        "./starling-serve-vulkan: error while loading shared libraries: libvulkan.so.1: cannot open shared object file: No such file or directory\n"
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

    #[test]
    fn parses_version_output() {
        let version = parse_version_output(
            "starling-serve 0.1.0\nabi-version: 8\nbackend: contract-fixture\nsupported-models: parakeet s1\n",
        )
        .expect("version parses");
        assert_eq!(version.version, "0.1.0");
        assert_eq!(version.abi, 8);
        assert_eq!(version.backend_family, "contract-fixture");

        assert!(matches!(
            parse_version_output("starling-serve 0.1.0\nbackend: cpu\n"),
            Err(ProbeFailure::BadOutput)
        ));
        assert!(matches!(
            parse_version_output("total garbage\n"),
            Err(ProbeFailure::BadOutput)
        ));
    }

    #[test]
    fn unix_exit_127_names_the_library() {
        match classify_exit(127, unix_lib_stderr(), ExitFamily::Unix) {
            ProbeFailure::MissingLibrary { library } => {
                assert_eq!(library, "libvulkan.so.1");
            }
            other => panic!("expected MissingLibrary, got {other:?}"),
        }
        // Exit 127 without the loader line is an ordinary crash.
        assert!(matches!(
            classify_exit(127, "command not found", ExitFamily::Unix),
            ProbeFailure::Crashed { code: 127, .. }
        ));
    }

    #[test]
    fn windows_loader_statuses_classify() {
        match classify_exit(crate::engine::STATUS_DLL_NOT_FOUND, "", ExitFamily::Windows) {
            ProbeFailure::MissingLibrary { library } => assert!(library.contains("vulkan-1.dll")),
            other => panic!("expected MissingLibrary, got {other:?}"),
        }
        match classify_exit(
            crate::engine::STATUS_INVALID_IMAGE_FORMAT,
            "",
            ExitFamily::Windows,
        ) {
            ProbeFailure::Crashed { code, stderr_tail } => {
                assert_eq!(code, crate::engine::STATUS_INVALID_IMAGE_FORMAT);
                assert!(stderr_tail.contains("bad image"));
            }
            other => panic!("expected Crashed, got {other:?}"),
        }
        // The raw numeric form of STATUS_DLL_NOT_FOUND as i32.
        assert_eq!(crate::engine::STATUS_DLL_NOT_FOUND, -1073741515);
    }

    #[test]
    fn other_nonzero_exits_crash_with_tail() {
        match classify_exit(1, "line one\nline two\n", ExitFamily::Unix) {
            ProbeFailure::Crashed { code, stderr_tail } => {
                assert_eq!(code, 1);
                assert!(stderr_tail.contains("line one"));
            }
            other => panic!("expected Crashed, got {other:?}"),
        }
        match classify_exit(2, "", ExitFamily::Unix) {
            ProbeFailure::Crashed { stderr_tail, .. } => {
                assert_eq!(stderr_tail, "(no output)");
            }
            other => panic!("expected Crashed, got {other:?}"),
        }
    }

    #[test]
    fn missing_library_extraction() {
        assert_eq!(
            missing_library_from_stderr(unix_lib_stderr()),
            Some("libvulkan.so.1".to_string())
        );
        assert_eq!(missing_library_from_stderr("unrelated"), None);
    }

    #[test]
    fn icd_scan_finds_json_manifests() {
        let root = tempfile::tempdir().expect("tempdir");
        let icd = root.path().join("vulkan/icd.d");
        std::fs::create_dir_all(&icd).expect("create icd dir");
        assert!(!icd_present_in(std::slice::from_ref(&icd)));
        std::fs::write(icd.join("radeon_icd.x86_64.json"), "{}").expect("write icd");
        assert!(icd_present_in(std::slice::from_ref(&icd)));
        assert!(!icd_present_in(&[root.path().join("elsewhere")]));
    }

    #[test]
    fn select_backend_falls_back_to_cpu_with_notice() {
        let dir = tempfile::tempdir().expect("tempdir");
        stage_fake_engines(dir.path(), true);
        let bundle = crate::engine::bundle::load_bundle(dir.path()).expect("bundle");
        // No ICD anywhere: Vulkan is skipped with the driver sentence.
        let empty_dirs: Vec<PathBuf> = vec![dir.path().join("no-icd")];
        let selection =
            select_backend(&bundle, None, Some(&empty_dirs)).expect("selection succeeds");
        assert_eq!(selection.chosen, Backend::Cpu);
        let notice = selection.fallback_notice.expect("fallback notice");
        assert!(notice.starts_with("Vulkan unavailable (No Vulkan driver (ICD) is installed)"));
        assert!(notice.ends_with("using the CPU engine."));
        assert_eq!(selection.rejected.len(), 1);
        assert_eq!(selection.rejected[0].0, Backend::Vulkan);
        // CPU override skips Vulkan entirely: no notice, no rejection.
        let selection =
            select_backend(&bundle, Some(Backend::Cpu), Some(&empty_dirs)).expect("selection");
        assert_eq!(selection.chosen, Backend::Cpu);
        assert!(selection.fallback_notice.is_none());
        assert!(selection.rejected.is_empty());
    }

    #[test]
    fn select_backend_rejects_bad_checksums() {
        let dir = tempfile::tempdir().expect("tempdir");
        stage_fake_engines(dir.path(), false); // sums deliberately wrong
        let bundle = crate::engine::bundle::load_bundle(dir.path()).expect("bundle");
        match select_backend(&bundle, None, None) {
            Err(EngineFailure::NoUsableEngine { rejected }) => {
                assert!(rejected
                    .iter()
                    .any(|(_, reason)| reason.contains("does not match its checksum")));
            }
            other => panic!("expected NoUsableEngine, got {other:?}"),
        }
    }

    #[test]
    fn select_backend_reports_missing_library_for_vulkan_only() {
        let dir = tempfile::tempdir().expect("tempdir");
        // A script whose --version exits 127 naming libvulkan: the classic
        // "Vulkan build without the driver" case.
        let engine = dir.path().join("starling-serve-vulkan");
        std::fs::write(
            &engine,
            "#!/bin/sh\necho 'error while loading shared libraries: libvulkan.so.1: cannot open' >&2\nexit 127\n",
        )
        .expect("write engine");
        make_executable(&engine);
        let hex = crate::engine::bundle::sha256_file(&engine).expect("hash");
        std::fs::write(
            dir.path().join("engines.json"),
            r#"{"version":"9.9.9","abi":8,"engines":[{"backend":"vulkan","file":"starling-serve-vulkan"}]}"#,
        )
        .expect("manifest");
        std::fs::write(
            dir.path().join("SHA256SUMS.txt"),
            format!("{hex}  starling-serve-vulkan\n"),
        )
        .expect("sums");
        let bundle = crate::engine::bundle::load_bundle(dir.path()).expect("bundle");
        match select_backend(&bundle, None, None) {
            Err(EngineFailure::MissingLibrary { backend, library }) => {
                assert_eq!(backend, Backend::Vulkan);
                assert_eq!(library, "libvulkan.so.1");
            }
            other => panic!("expected MissingLibrary, got {other:?}"),
        }
    }

    #[test]
    fn select_backend_surfaces_abi_mismatch() {
        let dir = tempfile::tempdir().expect("tempdir");
        let engine = dir.path().join("starling-serve-cpu");
        std::fs::write(
            &engine,
            "#!/bin/sh\necho 'starling-serve 0.1.0'\necho 'abi-version: 7'\necho 'backend: cpu'\n",
        )
        .expect("write engine");
        make_executable(&engine);
        let hex = crate::engine::bundle::sha256_file(&engine).expect("hash");
        std::fs::write(
            dir.path().join("engines.json"),
            r#"{"version":"0.1.0","abi":7,"engines":[{"backend":"cpu","file":"starling-serve-cpu"}]}"#,
        )
        .expect("manifest");
        std::fs::write(
            dir.path().join("SHA256SUMS.txt"),
            format!("{hex}  starling-serve-cpu\n"),
        )
        .expect("sums");
        let bundle = crate::engine::bundle::load_bundle(dir.path()).expect("bundle");
        match select_backend(&bundle, None, None) {
            Err(EngineFailure::AbiMismatch { found, expected }) => {
                assert_eq!(found, 7);
                assert_eq!(expected, EXPECTED_ENGINE_ABI);
            }
            other => panic!("expected AbiMismatch, got {other:?}"),
        }
    }

    /// Stages two fake engines: scripts that print a valid `--version`
    /// block. `correct_sums` false writes wrong hex so verification
    /// fails.
    fn stage_fake_engines(dir: &Path, correct_sums: bool) {
        for backend in ["vulkan", "cpu"] {
            let name = format!("starling-serve-{backend}");
            let path = dir.join(&name);
            std::fs::write(
                &path,
                format!(
                    "#!/bin/sh\ncat <<'EOF'\nstarling-serve 1.2.3\nabi-version: 8\nbackend: {backend}\nsupported-models: parakeet\nEOF\n"
                ),
            )
            .expect("write engine");
            make_executable(&path);
            let real = crate::engine::bundle::sha256_file(&path).expect("hash");
            let hex = if correct_sums { real } else { "0".repeat(64) };
            let sums_path = dir.join("SHA256SUMS.txt");
            let line = format!("{hex}  {name}\n");
            let existing = std::fs::read_to_string(&sums_path).unwrap_or_default();
            std::fs::write(&sums_path, existing + &line).expect("write sums");
        }
        std::fs::write(
            dir.join("engines.json"),
            r#"{"version":"1.2.3","abi":8,"engines":[
                {"backend":"vulkan","file":"starling-serve-vulkan"},
                {"backend":"cpu","file":"starling-serve-cpu"}]}"#,
        )
        .expect("manifest");
    }

    #[test]
    fn probe_detects_version_mismatch() {
        let dir = tempfile::tempdir().expect("tempdir");
        let engine = dir.path().join("engine");
        std::fs::write(
            &engine,
            "#!/bin/sh\necho 'starling-serve 0.9.0'; echo 'abi-version: 8'; echo 'backend: cpu'\n",
        )
        .expect("write engine");
        make_executable(&engine);
        match probe_engine(&engine, Some("1.0.0")) {
            Err(ProbeFailure::VersionMismatch { found, expected }) => {
                assert_eq!(found, "0.9.0");
                assert_eq!(expected, "1.0.0");
            }
            other => panic!("expected VersionMismatch, got {other:?}"),
        }
        assert!(probe_engine(&engine, Some("0.9.0")).is_ok());
    }

    #[test]
    fn probe_detects_abi_mismatch() {
        let dir = tempfile::tempdir().expect("tempdir");
        let engine = dir.path().join("engine");
        std::fs::write(
            &engine,
            "#!/bin/sh\necho 'starling-serve 0.1.0'; echo 'abi-version: 7'; echo 'backend: cpu'\n",
        )
        .expect("write engine");
        make_executable(&engine);
        match probe_engine(&engine, None) {
            Err(ProbeFailure::AbiMismatch { found, expected }) => {
                assert_eq!(found, 7);
                assert_eq!(expected, EXPECTED_ENGINE_ABI);
            }
            other => panic!("expected AbiMismatch, got {other:?}"),
        }
    }

    #[test]
    fn probe_classifies_spawn_failures() {
        match probe_engine(Path::new("/nonexistent/starling-serve"), None) {
            Err(ProbeFailure::NotExecutable { reason }) => {
                assert!(reason.contains("does not exist"));
            }
            other => panic!("expected NotExecutable, got {other:?}"),
        }
        let dir = tempfile::tempdir().expect("tempdir");
        let engine = dir.path().join("engine");
        std::fs::write(&engine, "not executable").expect("write engine");
        // No +x on unix: PermissionDenied at spawn.
        match probe_engine(&engine, None) {
            Err(ProbeFailure::NotExecutable { reason }) => {
                assert!(reason.contains("not executable"), "got {reason}");
            }
            other => panic!("expected NotExecutable, got {other:?}"),
        }
    }
}
