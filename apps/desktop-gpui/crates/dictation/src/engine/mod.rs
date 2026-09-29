//! Bundled-engine supervision for the desktop app (#362, #363).
//!
//! The desktop archive ships the native server (`starling-serve`) next to
//! the app executable. This module is the UI-free core that finds those
//! binaries, verifies them, picks the best backend, runs the server as a
//! supervised loopback sidecar, downloads and verifies models, and swaps
//! models at runtime without dropping in-flight takes. The gpui app (wave
//! C) only consumes [`EngineManager`]; every state decision in here is
//! observable through [`EngineManager::snapshot`].
//!
//! Why a sidecar at all: the server binds exactly one model per process
//! (`cpp/serve/main.cpp`), so a runtime model switch is a second process
//! on a second loopback port plus an atomic cutover — orchestrated here,
//! never by the UI (#363).

pub mod bundle;
pub mod catalog;
pub mod download;
pub mod manager;
pub mod memory;
pub mod probe;
pub mod registry;
pub mod sidecar;

pub use bundle::{discover_engine_dir, load_bundle, Bundle};
pub use catalog::{default_catalog, CatalogEntry};
pub use download::{
    delete_model_files, download_model, scan_install, verify_placed_file, DownloadError,
};
pub use manager::{
    ActiveEngineView, BackendSelectionView, EngineConfig, EngineError, EngineLease, EngineManager,
    EnginePhase, EngineSnapshot, InstallState, ModelView, SwapDecision, SwapMode, SwitchReport,
    SwitchStage, SwitchView,
};
pub use memory::{
    available_memory, estimate_resident, process_peak_rss, process_rss, swap_plan, SwapPlan,
    SWAP_MARGIN_BYTES,
};
pub use probe::{
    select_backend, try_select_backend, BackendSelection, EngineVersion, ProbeFailure,
};
pub use registry::{
    read_registration, registry_path, remove_registration, write_registration, SidecarRegistration,
};
pub use sidecar::{HealthSnapshot, ReadyError, ReadyStage, Sidecar};

/// The engine ABI this app speaks. It must match
/// `STARLING_GGML_ABI_VERSION` in `cpp/include/starling_ggml.h` — the same
/// lockstep contract the Python client pins in
/// `src/starling/_ggml/_native.py`. A mismatched engine is never run; it
/// is reported with an actionable message instead (#362).
pub const EXPECTED_ENGINE_ABI: u32 = 8;

/// An inference backend family. The bundle ships one server binary per
/// family; [`probe::select_backend`] prefers `Vulkan` and falls back to
/// `Cpu` with an honest notice (#362).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Backend {
    Vulkan,
    Cpu,
}

impl Backend {
    /// The lowercase key used in `engines.json` and `--version` output.
    pub fn as_str(&self) -> &'static str {
        match self {
            Backend::Vulkan => "vulkan",
            Backend::Cpu => "cpu",
        }
    }

    /// Parses the `engines.json` `backend` key. Unknown families are
    /// rejected rather than guessed: the manifest and this app must agree
    /// on what can be selected.
    pub fn parse(value: &str) -> Option<Backend> {
        match value {
            "vulkan" => Some(Backend::Vulkan),
            "cpu" => Some(Backend::Cpu),
            _ => None,
        }
    }
}

impl std::fmt::Display for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Every way the engine subsystem can end up unusable. Each variant
/// carries a human sentence that says what to do — these surface directly
/// in the app's engine status (#362 step 4).
#[derive(Clone, Debug, PartialEq)]
pub enum EngineFailure {
    /// No `engines/` layout was found next to the app (and no
    /// `STARLING_ENGINE_DIR`). In development builds the engines are not
    /// bundled; the user must stage them or use the Manual server mode.
    NoBundledEngine,

    /// Every candidate backend was rejected during selection; the payload
    /// lists each backend with the reason it cannot run.
    NoUsableEngine {
        /// `(backend, reason)` for every candidate that was tried.
        rejected: Vec<(Backend, String)>,
    },

    /// An engine file's sha256 does not match `SHA256SUMS.txt` (or has no
    /// entry there). An unverified engine is never run.
    ChecksumMismatch {
        file: String,
    },

    AbiMismatch {
        found: u32,
        expected: u32,
    },

    VersionMismatch {
        found: String,
        expected: String,
    },

    MissingLibrary {
        backend: Backend,
        library: String,
    },

    AnnounceTimeout,

    BindFailed(String),

    LoadFailed(String),

    CrashLoop {
        last_stderr: String,
    },
}

fn format_rejected(rejected: &[(Backend, String)]) -> String {
    rejected
        .iter()
        .map(|(backend, reason)| format!("{backend}: {reason}"))
        .collect::<Vec<_>>()
        .join("; ")
}

impl std::fmt::Display for EngineFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EngineFailure::NoBundledEngine => write!(
                f,
                "No bundled engine was found. Use the Manual server mode, or set \
                 STARLING_ENGINE_DIR to a directory staged by scripts/stage-engines.sh."
            ),
            EngineFailure::NoUsableEngine { rejected } => write!(
                f,
                "None of the bundled engines can run on this machine: {}.",
                format_rejected(rejected)
            ),
            EngineFailure::ChecksumMismatch { file } => write!(
                f,
                "The bundled engine file {file} does not match its checksum. \
                 Reinstall the app (or restage the engines directory); the engine was not run."
            ),
            EngineFailure::AbiMismatch { found, expected } => write!(
                f,
                "The bundled engine speaks ABI {found}, but this app expects ABI {expected}. \
                 Update the app so engine and app match; the engine was not run."
            ),
            EngineFailure::VersionMismatch { found, expected } => write!(
                f,
                "The bundled engine reports version {found}, but the bundle manifest \
                 says {expected}. Reinstall the app (or restage the engines directory)."
            ),
            EngineFailure::MissingLibrary { backend, library } => write!(
                f,
                "The {backend} engine could not start because {library} is missing. \
                 Install your GPU's Vulkan driver, or use the CPU engine."
            ),
            EngineFailure::AnnounceTimeout => write!(
                f,
                "The engine started but never reported its address within 30 seconds. \
                 Choose Retry; if it keeps happening, use the CPU engine or the Manual server mode."
            ),
            EngineFailure::BindFailed(message) => {
                write!(f, "The engine could not bind a loopback port: {message}.")
            }
            EngineFailure::LoadFailed(message) => write!(
                f,
                "The model failed to load: {message} The engine was stopped. \
                 Choose Retry, or activate a different model."
            ),
            EngineFailure::CrashLoop { last_stderr } => write!(
                f,
                "The engine kept crashing (5 times within 5 minutes) and was stopped. \
                 Last output: {last_stderr} Choose Retry, use the CPU engine, or the Manual server mode."
            ),
        }
    }
}

impl std::error::Error for EngineFailure {}

/// How a process's `--version`/exit information classifies into a probe
/// failure. Pure so the Windows status codes are unit-testable on any
/// host (#362 step 4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ExitFamily {
    Unix,
    Windows,
}

impl ExitFamily {
    /// The classification family for the host this code runs on.
    pub(crate) fn host() -> ExitFamily {
        if cfg!(windows) {
            ExitFamily::Windows
        } else {
            ExitFamily::Unix
        }
    }
}

/// `STATUS_DLL_NOT_FOUND` (`0xC0000135`) as the raw i32 Windows exit code
/// a loader failure produces — the shape `std::process::ExitStatus::code`
/// reports it in.
pub(crate) const STATUS_DLL_NOT_FOUND: i32 = 0xC0000135u32 as i32;

/// `STATUS_INVALID_IMAGE_FORMAT` (`0xC000007B`): wrong-architecture
/// binary, surfaced as a "bad image" crash rather than a missing library.
pub(crate) const STATUS_INVALID_IMAGE_FORMAT: i32 = 0xC000007Bu32 as i32;

#[cfg(test)]
mod tests {
    use super::*;

    /// The lockstep contract: `EXPECTED_ENGINE_ABI` must equal
    /// `STARLING_GGML_ABI_VERSION` in the C++ header, exactly like
    /// `src/starling/_ggml/_native.py` pins it on the Python side. If this
    /// fails, the app and the engine it bundles disagree about the native
    /// entry points and every bundled engine must be refused (#362).
    #[test]
    fn abi_version_matches_the_cpp_header() {
        let header = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../../cpp/include/starling_ggml.h"
        );
        let text = std::fs::read_to_string(header)
            .unwrap_or_else(|error| panic!("cannot read {header}: {error}"));
        let mut found = None;
        for line in text.lines() {
            let line = line.trim();
            if let Some(rest) = line.strip_prefix("#define STARLING_GGML_ABI_VERSION") {
                found = rest.trim().parse::<u32>().ok();
            }
        }
        assert_eq!(
            found,
            Some(EXPECTED_ENGINE_ABI),
            "STARLING_GGML_ABI_VERSION in {header} must equal EXPECTED_ENGINE_ABI"
        );
    }

    #[test]
    fn backend_parses_manifest_keys() {
        assert_eq!(Backend::parse("vulkan"), Some(Backend::Vulkan));
        assert_eq!(Backend::parse("cpu"), Some(Backend::Cpu));
        assert_eq!(Backend::parse("cuda"), None);
        assert_eq!(Backend::Cpu.to_string(), "cpu");
    }

    #[test]
    fn failure_sentences_say_what_to_do() {
        let missing = EngineFailure::MissingLibrary {
            backend: Backend::Vulkan,
            library: "libvulkan.so.1".to_string(),
        };
        let sentence = missing.to_string();
        assert!(
            sentence.contains("Install your GPU's Vulkan driver, or use the CPU engine."),
            "got: {sentence}"
        );
        let unusable = EngineFailure::NoUsableEngine {
            rejected: vec![
                (
                    Backend::Vulkan,
                    "No Vulkan driver (ICD) is installed".to_string(),
                ),
                (Backend::Cpu, "checksum mismatch".to_string()),
            ],
        };
        assert!(unusable.to_string().contains("vulkan: No Vulkan driver"));
        assert!(EngineFailure::NoBundledEngine
            .to_string()
            .contains("STARLING_ENGINE_DIR"));
    }
}
