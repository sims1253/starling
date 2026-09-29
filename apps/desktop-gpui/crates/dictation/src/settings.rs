//! Persistent app settings, replacing the `localStorage` keys used by
//! `apps/desktop/src/App.tsx` (`starling:endpoint`, `starling:model`,
//! `starling:terms`) with a JSON file. See
//! `apps/desktop-gpui/PORT.md`.

//! # The dropped `storageBackend` key (D14, deliberate)
//!
//! The cutover-era build persisted a `storageBackend` choice
//! (`"v1"`/`"v2"`). Since D14 (storage v2 is THE store — no backwards
//! compatibility of any kind) this build runs v2 unconditionally: the key
//! is unknown, ignored on load, and dropped on the next save. That drop is
//! intended, not an oversight — pinned by
//! `a_legacy_storage_backend_choice_is_ignored_since_d14` below. The same
//! posture applies to the v1 `sessions/`/`journals/` histories: they stay
//! on disk, untouched and no longer read anywhere.

use std::io;
use std::path::{Path, PathBuf};

/// The platform config directory could not be resolved (on Linux,
/// `$XDG_CONFIG_HOME` and `$HOME` are both unset). Returned instead of
/// silently reading/writing settings in an arbitrary working directory.
#[derive(Debug, thiserror::Error)]
#[error("could not resolve the user config directory; set XDG_CONFIG_HOME or HOME")]
pub struct ConfigDirUnavailable;

/// Which engine transcribes a take (#362, #363): the bundled sidecar
/// the app supervises itself, or the user's own hand-run server. A fresh
/// install is the self-contained experience (`builtin`); a settings file
/// written before this key existed loads as `manual` — see
/// [`Settings::load`] — because a working hand-run-server setup must
/// never silently change.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EngineMode {
    /// The bundled engine, supervised by the app (`engine::EngineManager`).
    Builtin,
    /// The user's own starling-serve / OpenAI-compatible endpoint.
    Manual,
}

/// The engine subsection of the settings file (#362): which engine runs,
/// which catalog model it serves, and whether the user pinned the
/// backend family. `active_model` is a catalog id (not the server slug):
/// the manager resolves it to an endpoint and slug at startup.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EngineSettings {
    pub mode: EngineMode,
    /// The catalog id of the model the built-in engine serves. Written by
    /// the app whenever the engine's active model changes (the engine is
    /// the source of truth; the file only restores it at launch).
    #[serde(default)]
    pub active_model: Option<String>,
    /// `"cpu"` when the user chose the CPU engine (skipping Vulkan
    /// selection); `None` is automatic (Vulkan preferred, CPU fallback).
    #[serde(default)]
    pub backend_override: Option<String>,
}

impl Default for EngineSettings {
    fn default() -> Self {
        Self {
            mode: EngineMode::Builtin,
            active_model: None,
            backend_override: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Settings {
    pub endpoint: String,
    pub model: String,
    pub expected_terms: Vec<String>,
    /// Whether the user explicitly chose the model (R02). While false, the
    /// app may sync `model` from the server's health response. Files written
    /// before this field existed deserialize it as `false` (`serde(default)`),
    /// which re-enables that sync instead of guessing from the file's text.
    #[serde(default)]
    pub user_set_model: bool,
    /// Text processing after transcription (#294/#295). A file without
    /// the key loads the defaults: raw transcripts, nothing sent anywhere.
    #[serde(default)]
    pub processing: ProcessingSettings,
    /// The transcription engine (#362). A file without the key is a
    /// legacy file and loads as `manual` — see [`Settings::load`].
    #[serde(default)]
    pub engine: EngineSettings,
}

/// Which processing mode runs after a take is transcribed, and where its
/// providers live. The key itself never goes into this file: only the
/// name of the environment variable that holds it.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct ProcessingSettings {
    /// A mode id from the app's built-in modes.
    pub mode: String,
    /// The starling-serve instance running S1-mini (its own process: one
    /// model is resident per server). Must be on this machine.
    pub s1_endpoint: String,
    /// An OpenAI-compatible base URL (`https://api.openai.com/v1`,
    /// OpenRouter, a local llama-server's `/v1`).
    pub api_endpoint: String,
    /// The model the API should use; empty means the API mode is not set up.
    pub api_model: String,
    /// The environment variable holding the API key.
    pub api_key_env: String,
}

impl Default for ProcessingSettings {
    fn default() -> Self {
        Self {
            mode: "verbatim".to_string(),
            s1_endpoint: "http://127.0.0.1:8182".to_string(),
            api_endpoint: "https://api.openai.com/v1".to_string(),
            api_model: String::new(),
            api_key_env: "OPENAI_API_KEY".to_string(),
        }
    }
}

impl Settings {
    /// Defaults mirroring `App.tsx`: `DEFAULT_ENDPOINT`, the `parakeet`
    /// model, and the "auth" expected-terms input. The engine defaults to
    /// `builtin` with no model (#362): a fresh install is the
    /// self-contained experience and picks its model on first run.
    pub fn default_settings() -> Self {
        Self {
            endpoint: "http://127.0.0.1:8181".to_string(),
            model: "parakeet".to_string(),
            expected_terms: vec!["auth".to_string()],
            user_set_model: false,
            processing: ProcessingSettings::default(),
            engine: EngineSettings::default(),
        }
    }

    /// `dirs::config_dir()/starling-gpui/settings.json`. A `None` from
    /// `dirs` is a typed error (R11): settings must never silently land in
    /// whatever directory the app happened to start in.
    pub fn default_path() -> Result<PathBuf, ConfigDirUnavailable> {
        Ok(dirs::config_dir()
            .ok_or(ConfigDirUnavailable)?
            .join("starling-gpui")
            .join("settings.json"))
    }

    /// Missing or corrupt file always falls back to the defaults; so does an
    /// unresolvable config directory (there is nothing to load from it).
    pub fn load_or_default() -> Self {
        match Self::default_path() {
            Ok(path) => Self::load(&path),
            Err(_) => Self::default_settings(),
        }
    }

    /// Missing or corrupt file always falls back to the defaults; never panics.
    ///
    /// A file that parses but carries no `engine` key is a legacy file
    /// (#362): it loads with `engine.mode = manual`, because every such
    /// file was written by a build whose only way to transcribe was a
    /// hand-run server — silently switching those users to the bundled
    /// engine would change where their audio goes. The key is detected on
    /// the parsed `serde_json::Value`, not sniffed from the text.
    pub fn load(path: &Path) -> Self {
        let Some(text) = std::fs::read_to_string(path).ok() else {
            return Self::default_settings();
        };
        let Some(value) = serde_json::from_str::<serde_json::Value>(&text).ok() else {
            return Self::default_settings();
        };
        let Ok(mut settings) = serde_json::from_value::<Settings>(value.clone()) else {
            return Self::default_settings();
        };
        if value.get("engine").is_none() {
            settings.engine.mode = EngineMode::Manual;
        }
        settings
    }

    /// Atomic write: serialize pretty JSON to a sibling `.tmp`, then rename.
    pub fn save(&self, path: &Path) -> io::Result<()> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }

        let json = serde_json::to_string_pretty(self)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;

        write_atomic(path, json.as_bytes())
    }

    /// The comma-joined form the settings UI edits, e.g. `"auth, Starling"`.
    pub fn expected_terms_input(&self) -> String {
        self.expected_terms.join(", ")
    }

    /// Split on commas, trim, and drop empty entries.
    pub fn set_expected_terms_input(&mut self, input: &str) {
        self.expected_terms = input
            .split(',')
            .map(str::trim)
            .filter(|term| !term.is_empty())
            .map(str::to_string)
            .collect();
    }
}

/// Write through a sibling `<file>.tmp`, then rename over the target.
fn write_atomic(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let mut file_name = path
        .file_name()
        .map(|name| name.to_os_string())
        .unwrap_or_default();
    file_name.push(".tmp");
    let tmp = path.with_file_name(file_name);

    let result = std::fs::write(&tmp, bytes).and_then(|()| std::fs::rename(&tmp, path));

    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn defaults_match_app_defaults() {
        let settings = Settings::default_settings();

        assert_eq!(settings.endpoint, "http://127.0.0.1:8181");
        assert_eq!(settings.model, "parakeet");
        assert_eq!(settings.expected_terms, vec!["auth".to_string()]);
        assert_eq!(settings.expected_terms_input(), "auth");
        assert!(!settings.user_set_model);
    }

    #[test]
    fn save_and_load_roundtrip() {
        let temp = TempDir::new().expect("tempdir");
        let path = temp.path().join("nested").join("settings.json");

        let settings = Settings {
            endpoint: "http://127.0.0.1:9999/".to_string(),
            model: "whisper-large-v3".to_string(),
            expected_terms: vec!["auth".to_string(), "Starling".to_string()],
            user_set_model: true,
            processing: ProcessingSettings {
                mode: "clean-local".to_string(),
                api_model: "gpt-4.1-mini".to_string(),
                ..ProcessingSettings::default()
            },
            engine: EngineSettings {
                mode: EngineMode::Builtin,
                active_model: Some("parakeet-v3-q8".to_string()),
                backend_override: Some("cpu".to_string()),
            },
        };

        settings.save(&path).expect("save");
        assert_eq!(Settings::load(&path), settings);

        // camelCase keys on disk.
        let raw = std::fs::read_to_string(&path).expect("read settings file");
        let value: serde_json::Value = serde_json::from_str(&raw).expect("parse settings");
        assert_eq!(value["endpoint"], "http://127.0.0.1:9999/");
        assert_eq!(value["model"], "whisper-large-v3");
        assert_eq!(
            value["expectedTerms"],
            serde_json::json!(["auth", "Starling"])
        );
        assert_eq!(value["userSetModel"], true);
        assert_eq!(value["processing"]["mode"], "clean-local");
        assert_eq!(value["processing"]["apiKeyEnv"], "OPENAI_API_KEY");
        // The engine subsection is camelCase like the rest of the file.
        assert_eq!(value["engine"]["mode"], "builtin");
        assert_eq!(value["engine"]["activeModel"], "parakeet-v3-q8");
        assert_eq!(value["engine"]["backendOverride"], "cpu");
    }

    #[test]
    fn fresh_defaults_use_the_builtin_engine_with_no_model() {
        // #362: a fresh install is the self-contained experience — the
        // bundled engine with no model picked yet (first run offers the
        // recommended download).
        let settings = Settings::default_settings();
        assert_eq!(settings.engine.mode, EngineMode::Builtin);
        assert_eq!(settings.engine.active_model, None);
        assert_eq!(settings.engine.backend_override, None);
    }

    #[test]
    fn a_legacy_file_without_the_engine_key_loads_as_manual() {
        let temp = TempDir::new().expect("tempdir");
        let path = temp.path().join("settings.json");
        // The exact shape a pre-engine build wrote (including a backend
        // choice this build ignores): the file still loads, and the engine
        // mode is manual so a working hand-run server keeps serving.
        std::fs::write(
            &path,
            r#"{"endpoint":"http://10.0.0.5:8181","model":"whisper-large-v3","expectedTerms":["auth"],"userSetModel":true,"storageBackend":"v1"}"#,
        )
        .expect("write legacy settings");
        let settings = Settings::load(&path);
        assert_eq!(settings.endpoint, "http://10.0.0.5:8181");
        assert_eq!(settings.engine.mode, EngineMode::Manual);
        assert_eq!(settings.engine.active_model, None);
    }

    #[test]
    fn a_file_with_the_engine_key_roundtrips_its_mode() {
        // Explicit manual must survive a roundtrip; only the ABSENCE of the
        // key means legacy (manual), so a deliberate manual choice and a
        // builtin choice both persist exactly.
        let temp = TempDir::new().expect("tempdir");
        let path = temp.path().join("settings.json");
        let settings = Settings {
            engine: EngineSettings {
                mode: EngineMode::Manual,
                active_model: None,
                backend_override: None,
            },
            ..Settings::default_settings()
        };
        settings.save(&path).expect("save");
        assert_eq!(Settings::load(&path).engine, settings.engine);
    }

    #[test]
    fn processing_defaults_to_raw_and_sends_nothing() {
        let settings = Settings::default_settings();
        assert_eq!(settings.processing.mode, "verbatim");
        assert!(settings.processing.api_model.is_empty());
        // A file written before processing existed loads the defaults.
        let temp = TempDir::new().expect("tempdir");
        let path = temp.path().join("settings.json");
        std::fs::write(
            &path,
            r#"{"endpoint":"http://127.0.0.1:8181","model":"parakeet","expectedTerms":["auth"]}"#,
        )
        .expect("write settings");
        assert_eq!(Settings::load(&path).processing, ProcessingSettings::default());
    }

    #[test]
    fn a_legacy_storage_backend_choice_is_ignored_since_d14() {
        let temp = TempDir::new().expect("tempdir");
        let path = temp.path().join("settings.json");

        // A file written by the cutover-era build carried a storageBackend
        // choice. Storage v2 is THE store (D14, no backwards compatibility);
        // the key is unknown to this build and simply ignored — the file
        // still loads, and saving drops the key.
        std::fs::write(
            &path,
            r#"{"endpoint":"http://127.0.0.1:8181","model":"parakeet","expectedTerms":["auth"],"storageBackend":"v1"}"#,
        )
        .expect("write legacy settings");
        let settings = Settings::load(&path);
        assert_eq!(settings.endpoint, "http://127.0.0.1:8181");

        settings.save(&path).expect("save");
        let raw = std::fs::read_to_string(&path).expect("read");
        assert!(!raw.contains("storageBackend"), "{raw}");
    }

    #[test]
    fn legacy_files_without_the_flag_load_as_not_user_set() {
        let temp = TempDir::new().expect("tempdir");
        let path = temp.path().join("settings.json");

        // A file written before `userSetModel` existed: every legacy save
        // contains a `model` key, but that never meant the user chose it, so
        // the missing flag must not be sniffed out of the text.
        std::fs::write(
            &path,
            r#"{"endpoint":"http://10.0.0.5:8181","model":"whisper-large-v3","expectedTerms":["auth"]}"#,
        )
        .expect("write legacy settings");

        let settings = Settings::load(&path);
        assert_eq!(settings.endpoint, "http://10.0.0.5:8181");
        assert_eq!(settings.model, "whisper-large-v3");
        assert!(!settings.user_set_model, "no recorded choice: auto-sync stays enabled");
    }

    #[test]
    fn missing_file_loads_defaults() {
        let temp = TempDir::new().expect("tempdir");
        let path = temp.path().join("settings.json");

        assert_eq!(Settings::load(&path), Settings::default_settings());
    }

    #[test]
    fn corrupt_file_loads_defaults() {
        let temp = TempDir::new().expect("tempdir");
        let path = temp.path().join("settings.json");

        std::fs::write(&path, "{not json at all").expect("write corrupt settings");
        assert_eq!(Settings::load(&path), Settings::default_settings());

        // Wrong field types are corrupt too, not a panic.
        std::fs::write(&path, r#"{"endpoint":42}"#)
            .expect("write mistyped settings");
        assert_eq!(Settings::load(&path), Settings::default_settings());
    }

    #[test]
    fn expected_terms_join_and_split_roundtrip() {
        let mut settings = Settings::default_settings();
        assert_eq!(settings.expected_terms_input(), "auth");

        settings.set_expected_terms_input("auth, Starling,,  um  ");
        assert_eq!(
            settings.expected_terms,
            vec!["auth".to_string(), "Starling".to_string(), "um".to_string()]
        );
        assert_eq!(settings.expected_terms_input(), "auth, Starling, um");

        settings.set_expected_terms_input("   ");
        assert!(settings.expected_terms.is_empty());
        assert_eq!(settings.expected_terms_input(), "");
    }
}
