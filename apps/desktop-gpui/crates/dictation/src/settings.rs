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
/// never silently change. Deserialization is lossy by design: an
/// unknown `mode` string (a file written by a newer build) resolves to
/// `manual` instead of failing the whole file.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EngineMode {
    /// The bundled engine, supervised by the app (`engine::EngineManager`).
    #[default]
    Builtin,
    /// The user's own starling-serve / OpenAI-compatible endpoint.
    #[serde(other)]
    Manual,
}

/// The engine subsection of the settings file (#362): which engine runs,
/// which catalog model it serves, and whether the user pinned the
/// backend family. `active_model` is a catalog id (not the server slug):
/// the manager resolves it to an endpoint and slug at startup. A partial
/// `engine` object loads with defaults for whatever it does not state —
/// one new key must never cost the user their whole settings file.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct EngineSettings {
    pub mode: EngineMode,
    /// The catalog id of the model the built-in engine serves. Written by
    /// the app whenever the engine's active model changes (the engine is
    /// the source of truth; the file only restores it at launch).
    pub active_model: Option<String>,
    /// `"cpu"` when the user chose the CPU engine (skipping Vulkan
    /// selection); `None` is automatic (Vulkan preferred, CPU fallback).
    pub backend_override: Option<String>,
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
    /// A file that parses but carries no `engine` key — or an explicit
    /// JSON `null` one — is a legacy file (#362): it loads with
    /// `engine.mode = manual`, because every such file was written by a
    /// build whose only way to transcribe was a hand-run server —
    /// silently switching those users to the bundled engine would change
    /// where their audio goes. The key is detected on the parsed
    /// `serde_json::Value`, not sniffed from the text.
    ///
    /// A file that parses as JSON but fails typed deserialization keeps
    /// its engine mode honest too: the fallback defaults to `manual`
    /// unless the file's `engine.mode` says exactly `"builtin"`. A
    /// partially unreadable file must never change where audio is sent,
    /// so the fallback errs toward the hand-run server the user had.
    pub fn load(path: &Path) -> Self {
        let Some(text) = std::fs::read_to_string(path).ok() else {
            return Self::default_settings();
        };
        let Some(mut value) = serde_json::from_str::<serde_json::Value>(&text).ok() else {
            return Self::default_settings();
        };
        // Checked before `value` is moved into `from_value` (a deep clone
        // of the whole document just to look at one key would be waste).
        // A JSON `null` engine key states no engine any more than an
        // absent one: drop it so the rest of the file still loads, and
        // the legacy rule below applies (#366).
        if matches!(value.get("engine"), Some(serde_json::Value::Null)) {
            if let Some(object) = value.as_object_mut() {
                object.remove("engine");
            }
        }
        let legacy_file = value.get("engine").is_none();
        // The engine mode a typed-deserialization failure falls back to
        // (#366): manual unless the file's `engine.mode` says exactly
        // `"builtin"` — a partially unreadable file must never change
        // where audio is sent, so it errs toward the hand-run server.
        // A legacy file (no engine key after the null drop above) has no
        // mode to read at all and lands on manual too.
        let safe_mode = if value
            .get("engine")
            .and_then(|engine| engine.get("mode"))
            == Some(&serde_json::json!("builtin"))
        {
            EngineMode::Builtin
        } else {
            EngineMode::Manual
        };
        // The fallback also keeps whichever engine fields are still
        // readable: forgetting the served model would re-download it, the
        // same silent state change the safe mode above avoids.
        let engine_field = |key: &str| {
            value
                .get("engine")
                .and_then(|engine| engine.get(key))
                .and_then(serde_json::Value::as_str)
                .map(str::to_string)
        };
        let (active_model, backend_override) =
            (engine_field("activeModel"), engine_field("backendOverride"));
        let Ok(mut settings) = serde_json::from_value::<Settings>(value) else {
            let mut fallback = Self::default_settings();
            fallback.engine = EngineSettings {
                mode: safe_mode,
                active_model,
                backend_override,
            };
            return fallback;
        };
        if legacy_file {
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

/// Counts `write_atomic` calls so concurrent writers stage through
/// distinct temp files: a shared fixed name would let one writer truncate
/// another's staging file mid-write.
static WRITE_SEQUENCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Write through a sibling `<file>.<pid>.<sequence>.tmp`, then rename over
/// the target. The temp name is unique per writer (process id plus a
/// process-wide counter) so two concurrent saves cannot clobber each
/// other's staging file; a failed write removes its temp again.
fn write_atomic(path: &Path, bytes: &[u8]) -> io::Result<()> {
    sweep_stale_tmp_siblings(path);
    let sequence =
        WRITE_SEQUENCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let mut file_name = path
        .file_name()
        .map(|name| name.to_os_string())
        .unwrap_or_default();
    file_name.push(format!(".{}.{}.tmp", std::process::id(), sequence));
    let tmp = path.with_file_name(file_name);

    let result = std::fs::write(&tmp, bytes).and_then(|()| std::fs::rename(&tmp, path));

    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }

    result
}

/// Best-effort sweep of orphaned staging files from crashed writers
/// (#366): a completed write renames its `<name>.<pid>.<seq>.tmp` away, so
/// any sibling still matching that pattern is a leftover from a process
/// that died mid-write. Only files older than an hour are removed — a
/// concurrent live writer's fresh temp must survive — and every error is
/// ignored: the sweep may never break the save it accompanies.
fn sweep_stale_tmp_siblings(path: &Path) {
    const MAX_TMP_AGE: std::time::Duration = std::time::Duration::from_secs(60 * 60);
    let Some(parent) = path.parent() else { return };
    let Some(file_name) = path.file_name() else { return };
    let mut prefix = file_name.to_os_string();
    prefix.push(".");
    let prefix = prefix.as_encoded_bytes();
    let Ok(entries) = std::fs::read_dir(parent) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.as_encoded_bytes();
        // Only the exact `<name>.<pid>.<seq>.tmp` shape write_atomic
        // stages; anything else (say a backup tool's `<name>.bak.tmp`) is
        // not ours to delete.
        let Some(middle) = name
            .strip_prefix(prefix)
            .and_then(|rest| rest.strip_suffix(b".tmp"))
        else {
            continue;
        };
        let mut parts = middle.split(|&b| b == b'.');
        let staged = parts.next().is_some_and(is_ascii_number)
            && parts.next().is_some_and(is_ascii_number)
            && parts.next().is_none();
        if !staged {
            continue;
        }
        let orphaned = entry
            .metadata()
            .and_then(|metadata| metadata.modified())
            .ok()
            .and_then(|modified| modified.elapsed().ok())
            .is_some_and(|age| age > MAX_TMP_AGE);
        if orphaned {
            let _ = std::fs::remove_file(entry.path());
        }
    }
}

fn is_ascii_number(bytes: &[u8]) -> bool {
    !bytes.is_empty() && bytes.iter().all(u8::is_ascii_digit)
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
    fn a_null_engine_key_loads_as_manual_like_a_missing_one() {
        // #366: JSON `null` is as legacy as an absent key — a file whose
        // engine choice says nothing must not fall back to builtin and
        // silently move audio to the bundled engine.
        let temp = TempDir::new().expect("tempdir");
        let path = temp.path().join("settings.json");
        std::fs::write(
            &path,
            r#"{"endpoint":"http://10.0.0.5:8181","model":"whisper-large-v3","expectedTerms":["auth"],"engine":null}"#,
        )
        .expect("write null-engine settings");
        let settings = Settings::load(&path);
        assert_eq!(settings.endpoint, "http://10.0.0.5:8181");
        assert_eq!(settings.engine.mode, EngineMode::Manual);
    }

    #[test]
    fn an_unreadable_file_falls_back_to_manual_unless_it_says_builtin_exactly(
    ) {
        // #366: the file parses as JSON but typed deserialization fails
        // (here: `endpoint` is a number). The fallback must not silently
        // flip a manual user to the bundled engine — only a file whose
        // `engine.mode` says exactly "builtin" keeps builtin.
        let temp = TempDir::new().expect("tempdir");
        for (index, (engine_key, expected)) in [
            (r#""engine":{"mode":"manual"}"#, EngineMode::Manual),
            // No mode at all, or an unknown mode: not exactly "builtin".
            (r#""engine":{}"#, EngineMode::Manual),
            (r#""engine":{"mode":"tensor-future"}"#, EngineMode::Manual),
            // No engine key: legacy, manual.
            ("", EngineMode::Manual),
            (r#""engine":null"#, EngineMode::Manual),
            // An explicit builtin survives the fallback.
            (r#""engine":{"mode":"builtin"}"#, EngineMode::Builtin),
        ]
        .into_iter()
        .enumerate()
        {
            let path = temp.path().join(format!("settings-{index}.json"));
            let body = if engine_key.is_empty() {
                r#""endpoint":42,"model":"parakeet","expectedTerms":["auth"]"#.to_string()
            } else {
                format!(
                    r#""endpoint":42,"model":"parakeet","expectedTerms":["auth"],{engine_key}"#
                )
            };
            std::fs::write(&path, format!("{{{body}}}"))
                .expect("write unreadable settings");
            let settings = Settings::load(&path);
            assert_eq!(settings.engine.mode, expected, "for engine key {engine_key}");
            // The rest of the fallback is the documented default shape.
            assert_eq!(settings.model, "parakeet");
        }
    }

    #[test]
    fn an_unreadable_file_keeps_its_readable_engine_fields() {
        // #366: a typed-deserialization failure elsewhere in the file must
        // not make the engine forget which model it serves or the CPU pin.
        let temp = TempDir::new().expect("tempdir");
        let path = temp.path().join("settings.json");
        std::fs::write(
            &path,
            r#"{"endpoint":42,"model":"parakeet","expectedTerms":["auth"],"engine":{"mode":"builtin","activeModel":"parakeet-tdt-0.6b-v3","backendOverride":"cpu"}}"#,
        )
        .expect("write unreadable settings");

        let settings = Settings::load(&path);

        assert_eq!(settings.engine.mode, EngineMode::Builtin);
        assert_eq!(settings.engine.active_model.as_deref(), Some("parakeet-tdt-0.6b-v3"));
        assert_eq!(settings.engine.backend_override.as_deref(), Some("cpu"));
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
    fn an_unknown_engine_mode_loads_as_manual_keeping_the_rest() {
        // A settings file written by a newer build may name an engine mode
        // this build does not know. The file must still load — losing every
        // setting over one new key would be far worse — and the unknown
        // mode resolves to manual, the server this build can always talk
        // to instead of guessing at a bundled engine it cannot select.
        let temp = TempDir::new().expect("tempdir");
        let path = temp.path().join("settings.json");
        std::fs::write(
            &path,
            r#"{"endpoint":"http://10.0.0.5:8181","model":"whisper-large-v3","expectedTerms":["auth"],"userSetModel":true,"engine":{"mode":"tensor-future","activeModel":"parakeet-v3-q8","backendOverride":"cpu"}}"#,
        )
        .expect("write settings");
        let settings = Settings::load(&path);
        assert_eq!(settings.endpoint, "http://10.0.0.5:8181");
        assert_eq!(settings.model, "whisper-large-v3");
        assert!(settings.user_set_model);
        assert_eq!(settings.engine.mode, EngineMode::Manual);
        assert_eq!(settings.engine.active_model, Some("parakeet-v3-q8".to_string()));
        assert_eq!(settings.engine.backend_override, Some("cpu".to_string()));
    }

    #[test]
    fn an_engine_object_without_mode_loads_the_default_mode() {
        // An `engine` object that states no `mode` is not a legacy file
        // (the key exists): it loads with the default mode, builtin, and
        // the fields it does state are kept — not the whole-file defaults.
        let temp = TempDir::new().expect("tempdir");
        let path = temp.path().join("settings.json");
        std::fs::write(
            &path,
            r#"{"endpoint":"http://10.0.0.5:8181","model":"whisper-large-v3","expectedTerms":["auth"],"engine":{"activeModel":"parakeet-v3-q8"}}"#,
        )
        .expect("write settings");
        let settings = Settings::load(&path);
        assert_eq!(settings.endpoint, "http://10.0.0.5:8181");
        assert_eq!(settings.model, "whisper-large-v3");
        assert_eq!(settings.engine.mode, EngineMode::Builtin);
        assert_eq!(settings.engine.active_model, Some("parakeet-v3-q8".to_string()));
        assert_eq!(settings.engine.backend_override, None);
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

        // Not JSON at all: nothing is knowable, pure defaults.
        std::fs::write(&path, "{not json at all").expect("write corrupt settings");
        assert_eq!(Settings::load(&path), Settings::default_settings());

        // Wrong field types are corrupt too, not a panic — but the file
        // carries no engine key, so it is legacy (#366): the fallback
        // keeps the manual engine mode instead of flipping a hand-run-
        // server user to the bundled engine. Everything else stays the
        // documented defaults.
        std::fs::write(&path, r#"{"endpoint":42}"#)
            .expect("write mistyped settings");
        let mut expected = Settings::default_settings();
        expected.engine.mode = EngineMode::Manual;
        assert_eq!(Settings::load(&path), expected);
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

    #[test]
    fn write_atomic_sweeps_orphaned_tmp_files_from_crashed_writers() {
        let temp = TempDir::new().expect("tempdir");
        let path = temp.path().join("settings.json");
        // An orphan from a writer that died mid-save hours ago, a fresh
        // temp that could belong to a concurrent live writer, and files
        // the pattern does not own.
        let orphan = temp.path().join(format!(
            "settings.json.{}.{}.tmp",
            std::process::id(),
            4_242
        ));
        std::fs::write(&orphan, b"partial").expect("write orphan");
        let old = std::time::SystemTime::now() - std::time::Duration::from_secs(2 * 60 * 60);
        std::fs::OpenOptions::new()
            .write(true)
            .open(&orphan)
            .expect("open orphan")
            .set_modified(old)
            .expect("age the orphan");
        let fresh = temp.path().join("settings.json.999999.7.tmp");
        std::fs::write(&fresh, b"partial").expect("write fresh temp");
        let unrelated = temp.path().join("other.json.1.1.tmp");
        std::fs::write(&unrelated, b"partial").expect("write unrelated");
        let foreign = temp.path().join("settings.json.backup.tmp");
        std::fs::write(&foreign, b"backup").expect("write foreign temp");
        std::fs::OpenOptions::new()
            .write(true)
            .open(&foreign)
            .expect("open foreign temp")
            .set_modified(old)
            .expect("age the foreign temp");

        write_atomic(&path, b"{}").expect("write settings");

        assert!(!orphan.exists(), "the aged orphan is swept");
        assert!(fresh.exists(), "a fresh temp may be a live writer's");
        assert!(unrelated.exists(), "other files' temps are not ours");
        assert!(foreign.exists(), "only the <pid>.<seq> staging shape is ours");
        assert_eq!(std::fs::read(&path).expect("read settings"), b"{}");
    }

    #[test]
    fn write_atomic_leaves_a_failed_writes_own_tmp_behind_only_on_error() {
        // The normal path stages and renames one temp; nothing matching
        // the pattern survives a successful save.
        let temp = TempDir::new().expect("tempdir");
        let path = temp.path().join("settings.json");
        write_atomic(&path, b"{}").expect("write settings");
        let leftovers: Vec<_> = std::fs::read_dir(temp.path())
            .expect("read dir")
            .map(|entry| entry.expect("entry").file_name().to_string_lossy().to_string())
            .collect();
        assert_eq!(leftovers, vec!["settings.json".to_string()]);
    }
}
