//! Persistent app settings, replacing the `localStorage` keys used by
//! `apps/desktop/src/App.tsx` (`starling:endpoint`, `starling:protocol`,
//! `starling:model`, `starling:terms`) with a JSON file. See
//! `apps/desktop-gpui/PORT.md`.

use std::io;
use std::path::{Path, PathBuf};

/// The transcription backend wire protocol. One enum for the whole crate
/// (R11): the client routes requests by it, the settings file persists it
/// (`"starling"` / `"openai"`, lowercase, unchanged on disk).
pub use crate::client::Protocol;

/// The platform config directory could not be resolved (on Linux,
/// `$XDG_CONFIG_HOME` and `$HOME` are both unset). Returned instead of
/// silently reading/writing settings in an arbitrary working directory.
#[derive(Debug, thiserror::Error)]
#[error("could not resolve the user config directory; set XDG_CONFIG_HOME or HOME")]
pub struct ConfigDirUnavailable;

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StorageBackend {
    /// The v1 file session store (`sessions/` + `journals/`). Default.
    V1,
    /// Storage v2 (SQLite `starling.db` + `audio/` sample journals). Set
    /// only through the reviewed migration flow (dry-run → verified
    /// import → cutover) or the `STARLING_STORAGE_V2` testing flag.
    V2,
}

impl Default for StorageBackend {
    fn default() -> Self {
        StorageBackend::V1
    }
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Settings {
    pub endpoint: String,
    pub protocol: Protocol,
    pub model: String,
    pub expected_terms: Vec<String>,
    /// Whether the user explicitly chose the model (R02). While false, the
    /// app may sync `model` from the server's health response. Files written
    /// before this field existed deserialize it as `false` (`serde(default)`),
    /// which re-enables that sync instead of guessing from the file's text.
    #[serde(default)]
    pub user_set_model: bool,
    /// Which storage backend the next start uses (E02 cutover). Additive:
    /// files written before the cutover deserialize it as `V1`, so the
    /// store of record never changes behind an old config's back. The
    /// `STARLING_STORAGE_V2` environment flag still wins over it while set.
    #[serde(default)]
    pub storage_backend: StorageBackend,
}

impl Settings {
    /// Defaults mirroring `App.tsx`: `DEFAULT_ENDPOINT`, the starling protocol,
    /// the `parakeet` model, and the "auth" expected-terms input.
    pub fn default_settings() -> Self {
        Self {
            endpoint: "http://127.0.0.1:8181".to_string(),
            protocol: Protocol::Starling,
            model: "parakeet".to_string(),
            expected_terms: vec!["auth".to_string()],
            user_set_model: false,
            storage_backend: StorageBackend::V1,
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
    pub fn load(path: &Path) -> Self {
        std::fs::read_to_string(path)
            .ok()
            .and_then(|text| serde_json::from_str(&text).ok())
            .unwrap_or_else(Self::default_settings)
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
        assert_eq!(settings.protocol, Protocol::Starling);
        assert_eq!(settings.model, "parakeet");
        assert_eq!(settings.expected_terms, vec!["auth".to_string()]);
        assert_eq!(settings.expected_terms_input(), "auth");
        assert!(!settings.user_set_model);
        assert_eq!(settings.storage_backend, StorageBackend::V1);
    }

    #[test]
    fn protocol_serializes_lowercase() {
        assert_eq!(
            serde_json::to_string(&Protocol::Starling).unwrap(),
            "\"starling\""
        );
        assert_eq!(
            serde_json::to_string(&Protocol::OpenAi).unwrap(),
            "\"openai\""
        );

        let starling: Protocol = serde_json::from_str("\"starling\"").unwrap();
        let openai: Protocol = serde_json::from_str("\"openai\"").unwrap();
        assert_eq!(starling, Protocol::Starling);
        assert_eq!(openai, Protocol::OpenAi);
    }

    #[test]
    fn save_and_load_roundtrip() {
        let temp = TempDir::new().expect("tempdir");
        let path = temp.path().join("nested").join("settings.json");

        let settings = Settings {
            endpoint: "http://127.0.0.1:9999/".to_string(),
            protocol: Protocol::OpenAi,
            model: "whisper-large-v3".to_string(),
            expected_terms: vec!["auth".to_string(), "Starling".to_string()],
            user_set_model: true,
            storage_backend: StorageBackend::V2,
        };

        settings.save(&path).expect("save");
        assert_eq!(Settings::load(&path), settings);

        // camelCase keys on disk, protocol lowercase.
        let raw = std::fs::read_to_string(&path).expect("read settings file");
        let value: serde_json::Value = serde_json::from_str(&raw).expect("parse settings");
        assert_eq!(value["endpoint"], "http://127.0.0.1:9999/");
        assert_eq!(value["protocol"], "openai");
        assert_eq!(value["model"], "whisper-large-v3");
        assert_eq!(
            value["expectedTerms"],
            serde_json::json!(["auth", "Starling"])
        );
        assert_eq!(value["userSetModel"], true);
        assert_eq!(value["storageBackend"], "v2");
    }

    #[test]
    fn the_storage_backend_choice_round_trips_and_legacy_files_stay_v1() {
        let temp = TempDir::new().expect("tempdir");
        let path = temp.path().join("settings.json");

        // A file written before the cutover: no storageBackend key, and the
        // store of record must stay v1 rather than being guessed from text.
        std::fs::write(
            &path,
            r#"{"endpoint":"http://127.0.0.1:8181","protocol":"starling","model":"parakeet","expectedTerms":["auth"]}"#,
        )
        .expect("write legacy settings");
        let settings = Settings::load(&path);
        assert_eq!(settings.storage_backend, StorageBackend::V1);

        // An explicit cutover persists and reloads.
        let mut settings = settings;
        settings.storage_backend = StorageBackend::V2;
        settings.save(&path).expect("save");
        assert_eq!(Settings::load(&path).storage_backend, StorageBackend::V2);
        let raw = std::fs::read_to_string(&path).expect("read");
        assert!(raw.contains("\"storageBackend\": \"v2\""), "{raw}");

        // A corrupt value falls back to defaults, never to v2.
        std::fs::write(
            &path,
            r#"{"endpoint":"http://127.0.0.1:8181","protocol":"starling","model":"parakeet","expectedTerms":["auth"],"storageBackend":"v3"}"#,
        )
        .expect("write unknown backend");
        assert_eq!(
            Settings::load(&path).storage_backend,
            StorageBackend::V1,
            "an unknown value is a corrupt file: defaults, not a guess"
        );
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
            r#"{"endpoint":"http://10.0.0.5:8181","protocol":"openai","model":"whisper-large-v3","expectedTerms":["auth"]}"#,
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
        std::fs::write(&path, r#"{"endpoint":42,"protocol":"bogus"}"#)
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
