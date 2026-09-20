//! Persistent app settings, replacing the `localStorage` keys used by
//! `apps/desktop/src/App.tsx` (`starling:endpoint`, `starling:protocol`,
//! `starling:model`, `starling:terms`) with a JSON file. See
//! `apps/desktop-gpui/PORT.md`.

use std::io;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Protocol {
    Starling,
    OpenAI,
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
        }
    }

    /// `dirs::config_dir()/starling-gpui/settings.json`.
    pub fn default_path() -> PathBuf {
        dirs::config_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("starling-gpui")
            .join("settings.json")
    }

    /// Missing or corrupt file always falls back to the defaults.
    pub fn load_or_default() -> Self {
        Self::load(&Self::default_path())
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
    }

    #[test]
    fn protocol_serializes_lowercase() {
        assert_eq!(
            serde_json::to_string(&Protocol::Starling).unwrap(),
            "\"starling\""
        );
        assert_eq!(
            serde_json::to_string(&Protocol::OpenAI).unwrap(),
            "\"openai\""
        );

        let starling: Protocol = serde_json::from_str("\"starling\"").unwrap();
        let openai: Protocol = serde_json::from_str("\"openai\"").unwrap();
        assert_eq!(starling, Protocol::Starling);
        assert_eq!(openai, Protocol::OpenAI);
    }

    #[test]
    fn save_and_load_roundtrip() {
        let temp = TempDir::new().expect("tempdir");
        let path = temp.path().join("nested").join("settings.json");

        let settings = Settings {
            endpoint: "http://127.0.0.1:9999/".to_string(),
            protocol: Protocol::OpenAI,
            model: "whisper-large-v3".to_string(),
            expected_terms: vec!["auth".to_string(), "Starling".to_string()],
            user_set_model: true,
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
