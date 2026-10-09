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
    /// System-wide dictation controls (#221): the recording shortcut and
    /// how it activates. The subsection is read field by field, so a
    /// value this build cannot read costs only its own field — never the
    /// rest of the subsection, and never the rest of the file.
    #[serde(default, deserialize_with = "lenient_dictation")]
    pub dictation: DictationSettings,
    /// The microphone choice. A file without the key follows the system
    /// default, which is what every earlier build recorded from. An
    /// unreadable value resets only this choice, never the rest of the
    /// file.
    #[serde(default, deserialize_with = "lenient_microphone")]
    pub microphone: MicrophoneSettings,
    /// Playback during recording (#361). An unreadable subsection loads
    /// its defaults (off) and never the rest of the file.
    #[serde(default, deserialize_with = "lenient_playback")]
    pub playback: PlaybackSettings,
}

/// Which microphone takes record from. Only the user changes this: a take
/// whose preferred device is missing falls back to the system default for
/// that take alone (see `crate::microphone`).
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct MicrophoneSettings {
    /// The preferred input device by name; `None` follows the system
    /// default input.
    pub preferred_device: Option<String>,
}

/// How the recording shortcut starts and stops a take (#221).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ActivationMode {
    /// Push-to-talk: recording runs while the shortcut is held.
    Hold,
    /// Each press starts or stops recording; release does nothing.
    Toggle,
    /// A short tap latches recording on (the next press stops it); holding
    /// the shortcut past the tap limit records until release. Also what
    /// an unknown value (a file from a newer build) loads as.
    #[default]
    #[serde(other)]
    HoldOrToggle,
}

/// The dictation subsection of the settings file (#221). Every field
/// has a default and is read on its own, so a partial object loads with
/// defaults for the rest — and one unreadable value costs only its own
/// field, never its readable neighbours.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct DictationSettings {
    /// The recording shortcut in `global-hotkey` notation
    /// (`CmdOrCtrl+Shift+Space`, `F9`, `Alt+D`). Validated by the app,
    /// which keeps the default when the stored text cannot be used.
    pub shortcut: String,
    pub activation: ActivationMode,
    /// Hold mode only: a quick double tap latches the take hands-free
    /// until the next press.
    pub double_tap_hands_free: bool,
}

/// The shortcut a fresh install uses: the one every earlier build had.
pub const DEFAULT_SHORTCUT: &str = "CmdOrCtrl+Shift+Space";

/// What happens to system playback while a recording runs. An unknown
/// value (a file from a newer build) loads as `Off`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PlaybackMode {
    Lower,
    Mute,
    #[default]
    #[serde(other)]
    Off,
}

/// The playback subsection of the settings file. Attenuation is opt-in:
/// a file without the key loads `Off`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct PlaybackSettings {
    pub during_recording: PlaybackMode,
    /// `Lower` mode's target volume in percent.
    #[serde(deserialize_with = "clamped_percent")]
    pub lower_level_percent: u8,
}

/// A hand-edited level outside 0..=100 (or a fraction) is clamped rather
/// than failing the whole settings document.
fn clamped_percent<'de, D>(deserializer: D) -> Result<u8, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let raw = <f64 as serde::Deserialize>::deserialize(deserializer)?;
    Ok(raw.round().clamp(0.0, 100.0) as u8)
}

fn lenient_playback<'de, D>(deserializer: D) -> Result<PlaybackSettings, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = <serde_json::Value as serde::Deserialize>::deserialize(deserializer)?;
    Ok(serde_json::from_value(value).unwrap_or_else(|err| {
        eprintln!("Unreadable playback settings; using the defaults: {err}");
        PlaybackSettings::default()
    }))
}

impl Default for PlaybackSettings {
    fn default() -> Self {
        Self {
            during_recording: PlaybackMode::Off,
            lower_level_percent: 30,
        }
    }
}

impl PlaybackSettings {
    /// The `Lower` target, never above 100% whatever the field holds.
    pub fn effective_lower_percent(&self) -> u32 {
        self.lower_level_percent.min(100).into()
    }
}

impl Default for DictationSettings {
    fn default() -> Self {
        Self {
            shortcut: DEFAULT_SHORTCUT.to_string(),
            activation: ActivationMode::default(),
            double_tap_hands_free: false,
        }
    }
}

/// Reads the `dictation` value without ever failing the whole file, and
/// without ever discarding one field's bad value along with its readable
/// ones: each field is read on its own ([`lenient_dictation_field`]), so
/// `{"shortcut": 7, "activation": "hold"}` keeps the activation. An
/// unknown `activation` string still resolves to `HoldOrToggle`
/// (`serde(other)`), and a non-object (`5`, `null`) is all-defaults —
/// with the reason logged, so a shortcut that "disappeared" is
/// explainable.
fn lenient_dictation<'de, D>(deserializer: D) -> Result<DictationSettings, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = <serde_json::Value as serde::Deserialize>::deserialize(deserializer)?;
    Ok(lenient_dictation_value(value))
}

/// One field of the `dictation` object, read on its own: a value this
/// build cannot read (a wrong type) costs only that field, which falls
/// back to `default` with the reason logged — the file's other choices
/// survive. A missing key is simply the default, silently, exactly as
/// `serde(default)` would have it.
fn lenient_dictation_field<T: serde::de::DeserializeOwned>(
    dictation: &serde_json::Value,
    key: &str,
    default: impl FnOnce() -> T,
) -> T {
    let Some(value) = dictation.get(key) else {
        return default();
    };
    match serde_json::from_value(value.clone()) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("Unreadable dictation field `{key}`; using its default: {err}");
            default()
        }
    }
}

/// The field-wise leniency itself, shared by the `dictation` attribute
/// and the [`Settings::load`] fallback below, so both paths read the
/// subsection the same way.
fn lenient_dictation_value(value: serde_json::Value) -> DictationSettings {
    if !value.is_object() {
        eprintln!("Unreadable dictation settings ({value}); using the defaults");
        return DictationSettings::default();
    }
    DictationSettings {
        shortcut: lenient_dictation_field(&value, "shortcut", || DEFAULT_SHORTCUT.to_string()),
        activation: lenient_dictation_field(&value, "activation", ActivationMode::default),
        double_tap_hands_free: lenient_dictation_field(&value, "doubleTapHandsFree", bool::default),
    }
}

/// Reads the `microphone` value without ever failing the whole file: an
/// unreadable value follows the system default, with the reason logged.
fn lenient_microphone<'de, D>(deserializer: D) -> Result<MicrophoneSettings, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = <serde_json::Value as serde::Deserialize>::deserialize(deserializer)?;
    Ok(lenient_microphone_value(value))
}

fn lenient_microphone_value(value: serde_json::Value) -> MicrophoneSettings {
    if value.is_object() {
        if let Ok(microphone) = serde_json::from_value(value.clone()) {
            return microphone;
        }
    }
    eprintln!("Unreadable microphone settings ({value}); following the system default");
    MicrophoneSettings::default()
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
            dictation: DictationSettings::default(),
            microphone: MicrophoneSettings::default(),
            playback: PlaybackSettings::default(),
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
    /// The post-read logic lives in [`Settings::from_json_bytes`] (the
    /// host's settings watcher reuses it on bytes it already holds).
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
        let Ok(text) = std::fs::read_to_string(path) else {
            return Self::default_settings();
        };
        Self::from_json_bytes(text.as_bytes()).unwrap_or_else(Self::default_settings)
    }

    /// The settings `bytes` state, or `None` when they are not valid
    /// JSON (an empty or truncated file — what a watcher sees while a
    /// writer is mid-atomic-write, and must not read as "the user chose
    /// the defaults"). Otherwise exactly what [`Settings::load`]
    /// derives from a file it read: the legacy-file and
    /// typed-deserialization-fallback rules documented there, applied
    /// once here so both callers share them.
    pub fn from_json_bytes(bytes: &[u8]) -> Option<Settings> {
        let mut value = serde_json::from_slice::<serde_json::Value>(bytes).ok()?;
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
        // The dictation subsection reads leniently on its own (#221), so
        // an unreadable sibling key does not reset the user's shortcut.
        // The subtree is taken before `value` moves into `from_value` and
        // parsed only inside its failure branch: a load that succeeds
        // relies on the field-level `lenient_dictation` alone, instead of
        // parsing (and logging) the subsection twice.
        let dictation_subtree = value.get("dictation").cloned();
        // Likewise the microphone choice: an unreadable sibling key must
        // not silently move recording to another device.
        let microphone_subtree = value.get("microphone").cloned();
        let Ok(mut settings) = serde_json::from_value::<Settings>(value) else {
            // The same field-wise leniency as the `dictation` attribute:
            // an unreadable sibling key must not cost the user their
            // shortcut — nor a bad shortcut value the rest of the
            // subsection.
            let dictation = dictation_subtree
                .map(lenient_dictation_value)
                .unwrap_or_default();
            let mut fallback = Self::default_settings();
            fallback.engine = EngineSettings {
                mode: safe_mode,
                active_model,
                backend_override,
            };
            fallback.microphone = microphone_subtree
                .map(lenient_microphone_value)
                .unwrap_or_default();
            fallback.dictation = dictation;
            return Some(fallback);
        };
        if legacy_file {
            settings.engine.mode = EngineMode::Manual;
        }
        Some(settings)
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
            dictation: DictationSettings {
                shortcut: "F9".to_string(),
                activation: ActivationMode::Hold,
                double_tap_hands_free: true,
            },
            microphone: MicrophoneSettings {
                preferred_device: Some("USB Mic".to_string()),
            },
            playback: PlaybackSettings::default(),
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
        assert_eq!(value["microphone"]["preferredDevice"], "USB Mic");
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
    fn from_json_bytes_rejects_what_is_not_json_and_parses_what_is() {
        // The watcher's read path (#220): an empty or truncated file —
        // what a mid-atomic-write looks like — is `None` ("the file
        // states nothing"), never the defaults.
        assert_eq!(Settings::from_json_bytes(b""), None);
        assert_eq!(Settings::from_json_bytes(b"{\"engine\":{\"mo"), None);
        assert_eq!(Settings::from_json_bytes(b"not json at all"), None);

        // Valid JSON parses exactly like `load` would: a full manual
        // file round-trips, and the legacy rules (no engine key →
        // manual; typed failure → the safe mode) apply unchanged.
        let manual = r#"{"endpoint":"http://10.0.0.5:8181","model":"whisper-large-v3","expectedTerms":["auth"],"engine":{"mode":"manual"}}"#;
        let parsed = Settings::from_json_bytes(manual.as_bytes()).expect("valid JSON parses");
        let temp = TempDir::new().expect("tempdir");
        let path = temp.path().join("settings.json");
        std::fs::write(&path, manual).expect("write settings");
        assert_eq!(Settings::load(&path), parsed);

        let legacy = r#"{"endpoint":"http://10.0.0.5:8181","model":"whisper-large-v3","expectedTerms":["auth"]}"#;
        assert_eq!(
            Settings::from_json_bytes(legacy.as_bytes())
                .expect("legacy parses")
                .engine
                .mode,
            EngineMode::Manual
        );
        let unreadable = r#"{"endpoint":42,"model":"parakeet","expectedTerms":["auth"],"engine":{"mode":"builtin"}}"#;
        assert_eq!(
            Settings::from_json_bytes(unreadable.as_bytes())
                .expect("the fallback parses")
                .engine
                .mode,
            EngineMode::Builtin
        );
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
    fn a_file_without_a_microphone_key_follows_the_system_default() {
        let temp = TempDir::new().expect("tempdir");
        let path = temp.path().join("settings.json");
        std::fs::write(
            &path,
            r#"{"endpoint":"http://127.0.0.1:8181","model":"parakeet","expectedTerms":["auth"],"engine":{"mode":"builtin"}}"#,
        )
        .expect("write settings");
        assert_eq!(Settings::load(&path).microphone.preferred_device, None);
    }

    #[test]
    fn an_unreadable_file_keeps_the_preferred_microphone() {
        let temp = TempDir::new().expect("tempdir");
        let path = temp.path().join("settings.json");
        std::fs::write(
            &path,
            r#"{"endpoint":42,"model":"parakeet","expectedTerms":["auth"],"engine":{"mode":"builtin"},"microphone":{"preferredDevice":"USB Mic"}}"#,
        )
        .expect("write unreadable settings");
        assert_eq!(
            Settings::load(&path).microphone.preferred_device.as_deref(),
            Some("USB Mic")
        );
    }

    #[test]
    fn an_unreadable_microphone_key_resets_only_the_microphone_choice() {
        let temp = TempDir::new().expect("tempdir");
        let path = temp.path().join("settings.json");
        for microphone in [
            serde_json::json!(null),
            serde_json::json!(5),
            serde_json::json!({ "preferredDevice": 42 }),
            serde_json::json!(["USB Mic"]),
        ] {
            let raw = serde_json::json!({
                "endpoint": "http://10.0.0.9:8181",
                "model": "whisper-large-v3",
                "expectedTerms": ["auth", "Starling"],
                "engine": { "mode": "builtin" },
                "dictation": { "shortcut": "F9", "activation": "hold" },
                "microphone": microphone,
            });
            std::fs::write(&path, raw.to_string()).expect("write");
            let loaded = Settings::load(&path);
            assert_eq!(loaded.endpoint, "http://10.0.0.9:8181", "{microphone}");
            assert_eq!(loaded.model, "whisper-large-v3", "{microphone}");
            assert_eq!(
                loaded.expected_terms,
                vec!["auth".to_string(), "Starling".to_string()],
                "{microphone}"
            );
            assert_eq!(loaded.dictation.shortcut, "F9", "{microphone}");
            assert_eq!(
                loaded.dictation.activation,
                ActivationMode::Hold,
                "{microphone}"
            );
            assert_eq!(
                loaded.microphone,
                MicrophoneSettings::default(),
                "{microphone}"
            );
        }
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
    fn dictation_defaults_keep_the_historic_shortcut() {
        let settings = Settings::default_settings();
        assert_eq!(settings.dictation.shortcut, "CmdOrCtrl+Shift+Space");
        assert_eq!(settings.dictation.activation, ActivationMode::HoldOrToggle);
        assert!(!settings.dictation.double_tap_hands_free);
    }

    #[test]
    fn playback_settings_load_leniently() {
        let temp = TempDir::new().expect("tempdir");
        let path = temp.path().join("settings.json");
        let load = |playback: &str| {
            std::fs::write(
                &path,
                format!(
                    r#"{{"endpoint":"http://10.0.0.5:8181","model":"m","expectedTerms":[]{playback}}}"#
                ),
            )
            .expect("write settings");
            Settings::load(&path)
        };

        // A file from before the key existed: off.
        assert_eq!(load("").playback, PlaybackSettings::default());
        assert_eq!(PlaybackSettings::default().during_recording, PlaybackMode::Off);
        // Partial object: defaults for the rest.
        assert_eq!(
            load(r#","playback":{"duringRecording":"mute"}"#).playback,
            PlaybackSettings {
                during_recording: PlaybackMode::Mute,
                lower_level_percent: 30,
            }
        );
        // An unknown mode is off, never a silent mute.
        let unknown = load(r#","playback":{"duringRecording":"duck","lowerLevelPercent":45}"#);
        assert_eq!(unknown.playback.during_recording, PlaybackMode::Off);
        assert_eq!(unknown.playback.lower_level_percent, 45);
        // Out-of-range levels clamp without resetting anything else.
        let high = load(r#","playback":{"duringRecording":"lower","lowerLevelPercent":300}"#);
        assert_eq!(high.endpoint, "http://10.0.0.5:8181");
        assert_eq!(high.playback.lower_level_percent, 100);
        let low = load(r#","playback":{"lowerLevelPercent":-20}"#);
        assert_eq!(low.playback.lower_level_percent, 0);
        let fraction = load(r#","playback":{"lowerLevelPercent":55.5}"#);
        assert_eq!(fraction.playback.lower_level_percent, 56);
        // An unreadable subsection is off, and costs nothing else.
        for playback in [
            r#","playback":null"#,
            r#","playback":{"duringRecording":"mute","lowerLevelPercent":"30%"}"#,
        ] {
            let loaded = load(playback);
            assert_eq!(loaded.playback, PlaybackSettings::default(), "{playback}");
            assert_eq!(loaded.endpoint, "http://10.0.0.5:8181", "{playback}");
            assert_eq!(loaded.model, "m", "{playback}");
        }
    }

    #[test]
    fn playback_settings_roundtrip() {
        let temp = TempDir::new().expect("tempdir");
        let path = temp.path().join("settings.json");
        let settings = Settings {
            playback: PlaybackSettings {
                during_recording: PlaybackMode::Lower,
                lower_level_percent: 45,
            },
            ..Settings::default_settings()
        };
        settings.save(&path).expect("save");
        let raw = std::fs::read_to_string(&path).expect("read");
        assert!(raw.contains("\"duringRecording\": \"lower\""), "{raw}");
        assert_eq!(Settings::load(&path), settings);
    }

    #[test]
    fn an_unreadable_dictation_key_loads_its_defaults_and_keeps_the_rest() {
        let temp = TempDir::new().expect("tempdir");
        let path = temp.path().join("settings.json");
        for dictation in [
            serde_json::json!(5),
            serde_json::json!(null),
            serde_json::json!({ "shortcut": 7 }),
        ] {
            let raw = serde_json::json!({
                "endpoint": "http://10.0.0.2:8181",
                "model": "m",
                "expectedTerms": [],
                "engine": { "mode": "manual" },
                "dictation": dictation,
            });
            std::fs::write(&path, raw.to_string()).expect("write");
            let loaded = Settings::load(&path);
            assert_eq!(loaded.endpoint, "http://10.0.0.2:8181", "{dictation}");
            assert_eq!(loaded.dictation, DictationSettings::default(), "{dictation}");
        }
    }

    #[test]
    fn a_partial_dictation_key_and_an_unknown_mode_load_leniently() {
        let temp = TempDir::new().expect("tempdir");
        let path = temp.path().join("settings.json");
        let raw = serde_json::json!({
            "endpoint": "http://127.0.0.1:8181",
            "model": "m",
            "expectedTerms": [],
            "engine": { "mode": "builtin" },
            "dictation": { "shortcut": "F9", "activation": "chord-of-the-future" },
        });
        std::fs::write(&path, raw.to_string()).expect("write");
        let loaded = Settings::load(&path).dictation;
        assert_eq!(loaded.shortcut, "F9");
        assert_eq!(loaded.activation, ActivationMode::HoldOrToggle);
        assert!(!loaded.double_tap_hands_free);
    }

    #[test]
    fn an_unreadable_dictation_field_costs_only_that_field() {
        // One unreadable value must cost only its own field: the readable
        // choices beside it survive, and the bad one falls back to its
        // default with the reason logged. An unknown `activation` string
        // is not even unreadable — `serde(other)` resolves it to
        // HoldOrToggle — so a wrong *type* is what exercises the fallback.
        let temp = TempDir::new().expect("tempdir");
        let cases = [
            (
                serde_json::json!({ "shortcut": 7, "activation": "hold", "doubleTapHandsFree": true }),
                DictationSettings {
                    shortcut: DEFAULT_SHORTCUT.to_string(),
                    activation: ActivationMode::Hold,
                    double_tap_hands_free: true,
                },
            ),
            (
                serde_json::json!({ "shortcut": "F9", "activation": 3, "doubleTapHandsFree": true }),
                DictationSettings {
                    shortcut: "F9".to_string(),
                    activation: ActivationMode::HoldOrToggle,
                    double_tap_hands_free: true,
                },
            ),
        ];
        for (index, (dictation, expected)) in cases.into_iter().enumerate() {
            let path = temp.path().join(format!("settings-{index}.json"));
            let raw = serde_json::json!({
                "endpoint": "http://127.0.0.1:8181",
                "model": "m",
                "expectedTerms": [],
                "engine": { "mode": "builtin" },
                "dictation": dictation,
            });
            std::fs::write(&path, raw.to_string()).expect("write");
            assert_eq!(Settings::load(&path).dictation, expected, "for {dictation}");
        }
        // The `Settings::load` fallback path (typed deserialization
        // fails elsewhere in the file) reads the subsection through the
        // same field-wise helper, so the same split holds there.
        let path = temp.path().join("settings-fallback.json");
        std::fs::write(
            &path,
            r#"{"endpoint":12,"dictation":{"shortcut":7,"activation":"hold","doubleTapHandsFree":true}}"#,
        )
        .expect("write");
        let loaded = Settings::load(&path);
        assert_eq!(loaded.dictation.shortcut, DEFAULT_SHORTCUT);
        assert_eq!(loaded.dictation.activation, ActivationMode::Hold);
        assert!(loaded.dictation.double_tap_hands_free);
    }

    #[test]
    fn an_unreadable_sibling_key_keeps_the_dictation_choices() {
        let temp = TempDir::new().expect("tempdir");
        let path = temp.path().join("settings.json");
        let raw = serde_json::json!({
            "endpoint": 12,
            "engine": { "mode": "builtin" },
            "dictation": { "shortcut": "F9", "activation": "hold", "doubleTapHandsFree": true },
        });
        std::fs::write(&path, raw.to_string()).expect("write");
        let loaded = Settings::load(&path);
        assert_eq!(loaded.endpoint, Settings::default_settings().endpoint);
        assert_eq!(loaded.dictation.shortcut, "F9");
        assert_eq!(loaded.dictation.activation, ActivationMode::Hold);
        assert!(loaded.dictation.double_tap_hands_free);
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
