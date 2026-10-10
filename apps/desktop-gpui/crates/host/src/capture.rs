//! The production capture source (#220): the recorder the app used to
//! open itself, opened by the host the way the app did — on the user's
//! preferred microphone (read from the settings file at every start, so
//! a choice saved in the app applies to the next take), journaled, with
//! the journal disk watched (#342). A disk too full to keep a take
//! refuses it before the microphone opens.
//!
//! A refusal's detail is a [`StartFailure`] in JSON, so the take feed can
//! hand the app the classified input problem behind it (the app's error
//! banner offers that problem's recovery actions); the `capture.error`
//! event itself stays the v1 `device_open_failed`.

use std::path::{Path, PathBuf};

use starling_dictation::disk::{DiskLevel, DiskWatch};
use starling_dictation::recorder::{self, CaptureRequest};
use starling_dictation::settings::Settings;
use starling_runtime::machine::capture::{recorder_session, CaptureSession, CaptureSource};

use crate::takes::StartFailure;

/// The device rate free-space estimates assume before a take opened one.
const PREFERRED_RATE_FOR_ESTIMATES: u32 = 48_000;

pub struct SettingsCaptureSource {
    /// The desktop settings file; `None` follows the system default
    /// microphone.
    settings_path: Option<PathBuf>,
}

impl SettingsCaptureSource {
    pub fn new(settings_path: Option<PathBuf>) -> SettingsCaptureSource {
        SettingsCaptureSource { settings_path }
    }

    fn preferred_device(&self) -> Option<String> {
        let path = self.settings_path.as_ref()?;
        Settings::load(path).microphone.preferred_device
    }
}

fn refusal(problem: Option<starling_dictation::microphone::InputProblem>, message: String) -> String {
    serde_json::to_string(&StartFailure { problem, message: message.clone() }).unwrap_or(message)
}

impl CaptureSource for SettingsCaptureSource {
    fn start(&self, journals_dir: &Path, _policy: &str) -> Result<Box<dyn CaptureSession>, String> {
        let disk_watch = DiskWatch::system();
        // A probe that cannot answer never blocks recording.
        if let Ok(reading) = disk_watch.policy.check(disk_watch.probe.as_ref(), journals_dir) {
            if reading.level == DiskLevel::Critical {
                let message = disk_watch
                    .policy
                    .warning(reading, PREFERRED_RATE_FOR_ESTIMATES)
                    .unwrap_or_else(|| "The disk is too full to record.".to_string());
                return Err(refusal(None, message));
            }
        }
        let preferred = self.preferred_device();
        recorder::start_capture(CaptureRequest {
            journals_dir: Some(journals_dir),
            preferred_device: preferred.as_deref(),
            disk_watch: Some(disk_watch),
        })
        .map(recorder_session)
        .map_err(|err| {
            let message = format!("{} {}", err.problem.message(), err.problem.recovery());
            refusal(Some(err.problem), message)
        })
    }
}
