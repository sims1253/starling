//! Microphone selection: which capture device a take opens, why, and how
//! the capture path explains an input that is missing, refused, dead or
//! silent.
//!
//! The user's preference ("follow the system default" or one named device)
//! is only ever changed by the user. Each take resolves it once, at start,
//! into an [`InputRoute`]: the preferred device when it is listed,
//! otherwise the system default with a visible [`InputRoute::notice`]. A
//! running take is never moved to another device; one that disappears
//! mid-take surfaces as a fatal [`crate::recorder::RecorderFault`].
//!
//! Device identity is the host's device name: cpal 0.15 exposes no stable
//! device id, so two devices reporting the same name resolve to the first
//! one listed.

use cpal::traits::{DeviceTrait, HostTrait};

/// A peak (|sample|, 0..=1) below this over a whole capture counts as
/// digital silence: a muted, unplugged-but-open, or permission-blocked
/// input (macOS feeds zeros to apps without microphone permission) rather
/// than a quiet room, whose noise floor sits well above it (~-80 dBFS).
pub const SILENT_PEAK: f32 = 1e-4;

/// One capture device as the host lists it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InputDevice {
    /// The host's name for the device — also its identity (see the module
    /// docs).
    pub name: String,
    /// Whether the host currently reports it as the default input.
    pub is_default: bool,
}

/// Why a take opened the device it did.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum RouteReason {
    /// No preferred device: the take follows the system default.
    FollowingDefault,
    /// The preferred device was listed and opened.
    Preferred,
    /// The preferred device was not listed (unplugged, renamed): the take
    /// uses the system default. The preference itself is unchanged.
    PreferredMissing,
    /// The host could not list devices at all: the take uses the system
    /// default. The preference itself is unchanged.
    ListingFailed(String),
}

/// The input a take actually opened, beside the preference it was
/// resolved from.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct InputRoute {
    /// The preferred device name at the time the take started (`None`:
    /// follow the system default).
    pub preferred: Option<String>,
    /// The device actually capturing.
    pub device: String,
    pub reason: RouteReason,
}

impl InputRoute {
    /// Whether the physical microphone behind this route is chosen (and
    /// may be switched mid-take) by the sound server, not by Starling: on
    /// Linux the ALSA `default`/`pulse`/`pipewire`/`jack` PCMs hand capture
    /// to PulseAudio, PipeWire or JACK, which Starling cannot see or pin.
    pub fn follows_sound_server(&self) -> bool {
        cfg!(target_os = "linux")
            && matches!(
                self.device.as_str(),
                "default" | "pulse" | "pipewire" | "jack"
            )
    }

    /// Whether the take records from something other than the device the
    /// user asked for.
    pub fn is_fallback(&self) -> bool {
        matches!(
            self.reason,
            RouteReason::PreferredMissing | RouteReason::ListingFailed(_)
        )
    }

    /// The visible fallback notice, `None` when the take uses what the
    /// user asked for.
    pub fn notice(&self) -> Option<String> {
        let preferred = self.preferred.as_deref().unwrap_or("your microphone");
        match &self.reason {
            RouteReason::FollowingDefault | RouteReason::Preferred => None,
            RouteReason::PreferredMissing => Some(format!(
                "{preferred} is not connected, so this recording uses the system default \
                 input ({}). Your preferred microphone is kept and will be used again once it \
                 is back.",
                self.device
            )),
            RouteReason::ListingFailed(error) => Some(format!(
                "Could not list microphones ({error}), so this recording uses the system \
                 default input ({}). Your preferred microphone ({preferred}) is kept.",
                self.device
            )),
        }
    }
}

/// Lists the default host's capture devices, default first-flagged.
/// Devices whose name cannot be read are skipped (they cannot be selected
/// or remembered). Blocking: on ALSA, listing probes each PCM, so call it
/// off the UI thread.
pub fn list_input_devices() -> Result<Vec<InputDevice>, String> {
    let host = cpal::default_host();
    let default_name = host
        .default_input_device()
        .and_then(|device| device.name().ok());
    let devices = host
        .input_devices()
        .map_err(|err| err.to_string())?
        .filter_map(|device| device.name().ok())
        .map(|name| InputDevice {
            is_default: default_name.as_deref() == Some(name.as_str()),
            name,
        })
        .collect();
    Ok(devices)
}

/// Opens the device a preference resolves to now, with the route that
/// explains it. A missing preferred device, or a failed listing, falls
/// back to the system default.
pub(crate) fn open_input(
    preferred: Option<&str>,
) -> Result<(cpal::Device, InputRoute), InputProblem> {
    let host = cpal::default_host();
    let reason = match preferred {
        None => RouteReason::FollowingDefault,
        Some(name) => match host.input_devices() {
            Ok(mut devices) => {
                if let Some(device) =
                    devices.find(|device| device.name().is_ok_and(|listed| listed == name))
                {
                    let route = InputRoute {
                        preferred: Some(name.to_string()),
                        device: name.to_string(),
                        reason: RouteReason::Preferred,
                    };
                    return Ok((device, route));
                }
                RouteReason::PreferredMissing
            }
            Err(err) => RouteReason::ListingFailed(err.to_string()),
        },
    };
    let device = host
        .default_input_device()
        .ok_or_else(|| InputProblem::NoDevice {
            detail: match (&reason, preferred) {
                (RouteReason::PreferredMissing, Some(name)) => format!(
                    "{name} is not connected and there is no other microphone to fall back to."
                ),
                _ => "No microphone was found.".to_string(),
            },
        })?;
    let name = device
        .name()
        .unwrap_or_else(|_| "system default input".to_string());
    Ok((
        device,
        InputRoute {
            preferred: preferred.map(str::to_string),
            device: name,
            reason,
        },
    ))
}

/// Why the input cannot be used, in the terms the user can act on. Each
/// kind has its own explanation and recovery ([`InputProblem::message`],
/// [`InputProblem::recovery`]); they must never collapse into one generic
/// "microphone error".
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum InputProblem {
    /// There is no capture device to open.
    NoDevice { detail: String },
    /// The device exists but could not be opened right now (unplugged
    /// between listing and open, busy, or gone mid-take).
    Unavailable { device: String, detail: String },
    /// The operating system refused access to the microphone.
    PermissionDenied { device: String, detail: String },
    /// The device opened and delivered audio, but only digital silence.
    Silent { device: String },
    /// Anything else the backend reported (unsupported format, …).
    Failed { device: String, detail: String },
}

impl InputProblem {
    /// One sentence saying what is wrong.
    pub fn message(&self) -> String {
        match self {
            InputProblem::NoDevice { detail } => {
                format!("{detail} Connect a microphone and try again.")
            }
            InputProblem::Unavailable { device, detail } => {
                format!("{device} is unavailable: {detail}")
            }
            InputProblem::PermissionDenied { device, detail } => {
                format!("Microphone access to {device} was denied: {detail}")
            }
            InputProblem::Silent { device } => {
                format!("{device} is delivering silence — no sound at all reached Starling.")
            }
            InputProblem::Failed { device, detail } => {
                format!("{device} could not be used: {detail}")
            }
        }
    }

    /// What the user can do about it.
    pub fn recovery(&self) -> &'static str {
        match self {
            InputProblem::NoDevice { .. } => {
                "Plug in a microphone (or enable one in your sound settings), then Refresh."
            }
            InputProblem::Unavailable { .. } => {
                "Check that it is plugged in and not used exclusively by another app, then \
                 retry or pick another microphone."
            }
            InputProblem::PermissionDenied { .. } => {
                "Allow microphone access for Starling in your system privacy settings, then \
                 retry."
            }
            InputProblem::Silent { .. } => {
                "Check the device's mute switch and input level. Missing microphone permission \
                 can also look like this (macOS records silence instead of reporting an \
                 error), so check the privacy settings too."
            }
            InputProblem::Failed { .. } => "Pick another microphone, or retry.",
        }
    }

    /// The OS settings pages worth offering, most likely first. Silence
    /// gets both: a muted device and a missing permission look the same.
    pub fn settings_pages(&self) -> &'static [SettingsPage] {
        match self {
            InputProblem::PermissionDenied { .. } => &[SettingsPage::Privacy],
            InputProblem::Silent { .. } => &[SettingsPage::Sound, SettingsPage::Privacy],
            _ => &[SettingsPage::Sound],
        }
    }
}

/// An OS settings page a recovery action can open.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SettingsPage {
    /// Sound input devices and levels.
    Sound,
    /// Microphone access permission.
    Privacy,
}

impl std::fmt::Display for InputProblem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message())
    }
}

/// Classifies a backend error by its typed cpal kind first: only an
/// opaque `BackendSpecific` description is inspected for the access-denied
/// wording each OS uses (WASAPI `E_ACCESSDENIED`, `EACCES`/`EPERM`), and
/// only to choose the explanation shown — never to decide whether a take
/// survives (that is [`crate::recorder::RecorderFault::is_fatal`]).
pub(crate) fn classify_backend(device: &str, unavailable: bool, detail: String) -> InputProblem {
    let device = device.to_string();
    if unavailable {
        return InputProblem::Unavailable { device, detail };
    }
    let lower = detail.to_ascii_lowercase();
    let denied = [
        "access is denied",
        "access denied",
        "permission",
        "not permitted",
        "0x80070005",
    ]
    .iter()
    .any(|needle| lower.contains(needle));
    if denied {
        InputProblem::PermissionDenied { device, detail }
    } else {
        InputProblem::Failed { device, detail }
    }
}

/// The level of a capture as a whole, for the microphone test.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SignalLevel {
    /// Largest |sample| (0..=1).
    pub peak: f32,
    /// Root mean square (0..=1).
    pub rms: f32,
}

impl SignalLevel {
    /// Measures `samples`; non-finite samples are ignored.
    pub fn measure(samples: &[f32]) -> Self {
        let mut peak = 0.0f32;
        let mut sum = 0.0f64;
        let mut count = 0usize;
        for &sample in samples.iter().filter(|sample| sample.is_finite()) {
            peak = peak.max(sample.abs());
            sum += f64::from(sample) * f64::from(sample);
            count += 1;
        }
        let rms = if count == 0 {
            0.0
        } else {
            (sum / count as f64).sqrt() as f32
        };
        Self { peak, rms }
    }

    /// Digital silence (see [`SILENT_PEAK`]).
    pub fn is_silent(&self) -> bool {
        self.peak < SILENT_PEAK
    }

    /// The RMS level in dBFS, floored at -90 so silence stays printable.
    pub fn rms_dbfs(&self) -> f32 {
        (20.0 * self.rms.max(1e-9).log10()).max(-90.0)
    }
}

/// Maps a dBFS level to a 0..=1 meter fill (-60 dBFS empty, 0 dBFS full),
/// so speech lands mid-meter and room noise near the bottom.
pub fn meter_fill(dbfs: f32) -> f32 {
    ((dbfs + 60.0) / 60.0).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn route(device: &str, reason: RouteReason) -> InputRoute {
        InputRoute {
            preferred: Some("USB Mic".to_string()),
            device: device.to_string(),
            reason,
        }
    }

    #[test]
    fn fallback_routes_announce_themselves_and_keep_the_preference() {
        let missing = route("Laptop Mic", RouteReason::PreferredMissing);
        assert!(missing.is_fallback());
        let notice = missing.notice().expect("a fallback is announced");
        assert!(notice.contains("USB Mic"), "{notice}");
        assert!(notice.contains("Laptop Mic"), "{notice}");
        assert!(notice.contains("kept"), "{notice}");

        let unlisted = route(
            "default",
            RouteReason::ListingFailed("ALSA lib busy".to_string()),
        );
        assert!(unlisted.is_fallback());
        let notice = unlisted.notice().expect("a fallback is announced");
        assert!(notice.contains("ALSA lib busy"), "{notice}");
        assert!(notice.contains("USB Mic"), "{notice}");

        for reason in [RouteReason::FollowingDefault, RouteReason::Preferred] {
            let route = route("Laptop Mic", reason);
            assert!(!route.is_fallback());
            assert_eq!(route.notice(), None);
        }
    }

    #[test]
    fn backend_errors_classify_by_kind_then_by_denial_wording() {
        assert!(matches!(
            classify_backend("USB Mic", true, "gone".to_string()),
            InputProblem::Unavailable { .. }
        ));
        for denied in [
            "Access is denied. (0x80070005)",
            "snd_pcm_open: Permission denied",
            "Operation not permitted",
        ] {
            assert!(
                matches!(
                    classify_backend("Mic", false, denied.to_string()),
                    InputProblem::PermissionDenied { .. }
                ),
                "{denied}"
            );
        }
        assert!(matches!(
            classify_backend("Mic", false, "format not supported".to_string()),
            InputProblem::Failed { .. }
        ));
    }

    #[test]
    fn each_problem_has_its_own_message_and_recovery() {
        let problems = [
            InputProblem::NoDevice {
                detail: "No microphone was found.".to_string(),
            },
            InputProblem::Unavailable {
                device: "USB Mic".to_string(),
                detail: "gone".to_string(),
            },
            InputProblem::PermissionDenied {
                device: "USB Mic".to_string(),
                detail: "denied".to_string(),
            },
            InputProblem::Silent {
                device: "USB Mic".to_string(),
            },
            InputProblem::Failed {
                device: "USB Mic".to_string(),
                detail: "odd".to_string(),
            },
        ];
        let recoveries: std::collections::HashSet<_> =
            problems.iter().map(InputProblem::recovery).collect();
        assert_eq!(
            recoveries.len(),
            problems.len(),
            "recoveries must be distinct"
        );
        let messages: std::collections::HashSet<_> =
            problems.iter().map(InputProblem::message).collect();
        assert_eq!(messages.len(), problems.len(), "messages must be distinct");
        assert_eq!(problems[2].settings_pages(), &[SettingsPage::Privacy]);
        // Silence: a muted device or a missing permission — offer both.
        assert_eq!(
            problems[3].settings_pages(),
            &[SettingsPage::Sound, SettingsPage::Privacy]
        );
        assert_eq!(problems[1].settings_pages(), &[SettingsPage::Sound]);
    }

    #[test]
    fn sound_server_routes_are_recognised_on_linux_only() {
        let linux = cfg!(target_os = "linux");
        for name in ["default", "pulse", "pipewire", "jack"] {
            assert_eq!(
                route(name, RouteReason::FollowingDefault).follows_sound_server(),
                linux,
                "{name}"
            );
        }
        for name in [
            "hw:CARD=USB,DEV=0",
            "sysdefault:CARD=PCH",
            "USB Mic",
            "wslg_mic",
        ] {
            assert!(
                !route(name, RouteReason::FollowingDefault).follows_sound_server(),
                "{name}"
            );
        }
    }

    #[test]
    fn signal_level_separates_silence_from_a_quiet_room() {
        assert!(SignalLevel::measure(&[]).is_silent());
        assert!(SignalLevel::measure(&[0.0; 1600]).is_silent());
        // A quiet room's noise floor (~-70 dBFS) is not silence.
        let room: Vec<f32> = (0..1600)
            .map(|i| if i % 2 == 0 { 0.0003 } else { -0.0003 })
            .collect();
        let level = SignalLevel::measure(&room);
        assert!(!level.is_silent());
        assert!(
            (level.rms_dbfs() - (-70.5)).abs() < 1.0,
            "{}",
            level.rms_dbfs()
        );
        // Non-finite samples are ignored rather than poisoning the level.
        let level = SignalLevel::measure(&[f32::NAN, 0.5, -0.5]);
        assert_eq!(level.peak, 0.5);
        assert!((level.rms - 0.5).abs() < 1e-6);
    }

    #[test]
    fn meter_fill_spans_minus_sixty_to_zero_dbfs() {
        assert_eq!(meter_fill(-90.0), 0.0);
        assert_eq!(meter_fill(-60.0), 0.0);
        assert!((meter_fill(-30.0) - 0.5).abs() < 1e-6);
        assert_eq!(meter_fill(0.0), 1.0);
        assert_eq!(meter_fill(6.0), 1.0);
    }
}
