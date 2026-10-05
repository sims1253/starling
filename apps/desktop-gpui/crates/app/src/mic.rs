//! Microphone selection and the repeatable microphone & shortcut check
//! (#222), app side: the device list the settings dialog shows, the
//! preferred-vs-active display, the live-take input health that ends a
//! dead take as interrupted, and the settings-only test recording.
//!
//! The test records through the same recorder start path as a real take
//! (same device resolution, same capture code), but it journals nothing,
//! saves nothing to history, runs no text processing, inserts nowhere and
//! never touches the output device: its transcript only appears in the
//! dialog's preview.

use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use gpui::{Context, Window};
use starling_dictation::client::StarlingClient;
use starling_dictation::microphone::{
    self, InputDevice, InputProblem, InputRoute, SignalLevel,
};
use starling_dictation::recorder::{self, CaptureRequest, RecorderFault, RecorderHandle};

use crate::app::StarlingApp;

/// No samples for this long ends a live take as interrupted: healthy
/// devices deliver every few milliseconds, and some backends report an
/// unplug only by going quiet. Generous enough for a Bluetooth headset's
/// slow first buffer.
pub(crate) const STALL_LIMIT: Duration = Duration::from_secs(4);

/// After this much of a take with digital silence only, the capture pane
/// stops claiming to listen and says the input is silent.
pub(crate) const SILENCE_GRACE: Duration = Duration::from_secs(3);

/// The microphone check stops itself after this long.
pub(crate) const CHECK_MAX: Duration = Duration::from_secs(8);

/// How long a check transcription may take before it is reported failed.
const CHECK_TRANSCRIBE_TIMEOUT_MS: u64 = 60_000;

/// Whether registering the global record shortcut worked, set once by
/// `main.rs` at startup (`Err` carries the platform's reason).
pub(crate) static SHORTCUT_REGISTRATION: OnceLock<Result<(), String>> = OnceLock::new();

/// The record shortcut as the UI spells it.
pub(crate) fn shortcut_label() -> &'static str {
    if cfg!(target_os = "macos") {
        "⌘ Shift Space"
    } else {
        "Ctrl Shift Space"
    }
}

/// What the settings dialog knows about the host's capture devices.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum DeviceList {
    /// Never listed in this dialog session.
    Unknown,
    Loading,
    Listed(Vec<InputDevice>),
    /// The host failed to list devices. A preference is never touched by
    /// this; the dialog says so.
    Failed(String),
}

/// One selectable row of the picker.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PickerRow {
    /// `None`: follow the system default.
    pub device: Option<String>,
    pub label: String,
    pub detail: Option<String>,
    pub selected: bool,
}

/// The picker rows for a listing and the draft preference: "follow the
/// system default" first, then every listed device, then — so the choice
/// stays visible and is never silently dropped — a preferred device that
/// is not connected right now.
pub(crate) fn picker_rows(list: &DeviceList, draft: Option<&str>) -> Vec<PickerRow> {
    let devices: &[InputDevice] = match list {
        DeviceList::Listed(devices) => devices,
        _ => &[],
    };
    let default_name = devices
        .iter()
        .find(|device| device.is_default)
        .map(|device| device.name.clone());
    let mut rows = vec![PickerRow {
        device: None,
        label: "Follow system default".to_string(),
        detail: Some(match (list, default_name) {
            (_, Some(name)) => format!("Currently {name}"),
            (DeviceList::Listed(_), None) => "No default input right now".to_string(),
            _ => "Whatever input your system selects".to_string(),
        }),
        selected: draft.is_none(),
    }];
    for device in devices {
        rows.push(PickerRow {
            device: Some(device.name.clone()),
            label: device.name.clone(),
            detail: device.is_default.then(|| "System default".to_string()),
            selected: draft == Some(device.name.as_str()),
        });
    }
    if let Some(preferred) = draft {
        if !devices.iter().any(|device| device.name == preferred) {
            rows.push(PickerRow {
                device: Some(preferred.to_string()),
                label: preferred.to_string(),
                detail: Some(match list {
                    DeviceList::Listed(_) => {
                        "Not connected — kept as your choice; recordings use the system default \
                         until it is back"
                            .to_string()
                    }
                    _ => "Your choice (connection unknown until the list loads)".to_string(),
                }),
                selected: true,
            });
        }
    }
    rows
}

/// Why a live take is no longer capturing, decided from the recorder's
/// typed fault and its stall evidence — never from message text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Interruption {
    /// The device or its stream failed (the error callback fired).
    DeviceFailed(String),
    /// The device stopped delivering samples without reporting an error.
    Stalled(Duration),
}

impl Interruption {
    pub(crate) fn describe(&self, device: &str) -> String {
        match self {
            Interruption::DeviceFailed(message) => {
                format!("{device} stopped working mid-recording ({message})")
            }
            Interruption::Stalled(for_) => format!(
                "{device} stopped delivering audio mid-recording (nothing for {:.0} s)",
                for_.as_secs_f32()
            ),
        }
    }
}

/// The interruption a take is under, if any: a fatal (device-side) fault,
/// or a stall past [`STALL_LIMIT`]. A journal fault is not an
/// interruption — capture continues in memory.
pub(crate) fn take_interruption(
    fault: Option<&RecorderFault>,
    stalled_for: Duration,
) -> Option<Interruption> {
    match fault {
        Some(fault) if fault.is_fatal() => {
            Some(Interruption::DeviceFailed(fault.message().to_string()))
        }
        _ if stalled_for >= STALL_LIMIT => Some(Interruption::Stalled(stalled_for)),
        _ => None,
    }
}

/// What the capture pane may claim about the live input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LiveInput {
    Listening,
    /// Samples arrive but they are digital silence.
    Silent,
    /// No samples for a while (not yet long enough to interrupt).
    NotResponding,
}

pub(crate) fn live_input(elapsed: Duration, source_peak: f32, stalled_for: Duration) -> LiveInput {
    if stalled_for >= Duration::from_millis(1_500) {
        LiveInput::NotResponding
    } else if elapsed >= SILENCE_GRACE && source_peak < microphone::SILENT_PEAK {
        LiveInput::Silent
    } else {
        LiveInput::Listening
    }
}

/// How a finished microphone check came out.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum CheckOutcome {
    /// The real transcription of the check recording.
    Transcript(String),
    /// The recording was digital silence: nothing was sent anywhere.
    Silent(InputProblem),
    /// Audio arrived, but there is no transcription engine to try it on.
    NoEngine,
    /// Audio arrived; the transcription request failed.
    TranscriptionFailed(String),
    /// The device failed or stalled during the check.
    Interrupted(String),
}

/// The microphone check's state machine.
pub(crate) enum MicCheck {
    Recording {
        handle: RecorderHandle,
        /// Current RMS meter fill, 0..=1.
        meter: f32,
    },
    Transcribing {
        route: Option<InputRoute>,
        level: SignalLevel,
        seconds: f32,
    },
    Done {
        route: Option<InputRoute>,
        level: SignalLevel,
        seconds: f32,
        outcome: CheckOutcome,
    },
    /// The input could not be opened.
    Failed {
        problem: InputProblem,
        route: Option<InputRoute>,
    },
    /// The check could not start for a reason that is not the input's.
    Blocked(&'static str),
}

/// The settings dialog's microphone state.
pub(crate) struct MicState {
    pub(crate) devices: DeviceList,
    /// Retires a listing that lands after a newer refresh.
    devices_generation: u64,
    pub(crate) check: Option<MicCheck>,
    /// Retires a check transcription that lands after the dialog closed
    /// or a newer check started.
    check_generation: u64,
    /// When the record shortcut last fired while the dialog was open (it
    /// does not toggle recording there; the check shows it was heard).
    pub(crate) shortcut_heard: Option<Instant>,
    /// The input the latest main take opened, for "last recorded from".
    pub(crate) last_route: Option<InputRoute>,
    /// An input problem that stopped a main take from starting (or ended
    /// it), with its recovery actions shown in the capture pane.
    pub(crate) problem: Option<InputProblem>,
    /// The outcome of the last "open sound settings" action, when it
    /// could not open anything.
    pub(crate) settings_launch_error: Option<String>,
}

impl Default for MicState {
    fn default() -> Self {
        Self {
            devices: DeviceList::Unknown,
            devices_generation: 0,
            check: None,
            check_generation: 0,
            shortcut_heard: None,
            last_route: None,
            problem: None,
            settings_launch_error: None,
        }
    }
}

/// Candidate commands that open the OS sound (or microphone privacy)
/// settings, tried in order. Linux has no single entry point: the common
/// desktop panels are tried, and the dialog says so when none exists.
pub(crate) fn settings_commands(privacy: bool) -> Vec<(&'static str, Vec<&'static str>)> {
    if cfg!(target_os = "macos") {
        let pane = if privacy {
            "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone"
        } else {
            "x-apple.systempreferences:com.apple.preference.sound?input"
        };
        vec![("open", vec![pane])]
    } else if cfg!(target_os = "windows") {
        let page = if privacy {
            "ms-settings:privacy-microphone"
        } else {
            "ms-settings:sound"
        };
        vec![("cmd", vec!["/C", "start", "", page])]
    } else {
        vec![
            ("gnome-control-center", vec!["sound"]),
            ("systemsettings", vec!["kcm_pulseaudio"]),
            ("pavucontrol", vec!["--tab=4"]),
        ]
    }
}

fn open_system_settings(privacy: bool) -> Result<(), String> {
    for (program, args) in settings_commands(privacy) {
        if std::process::Command::new(program)
            .args(&args)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .is_ok()
        {
            return Ok(());
        }
    }
    Err(
        "No sound settings app was found (tried GNOME Settings, KDE System Settings and \
         pavucontrol). Open your desktop's sound settings to check the input."
            .to_string(),
    )
}

impl StarlingApp {
    /// The microphone preference takes resolve against now (committed).
    pub(crate) fn preferred_microphone(&self) -> Option<&str> {
        self.microphone_settings.preferred_device.as_deref()
    }

    /// Lists capture devices off the UI thread (ALSA probes each PCM).
    pub(crate) fn refresh_input_devices(&mut self, cx: &mut Context<Self>) {
        self.mic.devices_generation += 1;
        let generation = self.mic.devices_generation;
        self.mic.devices = DeviceList::Loading;
        cx.notify();
        cx.spawn(async move |this, cx| {
            let listed = cx
                .background_spawn(async move { microphone::list_input_devices() })
                .await;
            this.update(cx, |app, cx| {
                if app.mic.devices_generation != generation {
                    return;
                }
                app.mic.devices = match listed {
                    Ok(devices) => DeviceList::Listed(devices),
                    Err(err) => DeviceList::Failed(err),
                };
                cx.notify();
            })
            .ok();
        })
        .detach();
    }

    pub(crate) fn pick_draft_microphone(&mut self, device: Option<String>, cx: &mut Context<Self>) {
        self.draft_microphone = device;
        cx.notify();
    }

    /// Opens the OS sound settings (or the microphone privacy page).
    pub(crate) fn open_input_settings(&mut self, privacy: bool, cx: &mut Context<Self>) {
        self.mic.settings_launch_error = open_system_settings(privacy).err();
        cx.notify();
    }

    /// Starts the microphone check on the dialog's draft choice, through
    /// the same start path as a real take — minus the journal.
    pub(crate) fn start_mic_check(&mut self, cx: &mut Context<Self>) {
        self.cancel_mic_check();
        if self.recorder.is_some() {
            self.mic.check = Some(MicCheck::Blocked(
                "A recording is in progress. Stop it first, then test the microphone.",
            ));
            cx.notify();
            return;
        }
        self.mic.check_generation += 1;
        self.mic.shortcut_heard = None;
        self.mic.check = Some(
            match recorder::start_capture(CaptureRequest {
                journals_dir: None,
                preferred_device: self.draft_microphone.as_deref(),
            }) {
                Ok(handle) => MicCheck::Recording { handle, meter: 0.0 },
                Err(err) => MicCheck::Failed {
                    problem: err.problem,
                    route: err.route,
                },
            },
        );
        cx.notify();
    }

    /// Drops a running check without a result (dialog closed, retried).
    pub(crate) fn cancel_mic_check(&mut self) {
        self.mic.check_generation += 1;
        if let Some(MicCheck::Recording { handle, .. }) = self.mic.check.take() {
            // Release the device; the samples are not wanted.
            let _ = handle.stop();
        }
    }

    /// Stops the check recording and evaluates it: silence is reported
    /// without sending anything; otherwise the take is transcribed raw.
    pub(crate) fn stop_mic_check(&mut self, cx: &mut Context<Self>) {
        let Some(MicCheck::Recording { handle, .. }) = self.mic.check.take() else {
            return;
        };
        let route = handle.input_route().cloned();
        let device = route
            .as_ref()
            .map(|route| route.device.clone())
            .unwrap_or_else(|| "The microphone".to_string());
        let interruption = take_interruption(
            handle.capture_fault().as_ref(),
            handle.input_stalled_for(),
        );
        let stopped = handle.stop();
        let audio = match stopped {
            Ok(take) => take.audio,
            Err(recorder::RecorderError::QuiesceTimeout { audio, .. }) => audio,
            Err(err) => {
                self.mic.check = Some(MicCheck::Done {
                    route,
                    level: SignalLevel::measure(&[]),
                    seconds: 0.0,
                    outcome: CheckOutcome::Interrupted(err.to_string()),
                });
                cx.notify();
                return;
            }
        };
        let level = SignalLevel::measure(&audio.samples);
        let seconds = audio.samples.len() as f32 / audio.sample_rate.max(1) as f32;
        let finish = |outcome| MicCheck::Done {
            route: route.clone(),
            level,
            seconds,
            outcome,
        };
        if let Some(interruption) = interruption {
            self.mic.check = Some(finish(CheckOutcome::Interrupted(
                interruption.describe(&device),
            )));
            cx.notify();
            return;
        }
        if level.is_silent() {
            self.mic.check = Some(finish(CheckOutcome::Silent(InputProblem::Silent {
                device,
            })));
            cx.notify();
            return;
        }
        let target = self.resolve_take_target();
        if target.endpoint().is_empty() {
            self.mic.check = Some(finish(CheckOutcome::NoEngine));
            cx.notify();
            return;
        }
        self.mic.check = Some(MicCheck::Transcribing {
            route: route.clone(),
            level,
            seconds,
        });
        cx.notify();
        let generation = self.mic.check_generation;
        cx.spawn(async move |this, cx| {
            let result = cx
                .background_spawn(async move {
                    let wav = starling_dictation::audio::encode_wav_16k(&audio)
                        .map_err(|err| err.to_string())?;
                    let client = StarlingClient::new(target.endpoint(), target.model())
                        .and_then(|client| client.with_timeout_ms(CHECK_TRANSCRIBE_TIMEOUT_MS))
                        .map_err(|err| err.to_string())?;
                    let result = client
                        .transcribe(Arc::new(wav), "microphone-check")
                        .map_err(|err| err.to_string());
                    // The engine lease is held until the request is done.
                    drop(target);
                    result
                })
                .await;
            this.update(cx, |app, cx| {
                if app.mic.check_generation != generation {
                    return;
                }
                app.mic.check = Some(MicCheck::Done {
                    route,
                    level,
                    seconds,
                    outcome: match result {
                        Ok(result) => CheckOutcome::Transcript(result.text),
                        Err(message) => CheckOutcome::TranscriptionFailed(message),
                    },
                });
                cx.notify();
            })
            .ok();
        })
        .detach();
    }

    /// Per-frame check upkeep while it records: the level meter, the
    /// time limit, and a device that dies mid-check.
    pub(crate) fn tick_mic_check(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(MicCheck::Recording { handle, meter }) = self.mic.check.as_mut() else {
            return;
        };
        let level = SignalLevel::measure(&handle.latest_window(1_024));
        *meter = microphone::meter_fill(level.rms_dbfs());
        let done = handle.elapsed() >= CHECK_MAX
            || take_interruption(handle.capture_fault().as_ref(), handle.input_stalled_for())
                .is_some();
        if done {
            cx.defer_in(window, |app, _window, cx| app.stop_mic_check(cx));
        }
        window.request_animation_frame();
    }

    /// The record shortcut fired while the dialog is open: it never
    /// toggles recording there (a take would start behind the scrim), but
    /// the check shows it arrived.
    pub(crate) fn note_shortcut_in_dialog(&mut self, cx: &mut Context<Self>) {
        self.mic.shortcut_heard = Some(Instant::now());
        cx.notify();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn device(name: &str, is_default: bool) -> InputDevice {
        InputDevice {
            name: name.to_string(),
            is_default,
        }
    }

    #[test]
    fn the_picker_keeps_a_disconnected_preferred_device_visible_and_selected() {
        let list = DeviceList::Listed(vec![device("Laptop Mic", true)]);
        let rows = picker_rows(&list, Some("USB Mic"));
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].device, None);
        assert_eq!(rows[0].detail.as_deref(), Some("Currently Laptop Mic"));
        assert!(!rows[0].selected);
        assert_eq!(rows[1].detail.as_deref(), Some("System default"));
        let missing = &rows[2];
        assert_eq!(missing.device.as_deref(), Some("USB Mic"));
        assert!(missing.selected);
        assert!(missing.detail.as_deref().unwrap().contains("Not connected"));
    }

    #[test]
    fn a_failed_listing_never_drops_the_preferred_device_from_the_picker() {
        let rows = picker_rows(&DeviceList::Failed("ALSA busy".to_string()), Some("USB Mic"));
        assert_eq!(rows.len(), 2);
        assert!(rows[1].selected);
        assert_eq!(rows[1].device.as_deref(), Some("USB Mic"));
        assert!(rows[1].detail.as_deref().unwrap().contains("unknown"));
    }

    #[test]
    fn following_the_default_selects_the_first_row_only() {
        let list = DeviceList::Listed(vec![device("Laptop Mic", true), device("USB Mic", false)]);
        let rows = picker_rows(&list, None);
        assert_eq!(rows.iter().filter(|row| row.selected).count(), 1);
        assert!(rows[0].selected);
        let rows = picker_rows(&list, Some("USB Mic"));
        assert_eq!(rows.iter().filter(|row| row.selected).count(), 1);
        assert!(rows[2].selected);
    }

    #[test]
    fn only_device_faults_and_long_stalls_interrupt_a_take() {
        let journal = RecorderFault::Journal("disk full".to_string());
        let device_fault = RecorderFault::Device("device unplugged".to_string());
        assert_eq!(take_interruption(None, Duration::ZERO), None);
        assert_eq!(take_interruption(Some(&journal), Duration::from_secs(1)), None);
        assert_eq!(
            take_interruption(Some(&device_fault), Duration::ZERO),
            Some(Interruption::DeviceFailed("device unplugged".to_string()))
        );
        assert_eq!(
            take_interruption(None, STALL_LIMIT),
            Some(Interruption::Stalled(STALL_LIMIT))
        );
        assert_eq!(take_interruption(None, STALL_LIMIT - Duration::from_millis(1)), None);
        let text = Interruption::Stalled(Duration::from_secs(5)).describe("USB Mic");
        assert!(text.contains("USB Mic") && text.contains("5 s"), "{text}");
    }

    #[test]
    fn the_pane_stops_claiming_to_listen_to_a_dead_or_silent_input() {
        let quick = Duration::from_millis(5);
        assert_eq!(live_input(Duration::from_secs(1), 0.0, quick), LiveInput::Listening);
        assert_eq!(live_input(SILENCE_GRACE, 0.0, quick), LiveInput::Silent);
        assert_eq!(live_input(SILENCE_GRACE, 0.2, quick), LiveInput::Listening);
        assert_eq!(
            live_input(Duration::from_secs(10), 0.2, Duration::from_secs(2)),
            LiveInput::NotResponding
        );
    }

    #[test]
    fn every_platform_has_a_settings_command_to_try() {
        assert!(!settings_commands(false).is_empty());
        assert!(!settings_commands(true).is_empty());
    }

    #[test]
    fn the_shortcut_label_names_the_platform_modifier() {
        let label = shortcut_label();
        assert!(label.ends_with("Shift Space"), "{label}");
    }
}
