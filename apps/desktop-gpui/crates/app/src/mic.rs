//! Microphone selection and the repeatable microphone & shortcut check,
//! app side: the device list the settings dialog shows, the
//! preferred-vs-active display, the live-take input health that ends a
//! dead take as interrupted, and the settings-only test recording.
//!
//! The test records through the same recorder start path as a real take
//! (same device resolution, same capture code), but it journals nothing,
//! saves nothing to history, runs no text processing, inserts nowhere and
//! never touches the output device: its transcript only appears in the
//! dialog's preview.

use std::sync::Arc;
use std::time::{Duration, Instant};

use gpui::{AppContext, Context, Window};
use starling_dictation::client::StarlingClient;
use starling_dictation::disk::DiskLevel;
use starling_dictation::microphone::{
    self, InputDevice, InputProblem, InputRoute, SettingsPage, SignalLevel,
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

/// What the settings dialog knows about the host's capture devices.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) enum DeviceList {
    #[default]
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
    /// The recording service ended the take on its own (#220) without a
    /// fault this window saw.
    Ended,
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
            Interruption::Ended => format!("{device} stopped recording mid-take"),
        }
    }
}

/// The interruption a take is under, if any: a fatal (device-side) fault,
/// or a stall past [`STALL_LIMIT`]. A journal fault is not an
/// interruption — capture continues in memory.
fn take_interruption(fault: Option<&RecorderFault>, stalled_for: Duration) -> Option<Interruption> {
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

/// `silent_for` is how long the input has carried no sound (since the
/// last audible window, or since the take started) — recent evidence, so
/// an input muted after speech is caught too.
pub(crate) fn live_input(silent_for: Duration, stalled_for: Duration) -> LiveInput {
    if stalled_for >= Duration::from_millis(1_500) {
        LiveInput::NotResponding
    } else if silent_for >= SILENCE_GRACE {
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
#[derive(Default)]
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
    /// it), beside the exact error-banner text it produced: the banner
    /// offers the problem's recovery actions only while it still shows
    /// that text, never under an unrelated later error.
    pub(crate) problem: Option<(InputProblem, String)>,
    /// The outcome of the last "open sound settings" action, when it
    /// could not open anything.
    pub(crate) settings_launch_error: Option<String>,
    /// Retires a settings-launch result that lands after a newer launch,
    /// or after the dialog it belonged to closed.
    pub(crate) settings_launch_generation: u64,
    /// Why the current take is being ended as interrupted, read by the
    /// cancel path (`CancelReason::InputLost`).
    pub(crate) interruption: Option<(String, InputProblem)>,
    /// When the live take's input last carried sound; `None` until it
    /// first does.
    pub(crate) last_sound_at: Option<Instant>,
    /// The live take already showed its low-disk warning (#342), so a
    /// dismissed warning is not raised again on every poll.
    pub(crate) disk_warned: bool,
    /// The low-disk warning text the live take shows, so a critical stop
    /// can drop it instead of joining a prediction that already came true.
    pub(crate) disk_low_warning: Option<String>,
    /// The live take shows [`DISK_UNCHECKED_NOTE`] (#342) because the
    /// free-space probe is failing.
    pub(crate) disk_unchecked: bool,
}

/// `warning` without the take's low-disk warning `low`: once the take
/// stopped for a full disk, "minutes left before Starling stops a take"
/// contradicts the stop note. Other warnings the stop set are kept.
fn without_low_disk_warning(warning: Option<String>, low: Option<String>) -> Option<String> {
    let (Some(warning), Some(low)) = (warning.clone(), low) else {
        return warning;
    };
    let rest = warning.replace(low.as_str(), "");
    let rest = rest.split_whitespace().collect::<Vec<_>>().join(" ");
    (!rest.is_empty()).then_some(rest)
}

/// Shown, without stopping the take, while the free-space probe fails
/// (#342); a journal write that finds the disk full still stops it.
const DISK_UNCHECKED_NOTE: &str = "Can't check free disk space.";

/// Candidate commands that open the OS sound (or microphone privacy)
/// settings, tried in order. Linux has no single entry point: the common
/// desktop panels are tried, and the dialog says so when none exists.
fn settings_commands(privacy: bool) -> Vec<(&'static str, Vec<&'static str>)> {
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

/// The settings pages a problem offers, as this platform can open them:
/// Linux desktops have no separate microphone-permission page, so a
/// privacy page folds into the sound settings there.
pub(crate) fn platform_pages(problem: &InputProblem) -> Vec<SettingsPage> {
    let mut pages = Vec::new();
    for &page in problem.settings_pages() {
        let page = if cfg!(target_os = "linux") {
            SettingsPage::Sound
        } else {
            page
        };
        if !pages.contains(&page) {
            pages.push(page);
        }
    }
    pages
}

pub(crate) fn page_label(page: SettingsPage) -> &'static str {
    match page {
        SettingsPage::Sound => "Open sound settings",
        SettingsPage::Privacy => "Open privacy settings",
    }
}

/// How long a settings candidate gets to prove itself: GUI apps stay
/// running once they open, while a launcher installed for the wrong
/// desktop (`gnome-control-center` outside GNOME) exits non-zero at once.
const SETTINGS_LAUNCH_WAIT: Duration = Duration::from_millis(1_500);

/// Tries each settings candidate in order; one that exits non-zero within
/// [`SETTINGS_LAUNCH_WAIT`] moves on to the next, one still running after
/// it (or exited 0) counts as opened. Blocking, so it runs off the UI
/// thread.
fn open_system_settings(privacy: bool) -> Result<(), String> {
    let commands = settings_commands(privacy);
    for (program, args) in &commands {
        let Ok(mut child) = std::process::Command::new(program)
            .args(args)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
        else {
            continue;
        };
        let deadline = Instant::now() + SETTINGS_LAUNCH_WAIT;
        let exited = loop {
            match child.try_wait() {
                Ok(Some(status)) => break Some(status.success()),
                Ok(None) if Instant::now() < deadline => {
                    std::thread::sleep(Duration::from_millis(50));
                }
                Ok(None) => break None,
                Err(_) => {
                    let _ = child.kill();
                    let _ = child.wait();
                    break Some(false);
                }
            }
        };
        match exited {
            Some(true) => return Ok(()),
            Some(false) => continue,
            None => {
                // Reap the settings app whenever the user closes it.
                std::thread::spawn(move || {
                    let _ = child.wait();
                });
                return Ok(());
            }
        }
    }
    let tried: Vec<_> = commands.iter().map(|(program, _)| *program).collect();
    Err(format!(
        "No sound settings app could be opened (tried {}). Open your desktop's sound settings \
         yourself to check the input.",
        tried.join(", ")
    ))
}

/// What the input-health checks read: the check's own recorder, or the
/// live take the recording service records (#220).
pub(crate) trait InputHealth {
    fn input_route(&self) -> Option<&InputRoute>;
    fn input_stalled_for(&self) -> Option<Duration>;
    fn capture_fault(&self) -> Option<RecorderFault>;
}

impl InputHealth for RecorderHandle {
    fn input_route(&self) -> Option<&InputRoute> {
        RecorderHandle::input_route(self)
    }
    fn input_stalled_for(&self) -> Option<Duration> {
        RecorderHandle::input_stalled_for(self)
    }
    fn capture_fault(&self) -> Option<RecorderFault> {
        RecorderHandle::capture_fault(self)
    }
}

impl InputHealth for crate::host_link::LiveCapture {
    fn input_route(&self) -> Option<&InputRoute> {
        crate::host_link::LiveCapture::input_route(self)
    }
    fn input_stalled_for(&self) -> Option<Duration> {
        crate::host_link::LiveCapture::input_stalled_for(self)
    }
    fn capture_fault(&self) -> Option<RecorderFault> {
        crate::host_link::LiveCapture::capture_fault(self)
    }
}

/// The device a take captures from, for messages.
pub(crate) fn device_name(handle: &impl InputHealth) -> String {
    handle
        .input_route()
        .map(|route| route.device.clone())
        .unwrap_or_else(|| "The microphone".to_string())
}

/// The interruption a live take is under. A stall only counts once
/// audio has arrived: a take that never delivered is the activation
/// machine's start stall (`activation::START_STALL`), not a lost input.
fn live_interruption(handle: &impl InputHealth) -> Option<Interruption> {
    let stalled_for = handle.input_stalled_for().unwrap_or_default();
    take_interruption(handle.capture_fault().as_ref(), stalled_for)
}

/// Where the microphone check sends its clip: the built-in engine's
/// endpoint (its lease held until the check is done) or the server from
/// Settings. Empty when no engine serves.
pub(crate) struct CheckTarget {
    pub endpoint: String,
    pub model: String,
    _lease: Option<starling_dictation::engine::EngineLease>,
}

impl StarlingApp {
    /// The microphone check's target, resolved now.
    pub(crate) fn check_target(&self) -> CheckTarget {
        match self.engine_settings.mode {
            starling_dictation::settings::EngineMode::Builtin => {
                match self.engine.as_ref().and_then(|engine| engine.lease()) {
                    Some(lease) => CheckTarget {
                        endpoint: lease.endpoint().to_string(),
                        model: lease.slug().to_string(),
                        _lease: Some(lease),
                    },
                    None => CheckTarget {
                        endpoint: String::new(),
                        model: String::new(),
                        _lease: None,
                    },
                }
            }
            starling_dictation::settings::EngineMode::Manual => CheckTarget {
                endpoint: self.endpoint.clone(),
                model: self.model.clone(),
                _lease: None,
            },
        }
    }

    /// Shows `text` in the error banner as the explanation of `problem`,
    /// so the banner can offer that problem's recovery actions.
    pub(crate) fn report_input_problem(&mut self, problem: InputProblem, text: String) {
        self.error = Some(text.clone());
        self.mic.problem = Some((problem, text));
        self.mic.settings_launch_error = None;
    }

    /// The input problem the error banner is currently explaining.
    pub(crate) fn shown_input_problem(&self) -> Option<&InputProblem> {
        let (problem, text) = self.mic.problem.as_ref()?;
        (self.error.as_deref() == Some(text.as_str())).then_some(problem)
    }

    /// The capture pane's headline override while a take records: never
    /// "listening" to an input that is silent or not responding.
    pub(crate) fn live_input_headline(&self) -> Option<&'static str> {
        let handle = self.recorder.as_ref()?;
        let silent_for = self
            .mic
            .last_sound_at
            .map(|at| at.elapsed())
            .unwrap_or_else(|| handle.elapsed());
        match live_input(silent_for, handle.input_stalled_for().unwrap_or_default()) {
            LiveInput::Listening => None,
            LiveInput::Silent => Some("No sound from the microphone."),
            LiveInput::NotResponding => Some("The microphone stopped responding."),
        }
    }

    /// The live take's free-space reading (#342): a low disk warns once;
    /// a critical one stops the take the normal way — finalized, saved
    /// and transcribed — while there is still room to do that, and says
    /// why.
    fn watch_take_disk(&mut self, take: crate::activation::TakeId, cx: &mut Context<Self>) {
        let Some(handle) = self.recorder.as_ref() else {
            return;
        };
        let probe_failing = handle.disk_probe_failing();
        if probe_failing != self.mic.disk_unchecked {
            self.mic.disk_unchecked = probe_failing;
            self.capture_warning =
                with_disk_unchecked_note(self.capture_warning.take(), probe_failing);
            cx.notify();
        }
        let Some(reading) = handle.disk_reading() else {
            return;
        };
        let rate = handle.sample_rate();
        let policy = starling_dictation::disk::DiskPolicy::default();
        match reading.level {
            DiskLevel::Ok => {}
            DiskLevel::Low => {
                if !self.mic.disk_warned {
                    self.mic.disk_warned = true;
                    self.capture_warning = policy.warning(reading, rate);
                    self.mic.disk_low_warning = self.capture_warning.clone();
                    cx.notify();
                }
            }
            DiskLevel::Critical => {
                self.activation_input(|machine| machine.storage_full(take), cx);
                // The stop above ran synchronously and set the take's own
                // warnings; this joins them. It never claims the take was
                // saved — saving is still under way, and a failure there
                // (or a journal fault) reports itself through `error`.
                let note = if reading.available == 0 {
                    "Recording stopped because the disk is full. Free up space before recording \
                     again."
                        .to_string()
                } else {
                    format!(
                        "Recording stopped because the disk is almost full ({} MB free). Free \
                         up space before recording again.",
                        reading.available / (1024 * 1024)
                    )
                };
                let low = self.mic.disk_low_warning.take();
                let existing = without_low_disk_warning(self.capture_warning.take(), low);
                self.capture_warning = Some(match existing {
                    Some(existing) => format!("{note} {existing}"),
                    None => note,
                });
                cx.notify();
            }
        }
    }

    /// Records why the running take lost its input, if it did; `true`
    /// when the take must end as interrupted.
    pub(crate) fn note_live_interruption(&mut self) -> bool {
        let Some(handle) = self.recorder.as_ref() else {
            return false;
        };
        let Some(interruption) = live_interruption(handle) else {
            return false;
        };
        let device = device_name(handle);
        self.note_interruption(&device, interruption);
        true
    }

    /// Records why the current take is ending as interrupted, for the
    /// `CancelReason::InputLost` save path.
    pub(crate) fn note_interruption(&mut self, device: &str, interruption: Interruption) {
        let problem = InputProblem::Unavailable {
            device: device.to_string(),
            detail: "it stopped during the last recording".to_string(),
        };
        self.mic.interruption = Some((interruption.describe(device), problem));
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

    /// Opens the OS sound settings (or the microphone privacy page), off
    /// the UI thread.
    pub(crate) fn open_input_settings(&mut self, page: SettingsPage, cx: &mut Context<Self>) {
        let privacy = page == SettingsPage::Privacy;
        self.mic.settings_launch_error = None;
        self.mic.settings_launch_generation += 1;
        let generation = self.mic.settings_launch_generation;
        cx.notify();
        cx.spawn(async move |this, cx| {
            let result = cx
                .background_spawn(async move { open_system_settings(privacy) })
                .await;
            this.update(cx, |app, cx| {
                if app.mic.settings_launch_generation != generation {
                    return;
                }
                app.mic.settings_launch_error = result.err();
                cx.notify();
            })
            .ok();
        })
        .detach();
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
        self.mic.shortcut_heard = None;
        // A preview or stop cue still sounding would be heard as signal.
        self.cue_take_starting();
        self.mic.check = Some(
            match recorder::start_capture(CaptureRequest {
                disk_watch: None,
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

    /// Drops the check and any running recording without a result
    /// (dialog opened or closed, retried).
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
        let device = device_name(&handle);
        let mut interruption = live_interruption(&handle).map(|i| i.describe(&device));
        let audio = match handle.stop() {
            Ok(take) => {
                // A fault posted after the check above still counts.
                if let (None, Some(fault)) = (&interruption, take.device_fault) {
                    interruption = Some(Interruption::DeviceFailed(fault).describe(&device));
                }
                take.audio
            }
            Err(recorder::RecorderError::QuiesceTimeout { audio, .. }) => {
                // The audio is salvaged, but the device did not stop
                // cleanly: never report that as a working microphone.
                interruption.get_or_insert_with(|| {
                    format!("{device} did not stop cleanly (the audio callback hung)")
                });
                audio
            }
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
            self.mic.check = Some(finish(CheckOutcome::Interrupted(interruption)));
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
        let target = self.check_target();
        if target.endpoint.is_empty() {
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
                    let client = StarlingClient::new(&target.endpoint, &target.model)
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

    /// Per-frame check upkeep while it records: the level meter.
    pub(crate) fn tick_mic_check(&mut self, window: &mut Window) {
        let Some(MicCheck::Recording { handle, meter }) = self.mic.check.as_mut() else {
            return;
        };
        let level = SignalLevel::measure(&handle.latest_window(1_024));
        *meter = microphone::meter_fill(level.rms_dbfs());
        window.request_animation_frame();
    }

    /// Microphone upkeep on the activation timer loop rather than in
    /// render, so it keeps running while a hidden window renders nothing:
    /// the silence evidence, the live-take watchdog that ends a dead take
    /// as interrupted, and the check's time limit.
    pub(crate) fn poll_microphone(&mut self, cx: &mut Context<Self>) {
        if let Some(handle) = self.recorder.as_ref() {
            if !SignalLevel::measure(&handle.latest_window(1_024)).is_silent() {
                self.mic.last_sound_at = Some(Instant::now());
            }
        }
        if let Some(take) = self.recording_take {
            if self.note_live_interruption() {
                self.activation_input(|machine| machine.input_lost(take), cx);
            } else {
                self.watch_take_disk(take, cx);
            }
        }
        let stop_check = match self.mic.check.as_ref() {
            Some(MicCheck::Recording { handle, .. }) => {
                handle.elapsed() >= CHECK_MAX || live_interruption(handle).is_some()
            }
            _ => false,
        };
        if stop_check {
            self.stop_mic_check(cx);
        }
    }

    /// Whether the settings dialog's check is recording right now (the
    /// activation loop keeps its fast interval then).
    pub(crate) fn mic_check_recording(&self) -> bool {
        matches!(self.mic.check, Some(MicCheck::Recording { .. }))
    }

    /// The record shortcut fired while the dialog is open: it never
    /// toggles recording there (a take would start behind the scrim), but
    /// the check shows it arrived.
    pub(crate) fn note_shortcut_in_dialog(&mut self, cx: &mut Context<Self>) {
        self.mic.shortcut_heard = Some(Instant::now());
        cx.notify();
    }
}

/// The capture warning with [`DISK_UNCHECKED_NOTE`] leading it while
/// `failing`, and without it once the probe answers again.
fn with_disk_unchecked_note(warning: Option<String>, failing: bool) -> Option<String> {
    let rest = warning.and_then(|text| match text.strip_prefix(DISK_UNCHECKED_NOTE) {
        Some(rest) => Some(rest.trim_start().to_string()).filter(|rest| !rest.is_empty()),
        None => Some(text),
    });
    match (failing, rest) {
        (true, Some(rest)) => Some(format!("{DISK_UNCHECKED_NOTE} {rest}")),
        (true, None) => Some(DISK_UNCHECKED_NOTE.to_string()),
        (false, rest) => rest,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_critical_stop_drops_the_low_disk_prediction_but_keeps_other_warnings() {
        let low = "Disk space is low (900 MB free): about 3 minutes of recording left.";
        assert_eq!(
            without_low_disk_warning(Some(low.to_string()), Some(low.to_string())),
            None
        );
        assert_eq!(
            without_low_disk_warning(
                Some(format!("Input is clipping. {low}")),
                Some(low.to_string())
            ),
            Some("Input is clipping.".to_string())
        );
        assert_eq!(
            without_low_disk_warning(Some("Input is clipping.".to_string()), None),
            Some("Input is clipping.".to_string())
        );
    }

    #[test]
    fn the_disk_unchecked_note_joins_and_leaves_the_capture_warning() {
        assert_eq!(
            with_disk_unchecked_note(None, true).as_deref(),
            Some(DISK_UNCHECKED_NOTE)
        );
        let joined = with_disk_unchecked_note(Some("Clipping.".to_string()), true);
        assert_eq!(
            joined.as_deref(),
            Some("Can't check free disk space. Clipping.")
        );
        assert_eq!(
            with_disk_unchecked_note(joined.clone(), true),
            joined,
            "never doubled"
        );
        assert_eq!(
            with_disk_unchecked_note(joined, false).as_deref(),
            Some("Clipping.")
        );
        assert_eq!(
            with_disk_unchecked_note(Some(DISK_UNCHECKED_NOTE.to_string()), false),
            None
        );
    }

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
        let rows = picker_rows(
            &DeviceList::Failed("ALSA busy".to_string()),
            Some("USB Mic"),
        );
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
        assert_eq!(
            take_interruption(Some(&journal), Duration::from_secs(1)),
            None
        );
        assert_eq!(
            take_interruption(Some(&device_fault), Duration::ZERO),
            Some(Interruption::DeviceFailed("device unplugged".to_string()))
        );
        assert_eq!(
            take_interruption(None, STALL_LIMIT),
            Some(Interruption::Stalled(STALL_LIMIT))
        );
        assert_eq!(
            take_interruption(None, STALL_LIMIT - Duration::from_millis(1)),
            None
        );
        let text = Interruption::Stalled(Duration::from_secs(5)).describe("USB Mic");
        assert!(text.contains("USB Mic") && text.contains("5 s"), "{text}");
    }

    #[test]
    fn the_pane_stops_claiming_to_listen_to_a_dead_or_silent_input() {
        let quick = Duration::from_millis(5);
        assert_eq!(
            live_input(Duration::from_secs(1), quick),
            LiveInput::Listening
        );
        assert_eq!(live_input(SILENCE_GRACE, quick), LiveInput::Silent);
        // Silence measured from the last audible window: speech followed
        // by a muted input becomes "silent" too, not only a take that was
        // silent from the start.
        assert_eq!(
            live_input(SILENCE_GRACE - Duration::from_millis(1), quick),
            LiveInput::Listening
        );
        assert_eq!(
            live_input(Duration::ZERO, Duration::from_secs(2)),
            LiveInput::NotResponding
        );
    }

    #[test]
    fn silence_offers_privacy_where_the_platform_has_such_a_page() {
        let silent = InputProblem::Silent {
            device: "Mic".to_string(),
        };
        let pages = platform_pages(&silent);
        if cfg!(target_os = "linux") {
            assert_eq!(pages, vec![SettingsPage::Sound]);
        } else {
            assert_eq!(pages, vec![SettingsPage::Sound, SettingsPage::Privacy]);
        }
        let denied = InputProblem::PermissionDenied {
            device: "Mic".to_string(),
            detail: "denied".to_string(),
        };
        assert_eq!(platform_pages(&denied).len(), 1);
    }
}
