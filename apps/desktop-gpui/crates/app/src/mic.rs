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

use std::sync::Arc;
use std::time::{Duration, Instant};

use gpui::{AppContext, Context, Window};
use starling_dictation::client::StarlingClient;
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
    /// When the live take's input last carried sound (render-loop
    /// evidence); `None` until it first does.
    pub(crate) last_sound_at: Option<Instant>,
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
            settings_launch_generation: 0,
            interruption: None,
            last_sound_at: None,
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

/// How long each settings candidate gets to prove itself (#222): GUI
/// apps stay running once they open, while a launcher installed for the
/// wrong desktop exits at once with a non-zero status — a `spawn` that
/// succeeds is never evidence on its own.
const SETTINGS_LAUNCH_WAIT: Duration = Duration::from_millis(1_500);

/// The message when no settings app could be opened, naming the programs
/// actually tried — built from the candidate list, never hard-coded, so
/// it never claims Linux panels were tried on macOS/Windows (#222).
pub(crate) fn settings_launch_failure(programs: &[&str]) -> String {
    format!(
        "No sound settings app could be opened (tried {}). Open your desktop's sound \
         settings yourself to check the input.",
        programs.join(", ")
    )
}

/// What `try_wait` reported about one settings candidate inside its
/// grace period (#222): the process exited (with its success flag), or
/// it is still running and `waited_out` says whether
/// [`SETTINGS_LAUNCH_WAIT`] has already elapsed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CandidateProgress {
    Exited { success: bool },
    Running { waited_out: bool },
}

/// What one launch attempt proved (#222). Pure over the exit outcome,
/// so the fallback order is unit-tested without spawning anything.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CandidateVerdict {
    /// Exited 0 (a handing-off launcher) or still running past the wait:
    /// the settings app opened — let it run, take no further candidate.
    Opened,
    /// Exited non-zero inside the wait (a wrong-desktop launcher) or an
    /// unreadable status: this candidate failed — try the next.
    TryNext,
    /// Still running inside the wait: poll again.
    KeepWaiting,
}

pub(crate) fn candidate_verdict(progress: CandidateProgress) -> CandidateVerdict {
    match progress {
        CandidateProgress::Exited { success: true } => CandidateVerdict::Opened,
        CandidateProgress::Exited { success: false } => CandidateVerdict::TryNext,
        CandidateProgress::Running { waited_out: true } => CandidateVerdict::Opened,
        CandidateProgress::Running { waited_out: false } => CandidateVerdict::KeepWaiting,
    }
}

/// What the one final `try_wait` after the launch wait ran out proves
/// (#222): `Some(success)` is a real exit — non-zero right at the
/// deadline fails like any other non-zero exit — while `None` (still
/// running) keeps the waited-out verdict that counts as opened. Pure,
/// so the deadline's last look is unit-tested like `candidate_verdict`.
fn final_candidate_progress(exited: Option<bool>) -> CandidateProgress {
    match exited {
        Some(success) => CandidateProgress::Exited { success },
        None => CandidateProgress::Running { waited_out: true },
    }
}

/// Tries each settings candidate in order. A candidate that spawns but
/// exits non-zero within [`SETTINGS_LAUNCH_WAIT`] (a wrong-desktop
/// launcher, e.g. `gnome-control-center` outside GNOME) moves on to the
/// next; one still running after the wait, or exited 0, counts as
/// opened. Blocking for up to ~1.5 s per failing candidate, so it runs
/// off the UI thread (`open_input_settings`).
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
            // Not installed: the next candidate.
            continue;
        };
        let deadline = Instant::now() + SETTINGS_LAUNCH_WAIT;
        loop {
            let progress = match child.try_wait() {
                Ok(Some(status)) => CandidateProgress::Exited {
                    success: status.success(),
                },
                Ok(None) if Instant::now() < deadline => CandidateProgress::Running {
                    waited_out: false,
                },
                // The wait ran out and it looked running: one last look
                // (#222) — a launcher exiting non-zero right at the
                // deadline failed, and the next candidate must get its
                // turn instead of trusting the earlier "still running".
                Ok(None) => final_candidate_progress(
                    child.try_wait().ok().flatten().map(|status| status.success()),
                ),
                // The status itself is unreadable: this candidate
                // counts as failed, like a non-zero exit — killed and
                // reaped, so it cannot linger as a zombie (#222).
                Err(_) => {
                    let _ = child.kill();
                    let _ = child.wait();
                    CandidateProgress::Exited { success: false }
                }
            };
            match candidate_verdict(progress) {
                CandidateVerdict::Opened => {
                    // Reap it whenever it exits, so a settings app the
                    // user closes later never lingers as a zombie.
                    std::thread::spawn(move || {
                        let _ = child.wait();
                    });
                    return Ok(());
                }
                CandidateVerdict::TryNext => break,
                CandidateVerdict::KeepWaiting => std::thread::sleep(Duration::from_millis(50)),
            }
        }
    }
    Err(settings_launch_failure(
        &commands
            .iter()
            .map(|(program, _)| *program)
            .collect::<Vec<_>>(),
    ))
}

/// Whether a capture window counts as sound (#222): a peak at or
/// above the digital-silence floor. Pure, so the silence evidence's
/// threshold is unit-tested without a device.
fn window_is_audible(samples: &[f32]) -> bool {
    SignalLevel::measure(samples).peak >= microphone::SILENT_PEAK
}

/// The interruption a live recorder is under. A stall only counts once
/// audio has arrived: a take that never delivered is the activation
/// machine's start stall (`activation::START_STALL`), not a lost input.
pub(crate) fn live_interruption(handle: &RecorderHandle) -> Option<Interruption> {
    let stalled_for = if handle.captured_sample_count() > 0 {
        handle.input_stalled_for()
    } else {
        Duration::ZERO
    };
    take_interruption(handle.capture_fault().as_ref(), stalled_for)
}

/// The mic check's stop decision, pure so it can be tested: its time
/// limit ([`CHECK_MAX`]), or an input that died mid-check
/// ([`live_interruption`]'s verdict). Called from the timer loop
/// (`poll_microphone`), not render, so a hidden window cannot let a
/// check run past its limit (#222).
pub(crate) fn check_should_stop(elapsed: Duration, interruption: Option<Interruption>) -> bool {
    elapsed >= CHECK_MAX || interruption.is_some()
}

impl StarlingApp {
    /// Shows `text` in the error banner as the explanation of `problem`,
    /// so the banner can offer that problem's recovery actions.
    pub(crate) fn report_input_problem(&mut self, problem: InputProblem, text: String) {
        self.error = Some(text.clone());
        self.mic.problem = Some((problem, text));
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
        let stalled_for = if handle.captured_sample_count() > 0 {
            handle.input_stalled_for()
        } else {
            Duration::ZERO
        };
        match live_input(silent_for, stalled_for) {
            LiveInput::Listening => None,
            LiveInput::Silent => Some("No sound from the microphone."),
            LiveInput::NotResponding => Some("The microphone stopped responding."),
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
        let device = handle
            .input_route()
            .map(|route| route.device.clone())
            .unwrap_or_else(|| "The microphone".to_string());
        let problem = InputProblem::Unavailable {
            device: device.clone(),
            detail: "it stopped during the last recording".to_string(),
        };
        self.mic.interruption = Some((interruption.describe(&device), problem));
        true
    }

    /// The live-take watchdog (the timer loop's `poll_microphone`, so a
    /// hidden window cannot pause it): a device that failed or stopped
    /// delivering ends the take as interrupted, its audio kept, instead
    /// of the pane presenting a dead input as listening.
    pub(crate) fn end_interrupted_take(&mut self, cx: &mut Context<Self>) {
        let Some(take) = self.recording_take else {
            return;
        };
        if self.note_live_interruption() {
            self.activation_input(|machine| machine.input_lost(take), cx);
        }
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
    /// Launched off the UI thread (#222): a failing candidate takes up
    /// to [`SETTINGS_LAUNCH_WAIT`] to prove it failed, and several may
    /// be tried before one opens.
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
        // `live_interruption`, not raw `take_interruption`: a stall only
        // counts once audio first arrived, so a device slow to deliver
        // its first buffer (Bluetooth) is never reported as "stopped
        // delivering" — a take that delivered nothing reaches the
        // silence/empty handling below instead (#222).
        let mut interruption = live_interruption(&handle).map(|i| i.describe(&device));
        let stopped = handle.stop();
        let audio = match stopped {
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

    /// Per-frame check upkeep while it records: the level meter only.
    /// The time limit and the mid-check interruption decision run on the
    /// timer loop (`poll_microphone`), which does not pause when a
    /// hidden or minimized window stops rendering (#222).
    pub(crate) fn tick_mic_check(&mut self, window: &mut Window) {
        let Some(MicCheck::Recording { handle, meter }) = self.mic.check.as_mut() else {
            return;
        };
        let level = SignalLevel::measure(&handle.latest_window(1_024));
        *meter = microphone::meter_fill(level.rms_dbfs());
        window.request_animation_frame();
    }

    /// Timer-driven microphone upkeep (#222), run from the activation
    /// loop (`poll_activation`) every 20–50 ms instead of from render: a
    /// hidden or minimized window stops rendering, and a dead input
    /// must not stay "recording", nor a check run past its limit, nor a
    /// healthy input look silent because the render-side sound stamp
    /// went stale while hidden. Render keeps only what rendering needs
    /// — the meter fill and the frames.
    pub(crate) fn poll_microphone(&mut self, cx: &mut Context<Self>) {
        if let Some(handle) = self.recorder.as_ref() {
            // #222: `last_sound_at` is stamped here, on the timer, so a
            // window hidden mid-take and restored cannot briefly claim
            // "No sound from the microphone." for a healthy input —
            // render's own stoppage is not the input's silence.
            if window_is_audible(&handle.latest_window(1_024)) {
                self.mic.last_sound_at = Some(Instant::now());
            }
        }
        if self.recording_take.is_some() {
            self.end_interrupted_take(cx);
        }
        let stop_check = match self.mic.check.as_ref() {
            Some(MicCheck::Recording { handle, .. }) => {
                check_should_stop(handle.elapsed(), live_interruption(handle))
            }
            _ => false,
        };
        if stop_check {
            self.stop_mic_check(cx);
        }
    }

    /// Whether the settings dialog's check is recording right now: the
    /// activation loop keeps its fast interval then, so the check's time
    /// limit is enforced on time and not only when a window renders
    /// (#222). It never stops the loop from polling.
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

    #[test]
    fn every_platform_has_a_settings_command_to_try() {
        assert!(!settings_commands(false).is_empty());
        assert!(!settings_commands(true).is_empty());
    }

    #[test]
    fn the_check_stops_at_its_limit_or_an_interruption() {
        // #222: the decision is timer-driven (`poll_microphone`), so it
        // must not need a window: pure elapsed-vs-limit plus the
        // interruption verdict.
        assert!(!check_should_stop(
            CHECK_MAX - Duration::from_millis(1),
            None
        ));
        assert!(check_should_stop(CHECK_MAX, None));
        assert!(check_should_stop(
            Duration::ZERO,
            Some(Interruption::DeviceFailed("unplugged".to_string()))
        ));
        assert!(check_should_stop(
            Duration::ZERO,
            Some(Interruption::Stalled(STALL_LIMIT))
        ));
    }

    #[test]
    fn the_settings_failure_message_names_the_programs_actually_tried() {
        // #222: built from the candidate list, never hard-coded — a
        // macOS/Windows build must not claim Linux panels were tried.
        let message = settings_launch_failure(&["gnome-control-center", "pavucontrol"]);
        assert!(message.contains("gnome-control-center"), "{message}");
        assert!(message.contains("pavucontrol"), "{message}");
        let programs: Vec<&str> = settings_commands(true)
            .iter()
            .map(|(program, _)| *program)
            .collect();
        let per_platform = settings_launch_failure(&programs);
        for program in &programs {
            assert!(per_platform.contains(program), "{per_platform}");
        }
    }

    #[test]
    fn a_wrong_desktop_launcher_moves_on_but_a_running_one_counts_as_opened() {
        // #222: a successful `spawn` is never evidence on its own — the
        // decision is what `try_wait` shows inside the grace period.
        use CandidateProgress::{Exited, Running};
        // gnome-control-center outside GNOME: starts, exits non-zero at
        // once — the next candidate must get its turn.
        assert_eq!(
            candidate_verdict(Exited { success: false }),
            CandidateVerdict::TryNext
        );
        // A handing-off launcher runs to completion: opened.
        assert_eq!(
            candidate_verdict(Exited { success: true }),
            CandidateVerdict::Opened
        );
        // Still running once the wait ran out: opened (never killed).
        assert_eq!(
            candidate_verdict(Running { waited_out: true }),
            CandidateVerdict::Opened
        );
        // Still inside the wait: keep polling, decide nothing yet.
        assert_eq!(
            candidate_verdict(Running { waited_out: false }),
            CandidateVerdict::KeepWaiting
        );
    }

    #[test]
    fn a_non_zero_exit_at_the_deadline_still_fails_the_candidate() {
        // #222: the waited-out verdict must rest on a final `try_wait`,
        // not on the earlier "still running" — a launcher exiting
        // non-zero right at the deadline failed, so the next candidate
        // gets its turn; exit 0 or still running counts as opened.
        use CandidateProgress::{Exited, Running};
        assert_eq!(
            final_candidate_progress(Some(false)),
            Exited { success: false }
        );
        assert_eq!(
            final_candidate_progress(Some(true)),
            Exited { success: true }
        );
        assert_eq!(
            final_candidate_progress(None),
            Running { waited_out: true }
        );
        // Composition: only the non-zero exit at the deadline diverges
        // from the plain waited-out "opened".
        assert_eq!(
            candidate_verdict(final_candidate_progress(Some(false))),
            CandidateVerdict::TryNext
        );
        assert_eq!(
            candidate_verdict(final_candidate_progress(None)),
            CandidateVerdict::Opened
        );
    }

    #[test]
    fn a_window_counts_as_sound_at_or_above_the_silence_floor() {
        // #222: the timer loop stamps `last_sound_at` from its own
        // window through this threshold, so a hidden window's stopped
        // rendering can never read as the input's silence.
        assert!(!window_is_audible(&[]));
        assert!(!window_is_audible(&[0.0; 64]));
        assert!(window_is_audible(&[0.0, 0.5, 0.0]));
        assert!(window_is_audible(&[0.0, microphone::SILENT_PEAK]));
        assert!(!window_is_audible(&[0.0, microphone::SILENT_PEAK * 0.5]));
    }
}
