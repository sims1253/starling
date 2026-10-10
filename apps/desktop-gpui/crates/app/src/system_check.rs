//! The Linux setup check (#221): what this desktop session gives Starling
//! for system-wide dictation, and what to do about anything missing.
//!
//! Two halves. [`gather`] probes the live session — environment, the
//! desktop portal over D-Bus, the insertion backends, the AT-SPI bus,
//! input-method daemons, the sound server's sockets, the capture devices
//! — off the UI thread, every probe bounded by [`PROBE_TIMEOUT`].
//! [`evaluate`] turns those [`Facts`] into the lines Settings shows, each
//! with a concrete fix where one exists. It is pure, so the wording and
//! the verdicts are tested with injected facts.
//!
//! Not reported: the `input` group. It only matters to an evdev
//! shortcut fallback, and Starling has none.

use std::time::Duration;

use gpui::{AppContext, Context};

use crate::app::StarlingApp;
use crate::portal::PortalStatus;

/// The longest any single probe may take before it counts as failed.
pub(crate) const PROBE_TIMEOUT: Duration = Duration::from_secs(3);

/// How a check came out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Verdict {
    Ok,
    /// Works, with a limitation worth knowing.
    Limited,
    /// Missing: a feature will not work until it is fixed.
    Missing,
    /// For information only; nothing depends on it yet.
    Info,
}

/// One line of the check.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CheckLine {
    pub topic: &'static str,
    pub verdict: Verdict,
    pub summary: String,
    pub fix: Option<String>,
}

/// The session as the environment describes it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct SessionFacts {
    /// `XDG_SESSION_TYPE` (`wayland`, `x11`, `tty`).
    pub session_type: Option<String>,
    pub wayland_display: bool,
    pub x11_display: bool,
    /// `XDG_CURRENT_DESKTOP` (`KDE`, `GNOME`, `COSMIC`, `Hyprland`).
    pub desktop: Option<String>,
}

impl SessionFacts {
    pub(crate) fn wayland(&self) -> bool {
        self.wayland_display || self.session_type.as_deref() == Some("wayland")
    }
}

/// The desktop portal, as introspected.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct PortalFacts {
    /// The `org.freedesktop.portal.*` interfaces it exports.
    pub interfaces: Vec<String>,
    /// GlobalShortcuts' `version`, when the interface answers.
    pub global_shortcuts_version: Option<u32>,
}

impl PortalFacts {
    fn has(&self, interface: &str) -> bool {
        self.interfaces.iter().any(|name| name == interface)
    }
}

/// A portal backend installed on this machine (`*.portal` file).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct PortalBackend {
    /// The file's stem: `kde`, `gnome`, `hyprland`, `cosmic`.
    pub name: String,
    pub implements_global_shortcuts: bool,
    /// Whether its `UseIn` names the current desktop.
    pub used_here: bool,
}

/// Which input-method frameworks are running or configured.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct InputMethodFacts {
    pub ibus_running: bool,
    pub fcitx_running: bool,
    /// `GTK_IM_MODULE` / `QT_IM_MODULE` / `XMODIFIERS`, as set.
    pub configured: Vec<(String, String)>,
}

/// The sound server's sockets.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct AudioFacts {
    pub pipewire: bool,
    pub pulse: bool,
}

/// One insertion backend, as [`starling_insertion::Inserter::availability`]
/// reports it, in capture order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct InsertionFacts {
    /// The backend's ref scheme (`x11`, `wl`, `win`), or `unknown` when
    /// the probe gave no answer.
    pub scheme: String,
    /// Whether it can check the target window (Wayland cannot).
    pub verifies_target: bool,
    pub availability: Result<(), String>,
}

/// Everything [`evaluate`] needs; each probe's failure is its own `Err`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Facts {
    pub session: SessionFacts,
    pub portal: Option<Result<PortalFacts, String>>,
    pub portal_backends: Vec<PortalBackend>,
    /// The insertion backends, in capture order.
    pub insertion: Vec<InsertionFacts>,
    pub atspi: Option<Result<(), String>>,
    pub input_methods: InputMethodFacts,
    pub audio: Option<Result<AudioFacts, String>>,
    /// Capture devices: (name, is default).
    pub microphones: Option<Result<Vec<(String, bool)>, String>>,
}

const GLOBAL_SHORTCUTS: &str = "org.freedesktop.portal.GlobalShortcuts";
const REMOTE_DESKTOP: &str = "org.freedesktop.portal.RemoteDesktop";

/// The live shortcut sources the check reports alongside its facts.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ShortcutState<'a> {
    /// How registering the X11 (or other platform) system-wide shortcut
    /// went.
    pub registration: &'a Result<(), String>,
    /// The portal binding, when the portal source is running.
    pub portal: Option<&'a PortalStatus>,
}

/// The check's lines, in a fixed order.
pub(crate) fn evaluate(facts: &Facts, shortcut: ShortcutState<'_>) -> Vec<CheckLine> {
    vec![
        session_line(&facts.session),
        shortcut_line(facts, shortcut),
        insertion_line(facts),
        atspi_line(facts.atspi.as_ref()),
        input_method_line(&facts.input_methods),
        audio_line(facts.audio.as_ref()),
        microphone_line(facts.microphones.as_ref()),
    ]
}

fn line(
    topic: &'static str,
    verdict: Verdict,
    summary: impl Into<String>,
    fix: Option<String>,
) -> CheckLine {
    CheckLine {
        topic,
        verdict,
        summary: summary.into(),
        fix,
    }
}

fn desktop_name(session: &SessionFacts) -> String {
    session
        .desktop
        .clone()
        .filter(|name| !name.is_empty())
        .unwrap_or_else(|| "this desktop".to_string())
}

fn session_line(session: &SessionFacts) -> CheckLine {
    let desktop = desktop_name(session);
    if session.wayland() {
        if session.x11_display {
            line(
                "Session",
                Verdict::Ok,
                format!("Wayland ({desktop}) with XWayland."),
                None,
            )
        } else {
            line(
                "Session",
                Verdict::Limited,
                format!("Wayland ({desktop}) without XWayland (DISPLAY is not set)."),
                Some(
                    "The X11 shortcut grab and X11 typing need XWayland; enable it in your \
                     compositor's settings, or rely on the desktop shortcut portal."
                        .to_string(),
                ),
            )
        }
    } else if session.x11_display {
        line("Session", Verdict::Ok, format!("X11 ({desktop})."), None)
    } else {
        line(
            "Session",
            Verdict::Missing,
            "No graphical session found (neither WAYLAND_DISPLAY nor DISPLAY is set).",
            Some("Start Starling from inside your desktop session.".to_string()),
        )
    }
}

fn shortcut_line(facts: &Facts, shortcut: ShortcutState<'_>) -> CheckLine {
    let session = &facts.session;
    if !session.wayland() {
        return match shortcut.registration {
            Ok(()) => line(
                SHORTCUT_TOPIC,
                Verdict::Ok,
                "X11: the shortcut is grabbed directly; no portal needed.",
                None,
            ),
            Err(reason) => line(
                SHORTCUT_TOPIC,
                Verdict::Missing,
                format!("X11: the shortcut could not be grabbed ({reason})."),
                Some(
                    "Another app may hold these keys: choose a different recording shortcut \
                     above. Until then the shortcut works only in the Starling window."
                        .to_string(),
                ),
            ),
        };
    }
    let mut checked = wayland_shortcut_line(facts, shortcut.portal);
    // Short of a portal binding, the X11 grab is what reaches XWayland apps.
    if let (false, Err(reason)) = (
        shortcut.portal.is_some_and(PortalStatus::is_bound),
        shortcut.registration,
    ) {
        checked.summary.push_str(&format!(
            " The X11 grab for XWayland apps failed too ({reason})."
        ));
        if checked.verdict == Verdict::Limited {
            checked.verdict = Verdict::Missing;
        }
    }
    checked
}

const SHORTCUT_TOPIC: &str = "System-wide shortcut";

fn wayland_shortcut_line(facts: &Facts, portal_status: Option<&PortalStatus>) -> CheckLine {
    const TOPIC: &str = SHORTCUT_TOPIC;
    let session = &facts.session;
    let desktop = desktop_name(session);
    let portal = match facts.portal.as_ref() {
        None => {
            return line(
                TOPIC,
                Verdict::Info,
                "The desktop portal was not checked.",
                None,
            );
        }
        Some(Err(reason)) => {
            return line(
                TOPIC,
                Verdict::Missing,
                format!("The desktop portal is not reachable ({reason})."),
                Some(
                    "Install xdg-desktop-portal and your desktop's backend \
                     (xdg-desktop-portal-kde, -gnome or -hyprland) and log in again. Until \
                     then the shortcut works only in XWayland apps and in the Starling window."
                        .to_string(),
                ),
            );
        }
        Some(Ok(portal)) => portal,
    };
    let providers: Vec<&str> = facts
        .portal_backends
        .iter()
        .filter(|backend| backend.implements_global_shortcuts)
        .map(|backend| backend.name.as_str())
        .collect();
    let in_use: Vec<&str> = facts
        .portal_backends
        .iter()
        .filter(|backend| backend.implements_global_shortcuts && backend.used_here)
        .map(|backend| backend.name.as_str())
        .collect();
    let Some(version) = portal
        .global_shortcuts_version
        .or_else(|| portal.has(GLOBAL_SHORTCUTS).then_some(1))
    else {
        let installed = if providers.is_empty() {
            String::new()
        } else {
            format!(
                " Installed backends with it ({}) are not used on {desktop}.",
                providers.join(", ")
            )
        };
        return line(
            TOPIC,
            Verdict::Limited,
            format!("{desktop}'s desktop portal has no GlobalShortcuts.{installed}"),
            Some(format!(
                "On {desktop}, hold to talk works while an XWayland app or the Starling window \
                 is focused. Desktops whose portal offers GlobalShortcuts (KDE Plasma, GNOME 48 \
                 or newer, Hyprland with xdg-desktop-portal-hyprland) deliver it in every app."
            )),
        );
    };
    // Which backend serves it is the portal's choice (portals.conf can
    // pick any installed one), so these are the installed candidates.
    let backend = if in_use.is_empty() {
        format!("no installed backend with it names {desktop}")
    } else {
        format!("installed for {desktop}: {}", in_use.join(", "))
    };
    let summary = format!("GlobalShortcuts portal version {version} ({backend}).");
    match portal_status {
        Some(status @ PortalStatus::Bound { .. }) => line(
            TOPIC,
            Verdict::Ok,
            format!("{summary} {}", status.describe()),
            None,
        ),
        Some(PortalStatus::Unavailable(reason)) => line(
            TOPIC,
            Verdict::Missing,
            format!("{summary} Starling could not use it: {reason}."),
            Some(if reason.contains("1.19") {
                "Update xdg-desktop-portal to 1.19 or newer, then restart Starling.".to_string()
            } else {
                "Check that xdg-desktop-portal and your desktop's backend are running \
                 (`systemctl --user status xdg-desktop-portal`), then restart Starling."
                    .to_string()
            }),
        ),
        Some(status) if status.can_set_up() => line(
            TOPIC,
            Verdict::Limited,
            format!("{summary} Not set up for Starling yet."),
            Some(
                "Use \"Set up desktop shortcut\" above and confirm the keys in your desktop's \
                 dialog."
                    .to_string(),
            ),
        ),
        Some(status) => line(
            TOPIC,
            Verdict::Info,
            format!("{summary} {}", status.describe()),
            None,
        ),
        None => line(TOPIC, Verdict::Info, summary, None),
    }
}

fn insertion_line(facts: &Facts) -> CheckLine {
    const TOPIC: &str = "Typing into other apps";
    const WAYLAND_FIX: &str = "Typing into native Wayland apps needs a compositor that offers \
         virtual keyboards (wlroots-based ones, niri, COSMIC; not GNOME or KDE). Elsewhere, \
         copy the transcript and paste it yourself.";
    let blocked: Vec<String> = facts
        .insertion
        .iter()
        .filter_map(|backend| {
            backend
                .availability
                .as_ref()
                .err()
                .map(|reason| format!("{}: {reason}", backend.scheme))
        })
        .collect();
    let remote_desktop = facts
        .portal
        .as_ref()
        .and_then(|portal| portal.as_ref().ok())
        .is_some_and(|portal| portal.has(REMOTE_DESKTOP));
    let portal_note = if remote_desktop {
        " The desktop's RemoteDesktop portal is present; Starling does not type through it yet."
    } else {
        ""
    };
    if facts.insertion.is_empty() {
        return line(
            TOPIC,
            Verdict::Missing,
            format!("No insertion backend exists for this platform.{portal_note}"),
            Some("Copy the transcript and paste it yourself.".to_string()),
        );
    }
    // Capture takes the first available backend in this order.
    let Some(used) = facts
        .insertion
        .iter()
        .find(|backend| backend.availability.is_ok())
    else {
        return line(
            TOPIC,
            Verdict::Missing,
            format!(
                "No insertion backend is available ({}).{portal_note}",
                blocked.join("; ")
            ),
            Some(if facts.session.wayland() {
                WAYLAND_FIX.to_string()
            } else {
                "Copy the transcript and paste it yourself.".to_string()
            }),
        );
    };
    let available: Vec<&str> = facts
        .insertion
        .iter()
        .filter(|backend| backend.availability.is_ok())
        .map(|backend| backend.scheme.as_str())
        .collect();
    if facts.session.wayland() && used.scheme == "x11" {
        let not_wayland = if blocked.is_empty() {
            String::new()
        } else {
            format!(" ({})", blocked.join("; "))
        };
        line(
            TOPIC,
            Verdict::Limited,
            format!(
                "X11 typing only{not_wayland}: it reaches XWayland apps, not native Wayland \
                 ones.{portal_note}"
            ),
            Some(WAYLAND_FIX.to_string()),
        )
    } else if !used.verifies_target {
        line(
            TOPIC,
            Verdict::Limited,
            format!(
                "Types through the compositor's virtual keyboard ({}), which cannot check \
                 which window has focus.{portal_note}",
                used.scheme
            ),
            Some(
                "Turn on \"Also type where the window cannot be checked\" under Settings → \
                 After a take; without it, takes are left for you to copy."
                    .to_string(),
            ),
        )
    } else {
        line(
            TOPIC,
            Verdict::Ok,
            format!("Available: {}.{portal_note}", available.join(", ")),
            None,
        )
    }
}

fn atspi_line(atspi: Option<&Result<(), String>>) -> CheckLine {
    const TOPIC: &str = "Accessibility bus (AT-SPI)";
    match atspi {
        Some(Ok(())) => line(TOPIC, Verdict::Ok, "Reachable.", None),
        Some(Err(reason)) => line(
            TOPIC,
            Verdict::Limited,
            format!("Not reachable ({reason}). Nothing in Starling needs it yet."),
            Some(
                "Install at-spi2-core; on GNOME also run `gsettings set \
                 org.gnome.desktop.interface toolkit-accessibility true`."
                    .to_string(),
            ),
        ),
        None => line(TOPIC, Verdict::Info, "Not checked.", None),
    }
}

fn input_method_line(ime: &InputMethodFacts) -> CheckLine {
    let mut running = Vec::new();
    if ime.ibus_running {
        running.push("IBus");
    }
    if ime.fcitx_running {
        running.push("Fcitx 5");
    }
    let configured = if ime.configured.is_empty() {
        String::new()
    } else {
        let pairs: Vec<String> = ime
            .configured
            .iter()
            .map(|(name, value)| format!("{name}={value}"))
            .collect();
        format!(" ({})", pairs.join(", "))
    };
    let summary = if running.is_empty() {
        format!("No IBus or Fcitx running{configured}.")
    } else {
        format!("{} running{configured}.", running.join(" and "))
    };
    line(
        "Input method",
        Verdict::Info,
        format!("{summary} For information: Starling has no input-method engine yet."),
        None,
    )
}

fn audio_line(audio: Option<&Result<AudioFacts, String>>) -> CheckLine {
    const TOPIC: &str = "Sound server";
    let audio = match audio {
        None => return line(TOPIC, Verdict::Info, "Not checked.", None),
        Some(Err(reason)) => {
            return line(
                TOPIC,
                Verdict::Missing,
                format!("The sound server's sockets could not be checked ({reason})."),
                Some("Restart it: `systemctl --user restart pipewire pipewire-pulse`.".to_string()),
            );
        }
        Some(Ok(audio)) => audio,
    };
    match (audio.pipewire, audio.pulse) {
        (true, true) => line(
            TOPIC,
            Verdict::Ok,
            "PipeWire with its PulseAudio socket.",
            None,
        ),
        (false, true) => line(TOPIC, Verdict::Ok, "PulseAudio.", None),
        (true, false) => line(
            TOPIC,
            Verdict::Limited,
            "PipeWire without its PulseAudio socket: microphone routing falls back to ALSA \
             device names.",
            Some(
                "Install and start pipewire-pulse: `systemctl --user enable --now pipewire-pulse`."
                    .to_string(),
            ),
        ),
        // Capture still works through ALSA directly; the microphone line
        // says whether any device is visible.
        (false, false) => line(
            TOPIC,
            Verdict::Limited,
            "Neither PipeWire nor PulseAudio answers: capture uses ALSA devices directly.",
            Some(
                "Start the sound server: `systemctl --user start pipewire pipewire-pulse` (or \
                 `pulseaudio --start`)."
                    .to_string(),
            ),
        ),
    }
}

fn microphone_line(microphones: Option<&Result<Vec<(String, bool)>, String>>) -> CheckLine {
    const TOPIC: &str = "Microphone";
    let fix = Some(
        "Connect a microphone and check it is enabled and not muted in your sound settings. \
         Linux has no per-app microphone permission for apps outside a sandbox."
            .to_string(),
    );
    match microphones {
        None => line(TOPIC, Verdict::Info, "Not checked.", None),
        Some(Err(reason)) => line(
            TOPIC,
            Verdict::Missing,
            format!("Capture devices could not be listed ({reason})."),
            fix,
        ),
        Some(Ok(devices)) if devices.is_empty() => line(
            TOPIC,
            Verdict::Missing,
            "No capture device is visible.",
            fix,
        ),
        Some(Ok(devices)) => {
            let default = devices
                .iter()
                .find(|(_, is_default)| *is_default)
                .map(|(name, _)| format!("; default: {name}"))
                .unwrap_or_default();
            let count = devices.len();
            let noun = if count == 1 { "device" } else { "devices" };
            line(
                TOPIC,
                Verdict::Ok,
                format!("{count} capture {noun} visible{default}."),
                None,
            )
        }
    }
}

/// Where PipeWire's socket is: `PIPEWIRE_REMOTE` is a socket *name*
/// (default `pipewire-0`) resolved in `PIPEWIRE_RUNTIME_DIR`, else
/// `XDG_RUNTIME_DIR`; an absolute path is taken as is.
fn pipewire_socket(
    remote: Option<&str>,
    pipewire_runtime: Option<&str>,
    xdg_runtime: Option<&std::path::Path>,
) -> Option<std::path::PathBuf> {
    let name = remote.unwrap_or("pipewire-0");
    if name.starts_with('/') {
        return Some(name.into());
    }
    pipewire_runtime
        .map(std::path::PathBuf::from)
        .or_else(|| xdg_runtime.map(std::path::Path::to_path_buf))
        .map(|dir| dir.join(name))
}

/// The check as plain text (for copying into a bug report).
pub(crate) fn report(lines: &[CheckLine]) -> String {
    let mut text = String::new();
    for line in lines {
        let mark = match line.verdict {
            Verdict::Ok => "ok",
            Verdict::Limited => "limited",
            Verdict::Missing => "missing",
            Verdict::Info => "info",
        };
        text.push_str(&format!("[{mark}] {}: {}\n", line.topic, line.summary));
        if let Some(fix) = &line.fix {
            text.push_str(&format!("        fix: {fix}\n"));
        }
    }
    text
}

/// Parse a `*.portal` file: whether it implements GlobalShortcuts and
/// whether its `UseIn` names one of `desktops` (case-insensitive).
pub(crate) fn parse_portal_file(name: &str, contents: &str, desktops: &[String]) -> PortalBackend {
    let mut backend = PortalBackend {
        name: name.to_string(),
        ..Default::default()
    };
    for raw in contents.lines() {
        let Some((key, value)) = raw.trim().split_once('=') else {
            continue;
        };
        let items = || {
            value
                .split(';')
                .map(str::trim)
                .filter(|item| !item.is_empty())
        };
        match key.trim() {
            "Interfaces" => {
                backend.implements_global_shortcuts =
                    items().any(|item| item == "org.freedesktop.impl.portal.GlobalShortcuts");
            }
            "UseIn" => {
                backend.used_here = items().any(|item| {
                    desktops
                        .iter()
                        .any(|desktop| desktop.eq_ignore_ascii_case(item))
                });
            }
            _ => {}
        }
    }
    backend
}

/// Run `probe` on its own thread. `Err` says why it gave no answer: it
/// took longer than [`PROBE_TIMEOUT`] (the thread is left to finish on
/// its own), or it crashed.
fn bounded<T: Send + 'static>(probe: impl FnOnce() -> T + Send + 'static) -> Result<T, String> {
    use std::sync::mpsc::RecvTimeoutError;
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::Builder::new()
        .name("starling-check".to_string())
        .spawn(move || {
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(probe));
            let _ = sender.send(outcome.map_err(|panic| panic_message(&*panic)));
        })
        .map_err(|err| format!("the probe could not start ({err})"))?;
    match receiver.recv_timeout(PROBE_TIMEOUT) {
        Ok(Ok(answer)) => Ok(answer),
        Ok(Err(message)) => Err(format!("the probe failed: {message}")),
        Err(RecvTimeoutError::Timeout) => {
            Err(format!("no answer within {} s", PROBE_TIMEOUT.as_secs()))
        }
        Err(RecvTimeoutError::Disconnected) => Err("the probe failed".to_string()),
    }
}

fn panic_message(panic: &(dyn std::any::Any + Send)) -> String {
    panic
        .downcast_ref::<&str>()
        .map(|text| text.to_string())
        .or_else(|| panic.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "it panicked".to_string())
}

/// A bounded probe that itself can fail: either failure is the reason.
fn answered<T>(outcome: Result<Result<T, String>, String>) -> Result<T, String> {
    outcome.and_then(|answer| answer)
}

fn env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|value| !value.is_empty())
}

/// Probe the live session. Blocking; the probes run in parallel, each
/// bounded by [`PROBE_TIMEOUT`].
#[cfg(target_os = "linux")]
pub(crate) fn gather() -> Facts {
    let session = SessionFacts {
        session_type: env("XDG_SESSION_TYPE"),
        wayland_display: env("WAYLAND_DISPLAY").is_some(),
        x11_display: env("DISPLAY").is_some(),
        desktop: env("XDG_CURRENT_DESKTOP"),
    };
    let desktops: Vec<String> = session
        .desktop
        .as_deref()
        .unwrap_or_default()
        .split(':')
        .map(str::to_string)
        .collect();

    let portal = std::thread::spawn(|| answered(bounded(live::portal)));
    let atspi = std::thread::spawn(|| answered(bounded(live::atspi)));
    let names = std::thread::spawn(|| bounded(live::input_method_names).unwrap_or_default());
    let insertion = std::thread::spawn(|| {
        bounded(|| {
            starling_insertion::Inserter::for_this_session()
                .availability()
                .into_iter()
                .map(|backend| InsertionFacts {
                    scheme: backend.kind.scheme().to_string(),
                    verifies_target: backend.verifies_target,
                    availability: backend.availability.map_err(|err| err.message()),
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(unknown_insertion)
    });
    let microphones = std::thread::spawn(|| {
        answered(bounded(|| {
            starling_dictation::microphone::list_input_devices().map(|devices| {
                devices
                    .into_iter()
                    .map(|device| (device.name, device.is_default))
                    .collect::<Vec<_>>()
            })
        }))
    });

    let (ibus_running, fcitx_running) = names.join().unwrap_or_default();
    let configured = ["GTK_IM_MODULE", "QT_IM_MODULE", "XMODIFIERS"]
        .into_iter()
        .filter_map(|name| env(name).map(|value| (name.to_string(), value)))
        .collect();
    let backends = std::thread::spawn(move || {
        bounded(move || live::portal_backends(&desktops)).unwrap_or_default()
    });
    let audio = bounded(live::audio);
    let failed = || "the probe failed".to_string();
    Facts {
        portal_backends: backends.join().unwrap_or_default(),
        session,
        portal: Some(portal.join().unwrap_or_else(|_| Err(failed()))),
        insertion: insertion
            .join()
            .unwrap_or_else(|_| unknown_insertion(failed())),
        atspi: Some(atspi.join().unwrap_or_else(|_| Err(failed()))),
        input_methods: InputMethodFacts {
            ibus_running,
            fcitx_running,
            configured,
        },
        audio: Some(audio),
        microphones: Some(microphones.join().unwrap_or_else(|_| Err(failed()))),
    }
}

/// The insertion probe gave no answer: which backends exist is unknown
/// too, so none is named.
fn unknown_insertion(reason: String) -> Vec<InsertionFacts> {
    vec![InsertionFacts {
        scheme: "unknown".to_string(),
        verifies_target: false,
        availability: Err(reason),
    }]
}

#[cfg(target_os = "linux")]
mod live {
    use std::collections::HashMap;
    use std::path::PathBuf;

    use zbus::blocking::{Connection, Proxy};

    use super::{
        AudioFacts, GLOBAL_SHORTCUTS, PortalBackend, PortalFacts, env, parse_portal_file,
        pipewire_socket,
    };

    const PORTAL: &str = "org.freedesktop.portal.Desktop";
    const PORTAL_PATH: &str = "/org/freedesktop/portal/desktop";

    pub(super) fn portal() -> Result<PortalFacts, String> {
        let connection = Connection::session().map_err(|err| format!("no session bus ({err})"))?;
        let introspectable = Proxy::new(
            &connection,
            PORTAL,
            PORTAL_PATH,
            "org.freedesktop.DBus.Introspectable",
        )
        .map_err(|err| err.to_string())?;
        let xml: String = introspectable
            .call("Introspect", &())
            .map_err(|err| format!("xdg-desktop-portal does not answer ({err})"))?;
        let interfaces = interface_names(&xml);
        let global_shortcuts_version = interfaces
            .iter()
            .any(|name| name == GLOBAL_SHORTCUTS)
            .then(|| {
                let properties = Proxy::new(
                    &connection,
                    PORTAL,
                    PORTAL_PATH,
                    "org.freedesktop.DBus.Properties",
                )
                .ok()?;
                let value: zbus::zvariant::OwnedValue = properties
                    .call("Get", &(GLOBAL_SHORTCUTS, "version"))
                    .ok()?;
                u32::try_from(value).ok()
            })
            .flatten();
        Ok(PortalFacts {
            interfaces,
            global_shortcuts_version,
        })
    }

    /// The `org.freedesktop.portal.*` interface names in introspection XML.
    pub(super) fn interface_names(xml: &str) -> Vec<String> {
        xml.split("<interface name=\"")
            .skip(1)
            .filter_map(|rest| rest.split('"').next())
            .filter(|name| name.starts_with("org.freedesktop.portal."))
            .map(str::to_string)
            .collect()
    }

    pub(super) fn atspi() -> Result<(), String> {
        let connection = Connection::session().map_err(|err| format!("no session bus ({err})"))?;
        let bus = Proxy::new(&connection, "org.a11y.Bus", "/org/a11y/bus", "org.a11y.Bus")
            .map_err(|err| err.to_string())?;
        let address: String = bus
            .call("GetAddress", &())
            .map_err(|err| format!("org.a11y.Bus does not answer ({err})"))?;
        zbus::blocking::connection::Builder::address(address.as_str())
            .and_then(|builder| builder.build())
            .map(drop)
            .map_err(|err| format!("its bus at {address} refused the connection ({err})"))
    }

    /// Whether IBus and Fcitx 5 own their names on the session bus.
    pub(super) fn input_method_names() -> (bool, bool) {
        let Ok(connection) = Connection::session() else {
            return (false, false);
        };
        let Ok(bus) = zbus::blocking::fdo::DBusProxy::new(&connection) else {
            return (false, false);
        };
        let owned = |name: &str| {
            zbus::names::BusName::try_from(name)
                .ok()
                .and_then(|name| bus.name_has_owner(name).ok())
                .unwrap_or(false)
        };
        (
            owned("org.freedesktop.IBus") || owned("org.freedesktop.portal.IBus"),
            owned("org.fcitx.Fcitx5"),
        )
    }

    pub(super) fn portal_backends(desktops: &[String]) -> Vec<PortalBackend> {
        let mut dirs: Vec<PathBuf> = Vec::new();
        if let Some(dir) = env("XDG_DESKTOP_PORTAL_DIR") {
            dirs.push(dir.into());
        }
        let data_dirs =
            env("XDG_DATA_DIRS").unwrap_or_else(|| "/usr/local/share:/usr/share".to_string());
        dirs.extend(
            data_dirs
                .split(':')
                .filter(|dir| !dir.is_empty())
                .map(|dir| PathBuf::from(dir).join("xdg-desktop-portal/portals")),
        );
        let mut seen: HashMap<String, PortalBackend> = HashMap::new();
        for dir in dirs {
            let Ok(entries) = std::fs::read_dir(&dir) else {
                continue;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|ext| ext.to_str()) != Some("portal") {
                    continue;
                }
                let Some(name) = path.file_stem().and_then(|stem| stem.to_str()) else {
                    continue;
                };
                if seen.contains_key(name) {
                    continue;
                }
                if let Ok(contents) = std::fs::read_to_string(&path) {
                    seen.insert(
                        name.to_string(),
                        parse_portal_file(name, &contents, desktops),
                    );
                }
            }
        }
        let mut backends: Vec<PortalBackend> = seen.into_values().collect();
        backends.sort_by(|a, b| a.name.cmp(&b.name));
        backends
    }

    pub(super) fn audio() -> AudioFacts {
        let runtime = env("XDG_RUNTIME_DIR").map(PathBuf::from);
        let reachable = |path: Option<PathBuf>| {
            path.is_some_and(|path| std::os::unix::net::UnixStream::connect(path).is_ok())
        };
        let pulse_path = match env("PULSE_SERVER") {
            Some(server) => server.strip_prefix("unix:").map(PathBuf::from),
            None => runtime.as_ref().map(|dir| dir.join("pulse/native")),
        };
        AudioFacts {
            pipewire: reachable(pipewire_socket(
                env("PIPEWIRE_REMOTE").as_deref(),
                env("PIPEWIRE_RUNTIME_DIR").as_deref(),
                runtime.as_deref(),
            )),
            pulse: reachable(pulse_path),
        }
    }
}

#[cfg(not(target_os = "linux"))]
pub(crate) fn gather() -> Facts {
    Facts::default()
}

/// The check's state in the app: the last result and whether one runs.
#[derive(Default)]
pub(crate) struct SystemCheck {
    pub facts: Option<Facts>,
    pub running: bool,
}

impl StarlingApp {
    /// Run the setup check off the UI thread; the panel repaints when the
    /// facts arrive. A second request while one runs is dropped.
    pub(crate) fn run_system_check(&mut self, cx: &mut Context<Self>) {
        if self.system_check.running {
            return;
        }
        self.system_check.running = true;
        cx.notify();
        let probing = cx.background_spawn(async move { gather() });
        cx.spawn(async move |this, cx| {
            let facts = probing.await;
            let _ = this.update(cx, |app, cx| {
                app.system_check.facts = Some(facts);
                app.system_check.running = false;
                cx.notify();
            });
        })
        .detach();
    }

    /// The check's lines for the last result, with the live portal status.
    pub(crate) fn system_check_lines(&self) -> Option<Vec<CheckLine>> {
        let facts = self.system_check.facts.as_ref()?;
        Some(evaluate(
            facts,
            ShortcutState {
                registration: &self.shortcut_registration,
                portal: self.portal_shortcuts.as_ref().map(|portal| portal.status()),
            },
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wayland(desktop: &str, xwayland: bool) -> SessionFacts {
        SessionFacts {
            session_type: Some("wayland".to_string()),
            wayland_display: true,
            x11_display: xwayland,
            desktop: Some(desktop.to_string()),
        }
    }

    fn healthy_kde() -> Facts {
        Facts {
            session: wayland("KDE", true),
            portal: Some(Ok(PortalFacts {
                interfaces: vec![GLOBAL_SHORTCUTS.to_string(), REMOTE_DESKTOP.to_string()],
                global_shortcuts_version: Some(2),
            })),
            portal_backends: vec![PortalBackend {
                name: "kde".to_string(),
                implements_global_shortcuts: true,
                used_here: true,
            }],
            insertion: vec![x11("x11", Ok(()))],
            atspi: Some(Ok(())),
            input_methods: InputMethodFacts::default(),
            audio: Some(Ok(AudioFacts {
                pipewire: true,
                pulse: true,
            })),
            microphones: Some(Ok(vec![
                ("Built-in".to_string(), false),
                ("USB mic".to_string(), true),
            ])),
        }
    }

    fn x11(name: &str, availability: Result<(), String>) -> InsertionFacts {
        InsertionFacts {
            scheme: name.to_string(),
            verifies_target: true,
            availability,
        }
    }

    fn wl(availability: Result<(), String>) -> InsertionFacts {
        InsertionFacts {
            scheme: "wl".to_string(),
            verifies_target: false,
            availability,
        }
    }

    /// The check with the X11 shortcut registered.
    fn check(facts: &Facts, portal: Option<&PortalStatus>) -> Vec<CheckLine> {
        evaluate(
            facts,
            ShortcutState {
                registration: &Ok(()),
                portal,
            },
        )
    }

    fn find<'a>(lines: &'a [CheckLine], topic: &str) -> &'a CheckLine {
        lines.iter().find(|line| line.topic == topic).expect(topic)
    }

    #[test]
    fn a_bound_portal_on_kde_reads_ok_with_its_keys_and_backend() {
        let status = PortalStatus::Bound {
            trigger: Some("Meta+Space".to_string()),
            configurable: true,
        };
        let lines = check(&healthy_kde(), Some(&status));
        let shortcut = find(&lines, "System-wide shortcut");
        assert_eq!(shortcut.verdict, Verdict::Ok);
        assert!(
            shortcut.summary.contains("version 2"),
            "{}",
            shortcut.summary
        );
        assert!(
            shortcut.summary.contains("installed for KDE: kde"),
            "{}",
            shortcut.summary
        );
        assert!(
            shortcut.summary.contains("Meta+Space"),
            "{}",
            shortcut.summary
        );
        assert_eq!(find(&lines, "Session").verdict, Verdict::Ok);
        let mic = find(&lines, "Microphone");
        assert_eq!(mic.summary, "2 capture devices visible; default: USB mic.");
        assert_eq!(find(&lines, "Sound server").verdict, Verdict::Ok);
    }

    #[test]
    fn cosmic_without_global_shortcuts_says_where_the_shortcut_still_works() {
        let mut facts = healthy_kde();
        facts.session = wayland("COSMIC", true);
        facts.portal = Some(Ok(PortalFacts {
            interfaces: vec![REMOTE_DESKTOP.to_string()],
            global_shortcuts_version: None,
        }));
        facts.portal_backends = vec![
            PortalBackend {
                name: "cosmic".to_string(),
                implements_global_shortcuts: false,
                used_here: true,
            },
            PortalBackend {
                name: "kde".to_string(),
                implements_global_shortcuts: true,
                used_here: false,
            },
        ];
        let lines = check(&facts, Some(&PortalStatus::Unavailable("x".into())));
        let shortcut = find(&lines, "System-wide shortcut");
        assert_eq!(shortcut.verdict, Verdict::Limited);
        assert!(
            shortcut
                .summary
                .contains("COSMIC's desktop portal has no GlobalShortcuts")
        );
        assert!(
            shortcut.summary.contains("(kde) are not used on COSMIC"),
            "{}",
            shortcut.summary
        );
        assert!(shortcut.fix.as_deref().unwrap().contains("XWayland app"));
        let typing = find(&lines, "Typing into other apps");
        assert_eq!(typing.verdict, Verdict::Limited);
        assert!(typing.summary.contains("RemoteDesktop portal is present"));
    }

    #[test]
    fn x11_needs_no_portal_and_a_missing_portal_on_wayland_is_named() {
        let mut facts = healthy_kde();
        facts.session = SessionFacts {
            session_type: Some("x11".to_string()),
            wayland_display: false,
            x11_display: true,
            desktop: Some("XFCE".to_string()),
        };
        let lines = check(&facts, None);
        assert_eq!(find(&lines, "Session").summary, "X11 (XFCE).");
        assert_eq!(find(&lines, "System-wide shortcut").verdict, Verdict::Ok);
        assert_eq!(find(&lines, "Typing into other apps").verdict, Verdict::Ok);

        let mut facts = healthy_kde();
        facts.portal = Some(Err("xdg-desktop-portal does not answer".to_string()));
        let shortcut = find(&check(&facts, None), "System-wide shortcut").clone();
        assert_eq!(shortcut.verdict, Verdict::Missing);
        assert!(shortcut.fix.unwrap().contains("xdg-desktop-portal-kde"));
    }

    #[test]
    fn an_unusable_portal_gets_advice_for_its_reason() {
        let fix = |reason: &str| {
            let status = PortalStatus::Unavailable(reason.to_string());
            find(
                &check(&healthy_kde(), Some(&status)),
                "System-wide shortcut",
            )
            .fix
            .clone()
            .unwrap()
        };
        assert!(
            fix(
                "this xdg-desktop-portal cannot identify Starling (its host app registry needs \
                 version 1.19 or newer)"
            )
            .starts_with("Update xdg-desktop-portal to 1.19")
        );
        let other = fix("the desktop portal did not answer");
        assert!(other.contains("systemctl --user status"), "{other}");
        assert!(!other.contains("1.19"));
    }

    #[test]
    fn a_portal_waiting_for_setup_points_at_the_button() {
        let lines = check(
            &healthy_kde(),
            Some(&PortalStatus::NeedsSetup { configurable: true }),
        );
        let shortcut = find(&lines, "System-wide shortcut");
        assert_eq!(shortcut.verdict, Verdict::Limited);
        assert!(
            shortcut
                .fix
                .as_deref()
                .unwrap()
                .contains("Set up desktop shortcut")
        );
    }

    #[test]
    fn wayland_without_xwayland_or_insertion_backends_says_so() {
        let mut facts = healthy_kde();
        facts.session = wayland("sway", false);
        facts.insertion = vec![
            wl(Err(
                "the compositor does not offer virtual keyboards".to_string()
            )),
            x11("x11", Err("no X11 display".to_string())),
        ];
        let lines = check(&facts, None);
        assert_eq!(find(&lines, "Session").verdict, Verdict::Limited);
        let typing = find(&lines, "Typing into other apps");
        assert_eq!(typing.verdict, Verdict::Missing);
        assert!(
            typing.summary.contains("x11: no X11 display"),
            "{}",
            typing.summary
        );
        assert!(typing.summary.contains("wl: the compositor does not offer"));
        assert!(typing.fix.as_deref().unwrap().contains("virtual keyboards"));
    }

    #[test]
    fn the_wayland_virtual_keyboard_is_reported_as_unverified_typing() {
        let mut facts = healthy_kde();
        facts.session = wayland("niri", true);
        facts.insertion = vec![wl(Ok(())), x11("x11", Ok(()))];
        let typing = find(&check(&facts, None), "Typing into other apps").clone();
        assert_eq!(typing.verdict, Verdict::Limited);
        assert!(
            typing.summary.contains("virtual keyboard (wl)"),
            "{}",
            typing.summary
        );
        assert!(
            typing
                .fix
                .unwrap()
                .contains("Also type where the window cannot be checked")
        );

        // A compositor without it leaves X11 typing, and says why.
        facts.insertion = vec![
            wl(Err(
                "the compositor does not offer virtual keyboards".to_string()
            )),
            x11("x11", Ok(())),
        ];
        let typing = find(&check(&facts, None), "Typing into other apps").clone();
        assert_eq!(typing.verdict, Verdict::Limited);
        assert!(
            typing
                .summary
                .starts_with("X11 typing only (wl: the compositor")
        );
        assert!(typing.fix.unwrap().contains("not GNOME or KDE"));
    }

    #[test]
    fn an_insertion_probe_without_an_answer_names_no_backend() {
        let mut facts = healthy_kde();
        facts.insertion = unknown_insertion("no answer within 3 s".to_string());
        let typing = find(&check(&facts, None), "Typing into other apps").clone();
        assert_eq!(typing.verdict, Verdict::Missing);
        assert!(
            typing.summary.contains("(unknown: no answer within 3 s)"),
            "{}",
            typing.summary
        );
        assert!(!typing.summary.contains("x11"));
    }

    #[test]
    fn a_failed_x11_grab_is_reported() {
        let mut facts = healthy_kde();
        facts.session = SessionFacts {
            session_type: Some("x11".to_string()),
            wayland_display: false,
            x11_display: true,
            desktop: Some("XFCE".to_string()),
        };
        let failed = Err("F9 is taken by another app".to_string());
        let lines = evaluate(
            &facts,
            ShortcutState {
                registration: &failed,
                portal: None,
            },
        );
        let shortcut = find(&lines, "System-wide shortcut");
        assert_eq!(shortcut.verdict, Verdict::Missing);
        assert!(
            shortcut.summary.contains("F9 is taken"),
            "{}",
            shortcut.summary
        );
        assert!(shortcut.fix.is_some());

        // On Wayland it is the XWayland fallback that fails, short of a
        // portal binding.
        let lines = evaluate(
            &healthy_kde(),
            ShortcutState {
                registration: &failed,
                portal: Some(&PortalStatus::NeedsSetup { configurable: true }),
            },
        );
        let shortcut = find(&lines, "System-wide shortcut");
        assert_eq!(shortcut.verdict, Verdict::Missing);
        assert!(
            shortcut
                .summary
                .contains("X11 grab for XWayland apps failed")
        );
        let bound = PortalStatus::Bound {
            trigger: None,
            configurable: true,
        };
        let lines = evaluate(
            &healthy_kde(),
            ShortcutState {
                registration: &failed,
                portal: Some(&bound),
            },
        );
        assert_eq!(find(&lines, "System-wide shortcut").verdict, Verdict::Ok);
    }

    #[test]
    fn a_crashing_probe_is_an_error_not_a_timeout() {
        let crashed = bounded(|| -> Result<(), String> { panic!("no bus") });
        assert_eq!(crashed, Err("the probe failed: no bus".to_string()));
        assert_eq!(answered(bounded(|| Ok::<_, String>(7))), Ok(7));
        assert_eq!(
            answered(bounded(|| Err::<(), _>("refused".to_string()))),
            Err("refused".to_string())
        );
    }

    #[test]
    fn missing_audio_microphones_and_atspi_carry_fixes() {
        let mut facts = healthy_kde();
        facts.audio = Some(Ok(AudioFacts::default()));
        facts.microphones = Some(Ok(Vec::new()));
        facts.atspi = Some(Err("org.a11y.Bus does not answer".to_string()));
        let lines = check(&facts, None);
        let audio = find(&lines, "Sound server");
        // ALSA still captures without a sound server.
        assert_eq!(audio.verdict, Verdict::Limited);
        assert!(audio.summary.contains("ALSA"));
        assert!(
            audio
                .fix
                .as_deref()
                .unwrap()
                .contains("systemctl --user start pipewire")
        );
        let mic = find(&lines, "Microphone");
        assert_eq!(mic.verdict, Verdict::Missing);
        assert!(mic.fix.is_some());
        let atspi = find(&lines, "Accessibility bus (AT-SPI)");
        assert_eq!(atspi.verdict, Verdict::Limited);
        assert!(atspi.fix.as_deref().unwrap().contains("at-spi2-core"));
    }

    #[test]
    fn a_sound_server_that_never_answers_is_named() {
        let mut facts = healthy_kde();
        facts.audio = Some(Err("no answer within 3 s".to_string()));
        let audio = find(&check(&facts, None), "Sound server").clone();
        assert_eq!(audio.verdict, Verdict::Missing);
        assert!(
            audio.summary.contains("no answer within 3 s"),
            "{}",
            audio.summary
        );
    }

    #[test]
    fn the_pipewire_socket_name_resolves_in_its_runtime_dir() {
        let xdg = std::path::Path::new("/run/user/1000");
        assert_eq!(
            pipewire_socket(None, None, Some(xdg)),
            Some(xdg.join("pipewire-0"))
        );
        assert_eq!(
            pipewire_socket(Some("pipewire-1"), Some("/tmp/pw"), Some(xdg)),
            Some("/tmp/pw/pipewire-1".into())
        );
        assert_eq!(
            pipewire_socket(Some("/srv/pw.sock"), None, Some(xdg)),
            Some("/srv/pw.sock".into())
        );
        assert_eq!(pipewire_socket(None, None, None), None);
    }

    #[test]
    fn input_methods_are_informational() {
        let mut facts = healthy_kde();
        facts.input_methods = InputMethodFacts {
            ibus_running: true,
            fcitx_running: false,
            configured: vec![("GTK_IM_MODULE".to_string(), "ibus".to_string())],
        };
        let line = find(&check(&facts, None), "Input method").clone();
        assert_eq!(line.verdict, Verdict::Info);
        assert!(
            line.summary
                .starts_with("IBus running (GTK_IM_MODULE=ibus).")
        );
        assert!(line.summary.contains("no input-method engine yet"));
    }

    #[test]
    fn the_report_lists_every_line_with_its_fix() {
        let mut facts = healthy_kde();
        facts.audio = Some(Ok(AudioFacts::default()));
        let text = report(&check(&facts, None));
        assert_eq!(text.lines().filter(|line| line.starts_with('[')).count(), 7);
        assert!(text.contains("[limited] Sound server: Neither PipeWire nor PulseAudio answers"));
        assert!(text.contains("        fix: Start the sound server"));
    }

    #[test]
    fn portal_files_are_matched_to_the_current_desktop() {
        let kde = "[portal]\nDBusName=org.freedesktop.impl.portal.desktop.kde\n\
                   Interfaces=org.freedesktop.impl.portal.FileChooser;org.freedesktop.impl.portal.GlobalShortcuts;\n\
                   UseIn=KDE\n";
        let desktops = vec!["kde".to_string()];
        let parsed = parse_portal_file("kde", kde, &desktops);
        assert!(parsed.implements_global_shortcuts && parsed.used_here);
        let parsed = parse_portal_file("kde", kde, &["COSMIC".to_string()]);
        assert!(parsed.implements_global_shortcuts && !parsed.used_here);
        let gtk = "Interfaces=org.freedesktop.impl.portal.FileChooser;\nUseIn=gnome\n";
        assert!(!parse_portal_file("gtk", gtk, &desktops).implements_global_shortcuts);
    }

    /// The probes against the session this test runs in (read-only).
    #[cfg(target_os = "linux")]
    #[test]
    #[ignore = "probes the live desktop session"]
    fn live_check_of_this_session() {
        let started = std::time::Instant::now();
        let facts = gather();
        eprintln!("{}", report(&check(&facts, None)));
        eprintln!("gathered in {:?}", started.elapsed());
        assert!(started.elapsed() < PROBE_TIMEOUT * 2);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn introspection_xml_yields_portal_interface_names() {
        let xml = r#"<node><interface name="org.freedesktop.DBus.Peer"/>
            <interface name="org.freedesktop.portal.GlobalShortcuts"><property name="version"/></interface>
            <interface name="org.freedesktop.portal.RemoteDesktop"></interface></node>"#;
        assert_eq!(
            live::interface_names(xml),
            vec![GLOBAL_SHORTCUTS.to_string(), REMOTE_DESKTOP.to_string()]
        );
    }
}
