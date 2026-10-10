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

/// Everything [`evaluate`] needs; each probe's failure is its own `Err`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Facts {
    pub session: SessionFacts,
    pub portal: Option<Result<PortalFacts, String>>,
    pub portal_backends: Vec<PortalBackend>,
    /// Each insertion backend: its scheme and availability.
    pub insertion: Vec<(String, Result<(), String>)>,
    pub atspi: Option<Result<(), String>>,
    pub input_methods: InputMethodFacts,
    pub audio: AudioFacts,
    /// Capture devices: (name, is default).
    pub microphones: Option<Result<Vec<(String, bool)>, String>>,
}

const GLOBAL_SHORTCUTS: &str = "org.freedesktop.portal.GlobalShortcuts";
const REMOTE_DESKTOP: &str = "org.freedesktop.portal.RemoteDesktop";

/// The check's lines, in a fixed order. `portal_status` is the live
/// shortcut binding, when the portal source is running.
pub(crate) fn evaluate(facts: &Facts, portal_status: Option<&PortalStatus>) -> Vec<CheckLine> {
    vec![
        session_line(&facts.session),
        shortcut_line(facts, portal_status),
        insertion_line(facts),
        atspi_line(facts.atspi.as_ref()),
        input_method_line(&facts.input_methods),
        audio_line(&facts.audio),
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

fn shortcut_line(facts: &Facts, portal_status: Option<&PortalStatus>) -> CheckLine {
    const TOPIC: &str = "System-wide shortcut";
    let session = &facts.session;
    if !session.wayland() {
        return line(
            TOPIC,
            Verdict::Ok,
            "X11: the shortcut is grabbed directly; no portal needed.",
            None,
        );
    }
    let desktop = desktop_name(session);
    let portal = match facts.portal.as_ref() {
        None => {
            return line(TOPIC, Verdict::Info, "The desktop portal was not checked.", None);
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
    let backend = if in_use.is_empty() {
        "backend not identified from the .portal files".to_string()
    } else {
        format!("backend: {}", in_use.join(", "))
    };
    let summary = format!("GlobalShortcuts portal version {version} ({backend}).");
    match portal_status {
        Some(status @ PortalStatus::Bound { .. }) => {
            line(TOPIC, Verdict::Ok, format!("{summary} {}", status.describe()), None)
        }
        Some(PortalStatus::Unavailable(reason)) => line(
            TOPIC,
            Verdict::Missing,
            format!("{summary} Starling could not use it: {reason}."),
            Some(
                "Restart Starling after updating xdg-desktop-portal; versions before 1.19 \
                 cannot identify apps that are not sandboxed."
                    .to_string(),
            ),
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
        Some(status) => line(TOPIC, Verdict::Info, format!("{summary} {}", status.describe()), None),
        None => line(TOPIC, Verdict::Info, summary, None),
    }
}

fn insertion_line(facts: &Facts) -> CheckLine {
    const TOPIC: &str = "Typing into other apps";
    let available: Vec<&str> = facts
        .insertion
        .iter()
        .filter(|(_, availability)| availability.is_ok())
        .map(|(scheme, _)| scheme.as_str())
        .collect();
    let blocked: Vec<String> = facts
        .insertion
        .iter()
        .filter_map(|(scheme, availability)| {
            availability
                .as_ref()
                .err()
                .map(|reason| format!("{scheme}: {reason}"))
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
    if available.is_empty() {
        return line(
            TOPIC,
            Verdict::Missing,
            format!("No insertion backend is available ({}).{portal_note}", blocked.join("; ")),
            Some(if facts.session.wayland() {
                "On Wayland, typing into other apps needs the compositor's virtual-keyboard \
                 protocol; until then copy the transcript and paste it yourself."
                    .to_string()
            } else {
                "Copy the transcript and paste it yourself.".to_string()
            }),
        );
    }
    let only_x11 = available.iter().all(|scheme| *scheme == "x11");
    if facts.session.wayland() && only_x11 {
        line(
            TOPIC,
            Verdict::Limited,
            format!("X11 typing only: it reaches XWayland apps, not native Wayland ones.{portal_note}"),
            Some("For native Wayland apps, copy the transcript and paste it yourself.".to_string()),
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

fn audio_line(audio: &AudioFacts) -> CheckLine {
    const TOPIC: &str = "Sound server";
    match (audio.pipewire, audio.pulse) {
        (true, true) => line(TOPIC, Verdict::Ok, "PipeWire with its PulseAudio socket.", None),
        (false, true) => line(TOPIC, Verdict::Ok, "PulseAudio.", None),
        (true, false) => line(
            TOPIC,
            Verdict::Limited,
            "PipeWire without its PulseAudio socket: microphone routing falls back to ALSA \
             device names.",
            Some("Install and start pipewire-pulse: `systemctl --user enable --now pipewire-pulse`.".to_string()),
        ),
        (false, false) => line(
            TOPIC,
            Verdict::Missing,
            "Neither PipeWire nor PulseAudio answers.",
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
        Some(Ok(devices)) if devices.is_empty() => {
            line(TOPIC, Verdict::Missing, "No capture device is visible.", fix)
        }
        Some(Ok(devices)) => {
            let default = devices
                .iter()
                .find(|(_, is_default)| *is_default)
                .map(|(name, _)| format!("; default: {name}"))
                .unwrap_or_default();
            let count = devices.len();
            let noun = if count == 1 { "device" } else { "devices" };
            line(TOPIC, Verdict::Ok, format!("{count} capture {noun} visible{default}."), None)
        }
    }
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
        let items = || value.split(';').map(str::trim).filter(|item| !item.is_empty());
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

/// Run `probe` on its own thread; `None` when it takes longer than
/// [`PROBE_TIMEOUT`] (the thread is left to finish on its own).
fn bounded<T: Send + 'static>(probe: impl FnOnce() -> T + Send + 'static) -> Option<T> {
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::Builder::new()
        .name("starling-check".to_string())
        .spawn(move || {
            let _ = sender.send(probe());
        })
        .ok()?;
    receiver.recv_timeout(PROBE_TIMEOUT).ok()
}

fn timed_out<T>(outcome: Option<Result<T, String>>) -> Result<T, String> {
    outcome.unwrap_or_else(|| Err(format!("no answer within {} s", PROBE_TIMEOUT.as_secs())))
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

    let portal = std::thread::spawn(|| timed_out(bounded(live::portal)));
    let atspi = std::thread::spawn(|| timed_out(bounded(live::atspi)));
    let names = std::thread::spawn(|| bounded(live::input_method_names).unwrap_or_default());
    let insertion = std::thread::spawn(|| {
        bounded(|| {
            starling_insertion::Inserter::for_this_session()
                .availability()
                .into_iter()
                .map(|(kind, availability)| {
                    (
                        kind.scheme().to_string(),
                        availability.map_err(|err| err.message()),
                    )
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(|| {
            vec![(
                "x11".to_string(),
                Err(format!("no answer within {} s", PROBE_TIMEOUT.as_secs())),
            )]
        })
    });
    let microphones = std::thread::spawn(|| {
        timed_out(bounded(|| {
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
    Facts {
        portal_backends: live::portal_backends(&desktops),
        session,
        portal: Some(portal.join().unwrap_or_else(|_| Err("the probe failed".to_string()))),
        insertion: insertion.join().unwrap_or_default(),
        atspi: Some(atspi.join().unwrap_or_else(|_| Err("the probe failed".to_string()))),
        input_methods: InputMethodFacts {
            ibus_running,
            fcitx_running,
            configured,
        },
        audio: live::audio(),
        microphones: Some(
            microphones
                .join()
                .unwrap_or_else(|_| Err("the probe failed".to_string())),
        ),
    }
}

#[cfg(target_os = "linux")]
mod live {
    use std::collections::HashMap;
    use std::path::PathBuf;

    use zbus::blocking::{Connection, Proxy};

    use super::{env, parse_portal_file, AudioFacts, PortalBackend, PortalFacts, GLOBAL_SHORTCUTS};

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
        let data_dirs = env("XDG_DATA_DIRS").unwrap_or_else(|| "/usr/local/share:/usr/share".to_string());
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
                    seen.insert(name.to_string(), parse_portal_file(name, &contents, desktops));
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
            pipewire: reachable(
                env("PIPEWIRE_REMOTE")
                    .map(PathBuf::from)
                    .or_else(|| runtime.as_ref().map(|dir| dir.join("pipewire-0"))),
            ),
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
            self.portal_shortcuts.as_ref().map(|portal| portal.status()),
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
            insertion: vec![("x11".to_string(), Ok(()))],
            atspi: Some(Ok(())),
            input_methods: InputMethodFacts::default(),
            audio: AudioFacts {
                pipewire: true,
                pulse: true,
            },
            microphones: Some(Ok(vec![
                ("Built-in".to_string(), false),
                ("USB mic".to_string(), true),
            ])),
        }
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
        let lines = evaluate(&healthy_kde(), Some(&status));
        let shortcut = find(&lines, "System-wide shortcut");
        assert_eq!(shortcut.verdict, Verdict::Ok);
        assert!(shortcut.summary.contains("version 2"), "{}", shortcut.summary);
        assert!(shortcut.summary.contains("backend: kde"), "{}", shortcut.summary);
        assert!(shortcut.summary.contains("Meta+Space"), "{}", shortcut.summary);
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
        let lines = evaluate(&facts, Some(&PortalStatus::Unavailable("x".into())));
        let shortcut = find(&lines, "System-wide shortcut");
        assert_eq!(shortcut.verdict, Verdict::Limited);
        assert!(shortcut.summary.contains("COSMIC's desktop portal has no GlobalShortcuts"));
        assert!(shortcut.summary.contains("(kde) are not used on COSMIC"), "{}", shortcut.summary);
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
        let lines = evaluate(&facts, None);
        assert_eq!(find(&lines, "Session").summary, "X11 (XFCE).");
        assert_eq!(find(&lines, "System-wide shortcut").verdict, Verdict::Ok);
        assert_eq!(find(&lines, "Typing into other apps").verdict, Verdict::Ok);

        let mut facts = healthy_kde();
        facts.portal = Some(Err("xdg-desktop-portal does not answer".to_string()));
        let shortcut = find(&evaluate(&facts, None), "System-wide shortcut").clone();
        assert_eq!(shortcut.verdict, Verdict::Missing);
        assert!(shortcut.fix.unwrap().contains("xdg-desktop-portal-kde"));
    }

    #[test]
    fn a_portal_waiting_for_setup_points_at_the_button() {
        let lines = evaluate(&healthy_kde(), Some(&PortalStatus::NeedsSetup { configurable: true }));
        let shortcut = find(&lines, "System-wide shortcut");
        assert_eq!(shortcut.verdict, Verdict::Limited);
        assert!(shortcut.fix.as_deref().unwrap().contains("Set up desktop shortcut"));
    }

    #[test]
    fn wayland_without_xwayland_or_insertion_backends_says_so() {
        let mut facts = healthy_kde();
        facts.session = wayland("sway", false);
        facts.insertion = vec![("x11".to_string(), Err("no X11 display".to_string()))];
        let lines = evaluate(&facts, None);
        assert_eq!(find(&lines, "Session").verdict, Verdict::Limited);
        let typing = find(&lines, "Typing into other apps");
        assert_eq!(typing.verdict, Verdict::Missing);
        assert!(typing.summary.contains("x11: no X11 display"));
        assert!(typing.fix.as_deref().unwrap().contains("virtual-keyboard"));
    }

    #[test]
    fn missing_audio_microphones_and_atspi_carry_fixes() {
        let mut facts = healthy_kde();
        facts.audio = AudioFacts::default();
        facts.microphones = Some(Ok(Vec::new()));
        facts.atspi = Some(Err("org.a11y.Bus does not answer".to_string()));
        let lines = evaluate(&facts, None);
        let audio = find(&lines, "Sound server");
        assert_eq!(audio.verdict, Verdict::Missing);
        assert!(audio.fix.as_deref().unwrap().contains("systemctl --user start pipewire"));
        let mic = find(&lines, "Microphone");
        assert_eq!(mic.verdict, Verdict::Missing);
        assert!(mic.fix.is_some());
        let atspi = find(&lines, "Accessibility bus (AT-SPI)");
        assert_eq!(atspi.verdict, Verdict::Limited);
        assert!(atspi.fix.as_deref().unwrap().contains("at-spi2-core"));
    }

    #[test]
    fn input_methods_are_informational() {
        let mut facts = healthy_kde();
        facts.input_methods = InputMethodFacts {
            ibus_running: true,
            fcitx_running: false,
            configured: vec![("GTK_IM_MODULE".to_string(), "ibus".to_string())],
        };
        let line = find(&evaluate(&facts, None), "Input method").clone();
        assert_eq!(line.verdict, Verdict::Info);
        assert!(line.summary.starts_with("IBus running (GTK_IM_MODULE=ibus)."));
        assert!(line.summary.contains("no input-method engine yet"));
    }

    #[test]
    fn the_report_lists_every_line_with_its_fix() {
        let mut facts = healthy_kde();
        facts.audio = AudioFacts::default();
        let text = report(&evaluate(&facts, None));
        assert_eq!(text.lines().filter(|line| line.starts_with('[')).count(), 7);
        assert!(text.contains("[missing] Sound server: Neither PipeWire nor PulseAudio answers."));
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
        eprintln!("{}", report(&evaluate(&facts, None)));
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
