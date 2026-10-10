//! The XDG GlobalShortcuts portal (#221): native Wayland's system-wide
//! source of the recording shortcut.
//!
//! On Wayland no client may grab keys; the desktop owns shortcuts and,
//! through `org.freedesktop.portal.GlobalShortcuts`, tells an app when
//! one of its bound shortcuts goes down (`Activated`) and comes up again
//! (`Deactivated`). Both edges arrive, so hold to talk works in every
//! app — not only the XWayland ones the X11 grab hears.
//!
//! The protocol runs on its own thread (`dbus::run`) and never on the UI
//! thread: connect, register the app id with the host portal registry
//! (host apps have none otherwise, and the portal refuses shortcuts
//! without one), create a session, and bind the one shortcut this app
//! has (`record`) with the configured shortcut as `preferred_trigger`.
//! Binding opens the desktop's shortcut dialog, so it only ever happens
//! on the user's request (Settings → Dictation → "Set up desktop
//! shortcut") — except when the desktop already remembers a binding from
//! an earlier run, which is re-bound at start without a dialog. The
//! desktop decides the keys: the dialog may change or refuse the
//! trigger, and later changes arrive as `ShortcutsChanged`.
//!
//! Edges reach the activation machine exactly like the X11 grab's
//! (`GlobalEvent::Pressed` / `Released`, timestamped where they were
//! received), so the machine's single-take, repeat and lost-release
//! rules apply unchanged. [`PortalKeys`] adds what is particular to the
//! portal: a `Deactivated` without a preceding `Activated` is dropped,
//! and a session that ends while the shortcut is down (the desktop closed
//! it, the portal restarted, the shortcut was rebound) releases it, like
//! a lost key release, so a held take finishes instead of recording
//! forever. While the portal holds a binding it is *the* system-wide
//! source: the X11 grab's presses are dropped (`activation.rs`), so one
//! physical press is never taken twice.
//!
//! Without a portal, or one whose backend lacks GlobalShortcuts (COSMIC,
//! wlroots desktops without xdg-desktop-portal-hyprland), the status says
//! so and today's sources (X11 grab for XWayland apps, in-window keys)
//! stay as they are.

use std::sync::mpsc;
use std::time::Instant;

use crate::shortcut::{GlobalEvent, Shortcut};

/// The one shortcut id this app binds.
const SHORTCUT_ID: &str = "record";

/// What the desktop's dialog lists the shortcut as.
const SHORTCUT_DESCRIPTION: &str = "Dictate with Starling (hold to talk, or tap)";

/// The app id registered with the host portal registry. The portal keeps
/// the binding under it, so it must stay stable across runs, and it only
/// accepts an id with an installed `<id>.desktop` entry (see
/// [`install_desktop_entry`]). The window carries it as its Wayland
/// app id too (`main.rs`).
pub(crate) const APP_ID: &str = "dev.starling.Starling";

/// Where the portal shortcut stands, for Settings and the system check.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum PortalStatus {
    /// Connecting and opening a session.
    Starting,
    /// No usable GlobalShortcuts portal; the reason says what is missing.
    Unavailable(String),
    /// The portal will not identify Starling: it registers host apps only
    /// by an installed desktop entry, and there is none. Setting up
    /// installs a per-user one, then binds.
    NeedsDesktopEntry,
    /// A session is open but nothing is bound yet: binding opens the
    /// desktop's dialog, so it waits for the user.
    NeedsSetup { configurable: bool },
    /// The desktop's dialog is (or may be) showing.
    Binding,
    /// The shortcut is bound. `trigger` is the desktop's description of
    /// the keys, when it gave one.
    Bound {
        trigger: Option<String>,
        configurable: bool,
    },
    /// The user dismissed the dialog or bound nothing.
    Declined { configurable: bool },
    /// The binding went away: the desktop closed the session, removed the
    /// shortcut, or a call failed. `reason` says which.
    Lost { reason: String, configurable: bool },
}

impl PortalStatus {
    pub(crate) fn is_bound(&self) -> bool {
        matches!(self, PortalStatus::Bound { .. })
    }

    /// Whether "Set up desktop shortcut" can act now.
    pub(crate) fn can_set_up(&self) -> bool {
        matches!(
            self,
            PortalStatus::NeedsSetup { .. }
                | PortalStatus::NeedsDesktopEntry
                | PortalStatus::Declined { .. }
                | PortalStatus::Lost { .. }
        )
    }

    /// Whether the desktop can show its own shortcut settings for this
    /// app (`ConfigureShortcuts`, portal version 2).
    pub(crate) fn configurable(&self) -> bool {
        matches!(
            self,
            PortalStatus::Bound {
                configurable: true,
                ..
            }
        )
    }

    /// One line for Settings and the system check.
    pub(crate) fn describe(&self) -> String {
        match self {
            PortalStatus::Starting => "Checking the desktop's shortcut portal…".to_string(),
            PortalStatus::Unavailable(reason) => {
                format!("The desktop shortcut portal is not available: {reason}.")
            }
            PortalStatus::NeedsDesktopEntry => format!(
                "Your desktop can deliver the shortcut to Starling in every app, press and \
                 release included, once it can identify Starling: setting it up adds a \
                 Starling entry to your applications ({APP_ID}.desktop in your data folder) \
                 and opens your desktop's shortcut dialog."
            ),
            PortalStatus::NeedsSetup { .. } => "Your desktop can deliver the shortcut to \
                 Starling in every app, press and release included. Setting it up opens your \
                 desktop's shortcut dialog, where you confirm or change the keys."
                .to_string(),
            PortalStatus::Binding => "Waiting for your desktop's shortcut dialog…".to_string(),
            PortalStatus::Bound {
                trigger: Some(trigger),
                ..
            } => format!(
                "Desktop shortcut: {trigger}. The desktop decides these keys; they may differ \
                 from the shortcut above, which still works in the Starling window."
            ),
            PortalStatus::Bound { trigger: None, .. } => "Desktop shortcut bound (the desktop \
                 did not say which keys; check its keyboard settings)."
                .to_string(),
            PortalStatus::Declined { .. } => "No desktop shortcut was assigned. Set it up \
                 again to choose keys in your desktop's dialog."
                .to_string(),
            PortalStatus::Lost { reason, .. } => {
                format!("The desktop shortcut stopped working ({reason}). Set it up again.")
            }
        }
    }
}

/// One edge or status change from the worker, received in order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum PortalSignal {
    /// `Activated` for this app's shortcut in the live session.
    Activated(Instant),
    /// `Deactivated` for it.
    Deactivated(Instant),
    /// The session the edges came from is gone (closed by the desktop or
    /// by a rebind, or the portal went away).
    SessionEnded(Instant),
    /// A new status, stamped when the worker published it (binding
    /// transitions are ordered against other sources' events by it).
    Status(PortalStatus, Instant),
}

/// The portal's key state: turns its edges into the activation machine's
/// presses and releases. Pure, so the mapping is tested without a bus.
#[derive(Debug, Default)]
pub(crate) struct PortalKeys {
    down: bool,
}

impl PortalKeys {
    /// The machine input for one portal edge, if any.
    pub(crate) fn map(&mut self, signal: &PortalSignal) -> Option<GlobalEvent> {
        match *signal {
            // A second `Activated` without a release is passed on: the
            // machine reads it as auto-repeat when it comes soon, and as a
            // lost release (finishing a stuck hold) when it does not.
            PortalSignal::Activated(at) => {
                self.down = true;
                Some(GlobalEvent::Pressed(at))
            }
            // A release for a press this source never reported (sent
            // twice, or across a session swap) is dropped.
            PortalSignal::Deactivated(at) | PortalSignal::SessionEnded(at) if self.down => {
                self.down = false;
                Some(GlobalEvent::Released(at))
            }
            _ => None,
        }
    }
}

/// Where a per-user desktop entry for Starling lives.
fn user_entry_dir() -> Option<std::path::PathBuf> {
    dirs::data_dir().map(|dir| dir.join("applications"))
}

/// Whether a desktop entry for [`APP_ID`] is installed in `dir` or any
/// system data directory.
fn desktop_entry_installed(dir: Option<&std::path::Path>) -> bool {
    let file = format!("{APP_ID}.desktop");
    let system = std::env::var("XDG_DATA_DIRS")
        .ok()
        .filter(|dirs| !dirs.is_empty())
        .unwrap_or_else(|| "/usr/local/share:/usr/share".to_string());
    dir.into_iter()
        .map(std::path::Path::to_path_buf)
        .chain(
            system
                .split(':')
                .filter(|dir| !dir.is_empty())
                .map(|dir| std::path::Path::new(dir).join("applications")),
        )
        .any(|dir| dir.join(&file).is_file())
}

/// The desktop entry's text for `exe`.
fn desktop_entry(exe: &std::path::Path) -> String {
    // Exec quoting per the desktop-entry spec: the path in double quotes,
    // with `"`, `` ` ``, `$` and `\` escaped.
    let mut quoted = String::from('"');
    for c in exe.to_string_lossy().chars() {
        if matches!(c, '"' | '`' | '$' | '\\') {
            quoted.push('\\');
        }
        quoted.push(c);
    }
    quoted.push('"');
    format!(
        "[Desktop Entry]\nType=Application\nName=Starling\nComment=Dictation on this \
         machine\nExec={quoted}\nTerminal=false\nCategories=Utility;Audio;\n\
         StartupWMClass={APP_ID}\n"
    )
}

/// Install the per-user desktop entry the portal needs to identify
/// Starling (on the user's request only: it also lists Starling among
/// their applications).
fn install_desktop_entry(dir: &std::path::Path) -> Result<(), String> {
    let exe = std::env::current_exe().map_err(|err| format!("no executable path ({err})"))?;
    std::fs::create_dir_all(dir).map_err(|err| format!("{} ({err})", dir.display()))?;
    let path = dir.join(format!("{APP_ID}.desktop"));
    std::fs::write(&path, desktop_entry(&exe)).map_err(|err| format!("{} ({err})", path.display()))
}

/// A request from the UI thread to the worker.
#[derive(Debug)]
enum Command {
    /// Bind the shortcut (opens the desktop's dialog).
    SetUp { trigger: Option<String> },
    /// The configured shortcut changed: offer it as the new preferred
    /// trigger, if a binding exists (a fresh session; the portal binds
    /// once per session).
    Rebind { trigger: Option<String> },
    /// Show the desktop's own shortcut settings for this app.
    Configure,
}

/// How many binding transitions [`PortalShortcuts::bound_at`] remembers.
const BOUND_HISTORY: usize = 16;

/// The UI thread's handle on the portal worker.
pub(crate) struct PortalShortcuts {
    commands: Option<tokio::sync::mpsc::UnboundedSender<Command>>,
    signals: mpsc::Receiver<PortalSignal>,
    keys: PortalKeys,
    /// Machine inputs received but not yet taken, oldest first.
    pending: std::collections::VecDeque<GlobalEvent>,
    /// When the binding came and went, oldest first (see [`Self::bound_at`]).
    bound_changes: Vec<(Instant, bool)>,
    status: PortalStatus,
    status_changed: bool,
}

impl PortalShortcuts {
    /// Start the portal source in a native Wayland session; `None`
    /// elsewhere (X11 sessions have the X11 grab, other platforms their
    /// own system-wide shortcut).
    pub(crate) fn start_for_session(shortcut: &Shortcut) -> Option<PortalShortcuts> {
        if !crate::shortcut::wayland_session() {
            return None;
        }
        Some(Self::spawn(None, user_entry_dir(), shortcut))
    }

    /// Spawn the worker on the session bus, or on `address` (tests);
    /// `entry_dir` is where a desktop entry is installed when the portal
    /// needs one.
    #[cfg(target_os = "linux")]
    fn spawn(
        address: Option<String>,
        entry_dir: Option<std::path::PathBuf>,
        shortcut: &Shortcut,
    ) -> PortalShortcuts {
        let (commands, inbox) = tokio::sync::mpsc::unbounded_channel();
        let (signals_out, signals) = mpsc::channel();
        let trigger = shortcut.xdg_trigger();
        let spawned = std::thread::Builder::new()
            .name("starling-portal".to_string())
            .spawn({
                let signals_out = signals_out.clone();
                move || dbus::thread_main(address, entry_dir, trigger, inbox, signals_out)
            });
        let mut portal = PortalShortcuts {
            commands: Some(commands),
            signals,
            keys: PortalKeys::default(),
            pending: Default::default(),
            bound_changes: Vec::new(),
            status: PortalStatus::Starting,
            status_changed: false,
        };
        if let Err(err) = spawned {
            portal.commands = None;
            portal.status = PortalStatus::Unavailable(format!("no worker thread ({err})"));
        }
        portal
    }

    #[cfg(not(target_os = "linux"))]
    fn spawn(
        _address: Option<String>,
        _entry_dir: Option<std::path::PathBuf>,
        _shortcut: &Shortcut,
    ) -> PortalShortcuts {
        let (_, signals) = mpsc::channel();
        PortalShortcuts {
            commands: None,
            signals,
            keys: PortalKeys::default(),
            pending: Default::default(),
            bound_changes: Vec::new(),
            status: PortalStatus::Unavailable("not a Linux desktop".to_string()),
            status_changed: false,
        }
    }

    pub(crate) fn status(&self) -> &PortalStatus {
        &self.status
    }

    /// Whether the portal is the system-wide source right now.
    pub(crate) fn is_bound(&self) -> bool {
        self.status.is_bound()
    }

    /// The user asked to bind the shortcut: the desktop's dialog opens.
    pub(crate) fn set_up(&self, shortcut: &Shortcut) {
        self.send(Command::SetUp {
            trigger: shortcut.xdg_trigger(),
        });
    }

    /// The configured shortcut changed. With a binding in place it is
    /// offered to the desktop as the new preferred trigger; the desktop
    /// may keep the keys the user chose there.
    pub(crate) fn rebind(&self, shortcut: &Shortcut) {
        self.send(Command::Rebind {
            trigger: shortcut.xdg_trigger(),
        });
    }

    /// Open the desktop's shortcut settings for this app.
    pub(crate) fn configure(&self) {
        self.send(Command::Configure);
    }

    fn send(&self, command: Command) {
        if let Some(commands) = self.commands.as_ref() {
            let _ = commands.send(command);
        }
    }

    /// Take in everything the worker sent: status changes apply now and
    /// binding transitions are remembered with their time (so other
    /// sources' events are judged by the binding as it was when they were
    /// received), machine inputs queue for [`Self::next_event`].
    pub(crate) fn poll(&mut self) {
        while let Ok(signal) = self.signals.try_recv() {
            if let PortalSignal::Status(status, at) = signal {
                if status.is_bound() != self.status.is_bound() {
                    self.bound_changes.push((at, status.is_bound()));
                    if self.bound_changes.len() > BOUND_HISTORY {
                        self.bound_changes.remove(0);
                    }
                }
                if status != self.status {
                    self.status = status;
                    self.status_changed = true;
                }
            } else if let Some(event) = self.keys.map(&signal) {
                self.pending.push_back(event);
            }
        }
    }

    /// Whether the portal held a binding at `at` (call [`Self::poll`]
    /// first for the latest transitions).
    pub(crate) fn bound_at(&self, at: Instant) -> bool {
        crate::activation::focused_at(&self.bound_changes, at)
    }

    /// The next machine input from the portal, oldest first.
    pub(crate) fn next_event(&mut self) -> Option<GlobalEvent> {
        self.poll();
        self.pending.pop_front()
    }

    /// Whether the status changed since the last call (the UI repaints).
    pub(crate) fn take_status_changed(&mut self) -> bool {
        std::mem::take(&mut self.status_changed)
    }
}

/// The D-Bus side: one worker thread per [`PortalShortcuts`], running the
/// portal protocol with `zbus` on a current-thread runtime. Every wait
/// that does not involve the user is bounded.
#[cfg(target_os = "linux")]
mod dbus {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex, mpsc};
    use std::time::{Duration, Instant};

    use futures_util::StreamExt;
    use serde::Deserialize;
    use tokio::sync::mpsc::UnboundedReceiver;
    use tokio::sync::oneshot;
    use zbus::proxy::{CacheProperties, OwnerChangedStream};
    use zbus::zvariant::{DeserializeDict, OwnedObjectPath, OwnedValue, Type, Value};
    use zbus::{Connection, Proxy};

    use super::{
        APP_ID, Command, PortalSignal, PortalStatus, SHORTCUT_DESCRIPTION, SHORTCUT_ID,
        desktop_entry_installed, install_desktop_entry,
    };

    pub(super) const DESTINATION: &str = "org.freedesktop.portal.Desktop";
    pub(super) const PATH: &str = "/org/freedesktop/portal/desktop";
    pub(super) const INTERFACE: &str = "org.freedesktop.portal.GlobalShortcuts";
    const REQUEST_INTERFACE: &str = "org.freedesktop.portal.Request";
    const SESSION_INTERFACE: &str = "org.freedesktop.portal.Session";
    const REGISTRY_INTERFACE: &str = "org.freedesktop.host.portal.Registry";

    /// CreateSession succeeded but left no live session: the handle was
    /// unusable, or the desktop closed it straight away.
    const SESSION_CLOSED_AT_ONCE: &str =
        "the portal returned no usable session handle, or closed it";

    /// Connecting to the session bus and every call that opens no dialog.
    const CALL_TIMEOUT: Duration = Duration::from_secs(10);

    /// Request responses: `0` success, `1` cancelled by the user.
    const RESPONSE_SUCCESS: u32 = 0;
    const RESPONSE_CANCELLED: u32 = 1;

    /// One entry of a `shortcuts` list (`a(sa{sv})`): the BindShortcuts /
    /// ListShortcuts results and the `ShortcutsChanged` signal.
    #[derive(Deserialize, Type, Debug)]
    pub(super) struct ShortcutEntry(pub String, pub ShortcutInfo);

    #[derive(DeserializeDict, Type, Debug, Default)]
    #[zvariant(signature = "dict")]
    pub(super) struct ShortcutInfo {
        pub trigger_description: Option<String>,
    }

    /// The results of BindShortcuts and ListShortcuts.
    #[derive(DeserializeDict, Type, Debug, Default)]
    #[zvariant(signature = "dict")]
    struct ShortcutResults {
        shortcuts: Option<Vec<ShortcutEntry>>,
    }

    /// This app's shortcut in a list: `None` when it is not there, else
    /// its trigger description (when the desktop gave one).
    pub(super) fn find_ours(entries: &[ShortcutEntry]) -> Option<Option<String>> {
        entries
            .iter()
            .find(|entry| entry.0 == SHORTCUT_ID)
            .map(|entry| {
                entry
                    .1
                    .trigger_description
                    .clone()
                    .filter(|text| !text.trim().is_empty())
            })
    }

    /// A request's `Response` body: the response code and its results.
    fn response_body<R>(message: &zbus::Message) -> Result<(u32, R), String>
    where
        R: serde::de::DeserializeOwned + Type,
    {
        message
            .body()
            .deserialize::<(u32, R)>()
            .map_err(|err| format!("unreadable portal response ({err})"))
    }

    /// The `session_handle` of a CreateSession response: specified as a
    /// string, sent as an object path by some backends.
    fn session_handle_of(results: &HashMap<String, OwnedValue>) -> Option<String> {
        match &**results.get("session_handle")? {
            Value::Str(text) => Some(text.to_string()),
            Value::ObjectPath(path) => Some(path.to_string()),
            _ => None,
        }
    }

    /// What a failed portal call means to the user.
    pub(super) fn describe_error(err: &zbus::Error) -> String {
        let name = match err {
            zbus::Error::MethodError(name, _, _) => name.to_string(),
            zbus::Error::FDO(fdo) => zbus::DBusError::name(fdo.as_ref()).to_string(),
            _ => String::new(),
        };
        match name.as_str() {
            "org.freedesktop.DBus.Error.ServiceUnknown"
            | "org.freedesktop.DBus.Error.NameHasNoOwner" => {
                "xdg-desktop-portal is not running on the session bus".to_string()
            }
            "org.freedesktop.DBus.Error.UnknownInterface"
            | "org.freedesktop.DBus.Error.UnknownProperty"
            | "org.freedesktop.DBus.Error.UnknownMethod"
            | "org.freedesktop.DBus.Error.InvalidArgs" => "the desktop's portal backend does \
                 not implement GlobalShortcuts"
                .to_string(),
            _ => err.to_string(),
        }
    }

    /// The worker thread: a current-thread runtime for the protocol. The
    /// connection's own I/O runs on zbus's internal executor.
    pub(super) fn thread_main(
        address: Option<String>,
        entry_dir: Option<std::path::PathBuf>,
        trigger: Option<String>,
        commands: UnboundedReceiver<Command>,
        out: mpsc::Sender<PortalSignal>,
    ) {
        let runtime = match tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
        {
            Ok(runtime) => runtime,
            Err(err) => {
                let _ = out.send(PortalSignal::Status(
                    PortalStatus::Unavailable(format!("no async runtime ({err})")),
                    Instant::now(),
                ));
                return;
            }
        };
        runtime.block_on(run(address, entry_dir, trigger, commands, out));
    }

    async fn connect(address: Option<String>) -> Result<Connection, String> {
        let connecting = async {
            match address {
                Some(address) => {
                    zbus::connection::Builder::address(address.as_str())?
                        .build()
                        .await
                }
                None => Connection::session().await,
            }
        };
        match tokio::time::timeout(CALL_TIMEOUT, connecting).await {
            Ok(Ok(connection)) => Ok(connection),
            Ok(Err(err)) => Err(format!("no session bus ({err})")),
            Err(_) => Err("the session bus did not answer".to_string()),
        }
    }

    /// Bounded wait for a call that opens no dialog.
    async fn bounded<T>(
        call: impl std::future::Future<Output = Result<T, String>>,
    ) -> Result<T, String> {
        tokio::time::timeout(CALL_TIMEOUT, call)
            .await
            .unwrap_or_else(|_| Err("the desktop portal did not answer".to_string()))
    }

    /// What the command side and the signal forwarder share: the live
    /// session, whether this app's shortcut is bound in it, and the one
    /// request waiting for its `Response`. Every status is published
    /// with this lock held, so the UI sees statuses in the order the
    /// state changed.
    #[derive(Default)]
    struct Live {
        session: Option<OwnedObjectPath>,
        bound: bool,
        configurable: bool,
        pending: Option<Pending>,
        /// The bus connection ended: nothing more is published.
        terminal: bool,
    }

    /// A request waiting for its `Response`. The response arrives on the
    /// forwarder's ordered stream, which applies what it means before any
    /// later signal is looked at.
    struct Pending {
        path: String,
        kind: PendingKind,
        reply: oneshot::Sender<Result<zbus::Message, String>>,
    }

    enum PendingKind {
        Plain,
        /// Commits the session handle.
        CreateSession,
        /// Commits the binding of `session` and publishes the outcome.
        Bind {
            session: OwnedObjectPath,
        },
    }

    type Shared = Arc<Mutex<Live>>;

    fn lock(live: &Shared) -> std::sync::MutexGuard<'_, Live> {
        live.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    impl Live {
        fn publish(&self, out: &mpsc::Sender<PortalSignal>, status: PortalStatus) {
            if !self.terminal {
                let _ = out.send(PortalSignal::Status(status, Instant::now()));
            }
        }

        /// The live session ended: release a held shortcut and say why.
        fn session_ended(&mut self, out: &mpsc::Sender<PortalSignal>, reason: &str) {
            let _ = out.send(PortalSignal::SessionEnded(Instant::now()));
            self.session = None;
            let status = if std::mem::take(&mut self.bound) {
                PortalStatus::Lost {
                    reason: reason.to_string(),
                    configurable: false,
                }
            } else {
                PortalStatus::NeedsSetup {
                    configurable: self.configurable,
                }
            };
            self.publish(out, status);
            self.fail_pending(out, reason);
        }

        /// The waiting request cannot be answered any more. A bind says
        /// so; its caller publishes nothing after it.
        fn fail_pending(&mut self, out: &mpsc::Sender<PortalSignal>, reason: &str) {
            let Some(pending) = self.pending.take() else {
                return;
            };
            if let PendingKind::Bind { .. } = pending.kind {
                self.publish(
                    out,
                    PortalStatus::Lost {
                        reason: reason.to_string(),
                        configurable: self.configurable,
                    },
                );
            }
            let _ = pending.reply.send(Err(reason.to_string()));
        }

        /// A `Response` for the waiting request.
        fn respond(&mut self, out: &mpsc::Sender<PortalSignal>, message: &zbus::Message) {
            let Some(pending) = self.pending.take() else {
                return;
            };
            match &pending.kind {
                PendingKind::Plain => {}
                PendingKind::CreateSession => {
                    self.session = response_body::<HashMap<String, OwnedValue>>(message)
                        .ok()
                        .filter(|(code, _)| *code == RESPONSE_SUCCESS)
                        .and_then(|(_, results)| session_handle_of(&results))
                        .and_then(|handle| OwnedObjectPath::try_from(handle).ok());
                }
                PendingKind::Bind { session } => {
                    let configurable = self.configurable;
                    let status = match response_body::<ShortcutResults>(message) {
                        Ok((RESPONSE_SUCCESS, results)) => {
                            match find_ours(results.shortcuts.as_deref().unwrap_or_default()) {
                                // Bound in a session that is still live.
                                Some(trigger) if self.session.as_ref() == Some(session) => {
                                    self.bound = true;
                                    Some(PortalStatus::Bound {
                                        trigger,
                                        configurable,
                                    })
                                }
                                // The session closed meanwhile (already said).
                                Some(_) => None,
                                None => Some(PortalStatus::Declined { configurable }),
                            }
                        }
                        Ok((RESPONSE_CANCELLED, _)) => {
                            Some(PortalStatus::Declined { configurable })
                        }
                        Ok((code, _)) => Some(PortalStatus::Lost {
                            reason: format!("the desktop refused the shortcut (response {code})"),
                            configurable,
                        }),
                        Err(reason) => Some(PortalStatus::Lost {
                            reason,
                            configurable,
                        }),
                    };
                    if let Some(status) = status {
                        self.publish(out, status);
                    }
                }
            }
            let _ = pending.reply.send(Ok(message.clone()));
        }
    }

    /// What the forwarder tells the command loop.
    enum Lifecycle {
        /// A (new) portal owns the name again: start over, registering
        /// the app id with it first.
        PortalBack,
        /// The bus connection ended.
        BusLost,
    }

    /// One portal signal, in wire order.
    fn handle_signal(message: &zbus::Message, live: &Shared, out: &mpsc::Sender<PortalSignal>) {
        let header = message.header();
        let (Some(interface), Some(member), Some(path)) =
            (header.interface(), header.member(), header.path())
        else {
            return;
        };
        let mut live = lock(live);
        match (interface.as_str(), member.as_str()) {
            (REQUEST_INTERFACE, "Response") => {
                let waiting = live
                    .pending
                    .as_ref()
                    .is_some_and(|p| p.path == path.as_str());
                if waiting {
                    live.respond(out, message);
                }
            }
            (INTERFACE, edge @ ("Activated" | "Deactivated")) => {
                let Ok((session, id, _timestamp, _options)) =
                    message
                        .body()
                        .deserialize::<(OwnedObjectPath, String, u64, HashMap<String, OwnedValue>)>(
                        )
                else {
                    return;
                };
                if live.bound && id == SHORTCUT_ID && live.session.as_ref() == Some(&session) {
                    let now = Instant::now();
                    let _ = out.send(if edge == "Activated" {
                        PortalSignal::Activated(now)
                    } else {
                        PortalSignal::Deactivated(now)
                    });
                }
            }
            (INTERFACE, "ShortcutsChanged") => {
                let Ok((session, entries)) = message
                    .body()
                    .deserialize::<(OwnedObjectPath, Vec<ShortcutEntry>)>()
                else {
                    return;
                };
                if live.session.as_ref() != Some(&session) || !live.bound {
                    return;
                }
                let configurable = live.configurable;
                match find_ours(&entries) {
                    Some(trigger) => live.publish(
                        out,
                        PortalStatus::Bound {
                            trigger,
                            configurable,
                        },
                    ),
                    None => {
                        // The user removed it in the desktop's settings.
                        let _ = out.send(PortalSignal::SessionEnded(Instant::now()));
                        live.bound = false;
                        live.publish(
                            out,
                            PortalStatus::Lost {
                                reason: "the shortcut was removed in the desktop's settings"
                                    .to_string(),
                                configurable,
                            },
                        );
                    }
                }
            }
            (SESSION_INTERFACE, "Closed") => {
                let ours =
                    live.session.as_ref().map(|session| session.as_str()) == Some(path.as_str());
                if ours {
                    live.session_ended(out, "the desktop closed Starling's shortcut session");
                }
            }
            _ => {}
        }
    }

    /// The forwarder: every signal from the portal — request responses
    /// included — in the order the bus delivered them, handled at once,
    /// never queued behind a command or an open dialog. One stream, so a
    /// release never overtakes its press and a binding is in place before
    /// the signals that follow it are judged.
    async fn forward(
        mut signals: zbus::MessageStream,
        mut owner: OwnerChangedStream<'static>,
        live: Shared,
        out: mpsc::Sender<PortalSignal>,
        lifecycle: tokio::sync::mpsc::UnboundedSender<Lifecycle>,
    ) {
        let bus_lost = |live: &Shared| {
            let mut live = lock(live);
            let reason = "the session bus connection ended";
            live.session_ended(&out, reason);
            live.publish(&out, PortalStatus::Unavailable(reason.to_string()));
            live.terminal = true;
            let _ = lifecycle.send(Lifecycle::BusLost);
        };
        loop {
            tokio::select! {
                message = signals.next() => match message {
                    Some(Ok(message)) => handle_signal(&message, &live, &out),
                    Some(Err(_)) => {}
                    None => return bus_lost(&live),
                },
                change = owner.next() => match change {
                    // The portal went away; its sessions and any open
                    // dialog with it.
                    Some(None) => lock(&live).session_ended(&out, "the desktop portal stopped"),
                    // Started again, or replaced outright (`--replace`):
                    // the old instance's session and any open dialog end.
                    Some(Some(_)) => {
                        lock(&live).session_ended(&out, "the desktop portal restarted");
                        let _ = lifecycle.send(Lifecycle::PortalBack);
                        return;
                    }
                    None => return bus_lost(&live),
                },
            }
        }
    }

    /// Why a request failed.
    enum RequestError {
        /// The call itself failed or timed out; nothing was published.
        Call(String),
        /// The forwarder ended the wait (session closed, portal or bus
        /// gone) and published what that means.
        Ended(String),
    }

    impl RequestError {
        fn reason(self) -> String {
            match self {
                RequestError::Call(reason) | RequestError::Ended(reason) => reason,
            }
        }
    }

    struct Worker {
        connection: Connection,
        portal: Proxy<'static>,
        out: mpsc::Sender<PortalSignal>,
        version: u32,
        live: Shared,
        /// Whether a bind was attempted on the live session (the portal
        /// allows one per session).
        bind_attempted: bool,
        /// The configured shortcut, offered as `preferred_trigger`.
        preferred: Option<String>,
        tokens: u64,
        lifecycle: tokio::sync::mpsc::UnboundedReceiver<Lifecycle>,
        forwarder: tokio::task::JoinHandle<()>,
    }

    impl Drop for Worker {
        fn drop(&mut self) {
            self.forwarder.abort();
        }
    }

    /// Why a worker could not open a session.
    enum OpenError {
        /// The portal needs an app id it can resolve to a desktop entry.
        NeedsDesktopEntry,
        Failed(String),
    }

    /// How the steady state ended.
    enum Exit {
        /// The UI dropped its handle, or the bus is gone.
        Done,
        /// The portal restarted: start over on a fresh connection.
        Restart,
    }

    pub(super) async fn run(
        address: Option<String>,
        entry_dir: Option<std::path::PathBuf>,
        mut preferred: Option<String>,
        mut commands: UnboundedReceiver<Command>,
        out: mpsc::Sender<PortalSignal>,
    ) {
        let status = |status| {
            let _ = out.send(PortalSignal::Status(status, Instant::now()));
        };
        let mut bind_now = false;
        loop {
            let connection = match connect(address.clone()).await {
                Ok(connection) => connection,
                Err(reason) => return status(PortalStatus::Unavailable(reason)),
            };
            // Who owns the portal name, watched from before the first call:
            // a portal that starts, restarts or is replaced while (or after)
            // opening fails gets a fresh start instead of a permanent
            // "unavailable".
            let watching = async {
                let bus = zbus::fdo::DBusProxy::new(&connection).await?;
                bus.receive_name_owner_changed_with_args(&[(0, DESTINATION)])
                    .await
            };
            let mut owners = tokio::time::timeout(CALL_TIMEOUT, watching)
                .await
                .ok()
                .and_then(Result::ok);
            let opened = Worker::open(
                connection,
                preferred.clone(),
                out.clone(),
                entry_dir.as_deref(),
                bind_now,
            )
            .await;
            match opened {
                Ok(mut worker) => {
                    let exit = worker.serve(&mut commands).await;
                    worker.close_session().await;
                    match exit {
                        Exit::Done => return,
                        Exit::Restart => {
                            // A new portal instance knows nothing of this
                            // connection: register again on a fresh one. A
                            // binding it remembers is re-bound by `open`.
                            preferred = worker.preferred.clone();
                            bind_now = false;
                        }
                    }
                }
                Err(OpenError::Failed(reason)) => {
                    status(PortalStatus::Unavailable(reason));
                    let Some(owners) = owners.as_mut() else {
                        return;
                    };
                    // Wait for a portal to (re)take the name; the other
                    // shortcut sources keep working meanwhile.
                    loop {
                        tokio::select! {
                            change = owners.next() => match change {
                                Some(change) => {
                                    let appeared = change
                                        .args()
                                        .is_ok_and(|args| args.new_owner().is_some());
                                    if appeared {
                                        break;
                                    }
                                }
                                None => return,
                            },
                            command = commands.recv() => {
                                if command.is_none() {
                                    return;
                                }
                            }
                        }
                    }
                    bind_now = false;
                }
                Err(OpenError::NeedsDesktopEntry) => {
                    status(PortalStatus::NeedsDesktopEntry);
                    // Wait for the user's set-up; it carries the current
                    // keys.
                    loop {
                        match commands.recv().await {
                            None => return,
                            Some(Command::SetUp { trigger }) => {
                                preferred = trigger;
                                break;
                            }
                            Some(Command::Rebind { .. } | Command::Configure) => {}
                        }
                    }
                    let Some(dir) = entry_dir.as_deref() else {
                        return status(PortalStatus::Unavailable(
                            "no data folder to install Starling's desktop entry in".to_string(),
                        ));
                    };
                    if let Err(reason) = install_desktop_entry(dir) {
                        return status(PortalStatus::Unavailable(format!(
                            "Starling's desktop entry could not be installed: {reason}"
                        )));
                    }
                    // Start over on a fresh connection (the portal fixes
                    // a connection's app id at its first call) and bind
                    // right away: the user asked for it.
                    bind_now = true;
                }
            }
        }
    }

    impl Worker {
        async fn open(
            connection: Connection,
            preferred: Option<String>,
            out: mpsc::Sender<PortalSignal>,
            entry_dir: Option<&std::path::Path>,
            bind_now: bool,
        ) -> Result<Worker, OpenError> {
            let portal = bounded(async {
                zbus::proxy::Builder::<Proxy<'static>>::new(&connection)
                    .destination(DESTINATION)
                    .and_then(|b| b.path(PATH))
                    .and_then(|b| b.interface(INTERFACE))
                    .map_err(|err| err.to_string())?
                    .cache_properties(CacheProperties::No)
                    .build()
                    .await
                    .map_err(|err| err.to_string())
            })
            .await
            .map_err(OpenError::Failed)?;
            // Host apps have no app id the portal can see; without one it
            // refuses shortcut sessions. Registering must be this
            // connection's first portal call (the portal fixes a sender's
            // app id at its first call, the version read included). A
            // missing registry (portal older than 1.19) is not fatal by
            // itself: an app id from the systemd scope still works, and
            // CreateSession says if not.
            let registered = bounded(async {
                connection
                    .call_method(
                        Some(DESTINATION),
                        PATH,
                        Some(REGISTRY_INTERFACE),
                        "Register",
                        &(APP_ID, HashMap::<&str, Value>::new()),
                    )
                    .await
                    .map(drop)
                    .map_err(|err| err.to_string())
            })
            .await;
            let version = bounded(async {
                portal
                    .get_property::<u32>("version")
                    .await
                    .map_err(|err| describe_error(&err))
            })
            .await
            .map_err(OpenError::Failed)?;
            if let Err(reason) = &registered {
                eprintln!("The portal registry did not take Starling's app id ({reason}).");
            }
            // Owner changes first, so a restart from here on is seen (and
            // starts over); then every signal from this portal instance on
            // one ordered stream, subscribed before the first call that can
            // make one fire. Matched on the instance's unique name: a
            // restarted portal is a new instance and a fresh start.
            let subscribed = bounded(async {
                let subscribing = async {
                    let owner_changes = portal.receive_owner_changed().await?;
                    let bus = zbus::fdo::DBusProxy::new(&connection).await?;
                    let owner = bus
                        .get_name_owner(zbus::names::BusName::try_from(DESTINATION)?)
                        .await?;
                    let rule = zbus::MatchRule::builder()
                        .msg_type(zbus::message::Type::Signal)
                        .sender(owner)?
                        .build();
                    let signals =
                        zbus::MessageStream::for_match_rule(rule, &connection, None).await?;
                    Ok::<_, zbus::Error>((signals, owner_changes))
                };
                subscribing.await.map_err(|err| describe_error(&err))
            })
            .await
            .map_err(OpenError::Failed)?;
            let (signals, owner_changes) = subscribed;
            let live: Shared = Arc::new(Mutex::new(Live {
                configurable: version >= 2,
                ..Default::default()
            }));
            let (lifecycle_tx, lifecycle) = tokio::sync::mpsc::unbounded_channel();
            let forwarder = tokio::spawn(forward(
                signals,
                owner_changes,
                live.clone(),
                out.clone(),
                lifecycle_tx,
            ));
            let mut worker = Worker {
                connection,
                portal,
                out,
                version,
                live,
                bind_attempted: false,
                preferred,
                tokens: 0,
                lifecycle,
                forwarder,
            };
            match worker.create_session().await {
                // The session ended at once: the worker stays, and the
                // user's set-up opens a fresh one.
                Err(reason) if reason == SESSION_CLOSED_AT_ONCE => {
                    let live = lock(&worker.live);
                    live.publish(
                        &worker.out,
                        PortalStatus::NeedsSetup {
                            configurable: live.configurable,
                        },
                    );
                    drop(live);
                    return Ok(worker);
                }
                Err(reason) => {
                    // No app id: the registry (xdg-desktop-portal 1.19+) only
                    // takes one with an installed desktop entry.
                    return Err(match registered {
                        Err(_)
                            if reason.contains("app id") && !desktop_entry_installed(entry_dir) =>
                        {
                            OpenError::NeedsDesktopEntry
                        }
                        Err(_) if reason.contains("app id") => OpenError::Failed(format!(
                            "{reason}; this xdg-desktop-portal cannot identify Starling (its host \
                         app registry needs version 1.19 or newer)"
                        )),
                        _ => OpenError::Failed(reason),
                    });
                }
                Ok(()) => {}
            }
            // A binding the desktop remembers from an earlier run is
            // re-bound now: the desktop shows no dialog for it. Anything
            // else waits for the user to ask.
            let remembered = match worker.list().await {
                Ok(entries) => find_ours(&entries).is_some(),
                Err(reason) => {
                    eprintln!("Could not list the desktop's shortcuts for Starling ({reason}).");
                    false
                }
            };
            if remembered || bind_now {
                worker.bind().await;
            } else {
                let live = lock(&worker.live);
                live.publish(
                    &worker.out,
                    PortalStatus::NeedsSetup {
                        configurable: live.configurable,
                    },
                );
            }
            Ok(worker)
        }

        fn configurable(&self) -> bool {
            self.version >= 2
        }

        fn session(&self) -> Option<OwnedObjectPath> {
            lock(&self.live).session.clone()
        }

        fn token(&mut self) -> String {
            self.tokens += 1;
            format!("starling{}_{}", std::process::id(), self.tokens)
        }

        /// The object path the portal will give a request or session made
        /// with `token`: `<base>/<sender without ':' and with '.' → '_'>/<token>`.
        fn handle_path(&self, kind: &str, token: &str) -> Result<String, String> {
            let sender = self
                .connection
                .unique_name()
                .ok_or("the bus gave this connection no name")?
                .trim_start_matches(':')
                .replace('.', "_");
            Ok(format!("{PATH}/{kind}/{sender}/{token}"))
        }

        /// Call a request-style method and wait for its `Response`, which
        /// the forwarder takes off the ordered stream (registered before
        /// the call, so a fast response is never missed). `dialog`
        /// requests wait as long as the user takes; a closed session or a
        /// vanished portal ends the wait.
        async fn request<B>(
            &mut self,
            method: &str,
            token: &str,
            body: &B,
            kind: PendingKind,
            dialog: bool,
        ) -> Result<zbus::Message, RequestError>
        where
            B: serde::Serialize + zbus::zvariant::DynamicType,
        {
            let path = self
                .handle_path("request", token)
                .map_err(RequestError::Call)?;
            let (reply, response) = oneshot::channel();
            {
                let mut live = lock(&self.live);
                if live.terminal {
                    return Err(RequestError::Ended(
                        "the session bus connection ended".into(),
                    ));
                }
                live.pending = Some(Pending {
                    path: path.clone(),
                    kind,
                    reply,
                });
            }
            let clear = |live: &Shared, path: &str| {
                let mut live = lock(live);
                if live.pending.as_ref().is_some_and(|p| p.path == path) {
                    live.pending = None;
                }
            };
            let called: Result<OwnedObjectPath, String> = bounded(async {
                self.portal
                    .call(method, body)
                    .await
                    .map_err(|err| describe_error(&err))
            })
            .await;
            let handle = match called {
                Ok(handle) => handle,
                Err(reason) => {
                    clear(&self.live, &path);
                    return Err(RequestError::Call(reason));
                }
            };
            if handle.as_str() != path {
                // An old portal ignoring `handle_token`: follow the handle
                // it returned (a response that already came is lost; the
                // wait below then times out or the dialog result arrives).
                if let Some(pending) = lock(&self.live).pending.as_mut() {
                    if pending.path == path {
                        pending.path = handle.to_string();
                    }
                }
            }
            let waited = async {
                response
                    .await
                    .unwrap_or_else(|_| Err("the portal worker stopped".to_string()))
                    .map_err(RequestError::Ended)
            };
            if dialog {
                waited.await
            } else {
                match tokio::time::timeout(CALL_TIMEOUT, waited).await {
                    Ok(outcome) => outcome,
                    Err(_) => {
                        clear(&self.live, handle.as_str());
                        Err(RequestError::Call(
                            "the desktop portal did not answer".to_string(),
                        ))
                    }
                }
            }
        }

        async fn create_session(&mut self) -> Result<(), String> {
            let handle_token = self.token();
            let session_token = self.token();
            let options: HashMap<&str, Value> = HashMap::from([
                ("handle_token", Value::from(handle_token.as_str())),
                ("session_handle_token", Value::from(session_token.as_str())),
            ]);
            // The forwarder knows the session's handle before it exists,
            // so a `Closed` right after creation is not missed; it commits
            // the handle the response names, in order with later signals.
            let predicted = OwnedObjectPath::try_from(self.handle_path("session", &session_token)?)
                .map_err(|err| err.to_string())?;
            {
                let mut live = lock(&self.live);
                live.session = Some(predicted);
                live.bound = false;
            }
            self.bind_attempted = false;
            let outcome = self
                .request(
                    "CreateSession",
                    &handle_token,
                    &(options,),
                    PendingKind::CreateSession,
                    false,
                )
                .await
                .map_err(RequestError::reason)
                .and_then(|message| {
                    let (code, _) = response_body::<HashMap<String, OwnedValue>>(&message)?;
                    if code != RESPONSE_SUCCESS {
                        return Err(format!(
                            "the desktop refused a shortcut session (response {code})"
                        ));
                    }
                    Ok(())
                });
            let mut live = lock(&self.live);
            match outcome {
                Ok(()) if live.session.is_some() => Ok(()),
                Ok(()) => Err(SESSION_CLOSED_AT_ONCE.into()),
                Err(reason) => {
                    live.session = None;
                    Err(reason)
                }
            }
        }

        /// Close the live session (a rebind, or the app going away). A
        /// shortcut held down through it is released.
        async fn close_session(&mut self) {
            let session = {
                let mut live = lock(&self.live);
                live.bound = false;
                live.session.take()
            };
            let Some(session) = session else {
                return;
            };
            let _ = self.out.send(PortalSignal::SessionEnded(Instant::now()));
            let _ = tokio::time::timeout(
                CALL_TIMEOUT,
                self.connection.call_method(
                    Some(DESTINATION),
                    session.as_str(),
                    Some(SESSION_INTERFACE),
                    "Close",
                    &(),
                ),
            )
            .await;
        }

        async fn list(&mut self) -> Result<Vec<ShortcutEntry>, String> {
            let session = self.session().ok_or("no session")?;
            let token = self.token();
            let options: HashMap<&str, Value> =
                HashMap::from([("handle_token", Value::from(token.as_str()))]);
            let message = self
                .request(
                    "ListShortcuts",
                    &token,
                    &(session, options),
                    PendingKind::Plain,
                    false,
                )
                .await
                .map_err(RequestError::reason)?;
            match response_body::<ShortcutResults>(&message)? {
                (RESPONSE_SUCCESS, results) => Ok(results.shortcuts.unwrap_or_default()),
                (code, _) => Err(format!("response {code}")),
            }
        }

        /// Bind the shortcut in the live session, opening a fresh session
        /// first when this one already had its one bind. The outcome is
        /// published by the forwarder as the response arrives.
        async fn bind(&mut self) {
            if self.session().is_none() || self.bind_attempted {
                self.close_session().await;
                if let Err(reason) = self.create_session().await {
                    lock(&self.live).publish(
                        &self.out,
                        PortalStatus::Lost {
                            reason,
                            configurable: false,
                        },
                    );
                    return;
                }
            }
            let Some(session) = self.session() else {
                return;
            };
            self.bind_attempted = true;
            lock(&self.live).publish(&self.out, PortalStatus::Binding);
            let token = self.token();
            let mut info: HashMap<&str, Value> =
                HashMap::from([("description", Value::from(SHORTCUT_DESCRIPTION))]);
            if let Some(trigger) = self.preferred.as_deref() {
                info.insert("preferred_trigger", Value::from(trigger.to_string()));
            }
            let shortcuts = vec![(SHORTCUT_ID, info)];
            let options: HashMap<&str, Value> =
                HashMap::from([("handle_token", Value::from(token.as_str()))]);
            let kind = PendingKind::Bind {
                session: session.clone(),
            };
            let outcome = self
                .request(
                    "BindShortcuts",
                    &token,
                    &(session, shortcuts, "", options),
                    kind,
                    true,
                )
                .await;
            if let Err(RequestError::Call(reason)) = outcome {
                let live = lock(&self.live);
                live.publish(
                    &self.out,
                    PortalStatus::Lost {
                        reason,
                        configurable: live.configurable,
                    },
                );
            }
        }

        async fn configure(&mut self) {
            let Some(session) = self.session() else {
                return;
            };
            if !self.configurable() {
                return;
            }
            let options: HashMap<&str, Value> = HashMap::new();
            let outcome = bounded(async {
                self.portal
                    .call_method("ConfigureShortcuts", &(session, "", options))
                    .await
                    .map(drop)
                    .map_err(|err| describe_error(&err))
            })
            .await;
            if let Err(reason) = outcome {
                eprintln!("The desktop's shortcut settings could not be opened ({reason}).");
            }
        }

        /// The steady state: UI commands, and what the forwarder reports
        /// about the portal's lifetime. Signals never wait here: the
        /// forwarder handles them as they arrive.
        async fn serve(&mut self, commands: &mut UnboundedReceiver<Command>) -> Exit {
            loop {
                tokio::select! {
                    command = commands.recv() => match command {
                        Some(command) => self.command(command).await,
                        None => return Exit::Done,
                    },
                    event = self.lifecycle.recv() => match event {
                        Some(Lifecycle::PortalBack) => return Exit::Restart,
                        Some(Lifecycle::BusLost) | None => return Exit::Done,
                    },
                }
            }
        }

        async fn command(&mut self, command: Command) {
            match command {
                Command::SetUp { trigger } => {
                    self.preferred = trigger;
                    self.bind().await;
                }
                Command::Rebind { trigger } => {
                    let changed = self.preferred != trigger;
                    self.preferred = trigger;
                    // Without a binding there is nothing to move: the next
                    // set-up offers the new keys.
                    if changed && lock(&self.live).bound {
                        self.bind().await;
                    }
                }
                Command::Configure => self.configure().await,
            }
        }
    }
}

#[cfg(all(test, target_os = "linux"))]
mod bus_tests;

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::activation::{Activation, ActivationConfig, CancelReason, Effect, TakeId};
    use starling_dictation::settings::ActivationMode;

    fn ms(n: u64) -> Duration {
        Duration::from_millis(n)
    }

    fn machine(mode: ActivationMode) -> Activation {
        Activation::new(ActivationConfig {
            mode,
            double_tap_hands_free: false,
        })
    }

    /// Feed portal signals through the mapping into the machine, the way
    /// `flush_system_events` does.
    fn feed(keys: &mut PortalKeys, machine: &mut Activation, signal: PortalSignal) -> Vec<Effect> {
        match keys.map(&signal) {
            Some(GlobalEvent::Pressed(at)) => machine.press(at, true),
            Some(GlobalEvent::Released(at)) => machine.release(at),
            Some(GlobalEvent::Escape(_)) => machine.escape(),
            None => Vec::new(),
        }
    }

    fn started(effects: &[Effect]) -> TakeId {
        let [Effect::Start(take)] = effects[..] else {
            panic!("expected a start, got {effects:?}");
        };
        take
    }

    #[test]
    fn activated_then_deactivated_is_a_hold_to_talk_take() {
        let (mut keys, mut m) = (PortalKeys::default(), machine(ActivationMode::Hold));
        let t0 = Instant::now();
        let take = started(&feed(&mut keys, &mut m, PortalSignal::Activated(t0)));
        m.samples_arrived(take);
        assert_eq!(
            feed(&mut keys, &mut m, PortalSignal::Deactivated(t0 + ms(2000))),
            vec![Effect::Finish(take)]
        );
        assert!(!m.is_active());
    }

    #[test]
    fn a_portal_tap_latches_in_hold_or_toggle_mode() {
        let (mut keys, mut m) = (PortalKeys::default(), machine(ActivationMode::HoldOrToggle));
        let t0 = Instant::now();
        let take = started(&feed(&mut keys, &mut m, PortalSignal::Activated(t0)));
        m.samples_arrived(take);
        assert!(feed(&mut keys, &mut m, PortalSignal::Deactivated(t0 + ms(120))).is_empty());
        assert!(m.is_active());
        assert_eq!(
            feed(&mut keys, &mut m, PortalSignal::Activated(t0 + ms(3000))),
            vec![Effect::Finish(take)]
        );
        assert!(feed(&mut keys, &mut m, PortalSignal::Deactivated(t0 + ms(3100))).is_empty());
    }

    #[test]
    fn repeated_activations_never_start_or_stop_a_take() {
        // A backend that repeats `Activated` while the keys are held.
        for mode in [
            ActivationMode::Toggle,
            ActivationMode::Hold,
            ActivationMode::HoldOrToggle,
        ] {
            let (mut keys, mut m) = (PortalKeys::default(), machine(mode));
            let t0 = Instant::now();
            let take = started(&feed(&mut keys, &mut m, PortalSignal::Activated(t0)));
            m.samples_arrived(take);
            let mut at = t0 + ms(500);
            for _ in 0..50 {
                assert!(
                    feed(&mut keys, &mut m, PortalSignal::Activated(at)).is_empty(),
                    "{mode:?}"
                );
                at += ms(33);
            }
            assert_eq!(m.active_take(), Some(take), "{mode:?}");
        }
    }

    #[test]
    fn a_stray_or_duplicate_deactivation_is_dropped() {
        let (mut keys, mut m) = (PortalKeys::default(), machine(ActivationMode::Toggle));
        let t0 = Instant::now();
        // A release with no press (out of order, or across a session swap).
        assert_eq!(keys.map(&PortalSignal::Deactivated(t0)), None);
        let take = started(&feed(
            &mut keys,
            &mut m,
            PortalSignal::Activated(t0 + ms(10)),
        ));
        m.samples_arrived(take);
        assert!(feed(&mut keys, &mut m, PortalSignal::Deactivated(t0 + ms(60))).is_empty());
        // Sent twice: the second one is not passed on.
        assert_eq!(keys.map(&PortalSignal::Deactivated(t0 + ms(61))), None);
        assert!(m.is_active(), "toggle keeps the take after the release");
    }

    #[test]
    fn a_session_closed_mid_hold_finishes_the_take_like_a_lost_release() {
        let (mut keys, mut m) = (PortalKeys::default(), machine(ActivationMode::Hold));
        let t0 = Instant::now();
        let take = started(&feed(&mut keys, &mut m, PortalSignal::Activated(t0)));
        m.samples_arrived(take);
        assert_eq!(
            feed(&mut keys, &mut m, PortalSignal::SessionEnded(t0 + ms(1500))),
            vec![Effect::Finish(take)]
        );
        // A late `Deactivated` from the dead session changes nothing, and
        // the next press starts normally.
        assert_eq!(keys.map(&PortalSignal::Deactivated(t0 + ms(1600))), None);
        assert_eq!(
            feed(&mut keys, &mut m, PortalSignal::Activated(t0 + ms(3000))),
            vec![Effect::Start(take + 1)]
        );
    }

    #[test]
    fn a_session_closed_before_any_audio_cancels_and_an_idle_close_does_nothing() {
        let (mut keys, mut m) = (PortalKeys::default(), machine(ActivationMode::Hold));
        let t0 = Instant::now();
        assert_eq!(keys.map(&PortalSignal::SessionEnded(t0)), None);
        let take = started(&feed(
            &mut keys,
            &mut m,
            PortalSignal::Activated(t0 + ms(10)),
        ));
        assert_eq!(
            feed(&mut keys, &mut m, PortalSignal::SessionEnded(t0 + ms(400))),
            vec![Effect::Cancel(take, CancelReason::NoAudioYet)]
        );
    }

    #[test]
    fn a_lost_portal_release_recovers_on_the_next_press() {
        // The `Deactivated` never came (and no session end either): the
        // machine's lost-release rule finishes the held take.
        let (mut keys, mut m) = (PortalKeys::default(), machine(ActivationMode::Hold));
        let t0 = Instant::now();
        let take = started(&feed(&mut keys, &mut m, PortalSignal::Activated(t0)));
        m.samples_arrived(take);
        assert_eq!(
            feed(&mut keys, &mut m, PortalSignal::Activated(t0 + ms(5000))),
            vec![Effect::Finish(take)]
        );
    }

    #[test]
    fn the_portal_and_the_window_seeing_one_press_take_it_once() {
        // A compositor that both triggers the portal and forwards the keys
        // to the focused Starling window: one take, one finish.
        let (mut keys, mut m) = (PortalKeys::default(), machine(ActivationMode::Toggle));
        let t0 = Instant::now();
        let take = started(&feed(&mut keys, &mut m, PortalSignal::Activated(t0)));
        m.samples_arrived(take);
        assert!(m.press_in_window(t0 + ms(5), true).is_empty());
        assert!(m.release(t0 + ms(80)).is_empty());
        assert!(feed(&mut keys, &mut m, PortalSignal::Deactivated(t0 + ms(82))).is_empty());
        assert_eq!(m.active_take(), Some(take));
    }

    #[test]
    fn the_desktop_entry_quotes_the_executable() {
        let entry = desktop_entry(std::path::Path::new("/opt/my apps/star$ling"));
        assert!(entry.starts_with("[Desktop Entry]\nType=Application\nName=Starling\n"));
        assert!(
            entry.contains("\nExec=\"/opt/my apps/star\\$ling\"\n"),
            "{entry}"
        );
        assert!(entry.contains(&format!("StartupWMClass={APP_ID}\n")));
        let dir = tempfile::tempdir().unwrap();
        assert!(!desktop_entry_installed(Some(dir.path())));
        install_desktop_entry(dir.path()).unwrap();
        assert!(desktop_entry_installed(Some(dir.path())));
    }

    #[test]
    fn grab_events_are_judged_by_the_binding_when_they_were_received() {
        let (sender, signals) = mpsc::channel();
        let mut portal = PortalShortcuts {
            commands: None,
            signals,
            keys: PortalKeys::default(),
            pending: Default::default(),
            bound_changes: Vec::new(),
            status: PortalStatus::Starting,
            status_changed: false,
        };
        let t0 = Instant::now();
        let bound = PortalStatus::Bound {
            trigger: None,
            configurable: false,
        };
        sender
            .send(PortalSignal::Status(bound.clone(), t0 + ms(100)))
            .unwrap();
        sender
            .send(PortalSignal::Status(bound, t0 + ms(150)))
            .unwrap();
        let lost = PortalStatus::Lost {
            reason: "gone".to_string(),
            configurable: false,
        };
        sender
            .send(PortalSignal::Status(lost, t0 + ms(300)))
            .unwrap();
        portal.poll();
        assert!(
            !portal.bound_at(t0 + ms(99)),
            "a grab press before the binding passes"
        );
        assert!(portal.bound_at(t0 + ms(100)));
        assert!(portal.bound_at(t0 + ms(299)));
        assert!(!portal.bound_at(t0 + ms(300)));
        assert_eq!(
            portal.bound_changes.len(),
            2,
            "a repeated Bound is no transition"
        );
    }

    #[test]
    fn status_text_names_what_to_do() {
        assert!(
            PortalStatus::NeedsSetup {
                configurable: false
            }
            .can_set_up()
        );
        assert!(!PortalStatus::Binding.can_set_up());
        let bound = PortalStatus::Bound {
            trigger: Some("Ctrl+Shift+Space".to_string()),
            configurable: true,
        };
        assert!(bound.is_bound() && bound.configurable());
        assert!(bound.describe().contains("Ctrl+Shift+Space"));
        let lost = PortalStatus::Lost {
            reason: "the desktop portal stopped".to_string(),
            configurable: false,
        };
        assert!(lost.can_set_up());
        assert!(lost.describe().contains("the desktop portal stopped"));
        assert!(
            PortalStatus::Unavailable("no GlobalShortcuts".to_string())
                .describe()
                .contains("no GlobalShortcuts")
        );
    }
}
