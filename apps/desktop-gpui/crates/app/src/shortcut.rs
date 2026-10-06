//! The configurable recording shortcut (#221): parsing and validation,
//! the in-window key matcher, and the system-wide registration.
//!
//! Two sources feed the activation machine, and one physical press is
//! taken from only one of them. The system-wide one (`global-hotkey`: X11
//! key grab, Windows `RegisterHotKey`, macOS Carbon hot keys) consumes the
//! key wherever its grab matches, so the window never sees that press;
//! the window's own key events cover the rest (no registration, an X11
//! lock modifier the grab does not list, a native Wayland window). The
//! one place both can fire is Wayland, where a compositor may forward a
//! native window's keys to XWayland as well (KDE's legacy X11 app
//! support): there, system-wide events that arrive while the Starling
//! window is focused are dropped (see `StarlingApp::system_event_is_ours`).
//!
//! Neither source ever raises or focuses the Starling window: the app the
//! user is dictating into keeps keyboard focus.

use std::str::FromStr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::Instant;

use global_hotkey::hotkey::{Code, HotKey, Modifiers};
use global_hotkey::{GlobalHotKeyEvent, GlobalHotKeyManager, HotKeyState};

/// A validated recording shortcut.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Shortcut {
    hotkey: HotKey,
    /// The text the user configured, as stored in settings.
    text: String,
    /// The key's name in `gpui::Keystroke::key`, and the character
    /// Shift makes of it on a US layout (the form some platforms report
    /// a shifted symbol key in), both precomputed once: matching a
    /// keystroke is the path every in-window key flows through, and it
    /// must not allocate or re-derive per key. Both follow from the
    /// hotkey, so equal shortcuts still compare equal.
    window_key: Option<String>,
    window_shifted: Option<&'static str>,
}

impl Shortcut {
    /// Parse and validate a shortcut in `global-hotkey` notation
    /// (`CmdOrCtrl+Shift+Space`, `F9`, `Alt+D`).
    pub(crate) fn parse(text: &str) -> Result<Shortcut, String> {
        let text = text.trim();
        if text.is_empty() {
            return Err("Enter a shortcut, for example Ctrl+Shift+Space or F9.".to_string());
        }
        let hotkey = HotKey::from_str(text).map_err(|err| {
            format!(
                "\"{text}\" is not a shortcut Starling can register ({err}). Use modifiers \
                 and one key, for example Ctrl+Shift+Space, Alt+D, or F9."
            )
        })?;
        if hotkey.key == Code::Escape {
            return Err("Escape is reserved for cancelling a take.".to_string());
        }
        if hotkey.mods.is_empty() && is_typing_key(hotkey.key) {
            return Err(format!(
                "{} on its own would take that key away from typing everywhere. Add a \
                 modifier, or use a key such as F9, Pause, or Insert.",
                key_label(hotkey.key)
            ));
        }
        let window_key = gpui_key(hotkey.key);
        let window_shifted = window_key.as_deref().and_then(us_shifted);
        Ok(Shortcut {
            hotkey,
            text: text.to_string(),
            window_key,
            window_shifted,
        })
    }

    pub(crate) fn hotkey(&self) -> HotKey {
        self.hotkey
    }

    pub(crate) fn text(&self) -> &str {
        &self.text
    }

    /// How the shortcut reads on screen: `Ctrl Shift Space`, `⌘ Shift Space`.
    pub(crate) fn label(&self) -> String {
        let mods = self.hotkey.mods;
        let mut parts: Vec<String> = Vec::new();
        if mods.contains(Modifiers::SUPER) {
            parts.push(if cfg!(target_os = "macos") { "⌘" } else { "Super" }.to_string());
        }
        if mods.contains(Modifiers::CONTROL) {
            parts.push("Ctrl".to_string());
        }
        if mods.contains(Modifiers::ALT) {
            parts.push(if cfg!(target_os = "macos") { "Option" } else { "Alt" }.to_string());
        }
        if mods.contains(Modifiers::SHIFT) {
            parts.push("Shift".to_string());
        }
        parts.push(key_label(self.hotkey.key));
        parts.join(" ")
    }

    /// Whether an in-window key-down is this shortcut (exact modifiers).
    /// With Shift, gpui may report the shifted character instead of the
    /// key (`?` rather than shift-`/` on a US layout), and a shifted
    /// letter as either the lowercase key with Shift held — the shape
    /// every gpui platform produces — or the uppercase name, which only
    /// x11's compose path emits. All of these forms match.
    pub(crate) fn matches_key_down(&self, keystroke: &gpui::Keystroke) -> bool {
        let Some(key) = self.window_key.as_deref() else {
            return false;
        };
        let mods = self.hotkey.mods;
        let others = keystroke.modifiers.control == mods.contains(Modifiers::CONTROL)
            && keystroke.modifiers.alt == mods.contains(Modifiers::ALT)
            && keystroke.modifiers.platform == mods.contains(Modifiers::SUPER);
        let shift = mods.contains(Modifiers::SHIFT);
        // A single ASCII letter matches case-insensitively when Shift is
        // part of the shortcut: gpui lowercases a shifted letter on every
        // platform (x11 and Wayland lowercase `key_utf8`, macOS reports
        // `charactersIgnoringModifiers`, Windows lowercases the vkey
        // character), but x11's compose path sets `key` straight from
        // `keysym_get_name`, which hands back the uppercase name.
        let plain = (keystroke.key == key
            || (shift && is_ascii_letter(key) && keystroke.key.eq_ignore_ascii_case(key)))
            && keystroke.modifiers.shift == shift;
        let shifted = shift && self.window_shifted.is_some_and(|symbol| keystroke.key == symbol);
        others && (plain || shifted)
    }

    /// Whether an in-window key-up ends this shortcut. Only the key
    /// counts: people let go of the modifiers first as often as last, and
    /// a hold must end either way. The key may arrive in either letter
    /// case or as the character Shift makes of it (see
    /// [`Shortcut::matches_key_down`]).
    pub(crate) fn matches_key_up(&self, keystroke: &gpui::Keystroke) -> bool {
        self.window_key.as_deref().is_some_and(|key| {
            keystroke.key == key
                || (is_ascii_letter(key) && keystroke.key.eq_ignore_ascii_case(key))
                || self
                    .window_shifted
                    .is_some_and(|symbol| keystroke.key == symbol)
        })
    }

    pub(crate) fn modifiers(&self) -> Modifiers {
        self.hotkey.mods
    }

    /// Whether the window can see this shortcut at all (keys gpui does
    /// not name, like Pause, only work system-wide).
    pub(crate) fn works_in_window(&self) -> bool {
        self.window_key.is_some()
    }
}

/// Whether an in-window key-down is the Escape that cancels a take. Any
/// modifiers count: the shortcut's own may still be held (Escape while
/// holding Ctrl+Shift+Space must cancel, not wait for the release).
pub(crate) fn is_escape(keystroke: &gpui::Keystroke) -> bool {
    keystroke.key == "escape"
}

/// Whether a gpui key name is one ASCII letter: the one key whose
/// shifted form reaches the window in either case (see
/// [`Shortcut::matches_key_down`]).
fn is_ascii_letter(key: &str) -> bool {
    let bytes = key.as_bytes();
    bytes.len() == 1 && bytes[0].is_ascii_alphabetic()
}

/// The character Shift turns a key into on a US layout, the form some
/// platforms report a shifted symbol key in.
fn us_shifted(key: &str) -> Option<&'static str> {
    Some(match key {
        "`" => "~",
        "1" => "!",
        "2" => "@",
        "3" => "#",
        "4" => "$",
        "5" => "%",
        "6" => "^",
        "7" => "&",
        "8" => "*",
        "9" => "(",
        "0" => ")",
        "-" => "_",
        "=" => "+",
        "[" => "{",
        "]" => "}",
        "\\" => "|",
        ";" => ":",
        "'" => "\"",
        "," => "<",
        "." => ">",
        "/" => "?",
        _ => return None,
    })
}

/// Keys that type text (or move through it) when pressed without a
/// modifier. Grabbing one of them system-wide would break typing —
/// the numpad included: `Numpad1` types a digit like `1` does.
fn is_typing_key(code: Code) -> bool {
    use Code::*;
    matches!(
        code,
        Backquote | Backslash | BracketLeft | BracketRight | Comma | Digit0 | Digit1 | Digit2
            | Digit3 | Digit4 | Digit5 | Digit6 | Digit7 | Digit8 | Digit9 | Equal | KeyA
            | KeyB | KeyC | KeyD | KeyE | KeyF | KeyG | KeyH | KeyI | KeyJ | KeyK | KeyL
            | KeyM | KeyN | KeyO | KeyP | KeyQ | KeyR | KeyS | KeyT | KeyU | KeyV | KeyW
            | KeyX | KeyY | KeyZ | Minus | Period | Quote | Semicolon | Slash | Space | Enter
            | Tab | Backspace | Delete | ArrowUp | ArrowDown | ArrowLeft | ArrowRight | Home
            | End | PageUp | PageDown | Numpad0 | Numpad1 | Numpad2 | Numpad3 | Numpad4
            | Numpad5 | Numpad6 | Numpad7 | Numpad8 | Numpad9 | NumpadAdd | NumpadDecimal
            | NumpadDivide | NumpadEnter | NumpadEqual | NumpadMultiply | NumpadSubtract
    )
}

/// The key's on-screen name.
fn key_label(code: Code) -> String {
    let name = code.to_string();
    let name = name
        .strip_prefix("Key")
        .or_else(|| name.strip_prefix("Digit"))
        .unwrap_or(&name);
    match name {
        "ArrowUp" => "Up".to_string(),
        "ArrowDown" => "Down".to_string(),
        "ArrowLeft" => "Left".to_string(),
        "ArrowRight" => "Right".to_string(),
        other => other.to_string(),
    }
}

/// The name gpui gives the key in `Keystroke::key`, when it has one.
fn gpui_key(code: Code) -> Option<String> {
    use Code::*;
    let name = match code {
        Space => "space",
        Enter => "enter",
        Tab => "tab",
        Backspace => "backspace",
        Delete => "delete",
        Insert => "insert",
        Home => "home",
        End => "end",
        PageUp => "pageup",
        PageDown => "pagedown",
        ArrowUp => "up",
        ArrowDown => "down",
        ArrowLeft => "left",
        ArrowRight => "right",
        Backquote => "`",
        Minus => "-",
        Equal => "=",
        BracketLeft => "[",
        BracketRight => "]",
        Backslash => "\\",
        Semicolon => ";",
        Quote => "'",
        Comma => ",",
        Period => ".",
        Slash => "/",
        _ => {
            let label = key_label(code);
            let lower = label.to_ascii_lowercase();
            let letter_or_digit = label.len() == 1 && label.chars().all(|c| c.is_ascii_alphanumeric());
            let function = lower.starts_with('f')
                && lower.len() > 1
                && lower[1..].chars().all(|c| c.is_ascii_digit());
            return (letter_or_digit || function).then_some(lower);
        }
    };
    Some(name.to_string())
}

/// One system-wide shortcut event, timestamped where it was received —
/// on the hotkey thread, not when the UI loop got round to it — so hold
/// and tap durations are measured honestly.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GlobalEvent {
    Pressed(Instant),
    Released(Instant),
    Escape(Instant),
}

/// One event as the hotkey thread received it.
pub(crate) type RawEvent = (u32, HotKeyState, Instant);

/// The Escape grabs a take wants: bare Escape, plus Escape with every
/// subset of the shortcut's modifiers — any of them may still be held
/// (or already let go) when Escape comes. Taking them is intended: for
/// the duration of the take, Escape — alone or with any of the
/// shortcut's modifiers — cancels the take and is not delivered to
/// other apps, so a chord the desktop may reserve (Ctrl+Shift+Escape,
/// Alt+Escape) counts as a cancel while a take is active.
fn escape_variants(shortcut: &Shortcut) -> Vec<HotKey> {
    let mut variants = vec![HotKey::new(None, Code::Escape)];
    let mods = shortcut.modifiers();
    let parts: Vec<Modifiers> = [
        Modifiers::CONTROL,
        Modifiers::SHIFT,
        Modifiers::ALT,
        Modifiers::SUPER,
    ]
    .into_iter()
    .filter(|part| mods.contains(*part))
    .collect();
    // Largest subsets first: the X11 worker registers one variant per
    // 50 ms, and the likeliest Escape mid-take comes with the whole chord
    // still held, so that grab must not wait behind the partial ones.
    let full = (1u32 << parts.len()) - 1;
    let mut masks: Vec<u32> = (1..=full).collect();
    masks.sort_by_key(|mask| std::cmp::Reverse(mask.count_ones()));
    for mask in masks {
        let subset = parts
            .iter()
            .enumerate()
            .filter(|(bit, _)| mask & (1 << bit) != 0)
            .fold(Modifiers::empty(), |acc, (_, part)| acc | *part);
        variants.push(HotKey::new(Some(subset), Code::Escape));
    }
    variants
}

/// Register one Escape grab, logging a refusal with the one message every
/// arming path shares (the inline Windows/macOS loop and the Linux
/// worker's); `Err` carries the refusal for callers that act on it.
fn register_escape_variant(
    manager: &GlobalHotKeyManager,
    variant: &HotKey,
) -> Result<(), String> {
    manager.register(*variant).map_err(|err| {
        eprintln!(
            "Escape could not be grabbed system-wide ({err}); the Starling window still \
             sees it."
        );
        err.to_string()
    })
}

/// Release one Escape grab, logging a failure with the one message every
/// disarm path shares (the inline loop and the Linux worker's).
fn release_escape_variant(manager: &GlobalHotKeyManager, variant: &HotKey) {
    if let Err(err) = manager.unregister(*variant) {
        eprintln!("An Escape grab could not be released ({err}).");
    }
}

/// Release the previous recording shortcut's grab once the new one holds
/// (both `set_record` paths). A failed release leaves the old and the new
/// shortcut live system-wide — logged, never swallowed, since either may
/// then toggle recording.
fn release_previous_record(manager: &GlobalHotKeyManager, old: Option<HotKey>) {
    if let Some(old) = old {
        if let Err(err) = manager.unregister(old) {
            eprintln!(
                "the previous shortcut's grab could not be released ({err}); both it and \
                 the new shortcut may now toggle recording"
            );
        }
    }
}

/// A command for the Linux worker thread that owns the manager.
enum WorkerCommand {
    /// Register `new`; once it holds, release `old`. The reply carries
    /// the refusal when `new` did not register (the old one stays).
    SetRecord {
        new: HotKey,
        old: Option<HotKey>,
        reply: mpsc::Sender<Result<(), String>>,
    },
    /// Register every Escape variant; the report names the refusals.
    Arm {
        generation: u64,
        variants: Vec<HotKey>,
    },
    /// Release the Escape grabs an arm took.
    Disarm {
        variants: Vec<HotKey>,
    },
}

/// The worker's answer to an [`WorkerCommand::Arm`]: which arming it
/// answers, and which Escape variants the platform refused.
struct ArmReport {
    generation: u64,
    refused: Vec<HotKey>,
}

/// The Linux manager's home: a plain thread that runs every registration
/// command in order. Arm and disarm arrive fire-and-forget — their
/// latency is the X11 backend's one-command-per-50-ms loop, which must
/// never land on the UI thread, and arming happens at every take start
/// and end — while `SetRecord` (a settings save) answers through its own
/// reply channel.
#[cfg(target_os = "linux")]
fn manage(
    manager: GlobalHotKeyManager,
    commands: mpsc::Receiver<WorkerCommand>,
    report: mpsc::Sender<ArmReport>,
) {
    while let Ok(command) = commands.recv() {
        match command {
            WorkerCommand::SetRecord { new, old, reply } => {
                let outcome = manager
                    .register(new)
                    .map_err(|err| err.to_string())
                    .map(|()| release_previous_record(&manager, old));
                let _ = reply.send(outcome);
            }
            WorkerCommand::Arm { generation, variants } => {
                // Each variant registers on its own, never `register_all`:
                // the backend replies over a one-slot channel, and a batch
                // carrying two failures would block its loop for good. A
                // refusal is logged and simply never fires.
                let refused = variants
                    .iter()
                    .filter(|variant| register_escape_variant(&manager, variant).is_err())
                    .copied()
                    .collect();
                let _ = report.send(ArmReport { generation, refused });
            }
            WorkerCommand::Disarm { variants } => {
                // One release per variant, for the same reason as arming.
                for variant in &variants {
                    release_escape_variant(&manager, variant);
                }
            }
        }
    }
}

/// Whether a `GlobalShortcuts` has taken the process-wide event handler.
/// `GlobalHotKeyEvent::set_event_handler` routes every hotkey event to one
/// closure, so a second instance would silently replace it and leave the
/// first's registrations firing into a dropped sender — the shortcut
/// would look registered but never act. One instance per process,
/// enforced: the claim `new` makes is handed back in `Drop`, so the
/// invariant survives a dropped instance instead of being assumed.
static HANDLER_INSTALLED: AtomicBool = AtomicBool::new(false);

/// The system-wide registrations: the recording shortcut for the app's
/// lifetime, and Escape only while a take is active (grabbing Escape the
/// rest of the time would take it away from every other app).
pub(crate) struct GlobalShortcuts {
    /// The manager, inline on Windows/macOS: their registrations are fast
    /// kernel/Carbon calls (and macOS wants the main thread). On Linux it
    /// moved to the worker thread (`worker`): every manager call there is
    /// a synchronous round trip to global-hotkey's X11 backend, which
    /// services one command per 50 ms loop.
    manager: Option<GlobalHotKeyManager>,
    /// Linux: where the manager lives; `None` on Windows/macOS.
    worker: Option<mpsc::Sender<WorkerCommand>>,
    events: mpsc::Receiver<RawEvent>,
    /// The worker's answers to an arm, whenever they land.
    reports: mpsc::Receiver<ArmReport>,
    record: Option<HotKey>,
    /// The Escape registrations a disarm must release, as requested:
    /// bare Escape and Escape with (subsets of) the shortcut's
    /// modifiers.
    escape: Vec<HotKey>,
    /// The variants that can actually fire: the request until the
    /// worker's report lands, then exactly the registered ones (a
    /// refused variant never fires, so it leaves here the moment the
    /// refusal is known).
    escape_live: Vec<HotKey>,
    /// Which arming the in-flight reports belong to.
    escape_generation: u64,
}

impl GlobalShortcuts {
    /// Create the manager and route its events here. Only one instance may
    /// exist — enforced below, not just documented: the event handler is
    /// process-wide, and a second one would orphan the first's grabs.
    pub(crate) fn new() -> Result<GlobalShortcuts, String> {
        // On Linux `global-hotkey` only speaks X11, and with no display it
        // fails silently (its registrations report success with no backend
        // behind them). Say so instead of claiming a registration.
        if cfg!(target_os = "linux") && std::env::var_os("DISPLAY").is_none_or(|d| d.is_empty()) {
            return Err(
                "no X11 display (DISPLAY is not set); the system-wide shortcut needs X11 or \
                 XWayland"
                    .to_string(),
            );
        }
        if HANDLER_INSTALLED.swap(true, Ordering::SeqCst) {
            return Err(
                "a GlobalShortcuts already exists in this process; the hotkey event \
                 handler is process-wide"
                    .to_string(),
            );
        }
        // The claim comes before any platform work — a second instance is
        // refused before it builds anything — and a manager that cannot
        // be built hands the slot back, so a failed construction does not
        // poison the next attempt.
        let manager = match GlobalHotKeyManager::new() {
            Ok(manager) => manager,
            Err(err) => {
                HANDLER_INSTALLED.store(false, Ordering::SeqCst);
                return Err(err.to_string());
            }
        };
        let (sender, events) = mpsc::channel();
        GlobalHotKeyEvent::set_event_handler(Some(move |event: GlobalHotKeyEvent| {
            let _ = sender.send((event.id(), event.state(), Instant::now()));
        }));
        #[cfg(target_os = "linux")]
        let (manager, worker, reports) = {
            let (report, reports) = mpsc::channel();
            let (commands, inbox) = mpsc::channel();
            // The manager moves off the UI thread for good: its X11 round
            // trips (one backend command per 50 ms) must never stall a
            // take's start or end. Moving it in pins that it is `Send`.
            std::thread::spawn(move || manage(manager, inbox, report));
            (None, Some(commands), reports)
        };
        #[cfg(not(target_os = "linux"))]
        let (manager, worker, reports) = {
            let (_, reports) = mpsc::channel();
            (Some(manager), None, reports)
        };
        Ok(GlobalShortcuts {
            manager,
            worker,
            events,
            reports,
            record: None,
            escape: Vec::new(),
            escape_live: Vec::new(),
            escape_generation: 0,
        })
    }

    /// Replace the recording shortcut. The new grab is taken first and
    /// the old one released only once it holds, so a platform refusal
    /// leaves the previous shortcut registered and running — never no
    /// shortcut at all.
    pub(crate) fn set_record(&mut self, shortcut: &Shortcut) -> Result<(), String> {
        if self.record == Some(shortcut.hotkey()) {
            return Ok(());
        }
        let new = shortcut.hotkey();
        let old = self.record;
        let outcome = if let Some(manager) = self.manager.as_ref() {
            // Windows/macOS: direct calls, fine to make inline.
            manager
                .register(new)
                .map_err(|err| err.to_string())
                .map(|()| release_previous_record(manager, old))
        } else if let Some(worker) = self.worker.as_ref() {
            // Linux: a request/reply through the worker. Blocking is fine —
            // this runs on a settings save, not per take.
            let (reply, replies) = mpsc::channel();
            worker
                .send(WorkerCommand::SetRecord { new, old, reply })
                .map_err(|err| err.to_string())?;
            replies
                .recv()
                .map_err(|_| "the shortcut worker is gone".to_string())?
        } else {
            Ok(())
        };
        self.record = match &outcome {
            Ok(()) => Some(new),
            // The old registration was never released.
            Err(_) => old,
        };
        outcome
    }

    /// Grab Escape system-wide while a take is active, release it after.
    /// The variants are the shortcut's modifier subsets too (see
    /// [`escape_variants`]): while a take is active, Escape — alone or
    /// with any of the shortcut's modifiers — cancels it and is not
    /// delivered to other apps. A variant the platform reserves
    /// (Ctrl+Shift+Escape on Windows) is best effort — the window still
    /// sees Escape either way. Inline (Windows/macOS) a refused bare
    /// Escape fails the arm and releases whatever did register, so the
    /// state never says armed on a partial set; on Linux the arm only
    /// posts to the worker, so even that refusal is known asynchronously
    /// — the worker logs it, it never reaches this call's result, and
    /// the window still sees Escape. The grabs come up off the UI thread.
    pub(crate) fn arm_escape(&mut self, armed: bool, shortcut: &Shortcut) -> Result<(), String> {
        if !armed {
            let variants = std::mem::take(&mut self.escape);
            let live = std::mem::take(&mut self.escape_live);
            if variants.is_empty() {
                return Ok(());
            }
            // Only what registered is released, on either path: a variant
            // the platform refused at arm time was never grabbed, and
            // unregistering it would only log an error. On Linux the
            // not-yet-reported refusals may still ride along in `live`;
            // the worker logs those and carries on.
            return match self.worker.as_ref() {
                Some(worker) => worker
                    .send(WorkerCommand::Disarm { variants: live })
                    .map_err(|err| err.to_string()),
                None => {
                    let manager = self
                        .manager
                        .as_ref()
                        .expect("the manager is inline without a worker");
                    // One release per variant, never `unregister_all`: it
                    // fails fast on a variant the platform refused at arm
                    // time and would leave every grab after it live. An
                    // individual failure is logged and the rest released.
                    for variant in &live {
                        release_escape_variant(manager, variant);
                    }
                    Ok(())
                }
            };
        }
        if !self.escape.is_empty() {
            return Ok(());
        }
        let variants = escape_variants(shortcut);
        self.escape = variants.clone();
        self.escape_live = variants.clone();
        self.escape_generation += 1;
        if let Some(worker) = self.worker.as_ref() {
            let generation = self.escape_generation;
            let sent = worker.send(WorkerCommand::Arm { generation, variants });
            // A failed send means nothing registered: the fields above
            // already claim the armed state, so hand it back — otherwise
            // `escape_armed()` stays true forever, every later arm
            // early-returns at the guard above, and the system-wide
            // Escape cancel is silently dead for the session. Cleared
            // here, the next arm (and the disarm at take end) retries.
            return match sent {
                Ok(()) => Ok(()),
                Err(err) => {
                    self.escape.clear();
                    self.escape_live.clear();
                    Err(err.to_string())
                }
            };
        }
        let manager = self
            .manager
            .as_ref()
            .expect("the manager is inline without a worker");
        // Each variant registers on its own — never `register_all`: the
        // X11 backend replies over a one-slot channel, and a batch
        // carrying two failures would block its loop for good. A refusal
        // is logged, never fires, and leaves `escape_live` without it;
        // only bare Escape failing fails the arm.
        let mut bare_refused = None;
        for variant in &variants {
            if let Err(reason) = register_escape_variant(manager, variant) {
                self.escape_live.retain(|live| live != variant);
                if variant.mods.is_empty() {
                    bare_refused = Some(reason);
                }
            }
        }
        if let Some(reason) = bare_refused {
            // Bare Escape is the one grab the arm promises, so a refusal
            // fails the arm: the variants that did register are released
            // and the armed state cleared, so the next arm retries
            // cleanly instead of sitting "armed" on a partial set.
            for variant in &self.escape_live {
                release_escape_variant(manager, variant);
            }
            self.escape.clear();
            self.escape_live.clear();
            return Err(reason);
        }
        Ok(())
    }

    pub(crate) fn escape_armed(&self) -> bool {
        !self.escape.is_empty()
    }

    /// The next received event, oldest first, not yet classified. The
    /// worker's refusal reports fold in first, so classification sees
    /// the Escape grabs as they ended up, not as they were requested.
    pub(crate) fn next_raw(&mut self) -> Option<RawEvent> {
        while let Ok(report) = self.reports.try_recv() {
            if report.generation == self.escape_generation {
                self.escape_live
                    .retain(|live| !report.refused.contains(live));
            }
        }
        self.events.try_recv().ok()
    }

    /// Classify a received event against the registrations as they are
    /// *now*: the caller classifies each event only when it processes it,
    /// so an event for a shortcut (or an Escape grab) replaced by an
    /// earlier event in the same batch is dropped, never misapplied.
    /// Only the Escape grabs that actually registered count.
    pub(crate) fn classify(&self, (id, state, at): RawEvent) -> Option<GlobalEvent> {
        if self.record.is_some_and(|hotkey| hotkey.id() == id) {
            Some(match state {
                HotKeyState::Pressed => GlobalEvent::Pressed(at),
                HotKeyState::Released => GlobalEvent::Released(at),
            })
        } else if state == HotKeyState::Pressed
            && self.escape_live.iter().any(|hotkey| hotkey.id() == id)
        {
            Some(GlobalEvent::Escape(at))
        } else {
            None
        }
    }
}

impl Drop for GlobalShortcuts {
    /// The process-wide event handler and the single-instance slot are
    /// handed back, so "one `GlobalShortcuts` per process" is enforced
    /// rather than assumed: after a drop a new instance works instead of
    /// failing forever with "already exists". The registrations go with
    /// it — the inline manager (Windows/macOS) drops as this struct tears
    /// down, and on Linux the worker's command sender is a field of this
    /// struct, so that teardown ends the worker loop, whose exit drops
    /// the manager on the worker thread.
    fn drop(&mut self) {
        // `None` uninstalls the handler; events global-hotkey still
        // delivers afterwards land in its default channel, not in the
        // sender this instance's handler closure captured.
        GlobalHotKeyEvent::set_event_handler(None::<fn(GlobalHotKeyEvent)>);
        HANDLER_INSTALLED.store(false, Ordering::SeqCst);
    }
}

/// Whether this is a Wayland session (the window is a native Wayland
/// client the X11 grab cannot hear).
pub(crate) fn wayland_session() -> bool {
    cfg!(target_os = "linux") && std::env::var_os("WAYLAND_DISPLAY").is_some_and(|d| !d.is_empty())
}

/// What the settings dialog says about where the shortcut works.
pub(crate) fn reach_note(registered: &Result<(), String>, shortcut: &Shortcut) -> String {
    let wayland = wayland_session();
    let in_window = if shortcut.works_in_window() {
        "It always works while the Starling window is focused."
    } else {
        "This key has no in-window fallback, so it works only where the system-wide shortcut does."
    };
    match registered {
        Err(reason) => format!(
            "The system-wide shortcut could not be registered ({reason}). {in_window}"
        ),
        Ok(()) if wayland => format!(
            "Wayland session: the system-wide shortcut reaches Starling only while an X11 \
             (XWayland) app is focused; the desktop portal is not supported yet. {in_window}"
        ),
        Ok(()) => format!("Registered system-wide. {in_window}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keystroke(text: &str) -> gpui::Keystroke {
        gpui::Keystroke::parse(text).expect("keystroke")
    }

    #[test]
    fn the_default_shortcut_parses_and_matches_its_keystroke() {
        let shortcut = Shortcut::parse(starling_dictation::settings::DEFAULT_SHORTCUT).unwrap();
        assert!(shortcut.matches_key_down(&keystroke("secondary-shift-space")));
        assert!(!shortcut.matches_key_down(&keystroke("secondary-space")));
        assert!(!shortcut.matches_key_down(&keystroke("shift-space")));
        assert!(shortcut.works_in_window());
        if cfg!(target_os = "macos") {
            assert_eq!(shortcut.label(), "⌘ Shift Space");
        } else {
            assert_eq!(shortcut.label(), "Ctrl Shift Space");
        }
    }

    #[test]
    fn a_single_function_key_is_a_valid_shortcut() {
        let shortcut = Shortcut::parse(" F9 ").unwrap();
        assert_eq!(shortcut.text(), "F9");
        assert_eq!(shortcut.label(), "F9");
        assert!(shortcut.matches_key_down(&keystroke("f9")));
        assert!(!shortcut.matches_key_down(&keystroke("ctrl-f9")));
    }

    #[test]
    fn keys_without_an_in_window_name_still_register_system_wide() {
        let shortcut = Shortcut::parse("Pause").unwrap();
        assert!(!shortcut.works_in_window());
        assert!(!shortcut.matches_key_down(&keystroke("space")));
    }

    #[test]
    fn a_release_matches_on_the_key_alone() {
        let shortcut = Shortcut::parse("Ctrl+Shift+Space").unwrap();
        assert!(shortcut.matches_key_up(&keystroke("space")));
        assert!(shortcut.matches_key_up(&keystroke("ctrl-space")));
        assert!(!shortcut.matches_key_up(&keystroke("ctrl-shift-a")));
    }

    #[test]
    fn letters_and_symbols_map_to_gpui_names() {
        let shortcut = Shortcut::parse("Alt+D").unwrap();
        assert!(shortcut.matches_key_down(&keystroke("alt-d")));
        assert_eq!(shortcut.label(), if cfg!(target_os = "macos") { "Option D" } else { "Alt D" });
        let shortcut = Shortcut::parse("Super+Slash").unwrap();
        assert!(shortcut.matches_key_down(&keystroke("cmd-/")));
    }

    #[test]
    fn typing_keys_need_a_modifier() {
        for text in [
            "Space",
            "A",
            "Enter",
            "7",
            "Slash",
            "Numpad1",
            "NumpadEnter",
            "NumpadDecimal",
            "NumpadAdd",
            "NumpadEqual",
        ] {
            let err = Shortcut::parse(text).unwrap_err();
            assert!(err.contains("modifier"), "{text}: {err}");
        }
        assert!(Shortcut::parse("Ctrl+A").is_ok());
        assert!(Shortcut::parse("Ctrl+Numpad1").is_ok());
    }

    #[test]
    fn escape_grabs_cover_the_bare_key_and_every_modifiers_subset() {
        let shortcut = Shortcut::parse("Ctrl+Shift+Space").unwrap();
        let variants = escape_variants(&shortcut);
        assert_eq!(variants.len(), 4, "bare + Ctrl + Shift + Ctrl+Shift");
        assert!(variants[0].mods.is_empty());
        // The whole chord right after bare Escape: the likeliest one held.
        assert_eq!(variants[1].mods, Modifiers::CONTROL | Modifiers::SHIFT);
        assert!(variants.iter().all(|hotkey| hotkey.key == Code::Escape));
        // Distinct registrations, so a refusal can be told apart.
        let ids: std::collections::HashSet<u32> =
            variants.iter().map(|hotkey| hotkey.id()).collect();
        assert_eq!(ids.len(), variants.len());
    }

    #[test]
    fn a_second_global_shortcuts_is_refused() {
        // The hotkey event handler is process-wide, so a second instance
        // would leave the first's registrations firing into a dropped
        // sender. With a display the refusal is observed through `new`
        // itself, and dropping the first hands the slot back: a new
        // instance is allowed again rather than refused forever.
        // Headless (CI) not even the first manager can be built, so the
        // claim `new` makes is exercised directly and restored.
        match GlobalShortcuts::new() {
            Ok(shortcuts) => {
                let err = match GlobalShortcuts::new() {
                    Ok(_second) => panic!("a second GlobalShortcuts must be refused"),
                    Err(err) => err,
                };
                assert!(err.contains("already exists"), "{err}");
                drop(shortcuts);
                let _revived = GlobalShortcuts::new()
                    .expect("a GlobalShortcuts is allowed again after the drop");
            }
            Err(_) => {
                assert!(!HANDLER_INSTALLED.swap(true, Ordering::SeqCst));
                assert!(HANDLER_INSTALLED.swap(true, Ordering::SeqCst));
                HANDLER_INSTALLED.store(false, Ordering::SeqCst);
            }
        }
    }

    #[test]
    fn escape_and_garbage_are_refused() {
        assert!(Shortcut::parse("Escape").unwrap_err().contains("reserved"));
        assert!(Shortcut::parse("Ctrl+Escape").unwrap_err().contains("reserved"));
        assert!(Shortcut::parse("").is_err());
        assert!(Shortcut::parse("Ctrl").is_err());
        assert!(Shortcut::parse("Ctrl+Shift+Banana").is_err());
        assert!(Shortcut::parse("Ctrl+A+B").is_err());
    }

    #[test]
    fn escape_cancels_even_with_the_shortcut_modifiers_held() {
        assert!(is_escape(&keystroke("escape")));
        assert!(is_escape(&keystroke("ctrl-shift-escape")));
        assert!(!is_escape(&keystroke("space")));
    }

    #[test]
    fn shifted_letter_shortcuts_match_both_event_shapes() {
        // The shape every gpui platform produces for a shifted letter:
        // the lowercase key with Shift held (x11 and Wayland lowercase
        // `key_utf8`, macOS reports `charactersIgnoringModifiers`, Windows
        // lowercases the vkey character) — also what `Keystroke::parse`
        // builds, which is why the parse-based event below is that shape.
        let shortcut = Shortcut::parse("Alt+Shift+D").unwrap();
        assert!(shortcut.matches_key_down(&keystroke("alt-shift-d")));
        // The x11 compose path sets `key` from `keysym_get_name` without
        // lowercasing, so the uppercase name must match as well.
        let mut upper = keystroke("alt-shift-d");
        upper.key = "D".to_string();
        assert!(shortcut.matches_key_down(&upper));
        // The release matches on the key alone, in either case.
        assert!(shortcut.matches_key_up(&keystroke("d")));
        let mut upper_up = keystroke("d");
        upper_up.key = "D".to_string();
        assert!(shortcut.matches_key_up(&upper_up));
        // Shift held still mismatches a shortcut that has no Shift.
        let plain = Shortcut::parse("Alt+D").unwrap();
        assert!(!plain.matches_key_down(&keystroke("alt-shift-d")));
        let mut plain_upper = keystroke("alt-d");
        plain_upper.key = "D".to_string();
        assert!(!plain_upper.modifiers.shift);
        assert!(!plain.matches_key_down(&plain_upper));
    }

    #[test]
    fn shifted_symbol_shortcuts_match_either_event_shape() {
        let shortcut = Shortcut::parse("Ctrl+Shift+Slash").unwrap();
        assert!(shortcut.matches_key_down(&keystroke("ctrl-shift-/")));
        // gpui on Linux/macOS: Shift absorbed into the character.
        let mut absorbed = keystroke("ctrl-?");
        assert!(!absorbed.modifiers.shift);
        assert!(shortcut.matches_key_down(&absorbed));
        absorbed.modifiers.shift = true;
        assert!(shortcut.matches_key_down(&absorbed));
        assert!(shortcut.matches_key_up(&keystroke("?")));
        // Without Shift in the shortcut, the symbol is a different key.
        let plain = Shortcut::parse("Ctrl+Slash").unwrap();
        assert!(!plain.matches_key_down(&keystroke("ctrl-?")));
        let digit = Shortcut::parse("Ctrl+Shift+1").unwrap();
        assert!(digit.matches_key_down(&keystroke("ctrl-!")));
    }
}
