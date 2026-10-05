//! The configurable recording shortcut (#221): parsing and validation,
//! the in-window key matcher, and the system-wide registration.
//!
//! Two sources feed the activation machine with presses and releases of
//! the same shortcut. The system-wide one (`global-hotkey`: X11 key grab,
//! Windows `RegisterHotKey`, macOS Carbon hot keys) consumes the key
//! where it works, so the window rarely sees it too; the in-window one
//! covers sessions where the system-wide grab cannot reach (a native
//! Wayland window) or registration failed. Where both do see one press,
//! the machine's key-down tracking makes the second report a repeat.
//!
//! Neither source ever raises or focuses the Starling window: the app the
//! user is dictating into keeps keyboard focus.

use std::str::FromStr;
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
}

impl Shortcut {
    /// Parse and validate a shortcut in `global-hotkey` notation
    /// (`CmdOrCtrl+Shift+Space`, `F9`, `Alt+D`).
    pub(crate) fn parse(text: &str) -> Result<Shortcut, String> {
        let text = text.trim();
        if text.is_empty() {
            return Err("Enter a shortcut, for example Ctrl+Shift+Space or F9.".to_string());
        }
        let hotkey = HotKey::from_str(text).map_err(|_| {
            format!(
                "\"{text}\" is not a shortcut Starling can register. Use modifiers and one \
                 key, for example Ctrl+Shift+Space, Alt+D, or F9."
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
        Ok(Shortcut {
            hotkey,
            text: text.to_string(),
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
    /// key (`?` rather than shift-`/` on a US layout); both forms match.
    pub(crate) fn matches_key_down(&self, keystroke: &gpui::Keystroke) -> bool {
        let Some(key) = gpui_key(self.hotkey.key) else {
            return false;
        };
        let mods = self.hotkey.mods;
        let others = keystroke.modifiers.control == mods.contains(Modifiers::CONTROL)
            && keystroke.modifiers.alt == mods.contains(Modifiers::ALT)
            && keystroke.modifiers.platform == mods.contains(Modifiers::SUPER);
        let shift = mods.contains(Modifiers::SHIFT);
        let plain = keystroke.key == key && keystroke.modifiers.shift == shift;
        let shifted = shift && us_shifted(&key).is_some_and(|symbol| keystroke.key == symbol);
        others && (plain || shifted)
    }

    /// Whether an in-window key-up ends this shortcut. Only the key
    /// counts: people let go of the modifiers first as often as last, and
    /// a hold must end either way.
    pub(crate) fn matches_key_up(&self, keystroke: &gpui::Keystroke) -> bool {
        gpui_key(self.hotkey.key).is_some_and(|key| {
            keystroke.key == key || us_shifted(&key).is_some_and(|symbol| keystroke.key == symbol)
        })
    }

    pub(crate) fn modifiers(&self) -> Modifiers {
        self.hotkey.mods
    }

    /// Whether the window can see this shortcut at all (keys gpui does
    /// not name, like Pause, only work system-wide).
    pub(crate) fn works_in_window(&self) -> bool {
        gpui_key(self.hotkey.key).is_some()
    }
}

/// Whether an in-window key-down is the Escape that cancels a take. Any
/// modifiers count: the shortcut's own may still be held (Escape while
/// holding Ctrl+Shift+Space must cancel, not wait for the release).
pub(crate) fn is_escape(keystroke: &gpui::Keystroke) -> bool {
    keystroke.key == "escape"
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
/// modifier. Grabbing one of them system-wide would break typing.
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
            | End | PageUp | PageDown
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

/// The system-wide registrations: the recording shortcut for the app's
/// lifetime, and Escape only while a take is active (grabbing Escape the
/// rest of the time would take it away from every other app).
pub(crate) struct GlobalShortcuts {
    manager: GlobalHotKeyManager,
    events: mpsc::Receiver<(u32, HotKeyState, Instant)>,
    record: Option<HotKey>,
    /// The Escape registrations while a take is active: bare Escape, and
    /// Escape with the recording shortcut's modifiers (still held during a
    /// push-to-talk take).
    escape: Vec<HotKey>,
}

impl GlobalShortcuts {
    /// Create the manager and route its events here. Only one instance may
    /// exist: the event handler is process-wide.
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
        let manager = GlobalHotKeyManager::new().map_err(|err| err.to_string())?;
        let (sender, events) = mpsc::channel();
        GlobalHotKeyEvent::set_event_handler(Some(move |event: GlobalHotKeyEvent| {
            let _ = sender.send((event.id(), event.state(), Instant::now()));
        }));
        Ok(GlobalShortcuts {
            manager,
            events,
            record: None,
            escape: Vec::new(),
        })
    }

    /// Replace the recording shortcut. On failure nothing is registered
    /// (the old shortcut is not kept: the user asked for the new one).
    pub(crate) fn set_record(&mut self, shortcut: &Shortcut) -> Result<(), String> {
        if self.record == Some(shortcut.hotkey()) {
            return Ok(());
        }
        if let Some(old) = self.record.take() {
            let _ = self.manager.unregister(old);
        }
        self.manager
            .register(shortcut.hotkey())
            .map_err(|err| err.to_string())?;
        self.record = Some(shortcut.hotkey());
        Ok(())
    }

    /// Grab Escape system-wide while a take is active, release it after.
    /// Bare Escape is required; Escape with the shortcut's modifiers is
    /// best effort (the platform may reserve it, like Ctrl+Shift+Escape on
    /// Windows), and the window still sees it either way.
    pub(crate) fn arm_escape(&mut self, armed: bool, shortcut: &Shortcut) -> Result<(), String> {
        if !armed {
            let mut result = Ok(());
            for escape in self.escape.drain(..) {
                if let Err(err) = self.manager.unregister(escape) {
                    result = Err(err.to_string());
                }
            }
            return result;
        }
        if !self.escape.is_empty() {
            return Ok(());
        }
        let bare = HotKey::new(None, Code::Escape);
        self.manager.register(bare).map_err(|err| err.to_string())?;
        self.escape.push(bare);
        let mods = shortcut.modifiers();
        if !mods.is_empty() {
            let held = HotKey::new(Some(mods), Code::Escape);
            if self.manager.register(held).is_ok() {
                self.escape.push(held);
            }
        }
        Ok(())
    }

    pub(crate) fn escape_armed(&self) -> bool {
        !self.escape.is_empty()
    }

    /// Events received since the last call, oldest first. Events for a
    /// shortcut that has since been replaced are dropped.
    pub(crate) fn drain(&self) -> Vec<GlobalEvent> {
        let record = self.record.map(|hotkey| hotkey.id());
        let escape: Vec<u32> = self.escape.iter().map(|hotkey| hotkey.id()).collect();
        self.events
            .try_iter()
            .filter_map(|(id, state, at)| {
                if Some(id) == record {
                    Some(match state {
                        HotKeyState::Pressed => GlobalEvent::Pressed(at),
                        HotKeyState::Released => GlobalEvent::Released(at),
                    })
                } else if escape.contains(&id) && state == HotKeyState::Pressed {
                    Some(GlobalEvent::Escape(at))
                } else {
                    None
                }
            })
            .collect()
    }
}

/// What the settings dialog says about where the shortcut works.
pub(crate) fn reach_note(registered: Result<(), &str>, shortcut: &Shortcut) -> String {
    let wayland = cfg!(target_os = "linux") && std::env::var_os("WAYLAND_DISPLAY").is_some();
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
        for text in ["Space", "A", "Enter", "7", "Slash"] {
            let err = Shortcut::parse(text).unwrap_err();
            assert!(err.contains("modifier"), "{text}: {err}");
        }
        assert!(Shortcut::parse("Ctrl+A").is_ok());
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
