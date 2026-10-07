//! The X11 backend: focus identity from EWMH + core focus, typing via
//! XTest synthetic keys (issue #221, slice 2 phase A).
//!
//! # Identity
//!
//! Capture reads `_NET_ACTIVE_WINDOW` (the window the WM says is
//! active) and `GetInputFocus` (the window keys actually go to — for a
//! real editor that is an input child of the active toplevel). The ref
//! is `x11:<active-hex>:<focus-hex>:<pid>`; revalidate compares both
//! ids and checks the window still exists, because focus and activity
//! move independently (a dialog can take focus while the active window
//! stays, and an app can close while focus lingers on a doomed child).
//!
//! `_NET_ACTIVE_WINDOW` needs an EWMH window manager. WSLg runs
//! "Weston WM" and does maintain it, but a bare Xvfb has none, so when
//! the property is absent or `None` the backend falls back to the
//! focus window as the active anchor — the one window keys go to is
//! the honest identity there. On a Wayland session
//! (`WAYLAND_DISPLAY` set) there is no honest fallback at all: native
//! Wayland windows are invisible to X11 and the X focus view is stale
//! or empty, so the backend reports itself unavailable rather than
//! guess a target (the portal or IBus, slice 2b, is the fix).
//!
//! # Typing
//!
//! XTest fake key events are keycode-level: the server delivers the
//! keycode to the focus window and the *target* decodes it through the
//! keyboard mapping. Only the base and Shift columns of that mapping
//! are trusted (column 0/1); the core protocol cannot tell a
//! single-group four-level mapping from a two-group two-level one, and
//! guessing the deeper columns wrong means *typing the wrong
//! character*. Everything else — a `ß` on a US layout, `→` anywhere,
//! `é` unless the layout has it on a base key — borrows the same
//! always-correct mechanism instead: one spare keycode (the highest
//! whose keysyms are all `NoSymbol`) is temporarily remapped
//! (`ChangeKeyboardMapping`) to the character's keysym, typed at
//! column 0, and *restored afterwards, on the error path too*
//! (`MappingGuard` is a drop guard) — leaving the user's keyboard
//! remapped would be a real-world breakage, not a test nit. The remap
//! is server-global by nature; it is brief (one press/release and a
//! round trip) and skipped entirely whenever the mapping already
//! produces the character.
//!
//! # Connections
//!
//! The backend opens a fresh connection per operation. Capture happens
//! when a take starts and insert minutes later; a cached connection
//! could be dead by then, and reconnecting (an async-safe Unix socket
//! round trip) is cheaper than the state machine that would have to
//! detect and replace it. Within `insert`, revalidation and typing
//! share one connection so the target check and the keys see the same
//! server state.

use x11rb::connection::Connection;
use x11rb::errors::ConnectError;
use x11rb::protocol::ErrorKind;
use x11rb::protocol::xproto::{
    Atom, ConnectionExt, GetPropertyReply, Keycode, Keysym, Window,
};
use x11rb::protocol::xtest;
use x11rb::rust_connection::RustConnection;

use crate::{
    format_ref, insertion_guards, parse_ref, Availability, BackendKind, InsertError,
    InsertReceipt, InsertionBackend, SurroundingText, TargetCheck, TargetSnapshot,
    EVIDENCE_SYNTHETIC_KEYS,
};

/// KeyPress as an event *type* number for XTest fake input (the core
/// protocol names the events but not their numbers).
const KEY_PRESS: u8 = 2;
/// KeyRelease as an event type number.
const KEY_RELEASE: u8 = 3;
/// `GetInputFocus` reports this when no window has focus.
const FOCUS_NONE: Window = 0;
/// ... and this when focus follows the pointer instead of a window.
const FOCUS_POINTER_ROOT: Window = 1;
/// `XK_Shift_L` / `XK_Shift_R` — either keycode pressed gives the
/// keyboard the Shift state a column-1 keysym needs.
const XK_SHIFT_L: Keysym = 0xffe1;
const XK_SHIFT_R: Keysym = 0xffe2;

/// How long a synthetic key stays "down". Real keys are down for tens
/// of milliseconds; a zero hold can be dropped or coalesced by
/// toolkits that watch press/release pairing, and the X server's own
/// auto-repeat only fires for keys held far longer than this.
const KEY_HOLD: std::time::Duration = std::time::Duration::from_millis(3);
/// Gap between characters, so event-driven targets (and their IME
/// layers, which often settle per key) can keep up with a burst.
const KEY_GAP: std::time::Duration = std::time::Duration::from_millis(1);

/// The X11 backend. Stateless: every call opens its own connection
/// (see the module docs for why that is simpler and *more* robust than
/// caching one).
#[derive(Debug, Default)]
pub struct X11Backend;

impl X11Backend {
    pub fn new() -> X11Backend {
        X11Backend
    }
}

/// One live X session: a connection plus the screen's root window.
struct Session {
    conn: RustConnection,
    root: Window,
}

impl Session {
    fn open() -> Result<Session, InsertError> {
        let (conn, screen) =
            x11rb::connect(None).map_err(|error| unavailable(&connect_error(&error)))?;
        let root = conn.setup().roots[screen].root;
        Ok(Session { conn, root })
    }

    /// A cheap round trip; used to prove the server has processed
    /// everything queued so far (before restoring a keyboard mapping,
    /// and before an insert receipt claims the keys were sent).
    fn sync(&self) -> Result<(), InsertError> {
        self.conn
            .get_input_focus()
            .map_err(x11_conn_error)?
            .reply()
            .map_err(reply_error)?;
        Ok(())
    }

    fn atom(&self, name: &str) -> Result<Atom, InsertError> {
        Ok(self
            .conn
            .intern_atom(false, name.as_bytes())
            .map_err(x11_conn_error)?
            .reply()
            .map_err(reply_error)?
            .atom)
    }

    /// Read a property's raw bytes; `None` when absent.
    fn get_property(
        &self,
        window: Window,
        property: Atom,
    ) -> Result<Option<GetPropertyReply>, InsertError> {
        let reply = self
            .conn
            .get_property(false, window, property, x11rb::NONE, 0, 64)
            .map_err(x11_conn_error)?
            .reply()
            .map_err(reply_error)?;
        if reply.type_ == x11rb::NONE {
            Ok(None)
        } else {
            Ok(Some(reply))
        }
    }

    /// Read a 32-bit CARDINAL property; `None` when absent.
    fn card32(&self, window: Window, property: Atom) -> Result<Option<u32>, InsertError> {
        let reply = self.get_property(window, property)?;
        match &reply {
            Some(reply) if reply.format == 32 && reply.value.len() >= 4 => Ok(Some(
                u32::from_ne_bytes(
                    reply.value[..4]
                        .try_into()
                        .expect("four bytes are four bytes"),
                ),
            )),
            _ => Ok(None),
        }
    }

    fn window_exists(&self, window: Window) -> Result<bool, InsertError> {
        match self
            .conn
            .get_window_attributes(window)
            .map_err(x11_conn_error)?
            .reply()
        {
            Ok(_) => Ok(true),
            Err(x11rb::errors::ReplyError::X11Error(error))
                if error.error_kind == ErrorKind::Window =>
            {
                Ok(false)
            }
            Err(error) => Err(reply_error(error)),
        }
    }

    /// The EWMH active window, `None` when the WM does not report one.
    fn ewmh_active_window(&self) -> Result<Option<Window>, InsertError> {
        let atom = self.atom("_NET_ACTIVE_WINDOW")?;
        let active = self.card32(self.root, atom)?;
        match active {
            // 0 is EWMH's "no active window"; anything else is a
            // window id (only the WM could have set it, so trust it).
            Some(id) if id != FOCUS_NONE => Ok(Some(id)),
            _ => Ok(None),
        }
    }

    /// `(active, focus)` as one consistent view. When no EWMH active
    /// window exists the focus window is the anchor (see the module
    /// docs); when the focus itself is `None`/`PointerRoot` there is
    /// no target at all and this returns `None` — never a guess.
    fn focus_pair(&self) -> Result<Option<(Window, Window)>, InsertError> {
        let focus = self
            .conn
            .get_input_focus()
            .map_err(x11_conn_error)?
            .reply()
            .map_err(reply_error)?
            .focus;
        if focus == FOCUS_NONE || focus == FOCUS_POINTER_ROOT {
            return Ok(None);
        }
        let active = match self.ewmh_active_window()? {
            Some(active) => active,
            None => focus,
        };
        Ok(Some((active, focus)))
    }

    /// `_NET_WM_PID`, the EWMH convention; plenty of apps set it, and
    /// the ones that do not leave the pid unknown (the ref then simply
    /// omits it, and the Starling-owns-it guard degrades to the
    /// inserting process only).
    fn wm_pid(&self, window: Window) -> Result<Option<u32>, InsertError> {
        let atom = self.atom("_NET_WM_PID")?;
        self.card32(window, atom)
    }

    /// `WM_CLASS`'s class half ("instance\0class\0" per ICCCM); the
    /// class is the stable, human-facing app name.
    fn wm_class(&self, window: Window) -> Result<Option<String>, InsertError> {
        let atom = self.atom("WM_CLASS")?;
        let Some(reply) = self.get_property(window, atom)? else {
            return Ok(None);
        };
        let parts: Vec<&[u8]> = reply
            .value
            .split(|&byte| byte == 0)
            .filter(|part| !part.is_empty())
            .collect();
        Ok(parts
            .last()
            .map(|class| String::from_utf8_lossy(class).into_owned()))
    }

    /// `_NET_WM_NAME` (UTF-8) with a `WM_NAME` (Latin-1 `STRING`)
    /// fallback — display-only, so lossy decoding is acceptable there.
    fn window_title(&self, window: Window) -> Result<Option<String>, InsertError> {
        let net = self.atom("_NET_WM_NAME")?;
        if let Some(reply) = self.get_property(window, net)? {
            return Ok(Some(String::from_utf8_lossy(&reply.value).into_owned()));
        }
        let wm = self.atom("WM_NAME")?;
        if let Some(reply) = self.get_property(window, wm)? {
            // Type STRING is Latin-1: every byte is a code point.
            return Ok(Some(reply.value.iter().map(|&byte| byte as char).collect()));
        }
        Ok(None)
    }
}

impl InsertionBackend for X11Backend {
    fn kind(&self) -> BackendKind {
        BackendKind::X11
    }

    fn availability(&self) -> Availability {
        // A Wayland session hides its native windows from X11 entirely;
        // the X focus view cannot be trusted to describe the user's
        // desktop, so this backend refuses to capture at all instead
        // of guessing (the portal / IBus backends, slice 2b, are the
        // supported route there).
        if std::env::var_os("WAYLAND_DISPLAY").is_some_and(|value| !value.is_empty()) {
            return Availability::Unavailable {
                reason: "Wayland session: native Wayland windows are invisible to X11; needs \
                         the portal or IBus (slice 2b)"
                    .to_string(),
                setup_hint: None,
            };
        }
        if std::env::var_os("DISPLAY").map_or(true, |value| value.is_empty()) {
            return Availability::Unavailable {
                reason: "no X display: DISPLAY is not set".to_string(),
                setup_hint: Some("run under X11/XWayland with DISPLAY set".to_string()),
            };
        }
        // DISPLAY being set does not mean the server answers; probe so
        // "Ready" is a claim this backend can defend.
        match x11rb::connect(None) {
            Ok(_) => Availability::Ready,
            Err(error) => Availability::Unavailable {
                reason: format!("cannot reach the X server: {}", connect_error(&error)),
                setup_hint: None,
            },
        }
    }

    fn capture(&self) -> Result<TargetSnapshot, InsertError> {
        // An unavailable backend has nothing to capture; reuse the
        // availability wording so callers see one story.
        match self.availability() {
            Availability::Ready => {}
            Availability::Unavailable { reason, setup_hint } => {
                return Err(InsertError::Unavailable { reason, setup_hint })
            }
        }
        let session = Session::open()?;
        let Some((active, focus)) = session.focus_pair()? else {
            return Err(InsertError::Rejected {
                reason: "no window has input focus".to_string(),
            });
        };
        let pid = session.wm_pid(active)?;
        if pid == Some(std::process::id()) {
            // Starling never types into Starling, and capture is where
            // the app learns its target, so the refusal starts here
            // (insert re-checks against the parsed ref).
            return Err(InsertError::TargetIsStarling);
        }
        let app = session.wm_class(active)?;
        let title = session.window_title(active)?;
        Ok(TargetSnapshot {
            backend: BackendKind::X11,
            target_ref: format_ref(BackendKind::X11, active as u64, focus as u64, pid),
            app,
            title,
            pid,
            capabilities: BackendKind::X11.capabilities(),
        })
    }

    fn revalidate(&self, target: &TargetSnapshot) -> Result<TargetCheck, InsertError> {
        let session = Session::open()?;
        self.revalidate_on(&session, target)
    }

    fn surrounding_text(
        &self,
        _target: &TargetSnapshot,
    ) -> Result<Option<SurroundingText>, InsertError> {
        // No portable X11 core protocol reports text around the
        // cursor; the capability table says so and the answer is None.
        Ok(None)
    }

    fn insert(&self, target: &TargetSnapshot, text: &str) -> Result<InsertReceipt, InsertError> {
        // One connection for the check and the keys (module docs): the
        // revalidation inside the guards is over the same server view
        // the typing rides.
        let session = Session::open()?;
        insertion_guards(text, target.pid, || self.revalidate_on(&session, target))?;

        let keyboard = Keyboard::load(&session)?;
        for character in text.chars() {
            type_character(&session, &keyboard, character)?;
        }
        // Prove the server processed every queued event before the
        // receipt claims keys were sent.
        session.sync()?;
        Ok(InsertReceipt {
            evidence: EVIDENCE_SYNTHETIC_KEYS,
        })
    }
}

impl X11Backend {
    /// The live comparison behind both standalone `revalidate` and
    /// the in-insert guard.
    fn revalidate_on(
        &self,
        session: &Session,
        target: &TargetSnapshot,
    ) -> Result<TargetCheck, InsertError> {
        let Some((kind, active, focus, _pid)) = parse_ref(&target.target_ref) else {
            return Err(InsertError::Rejected {
                reason: format!("malformed target ref: {}", target.target_ref),
            });
        };
        debug_assert_eq!(kind, BackendKind::X11, "the inserter routes by scheme");
        // A destroyed window is `Gone` even if focus also moved — the
        // app closing is the fact the user needs; where focus went is
        // secondary detail they can see.
        if !session.window_exists(active as Window)? {
            return Ok(TargetCheck::Gone);
        }
        match session.focus_pair()? {
            Some((live_active, live_focus))
                if live_active == active as Window && live_focus == focus as Window =>
            {
                Ok(TargetCheck::Same)
            }
            // Focus or activity moved: name the new state honestly —
            // a real ref when one can be captured, else a scheme tag
            // that says "not a window".
            Some((live_active, live_focus)) => {
                let live_pid = session.wm_pid(live_active).unwrap_or(None);
                Ok(TargetCheck::Changed {
                    expected: target.target_ref.clone(),
                    actual: format_ref(kind, live_active as u64, live_focus as u64, live_pid),
                })
            }
            None => Ok(TargetCheck::Changed {
                expected: target.target_ref.clone(),
                actual: format!("{}:none", kind.scheme()),
            }),
        }
    }
}

/// The keyboard mapping snapshot one insert types through, plus the
/// Shift keycode a column-1 keysym needs. The mapping is server-global
/// and *does* change underneath (the remap path itself does that), so
/// answers are consumed immediately, never cached across characters.
struct Keyboard {
    min_keycode: Keycode,
    /// Keysyms per keycode (the flat `GetKeyboardMapping` layout).
    width: usize,
    syms: Vec<Keysym>,
    /// A keycode that carries `Shift_L`/`Shift_R`; `None` only on a
    /// keyboard with no shift at all (none exists in practice, and
    /// column-1 characters then fall back to the remap path).
    shift: Option<Keycode>,
}

impl Keyboard {
    fn load(session: &Session) -> Result<Keyboard, InsertError> {
        let setup = session.conn.setup();
        let min = setup.min_keycode;
        let max = setup.max_keycode;
        let reply = session
            .conn
            .get_keyboard_mapping(min, max - min + 1)
            .map_err(x11_conn_error)?
            .reply()
            .map_err(reply_error)?;
        let width = reply.keysyms_per_keycode as usize;
        let syms = reply.keysyms;
        if width == 0 || syms.len() != (max - min + 1) as usize * width {
            // A server that lies about the mapping cannot be typed
            // through safely; refuse rather than index out of bounds.
            return Err(InsertError::Rejected {
                reason: "the X server reported an unusable keyboard mapping".to_string(),
            });
        }
        let keyboard = Keyboard {
            min_keycode: min,
            width,
            syms,
            shift: None,
        };
        // The Shift key: any keycode whose base columns carry a Shift
        // keysym. Modifier-map indirection is unnecessary — the
        // keycode's own keysyms are what a fake press delivers.
        let shift = (min..=max).find(|&keycode| {
            keyboard
                .keysyms(keycode)
                .iter()
                .take(2)
                .any(|&sym| sym == XK_SHIFT_L || sym == XK_SHIFT_R)
        });
        Ok(Keyboard { shift, ..keyboard })
    }

    /// The keysym columns of one keycode.
    fn keysyms(&self, keycode: Keycode) -> &[Keysym] {
        let start = (keycode - self.min_keycode) as usize * self.width;
        &self.syms[start..start + self.width]
    }

    /// Find a keysym in the trusted base/Shift columns:
    /// `(keycode, needs_shift)`. Deeper columns are deliberately not
    /// searched (module docs: their meaning is ambiguous without XKB).
    fn find(&self, keysym: Keysym) -> Option<(Keycode, bool)> {
        for (keycode, syms) in self.iter_keycodes() {
            if syms.first() == Some(&keysym) {
                return Some((keycode, false));
            }
            if syms.get(1) == Some(&keysym) {
                return Some((keycode, true));
            }
        }
        None
    }

    fn iter_keycodes(&self) -> impl Iterator<Item = (Keycode, &[Keysym])> {
        let count = self.syms.len() / self.width;
        (0..count).map(move |index| {
            let keycode = self.min_keycode + index as Keycode;
            (keycode, &self.syms[index * self.width..(index + 1) * self.width])
        })
    }

    /// The highest keycode with all-`NoSymbol` keysyms — "spare" for
    /// a temporary remap. From the top down, so the choice is
    /// deterministic (the integration test relies on that to predict
    /// which keycode carries a remapped character) and stays clear of
    /// the low keycodes real hardware and the test's own pre-mapped
    /// characters live on.
    fn spare_keycode(&self) -> Option<Keycode> {
        let count = self.syms.len() / self.width;
        (0..count).rev().find_map(|index| {
            let keycode = self.min_keycode + index as Keycode;
            let syms = self.keysyms(keycode);
            (syms.iter().all(|&sym| sym == x11rb::NO_SYMBOL)).then_some(keycode)
        })
    }
}

/// Restore a temporarily remapped keycode on scope exit; errors are
/// swallowed (a Drop cannot report, the keycode was spare anyway, and
/// the next remap of it rewrites it again).
struct MappingGuard<'a> {
    session: &'a Session,
    keycode: Keycode,
    width: u8,
    original: Vec<Keysym>,
}

impl Drop for MappingGuard<'_> {
    fn drop(&mut self) {
        if let Ok(cookie) = self.session.conn.change_keyboard_mapping(
            1,
            self.keycode,
            self.width,
            &self.original,
        ) {
            // check() is the round trip that proves the restore landed
            // before anyone else types.
            let _ = cookie.check();
        }
    }
}

/// Type one character: find (or briefly borrow) a keycode for it,
/// press Shift if the column needs it, tap the key, release in
/// reverse.
fn type_character(
    session: &Session,
    keyboard: &Keyboard,
    character: char,
) -> Result<(), InsertError> {
    let keysym = keysym_for_char(character);
    let (keycode, needs_shift, guard) = match keyboard.find(keysym) {
        // A found base/shift match is only usable when the Shift it
        // needs actually exists on this keyboard.
        Some((keycode, needs_shift)) if !needs_shift || keyboard.shift.is_some() => {
            (keycode, needs_shift, None)
        }
        _ => {
            // Not in the trusted columns (or Shift-less): borrow a
            // spare keycode at column 0 instead of giving up on the
            // character.
            let Some(spare) = keyboard.spare_keycode() else {
                return Err(InsertError::Rejected {
                    reason: format!(
                        "no keycode produces U+{:04X} and no spare keycode is free to remap",
                        character as u32
                    ),
                });
            };
            let mut remapped = vec![x11rb::NO_SYMBOL; keyboard.width];
            remapped[0] = keysym;
            session
                .conn
                .change_keyboard_mapping(1, spare, keyboard.width as u8, &remapped)
                .map_err(x11_conn_error)?
                .check()
                .map_err(reply_error)?;
            // Round trip: the mapping must be live before the keypress
            // is interpreted by anyone.
            session.sync()?;
            let guard = MappingGuard {
                session,
                keycode: spare,
                width: keyboard.width as u8,
                original: keyboard.keysyms(spare).to_vec(),
            };
            (spare, false, Some(guard))
        }
    };

    // A real Shift press (a key event the target sees, exactly like a
    // human typing `C`) rather than a synthetic modifier mask — XTest
    // only speaks keycodes.
    if needs_shift {
        let shift = keyboard
            .shift
            .expect("checked above: needs_shift implies a Shift keycode");
        fake_key(session, shift, KEY_PRESS)?;
    }
    fake_key(session, keycode, KEY_PRESS)?;
    std::thread::sleep(KEY_HOLD);
    fake_key(session, keycode, KEY_RELEASE)?;
    if needs_shift {
        let shift = keyboard
            .shift
            .expect("checked above: needs_shift implies a Shift keycode");
        fake_key(session, shift, KEY_RELEASE)?;
    }
    std::thread::sleep(KEY_GAP);
    if let Some(guard) = guard {
        // The character's key is up; prove the server saw the release
        // before restoring the mapping it was typed under.
        session.sync()?;
        drop(guard);
    }
    Ok(())
}

/// The standard keysym for a character: Latin-1 for the first 256
/// code points (the keysym space *is* Latin-1 there), the Unicode
/// convention `0x0100_0000 | codepoint` above.
fn keysym_for_char(character: char) -> Keysym {
    let codepoint = character as u32;
    if codepoint < 0x100 {
        codepoint
    } else {
        0x0100_0000 | codepoint
    }
}

/// One XTest fake key event; deviceid 0 addresses the core keyboard,
/// which is what every target's normal key handling consumes.
fn fake_key(session: &Session, keycode: Keycode, event_type: u8) -> Result<(), InsertError> {
    use xtest::ConnectionExt as _;
    session
        .conn
        .xtest_fake_input(event_type, keycode, 0, x11rb::NONE, 0, 0, 0)
        .map_err(x11_conn_error)?;
    session.conn.flush().map_err(x11_conn_error)?;
    Ok(())
}

fn unavailable(reason: &str) -> InsertError {
    InsertError::Unavailable {
        reason: reason.to_string(),
        setup_hint: None,
    }
}

fn connect_error(error: &ConnectError) -> String {
    match error {
        ConnectError::DisplayParsingError(inner) => format!("unparsable DISPLAY ({inner})"),
        other => other.to_string(),
    }
}

/// A dropped/broken connection makes the backend *unavailable*, not
/// "rejected": the target never got a chance to refuse anything.
fn x11_conn_error(error: x11rb::errors::ConnectionError) -> InsertError {
    unavailable(&format!("the X connection failed: {error}"))
}

fn reply_error(error: x11rb::errors::ReplyError) -> InsertError {
    match error {
        x11rb::errors::ReplyError::ConnectionError(inner) => x11_conn_error(inner),
        x11rb::errors::ReplyError::X11Error(inner) => InsertError::Rejected {
            reason: format!("the X server refused a request: {inner:?}"),
        },
    }
}
