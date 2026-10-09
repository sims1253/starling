//! The X11 backend: identity from EWMH and the core input focus, typing
//! through XTest.
//!
//! # Identity
//!
//! A ref is `x11:<_NET_ACTIVE_WINDOW>:<input focus>[:<_NET_WM_PID>]`.
//! Revalidation requires both windows to match, the active window to
//! exist, and the pid not to differ (a differing pid means a recycled
//! window id). Without an EWMH window manager (a bare Xvfb) the focus
//! window doubles as the active one. With `WAYLAND_DISPLAY` set the
//! backend is unavailable: native Wayland windows are invisible to X11,
//! so the X focus does not describe the desktop.
//!
//! # Threat model
//!
//! X11 has no isolation between clients: a hostile client can read keys,
//! inject its own and remap the keyboard, so it needs no race against
//! Starling to type Enter. This backend defends against benign
//! interference: a layout switcher, `setxkbmap` or an IME helper changing
//! the keymap, focus or modifiers while a transcript types. Its
//! `GrabServer` sections are atomic only on servers that honor the grab
//! (Xorg and Xvfb do; WSLg's XWayland does not, so there a remap landing
//! inside one keystroke cannot be prevented client-side).
//!
//! # Typing
//!
//! XTest sends keycodes; the target decodes them through its copy of the
//! keyboard mapping. A character uses a key the layout already has when
//! its keysym is in column 0, or in column 1 and a keycode exists that is
//! bound to the Shift modifier and carries Shift in its unshifted column.
//! Deeper columns depend on XKB key types and are not trusted. Every
//! other character borrows a spare (all-`NoSymbol`) keycode, remapped to
//! the keysym in every column so XKB has no lower/upper case pair to
//! expand, and restored afterwards.
//!
//! Before every chunk of [`X11_CHUNK_CHARS`] characters the backend waits
//! out held modifiers, refuses keyboard states it cannot reproduce (a
//! non-first group, any lock or latch except the bit the live modifier
//! mapping binds to `Num_Lock`), and revalidates the target. Each
//! character's keystroke then runs inside one server grab that first
//! re-verifies all of it: the XKB state must still equal the chunk-start
//! snapshot (no newly held modifier, lock, latch or group), the target
//! and its ownership must be unchanged, and the keys about to be pressed
//! must still carry the planned keysyms. A change between that check and
//! the key-down, or between key-down and key-up (targets act on releases
//! too), would otherwise make Starling's own key produce something else,
//! e.g. another client's Return, or text in a newly focused window.
//!
//! The keyboard mapping is server-global, so borrowing is transactional:
//!
//! - `INSERT_LOCK` serializes this process's inserts from loading the
//!   mapping to restoring it.
//! - A spare is re-read, written and its echo read back under a server
//!   grab; one another client took in the meantime is skipped.
//! - A keycode is restored only while it still holds exactly the echo;
//!   another client's mapping is left alone and logged. A failed restore
//!   is reported as [`InsertError::KeyboardRestoreFailed`], together with
//!   what happened to the text.
//!
//! Each operation opens its own connection: capture and insert can be
//! minutes apart, and reconnecting is simpler than detecting a dead
//! cached connection. Within one insert, checks and keys share one
//! connection so they see the same server state.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{Mutex, PoisonError};
use std::time::Duration;

use x11rb::connection::Connection;
use x11rb::errors::{ConnectionError, ReplyError};
use x11rb::protocol::xkb;
use x11rb::protocol::xproto::{Atom, ConnectionExt, GetPropertyReply, Keycode, Keysym, Window};
use x11rb::protocol::xtest;
use x11rb::protocol::ErrorKind;
use x11rb::rust_connection::RustConnection;
use x11rb::NO_SYMBOL;

use crate::{
    deliver_in_chunks, format_ref, insertion_guards, merge_excluded_pids, parse_ref,
    wait_modifiers_released, weighed_segments, BackendKind, ChunkFailure, InsertError,
    InsertReceipt, InsertionBackend, TargetCheck, TargetSnapshot,
};

const KEY_PRESS: u8 = 2;
const KEY_RELEASE: u8 = 3;
/// `GetInputFocus` values that are not a window.
const FOCUS_NONE: Window = 0;
const FOCUS_POINTER_ROOT: Window = 1;
const XK_SHIFT_L: Keysym = 0xffe1;
const XK_SHIFT_R: Keysym = 0xffe2;
const XK_NUM_LOCK: Keysym = 0xff7f;
/// `xkbUseCoreKeyboard`.
const USE_CORE_KEYBOARD: u16 = 0x100;
const MODIFIER_NAMES: [&str; 8] = [
    "Shift",
    "Lock",
    "Control",
    "Mod1 (usually Alt)",
    "Mod2 (usually NumLock)",
    "Mod3",
    "Mod4 (usually Super)",
    "Mod5 (usually AltGr)",
];

/// How long a synthetic key stays down: toolkits can drop a zero-length
/// press, and auto-repeat needs far longer. The only sleep taken while
/// the server is grabbed.
const KEY_HOLD: Duration = Duration::from_millis(3);
/// Gap between characters, so targets and their IME layers keep up.
const KEY_GAP: Duration = Duration::from_millis(1);

/// Characters per chunk; the full safety check runs before each one.
pub const X11_CHUNK_CHARS: usize = 16;

/// How long borrowed keycodes stay mapped after the last key was synced,
/// so a target that processes the remap's `MappingNotify` late still
/// decodes the typed keys. A heuristic: a slower target can still lose
/// them.
const KEYMAP_SETTLE: Duration = Duration::from_millis(50);

/// Serializes this process's inserts, from loading the keyboard mapping
/// until every borrowed keycode is restored. A mapping read while a
/// sibling insert holds a borrow would plan the sibling's keycodes as
/// pre-mapped, and the sibling's restore would unmap them before they
/// are typed.
static INSERT_LOCK: Mutex<()> = Mutex::new(());

#[derive(Debug)]
pub struct X11Backend {
    excluded_pids: Vec<u32>,
    key_hold: Duration,
}

impl Default for X11Backend {
    fn default() -> Self {
        X11Backend::new()
    }
}

impl X11Backend {
    pub fn new() -> X11Backend {
        X11Backend::with_excluded_pids(Vec::new())
    }

    /// Refuse targets owned by any of `excluded_pids` or this process.
    pub fn with_excluded_pids(excluded_pids: Vec<u32>) -> X11Backend {
        X11Backend {
            excluded_pids: merge_excluded_pids(excluded_pids),
            key_hold: KEY_HOLD,
        }
    }

    /// Hold every key this long, so a test can land interference inside
    /// one keystroke deterministically.
    #[cfg(any(test, feature = "test-doubles"))]
    pub fn with_key_hold_for_tests(mut self, key_hold: Duration) -> X11Backend {
        self.key_hold = key_hold;
        self
    }

    /// Wait out held modifiers, check the keyboard state, then the target:
    /// last, because the modifier wait can take seconds. Returns the
    /// keyboard state every keystroke of the chunk is verified against.
    fn chunk_check(
        &self,
        session: &Session,
        target_ref: &str,
    ) -> Result<KeyboardState, InsertError> {
        wait_modifiers_released(|| {
            let held = KeyboardState::read(session)?.base_mods;
            Ok((held != 0).then(|| modifier_names(held)))
        })?;
        let state = KeyboardState::read(session)?;
        state.check_reproducible(session)?;
        self.check_target(session, target_ref)?;
        Ok(state)
    }

    /// The live target is still the captured one and not Starling's.
    fn check_target(&self, session: &Session, target_ref: &str) -> Result<(), InsertError> {
        match live_target(session, target_ref)? {
            LiveTarget::Same { live_pid, active } => {
                let class = match live_pid {
                    Some(_) => None,
                    None => session.wm_class(active)?,
                };
                if self.owns_target(live_pid, class.as_ref()) {
                    return Err(InsertError::TargetIsStarling);
                }
                Ok(())
            }
            LiveTarget::Changed { actual } => Err(InsertError::TargetChanged {
                expected: target_ref.to_string(),
                actual,
            }),
            LiveTarget::Gone => Err(InsertError::TargetGone),
        }
    }

    /// Whether a window is Starling's: an excluded pid or, only when the
    /// window has no pid, a Starling `WM_CLASS`. The class is
    /// client-chosen, so a lookalike app can be refused; the copy fallback
    /// covers that, while not checking would let Starling type into its
    /// own editor.
    fn owns_target(&self, pid: Option<u32>, class: Option<&(String, String)>) -> bool {
        match pid {
            Some(pid) => self.excluded_pids.contains(&pid),
            None => class.is_some_and(|(instance, class)| {
                ["starling-gpui", "starling"].iter().any(|name| {
                    instance.eq_ignore_ascii_case(name) || class.eq_ignore_ascii_case(name)
                })
            }),
        }
    }
}

impl InsertionBackend for X11Backend {
    fn kind(&self) -> BackendKind {
        BackendKind::X11
    }

    fn availability(&self) -> Result<(), InsertError> {
        Session::open().map(drop)
    }

    fn capture(&self) -> Result<TargetSnapshot, InsertError> {
        let session = Session::open()?;
        let Some((active, focus)) = session.focus_pair()? else {
            return Err(InsertError::Rejected {
                reason: "no window has input focus".to_string(),
            });
        };
        let pid = session.wm_pid(active)?;
        let class = session.wm_class(active)?;
        if self.owns_target(pid, class.as_ref()) {
            return Err(InsertError::TargetIsStarling);
        }
        Ok(TargetSnapshot {
            backend: BackendKind::X11,
            target_ref: format_ref(BackendKind::X11, active, focus, pid),
            app: class.map(|(_, class)| class),
            title: session.window_title(active)?,
            pid,
        })
    }

    fn revalidate(&self, target: &TargetSnapshot) -> Result<TargetCheck, InsertError> {
        Ok(match live_target(&Session::open()?, &target.target_ref)? {
            LiveTarget::Same { .. } => TargetCheck::Same,
            LiveTarget::Changed { actual } => TargetCheck::Changed {
                expected: target.target_ref.clone(),
                actual,
            },
            LiveTarget::Gone => TargetCheck::Gone,
        })
    }

    fn insert(&self, target: &TargetSnapshot, text: &str) -> Result<InsertReceipt, InsertError> {
        insertion_guards(text, target.pid, &self.excluded_pids)?;
        let _insert_lock = INSERT_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
        let session = Session::open()?;

        // Plan everything before anything is borrowed or typed.
        let keyboard = Keyboard::load(&session)?;
        session.xkb_init()?;
        let mut plans = HashMap::new();
        let mut unmapped = Vec::new();
        for character in text.chars() {
            if plans.contains_key(&character) || unmapped.contains(&character) {
                continue;
            }
            match keyboard.find(keysym_for_char(character)) {
                Some(plan) => {
                    plans.insert(character, plan);
                }
                None => unmapped.push(character),
            }
        }
        let mut remap = RemapTransaction::begin(&session, &keyboard, &unmapped)?;
        for (character, borrowed) in unmapped.iter().zip(&remap.borrowed) {
            plans.insert(
                *character,
                CharPlan::Borrowed {
                    keycode: borrowed.keycode,
                    echoed: borrowed.echoed.clone(),
                },
            );
        }

        let segments = weighed_segments(text, X11_CHUNK_CHARS, |_| 1);
        let typed = deliver_in_chunks(
            text.chars().count(),
            &segments,
            || self.chunk_check(&session, &target.target_ref),
            |segment, state| {
                self.type_segment(&session, &plans, &target.target_ref, &state, segment)
            },
        );
        with_restore_outcome(typed, remap.finish())
    }
}

/// One connection plus the screen's root window.
struct Session {
    conn: RustConnection,
    root: Window,
    atoms: RefCell<HashMap<&'static str, Atom>>,
}

impl Session {
    fn open() -> Result<Session, InsertError> {
        let set = |name: &str| std::env::var_os(name).is_some_and(|value| !value.is_empty());
        if set("WAYLAND_DISPLAY") {
            return Err(unavailable(
                "Wayland session: native Wayland windows are invisible to X11",
            ));
        }
        if !set("DISPLAY") {
            return Err(unavailable("no X display (DISPLAY is not set)"));
        }
        let (conn, screen) = x11rb::connect(None)
            .map_err(|error| unavailable(&format!("cannot reach the X server: {error}")))?;
        let root = conn.setup().roots[screen].root;
        Ok(Session {
            conn,
            root,
            atoms: RefCell::default(),
        })
    }

    /// A round trip: every request queued so far has been processed.
    fn sync(&self) -> Result<(), InsertError> {
        self.input_focus().map(drop)
    }

    fn input_focus(&self) -> Result<Window, InsertError> {
        Ok(self
            .conn
            .get_input_focus()
            .map_err(x11_conn_error)?
            .reply()
            .map_err(reply_error)?
            .focus)
    }

    fn atom(&self, name: &'static str) -> Result<Atom, InsertError> {
        if let Some(&atom) = self.atoms.borrow().get(name) {
            return Ok(atom);
        }
        let atom = self
            .conn
            .intern_atom(false, name.as_bytes())
            .map_err(x11_conn_error)?
            .reply()
            .map_err(reply_error)?
            .atom;
        self.atoms.borrow_mut().insert(name, atom);
        Ok(atom)
    }

    fn property(
        &self,
        window: Window,
        name: &'static str,
    ) -> Result<Option<GetPropertyReply>, InsertError> {
        let reply = self
            .conn
            .get_property(false, window, self.atom(name)?, x11rb::NONE, 0, 64)
            .map_err(x11_conn_error)?
            .reply()
            .map_err(reply_error)?;
        Ok((reply.type_ != x11rb::NONE).then_some(reply))
    }

    fn card32(&self, window: Window, name: &'static str) -> Result<Option<u32>, InsertError> {
        Ok(self
            .property(window, name)?
            .filter(|reply| reply.format == 32)
            .and_then(|reply| reply.value32()?.next()))
    }

    fn window_exists(&self, window: Window) -> Result<bool, InsertError> {
        match self
            .conn
            .get_window_attributes(window)
            .map_err(x11_conn_error)?
            .reply()
        {
            Ok(_) => Ok(true),
            Err(ReplyError::X11Error(error)) if error.error_kind == ErrorKind::Window => Ok(false),
            Err(error) => Err(reply_error(error)),
        }
    }

    /// `(active, focus)`, or `None` when no window has the focus.
    fn focus_pair(&self) -> Result<Option<(Window, Window)>, InsertError> {
        let focus = self.input_focus()?;
        if focus == FOCUS_NONE || focus == FOCUS_POINTER_ROOT {
            return Ok(None);
        }
        let active = self
            .card32(self.root, "_NET_ACTIVE_WINDOW")?
            .filter(|&active| active != FOCUS_NONE)
            .unwrap_or(focus);
        Ok(Some((active, focus)))
    }

    fn wm_pid(&self, window: Window) -> Result<Option<u32>, InsertError> {
        self.card32(window, "_NET_WM_PID")
    }

    /// `WM_CLASS` as `(instance, class)`.
    fn wm_class(&self, window: Window) -> Result<Option<(String, String)>, InsertError> {
        let Some(reply) = self.property(window, "WM_CLASS")? else {
            return Ok(None);
        };
        let mut parts = reply
            .value
            .split(|&byte| byte == 0)
            .filter(|part| !part.is_empty())
            .map(|part| String::from_utf8_lossy(part).into_owned());
        Ok(match (parts.next(), parts.next()) {
            (Some(instance), Some(class)) => Some((instance, class)),
            (Some(class), None) => Some((String::new(), class)),
            _ => None,
        })
    }

    /// `_NET_WM_NAME` (UTF-8), else `WM_NAME` (Latin-1).
    fn window_title(&self, window: Window) -> Result<Option<String>, InsertError> {
        if let Some(reply) = self.property(window, "_NET_WM_NAME")? {
            return Ok(Some(String::from_utf8_lossy(&reply.value).into_owned()));
        }
        Ok(self
            .property(window, "WM_NAME")?
            .map(|reply| reply.value.iter().map(|&byte| char::from(byte)).collect()))
    }

    fn keycode_syms(&self, keycode: Keycode) -> Result<Vec<Keysym>, InsertError> {
        Ok(self
            .conn
            .get_keyboard_mapping(keycode, 1)
            .map_err(x11_conn_error)?
            .reply()
            .map_err(reply_error)?
            .keysyms)
    }

    /// The keycodes bound to each of the eight modifiers (Shift, Lock,
    /// Control, Mod1..Mod5).
    fn modifier_rows(&self) -> Result<Vec<Vec<Keycode>>, InsertError> {
        let reply = self
            .conn
            .get_modifier_mapping()
            .map_err(x11_conn_error)?
            .reply()
            .map_err(reply_error)?;
        let per_row = usize::from(reply.keycodes_per_modifier());
        if per_row == 0 || reply.keycodes.len() != per_row * 8 {
            return Err(InsertError::Rejected {
                reason: "the X server reported an unusable modifier mapping".to_string(),
            });
        }
        Ok(reply.keycodes.chunks(per_row).map(<[_]>::to_vec).collect())
    }

    /// The modifier bit the live mapping binds to `Num_Lock`, if any.
    /// Layouts bind it to any of Mod1..Mod5, so it is read, never assumed.
    fn num_lock_bit(&self) -> Result<Option<u16>, InsertError> {
        let setup = self.conn.setup();
        let keycodes = setup.min_keycode..=setup.max_keycode;
        for (bit, row) in self.modifier_rows()?.iter().enumerate() {
            for keycode in row
                .iter()
                .copied()
                .filter(|keycode| keycodes.contains(keycode))
            {
                if self
                    .keycode_syms(keycode)?
                    .iter()
                    .take(2)
                    .any(|&sym| sym == XK_NUM_LOCK)
                {
                    return Ok(Some(1 << bit));
                }
            }
        }
        Ok(None)
    }

    /// The XKB handshake; without XKB the keyboard state is unreadable and
    /// typing would be guessing.
    fn xkb_init(&self) -> Result<(), InsertError> {
        use xkb::ConnectionExt as _;
        let reply = self
            .conn
            .xkb_use_extension(1, 0)
            .map_err(x11_conn_error)?
            .reply()
            .map_err(reply_error)?;
        if !reply.supported {
            return Err(InsertError::KeyboardStateUnsupported {
                reason: "the X server does not support the XKB extension".to_string(),
            });
        }
        Ok(())
    }

    fn xkb_state(&self) -> Result<xkb::GetStateReply, InsertError> {
        use xkb::ConnectionExt as _;
        self.conn
            .xkb_get_state(USE_CORE_KEYBOARD)
            .map_err(x11_conn_error)?
            .reply()
            .map_err(reply_error)
    }
}

enum LiveTarget {
    Same {
        live_pid: Option<u32>,
        active: Window,
    },
    Changed {
        actual: String,
    },
    Gone,
}

fn live_target(session: &Session, target_ref: &str) -> Result<LiveTarget, InsertError> {
    let Some((_, active, focus, captured_pid)) = parse_ref(target_ref) else {
        return Err(InsertError::Rejected {
            reason: format!("malformed target ref: {target_ref}"),
        });
    };
    // A closed window is `Gone` even if focus moved too: that is the fact
    // the user can act on.
    if !session.window_exists(active)? {
        return Ok(LiveTarget::Gone);
    }
    let Some((live_active, live_focus)) = session.focus_pair()? else {
        return Ok(LiveTarget::Changed {
            actual: "x11:none".to_string(),
        });
    };
    if (live_active, live_focus) == (active, focus) {
        let live_pid = session.wm_pid(active)?;
        // A pid that differs from, or appeared since, capture means the
        // window id was recycled.
        if live_pid.is_none() || live_pid == captured_pid {
            return Ok(LiveTarget::Same { live_pid, active });
        }
        return Ok(LiveTarget::Changed {
            actual: format_ref(BackendKind::X11, active, focus, live_pid),
        });
    }
    // Only describes the new focus; it may vanish while being read.
    let live_pid = session.wm_pid(live_active).unwrap_or(None);
    Ok(LiveTarget::Changed {
        actual: format_ref(BackendKind::X11, live_active, live_focus, live_pid),
    })
}

fn modifier_names(mask: u16) -> Vec<String> {
    MODIFIER_NAMES
        .iter()
        .enumerate()
        .filter(|(bit, _)| mask & (1 << bit) != 0)
        .map(|(_, name)| name.to_string())
        .collect()
}

/// The XKB state of the core keyboard that decides what a key types.
#[derive(Debug, PartialEq, Eq)]
struct KeyboardState {
    /// Physically held modifiers.
    base_mods: u16,
    latched_mods: u16,
    locked_mods: u16,
    /// Effective, base, latched and locked group.
    groups: [i32; 4],
}

impl KeyboardState {
    fn read(session: &Session) -> Result<KeyboardState, InsertError> {
        let state = session.xkb_state()?;
        Ok(KeyboardState {
            base_mods: state.base_mods.into(),
            latched_mods: state.latched_mods.into(),
            locked_mods: state.locked_mods.into(),
            groups: [
                u8::from(state.group).into(),
                state.base_group.into(),
                state.latched_group.into(),
                u8::from(state.locked_group).into(),
            ],
        })
    }

    /// Refuse states in which columns 0 and 1 are not what a key types: a
    /// non-first group, or any locked or latched modifier except Num_Lock.
    fn check_reproducible(&self, session: &Session) -> Result<(), InsertError> {
        if self.groups != [0; 4] {
            return Err(InsertError::KeyboardStateUnsupported {
                reason: format!(
                    "an alternate keyboard group is active (group {})",
                    self.groups[0]
                ),
            });
        }
        let harmless = session.num_lock_bit()?.unwrap_or(0);
        for (what, mods) in [("locked", self.locked_mods), ("latched", self.latched_mods)] {
            let mods = mods & !harmless;
            if mods != 0 {
                return Err(InsertError::KeyboardStateUnsupported {
                    reason: format!("a modifier other than Num_Lock is {what} (mask {mods:#x})"),
                });
            }
        }
        Ok(())
    }

    /// Why `self` (read now) differs from the chunk-start `expected`.
    fn changed_from(&self, expected: &KeyboardState) -> Option<InsertError> {
        if self == expected {
            None
        } else if self.base_mods & !expected.base_mods != 0 {
            Some(InsertError::ModifiersHeld {
                held: modifier_names(self.base_mods & !expected.base_mods),
            })
        } else {
            Some(InsertError::KeyboardStateUnsupported {
                reason: "a modifier, lock or keyboard group changed while typing".to_string(),
            })
        }
    }
}

/// The keyboard mapping one insert plans with.
struct Keyboard {
    min_keycode: Keycode,
    max_keycode: Keycode,
    width: usize,
    syms: Vec<Keysym>,
    /// A keycode bound to the Shift modifier with Shift in its unshifted
    /// column, i.e. one that really sets Shift when pressed alone.
    shift: Option<Keycode>,
    /// All-`NoSymbol` keycodes, highest first (away from the low keycodes
    /// real hardware uses).
    spares: Vec<Keycode>,
}

impl Keyboard {
    fn load(session: &Session) -> Result<Keyboard, InsertError> {
        let setup = session.conn.setup();
        let (min_keycode, max_keycode) = (setup.min_keycode, setup.max_keycode);
        let reply = session
            .conn
            .get_keyboard_mapping(min_keycode, max_keycode - min_keycode + 1)
            .map_err(x11_conn_error)?
            .reply()
            .map_err(reply_error)?;
        let width = usize::from(reply.keysyms_per_keycode);
        if width == 0 || reply.keysyms.len() != usize::from(max_keycode - min_keycode + 1) * width {
            return Err(InsertError::Rejected {
                reason: "the X server reported an unusable keyboard mapping".to_string(),
            });
        }
        let mut keyboard = Keyboard {
            min_keycode,
            max_keycode,
            width,
            syms: reply.keysyms,
            shift: None,
            spares: Vec::new(),
        };
        keyboard.shift = session.modifier_rows()?[0]
            .iter()
            .copied()
            .filter(|keycode| (min_keycode..=max_keycode).contains(keycode))
            .find(|&keycode| is_shift_keysym(keyboard.keysyms(keycode).first()));
        keyboard.spares = (min_keycode..=max_keycode)
            .rev()
            .filter(|&keycode| {
                keyboard
                    .keysyms(keycode)
                    .iter()
                    .all(|&sym| sym == NO_SYMBOL)
            })
            .collect();
        Ok(keyboard)
    }

    fn keysyms(&self, keycode: Keycode) -> &[Keysym] {
        let start = usize::from(keycode - self.min_keycode) * self.width;
        &self.syms[start..start + self.width]
    }

    /// A key the layout already has for `keysym`, in a trusted column.
    fn find(&self, keysym: Keysym) -> Option<CharPlan> {
        (self.min_keycode..=self.max_keycode).find_map(|keycode| {
            let syms = self.keysyms(keycode);
            if syms.first() == Some(&keysym) {
                Some(CharPlan::Mapped {
                    keycode,
                    shift: None,
                })
            } else if self.shift.is_some() && syms.get(1) == Some(&keysym) {
                Some(CharPlan::Mapped {
                    keycode,
                    shift: self.shift,
                })
            } else {
                None
            }
        })
    }
}

enum CharPlan {
    /// A key of the layout, pressed with `shift` held for column 1.
    Mapped {
        keycode: Keycode,
        shift: Option<Keycode>,
    },
    /// A borrowed spare that must still hold exactly `echoed`.
    Borrowed {
        keycode: Keycode,
        echoed: Vec<Keysym>,
    },
}

struct BorrowedKey {
    keycode: Keycode,
    /// What the server reports after the write. Servers canonicalize what
    /// they store, so this, not what was written, identifies the mapping
    /// as Starling's.
    echoed: Vec<Keysym>,
    original: Vec<Keysym>,
}

/// Spare keycodes borrowed for one insert, restored by [`Self::finish`]
/// or, if that never ran, on drop.
struct RemapTransaction<'a> {
    session: &'a Session,
    /// One entry per requested character, in order.
    borrowed: Vec<BorrowedKey>,
    finished: bool,
}

impl<'a> RemapTransaction<'a> {
    /// Borrow one spare keycode per character. On failure everything
    /// borrowed so far is restored before returning.
    fn begin(
        session: &'a Session,
        keyboard: &Keyboard,
        characters: &[char],
    ) -> Result<RemapTransaction<'a>, InsertError> {
        let mut transaction = RemapTransaction {
            session,
            borrowed: Vec::new(),
            finished: false,
        };
        let mut spares = keyboard.spares.iter().copied();
        for &character in characters {
            let borrowed = loop {
                let Some(keycode) = spares.next() else {
                    break Err(InsertError::Rejected {
                        reason: format!(
                            "every spare keycode is in use, so U+{:04X} cannot be typed \
                             (keyboard busy)",
                            u32::from(character)
                        ),
                    });
                };
                match transaction.try_borrow(keycode, character) {
                    Ok(false) => continue,
                    Ok(true) => break Ok(()),
                    Err(error) => break Err(error),
                }
            };
            if let Err(error) = borrowed {
                return with_restore_outcome(Err(error), transaction.finish());
            }
        }
        // The new mappings must be live before any key is pressed.
        if !characters.is_empty() {
            if let Err(error) = session.sync() {
                return with_restore_outcome(Err(error), transaction.finish());
            }
        }
        Ok(transaction)
    }

    /// Borrow `keycode` for `character` unless another client took it.
    fn try_borrow(&mut self, keycode: Keycode, character: char) -> Result<bool, InsertError> {
        let session = self.session;
        let keysym = keysym_for_char(character);
        let _grab = ServerGrab::new(session)?;
        let live = session.keycode_syms(keycode)?;
        if live.is_empty() || live.iter().any(|&sym| sym != NO_SYMBOL) {
            return Ok(false);
        }
        let wrote = vec![keysym; live.len()];
        // Recorded before the write, so cleanup covers every outcome.
        self.borrowed.push(BorrowedKey {
            keycode,
            echoed: wrote.clone(),
            original: live,
        });
        session
            .conn
            .change_keyboard_mapping(1, keycode, wrote.len() as u8, &wrote)
            .map_err(x11_conn_error)?
            .check()
            .map_err(reply_error)?;
        let echoed = session.keycode_syms(keycode)?;
        let usable = echoed.first() == Some(&keysym);
        self.borrowed.last_mut().expect("pushed above").echoed = echoed;
        if !usable {
            return Err(InsertError::Rejected {
                reason: format!(
                    "the X server did not keep U+{:04X} in the base column of keycode {keycode} \
                     (keyboard busy)",
                    u32::from(character)
                ),
            });
        }
        Ok(true)
    }

    /// Restore every borrowed keycode, attempting all even if one fails.
    /// Returns the failure detail for [`InsertError::KeyboardRestoreFailed`].
    fn finish(&mut self) -> Result<(), String> {
        if self.finished || self.borrowed.is_empty() {
            self.finished = true;
            return Ok(());
        }
        let mut failures = Vec::new();
        if let Err(error) = self.session.sync() {
            failures.push(format!("syncing the typed keys: {error}"));
        }
        std::thread::sleep(KEYMAP_SETTLE);
        for entry in &self.borrowed {
            if let Err(error) = self.restore(entry) {
                failures.push(format!("restoring keycode {}: {error}", entry.keycode));
            }
        }
        self.finished = true;
        if failures.is_empty() {
            Ok(())
        } else {
            Err(failures.join("; "))
        }
    }

    fn restore(&self, entry: &BorrowedKey) -> Result<(), InsertError> {
        let _grab = ServerGrab::new(self.session)?;
        let live = self.session.keycode_syms(entry.keycode)?;
        if !same_keysyms(&live, &entry.echoed) {
            log::warn!(
                "starling-insertion: keycode {} was remapped by another client while borrowed; \
                 leaving its mapping {live:?} in place",
                entry.keycode
            );
            return Ok(());
        }
        self.session
            .conn
            .change_keyboard_mapping(
                1,
                entry.keycode,
                entry.original.len() as u8,
                &entry.original,
            )
            .map_err(x11_conn_error)?
            .check()
            .map_err(reply_error)
    }
}

impl Drop for RemapTransaction<'_> {
    fn drop(&mut self) {
        let _ = self.finish();
    }
}

/// Fold the restore of borrowed keycodes into an insert's outcome: a
/// failed restore is the error to report, and its detail keeps what
/// happened to the text.
fn with_restore_outcome<T>(
    outcome: Result<T, InsertError>,
    restore: Result<(), String>,
) -> Result<T, InsertError> {
    let Err(detail) = restore else {
        return outcome;
    };
    let detail = match outcome {
        Ok(_) => detail,
        Err(error @ InsertError::PartialDelivery { .. }) => format!("{detail}; and {error}"),
        Err(error) => format!("{detail}; and nothing was typed ({error})"),
    };
    Err(InsertError::KeyboardRestoreFailed { detail })
}

/// Equal up to trailing `NoSymbol` columns, which the server adds to every
/// keycode when another client widens the keymap.
fn same_keysyms(a: &[Keysym], b: &[Keysym]) -> bool {
    fn significant(syms: &[Keysym]) -> &[Keysym] {
        let end = syms
            .iter()
            .rposition(|&sym| sym != NO_SYMBOL)
            .map_or(0, |last| last + 1);
        &syms[..end]
    }
    significant(a) == significant(b)
}

fn is_shift_keysym(keysym: Option<&Keysym>) -> bool {
    matches!(keysym, Some(&XK_SHIFT_L | &XK_SHIFT_R))
}

/// A server grab, released on drop. While a conforming server holds it,
/// no other client's requests are processed.
struct ServerGrab<'a>(&'a Session);

impl<'a> ServerGrab<'a> {
    fn new(session: &'a Session) -> Result<ServerGrab<'a>, InsertError> {
        let cookie = session.conn.grab_server().map_err(x11_conn_error)?;
        // Constructed first so a refused grab is still (harmlessly) released.
        let grab = ServerGrab(session);
        // A security policy may refuse the grab yet allow mapping requests;
        // never continue ungrabbed.
        cookie.check().map_err(|error| match error {
            ReplyError::ConnectionError(error) => x11_conn_error(error),
            ReplyError::X11Error(error) => InsertError::Rejected {
                reason: format!("the X server refused a server grab (keyboard busy): {error:?}"),
            },
        })?;
        Ok(grab)
    }
}

impl Drop for ServerGrab<'_> {
    fn drop(&mut self) {
        let released = self.0.conn.ungrab_server().map(drop);
        if let Err(error) = released.and_then(|()| self.0.conn.flush()) {
            log::warn!("starling-insertion: releasing the server grab failed: {error}");
        }
    }
}

/// Keys pressed by one keystroke, released on drop so a failure never
/// leaves a key (worst: Shift) stuck down.
struct PressedKeys<'a> {
    session: &'a Session,
    keys: Vec<Keycode>,
}

impl PressedKeys<'_> {
    fn press(&mut self, keycode: Keycode) -> Result<(), ReplyError> {
        // Recorded first: a failed send may still have reached the server.
        self.keys.push(keycode);
        fake_key(self.session, keycode, KEY_PRESS)
    }

    fn release(&mut self, keycode: Keycode) -> Result<(), ReplyError> {
        self.keys.retain(|&pressed| pressed != keycode);
        fake_key(self.session, keycode, KEY_RELEASE)
    }
}

impl Drop for PressedKeys<'_> {
    fn drop(&mut self) {
        while let Some(keycode) = self.keys.pop() {
            if let Err(error) = fake_key(self.session, keycode, KEY_RELEASE) {
                log::warn!("starling-insertion: releasing keycode {keycode} failed: {error:?}");
            }
        }
    }
}

/// Where a keystroke failed relative to the character's key-down, which
/// decides whether the character counts as possibly delivered.
enum CharFailure {
    BeforeKeydown(InsertError),
    AfterKeydown(InsertError),
}

impl X11Backend {
    fn type_segment(
        &self,
        session: &Session,
        plans: &HashMap<char, CharPlan>,
        target_ref: &str,
        state: &KeyboardState,
        segment: &str,
    ) -> Result<(), ChunkFailure> {
        for (typed, character) in segment.chars().enumerate() {
            self.type_character(session, &plans[&character], target_ref, state, character)
                .map_err(|failure| match failure {
                    CharFailure::BeforeKeydown(cause) => ChunkFailure {
                        delivered: typed,
                        cause,
                    },
                    CharFailure::AfterKeydown(cause) => ChunkFailure {
                        delivered: typed + 1,
                        cause,
                    },
                })?;
        }
        Ok(())
    }

    /// Verify and type one character inside one server grab spanning the
    /// whole keystroke (see the module docs).
    fn type_character(
        &self,
        session: &Session,
        plan: &CharPlan,
        target_ref: &str,
        state: &KeyboardState,
        character: char,
    ) -> Result<(), CharFailure> {
        use CharFailure::{AfterKeydown, BeforeKeydown};
        let (keycode, shift) = match plan {
            CharPlan::Mapped { keycode, shift } => (*keycode, *shift),
            CharPlan::Borrowed { keycode, .. } => (*keycode, None),
        };
        {
            let _grab = ServerGrab::new(session).map_err(BeforeKeydown)?;
            // Declared after the grab, so its drop releases keys while grabbed.
            let mut pressed = PressedKeys {
                session,
                keys: Vec::new(),
            };
            if let Some(changed) = KeyboardState::read(session)
                .map_err(BeforeKeydown)?
                .changed_from(state)
            {
                return Err(BeforeKeydown(changed));
            }
            self.check_target(session, target_ref)
                .map_err(BeforeKeydown)?;
            verify_mapping(session, character, plan).map_err(BeforeKeydown)?;
            if let Some(shift) = shift {
                pressed
                    .press(shift)
                    .map_err(|e| BeforeKeydown(reply_error(e)))?;
            }
            // A refused key-down did not happen; a transport failure may
            // have delivered it.
            pressed.press(keycode).map_err(|error| match error {
                ReplyError::X11Error(_) => BeforeKeydown(reply_error(error)),
                ReplyError::ConnectionError(_) => AfterKeydown(reply_error(error)),
            })?;
            std::thread::sleep(self.key_hold);
            pressed
                .release(keycode)
                .map_err(|e| AfterKeydown(reply_error(e)))?;
            if let Some(shift) = shift {
                pressed
                    .release(shift)
                    .map_err(|e| AfterKeydown(reply_error(e)))?;
            }
        }
        std::thread::sleep(KEY_GAP);
        Ok(())
    }
}

/// The keys about to be pressed still carry what the plan expects.
fn verify_mapping(session: &Session, character: char, plan: &CharPlan) -> Result<(), InsertError> {
    match plan {
        CharPlan::Borrowed { keycode, echoed } => {
            let live = session.keycode_syms(*keycode)?;
            if !same_keysyms(&live, echoed) {
                return Err(mapping_drifted(*keycode, character, &live));
            }
        }
        CharPlan::Mapped { keycode, shift } => {
            let live = session.keycode_syms(*keycode)?;
            if live.get(usize::from(shift.is_some())) != Some(&keysym_for_char(character)) {
                return Err(mapping_drifted(*keycode, character, &live));
            }
            if let Some(shift) = *shift {
                // Shift is pressed with nothing else held, so it decodes
                // through column 0: `[Return, Shift_L]` would press Return.
                let live = session.keycode_syms(shift)?;
                if !is_shift_keysym(live.first()) || !session.modifier_rows()?[0].contains(&shift) {
                    return Err(mapping_drifted(shift, character, &live));
                }
            }
        }
    }
    Ok(())
}

fn mapping_drifted(keycode: Keycode, character: char, live: &[Keysym]) -> InsertError {
    InsertError::Rejected {
        reason: format!(
            "keycode {keycode} no longer carries the mapping planned for U+{:04X} (it now holds \
             {live:?}); another X client changed the keyboard (keyboard busy)",
            u32::from(character)
        ),
    }
}

/// Latin-1 keysyms are the code point; everything else uses the Unicode
/// convention `0x0100_0000 | code point`. Control characters never get
/// here (the guards refuse them), so this is never a function keysym.
fn keysym_for_char(character: char) -> Keysym {
    let codepoint = u32::from(character);
    if codepoint < 0x100 {
        codepoint
    } else {
        0x0100_0000 | codepoint
    }
}

/// One XTest key event on the core keyboard, checked: a refusal is an
/// error, not an unread event.
fn fake_key(session: &Session, keycode: Keycode, event_type: u8) -> Result<(), ReplyError> {
    use xtest::ConnectionExt as _;
    session
        .conn
        .xtest_fake_input(event_type, keycode, 0, x11rb::NONE, 0, 0, 0)?
        .check()
}

fn unavailable(reason: &str) -> InsertError {
    InsertError::Unavailable {
        reason: reason.to_string(),
    }
}

fn x11_conn_error(error: ConnectionError) -> InsertError {
    unavailable(&format!("the X connection failed: {error}"))
}

fn reply_error(error: ReplyError) -> InsertError {
    match error {
        ReplyError::ConnectionError(error) => x11_conn_error(error),
        ReplyError::X11Error(error) => InsertError::Rejected {
            reason: format!("the X server refused a request: {error:?}"),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_failed_restore_is_reported_with_what_happened_to_the_text() {
        let restore = || Err("restoring keycode 255: connection lost".to_string());
        let detail =
            |outcome: Result<(), InsertError>| match with_restore_outcome(outcome, restore()) {
                Err(InsertError::KeyboardRestoreFailed { detail }) => detail,
                other => panic!("a failed restore must be reported: {other:?}"),
            };

        assert_eq!(detail(Ok(())), "restoring keycode 255: connection lost");
        let partial = detail(Err(InsertError::PartialDelivery {
            delivered_chars: 16,
            total_chars: 20,
            cause: Box::new(InsertError::TargetGone),
        }));
        assert!(
            partial.contains("connection lost") && partial.contains("up to 16 of 20"),
            "{partial}"
        );
        let nothing = detail(Err(InsertError::TargetGone));
        assert!(nothing.contains("nothing was typed"), "{nothing}");

        // A clean restore leaves the outcome alone.
        assert_eq!(
            with_restore_outcome::<()>(Err(InsertError::TargetGone), Ok(())),
            Err(InsertError::TargetGone)
        );
    }

    #[test]
    fn keysym_comparison_ignores_trailing_padding_only() {
        assert!(same_keysyms(&[7, 7, 0, 0], &[7, 7]));
        assert!(!same_keysyms(&[7, 0, 7], &[7]));
        assert!(!same_keysyms(&[7], &[8]));
    }

    #[test]
    fn the_wm_class_fallback_applies_only_without_a_pid() {
        let backend = X11Backend::with_excluded_pids(vec![4213]);
        let starling = ("starling-gpui".to_string(), "Starling".to_string());
        assert!(backend.owns_target(Some(4213), None));
        assert!(!backend.owns_target(Some(1), Some(&starling)));
        assert!(backend.owns_target(None, Some(&starling)));
        assert!(!backend.owns_target(None, None));
    }
}
