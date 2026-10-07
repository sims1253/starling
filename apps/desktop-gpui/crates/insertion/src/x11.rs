//! The X11 backend: focus identity from EWMH + core focus, typing via
//! XTest synthetic keys (issue #221, slice 2 phase A).
//!
//! # Identity
//!
//! Capture reads `_NET_ACTIVE_WINDOW` (the window the WM says is
//! active) and `GetInputFocus` (the window keys actually go to — for a
//! real editor that is an input child of the active toplevel). The ref
//! is `x11:<active-hex>:<focus-hex>:<pid>`; revalidate compares both
//! ids, checks the window still exists, and compares pids when both
//! sides report one (a same window id with a different pid is a
//! recycled id, not the same target). Because focus and activity move
//! independently (a dialog can take focus while the active window
//! stays, and an app can close while focus lingers on a doomed child),
//! any mismatch is a change.
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
//! keyboard mapping. Before anything is typed the backend reads the
//! live keyboard state through XKB (`GetState`) — that one reply
//! answers every "is typing safe right now" question:
//!
//! - **Physically held modifiers** (`base_mods`): characters typed
//!   while the user still holds Ctrl/Alt/Shift/Super — typically the
//!   dictation shortcut's own keys — become commands. The backend
//!   waits up to [`MODIFIER_RELEASE_WAIT`] for release and then
//!   refuses with [`InsertError::ModifiersHeld`]; it never synthesizes
//!   a modifier release to fake readiness. After any wait the identity
//!   check runs again before a key moves.
//! - **Effective/locked group and Caps/Shift Lock** (`group`,
//!   `locked_mods`, `latched_mods`): if a group other than the first
//!   is active, or Caps/Shift Lock (or any locked modifier that
//!   changes what a bare keypress means — Alt, Super, AltGr) is
//!   engaged, the base/Shift columns do not identify the effective
//!   character, and the backend refuses with
//!   [`InsertError::KeyboardStateUnsupported`] instead of guessing.
//!
//! Pre-mapped keycodes are used only under the state those checks
//! establish: a keysym found in columns 0/1 of group 0 where Shift is
//! the only modifier needed, *and* the Shift modifier is actually
//! bound (via `GetModifierMapping`) to a keycode that carries a Shift
//! keysym. Everything else — a `ß` on a US layout, `→` anywhere, `é`
//! unless the layout has it on a base key — borrows the same
//! always-correct mechanism instead: spare keycodes (the highest whose
//! keysyms are all `NoSymbol`) are temporarily remapped
//! (`ChangeKeyboardMapping`) to the character's keysym, typed at
//! column 0, and restored afterwards.
//!
//! The remap is server-global and racing other clients would corrupt
//! *their* typing, so it is transactional:
//!
//! - a process-wide mutex serializes remap transactions of this
//!   process (concurrent inserts never share a spare);
//! - right before remapping, the keycode's *live* mapping is re-read
//!   and required to still be all-`NoSymbol` (another client's remap
//!   picks the next spare; none free is a `keyboard_busy` refusal);
//! - on restore, the live mapping is re-read and restored only if it
//!   still equals exactly what Starling wrote — another client's
//!   change is left alone (with a warning logged);
//! - the restore data is recorded *before* the mutating request, every
//!   synthetic press (Shift and character keys) is release-guarded,
//!   and after the last chunk the keys are flushed and synced and the
//!   borrowed mapping is held through a short settle delay
//!   ([`KEYMAP_SETTLE`]) before being restored, so a target that
//!   processes its `MappingNotify` late does not decode the already
//!   typed keys against the restored all-`NoSymbol` mapping. A failed
//!   restore on the success path surfaces as
//!   [`InsertError::KeyboardRestoreFailed`] rather than a fake
//!   success.
//!
//! Long texts type in chunks of at most [`X11_CHUNK_CHARS`]
//! characters, with the full safety check (held modifiers, keyboard
//! state, focus/identity/pid) repeated before *every* chunk; a change
//! part-way stops typing immediately and reports
//! [`InsertError::PartialDelivery`] with how much may have landed.
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

use std::cell::Cell;
use std::collections::HashMap;
use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};

use x11rb::connection::Connection;
use x11rb::errors::ConnectError;
use x11rb::protocol::xkb;
use x11rb::protocol::xproto::{Atom, ConnectionExt, GetPropertyReply, Keycode, Keysym, Window};
use x11rb::protocol::xtest;
use x11rb::protocol::ErrorKind;
use x11rb::rust_connection::RustConnection;

use crate::{
    cheap_insertion_guards, deliver_in_chunks, format_ref, merge_excluded_pids, parse_ref,
    weighed_segments, Availability, BackendKind, ChunkFailure, InsertError, InsertReceipt,
    InsertionBackend, SurroundingText, TargetCheck, TargetSnapshot, MODIFIER_POLL_INTERVAL,
    MODIFIER_RELEASE_WAIT,
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

/// `xkbUseCoreKeyboard` — the device `GetState` is asked about.
const USE_CORE_KEYBOARD: u16 = 0x100;

/// Locked modifiers that change what a bare keypress means, so typing
/// under them would not type what was asked: Shift and Lock (Caps Lock
/// and Shift Lock), Control, Mod1 (usually Alt), Mod4 (usually Super)
/// and Mod5 (usually AltGr). NumLock (Mod2) and Mod3 only reshape
/// keypad keys, which text characters never ride on, so a locked
/// NumLock stays allowed. Bit positions are the core protocol's
/// modifier mask bits (x11rb's `ModMask` constants in `u16` form).
const HAZARDOUS_LOCKED_MODS: u16 = 0b0000_0001 /* Shift */
        | 0b0000_0010 /* Lock: Caps Lock / Shift Lock */
        | 0b0000_0100 /* Control */
        | 0b0000_1000 /* Mod1: usually Alt */
        | 0b0100_0000 /* Mod4: usually Super */
        | 0b1000_0000; /* Mod5: usually AltGr */

/// How long a synthetic key stays "down". Real keys are down for tens
/// of milliseconds; a zero hold can be dropped or coalesced by
/// toolkits that watch press/release pairing, and the X server's own
/// auto-repeat only fires for keys held far longer than this.
const KEY_HOLD: Duration = Duration::from_millis(3);
/// Gap between characters, so event-driven targets (and their IME
/// layers, which often settle per key) can keep up with a burst.
const KEY_GAP: Duration = Duration::from_millis(1);

/// Maximum characters per typing chunk; the full safety recheck (held
/// modifiers, keyboard state, focus/identity) runs before each one.
pub const X11_CHUNK_CHARS: usize = 16;

/// How long a borrowed keycode stays mapped after the last key was
/// synced, before the mapping is restored. A slow target may process
/// the `MappingNotify` for the remap (and refresh its keymap copy)
/// noticeably after the key events themselves were queued; restoring
/// immediately would let such a target decode the character against
/// the restored all-`NoSymbol` mapping and drop it.
const KEYMAP_SETTLE: Duration = Duration::from_millis(50);

/// Serializes every remap *transaction* of this process: from the
/// first `ChangeKeyboardMapping` a transaction writes until it has
/// restored (or decided not to restore) every keycode it touched.
/// Concurrent inserts therefore never share a spare keycode, and the
/// "still all-`NoSymbol`?" re-read each transaction performs cannot
/// race a sibling transaction inside this process. Other X clients
/// are serialized by the server and handled by the same re-read plus
/// the restore-only-if-unchanged rule.
static REMAP_LOCK: Mutex<()> = Mutex::new(());

/// The X11 backend. Stateless: every call opens its own connection
/// (see the module docs for why that is simpler and *more* robust than
/// caching one), except for the pid exclusion policy it is configured
/// with (see [`Self::with_excluded_pids`]).
#[derive(Debug)]
pub struct X11Backend {
    excluded_pids: Vec<u32>,
}

impl X11Backend {
    pub fn new() -> X11Backend {
        X11Backend::with_excluded_pids(vec![std::process::id()])
    }

    /// Construct with an explicit Starling-ownership policy: every pid
    /// in `excluded_pids` marks a target this backend refuses to type
    /// into. Production callers get this through
    /// [`crate::Inserter::with_excluded_pids`], which unions the list
    /// with this process's own pid; constructing a backend directly
    /// with a list that omits it is for probes and the interactive
    /// tests that deliberately type into their own window.
    pub fn with_excluded_pids(excluded_pids: Vec<u32>) -> X11Backend {
        X11Backend {
            excluded_pids: merge_excluded_pids(excluded_pids),
        }
    }

    /// The configured exclusion set (always contains this process).
    pub fn excluded_pids(&self) -> &[u32] {
        &self.excluded_pids
    }
}

impl Default for X11Backend {
    fn default() -> Self {
        X11Backend::new()
    }
}

/// One live X session: a connection plus the screen's root window.
struct Session {
    conn: RustConnection,
    root: Window,
    /// Whether `xkb_use_extension` succeeded on this connection; XKB
    /// requests before that handshake are not guaranteed to work, and
    /// `GetState` is asked for on every chunk, so memoize the probe.
    xkb_ready: Cell<bool>,
}

impl Session {
    fn open() -> Result<Session, InsertError> {
        let (conn, screen) =
            x11rb::connect(None).map_err(|error| unavailable(&connect_error(&error)))?;
        let root = conn.setup().roots[screen].root;
        Ok(Session {
            conn,
            root,
            xkb_ready: Cell::new(false),
        })
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
            Some(reply) if reply.format == 32 && reply.value.len() >= 4 => {
                Ok(Some(u32::from_ne_bytes(
                    reply.value[..4]
                        .try_into()
                        .expect("four bytes are four bytes"),
                )))
            }
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
    /// omits it, and the Starling-owns-it guard falls back to the
    /// WM_CLASS heuristic).
    fn wm_pid(&self, window: Window) -> Result<Option<u32>, InsertError> {
        let atom = self.atom("_NET_WM_PID")?;
        self.card32(window, atom)
    }

    /// `WM_CLASS` as `(instance, class)` — "instance\0class\0" per the
    /// ICCCM. `None` when the property is absent; either half may be
    /// empty when a client set a degenerate value.
    fn wm_class_parts(&self, window: Window) -> Result<Option<(String, String)>, InsertError> {
        let atom = self.atom("WM_CLASS")?;
        let Some(reply) = self.get_property(window, atom)? else {
            return Ok(None);
        };
        let parts: Vec<String> = reply
            .value
            .split(|&byte| byte == 0)
            .filter(|part| !part.is_empty())
            .map(|part| String::from_utf8_lossy(part).into_owned())
            .collect();
        Ok(match parts.len() {
            0 => None,
            1 => Some((String::new(), parts[0].clone())),
            _ => Some((parts[0].clone(), parts[1].clone())),
        })
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

    /// The live keysyms of one keycode — the re-read the remap
    /// transaction's safety rules are built on.
    fn keycode_syms(&self, keycode: Keycode) -> Result<Vec<Keysym>, InsertError> {
        Ok(self
            .conn
            .get_keyboard_mapping(keycode, 1)
            .map_err(x11_conn_error)?
            .reply()
            .map_err(reply_error)?
            .keysyms)
    }

    /// The modifier mapping: eight rows (Shift, Lock, Control,
    /// Mod1..Mod5) of the keycodes bound to each modifier.
    fn modifier_rows(&self) -> Result<Vec<Vec<Keycode>>, InsertError> {
        let reply = self
            .conn
            .get_modifier_mapping()
            .map_err(x11_conn_error)?
            .reply()
            .map_err(reply_error)?;
        let per_row = reply.keycodes_per_modifier() as usize;
        if per_row == 0 || reply.keycodes.len() != per_row * 8 {
            return Err(InsertError::Rejected {
                reason: "the X server reported an unusable modifier mapping".to_string(),
            });
        }
        Ok((0..8)
            .map(|row| reply.keycodes[row * per_row..(row + 1) * per_row].to_vec())
            .collect())
    }

    /// Handshake the XKB extension. Without it the live keyboard state
    /// (held modifiers, active group, locks) is unreadable, and typing
    /// without reading it would be guessing — refuse instead.
    fn xkb_init(&self) -> Result<(), InsertError> {
        if self.xkb_ready.get() {
            return Ok(());
        }
        use xkb::ConnectionExt as _;
        let reply = self
            .conn
            .xkb_use_extension(1, 0)
            .map_err(x11_conn_error)?
            .reply()
            .map_err(reply_error)?;
        if !reply.supported {
            return Err(InsertError::KeyboardStateUnsupported {
                reason: "the X server does not support the XKB extension, so held modifiers \
                         and keyboard state cannot be read"
                    .to_string(),
            });
        }
        self.xkb_ready.set(true);
        Ok(())
    }

    /// One XKB state snapshot of the core keyboard.
    fn xkb_state(&self) -> Result<xkb::GetStateReply, InsertError> {
        use xkb::ConnectionExt as _;
        self.conn
            .xkb_get_state(USE_CORE_KEYBOARD)
            .map_err(x11_conn_error)?
            .reply()
            .map_err(reply_error)
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
        let class = session.wm_class_parts(active)?;
        if self.owns_target(pid, &class) {
            // Starling never types into Starling, and capture is where
            // the app learns its target, so the refusal starts here
            // (insert re-checks the *live* target the same way).
            return Err(InsertError::TargetIsStarling);
        }
        let title = session.window_title(active)?;
        Ok(TargetSnapshot {
            backend: BackendKind::X11,
            target_ref: format_ref(BackendKind::X11, active as u64, focus as u64, pid),
            app: class.map(|(_, class)| class),
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
        cheap_insertion_guards(text, target.pid, &self.excluded_pids)?;
        // One connection for the checks and the keys (module docs): the
        // revalidation inside the chunk loop rides the same server
        // view as the typing.
        let session = Session::open()?;

        // ---- prepare everything BEFORE the final revalidation ----
        // (a failed check must leave the keyboard exactly as it was,
        // so nothing is borrowed or locked until the plan is complete)
        let keyboard = Keyboard::load(&session)?;
        session.xkb_init()?;
        let mut plans: HashMap<char, CharPlan> = HashMap::new();
        let mut remap_keysyms: Vec<Keysym> = Vec::new();
        for character in text.chars() {
            if plans.contains_key(&character) {
                continue;
            }
            let keysym = keysym_for_char(character);
            match keyboard.find(keysym) {
                Some((keycode, needs_shift)) => {
                    plans.insert(
                        character,
                        CharPlan::Pre {
                            keycode,
                            needs_shift,
                        },
                    );
                }
                None => {
                    plans.insert(character, CharPlan::NeedsKeysym(keysym));
                    remap_keysyms.push(keysym);
                }
            }
        }
        let mut remap = RemapTransaction::begin(&session, &keyboard, &remap_keysyms)?;
        for plan in plans.values_mut() {
            if let CharPlan::NeedsKeysym(keysym) = *plan {
                let keycode = remap
                    .keycode_for(keysym)
                    .expect("begin assigned every keysym");
                *plan = CharPlan::Remapped { keycode };
            }
        }

        // ---- the final revalidation, then chunks with rechecks ----
        let segments = weighed_segments(text, X11_CHUNK_CHARS, |_| 1);
        let total = text.chars().count();
        let typed = deliver_in_chunks(
            total,
            &segments,
            || self.chunk_check(&session, &target.target_ref),
            |segment, delivered| type_segment(&session, &keyboard, &plans, segment, delivered),
        );
        match typed {
            Ok(receipt) => {
                // The receipt's claim is only honest once the borrowed
                // keycodes are given back (or deliberately left to
                // another client); a failed restore is the error to
                // surface, not a silent success.
                remap.finish()?;
                Ok(receipt)
            }
            Err(error) => Err(error), // `remap` drops: guarded best-effort restore
        }
    }
}

impl X11Backend {
    /// The live comparison behind both standalone `revalidate` and
    /// the in-insert checks.
    fn revalidate_on(
        &self,
        session: &Session,
        target: &TargetSnapshot,
    ) -> Result<TargetCheck, InsertError> {
        Ok(match live_target(session, target)? {
            LiveTarget::Same { .. } => TargetCheck::Same,
            LiveTarget::Changed { expected, actual } => TargetCheck::Changed { expected, actual },
            LiveTarget::Gone => TargetCheck::Gone,
        })
    }

    /// The full before-every-chunk check: held modifiers waited out,
    /// keyboard state verified, then the live identity (and Starling
    /// ownership) revalidated — immediately before the keys.
    fn chunk_check(&self, session: &Session, target_ref: &str) -> Result<(), InsertError> {
        wait_modifiers_released(session)?;
        check_keyboard_state(session)?;
        // After a (possibly seconds-long) modifier wait the world may
        // have moved; the identity check runs now, not earlier.
        match live_target_on_ref(session, target_ref)? {
            LiveTarget::Same { live_pid, active } => {
                if self.owns_target(live_pid, &None) {
                    return Err(InsertError::TargetIsStarling);
                }
                if live_pid.is_none() {
                    // No pid to compare: the documented WM_CLASS
                    // heuristic decides Starling ownership instead.
                    let class = session.wm_class_parts(active)?;
                    if self.owns_target(None, &class) {
                        return Err(InsertError::TargetIsStarling);
                    }
                }
                Ok(())
            }
            LiveTarget::Changed { expected, actual } => {
                Err(InsertError::TargetChanged { expected, actual })
            }
            LiveTarget::Gone => Err(InsertError::TargetGone),
        }
    }

    /// Whether a target with this (pid, WM_CLASS) is Starling itself:
    /// a pid in the exclusion set, or — only when no pid is known —
    /// the WM_CLASS heuristic below.
    fn owns_target(&self, pid: Option<u32>, class: &Option<(String, String)>) -> bool {
        if pid.is_some_and(|pid| self.excluded_pids.contains(&pid)) {
            return true;
        }
        pid.is_none()
            && class
                .as_ref()
                .is_some_and(|(instance, class)| is_starling_wm_class(instance, class))
    }
}

/// The live state of a frozen ref: identity first, then the pid
/// comparison a plain id match cannot see.
enum LiveTarget {
    /// Same window, same focus — and the pid the *live* window now
    /// reports (the ownership checks and heuristics read it from
    /// here, never from the frozen snapshot).
    Same {
        live_pid: Option<u32>,
        active: Window,
    },
    Changed {
        expected: String,
        actual: String,
    },
    Gone,
}

/// The `LiveTarget` of a snapshot's ref (its parsing twin, for the
/// chunk loop which works from the ref string directly).
fn live_target_on_ref(session: &Session, target_ref: &str) -> Result<LiveTarget, InsertError> {
    let Some((kind, active, focus, captured_pid)) = parse_ref(target_ref) else {
        return Err(InsertError::Rejected {
            reason: format!("malformed target ref: {target_ref}"),
        });
    };
    debug_assert_eq!(kind, BackendKind::X11, "the inserter routes by scheme");
    live_target_for(
        session,
        kind,
        active as Window,
        focus as Window,
        captured_pid,
        target_ref,
    )
}

/// The `LiveTarget` of a snapshot.
fn live_target(session: &Session, target: &TargetSnapshot) -> Result<LiveTarget, InsertError> {
    let Some((kind, active, focus, captured_pid)) = parse_ref(&target.target_ref) else {
        return Err(InsertError::Rejected {
            reason: format!("malformed target ref: {}", target.target_ref),
        });
    };
    debug_assert_eq!(kind, BackendKind::X11, "the inserter routes by scheme");
    live_target_for(
        session,
        kind,
        active as Window,
        focus as Window,
        captured_pid,
        &target.target_ref,
    )
}

fn live_target_for(
    session: &Session,
    kind: BackendKind,
    active: Window,
    focus: Window,
    captured_pid: Option<u32>,
    expected_ref: &str,
) -> Result<LiveTarget, InsertError> {
    // A destroyed window is `Gone` even if focus also moved — the
    // app closing is the fact the user needs; where focus went is
    // secondary detail they can see.
    if !session.window_exists(active)? {
        return Ok(LiveTarget::Gone);
    }
    match session.focus_pair()? {
        Some((live_active, live_focus)) if live_active == active && live_focus == focus => {
            let live_pid = session.wm_pid(active)?;
            match (captured_pid, live_pid) {
                // Same window id, different owning process: the id was
                // recycled by the server — the app that closed and a
                // new one that reused the id. Not the same target.
                (Some(captured), Some(live)) if captured != live => Ok(LiveTarget::Changed {
                    expected: expected_ref.to_string(),
                    actual: format_ref(kind, active as u64, focus as u64, live_pid),
                }),
                // A pid where capture saw none: the window content was
                // replaced under the same id (a set of circumstances
                // only a different window explains) — changed, not
                // same.
                (None, Some(live)) => Ok(LiveTarget::Changed {
                    expected: expected_ref.to_string(),
                    actual: format_ref(kind, active as u64, focus as u64, Some(live)),
                }),
                // (Some, None): the property vanished; no pid story to
                // tell, the ids still match. Same.
                _ => Ok(LiveTarget::Same { live_pid, active }),
            }
        }
        // Focus or activity moved: name the new state honestly —
        // a real ref when one can be captured, else a scheme tag
        // that says "not a window".
        Some((live_active, live_focus)) => {
            let live_pid = session.wm_pid(live_active).unwrap_or(None);
            Ok(LiveTarget::Changed {
                expected: expected_ref.to_string(),
                actual: format_ref(kind, live_active as u64, live_focus as u64, live_pid),
            })
        }
        None => Ok(LiveTarget::Changed {
            expected: expected_ref.to_string(),
            actual: format!("{}:none", kind.scheme()),
        }),
    }
}

/// The Starling WM_CLASS heuristic, used *only* when no `_NET_WM_PID`
/// is available to compare: a window whose WM_CLASS instance or class
/// half equals "starling-gpui" or "Starling" (case-insensitive) is
/// treated as Starling itself. Heuristic by construction — WM_CLASS
/// is client-chosen and two apps can share a class — but the failure
/// mode is a refused insert into a non-Starling window with a
/// Starling-lookalike class, which the copy fallback survives, while
/// the failure mode of *not* checking is Starling typing into its own
/// editor. Deliberately documented, deliberately narrow.
fn is_starling_wm_class(instance: &str, class: &str) -> bool {
    ["starling-gpui", "starling"]
        .into_iter()
        .any(|name| instance.eq_ignore_ascii_case(name) || class.eq_ignore_ascii_case(name))
}

/// The names of the modifier keys physically held right now (XKB
/// `base_mods`; latched/locked state is [`check_keyboard_state`]'s
/// business). `None` means the keyboard is at rest.
fn held_modifier_names(session: &Session) -> Result<Option<Vec<String>>, InsertError> {
    let state = session.xkb_state()?;
    let base = u16::from(state.base_mods);
    if base == 0 {
        return Ok(None);
    }
    const NAMES: [&str; 8] = [
        "Shift",
        "Lock",
        "Control",
        "Mod1 (usually Alt)",
        "Mod2 (usually NumLock)",
        "Mod3",
        "Mod4 (usually Super)",
        "Mod5 (usually AltGr)",
    ];
    Ok(Some(
        NAMES
            .iter()
            .enumerate()
            .filter(|(bit, _)| base & (1 << bit) != 0)
            .map(|(_, name)| (*name).to_string())
            .collect(),
    ))
}

/// Wait (bounded, polling) for physically held modifiers to clear; see
/// [`MODIFIER_RELEASE_WAIT`]. No modifier release is ever synthesized:
/// the user's keyboard is the user's.
fn wait_modifiers_released(session: &Session) -> Result<(), InsertError> {
    let deadline = Instant::now() + MODIFIER_RELEASE_WAIT;
    loop {
        if let Some(held) = held_modifier_names(session)? {
            if Instant::now() >= deadline {
                return Err(InsertError::ModifiersHeld { held });
            }
            std::thread::sleep(MODIFIER_POLL_INTERVAL);
        } else {
            return Ok(());
        }
    }
}

/// Refuse keyboard states synthetic typing cannot reproduce: an
/// active group other than the first (columns 0/1 would not be the
/// effective characters), or Caps/Shift Lock — or any locked or
/// latched modifier from the hazardous set — engaged. Do not guess.
fn check_keyboard_state(session: &Session) -> Result<(), InsertError> {
    let state = session.xkb_state()?;
    if u8::from(state.group) != 0
        || u8::from(state.locked_group) != 0
        || state.base_group != 0
        || state.latched_group != 0
    {
        return Err(InsertError::KeyboardStateUnsupported {
            reason: format!(
                "an alternate keyboard group is active (group {})",
                u8::from(state.group)
            ),
        });
    }
    let locked = u16::from(state.locked_mods) & HAZARDOUS_LOCKED_MODS;
    if locked != 0 {
        return Err(InsertError::KeyboardStateUnsupported {
            reason: format!(
                "a modifier that changes plain typing is locked on (mask {locked:#x}: Caps/Shift \
                 Lock, Alt, Super or AltGr)"
            ),
        });
    }
    let latched = u16::from(state.latched_mods) & HAZARDOUS_LOCKED_MODS;
    if latched != 0 {
        return Err(InsertError::KeyboardStateUnsupported {
            reason: format!(
                "a modifier that changes plain typing is latched on (mask {latched:#x}: sticky \
                 Shift, Alt, Super or AltGr)"
            ),
        });
    }
    Ok(())
}

/// The keyboard mapping snapshot one insert types through, plus what
/// the safety rules need from the modifier mapping. The mapping is
/// server-global and *does* change underneath (the remap path itself
/// does that), so answers are consumed immediately, never cached
/// across inserts; spare-keycode assignments additionally re-read the
/// live mapping before use.
struct Keyboard {
    min_keycode: Keycode,
    /// Keysyms per keycode (the flat `GetKeyboardMapping` layout).
    width: usize,
    syms: Vec<Keysym>,
    /// A keycode bound to the Shift *modifier* (via
    /// `GetModifierMapping`) whose own columns carry a Shift keysym —
    /// pressing it really presses Shift. `None` on a keyboard with no
    /// such binding, and then column-1 characters take the remap path
    /// instead of typing a Shift that is not there.
    shift: Option<Keycode>,
    /// All-`NoSymbol` keycodes of the snapshot, from the top down:
    /// the remap candidates, in the deterministic order the
    /// transaction tries them.
    spares: Vec<Keycode>,
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
            spares: Vec::new(),
        };
        // The Shift key, the policy way: the Shift *modifier's* row of
        // the modifier mapping, and among its keycodes one that
        // actually carries a Shift keysym in the trusted columns. A
        // Shift keysym on an unbound keycode would not set the Shift
        // modifier state targets decode with, so only a bound one
        // counts as usable.
        let rows = session.modifier_rows()?;
        let shift = rows
            .first()
            .expect("eight modifier rows always exist")
            .iter()
            .copied()
            .filter(|&keycode| keycode >= min && keycode <= max)
            .find(|&keycode| {
                keyboard
                    .keysyms(keycode)
                    .iter()
                    .take(2)
                    .any(|&sym| sym == XK_SHIFT_L || sym == XK_SHIFT_R)
            });
        // Spare candidates: every all-NoSymbol keycode, top down (the
        // choice stays clear of the low keycodes real hardware lives
        // on, and deterministic so a test can predict it).
        let count = keyboard.syms.len() / keyboard.width;
        let spares: Vec<Keycode> = (0..count)
            .rev()
            .map(|index| keyboard.min_keycode + index as Keycode)
            .filter(|&keycode| {
                keyboard
                    .keysyms(keycode)
                    .iter()
                    .all(|&sym| sym == x11rb::NO_SYMBOL)
            })
            .collect();
        Ok(Keyboard {
            shift,
            spares,
            ..keyboard
        })
    }

    /// The keysym columns of one keycode.
    fn keysyms(&self, keycode: Keycode) -> &[Keysym] {
        let start = (keycode - self.min_keycode) as usize * self.width;
        &self.syms[start..start + self.width]
    }

    /// Find a keysym in the trusted base/Shift columns:
    /// `(keycode, needs_shift)`. Deeper columns are deliberately not
    /// searched (module docs: their meaning is ambiguous without
    /// XKB), and column 1 matches only when a genuinely bound Shift
    /// keycode exists to press for it — Shift must be the one and
    /// only modifier the character needs.
    fn find(&self, keysym: Keysym) -> Option<(Keycode, bool)> {
        for (keycode, syms) in self.iter_keycodes() {
            if syms.first() == Some(&keysym) {
                return Some((keycode, false));
            }
            if self.shift.is_some() && syms.get(1) == Some(&keysym) {
                return Some((keycode, true));
            }
        }
        None
    }

    fn iter_keycodes(&self) -> impl Iterator<Item = (Keycode, &[Keysym])> {
        let count = self.syms.len() / self.width;
        (0..count).map(move |index| {
            let keycode = self.min_keycode + index as Keycode;
            (
                keycode,
                &self.syms[index * self.width..(index + 1) * self.width],
            )
        })
    }
}

/// How one character will be typed. `NeedsKeysym` is the planning
/// state before the remap transaction assigned a keycode.
enum CharPlan {
    /// Type the keycode the mapping already produces (with Shift when
    /// column 1 is the character's column).
    Pre { keycode: Keycode, needs_shift: bool },
    /// Ride a temporarily remapped spare keycode.
    Remapped { keycode: Keycode },
    /// Planning only: no pre-mapped key produces this character.
    NeedsKeysym(Keysym),
}

/// One temporarily-remapped keycode, with everything needed to give it
/// back. `(keycode, what Starling wrote, what was there before)` —
/// the before is all-`NoSymbol` by the spare rule, recorded from the
/// live re-read rather than assumed.
type BorrowedKey = (Keycode, Vec<Keysym>, Vec<Keysym>);

/// A keyboard-mapping borrow: the process-wide lock, the keycodes
/// written, and the discipline of giving them back. Created *before*
/// the first mutating request is sent (the entry list is what Drop
/// restores), finished explicitly on the success path (where a failed
/// restore is a real error) and by Drop everywhere else.
struct RemapTransaction<'a> {
    session: &'a Session,
    /// Held for its drop (the serialization itself); never read.
    #[allow(dead_code)]
    lock: Option<MutexGuard<'static, ()>>,
    borrowed: Vec<BorrowedKey>,
    finished: bool,
}

impl RemapTransaction<'_> {
    /// Borrow spare keycodes for every `needed` keysym. Each keycode's
    /// *live* mapping is re-read immediately before the write and
    /// must still be all-`NoSymbol` (another client's remap means the
    /// next candidate; none free is a `keyboard_busy` refusal). With
    /// nothing to borrow this is a no-op that takes no lock — an
    /// all-pre-mapped insert never serializes against anything.
    fn begin<'a>(
        session: &'a Session,
        keyboard: &Keyboard,
        needed: &[Keysym],
    ) -> Result<RemapTransaction<'a>, InsertError> {
        if needed.is_empty() {
            return Ok(RemapTransaction {
                session,
                lock: None,
                borrowed: Vec::new(),
                finished: false,
            });
        }
        let lock = REMAP_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut transaction = RemapTransaction {
            session,
            lock: Some(lock),
            borrowed: Vec::new(),
            finished: false,
        };
        let mut candidates = keyboard.spares.iter().copied();
        'needed: for &keysym in needed {
            while let Some(candidate) = candidates.next() {
                let live = session.keycode_syms(candidate)?;
                if !live.iter().all(|&sym| sym == x11rb::NO_SYMBOL) {
                    // Another client (or a Starling in another process)
                    // owns this keycode now; leave it alone.
                    continue;
                }
                let mut wrote = vec![x11rb::NO_SYMBOL; live.len()];
                wrote[0] = keysym;
                // The restore guard is recorded BEFORE the mutating
                // request: a failure between record and write restores
                // a no-op, a failure after it restores the original,
                // and nothing is left borrowed unrecorded.
                transaction.borrowed.push((candidate, wrote.clone(), live));
                match session
                    .conn
                    .change_keyboard_mapping(1, candidate, wrote.len() as u8, &wrote)
                {
                    Ok(cookie) => {
                        if let Err(error) = cookie.check() {
                            return Err(reply_error(error));
                        }
                    }
                    Err(error) => return Err(x11_conn_error(error)),
                }
                // The server canonicalizes what it stores (an XKB
                // server may echo a two-column write back padded or
                // duplicated per group), so "what Starling wrote" is
                // what the server says it now holds — re-read and
                // record that as the comparison baseline for the
                // restore, or the restore's other-client check would
                // disown our own mapping and leave it borrowed.
                let echoed = session.keycode_syms(candidate)?;
                transaction.borrowed.last_mut().expect("just pushed").1 = echoed;
                continue 'needed;
            }
            return Err(InsertError::Rejected {
                reason: format!(
                    "every spare keycode is already in use, so the character U+{:04X} cannot \
                     be typed (keyboard busy)",
                    keysym_codepoint(keysym)
                ),
            });
        }
        // Round trip: the mapping must be live before any keypress is
        // interpreted by anyone.
        session.sync()?;
        Ok(transaction)
    }

    /// The keycode this transaction borrowed for `keysym`.
    fn keycode_for(&self, keysym: Keysym) -> Option<Keycode> {
        self.borrowed
            .iter()
            .find(|(_, wrote, _)| wrote.first() == Some(&keysym))
            .map(|(keycode, _, _)| *keycode)
    }

    /// Give every borrowed keycode back. The server is synced (all
    /// queued keys processed) and the mapping held through
    /// [`KEYMAP_SETTLE`] first; each keycode's live mapping is re-read
    /// and restored only if it still equals exactly what Starling
    /// wrote — another client's change is left alone, with a warning.
    fn finish(&mut self) -> Result<(), InsertError> {
        if self.finished {
            return Ok(());
        }
        self.finished = true;
        if self.borrowed.is_empty() {
            return Ok(());
        }
        self.session.sync()?;
        std::thread::sleep(KEYMAP_SETTLE);
        let mut failure: Option<InsertError> = None;
        for (keycode, wrote, original) in &self.borrowed {
            match self.session.keycode_syms(*keycode) {
                Ok(live) if live == *wrote => {
                    let restore = self
                        .session
                        .conn
                        .change_keyboard_mapping(1, *keycode, original.len() as u8, original)
                        .map_err(x11_conn_error)
                        .and_then(|cookie| cookie.check().map_err(reply_error));
                    if let Err(error) = restore {
                        failure = failure.or_else(|| {
                            InsertError::KeyboardRestoreFailed {
                                detail: format!("restoring keycode {keycode}: {error}"),
                            }
                            .into()
                        });
                    }
                }
                Ok(live) => {
                    // Not ours anymore: another client re-mapped it
                    // while borrowed. Restoring would clobber *their*
                    // change; leave it and say so.
                    log::warn!(
                        "starling-insertion: keycode {keycode} was re-mapped by another \
                         client while Starling borrowed it; leaving their mapping {live:?} \
                         in place"
                    );
                }
                Err(error) => {
                    failure = failure.or_else(|| {
                        InsertError::KeyboardRestoreFailed {
                            detail: format!("re-reading keycode {keycode}: {error}"),
                        }
                        .into()
                    });
                }
            }
        }
        match failure {
            None => Ok(()),
            Some(error) => Err(error),
        }
    }
}

impl Drop for RemapTransaction<'_> {
    fn drop(&mut self) {
        if !self.finished {
            // Errors are swallowed (Drop cannot report; the insert's
            // own error is the honest one to surface), but the
            // restore discipline itself is identical.
            let _ = self.finish();
        }
    }
}

/// The keys a single segment press has put down, released in reverse
/// on drop — a failure mid-press may not leave a key (worst: Shift)
/// stuck down for the rest of the session.
struct PressedKeys<'a> {
    session: &'a Session,
    keys: Vec<Keycode>,
}

impl<'a> PressedKeys<'a> {
    fn new(session: &'a Session) -> PressedKeys<'a> {
        PressedKeys {
            session,
            keys: Vec::new(),
        }
    }

    fn press(&mut self, keycode: Keycode) -> Result<(), InsertError> {
        // Recorded before the send: a send that errors still gets a
        // release attempt, because "the server maybe did not see the
        // press" is not provable from here.
        self.keys.push(keycode);
        fake_key(self.session, keycode, KEY_PRESS)
    }

    fn release(&mut self, keycode: Keycode) -> Result<(), InsertError> {
        let result = fake_key(self.session, keycode, KEY_RELEASE);
        self.keys.retain(|&pressed| pressed != keycode);
        result
    }
}

impl Drop for PressedKeys<'_> {
    fn drop(&mut self) {
        while let Some(keycode) = self.keys.pop() {
            if let Err(error) = fake_key(self.session, keycode, KEY_RELEASE) {
                log::warn!(
                    "starling-insertion: releasing keycode {keycode} after a failed insert \
                     also failed: {error}"
                );
            }
        }
    }
}

/// Type one segment (up to [`X11_CHUNK_CHARS`] characters), reporting
/// exactly how many characters landed if typing fails part-way.
fn type_segment(
    session: &Session,
    keyboard: &Keyboard,
    plans: &HashMap<char, CharPlan>,
    segment: &str,
    delivered_before: usize,
) -> Result<usize, ChunkFailure> {
    let mut pressed = PressedKeys::new(session);
    let mut typed = 0usize;
    for character in segment.chars() {
        let plan = plans
            .get(&character)
            .expect("every character of the text was planned");
        if let Err(cause) = type_character(keyboard, plan, &mut pressed) {
            return Err(ChunkFailure {
                delivered: delivered_before + typed,
                cause,
            });
        }
        typed += 1;
    }
    Ok(typed)
}

/// Type one character through its plan: press Shift if the column
/// needs it, tap the key, release in reverse. The `pressed` guard
/// guarantees the releases even when a send fails mid-keystroke.
fn type_character(
    keyboard: &Keyboard,
    plan: &CharPlan,
    pressed: &mut PressedKeys<'_>,
) -> Result<(), InsertError> {
    let (keycode, needs_shift) = match plan {
        CharPlan::Pre {
            keycode,
            needs_shift,
        } => (*keycode, *needs_shift),
        CharPlan::Remapped { keycode } => (*keycode, false),
        CharPlan::NeedsKeysym(_) => {
            return Err(InsertError::Rejected {
                reason: "internal error: an unplanned character reached the typer".to_string(),
            })
        }
    };
    // A real Shift press (a key event the target sees, exactly like a
    // human typing `C`) rather than a synthetic modifier mask — XTest
    // only speaks keycodes — and only the modifier-mapping-bound Shift
    // keycode `find` verified before planning a column-1 character.
    if needs_shift {
        let shift = keyboard
            .shift
            .expect("a shift-needing plan exists only when a Shift keycode is bound");
        pressed.press(shift)?;
    }
    pressed.press(keycode)?;
    std::thread::sleep(KEY_HOLD);
    pressed.release(keycode)?;
    if needs_shift {
        let shift = keyboard
            .shift
            .expect("a shift-needing plan exists only when a Shift keycode is bound");
        pressed.release(shift)?;
    }
    std::thread::sleep(KEY_GAP);
    Ok(())
}

/// The standard keysym for a character: Latin-1 for the first 256
/// code points (the keysym space *is* Latin-1 there), the Unicode
/// convention `0x0100_0000 | codepoint` above. The control/function
/// keysym range (`0xfe00..=0xffff`) is unreachable: guards refuse
/// control characters (the only Latin-1 code points that map there),
/// and every other character takes the Unicode form, which the range
/// cannot collide with. A keysym this function returns is therefore
/// always a character keysym, never a control or function one.
fn keysym_for_char(character: char) -> Keysym {
    let codepoint = character as u32;
    if codepoint < 0x100 {
        codepoint
    } else {
        0x0100_0000 | codepoint
    }
}

/// The codepoint a keysym from [`keysym_for_char`] encodes (for
/// messages).
fn keysym_codepoint(keysym: Keysym) -> u32 {
    if keysym >= 0x0100_0000 {
        keysym - 0x0100_0000
    } else {
        keysym
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
