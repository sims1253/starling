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
//! # Threat model
//!
//! X11 has no isolation between clients: any client on the display can
//! read every key, inject its own (XTest) and rewrite the keyboard
//! mapping. A hostile X client therefore needs no race against Starling
//! to type Enter — it can simply send one. What this backend defends
//! against is *benign* interference: another program (a layout switcher,
//! `setxkbmap`, an IME helper) changing the keymap, the focus or the
//! modifier state while a transcript is being typed. Every keystroke is
//! verified against the live mapping inside a short `GrabServer` section
//! that spans the whole press–hold–release, which makes the check atomic
//! on servers that honor the grab (stock Xorg and Xvfb do). Some servers
//! do not: WSLg's XWayland lets other clients' requests through during a
//! grab (measured during #221), so there the per-key verification is the
//! remaining protection and a change that lands inside one keystroke's
//! few-millisecond hold cannot be prevented client-side. Wayland sessions
//! do not use this backend at all.
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
//! - **Effective/locked group and locked or latched modifiers**
//!   (`group`, `locked_mods`, `latched_mods`): if a group other than
//!   the first is active, or *any* modifier bit is locked or latched
//!   except the one bit the live modifier mapping actually binds to
//!   `Num_Lock`, the base/Shift columns do not identify the effective
//!   character, and the backend refuses with
//!   [`InsertError::KeyboardStateUnsupported`] instead of guessing.
//!   Which bit Num_Lock is must be *read* from the server, not
//!   assumed: layouts bind it anywhere in Mod1..Mod5 (or nowhere),
//!   and a locked Mod2 that is not this session's Num_Lock changes
//!   what a bare keypress means.
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
//! - a process-wide lock (`INSERT_LOCK`) serializes whole X11
//!   *inserts* of this process — taken before the keyboard mapping is
//!   loaded for planning and held through typing and cleanup — because
//!   even a mapping *read* can consume a sibling's borrow: a mapping
//!   loaded while another insert of this process holds a temporary
//!   remap plans "pre-mapped" keycodes that the sibling restores to
//!   all-`NoSymbol` before the first key is typed. One X11 insert runs
//!   at a time per process (the lock's own docs say why that is the
//!   honest granularity);
//! - each borrow's short read→write→echo-read sequence and each
//!   restore's compare→write sequence run inside a server grab
//!   (`GrabServer`/`UngrabServer`), so no *other X client* can
//!   interleave a remap of the same keycode between the ownership
//!   check and the write — and each character's *whole keystroke*
//!   (verify the live mapping, Shift↓, key↓, the key's hold time,
//!   key↑, Shift↑, flush) runs inside one grab, so no client can
//!   interleave a remap between the pre-press check and the key-down
//!   *or between the key-down and the key-up*: X translates
//!   KeyRelease events through the live mapping just as it does
//!   presses, and targets can act on a release, so a remap landing
//!   mid-hold would make Starling's own release decode as a foreign
//!   keysym — another client's `Return`, say, which is the no-Enter
//!   rule broken by a foreign hand. The keystroke grab holds
//!   [`KEY_HOLD`], the one sleep ever taken while grabbed (that is
//!   what makes the guarantee deterministic instead of a race an
//!   interloper might lose); a grab is therefore held for roughly
//!   `KEY_HOLD` plus a few requests per character — never across the
//!   inter-character gap, a wait or an event read — and the guard
//!   releases it even on error. This exclusion is the X protocol's
//!   `GrabServer` contract ("the processing of requests from other
//!   clients is curtailed"); a server that does not honor it — WSLg's
//!   XWayland does not, measured — voids the no-interleave guarantee
//!   for every grabbed section alike, and no client-side mechanism
//!   can restore it there. The grabs are held as specified
//!   regardless, so a conforming server gets the full guarantee;
//! - right before remapping, the keycode's *live* mapping is re-read
//!   and required to still be all-`NoSymbol` (another client's remap
//!   picks the next spare; none free is a `keyboard_busy`-style
//!   refusal). The requested keysym is written into **every** column
//!   of the keycode — all groups × levels the server reports — so
//!   there is no lowercase/uppercase pair for XKB to expand (writing
//!   only column 0 would let an uppercase letter come back expanded
//!   into a case pair and type in the wrong case, or not at all), and
//!   the server's echo is verified to still carry the requested
//!   keysym in the base column; an unexpected echo is a refusal,
//!   never a panic. The keysym a plan types is kept separately from
//!   the echoed mapping the restore comparison uses;
//! - immediately before each character's key goes down, that keycode's
//!   live mapping is re-read inside the same short grab and required
//!   to still be what the plan expects: a borrowed keycode carries
//!   exactly the recorded echo, a pre-mapped keycode still produces
//!   the planned keysym in the planned column, and the Shift keycode
//!   a column-1 plan presses still carries a Shift keysym in its
//!   unshifted column — the level that press itself decodes through,
//!   since Shift goes down with no other modifier held, so a
//!   `[Return, Shift_L]` remap is refused rather than pressed as
//!   Return — and stays bound to the Shift modifier. The chunk
//!   checks cannot see a remap that lands while they run (the
//!   held-modifier wait alone is seconds of exposure), and a press through a drifted mapping
//!   would type whatever the *new* mapping decodes to — another
//!   client's `Return`, say, which is the no-Enter rule broken by a
//!   foreign hand. A drift stops typing before the key-down with a
//!   keyboard-busy [`InsertError::Rejected`] naming the keycode
//!   (wrapped as [`InsertError::PartialDelivery`] when earlier
//!   characters may have landed), and the conditional restore then
//!   leaves the foreign mapping alone (logged) — an insert never
//!   reports a clean success over a mapping Starling no longer owns;
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
//! state, focus/identity/pid) repeated before *every* chunk and every
//! character's keycode re-verified against the live mapping
//! immediately before its key-down (the transaction rules above); a
//! change part-way stops typing immediately and reports
//! [`InsertError::PartialDelivery`] with how much may have landed — a
//! character counts as possibly delivered the moment its key-down was
//! issued, so the count never understates what the target may hold
//! (even a character whose keystroke did not complete).
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
/// `XK_Num_Lock` — the keysym whose bound modifier bit is the one
/// locked modifier that stays harmless (see [`check_keyboard_state`]:
/// which bit that is is read from the live modifier mapping, never
/// assumed).
const XK_NUM_LOCK: Keysym = 0xff7f;

/// `xkbUseCoreKeyboard` — the device `GetState` is asked about.
const USE_CORE_KEYBOARD: u16 = 0x100;

/// How long a synthetic key stays "down". Real keys are down for tens
/// of milliseconds; a zero hold can be dropped or coalesced by
/// toolkits that watch press/release pairing, and the X server's own
/// auto-repeat only fires for keys held far longer than this. This is
/// also the one sleep ever taken while a server grab is held — the
/// per-character keystroke grab spans it (see [`type_character`]).
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

/// Serializes every X11 *insert* of this process, from before the
/// keyboard mapping is loaded for planning until cleanup has given
/// back (or attempted to give back) every borrowed keycode. The
/// granularity is the whole insert, not just the remap write,
/// because the keyboard mapping is server-global state that even a
/// *read* consumes: a `GetKeyboardMapping` performed while a sibling
/// insert of this process holds a temporary remap reports the
/// sibling's keysyms as pre-mapped, the reader plans those keycodes,
/// and the sibling's restore then unmapps them before the reader's
/// first key is typed — one insert silently consuming another's
/// mapping. Holding the lock across load → plan → type → cleanup is
/// the only placement after which no such interleaving exists, so
/// one X11 insert runs at a time per process (inserts are seconds-long
/// typing bursts at most; queueing them is the honest behavior).
/// Other X clients are not covered by this lock — the
/// `GrabServer`-wrapped ownership sections and the restore's
/// compare-before-write handle them.
static INSERT_LOCK: Mutex<()> = Mutex::new(());

/// The X11 backend. Stateless: every call opens its own connection
/// (see the module docs for why that is simpler and *more* robust than
/// caching one), except for the pid exclusion policy it is configured
/// with (see [`Self::with_excluded_pids`]).
#[derive(Debug)]
pub struct X11Backend {
    excluded_pids: Vec<u32>,
    /// How long each key stays down ([`KEY_HOLD`] in production).
    key_hold: Duration,
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
            key_hold: KEY_HOLD,
        }
    }

    /// Hold every key this long instead of [`KEY_HOLD`] — tests only,
    /// so a regression can land interference deterministically inside
    /// one keystroke's hold (and therefore inside its server grab)
    /// instead of racing a 3 ms window against the scheduler.
    #[cfg(any(test, feature = "test-doubles"))]
    pub fn with_key_hold_for_tests(mut self, key_hold: Duration) -> X11Backend {
        self.key_hold = key_hold;
        self
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
    /// The backend's per-key hold (see [`X11Backend::with_key_hold_for_tests`]).
    key_hold: Duration,
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
            key_hold: KEY_HOLD,
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

    /// The modifier bit the *live* modifier mapping binds to `Num_Lock`:
    /// the first row (of the eight) with a keycode whose trusted
    /// columns carry the `Num_Lock` keysym. `None` when no row carries
    /// it — and then no locked or latched bit is harmless. Read from
    /// the server every time it matters, never assumed: which of
    /// Mod1..Mod5 carries NumLock is a property of the session's
    /// layout, and "Mod2 is probably NumLock" is exactly the guess
    /// that types wrong characters.
    fn num_lock_bit(&self) -> Result<Option<u16>, InsertError> {
        let setup = self.conn.setup();
        let (min, max) = (setup.min_keycode, setup.max_keycode);
        let rows = self.modifier_rows()?;
        for (bit, row) in rows.iter().enumerate() {
            for &keycode in row
                .iter()
                .filter(|&keycode| *keycode >= min && *keycode <= max)
            {
                if self
                    .keycode_syms(keycode)?
                    .iter()
                    .take(2)
                    .any(|&sym| sym == XK_NUM_LOCK)
                {
                    return Ok(Some(1u16 << bit));
                }
            }
        }
        Ok(None)
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
        let mut session = Session::open()?;
        session.key_hold = self.key_hold;
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
        let mut session = Session::open()?;
        session.key_hold = self.key_hold;
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
        // The process-wide insert lock, taken BEFORE anything reads the
        // keyboard mapping and held through typing and cleanup
        // ([`INSERT_LOCK`]'s docs say why the whole insert is the honest
        // granularity): one X11 insert at a time per process.
        let _insert_lock = insert_lock();
        // One connection for the checks and the keys (module docs): the
        // revalidation inside the chunk loop rides the same server
        // view as the typing.
        let mut session = Session::open()?;
        session.key_hold = self.key_hold;

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
                // `begin` assigned every keysym it was given or failed
                // the insert; a miss here is still an error, never a
                // panic — the typer must not be able to crash the
                // delivery path.
                let Some(keycode) = remap.keycode_for(keysym) else {
                    return Err(InsertError::Rejected {
                        reason: format!(
                            "internal error: no keycode was borrowed for U+{:04X}",
                            keysym_codepoint(keysym)
                        ),
                    });
                };
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
            |segment, delivered| {
                type_segment(&session, &keyboard, &remap, &plans, segment, delivered)
            },
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
            Err(delivery) => {
                // Cleanup is not skippable on a failed delivery either:
                // `finish` attempts every restore, and a restore that
                // fails must not hide the delivery outcome (nor may
                // the delivery error hide a broken keyboard) — the
                // combined error states both.
                Err(match remap.finish() {
                    Ok(()) => delivery,
                    Err(restore) => combine_delivery_and_restore_failure(&delivery, restore),
                })
            }
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
/// effective characters), or *any* locked or latched modifier bit
/// except the one the live modifier mapping binds to `Num_Lock` —
/// that bit is read from the server here, so a locked Mod2 that is
/// not this session's NumLock (or a NumLock bound elsewhere) is a
/// refusal, and a locked NumLock stays allowed. Do not guess.
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
    let harmless = session.num_lock_bit()?.unwrap_or(0);
    let locked = u16::from(state.locked_mods) & !harmless;
    if locked != 0 {
        return Err(InsertError::KeyboardStateUnsupported {
            reason: format!(
                "a modifier that changes plain typing is locked on (mask {locked:#x}: every \
                 locked modifier except Num_Lock refuses)"
            ),
        });
    }
    let latched = u16::from(state.latched_mods) & !harmless;
    if latched != 0 {
        return Err(InsertError::KeyboardStateUnsupported {
            reason: format!(
                "a modifier that changes plain typing is latched on (mask {latched:#x}: every \
                 latched modifier except Num_Lock refuses)"
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
        // actually carries a Shift keysym in its unshifted column. A
        // Shift keysym on an unbound keycode would not set the Shift
        // modifier state targets decode with, so only a bound one
        // counts as usable; and a Shift keysym in a *shifted* column
        // alone would not do either — this is the keycode the typer
        // presses with no modifier down, so the level that press
        // decodes through is column 0, and a `[Return, Shift_L]`
        // keycode would press as Return (the pre-press verification
        // applies the same effective-column rule live before every
        // press).
        let rows = session.modifier_rows()?;
        let shift = rows
            .first()
            .expect("eight modifier rows always exist")
            .iter()
            .copied()
            .filter(|&keycode| keycode >= min && keycode <= max)
            .find(|&keycode| {
                matches!(
                    keyboard.keysyms(keycode).first(),
                    Some(&XK_SHIFT_L) | Some(&XK_SHIFT_R)
                )
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
/// back. The keysym it was borrowed *for* (planning identity — which
/// keycode types this character) is kept separate from the echoed
/// mapping the server reports (the ownership baseline the restore
/// comparison uses), because the server canonicalizes what it stores
/// and the plan must not depend on that canonicalization. `original`
/// is what was there before (all-`NoSymbol` by the spare rule,
/// recorded from the live re-read rather than assumed).
struct BorrowedKey {
    keycode: Keycode,
    /// The keysym this borrow was requested for — column 0 of what
    /// was written, before the server canonicalized anything.
    keysym: Keysym,
    /// What the server says it now holds for this keycode — the exact
    /// value a restore compares the live mapping against.
    echoed: Vec<Keysym>,
    /// The all-`NoSymbol` columns to write back on restore.
    original: Vec<Keysym>,
}

/// A keyboard-mapping borrow: the keycodes written and the discipline
/// of giving them back. Process-wide serialization is *not* this
/// type's business — [`INSERT_LOCK`] covers the whole insert, and the
/// transaction is created inside it. Entries are recorded *before*
/// the mutating request is sent (the entry list is what Drop
/// restores), finished explicitly on the success path and on the
/// preparation-failure path (where a failed restore is a real error,
/// combined with the preparation cause — [`RemapTransaction::begin`])
/// and by Drop everywhere else.
struct RemapTransaction<'a> {
    session: &'a Session,
    borrowed: Vec<BorrowedKey>,
    finished: bool,
}

impl RemapTransaction<'_> {
    /// Borrow spare keycodes for every `needed` keysym. Each
    /// keycode's read→write→echo-read sequence runs inside a server
    /// grab, so no other X client can remap the same keycode between
    /// the ownership check and the write; the live mapping must still
    /// be all-`NoSymbol` (another client's remap means the next
    /// candidate; none free is a `keyboard_busy`-style refusal); the
    /// requested keysym is written into **every** column (all groups ×
    /// levels the server reports for that keycode, so there is no
    /// case pair for XKB to expand and an unshifted press yields
    /// exactly the requested keysym — the reason an uppercase letter
    /// must not be written into column 0 alone); and the server's
    /// echo must still carry the keysym in the base column, or the
    /// borrow is refused — never a panic. With nothing to borrow this
    /// is a no-op. A failure *part-way* (after some keycodes were
    /// borrowed) does not leave cleanup to Drop, which cannot report:
    /// [`Self::finish`] runs explicitly and a failed restore is
    /// combined with the preparation cause by
    /// [`combine_preparation_with_restore_outcome`] — nothing has been
    /// typed at that point, and the combined error says so.
    fn begin<'a>(
        session: &'a Session,
        keyboard: &Keyboard,
        needed: &[Keysym],
    ) -> Result<RemapTransaction<'a>, InsertError> {
        let mut transaction = RemapTransaction {
            session,
            borrowed: Vec::new(),
            finished: false,
        };
        if let Err(preparation) = transaction.borrow_needed(keyboard, needed) {
            // Every keycode borrowed before the failure is given back
            // here, out-of-band of the error: `finish` attempts every
            // restore (and sets `finished`, so Drop does not retry),
            // and a restore that failed must not hide behind the
            // preparation error (nor may the preparation error hide a
            // broken keyboard) — the combined error states both.
            return Err(combine_preparation_with_restore_outcome(
                preparation,
                transaction.finish(),
            ));
        }
        Ok(transaction)
    }

    /// The borrowing loop behind [`Self::begin`]: one spare keycode
    /// per `needed` keysym, or the first failure for `begin` to clean
    /// up after.
    fn borrow_needed(&mut self, keyboard: &Keyboard, needed: &[Keysym]) -> Result<(), InsertError> {
        let session = self.session;
        if needed.is_empty() {
            return Ok(());
        }
        let mut candidates = keyboard.spares.iter().copied();
        'needed: for &keysym in needed {
            while let Some(candidate) = candidates.next() {
                // The grabbed section: read, verify spare, write,
                // echo-read. Held for these few requests only (the
                // guard releases it on every exit, error included).
                // A grab the server refused means no remap may
                // happen: the character is refused (keyboard busy) —
                // never typed through an unprotected read→write.
                let _grab = ServerGrab::new(session).map_err(|error| match error {
                    // Connection trouble stays "unavailable" — it is
                    // the backend, not the keyboard, that failed.
                    unavailable @ InsertError::Unavailable { .. } => unavailable,
                    cause => InsertError::Rejected {
                        reason: format!(
                            "the X server refused the exclusive grab borrowing a keycode for \
                             U+{:04X} needs, so the character cannot be typed (keyboard busy): \
                             {}",
                            keysym_codepoint(keysym),
                            cause.message()
                        ),
                    },
                })?;
                let live = session.keycode_syms(candidate)?;
                if live.is_empty() {
                    // A keycode with no columns can carry nothing;
                    // not a usable spare (and writing an empty
                    // mapping would be meaningless).
                    continue;
                }
                if !live.iter().all(|&sym| sym == x11rb::NO_SYMBOL) {
                    // Another client (or a Starling in another process)
                    // owns this keycode now; leave it alone.
                    continue;
                }
                // Every column the server reports for this keycode:
                // identical columns leave XKB no pair to expand, so
                // the unshifted press produces exactly `keysym`.
                let wrote = vec![keysym; live.len()];
                // The restore guard is recorded BEFORE the mutating
                // request: a failure between record and write restores
                // a no-op, a failure after it restores the original,
                // and nothing is left borrowed unrecorded.
                self.borrowed.push(BorrowedKey {
                    keycode: candidate,
                    keysym,
                    echoed: wrote.clone(),
                    original: live,
                });
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
                // The server canonicalizes what it stores, so "what
                // Starling wrote" is what the server says it now
                // holds — re-read and record that as the comparison
                // baseline for the restore, or the restore's
                // other-client check would disown our own mapping and
                // leave it borrowed. The echo must also prove the
                // borrow usable: the base column (what an unshifted
                // press types) must still be the requested keysym —
                // an unexpected echo is a refusal, never a panic. It
                // is recorded BEFORE that usability decision: the
                // write has already landed, so a rejected echo is
                // still Starling's mapping and cleanup must compare
                // against — and restore — exactly it; had cleanup
                // compared against the requested write instead, the
                // server's canonicalized form would read as another
                // client's mapping and the borrow would leak.
                let echoed = session.keycode_syms(candidate)?;
                let base_column = echoed.first().copied();
                if let Some(entry) = self.borrowed.last_mut() {
                    entry.echoed = echoed;
                }
                if base_column != Some(keysym) {
                    return Err(InsertError::Rejected {
                        reason: format!(
                            "the X server did not keep U+{:04X} in the base column of keycode \
                             {candidate} (keyboard busy or an unusual XKB canonicalization)",
                            keysym_codepoint(keysym)
                        ),
                    });
                }
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
        // interpreted by anyone. (Outside any grab — it is a wait, and
        // grabs are never held across waits.)
        session.sync()?;
        Ok(())
    }

    /// The keycode this transaction borrowed for `keysym` — looked up
    /// by the keysym the plan asked for, never by the server's
    /// canonicalized echo (which an XKB server may have expanded into
    /// a case-pair shape that hides the requested symbol).
    fn keycode_for(&self, keysym: Keysym) -> Option<Keycode> {
        self.borrowed
            .iter()
            .find(|entry| entry.keysym == keysym)
            .map(|entry| entry.keycode)
    }

    /// The recorded echo for a keycode this transaction borrowed —
    /// the exact baseline the pre-press check compares the live
    /// mapping against (the same value a restore compares against, so
    /// "drifted" means the same thing at the press and at cleanup).
    fn echoed_for(&self, keycode: Keycode) -> Option<&[Keysym]> {
        self.borrowed
            .iter()
            .find(|entry| entry.keycode == keycode)
            .map(|entry| entry.echoed.as_slice())
    }

    /// Give every borrowed keycode back. The server is synced (all
    /// queued keys processed) and the mapping held through
    /// [`KEYMAP_SETTLE`] first; then each keycode's compare→restore
    /// sequence runs inside its own server grab and restores only if
    /// the live mapping still equals exactly what Starling wrote —
    /// another client's change is left alone, with a warning. A
    /// failed initial sync does **not** skip the restores: every one
    /// is attempted regardless (a dying connection makes each attempt
    /// fail on its own and those failures are collected), and
    /// `finished` is only set once restoration has been attempted —
    /// so Drop retries anything not yet attempted. A restore failure
    /// always surfaces; on the error path of an insert it is combined
    /// with the delivery outcome by
    /// [`combine_delivery_and_restore_failure`].
    fn finish(&mut self) -> Result<(), InsertError> {
        if self.finished {
            return Ok(());
        }
        if self.borrowed.is_empty() {
            self.finished = true;
            return Ok(());
        }
        let mut failure: Option<InsertError> = None;
        if let Err(error) = self.session.sync() {
            // Not a reason to skip the restores below: attempt them
            // anyway (the writes may still reach the server), but the
            // flush failure is part of the story.
            failure = Some(InsertError::KeyboardRestoreFailed {
                detail: format!("flushing the typed keys before restoring: {error}"),
            });
        }
        // The settle delay stays outside every grab (KEY_HOLD is the
        // only sleep ever held inside one); it exists so a target that
        // processes its MappingNotify late does not decode
        // already-typed keys against the restored mapping.
        std::thread::sleep(KEYMAP_SETTLE);
        for entry in &self.borrowed {
            let grabbed = ServerGrab::new(self.session);
            let outcome = match grabbed {
                Ok(_grab) => self.restore_one(entry),
                Err(error) => Err(InsertError::KeyboardRestoreFailed {
                    detail: format!(
                        "grabbing the server to restore keycode {}: {error}",
                        entry.keycode
                    ),
                }),
            };
            if let Err(error) = outcome {
                failure = failure.or(Some(error));
            }
        }
        // Only now may the transaction count as finished: every
        // restore has been attempted.
        self.finished = true;
        match failure {
            None => Ok(()),
            Some(error) => Err(error),
        }
    }

    /// The compare→restore of one borrowed keycode — the body the
    /// server grab in [`Self::finish`] wraps.
    fn restore_one(&self, entry: &BorrowedKey) -> Result<(), InsertError> {
        match self.session.keycode_syms(entry.keycode) {
            Ok(live) if live == entry.echoed => {
                let restore = self
                    .session
                    .conn
                    .change_keyboard_mapping(
                        1,
                        entry.keycode,
                        entry.original.len() as u8,
                        &entry.original,
                    )
                    .map_err(x11_conn_error)
                    .and_then(|cookie| cookie.check().map_err(reply_error));
                if let Err(error) = restore {
                    return Err(InsertError::KeyboardRestoreFailed {
                        detail: format!("restoring keycode {}: {error}", entry.keycode),
                    });
                }
                Ok(())
            }
            Ok(live) => {
                // Not ours anymore: another client re-mapped it while
                // borrowed. Restoring would clobber *their* change;
                // leave it and say so.
                log::warn!(
                    "starling-insertion: keycode {} was re-mapped by another client while \
                     Starling borrowed it; leaving their mapping {live:?} in place",
                    entry.keycode
                );
                Ok(())
            }
            Err(error) => Err(InsertError::KeyboardRestoreFailed {
                detail: format!("re-reading keycode {}: {error}", entry.keycode),
            }),
        }
    }
}

impl Drop for RemapTransaction<'_> {
    fn drop(&mut self) {
        if !self.finished {
            // Errors are swallowed (Drop cannot report; the insert's
            // own error is the honest one to surface), but the
            // restore discipline itself is identical — and anything
            // not yet attempted is retried here.
            let _ = self.finish();
        }
    }
}

/// A server grab held across a section and released on drop —
/// also on error, because Drop cannot report (an ungrab that itself
/// fails means the connection is dying and the grab dies with it;
/// both are logged). While grabbed, the X server processes no other
/// client's requests, so a read→write→read-back ownership sequence
/// cannot interleave with another client's remap of the same keycode.
/// Two section shapes exist: the few-request ownership sections (a
/// borrow's read→write→echo-read, a restore's compare→write) hold no
/// sleep at all; and a character's whole keystroke — verify the live
/// mapping, Shift↓, key↓, the [`KEY_HOLD`] sleep, key↑, Shift↑, flush
/// — deliberately holds that one sleep, because X translates
/// KeyRelease events through the live mapping too and a remap landing
/// between a key-down and its key-up would make the release decode as
/// a foreign keysym. A keystroke grab is therefore held for roughly
/// [`KEY_HOLD`] plus a few requests per character; no grab is ever
/// held across the inter-character gap, a modifier wait or an event
/// read. The exclusion itself is the X protocol's `GrabServer`
/// contract: a server that does not curtail other clients' processing
/// during a grab (WSLg's XWayland does not) voids it for every
/// grabbed section alike — the grabs are held as specified
/// regardless, so a conforming server gets the full guarantee.
struct ServerGrab<'a> {
    session: &'a Session,
}

impl<'a> ServerGrab<'a> {
    fn new(session: &'a Session) -> Result<ServerGrab<'a>, InsertError> {
        let cookie = session.conn.grab_server().map_err(x11_conn_error)?;
        // The guard is constructed before the grab's reply is checked,
        // so a rejected grab still unwinds through Drop's ungrab (an
        // ungrab for a grab the server never held is harmless).
        let guard = ServerGrab { session };
        // The grab's reply must be checked, never assumed: a security
        // policy such as XACE may deny GrabServer while still allowing
        // mapping requests, and a section that ran ungrabbed could
        // interleave its ownership read→write (or compare→restore)
        // with another client's remap of the same keycode. A rejected
        // grab is an error before any mapping read or write of the
        // section — never a fallback to an unprotected sequence.
        if let Err(error) = cookie.check() {
            return Err(grab_refused(error));
        }
        Ok(guard)
    }
}

impl Drop for ServerGrab<'_> {
    fn drop(&mut self) {
        if let Err(error) = self.session.conn.ungrab_server() {
            log::warn!("starling-insertion: releasing the server grab failed: {error}");
        }
        // Flush so the ungrab reaches the server now, not whenever
        // the next request happens to be queued.
        if let Err(error) = self.session.conn.flush() {
            log::warn!("starling-insertion: flushing the server-ungrab failed: {error}");
        }
    }
}

/// Take [`INSERT_LOCK`] for a whole insert. A panicked previous
/// holder poisons the lock; the state it protected (the keyboard
/// mapping) is server-global and repairable by the restore path, so
/// the next insert proceeds rather than every future insert failing.
fn insert_lock() -> MutexGuard<'static, ()> {
    INSERT_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
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

/// Where in its keystroke a character's typing failed: before the
/// character's own key went down (nothing of it can have landed — a
/// failed send, a failed Shift press, an unplanned plan), or after
/// the key-down was issued (the character may have landed even though
/// its keystroke — the release, the Shift release — did not complete).
/// The distinction is the delivery count: a character counts as
/// possibly delivered the moment its key-down was issued.
enum CharFailure {
    BeforeKeydown(InsertError),
    AfterKeydown(InsertError),
}

/// Type one segment (up to [`X11_CHUNK_CHARS`] characters), reporting
/// exactly how many characters may have landed if typing fails
/// part-way: every character whose key-down was issued counts,
/// including the one that failed mid-keystroke.
fn type_segment(
    session: &Session,
    keyboard: &Keyboard,
    remap: &RemapTransaction<'_>,
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
        match type_character(session, keyboard, remap, character, plan, &mut pressed) {
            Ok(()) => typed += 1,
            Err(CharFailure::BeforeKeydown(cause)) => {
                return Err(ChunkFailure {
                    delivered: delivered_before + typed,
                    cause,
                });
            }
            Err(CharFailure::AfterKeydown(cause)) => {
                // The character's key-down was issued: it may have
                // landed even though its keystroke failed, so it
                // counts — the report must never understate what the
                // target may hold.
                return Err(ChunkFailure {
                    delivered: delivered_before + typed + 1,
                    cause,
                });
            }
        }
    }
    Ok(typed)
}

/// Type one character through its plan: verify the keycode's live
/// mapping and run the *whole keystroke* — Shift press, key press,
/// the hold, both releases — inside one short server grab, so no
/// other client's remap can ride any part of the keystroke (X
/// translates KeyRelease events through the live mapping too, and
/// targets can act on a release); the inter-character gap follows
/// after the grab has dropped. The `pressed` guard guarantees the
/// releases even when a send fails mid-keystroke. Failures are
/// classified for the delivery count by [`CharFailure`]'s rule:
/// anything at or after the character's own key-down counts as
/// possibly delivered; a grab or verification failure happens
/// strictly before it.
fn type_character(
    session: &Session,
    keyboard: &Keyboard,
    remap: &RemapTransaction<'_>,
    character: char,
    plan: &CharPlan,
    pressed: &mut PressedKeys<'_>,
) -> Result<(), CharFailure> {
    let (keycode, needs_shift) = match plan {
        CharPlan::Pre {
            keycode,
            needs_shift,
        } => (*keycode, *needs_shift),
        CharPlan::Remapped { keycode } => (*keycode, false),
        CharPlan::NeedsKeysym(_) => {
            return Err(CharFailure::BeforeKeydown(InsertError::Rejected {
                reason: "internal error: an unplanned character reached the typer".to_string(),
            }))
        }
    };
    // A real Shift press (a key event the target sees, exactly like a
    // human typing `C`) rather than a synthetic modifier mask — XTest
    // only speaks keycodes — and only the modifier-mapping-bound Shift
    // keycode `find` verified before planning a column-1 character.
    // Even this planning invariant surfaces as an error rather than a
    // panic: the typer must not be able to crash the delivery path.
    let shift = if needs_shift {
        Some(keyboard.shift.ok_or_else(|| {
            CharFailure::BeforeKeydown(InsertError::Rejected {
                reason: "internal error: a shift-needing plan exists without a bound Shift \
                         keycode"
                    .to_string(),
            })
        })?)
    } else {
        None
    };
    // The grabbed verify→keystroke section: the chunk checks cannot
    // see a remap that lands while they run (the held-modifier wait
    // alone is seconds of exposure), so the keycode's live mapping is
    // re-read immediately before the key-down and the whole
    // keystroke is issued inside the same grab — no other X client
    // can interleave a remap between the read and the key-down, and
    // none between the key-down and the key-up either: Xlib
    // translates KeyRelease events through the live mapping just as
    // it does presses, and apps can act on a release, so a remap
    // landing mid-hold would make Starling's own release decode as
    // whatever the new mapping says — another client's `Return`,
    // say. The grab therefore spans verify → Shift↓ → key↓ → hold →
    // key↑ → Shift↑ → flush → ungrab: [`KEY_HOLD`] is the one sleep
    // ever taken while grabbed (it is what makes the
    // no-remap-mid-keystroke guarantee deterministic instead of a
    // race the interloper might lose), so a grab is held for roughly
    // KEY_HOLD plus a few requests per character — never across the
    // inter-character gap ([`KEY_GAP`]), a modifier wait or an event
    // read. A drift (or a refused grab) stops typing here, before
    // any key of this character goes down, with a keyboard-busy
    // refusal naming the keycode.
    {
        let _grab = ServerGrab::new(session).map_err(|error| match error {
            // Connection trouble stays "unavailable" — it is the
            // backend, not the keyboard, that failed.
            unavailable @ InsertError::Unavailable { .. } => {
                CharFailure::BeforeKeydown(unavailable)
            }
            cause => CharFailure::BeforeKeydown(InsertError::Rejected {
                reason: format!(
                    "the X server refused the exclusive grab re-verifying keycode {keycode} \
                     before its press, so U+{:04X} cannot be typed (keyboard busy): {}",
                    character as u32,
                    cause.message()
                ),
            }),
        })?;
        verify_press_mapping(session, remap, character, plan, keycode, shift)
            .map_err(CharFailure::BeforeKeydown)?;
        if let Some(shift) = shift {
            pressed.press(shift).map_err(CharFailure::BeforeKeydown)?;
        }
        // The key-down was issued (and flushed) under the verified
        // mapping: from here the character may have landed, whatever
        // happens to the rest of its keystroke.
        pressed.press(keycode).map_err(CharFailure::BeforeKeydown)?;
        std::thread::sleep(session.key_hold);
        pressed
            .release(keycode)
            .map_err(CharFailure::AfterKeydown)?;
        if let Some(shift) = shift {
            pressed.release(shift).map_err(CharFailure::AfterKeydown)?;
        }
        // Both releases flushed before the ungrab leaves this
        // connection, so the server lifts the grab only after it has
        // generated the release events (the requests of one
        // connection are processed in order, so this is belt and
        // braces over the per-event flushes `fake_key` already did).
        session
            .conn
            .flush()
            .map_err(x11_conn_error)
            .map_err(CharFailure::AfterKeydown)?;
    }
    // The inter-character gap: the deliberately ungrabbed part of
    // typing (a mapping that changes here is caught by the next
    // character's own verification — everything before it already
    // typed under a mapping this code verified).
    std::thread::sleep(KEY_GAP);
    Ok(())
}

/// The in-grab, pre-press comparison of the keycode a plan is about
/// to press against its live mapping — the guard that keeps a press
/// from ever riding a mapping another client changed after planning
/// (the chunk checks cannot see such a change; the held-modifier wait
/// alone is seconds of exposure). A borrowed keycode must still carry
/// exactly the recorded echo; a pre-mapped keycode must still produce
/// the planned keysym in the planned column, and the Shift keycode a
/// column-1 plan presses must still carry a Shift keysym in column 0
/// — the effective level of that very press, since Shift goes down
/// with no other modifier held — and remain bound to the Shift
/// modifier. Any drift is a keyboard-busy [`InsertError::Rejected`]
/// naming the keycode — never a press through a mapping Starling does
/// not own.
fn verify_press_mapping(
    session: &Session,
    remap: &RemapTransaction<'_>,
    character: char,
    plan: &CharPlan,
    keycode: Keycode,
    shift: Option<Keycode>,
) -> Result<(), InsertError> {
    let keysym = keysym_for_char(character);
    match plan {
        CharPlan::Remapped { .. } => {
            // The borrow's recorded echo is the baseline — the same
            // value a restore compares against, so "drifted" means
            // the same thing at the press and at cleanup. A keycode
            // pressed as borrowed with no recorded borrow is an
            // internal error: the typer must not be able to crash the
            // delivery path, but it may be stopped.
            let Some(expected) = remap.echoed_for(keycode) else {
                return Err(InsertError::Rejected {
                    reason: format!(
                        "internal error: keycode {keycode} is pressed as borrowed, but no \
                         borrow is recorded for it"
                    ),
                });
            };
            let live = session.keycode_syms(keycode)?;
            if live.as_slice() != expected {
                return Err(mapping_drifted_rejection(keycode, character, &live));
            }
        }
        CharPlan::Pre { needs_shift, .. } => {
            let live = session.keycode_syms(keycode)?;
            if live.get(usize::from(*needs_shift)) != Some(&keysym) {
                return Err(mapping_drifted_rejection(keycode, character, &live));
            }
            if let Some(shift) = shift {
                // The effective keysym of the press about to happen:
                // the Shift key goes down with no other modifier held,
                // so the level its own event decodes through is the
                // unshifted one, column 0. A Shift keysym somewhere in
                // the first two columns is not enough — `[Return,
                // Shift_L]` passes that check while still Shift-bound,
                // and the actual press then emits Return (the no-Enter
                // rule broken with Starling's own hand) — so column 0
                // itself must carry Shift_L or Shift_R.
                let shift_live = session.keycode_syms(shift)?;
                let effective = shift_live.first().copied();
                if effective != Some(XK_SHIFT_L) && effective != Some(XK_SHIFT_R) {
                    return Err(InsertError::Rejected {
                        reason: format!(
                            "keycode {shift}, the Shift key pressed for U+{:04X}, no longer \
                             carries a Shift keysym in its unshifted column (now {:?}), so the \
                             character cannot be typed (keyboard busy)",
                            character as u32, shift_live
                        ),
                    });
                }
                let still_bound = session
                    .modifier_rows()?
                    .first()
                    .is_some_and(|row| row.contains(&shift));
                if !still_bound {
                    return Err(InsertError::Rejected {
                        reason: format!(
                            "keycode {shift}, the Shift key pressed for U+{:04X}, is no longer \
                             bound to the Shift modifier, so the character cannot be typed \
                             (keyboard busy)",
                            character as u32
                        ),
                    });
                }
            }
        }
        CharPlan::NeedsKeysym(_) => {
            return Err(InsertError::Rejected {
                reason: "internal error: an unplanned character reached the typer".to_string(),
            });
        }
    }
    Ok(())
}

/// The keyboard-busy refusal for a keycode whose live mapping drifted
/// from the plan: typing stops before the key-down, the keycode is
/// named, and what the mapping now holds is stated — which is how a
/// foreign `Return` shows up in the report.
fn mapping_drifted_rejection(keycode: Keycode, character: char, live: &[Keysym]) -> InsertError {
    InsertError::Rejected {
        reason: format!(
            "keycode {keycode} no longer carries the mapping the insert planned for U+{:04X} \
             (another X client changed it while the text was being typed; it now holds {live:?}), \
             so the character cannot be typed (keyboard busy)",
            character as u32
        ),
    }
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

/// A checked server grab the server refused. A connection failure
/// means the backend is unavailable; an X11 error means the server
/// refused the exclusion the grabbed section needs (a security policy
/// such as XACE may deny GrabServer while still allowing mapping
/// requests) — a refusal the caller turns into the section's own
/// failure mode, never into an unprotected fallback.
fn grab_refused(error: x11rb::errors::ReplyError) -> InsertError {
    match error {
        x11rb::errors::ReplyError::ConnectionError(inner) => x11_conn_error(inner),
        x11rb::errors::ReplyError::X11Error(inner) => InsertError::Rejected {
            reason: format!("the X server refused the server grab: {inner:?}"),
        },
    }
}

/// Fold a failed delivery and a failed keyboard restore into the one
/// error an insert returns on that path. The restore failure is the
/// actionable headline (the user's keyboard may type wrong characters
/// until the layout is reloaded — `keyboard_restore_failed`), but its
/// message must also state the delivery outcome, so recovery sees
/// both facts: what may be in the target *and* what happened to the
/// keyboard. Neither error may hide the other.
fn combine_delivery_and_restore_failure(
    delivery: &InsertError,
    restore: InsertError,
) -> InsertError {
    let outcome = match delivery {
        InsertError::PartialDelivery {
            delivered_chars,
            total_chars,
            cause,
        } => format!(
            "up to {delivered_chars} of {total_chars} characters may have been typed before \
             the failure ({cause})"
        ),
        other => format!("no character's key-down was accepted before the failure ({other})"),
    };
    restore_with_outcome(outcome, restore)
}

/// The decision `RemapTransaction::begin` makes when borrowing fails
/// part-way: a *successful* restore changes nothing about the report
/// (the preparation error alone is the honest outcome — the keyboard
/// is exactly as it was), while a failed restore must not hide behind
/// the preparation error.
fn combine_preparation_with_restore_outcome(
    preparation: InsertError,
    restore: Result<(), InsertError>,
) -> InsertError {
    match restore {
        Ok(()) => preparation,
        Err(restore) => combine_preparation_and_restore_failure(&preparation, restore),
    }
}

/// Fold a preparation failure and a failed keyboard restore into the
/// one error `begin` returns on that path — the same discipline as
/// [`combine_delivery_and_restore_failure`]: the restore failure is
/// the actionable headline, and the message states both facts (the
/// original preparation cause *and* the delivery outcome). Here the
/// outcome is fixed: preparation precedes every key-down, so nothing
/// was typed — no "may have landed" needs reporting.
fn combine_preparation_and_restore_failure(
    preparation: &InsertError,
    restore: InsertError,
) -> InsertError {
    restore_with_outcome(
        format!("nothing was typed (the insert failed while preparing: {preparation})"),
        restore,
    )
}

/// The fold both combination helpers share: the restore failure stays
/// the headline error and the outcome sentence rides along in its
/// detail, so neither fact can hide the other.
fn restore_with_outcome(outcome: String, restore: InsertError) -> InsertError {
    match restore {
        InsertError::KeyboardRestoreFailed { detail } => InsertError::KeyboardRestoreFailed {
            detail: format!("{detail}; and {outcome}"),
        },
        // `finish` only produces `KeyboardRestoreFailed`; if that ever
        // changes, the restore error is still the one to return.
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_restore_failure_on_an_error_path_keeps_the_delivery_outcome() {
        let delivery = InsertError::PartialDelivery {
            delivered_chars: 16,
            total_chars: 20,
            cause: Box::new(InsertError::TargetChanged {
                expected: "x11:1:1".to_string(),
                actual: "x11:2:2".to_string(),
            }),
        };
        let combined = combine_delivery_and_restore_failure(
            &delivery,
            InsertError::KeyboardRestoreFailed {
                detail: "restoring keycode 255: the X connection failed".to_string(),
            },
        );
        // The restore failure is the headline (the actionable fact),
        // and both halves of the story survive in the message.
        assert_eq!(combined.code(), "keyboard_restore_failed");
        let message = combined.message();
        assert!(
            message.contains("could not be restored"),
            "the restore failure stays visible: {message}"
        );
        assert!(
            message.contains("keycode 255"),
            "the restore detail stays visible: {message}"
        );
        assert!(
            message.contains("up to 16 of 20 characters may have been typed"),
            "the delivery outcome stays visible: {message}"
        );

        // A delivery that failed before any key-down says that too.
        let bare = combine_delivery_and_restore_failure(
            &InsertError::TargetGone,
            InsertError::KeyboardRestoreFailed {
                detail: "re-reading keycode 255: the X connection failed".to_string(),
            },
        );
        let message = bare.message();
        assert!(
            message.contains("no character's key-down was accepted"),
            "the zero-delivery outcome is stated: {message}"
        );
        assert!(message.contains("could not be restored"));
    }

    #[test]
    fn a_preparation_failure_keeps_the_restore_failure_and_says_nothing_was_typed() {
        // Borrowing stopped part-way (the spare pool ran out) and the
        // restore of what *was* borrowed also failed: the restore
        // failure is the headline (the actionable fact), and both
        // halves of the story survive in the message — the preparation
        // cause and the fact that no key-down ever happened.
        let combined = combine_preparation_with_restore_outcome(
            InsertError::Rejected {
                reason: "every spare keycode is already in use, so the character U+1F984 \
                         cannot be typed (keyboard busy)"
                    .to_string(),
            },
            Err(InsertError::KeyboardRestoreFailed {
                detail: "restoring keycode 255: the X connection failed".to_string(),
            }),
        );
        assert_eq!(combined.code(), "keyboard_restore_failed");
        let message = combined.message();
        assert!(
            message.contains("could not be restored"),
            "the restore failure stays visible: {message}"
        );
        assert!(
            message.contains("keycode 255"),
            "the restore detail stays visible: {message}"
        );
        assert!(
            message.contains("every spare keycode is already in use"),
            "the preparation cause stays visible: {message}"
        );
        assert!(
            message.contains("nothing was typed"),
            "the zero-delivery outcome is stated: {message}"
        );
    }

    #[test]
    fn a_preparation_failure_with_a_clean_restore_surfaces_alone() {
        // The other arm of the decision: giving back every borrowed
        // keycode cleanly means the preparation error is the whole
        // story — no restore fact to fold in, no "nothing was typed"
        // rider on a failure the caller already understands.
        let combined = combine_preparation_with_restore_outcome(
            InsertError::Rejected {
                reason: "every spare keycode is already in use, so the character U+1F984 \
                         cannot be typed (keyboard busy)"
                    .to_string(),
            },
            Ok(()),
        );
        assert_eq!(combined.code(), "insertion_rejected");
        let message = combined.message();
        assert!(
            message.contains("every spare keycode is already in use"),
            "the preparation cause is the report: {message}"
        );
        assert!(
            !message.contains("could not be restored"),
            "a clean restore adds no restore failure: {message}"
        );
    }

    #[test]
    fn a_keyed_down_character_counts_in_the_chunk_failure_delivered() {
        // The X11 typer's counting rule, pinned at the seam the chunk
        // loop sees: a character whose key-down was issued counts as
        // delivered even when its keystroke failed — so the first
        // segment failing *after* its first key-down reports 1, which
        // `deliver_in_chunks` must wrap as a partial delivery (the
        // bare cause would claim nothing landed).
        let error = crate::deliver_in_chunks(
            3,
            &["abc"],
            || Ok(()),
            |_, _| {
                Err(ChunkFailure {
                    delivered: 0 + 0 + 1, // first char keyed down, keystroke failed
                    cause: InsertError::Unavailable {
                        reason: "the X connection failed".to_string(),
                        setup_hint: None,
                    },
                })
            },
        )
        .unwrap_err();
        match error {
            InsertError::PartialDelivery {
                delivered_chars,
                total_chars,
                cause,
            } => {
                assert_eq!(delivered_chars, 1, "the keyed-down character counts");
                assert_eq!(total_chars, 3);
                assert_eq!(cause.code(), "insertion_unavailable");
            }
            other => {
                panic!("a keyed-down character means partial delivery, not a bare cause: {other:?}")
            }
        }
    }
}
