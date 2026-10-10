//! The Wayland backend: synthetic typing through the
//! `zwp_virtual_keyboard_v1` protocol (what `wtype` uses), offered by
//! wlroots compositors, niri and other Smithay-based compositors. GNOME
//! and KDE do not offer it.
//!
//! What it cannot do, by the protocol's design:
//!
//! - **No target identity.** A Wayland client cannot see which surface has
//!   keyboard focus, so a capture records only "whatever is focused" and
//!   `revalidate` cannot notice a change: it always answers `Same`.
//!   [`InsertionBackend::verifies_target`] is `false`, and every caller
//!   must gate on it (the runtime bridge refuses such refs; the app asks
//!   for an explicit opt-in and checks its own window instead).
//! - **No held-modifier check.** The physical keyboard's modifiers cannot
//!   be read. The virtual keyboard sends its own (none); whether the
//!   compositor combines them with physically held keys is up to it.
//! - **No excluded-pid check.** The owner of the focused surface is
//!   unknown; only the caller can tell its own window has focus.
//!
//! Each insert uploads a keymap holding exactly the characters it types
//! (one keycode per distinct character, keysyms by value), so typing
//! never depends on the user's layout and Caps Lock cannot change case.
//! Control characters are refused before anything is sent, so Enter and
//! Tab are never typed.

use std::io::Write;
use std::os::fd::AsFd;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Mutex, PoisonError};
use std::time::Instant;

use wayland_client::globals::{registry_queue_init, GlobalListContents};
use wayland_client::protocol::{wl_keyboard, wl_registry, wl_seat};
use wayland_client::{delegate_noop, Connection, Dispatch, EventQueue, QueueHandle};

use crate::{
    format_ref, insertion_guards, merge_excluded_pids, BackendKind, InsertError, InsertReceipt,
    InsertionBackend, TargetCheck, TargetSnapshot, EVIDENCE_SYNTHETIC_KEYS,
};

#[allow(non_upper_case_globals, non_camel_case_types, missing_docs, clippy::all)]
mod protocol {
    use wayland_client;
    use wayland_client::protocol::*;

    pub mod __interfaces {
        use wayland_client::backend as wayland_backend;
        use wayland_client::protocol::__interfaces::*;
        wayland_scanner::generate_interfaces!("protocols/virtual-keyboard-unstable-v1.xml");
    }
    use self::__interfaces::*;

    wayland_scanner::generate_client_code!("protocols/virtual-keyboard-unstable-v1.xml");
}

use protocol::zwp_virtual_keyboard_manager_v1::ZwpVirtualKeyboardManagerV1;
use protocol::zwp_virtual_keyboard_v1::ZwpVirtualKeyboardV1;

/// Distinct characters per uploaded keymap; a text with more is typed in
/// segments, each with its own keymap. Well under the 255 keycodes every
/// client accepts.
const KEYMAP_CHARS: usize = 200;
/// The first keycode used, in evdev numbering (XKB adds 8).
const FIRST_EVDEV_KEY: u32 = 1;
const KEY_PRESSED: u32 = 1;
const KEY_RELEASED: u32 = 0;

/// Mints a fresh `wl:` ref per capture; it carries no identity.
static CAPTURES: AtomicU32 = AtomicU32::new(0);
/// Serializes this process's inserts: two virtual keyboards typing at
/// once would interleave their characters in the target.
static INSERT_LOCK: Mutex<()> = Mutex::new(());

#[derive(Debug)]
pub struct WaylandBackend {
    excluded_pids: Vec<u32>,
    epoch: Instant,
}

impl Default for WaylandBackend {
    fn default() -> Self {
        WaylandBackend::new()
    }
}

impl WaylandBackend {
    pub fn new() -> WaylandBackend {
        WaylandBackend::with_excluded_pids(Vec::new())
    }

    /// The pids are kept for the snapshot guard only: the protocol cannot
    /// name the focused surface's owner.
    pub fn with_excluded_pids(excluded_pids: Vec<u32>) -> WaylandBackend {
        WaylandBackend {
            excluded_pids: merge_excluded_pids(excluded_pids),
            epoch: Instant::now(),
        }
    }

    /// Milliseconds since the backend was made, for key timestamps.
    fn now_ms(&self) -> u32 {
        self.epoch.elapsed().as_millis() as u32
    }

    fn type_segment(
        &self,
        session: &mut Session,
        keyboard: &ZwpVirtualKeyboardV1,
        segment: &[char],
        delivered: &mut usize,
    ) -> Result<(), InsertError> {
        let mut distinct: Vec<char> = Vec::new();
        for character in segment {
            if !distinct.contains(character) {
                distinct.push(*character);
            }
        }
        let keymap = keymap_for(&distinct);
        let file = keymap_file(&keymap)?;
        keyboard.keymap(
            wl_keyboard::KeymapFormat::XkbV1.into(),
            file.as_fd(),
            keymap.len() as u32 + 1,
        );
        keyboard.modifiers(0, 0, 0, 0);
        session.roundtrip()?;
        for character in segment {
            let index = distinct
                .iter()
                .position(|known| known == character)
                .expect("every segment character is in its keymap") as u32;
            let key = FIRST_EVDEV_KEY + index;
            keyboard.key(self.now_ms(), key, KEY_PRESSED);
            // Counted once its press went out: from here on it may land.
            *delivered += 1;
            keyboard.key(self.now_ms(), key, KEY_RELEASED);
            session.roundtrip()?;
        }
        Ok(())
    }
}

impl InsertionBackend for WaylandBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Wayland
    }

    fn verifies_target(&self) -> bool {
        false
    }

    fn availability(&self) -> Result<(), InsertError> {
        Session::open().map(drop)
    }

    /// Only checks the protocol is there: what is focused cannot be read.
    fn capture(&self) -> Result<TargetSnapshot, InsertError> {
        Session::open()?;
        let capture = CAPTURES.fetch_add(1, Ordering::Relaxed).wrapping_add(1).max(1);
        Ok(TargetSnapshot {
            backend: BackendKind::Wayland,
            target_ref: format_ref(BackendKind::Wayland, capture, 0, None),
            app: None,
            title: None,
            pid: None,
        })
    }

    /// Always `Same`: the protocol cannot see focus. See the module docs.
    fn revalidate(&self, _target: &TargetSnapshot) -> Result<TargetCheck, InsertError> {
        Ok(TargetCheck::Same)
    }

    fn insert(&self, target: &TargetSnapshot, text: &str) -> Result<InsertReceipt, InsertError> {
        insertion_guards(text, target.pid, &self.excluded_pids)?;
        let _insert_lock = INSERT_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
        let mut session = Session::open()?;
        let keyboard = session.manager.create_virtual_keyboard(
            &session.seat,
            &session.queue.handle(),
            (),
        );
        let characters: Vec<char> = text.chars().collect();
        let mut delivered = 0;
        let mut typed = Ok(());
        for segment in distinct_segments(&characters, KEYMAP_CHARS) {
            typed = self.type_segment(&mut session, &keyboard, segment, &mut delivered);
            if typed.is_err() {
                break;
            }
        }
        keyboard.destroy();
        let _ = session.roundtrip();
        match typed {
            Ok(()) => Ok(InsertReceipt {
                evidence: EVIDENCE_SYNTHETIC_KEYS,
            }),
            Err(cause) if delivered == 0 => Err(cause),
            Err(cause) => Err(InsertError::PartialDelivery {
                delivered_chars: delivered,
                total_chars: characters.len(),
                cause: Box::new(cause),
            }),
        }
    }
}

/// One connection with the globals an insert needs.
struct Session {
    queue: EventQueue<State>,
    state: State,
    manager: ZwpVirtualKeyboardManagerV1,
    seat: wl_seat::WlSeat,
}

struct State;

impl Session {
    fn open() -> Result<Session, InsertError> {
        let unavailable = |reason: String| InsertError::Unavailable { reason };
        if std::env::var_os("WAYLAND_DISPLAY").is_none() {
            return Err(unavailable(
                "not a Wayland session (WAYLAND_DISPLAY is unset)".to_string(),
            ));
        }
        let connection = Connection::connect_to_env()
            .map_err(|err| unavailable(format!("cannot connect to the compositor: {err}")))?;
        let (globals, queue) = registry_queue_init::<State>(&connection)
            .map_err(|err| unavailable(format!("cannot read the compositor's globals: {err}")))?;
        let handle = queue.handle();
        let manager = globals
            .bind::<ZwpVirtualKeyboardManagerV1, _, _>(&handle, 1..=1, ())
            .map_err(|_| {
                unavailable(
                    "the compositor does not offer virtual keyboards \
                     (zwp_virtual_keyboard_manager_v1)"
                        .to_string(),
                )
            })?;
        let seat = globals
            .bind::<wl_seat::WlSeat, _, _>(&handle, 1..=1, ())
            .map_err(|_| unavailable("the compositor offers no seat".to_string()))?;
        Ok(Session {
            queue,
            state: State,
            manager,
            seat,
        })
    }

    /// Waits until the compositor processed everything sent so far. A
    /// refusal (an unauthorized client, a bad keymap) arrives here as a
    /// protocol error.
    fn roundtrip(&mut self) -> Result<(), InsertError> {
        self.queue
            .roundtrip(&mut self.state)
            .map(drop)
            .map_err(|err| InsertError::Rejected {
                reason: format!("the compositor refused the virtual keyboard: {err}"),
            })
    }
}

impl Dispatch<wl_registry::WlRegistry, GlobalListContents> for State {
    fn event(
        _: &mut Self,
        _: &wl_registry::WlRegistry,
        _: wl_registry::Event,
        _: &GlobalListContents,
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}

delegate_noop!(State: ignore wl_seat::WlSeat);
delegate_noop!(State: ZwpVirtualKeyboardManagerV1);
delegate_noop!(State: ZwpVirtualKeyboardV1);

/// The keysym typing `character`: Latin-1 printables are their own
/// keysym, everything else the Unicode keysym range.
fn keysym(character: char) -> u32 {
    let code = character as u32;
    if (0x20..=0x7e).contains(&code) || (0xa0..=0xff).contains(&code) {
        code
    } else {
        0x0100_0000 + code
    }
}

/// An XKB keymap with one single-level key per character, keycodes from
/// `FIRST_EVDEV_KEY + 8` in `characters` order.
fn keymap_for(characters: &[char]) -> String {
    let first = FIRST_EVDEV_KEY + 8;
    let last = first + characters.len().max(1) as u32 - 1;
    let mut keycodes = String::new();
    let mut symbols = String::new();
    for (index, character) in characters.iter().enumerate() {
        let keycode = first + index as u32;
        keycodes.push_str(&format!("<K{index}> = {keycode};\n"));
        symbols.push_str(&format!("key <K{index}> {{[ 0x{:x} ]}};\n", keysym(*character)));
    }
    format!(
        "xkb_keymap {{\n\
         xkb_keycodes \"starling\" {{\nminimum = 8;\nmaximum = {last};\n{keycodes}}};\n\
         xkb_types \"starling\" {{ include \"complete\" }};\n\
         xkb_compatibility \"starling\" {{ include \"complete\" }};\n\
         xkb_symbols \"starling\" {{\n{symbols}}};\n\
         }};\n"
    )
}

/// The keymap in an unlinked file the compositor can map, NUL-terminated
/// as `wl_keyboard.keymap` expects.
fn keymap_file(keymap: &str) -> Result<std::fs::File, InsertError> {
    let failed = |err: std::io::Error| InsertError::Rejected {
        reason: format!("cannot write the keymap for the virtual keyboard: {err}"),
    };
    let dir = std::env::var_os("XDG_RUNTIME_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    static FILES: AtomicU32 = AtomicU32::new(0);
    let path = dir.join(format!(
        "starling-keymap-{}-{}",
        std::process::id(),
        FILES.fetch_add(1, Ordering::Relaxed)
    ));
    let mut file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(&path)
        .map_err(failed)?;
    let _ = std::fs::remove_file(&path);
    file.write_all(keymap.as_bytes()).map_err(failed)?;
    file.write_all(&[0]).map_err(failed)?;
    file.flush().map_err(failed)?;
    Ok(file)
}

/// Splits `characters` so each segment has at most `max_distinct`
/// distinct characters.
fn distinct_segments(characters: &[char], max_distinct: usize) -> Vec<&[char]> {
    let mut segments = Vec::new();
    let mut start = 0;
    let mut seen: Vec<char> = Vec::new();
    for (index, character) in characters.iter().enumerate() {
        if !seen.contains(character) {
            if seen.len() == max_distinct {
                segments.push(&characters[start..index]);
                start = index;
                seen.clear();
            }
            seen.push(*character);
        }
    }
    if start < characters.len() {
        segments.push(&characters[start..]);
    }
    segments
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keysyms_use_latin1_values_and_the_unicode_range() {
        assert_eq!(keysym('a'), 0x61);
        assert_eq!(keysym(' '), 0x20);
        assert_eq!(keysym('é'), 0xe9);
        assert_eq!(keysym('€'), 0x0100_20ac);
        assert_eq!(keysym('😀'), 0x0101_f600);
    }

    #[test]
    fn the_keymap_has_one_key_per_character() {
        let keymap = keymap_for(&['H', 'i', '😀']);
        assert!(keymap.contains("maximum = 11;"), "{keymap}");
        assert!(keymap.contains("<K0> = 9;"), "{keymap}");
        assert!(keymap.contains("<K2> = 11;"), "{keymap}");
        assert!(keymap.contains("key <K0> {[ 0x48 ]};"), "{keymap}");
        assert!(keymap.contains("key <K2> {[ 0x101f600 ]};"), "{keymap}");
    }

    #[test]
    fn segments_cap_distinct_characters_not_length() {
        let text: Vec<char> = "abcabcabcd".chars().collect();
        let segments = distinct_segments(&text, 3);
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].iter().collect::<String>(), "abcabcabc");
        assert_eq!(segments[1].iter().collect::<String>(), "d");
        assert!(distinct_segments(&[], 3).is_empty());
    }

    #[test]
    fn captures_need_the_protocol_and_never_verify() {
        let backend = WaylandBackend::new();
        assert!(!backend.verifies_target());
        let snapshot = TargetSnapshot {
            backend: BackendKind::Wayland,
            target_ref: format_ref(BackendKind::Wayland, 1, 0, None),
            app: None,
            title: None,
            pid: None,
        };
        assert_eq!(backend.revalidate(&snapshot), Ok(TargetCheck::Same));
        assert_eq!(
            backend.insert(&snapshot, "two\nlines"),
            Err(InsertError::MultilineUnsupported)
        );
    }
}
