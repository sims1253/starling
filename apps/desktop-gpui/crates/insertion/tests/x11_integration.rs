//! The X11 backend against a real X server. Runs only with
//! `STARLING_X11_IT=1` and `DISPLAY` set (and `WAYLAND_DISPLAY` unset,
//! where the backend refuses by design); a private Xvfb is enough.
//!
//! The test window's key events are decoded while the insert runs,
//! refreshing the keymap at every `MappingNotify` like a target that keeps
//! up; decoding afterwards would read the restored mapping and hide remap
//! bugs. After every round the live keyboard mapping must be exactly what
//! it was before.
//!
//! Rounds:
//! 1. capture identity (class, title, no invented pid);
//! 2. a mixed string through layout keys, Shift and one borrowed spare;
//! 3. an astral character, then an uppercase letter (case kept), through
//!    borrowed spares; 3c. one more unmapped character than there are
//!    spares refuses and restores what it borrowed;
//! 4. a held Ctrl refuses after the bounded wait; 4b. a key remapped to
//!    Return during that wait (a borrowed spare, then a layout letter
//!    after one delivered character) stops typing before its key-down;
//!    4c. a remap of the pressed key during its hold is held back by the
//!    keystroke grab, where the server honors grabs; 4d. a Shift key
//!    remapped to `[Return, Shift_L]` is never pressed; 4e. a Shift held
//!    or Caps Lock tapped mid-chunk stops typing at the next character;
//! 5. two concurrent inserts land whole and in order; a focus steal to a
//!    window of an excluded process mid-chunk stops at the next character,
//!    and that window receives no key;
//! 6. a changed target refuses and a destroyed one reports `Gone`;
//! 7. a newline is refused before any key moves.
//!
//! Caps Lock is unlocked if needed, and the "unmapped" characters are
//! picked at runtime, so a keymap dirtied by an earlier run cannot turn a
//! remap round into a layout-key one.

#![cfg(target_os = "linux")]

use std::fmt::Display;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use x11rb::connection::Connection;
use x11rb::protocol::xproto::{
    Atom, ConnectionExt, CreateWindowAux, EventMask, InputFocus, MapState, PropMode, Window,
    WindowClass,
};
use x11rb::protocol::Event;
use x11rb::rust_connection::RustConnection;
use x11rb::wrapper::ConnectionExt as _;

use starling_insertion::x11::X11Backend;
use starling_insertion::{
    BackendKind, InsertError, InsertReceipt, Inserter, InsertionBackend, TargetCheck,
    TargetSnapshot, EVIDENCE_SYNTHETIC_KEYS, MODIFIER_RELEASE_WAIT,
};

const SETTLE: Duration = Duration::from_secs(5);
/// Lets events the server already queued reach this connection.
const DRAIN_GRACE: Duration = Duration::from_millis(150);
const INSERT_BUDGET: Duration = Duration::from_secs(30);

const MIXED_HEAD: &str = "Café ß ";
const MIXED_TAIL: &str = " → 42!";
const MIXED_REMAP_CANDIDATES: &[char] = &['ŋ', 'ƍ', 'Ƃ', 'ǅ', 'ƞ'];
const UNMAPPED_CANDIDATES: &[char] = &['🦄', '🥑', '🛰', '𝄞'];
/// Uppercase letters with a lowercase twin, so a single-column borrow
/// would be expanded into a case pair.
const UPPER_UNMAPPED_CANDIDATES: &[char] = &['Ŋ', 'Ƕ', 'Ǯ', 'Ǥ', 'Ɣ'];
const XK_CONTROL_L: u32 = 0xffe3;
const XK_CONTROL_R: u32 = 0xffe4;
const XK_CAPS_LOCK: u32 = 0xffe5;
const XK_RETURN: u32 = 0xff0d;
const XK_KP_ENTER: u32 = 0xff8b;
const XK_SHIFT_L: u32 = 0xffe1;
const XK_SHIFT_R: u32 = 0xffe2;
const USE_CORE_KEYBOARD: u16 = 0x100;

type ItResult<T> = Result<T, String>;
type Inserted = Result<InsertReceipt, InsertError>;

fn err(error: impl Display) -> String {
    error.to_string()
}

#[test]
fn x11_end_to_end_capture_type_revalidate() {
    let enabled = std::env::var("STARLING_X11_IT").ok().as_deref() == Some("1")
        && std::env::var_os("DISPLAY").is_some_and(|display| !display.is_empty());
    if !enabled {
        eprintln!("skipped: set DISPLAY and STARLING_X11_IT=1 to run the X11 integration test");
        return;
    }
    if let Err(error) = main_test() {
        panic!("X11 integration test failed: {error}");
    }
}

fn main_test() -> ItResult<()> {
    let (conn, screen) = x11rb::connect(None).map_err(err)?;
    let root = conn.setup().roots[screen].root;
    let atoms = Atoms::intern(&conn)?;
    let mut kb = Keyboard::load(&conn)?;
    let mut cleanup = Cleanup {
        conn: &conn,
        windows: Vec::new(),
        mappings: Vec::new(),
    };

    let a = make_window(&conn, root, &atoms, "first", true)?;
    cleanup.windows.push(a);
    focus_and_wait(&conn, root, &atoms, a)?;

    // é, ß and → on spare keycodes, so the layout-key path meets them.
    let spares = kb
        .spare_keycodes(3)
        .ok_or("this keyboard has no three spare keycodes")?;
    for (&keycode, character) in spares.iter().zip(['é', 'ß', '→']) {
        let mut columns = vec![0; kb.width];
        columns[0] = keysym_of(character);
        set_keysyms(&conn, keycode, &columns)?;
        cleanup.mappings.push((keycode, vec![0; kb.width]));
    }
    // The server canonicalizes those writes (adding case twins), so read
    // them back rather than trusting the local copy.
    kb.reload(&conn)?;
    let mixed_remap = unmapped_char(&kb, MIXED_REMAP_CANDIDATES);
    let unmapped = unmapped_char(&kb, UNMAPPED_CANDIDATES);
    let upper = unmapped_char(&kb, UPPER_UNMAPPED_CANDIDATES);
    let mixed = format!("{MIXED_HEAD}{mixed_remap}{MIXED_TAIL}");

    normalize_keyboard_state(&conn, &kb)?;
    let mapping_before = kb.syms.clone();
    let inserter = Inserter::with_backends(vec![Box::new(X11Backend::new())]);

    // 1. Capture.
    let snap_a = inserter.capture().map_err(err)?;
    assert_eq!(snap_a.backend, BackendKind::X11);
    assert_eq!(snap_a.app.as_deref(), Some("StarlingIt"));
    assert_eq!(
        snap_a.title.as_deref(),
        Some("starling x11 IT window (first)")
    );
    assert_eq!(snap_a.pid, None, "the test window sets no _NET_WM_PID");
    assert_eq!(snap_a.ids(), Some((a, a, None)));
    let backend = inserter.backend_for(&snap_a).expect("the X11 backend");

    // 7. A newline is refused before any key moves.
    assert_eq!(
        backend.insert(&snap_a, "line\nbreak"),
        Err(InsertError::MultilineUnsupported)
    );
    assert_eq!(drain_text(&conn, a, &mut kb)?, "");

    // 2, 3, 3b. Text through layout keys, Shift and borrowed spares.
    for text in [
        mixed.clone(),
        unmapped.to_string(),
        format!("a{upper}b{upper}"),
    ] {
        let handle = spawn_insert(X11Backend::new(), &snap_a, &text);
        let (results, seen) = listen(&conn, a, &mut kb, vec![handle], |_| Ok(()))?;
        let receipt = results[0]
            .clone()
            .map_err(|e| format!("inserting {text:?}: {e}"))?;
        assert_eq!(receipt.evidence, EVIDENCE_SYNTHETIC_KEYS);
        assert_eq!(
            text_of(&seen),
            text,
            "the window decodes the text it was sent"
        );
        kb.assert_mapping_is(&conn, &mapping_before)?;
    }

    // 3c. Borrowing runs out part-way: refused, nothing typed, restored.
    let spares_left = kb.spare_count();
    let overflow: String = unmapped_chars(&kb, spares_left + 1).into_iter().collect();
    match backend.insert(&snap_a, &overflow) {
        Err(InsertError::Rejected { reason }) => {
            assert!(reason.contains("keyboard busy"), "{reason}")
        }
        other => panic!("an exhausted spare pool must refuse: {other:?}"),
    }
    assert_eq!(drain_text(&conn, a, &mut kb)?, "");
    kb.assert_mapping_is(&conn, &mapping_before)?;

    // 4. A held Ctrl refuses after the bounded wait and is not released.
    let control = kb
        .keycode_of(XK_CONTROL_L)
        .or_else(|| kb.keycode_of(XK_CONTROL_R))
        .ok_or("this keyboard has no Control key")?;
    let control_keysym = kb.syms_of(control)[0];
    xtest_key(&conn, control, true)?;
    let started = Instant::now();
    match backend.insert(&snap_a, "blocked") {
        Err(InsertError::ModifiersHeld { held }) => {
            assert!(held.iter().any(|name| name.contains("Control")), "{held:?}")
        }
        other => panic!("held modifiers must refuse: {other:?}"),
    }
    assert!(started.elapsed() >= MODIFIER_RELEASE_WAIT);
    xtest_key(&conn, control, false)?;
    assert_eq!(drain_text(&conn, a, &mut kb)?, "");

    // 4b. While the insert waits out a held Ctrl, a key it is about to
    // press is remapped to Return; then Ctrl is released. The predicted
    // borrow is the highest spare, the backend's own choice.
    let borrowed = kb.spare_keycodes(1).ok_or("no spare keycode")?[0];
    let letter = ['q', 'x', 'z', 'v', 'k', 'j']
        .into_iter()
        .find(|&c| {
            kb.keycode_of(keysym_of(c))
                .is_some_and(|k| kb.syms_of(k)[0] == keysym_of(c))
        })
        .ok_or("this keyboard has no plain letter to remap")?;
    let letter_keycode = kb.keycode_of(keysym_of(letter)).expect("found above");
    let letter_original = kb.syms_of(letter_keycode).to_vec();
    let shift_keycode = shift_keycode_of(&conn, &kb)?;
    let shift_original = kb.syms_of(shift_keycode).to_vec();
    cleanup.mappings.push((borrowed, vec![0; kb.width]));
    cleanup
        .mappings
        .push((letter_keycode, letter_original.clone()));
    cleanup
        .mappings
        .push((shift_keycode, shift_original.clone()));
    let all_return = vec![XK_RETURN; kb.width];
    // (remapped keycode, its new columns, payload, delivered before the
    // stop, original columns)
    let drift_rounds: [(u8, &[u32], String, usize, &[u32]); 3] = [
        (
            borrowed,
            &all_return,
            unmapped.to_string(),
            0,
            &vec![0; kb.width],
        ),
        (
            letter_keycode,
            &all_return,
            format!("{unmapped}{letter}"),
            1,
            &letter_original,
        ),
        // 4d. Exactly two columns: a write padded with zeros widens this
        // server's global keymap and would shift every echo.
        (
            shift_keycode,
            &[XK_RETURN, XK_SHIFT_L],
            format!("{unmapped}C"),
            1,
            &shift_original,
        ),
    ];
    for (keycode, columns, payload, delivered, original) in drift_rounds {
        xtest_key(&conn, control, true)?;
        let handle = spawn_insert(X11Backend::new(), &snap_a, &payload);
        let mut interfered = false;
        let (results, seen) = listen(&conn, a, &mut kb, vec![handle], |_| {
            // The borrow is complete once its keysym is visible, and the
            // insert is parked in the modifier wait.
            if !interfered && live_keysyms(&conn, borrowed)?.first() == Some(&keysym_of(unmapped)) {
                interfered = true;
                set_keysyms(&conn, keycode, columns)?;
                xtest_key(&conn, control, false)?;
            }
            Ok(())
        })?;
        let reason = match results[0].clone() {
            Err(InsertError::Rejected { reason }) if delivered == 0 => reason,
            Err(InsertError::PartialDelivery {
                delivered_chars,
                cause,
                ..
            }) if delivered_chars == delivered => match *cause {
                InsertError::Rejected { reason } => reason,
                other => panic!("the remap must be the stop cause: {other:?}"),
            },
            other => panic!("remapped keycode {keycode} must stop typing: {other:?}"),
        };
        assert!(reason.contains("keyboard busy"), "{reason}");
        assert!(reason.contains(&format!("keycode {keycode}")), "{reason}");
        let mut expected = vec![control_keysym];
        expected.extend(payload.chars().take(delivered).map(keysym_of));
        assert_eq!(
            pressed_keysyms(&seen),
            expected,
            "no Return reached the window"
        );
        kb.reload(&conn)?;
        assert_eq!(
            kb.syms_of(keycode)[0],
            XK_RETURN,
            "the other client's mapping is left alone"
        );
        set_keysyms(&conn, keycode, original)?;
        kb.assert_mapping_is(&conn, &mapping_before)?;
    }

    // 4c. The pressed key is remapped to Return during its hold (made
    // long so the remap lands inside it). A server that honors GrabServer
    // processes the remap only after the key-up.
    let grab_excluded = grab_exclusion_honored(&conn)?;
    if !grab_excluded {
        println!(
            "this X server does not hold other clients back during GrabServer (WSLg's \
             XWayland); 4c checks only the press side"
        );
    }
    let char_keysym = keysym_of(unmapped);
    let handle = spawn_insert(
        X11Backend::new().with_key_hold_for_tests(Duration::from_millis(250)),
        &snap_a,
        &unmapped.to_string(),
    );
    let mut interfered = false;
    let (results, seen) = listen(&conn, a, &mut kb, vec![handle], |seen| {
        if !interfered && seen.contains(&Seen::Press(borrowed, char_keysym)) {
            interfered = true;
            set_keysyms(&conn, borrowed, &all_return)?;
        }
        Ok(())
    })?;
    assert!(interfered, "the remap fired during the hold: {seen:?}");
    assert_eq!(
        results[0].clone().map_err(err)?.evidence,
        EVIDENCE_SYNTHETIC_KEYS
    );
    let press = seen
        .iter()
        .position(|s| *s == Seen::Press(borrowed, char_keysym))
        .expect("checked above");
    let release = seen
        .iter()
        .position(|s| matches!(s, Seen::Release(keycode, _) if *keycode == borrowed))
        .unwrap_or_else(|| panic!("the key came back up: {seen:?}"));
    assert!(press < release, "{seen:?}");
    assert!(
        !pressed_keysyms(&seen).iter().any(|k| is_enter(*k)),
        "{seen:?}"
    );
    if grab_excluded {
        assert_eq!(
            seen[release],
            Seen::Release(borrowed, char_keysym),
            "{seen:?}"
        );
        assert!(!seen[press..release].contains(&Seen::Map), "{seen:?}");
        assert!(seen[release..].contains(&Seen::Map), "{seen:?}");
        assert!(
            !seen
                .iter()
                .any(|s| matches!(s, Seen::Press(_, k) | Seen::Release(_, k) if is_enter(*k))),
            "{seen:?}"
        );
    }
    kb.reload(&conn)?;
    assert_eq!(kb.syms_of(borrowed)[0], XK_RETURN);
    set_keysyms(&conn, borrowed, &vec![0; kb.width])?;
    kb.assert_mapping_is(&conn, &mapping_before)?;

    // 4e. The user holds Shift, or taps Caps Lock, after the first
    // character: typing stops before the next one, so no character comes
    // out in the wrong case.
    let caps = kb
        .keycode_of(XK_CAPS_LOCK)
        .ok_or("this keyboard has no Caps Lock")?;
    let lower = "abcdefghijkl";
    for (key, tap) in [(shift_keycode, false), (caps, true)] {
        let handle = spawn_insert(X11Backend::new(), &snap_a, lower);
        let mut fired = false;
        let (results, seen) = listen(&conn, a, &mut kb, vec![handle], |seen| {
            if !fired && !text_of(seen).is_empty() {
                fired = true;
                xtest_key(&conn, key, true)?;
                if tap {
                    xtest_key(&conn, key, false)?;
                }
            }
            Ok(())
        })?;
        if tap {
            normalize_keyboard_state(&conn, &kb)?;
        } else {
            xtest_key(&conn, key, false)?;
        }
        drain_text(&conn, a, &mut kb)?;
        match results[0].clone() {
            Err(InsertError::PartialDelivery {
                delivered_chars,
                cause,
                ..
            }) => {
                assert!(delivered_chars < lower.len());
                assert_eq!(text_of(&seen), lower[..delivered_chars]);
                match *cause {
                    InsertError::ModifiersHeld { held } if !tap => {
                        assert_eq!(held, ["Shift"])
                    }
                    InsertError::ModifiersHeld { .. }
                    | InsertError::KeyboardStateUnsupported { .. }
                        if tap => {}
                    other => panic!("unexpected stop cause: {other:?}"),
                }
            }
            other => panic!("a mid-chunk keyboard change must stop typing: {other:?}"),
        }
    }

    // 5. Two concurrent inserts are serialized.
    let handles = (0..2)
        .map(|_| spawn_insert(X11Backend::new(), &snap_a, &mixed))
        .collect();
    let (results, seen) = listen(&conn, a, &mut kb, handles, |_| Ok(()))?;
    for result in results {
        assert_eq!(result.map_err(err)?.evidence, EVIDENCE_SYNTHETIC_KEYS);
    }
    assert_eq!(text_of(&seen), format!("{mixed}{mixed}"));
    kb.assert_mapping_is(&conn, &mapping_before)?;

    // 5. A window B of an excluded process (on its own connection, like
    // another client) takes the focus after the first character: typing
    // stops before the next key, so A holds exactly the delivered prefix
    // and B receives nothing.
    let (conn_b, _) = x11rb::connect(None).map_err(err)?;
    let b = make_window(&conn_b, root, &atoms, "second", false)?;
    cleanup.windows.push(b);
    let excluded_pid = 4_000_001;
    conn_b
        .change_property32(
            PropMode::REPLACE,
            b,
            atoms.net_wm_pid,
            atoms.cardinal,
            &[excluded_pid],
        )
        .map_err(err)?
        .check()
        .map_err(err)?;
    let long = format!("{mixed} ").repeat(8);
    let handle = spawn_insert(
        X11Backend::with_excluded_pids(vec![excluded_pid]),
        &snap_a,
        &long,
    );
    let mut stolen = false;
    let (results, seen) = listen(&conn, a, &mut kb, vec![handle], |seen| {
        if !stolen && !text_of(seen).is_empty() {
            stolen = true;
            conn.map_window(b).map_err(err)?.check().map_err(err)?;
            wait_viewable(&conn, b)?;
            conn.set_input_focus(InputFocus::PARENT, b, x11rb::CURRENT_TIME)
                .map_err(err)?
                .check()
                .map_err(err)?;
        }
        Ok(())
    })?;
    match results[0].clone() {
        Err(InsertError::PartialDelivery {
            delivered_chars,
            total_chars,
            cause,
        }) => {
            assert!(0 < delivered_chars && delivered_chars < total_chars);
            assert!(
                matches!(*cause, InsertError::TargetChanged { .. }),
                "{cause:?}"
            );
            let prefix: String = long.chars().take(delivered_chars).collect();
            assert_eq!(
                text_of(&seen),
                prefix,
                "A holds exactly the delivered prefix"
            );
        }
        other => panic!("a focus steal must stop the insert part-way: {other:?}"),
    }
    std::thread::sleep(DRAIN_GRACE);
    while let Some(event) = conn_b.poll_for_event().map_err(err)? {
        assert!(
            !matches!(event, Event::KeyPress(_) | Event::KeyRelease(_)),
            "the excluded window received a key: {event:?}"
        );
    }
    kb.assert_mapping_is(&conn, &mapping_before)?;

    // 6. A's snapshot conflicts while B is focused; B's is gone once
    // destroyed.
    match backend.revalidate(&snap_a) {
        Ok(TargetCheck::Changed { expected, actual }) => {
            assert_eq!(expected, snap_a.target_ref);
            assert_ne!(actual, snap_a.target_ref);
        }
        other => panic!("A's snapshot must conflict: {other:?}"),
    }
    assert!(matches!(
        backend.insert(&snap_a, "must not type"),
        Err(InsertError::TargetChanged { .. })
    ));
    let snap_b = inserter.capture().map_err(err)?;
    assert_ne!(snap_b.target_ref, snap_a.target_ref);
    assert_eq!(drain_text(&conn, a, &mut kb)?, "");

    conn.destroy_window(b).map_err(err)?;
    conn.flush().map_err(err)?;
    std::thread::sleep(DRAIN_GRACE);
    // The server may recycle the id now; never touch it again.
    cleanup.windows.retain(|&window| window != b);
    assert_eq!(backend.revalidate(&snap_b), Ok(TargetCheck::Gone));

    println!(
        "starling-insertion X11 IT passed (mixed {mixed:?}, {spares_left} spares exhausted, \
         grab exclusion {})",
        if grab_excluded {
            "honored"
        } else {
            "not honored"
        }
    );
    Ok(())
}

fn spawn_insert(backend: X11Backend, target: &TargetSnapshot, text: &str) -> JoinHandle<Inserted> {
    let (target, text) = (target.clone(), text.to_string());
    std::thread::spawn(move || backend.insert(&target, &text))
}

fn keysym_of(character: char) -> u32 {
    let codepoint = u32::from(character);
    if codepoint < 0x100 {
        codepoint
    } else {
        0x0100_0000 | codepoint
    }
}

fn is_enter(keysym: u32) -> bool {
    keysym == XK_RETURN || keysym == XK_KP_ENTER
}

fn unmapped_char(kb: &Keyboard, candidates: &[char]) -> char {
    candidates
        .iter()
        .copied()
        .find(|&candidate| kb.keycode_of(keysym_of(candidate)).is_none())
        .unwrap_or_else(|| panic!("none of {candidates:?} is unmapped on this keyboard"))
}

/// `count` astral characters no key produces.
fn unmapped_chars(kb: &Keyboard, count: usize) -> Vec<char> {
    (0x2_0000..)
        .filter_map(char::from_u32)
        .filter(|&character| kb.keycode_of(keysym_of(character)).is_none())
        .take(count)
        .collect()
}

fn xtest_key(conn: &RustConnection, keycode: u8, press: bool) -> ItResult<()> {
    use x11rb::protocol::xtest::ConnectionExt as _;
    let event_type = if press { 2 } else { 3 };
    conn.xtest_fake_input(event_type, keycode, 0, x11rb::NONE, 0, 0, 0)
        .map_err(err)?;
    conn.flush().map_err(err)
}

fn set_keysyms(conn: &RustConnection, keycode: u8, columns: &[u32]) -> ItResult<()> {
    conn.change_keyboard_mapping(1, keycode, columns.len() as u8, columns)
        .map_err(err)?
        .check()
        .map_err(err)
}

fn live_keysyms(conn: &RustConnection, keycode: u8) -> ItResult<Vec<u32>> {
    Ok(conn
        .get_keyboard_mapping(keycode, 1)
        .map_err(err)?
        .reply()
        .map_err(err)?
        .keysyms)
}

/// Unlock Caps Lock with a real tap and require the first group: the
/// backend refuses to type otherwise. XKB runs on its own connection,
/// because an XKB client stops receiving the core `MappingNotify` events
/// this test decodes with.
fn normalize_keyboard_state(tapper: &RustConnection, kb: &Keyboard) -> ItResult<()> {
    use x11rb::protocol::xkb::ConnectionExt as _;
    let (xkb_conn, _) = x11rb::connect(None).map_err(err)?;
    let handshake = xkb_conn
        .xkb_use_extension(1, 0)
        .map_err(err)?
        .reply()
        .map_err(err)?;
    if !handshake.supported {
        return Err("this X server has no XKB extension".into());
    }
    let state = || -> ItResult<(u8, u16)> {
        let state = xkb_conn
            .xkb_get_state(USE_CORE_KEYBOARD)
            .map_err(err)?
            .reply()
            .map_err(err)?;
        Ok((u8::from(state.group), u16::from(state.locked_mods)))
    };
    let (group, locked) = state()?;
    if group != 0 {
        return Err("switch the keyboard to its first group before running this test".into());
    }
    if locked & 0x2 != 0 {
        if let Some(caps) = kb.keycode_of(XK_CAPS_LOCK) {
            xtest_key(tapper, caps, true)?;
            std::thread::sleep(Duration::from_millis(30));
            xtest_key(tapper, caps, false)?;
            std::thread::sleep(Duration::from_millis(100));
        }
        if state()?.1 & 0x2 != 0 {
            return Err("Caps Lock stayed locked; unlock it before running this test".into());
        }
    }
    Ok(())
}

/// One key event or keymap change seen by the test window, in server
/// order. Keysyms are decoded through the keymap current at that moment.
#[derive(Debug, PartialEq, Clone, Copy)]
enum Seen {
    Press(u8, u32),
    Release(u8, u32),
    Map,
}

fn pressed_keysyms(seen: &[Seen]) -> Vec<u32> {
    seen.iter()
        .filter_map(|s| match s {
            Seen::Press(_, keysym) => Some(*keysym),
            _ => None,
        })
        .collect()
}

/// The text the pressed keysyms spell, skipping modifiers.
fn text_of(seen: &[Seen]) -> String {
    pressed_keysyms(seen)
        .into_iter()
        .filter(|&keysym| keysym != 0 && !(0xfe00..=0xffff).contains(&keysym))
        .filter_map(|keysym| char::from_u32(keysym & !0x0100_0000))
        .collect()
}

/// Read `window`'s events while `handles` run, calling `hook` with
/// everything seen so far after each read. The threads are always joined,
/// so a failing hook never orphans one mid-keystroke (which would leave a
/// key stuck down); its error is returned after the join.
fn listen(
    conn: &RustConnection,
    window: Window,
    kb: &mut Keyboard,
    handles: Vec<JoinHandle<Inserted>>,
    mut hook: impl FnMut(&[Seen]) -> ItResult<()>,
) -> ItResult<(Vec<Inserted>, Vec<Seen>)> {
    let deadline = Instant::now() + INSERT_BUDGET;
    let mut seen = Vec::new();
    let mut error = None;
    let pump = |kb: &mut Keyboard, seen: &mut Vec<Seen>, error: &mut Option<String>| loop {
        let event = match conn.poll_for_event() {
            Ok(Some(event)) => event,
            Ok(None) => return,
            Err(e) => {
                error.get_or_insert(err(e));
                return;
            }
        };
        match event {
            Event::KeyPress(e) if e.event == window => {
                seen.push(Seen::Press(e.detail, kb.keysym(e.detail, e.state.into())))
            }
            Event::KeyRelease(e) if e.event == window => {
                seen.push(Seen::Release(e.detail, kb.keysym(e.detail, e.state.into())))
            }
            Event::MappingNotify(_) => {
                if let Err(e) = kb.reload(conn) {
                    error.get_or_insert(e);
                }
                seen.push(Seen::Map);
            }
            _ => {}
        }
    };
    loop {
        pump(kb, &mut seen, &mut error);
        if error.is_none() {
            if let Err(e) = hook(&seen) {
                error = Some(e);
            }
        }
        if handles.iter().all(JoinHandle::is_finished) {
            std::thread::sleep(DRAIN_GRACE);
            pump(kb, &mut seen, &mut error);
            break;
        }
        if Instant::now() > deadline {
            return Err("the insert did not finish within the test budget".into());
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    let results = handles
        .into_iter()
        .map(|handle| handle.join().expect("an inserting thread must not panic"))
        .collect();
    match error {
        Some(error) => Err(error),
        None => Ok((results, seen)),
    }
}

fn drain_text(conn: &RustConnection, window: Window, kb: &mut Keyboard) -> ItResult<String> {
    let (_, seen) = listen(conn, window, kb, Vec::new(), |_| Ok(()))?;
    Ok(text_of(&seen))
}

/// Whether another client's request waits out a grab held by `conn`. On a
/// conforming server (Xvfb, Xorg) the probe blocks for the whole grab;
/// WSLg's XWayland answers at once.
fn grab_exclusion_honored(conn: &RustConnection) -> ItResult<bool> {
    let (other, _) = x11rb::connect(None).map_err(err)?;
    // Acknowledged before the probe starts, so the probe cannot win.
    conn.grab_server().map_err(err)?.check().map_err(err)?;
    let probe = std::thread::spawn(move || -> ItResult<Duration> {
        let sent = Instant::now();
        other.get_input_focus().map_err(err)?.reply().map_err(err)?;
        Ok(sent.elapsed())
    });
    std::thread::sleep(Duration::from_millis(150));
    conn.ungrab_server().map_err(err)?;
    conn.flush().map_err(err)?;
    Ok(probe.join().expect("probe thread")? >= Duration::from_millis(100))
}

/// The Shift keycode the backend presses: the first of the Shift
/// modifier's keycodes with Shift in its unshifted column.
fn shift_keycode_of(conn: &RustConnection, kb: &Keyboard) -> ItResult<u8> {
    let reply = conn
        .get_modifier_mapping()
        .map_err(err)?
        .reply()
        .map_err(err)?;
    let per_row = usize::from(reply.keycodes_per_modifier());
    reply.keycodes[..per_row]
        .iter()
        .copied()
        .find(|&keycode| {
            kb.contains(keycode) && matches!(kb.syms_of(keycode)[0], XK_SHIFT_L | XK_SHIFT_R)
        })
        .ok_or_else(|| "this keyboard has no usable Shift keycode".into())
}

struct Atoms {
    wm_protocols: Atom,
    wm_delete_window: Atom,
    atom: Atom,
    wm_class: Atom,
    wm_name: Atom,
    string: Atom,
    utf8_string: Atom,
    net_wm_name: Atom,
    net_active_window: Atom,
    net_wm_pid: Atom,
    cardinal: Atom,
}

impl Atoms {
    fn intern(conn: &RustConnection) -> ItResult<Atoms> {
        let one = |name: &str| -> ItResult<Atom> {
            Ok(conn
                .intern_atom(false, name.as_bytes())
                .map_err(err)?
                .reply()
                .map_err(err)?
                .atom)
        };
        Ok(Atoms {
            wm_protocols: one("WM_PROTOCOLS")?,
            wm_delete_window: one("WM_DELETE_WINDOW")?,
            atom: one("ATOM")?,
            wm_class: one("WM_CLASS")?,
            wm_name: one("WM_NAME")?,
            string: one("STRING")?,
            utf8_string: one("UTF8_STRING")?,
            net_wm_name: one("_NET_WM_NAME")?,
            net_active_window: one("_NET_ACTIVE_WINDOW")?,
            net_wm_pid: one("_NET_WM_PID")?,
            cardinal: one("CARDINAL")?,
        })
    }
}

/// A named, classed window without `_NET_WM_PID` (capture must not invent
/// one); mapped and waited for unless `map` is false.
fn make_window(
    conn: &RustConnection,
    root: Window,
    atoms: &Atoms,
    label: &str,
    map: bool,
) -> ItResult<Window> {
    let window = conn.generate_id().map_err(err)?;
    conn.create_window(
        0,
        window,
        root,
        40,
        40,
        360,
        140,
        1,
        WindowClass::INPUT_OUTPUT,
        0,
        &CreateWindowAux::new().event_mask(
            EventMask::KEY_PRESS | EventMask::KEY_RELEASE | EventMask::STRUCTURE_NOTIFY,
        ),
    )
    .map_err(err)?
    .check()
    .map_err(err)?;
    let title = format!("starling x11 IT window ({label})");
    let protocols: Vec<u8> = atoms.wm_delete_window.to_ne_bytes().to_vec();
    for (property, type_, format, data) in [
        (
            atoms.wm_class,
            atoms.string,
            8,
            &b"starlingit\0StarlingIt\0"[..],
        ),
        (atoms.wm_name, atoms.string, 8, title.as_bytes()),
        (atoms.net_wm_name, atoms.utf8_string, 8, title.as_bytes()),
        (atoms.wm_protocols, atoms.atom, 32, &protocols[..]),
    ] {
        let length = data.len() as u32 / (u32::from(format) / 8);
        conn.change_property(
            PropMode::REPLACE,
            window,
            property,
            type_,
            format,
            length,
            data,
        )
        .map_err(err)?
        .check()
        .map_err(err)?;
    }
    if map {
        conn.map_window(window).map_err(err)?.check().map_err(err)?;
        wait_viewable(conn, window)?;
    }
    conn.flush().map_err(err)?;
    Ok(window)
}

fn wait_viewable(conn: &RustConnection, window: Window) -> ItResult<()> {
    let deadline = Instant::now() + SETTLE;
    loop {
        let attributes = conn
            .get_window_attributes(window)
            .map_err(err)?
            .reply()
            .map_err(err)?;
        if attributes.map_state == MapState::VIEWABLE {
            return Ok(());
        }
        if Instant::now() > deadline {
            return Err(format!("window {window:#x} never became viewable"));
        }
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// Focus `window` and wait until the input focus is it and the EWMH active
/// window (if a WM sets one) agrees.
fn focus_and_wait(
    conn: &RustConnection,
    root: Window,
    atoms: &Atoms,
    window: Window,
) -> ItResult<()> {
    conn.set_input_focus(InputFocus::PARENT, window, x11rb::CURRENT_TIME)
        .map_err(err)?
        .check()
        .map_err(err)?;
    let deadline = Instant::now() + SETTLE;
    loop {
        let focus = conn
            .get_input_focus()
            .map_err(err)?
            .reply()
            .map_err(err)?
            .focus;
        let active = conn
            .get_property(false, root, atoms.net_active_window, x11rb::NONE, 0, 1)
            .map_err(err)?
            .reply()
            .map_err(err)?
            .value32()
            .and_then(|mut values| values.next());
        if focus == window && active.map_or(true, |active| active == window) {
            return Ok(());
        }
        if Instant::now() > deadline {
            return Err("focus and the EWMH active window never agreed on the test window".into());
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// The test's copy of the keyboard mapping, reloaded at every
/// `MappingNotify` like a target toolkit does.
struct Keyboard {
    min: u8,
    width: usize,
    syms: Vec<u32>,
}

impl Keyboard {
    fn load(conn: &RustConnection) -> ItResult<Keyboard> {
        let setup = conn.setup();
        let reply = conn
            .get_keyboard_mapping(setup.min_keycode, setup.max_keycode - setup.min_keycode + 1)
            .map_err(err)?
            .reply()
            .map_err(err)?;
        Ok(Keyboard {
            min: setup.min_keycode,
            width: usize::from(reply.keysyms_per_keycode),
            syms: reply.keysyms,
        })
    }

    fn reload(&mut self, conn: &RustConnection) -> ItResult<()> {
        *self = Keyboard::load(conn)?;
        Ok(())
    }

    /// Reload from the server and require `expected`.
    fn assert_mapping_is(&mut self, conn: &RustConnection, expected: &[u32]) -> ItResult<()> {
        self.reload(conn)?;
        assert_eq!(self.syms, expected, "the keyboard mapping must be restored");
        Ok(())
    }

    fn keycodes(&self) -> impl DoubleEndedIterator<Item = u8> + '_ {
        (0..self.syms.len() / self.width).map(|index| self.min + index as u8)
    }

    fn contains(&self, keycode: u8) -> bool {
        self.keycodes().any(|k| k == keycode)
    }

    fn syms_of(&self, keycode: u8) -> &[u32] {
        &self.syms[usize::from(keycode - self.min) * self.width..][..self.width]
    }

    /// The keysym a key event decodes to: column 1 with Shift held.
    fn keysym(&self, keycode: u8, state: u16) -> u32 {
        let column = usize::from(state & 1 != 0);
        self.syms_of(keycode).get(column).copied().unwrap_or(0)
    }

    /// The first keycode with `keysym` in column 0 or 1.
    fn keycode_of(&self, keysym: u32) -> Option<u8> {
        self.keycodes().find(|&keycode| {
            self.syms_of(keycode)
                .iter()
                .take(2)
                .any(|&sym| sym == keysym)
        })
    }

    fn spare_count(&self) -> usize {
        self.keycodes()
            .filter(|&keycode| self.syms_of(keycode).iter().all(|&sym| sym == 0))
            .count()
    }

    /// The `count` highest all-`NoSymbol` keycodes, highest first: the
    /// backend's own borrowing order.
    fn spare_keycodes(&self, count: usize) -> Option<Vec<u8>> {
        let spares: Vec<u8> = self
            .keycodes()
            .rev()
            .filter(|&keycode| self.syms_of(keycode).iter().all(|&sym| sym == 0))
            .take(count)
            .collect();
        (spares.len() == count).then_some(spares)
    }
}

/// Restores the keyboard mappings and destroys the windows the test
/// touched, also when an assertion panics.
struct Cleanup<'a> {
    conn: &'a RustConnection,
    windows: Vec<Window>,
    mappings: Vec<(u8, Vec<u32>)>,
}

impl Drop for Cleanup<'_> {
    fn drop(&mut self) {
        for (keycode, original) in &self.mappings {
            let _ = set_keysyms(self.conn, *keycode, original);
        }
        for window in &self.windows {
            let _ = self.conn.destroy_window(*window);
        }
        let _ = self.conn.flush();
    }
}
