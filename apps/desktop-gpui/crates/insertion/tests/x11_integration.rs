//! The X11 backend against a real server (issue #221, slice 2 phase
//! A). Gated on `DISPLAY` **and** `STARLING_X11_IT=1` — this box's
//! WSLg session serves `DISPLAY=:0` — because the whole point of this
//! test is *end-to-end honesty*: a real window manager ("Weston WM"
//! under WSLg), a real keyboard mapping, real XTest events decoded
//! back through that mapping.
//!
//! What it pins, in order:
//!
//! 1. **Capture is honest**: a mapped, focused window with `WM_CLASS`
//!    and `_NET_WM_NAME` set captures as itself — right ids, right
//!    class, right title, no invented pid (the test window
//!    deliberately sets no `_NET_WM_PID`).
//! 2. **A mixed string lands as text**: insert ASCII (including a
//!    Shift-level `C` and `!`), `é`, `ß` and `→` — the last three
//!    pre-mapped by the test onto spare keycodes, so the backend's
//!    find-in-mapping path is exercised on real server state — plus a
//!    BMP character no key produces, so the remap path is exercised
//!    inside the same insert. The `KeyPress` events are decoded *as
//!    text at event time*: the insert runs on its own thread while
//!    this connection consumes the event stream as it arrives,
//!    refreshing its keymap copy at every `MappingNotify` (the
//!    backend's own remaps and restores) — a faithful model of a
//!    target processing events promptly, and the only honest measure
//!    of "what text did the window see" when mappings change
//!    underneath. The decoded text must equal the inserted text
//!    exactly. (Decoding *after the fact* would read the restored
//!    mapping and hide exactly this class of bug.)
//! 3. **The remap path types and restores**: a character no keyboard
//!    produces (an astral one, so it cannot ride any BMP column)
//!    rides a temporarily-remapped spare keycode, decodes as itself,
//!    and the whole keyboard mapping is byte-for-byte what it was
//!    before the insert — the leak-free invariant, asserted against
//!    the live server rather than the backend's return value.
//!    3b. The same remap path with an *unmapped uppercase letter*
//!    (picked at runtime like the BMP one): a borrow that wrote only
//!    the base column would let XKB expand a lowercase/uppercase pair
//!    (the planner would then find the letter gone or the unshifted
//!    press would type the wrong case), so this pins that an
//!    uppercase borrow decodes as text with its case intact.
//! 4. **Held modifiers refuse typing**: Ctrl held down via XTest
//!    during an insert makes the insert wait out the bounded release
//!    window and then refuse with `ModifiersHeld` — nothing typed,
//!    and no modifier was released behind the user's back.
//! 5. **Chunked typing rechecks the target**: a long string types in
//!    chunks; when a second window is mapped the moment the first
//!    chunk's first key is *observed* here (the clean seam: this
//!    connection owns the events, so the steal is timed off the
//!    observed `KeyPress`, not a sleep), the insert stops immediately
//!    with `PartialDelivery`, names how much may have landed, exactly
//!    that prefix decodes from the event stream, and the keyboard
//!    mapping still ends where it started.
//!    5b. **Concurrent inserts serialize**: two inserts racing from
//!    two threads land whole and in order (`text + text`, never a
//!    character-level interleave) — the process-wide insert lock in
//!    action, including the mapping load.
//! 6. **Focus safety at the edges**: with the second window focused,
//!    revalidate of the first snapshot reports `TargetChanged` (naming
//!    the new target), `insert` into the old snapshot refuses, and
//!    types nothing; destroying the second window makes its snapshot
//!    report `Gone`.
//! 7. **Control characters never type**: a `\n` payload is refused
//!    with `MultilineUnsupported` and produces no key events at all.
//!
//! The test also *normalizes* what it may: Caps Lock, if locked, is
//! unlocked (via a real XTest tap) and the active group must be the
//! first — the backend rightly refuses to type under either. The
//! remap-path characters are picked from candidate lists until one is
//! genuinely absent from the mapping, so a session dirtied by an old
//! leaked borrow (a crashed run of an earlier version, say) cannot
//! make the remap path silently degrade into the find path.

#![cfg(target_os = "linux")]

use std::time::{Duration, Instant};

use x11rb::connection::Connection;
use x11rb::protocol::xproto::{
    Atom, ConnectionExt, CreateWindowAux, EventMask, MapState, PropMode, Window, WindowClass,
};
use x11rb::rust_connection::RustConnection;

use starling_insertion::x11::{X11Backend, X11_CHUNK_CHARS};
use starling_insertion::{
    BackendKind, InsertError, InsertReceipt, Inserter, InsertionBackend, TargetCheck,
    EVIDENCE_SYNTHETIC_KEYS, MODIFIER_RELEASE_WAIT,
};

/// How long to wait for the WM (focus, active-window bookkeeping) or
/// the server to settle before an assertion gives up.
const SETTLE: Duration = Duration::from_secs(5);
/// Grace before the final event drain: covers delivery to *this*
/// connection of events the server already queued.
const DRAIN_GRACE: Duration = Duration::from_millis(150);
/// Overall budget for one insert-and-listen round.
const INSERT_BUDGET: Duration = Duration::from_secs(30);

/// The mixed string's fixed parts; the BMP remap character is chosen
/// at runtime (see the module docs).
const MIXED_HEAD: &str = "Café ß ";
const MIXED_TAIL: &str = " → 42!";
/// Candidate BMP characters for the mixed insert's remap-path slot.
const MIXED_REMAP_CANDIDATES: &[char] = &['ŋ', 'ƍ', 'Ƃ', 'ǅ', 'ƞ'];
/// Candidate astral characters for the dedicated remap-path insert.
const UNMAPPED_CANDIDATES: &[char] = &['🦄', '🥑', '🛰', '𝄞'];
/// Candidate *uppercase* BMP letters (each with a lowercase twin, so
/// XKB's case-pair expansion would apply to a single-column borrow)
/// for the uppercase remap-path insert.
const UPPER_UNMAPPED_CANDIDATES: &[char] = &['Ŋ', 'Ƕ', 'Ǯ', 'Ǥ', 'Ɣ'];
/// `XK_Control_L` — the held-modifier test's key.
const XK_CONTROL_L: u32 = 0xffe3;
/// `XK_Control_R`, the fallback if a layout has no left Control.
const XK_CONTROL_R: u32 = 0xffe4;
/// `XK_Caps_Lock` — the setup normalization's key.
const XK_CAPS_LOCK: u32 = 0xffe5;

/// A one-string failure type: every site has a different failure but
/// the handler only needs to report it.
#[derive(Debug)]
struct ItError(String);

impl std::fmt::Display for ItError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

fn x11(error: impl std::fmt::Display) -> ItError {
    ItError(error.to_string())
}

fn focus_error(why: &str) -> ItError {
    ItError(why.to_string())
}

fn main_test() -> Result<(), ItError> {
    let (conn, screen) =
        x11rb::connect(None).map_err(|e| x11(format!("connecting to DISPLAY: {e}")))?;
    let root = conn.setup().roots[screen].root;
    let atoms = Atoms::intern(&conn)?;
    let mut kb = Keyboard::load(&conn)?;
    let mut cleanup = Cleanup::new(&conn);

    let a = make_window(&conn, root, &atoms, "first", true)?;
    cleanup.windows.push(a);

    focus_and_wait(&conn, root, &atoms, a)?;

    // The pre-map: é/ß/→ onto spare keycodes so the backend's
    // find-in-mapping path meets them on real server state (the remap
    // path is exercised by the chosen unmapped characters). Originals
    // are recorded for cleanup so the session's keyboard leaves the
    // test exactly as it came in.
    let spares = kb
        .spare_keycodes(3)
        .ok_or_else(|| focus_error("this keyboard has no three spare keycodes to pre-map"))?;
    let premaps = ['é', 'ß', '→'];
    for (keycode, character) in spares.iter().zip(premaps) {
        kb.remap(&conn, *keycode, keysym_of(character))?;
        cleanup.premaps.push((*keycode, vec![0; kb.width]));
    }
    // The pre-maps are compared and decoded through the *server's*
    // canonical form: an XKB server normalizes a column-0 write (it
    // fills the shifted column with the uppercase twin and mirrors
    // the group — `é` alone becomes `é`/`É`, twice), so the local
    // hand-updated copy would drift from the truth the very first
    // reload sees. Re-read everything once the pre-maps are in.
    kb.reload(&conn)?;
    // The characters that must ride the remap path: picked until one
    // is genuinely absent from the live mapping.
    let mixed_remap = unmapped_char(&kb, MIXED_REMAP_CANDIDATES);
    let unmapped = unmapped_char(&kb, UNMAPPED_CANDIDATES);
    let upper = unmapped_char(&kb, UPPER_UNMAPPED_CANDIDATES);
    let mixed = format!("{MIXED_HEAD}{mixed_remap}{MIXED_TAIL}");

    // Normalize the keyboard state the backend is entitled to require:
    // Caps Lock unlocked, first group active (see the module docs).
    normalize_keyboard_state(&conn, &kb)?;
    // The leak-free invariant's baseline: the mapping with the test's
    // own pre-maps applied and nothing borrowed. Every insert below
    // must leave the live mapping exactly here.
    let mapping_before = kb.snapshot();

    let inserter = Inserter::with_backends(vec![Box::new(X11Backend::new())]);

    // 1. Capture.
    let snap_a = inserter
        .capture()
        .map_err(|e| x11(format!("capture: {e}")))?;
    assert_eq!(snap_a.backend, BackendKind::X11);
    assert_eq!(snap_a.app.as_deref(), Some("StarlingIt"));
    assert_eq!(
        snap_a.title.as_deref(),
        Some("starling x11 IT window (first)")
    );
    assert_eq!(snap_a.pid, None, "the test window sets no _NET_WM_PID");
    let (active_a, focus_a, _) = snap_a.ids().expect("a captured ref round trips");
    assert_eq!(active_a, a as u64, "active window is A");
    assert_eq!(focus_a, a as u64, "focus window is A");

    let backend = inserter
        .backend_for(&snap_a)
        .expect("the X11 backend is present");

    // 7. Control characters are refused before anything is typed; pin
    //    it while a real target is focused so the refusal cannot be
    //    blamed on focus state.
    match backend.insert(&snap_a, "line\nbreak") {
        Err(InsertError::MultilineUnsupported) => {}
        other => panic!("multiline must be refused on a live target: {other:?}"),
    }
    assert!(
        drain_text(&conn, a, &mut kb)?.is_empty(),
        "a refused insert may not type"
    );

    // 2. The mixed string, decoded as text at event time (the chosen
    //    BMP character rides the remap path inside this very insert).
    let (receipt, typed) = insert_and_decode(
        &conn,
        a,
        &mut kb,
        || {
            let backend_thread = X11Backend::new();
            let snapshot = snap_a.clone();
            let payload = mixed.clone();
            vec![std::thread::spawn(move || {
                backend_thread.insert(&snapshot, &payload)
            })]
        },
        None,
    )?;
    let receipt = receipt.into_iter().next().expect("one insert thread");
    let receipt = receipt.map_err(|e| x11(format!("inserting {mixed:?}: {e}")))?;
    assert_eq!(receipt.evidence, EVIDENCE_SYNTHETIC_KEYS);
    assert_eq!(typed, mixed, "the KeyPress stream decodes back to the text");
    kb.assert_mapping_is(&conn, &mapping_before)?;

    // 3. The remap path with an astral character: rides a borrowed
    //    spare keycode, decodes as itself, and leaves the mapping as
    //    it started.
    let unmapped_text = unmapped.to_string();
    let (receipt, typed) = insert_and_decode(
        &conn,
        a,
        &mut kb,
        || {
            let backend_thread = X11Backend::new();
            let snapshot = snap_a.clone();
            let payload = unmapped_text.clone();
            vec![std::thread::spawn(move || {
                backend_thread.insert(&snapshot, &payload)
            })]
        },
        None,
    )?;
    let receipt = receipt.into_iter().next().expect("one insert thread");
    let receipt = receipt.map_err(|e| x11(format!("inserting {unmapped}: {e}")))?;
    assert_eq!(receipt.evidence, EVIDENCE_SYNTHETIC_KEYS);
    assert_eq!(
        typed, unmapped_text,
        "the remapped character must arrive as text, not as a bare keycode"
    );
    kb.assert_mapping_is(&conn, &mapping_before)?;

    // 3b. The remap path with an unmapped *uppercase* letter. The
    //     borrow must write the keysym into every column (all groups ×
    //     levels), leaving XKB no case pair to expand: the unshifted
    //     press then yields exactly the uppercase letter, and the
    //     decode must carry the case. A single column-0 write would
    //     come back expanded into a lowercase/uppercase pair here and
    //     type the wrong case (or find nothing to type at all).
    let upper_text = format!("a{upper}b{upper}");
    let (receipt, typed) = insert_and_decode(
        &conn,
        a,
        &mut kb,
        || {
            let backend_thread = X11Backend::new();
            let snapshot = snap_a.clone();
            let payload = upper_text.clone();
            vec![std::thread::spawn(move || {
                backend_thread.insert(&snapshot, &payload)
            })]
        },
        None,
    )?;
    let receipt = receipt.into_iter().next().expect("one insert thread");
    let receipt = receipt.map_err(|e| x11(format!("inserting {upper_text:?}: {e}")))?;
    assert_eq!(receipt.evidence, EVIDENCE_SYNTHETIC_KEYS);
    assert_eq!(
        typed, upper_text,
        "an uppercase borrowed keycode must decode as text with its case intact"
    );
    kb.assert_mapping_is(&conn, &mapping_before)?;

    // 4. Held modifiers: Ctrl physically down (a real XTest press the
    //    server's keyboard state reflects) makes the insert wait out
    //    the bounded release window and then refuse — without typing
    //    and without releasing Ctrl itself.
    let control = kb
        .keycode_of(XK_CONTROL_L)
        .or_else(|| kb.keycode_of(XK_CONTROL_R))
        .ok_or_else(|| focus_error("this keyboard has no Control key"))?;
    xtest_press(&conn, control)?;
    let started = Instant::now();
    match backend.insert(&snap_a, "blocked") {
        Err(InsertError::ModifiersHeld { held }) => {
            assert!(
                held.iter().any(|name| name.contains("Control")),
                "the held set names Control: {held:?}"
            );
        }
        other => panic!("held modifiers must refuse the insert: {other:?}"),
    }
    let waited = started.elapsed();
    assert!(
        waited >= MODIFIER_RELEASE_WAIT,
        "the refusal comes after the bounded wait, not instantly ({waited:?})"
    );
    xtest_release(&conn, control)?;
    assert!(
        drain_text(&conn, a, &mut kb)?.is_empty(),
        "a held-modifier refusal may not type"
    );

    // 5b. Concurrent inserts serialize: two inserts racing from two
    //     threads must not interleave — the process-wide insert lock
    //     means one types (and borrows, and restores) to completion
    //     before the other even loads the keyboard mapping, so the
    //     event stream is exactly one text then the other, never a
    //     character-level interleave (and the second insert must not
    //     consume the first's temporary remap).
    let (results, typed) = insert_and_decode(
        &conn,
        a,
        &mut kb,
        || {
            (0..2)
                .map(|_| {
                    let backend_thread = X11Backend::new();
                    let snapshot = snap_a.clone();
                    let payload = mixed.clone();
                    std::thread::spawn(move || backend_thread.insert(&snapshot, &payload))
                })
                .collect()
        },
        None,
    )?;
    assert_eq!(results.len(), 2);
    for (index, result) in results.into_iter().enumerate() {
        let receipt = result.map_err(|e| x11(format!("concurrent insert {index}: {e}")))?;
        assert_eq!(receipt.evidence, EVIDENCE_SYNTHETIC_KEYS);
    }
    assert_eq!(
        typed,
        format!("{mixed}{mixed}"),
        "two concurrent inserts must land whole and in order, never interleaved"
    );
    kb.assert_mapping_is(&conn, &mapping_before)?;

    // 5. Chunked typing rechecks the target. B is created *unmapped*
    //    (creating a window takes no focus) and mapped the moment the
    //    first chunk's first key press is observed here — the
    //    deterministic seam. Mapping activates B under the WM, and the
    //    explicit SetInputFocus pins focus on it even if the WM is
    //    slow about bookkeeping.
    let b = make_window(&conn, root, &atoms, "second", false)?;
    cleanup.windows.push(b);
    let long = format!("{mixed} ").repeat(8);
    // Steal at the first *decoded character*: the current chunk keeps
    // typing to completion after the focus moves (there are no
    // mid-chunk checks, by design), so a few of its keys land on B —
    // which is exactly why the backend's count is an upper bound
    // ("up to N characters may have been typed") and the assertion
    // below checks a prefix, not byte-equality with the count.
    let steal = |decoded_chars: usize, conn: &RustConnection| -> Result<(), ItError> {
        if decoded_chars != 1 {
            return Ok(());
        }
        conn.map_window(b).map_err(x11)?.check().map_err(x11)?;
        // SetInputFocus on an unmapped window is a protocol error, so
        // wait until the WM actually made B viewable — the insert keeps
        // typing chunk by chunk meanwhile, and landing the steal in a
        // later chunk is equally valid for the assertions.
        let viewable_by = Instant::now() + Duration::from_secs(2);
        loop {
            let state = conn
                .get_window_attributes(b)
                .map_err(x11)?
                .reply()
                .map_err(x11)?
                .map_state;
            if state == MapState::VIEWABLE {
                break;
            }
            if Instant::now() > viewable_by {
                return Err(focus_error(
                    "window B never became viewable at the steal moment",
                ));
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        conn.set_input_focus(
            x11rb::protocol::xproto::InputFocus::PARENT,
            b,
            x11rb::CURRENT_TIME,
        )
        .map_err(x11)?
        .check()
        .map_err(x11)?;
        conn.flush().map_err(x11)?;
        Ok(())
    };
    let (result, typed) = insert_and_decode(
        &conn,
        a,
        &mut kb,
        || {
            let backend_thread = X11Backend::new();
            let snapshot = snap_a.clone();
            let payload = long.clone();
            vec![std::thread::spawn(move || {
                backend_thread.insert(&snapshot, &payload)
            })]
        },
        Some(&steal),
    )?;
    let result = result.into_iter().next().expect("one insert thread");
    match result {
        Err(InsertError::PartialDelivery {
            delivered_chars,
            total_chars,
            cause,
        }) => {
            assert_eq!(total_chars, long.chars().count());
            assert!(delivered_chars > 0, "the first chunk was observed typing");
            assert!(
                delivered_chars < total_chars,
                "the insert stopped before the end"
            );
            assert_eq!(
                delivered_chars % X11_CHUNK_CHARS,
                0,
                "delivery stops on chunk boundaries: {delivered_chars}"
            );
            assert!(
                matches!(*cause, InsertError::TargetChanged { .. }),
                "the stop cause is the focus change: {cause:?}"
            );
            assert!(
                !typed.is_empty(),
                "the first chunk was observed typing into A"
            );
            assert!(
                typed.chars().count() <= delivered_chars,
                "A received at most the reported upper bound: {} > {delivered_chars}",
                typed.chars().count()
            );
            let prefix = long.chars().take(typed.chars().count()).collect::<String>();
            assert_eq!(
                typed, prefix,
                "what A received is a prefix of the text, and only that"
            );
        }
        other => panic!("a mid-insert focus steal must stop partially: {other:?}"),
    }
    kb.assert_mapping_is(&conn, &mapping_before)?;

    // 6. Focus safety: the frozen A snapshot must conflict while B is
    //    focused.
    match backend.revalidate(&snap_a) {
        Ok(TargetCheck::Changed { expected, actual }) => {
            assert_eq!(expected, snap_a.target_ref);
            assert_ne!(actual, snap_a.target_ref, "actual names the new focus");
        }
        other => panic!("focus moved to B, so A's snapshot must conflict: {other:?}"),
    }
    match backend.insert(&snap_a, "must not type") {
        Err(InsertError::TargetChanged { .. }) => {}
        other => panic!("a changed target must refuse: {other:?}"),
    }
    let snap_b = inserter
        .capture()
        .map_err(|e| x11(format!("recapture: {e}")))?;
    assert_ne!(snap_b.target_ref, snap_a.target_ref);
    assert!(
        drain_text(&conn, a, &mut kb)?.is_empty(),
        "nothing may be typed into A while B is focused"
    );

    // And gone: destroy B, its snapshot reports Gone.
    conn.destroy_window(b).map_err(x11)?;
    conn.flush().map_err(x11)?;
    std::thread::sleep(DRAIN_GRACE);
    // The id may legally be recycled by the server now; never touch
    // it again (including in cleanup).
    cleanup.windows.retain(|&window| window != b);
    match backend.revalidate(&snap_b) {
        Ok(TargetCheck::Gone) => {}
        other => panic!("a destroyed window must report Gone: {other:?}"),
    }

    cleanup.run();
    println!(
        "starling-insertion X11 IT passed: capture identity, mixed-string decode as text \
         ({mixed:?}), remap+restore (mapping byte-for-byte unchanged), uppercase remap decode \
         ({upper_text:?}), held-modifier refusal, chunk-bounded partial delivery on focus \
         steal, serialized concurrent inserts, changed and gone revalidation"
    );
    Ok(())
}

#[test]
fn x11_end_to_end_capture_type_revalidate() {
    // Double gate (see the module docs): the test needs a display it
    // owns well enough to focus its own windows, which is a deliberate
    // opt-in, not a default part of `cargo test`.
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

/// The standard keysym for a character (the same convention the
/// backend encodes with, which is why the decode agrees).
fn keysym_of(character: char) -> u32 {
    let codepoint = character as u32;
    if codepoint < 0x100 {
        codepoint
    } else {
        0x0100_0000 | codepoint
    }
}

/// The first candidate character no key of the live mapping produces
/// in its trusted columns (see the module docs for why the test does
/// not hard-code one).
fn unmapped_char(kb: &Keyboard, candidates: &[char]) -> char {
    candidates
        .iter()
        .copied()
        .find(|&candidate| kb.keycode_of(keysym_of(candidate)).is_none())
        .unwrap_or_else(|| panic!("none of {candidates:?} is unmapped on this keyboard"))
}

/// One XTest fake key event on this test's own connection.
fn xtest_key(conn: &RustConnection, keycode: u8, event_type: u8) -> Result<(), ItError> {
    use x11rb::protocol::xtest::ConnectionExt as _;
    conn.xtest_fake_input(event_type, keycode, 0, x11rb::NONE, 0, 0, 0)
        .map_err(x11)?;
    conn.flush().map_err(x11)
}

fn xtest_press(conn: &RustConnection, keycode: u8) -> Result<(), ItError> {
    xtest_key(conn, keycode, 2)
}

fn xtest_release(conn: &RustConnection, keycode: u8) -> Result<(), ItError> {
    xtest_key(conn, keycode, 3)
}

/// `xkbUseCoreKeyboard`.
const USE_CORE_KEYBOARD: u16 = 0x100;

/// The keyboard state the backend requires (group 0, no locks) — or
/// an error saying what to fix. Caps Lock is *unlocked* here via a
/// real XTest tap (the same way a user would), because the backend
/// rightly refuses to type under it and the test wants a typable
/// keyboard, not a skipped assertion.
fn normalize_keyboard_state(tapper: &RustConnection, kb: &Keyboard) -> Result<(), ItError> {
    // The XKB handshake deliberately runs on its *own* connection: a
    // client that has used the XKB extension stops receiving core
    // MappingNotify events (the server switches it to the XKB event
    // stream), and this test's listening connection must stay a plain
    // core client to observe the backend's remaps the way the
    // MappingNotify-tracking decode depends on.
    use x11rb::protocol::xkb::ConnectionExt as _;
    let xkb_conn = x11rb::connect(None).map_err(x11)?.0;
    let handshake = xkb_conn
        .xkb_use_extension(1, 0)
        .map_err(x11)?
        .reply()
        .map_err(x11)?;
    if !handshake.supported {
        return Err(focus_error("this X server has no XKB extension"));
    }
    let state = || -> Result<(u8, u16), ItError> {
        let state = xkb_conn
            .xkb_get_state(USE_CORE_KEYBOARD)
            .map_err(x11)?
            .reply()
            .map_err(x11)?;
        Ok((u8::from(state.group), u16::from(state.locked_mods)))
    };
    let (group, locked) = state()?;
    if group != 0 {
        return Err(focus_error(
            "an alternate keyboard group is active; switch the layout back to its first \
             group before running this test (the backend refuses to type under it, by \
             design)",
        ));
    }
    if locked & 0x0002 != 0 {
        // Lock bit: tap Caps Lock like a user would (XTest needs no
        // XKB handshake), then re-read.
        if let Some(caps) = kb.keycode_of(XK_CAPS_LOCK) {
            xtest_press(tapper, caps)?;
            std::thread::sleep(Duration::from_millis(30));
            xtest_release(tapper, caps)?;
            std::thread::sleep(Duration::from_millis(100));
        }
        let (_group, locked) = state()?;
        if locked & 0x0002 != 0 {
            return Err(focus_error(
                "Caps Lock stayed locked after a tap; press Caps Lock once before running \
                 this test",
            ));
        }
    }
    Ok(())
}

/// Run one or more inserts (each on its own thread, via `spawn`)
/// while this thread consumes this connection's event stream *as it
/// arrives*: every `KeyPress` for `window` is decoded with the keymap
/// copy current at the moment the event is read, and the copy is
/// refreshed whenever a `MappingNotify` flows past — the faithful
/// "decode as text at event time" a promptly-reading target performs
/// (decoding after the fact would read the restored mapping and hide
/// remap/restore bugs). `on_first_press` runs (on this thread) when
/// the first `KeyPress` is observed — the deterministic hook the
/// focus-steal test uses. Every spawned handle is joined before the
/// result is returned, so a caller spawning concurrent inserts sees
/// all their outcomes.
fn insert_and_decode(
    conn: &RustConnection,
    window: Window,
    kb: &mut Keyboard,
    spawn: impl FnOnce() -> Vec<std::thread::JoinHandle<Result<InsertReceipt, InsertError>>>,
    on_press: Option<&dyn Fn(usize, &RustConnection) -> Result<(), ItError>>,
) -> Result<(Vec<Result<InsertReceipt, InsertError>>, String), ItError> {
    let handles = spawn();
    let deadline = Instant::now() + INSERT_BUDGET;
    let mut decoded = String::new();
    let mut first_press_seen = 0usize;
    // Listener-side problems (the steal callback failing, a reload
    // erroring) are *collected*, never propagated early: abandoning
    // the join would orphan a typing thread — and an orphan killed
    // mid-press leaves a stuck, auto-repeating key on the server for
    // whoever is focused next. The join always happens; the error, if
    // any, surfaces after it.
    let mut listener_error: Option<ItError> = None;
    let mut pump = |kb: &mut Keyboard, decoded: &mut String| {
        while let Some(event) = conn.poll_for_event().unwrap() {
            match event {
                x11rb::protocol::Event::KeyPress(press) if press.event == window => {
                    if let Some(character) = kb.decode(press.detail, u16::from(press.state)) {
                        decoded.push(character);
                        first_press_seen = decoded.chars().count();
                        if let Some(callback) = on_press {
                            if let Err(error) = callback(first_press_seen, conn) {
                                listener_error = Some(error);
                            }
                        }
                    }
                }
                x11rb::protocol::Event::MappingNotify(_) => {
                    if let Err(error) = kb.reload(conn) {
                        listener_error = Some(error);
                    }
                }
                _ => {}
            }
        }
    };
    loop {
        pump(&mut *kb, &mut decoded);
        if handles.iter().all(|handle| handle.is_finished()) {
            // Stragglers: events the server queued before the inserts
            // returned still count as "what the target saw".
            std::thread::sleep(DRAIN_GRACE);
            pump(&mut *kb, &mut decoded);
            break;
        }
        if Instant::now() > deadline {
            return Err(focus_error(
                "the insert did not finish within the test budget",
            ));
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    let joined = handles
        .into_iter()
        .map(|handle| handle.join().expect("an inserting thread must not panic"))
        .collect();
    match listener_error {
        Some(error) => Err(error),
        None => Ok((joined, decoded)),
    }
}

/// The atoms the test needs, interned once.
struct Atoms {
    wm_protocols: Atom,
    wm_delete_window: Atom,
    atom_type: Atom,
    wm_class: Atom,
    wm_name: Atom,
    string: Atom,
    utf8_string: Atom,
    net_wm_name: Atom,
    net_active_window: Atom,
}

impl Atoms {
    fn intern(conn: &RustConnection) -> Result<Atoms, ItError> {
        let one = |name: &str| -> Result<Atom, ItError> {
            Ok(conn
                .intern_atom(false, name.as_bytes())
                .map_err(x11)?
                .reply()
                .map_err(x11)?
                .atom)
        };
        Ok(Atoms {
            wm_protocols: one("WM_PROTOCOLS")?,
            wm_delete_window: one("WM_DELETE_WINDOW")?,
            atom_type: one("ATOM")?,
            wm_class: one("WM_CLASS")?,
            wm_name: one("WM_NAME")?,
            string: one("STRING")?,
            utf8_string: one("UTF8_STRING")?,
            net_wm_name: one("_NET_WM_NAME")?,
            net_active_window: one("_NET_ACTIVE_WINDOW")?,
        })
    }
}

/// Create, name and class one window; map it too unless `map` is
/// false (a window that must exist without taking focus — the
/// focus-steal step maps it at the steal moment). No `_NET_WM_PID`
/// on purpose: capture must report "unknown pid", not invent one.
fn make_window(
    conn: &RustConnection,
    root: Window,
    atoms: &Atoms,
    label: &str,
    map: bool,
) -> Result<Window, ItError> {
    let window = conn.generate_id().map_err(x11)?;
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
    .map_err(x11)?
    .check()
    .map_err(x11)?;
    set_property8(
        conn,
        window,
        atoms.wm_class,
        atoms.string,
        b"starlingit\0StarlingIt\0",
    )?;
    let title = format!("starling x11 IT window ({label})");
    set_property8(conn, window, atoms.wm_name, atoms.string, title.as_bytes())?;
    set_property8(
        conn,
        window,
        atoms.net_wm_name,
        atoms.utf8_string,
        title.as_bytes(),
    )?;
    // Ask for a clean WM_DELETE_WINDOW close so a cooperative WM
    // offers an orderly teardown instead of killing the window.
    set_property32(
        conn,
        window,
        atoms.wm_protocols,
        atoms.atom_type,
        &[atoms.wm_delete_window],
    )?;
    if !map {
        conn.flush().map_err(x11)?;
        return Ok(window);
    }
    conn.map_window(window).map_err(x11)?.check().map_err(x11)?;
    conn.flush().map_err(x11)?;
    // Wait until the server (and any WM) made it viewable.
    let deadline = Instant::now() + SETTLE;
    loop {
        let mapped = conn
            .get_window_attributes(window)
            .map_err(x11)?
            .reply()
            .map_err(x11)?
            .map_state;
        if mapped == MapState::VIEWABLE {
            return Ok(window);
        }
        if Instant::now() > deadline {
            return Err(focus_error("window never became viewable"));
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// Set an 8-bit property (STRING-shaped ones live here).
fn set_property8(
    conn: &RustConnection,
    window: Window,
    property: Atom,
    type_: Atom,
    data: &[u8],
) -> Result<(), ItError> {
    conn.change_property(
        PropMode::REPLACE,
        window,
        property,
        type_,
        8,
        data.len() as u32,
        data,
    )
    .map_err(x11)?
    .check()
    .map_err(x11)
}

/// Set a 32-bit property (ATOM-shaped ones live here).
fn set_property32(
    conn: &RustConnection,
    window: Window,
    property: Atom,
    type_: Atom,
    data: &[u32],
) -> Result<(), ItError> {
    let bytes: Vec<u8> = data.iter().flat_map(|word| word.to_ne_bytes()).collect();
    conn.change_property(
        PropMode::REPLACE,
        window,
        property,
        type_,
        32,
        data.len() as u32,
        &bytes,
    )
    .map_err(x11)?
    .check()
    .map_err(x11)
}

/// Focus `window` and wait until *both* the X focus and the EWMH
/// active window agree on it — the state capture is defined over
/// (the WM may take a moment between map and activation).
fn focus_and_wait(
    conn: &RustConnection,
    root: Window,
    atoms: &Atoms,
    window: Window,
) -> Result<(), ItError> {
    conn.set_input_focus(
        x11rb::protocol::xproto::InputFocus::PARENT,
        window,
        x11rb::CURRENT_TIME,
    )
    .map_err(x11)?
    .check()
    .map_err(x11)?;
    let deadline = Instant::now() + SETTLE;
    loop {
        conn.flush().map_err(x11)?;
        let focus = conn
            .get_input_focus()
            .map_err(x11)?
            .reply()
            .map_err(x11)?
            .focus;
        let active = conn
            .get_property(false, root, atoms.net_active_window, x11rb::NONE, 0, 4)
            .map_err(x11)?
            .reply()
            .map_err(x11)?;
        let active = if active.type_ == x11rb::NONE || active.value.len() < 4 {
            None
        } else {
            Some(u32::from_ne_bytes(
                active.value[..4].try_into().expect("four bytes"),
            ))
        };
        // The focus window is the decisive one for typing; the EWMH
        // active window may also be legitimately unset (the backend's
        // no-EWMH fallback), but must never name a *different*
        // window, or capture would not see `window`.
        if focus == window && active.map_or(true, |id| id == window) {
            return Ok(());
        }
        if Instant::now() > deadline {
            return Err(focus_error(
                "focus and the EWMH active window never agreed (the WM may keep a \
                 different window active; focus and active must both land on the \
                 test window for capture identity to be well-defined)",
            ));
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// The keyboard mapping snapshot the test decodes through — refreshed
/// from the live server whenever a `MappingNotify` flows past (the
/// same discipline a target toolkit applies). Only the base/Shift
/// columns are decoded — the same columns the backend trusts.
struct Keyboard {
    min: u8,
    width: usize,
    syms: Vec<u32>,
}

impl Keyboard {
    fn load(conn: &RustConnection) -> Result<Keyboard, ItError> {
        let setup = conn.setup();
        let reply = conn
            .get_keyboard_mapping(setup.min_keycode, setup.max_keycode - setup.min_keycode + 1)
            .map_err(x11)?
            .reply()
            .map_err(x11)?;
        let width = reply.keysyms_per_keycode as usize;
        assert!(width > 0, "a live keyboard mapping has columns");
        Ok(Keyboard {
            min: setup.min_keycode,
            width,
            syms: reply.keysyms,
        })
    }

    /// Re-read the whole mapping (the `MappingNotify` reaction).
    fn reload(&mut self, conn: &RustConnection) -> Result<(), ItError> {
        *self = Keyboard::load(conn)?;
        Ok(())
    }

    /// The full mapping, for before/after equality.
    fn snapshot(&self) -> Vec<u32> {
        self.syms.clone()
    }

    /// The live mapping equals `snapshot` — asked of the server, not
    /// the local copy: the leak-free invariant after any insert.
    fn assert_mapping_is(
        &mut self,
        conn: &RustConnection,
        snapshot: &[u32],
    ) -> Result<(), ItError> {
        self.reload(conn)?;
        assert_eq!(
            self.syms, snapshot,
            "the keyboard mapping must return to exactly what it was before the insert"
        );
        Ok(())
    }

    fn syms_of(&self, keycode: u8) -> &[u32] {
        &self.syms[((keycode - self.min) as usize) * self.width..][..self.width]
    }

    /// The first keycode producing `keysym` in the trusted columns.
    fn keycode_of(&self, keysym: u32) -> Option<u8> {
        let count = self.syms.len() / self.width;
        (0..count)
            .map(|index| self.min + index as u8)
            .find(|&keycode| {
                let syms = self.syms_of(keycode);
                syms.first() == Some(&keysym) || syms.get(1) == Some(&keysym)
            })
    }

    /// `count` spare (all-`NoSymbol`) keycodes in ascending order —
    /// taken from the top, the same policy as the backend.
    fn spare_keycodes(&self, count: usize) -> Option<Vec<u8>> {
        let keycode_count = self.syms.len() / self.width;
        let spares: Vec<u8> = (0..keycode_count)
            .rev()
            .map(|index| self.min + index as u8)
            .filter(|&keycode| self.syms_of(keycode).iter().all(|&sym| sym == 0))
            .take(count)
            .collect();
        if spares.len() == count {
            let mut spares = spares;
            spares.reverse(); // descending pick, reported ascending
            Some(spares)
        } else {
            None
        }
    }

    /// Point one keycode's column 0 at `keysym` (width preserved).
    /// The caller reloads afterwards: the server canonicalizes the
    /// stored columns (see the pre-map site), so a hand-updated
    /// snapshot would not be what any later reload sees.
    fn remap(&mut self, conn: &RustConnection, keycode: u8, keysym: u32) -> Result<(), ItError> {
        let mut columns = vec![0u32; self.width];
        columns[0] = keysym;
        conn.change_keyboard_mapping(1, keycode, self.width as u8, &columns)
            .map_err(x11)?
            .check()
            .map_err(x11)
    }

    /// Decode a `(keycode, state)` pair back to the character the
    /// target saw. `None` for modifiers and anything not a character
    /// (their key events are part of typing, not the text): the XK
    /// modifier/function block is `0xfe00..=0xffff` — Unicode
    /// keysyms at `0x01000000+` are characters and must pass.
    fn decode(&self, keycode: u8, state: u16) -> Option<char> {
        let column = if state & 0x01 != 0 { 1 } else { 0 };
        let keysym = *self.syms_of(keycode).get(column)?;
        if keysym == 0 || (0xfe00..=0xffff).contains(&keysym) {
            return None;
        }
        let codepoint = if keysym >= 0x0100_0000 {
            keysym - 0x0100_0000
        } else {
            keysym
        };
        char::from_u32(codepoint)
    }
}

/// Drain this connection's event queue in arrival order (refreshing
/// the keymap copy at `MappingNotify`s), decoding `KeyPress` events
/// for `window` into text. Used for the "nothing was typed"
/// assertions; the decode-as-text assertions use
/// [`insert_and_decode`], which reads while the insert runs.
fn drain_text(conn: &RustConnection, window: Window, kb: &mut Keyboard) -> Result<String, ItError> {
    std::thread::sleep(DRAIN_GRACE);
    let mut text = String::new();
    while let Ok(Some(event)) = conn.poll_for_event() {
        match event {
            x11rb::protocol::Event::KeyPress(press) if press.event == window => {
                if let Some(character) = kb.decode(press.detail, u16::from(press.state)) {
                    text.push(character);
                }
            }
            x11rb::protocol::Event::MappingNotify(_) => kb.reload(conn)?,
            _ => {}
        }
    }
    Ok(text)
}

/// Best-effort cleanup on scope exit *and* on panic: the test mutates
/// server-global state (keyboard mapping, windows) and must leave the
/// session as it found it even when an assertion fails halfway.
struct Cleanup<'a> {
    conn: &'a RustConnection,
    windows: Vec<Window>,
    premaps: Vec<(u8, Vec<u32>)>,
}

impl<'a> Cleanup<'a> {
    fn new(conn: &'a RustConnection) -> Cleanup<'a> {
        Cleanup {
            conn,
            windows: Vec::new(),
            premaps: Vec::new(),
        }
    }

    fn run(&mut self) {
        for (keycode, original) in &self.premaps {
            if let Ok(cookie) =
                self.conn
                    .change_keyboard_mapping(1, *keycode, original.len() as u8, original)
            {
                // check() is the round trip that proves the restore
                // landed (errors swallowed: cleanup must never mask
                // the original failure).
                let _ = cookie.check();
            }
        }
        for window in &self.windows {
            let _ = self.conn.destroy_window(*window);
        }
        let _ = self.conn.flush();
    }
}

impl Drop for Cleanup<'_> {
    fn drop(&mut self) {
        self.run();
    }
}
