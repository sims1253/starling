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
//!    3c. **Preparation failure restores what it borrowed**: one more
//!    unmapped character than the keyboard has spare keycodes makes
//!    borrowing stop part-way — the insert refuses (keyboard busy),
//!    types nothing, and the live mapping still ends exactly where it
//!    started, proving cleanup on the preparation-failure path runs
//!    (explicitly, not lost in Drop).
//! 4. **Held modifiers refuse typing**: Ctrl held down via XTest
//!    during an insert makes the insert wait out the bounded release
//!    window and then refuse with `ModifiersHeld` — nothing typed,
//!    and no modifier was released behind the user's back.
//! 4b. **A mapping that drifts between the chunk checks and the
//!    press stops typing before the key-down**: with Ctrl held (the
//!    insert waits), the test itself re-maps the exact keycode a plan
//!    is about to press to `Return` — once the borrowed spare of an
//!    unmapped character (bare refusal, nothing typed), once a
//!    pre-mapped letter's key after a first character was already
//!    delivered (partial delivery naming the keycode). After Ctrl is
//!    released the insert must refuse with a keyboard-busy cause, the
//!    window must have seen no `Return`/`KP_Enter` key at all, and
//!    the foreign mapping must survive untouched for the test to
//!    restore — the no-Enter rule cannot be broken by another
//!    client's hand.
//! 4c. **A remap aimed at the pressed key during its hold is blocked
//!    by the keystroke grab** (on a server that honors `GrabServer`):
//!    the payload is one unmapped character riding the predicted
//!    borrowed spare, and the moment this connection observes that
//!    keycode's `KeyPress` — the key is down right now, inside its
//!    3 ms hold — the test re-maps that same *pressed* keycode to
//!    `Return` (a second client as far as the server is concerned).
//!    Xlib translates `KeyRelease` events through the live mapping
//!    too and apps can act on a release, so an unguarded hold would
//!    let the window's `KeyRelease` decode as Return. The grab the
//!    backend holds through the whole keystroke must keep the
//!    foreign request unprocessed until the key-up: the `KeyRelease`
//!    decodes as the original character, no `MappingNotify` lands
//!    between the press and the release, the foreign remap takes
//!    effect only after the release, and no Return/KP_Enter press or
//!    release ever reaches the window. The round first *probes*
//!    whether this X server actually curtails other clients'
//!    requests during `GrabServer`: WSLg's XWayland does not
//!    (measured — a second connection round-trips freely mid-grab),
//!    and on such a server no client-side mechanism can keep a
//!    hostile remap from landing mid-hold, so the round degrades to
//!    the press-side guarantees (the press decodes as the character;
//!    no Return *press* reaches the window) and prints the gap.
//! 4d. **A Shift keycode whose unshifted column stopped being Shift
//!    is refused**: during the held-Ctrl wait the test re-maps the
//!    planned Shift keycode to `[Return, Shift_L]` — still bound to
//!    the Shift modifier, still carrying a Shift keysym in column 1,
//!    exactly the shape a keysym-in-either-column check lets
//!    through. Starling presses Shift with no other modifier down
//!    (column 0), so the press would emit Return; the insert must
//!    refuse before the Shift key goes down, naming the keycode,
//!    with no Return reaching the window, and the test restores the
//!    real Shift mapping itself (a cleanup guard holds it too, so
//!    even a failure leaves the session keyboard as it found it).
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
/// `XK_Return` and `XK_KP_Enter` — the keysyms the mapping-drift
/// regression must never let reach the window, however the drift
/// happened.
const XK_RETURN: u32 = 0xff0d;
const XK_KP_ENTER: u32 = 0xff8b;
/// `XK_Shift_L` / `XK_Shift_R` — the keysyms a usable Shift keycode
/// carries in its *unshifted* column, the level the backend's own
/// Shift press decodes through.
const XK_SHIFT_L: u32 = 0xffe1;
const XK_SHIFT_R: u32 = 0xffe2;

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

    // 3c. Preparation failure after a borrow: one more unmapped
    //     character than the keyboard has spare keycodes means
    //     borrowing stops part-way — the last character finds no
    //     candidate. The insert must refuse (keyboard busy), type
    //     nothing, and still give back every keycode borrowed before
    //     the failure (the mapping ends byte-for-byte where it
    //     started).
    let spares_left = spare_count(&kb);
    let overflow: String = unmapped_chars(&kb, spares_left + 1).iter().collect();
    match backend.insert(&snap_a, &overflow) {
        Err(InsertError::Rejected { reason }) => {
            assert!(
                reason.contains("keyboard busy"),
                "the refusal says why: {reason}"
            );
        }
        other => panic!("an exhausted spare pool must refuse the insert: {other:?}"),
    }
    assert!(
        drain_text(&conn, a, &mut kb)?.is_empty(),
        "a preparation refusal may not type"
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

    // 4b. A mapping that drifts between the chunk checks and the
    //     press must stop typing before the key-down. The chunk
    //     checks (held modifiers, keyboard state, identity) never
    //     read the keycode mappings, so a foreign client can flip the
    //     exact keycode a plan is about to press — during the
    //     held-modifier wait, say — and a blind press would type
    //     whatever the *new* mapping decodes to: a foreign `Return`
    //     is the no-Enter rule broken by another client's hand. Both
    //     plan kinds are pinned: a borrowed spare re-mapped to
    //     `Return` (bare refusal, nothing typed) and a pre-mapped
    //     letter re-mapped to `Return` after a first character was
    //     already delivered (partial delivery naming the keycode). In
    //     both, the window sees no Return/KP_Enter at all, Starling
    //     leaves the foreign mapping exactly as the interloper wrote
    //     it, and the test restores the session itself.
    {
        // The keycode the backend will borrow for an unmapped
        // character: the highest live all-`NoSymbol` one — the same
        // top-down policy the backend's spare pool draws from, so the
        // prediction is exact.
        let borrowed = kb
            .spare_keycodes(1)
            .ok_or_else(|| focus_error("this keyboard has no spare keycode to borrow"))?[0];
        let control_keysym = *kb
            .syms_of(control)
            .first()
            .expect("the Control keycode carries its keysym");

        // Variant A: the borrowed keycode itself is re-mapped to
        // Return while the insert waits out the held Ctrl. The payload
        // is the single unmapped character that rides it, so the
        // refusal must come before its key-down: a bare keyboard-busy
        // rejection naming the keycode, nothing typed.
        cleanup.premaps.push((borrowed, vec![0; kb.width]));
        xtest_press(&conn, control)?;
        let payload_a = unmapped.to_string();
        let (result, pressed) = drift_round(
            &conn,
            a,
            &mut kb,
            || {
                let backend_thread = X11Backend::new();
                let snapshot = snap_a.clone();
                let payload = payload_a.clone();
                std::thread::spawn(move || backend_thread.insert(&snapshot, &payload))
            },
            borrowed,
            keysym_of(unmapped),
            |conn, kb| {
                // The foreign remap: every column of the borrowed
                // keycode becomes Return (what an interloper's "make
                // this key Enter" looks like), then Ctrl is released
                // so typing proceeds straight into the drifted
                // mapping.
                let columns = vec![XK_RETURN; kb.width];
                conn.change_keyboard_mapping(1, borrowed, kb.width as u8, &columns)
                    .map_err(x11)?
                    .check()
                    .map_err(x11)?;
                xtest_release(conn, control)
            },
        )?;
        match result {
            Err(InsertError::Rejected { reason }) => {
                assert!(
                    reason.contains("keyboard busy"),
                    "the refusal says why: {reason}"
                );
                assert!(
                    reason.contains(&format!("keycode {borrowed}")),
                    "the refusal names the drifted keycode: {reason}"
                );
            }
            other => {
                panic!("a drifted borrowed keycode must refuse before any key-down: {other:?}")
            }
        }
        assert!(
            !pressed.contains(&XK_RETURN) && !pressed.contains(&XK_KP_ENTER),
            "no Return or KP_Enter may reach the window (saw {pressed:?})"
        );
        assert_eq!(
            pressed,
            vec![control_keysym],
            "nothing but the held Ctrl reached the window"
        );
        kb.reload(&conn)?;
        assert_eq!(
            kb.syms_of(borrowed).first(),
            Some(&XK_RETURN),
            "Starling must leave the interloper's mapping exactly as it was written"
        );
        // The test cleans up its own foreign mapping (all-`NoSymbol`
        // was there before the interloper touched it).
        let zeros = vec![0u32; kb.width];
        conn.change_keyboard_mapping(1, borrowed, kb.width as u8, &zeros)
            .map_err(x11)?
            .check()
            .map_err(x11)?;
        kb.assert_mapping_is(&conn, &mapping_before)?;

        // Variant B: a *pre-mapped* keycode drifts. The payload's
        // first character rides the borrow (delivered normally — its
        // mapping is untouched), the second is a plain layout letter
        // whose keycode the interloper re-maps to Return during the
        // same wait: typing stops between the two, reporting exactly
        // one character may have landed and a keyboard-busy cause
        // naming the letter's keycode.
        let letter = ['q', 'x', 'z', 'v', 'k', 'j']
            .iter()
            .copied()
            .find(|&candidate| {
                kb.keycode_of(keysym_of(candidate)).is_some_and(|keycode| {
                    kb.syms_of(keycode).first() == Some(&keysym_of(candidate))
                })
            })
            .ok_or_else(|| focus_error("this keyboard has no plain unshifted letter to drift"))?;
        let letter_keycode = kb
            .keycode_of(keysym_of(letter))
            .expect("the find above proved the letter is mapped");
        let letter_original = kb.syms_of(letter_keycode).to_vec();
        cleanup
            .premaps
            .push((letter_keycode, letter_original.clone()));
        xtest_press(&conn, control)?;
        let payload_b = format!("{unmapped}{letter}");
        let (result, pressed) = drift_round(
            &conn,
            a,
            &mut kb,
            || {
                let backend_thread = X11Backend::new();
                let snapshot = snap_a.clone();
                let payload = payload_b.clone();
                std::thread::spawn(move || backend_thread.insert(&snapshot, &payload))
            },
            borrowed,
            keysym_of(unmapped),
            |conn, kb| {
                let columns = vec![XK_RETURN; kb.width];
                conn.change_keyboard_mapping(1, letter_keycode, kb.width as u8, &columns)
                    .map_err(x11)?
                    .check()
                    .map_err(x11)?;
                xtest_release(conn, control)
            },
        )?;
        match result {
            Err(InsertError::PartialDelivery {
                delivered_chars,
                total_chars,
                cause,
            }) => {
                assert_eq!(
                    delivered_chars, 1,
                    "exactly the first character may have landed"
                );
                assert_eq!(total_chars, payload_b.chars().count());
                match *cause {
                    InsertError::Rejected { reason } => {
                        assert!(
                            reason.contains("keyboard busy"),
                            "the stop says why: {reason}"
                        );
                        assert!(
                            reason.contains(&format!("keycode {letter_keycode}")),
                            "the stop names the drifted keycode: {reason}"
                        );
                    }
                    other => panic!("the drift must be the stop cause, not {other:?}"),
                }
            }
            other => panic!(
                "a drifted pre-mapped keycode after a delivered character must stop partially: \
                 {other:?}"
            ),
        }
        assert!(
            !pressed.contains(&XK_RETURN) && !pressed.contains(&XK_KP_ENTER),
            "no Return or KP_Enter may reach the window (saw {pressed:?})"
        );
        assert_eq!(
            pressed,
            vec![control_keysym, keysym_of(unmapped)],
            "only the held Ctrl and the first character were pressed — never the drifted key"
        );
        kb.reload(&conn)?;
        assert_eq!(
            kb.syms_of(letter_keycode).first(),
            Some(&XK_RETURN),
            "Starling must leave the interloper's mapping on the pre-mapped key too"
        );
        // Restore the letter's real columns, then prove the whole
        // mapping is back at the baseline before the later rounds run.
        conn.change_keyboard_mapping(
            1,
            letter_keycode,
            letter_original.len() as u8,
            &letter_original,
        )
        .map_err(x11)?
        .check()
        .map_err(x11)?;
        kb.assert_mapping_is(&conn, &mapping_before)?;
    }

    // 4c. A remap aimed at the *pressed* key during its hold: the
    //     keystroke grab must keep it out. The payload is one
    //     unmapped character riding the predicted borrowed spare; the
    //     moment this connection observes that keycode's KeyPress —
    //     while the key is still down, inside KEY_HOLD — the test (a
    //     second client as far as the grab is concerned) re-maps the
    //     pressed keycode to Return. On a server that honors
    //     GrabServer's exclusion, the backend's whole-keystroke grab
    //     keeps the foreign request unprocessed until the key-up: the
    //     window's KeyRelease decodes as the original character, no
    //     MappingNotify lands between the press and the release, and
    //     the foreign remap takes effect only after the release. On a
    //     server that does not honor the exclusion (WSLg's XWayland,
    //     measured by the probe below), *no* client-side mechanism
    //     can keep a hostile remap from landing mid-hold — the round
    //     then degrades to what such a server can still prove (the
    //     press decodes as the character and no Return *press* ever
    //     reaches the window) and says so loudly.
    let grab_excluded = grab_exclusion_honored(&conn)?;
    if !grab_excluded {
        println!(
            "starling-insertion X11 IT: this X server does not curtail other clients' \
             requests during GrabServer (WSLg's XWayland); the mid-hold remap round \
             degrades to the press-side guarantees — the release-decode guarantee is \
             a server property this box does not provide"
        );
    }
    {
        let borrowed = kb
            .spare_keycodes(1)
            .ok_or_else(|| focus_error("this keyboard has no spare keycode to borrow"))?[0];
        // The failure guard: whatever happens below, the borrowed
        // keycode ends the test all-`NoSymbol` again (the interloper
        // takes it from Starling's borrow, whose baseline was zeros).
        cleanup.premaps.push((borrowed, vec![0; kb.width]));
        let payload_c = unmapped.to_string();
        let (result, seen, interfered) = hold_round(
            &conn,
            a,
            &mut kb,
            || {
                let backend_thread = X11Backend::new();
                let snapshot = snap_a.clone();
                let payload = payload_c.clone();
                std::thread::spawn(move || backend_thread.insert(&snapshot, &payload))
            },
            borrowed,
            |conn, kb| {
                // The foreign remap of the key that is down right
                // now: every column Return — what an interloper's
                // "make this key Enter" looks like. The check()
                // round trip can only complete once the server has
                // processed the request, so its timing relative to
                // the keystroke is exactly what the Seen order
                // asserts.
                let columns = vec![XK_RETURN; kb.width];
                conn.change_keyboard_mapping(1, borrowed, kb.width as u8, &columns)
                    .map_err(x11)?
                    .check()
                    .map_err(x11)?;
                Ok(())
            },
        )?;
        assert!(interfered, "the foreign remap must fire during the hold");
        let receipt = result.map_err(|e| x11(format!("inserting {unmapped}: {e}")))?;
        assert_eq!(receipt.evidence, EVIDENCE_SYNTHETIC_KEYS);
        let char_keysym = keysym_of(unmapped);
        let press = seen
            .iter()
            .position(|seen| matches!(seen, Seen::Press(keycode, _) if *keycode == borrowed))
            .unwrap_or_else(|| panic!("the character's key went down: {seen:?}"));
        assert_eq!(
            seen[press],
            Seen::Press(borrowed, char_keysym),
            "the press decodes as the character (the press always precedes the interloper's \
             trigger, so this holds even without grab exclusion): {seen:?}"
        );
        let release = seen
            .iter()
            .position(|seen| matches!(seen, Seen::Release(keycode, _) if *keycode == borrowed))
            .unwrap_or_else(|| panic!("the character's key came back up: {seen:?}"));
        assert!(press < release, "the press precedes the release: {seen:?}");
        assert!(
            seen.iter().all(|seen| match seen {
                Seen::Press(_, keysym) => *keysym != XK_RETURN && *keysym != XK_KP_ENTER,
                _ => true,
            }),
            "no Return or KP_Enter press may ever reach the window: {seen:?}"
        );
        if grab_excluded {
            // The keystroke grab held: the interloper's request sat
            // unprocessed until the ungrab, which the backend sends
            // only after the key-up — so the release decodes as the
            // character and the MappingNotify lands after it.
            assert_eq!(
                seen[release],
                Seen::Release(borrowed, char_keysym),
                "the release decodes as the character, never the interloper's Return: {seen:?}"
            );
            assert!(
                !seen[press + 1..release]
                    .iter()
                    .any(|seen| matches!(seen, Seen::Map)),
                "no mapping change may take effect while the pressed key is down: {seen:?}"
            );
            assert!(
                seen.iter()
                    .enumerate()
                    .any(|(index, seen)| { index > release && matches!(seen, Seen::Map) }),
                "the foreign remap lands (only after the release): {seen:?}"
            );
            assert!(
                seen.iter().all(|seen| match seen {
                    Seen::Press(_, keysym) | Seen::Release(_, keysym) => {
                        *keysym != XK_RETURN && *keysym != XK_KP_ENTER
                    }
                    Seen::Map => true,
                }),
                "no Return or KP_Enter press or release may reach the window: {seen:?}"
            );
        }
        kb.reload(&conn)?;
        assert_eq!(
            kb.syms_of(borrowed).first(),
            Some(&XK_RETURN),
            "Starling must leave the interloper's mapping exactly as it was written"
        );
        // Give the borrowed keycode back to the spare pool (it was
        // all-`NoSymbol` before the interloper touched it).
        let zeros = vec![0u32; kb.width];
        conn.change_keyboard_mapping(1, borrowed, kb.width as u8, &zeros)
            .map_err(x11)?
            .check()
            .map_err(x11)?;
        kb.assert_mapping_is(&conn, &mapping_before)?;
    }

    // 4d. The Shift keycode's *effective* keysym: during the same
    //     held-Ctrl wait the interloper re-maps the planned Shift
    //     keycode to `[Return, Shift_L]` — column 1 still carries a
    //     Shift keysym and the modifier binding is untouched, so only
    //     the column the press itself decodes through can catch it.
    //     Starling presses Shift with no modifier down (column 0), so
    //     the press would emit Return; the insert must refuse before
    //     the Shift key goes down, naming the keycode. The cleanup
    //     guard holds the real Shift columns from before the attack,
    //     so even a failure leaves the session keyboard as it was
    //     found.
    {
        let shift_keycode = shift_keycode_of(&conn, &kb)?;
        let shift_original = kb.syms_of(shift_keycode).to_vec();
        cleanup
            .premaps
            .push((shift_keycode, shift_original.clone()));
        let borrowed = kb
            .spare_keycodes(1)
            .ok_or_else(|| focus_error("this keyboard has no spare keycode to borrow"))?[0];
        let control_keysym = *kb
            .syms_of(control)
            .first()
            .expect("the Control keycode carries its keysym");
        xtest_press(&conn, control)?;
        let payload_d = format!("{unmapped}C");
        let (result, pressed) = drift_round(
            &conn,
            a,
            &mut kb,
            || {
                let backend_thread = X11Backend::new();
                let snapshot = snap_a.clone();
                let payload = payload_d.clone();
                std::thread::spawn(move || backend_thread.insert(&snapshot, &payload))
            },
            borrowed,
            keysym_of(unmapped),
            |conn, _kb| {
                // `[Return, Shift_L]`, written as exactly two
                // columns: the attack shape — a Shift keysym is still
                // present (column 1) and the keycode is still
                // Shift-bound, but the unshifted press Starling is
                // about to make decodes as Return. Exactly two
                // columns, not the keycode's full width, because a
                // write padded with trailing zeros makes this
                // server's XKB canonicalization widen the *global*
                // keymap width, which re-pads every keycode's
                // columns and would drift the borrowed keycode's
                // echo comparison for a reason that has nothing to
                // do with the Shift attack (measured).
                conn.change_keyboard_mapping(1, shift_keycode, 2, &[XK_RETURN, XK_SHIFT_L])
                    .map_err(x11)?
                    .check()
                    .map_err(x11)?;
                xtest_release(conn, control)
            },
        )?;
        match result {
            Err(InsertError::PartialDelivery {
                delivered_chars,
                total_chars,
                cause,
            }) => {
                assert_eq!(
                    delivered_chars, 1,
                    "exactly the first character may have landed"
                );
                assert_eq!(total_chars, payload_d.chars().count());
                match *cause {
                    InsertError::Rejected { reason } => {
                        assert!(
                            reason.contains("keyboard busy"),
                            "the refusal says why: {reason}"
                        );
                        assert!(
                            reason.contains(&format!("keycode {shift_keycode}")),
                            "the refusal names the Shift keycode: {reason}"
                        );
                    }
                    other => panic!("the Shift drift must be the stop cause, not {other:?}"),
                }
            }
            other => panic!(
                "a Shift keycode whose unshifted column is no longer Shift must refuse before \
                 its press: {other:?}"
            ),
        }
        assert!(
            !pressed.contains(&XK_RETURN) && !pressed.contains(&XK_KP_ENTER),
            "no Return or KP_Enter may reach the window (saw {pressed:?})"
        );
        assert_eq!(
            pressed,
            vec![control_keysym, keysym_of(unmapped)],
            "the Shift key itself never went down"
        );
        kb.reload(&conn)?;
        assert_eq!(
            kb.syms_of(shift_keycode).first(),
            Some(&XK_RETURN),
            "Starling must leave the interloper's mapping on the Shift key too"
        );
        // Restore the real Shift columns and prove the whole mapping
        // is back at the baseline before the later rounds run.
        conn.change_keyboard_mapping(
            1,
            shift_keycode,
            shift_original.len() as u8,
            &shift_original,
        )
        .map_err(x11)?
        .check()
        .map_err(x11)?;
        kb.assert_mapping_is(&conn, &mapping_before)?;
    }

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
         ({upper_text:?}), preparation-failure refusal with a restored mapping ({} spare \
         keycodes exhausted), held-modifier refusal, mapping-drift refusal (borrowed and \
         pre-mapped keycodes re-mapped to Return mid-wait: no Enter reached the window, the \
         foreign mappings survived for the test to restore), remap-during-the-hold blocked \
         by the keystroke grab (grab exclusion {}, the release still decoded as the character \
         where the server provides it, the foreign remap landed only after it), Shift \
         effective-keysym refusal (a [Return, Shift_L] Shift key never pressed), chunk-bounded partial delivery \
         on focus steal, serialized concurrent inserts, changed and gone revalidation",
        spares_left,
        if grab_excluded { "honored" } else { "not honored by this server" },
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

/// How many all-`NoSymbol` keycodes the live mapping holds — the same
/// pool, counted the same way, the backend's borrow loop draws from.
fn spare_count(kb: &Keyboard) -> usize {
    let keycode_count = kb.syms.len() / kb.width;
    (0..keycode_count)
        .map(|index| kb.min + index as u8)
        .filter(|&keycode| kb.syms_of(keycode).iter().all(|&sym| sym == 0))
        .count()
}

/// `count` distinct characters no trusted column of the live mapping
/// produces — astral codepoints (each with its Unicode-convention
/// keysym, which no real layout carries), so every one must ride the
/// remap path.
fn unmapped_chars(kb: &Keyboard, count: usize) -> Vec<char> {
    let mut found = Vec::with_capacity(count);
    let mut codepoint = 0x2_0000u32;
    while found.len() < count {
        let character = char::from_u32(codepoint)
            .expect("codepoints past the astral plane boundary are all valid chars");
        if kb.keycode_of(0x0100_0000 | codepoint).is_none() {
            found.push(character);
        }
        codepoint += 1;
    }
    found
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

/// The live keysyms of one keycode, straight from the server — the
/// drift round's poll of the backend's borrow.
fn live_keycode_syms(conn: &RustConnection, keycode: u8) -> Result<Vec<u32>, ItError> {
    Ok(conn
        .get_keyboard_mapping(keycode, 1)
        .map_err(x11)?
        .reply()
        .map_err(x11)?
        .keysyms)
}

/// One mapping-drift round (the regression for the pre-press
/// verification): the insert (`spawn`) runs while Ctrl is physically
/// held; this connection watches for the backend's borrow to land on
/// `borrow_keycode` (its base column carrying `borrow_keysym` —
/// preparation is done by then, and the insert is parked in the
/// held-modifier wait, the exact exposure window the regression
/// targets), then runs `interfere` — the foreign remap, plus the Ctrl
/// release that lets typing proceed into the drifted mapping — and
/// finally drains the stream until the insert returns. Every
/// `KeyPress` for `window` is collected as its event-time keysym
/// (modifier keysyms included, characters and all), so the caller can
/// assert exactly which keys the window saw. Modeled on
/// [`insert_and_decode`]: the join always happens, and listener-side
/// errors surface after it, never by orphaning the typing thread.
#[allow(clippy::type_complexity)]
fn drift_round(
    conn: &RustConnection,
    window: Window,
    kb: &mut Keyboard,
    spawn: impl FnOnce() -> std::thread::JoinHandle<Result<InsertReceipt, InsertError>>,
    borrow_keycode: u8,
    borrow_keysym: u32,
    interfere: impl FnOnce(&RustConnection, &mut Keyboard) -> Result<(), ItError>,
) -> Result<(Result<InsertReceipt, InsertError>, Vec<u32>), ItError> {
    fn pump(
        conn: &RustConnection,
        window: Window,
        kb: &mut Keyboard,
        pressed: &mut Vec<u32>,
        listener_error: &mut Option<ItError>,
    ) {
        while let Some(event) = conn.poll_for_event().unwrap() {
            match event {
                x11rb::protocol::Event::KeyPress(press) if press.event == window => {
                    // Event-time keysym: the column the event's own
                    // modifier state selects, decoded through the
                    // keymap copy current when the event is read.
                    let column = if u16::from(press.state) & 0x01 != 0 {
                        1
                    } else {
                        0
                    };
                    if let Some(&keysym) = kb.syms_of(press.detail).get(column) {
                        pressed.push(keysym);
                    }
                }
                x11rb::protocol::Event::MappingNotify(_) => {
                    if let Err(error) = kb.reload(conn) {
                        listener_error.get_or_insert(error);
                    }
                }
                _ => {}
            }
        }
    }
    let handle = spawn();
    let deadline = Instant::now() + INSERT_BUDGET;
    let mut pressed: Vec<u32> = Vec::new();
    let mut listener_error: Option<ItError> = None;
    let mut interfere = Some(interfere);
    let mut interfered = false;
    loop {
        pump(conn, window, kb, &mut pressed, &mut listener_error);
        if !interfered {
            // The borrow's write and echo-read both run inside the
            // backend's server grab, so a poll that sees the keysym
            // necessarily runs after the whole borrow — the interloper
            // cannot race the echo.
            let landed =
                live_keycode_syms(conn, borrow_keycode)?.first().copied() == Some(borrow_keysym);
            if landed {
                if let Some(interfere) = interfere.take() {
                    if let Err(error) = interfere(conn, kb) {
                        listener_error.get_or_insert(error);
                    }
                }
                interfered = true;
            }
        }
        if handle.is_finished() {
            // Stragglers: events the server queued before the insert
            // returned still count as "what the target saw".
            std::thread::sleep(DRAIN_GRACE);
            pump(conn, window, kb, &mut pressed, &mut listener_error);
            break;
        }
        if Instant::now() > deadline {
            return Err(focus_error(
                "the drifted insert did not finish within the test budget",
            ));
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    let result = handle.join().expect("an inserting thread must not panic");
    match listener_error {
        Some(error) => Err(error),
        None => Ok((result, pressed)),
    }
}

/// One observed keyboard event of the remap-during-the-hold
/// regression, in stream order: the `(keycode, event-time keysym)`
/// of a `KeyPress`/`KeyRelease` for the window (the keysym is the
/// column the event's own modifier state selects, decoded through
/// the keymap copy current when the event is read — which is the
/// decode a target performing the same reads would produce), or a
/// `MappingNotify` (the copy is refreshed before anything later
/// decodes). The stream order is the server's generation order, so
/// "a mapping change took effect between the press and the
/// release" is decidable from this sequence alone — and carrying
/// the keycode separately matters exactly when a foreign remap won
/// the race: the release of the pressed keycode then decodes as the
/// *foreign* keysym, which is the regression's failure signature.
#[derive(Debug, PartialEq, Clone, Copy)]
enum Seen {
    Press(u8, u32),
    Release(u8, u32),
    Map,
}

/// One remap-during-the-hold round (the regression for the
/// whole-keystroke server grab): the insert (`spawn`) runs while
/// this thread consumes this connection's event stream as it
/// arrives, recording [`Seen`] in order. The moment the `KeyPress`
/// of `watch_keycode` is observed — that key is down right now,
/// inside its hold — `interfere` runs: the foreign remap of that
/// same pressed keycode, issued and round-tripped from this
/// connection (a second client as far as the server grab is
/// concerned; the round trip completing at all proves the request
/// was accepted — when it completed relative to the keystroke is
/// what the [`Seen`] order asserts). Modeled on [`drift_round`]: the
/// join always happens, and listener-side errors surface after it,
/// never by orphaning the typing thread.
#[allow(clippy::type_complexity)]
fn hold_round(
    conn: &RustConnection,
    window: Window,
    kb: &mut Keyboard,
    spawn: impl FnOnce() -> std::thread::JoinHandle<Result<InsertReceipt, InsertError>>,
    watch_keycode: u8,
    interfere: impl FnOnce(&RustConnection, &mut Keyboard) -> Result<(), ItError>,
) -> Result<(Result<InsertReceipt, InsertError>, Vec<Seen>, bool), ItError> {
    fn pump<F>(
        conn: &RustConnection,
        window: Window,
        kb: &mut Keyboard,
        seen: &mut Vec<Seen>,
        watch_keycode: u8,
        interfere: &mut Option<F>,
        interfered: &mut bool,
        listener_error: &mut Option<ItError>,
    ) where
        F: FnOnce(&RustConnection, &mut Keyboard) -> Result<(), ItError>,
    {
        while let Some(event) = conn.poll_for_event().unwrap() {
            match event {
                x11rb::protocol::Event::KeyPress(press) if press.event == window => {
                    // Event-time keysym: the column the event's own
                    // modifier state selects, decoded through the
                    // keymap copy current when the event is read.
                    let column = if u16::from(press.state) & 0x01 != 0 {
                        1
                    } else {
                        0
                    };
                    if let Some(&keysym) = kb.syms_of(press.detail).get(column) {
                        seen.push(Seen::Press(press.detail, keysym));
                    }
                    if press.detail == watch_keycode && !*interfered {
                        *interfered = true;
                        if let Some(interfere) = interfere.take() {
                            if let Err(error) = interfere(conn, kb) {
                                listener_error.get_or_insert(error);
                            }
                        }
                    }
                }
                x11rb::protocol::Event::KeyRelease(release) if release.event == window => {
                    let column = if u16::from(release.state) & 0x01 != 0 {
                        1
                    } else {
                        0
                    };
                    if let Some(&keysym) = kb.syms_of(release.detail).get(column) {
                        seen.push(Seen::Release(release.detail, keysym));
                    }
                }
                x11rb::protocol::Event::MappingNotify(_) => {
                    if let Err(error) = kb.reload(conn) {
                        listener_error.get_or_insert(error);
                    }
                    seen.push(Seen::Map);
                }
                _ => {}
            }
        }
    }
    let handle = spawn();
    let deadline = Instant::now() + INSERT_BUDGET;
    let mut seen: Vec<Seen> = Vec::new();
    let mut listener_error: Option<ItError> = None;
    let mut interfere = Some(interfere);
    let mut interfered = false;
    loop {
        pump(
            conn,
            window,
            kb,
            &mut seen,
            watch_keycode,
            &mut interfere,
            &mut interfered,
            &mut listener_error,
        );
        if handle.is_finished() {
            // Stragglers: events the server queued before the insert
            // returned still count as "what the target saw" — the
            // foreign remap's MappingNotify included, when the
            // interference fired during the hold.
            std::thread::sleep(DRAIN_GRACE);
            pump(
                conn,
                window,
                kb,
                &mut seen,
                watch_keycode,
                &mut interfere,
                &mut interfered,
                &mut listener_error,
            );
            break;
        }
        if Instant::now() > deadline {
            return Err(focus_error(
                "the hold-round insert did not finish within the test budget",
            ));
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    let result = handle.join().expect("an inserting thread must not panic");
    match listener_error {
        Some(error) => Err(error),
        None => Ok((result, seen, interfered)),
    }
}

/// Whether this X server honors `GrabServer`'s contract of
/// curtailing the processing of other clients' requests: a second
/// connection round-trips one plain request during a grab held by
/// this connection across a short sleep — the exact shape the
/// backend's keystroke grab has (the grabber sleeps KEY_HOLD between
/// requests). On a conforming server the reply cannot come back
/// before the ungrab (measured: stock Xvfb blocks the probe for the
/// whole grab); a server that ignores the exclusion — WSLg's
/// XWayland does, measured — answers in microseconds. The probe
/// touches no keyboard state, so it cannot dirty the session.
fn grab_exclusion_honored(conn: &RustConnection) -> Result<bool, ItError> {
    let (other, _) =
        x11rb::connect(None).map_err(|e| x11(format!("connecting a probe connection: {e}")))?;
    let probe = std::thread::spawn(move || -> Result<Duration, ItError> {
        let sent = Instant::now();
        other.get_input_focus().map_err(x11)?.reply().map_err(x11)?;
        Ok(sent.elapsed())
    });
    conn.grab_server().map_err(x11)?.check().map_err(x11)?;
    std::thread::sleep(Duration::from_millis(150));
    conn.ungrab_server().map_err(x11)?;
    conn.flush().map_err(x11)?;
    let round_trip = probe.join().expect("the probe thread must not panic")?;
    Ok(round_trip >= Duration::from_millis(100))
}

/// The keycode the backend will press as Shift for a column-1
/// character: the first keycode of the live Shift modifier row whose
/// *unshifted* column carries Shift_L/Shift_R — the same
/// effective-column rule the backend's planning and pre-press
/// verification apply, so the test remaps the exact key the plan
/// will press.
fn shift_keycode_of(conn: &RustConnection, kb: &Keyboard) -> Result<u8, ItError> {
    let reply = conn
        .get_modifier_mapping()
        .map_err(x11)?
        .reply()
        .map_err(x11)?;
    let per_row = reply.keycodes_per_modifier() as usize;
    if per_row == 0 {
        return Err(focus_error(
            "the server reported an unusable modifier mapping",
        ));
    }
    let count = kb.syms.len() / kb.width;
    let (min, max) = (kb.min as usize, kb.min as usize + count - 1);
    reply.keycodes[0..per_row]
        .iter()
        .copied()
        .find(|&keycode| {
            let keycode = keycode as usize;
            keycode >= min
                && keycode <= max
                && matches!(
                    kb.syms_of(keycode as u8).first(),
                    Some(&XK_SHIFT_L) | Some(&XK_SHIFT_R)
                )
        })
        .ok_or_else(|| focus_error("this keyboard has no usable Shift keycode"))
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
