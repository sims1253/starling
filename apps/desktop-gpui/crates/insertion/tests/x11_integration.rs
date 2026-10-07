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
//! 2. **A mixed string lands as keys**: insert ASCII (including a
//!    Shift-level `C` and `!`), `é`, `ß` and `→` — the last three
//!    pre-mapped by the test onto spare keycodes, so the backend's
//!    find-in-mapping path is exercised on real server state — then
//!    read the `KeyPress` events the window received and decode them
//!    through the keyboard mapping. The decoded text must equal the
//!    inserted text exactly: no Enter, no Tab, no dropped or
//!    synthesized characters.
//! 3. **The remap path types and restores**: a character no key
//!    produces (🦄) rides the temporarily-remapped spare keycode, and
//!    the mapping is restored afterwards — checked on the live
//!    server, not inferred from the backend's return.
//! 4. **Focus safety**: focusing a second window makes revalidate of
//!    the first snapshot report `TargetChanged` (naming the new
//!    target), makes `insert` into the old snapshot refuse, and types
//!    nothing; destroying the second window makes its snapshot
//!    report `Gone`.
//! 5. **Control characters never type**: a `\n` payload is refused
//!    with `MultilineUnsupported` and produces no key events at all.

#![cfg(target_os = "linux")]

use std::time::{Duration, Instant};

use x11rb::connection::Connection;
use x11rb::protocol::xproto::{
    Atom, ConnectionExt, CreateWindowAux, EventMask, MapState, PropMode, Window, WindowClass,
};
use x11rb::rust_connection::RustConnection;

use starling_insertion::x11::X11Backend;
use starling_insertion::{
    BackendKind, Inserter, InsertError, TargetCheck, EVIDENCE_SYNTHETIC_KEYS,
};

/// How long to wait for the WM (focus, active-window bookkeeping) or
/// the server to settle before an assertion gives up.
const SETTLE: Duration = Duration::from_secs(5);
/// Grace before draining the event queue: insert has already round
/// tripped, this only covers delivery to *this* connection.
const DRAIN_GRACE: Duration = Duration::from_millis(150);

/// The string every decode round trip must survive: ASCII across both
/// shift levels plus the three pre-mapped characters.
const MIXED: &str = "Café ß → 42!";
/// A character no keyboard produces, for the remap path.
const UNMAPPED: char = '🦄';

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

    let a = make_window(&conn, root, &atoms, "first")?;
    cleanup.windows.push(a);

    // A WM activates each window as it maps, so B is only created
    // once A's part of the walk is done (see step 4).
    focus_and_wait(&conn, root, &atoms, a)?;

    // The pre-map: é/ß/→ onto spare keycodes so the backend's
    // find-in-mapping path meets them on real server state (the
    // remap path is exercised separately by UNMAPPED). Originals are
    // recorded for cleanup so the session's keyboard leaves the test
    // exactly as it came in.
    let spares = kb
        .spare_keycodes(3)
        .ok_or_else(|| focus_error("this keyboard has no three spare keycodes to pre-map"))?;
    for (keycode, character) in spares.iter().zip(['é', 'ß', '→']) {
        kb.remap(&conn, *keycode, keysym_of(character))?;
        cleanup.premaps.push((*keycode, vec![0; kb.width]));
    }

    let inserter = Inserter::with_backends(vec![Box::new(X11Backend::new())]);

    // 1. Capture.
    let snap_a = inserter.capture().map_err(|e| x11(format!("capture: {e}")))?;
    assert_eq!(snap_a.backend, BackendKind::X11);
    assert_eq!(snap_a.app.as_deref(), Some("StarlingIt"));
    assert_eq!(snap_a.title.as_deref(), Some("starling x11 IT window (first)"));
    assert_eq!(snap_a.pid, None, "the test window sets no _NET_WM_PID");
    let (active_a, focus_a, _) = snap_a.ids().expect("a captured ref round trips");
    assert_eq!(active_a, a as u64, "active window is A");
    assert_eq!(focus_a, a as u64, "focus window is A");

    // 5. Control characters are refused before anything is typed; pin
    //    it while a real target is focused so the refusal cannot be
    //    blamed on focus state.
    let backend = inserter.backend_for(&snap_a).expect("the X11 backend is present");
    match backend.insert(&snap_a, "line\nbreak") {
        Err(InsertError::MultilineUnsupported) => {}
        other => panic!("multiline must be refused on a live target: {other:?}"),
    }
    assert!(
        drain_keypress_events(&conn, a).is_empty(),
        "a refused insert may not type"
    );

    // 2. The mixed string, decoded back through the mapping.
    let receipt = backend
        .insert(&snap_a, MIXED)
        .map_err(|e| x11(format!("inserting {MIXED:?}: {e}")))?;
    assert_eq!(receipt.evidence, EVIDENCE_SYNTHETIC_KEYS);
    let typed = drain_keypresses(&conn, a, &kb);
    assert_eq!(typed, MIXED, "the KeyPress stream decodes back to the text");

    // 3. The remap path: UNMAPPED rides the highest remaining spare
    //    keycode (the backend scans top down past the pre-mapped,
    //    now non-zero keycodes), and the mapping is restored after.
    let spare = kb.spare_after(&spares).ok_or_else(|| {
        focus_error("this keyboard has no fourth spare keycode for the remap path")
    })?;
    let receipt = backend
        .insert(&snap_a, &UNMAPPED.to_string())
        .map_err(|e| x11(format!("inserting {UNMAPPED}: {e}")))?;
    assert_eq!(receipt.evidence, EVIDENCE_SYNTHETIC_KEYS);
    let presses = drain_keypress_events(&conn, a);
    let remap_presses: Vec<_> = presses
        .iter()
        .filter(|&&(keycode, state)| keycode == spare && state == 0)
        .collect();
    assert_eq!(
        remap_presses.len(),
        1,
        "the unmapped character typed exactly once on the remapped spare keycode {spare}, \
         events were {presses:?}"
    );
    kb.assert_restored(&conn, spare)?;

    // 4. Focus safety: a second window takes focus (its map activates
    //    it under a WM), and the frozen A snapshot must conflict.
    let b = make_window(&conn, root, &atoms, "second")?;
    cleanup.windows.push(b);
    focus_and_wait(&conn, root, &atoms, b)?;
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
    let snap_b = inserter.capture().map_err(|e| x11(format!("recapture: {e}")))?;
    assert_ne!(snap_b.target_ref, snap_a.target_ref);
    assert!(
        drain_keypress_events(&conn, a).is_empty(),
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
        "starling-insertion X11 IT passed: capture identity, mixed-string decode ({MIXED:?}), \
         remap+restore on keycode {spare}, changed and gone revalidation"
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

/// Create, name, class and map one test window. No `_NET_WM_PID` on
/// purpose: capture must report "unknown pid", not invent one.
fn make_window(
    conn: &RustConnection,
    root: Window,
    atoms: &Atoms,
    label: &str,
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
    set_property8(conn, window, atoms.net_wm_name, atoms.utf8_string, title.as_bytes())?;
    // Ask for a clean WM_DELETE_WINDOW close so a cooperative WM
    // offers an orderly teardown instead of killing the window.
    set_property32(
        conn,
        window,
        atoms.wm_protocols,
        atoms.atom_type,
        &[atoms.wm_delete_window],
    )?;
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
            Some(u32::from_ne_bytes(active.value[..4].try_into().expect("four bytes")))
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

/// The keyboard mapping snapshot the test decodes through. Only the
/// base/Shift columns are decoded — the same columns the backend
/// trusts — so agreement between the two is the assertion, not a
/// coincidence of wider layouts.
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

    fn max(&self) -> u8 {
        (self.min as usize + self.syms.len() / self.width - 1) as u8
    }

    fn syms_of(&self, keycode: u8) -> &[u32] {
        &self.syms[((keycode - self.min) as usize) * self.width..][..self.width]
    }

    /// `count` spare (all-`NoSymbol`) keycodes in ascending order —
    /// taken from the top, the same policy as the backend, which is
    /// why the remap prediction in the test can work.
    fn spare_keycodes(&self, count: usize) -> Option<Vec<u8>> {
        let spares: Vec<u8> = (self.min..=self.max())
            .rev()
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

    /// The highest remaining spare keycode after `used` — what the
    /// backend must pick for the remap (it scans top down past the
    /// pre-mapped, now non-zero keycodes).
    fn spare_after(&self, used: &[u8]) -> Option<u8> {
        (self.min..=self.max())
            .rev()
            .filter(|&keycode| {
                self.syms_of(keycode).iter().all(|&sym| sym == 0) && !used.contains(&keycode)
            })
            .next()
    }

    /// Point one keycode's column 0 at `keysym` (width preserved).
    /// The snapshot is updated too so decode stays truthful about the
    /// server state the events were decoded against.
    fn remap(&mut self, conn: &RustConnection, keycode: u8, keysym: u32) -> Result<(), ItError> {
        let mut columns = vec![0u32; self.width];
        columns[0] = keysym;
        conn.change_keyboard_mapping(1, keycode, self.width as u8, &columns)
            .map_err(x11)?
            .check()
            .map_err(x11)?;
        let start = (keycode - self.min) as usize * self.width;
        self.syms[start..start + self.width].copy_from_slice(&columns);
        Ok(())
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

    /// The live mapping of `keycode` is all-`NoSymbol` again (the
    /// backend's restore) — asked of the server, not the snapshot.
    fn assert_restored(&self, conn: &RustConnection, keycode: u8) -> Result<(), ItError> {
        let reply = conn
            .get_keyboard_mapping(keycode, 1)
            .map_err(x11)?
            .reply()
            .map_err(x11)?;
        assert!(
            reply.keysyms.iter().all(|&sym| sym == 0),
            "keycode {keycode} must be restored to NoSymbol, is {:?}",
            reply.keysyms
        );
        Ok(())
    }
}

/// Drain queued KeyPress events for `window` into decoded characters
/// (modifier events skipped: Shift presses are typing mechanics, not
/// text).
fn drain_keypresses(conn: &RustConnection, window: Window, kb: &Keyboard) -> String {
    let mut text = String::new();
    for (keycode, state) in drain_keypress_events(conn, window) {
        if let Some(character) = kb.decode(keycode, state) {
            text.push(character);
        }
    }
    text
}

/// Drain queued KeyPress events for `window` as `(keycode, state)`.
fn drain_keypress_events(conn: &RustConnection, window: Window) -> Vec<(u8, u16)> {
    std::thread::sleep(DRAIN_GRACE);
    let mut presses = Vec::new();
    while let Ok(Some(event)) = conn.poll_for_event() {
        // KeyPress = 2; the high bit is the send-event marker.
        if event.response_type() & 0x7f == 2 {
            if let x11rb::protocol::Event::KeyPress(press) = event {
                if press.event == window {
                    presses.push((press.detail, u16::from(press.state)));
                }
            }
        }
    }
    presses
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
