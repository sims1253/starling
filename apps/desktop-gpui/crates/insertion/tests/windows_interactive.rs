//! The Windows backend against a real interactive session (issue #221,
//! slice 2). `#[ignore]`d and *doubly* gated: it needs a real desktop
//! session (a foreground window to capture, a message pump, real input
//! injection), so it only runs when explicitly asked for:
//!
//! ```text
//! STARLING_WIN_IT=1 cargo test -p starling-insertion --test windows_interactive -- --ignored --nocapture
//! ```
//!
//! What it pins: a top-level EDIT control is created, shown and
//! brought to the foreground; the backend captures it as the focused
//! target (foreground + focus hwnd both the edit, pid = this process);
//! a mixed string — ASCII, Latin-1, an arrow, and a surrogate pair
//! (😀) — is inserted through `SendInput` `KEYEVENTF_UNICODE`; and the
//! control's own text is read back with `WM_GETTEXT` and must equal
//! the inserted string exactly (the EDIT control is the honest
//! decoder: it processes the injected input like any real target).
//!
//! # The self-target escape hatch, deliberately narrow
//!
//! The backend's ownership policy always refuses the *configuring*
//! process's pid, and [`crate::Inserter::with_excluded_pids`] unions
//! whatever the host passes with it. This test must type into its own
//! window (that is the point of an interactive test), so it uses
//! [`crate::windows::WindowsBackend::with_exact_excluded_pids_for_self_typing_tests`] directly
//! with an empty list — the documented advanced constructor for
//! probes. The Inserter-level guarantee (own pid always excluded) is
//! pinned by the crate's unit tests instead.

#![cfg(windows)]

use starling_insertion::windows::WindowsBackend;
use starling_insertion::{BackendKind, InsertError, InsertionBackend, EVIDENCE_SYNTHETIC_KEYS};
use windows_sys::Win32::Foundation::HWND;
use windows_sys::Win32::UI::Input::KeyboardAndMouse::GetAsyncKeyState;
use windows_sys::Win32::UI::WindowsAndMessaging::{
    CreateWindowExW, DestroyWindow, DispatchMessageW, PeekMessageW, SetForegroundWindow,
    ShowWindow, TranslateMessage, MSG, PM_REMOVE, SW_SHOW, WM_GETTEXT, WS_OVERLAPPEDWINDOW,
    WS_VISIBLE,
};

/// The mixed string: ASCII, Latin-1 (`é`), a symbol (`→`) and a
/// surrogate pair (😀, U+1F600) — every shape `KEYEVENTF_UNICODE`
/// must deliver as whole characters.
const MIXED: &str = "Café → 😀 42!";

#[test]
#[ignore = "interactive: needs a real desktop session; run with STARLING_WIN_IT=1 --ignored"]
fn windows_end_to_end_edit_control_round_trip() {
    let enabled = std::env::var("STARLING_WIN_IT").ok().as_deref() == Some("1");
    if !enabled {
        eprintln!("skipped: set STARLING_WIN_IT=1 to run the Windows interactive test");
        return;
    }
    unsafe { main_test().expect("Windows interactive test failed") };
}

unsafe fn main_test() -> Result<(), String> {
    // Held modifiers would (rightly) make the insert wait and refuse;
    // fail early with a clear message instead of a ModifiersHeld
    // mystery. Nothing is released synthetically — a human should not
    // be holding keys while running an interactive test.
    for key in ["Shift", "Ctrl", "Alt", "Left Windows", "Right Windows"] {
        let held = match key {
            "Shift" => GetAsyncKeyState(0x10),
            "Ctrl" => GetAsyncKeyState(0x11),
            "Alt" => GetAsyncKeyState(0x12),
            "Left Windows" => GetAsyncKeyState(0x5B),
            _ => GetAsyncKeyState(0x5C),
        };
        if held as u16 & 0x8000 != 0 {
            return Err(format!(
                "{key} is held down; release it before running this test"
            ));
        }
    }

    let class: Vec<u16> = "EDIT\0".encode_utf16().collect();
    // A top-level EDIT window's title *is* its text content: it must
    // start empty, or the read-back sees the title next to the insert.
    let title: Vec<u16> = "\0".encode_utf16().collect();
    let edit = CreateWindowExW(
        0,
        class.as_ptr(),
        title.as_ptr(),
        WS_OVERLAPPEDWINDOW | WS_VISIBLE, // ES_LEFT | ES_AUTOHSCROLL are both 0-compatible
        60,
        60,
        520,
        160,
        std::ptr::null_mut(),
        std::ptr::null_mut(),
        std::ptr::null_mut(),
        std::ptr::null(),
    );
    if edit.is_null() {
        return Err("CreateWindowExW(\"EDIT\") failed".to_string());
    }
    ShowWindow(edit, SW_SHOW);
    let foreground = SetForegroundWindow(edit);
    if foreground == 0 {
        DestroyWindow(edit);
        return Err(
            "SetForegroundWindow was refused (Windows only allows the foreground process to \
             take it); click a console once and rerun"
                .to_string(),
        );
    }
    pump(edit, 400);

    // The backend typed into directly with an empty exclusion list —
    // see the module docs for why this is deliberate and narrow.
    let backend = WindowsBackend::with_exact_excluded_pids_for_self_typing_tests(Vec::new());
    let snapshot = backend
        .capture()
        .map_err(|error| format!("capture: {error}"))?;
    if snapshot.backend != BackendKind::Windows {
        return Err(format!("wrong backend: {:?}", snapshot.backend));
    }
    let Some((active, focus, pid)) = snapshot.ids() else {
        return Err("the captured ref did not round trip".to_string());
    };
    if active != focus {
        return Err(format!(
            "a top-level EDIT control is its own focus window: active {active:#x}, focus \
             {focus:#x}"
        ));
    }
    if pid != Some(std::process::id()) {
        return Err(format!(
            "the edit control belongs to this process: pid {pid:?}"
        ));
    }

    let receipt = backend
        .insert(&snapshot, MIXED)
        .map_err(|error| format!("inserting {MIXED:?}: {error}"))?;
    if receipt.evidence != EVIDENCE_SYNTHETIC_KEYS {
        return Err(format!("unexpected evidence: {}", receipt.evidence));
    }
    // The EDIT control processes its WM_CHAR queue only while this
    // thread pumps; give it time, then read what it accumulated.
    pump(edit, 600);

    let mut buffer = [0u16; 256];
    let length = SendMessage_strlen(edit, &mut buffer);
    let text = String::from_utf16_lossy(&buffer[..length.max(0) as usize]);
    if text != MIXED {
        return Err(format!(
            "the edit control holds {text:?}, expected {MIXED:?}"
        ));
    }

    // The refused paths pin quickly too: a multiline payload never
    // reaches SendInput, so the control's text cannot change.
    match backend.insert(&snapshot, "line\nbreak") {
        Err(InsertError::MultilineUnsupported) => {}
        other => return Err(format!("multiline must be refused: {other:?}")),
    }
    pump(edit, 100);
    let mut check = [0u16; 256];
    let length = SendMessage_strlen(edit, &mut check);
    let text = String::from_utf16_lossy(&check[..length.max(0) as usize]);
    if text != MIXED {
        return Err(format!("a refused insert changed the text: {text:?}"));
    }

    DestroyWindow(edit);
    println!(
        "starling-insertion Windows IT passed: EDIT control round trip of {MIXED:?} via \
         SendInput, capture identity, multiline refusal"
    );
    Ok(())
}

/// `SendMessageW(WM_GETTEXT)` as the returned string length.
unsafe fn SendMessage_strlen(window: HWND, buffer: &mut [u16]) -> i32 {
    windows_sys::Win32::UI::WindowsAndMessaging::SendMessageW(
        window,
        WM_GETTEXT,
        buffer.len(),
        buffer.as_mut_ptr() as isize,
    ) as i32
}

/// Pump this thread's message queue for `millis` so the EDIT control
/// can process the input events `SendInput` queued onto it (WM_CHAR
/// handling, redraw) — a real target's event loop, miniature.
unsafe fn pump(_window: HWND, millis: u64) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(millis);
    let mut message: MSG = std::mem::zeroed();
    loop {
        while PeekMessageW(&mut message, std::ptr::null_mut(), 0, 0, PM_REMOVE) != 0 {
            let _ = TranslateMessage(&message);
            DispatchMessageW(&message);
        }
        if std::time::Instant::now() >= deadline {
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}
