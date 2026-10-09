//! The Windows backend against a real desktop session. Ignored by default:
//!
//! ```text
//! STARLING_WIN_IT=1 cargo test -p starling-insertion --test windows_interactive -- --ignored --nocapture
//! ```
//!
//! A top-level EDIT control is brought to the foreground, captured, typed
//! into through `SendInput`, and its text read back with `WM_GETTEXT`. The
//! test types into its own window, so it uses the constructor that does
//! not exclude this process.

#![cfg(windows)]

use std::time::{Duration, Instant};

use starling_insertion::windows::WindowsBackend;
use starling_insertion::{BackendKind, InsertError, InsertionBackend, EVIDENCE_SYNTHETIC_KEYS};
use windows_sys::Win32::Foundation::HWND;
use windows_sys::Win32::UI::Input::KeyboardAndMouse::{
    GetAsyncKeyState, VK_CONTROL, VK_LWIN, VK_MENU, VK_RWIN, VK_SHIFT,
};
use windows_sys::Win32::UI::WindowsAndMessaging::{
    CreateWindowExW, DestroyWindow, DispatchMessageW, PeekMessageW, SendMessageW,
    SetForegroundWindow, ShowWindow, TranslateMessage, MSG, PM_REMOVE, SW_SHOW, WM_GETTEXT,
    WS_OVERLAPPEDWINDOW, WS_VISIBLE,
};

/// ASCII, Latin-1, a symbol and a surrogate pair.
const MIXED: &str = "Café → 😀 42!";

#[test]
#[ignore = "interactive: needs a real desktop session; run with STARLING_WIN_IT=1 --ignored"]
fn windows_end_to_end_edit_control_round_trip() {
    if std::env::var("STARLING_WIN_IT").ok().as_deref() != Some("1") {
        eprintln!("skipped: set STARLING_WIN_IT=1 to run the Windows interactive test");
        return;
    }
    for key in [VK_SHIFT, VK_CONTROL, VK_MENU, VK_LWIN, VK_RWIN] {
        let held = unsafe { GetAsyncKeyState(i32::from(key)) } as u16 & 0x8000 != 0;
        assert!(!held, "release virtual key {key:#x} before running this test");
    }

    let class: Vec<u16> = "EDIT\0".encode_utf16().collect();
    // A top-level EDIT window's title is its text, so it must start empty.
    let title = [0u16];
    let edit = unsafe {
        CreateWindowExW(
            0,
            class.as_ptr(),
            title.as_ptr(),
            WS_OVERLAPPEDWINDOW | WS_VISIBLE,
            60,
            60,
            520,
            160,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null(),
        )
    };
    assert!(!edit.is_null(), "CreateWindowExW(\"EDIT\") failed");
    let result = round_trip(edit);
    unsafe { DestroyWindow(edit) };
    result.expect("Windows interactive test failed");
}

fn round_trip(edit: HWND) -> Result<(), String> {
    unsafe { ShowWindow(edit, SW_SHOW) };
    if unsafe { SetForegroundWindow(edit) } == 0 {
        return Err("SetForegroundWindow was refused; click a console once and rerun".into());
    }
    pump(400);

    let backend = WindowsBackend::with_exact_excluded_pids_for_self_typing_tests(Vec::new());
    let snapshot = backend.capture().map_err(|e| format!("capture: {e}"))?;
    assert_eq!(snapshot.backend, BackendKind::Windows);
    let (active, focus, pid) = snapshot.ids().expect("the captured ref parses");
    assert_eq!(active, focus, "a top-level EDIT control is its own focus");
    assert_eq!(pid, Some(std::process::id()));

    let receipt = backend
        .insert(&snapshot, MIXED)
        .map_err(|e| format!("inserting {MIXED:?}: {e}"))?;
    assert_eq!(receipt.evidence, EVIDENCE_SYNTHETIC_KEYS);
    pump(600);
    assert_eq!(edit_text(edit), MIXED);

    assert_eq!(
        backend.insert(&snapshot, "line\nbreak"),
        Err(InsertError::MultilineUnsupported)
    );
    pump(100);
    assert_eq!(edit_text(edit), MIXED, "a refused insert changed the text");
    Ok(())
}

fn edit_text(window: HWND) -> String {
    let mut buffer = [0u16; 256];
    let length = unsafe {
        SendMessageW(
            window,
            WM_GETTEXT,
            buffer.len(),
            buffer.as_mut_ptr() as isize,
        )
    };
    String::from_utf16_lossy(&buffer[..length.max(0) as usize])
}

/// Run this thread's message loop so the EDIT control processes the
/// queued input.
fn pump(millis: u64) {
    let deadline = Instant::now() + Duration::from_millis(millis);
    let mut message: MSG = unsafe { std::mem::zeroed() };
    while Instant::now() < deadline {
        while unsafe { PeekMessageW(&mut message, std::ptr::null_mut(), 0, 0, PM_REMOVE) } != 0 {
            unsafe {
                TranslateMessage(&message);
                DispatchMessageW(&message);
            }
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}
