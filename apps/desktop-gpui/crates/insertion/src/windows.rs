//! The Windows backend: identity from the foreground window and its
//! focused control, typing via `SendInput` `KEYEVENTF_UNICODE`
//! (issue #221, slice 2 phase A).
//!
//! # Identity
//!
//! Capture reads `GetForegroundWindow` (the active top-level window)
//! and, through `GetGUIThreadInfo` for that window's thread, the
//! actual focused child window (`hwndFocus`) — the control keystrokes
//! go to. The ref is `win:<hwnd-hex>:<focus-hex>:<pid>`; revalidate
//! compares both and checks the window still exists (`IsWindow`),
//! mirroring the X11 backend's two-id identity.
//!
//! # Typing
//!
//! `SendInput` with `KEYEVENTF_UNICODE` injects one UTF-16 unit per
//! down+up pair, so any BMP or supplementary character (surrogate
//! pairs ride as two units, exactly as a real IME would commit them)
//! can be typed without depending on the user's keyboard layout at
//! all. The send count is checked: UIPI silently blocks injected input
//! aimed at an elevated target, which surfaces as a short count and is
//! reported as [`InsertError::PermissionDenied`] with the settings
//! hint — the user's fix is to run Starling and the target at the same
//! integrity level, or use the copy fallback.
//!
//! # Deliberately absent in phase A
//!
//! No `SetForegroundWindow` (Starling never steals focus to type) and
//! no surrounding text (UIA is the later path; the capability table
//! reports it absent until then).

use crate::{
    format_ref, insertion_guards, parse_ref, Availability, BackendKind, InsertError,
    InsertReceipt, InsertionBackend, SurroundingText, TargetCheck, TargetSnapshot,
    EVIDENCE_SYNTHETIC_KEYS,
};

use windows_sys::Win32::Foundation::HWND;
use windows_sys::Win32::UI::Input::KeyboardAndMouse::{
    SendInput, INPUT, INPUT_KEYBOARD, KEYBDINPUT, KEYEVENTF_KEYUP, KEYEVENTF_UNICODE,
};
use windows_sys::Win32::UI::WindowsAndMessaging::{
    GetForegroundWindow, GetGUIThreadInfo, GetWindowTextW, GetWindowThreadProcessId, GUITHREADINFO,
    IsWindow,
};
use windows_sys::Win32::System::Threading::{
    OpenProcess, QueryFullProcessImageNameW, PROCESS_NAME_WIN32, PROCESS_QUERY_LIMITED_INFORMATION,
};

/// The Windows backend. Every call is a direct Win32 round trip, so
/// like the X11 backend it holds no state that can go stale between a
/// take's capture and its insert.
#[derive(Debug, Default)]
pub struct WindowsBackend;

impl WindowsBackend {
    pub fn new() -> WindowsBackend {
        WindowsBackend
    }
}

/// A null `HWND` is Windows' "no window"; windows-sys models HWND
/// as a raw pointer, so the null check is `is_null`.
fn is_null(hwnd: HWND) -> bool {
    hwnd.is_null()
}

/// HWND ↔ the u64 a target ref carries (`usize` is the honest
/// intermediate: pointers do not cast to `u64` directly).
fn hwnd_id(hwnd: HWND) -> u64 {
    hwnd as usize as u64
}

fn hwnd_from(id: u64) -> HWND {
    id as usize as HWND
}

/// `(foreground, focus)` plus the foreground window's pid, or `None`
/// when there is no honest target (no foreground window, or a
/// foreground window whose thread reports no focused control — typing
/// into those would be guessing).
fn focus_pair() -> Option<(HWND, HWND, u32)> {
    let active = unsafe { GetForegroundWindow() };
    if is_null(active) {
        return None;
    }
    let mut pid: u32 = 0;
    let thread_id = unsafe { GetWindowThreadProcessId(active, &mut pid) };
    if thread_id == 0 || pid == 0 {
        return None;
    }
    let mut info: GUITHREADINFO = unsafe { std::mem::zeroed() };
    info.cbSize = std::mem::size_of::<GUITHREADINFO>() as u32;
    if unsafe { GetGUIThreadInfo(thread_id, &mut info) } == 0 {
        return None;
    }
    if is_null(info.hwndFocus) {
        return None;
    }
    Some((active, info.hwndFocus, pid))
}

/// The window title, if it has one.
fn window_title(window: HWND) -> Option<String> {
    let mut buffer = [0u16; 512];
    let length = unsafe { GetWindowTextW(window, buffer.as_mut_ptr(), buffer.len() as i32) };
    if length <= 0 {
        return None;
    }
    Some(String::from_utf16_lossy(&buffer[..length as usize]))
}

/// The owning process's exe file name (without path), if it can be
/// opened for the (deliberately minimal) query.
fn process_exe_name(pid: u32) -> Option<String> {
    let process = unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid) };
    if process.is_null() {
        return None;
    }
    let mut buffer = [0u16; 512];
    let mut length = buffer.len() as u32;
    let ok = unsafe {
        QueryFullProcessImageNameW(process, PROCESS_NAME_WIN32, buffer.as_mut_ptr(), &mut length)
    };
    unsafe { windows_sys::Win32::Foundation::CloseHandle(process) };
    if ok == 0 {
        return None;
    }
    let path = String::from_utf16_lossy(&buffer[..length as usize]);
    // The bare file name is what a user recognizes; the directory is
    // noise in a recovery panel.
    path.rsplit(['\\', '/']).next().map(str::to_string)
}

impl InsertionBackend for WindowsBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Windows
    }

    fn availability(&self) -> Availability {
        // Windows always has a foreground-window model; the only
        // session-level refusal (UIPI against elevated targets) is
        // discovered per insert, where the settings hint has a target
        // to name.
        Availability::Ready
    }

    fn capture(&self) -> Result<TargetSnapshot, InsertError> {
        let Some((active, focus, pid)) = focus_pair() else {
            return Err(InsertError::Rejected {
                reason: "no foreground window with a focused control".to_string(),
            });
        };
        if pid == std::process::id() {
            // Starling never types into Starling (see the X11 twin of
            // this check for the reasoning).
            return Err(InsertError::TargetIsStarling);
        }
        Ok(TargetSnapshot {
            backend: BackendKind::Windows,
            target_ref: format_ref(
                BackendKind::Windows,
                hwnd_id(active),
                hwnd_id(focus),
                Some(pid),
            ),
            app: process_exe_name(pid),
            title: window_title(active),
            pid: Some(pid),
            capabilities: BackendKind::Windows.capabilities(),
        })
    }

    fn revalidate(&self, target: &TargetSnapshot) -> Result<TargetCheck, InsertError> {
        let Some((kind, active, focus, _pid)) = parse_ref(&target.target_ref) else {
            return Err(InsertError::Rejected {
                reason: format!("malformed target ref: {}", target.target_ref),
            });
        };
        debug_assert_eq!(kind, BackendKind::Windows, "the inserter routes by scheme");
        // A closed window is `Gone` even if focus also moved (the X11
        // backend's rule, for the same reason: "your target closed" is
        // the actionable fact).
        if unsafe { IsWindow(hwnd_from(active)) } == 0 {
            return Ok(TargetCheck::Gone);
        }
        match focus_pair() {
            Some((live_active, live_focus, _))
                if hwnd_id(live_active) == active && hwnd_id(live_focus) == focus =>
            {
                Ok(TargetCheck::Same)
            }
            Some((live_active, live_focus, live_pid)) => Ok(TargetCheck::Changed {
                expected: target.target_ref.clone(),
                actual: format_ref(
                    kind,
                    hwnd_id(live_active),
                    hwnd_id(live_focus),
                    Some(live_pid),
                ),
            }),
            None => Ok(TargetCheck::Changed {
                expected: target.target_ref.clone(),
                actual: format!("{}:none", kind.scheme()),
            }),
        }
    }

    fn surrounding_text(
        &self,
        _target: &TargetSnapshot,
    ) -> Result<Option<SurroundingText>, InsertError> {
        // Phase A: no UIA yet; the capability table says absent.
        Ok(None)
    }

    fn insert(&self, target: &TargetSnapshot, text: &str) -> Result<InsertReceipt, InsertError> {
        insertion_guards(text, target.pid, || self.revalidate(target))?;

        // One down+up pair per UTF-16 unit, `wVk` zero and the unit in
        // `wScan` per the `KEYEVENTF_UNICODE` contract. Surrogate
        // halves pass through as their own units, which is how a real
        // IME commits supplementary characters too.
        let units: Vec<u16> = text.encode_utf16().collect();
        let mut events = Vec::with_capacity(units.len() * 2);
        for &unit in &units {
            events.push(key_event(unit, false));
            events.push(key_event(unit, true));
        }
        let sent = unsafe {
            SendInput(
                events.len() as u32,
                events.as_ptr(),
                std::mem::size_of::<INPUT>() as i32,
            )
        };
        if sent != events.len() as u32 {
            // A short count is UIPI's silent block: the target runs
            // elevated above Starling. Nothing was typed (partially
            // delivered input is impossible for one SendInput call —
            // the return is zero when the block hits the first event),
            // so the honest report is the refusal, not a half receipt.
            return Err(InsertError::PermissionDenied {
                reason: "the system blocked the injected input (the target runs elevated; \
                         UIPI)"
                    .to_string(),
                settings_hint: "run Starling at the same integrity level as the target, or \
                                use the copy fallback"
                    .to_string(),
            });
        }
        Ok(InsertReceipt {
            evidence: EVIDENCE_SYNTHETIC_KEYS,
        })
    }
}

/// One `KEYEVENTF_UNICODE` event.
fn key_event(unit: u16, key_up: bool) -> INPUT {
    let mut flags = KEYEVENTF_UNICODE;
    if key_up {
        flags |= KEYEVENTF_KEYUP;
    }
    INPUT {
        r#type: INPUT_KEYBOARD,
        Anonymous: windows_sys::Win32::UI::Input::KeyboardAndMouse::INPUT_0 {
            ki: KEYBDINPUT {
                wVk: 0,
                wScan: unit,
                dwFlags: flags,
                time: 0,
                dwExtraInfo: 0,
            },
        },
    }
}
