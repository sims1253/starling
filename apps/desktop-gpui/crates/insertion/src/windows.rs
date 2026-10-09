//! The Windows backend: identity from the foreground window and its
//! focused control, typing through `SendInput` with `KEYEVENTF_UNICODE`.
//!
//! A ref is `win:<foreground hwnd>:<GetGUIThreadInfo focus hwnd>:<pid>`.
//! Revalidation requires both handles to match, the foreground window to
//! exist, and the pid to match (a differing pid means a recycled handle).
//!
//! `KEYEVENTF_UNICODE` sends one down/up pair per UTF-16 unit, independent
//! of the keyboard layout. Text goes out in chunks of at most
//! [`WIN_CHUNK_UTF16_UNITS`] units (a surrogate pair is never split), each
//! preceded by the held-modifier wait (`GetAsyncKeyState`) and the
//! identity check, and the foreground window is compared once more right
//! before the send. `SendInput` cannot bind input to a window, so a switch
//! inside that last gap is not caught; chunking bounds what it can
//! misdirect.
//!
//! Starling never calls `SetForegroundWindow`, and has no surrounding
//! text here (that needs UI Automation).

use windows_sys::Win32::Foundation::{CloseHandle, HWND};
use windows_sys::Win32::System::Threading::{
    OpenProcess, QueryFullProcessImageNameW, PROCESS_NAME_WIN32, PROCESS_QUERY_LIMITED_INFORMATION,
};
use windows_sys::Win32::UI::Input::KeyboardAndMouse::{
    GetAsyncKeyState, SendInput, INPUT, INPUT_0, INPUT_KEYBOARD, KEYBDINPUT, KEYEVENTF_KEYUP,
    KEYEVENTF_UNICODE, VK_CONTROL, VK_LWIN, VK_MENU, VK_RWIN, VK_SHIFT,
};
use windows_sys::Win32::UI::WindowsAndMessaging::{
    GetForegroundWindow, GetGUIThreadInfo, GetWindowTextW, GetWindowThreadProcessId, IsWindow,
    GUITHREADINFO,
};

use crate::{
    chars_keyed_down_within_events, deliver_in_chunks, format_ref, insertion_guards,
    merge_excluded_pids, parse_ref, wait_modifiers_released, weighed_segments, BackendKind,
    ChunkFailure, InsertError, InsertReceipt, InsertionBackend, TargetCheck, TargetSnapshot,
};

/// UTF-16 units per `SendInput` call.
const WIN_CHUNK_UTF16_UNITS: usize = 32;

const TRACKED_MODIFIERS: [(u16, &str); 5] = [
    (VK_SHIFT, "Shift"),
    (VK_CONTROL, "Ctrl"),
    (VK_MENU, "Alt"),
    (VK_LWIN, "Left Windows"),
    (VK_RWIN, "Right Windows"),
];

#[derive(Debug)]
pub struct WindowsBackend {
    excluded_pids: Vec<u32>,
}

impl Default for WindowsBackend {
    fn default() -> Self {
        WindowsBackend::new()
    }
}

impl WindowsBackend {
    pub fn new() -> WindowsBackend {
        WindowsBackend::with_excluded_pids(Vec::new())
    }

    /// Refuse targets owned by any of `excluded_pids` or this process.
    pub fn with_excluded_pids(excluded_pids: Vec<u32>) -> WindowsBackend {
        WindowsBackend {
            excluded_pids: merge_excluded_pids(excluded_pids),
        }
    }

    /// Exactly `excluded_pids`, without this process: for the interactive
    /// test, which types into its own window.
    #[cfg(any(test, feature = "test-doubles"))]
    pub fn with_exact_excluded_pids_for_self_typing_tests(
        excluded_pids: Vec<u32>,
    ) -> WindowsBackend {
        WindowsBackend { excluded_pids }
    }

    /// Wait out held modifiers, then the live identity: last, because the
    /// modifier wait can take seconds.
    fn chunk_check(&self, target_ref: &str) -> Result<(), InsertError> {
        wait_modifiers_released(|| Ok(held_modifier_names()))?;
        match live_target(target_ref)? {
            LiveTarget::Same { live_pid } if self.excluded_pids.contains(&live_pid) => {
                Err(InsertError::TargetIsStarling)
            }
            LiveTarget::Same { .. } => Ok(()),
            LiveTarget::Changed { actual } => Err(InsertError::TargetChanged {
                expected: target_ref.to_string(),
                actual,
            }),
            LiveTarget::Gone => Err(InsertError::TargetGone),
        }
    }
}

impl InsertionBackend for WindowsBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Windows
    }

    fn availability(&self) -> Result<(), InsertError> {
        Ok(())
    }

    fn capture(&self) -> Result<TargetSnapshot, InsertError> {
        let Some((active, focus, pid)) = focus_pair() else {
            return Err(InsertError::Rejected {
                reason: "no foreground window with a focused control".to_string(),
            });
        };
        if self.excluded_pids.contains(&pid) {
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
        })
    }

    fn revalidate(&self, target: &TargetSnapshot) -> Result<TargetCheck, InsertError> {
        Ok(match live_target(&target.target_ref)? {
            LiveTarget::Same { .. } => TargetCheck::Same,
            LiveTarget::Changed { actual } => TargetCheck::Changed {
                expected: target.target_ref.clone(),
                actual,
            },
            LiveTarget::Gone => TargetCheck::Gone,
        })
    }

    fn insert(&self, target: &TargetSnapshot, text: &str) -> Result<InsertReceipt, InsertError> {
        insertion_guards(text, target.pid, &self.excluded_pids)?;
        let Some((_, active, _, _)) = parse_ref(&target.target_ref) else {
            return Err(InsertError::Rejected {
                reason: format!("malformed target ref: {}", target.target_ref),
            });
        };
        let segments = weighed_segments(text, WIN_CHUNK_UTF16_UNITS, char::len_utf16);
        // Built before the first check, so only the send follows it.
        let mut chunks = segments
            .iter()
            .map(|segment| {
                segment
                    .encode_utf16()
                    .flat_map(|unit| [key_event(unit, 0), key_event(unit, KEYEVENTF_KEYUP)])
                    .collect::<Vec<INPUT>>()
            })
            .collect::<Vec<_>>()
            .into_iter();
        deliver_in_chunks(
            text.chars().count(),
            &segments,
            || self.chunk_check(&target.target_ref),
            |segment, ()| {
                let events = chunks.next().expect("one chunk per segment");
                if foreground_id() != active {
                    return Err(ChunkFailure {
                        delivered: 0,
                        cause: InsertError::TargetChanged {
                            expected: target.target_ref.clone(),
                            actual: format!("win:{:x}", foreground_id()),
                        },
                    });
                }
                let sent = unsafe {
                    SendInput(
                        events.len() as u32,
                        events.as_ptr(),
                        std::mem::size_of::<INPUT>() as i32,
                    )
                } as usize;
                if sent == events.len() {
                    return Ok(());
                }
                // Windows cannot tell why: an elevated target (UIPI), another
                // program blocking input, or a desktop switch.
                Err(ChunkFailure {
                    delivered: chars_keyed_down_within_events(segment, sent),
                    cause: InsertError::Rejected {
                        reason: "Windows blocked the input part-way (an elevated target, input \
                                 blocked by another program, or a desktop switch)"
                            .to_string(),
                    },
                })
            },
        )
    }
}

/// The ref carries 32 bits per handle: Windows guarantees only those are
/// significant, so truncating is safe and the way back sign-extends.
fn hwnd_id(hwnd: HWND) -> u32 {
    hwnd as usize as u32
}

fn foreground_id() -> u32 {
    hwnd_id(unsafe { GetForegroundWindow() })
}

fn hwnd_from(id: u32) -> HWND {
    id as i32 as isize as HWND
}

enum LiveTarget {
    Same { live_pid: u32 },
    Changed { actual: String },
    Gone,
}

/// `(foreground, focus, pid)`, or `None` without a foreground window that
/// has a focused control.
fn focus_pair() -> Option<(HWND, HWND, u32)> {
    let active = unsafe { GetForegroundWindow() };
    if active.is_null() {
        return None;
    }
    let mut pid = 0;
    let thread_id = unsafe { GetWindowThreadProcessId(active, &mut pid) };
    if thread_id == 0 || pid == 0 {
        return None;
    }
    let mut info: GUITHREADINFO = unsafe { std::mem::zeroed() };
    info.cbSize = std::mem::size_of::<GUITHREADINFO>() as u32;
    if unsafe { GetGUIThreadInfo(thread_id, &mut info) } == 0 || info.hwndFocus.is_null() {
        return None;
    }
    // The focus read belongs to `active` only if it is still foreground.
    (unsafe { GetForegroundWindow() } == active).then_some((active, info.hwndFocus, pid))
}

fn live_target(target_ref: &str) -> Result<LiveTarget, InsertError> {
    let Some((_, active, focus, captured_pid)) = parse_ref(target_ref) else {
        return Err(InsertError::Rejected {
            reason: format!("malformed target ref: {target_ref}"),
        });
    };
    // A closed window is `Gone` even if focus moved too.
    if unsafe { IsWindow(hwnd_from(active)) } == 0 {
        return Ok(LiveTarget::Gone);
    }
    let Some((live_active, live_focus, live_pid)) = focus_pair() else {
        return Ok(LiveTarget::Changed {
            actual: "win:none".to_string(),
        });
    };
    let live = (hwnd_id(live_active), hwnd_id(live_focus));
    if live == (active, focus) && captured_pid == Some(live_pid) {
        return Ok(LiveTarget::Same { live_pid });
    }
    Ok(LiveTarget::Changed {
        actual: format_ref(BackendKind::Windows, live.0, live.1, Some(live_pid)),
    })
}

fn held_modifier_names() -> Option<Vec<String>> {
    let held: Vec<String> = TRACKED_MODIFIERS
        .iter()
        .filter(|(key, _)| unsafe { GetAsyncKeyState(i32::from(*key)) } as u16 & 0x8000 != 0)
        .map(|(_, name)| name.to_string())
        .collect();
    (!held.is_empty()).then_some(held)
}

fn window_title(window: HWND) -> Option<String> {
    let mut buffer = [0u16; 512];
    let length = unsafe { GetWindowTextW(window, buffer.as_mut_ptr(), buffer.len() as i32) };
    (length > 0).then(|| String::from_utf16_lossy(&buffer[..length as usize]))
}

/// The owning process's exe file name, without its directory.
fn process_exe_name(pid: u32) -> Option<String> {
    let process = unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid) };
    if process.is_null() {
        return None;
    }
    let mut buffer = [0u16; 512];
    let mut length = buffer.len() as u32;
    let ok = unsafe {
        QueryFullProcessImageNameW(
            process,
            PROCESS_NAME_WIN32,
            buffer.as_mut_ptr(),
            &mut length,
        )
    };
    unsafe { CloseHandle(process) };
    if ok == 0 {
        return None;
    }
    let path = String::from_utf16_lossy(&buffer[..length as usize]);
    path.rsplit(['\\', '/']).next().map(str::to_string)
}

fn key_event(unit: u16, flags: u32) -> INPUT {
    INPUT {
        r#type: INPUT_KEYBOARD,
        Anonymous: INPUT_0 {
            ki: KEYBDINPUT {
                wVk: 0,
                wScan: unit,
                dwFlags: KEYEVENTF_UNICODE | flags,
                time: 0,
                dwExtraInfo: 0,
            },
        },
    }
}
