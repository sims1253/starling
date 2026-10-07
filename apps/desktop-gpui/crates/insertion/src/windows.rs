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
//! compares both, checks the window still exists (`IsWindow`), and
//! compares pids (always known on Windows, and a same hwnd with a
//! different pid is a recycled handle, not the same target), mirroring
//! the X11 backend's identity rules.
//!
//! # Typing
//!
//! `SendInput` with `KEYEVENTF_UNICODE` injects one UTF-16 unit per
//! down+up pair, so any BMP or supplementary character (surrogate
//! pairs ride as two units, exactly as a real IME would commit them)
//! can be typed without depending on the user's keyboard layout at
//! all.
//!
//! Before the first key — and before every chunk of a long text —
//! the backend checks the *physically held* modifier keys
//! (`GetAsyncKeyState` on Ctrl/Alt/Shift/Win): characters typed while
//! the user still holds the dictation shortcut's modifiers become
//! commands. It waits up to [`MODIFIER_RELEASE_WAIT`] for release and
//! then refuses with [`InsertError::ModifiersHeld`]; no modifier
//! release is ever synthesized. After any wait the identity check
//! runs again before a key moves.
//!
//! The full `INPUT` array is prepared *before* the final
//! revalidation, then sent in chunks of at most
//! [`WIN_CHUNK_UTF16_UNITS`] UTF-16 units per `SendInput` call (a
//! surrogate pair never straddles a chunk), with the held-modifier and
//! identity checks repeated before every chunk. A send that comes
//! back short — input blocked mid-stream, classically UIPI against a
//! higher-privilege target, but also a switch of the input desktop —
//! is reported honestly as [`InsertError::PartialDelivery`]: a
//! character counts as possibly delivered the moment its key-DOWN
//! was accepted (the index of its first keydown event is below
//! `SendInput`'s returned count; a surrogate pair counts once its
//! first unit's keydown went in — a half-typed pair is
//! possibly-delivered garbage, not "nothing"), and the message never
//! claims nothing landed. With no key-down accepted at all the
//! underlying refusal (typically [`InsertError::PermissionDenied`])
//! is returned bare.
//!
//! # Deliberately absent in phase A
//!
//! No `SetForegroundWindow` (Starling never steals focus to type) and
//! no surrounding text (UIA is the later path; the capability table
//! reports it absent until then).

use crate::{
    chars_keyed_down_within_events, cheap_insertion_guards, deliver_in_chunks, format_ref,
    merge_excluded_pids, parse_ref, weighed_segments, Availability, BackendKind, ChunkFailure,
    InsertError, InsertReceipt, InsertionBackend, SurroundingText, TargetCheck, TargetSnapshot,
    MODIFIER_POLL_INTERVAL, MODIFIER_RELEASE_WAIT,
};

use windows_sys::Win32::Foundation::HWND;
use windows_sys::Win32::System::Threading::{
    OpenProcess, QueryFullProcessImageNameW, PROCESS_NAME_WIN32, PROCESS_QUERY_LIMITED_INFORMATION,
};
use windows_sys::Win32::UI::Input::KeyboardAndMouse::{
    GetAsyncKeyState, SendInput, INPUT, INPUT_KEYBOARD, KEYBDINPUT, KEYEVENTF_KEYUP,
    KEYEVENTF_UNICODE, VK_CONTROL, VK_LWIN, VK_MENU, VK_RWIN, VK_SHIFT,
};
use windows_sys::Win32::UI::WindowsAndMessaging::{
    GetForegroundWindow, GetGUIThreadInfo, GetWindowTextW, GetWindowThreadProcessId, IsWindow,
    GUITHREADINFO,
};

/// Maximum UTF-16 units per `SendInput` call; the held-modifier and
/// identity checks run before each chunk.
pub const WIN_CHUNK_UTF16_UNITS: usize = 32;

/// The modifier keys typing must not ride on, as (virtual key, name):
/// Ctrl and Alt turn characters into commands, Shift alters case, and
/// the Windows keys open the shell — none of them may be physically
/// held when dictation text starts (or continues) typing.
const TRACKED_MODIFIERS: &[(i32, &str)] = &[
    (VK_SHIFT as i32, "Shift"),
    (VK_CONTROL as i32, "Ctrl"),
    (VK_MENU as i32, "Alt"),
    (VK_LWIN as i32, "Left Windows"),
    (VK_RWIN as i32, "Right Windows"),
];

/// The Windows backend. Every call is a direct Win32 round trip, so
/// like the X11 backend it holds no state that can go stale between a
/// take's capture and its insert — except the pid exclusion policy it
/// is configured with (see [`Self::with_excluded_pids`]).
#[derive(Debug)]
pub struct WindowsBackend {
    excluded_pids: Vec<u32>,
}

impl WindowsBackend {
    pub fn new() -> WindowsBackend {
        WindowsBackend::with_excluded_pids(vec![std::process::id()])
    }

    /// Construct with an explicit Starling-ownership policy: every pid
    /// in `excluded_pids` marks a target this backend refuses to type
    /// into, checked against the **live** foreground window's pid at
    /// insert time. This process's own pid is always added, so the
    /// backend can never type into Starling itself.
    pub fn with_excluded_pids(excluded_pids: Vec<u32>) -> WindowsBackend {
        WindowsBackend {
            excluded_pids: merge_excluded_pids(excluded_pids),
        }
    }

    /// Exactly `excluded_pids`, **without** adding this process: only for
    /// the interactive test and probes that deliberately type into a
    /// window of their own. Production code uses
    /// [`Self::with_excluded_pids`] (or the [`crate::Inserter`]), which
    /// always protects this process.
    pub fn with_exact_excluded_pids_for_self_typing_tests(
        excluded_pids: Vec<u32>,
    ) -> WindowsBackend {
        WindowsBackend { excluded_pids }
    }

    /// The configured exclusion set (always contains this process).
    pub fn excluded_pids(&self) -> &[u32] {
        &self.excluded_pids
    }
}

impl Default for WindowsBackend {
    fn default() -> Self {
        WindowsBackend::new()
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

/// The live state of a frozen ref (the insert-time checks need the
/// live pid, which a `TargetCheck` cannot carry).
enum LiveTarget {
    Same { live_pid: u32 },
    Changed { expected: String, actual: String },
    Gone,
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

/// The live state of `target`'s ref: ids first, then the pid
/// comparison a plain id match cannot see (a recycled hwnd).
fn live_target(target: &TargetSnapshot) -> Result<LiveTarget, InsertError> {
    let Some((kind, active, focus, captured_pid)) = parse_ref(&target.target_ref) else {
        return Err(InsertError::Rejected {
            reason: format!("malformed target ref: {}", target.target_ref),
        });
    };
    debug_assert_eq!(kind, BackendKind::Windows, "the inserter routes by scheme");
    // A closed window is `Gone` even if focus also moved (the X11
    // backend's rule, for the same reason: "your target closed" is
    // the actionable fact).
    if unsafe { IsWindow(hwnd_from(active)) } == 0 {
        return Ok(LiveTarget::Gone);
    }
    match focus_pair() {
        Some((live_active, live_focus, live_pid))
            if hwnd_id(live_active) == active && hwnd_id(live_focus) == focus =>
        {
            match captured_pid {
                // Same hwnds, different owning process: the id was
                // recycled — not the same target.
                Some(captured) if captured != live_pid => Ok(LiveTarget::Changed {
                    expected: target.target_ref.clone(),
                    actual: format_ref(
                        kind,
                        hwnd_id(live_active),
                        hwnd_id(live_focus),
                        Some(live_pid),
                    ),
                }),
                // A pid where capture saw none: same shape as X11 —
                // treat as changed rather than trust the ids alone.
                None => Ok(LiveTarget::Changed {
                    expected: target.target_ref.clone(),
                    actual: format_ref(
                        kind,
                        hwnd_id(live_active),
                        hwnd_id(live_focus),
                        Some(live_pid),
                    ),
                }),
                _ => Ok(LiveTarget::Same { live_pid }),
            }
        }
        Some((live_active, live_focus, live_pid)) => Ok(LiveTarget::Changed {
            expected: target.target_ref.clone(),
            actual: format_ref(
                kind,
                hwnd_id(live_active),
                hwnd_id(live_focus),
                Some(live_pid),
            ),
        }),
        None => Ok(LiveTarget::Changed {
            expected: target.target_ref.clone(),
            actual: format!("{}:none", kind.scheme()),
        }),
    }
}

/// The names of the tracked modifier keys physically held right now
/// (`GetAsyncKeyState`'s high bit is "currently down", regardless of
/// which window would receive the key). `None` means none are held.
fn held_modifier_names() -> Option<Vec<String>> {
    let held: Vec<String> = TRACKED_MODIFIERS
        .iter()
        .filter(|(key, _)| (unsafe { GetAsyncKeyState(*key) } as u16) & 0x8000 != 0)
        .map(|(_, name)| (*name).to_string())
        .collect();
    (!held.is_empty()).then_some(held)
}

/// Wait (bounded, polling) for physically held modifiers to clear;
/// see [`MODIFIER_RELEASE_WAIT`]. No modifier release is ever
/// synthesized: the user's keyboard is the user's.
fn wait_modifiers_released() -> Result<(), InsertError> {
    let deadline = std::time::Instant::now() + MODIFIER_RELEASE_WAIT;
    loop {
        if let Some(held) = held_modifier_names() {
            if std::time::Instant::now() >= deadline {
                return Err(InsertError::ModifiersHeld { held });
            }
            std::thread::sleep(MODIFIER_POLL_INTERVAL);
        } else {
            return Ok(());
        }
    }
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
        QueryFullProcessImageNameW(
            process,
            PROCESS_NAME_WIN32,
            buffer.as_mut_ptr(),
            &mut length,
        )
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
        if self.excluded_pids.contains(&pid) {
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
        Ok(match live_target(target)? {
            LiveTarget::Same { .. } => TargetCheck::Same,
            LiveTarget::Changed { expected, actual } => TargetCheck::Changed { expected, actual },
            LiveTarget::Gone => TargetCheck::Gone,
        })
    }

    fn surrounding_text(
        &self,
        _target: &TargetSnapshot,
    ) -> Result<Option<SurroundingText>, InsertError> {
        // Phase A: no UIA yet; the capability table says absent.
        Ok(None)
    }

    fn insert(&self, target: &TargetSnapshot, text: &str) -> Result<InsertReceipt, InsertError> {
        cheap_insertion_guards(text, target.pid, &self.excluded_pids)?;

        // ---- prepare everything BEFORE the final revalidation ----
        // One down+up pair per UTF-16 unit, `wVk` zero and the unit in
        // `wScan` per the `KEYEVENTF_UNICODE` contract. Surrogate
        // halves pass through as their own units, which is how a real
        // IME commits supplementary characters too.
        let events: Vec<INPUT> = text
            .encode_utf16()
            .flat_map(|unit| [key_event(unit, false), key_event(unit, true)])
            .collect();
        let segments = weighed_segments(text, WIN_CHUNK_UTF16_UNITS, char::len_utf16);
        let total_chars = text.chars().count();

        // ---- the final revalidation, then chunks with rechecks ----
        // (`unit_cursor` tracks where each segment's events start: the
        // array is one flat down/up stream, segments are unit-aligned)
        let mut unit_cursor = 0usize;
        deliver_in_chunks(
            total_chars,
            &segments,
            || self.chunk_check(target),
            |segment, _delivered| {
                let units = segment.encode_utf16().count();
                let chunk = &events[unit_cursor * 2..(unit_cursor + units) * 2];
                unit_cursor += units;
                let sent = unsafe {
                    SendInput(
                        chunk.len() as u32,
                        chunk.as_ptr(),
                        std::mem::size_of::<INPUT>() as i32,
                    )
                };
                if sent != chunk.len() as u32 {
                    // A short count means the stream was cut: blocked
                    // input is *not* all-or-nothing across a chunked
                    // delivery, so never claim nothing landed. What
                    // went in are the first `sent` events of this
                    // chunk's slice, i.e. the first
                    // `(unit_cursor - units) * 2 + sent` events of the
                    // whole text's array — and a character counts as
                    // possibly delivered the moment its key-DOWN (an
                    // even event index) was accepted: a cut right
                    // after a keydown still counts that character,
                    // and a surrogate pair counts once its first
                    // unit's keydown went in.
                    let accepted_events = (unit_cursor - units) * 2 + sent as usize;
                    return Err(ChunkFailure {
                        delivered: chars_keyed_down_within_events(text, accepted_events),
                        cause: InsertError::PermissionDenied {
                            reason: "the input stream was cut part-way, possibly blocked by a \
                                     higher-privilege target (UIPI) or an input desktop change"
                                .to_string(),
                            settings_hint: "run Starling at the same integrity level as the \
                                            target, or use the copy fallback"
                                .to_string(),
                        },
                    });
                }
                Ok(segment.chars().count())
            },
        )
    }
}

impl WindowsBackend {
    /// The before-every-chunk check: held modifiers waited out, then
    /// the live identity (and Starling ownership) revalidated —
    /// immediately before the keys. After a seconds-long modifier wait
    /// the world may have moved, which is exactly why the identity
    /// check lives here and not only at the top of `insert`.
    fn chunk_check(&self, target: &TargetSnapshot) -> Result<(), InsertError> {
        wait_modifiers_released()?;
        match live_target(target)? {
            LiveTarget::Same { live_pid } => {
                if self.excluded_pids.contains(&live_pid) {
                    return Err(InsertError::TargetIsStarling);
                }
                Ok(())
            }
            LiveTarget::Changed { expected, actual } => {
                Err(InsertError::TargetChanged { expected, actual })
            }
            LiveTarget::Gone => Err(InsertError::TargetGone),
        }
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
