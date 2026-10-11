//! Process liveness by pid.

/// Whether process `pid` is alive. A process owned by another user
/// counts as alive; pid recycling is not detected.
#[cfg(unix)]
pub fn process_alive(pid: u32) -> bool {
    // 0 and values past pid_t would address process groups, not a pid.
    let Ok(pid) = libc::pid_t::try_from(pid) else {
        return false;
    };
    if pid <= 0 {
        return false;
    }
    // SAFETY: kill(2) with signal 0 performs existence and permission
    // checks only; no signal is delivered.
    if unsafe { libc::kill(pid, 0) } == 0 {
        return true;
    }
    // EPERM: exists but belongs to another user. Only ESRCH means gone.
    std::io::Error::last_os_error().raw_os_error() != Some(libc::ESRCH)
}

/// Whether process `pid` is alive. A process that cannot be opened
/// because it belongs to someone else counts as alive, as does one whose
/// exit code cannot be read.
#[cfg(windows)]
pub fn process_alive(pid: u32) -> bool {
    use windows_sys::Win32::Foundation::{CloseHandle, GetLastError, ERROR_ACCESS_DENIED};
    use windows_sys::Win32::System::Threading::{
        GetExitCodeProcess, OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION,
    };
    /// The exit code a running process reports.
    const STILL_ACTIVE: u32 = 259;
    // SAFETY: plain Win32 calls on a handle closed on every path that
    // opened it.
    unsafe {
        let handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
        if handle.is_null() {
            return GetLastError() == ERROR_ACCESS_DENIED;
        }
        let mut code = 0u32;
        let queried = GetExitCodeProcess(handle, &mut code);
        CloseHandle(handle);
        queried == 0 || code == STILL_ACTIVE
    }
}

/// Without a primitive to ask, a pid reads as gone.
#[cfg(not(any(unix, windows)))]
pub fn process_alive(_pid: u32) -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::process_alive;

    #[cfg(unix)]
    #[test]
    fn this_process_is_alive_and_an_exited_child_is_not() {
        assert!(process_alive(std::process::id()));
        let mut child = std::process::Command::new("true").spawn().expect("spawn true");
        let pid = child.id();
        child.wait().expect("wait for the child");
        assert!(!process_alive(pid), "pid {pid} has exited");
    }
}
