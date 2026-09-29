//! Resource policy for model swaps (#363 step 4).
//!
//! A rolling swap briefly holds two models resident. On the reference
//! notebook (Ryzen 5 PRO 5650U, Vega 7 iGPU) system RAM is the binding
//! limit: UMA iGPUs allocate Vulkan buffers from system memory, so
//! "free VRAM" is not a separate pool. `estimate_resident` is therefore a
//! deliberately conservative *system-memory* estimate; discrete-GPU VRAM
//! is not checked (a 12 GB model on an 8 GB-dGPU/32 GB-RAM box looks
//! fine to this policy — that limitation is stated rather than hidden).

use std::fs;
use std::path::Path;

/// The safety margin a swap must leave beyond the incoming model's
/// estimate (default 512 MiB; tests pass smaller values).
pub const SWAP_MARGIN_BYTES: u64 = 512 * 1024 * 1024;

/// The fixed overhead part of [`estimate_resident`]: runtime buffers,
/// activations, and the audio/journal pipeline around the weights.
pub const RESIDENT_OVERHEAD_BYTES: u64 = 384 * 1024 * 1024;

/// Conservative system-RAM estimate for a loaded model of `file_size`
/// bytes: weights + 1/8 of the weights (quantization scratch, KV-ish
/// state) + a fixed pipeline overhead. Prefer over-asking to OOM: when
/// this estimate refuses, the drain-then-swap path still completes the
/// switch, just without the two-models-resident window.
pub fn estimate_resident(file_size: u64) -> u64 {
    file_size + file_size / 8 + RESIDENT_OVERHEAD_BYTES
}

/// The decision the swap policy reaches for a given memory situation.
/// Pure — every branch is unit-tested, and the manager treats
/// [`SwapPlan::Unknown`] as rolling-with-a-notice rather than refusing to
/// ever switch on platforms without a reading (macOS).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SwapPlan {
    /// Both models fit: load the incoming one while the outgoing serves.
    Rolling,
    /// The pair fits only once the outgoing model is unloaded: drain
    /// (finish the open take), stop the old, then load the new.
    NeedsDrain { short_by: u64 },
    /// Not enough memory even after draining: refuse with numbers.
    Refuse { needed: u64, available: u64 },
    /// No memory reading available.
    Unknown,
}

/// Decides how a swap can proceed:
///
/// - `Rolling` when `available >= incoming + margin`;
/// - else `NeedsDrain` when `available + outgoing >= incoming + margin`
///   (the outgoing model's memory is freed first);
/// - else `Refuse`.
///
/// `outgoing` being `None` (no model resident) collapses `NeedsDrain`
/// into `Rolling`-or-`Refuse`, which is the correct first-activation
/// behavior: there is nothing to drain.
pub fn swap_plan(
    available: Option<u64>,
    incoming_est: u64,
    outgoing_est: Option<u64>,
    margin: u64,
) -> SwapPlan {
    let Some(available) = available else {
        return SwapPlan::Unknown;
    };
    let needed = incoming_est.saturating_add(margin);
    if available >= needed {
        return SwapPlan::Rolling;
    }
    let with_outgoing_freed = available.saturating_add(outgoing_est.unwrap_or(0));
    if with_outgoing_freed >= needed {
        return SwapPlan::NeedsDrain {
            short_by: needed - available,
        };
    }
    SwapPlan::Refuse { needed, available }
}

/// System memory currently available for new allocations.
///
/// - Linux: `/proc/meminfo` `MemAvailable` (the honest number, unlike
///   `MemFree` which ignores reclaimable caches);
/// - Windows: `GlobalMemoryStatusEx().ullAvailPhys`;
/// - macOS: `None` — there is no stable supported API for this
///   (private `host_statistics64` lies under memory compression), so the
///   swap policy runs in `Unknown` mode instead of trusting a bad number.
pub fn available_memory() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        read_meminfo_available(Path::new("/proc/meminfo"))
    }
    #[cfg(target_os = "windows")]
    {
        windows_available_memory()
    }
    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    {
        None
    }
}

/// Parses `MemAvailable: <n> kB` out of `/proc/meminfo`; injectable so
/// the parsing (not the file) is unit-testable.
pub fn read_meminfo_available(path: &Path) -> Option<u64> {
    let text = fs::read_to_string(path).ok()?;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            let kb: u64 = rest.trim().trim_end_matches("kB").trim().parse().ok()?;
            return Some(kb.saturating_mul(1024));
        }
    }
    None
}

#[cfg(target_os = "windows")]
fn windows_available_memory() -> Option<u64> {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    unsafe {
        let mut status: MEMORYSTATUSEX = std::mem::zeroed();
        status.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
        if GlobalMemoryStatusEx(&mut status) != 0 {
            Some(status.ullAvailPhys)
        } else {
            None
        }
    }
}

/// Peak resident set size of a process in bytes (Linux `VmHWM`, Windows
/// `GetProcessMemoryInfo` `PeakWorkingSetSize`); `None` when unknown.
/// Used for the switch report's peak-memory measurement (#363 step 4).
pub fn process_peak_rss(pid: u32) -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        read_proc_status_field(&format!("/proc/{pid}/status"), "VmHWM")
    }
    #[cfg(target_os = "windows")]
    {
        windows_process_memory(pid).map(|(peak, _rss)| peak)
    }
    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    {
        let _ = pid;
        None
    }
}

/// Current resident set size of a process in bytes (Linux `VmRSS`,
/// Windows working set); `None` when unknown.
pub fn process_rss(pid: u32) -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        read_proc_status_field(&format!("/proc/{pid}/status"), "VmRSS")
    }
    #[cfg(target_os = "windows")]
    {
        windows_process_memory(pid).map(|(_peak, rss)| rss)
    }
    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    {
        let _ = pid;
        None
    }
}

#[cfg(target_os = "linux")]
fn read_proc_status_field(path: &str, field: &str) -> Option<u64> {
    let text = fs::read_to_string(path).ok()?;
    let prefix = format!("{field}:");
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix(&prefix) {
            let kb: u64 = rest.trim().trim_end_matches("kB").trim().parse().ok()?;
            return Some(kb.saturating_mul(1024));
        }
    }
    None
}

#[cfg(target_os = "windows")]
fn windows_process_memory(pid: u32) -> Option<(u64, u64)> {
    use windows_sys::Win32::System::ProcessStatus::{
        GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS,
    };
    use windows_sys::Win32::System::Threading::{OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION};
    unsafe {
        let handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
        if handle.is_null() {
            return None;
        }
        let mut counters: PROCESS_MEMORY_COUNTERS = std::mem::zeroed();
        let mut ok = false;
        if GetProcessMemoryInfo(
            handle,
            &mut counters,
            std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
        ) != 0
        {
            ok = true;
        }
        windows_sys::Win32::Foundation::CloseHandle(handle);
        if ok {
            Some((
                counters.PeakWorkingSetSize as u64,
                counters.WorkingSetSize as u64,
            ))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimate_is_conservative() {
        // 552 670 624-byte parakeet q4: weights + 12.5% + 384 MiB.
        assert_eq!(
            estimate_resident(552_670_624),
            552_670_624 + 552_670_624 / 8 + RESIDENT_OVERHEAD_BYTES
        );
        assert_eq!(estimate_resident(0), RESIDENT_OVERHEAD_BYTES);
    }

    #[test]
    fn swap_plan_covers_every_branch() {
        let incoming = 1_000;
        let outgoing = 800;
        let margin = 512;

        // Both fit.
        assert_eq!(
            swap_plan(Some(2_000), incoming, Some(outgoing), margin),
            SwapPlan::Rolling
        );
        // Exactly at the rolling boundary.
        assert_eq!(
            swap_plan(Some(incoming + margin), incoming, Some(outgoing), margin),
            SwapPlan::Rolling
        );
        // Pair fits only after the outgoing is freed (exactly at the
        // drain boundary).
        assert_eq!(
            swap_plan(
                Some(incoming + margin - outgoing),
                incoming,
                Some(outgoing),
                margin
            ),
            SwapPlan::NeedsDrain { short_by: outgoing }
        );
        // Even draining does not help.
        assert_eq!(
            swap_plan(Some(100), incoming, Some(outgoing), margin),
            SwapPlan::Refuse {
                needed: incoming + margin,
                available: 100
            }
        );
        // No outgoing model: NeedsDrain collapses to Rolling/Refuse.
        assert_eq!(
            swap_plan(Some(2_000), incoming, None, margin),
            SwapPlan::Rolling
        );
        assert_eq!(
            swap_plan(Some(incoming + margin - 1), incoming, None, margin),
            SwapPlan::Refuse {
                needed: incoming + margin,
                available: incoming + margin - 1
            }
        );
        // No reading.
        assert_eq!(
            swap_plan(None, incoming, Some(outgoing), margin),
            SwapPlan::Unknown
        );
    }

    #[test]
    fn meminfo_parsing_reads_mem_available() {
        let dir = tempfile::tempdir().expect("tempdir");
        let file = dir.path().join("meminfo");
        fs::write(
            &file,
            "MemTotal:       16000000 kB\nMemFree:         1000000 kB\nMemAvailable:   12345678 kB\n",
        )
        .expect("write meminfo");
        assert_eq!(read_meminfo_available(&file), Some(12_345_678 * 1024));
        assert_eq!(read_meminfo_available(&dir.path().join("absent")), None);
        fs::write(&file, "MemTotal: 16000000 kB\n").expect("rewrite");
        assert_eq!(read_meminfo_available(&file), None);
    }

    #[test]
    fn proc_status_fields_parse() {
        // The current process always has a VmRSS line on Linux; other
        // platforms simply report None.
        if cfg!(target_os = "linux") {
            let rss = process_rss(std::process::id()).expect("own VmRSS");
            assert!(rss > 0);
            assert!(process_peak_rss(std::process::id()).is_some());
        }
        assert_eq!(process_rss(u32::MAX), None);
    }
}
