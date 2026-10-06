//! Shared helpers for the host's integration suites — each test binary
//! pulls this in with `mod common;`, so the suites cannot drift apart
//! on the helpers they share.

/// Whether the endpoint still exists: the socket file on unix; on
/// Windows (no filesystem entry — `\\.\pipe\…` lives in the kernel
/// namespace) whether a probe still finds a server bound to the name.
pub fn endpoint_present(socket: &std::path::Path) -> bool {
    #[cfg(unix)]
    {
        socket.exists()
    }
    #[cfg(windows)]
    {
        starling_runtime_host::platform::probe(socket)
            != starling_runtime_host::platform::Probe::Dead
    }
}
