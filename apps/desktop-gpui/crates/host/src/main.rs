//! The `starling-runtime-host` binary: one per user session.
//!
//! Launch → serve until SIGINT/SIGTERM (graceful: `bye` to clients,
//! machines join, lease released, endpoint removed) → exit 0.
//!
//! Exit contract for launchers (stdout is one JSON line each):
//! - `{"status":"owner",…}` then `{"status":"stopped"}` — this process
//!   was the host and served.
//! - `{"status":"already-running","socket":…,"owner":…}` exit 0 — a live
//!   owner holds the data-root lease; the launcher connects a client to
//!   `socket` instead of starting anything.
//! - anything else: stderr explains, exit 1.

use std::sync::atomic::{AtomicBool, Ordering};

use starling_runtime_host::{default_data_root, platform, serve, HostConfig};

static SHUTDOWN: AtomicBool = AtomicBool::new(false);

#[cfg(unix)]
extern "C" fn on_signal(_signal: i32) {
    // Async-signal-safe: one atomic store, nothing else.
    SHUTDOWN.store(true, Ordering::SeqCst);
}

fn usage() -> ! {
    eprintln!(
        "starling-runtime-host — the Starling runtime service host (E17 I4)

USAGE:
    starling-runtime-host [--root <dir>] [--runtime-dir <dir>]

OPTIONS:
    --root <dir>         storage v2 data root to own
                         (default: the platform default root)
    --runtime-dir <dir>  directory for the IPC endpoint
                         (default: {})",
        platform::default_runtime_dir().display()
    );
    std::process::exit(2);
}

/// `--help` is not an error: print to stdout and exit 0, the convention
/// launchers and scripts probing the interface rely on.
fn help() -> ! {
    println!(
        "starling-runtime-host — the Starling runtime service host (E17 I4)

USAGE:
    starling-runtime-host [--root <dir>] [--runtime-dir <dir>]

OPTIONS:
    --root <dir>         storage v2 data root to own
                         (default: the platform default root)
    --runtime-dir <dir>  directory for the IPC endpoint
                         (default: {})",
        platform::default_runtime_dir().display()
    );
    std::process::exit(0);
}

fn main() {
    // Signals first, before any fallible startup step: store open, lease
    // acquisition (with stale-lease breaking) and the endpoint probe can
    // all take a while, and a SIGTERM in that window must set the flag
    // this loop reads rather than hit the default disposition (which
    // would kill the process with the lease held and the endpoint
    // half-prepared). The handler is one atomic store, so registering it
    // this early is safe.
    #[cfg(unix)]
    unsafe {
        // SAFETY: `on_signal` only stores to a static atomic; `signal`
        // registers it. This is the standard no-dependency Ctrl-C path.
        // SIG_ERR would leave the default disposition in place — the
        // exact abrupt-SIGTERM-with-lease-held death the early
        // registration exists to prevent — so it is checked, not
        // ignored.
        if libc::signal(libc::SIGINT, on_signal as *const () as libc::sighandler_t) == libc::SIG_ERR
            || libc::signal(libc::SIGTERM, on_signal as *const () as libc::sighandler_t)
                == libc::SIG_ERR
        {
            eprintln!("starling-runtime-host: could not install signal handlers");
            std::process::exit(1);
        }
    }
    #[cfg(not(unix))]
    {
        // Windows: no console-ctrl handler wired yet (recorded gap);
        // the process still exits on window-close / taskkill and the OS
        // closes the pipe handles, so clients observe EOF, and the lease
        // flock-equivalent (the DACL'd pipe name) disappears with it.
    }

    let mut root: Option<std::path::PathBuf> = None;
    let mut runtime_dir: Option<std::path::PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--root" => root = Some(value_of(&mut args, &flag)),
            "--runtime-dir" => runtime_dir = Some(value_of(&mut args, &flag)),
            "--help" | "-h" => help(),
            other => {
                eprintln!("unknown argument {other:?}");
                usage();
            }
        }
    }

    let root = match root {
        Some(root) => root,
        None => match default_data_root() {
            Ok(root) => root,
            Err(err) => {
                eprintln!("starling-runtime-host: {err}");
                std::process::exit(1);
            }
        },
    };

    // The runtime dir rides into `production` so only the *final*
    // endpoint directory is created — applying an override afterwards
    // would leave the default directory behind as stray residue.
    let mut host = match HostConfig::production(&root, runtime_dir) {
        Ok(config) => match serve(config) {
            Ok(host) => host,
            Err(starling_runtime_host::HostError::OwnerLive {
                owner_id,
                owner_pid,
                socket_path,
            }) => {
                println!(
                    "{}",
                    serde_json::json!({
                        "status": "already-running",
                        "socket": socket_path,
                        "owner": owner_id,
                        "ownerPid": owner_pid,
                    })
                );
                std::process::exit(0);
            }
            Err(err) => {
                eprintln!("starling-runtime-host: {err}");
                std::process::exit(1);
            }
        },
        Err(err) => {
            eprintln!("starling-runtime-host: {err}");
            std::process::exit(1);
        }
    };

    println!(
        "{}",
        serde_json::json!({
            "status": "owner",
            "socket": host.socket_path(),
            "owner": host.owner_id(),
            "pid": std::process::id(),
        })
    );

    while !SHUTDOWN.load(Ordering::SeqCst) {
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    host.shutdown();
    println!("{}", serde_json::json!({ "status": "stopped" }));
}

fn value_of(args: &mut impl Iterator<Item = String>, flag: &str) -> std::path::PathBuf {
    match args.next() {
        Some(value) => value.into(),
        None => {
            eprintln!("{flag} needs a value");
            usage();
        }
    }
}
