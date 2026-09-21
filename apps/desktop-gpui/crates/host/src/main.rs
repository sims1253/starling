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

fn main() {
    let mut root: Option<std::path::PathBuf> = None;
    let mut runtime_dir: Option<std::path::PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--root" => root = Some(value_of(&mut args, &flag)),
            "--runtime-dir" => runtime_dir = Some(value_of(&mut args, &flag)),
            "--help" | "-h" => usage(),
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

    let mut config = match HostConfig::production(&root) {
        Ok(config) => config,
        Err(err) => {
            eprintln!("starling-runtime-host: {err}");
            std::process::exit(1);
        }
    };
    if let Some(dir) = runtime_dir {
        config.runtime_dir = dir;
    }

    let mut host = match serve(config) {
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

    #[cfg(unix)]
    unsafe {
        // SAFETY: `on_signal` only stores to a static atomic; `signal`
        // registers it. This is the standard no-dependency Ctrl-C path.
        libc::signal(libc::SIGINT, on_signal as *const () as libc::sighandler_t);
        libc::signal(libc::SIGTERM, on_signal as *const () as libc::sighandler_t);
    }
    #[cfg(not(unix))]
    {
        // Windows: no console-ctrl handler wired yet (recorded gap);
        // the process still exits on window-close / taskkill and the OS
        // closes the pipe handles, so clients observe EOF, and the lease
        // flock-equivalent (the DACL'd pipe name) disappears with it.
    }

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
