//! The `starling-runtime-host` binary: one per user session.
//!
//! Launch → serve until SIGINT/SIGTERM (Windows: console Ctrl+C/Break,
//! close, logoff, shutdown — see `on_console_ctrl`) → graceful stop
//! (`bye` to clients, machines join, the supervised engine stops, lease
//! released, endpoint removed) → exit 0.
//!
//! Exit contract for launchers (stdout is one JSON line each):
//! - `{"status":"owner",…}` then `{"status":"stopped"}` — this process
//!   was the host and served.
//! - `{"status":"already-running","socket":…,"owner":…}` exit 0 — a live
//!   owner holds the data-root lease; the launcher connects a client to
//!   `socket` instead of starting anything.
//! - anything else: stderr explains, exit 1.

use std::sync::atomic::{AtomicBool, Ordering};

use starling_runtime_host::engine::EngineChoice;
use starling_runtime_host::{default_data_root, platform, serve, HostConfig};

static SHUTDOWN: AtomicBool = AtomicBool::new(false);
/// Set once `host.shutdown()` returned — what the Windows close/logoff/shutdown
/// handler waits for before letting the OS end the process.
static STOPPED: AtomicBool = AtomicBool::new(false);

#[cfg(unix)]
extern "C" fn on_signal(_signal: i32) {
    // Async-signal-safe: one atomic store, nothing else.
    SHUTDOWN.store(true, Ordering::SeqCst);
}

/// The Windows analogue of SIGINT/SIGTERM: Ctrl+C/Ctrl+Break, console
/// close, logoff and system shutdown all request the same graceful
/// stop. The handler runs on its own thread; for the close/logoff/
/// shutdown events the OS ends the process as soon as the handler
/// returns, so it waits (bounded, inside the OS's grace period) for the
/// main loop to finish shutting down — lease released, engine stopped.
/// (Windows withholds logoff/shutdown from console processes that load
/// user32/gdi32; then, like any force-kill past the grace period, the
/// process simply dies — still safe: the lease lock (`LockFileEx`) and
/// every pipe handle are kernel objects the OS releases at process
/// death, so ownership cannot be stranded; the successor's
/// crash-recovery sweep handles the rest.)
#[cfg(windows)]
unsafe extern "system" fn on_console_ctrl(ctrl_type: u32) -> windows_sys::Win32::Foundation::BOOL {
    use windows_sys::Win32::System::Console::{
        CTRL_CLOSE_EVENT, CTRL_LOGOFF_EVENT, CTRL_SHUTDOWN_EVENT,
    };
    SHUTDOWN.store(true, Ordering::SeqCst);
    if matches!(
        ctrl_type,
        CTRL_CLOSE_EVENT | CTRL_LOGOFF_EVENT | CTRL_SHUTDOWN_EVENT
    ) {
        let deadline = std::time::Instant::now() + std::time::Duration::from_millis(4500);
        while !STOPPED.load(Ordering::SeqCst) && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
    }
    1
}

fn usage() -> ! {
    eprintln!(
        "starling-runtime-host — the Starling runtime service host (E17 I4)

USAGE:
    starling-runtime-host [--root <dir>] [--runtime-dir <dir>] [--engine <source>]

OPTIONS:
    --root <dir>         storage v2 data root to own
                         (default: the platform default root)
    --runtime-dir <dir>  directory for the IPC endpoint
                         (default: {})
    --engine <source>    settings: the transcription engine the desktop
                         settings choose (bundled engine supervised by
                         this host, or the manual server) — the default;
                         the file is followed while the host runs, so
                         engine changes apply without a restart;
                         none: no engine (jobs fail no_provider_configured)",
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
    starling-runtime-host [--root <dir>] [--runtime-dir <dir>] [--engine <source>]

OPTIONS:
    --root <dir>         storage v2 data root to own
                         (default: the platform default root)
    --runtime-dir <dir>  directory for the IPC endpoint
                         (default: {})
    --engine <source>    settings: the transcription engine the desktop
                         settings choose (bundled engine supervised by
                         this host, or the manual server) — the default;
                         the file is followed while the host runs, so
                         engine changes apply without a restart;
                         none: no engine (jobs fail no_provider_configured)",
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
        if libc::signal(libc::SIGINT, on_signal as *const () as libc::sighandler_t)
            == libc::SIG_ERR
            || libc::signal(libc::SIGTERM, on_signal as *const () as libc::sighandler_t)
                == libc::SIG_ERR
        {
            eprintln!("starling-runtime-host: could not install signal handlers");
            std::process::exit(1);
        }
    }
    #[cfg(windows)]
    unsafe {
        // SAFETY: registers a handler that only touches static atomics.
        if windows_sys::Win32::System::Console::SetConsoleCtrlHandler(Some(on_console_ctrl), 1)
            == 0
        {
            eprintln!("starling-runtime-host: could not install the console control handler");
            std::process::exit(1);
        }
    }

    let mut root: Option<std::path::PathBuf> = None;
    let mut runtime_dir: Option<std::path::PathBuf> = None;
    let mut engine_source = "settings".to_string();
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--root" => root = Some(value_of(&mut args, &flag)),
            "--runtime-dir" => runtime_dir = Some(value_of(&mut args, &flag)),
            "--engine" => {
                engine_source = value_of(&mut args, &flag).to_string_lossy().into_owned();
                if engine_source != "settings" && engine_source != "none" {
                    eprintln!("--engine takes settings or none, not {engine_source:?}");
                    usage();
                }
            }
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
    //
    // `--engine settings` also *follows* the file while the host runs
    // (the app applies engine changes immediately; the host must not
    // freeze its startup choice): the watcher reads the same path the
    // startup load read. An unresolvable settings path cannot change
    // under us either — nothing to watch, the choice stays the startup
    // one. `--engine none` watches nothing.
    let (engine, settings_path) = if engine_source == "none" {
        (EngineChoice::None, None)
    } else {
        let settings = starling_dictation::settings::Settings::load_or_default();
        let choice = match EngineChoice::from_settings(&settings) {
            Ok(engine) => engine,
            Err(err) => {
                eprintln!("starling-runtime-host: {err}");
                std::process::exit(1);
            }
        };
        (choice, starling_dictation::settings::Settings::default_path().ok())
    };

    let mut host = match HostConfig::production(&root, runtime_dir) {
        Ok(config) => {
            let config = match settings_path {
                Some(path) => config.with_engine(engine).with_settings_path(path),
                None => config.with_engine(engine),
            };
            match serve(config) {
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
            }
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
            // The effective engine, not the startup choice's label: a
            // manual endpoint that did not validate reads as
            // `unconfigured` here, matching what jobs actually face.
            "engine": host.engine_label(),
        })
    );

    while !SHUTDOWN.load(Ordering::SeqCst) {
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    host.shutdown();
    // Before the println: the console-close handler's bounded wait ends
    // as soon as the flag is set, so the stop work must be accounted
    // first — printing after it keeps the handler's remaining grace
    // budget real.
    STOPPED.store(true, Ordering::SeqCst);
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
