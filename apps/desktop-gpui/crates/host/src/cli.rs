//! The host's command line: the `starling-runtime-host` binary, and the
//! desktop app's `starling-gpui --runtime-host` (#220: the app launches
//! its own binary as the host, so the two are always the same build and
//! packaging ships one executable). One host per user session.
//!
//! Launch → serve until SIGINT/SIGTERM (Windows: console Ctrl+C/Break
//! or close; at logoff/shutdown an interactive-session process is
//! terminated by the OS — see `on_console_ctrl`) → graceful stop
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
//!
//! With `--exit-when-idle <seconds>` (what the app passes) the host also
//! stops gracefully once it has been idle that long — no client
//! connected, no take recording or being stored, no job — so a host the
//! app started does not outlive the app by more than that.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use crate::engine::EngineChoice;
use crate::{default_data_root, platform, serve, HostConfig};

static SHUTDOWN: AtomicBool = AtomicBool::new(false);
/// Set once `host.shutdown()` returned — what the Windows console-close
/// handler waits for before letting the OS end the process.
static STOPPED: AtomicBool = AtomicBool::new(false);

#[cfg(unix)]
extern "C" fn on_signal(_signal: i32) {
    // Async-signal-safe: one atomic store, nothing else.
    SHUTDOWN.store(true, Ordering::SeqCst);
}

/// The Windows analogue of SIGINT/SIGTERM: Ctrl+C/Ctrl+Break and a
/// console close request the graceful stop. The handler runs on its own
/// thread; for a close the OS ends the process as soon as the handler
/// returns, so it waits (bounded, inside the OS's grace period) for the
/// main loop to finish shutting down — lease released, engine stopped.
/// `CTRL_LOGOFF_EVENT`/`CTRL_SHUTDOWN_EVENT` reach only services (an
/// interactive-session process is terminated before they are sent);
/// their arms stay so a host run as a service stops gracefully too.
/// Termination without the handshake is still safe: the lease lock
/// (`LockFileEx`) and every pipe handle are kernel objects the OS
/// releases at process death, so ownership cannot be stranded — the
/// successor's crash-recovery sweep handles the rest.
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
                          [--exit-when-idle <seconds>] [--orphan-grace <seconds>]

OPTIONS:
    --root <dir>         storage v2 data root to own
                         (default: the platform default root)
    --runtime-dir <dir>  directory for the IPC endpoint
                         (default: {})
    --engine <source>    settings: the transcription engine the desktop
                         settings choose (bundled engine supervised by
                         this host, or the manual server) — the default;
                         the file is followed while the host runs, so
                         model and mode changes apply without a restart
                         (a CPU/automatic backend change applies at the
                         next start);
                         none: no engine (jobs fail no_provider_configured)
    --exit-when-idle <seconds>
                         stop once idle this long (no client, no take,
                         no job); the desktop app starts its host so
                         (default: serve until signalled)
    --orphan-grace <seconds>
                         how long a take records with no app following
                         it before the host stops and stores it
                         (default {})",
        platform::default_runtime_dir().display(),
        crate::takes::DEFAULT_ORPHAN_GRACE.as_secs()
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
                          [--exit-when-idle <seconds>] [--orphan-grace <seconds>]

OPTIONS:
    --root <dir>         storage v2 data root to own
                         (default: the platform default root)
    --runtime-dir <dir>  directory for the IPC endpoint
                         (default: {})
    --engine <source>    settings: the transcription engine the desktop
                         settings choose (bundled engine supervised by
                         this host, or the manual server) — the default;
                         the file is followed while the host runs, so
                         model and mode changes apply without a restart
                         (a CPU/automatic backend change applies at the
                         next start);
                         none: no engine (jobs fail no_provider_configured)
    --exit-when-idle <seconds>
                         stop once idle this long (no client, no take,
                         no job); the desktop app starts its host so
                         (default: serve until signalled)
    --orphan-grace <seconds>
                         how long a take records with no app following
                         it before the host stops and stores it
                         (default {})",
        platform::default_runtime_dir().display(),
        crate::takes::DEFAULT_ORPHAN_GRACE.as_secs()
    );
    std::process::exit(0);
}

/// Runs the host with `args` (the arguments after the program name, or
/// after `--runtime-host`) and returns the process exit code; the caller
/// exits with it.
pub fn run(args: Vec<String>) -> i32 {
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
            return 1;
        }
    }
    #[cfg(windows)]
    unsafe {
        // SAFETY: registers a handler that only touches static atomics.
        if windows_sys::Win32::System::Console::SetConsoleCtrlHandler(Some(on_console_ctrl), 1)
            == 0
        {
            eprintln!("starling-runtime-host: could not install the console control handler");
            return 1;
        }
    }

    let mut root: Option<std::path::PathBuf> = None;
    let mut runtime_dir: Option<std::path::PathBuf> = None;
    let mut engine_source = "settings".to_string();
    let mut exit_when_idle: Option<Duration> = None;
    let mut orphan_grace: Option<Duration> = None;
    let mut args = args.into_iter();
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
            "--exit-when-idle" => exit_when_idle = Some(seconds_of(&mut args, &flag)),
            "--orphan-grace" => orphan_grace = Some(seconds_of(&mut args, &flag)),
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
                return 1;
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
        use starling_dictation::settings::Settings;
        // One read decides both the settings and the diagnostic: a file
        // that is not JSON falls back to the defaults (as
        // `load_or_default` would) and says so, rather than silently
        // starting the bundled engine on a corrupt file (the watcher then
        // applies the file's real choice once it parses).
        let settings = match Settings::default_path() {
            Ok(path) => match std::fs::read(&path) {
                Ok(bytes) => Settings::from_json_bytes(&bytes).unwrap_or_else(|| {
                    eprintln!(
                        "starling-runtime-host: settings at {} are not valid JSON; \
                         starting with the default engine settings",
                        path.display()
                    );
                    Settings::default_settings()
                }),
                Err(_) => Settings::default_settings(),
            },
            Err(_) => Settings::default_settings(),
        };
        // An engine choice that cannot resolve (no user data directory)
        // must not cost a second launch its ownership answer: serve
        // without an engine and say so, rather than exit before the
        // lease ladder ran.
        let choice = match EngineChoice::from_settings(&settings) {
            Ok(engine) => engine,
            Err(err) => {
                eprintln!("starling-runtime-host: {err}; serving without an engine");
                EngineChoice::None
            }
        };
        (choice, starling_dictation::settings::Settings::default_path().ok())
    };

    let mut host = match HostConfig::production(&root, runtime_dir) {
        Ok(config) => {
            let config = config.with_agent_allowlist(Some(
                root.join(crate::agent::ALLOWLIST_FILE),
            ));
            let config = match settings_path {
                Some(path) => config.with_engine(engine).with_settings_path(path),
                None => config.with_engine(engine),
            };
            let config = match orphan_grace {
                Some(grace) => config.with_orphan_grace(grace),
                None => config,
            };
            match serve(config) {
                Ok(host) => host,
                Err(crate::HostError::OwnerLive {
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
                    return 0;
                }
                Err(err) => {
                    eprintln!("starling-runtime-host: {err}");
                    return 1;
                }
            }
        }
        Err(err) => {
            eprintln!("starling-runtime-host: {err}");
            return 1;
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

    let mut idle_since: Option<Instant> = None;
    while !SHUTDOWN.load(Ordering::SeqCst) {
        std::thread::sleep(Duration::from_millis(100));
        if let Some(limit) = exit_when_idle {
            if host.idle() {
                if idle_since.get_or_insert_with(Instant::now).elapsed() >= limit {
                    break;
                }
            } else {
                idle_since = None;
            }
        }
    }
    host.shutdown();
    // Print (and flush) before setting the flag: on a console close the
    // OS ends the process as soon as the handler's wait sees `STOPPED`,
    // so a line written after it could be lost. The write is fallible:
    // stdout may already be gone on a console close, and a panicking
    // `println!` would never set the flag.
    {
        use std::io::Write;
        let mut stdout = std::io::stdout().lock();
        let _ = writeln!(stdout, "{}", serde_json::json!({ "status": "stopped" }));
        let _ = stdout.flush();
    }
    STOPPED.store(true, Ordering::SeqCst);
    0
}

fn seconds_of(args: &mut impl Iterator<Item = String>, flag: &str) -> Duration {
    let value = value_of(args, flag);
    match value.to_string_lossy().parse::<u64>() {
        Ok(seconds) => Duration::from_secs(seconds),
        Err(_) => {
            eprintln!("{flag} takes a whole number of seconds, not {value:?}");
            usage();
        }
    }
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
