//! The app's link to the runtime host (#220).
//!
//! Takes are recorded, journaled and stored by the per-user runtime host
//! (`starling-runtime-host`, or this binary run as `--runtime-host`), not
//! by the app: the host holds the store lease and the recorder's journal
//! tree, so killing the app never costs a take and a second app window
//! is just another client. The host also transcribes the takes it
//! stores: live text while a take records, the transcript once it is
//! stored, the retries the app asks for — and its store is the app's: the
//! history the app shows and changes goes through the same connection
//! ([`crate::store`]). The app keeps what needs its window — the
//! activation machine, the live view, staging, delivery.
//!
//! [`HostLink`] owns the connection on its own thread: it connects to the
//! host serving the default data root, starts one when nothing serves
//! (this executable, detached, idle-exiting), follows the take feed and
//! reconnects — starting the host again if it is gone — whenever the
//! connection drops. The UI hears about it through [`HostUpdate`]s.

use std::io::BufRead;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use starling_dictation::disk::DiskReading;
use starling_dictation::microphone::{InputProblem, InputRoute};
use starling_dictation::recorder::RecorderFault;
use starling_runtime::machine::capture::LiveTakeStatus;
use starling_runtime::protocol::Command;
use starling_runtime_host::client::{EventWire, HostClient, TakeWire};
use starling_runtime_host::frame::{
    HostRecovery, LivePartial, TakeBusy, TakeOwner, TranscribeWith, TranscriptionState,
};
use starling_runtime_host::version::{BuildStamp, RetireAnswer};
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};

/// How long a host this app started stays up with nothing to do.
const HOST_IDLE_EXIT: Duration = Duration::from_secs(60);

/// How long a started host may take to say whether it serves.
const HOST_START_TIMEOUT: Duration = Duration::from_secs(30);

/// Reconnect backoff: first retry, and the cap it doubles up to.
const RETRY_FIRST: Duration = Duration::from_millis(250);
const RETRY_MAX: Duration = Duration::from_secs(4);

/// How many times in a row the link starts a host that never serves
/// before it stops starting one (it still connects to one that appears)
/// until the user asks again ([`HostLink::retry`]).
const LAUNCH_ATTEMPTS: u32 = 3;

/// The host's endpoint for the default data root (what a host started
/// with no `--root`/`--runtime-dir` serves).
pub(crate) fn default_endpoint() -> Result<PathBuf, String> {
    let root = starling_runtime_host::default_data_root()?;
    Ok(starling_runtime_host::platform::socket_path(
        &starling_runtime_host::platform::default_runtime_dir(),
        &root,
    ))
}

/// What the UI hears from the link.
pub(crate) enum HostUpdate {
    Connected {
        client: Arc<HostClient>,
        recovery: Option<HostRecovery>,
    },
    /// The connection is gone (or never came up); the link is retrying.
    /// `gave_up`: starting the recording service failed
    /// [`LAUNCH_ATTEMPTS`] times in a row, and the link no longer starts
    /// it until [`HostLink::retry`]. `host_gone`: the service itself
    /// stopped (it took any take it was recording with it), not just the
    /// connection to it.
    Disconnected {
        reason: String,
        gave_up: bool,
        host_gone: bool,
        /// `reason` is the whole sentence to show (a version mismatch:
        /// it says what is going on and what to do).
        plain: bool,
    },
    /// A take-feed frame.
    Take(TakeUpdate),
    /// A runtime event the UI acts on (`capture.error`).
    Event(EventWire),
    /// A command for `take` was not carried out (refused by the runtime,
    /// or no connection to send it on). `busy`: a refused start, and what
    /// holds the microphone.
    Refused {
        take: String,
        command: &'static str,
        reason: String,
        busy: Option<TakeBusy>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum TakeUpdate {
    Live {
        take: String,
        rate: u32,
        status: Option<LiveTakeStatus>,
        owner: TakeOwner,
        ended: Option<u64>,
        /// With `ended`: a [`TakeUpdate::Persisted`] follows.
        kept: bool,
        /// The take's newest samples, for this window's level meter (on
        /// the owner's status ticks).
        meter: Option<Vec<f32>>,
    },
    StartFailed {
        take: String,
        problem: Option<InputProblem>,
        message: String,
    },
    Persisted {
        take: String,
        stored_id: Option<String>,
        interrupted: bool,
        error: Option<String>,
        orphan: bool,
    },
    Notice(HostRecovery),
    /// What a pass of the host's history audio upkeep did.
    Upkeep { report: String, retired: bool },
    /// Another window (or this one) changed the history list.
    HistoryChanged,
    /// A recording take's live text, or why it stopped.
    LiveText {
        take: String,
        partial: Option<LivePartial>,
        degraded: Option<String>,
    },
    /// The host's engine as it stands now (see
    /// [`starling_runtime_host::frame::Frame::EngineState`]).
    Engine(Box<starling_runtime_host::engine::EngineStatus>),
    /// Where a stored take's transcription stands (see
    /// [`starling_runtime_host::frame::Frame::Transcription`]).
    Transcription {
        stored_id: String,
        take: Option<String>,
        req: Option<String>,
        attempt: Option<String>,
        state: TranscriptionState,
        yours: bool,
    },
}

/// How the link reaches a host when none serves.
#[derive(Clone)]
pub(crate) enum Launch {
    /// Start this executable as `--runtime-host`, logging to `log`.
    SelfAsHost { log: PathBuf },
    /// Never start one (tests serve their own host).
    Never,
}

/// The connection's owner: a thread that keeps the app connected, and
/// one that sends the app's commands — in the order the app issued them,
/// so a take's stop or abort can never overtake its start.
pub(crate) struct HostLink {
    commands: std::sync::mpsc::Sender<Outgoing>,
    stop: Arc<AtomicBool>,
    relaunch: Arc<AtomicBool>,
    current: Arc<Mutex<Option<Arc<HostClient>>>>,
}

enum Outgoing {
    Command { take: String, command: Command },
    Adopt { take: String },
    Transcribe {
        req: String,
        stored_id: String,
        with: TranscribeWith,
    },
    TranscribeDue { stored_id: String },
}

impl HostLink {
    /// Starts the link thread for `endpoint`; updates arrive on the
    /// returned receiver.
    pub(crate) fn start(
        endpoint: PathBuf,
        launch: Launch,
    ) -> (HostLink, UnboundedReceiver<HostUpdate>) {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let stop = Arc::new(AtomicBool::new(false));
        let relaunch = Arc::new(AtomicBool::new(false));
        let current: Arc<Mutex<Option<Arc<HostClient>>>> = Arc::default();
        let (commands, outgoing) = std::sync::mpsc::channel();
        let spawned = std::thread::Builder::new()
            .name("starling-host-link".to_string())
            .spawn({
                let stop = Arc::clone(&stop);
                let relaunch = Arc::clone(&relaunch);
                let current = Arc::clone(&current);
                let tx = tx.clone();
                move || link_loop(endpoint, launch, current, stop, relaunch, tx)
            })
            .and_then(|_| {
                std::thread::Builder::new()
                    .name("starling-host-commands".to_string())
                    .spawn({
                        let tx = tx.clone();
                        let current = Arc::clone(&current);
                        move || command_loop(outgoing, current, tx)
                    })
            });
        if let Err(err) = spawned {
            let _ = tx.send(HostUpdate::Disconnected {
                reason: format!("the connection thread could not start: {err}"),
                gave_up: true,
                host_gone: false,
                plain: false,
            });
        }
        (
            HostLink {
                commands,
                stop,
                relaunch,
                current,
            },
            rx,
        )
    }

    /// Sends `command` for `take` after every command issued before it.
    pub(crate) fn command(&self, take: &str, command: Command) {
        let _ = self.commands.send(Outgoing::Command {
            take: take.to_string(),
            command,
        });
    }

    /// Takes on running take `take`, whose window is gone (see
    /// [`starling_runtime_host::frame::Frame::TakeAdopt`]).
    pub(crate) fn adopt(&self, take: &str) {
        let _ = self.commands.send(Outgoing::Adopt {
            take: take.to_string(),
        });
    }

    /// Asks the host to transcribe stored take `stored_id` with `with`;
    /// [`TakeUpdate::Transcription`]s carrying `req` follow.
    pub(crate) fn transcribe(&self, req: &str, stored_id: &str, with: TranscribeWith) {
        let _ = self.commands.send(Outgoing::Transcribe {
            req: req.to_string(),
            stored_id: stored_id.to_string(),
            with,
        });
    }

    /// Asks the host to run the transcription stored take `stored_id`
    /// waits for (an import stored with its intent).
    pub(crate) fn transcribe_due(&self, stored_id: &str) {
        let _ = self.commands.send(Outgoing::TranscribeDue {
            stored_id: stored_id.to_string(),
        });
    }

    /// The connection the link holds now (`None` while it has none), as
    /// the app's store reaches the host through it.
    pub(crate) fn connection(&self) -> Arc<Mutex<Option<Arc<HostClient>>>> {
        Arc::clone(&self.current)
    }

    /// Starts the recording service again after the link gave up on it.
    pub(crate) fn retry(&self) {
        self.relaunch.store(true, Ordering::SeqCst);
    }

}

impl Drop for HostLink {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
    }
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Sends the app's commands one by one on the current connection.
fn command_loop(
    outgoing: std::sync::mpsc::Receiver<Outgoing>,
    current: Arc<Mutex<Option<Arc<HostClient>>>>,
    tx: UnboundedSender<HostUpdate>,
) {
    while let Ok(next) = outgoing.recv() {
        let client = lock(&current).clone();
        match next {
            Outgoing::Command { take, command } => {
                let name = command_name(&command);
                let result = match client {
                    Some(client) => client
                        .send_reporting_busy(Some(&take), command)
                        .map(|_| ())
                        .map_err(|(err, busy)| (err.to_string(), busy)),
                    None => Err(("not connected to the recording service".to_string(), None)),
                };
                if let Err((reason, busy)) = result {
                    if tx
                        .send(HostUpdate::Refused {
                            take,
                            command: name,
                            reason,
                            busy,
                        })
                        .is_err()
                    {
                        return;
                    }
                }
            }
            // An adoption that does not go out leaves the take with
            // nobody here: the UI hears so and asks again.
            Outgoing::Adopt { take } => {
                let result = match client {
                    Some(client) => client.take_adopt(&take).map_err(|err| err.to_string()),
                    None => Err("not connected to the recording service".to_string()),
                };
                if let Err(reason) = result {
                    if tx
                        .send(HostUpdate::Refused {
                            take,
                            command: "take.adopt",
                            reason,
                            busy: None,
                        })
                        .is_err()
                    {
                        return;
                    }
                }
            }
            // Not sent, the take still waits in the store: the host finds
            // it at its next look.
            Outgoing::TranscribeDue { stored_id } => {
                if let Some(client) = client {
                    let _ = client.transcribe_due(&stored_id);
                }
            }
            Outgoing::Transcribe {
                req,
                stored_id,
                with,
            } => {
                let result = match client {
                    Some(client) => client
                        .transcribe(&req, &stored_id, with)
                        .map_err(|err| err.to_string()),
                    None => Err("not connected to the recording service".to_string()),
                };
                if let Err(reason) = result {
                    if tx
                        .send(HostUpdate::Refused {
                            take: req,
                            command: "transcribe",
                            reason,
                            busy: None,
                        })
                        .is_err()
                    {
                        return;
                    }
                }
            }
        }
    }
}

fn command_name(command: &Command) -> &'static str {
    match command {
        Command::CaptureStart { .. } => "capture.start",
        Command::CaptureStop { .. } => "capture.stop",
        Command::CaptureAbort => "capture.abort",
        _ => "command",
    }
}

#[allow(clippy::too_many_arguments)]
fn link_loop(
    endpoint: PathBuf,
    launch: Launch,
    current: Arc<Mutex<Option<Arc<HostClient>>>>,
    stop: Arc<AtomicBool>,
    relaunch: Arc<AtomicBool>,
    tx: UnboundedSender<HostUpdate>,
) {
    let mut backoff = RETRY_FIRST;
    // What the UI was last told, so a retry loop does not repeat it.
    let mut last_said: Option<(String, bool)> = None;
    // Failed tries in a row while the link may start the host: a launch
    // that keeps failing is given up on, not repeated forever.
    let mut failed_launches = 0;
    let launches = matches!(launch, Launch::SelfAsHost { .. });
    let mine = BuildStamp::current();
    while !stop.load(Ordering::SeqCst) {
        if relaunch.swap(false, Ordering::SeqCst) {
            failed_launches = 0;
            backoff = RETRY_FIRST;
            last_said = None;
        }
        let may_launch = failed_launches < LAUNCH_ATTEMPTS;
        let launch_now = if may_launch { launch.clone() } else { Launch::Never };
        let connected = connect_or_launch(&endpoint, &launch_now, &mine).and_then(|client| {
            let recovery = client.take_watch_as(&mine).map_err(|err| {
                LinkError::Failed(format!("the recording service did not answer: {err}"))
            })?;
            Ok((Arc::new(client), recovery))
        });
        let (client, recovery) = match connected {
            Ok(connected) => connected,
            // Another build serves (see `starling_runtime_host::version`):
            // nothing to launch — wait for the older one to finish, or
            // for the user to restart this older window.
            Err(LinkError::Version(reason)) => {
                if last_said.as_ref() != Some(&(reason.clone(), false)) {
                    eprintln!("Starling: {reason}");
                    if tx
                        .send(HostUpdate::Disconnected {
                            reason: reason.clone(),
                            gave_up: false,
                            host_gone: false,
                            plain: true,
                        })
                        .is_err()
                    {
                        return;
                    }
                    last_said = Some((reason, false));
                }
                sleep_unless(&stop, &relaunch, VERSION_RETRY);
                continue;
            }
            Err(LinkError::Failed(reason)) => {
                if launches && may_launch {
                    failed_launches += 1;
                }
                let gave_up = launches && failed_launches >= LAUNCH_ATTEMPTS;
                // Once given up, the link only looks for a host someone
                // else started; the launch failure stays the reason shown.
                let said = (reason, gave_up);
                if (may_launch || !launches) && last_said.as_ref() != Some(&said) {
                    eprintln!("Starling: {}", said.0);
                    if tx
                        .send(HostUpdate::Disconnected {
                            reason: said.0.clone(),
                            gave_up,
                            host_gone: false,
                            plain: false,
                        })
                        .is_err()
                    {
                        return;
                    }
                    last_said = Some(said);
                }
                sleep_unless(&stop, &relaunch, if gave_up { RETRY_MAX } else { backoff });
                backoff = (backoff * 2).min(RETRY_MAX);
                continue;
            }
        };
        backoff = RETRY_FIRST;
        failed_launches = 0;
        last_said = None;
        *lock(&current) = Some(Arc::clone(&client));
        if tx
            .send(HostUpdate::Connected {
                client: Arc::clone(&client),
                recovery,
            })
            .is_err()
        {
            return;
        }
        let reason = follow(&client, &stop, &tx);
        *lock(&current) = None;
        if stop.load(Ordering::SeqCst) {
            return;
        }
        // The connection, or the service itself? A service that went away
        // took any take it was recording with it. A killed host this app
        // started reads alive until its reaper collected it: give it a
        // moment.
        let host_gone = host_exited(client.info.pid);
        eprintln!("Starling: the recording service connection ended: {reason}");
        if tx
            .send(HostUpdate::Disconnected {
                reason,
                gave_up: false,
                host_gone,
                plain: false,
            })
            .is_err()
        {
            return;
        }
    }
}

/// Whether host process `pid` is gone, allowing up to a second for an
/// exiting one to be reaped.
fn host_exited(pid: u32) -> bool {
    let until = Instant::now() + Duration::from_secs(1);
    loop {
        if !starling_dictation::engine::registry::process_alive(pid) {
            return true;
        }
        if Instant::now() >= until {
            return false;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

/// Sleeps `duration`, or until the app closes or asks to start the
/// recording service again.
fn sleep_unless(stop: &AtomicBool, relaunch: &AtomicBool, duration: Duration) {
    let until = Instant::now() + duration;
    while Instant::now() < until
        && !stop.load(Ordering::SeqCst)
        && !relaunch.load(Ordering::SeqCst)
    {
        std::thread::sleep(Duration::from_millis(25));
    }
}

/// Routes the connection's take feed and events until it closes; returns
/// why it closed.
fn follow(client: &HostClient, stop: &AtomicBool, tx: &UnboundedSender<HostUpdate>) -> String {
    loop {
        if stop.load(Ordering::SeqCst) {
            return "the app is closing".to_string();
        }
        // Events are drained too: an undrained event stream would fail
        // the connection. Only `capture.error` matters to the UI.
        while let Ok(event) = client.try_recv_event() {
            if event.type_name() == "capture.error" && tx.send(HostUpdate::Event(event)).is_err() {
                return "the app is closing".to_string();
            }
        }
        match client.recv_take_timeout(Duration::from_millis(100)) {
            Ok(frame) => {
                if tx.send(HostUpdate::Take(route(frame))).is_err() {
                    return "the app is closing".to_string();
                }
            }
            Err(starling_runtime::channel::RecvError::Timeout) => {}
            Err(starling_runtime::channel::RecvError::Closed) => return client.close_reason(),
        }
        if client.is_closed() {
            return client.close_reason();
        }
    }
}

/// A take-feed frame as the UI reads it.
fn route(frame: TakeWire) -> TakeUpdate {
    match frame {
        TakeWire::Live {
            take,
            rate,
            status,
            owner,
            ended,
            kept,
            meter,
            ..
        } => TakeUpdate::Live {
            take,
            rate,
            status,
            owner,
            ended,
            kept,
            meter,
        },
        TakeWire::StartFailed {
            take,
            problem,
            message,
        } => TakeUpdate::StartFailed {
            take,
            problem,
            message,
        },
        TakeWire::Persisted {
            take,
            stored_id,
            interrupted,
            error,
            orphan,
        } => TakeUpdate::Persisted {
            take,
            stored_id,
            interrupted,
            error,
            orphan,
        },
        TakeWire::Notice(recovery) => TakeUpdate::Notice(recovery),
        TakeWire::Upkeep { report, retired } => TakeUpdate::Upkeep { report, retired },
        TakeWire::HistoryChanged => TakeUpdate::HistoryChanged,
        TakeWire::LiveText {
            take,
            partial,
            degraded,
        } => TakeUpdate::LiveText {
            take,
            partial,
            degraded,
        },
        TakeWire::Engine(status) => TakeUpdate::Engine(status),
        TakeWire::Transcription {
            stored_id,
            take,
            req,
            attempt,
            state,
            yours,
        } => TakeUpdate::Transcription {
            stored_id,
            take,
            req,
            attempt,
            state,
            yours,
        },
    }
}

/// Why the link has no connection to offer.
#[derive(Debug)]
enum LinkError {
    /// Reaching (or starting) the recording service failed.
    Failed(String),
    /// A recording service of another build serves, and this window
    /// cannot use it yet — or, a newer one, at all. The sentence says
    /// which, and what to do.
    Version(String),
}

/// Connects to the host at `endpoint`, starting one first when nothing
/// serves there (and `launch` allows it), and settles the version
/// handshake with it as an app of build `mine`.
fn connect_or_launch(
    endpoint: &Path,
    launch: &Launch,
    mine: &BuildStamp,
) -> Result<HostClient, LinkError> {
    match launch {
        Launch::SelfAsHost { log } => {
            connect_versioned(endpoint, mine, || launch_self(log), host_stopped)
        }
        Launch::Never => connect_versioned(
            endpoint,
            mine,
            || Err("the recording service is not running".to_string()),
            host_stopped,
        ),
    }
}

/// How long a window waits between looks while a host of another build
/// serves.
const VERSION_RETRY: Duration = Duration::from_secs(2);

/// How long an older host that agreed to step aside may take to stop.
const RETIRE_TIMEOUT: Duration = Duration::from_secs(30);

/// Whether the host `info` describes has exited (allowing it
/// [`RETIRE_TIMEOUT`] to finish its shutdown — its engine stops on the
/// way).
fn host_stopped(info: &starling_runtime_host::client::HostInfo) -> bool {
    let until = Instant::now() + RETIRE_TIMEOUT;
    while starling_dictation::engine::registry::process_alive(info.pid) {
        if Instant::now() >= until {
            return false;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    true
}

/// [`connect_or_start`], then the version handshake
/// (`starling_runtime_host::version`): a host of this build is used; an
/// older one is asked to step aside and, once `stopped` says it has gone,
/// this window starts its own; a newer one is not used — this window is
/// the one that is out of date.
fn connect_versioned(
    endpoint: &Path,
    mine: &BuildStamp,
    mut start: impl FnMut() -> Result<(), String>,
    stopped: impl Fn(&starling_runtime_host::client::HostInfo) -> bool,
) -> Result<HostClient, LinkError> {
    // Twice at most: the older host, then the one this window starts.
    for _ in 0..2 {
        let client = connect_or_start(endpoint, &mut start).map_err(LinkError::Failed)?;
        let Some(theirs) = client.info.build.clone() else {
            // From before the handshake: it cannot be asked to stop.
            return Err(LinkError::Version(
                "An older version of Starling's recording service is still running. It stops \
                 by itself about a minute after its last window closed; this window connects \
                 then."
                    .to_string(),
            ));
        };
        if theirs == *mine {
            return Ok(client);
        }
        if mine.older_than(&theirs) {
            return Err(LinkError::Version(
                starling_runtime_host::version::older_app_refusal(),
            ));
        }
        match client.retire(mine) {
            Ok(RetireAnswer::Retiring) => {
                let info = client.info.clone();
                drop(client);
                eprintln!(
                    "Starling: the recording service of an older build ({}) is stepping aside",
                    theirs.id
                );
                if !stopped(&info) {
                    return Err(LinkError::Version(
                        "An older version of Starling's recording service is still stopping; \
                         this window connects once it has."
                            .to_string(),
                    ));
                }
            }
            Ok(RetireAnswer::Busy { reason }) => {
                return Err(LinkError::Version(format!(
                    "An older version of Starling's recording service is still running \
                     ({reason}). This window connects once it is done; close the older \
                     Starling window if one is open."
                )));
            }
            Ok(RetireAnswer::Refused { reason }) => return Err(LinkError::Version(reason)),
            Err(err) => {
                return Err(LinkError::Failed(format!(
                    "the recording service did not answer ({err})"
                )))
            }
        }
    }
    Err(LinkError::Failed(
        "the recording service was replaced while this window connected".to_string(),
    ))
}

/// Connects to the host at `endpoint`, running `start` first when nothing
/// serves there. A started host is given [`HOST_START_TIMEOUT`] from its
/// start to serve, however it reported: one that found another host
/// starting ("already-running") waits out that host's startup recovery
/// rather than giving up early, and a slow start is never started again
/// over itself.
fn connect_or_start(
    endpoint: &Path,
    start: impl FnOnce() -> Result<(), String>,
) -> Result<HostClient, String> {
    if let Ok(client) = HostClient::connect(endpoint) {
        return Ok(client);
    }
    let deadline = Instant::now() + HOST_START_TIMEOUT;
    start()?;
    loop {
        match HostClient::connect(endpoint) {
            Ok(client) => return Ok(client),
            Err(err) if Instant::now() >= deadline => {
                return Err(format!("the recording service started but cannot be reached ({err})"));
            }
            Err(_) => std::thread::sleep(Duration::from_millis(50)),
        }
    }
}

/// The host's log beside the store, kept from growing without bound.
pub(crate) fn host_log_path() -> Option<PathBuf> {
    starling_runtime_host::default_data_root()
        .ok()
        .map(|root| root.join("runtime-host.log"))
}

/// Starts this executable as the host, detached from the app (its own
/// process group: a terminal's Ctrl+C to the app does not reach it, and
/// it outlives an app that is killed), and waits for its status line.
fn launch_self(log: &Path) -> Result<(), String> {
    let exe = std::env::current_exe()
        .map_err(|err| format!("cannot find the Starling executable to start the recording service: {err}"))?;
    let mut command = std::process::Command::new(exe);
    command
        .arg("--runtime-host")
        // The host transcribes the takes it stores, on the engine the
        // settings choose: the bundled engine is shared with this app
        // through the engine registry (one sidecar, whoever started it).
        .arg("--exit-when-idle")
        .arg(HOST_IDLE_EXIT.as_secs().to_string());
    start_host(command, log)
}

/// Runs `command` as the host, logging to `log`, and waits for its status
/// line. Its stdout is read for as long as it runs: a pipe nobody reads
/// would block (or break) a host that writes more than that line.
fn start_host(mut command: std::process::Command, log: &Path) -> Result<(), String> {
    if let Some(dir) = log.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    // Keep one previous log; never let it grow without bound.
    if std::fs::metadata(log).is_ok_and(|meta| meta.len() > 4 * 1024 * 1024) {
        let _ = std::fs::rename(log, log.with_extension("log.old"));
    }
    let stderr = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log)
        .map(std::process::Stdio::from)
        .unwrap_or_else(|_| std::process::Stdio::null());
    command
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(stderr);
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        command.creation_flags(CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW);
    }
    let mut child = command
        .spawn()
        .map_err(|err| format!("the recording service could not start: {err}"))?;
    let stdout = child.stdout.take();
    let (line_tx, line_rx) = std::sync::mpsc::channel::<String>();
    std::thread::spawn(move || {
        let mut line = String::new();
        let mut stdout = stdout.map(std::io::BufReader::new);
        if let Some(stdout) = stdout.as_mut() {
            let _ = stdout.read_line(&mut line);
        }
        let _ = line_tx.send(line);
        if let Some(stdout) = stdout.as_mut() {
            let _ = std::io::copy(stdout, &mut std::io::sink());
        }
    });
    // Reap the host whenever it exits, so it never lingers as a zombie of
    // this app.
    std::thread::spawn(move || {
        let _ = child.wait();
    });
    let line = line_rx
        .recv_timeout(HOST_START_TIMEOUT)
        .map_err(|_| "the recording service did not report within 30 s".to_string())?;
    let status = serde_json::from_str::<serde_json::Value>(line.trim())
        .ok()
        .and_then(|value| value.get("status").and_then(|s| s.as_str()).map(str::to_string));
    match status.as_deref() {
        Some("owner") | Some("already-running") => Ok(()),
        _ => Err(format!(
            "the recording service could not start; see {}",
            log.display()
        )),
    }
}

// --------------------------------------------------------------------- //
// The take as the app sees it
// --------------------------------------------------------------------- //

/// The take the recording service records for this window, as the UI
/// reads it: what the app's recorder handle used to answer, from the
/// host's status ticks.
pub(crate) struct LiveCapture {
    /// The take's id on the wire (`capture.start`'s corr).
    pub take: String,
    pub status: Option<LiveTakeStatus>,
    /// Whether the host has confirmed the take is recording.
    pub confirmed: bool,
    /// The device rate, from the host's first tick.
    pub rate: u32,
    /// The take's newest samples, for the level meter.
    pub meter: Vec<f32>,
    started_at: Instant,
}

impl LiveCapture {
    pub(crate) fn new(take: String) -> LiveCapture {
        LiveCapture {
            take,
            status: None,
            confirmed: false,
            rate: 0,
            meter: Vec::new(),
            started_at: Instant::now(),
        }
    }

    pub(crate) fn sample_rate(&self) -> u32 {
        self.rate
    }

    pub(crate) fn captured_sample_count(&self) -> u64 {
        self.status
            .as_ref()
            .map(|status| status.captured)
            .unwrap_or(0)
    }

    /// The newest `n` samples, for the level meter.
    pub(crate) fn latest_window(&self, n: usize) -> Vec<f32> {
        let start = self.meter.len().saturating_sub(n);
        self.meter[start..].to_vec()
    }

    pub(crate) fn source_clip_ratio(&self) -> f64 {
        self.status.as_ref().map(|status| status.clip_ratio).unwrap_or(0.0)
    }

    pub(crate) fn capture_fault(&self) -> Option<RecorderFault> {
        self.status.as_ref().and_then(|status| status.fault.clone())
    }

    pub(crate) fn capture_error(&self) -> Option<String> {
        self.capture_fault().map(|fault| fault.to_string())
    }

    pub(crate) fn input_route(&self) -> Option<&InputRoute> {
        self.status.as_ref().and_then(|status| status.route.as_ref())
    }

    pub(crate) fn input_stalled_for(&self) -> Option<Duration> {
        self.status
            .as_ref()
            .and_then(|status| status.stalled_ms)
            .map(Duration::from_millis)
    }

    /// Time since the take started, from the host's clock once it
    /// reported.
    pub(crate) fn elapsed(&self) -> Duration {
        match &self.status {
            Some(status) => Duration::from_millis(status.elapsed_ms),
            None => self.started_at.elapsed(),
        }
    }

    pub(crate) fn disk_reading(&self) -> Option<DiskReading> {
        self.status.as_ref().and_then(|status| status.disk)
    }

    pub(crate) fn disk_probe_failing(&self) -> bool {
        self.status
            .as_ref()
            .is_some_and(|status| status.disk_probe_failing)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(tag: &str) -> PathBuf {
        // Short: the host's socket path lives under it.
        let root = std::env::temp_dir().join(format!("sl-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).expect("scratch");
        root
    }

    #[cfg(unix)]
    fn shell(script: &str) -> std::process::Command {
        let mut command = std::process::Command::new("sh");
        command.args(["-c", script]);
        command
    }

    #[cfg(unix)]
    #[test]
    fn a_host_that_starts_slowly_is_waited_for_and_started_once() {
        // The launched process finds another host starting (it holds the
        // lease and is still recovering) and says so at once; that host
        // serves only seconds later.
        let root = scratch("slow");
        let config = starling_runtime_host::HostConfig::new(&root, root.join("endpoints"));
        let endpoint = config.socket_path();
        let serving = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_secs(6));
            starling_runtime_host::serve(config).expect("host serves")
        });
        let launches = std::sync::atomic::AtomicUsize::new(0);
        let log = root.join("runtime-host.log");
        let started = Instant::now();
        let client = connect_or_start(&endpoint, || {
            launches.fetch_add(1, Ordering::SeqCst);
            start_host(shell(r#"echo '{"status":"already-running"}'"#), &log)
        });
        assert!(client.is_ok(), "connected: {:?}", client.err());
        assert!(started.elapsed() >= Duration::from_secs(6));
        assert_eq!(launches.load(Ordering::SeqCst), 1, "started once");
        drop(client);
        serving.join().expect("host thread").shutdown();
        let _ = std::fs::remove_dir_all(&root);
    }

    /// #220's version handshake, app side, over the real socket: a
    /// window of a newer build finds an older host — told to wait while a
    /// window of that version is open, then the idle host steps aside and
    /// the window starts its own; an older window reaching that newer host
    /// is told plainly to restart.
    #[test]
    fn a_newer_window_replaces_an_older_idle_host() {
        use starling_runtime_host::{serve, HostConfig};
        let root = scratch("ver");
        let old = BuildStamp {
            id: "an-older-build".to_string(),
            built: 1,
        };
        let mine = BuildStamp::current();
        let older = HostConfig::new(&root, root.join("endpoints")).with_build(old.clone());
        let endpoint = older.socket_path();
        let older_host = serve(older).expect("the older host serves");

        // A window of the older version is open: the host keeps serving it.
        let old_window = HostClient::connect(&endpoint).expect("connects");
        old_window.take_watch_as(&old).expect("the older window is served");
        match connect_versioned(&endpoint, &mine, || Err("not expected".to_string()), |_| true) {
            Err(LinkError::Version(text)) => {
                assert!(text.contains("connects once it is done"), "{text}")
            }
            other => panic!("the busy older host must not be replaced: {:?}", other.err()),
        }
        drop(old_window);

        // Whoever runs the older host stops it once it agreed to step aside.
        let (stopped_tx, stopped_rx) = std::sync::mpsc::channel();
        let runner = std::thread::spawn(move || {
            let mut host = older_host;
            let deadline = Instant::now() + Duration::from_secs(20);
            while !host.retire_requested() && Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(20));
            }
            host.shutdown();
            let _ = stopped_tx.send(());
        });
        let newer: Arc<Mutex<Option<starling_runtime_host::HostHandle>>> = Arc::default();
        let starts = std::sync::atomic::AtomicUsize::new(0);
        let deadline = Instant::now() + Duration::from_secs(20);
        let client = loop {
            let result = connect_versioned(
                &endpoint,
                &mine,
                || {
                    starts.fetch_add(1, Ordering::SeqCst);
                    let host = serve(HostConfig::new(&root, root.join("endpoints")))
                        .map_err(|err| err.to_string())?;
                    *lock(&newer) = Some(host);
                    Ok(())
                },
                |_| stopped_rx.recv_timeout(Duration::from_secs(20)).is_ok(),
            );
            match result {
                Ok(client) => break client,
                // The older window's connection may still be closing.
                Err(LinkError::Version(text)) if text.contains("connects once it is done") => {
                    assert!(Instant::now() < deadline, "the idle older host never stepped aside");
                    std::thread::sleep(Duration::from_millis(50));
                }
                Err(other) => panic!("{other:?}"),
            }
        };
        runner.join().expect("the older host's runner");
        assert_eq!(client.info.build.as_ref(), Some(&mine), "this window's own build serves");
        assert_eq!(starts.load(Ordering::SeqCst), 1, "started once");
        client.take_watch_as(&mine).expect("watching");

        // An older window reaching the newer host is refused, plainly.
        match connect_versioned(&endpoint, &old, || Err("not expected".to_string()), |_| true) {
            Err(LinkError::Version(text)) => {
                assert!(text.contains("older version"), "{text}");
                assert!(text.contains("start Starling again"), "{text}");
            }
            other => panic!("an older window must be refused: {:?}", other.err()),
        }
        drop(client);
        lock(&newer).take().expect("the newer host").shutdown();
        let _ = std::fs::remove_dir_all(&root);
    }

    #[cfg(unix)]
    #[test]
    fn a_started_host_can_keep_writing_to_its_stdout() {
        let root = scratch("out");
        let done = root.join("done");
        let script = format!(
            r#"echo '{{"status":"owner"}}'; head -c 1000000 /dev/zero && touch '{}'"#,
            done.display()
        );
        start_host(shell(&script), &root.join("runtime-host.log")).expect("reported");
        let deadline = Instant::now() + Duration::from_secs(20);
        while !done.exists() {
            assert!(Instant::now() < deadline, "the host's writes after its status line blocked or broke");
            std::thread::sleep(Duration::from_millis(20));
        }
        let _ = std::fs::remove_dir_all(&root);
    }
}
