//! The app's link to the runtime host (#220).
//!
//! Takes are recorded, journaled and stored by the per-user runtime host
//! (`starling-runtime-host`, or this binary run as `--runtime-host`), not
//! by the app: the host holds the store lease and the recorder's journal
//! tree, so killing the app never costs a take and a second app window
//! is just another client. The app keeps what needs its window — the
//! activation machine, the live view, staging, delivery — and the
//! transcription of its takes for now.
//!
//! [`HostLink`] owns the connection on its own thread: it connects to the
//! host serving the default data root, starts one when nothing serves
//! (this executable, detached, idle-exiting), follows the take feed and
//! reconnects — starting the host again if it is gone — whenever the
//! connection drops. The UI hears about it through [`HostUpdate`]s; take
//! audio skips the UI and lands in the take's [`TakeFeed`], which the
//! live stream drains on its own worker.

use std::collections::HashMap;
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
use starling_runtime_host::frame::{HostRecovery, TakeOwner};
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};

use starling_runtime_host::live::pump::AudioTap;

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
    /// it until [`HostLink::retry`].
    Disconnected { reason: String, gave_up: bool },
    /// A take-feed frame, its audio already in the take's feed.
    Take(TakeUpdate),
    /// A runtime event the UI acts on (`capture.error`).
    Event(EventWire),
    /// A command for `take` was not carried out (refused by the runtime,
    /// or no connection to send it on).
    Refused {
        take: String,
        command: &'static str,
        reason: String,
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
    feeds: Arc<Mutex<HashMap<String, Arc<TakeFeed>>>>,
    commands: std::sync::mpsc::Sender<Outgoing>,
    stop: Arc<AtomicBool>,
    relaunch: Arc<AtomicBool>,
}

enum Outgoing {
    Command { take: String, command: Command },
    Tap { take: String, from: u64 },
    Handled { stored_id: String, handed_back: bool },
}

impl HostLink {
    /// Starts the link thread for `endpoint`; updates arrive on the
    /// returned receiver.
    pub(crate) fn start(
        endpoint: PathBuf,
        launch: Launch,
    ) -> (HostLink, UnboundedReceiver<HostUpdate>) {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let feeds: Arc<Mutex<HashMap<String, Arc<TakeFeed>>>> = Arc::default();
        let stop = Arc::new(AtomicBool::new(false));
        let relaunch = Arc::new(AtomicBool::new(false));
        let current: Arc<Mutex<Option<Arc<HostClient>>>> = Arc::default();
        let (commands, outgoing) = std::sync::mpsc::channel();
        let spawned = std::thread::Builder::new()
            .name("starling-host-link".to_string())
            .spawn({
                let feeds = Arc::clone(&feeds);
                let stop = Arc::clone(&stop);
                let relaunch = Arc::clone(&relaunch);
                let current = Arc::clone(&current);
                let tx = tx.clone();
                move || link_loop(endpoint, launch, feeds, current, stop, relaunch, tx)
            })
            .and_then(|_| {
                std::thread::Builder::new()
                    .name("starling-host-commands".to_string())
                    .spawn({
                        let tx = tx.clone();
                        move || command_loop(outgoing, current, tx)
                    })
            });
        if let Err(err) = spawned {
            let _ = tx.send(HostUpdate::Disconnected {
                reason: format!("the connection thread could not start: {err}"),
                gave_up: true,
            });
        }
        (
            HostLink {
                feeds,
                commands,
                stop,
                relaunch,
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

    /// Asks for `take`'s audio from sample `from` on, into its feed.
    pub(crate) fn tap(&self, take: &str, from: u64) {
        let _ = self.commands.send(Outgoing::Tap {
            take: take.to_string(),
            from,
        });
    }

    /// Tells the host the stored take `stored_id` it handed this app is
    /// handled (transcribed, or its failure recorded): until then the
    /// host keeps it for whichever app comes next.
    pub(crate) fn handled(&self, stored_id: &str) {
        let _ = self.commands.send(Outgoing::Handled {
            stored_id: stored_id.to_string(),
            handed_back: false,
        });
    }

    /// Hands the stored take `stored_id` back to the host: this app
    /// cannot transcribe it, so the host gives it to another.
    pub(crate) fn handed_back(&self, stored_id: &str) {
        let _ = self.commands.send(Outgoing::Handled {
            stored_id: stored_id.to_string(),
            handed_back: true,
        });
    }

    /// Starts the recording service again after the link gave up on it.
    pub(crate) fn retry(&self) {
        self.relaunch.store(true, Ordering::SeqCst);
    }

    /// The feed `take`'s audio lands in from now on (one per take).
    pub(crate) fn feed(&self, take: &str) -> Arc<TakeFeed> {
        Arc::clone(
            lock(&self.feeds)
                .entry(take.to_string())
                .or_insert_with(|| Arc::new(TakeFeed::default())),
        )
    }

    /// Stops routing audio to `take`'s feed.
    pub(crate) fn forget(&self, take: &str) {
        lock(&self.feeds).remove(take);
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
                    Some(client) => client.send(Some(&take), command).map(|_| ()).map_err(|err| err.to_string()),
                    None => Err("not connected to the recording service".to_string()),
                };
                if let Err(reason) = result {
                    if tx
                        .send(HostUpdate::Refused {
                            take,
                            command: name,
                            reason,
                        })
                        .is_err()
                    {
                        return;
                    }
                }
            }
            // A tap that does not go out costs the take's live audio
            // here (live text, levels), never the take: the host records
            // and stores it either way, a take this window does not tap
            // still gets its end and stored row, and a finished take is
            // transcribed from the stored audio when the feed is not
            // whole. The UI hears about it so it can tap again.
            Outgoing::Tap { take, from } => {
                let result = match client {
                    Some(client) => client.take_tap(&take, from).map_err(|err| err.to_string()),
                    None => Err("not connected to the recording service".to_string()),
                };
                if let Err(reason) = result {
                    if tx
                        .send(HostUpdate::Refused {
                            take,
                            command: "take.tap",
                            reason,
                        })
                        .is_err()
                    {
                        return;
                    }
                }
            }
            // An ack that does not go out leaves the take with the host,
            // which hands it to the next connection: at worst it is
            // transcribed once more, never lost.
            Outgoing::Handled {
                stored_id,
                handed_back,
            } => {
                if let Some(client) = client {
                    let _ = if handed_back {
                        client.take_handed_back(&stored_id)
                    } else {
                        client.take_handled(&stored_id)
                    };
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
    feeds: Arc<Mutex<HashMap<String, Arc<TakeFeed>>>>,
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
    while !stop.load(Ordering::SeqCst) {
        if relaunch.swap(false, Ordering::SeqCst) {
            failed_launches = 0;
            backoff = RETRY_FIRST;
            last_said = None;
        }
        let may_launch = failed_launches < LAUNCH_ATTEMPTS;
        let launch_now = if may_launch { launch.clone() } else { Launch::Never };
        let connected = connect_or_launch(&endpoint, &launch_now).and_then(|client| {
            let recovery = client
                .take_watch()
                .map_err(|err| format!("the recording service did not answer: {err}"))?;
            Ok((Arc::new(client), recovery))
        });
        let (client, recovery) = match connected {
            Ok(connected) => connected,
            Err(reason) => {
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
        let reason = follow(&client, &feeds, &stop, &tx);
        *lock(&current) = None;
        // Feeds belong to the connection that tapped them: a reconnect
        // re-adopts the take and taps it again from its start.
        lock(&feeds).clear();
        if stop.load(Ordering::SeqCst) {
            return;
        }
        eprintln!("Starling: the recording service connection ended: {reason}");
        if tx
            .send(HostUpdate::Disconnected {
                reason,
                gave_up: false,
            })
            .is_err()
        {
            return;
        }
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
fn follow(
    client: &HostClient,
    feeds: &Mutex<HashMap<String, Arc<TakeFeed>>>,
    stop: &AtomicBool,
    tx: &UnboundedSender<HostUpdate>,
) -> String {
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
        match client.recv_take_timeout(Duration::from_millis(20)) {
            Ok(frame) => {
                let Some(update) = route(frame, feeds, client) else {
                    continue;
                };
                if tx.send(HostUpdate::Take(update)).is_err() {
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

/// Puts a frame's audio into its take's feed and hands back the rest.
/// A feed that finds a gap in its audio asks the host for the take again
/// from where it breaks off (the host reads it from the recorder, or the
/// finished take's record).
fn route(
    frame: TakeWire,
    feeds: &Mutex<HashMap<String, Arc<TakeFeed>>>,
    client: &HostClient,
) -> Option<TakeUpdate> {
    Some(match frame {
        TakeWire::Live {
            take,
            rate,
            status,
            audio,
            owner,
            ended,
            kept,
            ..
        } => {
            let feed = lock(feeds).get(&take).cloned();
            if let Some(feed) = feed {
                if let Some(from) = feed.receive(rate, status.as_ref(), audio, ended) {
                    eprintln!(
                        "Starling: take {take}'s audio has a gap at sample {from}; asking the \
                         recording service for it again"
                    );
                    if let Err(err) = client.take_tap(&take, from) {
                        eprintln!("Starling: could not ask for take {take}'s audio again: {err}");
                    }
                }
            }
            TakeUpdate::Live {
                take,
                rate,
                status,
                owner,
                ended,
                kept,
            }
        }
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
        TakeWire::LiveText { .. } | TakeWire::Transcription { .. } => return None,
    })
}

/// Connects to the host at `endpoint`, starting one first when nothing
/// serves there (and `launch` allows it).
fn connect_or_launch(endpoint: &Path, launch: &Launch) -> Result<HostClient, String> {
    match launch {
        Launch::SelfAsHost { log } => connect_or_start(endpoint, || launch_self(log)),
        Launch::Never => HostClient::connect(endpoint)
            .map_err(|err| format!("the recording service is not running ({err})")),
    }
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
        // The app transcribes its own takes for now (#220 later work
        // moves jobs into the host): the host runs no engine of its own,
        // so the app's engine stays the only one.
        .args(["--engine", "none"])
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

/// A take's audio as it arrives from the host, for the live stream (an
/// [`AudioTap`]) and the level meter. The host sends the take in order,
/// each sample once; the end says how many there were.
#[derive(Default)]
pub(crate) struct TakeFeed {
    state: Mutex<FeedState>,
}

#[derive(Default)]
struct FeedState {
    samples: Vec<f32>,
    /// Where a gap was found and the take asked for again from: asked
    /// once per position, not once per chunk past it.
    resync_from: Option<usize>,
    /// What [`AudioTap::drain`] handed out.
    drained: usize,
    /// The journal-durable count from the latest tick.
    acknowledged: u64,
    rate: u32,
    /// The take's final sample count, once its last sample arrived.
    ended: Option<u64>,
}

impl TakeFeed {
    /// Takes one frame's audio; `Some(from)` when it found a gap the take
    /// should be tapped again from. Audio is only ever spliced where it
    /// belongs: a repeat (a re-tap replaying what is here) adds only what
    /// is new, and audio past a gap waits for the gap to be filled. A
    /// feed with a gap is never [`TakeFeed::complete`], so a finished
    /// take is then transcribed from the stored audio, not from this.
    fn receive(
        &self,
        rate: u32,
        status: Option<&LiveTakeStatus>,
        audio: Option<(u64, Vec<f32>)>,
        ended: Option<u64>,
    ) -> Option<u64> {
        let mut state = lock(&self.state);
        if rate > 0 {
            state.rate = rate;
        }
        if let Some(status) = status {
            state.acknowledged = state.acknowledged.max(status.acknowledged);
        }
        let mut resync = None;
        if let Some((start, chunk)) = audio {
            let start = start as usize;
            let have = state.samples.len();
            if start <= have {
                let new = chunk.get(have - start..).unwrap_or_default();
                state.samples.extend_from_slice(new);
                state.resync_from = None;
            } else if state.resync_from != Some(have) {
                state.resync_from = Some(have);
                resync = Some(have as u64);
            }
        }
        if let Some(total) = ended {
            state.ended = Some(total);
        }
        resync
    }

    /// Whether the feed is short of the take: it ended with fewer samples
    /// than the take has.
    pub(crate) fn short_of(&self) -> Option<(usize, u64)> {
        let state = lock(&self.state);
        state
            .ended
            .filter(|total| (state.samples.len() as u64) < *total)
            .map(|total| (state.samples.len(), total))
    }

    /// Whether the take's last sample is here.
    pub(crate) fn complete(&self) -> bool {
        let state = lock(&self.state);
        state
            .ended
            .is_some_and(|total| state.samples.len() as u64 >= total)
    }

    pub(crate) fn sample_rate(&self) -> u32 {
        lock(&self.state).rate
    }

    pub(crate) fn len(&self) -> usize {
        lock(&self.state).samples.len()
    }

    /// The samples from `from` on.
    pub(crate) fn samples_from(&self, from: usize) -> Vec<f32> {
        let state = lock(&self.state);
        state.samples.get(from..).map(<[f32]>::to_vec).unwrap_or_default()
    }

    /// The newest `n` samples, for the level meter.
    pub(crate) fn latest_window(&self, n: usize) -> Vec<f32> {
        let state = lock(&self.state);
        let start = state.samples.len().saturating_sub(n);
        state.samples[start..].to_vec()
    }
}

impl AudioTap for TakeFeed {
    fn drain(&self) -> Vec<f32> {
        let mut state = lock(&self.state);
        let drained = state.samples[state.drained..].to_vec();
        state.drained = state.samples.len();
        drained
    }

    fn acknowledged(&self) -> u64 {
        let state = lock(&self.state);
        if state.ended.is_some() {
            // A finished take's audio is the stored take.
            state.samples.len() as u64
        } else {
            state.acknowledged
        }
    }
}

/// The take the host records for this app, as the UI reads it: what the
/// app's recorder handle used to answer, from the host's status ticks
/// and the take's feed.
pub(crate) struct LiveCapture {
    /// The take's id on the wire (`capture.start`'s corr).
    pub take: String,
    pub feed: Arc<TakeFeed>,
    pub status: Option<LiveTakeStatus>,
    /// Whether the host has confirmed the take is recording.
    pub confirmed: bool,
    /// Whether this window asked for the take's audio.
    pub tapped: bool,
    started_at: Instant,
}

impl LiveCapture {
    pub(crate) fn new(take: String, feed: Arc<TakeFeed>) -> LiveCapture {
        LiveCapture {
            take,
            feed,
            status: None,
            confirmed: false,
            tapped: false,
            started_at: Instant::now(),
        }
    }

    pub(crate) fn sample_rate(&self) -> u32 {
        self.feed.sample_rate()
    }

    pub(crate) fn captured_sample_count(&self) -> u64 {
        self.status
            .as_ref()
            .map(|status| status.captured)
            .unwrap_or(0)
            .max(self.feed.len() as u64)
    }

    pub(crate) fn latest_window(&self, n: usize) -> Vec<f32> {
        self.feed.latest_window(n)
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

    #[test]
    fn a_gap_is_asked_for_once_however_much_audio_arrives_past_it() {
        let feed = Arc::new(TakeFeed::default());
        feed.receive(16_000, None, Some((0, vec![0.1; 4])), None);
        assert_eq!(feed.receive(16_000, None, Some((8, vec![0.2; 4])), None), Some(4));
        for start in (12..400).step_by(4) {
            assert_eq!(
                feed.receive(16_000, None, Some((start, vec![0.2; 4])), None),
                None,
                "asked again at {start}"
            );
        }
        // The replay from the gap fills it; a later gap is asked for once.
        assert_eq!(feed.receive(16_000, None, Some((4, vec![0.3; 8])), None), None);
        assert_eq!(feed.samples_from(0).len(), 12);
        assert_eq!(feed.receive(16_000, None, Some((20, vec![0.4; 4])), None), Some(12));
        assert_eq!(feed.receive(16_000, None, Some((24, vec![0.4; 4])), None), None);
    }

    #[test]
    fn a_feed_takes_audio_in_order_once_and_knows_when_it_is_complete() {
        let feed = Arc::new(TakeFeed::default());
        assert_eq!(feed.receive(16_000, None, Some((0, vec![0.1, 0.2])), None), None);
        // A repeat adds nothing; audio past a gap is not spliced in, and
        // the gap is asked for again once.
        assert_eq!(feed.receive(16_000, None, Some((0, vec![0.1, 0.2])), None), None);
        assert_eq!(feed.receive(16_000, None, Some((5, vec![0.9])), None), Some(2));
        assert_eq!(feed.receive(16_000, None, Some((6, vec![0.9])), None), None);
        assert_eq!(feed.samples_from(0), vec![0.1, 0.2]);
        assert_eq!(feed.drain(), vec![0.1, 0.2]);
        feed.receive(16_000, None, Some((2, vec![0.3])), Some(3));
        assert!(feed.complete());
        assert_eq!(feed.drain(), vec![0.3], "drain hands out each sample once");
        assert_eq!(feed.acknowledged(), 3, "a finished take is all acknowledged");
        assert_eq!(feed.latest_window(2), vec![0.2, 0.3]);
    }

    #[test]
    fn a_replay_that_overlaps_the_feed_adds_only_the_new_audio() {
        let feed = Arc::new(TakeFeed::default());
        feed.receive(16_000, None, Some((0, vec![0.1, 0.2])), None);
        feed.receive(16_000, None, Some((1, vec![0.2, 0.3, 0.4])), None);
        assert_eq!(feed.samples_from(0), vec![0.1, 0.2, 0.3, 0.4]);
    }

    #[test]
    fn a_feed_with_a_gap_at_its_end_is_short_of_the_take() {
        let feed = Arc::new(TakeFeed::default());
        feed.receive(16_000, None, Some((0, vec![0.1])), None);
        feed.receive(16_000, None, Some((3, vec![0.4])), Some(4));
        assert!(!feed.complete());
        assert_eq!(feed.short_of(), Some((1, 4)));
    }

    #[test]
    fn a_feed_is_incomplete_until_its_last_sample_arrives() {
        let feed = Arc::new(TakeFeed::default());
        feed.receive(16_000, None, Some((0, vec![0.1])), Some(2));
        assert!(!feed.complete());
        feed.receive(16_000, None, Some((1, vec![0.2])), None);
        assert!(feed.complete());
    }
}
