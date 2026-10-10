//! Lowers or mutes system playback while a take records, and restores it
//! the moment recording stops (not when transcription finishes).
//!
//! The app's recorder path brackets each take with
//! [`PlaybackHandle::begin`] and [`PlaybackHandle::end`]. Both only send a
//! message: every backend call (a `pactl` subprocess on Linux) runs on the
//! service's own thread.
//!
//! Starling never overwrites a volume or mute state it did not write, and
//! restores what it did write:
//! * The first `begin` of an attenuation group (takes that overlap share
//!   one) snapshots the default output and keys the group to that device;
//!   a device that has gone away restores nothing (reported as
//!   [`NoticeKind::OutputRemoved`]).
//! * The volume (the whole raw channel vector, as pactl writes it) and the
//!   mute flag are owned separately. Before every later write and at
//!   restore the device is re-read; a field holding anything other than
//!   the snapshot or our last write was changed by the user and is never
//!   written again in this group. (A user change back to our exact value
//!   is indistinguishable. pactl has no compare-and-set, so a change
//!   landing between our read and our write is still overwritten.)
//! * Lower caps each channel at the level, never raising one; Mute mutes.
//!   Overlapping requests combine to the quieter outcome, and the last
//!   owner's `end` writes the snapshot back to every field still holding
//!   our write. A volume the user changed while some channel still holds
//!   our lowering is reported instead.
//! * A write is recorded before it is sent: one that reported an error may
//!   still have landed and is restored like any other.
//! * A failing restore is retried, then reported as
//!   [`NoticeKind::RestoreFailed`].
//! * Shutting the service down restores synchronously. Nothing is
//!   persisted: after a crash playback stays as the take left it, and the
//!   next launch never reapplies a stale snapshot.

use std::collections::VecDeque;
use std::process::{Command, Stdio};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crate::settings::{PlaybackMode, PlaybackSettings};

/// One output device's state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutputSnapshot {
    /// Backend device id (the sink name on Linux).
    pub device: String,
    /// Volume per channel in raw units ([`VOLUME_NORM`] is 100%); empty
    /// when it cannot be read.
    pub volumes: Vec<u32>,
    pub muted: bool,
}

/// 100% volume in raw units (PulseAudio's `PA_VOLUME_NORM`).
pub const VOLUME_NORM: u32 = 65536;

/// A change to one output; `None` fields are left alone.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OutputChange {
    /// Volume per channel, in the order [`OutputSnapshot::volumes`] has.
    pub volumes: Option<Vec<u32>>,
    pub set_muted: Option<bool>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlaybackError {
    /// The device no longer exists.
    NoSuchDevice(String),
    /// This system cannot control playback (no `pactl`, no audio server,
    /// no backend for the platform).
    Unsupported(String),
    Backend(String),
}

impl std::fmt::Display for PlaybackError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PlaybackError::NoSuchDevice(detail) => write!(f, "output device is gone: {detail}"),
            PlaybackError::Unsupported(detail) => {
                write!(f, "playback control unavailable: {detail}")
            }
            PlaybackError::Backend(detail) => write!(f, "{detail}"),
        }
    }
}

/// What the service needs from an audio system. Only ever called from the
/// service thread, so calls may block.
pub trait PlaybackBackend: Send + Sync + 'static {
    /// The current default output, discovered fresh on every call.
    fn active_output(&self) -> Result<OutputSnapshot, PlaybackError>;
    fn output(&self, id: &str) -> Result<OutputSnapshot, PlaybackError>;
    /// Applies every field of `change`; a failing field does not skip the
    /// others.
    fn apply(&self, id: &str, change: &OutputChange) -> Result<(), PlaybackError>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NoticeKind {
    /// Playback could not be lowered or muted for a take. The recording
    /// itself is unaffected.
    AdjustFailed,
    /// Restoring failed after retries; the user has to fix the volume.
    RestoreFailed,
    /// The attenuated output disappeared, so there was nothing to restore.
    OutputRemoved,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlaybackNotice {
    pub kind: NoticeKind,
    pub message: String,
}

/// Undrained notices kept; the oldest go first.
const NOTICE_LIMIT: usize = 32;
const RESTORE_ATTEMPTS: u32 = 3;
const RESTORE_RETRY_DELAY: Duration = Duration::from_millis(250);

enum Msg {
    Begin {
        epoch: u64,
        settings: PlaybackSettings,
    },
    End {
        epoch: u64,
    },
    Shutdown,
    /// Answered once every earlier message is handled: whether nothing is
    /// attenuated and no restore failed since the last take began.
    Flush(Sender<bool>),
}

struct Shared {
    tx: Sender<Msg>,
    state: Mutex<SharedState>,
}

#[derive(Default)]
struct SharedState {
    next_epoch: u64,
    /// Why the last take could not touch playback on this system.
    unsupported: Option<String>,
    notices: VecDeque<PlaybackNotice>,
}

impl Shared {
    fn state(&self) -> MutexGuard<'_, SharedState> {
        self.state.lock().expect("playback state lock")
    }

    fn notice(&self, kind: NoticeKind, message: String) {
        let mut state = self.state();
        if state.notices.len() == NOTICE_LIMIT {
            state.notices.pop_front();
        }
        state.notices.push_back(PlaybackNotice { kind, message });
    }
}

/// One take's claim on the attenuation. Ending it twice is harmless.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlaybackLease(u64);

/// Cloneable, non-blocking handle to a [`PlaybackAttenuation`].
#[derive(Clone)]
pub struct PlaybackHandle {
    shared: Arc<Shared>,
}

impl PlaybackHandle {
    /// A take is starting. `Off` settings touch nothing.
    pub fn begin(&self, settings: &PlaybackSettings) -> PlaybackLease {
        let epoch = {
            let mut state = self.shared.state();
            state.next_epoch += 1;
            state.next_epoch
        };
        let _ = self.shared.tx.send(Msg::Begin {
            epoch,
            settings: *settings,
        });
        PlaybackLease(epoch)
    }

    /// The take behind `lease` stopped recording.
    pub fn end(&self, lease: PlaybackLease) {
        let _ = self.shared.tx.send(Msg::End { epoch: lease.0 });
    }

    /// Why playback cannot be controlled here, as of the last take that
    /// tried.
    pub fn unsupported_reason(&self) -> Option<String> {
        self.shared.state().unsupported.clone()
    }

    pub fn take_notices(&self) -> Vec<PlaybackNotice> {
        self.shared.state().notices.drain(..).collect()
    }

    /// Answers once every request sent before it was handled, including
    /// an `end`'s restore (retries and all), with whether playback is now
    /// as the user left it: `false` while attenuated, and after a restore
    /// that failed until the next take begins (the user was told; what
    /// they fix by hand is theirs). The start/stop cues (#221) wait on it so they are not
    /// played into a lowered or muted output. Disconnects without an
    /// answer once the service has shut down.
    pub fn settled(&self) -> Receiver<bool> {
        let (tx, rx) = mpsc::channel();
        let _ = self.shared.tx.send(Msg::Flush(tx));
        rx
    }

    #[cfg(test)]
    fn flush(&self) {
        self.settled().recv().expect("service flushed");
    }
}

/// The attenuation service: one thread, one backend. Shutting it down (or
/// dropping it) restores any live attenuation before the thread joins.
pub struct PlaybackAttenuation {
    shared: Arc<Shared>,
    thread: Option<JoinHandle<()>>,
}

impl PlaybackAttenuation {
    pub fn start(backend: Arc<dyn PlaybackBackend>) -> PlaybackAttenuation {
        let (tx, rx) = mpsc::channel();
        let shared = Arc::new(Shared {
            tx,
            state: Mutex::default(),
        });
        let mut worker = Worker {
            backend,
            shared: Arc::clone(&shared),
            attenuation: None,
            restore_failed: false,
        };
        let thread = std::thread::Builder::new()
            .name("starling-playback-attenuation".to_string())
            .spawn(move || worker.run(rx))
            .expect("playback attenuation thread spawn");
        PlaybackAttenuation {
            shared,
            thread: Some(thread),
        }
    }

    pub fn handle(&self) -> PlaybackHandle {
        PlaybackHandle {
            shared: Arc::clone(&self.shared),
        }
    }

    /// Restores any live attenuation and joins the service thread.
    pub fn shutdown(&mut self) {
        if let Some(thread) = self.thread.take() {
            let _ = self.shared.tx.send(Msg::Shutdown);
            let _ = thread.join();
        }
    }
}

impl Drop for PlaybackAttenuation {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// The volume vector or the mute flag.
#[derive(Clone, Debug)]
enum Field<T> {
    /// `held`: the last confirmed value (the snapshot or our last write),
    /// plus the target of a write that reported an error but may have
    /// landed.
    Owned { snapshot: T, held: Vec<T> },
    /// The user changed it: never written again.
    Relinquished,
}

impl<T: Clone + PartialEq> Field<T> {
    fn new(snapshot: &T) -> Self {
        Field::Owned {
            snapshot: snapshot.clone(),
            held: vec![snapshot.clone()],
        }
    }

    /// Reconciles with a fresh read. Relinquishing a field we had changed
    /// returns its snapshot and the value we wrote.
    fn observe(&mut self, current: &T) -> Option<(T, T)> {
        match self {
            Field::Owned { held, .. } if held.contains(current) => {
                *held = vec![current.clone()];
                None
            }
            Field::Owned { snapshot, held } => {
                let written = held
                    .iter()
                    .find(|value| *value != snapshot)
                    .map(|value| (snapshot.clone(), value.clone()));
                *self = Field::Relinquished;
                written
            }
            Field::Relinquished => None,
        }
    }

    /// The value to send: `target(snapshot, current)` while owned, recorded
    /// before it is sent; `current` otherwise.
    fn plan(&mut self, current: &T, target: impl Fn(&T, &T) -> T) -> T {
        match self {
            Field::Owned { snapshot, held } => {
                let value = target(snapshot, current);
                if value != *current {
                    held.push(value.clone());
                }
                value
            }
            Field::Relinquished => current.clone(),
        }
    }

    /// The write was acknowledged: the field holds the last value sent.
    fn confirm(&mut self) {
        if let Field::Owned { held, .. } = self {
            held.drain(..held.len() - 1);
        }
    }

    fn changed(&self) -> bool {
        matches!(self, Field::Owned { snapshot, held } if held.iter().any(|value| value != snapshot))
    }
}

struct Attenuation {
    owners: Vec<u64>,
    device: String,
    volume: Field<Vec<u32>>,
    mute: Field<bool>,
    /// The user changed the volume while some channel still held our
    /// lowering (or the channel layout changed): it is not restored.
    volume_stranded: bool,
}

impl Attenuation {
    fn new(snapshot: &OutputSnapshot) -> Self {
        Attenuation {
            owners: Vec::new(),
            device: snapshot.device.clone(),
            volume: Field::new(&snapshot.volumes),
            mute: Field::new(&snapshot.muted),
            volume_stranded: false,
        }
    }

    fn observe(&mut self, current: &OutputSnapshot) {
        let now = &current.volumes;
        if let Some((snapshot, written)) = self.volume.observe(now) {
            self.volume_stranded |= written.len() != now.len()
                || (written.iter().zip(&snapshot).zip(now))
                    .any(|((ours, before), now)| ours != before && ours == now);
        }
        self.mute.observe(&current.muted);
    }

    fn changed(&self) -> bool {
        self.volume.changed() || self.mute.changed()
    }

    /// Moves every owned field to `target(snapshot, current)` in one write.
    fn apply(
        &mut self,
        backend: &dyn PlaybackBackend,
        current: &OutputSnapshot,
        volume: impl Fn(&Vec<u32>, &Vec<u32>) -> Vec<u32>,
        mute: impl Fn(&bool, &bool) -> bool,
    ) -> Result<(), PlaybackError> {
        let volumes = self.volume.plan(&current.volumes, volume);
        let muted = self.mute.plan(&current.muted, mute);
        let change = OutputChange {
            volumes: (volumes != current.volumes).then_some(volumes),
            set_muted: (muted != current.muted).then_some(muted),
        };
        if change == OutputChange::default() {
            return Ok(());
        }
        backend.apply(&self.device, &change)?;
        self.volume.confirm();
        self.mute.confirm();
        Ok(())
    }
}

struct Worker {
    backend: Arc<dyn PlaybackBackend>,
    shared: Arc<Shared>,
    attenuation: Option<Attenuation>,
    /// A restore gave up with playback still adjusted since the last
    /// `begin`.
    restore_failed: bool,
}

impl Worker {
    fn run(&mut self, rx: Receiver<Msg>) {
        while let Ok(msg) = rx.recv() {
            match msg {
                Msg::Begin { epoch, settings } => self.begin(epoch, settings),
                Msg::End { epoch } => self.end(epoch),
                Msg::Shutdown => break,
                Msg::Flush(done) => {
                    let _ = done.send(self.attenuation.is_none() && !self.restore_failed);
                }
            }
        }
        if let Some(attenuation) = self.attenuation.take() {
            self.restore(attenuation);
        }
    }

    fn begin(&mut self, epoch: u64, settings: PlaybackSettings) {
        self.restore_failed = false;
        if settings.during_recording == PlaybackMode::Off {
            return;
        }
        let result = self.attenuate(epoch, settings);
        self.shared.state().unsupported = match &result {
            Err(PlaybackError::Unsupported(reason)) => Some(reason.clone()),
            _ => None,
        };
        if let Err(err) = result {
            self.shared.notice(
                NoticeKind::AdjustFailed,
                format!("Playback could not be adjusted for this recording ({err})."),
            );
        }
    }

    fn attenuate(&mut self, epoch: u64, settings: PlaybackSettings) -> Result<(), PlaybackError> {
        let current = if let Some(live) = &mut self.attenuation {
            live.owners.push(epoch);
            let current = self.backend.output(&live.device)?;
            live.observe(&current);
            current
        } else {
            let snapshot = self.backend.active_output()?;
            let mut attenuation = Attenuation::new(&snapshot);
            attenuation.owners.push(epoch);
            self.attenuation = Some(attenuation);
            snapshot
        };
        let attenuation = self.attenuation.as_mut().expect("attenuation is live");
        let backend = &*self.backend;
        match settings.during_recording {
            PlaybackMode::Off => Ok(()),
            PlaybackMode::Mute => {
                attenuation.apply(backend, &current, |_, now| now.clone(), |_, _| true)
            }
            PlaybackMode::Lower => {
                let cap = settings.effective_lower_percent() * VOLUME_NORM / 100;
                let capped =
                    |_: &Vec<u32>, now: &Vec<u32>| now.iter().map(|&v| v.min(cap)).collect();
                attenuation.apply(backend, &current, capped, |_, now| *now)
            }
        }
    }

    fn end(&mut self, epoch: u64) {
        let Some(attenuation) = &mut self.attenuation else {
            return;
        };
        attenuation.owners.retain(|owner| *owner != epoch);
        if attenuation.owners.is_empty() {
            let attenuation = self.attenuation.take().expect("attenuation is live");
            self.restore(attenuation);
        }
    }

    fn restore(&mut self, mut attenuation: Attenuation) {
        let device = attenuation.device.clone();
        let mut attempt = 1;
        loop {
            match self.try_restore(&mut attenuation) {
                Ok(()) => break,
                Err(PlaybackError::NoSuchDevice(detail)) => {
                    self.shared.notice(
                        NoticeKind::OutputRemoved,
                        format!(
                            "The playback device Starling adjusted ('{device}') was removed \
                             ({detail}); there was nothing to restore."
                        ),
                    );
                    return;
                }
                Err(err) if attempt >= RESTORE_ATTEMPTS => {
                    self.restore_failed = true;
                    self.shared.notice(
                        NoticeKind::RestoreFailed,
                        format!(
                            "Starling could not restore playback on '{device}' ({err}). Please \
                             adjust your system volume or unmute manually."
                        ),
                    );
                    return;
                }
                Err(_) => {
                    attempt += 1;
                    std::thread::sleep(RESTORE_RETRY_DELAY);
                }
            }
        }
        if attenuation.volume_stranded {
            self.shared.notice(
                NoticeKind::RestoreFailed,
                format!(
                    "The volume of '{device}' changed during recording, so Starling left it \
                     as it is. Please check it."
                ),
            );
        }
    }

    fn try_restore(&self, attenuation: &mut Attenuation) -> Result<(), PlaybackError> {
        if !attenuation.changed() {
            return Ok(());
        }
        let current = self.backend.output(&attenuation.device)?;
        attenuation.observe(&current);
        attenuation.apply(
            &*self.backend,
            &current,
            |snapshot, _| snapshot.clone(),
            |snapshot, _| *snapshot,
        )
    }
}

/// The platform's backend: `pactl` on Linux. Windows (Core Audio) and
/// macOS have none yet.
pub fn platform_backend() -> Arc<dyn PlaybackBackend> {
    if cfg!(target_os = "linux") {
        Arc::new(PactlBackend)
    } else {
        Arc::new(UnsupportedBackend)
    }
}

struct UnsupportedBackend;

impl UnsupportedBackend {
    fn error() -> PlaybackError {
        PlaybackError::Unsupported("not implemented on this platform yet".to_string())
    }
}

impl PlaybackBackend for UnsupportedBackend {
    fn active_output(&self) -> Result<OutputSnapshot, PlaybackError> {
        Err(Self::error())
    }
    fn output(&self, _id: &str) -> Result<OutputSnapshot, PlaybackError> {
        Err(Self::error())
    }
    fn apply(&self, _id: &str, _change: &OutputChange) -> Result<(), PlaybackError> {
        Err(Self::error())
    }
}

/// PulseAudio and PipeWire (pipewire-pulse) through the `pactl` CLI.
struct PactlBackend;

/// A wedged audio server must not hold the service thread, and with it
/// restores and app quit.
const PACTL_DEADLINE: Duration = Duration::from_secs(5);

fn pactl(args: &[&str]) -> Result<String, PlaybackError> {
    // pactl localizes its output; the parsers match the English keys.
    let mut child = Command::new("pactl")
        .args(args)
        .env("LC_ALL", "C")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|err| {
            PlaybackError::Unsupported(if err.kind() == std::io::ErrorKind::NotFound {
                "pactl not found; install pipewire-pulse or pulseaudio".to_string()
            } else {
                format!("pactl could not be run: {err}")
            })
        })?;
    let backend_error = |err: std::io::Error| PlaybackError::Backend(format!("pactl: {err}"));
    let started = Instant::now();
    while child.try_wait().map_err(backend_error)?.is_none() {
        if started.elapsed() >= PACTL_DEADLINE {
            let _ = child.kill();
            let _ = child.wait();
            return Err(PlaybackError::Backend(format!(
                "pactl did not answer within {} s",
                PACTL_DEADLINE.as_secs()
            )));
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    let output = child.wait_with_output().map_err(backend_error)?;
    if output.status.success() {
        return Ok(String::from_utf8_lossy(&output.stdout).into_owned());
    }
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    Err(if stderr.contains("No such entity") {
        PlaybackError::NoSuchDevice(stderr)
    } else {
        PlaybackError::Backend(stderr)
    })
}

impl PlaybackBackend for PactlBackend {
    fn active_output(&self) -> Result<OutputSnapshot, PlaybackError> {
        // `pactl info` failing means no reachable audio server.
        let info = pactl(&["info"]).map_err(|err| match err {
            PlaybackError::Unsupported(_) => err,
            other => PlaybackError::Unsupported(other.to_string()),
        })?;
        let sink = parse_default_sink(&info).ok_or_else(|| {
            PlaybackError::Backend("`pactl info` named no default sink".to_string())
        })?;
        self.output(&sink)
    }

    fn output(&self, id: &str) -> Result<OutputSnapshot, PlaybackError> {
        let volumes = parse_raw_volumes(&pactl(&["get-sink-volume", id])?);
        let muted = parse_mute(&pactl(&["get-sink-mute", id])?).ok_or_else(|| {
            PlaybackError::Backend(format!("`pactl get-sink-mute {id}` printed no mute state"))
        })?;
        Ok(OutputSnapshot {
            device: id.to_string(),
            volumes,
            muted,
        })
    }

    fn apply(&self, id: &str, change: &OutputChange) -> Result<(), PlaybackError> {
        let volume = change.volumes.as_ref().map(|volumes| {
            let raw: Vec<String> = volumes.iter().map(u32::to_string).collect();
            let mut args = vec!["set-sink-volume", id];
            args.extend(raw.iter().map(String::as_str));
            pactl(&args)
        });
        let mute = change
            .set_muted
            .map(|muted| pactl(&["set-sink-mute", id, if muted { "1" } else { "0" }]));
        volume
            .into_iter()
            .chain(mute)
            .try_for_each(|result| result.map(drop))
    }
}

/// `pactl info` → the `Default Sink:` value.
fn parse_default_sink(info: &str) -> Option<String> {
    info.lines()
        .find_map(|line| line.strip_prefix("Default Sink:"))
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(str::to_string)
}

/// `pactl get-sink-volume` → each channel's raw volume, in channel order,
/// e.g. `Volume: front-left: 39321 /  60% / -13.81 dB,   front-right: ...`.
fn parse_raw_volumes(text: &str) -> Vec<u32> {
    let tokens: Vec<&str> = text.split_whitespace().collect();
    tokens
        .windows(2)
        .filter(|pair| pair[0].ends_with(':'))
        .filter_map(|pair| pair[1].parse().ok())
        .collect()
}

/// `pactl get-sink-mute` (`Mute: yes` / `Mute: no`) → the flag.
fn parse_mute(text: &str) -> Option<bool> {
    match text
        .lines()
        .find_map(|line| line.strip_prefix("Mute:"))?
        .trim()
    {
        "yes" => Some(true),
        "no" => Some(false),
        _ => None,
    }
}

#[cfg(test)]
mod tests;
