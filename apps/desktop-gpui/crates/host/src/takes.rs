//! The take feed (#220): the app's projection of the takes this host
//! records.
//!
//! The app starts and stops takes with the v1 `capture.*` commands like
//! any client; the I3 events say what the machine did. What the app
//! needs beyond them — the take's health while it records (input route,
//! stall, disk, fault), its audio for live transcription, and which
//! history row it landed in — rides these host-level frames, the same
//! way the agent ask frames ride beside the envelope:
//!
//! - [`Frame::TakeWatch`]: the connection follows the feed. Every
//!   watching connection gets a [`Frame::LiveTake`] status tick every
//!   [`TICK`] while a take records, one `ended` frame when it stops,
//!   [`Frame::TakeStartFailed`] and [`Frame::TakePersisted`].
//! - [`Frame::TakeTap`]: the connection also wants the take's audio from
//!   a sample index on. The hub reads it from the recorder without
//!   taking it from the stop handshake ([`LiveTakeMonitor`]), then from
//!   the finished take's record, and sends the final `ended` frame only
//!   after the last sample — so the app's live stream can finish the
//!   take exactly. Audio waits for room: it never takes more than a
//!   quarter of a connection's queue, so events and receipts are never
//!   crowded out (a slow consumer would be closed).
//!
//! Each take has an **owner**: the connection whose `capture.start`
//! opened it, or the one that tapped it after its owner was gone.
//! Each tick tells its connection whose take it is ([`TakeOwner`]: the
//! connection itself, another live one, or nobody), so a second app
//! window never takes over a take another live window records — it only
//! adopts one whose owner died, and of two windows racing to adopt, the
//! one whose tap arrives second sees the take is another's.
//!
//! A take outlives its app (the renderer-kill acceptance): when no
//! connection watches a recording take for [`HostConfig::orphan_grace`],
//! the hub stops it itself — the take is finalized and stored like any
//! other. A take that is stored after its owner is gone is an
//! **orphan** (`orphan: true` on its [`Frame::TakePersisted`]). An app
//! that reconnects while the take still records sees its status ticks
//! (`owner: nobody`) and adopts it ([`Frame::TakeAdopt`]).
//!
//! The host transcribes every take it stores ([`crate::transcribe`]):
//! watchers hear each take stored, its live text while it records
//! ([`Frame::LiveText`]) and its transcription ([`Frame::Transcription`]).
//! A frame the connection it is for has to act on (the owner's stored
//! row, the result its window delivers) waits for room in its queue
//! rather than being dropped.
//!
//! Agent asks (`ask_*` corrs) are the broker's business: their takes are
//! not part of this feed, and agent connections may not follow it. Any
//! other connection may — like every command, the feed is open to the
//! user's own processes (peer-credential auth), which can already read
//! the journals and the store on disk; it grants no access they lack.
//!
//! [`HostConfig::orphan_grace`]: crate::config::HostConfig::orphan_grace

use std::collections::VecDeque;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use starling_runtime::machine::capture::{
    CaptureObserver, LiveTakeMonitor, TakeRecord, TakeStatus,
};
use starling_runtime::protocol::Command;
use starling_runtime::RuntimeClient;

use crate::agent::ASK_PREFIX;
use crate::frame::{
    Frame, HostRecovery, LivePartial, TakeAudio, TakeOwner, TranscriptionState, METER_SAMPLES,
};
use crate::server::{lock_registry, ConnState};
use crate::transcribe::TranscriberLink;

/// How often watching connections hear about a recording take.
pub const TICK: Duration = Duration::from_millis(50);

/// The default for [`crate::config::HostConfig::orphan_grace`]: long
/// enough for an app that crashed to be relaunched and adopt its take,
/// short enough that a microphone nobody watches does not stay open.
pub const DEFAULT_ORPHAN_GRACE: Duration = Duration::from_secs(20);

/// The most samples one audio frame carries (a second at 48 kHz: 256 KB
/// of base64, well under the default frame cap).
const FRAME_SAMPLES: usize = 48_000;

/// The most audio frames one connection gets per tick: a replay after a
/// reconnect catches up at up to 80 s of audio per second.
const FRAMES_PER_TICK: usize = 4;

/// Finished takes whose audio stays readable for taps still catching up.
const ENDED_KEEP: usize = 4;

/// What a `capture.start` failure's detail carries when the host's own
/// capture source produced it (see [`crate::capture`]).
#[derive(serde::Serialize, serde::Deserialize)]
pub(crate) struct StartFailure {
    pub problem: Option<starling_dictation::microphone::InputProblem>,
    pub message: String,
}

pub struct TakeHub {
    state: Mutex<HubState>,
    client: OnceLock<RuntimeClient>,
    /// The host's transcriber, told about every take (#220).
    transcriber: OnceLock<TranscriberLink>,
    orphan_grace: Duration,
}

#[derive(Default)]
struct HubState {
    /// The connection that sent each recent `capture.start`, by corr.
    starters: VecDeque<(String, Arc<ConnState>)>,
    live: Option<Live>,
    ended: VecDeque<Ended>,
    watchers: Vec<Watcher>,
    /// Startup recovery's findings until an app hears them.
    recovery: Option<HostRecovery>,
    /// A `capture.start` still opening its device, and who sent it: it
    /// owns the take before the take exists.
    acquiring: Option<(String, Arc<ConnState>)>,
    /// Since when the live take has had no watcher.
    orphaned_since: Option<Instant>,
    /// Takes the hub stopped itself (orphaned).
    orphan_stopped: Vec<String>,
    /// Takes whose persist has not reported yet, with their owner (the
    /// `ended` cache is bounded; this is not, and shrinks per persist).
    persisting: Vec<(String, Option<Arc<ConnState>>)>,
    /// Frames the connection they are for has to get (a transcription
    /// result it acts on), waiting for room in its queue.
    owed: Vec<(Arc<ConnState>, Frame)>,
}

struct Live {
    take: String,
    rate: u32,
    monitor: Option<Arc<dyn LiveTakeMonitor>>,
    owner: Option<Arc<ConnState>>,
    /// The take's newest live text, for an app that adopts it.
    text: Option<LivePartial>,
    degraded: Option<String>,
}

struct Ended {
    take: String,
    rate: u32,
    owner: Option<Arc<ConnState>>,
    /// Everything the take kept; `None` when it kept nothing.
    record: Option<Arc<TakeRecord>>,
}

struct Watcher {
    conn: Arc<ConnState>,
    /// One per take: a finished take's tap keeps catching up after the
    /// app started its next take.
    taps: Vec<Tap>,
}

impl Watcher {
    /// Sends `frame` now, or after the tap on `take` finished.
    fn deliver_after_tap(&mut self, take: &str, frame: Frame) -> bool {
        match self.taps.iter_mut().find(|tap| tap.take == take) {
            Some(tap) => {
                tap.held.push(frame);
                true
            }
            None => self.conn.try_deliver(frame).is_ok(),
        }
    }

    fn tapping(&self, take: &str) -> bool {
        self.taps.iter().any(|tap| tap.take == take)
    }
}

struct Tap {
    take: String,
    cursor: u64,
    /// Frames held until this tap's `ended` went out: a tapping app hears
    /// its take stored only after the last of its audio.
    held: Vec<Frame>,
    /// The finished take's `ended` went out; only `held` is left.
    ended_sent: bool,
}

fn is_agent_take(corr: &str) -> bool {
    corr.starts_with(ASK_PREFIX)
}

fn alive(owner: &Option<Arc<ConnState>>) -> bool {
    owner
        .as_ref()
        .is_some_and(|conn| !conn.closed.load(Ordering::SeqCst))
}

/// Whose take it is, as `conn` sees it.
fn owner_for(owner: &Option<Arc<ConnState>>, conn: &Arc<ConnState>) -> TakeOwner {
    match owner {
        Some(owner) if Arc::ptr_eq(owner, conn) => TakeOwner::You,
        _ if alive(owner) => TakeOwner::Another,
        _ => TakeOwner::Nobody,
    }
}

impl TakeHub {
    pub fn new(orphan_grace: Duration) -> Arc<TakeHub> {
        Arc::new(TakeHub {
            state: Mutex::new(HubState::default()),
            client: OnceLock::new(),
            transcriber: OnceLock::new(),
            orphan_grace,
        })
    }

    /// The runtime the hub stops orphaned takes through (set once the
    /// runtime started; the hub exists before it, as its observer).
    pub(crate) fn attach(&self, client: RuntimeClient) {
        let _ = self.client.set(client);
    }

    /// The transcriber the hub tells about takes (set once it started).
    pub(crate) fn attach_transcriber(&self, link: TranscriberLink) {
        let _ = self.transcriber.set(link);
    }

    /// The live take's newest preview (or why live text stopped): to every
    /// watcher, and kept for one that adopts the take later. A watcher
    /// whose queue is full misses this one; the next preview holds the
    /// whole text again.
    pub(crate) fn live_text(
        &self,
        take: &str,
        partial: Option<LivePartial>,
        degraded: Option<String>,
    ) {
        let mut state = lock_registry(&self.state);
        if let Some(live) = state.live.as_mut().filter(|live| live.take == take) {
            if partial.is_some() {
                live.text = partial.clone();
            }
            if degraded.is_some() {
                live.degraded = degraded.clone();
            }
        }
        for watcher in &state.watchers {
            if has_room(&watcher.conn) {
                let _ = watcher.conn.try_deliver(Frame::LiveText {
                    take: take.to_string(),
                    partial: partial.clone(),
                    degraded: degraded.clone(),
                });
            }
        }
    }

    /// Where the transcription of stored take `stored_id` stands: to every
    /// watcher (and to a requesting `owner` that does not watch). The one
    /// it is for — `owner` while it lives (a take's owner only while it
    /// watches), else the first watching app — acts on it. Every watcher
    /// gets it whatever its queue says (it waits for room, in order): a
    /// window that saw the transcription start shows the take busy until
    /// it hears the end, and nothing else would tell it.
    pub(crate) fn transcription(
        &self,
        stored_id: &str,
        take: Option<&str>,
        req: Option<&str>,
        attempt: Option<&str>,
        transcription: TranscriptionState,
        owner: Option<&Arc<ConnState>>,
    ) {
        let mut state = lock_registry(&self.state);
        // A recorded take's owner acts on it while it follows the feed (a
        // command-only owner cannot show anything); a requester always.
        let recipient = owner
            .filter(|owner| !owner.closed.load(Ordering::SeqCst))
            .filter(|owner| {
                (take.is_none() && req.is_some())
                    || state
                        .watchers
                        .iter()
                        .any(|watcher| Arc::ptr_eq(&watcher.conn, owner))
            })
            .cloned()
            .or_else(|| state.watchers.first().map(|watcher| Arc::clone(&watcher.conn)));
        let frame = |yours: bool| Frame::Transcription {
            stored_id: stored_id.to_string(),
            take: take.map(str::to_string),
            req: req.map(str::to_string),
            attempt: attempt.map(str::to_string),
            state: transcription.clone(),
            yours,
        };
        let HubState {
            watchers, owed, ..
        } = &mut *state;
        let mut reached = false;
        for watcher in watchers.iter_mut() {
            let yours = recipient
                .as_ref()
                .is_some_and(|recipient| Arc::ptr_eq(recipient, &watcher.conn));
            reached |= yours;
            match take {
                Some(take) if watcher.tapping(take) => {
                    watcher.deliver_after_tap(take, frame(yours));
                }
                _ => deliver_owed(owed, &watcher.conn, frame(yours)),
            }
        }
        if let (Some(recipient), false) = (recipient, reached) {
            deliver_owed(owed, &recipient, frame(true));
        }
    }

    /// `conn` takes on running take `take` if its owner is gone (an app
    /// that came back), and hears its latest live text.
    pub(crate) fn adopt(&self, conn: &Arc<ConnState>, take: &str) {
        let mut state = lock_registry(&self.state);
        if let Some(live) = state.live.as_mut().filter(|live| live.take == take) {
            if !alive(&live.owner) {
                live.owner = Some(Arc::clone(conn));
            }
            if live.text.is_some() || live.degraded.is_some() {
                let _ = conn.try_deliver(Frame::LiveText {
                    take: take.to_string(),
                    partial: live.text.clone(),
                    degraded: live.degraded.clone(),
                });
            }
        }
        if let Some(done) = state.ended.iter_mut().find(|done| done.take == take) {
            if !alive(&done.owner) {
                done.owner = Some(Arc::clone(conn));
            }
        }
        if let Some((_, owner)) = state.persisting.iter_mut().find(|(known, _)| known == take) {
            if !alive(owner) {
                *owner = Some(Arc::clone(conn));
            }
        }
    }

    /// Startup recovery's findings, for the first app that watches.
    pub(crate) fn set_recovery(&self, recovery: HostRecovery) {
        if recovery.notice.is_empty() && recovery.problems.is_empty() {
            return;
        }
        lock_registry(&self.state).recovery = Some(recovery);
    }

    /// A later recovery finding: to every watcher, or kept for the next.
    pub(crate) fn notice(&self, recovery: HostRecovery) {
        if recovery.notice.is_empty() && recovery.problems.is_empty() {
            return;
        }
        let mut state = lock_registry(&self.state);
        let mut told = false;
        for watcher in &state.watchers {
            told |= watcher
                .conn
                .try_deliver(Frame::HostNotice {
                    recovery: recovery.clone(),
                })
                .is_ok();
        }
        if !told {
            let merged = match state.recovery.take() {
                Some(earlier) => HostRecovery {
                    notice: join(&earlier.notice, &recovery.notice),
                    problems: join(&earlier.problems, &recovery.problems),
                },
                None => recovery,
            };
            state.recovery = Some(merged);
        }
    }

    /// Whether a take records or is still being stored.
    pub fn busy(&self) -> bool {
        let state = lock_registry(&self.state);
        state.live.is_some() || !state.persisting.is_empty()
    }

    /// `conn` follows the feed from now on.
    pub(crate) fn watch(&self, conn: &Arc<ConnState>, req: String) -> Result<(), ()> {
        let mut state = lock_registry(&self.state);
        // Startup's findings are kept until a watcher was told them.
        let recovery = state.recovery.take();
        if conn
            .try_deliver(Frame::TakeWatching {
                req,
                recovery: recovery.clone(),
            })
            .is_err()
        {
            state.recovery = recovery;
            return Err(());
        }
        if !state.watchers.iter().any(|watcher| Arc::ptr_eq(&watcher.conn, conn)) {
            state.watchers.push(Watcher {
                conn: Arc::clone(conn),
                taps: Vec::new(),
            });
        }
        state.orphaned_since = None;
        Ok(())
    }

    /// Whether `conn` may stop or cancel take `corr` (`None`: whatever
    /// take is current): not while another live connection owns it (a
    /// second window must not end, and then transcribe, a take another
    /// window records).
    pub(crate) fn may_end(&self, corr: Option<&str>, conn: &Arc<ConnState>) -> bool {
        let state = lock_registry(&self.state);
        if let Some(live) = state
            .live
            .as_ref()
            .filter(|live| corr.is_none_or(|corr| live.take == corr))
        {
            return owner_for(&live.owner, conn) != TakeOwner::Another;
        }
        // The take may still be opening its device: the command would run
        // after the start, on that take.
        match state
            .acquiring
            .as_ref()
            .filter(|(take, _)| corr.is_none_or(|corr| take == corr))
        {
            Some((_, starter)) => {
                owner_for(&Some(Arc::clone(starter)), conn) != TakeOwner::Another
            }
            None => true,
        }
    }

    /// `conn` sent `capture.start` for `corr`: it owns that take — unless
    /// a take by that corr is recording or opening its device already
    /// (the machine refuses the start; a start is never a way to take a
    /// take over, nor to lose one's own). A corr a finished take used is
    /// free again. `true` when this registered the start, for
    /// [`TakeHub::start_refused`] to undo.
    pub(crate) fn starting(&self, corr: &str, conn: &Arc<ConnState>) -> bool {
        if is_agent_take(corr) {
            return false;
        }
        let mut state = lock_registry(&self.state);
        let current = state.live.as_ref().is_some_and(|live| live.take == corr)
            || state.acquiring.as_ref().is_some_and(|(take, _)| take == corr);
        if current {
            return false;
        }
        state.starters.retain(|(known, _)| known != corr);
        state.starters.push_back((corr.to_string(), Arc::clone(conn)));
        // A start that arrives while another is still opening its device
        // is refused by the machine; it must not take over that ownership.
        let acquiring_alive = state
            .acquiring
            .as_ref()
            .is_some_and(|(_, starter)| !starter.closed.load(Ordering::SeqCst));
        if state.live.is_none() && !acquiring_alive {
            state.acquiring = Some((corr.to_string(), Arc::clone(conn)));
        }
        // The oldest go first — never the take still opening its device:
        // its start report resolves its owner from here.
        while state.starters.len() > ENDED_KEEP {
            let acquiring = state.acquiring.as_ref().map(|(take, _)| take.clone());
            let Some(index) = state
                .starters
                .iter()
                .position(|(known, _)| Some(known) != acquiring.as_ref())
            else {
                break;
            };
            state.starters.remove(index);
        }
        true
    }

    /// The runtime refused the `capture.start` [`TakeHub::starting`]
    /// registered: no take opens, so `conn` owns nothing by it.
    pub(crate) fn start_refused(&self, corr: &str, conn: &Arc<ConnState>) {
        let mut state = lock_registry(&self.state);
        state
            .starters
            .retain(|(known, starter)| !(known == corr && Arc::ptr_eq(starter, conn)));
        if state
            .acquiring
            .as_ref()
            .is_some_and(|(take, starter)| take == corr && Arc::ptr_eq(starter, conn))
        {
            state.acquiring = None;
        }
    }

    /// `conn` wants `take`'s audio from sample `from` on. Tapping a take
    /// whose owner is gone adopts it.
    pub(crate) fn tap(&self, conn: &Arc<ConnState>, take: String, from: u64) {
        let mut state = lock_registry(&self.state);
        if let Some(live) = state.live.as_mut().filter(|live| live.take == take) {
            if !alive(&live.owner) {
                live.owner = Some(Arc::clone(conn));
            }
        }
        if let Some(done) = state.ended.iter_mut().find(|done| done.take == take) {
            if !alive(&done.owner) {
                done.owner = Some(Arc::clone(conn));
            }
        }
        if let Some((_, owner)) = state.persisting.iter_mut().find(|(known, _)| *known == take) {
            if !alive(owner) {
                *owner = Some(Arc::clone(conn));
            }
        }
        if let Some(watcher) = state
            .watchers
            .iter_mut()
            .find(|watcher| Arc::ptr_eq(&watcher.conn, conn))
        {
            // Re-tapping a take restarts its audio but keeps what was held
            // for after its end.
            match watcher.taps.iter_mut().find(|tap| tap.take == take) {
                Some(tap) => tap.cursor = from,
                None => watcher.taps.push(Tap {
                    take,
                    cursor: from,
                    held: Vec::new(),
                    ended_sent: false,
                }),
            }
        }
    }

    /// One feed tick: status and audio out, closed watchers dropped, an
    /// orphaned take stopped once its grace ran out.
    pub(crate) fn tick(&self) {
        let mut orphan_stop = None;
        {
            let mut state = lock_registry(&self.state);
            state
                .watchers
                .retain(|watcher| !watcher.conn.closed.load(Ordering::SeqCst));
            // In order, per connection: one that still has no room keeps
            // the rest of its frames behind the first it could not take.
            let mut stuck: Vec<Arc<ConnState>> = Vec::new();
            state.owed.retain(|(conn, frame)| {
                if conn.closed.load(Ordering::SeqCst) {
                    return false;
                }
                if stuck.iter().any(|known| Arc::ptr_eq(known, conn)) {
                    return true;
                }
                let held = conn.try_deliver(frame.clone()).is_err();
                if held {
                    stuck.push(Arc::clone(conn));
                }
                held
            });
            let HubState {
                live,
                ended,
                watchers,
                orphaned_since,
                orphan_stopped,
                ..
            } = &mut *state;
            let status = live
                .as_ref()
                .and_then(|live| live.monitor.as_ref().map(|monitor| monitor.status()));
            for watcher in watchers.iter_mut() {
                serve_watcher(watcher, live.as_ref(), status.as_ref(), ended);
            }
            match live {
                Some(live) if watchers.is_empty() => {
                    let since = *orphaned_since.get_or_insert_with(Instant::now);
                    if since.elapsed() >= self.orphan_grace
                        && !orphan_stopped.contains(&live.take)
                    {
                        orphan_stopped.push(live.take.clone());
                        orphan_stop = Some(live.take.clone());
                    }
                }
                _ => *orphaned_since = None,
            }
        }
        // Outside the lock: the actor calls back into the hub.
        if let (Some(take), Some(client)) = (orphan_stop, self.client.get()) {
            eprintln!(
                "starling-runtime-host: no app has followed take {take} for {:?}; \
                 stopping and storing it",
                self.orphan_grace
            );
            if let Err(err) = client.send(Some(&take), Command::CaptureStop { drain: Some(true) }) {
                eprintln!(
                    "starling-runtime-host: stopping orphaned take {take} failed ({err}); \
                     retrying in a second"
                );
                // Never leave the microphone open on a refused stop: ask
                // again a second from now.
                let mut state = lock_registry(&self.state);
                state.orphan_stopped.retain(|stopped| stopped != &take);
                let now = Instant::now();
                state.orphaned_since = Some(
                    now.checked_sub(self.orphan_grace.saturating_sub(Duration::from_secs(1)))
                        .unwrap_or(now),
                );
            }
        }
    }
}

/// One watcher's share of a tick.
fn serve_watcher(
    watcher: &mut Watcher,
    live: Option<&Live>,
    status: Option<&starling_runtime::machine::capture::LiveTakeStatus>,
    ended: &VecDeque<Ended>,
) {
    let conn = Arc::clone(&watcher.conn);
    if let Some(live) = live {
        let owner = owner_for(&live.owner, &conn);
        let tapped = watcher.taps.iter_mut().find(|tap| tap.take == live.take);
        match (tapped, live.monitor.as_ref()) {
            (Some(tap), Some(monitor)) => {
                let mut first = true;
                for _ in 0..FRAMES_PER_TICK {
                    if !has_room(&conn) {
                        break;
                    }
                    let samples = monitor.samples_from(tap.cursor as usize, FRAME_SAMPLES);
                    if samples.is_empty() && !first {
                        break;
                    }
                    let frame = Frame::LiveTake {
                        take: live.take.clone(),
                        rate: live.rate,
                        status: first.then(|| status.cloned()).flatten(),
                        audio: (!samples.is_empty())
                            .then(|| TakeAudio::encode(tap.cursor, &samples)),
                        owner,
                        ended: None,
                        kept: false,
                        meter: None,
                    };
                    if conn.try_deliver(frame).is_err() {
                        break;
                    }
                    tap.cursor += samples.len() as u64;
                    first = false;
                    if samples.len() < FRAME_SAMPLES {
                        break;
                    }
                }
            }
            (_, monitor) => {
                if has_room(&conn) {
                    // The owner's level meter: the newest samples.
                    let meter = monitor.filter(|_| owner == TakeOwner::You).map(|monitor| {
                        let from = monitor.sample_count().saturating_sub(METER_SAMPLES);
                        TakeAudio::encode(from as u64, &monitor.samples_from(from, METER_SAMPLES))
                    });
                    let _ = conn.try_deliver(Frame::LiveTake {
                        take: live.take.clone(),
                        rate: live.rate,
                        status: status.cloned(),
                        audio: None,
                        owner,
                        ended: None,
                        kept: false,
                        meter,
                    });
                }
            }
        }
    }
    // Taps on finished takes: the rest of their audio, then the end, then
    // what was held for after it — a frame the queue has no room for yet
    // stays held for the next tick, never dropped.
    let live_take = live.map(|live| live.take.as_str());
    watcher.taps.retain_mut(|tap| {
        if Some(tap.take.as_str()) == live_take {
            return true;
        }
        if !tap.ended_sent {
            tap.ended_sent = serve_finished_tap(&conn, tap, ended);
            if !tap.ended_sent {
                return true;
            }
        }
        while let Some(frame) = tap.held.first() {
            if conn.try_deliver(frame.clone()).is_err() {
                return true;
            }
            tap.held.remove(0);
        }
        false
    });
}

/// Sends a finished take's remaining audio to a tap; `true` once its end
/// went out.
fn serve_finished_tap(conn: &Arc<ConnState>, tap: &mut Tap, ended: &VecDeque<Ended>) -> bool {
    let Some(done) = ended.iter().find(|done| done.take == tap.take) else {
        // Nothing known by that name (long gone, or never ours): end it.
        return conn
            .try_deliver(Frame::LiveTake {
                take: tap.take.clone(),
                rate: 0,
                status: None,
                audio: None,
                owner: TakeOwner::Nobody,
                ended: Some(tap.cursor),
                kept: false,
                meter: None,
            })
            .is_ok();
    };
    let samples: &[f32] = done
        .record
        .as_ref()
        .map(|record| record.samples.as_slice())
        .unwrap_or(&[]);
    let owner = owner_for(&done.owner, conn);
    for _ in 0..FRAMES_PER_TICK {
        if !has_room(conn) {
            return false;
        }
        let start = (tap.cursor as usize).min(samples.len());
        if start >= samples.len() {
            break;
        }
        let end = (start + FRAME_SAMPLES).min(samples.len());
        let frame = Frame::LiveTake {
            take: done.take.clone(),
            rate: done.rate,
            status: None,
            audio: Some(TakeAudio::encode(start as u64, &samples[start..end])),
            owner,
            ended: None,
            kept: false,
            meter: None,
        };
        if conn.try_deliver(frame).is_err() {
            return false;
        }
        tap.cursor = end as u64;
    }
    (tap.cursor as usize) >= samples.len()
        && has_room(conn)
        && conn
            .try_deliver(Frame::LiveTake {
                take: done.take.clone(),
                rate: done.rate,
                status: None,
                audio: None,
                owner,
                ended: Some(samples.len() as u64),
                kept: done.record.is_some(),
                meter: None,
            })
            .is_ok()
}

/// Audio and status wait for room: they may use at most a quarter of the
/// connection's queue, so events and receipts always fit.
fn has_room(conn: &ConnState) -> bool {
    let (queued, capacity) = conn.outbound_depth();
    queued < capacity / 4
}

fn join(first: &str, second: &str) -> String {
    match (first.is_empty(), second.is_empty()) {
        (true, _) => second.to_string(),
        (_, true) => first.to_string(),
        _ => format!("{first} {second}"),
    }
}

impl CaptureObserver for TakeHub {
    fn take_started(&self, corr: &str, monitor: Option<Arc<dyn LiveTakeMonitor>>) {
        if is_agent_take(corr) {
            return;
        }
        let mut state = lock_registry(&self.state);
        let acquiring = state.acquiring.take();
        let rate = monitor.as_ref().map(|monitor| monitor.sample_rate()).unwrap_or(0);
        // The connection whose start is opening this device owns it; its
        // starter record is the fallback.
        let owner = acquiring
            .filter(|(take, _)| take == corr)
            .map(|(_, conn)| conn)
            .or_else(|| {
                state
                    .starters
                    .iter()
                    .find(|(known, _)| known == corr)
                    .map(|(_, conn)| Arc::clone(conn))
            });
        state.live = Some(Live {
            take: corr.to_string(),
            rate,
            monitor: monitor.clone(),
            owner,
            text: None,
            degraded: None,
        });
        state.orphaned_since = None;
        if let Some(transcriber) = self.transcriber.get() {
            transcriber.started(corr, monitor);
        }
    }

    fn take_start_failed(&self, corr: &str, detail: &str) {
        if is_agent_take(corr) {
            return;
        }
        let (problem, message) = match serde_json::from_str::<StartFailure>(detail) {
            Ok(failure) => (failure.problem, failure.message),
            Err(_) => (None, detail.to_string()),
        };
        let mut state = lock_registry(&self.state);
        let starter = match state.acquiring.take() {
            Some((take, conn)) if take == corr => Some(conn),
            other => {
                state.acquiring = other;
                state
                    .starters
                    .iter()
                    .find(|(known, _)| known == corr)
                    .map(|(_, conn)| Arc::clone(conn))
            }
        };
        let HubState {
            watchers, owed, ..
        } = &mut *state;
        for watcher in watchers.iter() {
            let frame = Frame::TakeStartFailed {
                take: corr.to_string(),
                problem: problem.clone(),
                message: message.clone(),
            };
            // The window that asked waits on this answer.
            if starter
                .as_ref()
                .is_some_and(|starter| Arc::ptr_eq(starter, &watcher.conn))
            {
                deliver_owed(owed, &watcher.conn, frame);
            } else {
                let _ = watcher.conn.try_deliver(frame);
            }
        }
    }

    fn take_ended(&self, corr: &str, record: Option<&Arc<TakeRecord>>) {
        if is_agent_take(corr) {
            return;
        }
        if let Some(transcriber) = self.transcriber.get() {
            transcriber.ended(corr, record.cloned());
        }
        let mut state = lock_registry(&self.state);
        let (rate, owner) = match state.live.take() {
            Some(live) if live.take == corr => (live.rate, live.owner),
            other => {
                // Not the take the hub saw start (its start report was
                // missed): its starter owns it.
                state.live = other;
                let starter = state
                    .starters
                    .iter()
                    .find(|(known, _)| known == corr)
                    .map(|(_, conn)| Arc::clone(conn));
                (record.map(|record| record.sample_rate).unwrap_or(0), starter)
            }
        };
        if record.is_some() {
            state.persisting.retain(|(known, _)| known != corr);
            state.persisting.push((corr.to_string(), owner.clone()));
        }
        state.ended.retain(|done| done.take != corr);
        state.ended.push_back(Ended {
            take: corr.to_string(),
            rate,
            owner: owner.clone(),
            record: record.cloned(),
        });
        // The oldest go first — but never one a tap still replays: its
        // audio, end and ownership are what that tap resolves on.
        while state.ended.len() > ENDED_KEEP {
            let Some(index) = state.ended.iter().position(|done| {
                !state.watchers.iter().any(|watcher| watcher.tapping(&done.take))
            }) else {
                break;
            };
            state.ended.remove(index);
        }
        // Watchers that do not tap this take hear the end now; a tap
        // hears it after its last sample (the tick).
        let total = record.map(|record| record.samples.len() as u64).unwrap_or(0);
        let HubState {
            watchers, owed, ..
        } = &mut *state;
        for watcher in watchers.iter() {
            if watcher.tapping(corr) {
                continue;
            }
            let frame = Frame::LiveTake {
                take: corr.to_string(),
                rate,
                status: None,
                audio: None,
                owner: owner_for(&owner, &watcher.conn),
                ended: Some(total),
                kept: record.is_some(),
                meter: None,
            };
            // The owner's window ends its take on this frame (and only then
            // finds its stored row): it waits for room, never dropped.
            if owner_for(&owner, &watcher.conn) == TakeOwner::You {
                deliver_owed(owed, &watcher.conn, frame);
            } else {
                let _ = watcher.conn.try_deliver(frame);
            }
        }
    }

    fn take_persisted(
        &self,
        corr: &str,
        record: &Arc<TakeRecord>,
        stored: Result<Option<String>, String>,
    ) {
        if is_agent_take(corr) {
            return;
        }
        {
            let mut state = lock_registry(&self.state);
            let owner = match state.persisting.iter().position(|(known, _)| known == corr) {
                Some(index) => state.persisting.remove(index).1,
                None => None,
            };
            let stopped_here = match state.orphan_stopped.iter().position(|take| take == corr) {
                Some(index) => {
                    state.orphan_stopped.remove(index);
                    true
                }
                None => false,
            };
            let orphan = stopped_here || !alive(&owner);
            let (stored_id, error) = match stored {
                Ok(id) => (id, None),
                Err(err) => (None, Some(err)),
            };
            if let Some(transcriber) = self.transcriber.get() {
                // Only a complete take is transcribed: an interrupted one
                // waits for the user, a failed store has nothing to read.
                let complete = record.status != TakeStatus::Interrupted && error.is_none();
                transcriber.persisted(
                    corr,
                    stored_id.clone().filter(|_| complete),
                    owner.clone(),
                );
            }
            let frame = Frame::TakePersisted {
                take: corr.to_string(),
                stored_id,
                interrupted: record.status == TakeStatus::Interrupted,
                error,
                orphan,
            };
            // Every window hears it is in history, and its owner (which
            // binds its delivery to the stored row) whatever its queue
            // says; the host transcribes it.
            let HubState {
                watchers, owed, ..
            } = &mut *state;
            for watcher in watchers.iter_mut() {
                let theirs = owner
                    .as_ref()
                    .is_some_and(|owner| Arc::ptr_eq(&watcher.conn, owner));
                if watcher.tapping(corr) {
                    watcher.deliver_after_tap(corr, frame.clone());
                } else if theirs {
                    deliver_owed(owed, &watcher.conn, frame.clone());
                } else {
                    let _ = watcher.conn.try_deliver(frame.clone());
                }
            }
        }
    }
}

/// Sends `frame` to `conn` now, or — when its queue is full, or frames
/// owed to it wait already — after those, in order.
fn deliver_owed(owed: &mut Vec<(Arc<ConnState>, Frame)>, conn: &Arc<ConnState>, frame: Frame) {
    let behind = owed.iter().any(|(owed_to, _)| Arc::ptr_eq(owed_to, conn));
    if behind || conn.try_deliver(frame.clone()).is_err() {
        owed.push((Arc::clone(conn), frame));
    }
}

/// The feed's tick thread body.
pub(crate) fn tick_loop(hub: Arc<TakeHub>, shared: Arc<crate::server::HostShared>) {
    while !shared.shutdown.load(Ordering::SeqCst) {
        hub.tick();
        std::thread::sleep(TICK);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn drained(inbound: &starling_runtime::channel::Receiver<Frame>) -> Vec<Frame> {
        std::iter::from_fn(|| inbound.try_recv().ok()).collect()
    }

    /// A window that is not the one to act on a transcription still hears
    /// how it ended when its queue was full at the time: it showed the take
    /// busy since the start, and nothing else would clear that.
    #[test]
    fn a_watcher_with_a_full_queue_still_hears_the_transcription_end() {
        let hub = TakeHub::new(Duration::from_secs(60));
        let (acting, _acting_inbound) = ConnState::for_test(16);
        let (other, other_inbound) = ConnState::for_test(2);
        hub.watch(&acting, "w1".into()).unwrap();
        hub.watch(&other, "w2".into()).unwrap();
        // The other window's queue is full.
        while other
            .try_deliver(Frame::GetSnapshot { req: "fill".into() })
            .is_ok()
        {}
        hub.transcription(
            "stored",
            None,
            None,
            None,
            TranscriptionState::Completed {
                text: "words".into(),
                kept_earlier: false,
            },
            Some(&acting),
        );
        drained(&other_inbound);
        hub.tick();
        let frames = drained(&other_inbound);
        assert!(
            frames.iter().any(|frame| matches!(
                frame,
                Frame::Transcription {
                    yours: false,
                    state: TranscriptionState::Completed { .. },
                    ..
                }
            )),
            "{frames:?}"
        );
    }
}
