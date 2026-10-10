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
//! **orphan**: one watching connection — the first, or the next to watch
//! — receives its [`Frame::TakePersisted`] with `orphan: true`, and
//! transcribes it. An app that reconnects while the take still records
//! sees its status ticks (`owner: nobody`) and adopts it.
//!
//! A stored take an app has to transcribe stays the host's until that
//! app says it handled it ([`Frame::TakeHandled`]): its owner's, or the
//! orphan's one app. If that app goes away first, the take is handed to
//! the next one as an orphan. Until then its id is kept in
//! [`UNCLAIMED_FILE`], so neither the host's idle exit nor a crash of
//! either side forgets it.
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
use crate::frame::{Frame, HostRecovery, TakeAudio, TakeOwner};
use crate::server::{lock_registry, ConnState};

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
    orphan_grace: Duration,
    /// Where pending takes are kept across host restarts (the data
    /// root's [`UNCLAIMED_FILE`]); `None` keeps them in memory only.
    unclaimed_file: Option<std::path::PathBuf>,
    /// The [`HubState::pending_gen`] last written to `unclaimed_file`, and
    /// whether the last write failed; the lock serializes the writes,
    /// which run outside the hub's lock.
    saved_gen: Mutex<(u64, bool)>,
}

/// The pending-takes file in the data root: the stored ids of takes an
/// app still has to transcribe, until one says it did — a host that
/// exits idle (or is killed) before any app handled them must not forget
/// them.
pub const UNCLAIMED_FILE: &str = "unclaimed-takes.json";

#[derive(Default)]
struct HubState {
    /// The connection that sent each recent `capture.start`, by corr.
    starters: VecDeque<(String, Arc<ConnState>)>,
    live: Option<Live>,
    ended: VecDeque<Ended>,
    watchers: Vec<Watcher>,
    /// Stored takes an app has to hear about, until it did (and, for one
    /// it has to transcribe, until it said it handled it).
    pending: Vec<Pending>,
    /// The durable part of `pending` changed since it was last written.
    pending_dirty: bool,
    pending_gen: u64,
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
}

/// A stored take an app has to hear about.
struct Pending {
    /// A [`Frame::TakePersisted`].
    frame: Frame,
    /// The connection it is for: its owner, or the app an orphan was
    /// handed to. `None` until an app watches.
    offered: Option<Arc<ConnState>>,
    /// The frame went out to `offered` (or waits behind its tap; an
    /// owner that does not watch is sent nothing — it holds the take
    /// until it goes away).
    sent: bool,
    /// Apps that handed it back (they cannot transcribe it): it goes to
    /// another.
    declined: Vec<Arc<ConnState>>,
}

impl Pending {
    fn take(&self) -> &str {
        match &self.frame {
            Frame::TakePersisted { take, .. } => take,
            _ => "",
        }
    }

    /// The take the app has to transcribe: kept until it says it handled
    /// it, and durably. Anything else is done once it went out.
    fn stored_id(&self) -> Option<&str> {
        handling_id(&self.frame)
    }

    fn pending(frame: Frame, offered: Option<Arc<ConnState>>, sent: bool) -> Pending {
        Pending {
            frame,
            offered,
            sent,
            declined: Vec::new(),
        }
    }

    /// Hands the take to whichever app comes next, as an orphan.
    fn reoffer(&mut self) {
        self.offered = None;
        self.sent = false;
        if let Frame::TakePersisted { orphan, .. } = &mut self.frame {
            *orphan = true;
        }
    }
}

struct Live {
    take: String,
    rate: u32,
    monitor: Option<Arc<dyn LiveTakeMonitor>>,
    owner: Option<Arc<ConnState>>,
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
    pub fn new(orphan_grace: Duration, unclaimed_file: Option<std::path::PathBuf>) -> Arc<TakeHub> {
        let mut state = HubState::default();
        if let Some(path) = &unclaimed_file {
            state.pending = load_unclaimed(path);
        }
        Arc::new(TakeHub {
            state: Mutex::new(state),
            client: OnceLock::new(),
            orphan_grace,
            unclaimed_file,
            saved_gen: Mutex::new((0, false)),
        })
    }

    /// Adds a stored take for an app and hands it out if it can.
    fn add_pending(state: &mut HubState, pending: Pending) {
        state.pending_dirty |= pending.stored_id().is_some();
        state.pending.push(pending);
        offer_pending(state);
    }

    /// Writes the pending takes' ids if they changed — outside the hub's
    /// lock, so a slow disk never holds up the capture actor's callbacks.
    fn save_pending(&self) {
        let Some(path) = &self.unclaimed_file else {
            return;
        };
        let (ids, generation) = {
            let mut state = lock_registry(&self.state);
            if !state.pending_dirty {
                return;
            }
            state.pending_dirty = false;
            state.pending_gen += 1;
            let ids: Vec<String> = state
                .pending
                .iter()
                .filter_map(|pending| pending.stored_id().map(str::to_string))
                .collect();
            (ids, state.pending_gen)
        };
        let mut saved = lock_registry(&self.saved_gen);
        if saved.0 >= generation {
            // A newer snapshot is on disk already.
            return;
        }
        let result = if ids.is_empty() {
            match std::fs::remove_file(path) {
                Err(err) if err.kind() != std::io::ErrorKind::NotFound => Err(err),
                _ => Ok(()),
            }
        } else {
            let tmp = path.with_extension("json.tmp");
            serde_json::to_vec(&ids)
                .map_err(std::io::Error::other)
                .and_then(|bytes| std::fs::write(&tmp, bytes))
                .and_then(|()| std::fs::rename(&tmp, path))
        };
        match result {
            Ok(()) => {
                if saved.1 {
                    eprintln!("starling-runtime-host: recorded pending takes again");
                }
                *saved = (generation, false);
            }
            Err(err) => {
                // Tried again on the next tick (said once, not per tick).
                if !saved.1 {
                    eprintln!(
                        "starling-runtime-host: could not record pending takes in {}: {err}; \
                         retrying",
                        path.display()
                    );
                }
                saved.1 = true;
                drop(saved);
                lock_registry(&self.state).pending_dirty = true;
            }
        }
    }

    /// The runtime the hub stops orphaned takes through (set once the
    /// runtime started; the hub exists before it, as its observer).
    pub(crate) fn attach(&self, client: RuntimeClient) {
        let _ = self.client.set(client);
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
        // The takes nobody has go to an app now.
        offer_pending(&mut state);
        state.orphaned_since = None;
        Ok(())
    }

    /// An app handled the stored take `stored_id`: it is no longer the
    /// host's to offer. (Whichever connection says so — an app that
    /// reconnected answers for the take on its new one.)
    pub(crate) fn handled(&self, stored_id: &str) {
        {
            let mut state = lock_registry(&self.state);
            let before = state.pending.len();
            state
                .pending
                .retain(|pending| pending.stored_id() != Some(stored_id));
            if state.pending.len() == before {
                return;
            }
            state.pending_dirty = true;
        }
        self.save_pending();
    }

    /// `conn` cannot transcribe the stored take `stored_id`: it goes to
    /// another app (or waits for one).
    pub(crate) fn handed_back(&self, conn: &Arc<ConnState>, stored_id: &str) {
        let mut state = lock_registry(&self.state);
        for pending in state.pending.iter_mut().filter(|pending| {
            pending.stored_id() == Some(stored_id)
                && pending
                    .offered
                    .as_ref()
                    .is_some_and(|offered| Arc::ptr_eq(offered, conn))
        }) {
            pending.reoffer();
            pending.declined.push(Arc::clone(conn));
        }
        offer_pending(&mut state);
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
            // A watcher that went away takes nothing with it: what it held
            // behind its taps is pending until an app handles it, and a
            // pending take whose app is gone goes to the next one.
            state
                .watchers
                .retain(|watcher| !watcher.conn.closed.load(Ordering::SeqCst));
            offer_pending(&mut state);
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
        self.save_pending();
    }
}

/// The stored id of a [`Frame::TakePersisted`] an app has to transcribe
/// (stored complete), if it is one.
fn handling_id(frame: &Frame) -> Option<&str> {
    match frame {
        Frame::TakePersisted {
            stored_id: Some(id),
            interrupted: false,
            error: None,
            ..
        } => Some(id),
        _ => None,
    }
}

/// Hands every pending take to its app: one whose app went away (or that
/// has none yet) to the first watcher, as an orphan; one not sent yet
/// (its app's queue was full) again. A take only to be heard about is
/// done once it went out.
fn offer_pending(state: &mut HubState) {
    let HubState {
        pending, watchers, ..
    } = state;
    pending.retain_mut(|pending| {
        if pending
            .offered
            .as_ref()
            .is_some_and(|conn| conn.closed.load(Ordering::SeqCst))
        {
            pending.reoffer();
        }
        pending
            .declined
            .retain(|conn| !conn.closed.load(Ordering::SeqCst));
        if pending.offered.is_none() {
            pending.offered = watchers
                .iter()
                .map(|watcher| &watcher.conn)
                .find(|conn| !pending.declined.iter().any(|declined| Arc::ptr_eq(declined, conn)))
                .cloned();
        }
        if let (Some(conn), false) = (pending.offered.as_ref(), pending.sent) {
            let take = pending.take().to_string();
            pending.sent = match watchers
                .iter_mut()
                .find(|watcher| Arc::ptr_eq(&watcher.conn, conn))
            {
                Some(watcher) => watcher.deliver_after_tap(&take, pending.frame.clone()),
                None => conn.try_deliver(pending.frame.clone()).is_ok(),
            };
        }
        !(pending.sent && pending.stored_id().is_none())
    });
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
            _ => {
                if has_room(&conn) {
                    let _ = conn.try_deliver(Frame::LiveTake {
                        take: live.take.clone(),
                        rate: live.rate,
                        status: status.cloned(),
                        audio: None,
                        owner,
                        ended: None,
                        kept: false,
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
            monitor,
            owner,
        });
        state.orphaned_since = None;
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
        if state.acquiring.as_ref().is_some_and(|(take, _)| take == corr) {
            state.acquiring = None;
        }
        for watcher in &state.watchers {
            let _ = watcher.conn.try_deliver(Frame::TakeStartFailed {
                take: corr.to_string(),
                problem: problem.clone(),
                message: message.clone(),
            });
        }
    }

    fn take_ended(&self, corr: &str, record: Option<&Arc<TakeRecord>>) {
        if is_agent_take(corr) {
            return;
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
        for watcher in &state.watchers {
            if watcher.tapping(corr) {
                continue;
            }
            let _ = watcher.conn.try_deliver(Frame::LiveTake {
                take: corr.to_string(),
                rate,
                status: None,
                audio: None,
                owner: owner_for(&owner, &watcher.conn),
                ended: Some(total),
                kept: record.is_some(),
            });
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
            let frame = Frame::TakePersisted {
                take: corr.to_string(),
                stored_id,
                interrupted: record.status == TakeStatus::Interrupted,
                error,
                orphan,
            };
            // The connection the take is for: its owner while it lives,
            // else (an orphan) exactly one app — the first watching one,
            // or the next to watch. An owner that only sends commands
            // (no app) is sent nothing; once it goes away an app gets
            // the take as an orphan, unless it said it handled it.
            let owner_watches = owner.as_ref().is_some_and(|owner| {
                state
                    .watchers
                    .iter()
                    .any(|watcher| Arc::ptr_eq(&watcher.conn, owner))
            });
            if !orphan {
                // Every other window hears the take is in history (a
                // full queue only costs that window a history refresh).
                for watcher in state.watchers.iter_mut() {
                    let theirs = owner
                        .as_ref()
                        .is_some_and(|owner| Arc::ptr_eq(&watcher.conn, owner));
                    if !theirs {
                        let _ = watcher.deliver_after_tap(corr, frame.clone());
                    }
                }
            }
            let held_by_owner = !orphan && !owner_watches;
            if orphan || owner_watches || handling_id(&frame).is_some() {
                Self::add_pending(
                    &mut state,
                    Pending::pending(frame, owner.filter(|_| !orphan), held_by_owner),
                );
            }
        }
        self.save_pending();
    }
}

/// The pending takes a previous host left (see [`UNCLAIMED_FILE`]).
fn load_unclaimed(path: &std::path::Path) -> Vec<Pending> {
    let Ok(bytes) = std::fs::read(path) else {
        return Vec::new();
    };
    match serde_json::from_slice::<Vec<String>>(&bytes) {
        Ok(ids) => ids
            .into_iter()
            .map(|id| {
                Pending::pending(
                    Frame::TakePersisted {
                        take: id.clone(),
                        stored_id: Some(id),
                        interrupted: false,
                        error: None,
                        orphan: true,
                    },
                    None,
                    false,
                )
            })
            .collect(),
        Err(err) => {
            eprintln!(
                "starling-runtime-host: ignoring unreadable {}: {err}",
                path.display()
            );
            Vec::new()
        }
    }
}

/// The feed's tick thread body.
pub(crate) fn tick_loop(hub: Arc<TakeHub>, shared: Arc<crate::server::HostShared>) {
    while !shared.shutdown.load(Ordering::SeqCst) {
        hub.tick();
        std::thread::sleep(TICK);
    }
}
