//! Takes recorded by the runtime host, app side (#220).
//!
//! The app asks the host to start, stop or cancel a take and follows it
//! through the host's take feed ([`crate::host_link`]). A take the app
//! stopped or cancelled is *finishing* until the host says it is stored:
//! a stopped take is then transcribed from the stored audio (the live
//! stream finishes on the tail the feed delivered), a cancelled one is
//! announced as kept in history.
//!
//! A take outlives this window. When the connection drops mid-take the
//! window lets go of it (the host keeps recording); after reconnecting —
//! or after the app restarted — a take the host still records with no
//! live owner is adopted: it becomes the active take, latched, its live
//! text replayed from its start, and it stops like any other. A take the
//! host stopped and stored while no app followed it comes back as an
//! orphan and is transcribed into history (nothing is typed: the window
//! it was meant for is gone).
//!
//! A stored take this window has to transcribe (its own, or an orphan)
//! is acknowledged to the host only once its transcription finished
//! ([`StarlingApp::take_handled`]); a window that goes away before that
//! leaves it to the next one.

use std::collections::{HashSet, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

use gpui::{AppContext, Context};
use starling_dictation::audio;
use starling_runtime::protocol::Command;
use starling_runtime_host::client::HostClient;
use starling_runtime_host::frame::TakeOwner;

use crate::activation::{CancelReason, TakeId};
use crate::app::StarlingApp;
use crate::host_link::{HostLink, HostUpdate, LiveCapture, TakeFeed, TakeUpdate};
use crate::live_stream::LiveStream;
use crate::stream_pump::Handoff;
use crate::store::Recognition;
use crate::upload::{TakeTarget, refresh_sessions};

/// How often a stop or cancel the host has not acted on is asked again.
const REASK: Duration = Duration::from_secs(1);

/// How many acknowledged stored takes are remembered, to answer a take
/// the host offers again (its ack crossed a reconnect) without
/// transcribing it twice.
const HANDLED_KEEP: usize = 64;

/// How often another window's stored take is looked up again while that
/// window transcribes it (the host tells only the transcribing window
/// when it is done): first, and the cap it doubles up to.
pub(crate) const FOREIGN_POLL: Duration = Duration::from_secs(2);
const FOREIGN_POLL_MAX: Duration = Duration::from_secs(30);

/// The app's side of the host connection.
#[derive(Default)]
pub(crate) struct HostState {
    pub(crate) link: Option<HostLink>,
    pub(crate) client: Option<Arc<HostClient>>,
    /// Why the app is not connected, while it is not.
    pub(crate) down: Option<String>,
    /// The link stopped starting the recording service (it kept failing)
    /// until the user asks again.
    pub(crate) gave_up: bool,
    /// Stored takes this window transcribes for the host, acknowledged
    /// once their transcription finished.
    pub(crate) handling: HashSet<String>,
    /// Stored takes acknowledged lately.
    pub(crate) handled: VecDeque<String>,
    /// Takes this window stopped or cancelled, until they are stored.
    pub(crate) finishing: Vec<FinishingTake>,
    /// The live take's stream waits for the take's device rate, which the
    /// host's first tick brings: the endpoint it will open.
    pub(crate) stream_endpoint: Option<String>,
    /// A running take with no live owner this window asked for (by
    /// tapping it), and when: it is adopted only once the host says it is
    /// this window's — of two windows asking, only one gets it.
    pub(crate) claiming: Option<(String, Instant)>,
}

/// A take this window stopped or cancelled, until the host stored it.
pub(crate) struct FinishingTake {
    pub(crate) take: String,
    pub(crate) activation: TakeId,
    pub(crate) feed: Arc<TakeFeed>,
    /// What the live stream drained and sent before the stop.
    pub(crate) handoff: Handoff<LiveStream>,
    pub(crate) kind: FinishKind,
    /// The take's final sample count and whether it kept anything.
    pub(crate) ended: Option<(u64, bool)>,
    asked_at: Instant,
}

pub(crate) enum FinishKind {
    /// Stopped: transcribe it once stored.
    Transcribe {
        target: TakeTarget,
        stopped_at: Instant,
        staging: Option<u64>,
        delivery: Option<crate::delivery::Capture>,
    },
    /// Cancelled: kept in history untranscribed; `saved_notice` is shown
    /// once it is there, `empty_notice` when it kept nothing.
    Cancel {
        saved_notice: Option<String>,
        empty_notice: Option<String>,
        staging: Option<u64>,
    },
}

impl FinishingTake {
    pub(crate) fn new(
        take: String,
        activation: TakeId,
        feed: Arc<TakeFeed>,
        handoff: Handoff<LiveStream>,
        kind: FinishKind,
    ) -> FinishingTake {
        FinishingTake {
            take,
            activation,
            feed,
            handoff,
            kind,
            ended: None,
            asked_at: Instant::now(),
        }
    }

    fn command(&self) -> Command {
        match self.kind {
            FinishKind::Transcribe { .. } => Command::CaptureStop { drain: Some(true) },
            FinishKind::Cancel { .. } => Command::CaptureAbort,
        }
    }

}

/// The whole take as a window holds it: what the stream drained, then
/// the rest of the feed. A copy of the take, so built off the UI thread.
fn held_samples(drained: Vec<f32>, feed: &TakeFeed) -> Vec<f32> {
    let mut samples = drained;
    samples.extend(feed.samples_from(samples.len()));
    samples
}

impl StarlingApp {
    /// Connects to the runtime host (starting it when nothing serves) and
    /// follows it for the app's lifetime.
    pub(crate) fn start_host_link(&mut self, cx: &mut Context<Self>) {
        let endpoint = match crate::host_link::default_endpoint() {
            Ok(endpoint) => endpoint,
            Err(err) => {
                self.host.down = Some(err.clone());
                self.error = Some(format!(
                    "Starling's recording service has no place to run ({err}); recording is \
                     unavailable."
                ));
                return;
            }
        };
        let launch = match crate::host_link::host_log_path() {
            Some(log) => crate::host_link::Launch::SelfAsHost { log },
            None => crate::host_link::Launch::Never,
        };
        self.follow_host(endpoint, launch, cx);
    }

    /// Follows the host at `endpoint` (tests point this at their own).
    pub(crate) fn follow_host(
        &mut self,
        endpoint: std::path::PathBuf,
        launch: crate::host_link::Launch,
        cx: &mut Context<Self>,
    ) {
        let (link, mut updates) = HostLink::start(endpoint, launch);
        self.host.link = Some(link);
        self.host.down = Some("connecting".to_string());
        cx.spawn(async move |this, cx| {
            while let Some(update) = updates.recv().await {
                if this
                    .update(cx, |app, cx| app.host_update(update, cx))
                    .is_err()
                {
                    break;
                }
            }
        })
        .detach();
    }

    /// Why recording is unavailable right now, if it is.
    pub(crate) fn host_unavailable(&self) -> Option<String> {
        match (&self.host.client, &self.host.down) {
            (Some(_), _) => None,
            (None, Some(reason)) if self.host.gave_up => Some(format!(
                "Starling's recording service is unavailable ({reason}). Press record to try \
                 starting it again."
            )),
            (None, Some(reason)) if reason == "connecting" => {
                Some("Starling's recording service is starting.".to_string())
            }
            (None, Some(reason)) => Some(format!(
                "Starling's recording service is not reachable ({reason}); reconnecting."
            )),
            (None, None) => Some("Starling's recording service is starting.".to_string()),
        }
    }

    /// Sends `command` for `take`, in order with every earlier one.
    pub(crate) fn host_command(&self, take: &str, command: Command) {
        if let Some(link) = &self.host.link {
            link.command(take, command);
        }
    }

    pub(crate) fn host_update(&mut self, update: HostUpdate, cx: &mut Context<Self>) {
        match update {
            HostUpdate::Connected { client, recovery } => {
                self.host.client = Some(client);
                self.host.down = None;
                self.host.gave_up = false;
                if let Some(recovery) = recovery {
                    crate::upload::add_recovery_messages(self, recovery.problems, recovery.notice);
                }
                self.refresh_history(cx);
            }
            HostUpdate::Disconnected { reason, gave_up } => {
                self.host.client = None;
                self.host.down = Some(reason);
                self.host.gave_up = gave_up;
                self.host.claiming = None;
                self.let_go_of_takes(cx);
            }
            HostUpdate::Take(update) => self.take_update(update, cx),
            HostUpdate::Event(event) => {
                let code = event.payload().get("code").and_then(|code| code.as_str());
                let ours = event
                    .corr()
                    .is_some_and(|corr| self.host.finishing.iter().any(|take| take.take == corr));
                if ours && code == Some("quiesce_timeout") {
                    self.error = Some(
                        "The microphone did not stop cleanly within the quiesce timeout; the \
                         audio it captured was kept and saved to your history as an \
                         interrupted recording."
                            .to_string(),
                    );
                }
            }
            HostUpdate::Refused {
                take,
                command,
                reason,
            } => {
                if command == "capture.start" {
                    self.take_start_failed(&take, None, format!(
                        "The recording could not start: {reason}"
                    ), cx);
                }
                if command == "take.tap" {
                    // The live take's audio did not get asked for: ask on
                    // the next tick (a finishing take needs no tap — its
                    // end and stored row come either way).
                    if let Some(live) = self.recorder.as_mut().filter(|live| live.take == take) {
                        live.tapped = false;
                    }
                }
                // A stop or cancel that was refused is asked again while
                // the take still records (see `take_update`).
            }
        }
        cx.notify();
    }

    fn refresh_history(&mut self, cx: &mut Context<Self>) {
        if let Some(store) = self.store.clone() {
            cx.spawn(async move |this, cx| refresh_sessions(&this, &store, cx).await)
                .detach();
        }
    }

    /// The connection dropped: the host keeps any take this window had.
    /// The live view lets go of it (it is adopted again on reconnect);
    /// finishing takes come back as orphans once stored.
    fn let_go_of_takes(&mut self, cx: &mut Context<Self>) {
        if let Some(live) = self.recorder.take() {
            if let Some(link) = &self.host.link {
                link.forget(&live.take);
            }
            self.end_live_take_locally(cx);
            self.error = Some(
                "Lost the connection to Starling's recording service. The recording continues \
                 there and comes back here once the connection does."
                    .to_string(),
            );
        }
        for finishing in std::mem::take(&mut self.host.finishing) {
            if let Some(link) = &self.host.link {
                link.forget(&finishing.take);
            }
            match finishing.kind {
                FinishKind::Transcribe {
                    staging, stopped_at, ..
                } => {
                    if let Some(token) = staging {
                        self.staging_save_failed(token, cx);
                    }
                    self.overlay.model.save_failed(stopped_at, Instant::now());
                }
                FinishKind::Cancel { staging, .. } => {
                    if let Some(token) = staging {
                        self.staging_save_failed(token, cx);
                    }
                }
            }
        }
    }

    /// A tick for the take this window asked for.
    fn claim_update(
        &mut self,
        take: String,
        rate: u32,
        status: Option<starling_runtime::machine::capture::LiveTakeStatus>,
        owner: TakeOwner,
        ended: bool,
        cx: &mut Context<Self>,
    ) {
        let forget = |app: &mut StarlingApp, take: &str| {
            app.host.claiming = None;
            if let Some(link) = &app.host.link {
                link.forget(take);
            }
        };
        if owner == TakeOwner::Another || (ended && owner == TakeOwner::Nobody) {
            // Another window got it first, or it ended unowned (the host
            // hands it over as an orphan once stored).
            forget(self, &take);
            return;
        }
        if owner == TakeOwner::Nobody {
            // The tap has not landed yet (or was lost to a reconnect):
            // ask again now and then.
            if let Some((_, asked_at)) = self.host.claiming.as_mut() {
                if asked_at.elapsed() >= REASK {
                    *asked_at = Instant::now();
                    if let Some(link) = &self.host.link {
                        link.tap(&take, 0);
                    }
                }
            }
            return;
        }
        self.host.claiming = None;
        if !ended
            && self.recorder.is_none()
            && !self.activation.is_active()
            && self.adopt_take(take.clone(), rate, status, cx)
        {
            return;
        }
        // It is ours but cannot be shown live — it already ended, or this
        // window started a take of its own meanwhile: finish it into
        // history.
        self.finish_unshown_take(take, !ended);
    }

    /// A take this window owns but does not show (adopting it lost a race
    /// with a take of the window's own, or a tap from before a reconnect
    /// landed on the new connection): stopped (`stop`: unless it ended
    /// already) and transcribed into history — its end and stored row
    /// follow — so it never records on with nobody able to stop it, and
    /// its stored row is never left with a window that ignores it.
    fn finish_unshown_take(&mut self, take: String, stop: bool) {
        let Some(link) = &self.host.link else {
            return;
        };
        let feed = link.feed(&take);
        let target = self.resolve_take_target();
        let finishing = FinishingTake::new(
            take.clone(),
            self.activation.last_started(),
            feed,
            Handoff::default(),
            FinishKind::Transcribe {
                target,
                stopped_at: Instant::now(),
                staging: None,
                delivery: None,
            },
        );
        if stop {
            self.host_command(&take, finishing.command());
        }
        self.host.finishing.push(finishing);
    }

    /// Another window holds the take this window adopted: it is theirs.
    fn lose_live_take(&mut self, cx: &mut Context<Self>) {
        if let Some(live) = self.recorder.take() {
            if let Some(link) = &self.host.link {
                link.forget(&live.take);
            }
            self.end_live_take_locally(cx);
            self.service_notice = Some(
                "Another Starling window picked up the running recording.".to_string(),
            );
        }
    }

    /// Everything the window did for its live take, undone without a
    /// stop: the take itself goes on in the host.
    fn end_live_take_locally(&mut self, cx: &mut Context<Self>) {
        if let Some(lease) = self.playback_lease.take() {
            self.playback.handle().end(lease);
        }
        self.audio_upkeep.set_recording(false);
        let _ = self.finish_stream_pump();
        self.host.stream_endpoint = None;
        self.active_take = None;
        self.live_partial.clear();
        self.levels = vec![0.06; 52];
        self.delivery_take_stopped();
        self.staging_interrupted(cx);
        if let Some(take) = self.recording_take.take() {
            self.activation.ended(take);
            self.overlay.model.take_cancelled(Instant::now());
        }
        self.activation_settled(cx);
    }

    fn take_update(&mut self, update: TakeUpdate, cx: &mut Context<Self>) {
        match update {
            TakeUpdate::Live {
                take,
                rate,
                status,
                owner,
                ended,
                kept,
            } => {
                if self.recorder.as_ref().is_some_and(|live| live.take == take) {
                    if owner == TakeOwner::Another {
                        // Another window adopted it first.
                        self.lose_live_take(cx);
                        return;
                    }
                    self.live_tick(&take, rate, status.clone(), cx);
                    if ended.is_some() {
                        // The take ended without this window stopping it:
                        // its microphone failed (the host kept what it
                        // captured) or something else stopped it.
                        self.live_take_ended_elsewhere(cx);
                    }
                }
                if self
                    .host
                    .claiming
                    .as_ref()
                    .is_some_and(|(claimed, _)| *claimed == take)
                {
                    self.claim_update(take.clone(), rate, status.clone(), owner, ended.is_some(), cx);
                    // A claimed take that had already ended is finishing
                    // now: this same frame's end belongs to it.
                    if self.finishing_index(&take).is_none() {
                        return;
                    }
                }
                if owner == TakeOwner::You
                    && (ended.is_none() || kept)
                    && self.finishing_index(&take).is_none()
                    && !self.recorder.as_ref().is_some_and(|live| live.take == take)
                {
                    // Ours, but nothing here follows it: finishing now
                    // (this frame's end, if it has one, belongs to it).
                    // A claim on this take was settled above; one on
                    // another take does not hold this one up.
                    self.finish_unshown_take(take.clone(), ended.is_none());
                }
                if let Some(index) = self.finishing_index(&take) {
                    match ended {
                        Some(total) => {
                            self.host.finishing[index].ended = Some((total, kept));
                            self.finishing_progress(&take, cx);
                        }
                        None => self.reask(index),
                    }
                    return;
                }
                if ended.is_none()
                    && owner == TakeOwner::Nobody
                    && self.recorder.is_none()
                    && self.host.claiming.is_none()
                    && !self.activation.is_active()
                {
                    // Ask for it; adopt it once it is ours.
                    if let Some(link) = &self.host.link {
                        link.feed(&take);
                        link.tap(&take, 0);
                        self.host.claiming = Some((take, Instant::now()));
                    }
                }
            }
            TakeUpdate::StartFailed {
                take,
                problem,
                message,
            } => self.take_start_failed(&take, problem, message, cx),
            TakeUpdate::Persisted {
                take,
                stored_id,
                interrupted,
                error,
                orphan,
            } => {
                if self.finishing_index(&take).is_some() {
                    self.take_stored(&take, stored_id, interrupted, error, cx);
                } else if orphan {
                    match (stored_id, interrupted || error.is_some()) {
                        (Some(id), false) if self.host.handled.contains(&id) => {
                            // Handled here already; the ack crossed a
                            // reconnect. Say so again.
                            if let Some(link) = &self.host.link {
                                link.handled(&id);
                            }
                        }
                        // Being transcribed here already: its job acks it.
                        (Some(id), false) if self.host.handling.contains(&id) => {}
                        (Some(id), false) => self.transcribe_orphan(id, cx),
                        _ => self.refresh_history(cx),
                    }
                } else {
                    // Another window's take: it is in history now, and
                    // its transcript once that window has it.
                    self.refresh_history(cx);
                    if let Some(id) = stored_id {
                        self.follow_foreign_take(id, cx);
                    }
                }
            }
            TakeUpdate::Notice(recovery) => {
                crate::upload::add_recovery_messages(self, recovery.problems, recovery.notice);
                self.refresh_history(cx);
            }
        }
    }

    /// Reads history again once another window settled stored take `id`
    /// (transcribed, failed or deleted), so it does not read "Sending to
    /// server…" here forever.
    fn follow_foreign_take(&mut self, id: String, cx: &mut Context<Self>) {
        let Some(store) = self.store.clone() else {
            return;
        };
        cx.spawn(async move |this, cx| {
            // As long as it is pending and this window is open: a job may
            // take many minutes (a stream's wait, then a long batch).
            let mut every = FOREIGN_POLL;
            loop {
                cx.background_executor().timer(every).await;
                every = (every * 2).min(FOREIGN_POLL_MAX);
                let pending = {
                    let store = store.clone();
                    let id = id.clone();
                    cx.background_spawn(async move { store.pending(&id) }).await
                };
                // A failed read stops looking; the next stored take
                // refreshes history anyway.
                if pending.unwrap_or(false) && this.upgrade().is_some() {
                    continue;
                }
                refresh_sessions(&this, &store, cx).await;
                return;
            }
        })
        .detach();
    }

    fn finishing_index(&self, take: &str) -> Option<usize> {
        self.host.finishing.iter().position(|finishing| finishing.take == take)
    }

    /// The transcription of stored take `id` finished (or failed and
    /// said so): if the host handed it to this window, it hears so now.
    pub(crate) fn take_handled(&mut self, id: &str) {
        if !self.host.handling.remove(id) {
            return;
        }
        if let Some(link) = &self.host.link {
            link.handled(id);
        }
        self.host.handled.push_back(id.to_string());
        while self.host.handled.len() > HANDLED_KEEP {
            self.host.handled.pop_front();
        }
    }

    /// This window cannot transcribe stored take `id` (no store, its
    /// audio unreadable here): the host hands it to another window — or
    /// to this one again after a reconnect.
    pub(crate) fn take_handed_back(&mut self, id: &str) {
        if !self.host.handling.remove(id) {
            return;
        }
        if let Some(link) = &self.host.link {
            link.handed_back(id);
        }
    }

    /// A stop or cancel the take still records past: ask again.
    fn reask(&mut self, index: usize) {
        let finishing = &mut self.host.finishing[index];
        if finishing.asked_at.elapsed() >= REASK {
            finishing.asked_at = Instant::now();
            let (take, command) = (finishing.take.clone(), finishing.command());
            self.host_command(&take, command);
        }
    }

    /// A status tick for the window's live take.
    fn live_tick(
        &mut self,
        take: &str,
        rate: u32,
        status: Option<starling_runtime::machine::capture::LiveTakeStatus>,
        cx: &mut Context<Self>,
    ) {
        let Some(live) = self.recorder.as_mut() else {
            return;
        };
        let first = !live.confirmed;
        if let Some(status) = status {
            if let Some(route) = status.route.clone() {
                self.mic.last_route = Some(route);
            }
            live.status = Some(status);
        }
        // Listening is announced as soon as the service reports audio.
        self.check_readiness(cx);
        let Some(live) = self.recorder.as_mut() else {
            return;
        };
        // The take's audio, from the start (or, after a tap that did not
        // go out, from what is here).
        if !live.tapped {
            live.tapped = true;
            if let Some(link) = &self.host.link {
                link.tap(take, live.feed.len() as u64);
            }
        }
        if first {
            live.confirmed = true;
            if let Some(endpoint) = self.host.stream_endpoint.take() {
                let feed = Arc::clone(&live.feed);
                if let Err(reason) = self.start_stream_pump(feed, rate, &endpoint, cx) {
                    self.stream_degradation = Some(format!(
                        "Live transcription is unavailable ({reason}); the recording will be \
                         uploaded in full after you stop."
                    ));
                }
            }
        }
    }

    /// The live take ended in the host without this window asking.
    fn live_take_ended_elsewhere(&mut self, cx: &mut Context<Self>) {
        let Some(take) = self.recording_take else {
            return;
        };
        let healthy = self
            .recorder
            .as_ref()
            .is_some_and(|live| !live.capture_fault().is_some_and(|fault| fault.is_fatal()));
        if healthy {
            // Stopped cleanly elsewhere (a stop an earlier window of this
            // app sent before it went away): finish it like any stop.
            self.recording_take = None;
            self.activation.ended(take);
            self.stop_recording(take, cx);
            if !self.overlay.model.is_saving() {
                self.overlay.model.take_cancelled(Instant::now());
            }
            self.cue_take_ended(take, cx);
            self.activation_settled(cx);
            return;
        }
        if !self.note_live_interruption() {
            let device = self
                .recorder
                .as_ref()
                .map(crate::mic::device_name)
                .unwrap_or_else(|| "The microphone".to_string());
            self.note_interruption(&device, crate::mic::Interruption::Ended);
        }
        self.activation_input(|machine| machine.input_lost(take), cx);
    }

    /// The host could not start `take`.
    pub(crate) fn take_start_failed(
        &mut self,
        take: &str,
        problem: Option<starling_dictation::microphone::InputProblem>,
        message: String,
        cx: &mut Context<Self>,
    ) {
        if let Some(index) = self.finishing_index(take) {
            // Stopped or cancelled before the host said it records: no
            // tick, end or stored row will ever come for it.
            self.finishing_never_started(index, problem, message, cx);
            return;
        }
        if !self.recorder.as_ref().is_some_and(|live| live.take == take) {
            return;
        }
        self.recorder = None;
        if let Some(link) = &self.host.link {
            link.forget(take);
        }
        if let Some(lease) = self.playback_lease.take() {
            self.playback.handle().end(lease);
        }
        self.audio_upkeep.set_recording(false);
        let _ = self.finish_stream_pump();
        self.host.stream_endpoint = None;
        self.active_take = None;
        self.delivery_take_stopped();
        self.retire_staging(cx);
        if let Some(take) = self.recording_take.take() {
            self.activation.start_failed(take);
            self.overlay.model.take_cancelled(Instant::now());
            self.cue_take_ended(take, cx);
        }
        match problem {
            Some(problem) => self.report_input_problem(problem, message),
            None => self.error = Some(message),
        }
        self.activation_settled(cx);
    }

    /// A take this window stopped or cancelled before the host confirmed
    /// it, which never started: resolved here.
    fn finishing_never_started(
        &mut self,
        index: usize,
        problem: Option<starling_dictation::microphone::InputProblem>,
        message: String,
        cx: &mut Context<Self>,
    ) {
        let finishing = self.host.finishing.remove(index);
        if let Some(link) = &self.host.link {
            link.forget(&finishing.take);
        }
        match finishing.kind {
            FinishKind::Transcribe {
                staging, stopped_at, ..
            } => {
                if let Some(token) = staging {
                    self.staging_save_failed(token, cx);
                }
                self.overlay.model.save_failed(stopped_at, Instant::now());
                match problem {
                    Some(problem) => self.report_input_problem(problem, message),
                    None => self.error = Some(message),
                }
            }
            FinishKind::Cancel {
                staging,
                empty_notice,
                ..
            } => {
                if let Some(token) = staging {
                    self.staging_save_failed(token, cx);
                }
                if let Some(notice) = empty_notice {
                    if self.activation.last_started() == finishing.activation {
                        self.take_notice = Some(notice);
                    }
                }
            }
        }
    }

    /// Makes a take the host records — with no live owner — this
    /// window's active take; `false` when the window cannot take it on
    /// (its activation is busy).
    fn adopt_take(
        &mut self,
        take: String,
        rate: u32,
        status: Option<starling_runtime::machine::capture::LiveTakeStatus>,
        cx: &mut Context<Self>,
    ) -> bool {
        let Some(link) = &self.host.link else {
            return false;
        };
        let Some(activation) = self.activation.adopt(Instant::now()) else {
            return false;
        };
        let feed = link.feed(&take);
        let mut live = LiveCapture::new(take.clone(), feed);
        // Claiming already asked for its audio from the start.
        live.tapped = true;
        self.recorder = Some(live);
        self.recording_take = Some(activation);
        self.audio_upkeep.set_recording(true);
        self.live_partial.clear();
        self.stream_degradation = None;
        if self.staged_mode() {
            self.begin_staging(cx);
        } else {
            self.retire_staging(cx);
        }
        let target = self.resolve_take_target();
        self.host.stream_endpoint = (!target.endpoint().is_empty()).then(|| target.endpoint().to_string());
        self.active_take = Some(target);
        self.levels = vec![0.06; 52];
        self.service_notice = Some(
            "A recording was still running when this window opened; it continues here. Stop it \
             as usual. Nothing will be typed into another app for it."
                .to_string(),
        );
        self.overlay_take_started();
        // The tap replays the take from its start, so live text catches up.
        self.live_tick(&take, rate, status, cx);
        self.activation_settled(cx);
        true
    }

    /// A finishing take's audio or end arrived: a cancel that kept
    /// nothing is resolved now.
    fn finishing_progress(&mut self, take: &str, cx: &mut Context<Self>) {
        let Some(index) = self.finishing_index(take) else {
            return;
        };
        let nothing_kept = matches!(self.host.finishing[index].ended, Some((_, false)));
        if !nothing_kept {
            return;
        }
        let finishing = self.host.finishing.remove(index);
        if let Some(link) = &self.host.link {
            link.forget(take);
        }
        match finishing.kind {
            FinishKind::Cancel {
                staging,
                empty_notice,
                ..
            } => {
                if let Some(notice) = empty_notice {
                    if self.activation.last_started() == finishing.activation {
                        self.take_notice = Some(notice);
                    }
                }
                if let Some(token) = staging {
                    self.staging_save_failed(token, cx);
                }
            }
            FinishKind::Transcribe {
                staging, stopped_at, ..
            } => {
                self.take_notice = Some(
                    "The microphone delivered no audio, so nothing was recorded.".to_string(),
                );
                if let Some(token) = staging {
                    self.staging_save_failed(token, cx);
                }
                self.overlay.model.save_failed(stopped_at, Instant::now());
            }
        }
    }

    /// The host stored a take this window stopped or cancelled.
    fn take_stored(
        &mut self,
        take: &str,
        stored_id: Option<String>,
        interrupted: bool,
        error: Option<String>,
        cx: &mut Context<Self>,
    ) {
        let Some(index) = self.finishing_index(take) else {
            return;
        };
        let finishing = self.host.finishing.remove(index);
        if let Some(link) = &self.host.link {
            link.forget(take);
        }
        let complete = finishing.feed.complete();
        if let Some((have, total)) = finishing.feed.short_of() {
            eprintln!(
                "Starling: take {take} reached this window with {have} of its {total} samples; \
                 it is transcribed from the stored recording"
            );
        }
        let rate = finishing.feed.sample_rate();
        let FinishingTake {
            activation,
            handoff,
            kind,
            feed,
            ..
        } = finishing;
        // The store refused it: the audio this window holds is the only
        // copy that can be offered (the journal stays for recovery).
        if let Some(err) = error {
            if self.error.is_none() {
                self.error = Some(format!("Local storage failed: {err}"));
            }
            let drained = handoff.samples;
            cx.spawn(async move |this, cx| {
                let wav = cx
                    .background_spawn(async move {
                        let samples = held_samples(drained, &feed);
                        if samples.is_empty() {
                            return None;
                        }
                        audio::encode_wav_16k_parts(&samples, rate.max(1), 1).ok()
                    })
                    .await;
                if let Some(wav) = wav {
                    let short = if complete {
                        ""
                    } else {
                        " Only part of the take reached this window, so this copy is \
                         incomplete; the full take stays in the recording journal and is \
                         recovered when Starling's recording service next starts."
                    };
                    this.update(cx, |app, cx| {
                        app.stash_unsaved(
                            Arc::new(wav),
                            &format!(
                                "Local storage failed: {err} Keep this window open and download \
                                 the unsaved WAV to recover it.{short}"
                            ),
                        );
                        cx.notify();
                    })
                    .ok();
                }
            })
            .detach();
            match kind {
                FinishKind::Transcribe {
                    staging, stopped_at, ..
                } => {
                    if let Some(token) = staging {
                        self.staging_save_failed(token, cx);
                    }
                    self.overlay.model.save_failed(stopped_at, Instant::now());
                }
                FinishKind::Cancel { staging, .. } => {
                    if let Some(token) = staging {
                        self.staging_save_failed(token, cx);
                    }
                }
            }
            return;
        }
        let Some(id) = stored_id else {
            // The host could not confirm which row the take landed in
            // (its commit outcome was unreadable): nothing is transcribed
            // or typed against a guess. Whatever landed is in history.
            let staging = match kind {
                FinishKind::Transcribe {
                    staging, stopped_at, ..
                } => {
                    self.overlay.model.save_failed(stopped_at, Instant::now());
                    self.error = Some(
                        "The recording was saved, but Starling could not confirm which history \
                         entry holds it, so it was not transcribed. Find it in your history and \
                         transcribe it from there."
                            .to_string(),
                    );
                    staging
                }
                FinishKind::Cancel { staging, .. } => staging,
            };
            if let Some(token) = staging {
                self.staging_ended_without_transcript(token, cx);
            }
            self.refresh_history(cx);
            return;
        };
        match kind {
            FinishKind::Cancel {
                saved_notice,
                staging,
                ..
            } => {
                if let Some(token) = staging {
                    self.bind_staging(token, &id);
                }
                // The take is in history only now (#221): a notice that
                // says so lands here — unless a newer take started.
                if let Some(notice) = saved_notice {
                    if self.activation.last_started() == activation {
                        self.take_notice = Some(notice);
                    }
                }
                self.refresh_history(cx);
            }
            FinishKind::Transcribe {
                target,
                stopped_at,
                staging,
                delivery,
            } if interrupted => {
                // Saved as interrupted (the microphone did not stop
                // cleanly): kept, never transcribed as complete.
                let _ = (target, delivery);
                if let Some(token) = staging {
                    self.staging_ended_without_transcript(token, cx);
                }
                self.overlay.model.save_failed(stopped_at, Instant::now());
                if self.error.is_none() {
                    self.error = Some(
                        "The recording was saved to your history as an interrupted recording; \
                         transcribe it from there."
                            .to_string(),
                    );
                }
                self.refresh_history(cx);
            }
            FinishKind::Transcribe {
                target,
                stopped_at,
                staging,
                delivery,
            } => {
                let Handoff {
                    sent,
                    stream,
                    samples: drained,
                    ..
                } = handoff;
                // The stream finishes on the tail only when this window
                // holds the whole take; otherwise the stored take is
                // uploaded in full.
                let stream = stream.filter(|_| complete);
                let Some(store) = self.store.clone() else {
                    self.take_handed_back(&id);
                    return;
                };
                // Acknowledged to the host once its transcription ends.
                self.host.handling.insert(id.clone());
                let load_id = id.clone();
                cx.spawn(async move |this, cx| {
                    let (stream, wav) = cx
                        .background_spawn(async move {
                            let mut stream = stream;
                            // The remainder past the send watermark goes
                            // out beside the load, never on the UI thread.
                            let remainder_sent = match stream.as_ref() {
                                Some(live) => {
                                    let samples = held_samples(drained, &feed);
                                    sent >= samples.len()
                                        || audio::encode_wav_16k_parts(
                                            &samples[sent..],
                                            rate.max(1),
                                            1,
                                        )
                                        .map(|wav| live.send_audio(wav))
                                        .unwrap_or(false)
                                }
                                None => true,
                            };
                            if !remainder_sent {
                                stream = None;
                            }
                            (stream, store.audio_wav(&load_id))
                        })
                        .await;
                    this.update(cx, |app, cx| match wav {
                        Ok(Some(wav)) => {
                            app.stop_instants.insert(id.clone(), stopped_at);
                            app.overlay.model.take_saved(stopped_at, &id);
                            if let Some(token) = staging {
                                app.bind_staging(token, &id);
                            }
                            app.bind_delivery(delivery, &id);
                            app.transcribe_with_stream(id, wav, stream, target, None, false, cx);
                        }
                        Ok(None) | Err(_) => {
                            let reason = match wav {
                                Err(err) => {
                                    // Back to the host for another try.
                                    app.take_handed_back(&id);
                                    err.to_string()
                                }
                                _ => {
                                    // Deleted: nothing is left to do.
                                    app.take_handled(&id);
                                    "it is no longer in history".to_string()
                                }
                            };
                            app.error = Some(format!(
                                "The recording was saved but could not be read back for \
                                 transcription ({reason})."
                            ));
                            if let Some(token) = staging {
                                app.staging_save_failed(token, cx);
                            }
                            app.overlay.model.save_failed(stopped_at, Instant::now());
                            cx.notify();
                        }
                    })
                    .ok();
                })
                .detach();
            }
        }
    }

    /// A take the host stored while no app followed it: transcribed into
    /// history, never typed (no window's delivery is bound to it). One
    /// that already has a transcript (its app went before telling the
    /// host), or that another live app is transcribing, is acknowledged
    /// instead of transcribed again.
    fn transcribe_orphan(&mut self, id: String, cx: &mut Context<Self>) {
        let Some(store) = self.store.clone() else {
            if let Some(link) = &self.host.link {
                link.handed_back(&id);
            }
            return;
        };
        // Acknowledged to the host once its transcription ends; one that
        // cannot be read here stays the host's for the next window.
        self.host.handling.insert(id.clone());
        cx.spawn(async move |this, cx| {
            let (progress, wav) = {
                let store = store.clone();
                let id = id.clone();
                cx.background_spawn(async move {
                    let progress = store.recognition(&id);
                    let wav = matches!(progress, Ok(Recognition::Due)).then(|| store.audio_wav(&id));
                    (progress, wav)
                })
                .await
            };
            match (progress, wav) {
                (Ok(Recognition::Due), Some(Ok(Some(wav)))) => {
                    this.update(cx, |app, cx| {
                        app.service_notice = Some(
                            "A recording that was still running when Starling closed was saved; \
                             it is being transcribed into your history."
                                .to_string(),
                        );
                        let target = app.resolve_take_target();
                        app.transcribe_with_stream(id, wav, None, target, None, false, cx);
                    })
                    .ok();
                }
                (progress, wav) => {
                    this.update(cx, |app, cx| match (progress, wav) {
                        // Transcribed already; or deleted from history:
                        // nothing is left to do.
                        (Ok(Recognition::Done), _) | (_, Some(Ok(None))) => app.take_handled(&id),
                        // Another app's job settles it; this window's
                        // history follows.
                        (Ok(Recognition::InFlight), _) => {
                            app.take_handled(&id);
                            app.follow_foreign_take(id.clone(), cx);
                        }
                        // Back to the host for another try.
                        _ => app.take_handed_back(&id),
                    })
                    .ok();
                    refresh_sessions(&this, &store, cx).await
                }
            }
        })
        .detach();
    }

    /// Stop the window's live take: the host finalizes and stores it; the
    /// take is transcribed once it is stored. `CancelReason`-free: the
    /// cancel path is [`Self::cancel_live_take`].
    pub(crate) fn finish_live_take(
        &mut self,
        live: LiveCapture,
        activation: TakeId,
        handoff: Handoff<LiveStream>,
        kind: FinishKind,
    ) {
        let finishing = FinishingTake::new(live.take.clone(), activation, Arc::clone(&live.feed), handoff, kind);
        self.host_command(&live.take, finishing.command());
        if !live.tapped {
            // Never confirmed, so never tapped: tap now so the end still
            // arrives after the last sample.
            if let Some(link) = &self.host.link {
                link.tap(&live.take, 0);
            }
        }
        self.host.finishing.push(finishing);
    }

    /// Whether a cancel with `reason` keeps a notice for when the take is
    /// in history.
    pub(crate) fn cancel_saved_notice(reason: CancelReason) -> Option<String> {
        match reason {
            CancelReason::Escape => Some(
                "Cancelled with Escape. No transcript was kept and nothing was inserted anywhere; \
                 the audio is in your history, ready to transcribe if you need the words."
                    .to_string(),
            ),
            CancelReason::NoAudioYet => Some(
                "Stopped before the microphone was fully ready. The little audio it captured is \
                 in your history; no transcript was kept."
                    .to_string(),
            ),
            CancelReason::MicStalled => Some(
                "The microphone stalled at the start, so the take was stopped. The audio it did \
                 capture is in your history; no transcript was kept."
                    .to_string(),
            ),
            CancelReason::InputLost => Some(
                "The recording was interrupted when the microphone stopped. Everything captured \
                 before that is in your history as an interrupted recording; transcribe it from \
                 there."
                    .to_string(),
            ),
        }
    }
}
