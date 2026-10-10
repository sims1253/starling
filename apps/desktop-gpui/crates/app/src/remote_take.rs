//! Takes recorded and transcribed by the runtime host, app side (#220).
//!
//! The app asks the host to start, stop or cancel a take and follows it
//! through the host's take feed ([`crate::host_link`]). The host streams
//! the take's live text while it records and transcribes it once it is
//! stored; the app shows both. A take the app stopped or cancelled is
//! *finishing* until the host says it is stored: a stopped take's
//! delivery and staging bind to the stored row then, and run when the
//! host reports the transcript — typed only for the take this window
//! recorded, never for a retry or a take another window left.
//!
//! A take outlives this window. When the connection drops mid-take the
//! window lets go of it (the host keeps recording); after reconnecting —
//! or after the app restarted — a take the host still records with no
//! live owner is adopted: it becomes the active take, latched, its live
//! text shown from the host's latest, and it stops like any other. A take
//! the host stopped and stored while no app followed it is transcribed by
//! the host into history (nothing is typed: the window it was meant for
//! is gone).

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

use gpui::Context;
use starling_runtime::protocol::Command;
use starling_runtime_host::client::HostClient;
use starling_runtime_host::frame::{LivePartial, TakeBusy, TakeOwner, TranscriptionState};
use starling_runtime_host::live::stream::Partial;

use crate::activation::{CancelReason, TakeId};
use crate::app::{HealthCheckPurpose, StarlingApp};
use crate::host_link::{HostLink, HostUpdate, LiveCapture, TakeUpdate};
use crate::store::AudioHold;
use crate::upload::refresh_sessions;

/// How often a stop or cancel the host has not acted on is asked again.
const REASK: Duration = Duration::from_secs(1);

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
    /// Takes this window stopped or cancelled, until they are stored.
    pub(crate) finishing: Vec<FinishingTake>,
    /// A running take with no live owner this window asked for, and
    /// when: it is adopted only once the host says it is this window's —
    /// of two windows asking, only one gets it.
    pub(crate) claiming: Option<(String, Instant)>,
    /// This window's stored takes whose transcript it waits for (their
    /// delivery and staging are bound to them).
    pub(crate) awaiting: HashSet<String>,
    /// Stored takes the host reports it is transcribing.
    pub(crate) transcribing: HashSet<String>,
    /// Transcriptions this window asked for, by request, holding their
    /// take's audio until the host started them (#342).
    pub(crate) requests: HashMap<String, Request>,
    /// The host a live take was let go of on a lost connection (its pid):
    /// a reconnect that finds another host corrects what the window said.
    pub(crate) lost_take_on: Option<u32>,
    /// The pid of the host the last connection went to.
    pub(crate) client_pid: Option<u32>,
    /// The exact transcript (attempt id and text) this window's own takes
    /// deliver and stage — never a later result another window asked
    /// for — newest last. Kept past the result's arrival: a staged take's
    /// Insert may come much later ([`OWN_RESULTS_KEPT`] of them).
    pub(crate) own_results: VecDeque<OwnResult>,
    /// The live text the host sent with an adoption still being
    /// confirmed, shown once the take is this window's.
    pub(crate) claimed_text: Option<(Option<LivePartial>, Option<String>)>,
}

/// How many of this window's own results are remembered.
const OWN_RESULTS_KEPT: usize = 32;

/// One own take's transcript ([`HostState::own_results`]).
pub(crate) struct OwnResult {
    stored_id: String,
    attempt: Option<String>,
    text: String,
}

/// What the window says when the service went away with its take.
const SERVICE_STOPPED: &str = "Starling's recording service stopped while recording. It is \
     starting again; the audio it had saved is recovered into your history as an interrupted \
     recording.";

/// A transcription this window asked the host for.
pub(crate) struct Request {
    pub(crate) stored_id: String,
    /// Held (never read) until the host's attempt holds the audio: any
    /// process's upkeep leaves it alone meanwhile.
    pub(crate) _hold: Option<AudioHold>,
    /// A retry: its transcript is offered for Copy / Paste last.
    pub(crate) offer: bool,
}

/// A take this window stopped or cancelled, until the host stored it.
pub(crate) struct FinishingTake {
    pub(crate) take: String,
    pub(crate) activation: TakeId,
    pub(crate) kind: FinishKind,
    /// The take's final sample count and whether it kept anything.
    pub(crate) ended: Option<(u64, bool)>,
    asked_at: Instant,
}

pub(crate) enum FinishKind {
    /// Stopped: the host transcribes it once stored, and this window
    /// delivers it.
    Transcribe {
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
    pub(crate) fn new(take: String, activation: TakeId, kind: FinishKind) -> FinishingTake {
        FinishingTake {
            take,
            activation,
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

/// A host preview as the staging draft takes it.
fn partial_of(live: LivePartial) -> Partial {
    Partial {
        text: live.text,
        stable_words: live.stable_words,
        covered_s: live.covered_s,
    }
}

/// What a refused start says, from what the host says holds the
/// microphone.
fn start_refusal(busy: Option<TakeBusy>, reason: &str) -> String {
    match busy {
        Some(TakeBusy::Recording { yours: false }) => "Another Starling window is recording \
             right now. Stop that recording first, then start a new one."
            .to_string(),
        Some(TakeBusy::Recording { yours: true }) => "This window's previous recording is still \
             running. Stop it first, then start a new one."
            .to_string(),
        Some(TakeBusy::Saving) => "The previous recording is still being saved. Start again in a \
             moment."
            .to_string(),
        None => format!("The recording could not start: {reason}"),
    }
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
                // The connection was lost mid-take, and this is another
                // host: the take did not carry on.
                if self
                    .host
                    .lost_take_on
                    .take()
                    .is_some_and(|pid| pid != client.info.pid)
                {
                    self.error = Some(SERVICE_STOPPED.to_string());
                }
                self.host.client = Some(client);
                self.host.down = None;
                self.host.gave_up = false;
                if let Some(recovery) = recovery {
                    crate::upload::add_recovery_messages(self, recovery.problems, recovery.notice);
                }
                self.refresh_history(cx);
            }
            HostUpdate::Disconnected {
                reason,
                gave_up,
                host_gone,
            } => {
                self.host.client_pid = self.host.client.as_ref().map(|client| client.info.pid);
                self.host.client = None;
                self.host.down = Some(reason);
                self.host.gave_up = gave_up;
                self.host.claiming = None;
                self.let_go_of_takes(host_gone, cx);
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
                busy,
            } => match command {
                "capture.start" => {
                    self.take_start_failed(&take, None, start_refusal(busy, &reason), cx)
                }
                "take.adopt" => {
                    // Asked again on the next tick while it is unowned.
                    if let Some((_, asked_at)) = self
                        .host
                        .claiming
                        .as_mut()
                        .filter(|(claimed, _)| *claimed == take)
                    {
                        *asked_at = Instant::now() - REASK;
                    }
                }
                // `take` is the request: nothing was attempted.
                "transcribe" if self.host.requests.remove(&take).is_some() => {
                    self.error = Some(format!(
                        "Could not ask Starling's recording service to transcribe the \
                         recording ({reason}). The recording is unchanged."
                    ));
                }
                // A stop or cancel that was refused is asked again while
                // the take still records (see `take_update`).
                _ => {}
            },
        }
        cx.notify();
    }

    pub(crate) fn refresh_history(&mut self, cx: &mut Context<Self>) {
        if let Some(store) = self.store.clone() {
            cx.spawn(async move |this, cx| refresh_sessions(&this, &store, cx).await)
                .detach();
        }
    }

    /// The connection dropped. A take the window had keeps recording in
    /// the host when only the connection went (it is adopted again on
    /// reconnect); when the host itself went, it took the take with it,
    /// and the next one recovers what was saved. Either way the takes this
    /// window waits for are transcribed into history without it: nothing
    /// is typed for them later.
    fn let_go_of_takes(&mut self, host_gone: bool, cx: &mut Context<Self>) {
        if self.recorder.take().is_some() {
            self.end_live_take_locally(cx);
            self.host.lost_take_on = (!host_gone).then_some(self.host.client_pid).flatten();
            self.error = Some(if host_gone {
                SERVICE_STOPPED.to_string()
            } else {
                "Lost the connection to Starling's recording service. The recording continues \
                 there and comes back here once the connection does."
                    .to_string()
            });
        }
        for finishing in std::mem::take(&mut self.host.finishing) {
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
        for id in std::mem::take(&mut self.host.awaiting) {
            self.active_ids.remove(&id);
            self.no_transcript_here(&id, cx);
        }
        for id in std::mem::take(&mut self.host.transcribing) {
            self.active_ids.remove(&id);
        }
        self.host.requests.clear();
        self.refresh_history(cx);
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
        if owner == TakeOwner::Another || (ended && owner == TakeOwner::Nobody) {
            // Another window got it first, or it ended unowned (the host
            // stores and transcribes it).
            self.host.claiming = None;
            self.host.claimed_text = None;
            return;
        }
        if owner == TakeOwner::Nobody {
            // The adoption has not landed yet (or was lost to a
            // reconnect): ask again now and then.
            if let Some((_, asked_at)) = self.host.claiming.as_mut() {
                if asked_at.elapsed() >= REASK {
                    *asked_at = Instant::now();
                    if let Some(link) = &self.host.link {
                        link.adopt(&take);
                    }
                }
            }
            return;
        }
        self.host.claiming = None;
        if ended || self.recorder.is_some() || self.activation.is_active() {
            self.host.claimed_text = None;
        }
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
    /// with a take of the window's own): stopped (`stop`: unless it ended
    /// already) and left to the host to transcribe into history, so it
    /// never records on with nobody able to stop it.
    fn finish_unshown_take(&mut self, take: String, stop: bool) {
        let finishing = FinishingTake::new(
            take.clone(),
            self.activation.last_started(),
            FinishKind::Transcribe {
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
        if self.recorder.take().is_some() {
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
                meter,
            } => {
                if self.recorder.as_ref().is_some_and(|live| live.take == take) {
                    if owner == TakeOwner::Another {
                        // Another window adopted it first.
                        self.lose_live_take(cx);
                        return;
                    }
                    self.live_tick(rate, status.clone(), meter, cx);
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
                        link.adopt(&take);
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
                ..
            } => {
                if self.finishing_index(&take).is_some() {
                    self.take_stored(&take, stored_id, interrupted, error, cx);
                } else {
                    // Another window's take (or one nobody followed): it is
                    // in history now, and its transcript once the host has
                    // it.
                    self.refresh_history(cx);
                }
            }
            TakeUpdate::Notice(recovery) => {
                crate::upload::add_recovery_messages(self, recovery.problems, recovery.notice);
                self.refresh_history(cx);
            }
            TakeUpdate::LiveText {
                take,
                partial,
                degraded,
            } => {
                // Only the running take's: a preview still in flight when
                // its take stopped is never shown — the final replaces it.
                // One for a take being adopted waits for the adoption.
                if self
                    .host
                    .claiming
                    .as_ref()
                    .is_some_and(|(claimed, _)| *claimed == take)
                {
                    self.host.claimed_text = Some((partial, degraded));
                    return;
                }
                if !self.recorder.as_ref().is_some_and(|live| live.take == take) {
                    return;
                }
                self.show_live_text(partial, degraded, cx);
            }
            TakeUpdate::Transcription {
                stored_id,
                take,
                req,
                attempt,
                state,
                yours,
            } => self.transcription_update(stored_id, take, req, attempt, state, yours, cx),
        }
    }

    /// The running take's live text, or why it stopped.
    fn show_live_text(
        &mut self,
        partial: Option<LivePartial>,
        degraded: Option<String>,
        cx: &mut Context<Self>,
    ) {
        if let Some(reason) = degraded {
            self.stream_degradation = Some(reason);
        }
        if let Some(partial) = partial {
            if self.staging.is_some() {
                self.staging_partial(partial_of(partial), cx);
            } else {
                self.live_partial = partial.text;
            }
        }
    }

    fn finishing_index(&self, take: &str) -> Option<usize> {
        self.host.finishing.iter().position(|finishing| finishing.take == take)
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
        rate: u32,
        status: Option<starling_runtime::machine::capture::LiveTakeStatus>,
        meter: Option<Vec<f32>>,
        cx: &mut Context<Self>,
    ) {
        let Some(live) = self.recorder.as_mut() else {
            return;
        };
        if rate > 0 {
            live.rate = rate;
        }
        if let Some(meter) = meter {
            live.meter = meter;
        }
        live.confirmed = true;
        if let Some(status) = status {
            if let Some(route) = status.route.clone() {
                self.mic.last_route = Some(route);
            }
            live.status = Some(status);
        }
        // Listening is announced as soon as the service reports audio.
        self.check_readiness(cx);
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
        if let Some(lease) = self.playback_lease.take() {
            self.playback.handle().end(lease);
        }
        self.audio_upkeep.set_recording(false);
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
        let Some(activation) = self.activation.adopt(Instant::now()) else {
            return false;
        };
        self.recorder = Some(LiveCapture::new(take));
        self.recording_take = Some(activation);
        self.audio_upkeep.set_recording(true);
        self.live_partial.clear();
        self.stream_degradation = None;
        if self.staged_mode() {
            self.begin_staging(cx);
        } else {
            self.retire_staging(cx);
        }
        self.levels = vec![0.06; 52];
        self.service_notice = Some(
            "A recording was still running when this window opened; it continues here. Stop it \
             as usual. Nothing will be typed into another app for it."
                .to_string(),
        );
        self.overlay_take_started();
        self.live_tick(rate, status, None, cx);
        // The take's live text so far, sent with the adoption.
        if let Some((partial, degraded)) = self.host.claimed_text.take() {
            self.show_live_text(partial, degraded, cx);
        }
        self.activation_settled(cx);
        true
    }

    /// A finishing take's end arrived: a cancel or stop that kept nothing
    /// is resolved now.
    fn finishing_progress(&mut self, take: &str, cx: &mut Context<Self>) {
        let Some(index) = self.finishing_index(take) else {
            return;
        };
        let nothing_kept = matches!(self.host.finishing[index].ended, Some((_, false)));
        if !nothing_kept {
            return;
        }
        let finishing = self.host.finishing.remove(index);
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
        let FinishingTake {
            activation, kind, ..
        } = self.host.finishing.remove(index);
        // The store refused it: the recording stays in the service's
        // capture journal, which it recovers at its next start.
        if let Some(err) = error {
            if self.error.is_none() {
                self.error = Some(format!(
                    "Local storage failed: {err} The recording stays in Starling's capture \
                     journal and is recovered into your history the next time Starling's \
                     recording service starts."
                ));
            }
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
            // (its commit outcome was unreadable): nothing is typed
            // against a guess. Whatever landed is in history.
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
            }
            FinishKind::Transcribe {
                stopped_at,
                staging,
                delivery,
            } if interrupted => {
                // Saved as interrupted (the microphone did not stop
                // cleanly): kept, never transcribed as complete.
                let _ = delivery;
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
            }
            FinishKind::Transcribe {
                stopped_at,
                staging,
                delivery,
            } => {
                // The host transcribes it now; its text goes where the
                // take was meant to go once the host has it.
                self.stop_instants.insert(id.clone(), stopped_at);
                self.overlay.model.take_saved(stopped_at, &id);
                if let Some(token) = staging {
                    self.bind_staging(token, &id);
                }
                self.bind_delivery(delivery, &id);
                self.selected_id = Some(id.clone());
                self.host.awaiting.insert(id.clone());
                self.active_ids.insert(id);
            }
        }
        self.refresh_history(cx);
    }

    /// Where the host's transcription of stored take `stored_id` stands.
    /// Every window shows it; only the one it is for (`yours`) acts on the
    /// result. A take's own transcription (no `req`) is delivered by the
    /// window that recorded it, with exactly that result's text; a
    /// request's result (a retry, an import) is offered to the window that
    /// asked, never typed, and never touches the take's own delivery.
    #[allow(clippy::too_many_arguments)]
    fn transcription_update(
        &mut self,
        stored_id: String,
        take: Option<String>,
        req: Option<String>,
        attempt: Option<String>,
        state: TranscriptionState,
        yours: bool,
        cx: &mut Context<Self>,
    ) {
        // A request answered for another take is not this one's: it stays
        // (with its hold) for the frames that are.
        let request = match req.as_ref() {
            Some(req)
                if self
                    .host
                    .requests
                    .get(req)
                    .is_some_and(|request| request.stored_id == stored_id) =>
            {
                self.host.requests.remove(req)
            }
            _ => None,
        };
        let offer = request.as_ref().is_some_and(|request| request.offer);
        let own_job = req.is_none();
        match state {
            TranscriptionState::Started { .. } => {
                if let (Some(req), Some(request)) = (&req, request) {
                    // Kept for the end; the attempt holds the audio now.
                    self.host.requests.insert(
                        req.clone(),
                        Request {
                            _hold: None,
                            ..request
                        },
                    );
                }
                self.host.transcribing.insert(stored_id.clone());
                self.active_ids.insert(stored_id.clone());
                if yours && own_job && !self.host.awaiting.contains(&stored_id) {
                    self.service_notice = Some(if take.is_some() {
                        "A recording that was still running when its window closed was saved; \
                         it is being transcribed into your history."
                            .to_string()
                    } else {
                        "A recording Starling saved but had not transcribed yet is being \
                         transcribed into your history."
                            .to_string()
                    });
                }
                self.refresh_history(cx);
            }
            // Nothing started: nothing else is touched (a job on the take
            // may still be running).
            TranscriptionState::Refused { message } => {
                if yours && (request.is_some() || req.is_none()) {
                    self.error = Some(message);
                }
                if own_job && self.host.awaiting.remove(&stored_id) {
                    // Its wait ends here: the take stays busy only while
                    // a job on it still runs.
                    if !self.host.transcribing.contains(&stored_id) {
                        self.active_ids.remove(&stored_id);
                    }
                    self.no_transcript_here(&stored_id, cx);
                }
                self.refresh_history(cx);
            }
            state => {
                self.host.transcribing.remove(&stored_id);
                self.active_ids.remove(&stored_id);
                let own = own_job && self.host.awaiting.remove(&stored_id);
                if !yours {
                    // Shown here, acted on elsewhere.
                    if own {
                        self.no_transcript_here(&stored_id, cx);
                    }
                    self.refresh_history(cx);
                    return;
                }
                match state {
                    TranscriptionState::Completed { kept_earlier, text } => {
                        let Some(store) = self.store.clone() else {
                            // Nowhere to read the take back from: nothing
                            // is typed or staged for it here.
                            if own {
                                self.no_transcript_here(&stored_id, cx);
                            }
                            return;
                        };
                        cx.spawn(async move |this, cx| {
                            // Raw text is in history first; the active
                            // mode's processing follows as a proposal
                            // (#295). A blank retry changed no shown text:
                            // no draft is dropped, nothing is reprocessed
                            // or offered again.
                            refresh_sessions(&this, &store, cx).await;
                            if kept_earlier {
                                return;
                            }
                            this.update(cx, |app, cx| {
                                // Only the take's own transcription is
                                // delivered and staged, with exactly its
                                // text; anything else this window acts on
                                // is processed and, if it asked, offered.
                                // A retry this window asked for recovers a
                                // staging panel that failed with its edits
                                // kept (staged: never typed — its delivery
                                // went when it was asked for).
                                let recovers = offer && app.staging_failed_for(&stored_id);
                                if own || recovers {
                                    app.remember_own_result(&stored_id, attempt, text.clone());
                                    app.after_transcription(stored_id.clone(), cx);
                                } else {
                                    app.after_other_result(stored_id.clone(), cx);
                                }
                                if offer {
                                    app.offer_retried_text(&stored_id, &text, cx);
                                }
                            })
                            .ok();
                        })
                        .detach();
                    }
                    TranscriptionState::Failed { message, transport } => {
                        self.error = Some(message);
                        // A server that could not be reached is probed
                        // again (Diagnostic, #207: the badge, never this
                        // explanation). The built-in engine's own endpoint
                        // is never probed here.
                        if transport
                            && self.engine_settings.mode
                                == starling_dictation::settings::EngineMode::Manual
                        {
                            self.check_health(
                                HealthCheckPurpose::Diagnostic,
                                self.endpoint.clone(),
                                cx,
                            );
                        }
                        if own {
                            self.no_transcript_here(&stored_id, cx);
                        }
                        self.refresh_history(cx);
                    }
                    TranscriptionState::Gone => {
                        if own || request.is_some() {
                            self.error = Some(
                                "This recording was deleted while it was being transcribed, so \
                                 its transcript was not kept."
                                    .to_string(),
                            );
                        }
                        if own {
                            self.no_transcript_here(&stored_id, cx);
                        }
                        self.refresh_history(cx);
                    }
                    TranscriptionState::Started { .. } | TranscriptionState::Refused { .. } => {}
                }
            }
        }
    }

    /// Remembers take `id`'s own result, replacing an earlier one.
    fn remember_own_result(&mut self, id: &str, attempt: Option<String>, text: String) {
        self.forget_own_result(id);
        if self.host.own_results.len() >= OWN_RESULTS_KEPT {
            self.host.own_results.pop_front();
        }
        self.host.own_results.push_back(OwnResult {
            stored_id: id.to_string(),
            attempt,
            text,
        });
    }

    /// Take `id` delivers or stages nothing more of its own here.
    pub(crate) fn forget_own_result(&mut self, id: &str) {
        self.host.own_results.retain(|own| own.stored_id != id);
    }

    fn own(&self, id: &str) -> Option<&OwnResult> {
        self.host.own_results.iter().find(|own| own.stored_id == id)
    }

    /// The transcript stored take `id` delivers here: its own result's
    /// text (a later result another window asked for must not be typed in
    /// its place).
    pub(crate) fn own_result(&self, id: &str) -> Option<String> {
        self.own(id).map(|own| own.text.clone())
    }

    /// [`Self::own_result`] with the attempt that produced it.
    pub(crate) fn own_result_attempt(&self, id: &str) -> Option<(String, String)> {
        self.own(id)
            .and_then(|own| own.attempt.clone().map(|attempt| (attempt, own.text.clone())))
    }

    /// No transcript will come for stored take `id` in this window:
    /// nothing is typed or processed for it.
    fn no_transcript_here(&mut self, id: &str, cx: &mut Context<Self>) {
        self.forget_own_result(id);
        self.stop_instants.remove(id);
        self.forget_delivery(id);
        self.staging_transcription_failed(id, cx);
    }

    /// Stop the window's live take: the host finalizes and stores it; the
    /// take is transcribed once it is stored. `CancelReason`-free: the
    /// cancel path is [`Self::cancel_recording`].
    pub(crate) fn finish_live_take(&mut self, live: LiveCapture, activation: TakeId, kind: FinishKind) {
        let finishing = FinishingTake::new(live.take.clone(), activation, kind);
        self.host_command(&live.take, finishing.command());
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_refused_start_says_what_holds_the_microphone() {
        let reason = "the runtime refused the command: capture.start illegal in Recording";
        let told = start_refusal(Some(TakeBusy::Recording { yours: false }), reason);
        assert!(told.contains("Another Starling window is recording"), "{told}");
        assert!(!told.contains("illegal"));
        let told = start_refusal(Some(TakeBusy::Saving), reason);
        assert!(told.contains("still being saved"), "{told}");
        // The wording of the rejection decides nothing.
        assert_eq!(
            start_refusal(None, reason),
            format!("The recording could not start: {reason}")
        );
        assert_eq!(
            start_refusal(None, "not connected to the recording service"),
            "The recording could not start: not connected to the recording service"
        );
    }
}
