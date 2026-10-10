//! The recording -> encode -> persist -> transcribe pipeline and the file
//! import flow, split out of `app.rs`.

use std::sync::Arc;
use std::time::Instant;

use gpui::{AppContext, AsyncApp, Context, PathPromptOptions, WeakEntity};
use starling_dictation::{
    audio,
    client::{ClientError, StarlingClient},
    engine::EngineLease,
    recorder,
    settings::EngineMode,
    storage,
};

use crate::app::{HealthCheckPurpose, StarlingApp, UnsavedWav};
use crate::live_stream::LiveStream;
use crate::store::{AudioPin, Store};

/// A take's endpoint/model binding (#363), resolved once at the moment
/// the take starts and carried with it to the end of its transcription
/// job. The lease (builtin mode) pins the engine: a model switch
/// mid-take spawns the new engine, but this take keeps talking to the
/// endpoint it started on, and the draining engine waits for the lease
/// to drop — which happens exactly when the job finishes (success,
/// failure, or session gone) because the target moves into the job and
/// is dropped with it.
///
/// Manual mode binds the committed `endpoint`/`model` at start instead,
/// so even a settings save mid-take cannot move a take that already
/// began. Retries, re-transcribes, and file imports resolve a fresh
/// target at the moment they start — they are new jobs, not continuations.
pub(crate) struct TakeTarget {
    endpoint: String,
    model: String,
    provenance: storage::BackendLabel,
    lease: Option<EngineLease>,
    builtin: bool,
}

impl TakeTarget {
    /// A manual-mode target: the committed endpoint and model, with the
    /// unchanged `openai:<model>` attempt label.
    fn manual(endpoint: String, model: String) -> TakeTarget {
        let provenance = storage::BackendLabel::OpenAi {
            model: model.clone(),
        };
        TakeTarget {
            endpoint,
            model,
            provenance,
            lease: None,
            builtin: false,
        }
    }

    /// A builtin-mode target holding its engine lease.
    fn from_lease(lease: EngineLease) -> TakeTarget {
        TakeTarget {
            model: lease.slug().to_string(),
            provenance: storage::BackendLabel::Engine {
                model_id: lease.model_id().to_string(),
            },
            endpoint: lease.endpoint().to_string(),
            lease: Some(lease),
            builtin: true,
        }
    }

    /// A builtin-mode target with no engine to talk to (the engine was
    /// not ready when the take started): the take still records — audio
    /// is journaled — and transcription is retried once the engine is.
    fn builtin_unready() -> TakeTarget {
        TakeTarget {
            endpoint: String::new(),
            model: String::new(),
            provenance: storage::BackendLabel::Engine {
                model_id: "none".to_string(),
            },
            lease: None,
            builtin: true,
        }
    }

    pub(crate) fn endpoint(&self) -> &str {
        &self.endpoint
    }

    pub(crate) fn model(&self) -> &str {
        &self.model
    }

    /// The attempt row's backend label (`engine:<model_id>` builtin,
    /// `openai:<model>` manual) — the provenance history shows per take.
    /// Rendered through [`storage::BackendLabel`], the typed form of the
    /// persisted string shape.
    fn provenance(&self) -> String {
        self.provenance.to_string()
    }

    /// Whether this is a builtin take whose engine was not ready — the
    /// job may re-resolve once before failing honestly.
    fn needs_engine(&self) -> bool {
        self.builtin && self.lease.is_none()
    }

    /// Whether failures of this take belong to the built-in engine —
    /// they must never be routed into the manual-endpoint health prober.
    fn is_builtin(&self) -> bool {
        self.builtin
    }
}

impl StarlingApp {
    /// Resolve the target for a take starting now (#363): the engine's
    /// lease in builtin mode (recording proceeds even without one — the
    /// audio is journaled and transcription retries later), the
    /// committed endpoint/model in manual mode.
    pub(crate) fn resolve_take_target(&self) -> TakeTarget {
        match self.engine_settings.mode {
            EngineMode::Builtin => match self.engine.as_ref().and_then(|engine| engine.lease()) {
                Some(lease) => TakeTarget::from_lease(lease),
                None => TakeTarget::builtin_unready(),
            },
            EngineMode::Manual => TakeTarget::manual(self.endpoint.clone(), self.model.clone()),
        }
    }
}

/// What a transport-class failure against the built-in engine's own
/// endpoint says (#363): the sidecar went away, the audio is safe, and
/// the manual-endpoint prober has nothing to do with it. Other failure
/// classes (the server answered and objected) keep their own message.
pub(crate) fn builtin_take_failure_message(builtin: bool, class: FailureClass, err: &ClientError) -> String {
    if builtin && class == FailureClass::Transport {
        "The built-in engine stopped while transcribing; the recording is saved — retry when \
         the engine is ready."
            .to_string()
    } else {
        err.to_string()
    }
}

/// What a take says when the built-in engine never became ready (#363
/// first run): the audio is saved and retry works once a model serves.
pub(crate) fn engine_not_ready_message() -> String {
    "The built-in engine is not ready; the recording is saved — pick a model in Settings, or \
     retry once the engine is ready."
        .to_string()
}

/// What a failed job says about retrying it (R13).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum FailureClass {
    /// A transport-level failure — connection refused, timeout, network
    /// unreachable. The server may merely be down or slow, so a retry can
    /// still get through and the connection badge needs a fresh health
    /// probe.
    Transport,
    /// A deterministic failure: retrying the same request cannot change
    /// the outcome, so no probe may be prompted. This covers failures
    /// that never left this machine (local storage errors, input
    /// validation, `Cancelled`) and ones where the server provably
    /// answered and would answer the same way again — blocked redirects,
    /// HTTP error statuses, protocol/parse errors, and oversized
    /// responses (issue #235). "Local" here means "retry is pointless",
    /// not "nothing left this machine".
    Local,
}

/// Classify a transcription-client failure by whether a retry could
/// still succeed (R13). Only [`ClientError::Transport`] (connection
/// refused, network unreachable, DNS and socket failures) and
/// [`ClientError::Timeout`] are nondeterministic — the server may be
/// reachable on a later attempt. Everything else lands in
/// [`FailureClass::Local`]: `Input`/`Protocol` failures never left this
/// machine, while an HTTP error status, a blocked redirect, or an
/// oversized response (issue #235) proves the server answered and would
/// answer the same way again — deterministic, so no probe. `Cancelled`
/// (the abort signal of issue #251) is moot here: this upload path never
/// passes a cancel token, so it cannot occur.
pub(crate) fn failure_class(err: &ClientError) -> FailureClass {
    match err {
        ClientError::Transport(_) | ClientError::Timeout(_) => FailureClass::Transport,
        ClientError::Input(_)
        | ClientError::Cancelled
        | ClientError::Redirect(_)
        | ClientError::Http { .. }
        | ClientError::Protocol(_)
        | ClientError::ResponseTooLarge(_) => FailureClass::Local,
    }
}

/// What a transcript-job store write does when its session may have been
/// deleted mid-flight (R05).
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum SaveRaceDecision {
    /// The session is present and the write went through.
    Written,
    /// The session is gone — the user's delete landed while the job was in
    /// flight. Never resurrect it: keep the job's in-memory audio
    /// recoverable and surface what happened.
    SessionDeleted,
    /// A genuine storage failure (think a full disk): the write failed for
    /// reasons that have nothing to do with deletion.
    Failed(String),
}

/// Decide a store write's outcome in the delete race (R05).
///
/// `session_found` is what the pre-write re-check saw; `write_error` is the
/// write's own error, if the write was attempted. A `NotFound` from either
/// side is a delete that landed in the check→write gap and gets the same
/// `SessionDeleted` decision as a failed re-check — the app can never tell
/// the two orderings apart, and must not treat either as a job failure.
pub(crate) fn save_race_decision(
    session_found: bool,
    write_error: Option<&storage::StorageError>,
) -> SaveRaceDecision {
    if !session_found {
        return SaveRaceDecision::SessionDeleted;
    }
    match write_error {
        None => SaveRaceDecision::Written,
        Some(storage::StorageError::NotFound(_)) => SaveRaceDecision::SessionDeleted,
        Some(err) => SaveRaceDecision::Failed(err.to_string()),
    }
}

/// What the surfaced error says when a recording is deleted while it is
/// being transcribed (R05): the transcript could not land anywhere, the
/// audio is kept recoverable, and nothing is dropped silently.
pub(crate) fn session_deleted_message() -> String {
    "This recording was deleted while it was being transcribed, so the transcript could not \
     be saved to history. The audio is kept in the recovery banner — download it if you still \
     want it, or discard it if the delete was on purpose."
        .to_string()
}

/// What a finished transcription job may apply to global app state (R03,
/// revised by R13).
///
/// There is still deliberately no connection variant: a job outcome
/// conflates transport failures with local storage failures
/// (`mark_attempt`, `save_transcript`, `save_failure` — think a full
/// disk), so it cannot decide `Connection::Offline` or
/// `Connection::Ready` itself; `check_health` remains the single writer
/// of connection state. What R13 adds: a failure classified
/// [`FailureClass::Transport`] prompts a health probe, whose *result* —
/// not the job — updates the badge, so a server that dies mid-session no
/// longer shows a stale `Ready` until the next probe. That probe runs as
/// a Diagnostic check (#207): it owns the badge and the server model
/// only, never the error banner, so the job's explanation for the
/// missing transcript survives a successful probe.
pub(crate) enum FinishedJob {
    /// Transcript saved: no global-state effects.
    Saved,
    /// The job failed: surface `message` as the app error, and prompt a
    /// connection re-probe only when `class` is
    /// [`FailureClass::Transport`].
    Failed {
        message: String,
        class: FailureClass,
    },
}

/// Classify a finished transcription job (R03, revised by R13).
///
/// `job_failure` is the error that failed the job (with its
/// [`FailureClass`]), if any; `history_update_failure` is the error from
/// the follow-up attempt to record that failure in local history, if
/// that write also failed.
pub(crate) fn finished_job(
    job_failure: Option<(&str, FailureClass)>,
    history_update_failure: Option<&str>,
) -> FinishedJob {
    match job_failure {
        None => FinishedJob::Saved,
        Some((failure, class)) => FinishedJob::Failed {
            message: match history_update_failure {
                Some(storage_err) => joined_failure(failure, storage_err),
                None => failure.to_string(),
            },
            class,
        },
    }
}

/// Join a job failure with a follow-up history-write failure (R14): one
/// sentence boundary between the two messages — never a bare space, and
/// never a doubled period when the failure already ends with one (the
/// client's `Timeout` and `Protocol` messages do).
fn joined_failure(failure: &str, history_err: &str) -> String {
    let trimmed = failure.trim_end_matches('.');
    format!("{trimmed}. Local history update also failed: {history_err}")
}

impl StarlingApp {
    /// The on-screen record button: a toggle in every activation mode,
    /// through the same machine as the shortcut (#221). `showed` is what
    /// the button showed when it was rendered (Stop or Start).
    pub fn toggle_recording(
        &mut self,
        showed: crate::activation::RecordButton,
        cx: &mut Context<Self>,
    ) {
        self.flush_system_events(cx);
        if !crate::activation::click_matches(showed, self.recorder.is_some()) {
            cx.notify();
            return;
        }
        self.activation_input(|machine| machine.click(Instant::now()), cx);
    }

    /// Stop the window's live take: the host finalizes and stores it, and
    /// it is transcribed once stored (#220). Only the activation machine
    /// calls this (an `Effect::Finish`).
    pub(crate) fn stop_recording(
        &mut self,
        finished_take: crate::activation::TakeId,
        cx: &mut Context<Self>,
    ) {
        self.error = None;
        // Playback comes back when recording stops, not after transcription.
        self.end_playback_lease();
        let Some(live) = self.recorder.take() else {
            return;
        };
        self.audio_upkeep.set_recording(false);
        // The stream worker stops here; what it drained and sent travels
        // with the take, and the rest arrives on the take's feed.
        let mut pumped = self.finish_stream_pump();
        self.host.stream_endpoint = None;
        if let Some(reason) = pumped.degradation.take() {
            self.stream_degradation = Some(reason);
        }
        // The binding resolved at START leaves with the take (#363); a
        // stop without one (not a normal path) resolves fresh rather than
        // transcribing against nothing.
        let target = self
            .active_take
            .take()
            .unwrap_or_else(|| self.resolve_take_target());
        self.live_partial.clear();
        // Storage-fault honesty (I1 phase 2): a journal fault that froze
        // acknowledgment is surfaced, and the stream (which only had
        // acknowledged audio) does not commit over the stored take.
        if let Some(fault) = live.capture_error() {
            pumped.stream = None;
            self.error = Some(fault);
        }
        // G03: clipping is measured on the raw captured samples.
        self.capture_warning = recorder::clipping_warning(live.source_clip_ratio())
            .or_else(|| self.stream_degradation.take());
        self.levels = vec![0.06; 52];
        // Stop-to-processed latency (#295) starts here.
        let stopped_at = Instant::now();
        // The staging panel keeps its draft while the take is stored.
        let staging = self.stop_staging();
        // Where the take's text goes (#221) travels with it too.
        let delivery = self.delivery_take_stopped();
        self.overlay.model.take_finished(stopped_at);
        self.finish_live_take(
            live,
            finished_take,
            pumped,
            crate::remote_take::FinishKind::Transcribe {
                target,
                stopped_at,
                staging,
                delivery,
            },
        );
        cx.notify();
    }

    /// Ask the recording service for a take (#220); `false` when it
    /// cannot be asked (not connected — the reason is in the error
    /// banner). The microphone opens in the service: a microphone that
    /// will not open is reported back and ends the take then. Only the
    /// activation machine calls this (an `Effect::Start`).
    pub(crate) fn start_recording(&mut self, cx: &mut Context<Self>) -> bool {
        // Where the take's text goes (#221): captured first, before the
        // microphone or anything else can move focus.
        self.delivery_take_started();
        self.error = None;
        self.take_notice = None;
        if let Some(reason) = self.host_unavailable() {
            self.delivery_take_stopped();
            self.error = Some(match self.host.link.as_ref().filter(|_| self.host.gave_up) {
                // Pressing record is how the user asks for another try.
                Some(link) => {
                    link.retry();
                    self.host.gave_up = false;
                    self.host.down = Some("connecting".to_string());
                    "Starting Starling's recording service again; record once it is ready."
                        .to_string()
                }
                None => format!("{reason} Try again in a moment."),
            });
            cx.notify();
            return false;
        }
        let Some(link) = self.host.link.as_ref() else {
            self.delivery_take_stopped();
            self.error = Some(
                "Starling's recording service is not ready; try again in a moment.".to_string(),
            );
            cx.notify();
            return false;
        };
        // Attenuation begins with the attempt, before the microphone opens
        // — unless a start cue is due, which it would swallow: then it
        // begins once the cue has played (`cues.rs`).
        let playback_lease = (!crate::cues::attenuation_waits_for_cue(
            &self.feedback,
            self.playback_settings.during_recording,
        ))
        .then(|| self.playback.handle().begin(&self.playback_settings));
        let take = starling_runtime::bus::new_id("take");
        let feed = link.feed(&take);
        self.mic.problem = None;
        self.mic.interruption = None;
        self.mic.last_sound_at = None;
        self.live_partial.clear();
        if self.staged_mode() {
            self.begin_staging(cx);
        } else {
            self.retire_staging(cx);
        }
        self.stream_degradation = None;
        // #363: the take's endpoint/model binding resolves at START and
        // travels with the take — a model switch or settings save mid-take
        // cannot move it.
        let target = self.resolve_take_target();
        self.host.stream_endpoint = None;
        if target.endpoint().is_empty() {
            // Builtin mode with no ready engine: recording proceeds (the
            // audio is journaled either way); the note says transcription
            // will need a retry.
            self.stream_degradation = Some(
                "The built-in engine is not ready; the recording is still saved, and \
                 transcription will need a retry once the engine is ready."
                    .to_string(),
            );
        } else {
            // The live stream opens once the service reports the take's
            // device rate (its first status tick).
            self.host.stream_endpoint = Some(target.endpoint().to_string());
        }
        self.active_take = Some(target);
        self.recorder = Some(crate::host_link::LiveCapture::new(take.clone(), feed));
        self.audio_upkeep.set_recording(true);
        self.playback_lease = playback_lease;
        self.elapsed_ms = 0.0;
        self.levels = vec![0.06; 52];
        self.capture_warning = None;
        self.mic.disk_warned = false;
        self.mic.disk_low_warning = None;
        self.mic.disk_unchecked = false;
        self.host_command(
            &take,
            starling_runtime::protocol::Command::CaptureStart {
                policy: "push-to-talk".to_string(),
            },
        );
        cx.notify();
        true
    }

    /// Restores playback for the live take (asynchronously).
    fn end_playback_lease(&mut self) {
        if let Some(lease) = self.playback_lease.take() {
            self.playback.handle().end(lease);
        }
    }

    /// Cancel the window's live take without transcribing or delivering
    /// anything (#221): Escape, a stop before the microphone delivered
    /// audio, or a microphone that never did. The recording service keeps
    /// whatever audio was captured — saved to history as an interrupted
    /// take the user can transcribe later — so a cancel never loses words.
    pub(crate) fn cancel_recording(
        &mut self,
        cancelled_take: crate::activation::TakeId,
        reason: crate::activation::CancelReason,
        cx: &mut Context<Self>,
    ) {
        use crate::activation::CancelReason;
        self.end_playback_lease();
        // A cancelled take delivers nothing (#221).
        self.delivery_take_stopped();
        let Some(live) = self.recorder.take() else {
            return;
        };
        self.audio_upkeep.set_recording(false);
        // The stream and the engine lease leave with the take: nothing is
        // transcribed.
        let mut pumped = self.finish_stream_pump();
        pumped.stream = None;
        self.host.stream_endpoint = None;
        self.active_take = None;
        self.live_partial.clear();
        self.stream_degradation = None;
        self.levels = vec![0.06; 52];
        let staging = self.staging_cancelled(cx);
        // What a cancel that kept no audio says (one that kept audio says
        // so once it is in history).
        let empty_notice = match reason {
            CancelReason::Escape => {
                Some("Cancelled with Escape before any audio was captured.".to_string())
            }
            CancelReason::NoAudioYet => Some(
                "Stopped before the microphone delivered any audio; nothing was recorded."
                    .to_string(),
            ),
            CancelReason::MicStalled | CancelReason::InputLost => None,
        };
        match reason {
            CancelReason::MicStalled => {
                self.error = Some(format!(
                    "The microphone delivered no audio within {} seconds, so the take was \
                     stopped. Check that the input device is connected and not muted.",
                    crate::activation::START_STALL.as_secs()
                ));
            }
            CancelReason::InputLost => {
                let (what, problem) = self.mic.interruption.take().unwrap_or_else(|| {
                    (
                        "The microphone stopped mid-recording".to_string(),
                        starling_dictation::microphone::InputProblem::Unavailable {
                            device: "The microphone".to_string(),
                            detail: "it stopped during the last recording".to_string(),
                        },
                    )
                });
                self.report_input_problem(
                    problem,
                    format!(
                        "{what}. Check the microphone (Settings → Microphone can test it or pick \
                         another), then record again."
                    ),
                );
            }
            CancelReason::Escape | CancelReason::NoAudioYet => {}
        }
        self.finish_live_take(
            live,
            cancelled_take,
            pumped,
            crate::remote_take::FinishKind::Cancel {
                saved_notice: Self::cancel_saved_notice(reason),
                empty_notice,
                staging,
            },
        );
        cx.notify();
    }

    /// The target travels with the take through here (#363); the lease it
    /// may hold is dropped when the transcription job finishes.
    #[allow(clippy::too_many_arguments)]
    pub fn save_and_transcribe(
        &mut self,
        wav: Arc<Vec<u8>>,
        stream: Option<LiveStream>,
        stopped_at: Option<Instant>,
        staging: Option<u64>,
        delivery: Option<crate::delivery::Capture>,
        target: TakeTarget,
        cx: &mut Context<Self>,
    ) {
        // Deliberately no `self.error = None` here: every caller clears the
        // slot at its own start, and a capture-path fault (e.g. a journal
        // fsync failure that froze acknowledgment) must survive until
        // something replaces it — a clean save is not a reason to un-say
        // it (I1 phase 2 storage-fault honesty).
        let Some(store) = self.store.clone() else {
            let reason = self
                .store_error
                .clone()
                .unwrap_or_else(|| "session store unavailable".to_string());
            self.stash_unsaved(wav, &format!(
                "Local storage failed: {reason} Keep this window open and download the unsaved WAV to recover it."
            ));
            if let Some(token) = staging {
                self.staging_save_failed(token, cx);
            }
            if let Some(stopped_at) = stopped_at {
                self.overlay.model.save_failed(stopped_at, Instant::now());
            }
            cx.notify();
            return;
        };
        cx.spawn(async move |this, cx| {
            let job_wav = wav.clone();
            let create_store = store.clone();
            let created = cx
                .background_spawn(async move {
                    create_store.save_capture(job_wav)
                })
                .await;
            match created {
                Ok(saved) => {
                    refresh_sessions(&this, &store, cx).await;
                    this.update(cx, |app, cx| {
                        // `saved.wav` is the audio to transcribe: the
                        // stored evidence itself when the journal was
                        // adopted (the same bytes a retry loads), the
                        // caller's WAV otherwise.
                        if let Some(stopped_at) = stopped_at {
                            app.stop_instants.insert(saved.id.clone(), stopped_at);
                            app.overlay.model.take_saved(stopped_at, &saved.id);
                        }
                        if let Some(token) = staging {
                            app.bind_staging(token, &saved.id);
                        }
                        app.bind_delivery(delivery, &saved.id);
                        app.transcribe_with_stream(
                            saved.id, saved.wav, stream, target, None, false, cx,
                        );
                    })
                    .ok();
                }
                Err(err) => {
                    this.update(cx, |app, cx| {
                        app.stash_unsaved(wav, &format!(
                            "Local storage failed: {err} Keep this window open and download the unsaved WAV to recover it."
                        ));
                        if let Some(token) = staging {
                            app.staging_save_failed(token, cx);
                        }
                        if let Some(stopped_at) = stopped_at {
                            app.overlay.model.save_failed(stopped_at, Instant::now());
                        }
                        cx.notify();
                    })
                    .ok();
                }
            }
        })
        .detach();
    }

    /// The never-drop-audio fallback shared by every persist path (I1
    /// phase 2 folds the salvage paths into it too): when the session
    /// store cannot take the WAV, it lands in the unsaved list with
    /// `message` surfaced, so the only copy is never discarded.
    pub(crate) fn stash_unsaved(&mut self, wav: Arc<Vec<u8>>, message: &str) {
        self.unsaved.push(UnsavedWav {
            id: format!("unsaved-{}", storage::now_iso()),
            wav,
            created_at: storage::now_iso(),
        });
        self.error = Some(message.to_string());
    }

    /// Transcribe a saved take again on `target` (#356): a new attempt
    /// beside the earlier ones, never typed into an editor — the take's
    /// own delivery is gone with its first job, and a retried transcript
    /// is offered for Copy / Paste last instead. `pin` holds the take's
    /// audio until the attempt is marked started (#342).
    fn retry_on(
        &mut self,
        id: String,
        wav: Arc<Vec<u8>>,
        target: TakeTarget,
        pin: AudioPin,
        cx: &mut Context<Self>,
    ) {
        self.forget_delivery(&id);
        self.transcribe_with_stream(id, wav, None, target, Some(pin), true, cx);
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn transcribe_with_stream(
        &mut self,
        id: String,
        wav: Arc<Vec<u8>>,
        stream: Option<LiveStream>,
        mut target: TakeTarget,
        pin: Option<AudioPin>,
        retry: bool,
        cx: &mut Context<Self>,
    ) {
        if self.active_ids.contains(&id) {
            return;
        }
        let Some(store) = self.store.clone() else {
            return;
        };
        self.active_ids.insert(id.clone());
        self.selected_id = Some(id.clone());
        self.error = None;
        cx.notify();

        // #363: a take that started before the engine was ready gets one
        // fresh resolution here — the engine may have come up while it
        // recorded. Still no engine and the job fails honestly below.
        if target.needs_engine() {
            target = self.resolve_take_target();
        }
        let endpoint = target.endpoint().to_string();
        let model = target.model().to_string();
        // The attempt row's backend label (v2 keeps it on the recognition
        // attempt): the target's provenance — `engine:<model_id>` for a
        // built-in take, `openai:<model>` for a manual one.
        let backend = target.provenance();
        let builtin = target.is_builtin();
        let unready = target.needs_engine();
        let store_for_job = store.clone();

        cx.spawn(async move |this, cx| {
            // The take's engine hold (#363): nothing reads the target
            // anymore — endpoint/model/provenance were copied out above —
            // but owning it here means the lease is dropped only when
            // this job finishes (success, failure, or session gone),
            // releasing a draining engine exactly then. (If it were left
            // behind in this function's scope, a model switch mid-take
            // could stop the old engine while this take still transcribes
            // against it.)
            let _take_target = target;
            enum Outcome {
                Success,
                Failure { message: String, class: FailureClass },
                /// The session was deleted while this job was in flight
                /// (R05): not a failure — nothing to probe, nothing to
                /// record in history, just keep the audio and surface.
                SessionGone,
            }
            let mut outcome = Outcome::Success;
            // #356: a blank retry is kept as a result but leaves the
            // take's earlier words shown; nothing follows up on it.
            let mut kept_earlier_text = false;

            let marked = {
                let id = id.clone();
                let backend = backend.clone();
                let store = store_for_job.clone();
                cx.background_spawn(async move {
                    let marked = store.mark_attempt(&id, &backend);
                    // The started attempt holds the audio from here on.
                    drop(pin);
                    marked
                })
                .await
            };
            match marked {
                Ok(_) => {
                    refresh_sessions(&this, &store_for_job, cx).await;
                    let attempt = {
                        let wav = wav.clone();
                        let id = id.clone();
                        cx.background_spawn(async move {
                            // #363: no engine to talk to (builtin, never
                            // became ready): fail the request with the
                            // honest sentence — Local class, so nothing
                            // prompts a probe.
                            let client = if unready {
                                Err(ClientError::Input(engine_not_ready_message()))
                            } else {
                                StarlingClient::new(&endpoint, &model).and_then(|client| {
                                    client.with_timeout_ms(request_timeout_ms(wav.len()))
                                })
                            }?;
                            // The stream failure reason is kept (and logged
                            // when the batch fallback also fails): silently
                            // re-uploading after a dead stream made
                            // stream-mode regressions undiagnosable.
                            let mut stream_failure = None;
                            if let Some(stream) = stream {
                                // Guarding on is_closed first collapses the
                                // dead-stream case to an immediate failure
                                // instead of a commit that can only time
                                // out into the fallback.
                                if !stream.is_closed() && stream.commit() {
                                    match stream.final_result() {
                                        Ok(result) => return Ok(result),
                                        Err(err) => stream_failure = Some(err),
                                    }
                                } else {
                                    stream_failure =
                                        Some("stream closed before commit".to_string());
                                }
                            }
                            // Shares the upload buffer with the client by
                            // reference count (issue #235): the WAV bytes
                            // are never duplicated for this request.
                            let outcome = client.transcribe(wav, &id);
                            if let (Err(err), Some(stream_err)) = (&outcome, stream_failure) {
                                eprintln!(
                                    "STARLING stream failed ({stream_err}); \
                                     batch upload fallback also failed: {err}"
                                );
                            }
                            outcome
                        })
                        .await
                    };
                    match attempt {
                        Ok(result) => {
                            // R05: re-check that the session still exists
                            // before writing the transcript.
                            // `remove_session` refuses to run while the id
                            // is in `active_ids`, but that guard cannot
                            // cover a DictationSession created before
                            // `transcribe` ran (the create→transcribe
                            // window) — so the user's delete can land
                            // while the request is in flight. A NotFound
                            // from either the re-check or the write itself
                            // is that delete race: never a job failure,
                            // never a resurrection.
                            let saved = {
                                let id = id.clone();
                                let store = store_for_job.clone();
                                cx.background_spawn(async move {
                                    let session_found = store.exists(&id)?;
                                    let blank = result.text.trim().is_empty();
                                    let write_error = if session_found {
                                        store.save_transcript(&id, result).err()
                                    } else {
                                        None
                                    };
                                    // Unreadable afterwards: assume the
                                    // earlier words still show — a blank
                                    // result has nothing to follow up on.
                                    let kept_earlier = retry
                                        && blank
                                        && session_found
                                        && write_error.is_none()
                                        && store.latest_raw(&id).map_or(true, |shown| {
                                            shown.is_some_and(|(_, text)| !text.trim().is_empty())
                                        });
                                    Ok::<_, storage::StorageError>((
                                        session_found,
                                        write_error,
                                        kept_earlier,
                                    ))
                                })
                                .await
                            };
                            match saved {
                                Ok((session_found, write_error, kept_earlier)) => {
                                    kept_earlier_text = kept_earlier;
                                    match save_race_decision(
                                        session_found,
                                        write_error.as_ref(),
                                    ) {
                                        SaveRaceDecision::Written => {}
                                        SaveRaceDecision::SessionDeleted => {
                                            outcome = Outcome::SessionGone;
                                        }
                                        SaveRaceDecision::Failed(message) => {
                                            // A local storage failure says
                                            // nothing about reachability
                                            // (R13).
                                            outcome = Outcome::Failure {
                                                message,
                                                class: FailureClass::Local,
                                            };
                                        }
                                    }
                                }
                                Err(err) => {
                                    // The re-check itself failed — an I/O
                                    // error, not a delete.
                                    outcome = Outcome::Failure {
                                        message: err.to_string(),
                                        class: FailureClass::Local,
                                    };
                                }
                            }
                        }
                        Err(err) => {
                            // R13: the client error is classified while it
                            // is still typed — string matching later would
                            // be fragile. #363: a transport failure on a
                            // built-in take means the sidecar went away;
                            // the message says so instead of naming a
                            // socket error.
                            let class = failure_class(&err);
                            outcome = Outcome::Failure {
                                message: builtin_take_failure_message(builtin, class, &err),
                                class,
                            };
                        }
                    }
                }
                Err(err) => {
                    // The delete may have landed before the first write
                    // too (R05): a session that existed when the job
                    // started but is gone by `mark_attempt` is the same
                    // race, not a storage fault. (`Written` is unreachable
                    // here — the decision was handed an error.)
                    outcome = match save_race_decision(true, Some(&err)) {
                        SaveRaceDecision::Failed(message) => Outcome::Failure {
                            message,
                            class: FailureClass::Local,
                        },
                        SaveRaceDecision::SessionDeleted | SaveRaceDecision::Written => {
                            Outcome::SessionGone
                        }
                    };
                }
            }

            let session_gone = matches!(outcome, Outcome::SessionGone);
            let job_failure = match outcome {
                Outcome::Success | Outcome::SessionGone => None,
                Outcome::Failure { message, class } => Some((message, class)),
            };

            // Record a failure in local history; keep that write's own
            // error, if any, to append to the surfaced message.
            let history_failure = match job_failure.as_ref() {
                Some((failure, _class)) => {
                    let store = store_for_job.clone();
                    let id_for_failure = id.clone();
                    let failure_message = failure.clone();
                    cx.background_spawn(async move {
                        if store.exists(&id_for_failure)? {
                            store.save_failure(&id_for_failure, &failure_message)?;
                        }
                        Ok::<(), storage::StorageError>(())
                    })
                    .await
                    .err()
                    .map(|err| err.to_string())
                }
                None => None,
            };

            match finished_job(
                job_failure.as_ref().map(|(message, class)| (message.as_str(), *class)),
                history_failure.as_deref(),
            ) {
                FinishedJob::Failed { message, class } => {
                    this.update(cx, |app, cx| {
                        app.error = Some(message);
                        // R13: a transport-classified failure is the one
                        // job outcome allowed to ask for a connection
                        // re-check — and even it only triggers the probe.
                        // `check_health` stays the single writer of
                        // connection state, so a local failure still
                        // cannot flip the badge. The probe is Diagnostic
                        // (#207): it corrects the badge (a transport
                        // failure whose probe succeeds shows the server
                        // recovered) but never touches `app.error`, so
                        // the explanation for the missing transcript is
                        // not wiped ~5 s after it appeared. #363: a
                        // built-in take's endpoint is the engine's own —
                        // its failures never probe the manual endpoint.
                        if class == FailureClass::Transport && !builtin {
                            app.check_health(
                                HealthCheckPurpose::Diagnostic,
                                app.endpoint.clone(),
                                cx,
                            );
                        }
                        cx.notify();
                    })
                    .ok();
                }
                // R03: a finished job applies no connection state. A success
                // says nothing about readiness or busyness, and a failure may
                // be a local storage error rather than a dead server; only
                // health probes (`check_health`) write connection state.
                FinishedJob::Saved => {}
            }

            if session_gone {
                // R05: the transcript had nowhere to land because its
                // session was deleted mid-flight. Keep the job's audio
                // recoverable in the unsaved list and surface what
                // happened — never drop it silently. This stash path
                // deliberately performs no journal operation (R21): the
                // confirmed delete that won the race already quarantined
                // the capture journal via `remove_session`, and the stashed
                // in-memory WAV is the kept copy the user can download.
                let message = session_deleted_message();
                this.update(cx, |app, cx| {
                    app.stash_unsaved(wav, &message);
                    cx.notify();
                })
                .ok();
            }

            let transcribed = job_failure.is_none() && !session_gone;
            this.update(cx, |app, cx| {
                app.active_ids.remove(&id);
                // Transcribed or its failure recorded: a take the host
                // handed this window is handled. A failure history could
                // not record leaves the take unsettled: it goes back to
                // the host, which keeps it for the next try.
                if history_failure.is_some() {
                    app.take_handed_back(&id);
                } else {
                    app.take_handled(&id);
                }
                if !transcribed {
                    // No processing will run for this take, and nothing
                    // is typed for it: a retry is a new, explicit job.
                    app.stop_instants.remove(&id);
                    app.forget_delivery(&id);
                    app.staging_transcription_failed(&id, cx);
                }
                cx.notify();
            })
            .ok();
            refresh_sessions(&this, &store_for_job, cx).await;
            // Raw text is in history now; the active mode's processing
            // follows as a proposal (#295). A blank retry changed no
            // shown text: no draft is dropped, nothing is reprocessed or
            // offered again.
            if transcribed && !kept_earlier_text {
                this.update(cx, |app, cx| {
                    app.after_transcription(id.clone(), cx);
                    if retry {
                        app.offer_retried_text(&id, cx);
                    }
                })
                .ok();
            }
        })
        .detach();
    }

    pub fn import_audio(&mut self, cx: &mut Context<Self>) {
        self.error = None;
        let receiver = cx.prompt_for_paths(PathPromptOptions {
            files: true,
            directories: false,
            multiple: false,
            prompt: None,
        });
        cx.spawn(async move |this, cx| {
            if let Ok(Ok(Some(paths))) = receiver.await {
                if let Some(path) = paths.first().cloned() {
                    let prepared = cx
                        .background_spawn(async move {
                            let bytes = std::fs::read(&path)
                                .map_err(|err| format!("Could not read the file: {err}"))?;
                            audio::prepare_wav_16k(&bytes).map_err(|err| err.to_string())
                        })
                        .await;
                    match prepared {
                        Ok(prepared) => {
                            this.update(cx, |app, cx| {
                                // A file import is a new job: it resolves a
                                // fresh target at the moment it starts
                                // (#363) — whatever the engine serves now.
                                let target = app.resolve_take_target();
                                app.save_and_transcribe(
                                    Arc::new(prepared.wav),
                                    None,
                                    None,
                                    None,
                                    None,
                                    target,
                                    cx,
                                );
                            })
                            .ok();
                        }
                        Err(message) => {
                            this.update(cx, |app, cx| {
                                app.error = Some(message);
                                cx.notify();
                            })
                            .ok();
                        }
                    }
                }
            }
        })
        .detach();
    }

    /// Retry the selected take with whatever the app transcribes with
    /// now (#363: a retry resolves a fresh target when it starts).
    pub fn retry_selected(&mut self, cx: &mut Context<Self>) {
        self.retry_selected_with(RetryWith::Current, cx);
    }

    /// Retry the selected take with `with` (#356), from its stored audio:
    /// no re-recording, and a success adds a result beside the earlier
    /// ones.
    pub(crate) fn retry_selected_with(&mut self, with: RetryWith, cx: &mut Context<Self>) {
        let Some(session) = self.selected() else {
            return;
        };
        let id = session.id.clone();
        if self.active_ids.contains(&id) || self.is_deleting(&id) {
            return;
        }
        self.retry_menu = None;
        if with == RetryWith::Server && self.endpoint.trim().is_empty() {
            // A choice that cannot run replaces nothing: a retry still
            // waiting for its model carries on.
            self.error = Some(
                "No server is set up: add its endpoint in Settings → Engine, then retry."
                    .to_string(),
            );
            cx.notify();
            return;
        }
        // A newer choice replaces a retry still loading its audio or
        // waiting for its model.
        self.pending_retry = None;
        self.retry_seq += 1;
        let seq = self.retry_seq;
        // G02: history holds metadata only — fetch this one recording's
        // audio on demand (a damaged record surfaces its reason here).
        let Some(store) = self.store.clone() else {
            return;
        };
        cx.spawn(async move |this, cx| {
            // #342: pinned from before the load until the retry's attempt
            // is marked started, so upkeep cannot remove the audio between
            // — including while the engine switches to the retry's model.
            let loaded = {
                let store = store.clone();
                let id = id.clone();
                cx.background_spawn(async move {
                    let pin = store.pin_audio(&id);
                    store
                        .audio_wav(&id)
                        .map(|wav| wav.map(|wav| (wav, pin)))
                })
                .await
            };
            this.update(cx, |app, cx| match loaded {
                // Superseded by a newer choice, or the take was deleted
                // while its audio loaded: the pin drops, nothing runs.
                _ if app.retry_seq != seq || app.retry_target_gone(&id) => {}
                Ok(Some((wav, pin))) => match with {
                    RetryWith::Current => {
                        let target = app.resolve_take_target();
                        app.retry_on(id, wav, target, pin, cx);
                    }
                    RetryWith::Server => {
                        let target = TakeTarget::manual(app.endpoint.clone(), app.model.clone());
                        app.retry_on(id, wav, target, pin, cx);
                    }
                    RetryWith::Model(model_id) => {
                        app.retry_after_switch(id, model_id, seq, wav, pin, cx);
                    }
                },
                Ok(None) => {
                    app.error = Some(format!("Recording {id} was not found."));
                    cx.notify();
                }
                Err(err) => {
                    app.error = Some(err.to_string());
                    cx.notify();
                }
            })
            .ok();
        })
        .detach();
    }

    /// Retry on an installed built-in model (#356): the engine switches
    /// to it the way Settings would (it stays the active model), and the
    /// retry runs once that model serves. Nothing is attempted — and the
    /// take is untouched — when the switch does not happen.
    fn retry_after_switch(
        &mut self,
        id: String,
        model_id: String,
        seq: u64,
        wav: Arc<Vec<u8>>,
        pin: AudioPin,
        cx: &mut Context<Self>,
    ) {
        let Some(engine) = self.engine.clone() else {
            self.error = Some(
                "The built-in engine is off (Settings → Engine uses your own server), so this \
                 model cannot transcribe. The recording is unchanged."
                    .to_string(),
            );
            cx.notify();
            return;
        };
        let serving = engine.lease().filter(|lease| lease.model_id() == model_id);
        if let Some(lease) = serving {
            self.retry_on(id, wav, TakeTarget::from_lease(lease), pin, cx);
            return;
        }
        let request = engine.activate(&model_id);
        let pending = PendingRetry {
            take_id: id.clone(),
            model_id: model_id.clone(),
            seq,
        };
        self.pending_retry = Some(pending.clone());
        cx.notify();
        let instance = self.engine_instance;
        let started = Instant::now();
        cx.spawn(async move |this, cx| {
            let mut job = Some((wav, pin));
            // Since when the engine has reported the model serving without
            // handing out a lease on it.
            let mut unleased_since: Option<Instant> = None;
            loop {
                cx.background_executor()
                    .timer(std::time::Duration::from_millis(250))
                    .await;
                let done = this.update(cx, |app, cx| {
                    if app.pending_retry.as_ref() != Some(&pending) {
                        // Replaced by another choice: this wait is over.
                        return true;
                    }
                    if app.engine_instance != instance {
                        // The engine itself was replaced or stopped (a
                        // mode switch): the model will not load for this
                        // retry, and its audio is released.
                        app.pending_retry = None;
                        app.error = Some(format!(
                            "The retry with {} was cancelled: the engine changed in \
                             Settings while the model loaded. The recording is unchanged.",
                            crate::views::drawer::provenance_label(&format!(
                                "engine:{model_id}"
                            ))
                        ));
                        cx.notify();
                        return true;
                    }
                    if app.retry_target_gone(&pending.take_id) {
                        // Deleted while the model loaded: nothing to retry.
                        app.pending_retry = None;
                        cx.notify();
                        return true;
                    }
                    match switch_progress(
                        &engine.snapshot(),
                        &model_id,
                        request,
                        started.elapsed(),
                    ) {
                        SwitchProgress::Waiting => false,
                        SwitchProgress::Ready => {
                            let Some(lease) =
                                engine.lease().filter(|lease| lease.model_id() == model_id)
                            else {
                                // A moment between the snapshot and the
                                // lease is fine; a lasting gap ends the
                                // retry visibly instead of waiting on.
                                let since = *unleased_since.get_or_insert_with(Instant::now);
                                let Some(reason) = unleased_retry_failure(since.elapsed()) else {
                                    return false;
                                };
                                app.pending_retry = None;
                                app.error = Some(format!(
                                    "Could not retry with {}: {reason} The recording is \
                                     unchanged.",
                                    crate::views::drawer::provenance_label(&format!(
                                        "engine:{model_id}"
                                    ))
                                ));
                                cx.notify();
                                return true;
                            };
                            app.pending_retry = None;
                            if let Some((wav, pin)) = job.take() {
                                app.retry_on(
                                    pending.take_id.clone(),
                                    wav,
                                    TakeTarget::from_lease(lease),
                                    pin,
                                    cx,
                                );
                            }
                            true
                        }
                        SwitchProgress::Failed(reason) => {
                            app.pending_retry = None;
                            app.error = Some(format!(
                                "Could not retry with {}: {reason} The recording is unchanged.",
                                crate::views::drawer::provenance_label(&format!(
                                    "engine:{model_id}"
                                ))
                            ));
                            cx.notify();
                            true
                        }
                    }
                });
                if done.unwrap_or(true) {
                    break;
                }
            }
        })
        .detach();
    }

    /// What a retry can run on right now (#356), for the drawer: every
    /// installed built-in model (the active one first) and the server
    /// from Settings when one is set up.
    pub(crate) fn retry_choices(&self) -> Vec<RetryChoice> {
        let mut choices = Vec::new();
        if let Some(snapshot) = self.engine.as_ref().map(|engine| engine.snapshot()) {
            let mut models: Vec<_> = snapshot
                .models
                .iter()
                .filter(|model| {
                    model.install == starling_dictation::engine::InstallState::Installed
                })
                .collect();
            models.sort_by_key(|model| !model.active);
            for model in models {
                choices.push(RetryChoice {
                    label: if model.active {
                        format!("{} (current)", model.label)
                    } else {
                        model.label.clone()
                    },
                    with: RetryWith::Model(model.id.clone()),
                    switches_engine: !model.active,
                });
            }
        }
        if !self.endpoint.trim().is_empty() {
            let model = if self.model.trim().is_empty() {
                "default model".to_string()
            } else {
                self.model.clone()
            };
            choices.push(RetryChoice {
                label: format!("{model} (your server)"),
                with: RetryWith::Server,
                switches_engine: false,
            });
        }
        choices
    }

    /// Whether a retry's take was deleted — confirmed (in flight) or
    /// already gone from history — so the retry must not run: it would
    /// only stash the deleted audio back as unsaved.
    fn retry_target_gone(&self, id: &str) -> bool {
        self.is_deleting(id) || !self.sessions.iter().any(|session| session.id == id)
    }

    /// Opens or closes the drawer's "Retry with" choices for `id`.
    pub(crate) fn toggle_retry_menu(&mut self, id: &str, cx: &mut Context<Self>) {
        self.retry_menu = match self.retry_menu.as_deref() {
            Some(open) if open == id => None,
            _ => Some(id.to_string()),
        };
        cx.notify();
    }
}

/// What a retry transcribes with (#356).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum RetryWith {
    /// Whatever the app transcribes with now.
    Current,
    /// An installed built-in model; the engine switches to it first.
    Model(String),
    /// The server from Settings (endpoint + model), in either mode.
    Server,
}

/// One "Retry with" entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct RetryChoice {
    pub label: String,
    pub with: RetryWith,
    /// Choosing it makes this model the engine's active one.
    pub switches_engine: bool,
}

/// A retry waiting for the engine to serve its model.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PendingRetry {
    pub take_id: String,
    pub model_id: String,
    /// The request's `retry_seq`: a newer request for the same take and
    /// model is a different wait.
    pub seq: u64,
}

/// Where an engine switch a retry waits on stands.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum SwitchProgress {
    Waiting,
    Ready,
    /// The switch will not happen; the sentence says why.
    Failed(String),
}

/// How long a retry waits for its model before giving up. Covers a
/// download-free load of the largest catalog model with room to spare.
const RETRY_SWITCH_CAP: std::time::Duration = std::time::Duration::from_secs(10 * 60);
/// How long the engine may take to pick up the retry's activation (its
/// command queue can be busy winding down an earlier switch) before the
/// retry gives up.
const RETRY_SWITCH_PICKUP: std::time::Duration = std::time::Duration::from_secs(30);

/// Why a retry whose model the engine reports serving gives up after
/// `unleased` without a lease on it — never, until
/// [`RETRY_SWITCH_PICKUP`] has passed.
fn unleased_retry_failure(unleased: std::time::Duration) -> Option<&'static str> {
    (unleased >= RETRY_SWITCH_PICKUP).then_some("the model loaded but could not be used.")
}

/// Read an engine snapshot for a retry waiting on `model_id`, whose
/// switch is the engine's activation `request`.
pub(crate) fn switch_progress(
    snapshot: &starling_dictation::engine::EngineSnapshot,
    model_id: &str,
    request: u64,
    waited: std::time::Duration,
) -> SwitchProgress {
    use starling_dictation::engine::{EnginePhase, SwapDecision};
    let serving = snapshot
        .active
        .as_ref()
        .is_some_and(|active| active.model_id == model_id);
    if serving && snapshot.phase == EnginePhase::Ready && snapshot.switch.is_none() {
        return SwitchProgress::Ready;
    }
    if waited >= RETRY_SWITCH_CAP {
        return SwitchProgress::Failed("the model did not finish loading in time.".to_string());
    }
    if snapshot.activations_handled < request {
        // Not taken up yet: any switch, refusal or error in view is left
        // over from an earlier request and says nothing about this one.
        return if waited < RETRY_SWITCH_PICKUP {
            SwitchProgress::Waiting
        } else {
            SwitchProgress::Failed("the engine did not start the switch in time.".to_string())
        };
    }
    // From here on the snapshot answers this request or a later intent.
    if let Some(switch) = &snapshot.switch {
        return if switch.target_model_id == model_id {
            SwitchProgress::Waiting
        } else {
            SwitchProgress::Failed("another model switch replaced it.".to_string())
        };
    }
    if snapshot.activations_handled > request && !serving {
        return SwitchProgress::Failed("another model switch replaced it.".to_string());
    }
    if let Some(SwapDecision::Refused { .. }) = snapshot.pending_decision {
        return SwitchProgress::Failed(
            "there is not enough free memory to load that model.".to_string(),
        );
    }
    if let EnginePhase::Failed(failure) = &snapshot.phase {
        return SwitchProgress::Failed(format!("{failure}"));
    }
    if serving {
        // Warming or restarting after the cutover.
        return SwitchProgress::Waiting;
    }
    SwitchProgress::Failed(
        snapshot
            .last_error
            .clone()
            .unwrap_or_else(|| "the engine did not switch to it.".to_string()),
    )
}

/// The request timeout for one upload (#356): the client's 180 s
/// default, plus the take's own length, capped at the client's 10-minute
/// limit — a long take is not timed out by a budget sized for short
/// ones, and a hung engine still fails in bounded time.
pub(crate) fn request_timeout_ms(wav_bytes: usize) -> u64 {
    const BASE_MS: u64 = 180_000;
    const MAX_MS: u64 = 600_000;
    // 16 kHz mono PCM16: 32 000 bytes a second, after the 44-byte header.
    let audio_ms = (wav_bytes.saturating_sub(44) as u64) / 32;
    (BASE_MS + audio_ms).min(MAX_MS)
}

/// Show what a recovery pass found beside whatever the banners already
/// say: the pass runs while the app is in use, and an error or notice
/// the user has not seen yet is never replaced by it.
pub(crate) fn add_recovery_messages(app: &mut StarlingApp, error: String, notice: String) {
    fn joined(existing: Option<String>, added: String) -> Option<String> {
        match existing {
            _ if added.is_empty() => existing,
            Some(existing) if !existing.is_empty() => Some(format!("{existing} {added}")),
            _ => Some(added),
        }
    }
    app.error = joined(app.error.take(), error);
    app.recovery_notice = joined(app.recovery_notice.take(), notice);
}

pub(crate) async fn refresh_sessions(
    this: &WeakEntity<StarlingApp>,
    store: &Store,
    cx: &mut AsyncApp,
) {
    let store = store.clone();
    // G02: metadata-only listing — good and orphan-recovered records arrive
    // as summaries, damaged ones flagged with their reason.
    let result = cx.background_spawn(async move { store.list() }).await;
    match result {
        Ok(list) => {
            this.update(cx, |app, cx| {
                app.apply_sessions(list);
                if let Some(id) = app.selected_id.clone() {
                    app.load_processing(id, cx);
                }
                cx.notify();
            })
            .ok();
        }
        Err(err) => {
            this.update(cx, |app, cx| {
                app.error = Some(err.to_string());
                cx.notify();
            })
            .ok();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_manual_target_keeps_the_openai_provenance_label() {
        // #363: manual takes bind endpoint+model at start and label
        // their attempts exactly as before — `openai:<model>`.
        let target = TakeTarget::manual(
            "http://10.0.0.5:8181".to_string(),
            "whisper-large-v3".to_string(),
        );
        assert_eq!(target.endpoint(), "http://10.0.0.5:8181");
        assert_eq!(target.model(), "whisper-large-v3");
        assert_eq!(target.provenance(), "openai:whisper-large-v3");
        assert!(!target.is_builtin());
        assert!(!target.needs_engine());
    }

    #[test]
    fn an_unready_builtin_target_has_no_endpoint_and_needs_an_engine() {
        // Recording started with no ready engine: the take proceeds, but
        // the target carries nothing to send to — the job may re-resolve
        // once, then fails honestly.
        let target = TakeTarget::builtin_unready();
        assert!(target.is_builtin());
        assert!(target.needs_engine());
        assert_eq!(target.endpoint(), "");
        assert_eq!(target.provenance(), "engine:none");
    }

    #[test]
    fn a_transport_failure_on_a_builtin_take_names_the_stopped_engine() {
        // #363: connection-refused against the engine's own endpoint is
        // the sidecar going away, not a flaky network — and the sentence
        // says the recording is saved and retry is the path.
        let refused = ClientError::Transport(
            "error sending request for url (http://127.0.0.1:51309/v1/audio/transcriptions): \
             Connection refused (os error 111)"
                .to_string(),
        );
        let message = builtin_take_failure_message(true, FailureClass::Transport, &refused);
        assert_eq!(
            message,
            "The built-in engine stopped while transcribing; the recording is saved — retry \
             when the engine is ready."
        );
        // The same error on a manual take keeps the client's message (the
        // manual prober owns that story), and non-transport builtin
        // failures keep theirs — the server answered, the engine lives.
        assert_eq!(
            builtin_take_failure_message(false, FailureClass::Transport, &refused),
            refused.to_string()
        );
        let http = ClientError::Http {
            status: 500,
            message: "model is loading".to_string(),
        };
        assert_eq!(
            builtin_take_failure_message(true, FailureClass::Local, &http),
            http.to_string()
        );
    }

    #[test]
    fn the_engine_not_ready_message_points_at_retry() {
        let message = engine_not_ready_message();
        assert!(message.contains("not ready"), "{message}");
        assert!(message.contains("recording is saved"), "{message}");
        assert!(message.contains("retry"), "{message}");
    }

    #[test]
    fn a_local_storage_failure_surfaces_as_an_error_not_a_connection_change() {
        // R03: a full disk in `save_transcript` fails the job with a local
        // storage error. The job's only global effect is the error banner —
        // `FinishedJob` carries no connection state at all, so the app can
        // no longer flip to `Connection::Offline` (or `Ready`) from a job
        // outcome; only a health probe may. R13 adds that this class also
        // never prompts a probe: a full disk says nothing about the server.
        let job = finished_job(
            Some((
                "Local storage failed: No space left on device (os error 28)",
                FailureClass::Local,
            )),
            None,
        );
        match job {
            FinishedJob::Failed { message, class } => {
                assert_eq!(
                    message,
                    "Local storage failed: No space left on device (os error 28)"
                );
                assert_eq!(class, FailureClass::Local);
            }
            FinishedJob::Saved => panic!("a failed job must surface its error"),
        }
    }

    #[test]
    fn a_failed_history_update_is_appended_to_the_surfaced_error() {
        let job = finished_job(
            Some(("request failed", FailureClass::Local)),
            Some("history is read-only"),
        );
        match job {
            FinishedJob::Failed { message, .. } => assert_eq!(
                message,
                "request failed. Local history update also failed: history is read-only"
            ),
            FinishedJob::Saved => panic!("a failed job must surface its error"),
        }
    }

    #[test]
    fn a_successful_job_has_no_global_state_effects() {
        // Success says nothing about server readiness or busyness, so it
        // cannot set `Connection::Ready` from its outcome either.
        assert!(matches!(finished_job(None, None), FinishedJob::Saved));
    }

    #[test]
    fn transport_level_client_failures_classify_as_transport() {
        // R13: only failures that mean "could not reach the server" may
        // prompt a probe — connection refused, timeouts, unreachable
        // networks all arrive as these two variants.
        assert_eq!(
            failure_class(&ClientError::Transport(
                "error sending request for url (http://127.0.0.1:8181/v1/audio/transcriptions): Connection \
                 refused (os error 111)"
                    .to_string()
            )),
            FailureClass::Transport
        );
        assert_eq!(
            failure_class(&ClientError::Timeout(180_000)),
            FailureClass::Transport
        );
    }

    #[test]
    fn answered_or_purely_local_failures_do_not_prompt_a_probe() {
        // An HTTP error status or a blocked redirect proves something
        // answered on the endpoint; Input/Protocol failures never left
        // this machine. An oversized response (issue #235) proves the
        // server answered — deterministically wrong.
        assert_eq!(
            failure_class(&ClientError::Input("Invalid server endpoint.".to_string())),
            FailureClass::Local
        );
        assert_eq!(failure_class(&ClientError::Redirect(302)), FailureClass::Local);
        assert_eq!(
            failure_class(&ClientError::Http {
                status: 500,
                message: "model is loading".to_string()
            }),
            FailureClass::Local
        );
        assert_eq!(
            failure_class(&ClientError::Protocol("transcription")),
            FailureClass::Local
        );
        assert_eq!(
            failure_class(&ClientError::ResponseTooLarge(64)),
            FailureClass::Local
        );
    }

    #[test]
    fn a_transport_classified_job_failure_asks_for_a_probe_without_deciding_state() {
        // R13: the class tells the caller to probe; the job outcome itself
        // still carries no connection state — the probe's result is what
        // writes `Connection::Offline`/`Ready`/`Busy`.
        let job = finished_job(
            Some((
                "error sending request: connection refused",
                FailureClass::Transport,
            )),
            None,
        );
        match job {
            FinishedJob::Failed { message, class } => {
                assert_eq!(message, "error sending request: connection refused");
                assert_eq!(class, FailureClass::Transport);
            }
            FinishedJob::Saved => panic!("a failed job must surface its error"),
        }
    }

    #[test]
    fn a_history_failure_is_joined_with_a_sentence_boundary() {
        // R14: no bare-space run-on.
        assert_eq!(
            joined_failure("request failed", "history is read-only"),
            "request failed. Local history update also failed: history is read-only"
        );
        // Client messages that already end in a period must not double it.
        assert_eq!(
            joined_failure("Request timed out after 180000 ms.", "disk full"),
            "Request timed out after 180000 ms. Local history update also failed: disk full"
        );
    }

    #[test]
    fn a_missing_session_at_the_recheck_is_the_delete_race_not_a_failure() {
        // R05: the pre-write re-check came back empty — the delete won, so
        // no write is attempted. The decision is SessionDeleted (keep the
        // audio, surface), never a job failure with a raw NotFound.
        assert_eq!(save_race_decision(false, None), SaveRaceDecision::SessionDeleted);
    }

    #[test]
    fn a_not_found_from_the_write_itself_is_the_same_delete_race() {
        // The delete landed between the re-check and the write, or before
        // the job's first `mark_attempt` write (which hands the decision an
        // error for a session it expects to exist). Same decision.
        assert_eq!(
            save_race_decision(
                true,
                Some(&storage::StorageError::NotFound("session-id".to_string()))
            ),
            SaveRaceDecision::SessionDeleted
        );
    }

    #[test]
    fn a_present_session_with_a_clean_write_is_written() {
        assert_eq!(save_race_decision(true, None), SaveRaceDecision::Written);
    }

    #[test]
    fn a_genuine_storage_failure_is_still_a_failure_not_a_delete() {
        // A full disk is not a delete: the job must fail (Local class), not
        // claim the session vanished.
        let disk_full = storage::StorageError::Io(std::io::Error::other(
            "No space left on device (os error 28)",
        ));
        match save_race_decision(true, Some(&disk_full)) {
            SaveRaceDecision::Failed(message) => {
                assert!(message.contains("No space left on device"), "{message}");
            }
            other => panic!("a storage fault must fail the job, got {other:?}"),
        }
    }

    #[test]
    fn the_session_deleted_message_keeps_the_audio_and_surfaces_it() {
        // R05: never orphan data silently — the message must say the
        // session was deleted mid-transcription, that the audio is kept,
        // and offer the discard path for a deliberate delete.
        let message = session_deleted_message();
        assert!(message.contains("deleted while it was being transcribed"), "{message}");
        assert!(message.contains("transcript could not be saved"), "{message}");
        assert!(message.contains("audio is kept"), "{message}");
        assert!(message.contains("discard"), "{message}");
    }

    // ---- #356: retries, engine faults, focus safety ----------------------

    use std::io::{Read, Write};
    use std::net::{SocketAddr, TcpListener, TcpStream};
    use std::sync::Mutex;

    use starling_dictation::engine::{
        ActiveEngineView, EngineFailure, EnginePhase, EngineSnapshot, SwapDecision, SwitchStage,
        SwitchView,
    };
    use starling_dictation::settings::InsertionSettings;
    use starling_dictation::storage::{ListedRecord, SessionStatus, SessionSummary};
    use starling_insertion::testing::{FakeBackend, FakeTarget};
    use starling_insertion::Inserter;

    use crate::delivery::{DeliveryState, Failure};

    fn snapshot(active: Option<&str>, phase: EnginePhase) -> EngineSnapshot {
        EngineSnapshot {
            backend: None,
            phase,
            active: active.map(|model_id| ActiveEngineView {
                model_id: model_id.to_string(),
                endpoint: "http://127.0.0.1:1".to_string(),
                pid: 1,
                owned: true,
                device: None,
            }),
            switch: None,
            pending_decision: None,
            last_switch: None,
            models: Vec::new(),
            notices: Vec::new(),
            last_error: None,
            activations_handled: 0,
        }
    }

    fn switching_to(model_id: &str) -> Option<SwitchView> {
        Some(SwitchView {
            target_model_id: model_id.to_string(),
            stage: SwitchStage::Loading,
            started: Instant::now(),
        })
    }

    /// A snapshot that has taken up activation `handled`.
    fn handled(mut snapshot: EngineSnapshot, handled: u64) -> EngineSnapshot {
        snapshot.activations_handled = handled;
        snapshot
    }

    #[test]
    fn a_retry_waits_for_its_model_and_gives_up_honestly() {
        let secs = std::time::Duration::from_secs;
        // Serving the model: go.
        assert_eq!(
            switch_progress(&snapshot(Some("b"), EnginePhase::Ready), "b", 5, secs(1)),
            SwitchProgress::Ready
        );
        // The switch to it is running, or the command is not picked up yet.
        let mut loading = handled(snapshot(Some("a"), EnginePhase::Ready), 5);
        loading.switch = switching_to("b");
        assert_eq!(switch_progress(&loading, "b", 5, secs(30)), SwitchProgress::Waiting);
        assert_eq!(
            switch_progress(&snapshot(Some("a"), EnginePhase::Ready), "b", 5, secs(1)),
            SwitchProgress::Waiting
        );
        assert_eq!(
            switch_progress(&handled(snapshot(Some("b"), EnginePhase::Loading), 5), "b", 5, secs(30)),
            SwitchProgress::Waiting
        );
        // Everything else ends the wait with a reason.
        let mut replaced = handled(snapshot(Some("a"), EnginePhase::Ready), 5);
        replaced.switch = switching_to("c");
        assert!(matches!(
            switch_progress(&replaced, "b", 5, secs(5)),
            SwitchProgress::Failed(_)
        ));
        // A later activation finished on another model.
        assert!(matches!(
            switch_progress(&handled(snapshot(Some("c"), EnginePhase::Ready), 6), "b", 5, secs(5)),
            SwitchProgress::Failed(reason) if reason.contains("another model switch")
        ));
        let mut refused = handled(snapshot(Some("a"), EnginePhase::Ready), 5);
        refused.pending_decision = Some(SwapDecision::Refused {
            needed: 2,
            available: 1,
        });
        assert!(matches!(
            switch_progress(&refused, "b", 5, secs(1)),
            SwitchProgress::Failed(reason) if reason.contains("memory")
        ));
        assert!(matches!(
            switch_progress(
                &handled(
                    snapshot(None, EnginePhase::Failed(EngineFailure::LoadFailed("bad file".into()))),
                    5
                ),
                "b",
                5,
                secs(5)
            ),
            SwitchProgress::Failed(_)
        ));
        let mut ignored = handled(snapshot(Some("a"), EnginePhase::Ready), 5);
        ignored.last_error = Some("the model file failed verification".to_string());
        assert_eq!(
            switch_progress(&ignored, "b", 5, secs(5)),
            SwitchProgress::Failed("the model file failed verification".to_string())
        );
        assert!(matches!(
            switch_progress(&loading, "b", 5, RETRY_SWITCH_CAP),
            SwitchProgress::Failed(_)
        ));
    }

    /// Until the engine takes up the retry's own activation, an earlier
    /// switch's refusal, error or failed phase is not this retry's
    /// outcome: the wait goes on, bounded, and then fails with its own
    /// reason (#356).
    #[test]
    fn a_retry_ignores_outcomes_from_before_its_activation() {
        let secs = std::time::Duration::from_secs;
        let mut stale = handled(
            snapshot(Some("a"), EnginePhase::Failed(EngineFailure::LoadFailed("old".into()))),
            4,
        );
        stale.pending_decision = Some(SwapDecision::Refused {
            needed: 2,
            available: 1,
        });
        stale.last_error = Some("an earlier switch failed".to_string());
        stale.switch = switching_to("c");
        assert_eq!(switch_progress(&stale, "b", 5, secs(1)), SwitchProgress::Waiting);
        assert_eq!(
            switch_progress(&stale, "b", 5, RETRY_SWITCH_PICKUP - secs(1)),
            SwitchProgress::Waiting
        );
        assert_eq!(
            switch_progress(&stale, "b", 5, RETRY_SWITCH_PICKUP),
            SwitchProgress::Failed("the engine did not start the switch in time.".to_string())
        );
        // Taken up: the engine's own state for this request decides.
        let mut taken = handled(snapshot(Some("a"), EnginePhase::Ready), 5);
        taken.switch = switching_to("b");
        assert_eq!(switch_progress(&taken, "b", 5, secs(40)), SwitchProgress::Waiting);
    }

    #[test]
    fn long_takes_get_a_longer_but_bounded_request_timeout() {
        assert_eq!(request_timeout_ms(44), 180_000);
        // Ten seconds of 16 kHz PCM16.
        assert_eq!(request_timeout_ms(44 + 320_000), 190_000);
        // A ten-minute take: the client's own ceiling.
        assert_eq!(request_timeout_ms(44 + 600 * 32_000), 600_000);
    }

    /// What the fake transcription server does with one request.
    enum Reply {
        Status(&'static str),
        /// Reads the request and hangs up without an answer — a crashed
        /// engine.
        HangUp,
        Text(&'static str),
    }

    /// A transcription server scripted per request; anything else (the
    /// health probe a transport failure prompts) answers without using up
    /// a reply.
    fn fake_server(replies: Vec<Reply>) -> (SocketAddr, Arc<Mutex<usize>>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");
        let served = Arc::new(Mutex::new(0usize));
        let count = served.clone();
        let mut replies = std::collections::VecDeque::from(replies);
        std::thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                let mut stream = stream;
                let path = read_request(&mut stream);
                if path != "/v1/audio/transcriptions" {
                    // Health probes after a transport failure.
                    respond(&mut stream, "200 OK", r#"{"status":"ok"}"#);
                    continue;
                }
                *count.lock().unwrap() += 1;
                match replies.pop_front() {
                    Some(Reply::Status(status)) => respond(&mut stream, status, r#"{"error":"boom"}"#),
                    Some(Reply::HangUp) | None => drop(stream),
                    Some(Reply::Text(text)) => {
                        respond(&mut stream, "200 OK", &format!(r#"{{"text":"{text}"}}"#))
                    }
                }
            }
        });
        (addr, served)
    }

    fn read_request(stream: &mut TcpStream) -> String {
        let mut buffer = Vec::new();
        let mut chunk = [0u8; 8192];
        let head_end = loop {
            let read = stream.read(&mut chunk).unwrap_or(0);
            if read == 0 {
                return String::new();
            }
            buffer.extend_from_slice(&chunk[..read]);
            if let Some(at) = buffer.windows(4).position(|window| window == b"\r\n\r\n") {
                break at;
            }
        };
        let head = String::from_utf8_lossy(&buffer[..head_end]).to_string();
        let length = head
            .lines()
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                name.eq_ignore_ascii_case("content-length")
                    .then(|| value.trim().parse::<usize>().ok())
                    .flatten()
            })
            .unwrap_or(0);
        // The multipart upload streams: a chunked body ends with its
        // zero-length chunk.
        let chunked = head.to_ascii_lowercase().contains("transfer-encoding: chunked");
        let mut body = buffer[head_end + 4..].to_vec();
        loop {
            let complete = if chunked {
                body.ends_with(b"0\r\n\r\n")
            } else {
                body.len() >= length
            };
            if complete {
                break;
            }
            let read = stream.read(&mut chunk).unwrap_or(0);
            if read == 0 {
                break;
            }
            body.extend_from_slice(&chunk[..read]);
        }
        head.split_whitespace().nth(1).unwrap_or_default().to_string()
    }

    fn respond(stream: &mut TcpStream, status: &str, body: &str) {
        let _ = write!(
            stream,
            "HTTP/1.1 {status}\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\
             connection: close\r\n\r\n{body}",
            body.len()
        );
        let _ = stream.flush();
    }

    fn scratch(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("starling-356-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("scratch");
        dir
    }

    fn one_second_wav() -> Arc<Vec<u8>> {
        let pcm = audio::PcmAudio {
            samples: (0..16_000).map(|i| ((i as f32) * 0.02).sin() * 0.2).collect(),
            sample_rate: 16_000,
            channels: 1,
        };
        Arc::new(audio::encode_wav_16k(&pcm).expect("wav"))
    }

    fn summary(store: &Store, id: &str) -> SessionSummary {
        store
            .list()
            .expect("list")
            .into_iter()
            .find_map(|record| match record {
                ListedRecord::Session(summary) if summary.id == id => Some(summary),
                _ => None,
            })
            .expect("listed")
    }

    fn retry(app: &gpui::Entity<StarlingApp>, cx: &mut gpui::TestAppContext, with: RetryWith) {
        app.update(cx, |app, cx| app.retry_selected_with(with, cx));
        cx.run_until_parked();
    }

    /// The engine-client fault matrix on a saved take (#356): an error
    /// status, a server that hangs up mid-request (a crashed engine), and
    /// an empty answer each keep the audio and the take's history; a
    /// successful retry adds a result beside the earlier one, and nothing
    /// is ever typed into another app — the text is offered for Copy /
    /// Paste last instead.
    #[gpui::test]
    fn engine_faults_keep_the_take_and_a_retry_adds_a_result_without_typing(
        cx: &mut gpui::TestAppContext,
    ) {
        let root = scratch("faults");
        let store = Store::at_test_root(&root);
        let wav = one_second_wav();
        let id = store.save_capture(wav.clone()).expect("save").id;
        let stored = store.audio_wav(&id).expect("load").expect("present");
        store.mark_attempt(&id, "engine:model-a").expect("begin");
        store
            .save_transcript(
                &id,
                storage::TranscriptionResult {
                    text: "first words".to_string(),
                    segments: Vec::new(),
                    duration_seconds: None,
                    request_id: None,
                },
            )
            .expect("first");

        let (addr, served) = fake_server(vec![
            Reply::Status("500 Internal Server Error"),
            Reply::HangUp,
            Reply::Text(" "),
            Reply::Text("second words"),
        ]);
        let fake = Arc::new(FakeBackend::new());
        fake.focus(FakeTarget::named("Editor", "notes.txt"));
        let inserter = Arc::new(Inserter::with_backends(vec![Box::new(fake.clone())]));
        let app = cx.new(|cx| {
            let mut app = StarlingApp::for_test(Some(store.clone()), cx);
            app.endpoint = format!("http://{addr}");
            app.model = "fake-model".to_string();
            app.delivery = DeliveryState::new(inserter, InsertionSettings::default());
            app.apply_sessions(store.list().expect("list"));
            app.selected_id = Some(id.clone());
            // The take was dictated into the editor: its first delivery
            // capture is still bound, the shape a retry must never use.
            app.delivery_take_started();
            let capture = app.delivery_take_stopped();
            app.bind_delivery(capture, &id);
            app
        });

        let audio_intact = |store: &Store| {
            assert_eq!(*store.audio_wav(&id).expect("load").expect("present"), *stored);
        };

        // An HTTP error: a failed attempt, the earlier transcript stays.
        retry(&app, cx, RetryWith::Server);
        let take = summary(&store, &id);
        assert_eq!(take.status, SessionStatus::Failed);
        assert_eq!(take.transcript.as_ref().expect("kept").text, "first words");
        assert!(app.read_with(cx, |app, _| app.error.is_some()));
        audio_intact(&store);

        // The engine dies mid-request.
        retry(&app, cx, RetryWith::Server);
        let take = summary(&store, &id);
        assert_eq!(take.status, SessionStatus::Failed);
        assert_eq!(take.attempt_count, 3);
        audio_intact(&store);

        // An empty answer is kept as a result, never shown over words —
        // and the words it did not replace are not offered again.
        retry(&app, cx, RetryWith::Server);
        let take = summary(&store, &id);
        assert_eq!(take.transcript.as_ref().expect("kept").text, "first words");
        assert_eq!(take.results.len(), 2);
        assert!(app.read_with(cx, |app, _| app.delivery.recovery.is_none()));

        // A real answer: a new result, attached to the same recording.
        retry(&app, cx, RetryWith::Server);
        let take = summary(&store, &id);
        assert_eq!(take.status, SessionStatus::Transcribed);
        assert_eq!(take.transcript.as_ref().expect("shown").text, "second words");
        assert_eq!(take.model_label.as_deref(), Some("openai:fake-model"));
        let texts: Vec<_> = take.results.iter().map(|result| result.text.trim()).collect();
        assert_eq!(texts, vec!["first words", "", "second words"]);
        audio_intact(&store);
        assert_eq!(*served.lock().unwrap(), 4);

        // Focus safety: nothing was typed; the text waits for an explicit
        // Copy or Paste last.
        assert!(fake.insertions().is_empty(), "{:?}", fake.insertions());
        let offered = app.read_with(cx, |app, _| {
            app.delivery
                .recovery
                .as_ref()
                .map(|recovery| (recovery.failure.clone(), recovery.text.clone(), recovery.armed))
        });
        assert_eq!(
            offered,
            Some((Failure::Retried, "second words".to_string(), None))
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    /// A take deleted while its retry's audio loads is not retried: no
    /// attempt, and its audio is not stashed back as "unsaved".
    #[gpui::test]
    fn a_take_deleted_while_its_retry_loads_is_left_deleted(cx: &mut gpui::TestAppContext) {
        let root = scratch("deleted-mid-load");
        let store = Store::at_test_root(&root);
        let id = store.save_capture(one_second_wav()).expect("save").id;
        let (addr, served) = fake_server(vec![Reply::Text("never")]);
        let app = cx.new(|cx| {
            let mut app = StarlingApp::for_test(Some(store.clone()), cx);
            app.endpoint = format!("http://{addr}");
            app.apply_sessions(store.list().expect("list"));
            app.selected_id = Some(id.clone());
            app
        });
        app.update(cx, |app, cx| {
            app.retry_selected_with(RetryWith::Server, cx);
            // The confirmed delete is in flight when the load finishes:
            // history still lists the take.
            app.deleting_ids.insert(id.clone());
            store.delete(&id).expect("delete");
        });
        cx.run_until_parked();
        assert_eq!(*served.lock().unwrap(), 0, "nothing was sent");
        app.read_with(cx, |app, _| {
            assert!(app.unsaved.is_empty(), "the deleted audio is not resurrected");
            assert!(!app.active_ids.contains(&id));
        });
        let _ = std::fs::remove_dir_all(&root);
    }

    /// The delayed recovery pass lands while the app is in use: what it
    /// found joins the banners, never replaces an error or notice the
    /// user has not seen yet.
    #[gpui::test]
    fn a_recovery_pass_adds_to_the_banners_it_finds(cx: &mut gpui::TestAppContext) {
        let app = cx.new(|cx| StarlingApp::for_test(None, cx));
        app.update(cx, |app, _| {
            app.error = Some("The save failed.".to_string());
            add_recovery_messages(
                app,
                "1 recording could not be recovered.".to_string(),
                String::new(),
            );
            assert_eq!(
                app.error.as_deref(),
                Some("The save failed. 1 recording could not be recovered.")
            );
            assert!(app.recovery_notice.is_none());
            add_recovery_messages(app, String::new(), "Recovered 1 recording.".to_string());
            add_recovery_messages(app, String::new(), "Recovered 2 recordings.".to_string());
            assert_eq!(
                app.recovery_notice.as_deref(),
                Some("Recovered 1 recording. Recovered 2 recordings.")
            );
            assert_eq!(
                app.error.as_deref(),
                Some("The save failed. 1 recording could not be recovered.")
            );
        });
    }

    /// An engine without binaries: it never serves, so a retry waiting
    /// on one of its models waits until something else ends it.
    fn idle_engine(root: &std::path::Path) -> starling_dictation::engine::EngineManager {
        std::fs::create_dir_all(root.join("engines")).expect("engines");
        starling_dictation::engine::EngineManager::start(
            starling_dictation::engine::EngineConfig {
                engine_dir: Some(root.join("engines")),
                models_dir: root.join("models"),
                state_dir: root.join("state"),
                catalog: Vec::new(),
                backend_override: None,
                icd_dirs: None,
                available_memory_override: None,
                backoff_schedule: None,
            },
            None,
        )
    }

    /// A "Retry with your server" click with no server set up says so
    /// and replaces nothing: a retry waiting for its model carries on.
    #[gpui::test]
    fn a_server_retry_that_cannot_run_leaves_a_waiting_retry_alone(
        cx: &mut gpui::TestAppContext,
    ) {
        let root = scratch("server-retry-keeps-pending");
        let store = Store::at_test_root(&root);
        let id = store.save_capture(one_second_wav()).expect("save").id;
        let engine = idle_engine(&root);
        let app = cx.new(|cx| {
            let mut app = StarlingApp::for_test(Some(store.clone()), cx);
            app.engine = Some(engine.clone());
            app.endpoint = String::new();
            app.apply_sessions(store.list().expect("list"));
            app.selected_id = Some(id.clone());
            app
        });
        retry(&app, cx, RetryWith::Model("model-b".to_string()));
        let waiting = app.read_with(cx, |app, _| app.pending_retry.clone());
        assert!(waiting.is_some());
        retry(&app, cx, RetryWith::Server);
        app.read_with(cx, |app, _| {
            let error = app.error.as_deref().expect("explained");
            assert!(error.contains("No server"), "{error}");
            assert_eq!(app.pending_retry, waiting, "the waiting retry carries on");
        });
        engine.shutdown();
        let _ = std::fs::remove_dir_all(&root);
    }

    /// The engine reporting the model serving without a lease on it is a
    /// moment's gap at most: past the pickup bound the retry ends with a
    /// sentence, never waits on unseen.
    #[test]
    fn a_served_model_without_a_lease_ends_the_retry_visibly() {
        assert_eq!(unleased_retry_failure(std::time::Duration::ZERO), None);
        assert_eq!(
            unleased_retry_failure(RETRY_SWITCH_PICKUP - std::time::Duration::from_millis(1)),
            None
        );
        let reason = unleased_retry_failure(RETRY_SWITCH_PICKUP).expect("gives up");
        assert!(reason.contains("could not be used"), "{reason}");
    }

    /// Settings switching the engine off while a retry waits for its
    /// model ends that wait visibly: no "Loading" left in the drawer, a
    /// sentence saying why, and nothing sent.
    #[gpui::test]
    fn an_engine_change_cancels_a_retry_waiting_for_its_model(cx: &mut gpui::TestAppContext) {
        let root = scratch("engine-change");
        let store = Store::at_test_root(&root);
        let id = store.save_capture(one_second_wav()).expect("save").id;
        let engine = idle_engine(&root);
        let app = cx.new(|cx| {
            let mut app = StarlingApp::for_test(Some(store.clone()), cx);
            app.engine = Some(engine.clone());
            app.apply_sessions(store.list().expect("list"));
            app.selected_id = Some(id.clone());
            app
        });
        retry(&app, cx, RetryWith::Model("model-b".to_string()));
        assert!(app.read_with(cx, |app, _| app.pending_retry.is_some()));
        // Settings → Engine: your own server.
        app.update(cx, |app, _| {
            app.engine = None;
            app.engine_instance += 1;
        });
        cx.executor().advance_clock(std::time::Duration::from_secs(1));
        cx.run_until_parked();
        app.read_with(cx, |app, _| {
            assert!(app.pending_retry.is_none());
            let error = app.error.as_deref().expect("explained");
            assert!(error.contains("cancelled"), "{error}");
            assert!(!app.active_ids.contains(&id));
        });
        assert_eq!(summary(&store, &id).attempt_count, 0);
        engine.shutdown();
        let _ = std::fs::remove_dir_all(&root);
    }

    /// A retry with the server when none is set up does nothing to the
    /// take and says what to do.
    #[gpui::test]
    fn a_retry_without_a_server_leaves_the_take_alone(cx: &mut gpui::TestAppContext) {
        let root = scratch("no-server");
        let store = Store::at_test_root(&root);
        let id = store.save_capture(one_second_wav()).expect("save").id;
        let app = cx.new(|cx| {
            let mut app = StarlingApp::for_test(Some(store.clone()), cx);
            app.endpoint = String::new();
            app.apply_sessions(store.list().expect("list"));
            app.selected_id = Some(id.clone());
            app
        });
        retry(&app, cx, RetryWith::Server);
        assert_eq!(summary(&store, &id).attempt_count, 0);
        let error = app.read_with(cx, |app, _| app.error.clone()).expect("explained");
        assert!(error.contains("No server"), "{error}");
        // No built-in engine runs in manual mode: a model choice says so.
        retry(&app, cx, RetryWith::Model("parakeet".to_string()));
        assert_eq!(summary(&store, &id).attempt_count, 0);
        let error = app.read_with(cx, |app, _| app.error.clone()).expect("explained");
        assert!(error.contains("recording is unchanged"), "{error}");
        let _ = std::fs::remove_dir_all(&root);
    }

    // ---- #220: takes recorded through the runtime host --------------------

    /// A host serving `root` (the app's store root) over the real socket,
    /// recording scripted takes.
    fn host_at(
        root: &std::path::Path,
        scripts: Vec<starling_runtime::testing::FakeTakeScript>,
        orphan_grace: std::time::Duration,
    ) -> starling_runtime_host::HostHandle {
        use starling_runtime::machine::capture::{CaptureConfig, V2CaptureStore};
        let mut config = starling_runtime_host::HostConfig::new(root, root.join("endpoints"))
            .with_orphan_grace(orphan_grace);
        config.runtime = config
            .runtime
            .with_capture_source(starling_runtime::testing::FakeCaptureSource::new(scripts))
            .with_capture_store(Arc::new(V2CaptureStore::open(root).expect("v2 store")))
            .with_capture_config(CaptureConfig {
                journals_dir: root.join("journals"),
                poll_interval: std::time::Duration::from_millis(10),
                ..CaptureConfig::default()
            });
        starling_runtime_host::serve(config).expect("host serves")
    }

    fn scripted_take() -> starling_runtime::testing::FakeTakeScript {
        starling_runtime::testing::FakeTakeScript::clean()
    }

    /// Runs the app until `done` holds (the host and the transcription
    /// server answer on real threads).
    fn settle(
        cx: &mut gpui::TestAppContext,
        what: &str,
        mut done: impl FnMut(&mut gpui::TestAppContext) -> bool,
    ) {
        let deadline = Instant::now() + std::time::Duration::from_secs(20);
        loop {
            cx.run_until_parked();
            if done(cx) {
                return;
            }
            assert!(Instant::now() < deadline, "timed out waiting for {what}");
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
    }

    fn app_on_host(
        cx: &mut gpui::TestAppContext,
        store: &Store,
        host: &starling_runtime_host::HostHandle,
        server: SocketAddr,
    ) -> gpui::Entity<StarlingApp> {
        let socket = host.socket_path().to_path_buf();
        let app = cx.new(|cx| {
            let mut app = StarlingApp::for_test(Some(store.clone()), cx);
            app.endpoint = format!("http://{server}");
            app.model = "fake-model".to_string();
            app.follow_host(socket, crate::host_link::Launch::Never, cx);
            app
        });
        settle(cx, "the connection", |cx| {
            app.read_with(cx, |app, _| app.host.client.is_some())
        });
        app
    }

    fn click(app: &gpui::Entity<StarlingApp>, cx: &mut gpui::TestAppContext) {
        app.update(cx, |app, cx| {
            app.activation_input(|machine| machine.click(Instant::now()), cx)
        });
    }

    fn records(store: &Store) -> Vec<SessionSummary> {
        store
            .list()
            .expect("list")
            .into_iter()
            .filter_map(|record| match record {
                ListedRecord::Session(summary) => Some(summary),
                ListedRecord::Damaged(_) => None,
            })
            .collect()
    }

    #[gpui::test]
    fn a_take_recorded_through_the_host_is_stored_once_and_transcribed(
        cx: &mut gpui::TestAppContext,
    ) {
        let root = scratch("host-take");
        let store = Store::at_test_root(&root);
        let mut host = host_at(&root, vec![scripted_take()], std::time::Duration::from_secs(20));
        let (server, _) = fake_server(vec![Reply::Text("hello from the host")]);
        let app = app_on_host(cx, &store, &host, server);

        click(&app, cx);
        settle(cx, "the take listening", |cx| {
            app.read_with(cx, |app, _| {
                app.recorder
                    .as_ref()
                    .is_some_and(|live| live.confirmed && live.captured_sample_count() > 0)
                    && app.activation.readiness()
                        == Some(crate::activation::Readiness::Listening)
            })
        });
        assert!(
            app.read_with(cx, |app, _| app.recorder.as_ref().unwrap().sample_rate()) > 0,
            "the device rate arrived with the take's status"
        );
        std::thread::sleep(std::time::Duration::from_millis(150));
        click(&app, cx);
        assert!(app.read_with(cx, |app, _| app.recorder.is_none()), "stopping ends the live take");
        settle(cx, "the transcript", |_| {
            records(&store)
                .first()
                .and_then(|take| take.transcript.as_ref())
                .is_some_and(|transcript| transcript.text == "hello from the host")
        });
        assert_eq!(records(&store).len(), 1, "the host stored the take once; the app stored nothing");
        assert!(app.read_with(cx, |app, _| app.host.finishing.is_empty()));
        drop(app);
        host.shutdown();
        let _ = std::fs::remove_dir_all(&root);
    }

    #[gpui::test]
    fn a_take_still_recording_when_the_window_returns_is_adopted_and_finished(
        cx: &mut gpui::TestAppContext,
    ) {
        let root = scratch("host-adopt");
        let store = Store::at_test_root(&root);
        let mut host = host_at(&root, vec![scripted_take()], std::time::Duration::from_secs(20));
        // The window that started the take dies mid-take.
        {
            let gone = starling_runtime_host::client::HostClient::connect(host.socket_path())
                .expect("connect");
            gone.take_watch().expect("watch");
            gone.send(
                Some("take_before_crash"),
                starling_runtime::protocol::Command::CaptureStart {
                    policy: "push-to-talk".into(),
                },
            )
            .expect("start");
            std::thread::sleep(std::time::Duration::from_millis(200));
        }
        let (server, _) = fake_server(vec![Reply::Text("after the restart")]);
        let app = app_on_host(cx, &store, &host, server);
        settle(cx, "the adoption", |cx| {
            app.read_with(cx, |app, _| {
                app.recorder
                    .as_ref()
                    .is_some_and(|live| live.take == "take_before_crash")
                    && app.activation.is_active()
            })
        });
        assert!(
            app.read_with(cx, |app, _| app.service_notice.clone())
                .is_some_and(|notice| notice.contains("still running")),
            "the window says it picked the take up"
        );
        click(&app, cx);
        settle(cx, "the adopted take's transcript", |_| {
            records(&store)
                .first()
                .and_then(|take| take.transcript.as_ref())
                .is_some_and(|transcript| transcript.text == "after the restart")
        });
        assert_eq!(records(&store).len(), 1);
        drop(app);
        host.shutdown();
        let _ = std::fs::remove_dir_all(&root);
    }

    #[gpui::test]
    fn a_take_the_host_stored_with_no_app_is_transcribed_when_one_connects(
        cx: &mut gpui::TestAppContext,
    ) {
        let root = scratch("host-orphan");
        let store = Store::at_test_root(&root);
        let mut host = host_at(&root, vec![scripted_take()], std::time::Duration::from_millis(200));
        {
            let gone = starling_runtime_host::client::HostClient::connect(host.socket_path())
                .expect("connect");
            gone.send(
                Some("take_orphaned"),
                starling_runtime::protocol::Command::CaptureStart {
                    policy: "push-to-talk".into(),
                },
            )
            .expect("start");
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        let deadline = Instant::now() + std::time::Duration::from_secs(10);
        while records(&store).is_empty() {
            assert!(Instant::now() < deadline, "the host never stored the orphan");
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        let (server, _) = fake_server(vec![Reply::Text("nobody was watching")]);
        let app = app_on_host(cx, &store, &host, server);
        settle(cx, "the orphan's transcript", |_| {
            records(&store)
                .first()
                .and_then(|take| take.transcript.as_ref())
                .is_some_and(|transcript| transcript.text == "nobody was watching")
        });
        assert!(
            app.read_with(cx, |app, _| app.service_notice.clone())
                .is_some_and(|notice| notice.contains("still running when Starling closed"))
        );
        drop(app);
        host.shutdown();
        let _ = std::fs::remove_dir_all(&root);
    }

    /// A take the host stores with no app following it; its stored id.
    fn orphan_on(host: &starling_runtime_host::HostHandle, store: &Store) -> String {
        {
            let gone = starling_runtime_host::client::HostClient::connect(host.socket_path())
                .expect("connect");
            gone.send(
                Some("take_orphaned"),
                starling_runtime::protocol::Command::CaptureStart {
                    policy: "push-to-talk".into(),
                },
            )
            .expect("start");
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        let deadline = Instant::now() + std::time::Duration::from_secs(10);
        loop {
            if let Some(take) = records(store).first() {
                return take.id.clone();
            }
            assert!(Instant::now() < deadline, "the host never stored the orphan");
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
    }

    /// The stored takes the host still holds for an app to transcribe.
    fn unclaimed(root: &std::path::Path) -> Vec<String> {
        std::fs::read(root.join(starling_runtime_host::takes::UNCLAIMED_FILE))
            .map(|bytes| serde_json::from_slice(&bytes).expect("unclaimed list"))
            .unwrap_or_default()
    }

    fn words(text: &str) -> storage::TranscriptionResult {
        storage::TranscriptionResult {
            text: text.to_string(),
            segments: Vec::new(),
            duration_seconds: None,
            request_id: None,
        }
    }

    #[gpui::test]
    fn an_orphan_transcribed_before_its_app_told_the_host_is_not_transcribed_again(
        cx: &mut gpui::TestAppContext,
    ) {
        let root = scratch("host-orphan-done");
        let store = Store::at_test_root(&root);
        let mut host = host_at(&root, vec![scripted_take()], std::time::Duration::from_millis(200));
        let id = orphan_on(&host, &store);
        // An app transcribed it and went before its acknowledgement did.
        store.mark_attempt(&id, "openai:fake-model").expect("attempt");
        store.save_transcript(&id, words("already here")).expect("transcript");
        assert_eq!(unclaimed(&root), vec![id.clone()]);
        let (server, served) = fake_server(vec![Reply::Text("a second transcript")]);
        let app = app_on_host(cx, &store, &host, server);
        settle(cx, "the host to hear it handled", |_| unclaimed(&root).is_empty());
        assert_eq!(*served.lock().unwrap(), 0, "not transcribed again");
        assert_eq!(records(&store)[0].attempt_count, 1);
        drop(app);
        host.shutdown();
        let _ = std::fs::remove_dir_all(&root);
    }

    #[gpui::test]
    fn an_orphan_another_app_is_still_transcribing_is_left_to_it(cx: &mut gpui::TestAppContext) {
        let root = scratch("host-orphan-busy");
        let store = Store::at_test_root(&root);
        let mut host = host_at(&root, vec![scripted_take()], std::time::Duration::from_millis(200));
        let id = orphan_on(&host, &store);
        // Another app (its own store handle) is transcribing it; its
        // connection to the host is gone (a host restart) but it is not.
        let other = Store::at_test_root(&root);
        other.mark_attempt(&id, "openai:fake-model").expect("attempt");
        let (server, served) = fake_server(vec![Reply::Text("a second transcript")]);
        let app = app_on_host(cx, &store, &host, server);
        settle(cx, "the host to hear it handled", |_| unclaimed(&root).is_empty());
        assert_eq!(*served.lock().unwrap(), 0, "left to the app transcribing it");
        other.save_transcript(&id, words("the other app's words")).expect("transcript");
        settle(cx, "the other app's transcript here", |cx| {
            cx.executor().advance_clock(crate::remote_take::FOREIGN_POLL);
            app.read_with(cx, |app, _| {
                app.sessions
                    .first()
                    .and_then(|take| take.transcript.as_ref())
                    .is_some_and(|transcript| transcript.text == "the other app's words")
            })
        });
        drop(app);
        host.shutdown();
        let _ = std::fs::remove_dir_all(&root);
    }

    #[gpui::test]
    fn a_failure_history_could_not_record_leaves_the_take_with_the_host(
        cx: &mut gpui::TestAppContext,
    ) {
        let root = scratch("host-unrecorded");
        let store = Store::at_test_root(&root);
        let mut host = host_at(&root, vec![scripted_take()], std::time::Duration::from_secs(20));
        let (server, served) = fake_server(vec![Reply::Status("500 Internal Server Error")]);
        let app = app_on_host(cx, &store, &host, server);
        // The disk refuses the failure's write (as a full disk would).
        rusqlite::Connection::open(root.join("starling.db"))
            .expect("db")
            .execute_batch(
                "CREATE TRIGGER refuse_failures BEFORE UPDATE ON recognition_attempts
                 WHEN NEW.status = 'failed' BEGIN SELECT RAISE(ABORT, 'disk full'); END;",
            )
            .expect("trigger");
        click(&app, cx);
        settle(cx, "listening", |cx| {
            app.read_with(cx, |app, _| {
                app.activation.readiness() == Some(crate::activation::Readiness::Listening)
            })
        });
        click(&app, cx);
        settle(cx, "the failed job", |cx| {
            *served.lock().unwrap() == 1
                && app.read_with(cx, |app, _| app.active_ids.is_empty() && app.host.handling.is_empty())
        });
        std::thread::sleep(std::time::Duration::from_millis(300));
        cx.run_until_parked();
        let id = records(&store).first().expect("stored").id.clone();
        assert_eq!(unclaimed(&root), vec![id], "still the host's to hand out again");
        drop(app);
        host.shutdown();
        let _ = std::fs::remove_dir_all(&root);
    }

    #[gpui::test]
    fn two_windows_share_one_host_and_only_the_starter_finishes_its_take(
        cx: &mut gpui::TestAppContext,
    ) {
        let root = scratch("host-two");
        let store = Store::at_test_root(&root);
        let mut host = host_at(&root, vec![scripted_take()], std::time::Duration::from_secs(20));
        let (server, served) = fake_server(vec![Reply::Text("one window's words")]);
        let first = app_on_host(cx, &store, &host, server);
        let second = app_on_host(cx, &Store::at_test_root(&root), &host, server);

        click(&first, cx);
        settle(cx, "the first window listening", |cx| {
            first.read_with(cx, |app, _| {
                app.activation.readiness() == Some(crate::activation::Readiness::Listening)
            })
        });
        std::thread::sleep(std::time::Duration::from_millis(150));
        cx.run_until_parked();
        assert!(
            second.read_with(cx, |app, _| app.recorder.is_none() && !app.activation.is_active()),
            "the second window never takes over a take its live owner records"
        );
        click(&first, cx);
        settle(cx, "the transcript", |_| {
            records(&store)
                .first()
                .and_then(|take| take.transcript.as_ref())
                .is_some_and(|transcript| transcript.text == "one window's words")
        });
        cx.run_until_parked();
        assert_eq!(*served.lock().unwrap(), 1, "transcribed once, by the window that recorded it");
        drop(first);
        drop(second);
        host.shutdown();
        let _ = std::fs::remove_dir_all(&root);
    }

    #[gpui::test]
    fn another_windows_take_settles_here_once_that_window_settled_it(
        cx: &mut gpui::TestAppContext,
    ) {
        let root = scratch("host-foreign");
        let store = Store::at_test_root(&root);
        let mut host = host_at(&root, vec![scripted_take()], std::time::Duration::from_secs(20));
        let (server, served) = fake_server(Vec::new());
        let app = app_on_host(cx, &store, &host, server);
        // The other window: it starts and stops its take, then transcribes
        // it (here: records a failed attempt) without the host hearing.
        let other = starling_runtime_host::client::HostClient::connect(host.socket_path())
            .expect("connect");
        other
            .send(
                Some("take_other"),
                starling_runtime::protocol::Command::CaptureStart {
                    policy: "push-to-talk".into(),
                },
            )
            .expect("start");
        std::thread::sleep(std::time::Duration::from_millis(200));
        other
            .send(
                Some("take_other"),
                starling_runtime::protocol::Command::CaptureStop { drain: Some(true) },
            )
            .expect("stop");
        let row = |cx: &mut gpui::TestAppContext| {
            app.read_with(cx, |app, _| app.sessions.first().map(|take| take.status))
        };
        settle(cx, "the other window's take listed here", |cx| row(cx).is_some());
        assert_eq!(row(cx), Some(SessionStatus::Captured), "listed as being sent");
        let id = records(&store).first().expect("stored").id.clone();
        store.mark_attempt(&id, "openai:fake-model").expect("attempt");
        store.save_failure(&id, "the other window's server failed").expect("failure");
        settle(cx, "the settled take here", |cx| {
            cx.executor().advance_clock(crate::remote_take::FOREIGN_POLL);
            row(cx) == Some(SessionStatus::Failed)
        });
        assert_eq!(*served.lock().unwrap(), 0, "never transcribed here");
        drop(other);
        drop(app);
        host.shutdown();
        let _ = std::fs::remove_dir_all(&root);
    }

    #[gpui::test]
    fn of_two_windows_that_could_adopt_a_take_only_one_keeps_and_transcribes_it(
        cx: &mut gpui::TestAppContext,
    ) {
        let root = scratch("host-adopt-race");
        let store = Store::at_test_root(&root);
        let mut host = host_at(&root, vec![scripted_take()], std::time::Duration::from_secs(20));
        {
            let gone = starling_runtime_host::client::HostClient::connect(host.socket_path())
                .expect("connect");
            gone.send(
                Some("take_race"),
                starling_runtime::protocol::Command::CaptureStart {
                    policy: "push-to-talk".into(),
                },
            )
            .expect("start");
            std::thread::sleep(std::time::Duration::from_millis(200));
        }
        let (server, served) = fake_server(vec![Reply::Text("adopted once")]);
        let first = app_on_host(cx, &store, &host, server);
        let second = app_on_host(cx, &Store::at_test_root(&root), &host, server);
        let holding = |cx: &mut gpui::TestAppContext| {
            [&first, &second]
                .iter()
                .filter(|app| app.read_with(cx, |app, _| app.recorder.is_some()))
                .count()
        };
        settle(cx, "exactly one window holding the take", |cx| holding(cx) == 1);
        std::thread::sleep(std::time::Duration::from_millis(300));
        cx.run_until_parked();
        assert_eq!(holding(cx), 1, "the other window let go");
        let owner = if first.read_with(cx, |app, _| app.recorder.is_some()) {
            first.clone()
        } else {
            second.clone()
        };
        click(&owner, cx);
        settle(cx, "the transcript", |_| {
            records(&store)
                .first()
                .and_then(|take| take.transcript.as_ref())
                .is_some_and(|transcript| transcript.text == "adopted once")
        });
        cx.run_until_parked();
        assert_eq!(*served.lock().unwrap(), 1, "transcribed once");
        drop(first);
        drop(second);
        host.shutdown();
        let _ = std::fs::remove_dir_all(&root);
    }

    #[gpui::test]
    fn recording_is_refused_while_the_service_cannot_be_reached(cx: &mut gpui::TestAppContext) {
        let root = scratch("host-down");
        let store = Store::at_test_root(&root);
        let app = cx.new(|cx| {
            let mut app = StarlingApp::for_test(Some(store.clone()), cx);
            app.follow_host(root.join("nobody-serves-here"), crate::host_link::Launch::Never, cx);
            app
        });
        settle(cx, "the link to give up once", |cx| {
            app.read_with(cx, |app, _| {
                app.host.down.as_deref().is_some_and(|reason| reason != "connecting")
            })
        });
        click(&app, cx);
        let error = app.read_with(cx, |app, _| app.error.clone()).expect("explained");
        assert!(error.contains("recording service"), "{error}");
        assert!(app.read_with(cx, |app, _| app.recorder.is_none() && !app.activation.is_active()));
        let _ = std::fs::remove_dir_all(&root);
    }
}
