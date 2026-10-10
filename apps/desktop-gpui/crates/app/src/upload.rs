//! Recording start/stop/cancel, file imports and retries, split out of
//! `app.rs`. The runtime host records and transcribes (#220); this asks it
//! to and keeps what the window owns.

use std::sync::Arc;
use std::time::Instant;

use gpui::{AppContext, AsyncApp, Context, PathPromptOptions, WeakEntity};
use starling_dictation::{audio, recorder, storage};
use starling_runtime_host::frame::TranscribeWith;

use crate::app::{StarlingApp, UnsavedWav};
use crate::store::{AudioHold, Store};

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

    /// Stop the window's live take: the host finalizes, stores and
    /// transcribes it (#220). Only the activation machine calls this (an
    /// `Effect::Finish`).
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
        self.live_partial.clear();
        // Storage-fault honesty (I1 phase 2): a journal fault that froze
        // acknowledgment is surfaced.
        if let Some(fault) = live.capture_error() {
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
            crate::remote_take::FinishKind::Transcribe {
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
        if self.host.link.is_none() {
            self.delivery_take_stopped();
            self.error = Some(
                "Starling's recording service is not ready; try again in a moment.".to_string(),
            );
            cx.notify();
            return false;
        }
        // Attenuation begins with the attempt, before the microphone opens
        // — unless a start cue is due, which it would swallow: then it
        // begins once the cue has played (`cues.rs`).
        let playback_lease = (!crate::cues::attenuation_waits_for_cue(
            &self.feedback,
            self.playback_settings.during_recording,
        ))
        .then(|| self.playback.handle().begin(&self.playback_settings));
        let take = starling_runtime::bus::new_id("take");
        self.mic.problem = None;
        self.mic.interruption = None;
        self.mic.last_sound_at = None;
        self.live_partial.clear();
        if self.staged_mode() {
            self.begin_staging(cx);
        } else {
            self.retire_staging(cx);
        }
        // The host binds the take's engine at its start (#363) and says
        // when there is no live text for it.
        self.stream_degradation = None;
        self.recorder = Some(crate::host_link::LiveCapture::new(take.clone()));
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
            crate::remote_take::FinishKind::Cancel {
                saved_notice: Self::cancel_saved_notice(reason),
                empty_notice,
                staging,
            },
        );
        cx.notify();
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

    /// Asks the recording service to transcribe stored take `id` with
    /// `with` (#220): a new attempt beside the earlier ones, never typed
    /// into an editor — the take's own delivery is gone with its first
    /// job; a retried transcript (`offer`) is offered for Copy / Paste
    /// last instead. `hold` keeps the take's audio from every process's
    /// upkeep until the service's attempt does (#342). `false` when the
    /// service cannot be asked.
    pub(crate) fn ask_host_to_transcribe(
        &mut self,
        id: String,
        with: TranscribeWith,
        hold: Option<AudioHold>,
        offer: bool,
        cx: &mut Context<Self>,
    ) -> bool {
        let Some(link) = self.host.link.as_ref().filter(|_| self.host.client.is_some()) else {
            self.error = Some(
                "Starling's recording service is not reachable right now; try again once it is. \
                 The recording is unchanged."
                    .to_string(),
            );
            cx.notify();
            return false;
        };
        let req = starling_runtime::bus::new_id("tr");
        link.transcribe(&req, &id, with);
        self.forget_delivery(&id);
        self.host.requests.insert(
            req,
            crate::remote_take::Request {
                stored_id: id.clone(),
                _hold: hold,
                offer,
            },
        );
        self.selected_id = Some(id);
        self.error = None;
        cx.notify();
        true
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
            let Ok(Ok(Some(paths))) = receiver.await else {
                return;
            };
            let Some(path) = paths.first().cloned() else {
                return;
            };
            let prepared = cx
                .background_spawn(async move {
                    let bytes = std::fs::read(&path)
                        .map_err(|err| format!("Could not read the file: {err}"))?;
                    audio::prepare_wav_16k(&bytes).map_err(|err| err.to_string())
                })
                .await;
            match prepared {
                Ok(prepared) => {
                    this.update(cx, |app, cx| app.save_import(Arc::new(prepared.wav), cx))
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
        })
        .detach();
    }

    /// Stores an imported take with the intent to transcribe it and asks
    /// the recording service to run that now, with whatever it
    /// transcribes with. A store that cannot take it leaves the WAV in the
    /// unsaved list.
    pub(crate) fn save_import(&mut self, wav: Arc<Vec<u8>>, cx: &mut Context<Self>) {
        let Some(store) = self.store.clone() else {
            let reason = self
                .store_error
                .clone()
                .unwrap_or_else(|| "session store unavailable".to_string());
            self.stash_unsaved(wav, &format!(
                "Local storage failed: {reason} Keep this window open and download the unsaved WAV to recover it."
            ));
            cx.notify();
            return;
        };
        cx.spawn(async move |this, cx| {
            let saved = {
                let store = store.clone();
                let wav = wav.clone();
                cx.background_spawn(async move { store.save_import(wav).map(|saved| saved.id) })
                    .await
            };
            match saved {
                Ok(id) => {
                    // Stored with its intent: the service transcribes it
                    // whether or not this window is still here to ask.
                    this.update(cx, |app, cx| {
                        if let Some(link) = &app.host.link {
                            link.transcribe_due(&id);
                        }
                        app.host.awaiting.insert(id.clone());
                        app.active_ids.insert(id.clone());
                        app.selected_id = Some(id);
                        cx.notify();
                    })
                    .ok();
                    refresh_sessions(&this, &store, cx).await;
                }
                Err(err) => {
                    this.update(cx, |app, cx| {
                        app.stash_unsaved(wav, &format!(
                            "Local storage failed: {err} Keep this window open and download the unsaved WAV to recover it."
                        ));
                        cx.notify();
                    })
                    .ok();
                }
            }
        })
        .detach();
    }

    /// Retry the selected take with whatever the service transcribes
    /// with now.
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
        if self.is_active(&id) || self.is_deleting(&id) {
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
        // A newer choice replaces a retry still waiting for its model.
        self.pending_retry = None;
        self.retry_seq += 1;
        let seq = self.retry_seq;
        let Some(store) = self.store.clone() else {
            return;
        };
        cx.spawn(async move |this, cx| {
            // #342: held from before the check until the service's
            // attempt holds the audio — including while the engine
            // switches to the retry's model — against every process's
            // upkeep (#220: the attempt starts in the service).
            let hold = {
                let store = store.clone();
                let id = id.clone();
                cx.background_spawn(async move { store.hold_audio(&id) }).await
            };
            this.update(cx, |app, cx| {
                // Superseded by a newer choice, or the take was deleted
                // meanwhile: the hold drops, nothing runs.
                if app.retry_seq != seq || app.retry_target_gone(&id) {
                    return;
                }
                let pin = match hold {
                    Ok(hold) => hold,
                    Err(err) => {
                        app.error = Some(format!(
                            "The recording could not be kept for the retry ({err}); it is \
                             unchanged."
                        ));
                        cx.notify();
                        return;
                    }
                };
                match with {
                    RetryWith::Current => {
                        app.ask_host_to_transcribe(id, TranscribeWith::Current, Some(pin), true, cx);
                    }
                    RetryWith::Server => {
                        let with = TranscribeWith::Server {
                            endpoint: app.endpoint.clone(),
                            model: app.model.clone(),
                        };
                        app.ask_host_to_transcribe(id, with, Some(pin), true, cx);
                    }
                    RetryWith::Model(model_id) => app.retry_after_switch(id, model_id, seq, pin, cx),
                }
            })
            .ok();
        })
        .detach();
    }

    /// Retry on an installed built-in model (#356): the engine switches
    /// to it the way Settings would (it stays the active model), and the
    /// recording service transcribes once that model serves. Nothing is
    /// attempted — and the take is untouched — when the switch does not
    /// happen.
    fn retry_after_switch(
        &mut self,
        id: String,
        model_id: String,
        seq: u64,
        pin: AudioHold,
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
        let with = TranscribeWith::Model {
            model_id: model_id.clone(),
        };
        if engine.lease().is_some_and(|lease| lease.model_id() == model_id) {
            self.ask_host_to_transcribe(id, with, Some(pin), true, cx);
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
            let mut job = Some((with, pin));
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
                    match switch_progress(&engine.snapshot(), &model_id, request, started.elapsed()) {
                        SwitchProgress::Waiting => false,
                        SwitchProgress::Ready => {
                            // The service follows the switch (the same
                            // settings, the same engine) and waits for the
                            // model itself.
                            app.pending_retry = None;
                            if let Some((with, pin)) = job.take() {
                                app.ask_host_to_transcribe(
                                    pending.take_id.clone(),
                                    with,
                                    Some(pin),
                                    true,
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
    use starling_runtime_host::frame::TranscriptionState;

    use crate::delivery::{DeliveryState, Failure};
    use crate::host_link::{HostUpdate, TakeUpdate};

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

    /// Runs `with` on the selected take and waits until the host settled
    /// it (its attempt count reached `attempts`).
    fn retry(
        app: &gpui::Entity<StarlingApp>,
        cx: &mut gpui::TestAppContext,
        store: &Store,
        with: RetryWith,
        attempts: u32,
    ) {
        let id = app.read_with(cx, |app, _| app.selected_id.clone()).expect("selected");
        app.update(cx, |app, cx| app.retry_selected_with(with, cx));
        settle(cx, "the retry", |cx| {
            summary(store, &id).attempt_count >= attempts
                && app.read_with(cx, |app, _| {
                    app.host.requests.is_empty() && !app.active_ids.contains(&id)
                })
        });
    }

    /// The engine-client fault matrix on a saved take (#356), through the
    /// recording service (#220): an error status, a server that hangs up
    /// mid-request (a crashed engine), and an empty answer each keep the
    /// audio and the take's history; a successful retry adds a result
    /// beside the earlier one, and nothing is ever typed into another app
    /// — the text is offered for Copy / Paste last instead.
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
        let mut host = host_at(&root, Vec::new(), std::time::Duration::from_secs(20), addr);
        let fake = Arc::new(FakeBackend::new());
        fake.focus(FakeTarget::named("Editor", "notes.txt"));
        let inserter = Arc::new(Inserter::with_backends(vec![Box::new(fake.clone())]));
        let app = app_on_host(cx, &store, &host, addr);
        app.update(cx, |app, _| {
            app.delivery = DeliveryState::new(inserter, InsertionSettings::default());
            app.apply_sessions(store.list().expect("list"));
            app.selected_id = Some(id.clone());
            // The take was dictated into the editor: its first delivery
            // capture is still bound, the shape a retry must never use.
            app.delivery_take_started();
            let capture = app.delivery_take_stopped();
            app.bind_delivery(capture, &id);
        });

        let audio_intact = |store: &Store| {
            assert_eq!(*store.audio_wav(&id).expect("load").expect("present"), *stored);
        };

        // An HTTP error: a failed attempt, the earlier transcript stays.
        retry(&app, cx, &store, RetryWith::Server, 2);
        let take = summary(&store, &id);
        assert_eq!(take.status, SessionStatus::Failed);
        assert_eq!(take.transcript.as_ref().expect("kept").text, "first words");
        assert!(app.read_with(cx, |app, _| app.error.is_some()));
        audio_intact(&store);

        // The engine dies mid-request.
        retry(&app, cx, &store, RetryWith::Server, 3);
        let take = summary(&store, &id);
        assert_eq!(take.status, SessionStatus::Failed);
        assert_eq!(take.attempt_count, 3);
        audio_intact(&store);

        // An empty answer is kept as a result, never shown over words —
        // and the words it did not replace are not offered again.
        retry(&app, cx, &store, RetryWith::Server, 4);
        let take = summary(&store, &id);
        assert_eq!(take.transcript.as_ref().expect("kept").text, "first words");
        assert_eq!(take.results.len(), 2);
        assert!(app.read_with(cx, |app, _| app.delivery.recovery.is_none()));

        // A real answer: a new result, attached to the same recording.
        retry(&app, cx, &store, RetryWith::Server, 5);
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
        settle(cx, "the offer", |cx| {
            app.read_with(cx, |app, _| app.delivery.recovery.is_some())
        });
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
        drop(app);
        host.shutdown();
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
        app.update(cx, |app, cx| app.retry_selected_with(RetryWith::Model("model-b".to_string()), cx));
        cx.run_until_parked();
        let waiting = app.read_with(cx, |app, _| app.pending_retry.clone());
        assert!(waiting.is_some());
        app.update(cx, |app, cx| app.retry_selected_with(RetryWith::Server, cx));
        cx.run_until_parked();
        app.read_with(cx, |app, _| {
            let error = app.error.as_deref().expect("explained");
            assert!(error.contains("No server"), "{error}");
            assert_eq!(app.pending_retry, waiting, "the waiting retry carries on");
        });
        engine.shutdown();
        let _ = std::fs::remove_dir_all(&root);
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
        app.update(cx, |app, cx| app.retry_selected_with(RetryWith::Model("model-b".to_string()), cx));
        cx.run_until_parked();
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
        app.update(cx, |app, cx| app.retry_selected_with(RetryWith::Server, cx));
        cx.run_until_parked();
        assert_eq!(summary(&store, &id).attempt_count, 0);
        let error = app.read_with(cx, |app, _| app.error.clone()).expect("explained");
        assert!(error.contains("No server"), "{error}");
        // No built-in engine runs in manual mode: a model choice says so.
        app.update(cx, |app, cx| app.retry_selected_with(RetryWith::Model("parakeet".to_string()), cx));
        cx.run_until_parked();
        assert_eq!(summary(&store, &id).attempt_count, 0);
        let error = app.read_with(cx, |app, _| app.error.clone()).expect("explained");
        assert!(error.contains("recording is unchanged"), "{error}");
        let _ = std::fs::remove_dir_all(&root);
    }

    // ---- #220: takes recorded and transcribed by the runtime host ---------

    /// A host serving `root` (the app's store root) over the real socket,
    /// recording scripted takes and transcribing them on the server at
    /// `server`.
    fn host_at(
        root: &std::path::Path,
        scripts: Vec<starling_runtime::testing::FakeTakeScript>,
        orphan_grace: std::time::Duration,
        server: SocketAddr,
    ) -> starling_runtime_host::HostHandle {
        use starling_runtime::machine::capture::{CaptureConfig, V2CaptureStore};
        let mut config = starling_runtime_host::HostConfig::new(root, root.join("endpoints"))
            .with_orphan_grace(orphan_grace)
            .with_engine(starling_runtime_host::engine::EngineChoice::Manual {
                endpoint: format!("http://{server}"),
                model: "fake-model".to_string(),
            });
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

    fn transcript_of(store: &Store) -> Option<String> {
        records(store)
            .first()
            .and_then(|take| take.transcript.as_ref())
            .map(|transcript| transcript.text.clone())
    }

    /// Records one take in `app` and stops it once it is listening.
    fn record_once(app: &gpui::Entity<StarlingApp>, cx: &mut gpui::TestAppContext) {
        click(app, cx);
        settle(cx, "the take listening", |cx| {
            app.read_with(cx, |app, _| {
                app.recorder
                    .as_ref()
                    .is_some_and(|live| live.confirmed && live.captured_sample_count() > 0)
                    && app.activation.readiness()
                        == Some(crate::activation::Readiness::Listening)
            })
        });
        std::thread::sleep(std::time::Duration::from_millis(150));
        click(app, cx);
    }

    #[gpui::test]
    fn a_take_is_stored_and_transcribed_by_the_host_into_the_window_that_dictated_it(
        cx: &mut gpui::TestAppContext,
    ) {
        let root = scratch("host-take");
        let store = Store::at_test_root(&root);
        let (server, served) = fake_server(vec![Reply::Text("hello from the host")]);
        let mut host = host_at(&root, vec![scripted_take()], std::time::Duration::from_secs(20), server);
        let fake = Arc::new(FakeBackend::new());
        fake.focus(FakeTarget::named("Editor", "notes.txt"));
        let inserter = Arc::new(Inserter::with_backends(vec![Box::new(fake.clone())]));
        let app = app_on_host(cx, &store, &host, server);
        app.update(cx, |app, _| {
            app.delivery = DeliveryState::new(inserter, InsertionSettings::default());
        });
        record_once(&app, cx);
        assert!(app.read_with(cx, |app, _| app.recorder.is_none()), "stopping ends the live take");
        // The default mode stages: the transcript lands in this window's
        // staging panel, typed where it was dictated once the user inserts.
        settle(cx, "the transcript in the staging panel", |cx| {
            let id = records(&store).first().map(|take| take.id.clone());
            transcript_of(&store).as_deref() == Some("hello from the host")
                && app.read_with(cx, |app, _| {
                    app.host.awaiting.is_empty()
                        && id.is_some_and(|id| {
                            app.staged_text_for(&id).as_deref() == Some("hello from the host")
                        })
                })
        });
        assert_eq!(records(&store).len(), 1, "the host stored the take once; the app stored nothing");
        assert_eq!(*served.lock().unwrap(), 1, "transcribed once, by the host");
        assert!(app.read_with(cx, |app, _| app.host.finishing.is_empty() && app.active_ids.is_empty()));
        assert!(fake.insertions().is_empty(), "staged: nothing typed before Insert");
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
        let (server, _) = fake_server(vec![Reply::Text("after the restart")]);
        let mut host = host_at(&root, vec![scripted_take()], std::time::Duration::from_secs(20), server);
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
        let fake = Arc::new(FakeBackend::new());
        fake.focus(FakeTarget::named("Editor", "notes.txt"));
        let inserter = Arc::new(Inserter::with_backends(vec![Box::new(fake.clone())]));
        let app = app_on_host(cx, &store, &host, server);
        app.update(cx, |app, _| {
            app.delivery = DeliveryState::new(inserter, InsertionSettings::default());
        });
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
        settle(cx, "the adopted take's transcript", |cx| {
            transcript_of(&store).as_deref() == Some("after the restart")
                && app.read_with(cx, |app, _| app.host.awaiting.is_empty())
        });
        assert_eq!(records(&store).len(), 1);
        assert!(fake.insertions().is_empty(), "an adopted take is never typed into another app");
        drop(app);
        host.shutdown();
        let _ = std::fs::remove_dir_all(&root);
    }

    #[gpui::test]
    fn a_take_the_host_stored_with_no_app_is_transcribed_without_one(
        cx: &mut gpui::TestAppContext,
    ) {
        let root = scratch("host-orphan");
        let store = Store::at_test_root(&root);
        let (server, served) = fake_server(vec![Reply::Text("nobody was watching")]);
        let mut host = host_at(&root, vec![scripted_take()], std::time::Duration::from_millis(200), server);
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
        while transcript_of(&store).is_none() {
            assert!(Instant::now() < deadline, "the host never transcribed the orphan");
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        assert_eq!(transcript_of(&store).as_deref(), Some("nobody was watching"));
        // A window that opens later finds it in history, and nothing to do.
        let app = app_on_host(cx, &store, &host, server);
        settle(cx, "the history", |cx| app.read_with(cx, |app, _| !app.sessions.is_empty()));
        cx.run_until_parked();
        assert_eq!(*served.lock().unwrap(), 1, "transcribed once");
        assert!(app.read_with(cx, |app, _| app.active_ids.is_empty()));
        drop(app);
        host.shutdown();
        let _ = std::fs::remove_dir_all(&root);
    }

    #[gpui::test]
    fn two_windows_share_one_host_and_only_the_starter_gets_its_take(
        cx: &mut gpui::TestAppContext,
    ) {
        let root = scratch("host-two");
        let store = Store::at_test_root(&root);
        let (server, served) = fake_server(vec![Reply::Text("one window's words")]);
        let mut host = host_at(&root, vec![scripted_take()], std::time::Duration::from_secs(20), server);
        let first = app_on_host(cx, &store, &host, server);
        let second = app_on_host(cx, &Store::at_test_root(&root), &host, server);
        let typed = |app: &gpui::Entity<StarlingApp>, cx: &mut gpui::TestAppContext| {
            let fake = Arc::new(FakeBackend::new());
            fake.focus(FakeTarget::named("Editor", "notes.txt"));
            let inserter = Arc::new(Inserter::with_backends(vec![Box::new(fake.clone())]));
            app.update(cx, |app, _| {
                app.delivery = DeliveryState::new(inserter, InsertionSettings::default());
            });
            fake
        };
        let first_typed = typed(&first, cx);
        let second_typed = typed(&second, cx);

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
        settle(cx, "the transcript, staged in the first window", |cx| {
            let id = records(&store).first().map(|take| take.id.clone());
            transcript_of(&store).as_deref() == Some("one window's words")
                && first.read_with(cx, |app, _| {
                    id.is_some_and(|id| app.staged_text_for(&id).is_some())
                })
        });
        settle(cx, "the second window showing it", |cx| {
            second.read_with(cx, |app, _| {
                app.sessions
                    .first()
                    .is_some_and(|take| take.status == SessionStatus::Transcribed)
            })
        });
        assert_eq!(*served.lock().unwrap(), 1, "transcribed once, by the host");
        let id = records(&store).first().expect("stored").id.clone();
        assert!(second.read_with(cx, |app, _| app.staged_text_for(&id).is_none()));
        assert!(first_typed.insertions().is_empty() && second_typed.insertions().is_empty());
        drop(first);
        drop(second);
        host.shutdown();
        let _ = std::fs::remove_dir_all(&root);
    }

    #[gpui::test]
    fn of_two_windows_that_could_adopt_a_take_only_one_keeps_it(
        cx: &mut gpui::TestAppContext,
    ) {
        let root = scratch("host-adopt-race");
        let store = Store::at_test_root(&root);
        let (server, served) = fake_server(vec![Reply::Text("adopted once")]);
        let mut host = host_at(&root, vec![scripted_take()], std::time::Duration::from_secs(20), server);
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
        settle(cx, "the transcript", |_| transcript_of(&store).as_deref() == Some("adopted once"));
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
    // ---- the app's side of the host's frames, with no host ----------------

    /// A window with a fake host: frames are fed to it directly.
    fn window_with_typing(
        cx: &mut gpui::TestAppContext,
        store: &Store,
    ) -> (gpui::Entity<StarlingApp>, Arc<FakeBackend>) {
        let fake = Arc::new(FakeBackend::new());
        fake.focus(FakeTarget::named("Editor", "notes.txt"));
        let inserter = Arc::new(Inserter::with_backends(vec![Box::new(fake.clone())]));
        let app = cx.new(|cx| {
            let mut app = StarlingApp::for_test(Some(store.clone()), cx);
            app.delivery = DeliveryState::new(inserter, InsertionSettings::default());
            app
        });
        (app, fake)
    }

    fn transcribed(store: &Store, text: &str) -> String {
        let id = store.save_capture(one_second_wav()).expect("save").id;
        store.mark_attempt(&id, "openai:fake-model").expect("attempt");
        store
            .save_transcript(
                &id,
                storage::TranscriptionResult {
                    text: text.to_string(),
                    segments: Vec::new(),
                    duration_seconds: None,
                    request_id: None,
                },
            )
            .expect("transcript");
        id
    }

    fn frame(
        app: &gpui::Entity<StarlingApp>,
        cx: &mut gpui::TestAppContext,
        update: TakeUpdate,
    ) {
        app.update(cx, |app, cx| app.host_update(HostUpdate::Take(update), cx));
        cx.run_until_parked();
    }

    fn completed(stored_id: &str, req: Option<&str>, yours: bool) -> TakeUpdate {
        completed_with(stored_id, req, yours, "words")
    }

    fn completed_with(stored_id: &str, req: Option<&str>, yours: bool, text: &str) -> TakeUpdate {
        TakeUpdate::Transcription {
            stored_id: stored_id.to_string(),
            take: None,
            req: req.map(str::to_string),
            attempt: None,
            state: TranscriptionState::Completed {
                text: text.to_string(),
                kept_earlier: false,
            },
            yours,
        }
    }

    /// A window's own take with its delivery bound, waiting for the host.
    fn own_take(app: &gpui::Entity<StarlingApp>, cx: &mut gpui::TestAppContext, id: &str) {
        app.update(cx, |app, _| {
            app.delivery_take_started();
            let capture = app.delivery_take_stopped();
            app.bind_delivery(capture, id);
            app.host.awaiting.insert(id.to_string());
        });
    }

    /// The window types its own take's result — exactly that one, even
    /// when a retry another window asked for has landed in history since.
    #[gpui::test]
    fn the_own_take_types_its_own_result_not_a_later_retry(cx: &mut gpui::TestAppContext) {
        let root = scratch("frames-exact");
        let store = Store::at_test_root(&root);
        let id = transcribed(&store, "my own words");
        // Another window's retry landed after it.
        store.mark_attempt(&id, "openai:other").expect("attempt");
        store
            .save_transcript(
                &id,
                storage::TranscriptionResult {
                    text: "somebody else's retry".to_string(),
                    segments: Vec::new(),
                    duration_seconds: None,
                    request_id: None,
                },
            )
            .expect("retry");
        let (app, fake) = window_with_typing(cx, &store);
        own_take(&app, cx, &id);
        frame(&app, cx, completed_with(&id, None, true, "my own words"));
        settle(cx, "the own take typed", |_| !fake.insertions().is_empty());
        assert_eq!(fake.insertions().len(), 1);
        assert_eq!(fake.insertions()[0].1, "my own words", "{:?}", fake.insertions());
        let _ = std::fs::remove_dir_all(&root);
    }

    /// Another window's retry that the host refused (or that finished)
    /// leaves this window's own take waiting, its delivery bound.
    #[gpui::test]
    fn another_windows_request_never_touches_the_own_takes_delivery(
        cx: &mut gpui::TestAppContext,
    ) {
        let root = scratch("frames-foreign-request");
        let store = Store::at_test_root(&root);
        let id = transcribed(&store, "mine at last");
        let (app, fake) = window_with_typing(cx, &store);
        own_take(&app, cx, &id);
        frame(
            &app,
            cx,
            TakeUpdate::Transcription {
                stored_id: id.clone(),
                take: None,
                req: Some("tr_elsewhere".to_string()),
                attempt: None,
                state: TranscriptionState::Refused {
                    message: "This recording is being transcribed already.".to_string(),
                },
                yours: false,
            },
        );
        frame(&app, cx, completed_with(&id, Some("tr_elsewhere"), false, "theirs"));
        assert!(app.read_with(cx, |app, _| app.host.awaiting.contains(&id)), "still waiting");
        assert!(fake.insertions().is_empty());
        frame(&app, cx, completed_with(&id, None, true, "mine at last"));
        settle(cx, "the own take typed", |_| !fake.insertions().is_empty());
        assert_eq!(fake.insertions()[0].1, "mine at last");
        let _ = std::fs::remove_dir_all(&root);
    }

    /// A retry another window asked for, handed to this window to act on
    /// (its asker went away), is processed here but never typed — even
    /// while this window's own take of it still waits for its result.
    #[gpui::test]
    fn a_retry_handed_to_this_window_is_never_typed(cx: &mut gpui::TestAppContext) {
        let root = scratch("frames-handed-retry");
        let store = Store::at_test_root(&root);
        let id = transcribed(&store, "their retry");
        let (app, fake) = window_with_typing(cx, &store);
        own_take(&app, cx, &id);
        frame(&app, cx, completed_with(&id, Some("tr_gone"), true, "their retry"));
        cx.run_until_parked();
        assert!(fake.insertions().is_empty(), "{:?}", fake.insertions());
        assert!(app.read_with(cx, |app, _| app.host.awaiting.contains(&id)), "still waiting");
        frame(&app, cx, completed_with(&id, None, true, "my own"));
        settle(cx, "the own take typed", |_| !fake.insertions().is_empty());
        assert_eq!(fake.insertions()[0].1, "my own");
        let _ = std::fs::remove_dir_all(&root);
    }

    /// The live text the host sends with an adoption shows once the take
    /// is this window's (it arrives before the confirming tick).
    #[gpui::test]
    fn an_adopted_take_shows_the_live_state_sent_with_the_adoption(
        cx: &mut gpui::TestAppContext,
    ) {
        let app = cx.new(|cx| StarlingApp::for_test(None, cx));
        app.update(cx, |app, _| {
            app.host.claiming = Some(("take_back".to_string(), Instant::now()));
        });
        frame(
            &app,
            cx,
            TakeUpdate::LiveText {
                take: "take_back".to_string(),
                partial: None,
                degraded: Some("live text stopped earlier".to_string()),
            },
        );
        assert!(app.read_with(cx, |app, _| app.stream_degradation.is_none()), "not ours yet");
        frame(
            &app,
            cx,
            TakeUpdate::Live {
                take: "take_back".to_string(),
                rate: 16_000,
                status: None,
                owner: starling_runtime_host::frame::TakeOwner::You,
                ended: None,
                kept: false,
                meter: None,
            },
        );
        app.read_with(cx, |app, _| {
            assert!(app.recorder.as_ref().is_some_and(|live| live.take == "take_back"));
            assert_eq!(app.stream_degradation.as_deref(), Some("live text stopped earlier"));
        });
    }

    /// The window types the take it recorded when the host reports it
    /// transcribed for it; a result for another window, or one this window
    /// is not the one to act on, is shown and never typed.
    #[gpui::test]
    fn only_the_windows_own_take_is_typed_and_only_when_it_is_the_one_to_act(
        cx: &mut gpui::TestAppContext,
    ) {
        let root = scratch("frames-own");
        let store = Store::at_test_root(&root);
        let own = transcribed(&store, "my words");
        let other = transcribed(&store, "their words");
        let (app, fake) = window_with_typing(cx, &store);
        app.update(cx, |app, _| {
            for id in [&own, &other] {
                app.delivery_take_started();
                let capture = app.delivery_take_stopped();
                app.bind_delivery(capture, id);
                app.host.awaiting.insert(id.clone());
            }
        });
        // Acted on elsewhere: shown here, typed nowhere here.
        frame(&app, cx, completed(&other, None, false));
        assert!(fake.insertions().is_empty(), "{:?}", fake.insertions());
        assert!(app.read_with(cx, |app, _| !app.host.awaiting.contains(&other)));
        frame(&app, cx, completed_with(&own, None, true, "my words"));
        settle(cx, "the own take typed", |_| !fake.insertions().is_empty());
        assert_eq!(fake.insertions().len(), 1);
        assert!(fake.insertions()[0].1.contains("my words"), "{:?}", fake.insertions());
        let _ = std::fs::remove_dir_all(&root);
    }

    /// A retry's result is offered for Copy / Paste last, never typed —
    /// even into the editor the take was first dictated into.
    #[gpui::test]
    fn a_retried_result_is_offered_not_typed(cx: &mut gpui::TestAppContext) {
        let root = scratch("frames-retry");
        let store = Store::at_test_root(&root);
        let id = transcribed(&store, "retried words");
        let (app, fake) = window_with_typing(cx, &store);
        app.update(cx, |app, _| {
            app.host.requests.insert(
                "tr_1".to_string(),
                crate::remote_take::Request {
                    stored_id: id.clone(),
                    _hold: None,
                    offer: true,
                },
            );
        });
        frame(&app, cx, completed(&id, Some("tr_1"), true));
        settle(cx, "the offer", |cx| app.read_with(cx, |app, _| app.delivery.recovery.is_some()));
        assert!(fake.insertions().is_empty());
        assert!(app.read_with(cx, |app, _| app.host.requests.is_empty()));
        let _ = std::fs::remove_dir_all(&root);
    }

    /// Live text from the host lands in the running take's live line; a
    /// preview for a take that is not running here is ignored, and a
    /// degradation says why live text stopped.
    #[gpui::test]
    fn live_text_follows_the_running_take_only(cx: &mut gpui::TestAppContext) {
        let app = cx.new(|cx| StarlingApp::for_test(None, cx));
        app.update(cx, |app, _| {
            app.recorder = Some(crate::host_link::LiveCapture::new("take_live".to_string()));
        });
        let text = |take: &str, words: &str| TakeUpdate::LiveText {
            take: take.to_string(),
            partial: Some(starling_runtime_host::frame::LivePartial {
                text: words.to_string(),
                stable_words: 0,
                covered_s: None,
            }),
            degraded: None,
        };
        frame(&app, cx, text("take_live", "hello there"));
        assert_eq!(app.read_with(cx, |app, _| app.live_partial.clone()), "hello there");
        frame(&app, cx, text("take_other", "not mine"));
        assert_eq!(app.read_with(cx, |app, _| app.live_partial.clone()), "hello there");
        frame(
            &app,
            cx,
            TakeUpdate::LiveText {
                take: "take_live".to_string(),
                partial: None,
                degraded: Some("the stream went away".to_string()),
            },
        );
        assert_eq!(
            app.read_with(cx, |app, _| app.stream_degradation.clone()).as_deref(),
            Some("the stream went away")
        );
    }

    /// The service going away mid-take says so — the take did not carry
    /// on there — while a dropped connection says the take continues.
    #[gpui::test]
    fn a_lost_service_and_a_lost_connection_say_different_things(cx: &mut gpui::TestAppContext) {
        for (host_gone, says) in [(true, "stopped while recording"), (false, "continues there")] {
            let app = cx.new(|cx| StarlingApp::for_test(None, cx));
            app.update(cx, |app, cx| {
                app.recorder = Some(crate::host_link::LiveCapture::new("take_x".to_string()));
                app.host_update(
                    HostUpdate::Disconnected {
                        reason: "gone".to_string(),
                        gave_up: false,
                        host_gone,
                    },
                    cx,
                );
            });
            let error = app.read_with(cx, |app, _| app.error.clone()).expect("explained");
            assert!(error.contains(says), "{error}");
            assert!(app.read_with(cx, |app, _| app.recorder.is_none()));
        }
    }
}
