//! The recording -> encode -> persist -> transcribe pipeline and the file
//! import flow, split out of `app.rs`.

use std::sync::Arc;
use std::time::Instant;

use gpui::{AppContext, AsyncApp, Context, PathPromptOptions, WeakEntity};
use starling_dictation::{
    audio,
    client::{ClientError, StarlingClient},
    disk::{self, DiskLevel},
    engine::EngineLease,
    journal,
    recorder,
    settings::EngineMode,
    storage,
};

use crate::app::{HealthCheckPurpose, StarlingApp, UnsavedWav};
use crate::live_stream::LiveStream;
use crate::mic::Interruption;
use crate::store::{AudioPin, Store};

/// The capture rate a pre-start disk estimate assumes (the device rate
/// is only known once the microphone opened).
const PREFERRED_RATE_FOR_ESTIMATES: u32 = 48_000;

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

/// The note recorded on a quiesce-timeout salvage (I1 phase 2, R17): the
/// salvaged take is persisted as interrupted rather than dropped, and the
/// note states exactly what was kept.
pub(crate) fn quiesce_salvage_note(samples: u64, sample_rate: u32) -> String {
    format!(
        "The microphone did not stop cleanly within the quiesce timeout; all {samples} \
         captured samples were salvaged and kept as this interrupted recording (device rate \
         {sample_rate} Hz). You can retry transcription on it."
    )
}

/// The cancelled take's audio (see `StarlingApp::cancel_recording`):
/// whatever the recorder's stop handed back, with the live-stream prefix
/// the take already drained spliced back in front — or, when the stop
/// handed back nothing at all, the prefix alone, rebuilt at the take's
/// device rate. The prefix is samples this app already owns
/// (`drain_chunks` hands them out for live streaming), so a stop that
/// reports `Empty` or fails outright must not lose them: a cancel never
/// loses words.
fn salvaged_take_audio(
    stopped: Option<audio::PcmAudio>,
    streamed_prefix: Vec<f32>,
    sample_rate: u32,
) -> Option<audio::PcmAudio> {
    match stopped {
        Some(mut audio) => {
            audio.samples.splice(0..0, streamed_prefix);
            Some(audio)
        }
        None if !streamed_prefix.is_empty() => Some(audio::PcmAudio {
            samples: streamed_prefix,
            sample_rate,
            channels: 1,
        }),
        None => None,
    }
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

    /// Stop the running recorder and process the take as usual. Only the
    /// activation machine calls this (an `Effect::Finish`).
    pub(crate) fn stop_recording(
        &mut self,
        finished_take: crate::activation::TakeId,
        cx: &mut Context<Self>,
    ) {
        self.error = None;
        // Playback comes back when recording stops, not after transcription.
        self.end_playback_lease();
        if let Some(handle) = self.recorder.take() {
            self.audio_upkeep.set_recording(false);
            // The worker stops before the recorder does, so `stop` below
            // returns exactly the samples after the drained ones.
            let pumped = self.finish_stream_pump();
            let mut stream = pumped.stream;
            if let Some(reason) = pumped.degradation {
                self.stream_degradation = Some(reason);
            }
            // The binding resolved at START leaves with the take (#363);
            // a stop without one (not a normal path) resolves fresh
            // rather than transcribing against nothing.
            let target = self
                .active_take
                .take()
                .unwrap_or_else(|| self.resolve_take_target());
            self.live_partial.clear();
            let streamed_samples = pumped.samples;
            let sent_samples = pumped.sent;
            // G03: clipping is measured on the raw captured samples (before
            // the attenuation-only auto gain), so an already-clipped source
            // stays visible even when its attenuated copy peaks below
            // full scale.
            let source_clip_ratio = handle.source_clip_ratio();
            // Storage-fault honesty (I1 phase 2): read the capture-path
            // error before `stop` consumes the handle — a successful take
            // must not swallow a journal fault that froze acknowledgment.
            let capture_fault = handle.capture_error();
            if capture_fault.is_some() {
                stream = None;
            }
            let device = crate::mic::device_name(&handle);
            let device_sample_rate = handle.sample_rate();
            // Stop-to-processed latency (#295) starts here.
            let stopped_at = Instant::now();
            // The staging panel keeps its draft while the take is saved.
            let staging = self.stop_staging();
            // Where the take's text goes (#221) travels with it too.
            let delivery = self.delivery_take_stopped();
            match handle.stop() {
                // The device failed after the last input check: the take
                // was interrupted and is never transcribed as complete.
                Ok(take) if take.device_fault.is_some() => {
                    if let Some(fault) = &take.device_fault {
                        self.note_interruption(&device, Interruption::DeviceFailed(fault.clone()));
                    }
                    self.keep_untranscribed_take(
                        finished_take,
                        crate::activation::CancelReason::InputLost,
                        Ok(take),
                        streamed_samples,
                        device_sample_rate,
                        cx,
                    );
                }
                Ok(mut take) => {
                    // `sent_samples` indexes device-rate samples of the
                    // spliced layout (drained stream prefix + journal
                    // tail) — the same units the stream worker advanced
                    // it in, so the remainder slice below lines up exactly.
                    take.audio.samples.splice(0..0, streamed_samples);
                    // Streaming requires a finalized, fault-free journal:
                    // production captures always journal (see the start
                    // branch), so a missing or faulted report means the
                    // durable copy cannot be trusted to match what the
                    // stream already sent — the take must not commit a
                    // stream result over it.
                    if !matches!(take.journal.as_ref(), Some(report) if report.finalized && report.fault.is_none()) {
                        stream = None;
                    }
                    self.levels = vec![0.06; 52];
                    self.capture_warning = recorder::clipping_warning(source_clip_ratio)
                        .or_else(|| self.stream_degradation.take());
                    if let Some(fault) = capture_fault {
                        self.error = Some(fault);
                    }
                    // The journal itself becomes the stored audio (adopted
                    // by the facade).
                    let journal_report = take.journal.clone();
                    self.overlay.model.take_finished(stopped_at);
                    cx.notify();
                    cx.spawn(async move |this, cx| {
                        let encoded = cx
                            .background_spawn(async move {
                                // The un-streamed remainder — everything
                                // past the send watermark — is encoded and
                                // sent here, beside the save-path encode,
                                // never on the UI thread: after a mid-take
                                // stall it can be as long as the rest of
                                // the take. A failed encode or send drops
                                // the stream, so the job's final result
                                // stays gated on the remainder actually
                                // reaching the server (otherwise the full
                                // upload below covers it).
                                let remainder_sent = match stream.as_ref() {
                                    Some(live) if sent_samples < take.audio.samples.len() => {
                                        audio::encode_wav_16k_parts(
                                            &take.audio.samples[sent_samples..],
                                            take.audio.sample_rate,
                                            1,
                                        )
                                        .map(|wav| live.send_audio(wav))
                                        .unwrap_or(false)
                                    }
                                    _ => true,
                                };
                                if !remainder_sent {
                                    stream = None;
                                }
                                (audio::encode_wav_16k(&take.audio), stream)
                            })
                            .await;
                        match encoded {
                            (Ok(wav), stream) => {
                                this.update(cx, |app, cx| {
                                    app.save_and_transcribe(
                                        Arc::new(wav),
                                        journal_report,
                                        stream,
                                        Some(stopped_at),
                                        staging,
                                        delivery,
                                        target,
                                        cx,
                                    );
                                })
                                .ok();
                            }
                            (Err(err), _) => {
                                this.update(cx, |app, cx| {
                                    app.overlay.model.save_failed(stopped_at, Instant::now());
                                    app.error = Some(err.to_string());
                                    if let Some(token) = staging {
                                        app.staging_save_failed(token, cx);
                                    }
                                    cx.notify();
                                })
                                .ok();
                            }
                        }
                    })
                    .detach();
                }
                Err(recorder::RecorderError::QuiesceTimeout {
                    acknowledged_samples,
                    mut audio,
                    journal,
                }) => {
                    audio.samples.splice(0..0, streamed_samples);
                    // R17 / I1 phase 2: a device hiccup must not silently
                    // discard acknowledged audio. The salvaged samples are
                    // persisted as an interrupted-but-usable take, linked
                    // to its (already finalized) journal; the quiesce gap
                    // is recorded in the session note.
                    let journal_report = journal.clone();
                    let note = quiesce_salvage_note(acknowledged_samples, audio.sample_rate);
                    self.staging_interrupted(cx);
                    self.levels = vec![0.06; 52];
                    self.capture_warning = recorder::clipping_warning(source_clip_ratio);
                    self.error = Some(format!(
                        "The microphone did not stop cleanly within the quiesce timeout; \
                         {acknowledged_samples} captured samples were preserved and not \
                         lost. It was saved to your history as an interrupted recording."
                    ));
                    cx.notify();
                    cx.spawn(async move |this, cx| {
                        let encoded = cx
                            .background_spawn(async move {
                                audio::encode_wav_16k(&audio)
                            })
                            .await;
                        match encoded {
                            Ok(wav) => {
                                this.update(cx, |app, cx| {
                                    app.save_interrupted_take(
                                        Arc::new(wav),
                                        journal_report,
                                        note,
                                        None,
                                        None,
                                        cx,
                                    );
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
                    })
                    .detach();
                }
                Err(err) => {
                    self.error = Some(err.to_string());
                    self.staging_interrupted(cx);
                    cx.notify();
                }
            }
        }
    }

    /// Start a take's recorder; `false` when the microphone could not be
    /// opened (the reason is in the error banner). Only the activation
    /// machine calls this (an `Effect::Start`).
    pub(crate) fn start_recording(&mut self, cx: &mut Context<Self>) -> bool {
        // Where the take's text goes (#221): captured first, before the
        // microphone or anything else can move focus.
        self.delivery_take_started();
        self.error = None;
        self.take_notice = None;
        // Attenuation begins with the attempt, before the microphone opens
        // — unless a start cue is due, which it would swallow: then it
        // begins once the cue has played (`cues.rs`).
        let playback_lease = (!crate::cues::attenuation_waits_for_cue(
            &self.feedback,
            self.playback_settings.during_recording,
        ))
        .then(|| self.playback.handle().begin(&self.playback_settings));
        {
            // I1 phase 2: production captures journal to the durable
            // per-take file; only fsynced-boundary samples are
            // acknowledged (see recorder::start_recording_with_journal).
            let journals_root = journal::default_journals_root();
            // #342: no take starts on a disk too full to keep it; a low
            // disk starts with a warning. A probe that cannot answer
            // never blocks recording.
            let disk_watch = disk::DiskWatch::system();
            let disk_reading = disk_watch
                .policy
                .check(disk_watch.probe.as_ref(), &journals_root)
                .ok();
            if let Some(reading) = disk_reading.filter(|reading| reading.level == DiskLevel::Critical)
            {
                self.delivery_take_stopped();
                if let Some(lease) = playback_lease {
                    self.playback.handle().end(lease);
                }
                self.error = disk_watch.policy.warning(reading, PREFERRED_RATE_FOR_ESTIMATES);
                cx.notify();
                return false;
            }
            match recorder::start_capture(recorder::CaptureRequest {
                journals_dir: Some(&journals_root),
                preferred_device: self.microphone_settings.preferred_device.as_deref(),
                disk_watch: Some(disk_watch.clone()),
            }) {
                Ok(handle) => {
                    self.mic.problem = None;
                    self.mic.interruption = None;
                    self.mic.last_sound_at = None;
                    self.mic.last_route = handle.input_route().cloned();
                    self.live_partial.clear();
                    if self.staged_mode() {
                        self.begin_staging(cx);
                    } else {
                        self.retire_staging(cx);
                    }
                    self.stream_degradation = None;
                    // #363: the take's endpoint/model binding resolves at
                    // START and travels with the take — a model switch or
                    // settings save mid-take cannot move it.
                    let target = self.resolve_take_target();
                    if target.endpoint().is_empty() {
                        // Builtin mode with no ready engine: recording
                        // proceeds (the audio is journaled either way);
                        // the note says transcription will need a retry.
                        self.stream_degradation = Some(
                            "The built-in engine is not ready; the recording is still saved, \
                             and transcription will need a retry once the engine is ready."
                                .to_string(),
                        );
                    } else {
                        // A URL-shape failure is deterministic, so it gets the
                        // same visible degradation note as a mid-recording
                        // death instead of a silent `.ok()` downgrade.
                        if let Err(reason) = self.start_stream_pump(&handle, target.endpoint(), cx) {
                            self.stream_degradation = Some(format!(
                                "Live transcription is unavailable ({reason}); the recording \
                                 will be uploaded in full after you stop."
                            ));
                        }
                    }
                    self.active_take = Some(target);
                    let handle_rate = handle.sample_rate();
                    self.recorder = Some(handle);
                    self.audio_upkeep.set_recording(true);
                    self.playback_lease = playback_lease;
                    self.elapsed_ms = 0.0;
                    self.levels = vec![0.06; 52];
                    self.capture_warning = disk_reading.and_then(|reading| {
                        disk_watch.policy.warning(reading, handle_rate)
                    });
                    self.mic.disk_warned = self.capture_warning.is_some();
                    self.mic.disk_low_warning = self.capture_warning.clone();
                    self.mic.disk_unchecked = false;
                    cx.notify();
                    true
                }
                Err(err) => {
                    self.delivery_take_stopped();
                    if let Some(lease) = playback_lease {
                        self.playback.handle().end(lease);
                    }
                    let text = format!("{} {}", err.problem.message(), err.problem.recovery());
                    self.report_input_problem(err.problem, text);
                    cx.notify();
                    false
                }
            }
        }
    }

    /// Restores playback for the live take (asynchronously).
    fn end_playback_lease(&mut self) {
        if let Some(lease) = self.playback_lease.take() {
            self.playback.handle().end(lease);
        }
    }

    /// Stop the running recorder without transcribing or delivering
    /// anything (#221): Escape, a stop before the microphone delivered
    /// audio, or a microphone that never did. Whatever audio was captured
    /// is kept — saved to history as an interrupted take the user can
    /// transcribe later — so a cancel never loses words.
    pub(crate) fn cancel_recording(
        &mut self,
        cancelled_take: crate::activation::TakeId,
        reason: crate::activation::CancelReason,
        cx: &mut Context<Self>,
    ) {
        self.end_playback_lease();
        // A cancelled take delivers nothing (#221).
        self.delivery_take_stopped();
        let Some(handle) = self.recorder.take() else {
            return;
        };
        self.audio_upkeep.set_recording(false);
        // The stream and the engine lease leave with the take: nothing
        // is transcribed.
        let streamed_samples = self.finish_stream_pump().samples;
        self.active_take = None;
        self.live_partial.clear();
        // Read before `stop` consumes the handle: a stop that hands back
        // no audio still rebuilds the drained prefix at the take's own
        // device rate.
        let device_sample_rate = handle.sample_rate();
        let stopped = handle.stop();
        self.keep_untranscribed_take(
            cancelled_take,
            reason,
            stopped,
            streamed_samples,
            device_sample_rate,
            cx,
        );
    }

    /// Saves a take that ends without transcription — cancelled, or
    /// interrupted by its microphone — to history as an interrupted take,
    /// from the recorder's `stopped` result and the live-stream prefix
    /// already drained from it.
    fn keep_untranscribed_take(
        &mut self,
        cancelled_take: crate::activation::TakeId,
        reason: crate::activation::CancelReason,
        stopped: Result<recorder::CapturedTake, recorder::RecorderError>,
        streamed_samples: Vec<f32>,
        device_sample_rate: u32,
        cx: &mut Context<Self>,
    ) {
        use crate::activation::CancelReason;
        self.stream_degradation = None;
        self.levels = vec![0.06; 52];
        let staging = self.staging_cancelled(cx);
        // This stop's own failure, kept apart from `self.error` (which may
        // hold an unrelated earlier message) so the stall report below
        // only ever joins what actually happened to this take.
        let mut stop_error = None;
        let (stopped, journal_report) = match stopped {
            Ok(take) => (Some(take.audio), take.journal),
            Err(recorder::RecorderError::QuiesceTimeout { audio, journal, .. }) => {
                (Some(audio), journal)
            }
            Err(recorder::RecorderError::Empty) => (None, None),
            Err(err) => {
                stop_error = Some(format!(
                    "{}. The capture journal, if this take had one, stays on disk \
                     for recovery.",
                    err.to_string().trim_end_matches('.')
                ));
                // The stop failed, so no take came back to bind the kept
                // draft to; the salvage below may still produce audio from
                // the drained prefix, and when it does not, the no-audio
                // guard below resolves the draft — once.
                (None, None)
            }
        };
        // The prefix drained for live streaming is spliced in front of
        // whatever the stop handed back, or becomes the whole take when it
        // handed back nothing (`Empty`, a failed stop): samples this app
        // already owns are never dropped by a cancel.
        let audio = salvaged_take_audio(stopped, streamed_samples, device_sample_rate);
        // The journal rides along whatever its state: the store adopts
        // only a clean, finalized one (a faulted one holds just the prefix
        // before the fault, so the full in-memory take is stored instead)
        // and moves a journal it did not adopt aside (#356).
        let kept = audio.as_ref().is_some_and(|audio| !audio.samples.is_empty());
        // Notices that promise history are shown only once the save below
        // lands — a failed save explains itself through the error banner
        // instead, so the notice can never contradict what happened.
        // The notice belongs to `cancelled_take` (the effect's own id, not
        // the newest: a late second tap cancels one take and starts the
        // next in the same step) and is shown only while no later take has
        // started.
        let saved_notice = match reason {
            CancelReason::Escape if kept => {
                Some(
                    "Cancelled with Escape. No transcript was kept and nothing was inserted \
                     anywhere; the audio is in your history, ready to transcribe if you need \
                     the words."
                        .to_string(),
                )
            }
            CancelReason::Escape => {
                self.take_notice = Some(
                    "Cancelled with Escape before any audio was captured.".to_string(),
                );
                None
            }
            CancelReason::NoAudioYet if kept => {
                Some(
                    "Stopped before the microphone was fully ready. The little audio it \
                     captured is in your history; no transcript was kept."
                        .to_string(),
                )
            }
            CancelReason::NoAudioYet => {
                self.take_notice = Some(
                    "Stopped before the microphone delivered any audio; nothing was \
                     recorded."
                        .to_string(),
                );
                None
            }
            CancelReason::MicStalled => {
                let mut stall = format!(
                    "The microphone delivered no audio within {} seconds, so the take was \
                     stopped. Check that the input device is connected and not muted.",
                    crate::activation::START_STALL.as_secs()
                );
                // The stop above may already have surfaced its own failure
                // (the generic `Err` arm); a stall must not overwrite it —
                // both stay, joined at a sentence boundary.
                if let Some(stop_error) = stop_error.take() {
                    stall = format!(
                        "{stall} Stopping it also failed: {}",
                        stop_error.trim_end_matches('.')
                    );
                }
                self.error = Some(stall);
                // With audio kept, the saved notice tells the user the words
                // survived once the save lands — the error above only says
                // why the take was stopped.
                kept.then(|| {
                    "The microphone stalled at the start, so the take was stopped. The audio \
                     it did capture is in your history; no transcript was kept."
                        .to_string()
                })
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
                let mut lost = format!(
                    "{what}. Check the microphone (Settings → Microphone can test it or pick \
                     another), then record again."
                );
                if let Some(stop_error) = stop_error.take() {
                    lost = format!(
                        "{lost} Stopping it also failed: {}",
                        stop_error.trim_end_matches('.')
                    );
                }
                self.report_input_problem(problem, lost);
                kept.then(|| {
                    "The recording was interrupted when the microphone stopped. Everything \
                     captured before that is in your history as an interrupted recording; \
                     transcribe it from there."
                        .to_string()
                })
            }
        };
        if let Some(stop_error) = stop_error {
            self.error = Some(stop_error);
        }
        cx.notify();
        let Some(audio) = audio.filter(|audio| !audio.samples.is_empty()) else {
            // Nothing was captured, so no take is saved for the kept draft
            // to bind to: fail it here rather than leave it dangling — the
            // single resolution for every path that arrives with no audio.
            if let Some(token) = staging {
                self.staging_save_failed(token, cx);
            }
            return;
        };
        let note = match reason {
            CancelReason::Escape => "Cancelled with Escape before transcription; the audio was kept.",
            CancelReason::NoAudioYet => "Stopped before the microphone was ready; the audio was kept.",
            CancelReason::MicStalled => "The microphone stalled at the start; the audio was kept.",
            CancelReason::InputLost => {
                "The microphone failed or stopped delivering mid-recording; the audio captured \
                 before that was kept."
            }
        }
        .to_string();
        cx.spawn(async move |this, cx| {
            let encoded = cx
                .background_spawn(async move { audio::encode_wav_16k(&audio) })
                .await;
            let saved = this.update(cx, |app, cx| {
                match encoded {
                    Ok(wav) => app.save_interrupted_take(
                        Arc::new(wav),
                        journal_report,
                        note,
                        saved_notice.map(|notice| (notice, cancelled_take)),
                        staging,
                        cx,
                    ),
                    Err(err) => {
                        app.error = Some(err.to_string());
                        // The take exists only as audio; with no history
                        // entry to bind to, the kept draft fails here.
                        if let Some(token) = staging {
                            app.staging_save_failed(token, cx);
                        }
                        cx.notify();
                    }
                }
            });
            if saved.is_err() {
                // The window closed while the take was encoding: nothing
                // can persist the WAV, and the journal, if any, stays on
                // disk.
                eprintln!("a cancelled take could not be saved: the app window is gone");
            }
        })
        .detach();
    }

    /// The target travels with the take through here (#363); the lease it
    /// may hold is dropped when the transcription job finishes.
    #[allow(clippy::too_many_arguments)]
    pub fn save_and_transcribe(
        &mut self,
        wav: Arc<Vec<u8>>,
        journal: Option<recorder::JournalReport>,
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
                    create_store.save_capture(job_wav, journal.as_ref())
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

    /// Persists a salvaged or recovered take as interrupted-but-usable (I1
    /// phase 2): the audio goes through the same storage path, then the
    /// session is marked interrupted with `note` stating exactly what
    /// survived. No transcription is started — the user decides to retry.
    /// `notice`, when set, is the user-facing take notice, shown only
    /// once the take really is in history.
    #[allow(clippy::too_many_arguments)]
    pub fn save_interrupted_take(
        &mut self,
        wav: Arc<Vec<u8>>,
        journal: Option<recorder::JournalReport>,
        note: String,
        notice: Option<(String, crate::activation::TakeId)>,
        staging: Option<u64>,
        cx: &mut Context<Self>,
    ) {
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
            cx.notify();
            return;
        };
        cx.spawn(async move |this, cx| {
            let job_wav = wav.clone();
            let note_store = store.clone();
            let created = cx
                .background_spawn(async move {
                    note_store.save_interrupted_capture(job_wav, journal.as_ref(), &note)
                })
                .await;
            match created {
                Ok(id) => {
                    if notice.is_some() || staging.is_some() {
                        this.update(cx, |app, cx| {
                            if let Some(token) = staging {
                                app.bind_staging(token, &id);
                            }
                            // The take is in history only now (#221): the
                            // cancel notice that says so lands here — unless
                            // a newer take started meanwhile, which it would
                            // misdescribe.
                            if let Some((notice, take)) = notice {
                                if app.activation.last_started() == take {
                                    app.take_notice = Some(notice);
                                }
                            }
                            cx.notify();
                        })
                        .ok();
                    }
                    refresh_sessions(&this, &store, cx).await;
                }
                Err(err) => {
                    this.update(cx, |app, cx| {
                        if let Some(token) = staging {
                            app.staging_save_failed(token, cx);
                        }
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

    /// The never-drop-audio fallback shared by every persist path (I1
    /// phase 2 folds the salvage paths into it too): when the session
    /// store cannot take the WAV, it lands in the unsaved list with
    /// `message` surfaced, so the only copy is never discarded.
    fn stash_unsaved(&mut self, wav: Arc<Vec<u8>>, message: &str) {
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
    fn transcribe_with_stream(
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
                                    let write_error = if session_found {
                                        store.save_transcript(&id, result).err()
                                    } else {
                                        None
                                    };
                                    Ok::<_, storage::StorageError>((session_found, write_error))
                                })
                                .await
                            };
                            match saved {
                                Ok((session_found, write_error)) => {
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
            // follows as a proposal (#295).
            if transcribed {
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
        // A newer choice replaces a retry still loading its audio or
        // waiting for its model.
        self.pending_retry = None;
        self.retry_seq += 1;
        let seq = self.retry_seq;
        if with == RetryWith::Server && self.endpoint.trim().is_empty() {
            self.error = Some(
                "No server is set up: add its endpoint in Settings → Engine, then retry."
                    .to_string(),
            );
            cx.notify();
            return;
        }
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
        engine.activate(&model_id);
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
            loop {
                cx.background_executor()
                    .timer(std::time::Duration::from_millis(250))
                    .await;
                let done = this.update(cx, |app, cx| {
                    if app.pending_retry.as_ref() != Some(&pending)
                        || app.engine_instance != instance
                    {
                        // Replaced by another choice, or the engine itself
                        // was replaced (a mode switch): this wait is over.
                        return true;
                    }
                    if app.retry_target_gone(&pending.take_id) {
                        // Deleted while the model loaded: nothing to retry.
                        app.pending_retry = None;
                        cx.notify();
                        return true;
                    }
                    match switch_progress(&engine.snapshot(), &model_id, started.elapsed()) {
                        SwitchProgress::Waiting => false,
                        SwitchProgress::Ready => {
                            let Some(lease) =
                                engine.lease().filter(|lease| lease.model_id() == model_id)
                            else {
                                return false;
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
/// How long the engine may show no sign of the requested switch before
/// the request counts as not taken (the command is asynchronous).
const RETRY_SWITCH_GRACE: std::time::Duration = std::time::Duration::from_secs(3);

/// Read an engine snapshot for a retry waiting on `model_id`.
pub(crate) fn switch_progress(
    snapshot: &starling_dictation::engine::EngineSnapshot,
    model_id: &str,
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
    if let Some(SwapDecision::Refused { .. }) = snapshot.pending_decision {
        return SwitchProgress::Failed(
            "there is not enough free memory to load that model.".to_string(),
        );
    }
    if let Some(switch) = &snapshot.switch {
        return if switch.target_model_id == model_id {
            SwitchProgress::Waiting
        } else {
            SwitchProgress::Failed("another model switch replaced it.".to_string())
        };
    }
    if let EnginePhase::Failed(failure) = &snapshot.phase {
        return SwitchProgress::Failed(format!("{failure}"));
    }
    if serving || waited < RETRY_SWITCH_GRACE {
        // Loading, warming, restarting — or the command is not picked up
        // yet.
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

/// Surfaces what startup recovery found: problems in the error banner,
/// takes brought back in a notice. Returns the journal ids to look at
/// again (see [`recheck_capture_journals`]).
pub(crate) fn show_startup_recovery(
    this: &WeakEntity<StarlingApp>,
    recovery: crate::store::StartupRecovery,
    cx: &mut AsyncApp,
) -> Vec<String> {
    let crate::store::StartupRecovery {
        summary,
        notice,
        recheck,
        failure,
    } = recovery;
    let error = [
        failure.map(|err| format!("Could not recover interrupted recordings: {err}")),
        (!summary.is_empty()).then_some(summary),
    ]
    .into_iter()
    .flatten()
    .collect::<Vec<_>>()
    .join(" ");
    this.update(cx, |app, cx| {
        if !error.is_empty() {
            app.error = Some(error);
        }
        if !notice.is_empty() {
            app.recovery_notice = Some(notice);
        }
        cx.notify();
    })
    .ok();
    recheck
}

/// The second look at the recorder's tree (#356): journals the startup
/// pass left to a writer or a save that may have been under way are
/// recovered once that save would long have finished, in this launch
/// rather than the next. Only those ids: a take this launch started
/// recording since is never a recovery candidate.
pub(crate) async fn recheck_capture_journals(
    this: &WeakEntity<StarlingApp>,
    store: &Store,
    ids: Vec<String>,
    cx: &mut AsyncApp,
) {
    cx.background_executor()
        .timer(starling_dictation::store_v2::FINALIZED_ADOPTION_GRACE + std::time::Duration::from_secs(2))
        .await;
    let recovered = {
        let store = store.clone();
        cx.background_spawn(async move {
            store.recover_capture_journals(|id| ids.iter().any(|wanted| wanted == id))
        })
        .await
    };
    let (problems, notice) = match recovered {
        Ok(recovery) => (recovery.problems(), recovery.recovered_summary()),
        Err(err) => (
            format!("Could not scan for interrupted recordings: {err}"),
            String::new(),
        ),
    };
    if problems.is_empty() && notice.is_empty() {
        return;
    }
    this.update(cx, |app, cx| {
        if !problems.is_empty() {
            app.error = Some(problems);
        }
        if !notice.is_empty() {
            app.recovery_notice = Some(notice);
        }
        cx.notify();
    })
    .ok();
    refresh_sessions(this, store, cx).await;
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
    fn the_quiesce_salvage_note_states_exactly_what_was_kept() {
        // R17 / I1 phase 2: the note must say the audio was kept (not lost,
        // not silently trimmed), carry the exact sample count and device
        // rate, and point at retry.
        let note = quiesce_salvage_note(12_345, 48_000);
        assert!(note.contains("12345"), "{note}");
        assert!(note.contains("48000"), "{note}");
        assert!(note.contains("salvaged and kept"), "{note}");
        assert!(note.contains("interrupted recording"), "{note}");
        assert!(note.contains("retry"), "{note}");
        assert!(!note.contains("lost"), "{note}");
    }

    #[test]
    fn a_cancelled_take_splices_the_streamed_prefix_in_front_of_the_stop_audio() {
        // The drained live-stream prefix is the start of the take, so it
        // goes in front of whatever the stop handed back — never after.
        let stopped = audio::PcmAudio {
            samples: vec![0.5, 0.25],
            sample_rate: 48_000,
            channels: 1,
        };
        let salvaged = salvaged_take_audio(Some(stopped), vec![0.1, 0.2, 0.3], 48_000).unwrap();
        assert_eq!(salvaged.samples, vec![0.1, 0.2, 0.3, 0.5, 0.25]);
        assert_eq!(salvaged.sample_rate, 48_000);
    }

    #[test]
    fn a_stop_that_hands_back_nothing_still_keeps_the_streamed_prefix() {
        // `Empty` and a failed stop must not drop samples the app already
        // drained: the prefix alone is the interrupted take, mono, at the
        // device rate captured before `stop` consumed the handle.
        let salvaged = salvaged_take_audio(None, vec![0.1, 0.2], 44_100).unwrap();
        assert_eq!(salvaged.samples, vec![0.1, 0.2]);
        assert_eq!(salvaged.sample_rate, 44_100);
        assert_eq!(salvaged.channels, 1);
    }

    #[test]
    fn a_stop_with_no_audio_and_no_prefix_salvages_nothing() {
        assert!(salvaged_take_audio(None, Vec::new(), 48_000).is_none());
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
        }
    }

    fn switching_to(model_id: &str) -> Option<SwitchView> {
        Some(SwitchView {
            target_model_id: model_id.to_string(),
            stage: SwitchStage::Loading,
            started: Instant::now(),
        })
    }

    #[test]
    fn a_retry_waits_for_its_model_and_gives_up_honestly() {
        let secs = std::time::Duration::from_secs;
        // Serving the model: go.
        assert_eq!(
            switch_progress(&snapshot(Some("b"), EnginePhase::Ready), "b", secs(1)),
            SwitchProgress::Ready
        );
        // The switch to it is running, or the command is not picked up yet.
        let mut loading = snapshot(Some("a"), EnginePhase::Ready);
        loading.switch = switching_to("b");
        assert_eq!(switch_progress(&loading, "b", secs(30)), SwitchProgress::Waiting);
        assert_eq!(
            switch_progress(&snapshot(Some("a"), EnginePhase::Ready), "b", secs(1)),
            SwitchProgress::Waiting
        );
        assert_eq!(
            switch_progress(&snapshot(Some("b"), EnginePhase::Loading), "b", secs(30)),
            SwitchProgress::Waiting
        );
        // Everything else ends the wait with a reason.
        let mut replaced = snapshot(Some("a"), EnginePhase::Ready);
        replaced.switch = switching_to("c");
        assert!(matches!(switch_progress(&replaced, "b", secs(5)), SwitchProgress::Failed(_)));
        let mut refused = snapshot(Some("a"), EnginePhase::Ready);
        refused.pending_decision = Some(SwapDecision::Refused {
            needed: 2,
            available: 1,
        });
        assert!(matches!(
            switch_progress(&refused, "b", secs(1)),
            SwitchProgress::Failed(reason) if reason.contains("memory")
        ));
        assert!(matches!(
            switch_progress(
                &snapshot(None, EnginePhase::Failed(EngineFailure::LoadFailed("bad file".into()))),
                "b",
                secs(5)
            ),
            SwitchProgress::Failed(_)
        ));
        let mut ignored = snapshot(Some("a"), EnginePhase::Ready);
        ignored.last_error = Some("the model file failed verification".to_string());
        assert_eq!(
            switch_progress(&ignored, "b", secs(5)),
            SwitchProgress::Failed("the model file failed verification".to_string())
        );
        assert!(matches!(
            switch_progress(&loading, "b", RETRY_SWITCH_CAP),
            SwitchProgress::Failed(_)
        ));
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
        let id = store.save_capture(wav.clone(), None).expect("save").id;
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

        // An empty answer is kept as a result, never shown over words.
        retry(&app, cx, RetryWith::Server);
        let take = summary(&store, &id);
        assert_eq!(take.transcript.as_ref().expect("kept").text, "first words");
        assert_eq!(take.results.len(), 2);

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
        let id = store.save_capture(one_second_wav(), None).expect("save").id;
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

    /// A retry with the server when none is set up does nothing to the
    /// take and says what to do.
    #[gpui::test]
    fn a_retry_without_a_server_leaves_the_take_alone(cx: &mut gpui::TestAppContext) {
        let root = scratch("no-server");
        let store = Store::at_test_root(&root);
        let id = store.save_capture(one_second_wav(), None).expect("save").id;
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
}
