//! The recording -> encode -> persist -> transcribe pipeline and the file
//! import flow, split out of `app.rs`.

use std::sync::Arc;

use gpui::{AppContext, AsyncApp, Context, PathPromptOptions, WeakEntity};
use starling_dictation::{
    audio,
    client::{ClientError, StarlingClient},
    journal,
    recorder,
    storage::{self, FileSessionStore},
};

use crate::app::{StarlingApp, UnsavedWav, client_protocol};

/// What a failed job says about server reachability (R13).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum FailureClass {
    /// A transport-level failure — connection refused, timeout, network
    /// unreachable. The server may be down, so the connection badge needs
    /// a fresh health probe.
    Transport,
    /// Everything else: local storage failures (full disk), input
    /// validation, blocked redirects, HTTP error statuses, and
    /// protocol/parse errors. None of them says the server is
    /// unreachable, so none of them may prompt a probe.
    Local,
}

/// Classify a transcription-client failure by what it says about server
/// reachability (R13). Only [`ClientError::Transport`] (connection
/// refused, network unreachable, DNS and socket failures) and
/// [`ClientError::Timeout`] qualify: an HTTP error status or a blocked
/// redirect proves something answered on the endpoint, and
/// `Input`/`Protocol` failures never left this machine.
pub(crate) fn failure_class(err: &ClientError) -> FailureClass {
    match err {
        ClientError::Transport(_) | ClientError::Timeout(_) => FailureClass::Transport,
        ClientError::Input(_)
        | ClientError::Redirect(_)
        | ClientError::Http { .. }
        | ClientError::Protocol(_) => FailureClass::Local,
    }
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
/// longer shows a stale `Ready` until the next probe.
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

impl StarlingApp {
    pub fn toggle_recording(&mut self, cx: &mut Context<Self>) {
        self.error = None;
        if let Some(handle) = self.recorder.take() {
            let duration_ms = handle.elapsed().as_secs_f64() * 1000.0;
            // G03: clipping is measured on the raw captured samples (before
            // the attenuation-only auto gain), so an already-clipped source
            // stays visible even when its attenuated copy peaks below
            // full scale.
            let source_clip_ratio = handle.source_clip_ratio();
            // Storage-fault honesty (I1 phase 2): read the capture-path
            // error before `stop` consumes the handle — a successful take
            // must not swallow a journal fault that froze acknowledgment.
            let capture_fault = handle.capture_error();
            match handle.stop() {
                Ok(take) => {
                    self.levels = vec![0.06; 52];
                    self.capture_warning = recorder::clipping_warning(source_clip_ratio);
                    if let Some(fault) = capture_fault {
                        self.error = Some(fault);
                    }
                    // The journal→session linkage is additive metadata on
                    // the manifest; the audio goes through the existing
                    // storage path as before.
                    let journal_id = take
                        .journal
                        .as_ref()
                        .map(|report| report.id.clone());
                    cx.notify();
                    cx.spawn(async move |this, cx| {
                        let encoded = cx
                            .background_spawn(async move { audio::encode_wav_16k(&take.audio) })
                            .await;
                        match encoded {
                            Ok(wav) => {
                                this.update(cx, |app, cx| {
                                    app.save_and_transcribe(
                                        Arc::new(wav),
                                        Some(duration_ms),
                                        journal_id,
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
                Err(recorder::RecorderError::QuiesceTimeout {
                    acknowledged_samples,
                    audio,
                    journal,
                }) => {
                    // R17 / I1 phase 2: a device hiccup must not silently
                    // discard acknowledged audio. The salvaged samples are
                    // persisted as an interrupted-but-usable take, linked
                    // to its (already finalized) journal; the quiesce gap
                    // is recorded in the session note.
                    let journal_id = journal.as_ref().map(|report| report.id.clone());
                    let note = quiesce_salvage_note(acknowledged_samples, audio.sample_rate);
                    self.levels = vec![0.06; 52];
                    self.capture_warning = recorder::clipping_warning(source_clip_ratio);
                    self.error = Some(format!(
                        "The microphone did not stop cleanly within the quiesce timeout; \
                         {acknowledged_samples} captured samples were preserved and not \
                         lost. It was saved to your history as an interrupted recording."
                    ));
                    cx.notify();
                    cx.spawn(async move |this, cx| {
                        // Duration from the salvaged sample count at the
                        // device rate — not the wall clock, which includes
                        // the drain wait.
                        let duration_ms = audio.samples.len() as f64 * 1000.0
                            / audio.sample_rate as f64;
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
                                        duration_ms,
                                        journal_id,
                                        note,
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
                    cx.notify();
                }
            }
        } else {
            // I1 phase 2: production captures journal to the durable
            // per-take file; only fsynced-boundary samples are
            // acknowledged (see recorder::start_recording_with_journal).
            match recorder::start_recording_with_journal(&journal::default_journals_root()) {
                Ok(handle) => {
                    self.recorder = Some(handle);
                    self.elapsed_ms = 0.0;
                    self.levels = vec![0.06; 52];
                    self.capture_warning = None;
                    cx.notify();
                }
                Err(err) => {
                    self.error = Some(err.to_string());
                    cx.notify();
                }
            }
        }
    }

    pub fn save_and_transcribe(
        &mut self,
        wav: Arc<Vec<u8>>,
        duration_ms: Option<f64>,
        journal_id: Option<String>,
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
            cx.notify();
            return;
        };
        cx.spawn(async move |this, cx| {
            let bytes = (*wav).clone();
            let create_store = store.clone();
            let created = cx
                .background_spawn(async move {
                    create_store.create_with_journal(bytes, duration_ms, journal_id.as_deref())
                })
                .await;
            match created {
                Ok(session) => {
                    refresh_sessions(&this, &store, cx).await;
                    this.update(cx, |app, cx| {
                        app.transcribe(session.id.clone(), session.wav.clone(), cx);
                    })
                    .ok();
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

    /// Persists a salvaged or recovered take as interrupted-but-usable (I1
    /// phase 2): the audio goes through the same storage path, then the
    /// session is marked interrupted with `note` stating exactly what
    /// survived. No transcription is started — the user decides to retry.
    pub fn save_interrupted_take(
        &mut self,
        wav: Arc<Vec<u8>>,
        duration_ms: f64,
        journal_id: Option<String>,
        note: String,
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
            cx.notify();
            return;
        };
        cx.spawn(async move |this, cx| {
            let bytes = (*wav).clone();
            let note_store = store.clone();
            let created = cx
                .background_spawn(async move {
                    let session = note_store
                        .create_with_journal(bytes, Some(duration_ms), journal_id.as_deref())?;
                    note_store.mark_interrupted(&session.id, &note)?;
                    Ok::<_, storage::StorageError>(session)
                })
                .await;
            match created {
                Ok(_session) => {
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

    pub fn transcribe(&mut self, id: String, wav: Arc<Vec<u8>>, cx: &mut Context<Self>) {
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

        let endpoint = self.endpoint.clone();
        let protocol = client_protocol(self.protocol);
        let model = self.model.clone();
        let store_for_job = store.clone();

        cx.spawn(async move |this, cx| {
            enum Outcome {
                Success,
                Failure { message: String, class: FailureClass },
            }
            let mut outcome = Outcome::Success;

            let marked = {
                let id = id.clone();
                let store = store_for_job.clone();
                cx.background_spawn(async move { store.mark_attempt(&id) })
                    .await
            };
            match marked {
                Ok(_) => {
                    refresh_sessions(&this, &store_for_job, cx).await;
                    let attempt = {
                        let wav = wav.clone();
                        let id = id.clone();
                        cx.background_spawn(async move {
                            let client = StarlingClient::new(&endpoint, protocol, &model)?;
                            client.transcribe(wav.as_slice(), &id)
                        })
                        .await
                    };
                    match attempt {
                        Ok(result) => {
                            let saved = {
                                let id = id.clone();
                                let store = store_for_job.clone();
                                cx.background_spawn(
                                    async move { store.save_transcript(&id, result) },
                                )
                                .await
                            };
                            if let Err(err) = saved {
                                // A local storage failure says nothing
                                // about reachability (R13).
                                outcome = Outcome::Failure {
                                    message: err.to_string(),
                                    class: FailureClass::Local,
                                };
                            }
                        }
                        Err(err) => {
                            // R13: the client error is classified while it
                            // is still typed — string matching later would
                            // be fragile.
                            outcome = Outcome::Failure {
                                message: err.to_string(),
                                class: failure_class(&err),
                            };
                        }
                    }
                }
                Err(err) => {
                    outcome = Outcome::Failure {
                        message: err.to_string(),
                        class: FailureClass::Local,
                    };
                }
            }

            let job_failure = match outcome {
                Outcome::Success => None,
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
                        if store.get(&id_for_failure)?.is_some() {
                            store.save_failure(&id_for_failure, failure_message)?;
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
                        // cannot flip the badge, and a transport failure
                        // whose probe succeeds shows the server recovered.
                        if class == FailureClass::Transport {
                            app.check_health(app.endpoint.clone(), cx);
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

            this.update(cx, |app, cx| {
                app.active_ids.remove(&id);
                cx.notify();
            })
            .ok();
            refresh_sessions(&this, &store_for_job, cx).await;
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
                                app.save_and_transcribe(
                                    Arc::new(prepared.wav),
                                    Some(prepared.duration_ms),
                                    None,
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

    pub fn retry_selected(&mut self, cx: &mut Context<Self>) {
        if let Some(session) = self.selected() {
            let id = session.id.clone();
            let wav = session.wav.clone();
            self.transcribe(id, wav, cx);
        }
    }
}

pub(crate) async fn refresh_sessions(
    this: &WeakEntity<StarlingApp>,
    store: &Arc<FileSessionStore>,
    cx: &mut AsyncApp,
) {
    let store = store.clone();
    let result = cx.background_spawn(async move { store.list() }).await;
    match result {
        Ok(list) => {
            this.update(cx, |app, cx| {
                app.apply_sessions(list);
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
                "error sending request for url (http://127.0.0.1:8181/transcribe): Connection \
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
        // this machine.
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
}
