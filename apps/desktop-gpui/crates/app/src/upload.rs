//! The recording -> encode -> persist -> transcribe pipeline and the file
//! import flow, split out of `app.rs`.

use std::sync::Arc;

use gpui::{AppContext, AsyncApp, Context, PathPromptOptions, WeakEntity};
use starling_dictation::{
    audio,
    client::StarlingClient,
    recorder,
    storage::{self, FileSessionStore},
};

use crate::app::{StarlingApp, UnsavedWav, client_protocol};

/// What a finished transcription job may apply to global app state (R03).
///
/// There is deliberately no connection variant: a job outcome conflates
/// transport failures with local storage failures (`mark_attempt`,
/// `save_transcript`, `save_failure` — think a full disk), so it cannot
/// decide `Connection::Offline` or `Connection::Ready`. `check_health` is
/// the single writer of connection state; a failed job surfaces as the
/// per-session error instead.
pub(crate) enum FinishedJob {
    /// Transcript saved: no global-state effects.
    Saved,
    /// The job failed: surface `message` as the app error.
    Failed { message: String },
}

/// Classify a finished transcription job (R03).
///
/// `job_failure` is the error that failed the job, if any;
/// `history_update_failure` is the error from the follow-up attempt to
/// record that failure in local history, if that write also failed.
pub(crate) fn finished_job(
    job_failure: Option<&str>,
    history_update_failure: Option<&str>,
) -> FinishedJob {
    match job_failure {
        None => FinishedJob::Saved,
        Some(failure) => FinishedJob::Failed {
            message: match history_update_failure {
                Some(storage_err) => {
                    format!("{failure} Local history update also failed: {storage_err}")
                }
                None => failure.to_string(),
            },
        },
    }
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
            match handle.stop() {
                Ok(pcm) => {
                    self.levels = vec![0.06; 52];
                    self.capture_warning = recorder::clipping_warning(source_clip_ratio);
                    cx.notify();
                    cx.spawn(async move |this, cx| {
                        let encoded = cx
                            .background_spawn(async move { audio::encode_wav_16k(&pcm) })
                            .await;
                        match encoded {
                            Ok(wav) => {
                                this.update(cx, |app, cx| {
                                    app.save_and_transcribe(Arc::new(wav), Some(duration_ms), cx);
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
            match recorder::start_recording() {
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
        cx: &mut Context<Self>,
    ) {
        self.error = None;
        let Some(store) = self.store.clone() else {
            let reason = self
                .store_error
                .clone()
                .unwrap_or_else(|| "session store unavailable".to_string());
            self.unsaved.push(UnsavedWav {
                id: format!("unsaved-{}", storage::now_iso()),
                wav,
                created_at: storage::now_iso(),
            });
            self.error = Some(format!(
                "Local storage failed: {reason} Keep this window open and download the unsaved WAV to recover it."
            ));
            cx.notify();
            return;
        };
        cx.spawn(async move |this, cx| {
            let bytes = (*wav).clone();
            let create_store = store.clone();
            let created = cx
                .background_spawn(async move { create_store.create(bytes, duration_ms) })
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
                        app.unsaved.push(UnsavedWav {
                            id: format!("unsaved-{}", storage::now_iso()),
                            wav,
                            created_at: storage::now_iso(),
                        });
                        app.error = Some(format!(
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
                Failure(String),
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
                                outcome = Outcome::Failure(err.to_string());
                            }
                        }
                        Err(err) => {
                            outcome = Outcome::Failure(err.to_string());
                        }
                    }
                }
                Err(err) => {
                    outcome = Outcome::Failure(err.to_string());
                }
            }

            let job_failure = match outcome {
                Outcome::Success => None,
                Outcome::Failure(message) => Some(message),
            };

            // Record a failure in local history; keep that write's own
            // error, if any, to append to the surfaced message.
            let history_failure = match job_failure.as_ref() {
                Some(failure) => {
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

            match finished_job(job_failure.as_deref(), history_failure.as_deref()) {
                FinishedJob::Failed { message } => {
                    this.update(cx, |app, cx| {
                        app.error = Some(message);
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
        // outcome; only a health probe may.
        let job = finished_job(
            Some("Local storage failed: No space left on device (os error 28)"),
            None,
        );
        match job {
            FinishedJob::Failed { message } => assert_eq!(
                message,
                "Local storage failed: No space left on device (os error 28)"
            ),
            FinishedJob::Saved => panic!("a failed job must surface its error"),
        }
    }

    #[test]
    fn a_failed_history_update_is_appended_to_the_surfaced_error() {
        let job = finished_job(Some("request failed"), Some("history is read-only"));
        match job {
            FinishedJob::Failed { message } => assert_eq!(
                message,
                "request failed Local history update also failed: history is read-only"
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
}
