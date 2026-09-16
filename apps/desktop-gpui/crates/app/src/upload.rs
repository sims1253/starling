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

use crate::app::{Connection, StarlingApp, UnsavedWav, client_protocol};

impl StarlingApp {
    pub fn toggle_recording(&mut self, cx: &mut Context<Self>) {
        self.error = None;
        if let Some(handle) = self.recorder.take() {
            let duration_ms = handle.elapsed().as_secs_f64() * 1000.0;
            match handle.stop() {
                Ok(pcm) => {
                    self.levels = vec![0.06; 52];
                    let clipped = pcm
                        .samples
                        .iter()
                        .filter(|sample| sample.abs() >= 0.999)
                        .count();
                    let ratio = clipped as f64 / pcm.samples.len().max(1) as f64;
                    if ratio > 0.02 {
                        self.capture_warning = Some(format!(
                            "That recording was heavily clipped ({ratio:.0}% of samples at full \
                             scale). The microphone input level is too high — lower it in your \
                             sound settings and record again for a cleaner take."
                        ));
                    }
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

            if let Outcome::Failure(message) = outcome {
                let store = store_for_job.clone();
                let id_for_failure = id.clone();
                let failure_message = message.clone();
                let stored = cx
                    .background_spawn(async move {
                        if store.get(&id_for_failure)?.is_some() {
                            store.save_failure(&id_for_failure, failure_message)?;
                        }
                        Ok::<(), storage::StorageError>(())
                    })
                    .await;
                let mut full = message;
                if let Err(storage_err) = stored {
                    full = format!("{full} Local history update also failed: {storage_err}");
                }
                this.update(cx, |app, cx| {
                    app.connection = Connection::Offline;
                    app.error = Some(full);
                    cx.notify();
                })
                .ok();
            } else {
                this.update(cx, |app, cx| {
                    app.connection = Connection::Ready;
                    cx.notify();
                })
                .ok();
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
