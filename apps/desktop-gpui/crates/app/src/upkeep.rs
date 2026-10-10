//! History audio upkeep (#342): settled takes are compressed to lossless
//! FLAC and the user's retention limits (off by default) are applied, in
//! the background — after startup recovery, then on a slow timer, and
//! right after the settings are saved. Never while a take records: the
//! pass waits for the next turn instead.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use gpui::{AppContext, Context};
use starling_dictation::settings::StorageSettings;

use crate::app::StarlingApp;

/// How often upkeep runs while the app is open.
const UPKEEP_INTERVAL: Duration = Duration::from_secs(15 * 60);

/// The storage settings and the state of the upkeep pass.
#[derive(Default)]
pub(crate) struct AudioUpkeep {
    /// The committed settings (what the pass applies).
    pub(crate) settings: StorageSettings,
    /// The committed settings as a running pass reads them: right before
    /// it removes anything, so a limit lifted mid-pass is not applied.
    live: Arc<Mutex<StorageSettings>>,
    /// The settings dialog's draft, committed on save.
    pub(crate) draft: StorageSettings,
    /// What the latest pass did, for the settings dialog.
    pub(crate) last_report: Option<String>,
    running: bool,
    /// The settings changed while a pass ran: run again when it ends.
    rerun: bool,
}

impl AudioUpkeep {
    pub(crate) fn new(settings: StorageSettings) -> Self {
        Self {
            settings,
            live: Arc::new(Mutex::new(settings)),
            draft: settings,
            last_report: None,
            running: false,
            rerun: false,
        }
    }
}

impl StarlingApp {
    /// Start the periodic pass; the first one runs now.
    pub(crate) fn start_audio_upkeep(&mut self, cx: &mut Context<Self>) {
        cx.spawn(async move |this, cx| {
            loop {
                if this.update(cx, |app, cx| app.run_audio_upkeep(cx)).is_err() {
                    return;
                }
                cx.background_executor().timer(UPKEEP_INTERVAL).await;
            }
        })
        .detach();
    }

    /// One upkeep pass in the background, unless one is running or a take
    /// is recording.
    pub(crate) fn run_audio_upkeep(&mut self, cx: &mut Context<Self>) {
        let Some(store) = self.store.clone() else {
            return;
        };
        if self.audio_upkeep.running || self.recorder.is_some() {
            return;
        }
        self.audio_upkeep.running = true;
        let live = Arc::clone(&self.audio_upkeep.live);
        cx.spawn(async move |this, cx| {
            let outcome = {
                let store = store.clone();
                cx.background_spawn(async move {
                    store.audio_upkeep(|| {
                        live.lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner())
                            .retention_policy()
                    })
                })
                .await
            };
            let retired = matches!(&outcome, Ok(report) if !report.retention.retired.is_empty());
            for (id, reason) in outcome
                .as_ref()
                .map(|report| report.failures.as_slice())
                .unwrap_or_default()
            {
                eprintln!("Could not compress the audio of {id}: {reason}");
            }
            this.update(cx, |app, cx| {
                app.audio_upkeep.running = false;
                if std::mem::take(&mut app.audio_upkeep.rerun) {
                    app.run_audio_upkeep(cx);
                }
                match &outcome {
                    Ok(report) => {
                        if let Some(summary) = report.summary() {
                            app.audio_upkeep.last_report = Some(summary);
                        }
                    }
                    Err(err) => {
                        app.audio_upkeep.last_report =
                            Some(format!("History audio upkeep failed: {err}"));
                    }
                }
                cx.notify();
            })
            .ok();
            if retired {
                crate::upload::refresh_sessions(&this, &store, cx).await;
            }
        })
        .detach();
    }

    /// Move the selected take into the archival class or back (#342).
    pub(crate) fn toggle_archival_selected(&mut self, cx: &mut Context<Self>) {
        let (Some(store), Some(session)) = (self.store.clone(), self.selected()) else {
            return;
        };
        let id = session.id.clone();
        let archival = !session.archival;
        cx.spawn(async move |this, cx| {
            let result = {
                let store = store.clone();
                let id = id.clone();
                cx.background_spawn(async move { store.set_archival(&id, archival) })
                    .await
            };
            if let Err(err) = result {
                this.update(cx, |app, cx| {
                    app.error = Some(err.to_string());
                    cx.notify();
                })
                .ok();
                return;
            }
            crate::upload::refresh_sessions(&this, &store, cx).await;
        })
        .detach();
    }

    /// Commit the dialog's storage draft; a changed policy applies now.
    pub(crate) fn commit_storage_draft(&mut self, cx: &mut Context<Self>) {
        let changed = self.audio_upkeep.draft != self.audio_upkeep.settings;
        self.audio_upkeep.settings = self.audio_upkeep.draft;
        *self
            .audio_upkeep
            .live
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = self.audio_upkeep.settings;
        if changed {
            if self.audio_upkeep.running {
                self.audio_upkeep.rerun = true;
            } else {
                self.run_audio_upkeep(cx);
            }
        }
    }
}
