//! History audio upkeep (#342): settled takes are compressed to lossless
//! FLAC, the user's retention limits (off by default) are applied, and
//! the retention sweep removes deleted takes' audio and superseded
//! recorder journals for good, in the background — after startup
//! recovery, then on a slow timer, and right after the settings are
//! saved. Never while a take records: a pass does not start then, and a
//! running one stops at its next step.

use std::borrow::Borrow;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use gpui::{AppContext, Context};
use starling_dictation::settings::StorageSettings;
use starling_dictation::store_v2::RetentionPolicy;

use crate::app::StarlingApp;

/// How often upkeep runs while the app is open.
const UPKEEP_INTERVAL: Duration = Duration::from_secs(15 * 60);

/// The storage settings and the state of the upkeep pass.
#[derive(Default)]
pub(crate) struct AudioUpkeep {
    /// The committed settings (what the pass applies).
    pub(crate) settings: StorageSettings,
    /// The saved settings as a running pass reads them: right before it
    /// removes anything, so a limit lifted mid-pass is not applied. Only
    /// a written settings file updates them.
    live: Arc<Mutex<StorageSettings>>,
    /// The save sequence `live` was last published from.
    published: u64,
    /// A take is recording: a running pass stops.
    recording: Arc<AtomicBool>,
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
            published: 0,
            recording: Arc::new(AtomicBool::new(false)),
            draft: settings,
            last_report: None,
            running: false,
            rerun: false,
        }
    }

    pub(crate) fn set_recording(&self, recording: bool) {
        self.recording.store(recording, Ordering::Release);
    }

    /// Publish the storage settings of the settings document written as
    /// save `sequence`; `true` when the policy a pass applies changed.
    /// An older document that lands after a newer one changes nothing.
    fn publish_saved(&mut self, sequence: u64, storage: StorageSettings) -> bool {
        if sequence <= self.published {
            return false;
        }
        self.published = sequence;
        let mut live = self.live.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let changed = *live != storage;
        *live = storage;
        changed
    }
}

/// The live policy, holding the lock on the settings it was read from. A
/// pass keeps it from its last check until that removal commits, so a
/// save that lifts a limit waits for at most that one removal, and the
/// pass stops before the next.
struct LivePolicy<'a> {
    _settings: MutexGuard<'a, StorageSettings>,
    policy: RetentionPolicy,
}

impl<'a> LivePolicy<'a> {
    fn lock(live: &'a Mutex<StorageSettings>) -> Self {
        let settings = live.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let policy = settings.retention_policy();
        Self {
            _settings: settings,
            policy,
        }
    }
}

impl Borrow<RetentionPolicy> for LivePolicy<'_> {
    fn borrow(&self) -> &RetentionPolicy {
        &self.policy
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
        let recording = Arc::clone(&self.audio_upkeep.recording);
        cx.spawn(async move |this, cx| {
            let outcome = {
                let store = store.clone();
                cx.background_spawn(async move {
                    store.audio_upkeep(
                        || LivePolicy::lock(&live),
                        || recording.load(Ordering::Acquire),
                    )
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
            if let Ok(report) = &outcome {
                for file in &report.sweep.swept {
                    eprintln!("Swept {} {} ({} bytes)", file.kind, file.id, file.bytes);
                }
                for (name, reason) in &report.sweep.retained {
                    eprintln!("Kept {name} at the retention sweep: {reason}");
                }
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

    /// Commit the dialog's storage draft into the settings the next save
    /// writes. A pass applies them only once that save succeeded
    /// ([`Self::storage_settings_saved`]).
    pub(crate) fn commit_storage_draft(&mut self) {
        self.audio_upkeep.settings = self.audio_upkeep.draft;
    }

    /// The settings document written as save `sequence` holds `storage`:
    /// a changed policy applies now.
    pub(crate) fn storage_settings_saved(
        &mut self,
        sequence: u64,
        storage: StorageSettings,
        cx: &mut Context<Self>,
    ) {
        if !self.audio_upkeep.publish_saved(sequence, storage) {
            return;
        }
        if self.audio_upkeep.running {
            self.audio_upkeep.rerun = true;
        } else {
            self.run_audio_upkeep(cx);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_a_newer_saved_document_publishes_its_policy() {
        let mut upkeep = AudioUpkeep::new(StorageSettings::default());
        let mut limited = StorageSettings::default();
        limited.include_referenced = true;
        // A draft committed but never saved changes nothing a pass reads.
        upkeep.settings = limited;
        assert_eq!(*upkeep.live.lock().unwrap(), StorageSettings::default());
        assert!(upkeep.publish_saved(2, limited));
        assert_eq!(*upkeep.live.lock().unwrap(), limited);
        // An older save that lands later does not undo the newer one.
        assert!(!upkeep.publish_saved(1, StorageSettings::default()));
        assert_eq!(*upkeep.live.lock().unwrap(), limited);
        // The same settings saved again need no new pass.
        assert!(!upkeep.publish_saved(3, limited));
    }
}
