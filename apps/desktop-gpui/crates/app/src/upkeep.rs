//! History audio upkeep (#342), app side. The pass itself — settled takes
//! compressed to lossless FLAC, the user's retention limits (off by
//! default) applied, the retention sweep — runs in the runtime host on
//! its own schedule and waits while a take records (#220). The app keeps
//! the storage settings dialog, asks the host for a pass when a saved
//! settings document changes the policy, and shows the host's latest
//! report.

use gpui::{AppContext, Context};
use starling_dictation::settings::StorageSettings;

use crate::app::StarlingApp;

/// The storage settings and what the host's upkeep last reported.
#[derive(Default)]
pub(crate) struct AudioUpkeep {
    /// The committed settings (what the next save writes).
    pub(crate) settings: StorageSettings,
    /// The settings as last written: the host reads its policy from the
    /// file.
    saved: StorageSettings,
    /// The save sequence `saved` was last published from.
    published: u64,
    /// The settings dialog's draft, committed on save.
    pub(crate) draft: StorageSettings,
    /// What the host's latest pass did, for the settings dialog.
    pub(crate) last_report: Option<String>,
}

impl AudioUpkeep {
    pub(crate) fn new(settings: StorageSettings) -> Self {
        Self {
            settings,
            saved: settings,
            published: 0,
            draft: settings,
            last_report: None,
        }
    }

    /// Publish the storage settings of the settings document written as
    /// save `sequence`; `true` when the policy a pass applies changed.
    /// An older document that lands after a newer one changes nothing.
    fn publish_saved(&mut self, sequence: u64, storage: StorageSettings) -> bool {
        if sequence <= self.published {
            return false;
        }
        self.published = sequence;
        let changed = self.saved != storage;
        self.saved = storage;
        changed
    }
}

impl StarlingApp {
    /// Asks the host for its latest upkeep report (`run`: and for a pass
    /// now), shown in the storage settings.
    pub(crate) fn ask_upkeep(&mut self, run: bool, cx: &mut Context<Self>) {
        let Some(store) = self.store.clone() else {
            return;
        };
        cx.spawn(async move |this, cx| {
            let report = cx.background_spawn(async move { store.upkeep(run) }).await;
            this.update(cx, |app, cx| {
                match report {
                    Ok(Some(report)) => app.audio_upkeep.last_report = Some(report),
                    Ok(None) => {}
                    Err(err) if run => {
                        app.audio_upkeep.last_report =
                            Some(format!("Could not ask for history audio upkeep: {err}"));
                    }
                    Err(_) => {}
                }
                cx.notify();
            })
            .ok();
        })
        .detach();
    }

    /// A pass of the host's upkeep reported (`retired`: audio was
    /// removed, so the history list changed).
    pub(crate) fn upkeep_reported(&mut self, report: String, retired: bool, cx: &mut Context<Self>) {
        self.audio_upkeep.last_report = Some(report);
        if retired {
            self.refresh_history(cx);
        }
        cx.notify();
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
    /// writes. The host applies them once that save succeeded
    /// ([`Self::storage_settings_saved`]).
    pub(crate) fn commit_storage_draft(&mut self) {
        self.audio_upkeep.settings = self.audio_upkeep.draft;
    }

    /// The settings document written as save `sequence` holds `storage`:
    /// a changed policy applies now — the host reads it from the file.
    pub(crate) fn storage_settings_saved(
        &mut self,
        sequence: u64,
        storage: StorageSettings,
        cx: &mut Context<Self>,
    ) {
        if self.audio_upkeep.publish_saved(sequence, storage) {
            self.ask_upkeep(true, cx);
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
        assert_eq!(upkeep.saved, StorageSettings::default());
        assert!(upkeep.publish_saved(2, limited));
        assert_eq!(upkeep.saved, limited);
        // An older save that lands later does not undo the newer one.
        assert!(!upkeep.publish_saved(1, StorageSettings::default()));
        assert_eq!(upkeep.saved, limited);
        // The same settings saved again need no new pass.
        assert!(!upkeep.publish_saved(3, limited));
    }
}
