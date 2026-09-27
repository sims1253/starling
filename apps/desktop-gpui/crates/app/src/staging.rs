//! The live staging panel (#297): the take's transcript as an editable
//! draft while it is being spoken, and until the user is done with it.
//!
//! One [`Staging`] per staged take. While recording, its draft is fed by
//! the `/stream` partials through [`LiveSegmenter`] (stable words become
//! final attempts, the tail stays live) and by the user's edits from the
//! [`StagingEditor`]; an edit closes the live segment, so later speech
//! lands after it and never overwrites it. Once the take is saved and its
//! transcript is in, the draft is rebased onto the take: the stored
//! transcript becomes its one raw attempt, the edited text becomes the
//! take's head (what Copy and Export use), and the draft moves into
//! `app.drafts` where processing already looks for it. From then on,
//! edits are written to the take's processing document shortly after the
//! user pauses, and processing results arrive as proposals next to the
//! text, never in it.
//!
//! Stopping and starting another take right away is safe: a staging whose
//! transcript has not landed yet moves to the background and rebases on
//! its own. Edits made while recording live in memory until the rebase;
//! the audio itself is journaled as always.

use std::ops::Range;
use std::time::Duration;

use gpui::{AppContext, ClipboardItem, Context, Entity, Subscription};
use starling_processing::contract::ProcessingDelivery;
use starling_processing::live::LiveSegmenter;
use starling_processing::staging::{Attempt, Draft, Outcome, RegionKind};

use crate::app::StarlingApp;
use crate::editor::{EditorEvent, StagingEditor, TextEdit};
use crate::live_stream::Partial;
use crate::processing::{ProcessingState, TakeProcessing, draft_from_doc};
use crate::store::ProcessingDoc;

/// Edits are written this long after the last keystroke.
const PERSIST_DELAY: Duration = Duration::from_millis(800);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum StagingPhase {
    /// The microphone is live; partials arrive.
    Recording,
    /// Stopped; waiting for the take to be saved and transcribed.
    Finishing,
    /// Rebased onto its saved take; edits persist.
    Ready,
    /// The take could not be saved or transcribed; the text is only here.
    Failed,
}

pub(crate) struct Staging {
    pub token: u64,
    pub editor: Entity<StagingEditor>,
    pub phase: StagingPhase,
    /// The draft until the rebase; afterwards it is `app.drafts[take_id]`.
    live: Option<Draft>,
    segmenter: LiveSegmenter,
    pub take_id: Option<String>,
    pub notice: Option<String>,
    pub copied: bool,
    persist_generation: u64,
    persisted_revision: u64,
    _subscription: Subscription,
}

/// Region kinds over the draft text, in code points, for the editor.
fn region_kinds(draft: &Draft) -> Vec<(Range<usize>, RegionKind)> {
    draft
        .snapshot()
        .regions
        .iter()
        .map(|region| (region.span[0]..region.span[1], region.kind))
        .collect()
}

/// `text` with `edit` applied (code points), for deciding how to apply it.
fn edited_text(text: &str, edit: &TextEdit) -> String {
    let byte = |at: usize| {
        text.char_indices()
            .nth(at)
            .map_or(text.len(), |(index, _)| index)
    };
    let (start, end) = (byte(edit.start), byte(edit.end.max(edit.start)));
    format!("{}{}{}", &text[..start], edit.text, &text[end..])
}

/// Applies one editor edit to the draft. An edit that restores the raw
/// transcript exactly (undoing an accepted proposal) goes back to raw, so
/// the text is raw again rather than a user copy of it.
fn apply_edit(draft: &mut Draft, edit: &TextEdit) {
    let result = edited_text(&draft.text(), edit);
    let live_tail = draft
        .snapshot()
        .regions
        .iter()
        .any(|region| region.kind == RegionKind::Partial);
    if !live_tail && !result.is_empty() && result == draft.raw_text() {
        draft.revert_raw();
        return;
    }
    if edit.end > edit.start {
        draft.delete(edit.start, edit.end);
    }
    if !edit.text.is_empty() {
        draft.insert(edit.start, &edit.text);
    }
}

/// Whether the user changed anything the recognition did not produce.
fn user_edited(draft: &Draft) -> bool {
    let snapshot = draft.snapshot();
    !snapshot.pinned_segments.is_empty()
        || snapshot
            .regions
            .iter()
            .any(|region| region.kind != RegionKind::Raw)
        || snapshot.text != draft.raw_text()
}

impl StarlingApp {
    /// Whether the active mode stages its takes (every built-in mode
    /// does; a direct mode keeps the read-only live line).
    pub(crate) fn staged_mode(&self) -> bool {
        self.active_mode().processing_delivery == ProcessingDelivery::Staged
    }

    fn staging_mut(&mut self, token: u64) -> Option<&mut Staging> {
        self.staging
            .iter_mut()
            .chain(self.background_stagings.iter_mut())
            .find(|staging| staging.token == token)
    }

    fn staging_token_for(&self, id: &str) -> Option<u64> {
        self.staging
            .iter()
            .chain(self.background_stagings.iter())
            .find(|staging| staging.take_id.as_deref() == Some(id))
            .map(|staging| staging.token)
    }

    /// The draft a staging edits: its own until the rebase, the take's
    /// afterwards.
    fn staging_draft(&mut self, token: u64) -> Option<&mut Draft> {
        let staging = self
            .staging
            .iter_mut()
            .chain(self.background_stagings.iter_mut())
            .find(|staging| staging.token == token)?;
        if staging.live.is_some() {
            return staging.live.as_mut();
        }
        let id = staging.take_id.clone()?;
        self.drafts.get_mut(&id)
    }

    /// The visible staging's draft, read-only (for the panel).
    pub(crate) fn visible_staging_draft(&self) -> Option<&Draft> {
        let staging = self.staging.as_ref()?;
        match &staging.live {
            Some(draft) => Some(draft),
            None => self.drafts.get(staging.take_id.as_ref()?),
        }
    }

    /// Whether the visible staging panel shows this take.
    pub(crate) fn staging_shows(&self, id: &str) -> bool {
        self.staging
            .as_ref()
            .is_some_and(|staging| staging.take_id.as_deref() == Some(id))
    }

    /// A new staged take starts with the recording.
    pub(crate) fn begin_staging(&mut self, cx: &mut Context<Self>) {
        self.retire_staging(cx);
        self.next_staging_token += 1;
        let token = self.next_staging_token;
        let editor = cx.new(|cx| StagingEditor::new("Start speaking, or type…", cx));
        let subscription = cx.subscribe(&editor, move |app, _editor, event: &EditorEvent, cx| {
            app.on_staging_event(token, event, cx);
        });
        self.staging = Some(Staging {
            token,
            editor,
            phase: StagingPhase::Recording,
            live: Some(Draft::new(format!("live-{token}"), format!("live-{token}"))),
            segmenter: LiveSegmenter::new(),
            take_id: None,
            notice: None,
            copied: false,
            persist_generation: 0,
            persisted_revision: 0,
            _subscription: subscription,
        });
        self.focus_staging_pending = true;
    }

    /// The visible staging leaves the screen: written now when it is
    /// ready, kept in the background while its transcript is pending.
    pub(crate) fn retire_staging(&mut self, cx: &mut Context<Self>) {
        let Some(staging) = self.staging.take() else {
            return;
        };
        match staging.phase {
            StagingPhase::Ready => {
                let token = staging.token;
                self.background_stagings.push(staging);
                self.persist_staging_now(token, cx);
                self.background_stagings
                    .retain(|staging| staging.token != token);
            }
            StagingPhase::Recording | StagingPhase::Finishing => {
                self.background_stagings.push(staging);
            }
            StagingPhase::Failed => {}
        }
        self.focus_root_pending = true;
        cx.notify();
    }

    /// Recording stopped: the take is being saved. Returns the staging's
    /// token for the save path.
    pub(crate) fn stop_staging(&mut self) -> Option<u64> {
        let staging = self.staging.as_mut()?;
        if staging.phase != StagingPhase::Recording {
            return None;
        }
        staging.phase = StagingPhase::Finishing;
        Some(staging.token)
    }

    /// The recording ended without a take to transcribe (a device that did
    /// not stop cleanly is saved as interrupted).
    pub(crate) fn staging_interrupted(&mut self, cx: &mut Context<Self>) {
        if let Some(staging) = self.staging.as_mut() {
            if staging.phase == StagingPhase::Recording {
                staging.phase = StagingPhase::Failed;
                staging.notice = Some(
                    "The recording was interrupted and saved to history untranscribed. The text \
                     here is not saved; copy it if you need it."
                        .to_string(),
                );
                cx.notify();
            }
        }
    }

    /// The live stream delivered a partial.
    pub(crate) fn staging_partial(&mut self, partial: Partial, cx: &mut Context<Self>) {
        let Some(staging) = self.staging.as_mut() else {
            return;
        };
        let Some(draft) = staging.live.as_mut() else {
            return;
        };
        staging
            .segmenter
            .partial(draft, &partial.text, partial.stable_words);
        let token = staging.token;
        self.sync_staging_editor(token, false, cx);
    }

    /// Pushes the draft's text and regions to the staging's editor.
    pub(crate) fn sync_staging_editor(
        &mut self,
        token: u64,
        undoable: bool,
        cx: &mut Context<Self>,
    ) {
        let Some(editor) = self
            .staging
            .iter()
            .chain(self.background_stagings.iter())
            .find(|staging| staging.token == token)
            .map(|staging| staging.editor.clone())
        else {
            return;
        };
        let Some(draft) = self.staging_draft(token) else {
            return;
        };
        let text = draft.text();
        let kinds = region_kinds(draft);
        editor.update(cx, |editor, cx| {
            editor.set_content(&text, &kinds, undoable, cx)
        });
        cx.notify();
    }

    fn on_staging_event(&mut self, token: u64, event: &EditorEvent, cx: &mut Context<Self>) {
        match event {
            EditorEvent::Edit(edit) => self.apply_staging_edit(token, edit, cx),
            EditorEvent::Submit => {
                if self
                    .staging
                    .as_ref()
                    .is_some_and(|staging| staging.token == token)
                {
                    self.finish_staging(cx);
                }
            }
            EditorEvent::Leave => {
                self.focus_root_pending = true;
                cx.notify();
            }
        }
    }

    fn apply_staging_edit(&mut self, token: u64, edit: &TextEdit, cx: &mut Context<Self>) {
        let Some(staging) = self.staging_mut(token) else {
            return;
        };
        staging.copied = false;
        if matches!(
            staging.phase,
            StagingPhase::Recording | StagingPhase::Finishing
        ) {
            staging.segmenter.cut();
        }
        let ready = staging.phase == StagingPhase::Ready;
        let Some(draft) = self.staging_draft(token) else {
            return;
        };
        apply_edit(draft, edit);
        // Whatever the draft made of the edit (a pinned partial becomes the
        // user's), the editor shows the draft.
        self.sync_staging_editor(token, false, cx);
        if ready {
            self.schedule_staging_persist(token, cx);
        }
    }

    /// The take a staging recorded was saved under `id`.
    pub(crate) fn bind_staging(&mut self, token: u64, id: &str) {
        if let Some(staging) = self.staging_mut(token) {
            staging.take_id = Some(id.to_string());
        }
    }

    /// The take could not be saved: its text lives only in the panel.
    pub(crate) fn staging_save_failed(&mut self, token: u64, cx: &mut Context<Self>) {
        if let Some(staging) = self.staging_mut(token) {
            staging.phase = StagingPhase::Failed;
            staging.notice = Some(
                "This take could not be saved, so the text here is not saved either. Copy it \
                 before you close the panel."
                    .to_string(),
            );
        }
        self.background_stagings
            .retain(|staging| staging.phase != StagingPhase::Failed);
        cx.notify();
    }

    /// The take's transcription failed: the live text stays in the panel.
    pub(crate) fn staging_transcription_failed(&mut self, id: &str, cx: &mut Context<Self>) {
        let Some(token) = self.staging_token_for(id) else {
            return;
        };
        if let Some(staging) = self.staging_mut(token) {
            if staging.phase != StagingPhase::Finishing {
                return;
            }
            staging.phase = StagingPhase::Failed;
            staging.notice = Some(
                "Transcription failed. The audio is in history (retry it there); the live text \
                 here is not saved, so copy it if you need it."
                    .to_string(),
            );
        }
        self.background_stagings
            .retain(|staging| staging.phase != StagingPhase::Failed);
        cx.notify();
    }

    /// The take was deleted: its staging goes with it.
    pub(crate) fn drop_staging_for(&mut self, id: &str) {
        if self.staging_shows(id) {
            self.staging = None;
            self.focus_root_pending = true;
        }
        self.background_stagings
            .retain(|staging| staging.take_id.as_deref() != Some(id));
    }

    /// The transcript of a staged take landed. Returns false when `id` is
    /// not a staged take waiting for it (the normal path runs instead).
    pub(crate) fn staged_transcript(&mut self, id: &str, cx: &mut Context<Self>) -> bool {
        let Some(token) = self.staging_token_for(id) else {
            return false;
        };
        let Some(store) = self.store.clone() else {
            return false;
        };
        let final_text = self
            .sessions
            .iter()
            .find(|session| session.id == id)
            .and_then(|session| session.transcript.as_ref())
            .map(|transcript| transcript.text.clone());
        let Some(staging) = self.staging_mut(token) else {
            return false;
        };
        // A failed transcription retried from history rebases the kept
        // text like a first transcript would.
        let waiting = staging.phase == StagingPhase::Finishing
            || (staging.phase == StagingPhase::Failed && staging.live.is_some());
        if !waiting {
            // A new transcript of a take whose staging already settled:
            // that draft was for the old text, and the normal path starts
            // over. The panel closes rather than edit a draft that is gone.
            self.drop_staging_for(id);
            return false;
        }
        staging.phase = StagingPhase::Finishing;
        staging.notice = None;
        let Some(draft) = staging.live.as_mut() else {
            return false;
        };
        let Some(final_text) = final_text else {
            return false;
        };
        if !staging.segmenter.finish(draft, &final_text) {
            staging.notice = Some(
                "The final transcript differs from the live one. Your text is kept; \
                 \"Back to raw\" shows the final transcript."
                    .to_string(),
            );
        }
        let edited = user_edited(draft).then(|| draft.text());
        let revision_read = draft.revision();
        self.sync_staging_editor(token, false, cx);
        // Stale state for an earlier transcript of this take does not
        // apply; the staged draft replaces it.
        self.processing.remove(id);
        self.drafts.remove(id);
        self.processing_loading.remove(id);

        let id = id.to_string();
        cx.spawn(async move |this, cx| {
            let prepared = {
                let id = id.clone();
                cx.background_spawn(async move {
                    let (attempt_id, raw) = store.latest_raw(&id)?.ok_or_else(|| {
                        starling_dictation::storage::StorageError::NotFound(id.clone())
                    })?;
                    let mut doc = store.start_processing_doc(&id, &attempt_id, &raw)?;
                    if let Some(text) = edited.filter(|text| *text != doc.head_text) {
                        let revision = doc.head_revision + 1;
                        store.commit_processing_head(
                            &id,
                            revision,
                            &text,
                            false,
                            &attempt_id,
                            None,
                        )?;
                        doc.head_revision = revision;
                        doc.head_text = text;
                        doc.head_is_raw = false;
                    }
                    Ok::<ProcessingDoc, starling_dictation::storage::StorageError>(doc)
                })
                .await
            };
            this.update(cx, |app, cx| {
                app.finish_rebase(token, &id, revision_read, prepared, cx)
            })
            .ok();
        })
        .detach();
        true
    }

    fn finish_rebase(
        &mut self,
        token: u64,
        id: &str,
        revision_read: u64,
        prepared: Result<ProcessingDoc, starling_dictation::storage::StorageError>,
        cx: &mut Context<Self>,
    ) {
        let doc = match prepared {
            Ok(doc) => doc,
            Err(err) => {
                if let Some(staging) = self.staging_mut(token) {
                    staging.phase = StagingPhase::Failed;
                    staging.notice = Some(format!(
                        "Could not save your edits with the take ({err}). The text here is not \
                         saved; copy it if you need it."
                    ));
                }
                self.background_stagings
                    .retain(|staging| staging.phase != StagingPhase::Failed);
                cx.notify();
                return;
            }
        };
        let Some(staging) = self.staging_mut(token) else {
            return;
        };
        let Some(live) = staging.live.take() else {
            return;
        };
        staging.phase = StagingPhase::Ready;
        staging.persisted_revision = doc.head_revision;
        let mut draft = draft_from_doc(id, &doc);
        // The raw attempt is the stored transcript itself.
        debug_assert_eq!(
            draft
                .attempts()
                .first()
                .map(|attempt: &Attempt| attempt.text.as_str()),
            Some(doc.raw_text.as_str())
        );
        // Edits made while the take was being written carry over.
        let typed = live.text();
        let moved = live.revision() != revision_read && typed != draft.text();
        if moved {
            let len = draft.text().chars().count();
            draft.delete(0, len);
            draft.insert(0, &typed);
        }
        self.drafts.insert(id.to_string(), draft);
        self.processing
            .insert(id.to_string(), TakeProcessing::from_doc(&doc));
        self.sync_staging_editor(token, false, cx);
        if moved {
            self.schedule_staging_persist(token, cx);
        }
        let id = id.to_string();
        if self.mode_processes() {
            self.process_take(id, cx);
        } else {
            self.stop_instants.remove(&id);
        }
        // A background staging is done once it is written.
        if !moved {
            self.background_stagings
                .retain(|staging| staging.token != token);
        }
        cx.notify();
    }

    fn schedule_staging_persist(&mut self, token: u64, cx: &mut Context<Self>) {
        let Some(staging) = self.staging_mut(token) else {
            return;
        };
        staging.persist_generation += 1;
        let generation = staging.persist_generation;
        cx.spawn(async move |this, cx| {
            cx.background_executor().timer(PERSIST_DELAY).await;
            this.update(cx, |app, cx| {
                let current = app
                    .staging_mut(token)
                    .is_some_and(|staging| staging.persist_generation == generation);
                if current {
                    app.persist_staging_now(token, cx);
                    app.background_stagings
                        .retain(|staging| staging.token != token);
                }
            })
            .ok();
        })
        .detach();
    }

    /// Writes the staged draft as the take's head, if it moved since the
    /// last write.
    pub(crate) fn persist_staging_now(&mut self, token: u64, cx: &mut Context<Self>) {
        let Some(store) = self.store.clone() else {
            return;
        };
        let Some(staging) = self.staging_mut(token) else {
            return;
        };
        if staging.phase != StagingPhase::Ready {
            return;
        }
        let persisted = staging.persisted_revision;
        let Some(id) = staging.take_id.clone() else {
            return;
        };
        let Some(draft) = self.drafts.get(&id) else {
            return;
        };
        if draft.revision() == persisted {
            return;
        }
        let revision = draft.revision();
        let text = draft.text();
        let is_raw = text == draft.raw_text();
        let attempt_id = draft
            .attempts()
            .first()
            .map(|attempt| attempt.attempt_id.clone())
            .unwrap_or_default();
        if let Some(staging) = self.staging_mut(token) {
            staging.persisted_revision = revision;
        }
        if let Some(take) = self.processing.get_mut(&id) {
            take.processed_head = (!is_raw).then(|| text.clone());
        }
        self.persist_head(store, id, revision, text, is_raw, attempt_id, None, cx);
    }

    /// "Done" (or Secondary+Enter): the draft is written and the panel
    /// closes. A take still being transcribed finishes in the background.
    pub(crate) fn finish_staging(&mut self, cx: &mut Context<Self>) {
        if self
            .staging
            .as_ref()
            .is_some_and(|staging| staging.phase == StagingPhase::Recording)
        {
            return;
        }
        self.retire_staging(cx);
    }

    /// Copies the draft's text; the delivery is recorded once the take is
    /// final.
    pub(crate) fn copy_staging(&mut self, cx: &mut Context<Self>) {
        let Some(token) = self.staging.as_ref().map(|staging| staging.token) else {
            return;
        };
        let Some(draft) = self.staging_draft(token) else {
            return;
        };
        let text = draft.text();
        let delivery = format!("copy-{}", draft.revision());
        // A live tail refuses the record (the text is not final yet); the
        // copy happens either way.
        let _: Outcome = draft.deliver(&delivery, "clipboard");
        cx.write_to_clipboard(ClipboardItem::new_string(text));
        if let Some(staging) = self.staging.as_mut() {
            staging.copied = true;
        }
        cx.notify();
    }

    /// Runs the active mode on the staged take (after its transcript).
    pub(crate) fn process_staging(&mut self, cx: &mut Context<Self>) {
        let Some(staging) = self.staging.as_ref() else {
            return;
        };
        if staging.phase != StagingPhase::Ready {
            return;
        }
        let (token, id) = (staging.token, staging.take_id.clone());
        // The job reads the stored head; write pending edits first.
        self.persist_staging_now(token, cx);
        if let Some(id) = id {
            self.process_take(id, cx);
        }
    }

    /// "Use processed" in the panel: the proposal replaces the text, as
    /// one undo step.
    pub(crate) fn accept_staging(&mut self, force: bool, cx: &mut Context<Self>) {
        let Some((token, id)) = self.ready_staging() else {
            return;
        };
        self.accept_processed(&id, force, cx);
        self.after_head_change(token, &id, cx);
    }

    /// "Back to raw" in the panel, as one undo step.
    pub(crate) fn revert_staging(&mut self, cx: &mut Context<Self>) {
        let Some((token, id)) = self.ready_staging() else {
            return;
        };
        self.revert_to_raw(&id, cx);
        self.after_head_change(token, &id, cx);
    }

    fn ready_staging(&self) -> Option<(u64, String)> {
        let staging = self.staging.as_ref()?;
        (staging.phase == StagingPhase::Ready)
            .then(|| staging.take_id.clone().map(|id| (staging.token, id)))
            .flatten()
    }

    /// Accept and revert wrote the head themselves; the panel follows.
    fn after_head_change(&mut self, token: u64, id: &str, cx: &mut Context<Self>) {
        let revision = self.drafts.get(id).map(Draft::revision);
        if let (Some(staging), Some(revision)) = (self.staging_mut(token), revision) {
            staging.persisted_revision = revision;
            staging.copied = false;
        }
        self.sync_staging_editor(token, true, cx);
    }

    /// The processing state the panel shows, with "current" judged
    /// against the draft as it is now (an edit makes a proposal stale).
    pub(crate) fn staging_processing(&self) -> Option<(String, ProcessingState)> {
        let staging = self.staging.as_ref()?;
        let id = staging.take_id.as_ref()?;
        let take = self.processing.get(id)?;
        let state = match &take.state {
            ProcessingState::Proposal { row, .. } => ProcessingState::Proposal {
                row: row.clone(),
                current: self.drafts.get(id).is_some_and(|draft| {
                    draft.proposal_status(&row.request_id)
                        == Some(starling_processing::staging::ProposalStatus::Current)
                }),
            },
            state => state.clone(),
        };
        Some((take.label.clone(), state))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn draft(raw: &str) -> Draft {
        let mut draft = Draft::new("d", "c");
        draft.final_attempt(0, "a", raw);
        draft
    }

    #[test]
    fn an_edit_applies_in_code_points() {
        let mut draft = draft("héllo wörld");
        apply_edit(
            &mut draft,
            &TextEdit {
                start: 6,
                end: 11,
                text: "there".into(),
            },
        );
        assert_eq!(draft.text(), "héllo there");
        assert_eq!(draft.raw_text(), "héllo wörld");
        assert!(user_edited(&draft));
    }

    #[test]
    fn undoing_an_accept_back_to_the_raw_text_is_raw_again() {
        let mut draft = draft("um the thing");
        draft.request_transform("r", None);
        draft.result(
            "r",
            starling_processing::staging::ResultKind::Completed,
            Some("The thing."),
        );
        draft.accept("r", false);
        assert_eq!(draft.text(), "The thing.");
        // The editor's undo of the accept: the whole text back to raw.
        apply_edit(
            &mut draft,
            &TextEdit {
                start: 0,
                end: 10,
                text: "um the thing".into(),
            },
        );
        assert_eq!(draft.text(), "um the thing");
        assert!(
            !user_edited(&draft),
            "the text is raw again, not a user copy"
        );
    }

    #[test]
    fn an_untouched_live_take_is_not_edited() {
        let mut draft = Draft::new("d", "c");
        let mut live = LiveSegmenter::new();
        live.partial(&mut draft, "one two three", 2);
        live.finish(&mut draft, "one two three four");
        assert!(!user_edited(&draft));
    }
}
