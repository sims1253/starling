//! Focus-safe delivery of finished takes (#221): the text goes into the
//! window that had focus when the take started, without Starling's window
//! ever coming forward.
//!
//! - **Capture at start.** `start_recording` captures the focused target
//!   before anything else can move focus; the capture travels with the
//!   take (`stop_recording` → the save → its store id).
//! - **Deliver once, when the text is final.** When the take's transcript
//!   lands, its text (the staged draft with the user's edits, or the raw
//!   transcript) is typed into the captured target. The backend
//!   revalidates the target immediately before typing and before every
//!   chunk; nothing is ever submitted (control characters are refused, so
//!   no Enter). A capture is consumed by its first delivery attempt: a
//!   retried transcription never types again.
//! - **Recovery.** Anything that keeps the text from landing (completely)
//!   leaves a notice with the text, the specific reason, Copy, and Paste
//!   last. Paste last types into a target captured anew, after the user
//!   moved focus away from Starling on purpose; it never re-targets the
//!   original window by itself. Dismissing the notice only hides it: the
//!   take, its transcript and its audio stay in history.
//! - **Unverifiable targets.** Where the backend cannot identify the
//!   target (Wayland), typing needs the user's opt-in, and a take during
//!   which Starling's own window gained focus is not typed at all.
//!
//! The clipboard is written only by an explicit Copy: delivery types, it
//! never pastes through the clipboard, so a user's clipboard is never
//! replaced behind their back.
//!
//! The seam to the host (#220): everything platform-facing goes through
//! the [`Inserter`]; moving delivery into the runtime host swaps
//! [`StarlingApp::spawn_insert`] for a `delivery.*` call with the same
//! captured `target_ref`.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use gpui::{AppContext, ClipboardItem, Context};
use starling_dictation::settings::InsertionSettings;
use starling_insertion::{InsertError, Inserter, TargetSnapshot};

use crate::app::StarlingApp;

/// How long an armed Paste last waits for the window to lose focus.
pub(crate) const PASTE_ARM_TIMEOUT: Duration = Duration::from_secs(15);
/// After Starling's window loses focus, the new focus settles this long
/// before it is captured (a window switcher may pass through first).
pub(crate) const PASTE_SETTLE: Duration = Duration::from_millis(350);

/// What a take captured when it started.
#[derive(Clone, Debug)]
pub(crate) struct Capture {
    pub target: Result<TargetSnapshot, InsertError>,
    pub at: Instant,
}

/// Why text did not land (completely).
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Failure {
    Insert(InsertError),
    /// The target cannot be verified here and the user has not opted in.
    UnverifiedOff,
    /// The target cannot be verified, and Starling's own window had focus
    /// since the capture: the original target cannot be assumed.
    FocusMovedThroughStarling,
}

/// What to do with a capture and a text.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Plan {
    /// Nothing to deliver, and nothing to report: the take was dictated
    /// into Starling itself, or it has no text.
    Skip,
    Insert(TargetSnapshot),
    Fail(Failure),
}

/// The delivery decision, before any platform call. `verified` is whether
/// the target's backend can verify it; `focused_since` whether Starling's
/// window had focus at any point since the capture.
pub(crate) fn plan(
    capture: &Capture,
    settings: InsertionSettings,
    verified: bool,
    focused_since: bool,
    text: &str,
) -> Plan {
    if text.trim().is_empty() {
        return Plan::Skip;
    }
    match &capture.target {
        Err(InsertError::TargetIsStarling) => Plan::Skip,
        Err(error) => Plan::Fail(Failure::Insert(error.clone())),
        Ok(_) if !verified && !settings.allow_unverified => Plan::Fail(Failure::UnverifiedOff),
        Ok(_) if !verified && focused_since => Plan::Fail(Failure::FocusMovedThroughStarling),
        Ok(target) => Plan::Insert(target.clone()),
    }
}

/// The failed-insertion notice: the take's text, why it did not land,
/// and the ways to get it where it belongs.
#[derive(Clone, Debug)]
pub(crate) struct Recovery {
    pub take_id: String,
    pub text: String,
    pub failure: Failure,
    /// Paste last is armed: the next focus away from Starling receives
    /// the text. The instant tells one arming from the next.
    pub armed: Option<Instant>,
    /// A Paste last insert is running.
    pub pasting: bool,
    pub copied: bool,
}

impl Recovery {
    fn new(take_id: &str, text: String, failure: Failure) -> Recovery {
        Recovery {
            take_id: take_id.to_string(),
            text,
            failure,
            armed: None,
            pasting: false,
            copied: false,
        }
    }

    pub(crate) fn title(&self) -> &'static str {
        match &self.failure {
            Failure::Insert(InsertError::PartialDelivery { .. }) => "Inserted only in part",
            Failure::Insert(InsertError::TargetChanged { .. })
            | Failure::FocusMovedThroughStarling => "Not inserted: focus moved",
            Failure::Insert(InsertError::TargetGone) => "Not inserted: the window closed",
            Failure::Insert(InsertError::ModifiersHeld { .. }) => "Not inserted: keys were held",
            Failure::Insert(InsertError::Unavailable { .. }) => "Insertion unavailable",
            Failure::UnverifiedOff => "Not inserted: the target cannot be checked",
            Failure::Insert(_) => "Not inserted",
        }
    }

    /// The specific explanation the notice shows.
    pub(crate) fn explanation(&self) -> String {
        match &self.failure {
            Failure::Insert(InsertError::TargetChanged { .. }) => {
                "The window you dictated into no longer had focus when the text was ready, so \
                 nothing was typed."
                    .to_string()
            }
            Failure::Insert(InsertError::TargetGone) => {
                "The window you dictated into was closed, so nothing was typed.".to_string()
            }
            Failure::Insert(InsertError::TargetIsStarling) => {
                "Starling's own window had focus, so nothing was typed.".to_string()
            }
            Failure::Insert(InsertError::PartialDelivery {
                delivered_chars,
                total_chars,
                cause,
            }) => format!(
                "Typing stopped part-way ({cause}). Up to {delivered_chars} of {total_chars} \
                 characters may already be in the window; check it before pasting again."
            ),
            Failure::Insert(error @ InsertError::MultilineUnsupported) => format!(
                "Nothing was typed: {}. Copy it and paste it yourself.",
                error.message()
            ),
            Failure::Insert(error) => format!("Nothing was typed: {}.", error.message()),
            Failure::UnverifiedOff => {
                "On Wayland, Starling cannot check which window receives typed text, and \
                 typing without that check is off. Turn it on in Settings, or copy the text."
                    .to_string()
            }
            Failure::FocusMovedThroughStarling => {
                "Starling's window had focus during the take, and on Wayland the original window \
                 cannot be checked, so nothing was typed."
                    .to_string()
            }
        }
    }

    /// Whether the fix is in Starling's settings: opting in to unverified
    /// typing, or switching to copy-only where insertion cannot work.
    pub(crate) fn offers_settings(&self) -> bool {
        matches!(
            self.failure,
            Failure::UnverifiedOff | Failure::Insert(InsertError::Unavailable { .. })
        )
    }
}

/// The app's delivery state.
pub(crate) struct DeliveryState {
    pub(crate) inserter: Arc<Inserter>,
    pub(crate) settings: InsertionSettings,
    /// The running take's capture.
    live: Option<Capture>,
    /// Saved takes waiting for their transcript, by store id.
    by_take: HashMap<String, Capture>,
    pub(crate) recovery: Option<Recovery>,
    /// Bumped whenever the recovery is replaced or dismissed, so a late
    /// Paste last result lands only on the notice it was started from.
    generation: u64,
    /// Bumped on every focus change of Starling's window, so only the
    /// settle timer of the latest focus loss fires Paste last.
    focus_changes: u64,
    /// Whether Starling's window has focus, for typing off the UI thread:
    /// an insert stops before its next key once it turns true (Wayland
    /// cannot tell that Starling's window took focus; Starling can).
    own_focus: Arc<AtomicBool>,
}

impl DeliveryState {
    pub(crate) fn new(inserter: Arc<Inserter>, settings: InsertionSettings) -> DeliveryState {
        DeliveryState {
            inserter,
            settings,
            live: None,
            by_take: HashMap::new(),
            recovery: None,
            generation: 0,
            focus_changes: 0,
            own_focus: Arc::new(AtomicBool::new(false)),
        }
    }

    fn replace_recovery(&mut self, recovery: Option<Recovery>) {
        self.generation += 1;
        self.recovery = recovery;
    }
}

impl StarlingApp {
    /// Whether Starling's window has focus now.
    fn starling_focused(&self) -> bool {
        self.window_focus.last().is_some_and(|(_, active)| *active)
    }

    /// Whether Starling's window had focus at any point since `at`.
    fn starling_focused_since(&self, at: Instant) -> bool {
        self.starling_focused()
            || self
                .window_focus
                .iter()
                .any(|(changed, active)| *active && *changed >= at)
    }

    fn capture_now(&self) -> Capture {
        let at = Instant::now();
        let target = if self.starling_focused() {
            // Dictating into Starling itself. On Wayland only the app can
            // tell; elsewhere the backend refuses Starling's pid too.
            Err(InsertError::TargetIsStarling)
        } else {
            self.delivery.inserter.capture()
        };
        Capture { target, at }
    }

    /// A take started: capture where its text goes, before anything can
    /// move focus. An armed Paste last is disarmed: the user is dictating
    /// again, not choosing a field.
    pub(crate) fn delivery_take_started(&mut self) {
        if let Some(recovery) = self.delivery.recovery.as_mut() {
            recovery.armed = None;
        }
        self.delivery.live = self
            .delivery
            .settings
            .auto_insert
            .then(|| self.capture_now());
    }

    /// The take stopped (or was cancelled): its capture leaves with it.
    pub(crate) fn delivery_take_stopped(&mut self) -> Option<Capture> {
        self.delivery.live.take()
    }

    /// The stopped take was saved as `id`.
    pub(crate) fn bind_delivery(&mut self, capture: Option<Capture>, id: &str) {
        if let Some(capture) = capture {
            self.delivery.by_take.insert(id.to_string(), capture);
        }
    }

    /// The take will not get a transcript from this job: nothing is
    /// delivered for it later (a retry is a new, explicit job).
    pub(crate) fn forget_delivery(&mut self, id: &str) {
        self.delivery.by_take.remove(id);
    }

    /// The take's transcript landed: type its text into the captured
    /// target, once.
    pub(crate) fn deliver_finished_take(&mut self, id: &str, cx: &mut Context<Self>) {
        let Some(capture) = self.delivery.by_take.remove(id) else {
            return;
        };
        if !self.delivery.settings.auto_insert {
            return;
        }
        let Some(text) = self.staged_text_for(id).or_else(|| self.head_text(id)) else {
            return;
        };
        let verified = capture
            .target
            .as_ref()
            .is_ok_and(|target| self.delivery.inserter.verifies(target));
        let focused_since = self.starling_focused_since(capture.at);
        match plan(
            &capture,
            self.delivery.settings,
            verified,
            focused_since,
            &text,
        ) {
            Plan::Skip => {}
            Plan::Fail(failure) => {
                self.delivery
                    .replace_recovery(Some(Recovery::new(id, text, failure)));
                cx.notify();
            }
            Plan::Insert(target) => self.spawn_insert(id.to_string(), target, text, None, cx),
        }
    }

    /// Types `text` into `target` off the UI thread. `paste` is the
    /// recovery generation a Paste last runs for; `None` for the take's
    /// own delivery.
    fn spawn_insert(
        &mut self,
        id: String,
        target: TargetSnapshot,
        text: String,
        paste: Option<u64>,
        cx: &mut Context<Self>,
    ) {
        let inserter = self.delivery.inserter.clone();
        let own_focus = self.delivery.own_focus.clone();
        let typed_text = text.clone();
        cx.spawn(async move |this, cx| {
            let result = cx
                .background_spawn(async move {
                    let stop = || {
                        own_focus
                            .load(Ordering::SeqCst)
                            .then_some(InsertError::TargetIsStarling)
                    };
                    inserter.insert(&target, &typed_text, &stop)
                })
                .await;
            this.update(cx, |app, cx| {
                app.insert_finished(id, text, paste, result, cx)
            })
            .ok();
        })
        .detach();
    }

    fn insert_finished(
        &mut self,
        id: String,
        text: String,
        paste: Option<u64>,
        result: Result<starling_insertion::InsertReceipt, InsertError>,
        cx: &mut Context<Self>,
    ) {
        match paste {
            None => {
                if let Err(error) = result {
                    self.delivery.replace_recovery(Some(Recovery::new(
                        &id,
                        text,
                        Failure::Insert(error),
                    )));
                }
            }
            Some(generation) => {
                if generation != self.delivery.generation {
                    // The notice was dismissed or replaced meanwhile.
                    return;
                }
                match result {
                    Ok(_) => self.delivery.replace_recovery(None),
                    Err(error) => {
                        if let Some(recovery) = self.delivery.recovery.as_mut() {
                            recovery.pasting = false;
                            recovery.failure = Failure::Insert(error);
                        }
                    }
                }
            }
        }
        cx.notify();
    }

    /// "Copy" in the notice: an explicit clipboard write, never undone.
    pub(crate) fn copy_recovery(&mut self, cx: &mut Context<Self>) {
        let Some(recovery) = self.delivery.recovery.as_mut() else {
            return;
        };
        cx.write_to_clipboard(ClipboardItem::new_string(recovery.text.clone()));
        recovery.copied = true;
        cx.notify();
    }

    /// Hides the notice. The take stays in history with its transcript
    /// and audio.
    pub(crate) fn dismiss_recovery(&mut self, cx: &mut Context<Self>) {
        self.delivery.replace_recovery(None);
        cx.notify();
    }

    /// "Paste last": the next window the user moves focus to (within
    /// [`PASTE_ARM_TIMEOUT`]) receives the text, captured and checked
    /// anew. Pressing it again cancels.
    pub(crate) fn toggle_paste_last(&mut self, cx: &mut Context<Self>) {
        let focused = self.starling_focused();
        let Some(recovery) = self.delivery.recovery.as_mut() else {
            return;
        };
        if recovery.pasting {
            return;
        }
        if recovery.armed.take().is_some() {
            cx.notify();
            return;
        }
        let armed_at = Instant::now();
        recovery.armed = Some(armed_at);
        cx.spawn(async move |this, cx| {
            cx.background_executor().timer(PASTE_ARM_TIMEOUT).await;
            this.update(cx, |app, cx| {
                if let Some(recovery) = app.delivery.recovery.as_mut() {
                    if recovery.armed == Some(armed_at) {
                        recovery.armed = None;
                        cx.notify();
                    }
                }
            })
            .ok();
        })
        .detach();
        if !focused {
            // Armed from outside the window (a keyboard path, a test):
            // focus is already elsewhere.
            self.schedule_paste_last(armed_at, cx);
        }
        cx.notify();
    }

    /// Starling's window gained or lost focus.
    pub(crate) fn delivery_window_activation(&mut self, active: bool, cx: &mut Context<Self>) {
        self.delivery.focus_changes += 1;
        self.delivery.own_focus.store(active, Ordering::SeqCst);
        if active {
            return;
        }
        if let Some(armed_at) = self
            .delivery
            .recovery
            .as_ref()
            .and_then(|recovery| recovery.armed)
        {
            self.schedule_paste_last(armed_at, cx);
        }
    }

    fn schedule_paste_last(&mut self, armed_at: Instant, cx: &mut Context<Self>) {
        let focus_changes = self.delivery.focus_changes;
        cx.spawn(async move |this, cx| {
            cx.background_executor().timer(PASTE_SETTLE).await;
            this.update(cx, |app, cx| {
                // Focus moved again meanwhile: that move's own timer
                // decides, after its full settle.
                if app.delivery.focus_changes == focus_changes {
                    app.fire_paste_last(armed_at, cx);
                }
            })
            .ok();
        })
        .detach();
    }

    /// The armed Paste last fires: capture the field the user moved to,
    /// check it, and type there.
    fn fire_paste_last(&mut self, armed_at: Instant, cx: &mut Context<Self>) {
        let still_armed = self
            .delivery
            .recovery
            .as_ref()
            .is_some_and(|recovery| recovery.armed == Some(armed_at));
        // Back in Starling before the focus settled: stay armed.
        if !still_armed || self.starling_focused() {
            return;
        }
        let capture = self.capture_now();
        let generation = self.delivery.generation;
        let Some(recovery) = self.delivery.recovery.as_mut() else {
            return;
        };
        recovery.armed = None;
        let (id, text) = (recovery.take_id.clone(), recovery.text.clone());
        let verified = capture
            .target
            .as_ref()
            .is_ok_and(|target| self.delivery.inserter.verifies(target));
        // The capture is fresh: a focus change before it is the user's
        // choice, not a reason to refuse.
        match plan(&capture, self.delivery.settings, verified, false, &text) {
            Plan::Insert(target) => {
                if let Some(recovery) = self.delivery.recovery.as_mut() {
                    recovery.pasting = true;
                }
                self.spawn_insert(id, target, text, Some(generation), cx);
            }
            Plan::Skip => {
                if let Some(recovery) = self.delivery.recovery.as_mut() {
                    recovery.failure = Failure::Insert(InsertError::TargetIsStarling);
                }
            }
            Plan::Fail(failure) => {
                if let Some(recovery) = self.delivery.recovery.as_mut() {
                    recovery.failure = failure;
                }
            }
        }
        cx.notify();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use starling_insertion::InsertionBackend;
    use starling_insertion::testing::{FakeBackend, FakeTarget, InsertBehavior};

    fn on() -> InsertionSettings {
        InsertionSettings::default()
    }

    fn captured(target: Result<TargetSnapshot, InsertError>) -> Capture {
        Capture {
            target,
            at: Instant::now(),
        }
    }

    fn fake_session() -> (Arc<FakeBackend>, Arc<Inserter>) {
        let fake = Arc::new(FakeBackend::new());
        let inserter = Inserter::with_backends(vec![Box::new(fake.clone())]);
        (fake, Arc::new(inserter))
    }

    /// A test app with a fake backend and a take bound to `id` whose
    /// capture saw `focus`.
    fn app_with(
        cx: &mut gpui::TestAppContext,
        settings: InsertionSettings,
    ) -> (gpui::Entity<StarlingApp>, Arc<FakeBackend>) {
        let (fake, inserter) = fake_session();
        let app = cx.new(|cx| {
            let mut app = StarlingApp::for_test(None, cx);
            app.delivery = DeliveryState::new(inserter, settings);
            app
        });
        (app, fake)
    }

    /// One take through the app's hooks: started while `focus` had
    /// focus, saved as `id`, then (after `between`) its text is ready.
    fn take(
        app: &gpui::Entity<StarlingApp>,
        cx: &mut gpui::TestAppContext,
        id: &str,
        text: &str,
        between: impl FnOnce(&mut StarlingApp),
    ) {
        app.update(cx, |app, cx| {
            app.delivery_take_started();
            let capture = app.delivery_take_stopped();
            app.bind_delivery(capture, id);
            between(app);
            app.sessions.push(session(id, text));
            app.deliver_finished_take(id, cx);
        });
        cx.run_until_parked();
    }

    fn session(id: &str, text: &str) -> starling_dictation::storage::SessionSummary {
        use starling_dictation::storage::{SessionStatus, SessionSummary, TranscriptionResult};
        SessionSummary {
            id: id.to_string(),
            created_at: "2026-10-10T00:00:00.000Z".to_string(),
            updated_at: "2026-10-10T00:00:00.000Z".to_string(),
            status: SessionStatus::Transcribed,
            duration_ms: None,
            attempt_count: 1,
            transcript: Some(TranscriptionResult {
                text: text.to_string(),
                segments: Vec::new(),
                duration_seconds: None,
                request_id: None,
            }),
            last_error: None,
            model_label: None,
            journal_id: None,
            archival: false,
        }
    }

    fn failure(app: &gpui::Entity<StarlingApp>, cx: &mut gpui::TestAppContext) -> Option<Failure> {
        app.read_with(cx, |app, _| {
            app.delivery.recovery.as_ref().map(|r| r.failure.clone())
        })
    }

    #[test]
    fn the_plan_skips_starling_and_empty_text_and_gates_unverified_targets() {
        let fake = FakeBackend::new();
        fake.focus(FakeTarget::named("Editor", "notes.txt"));
        let target = fake.capture().unwrap();
        let ok = captured(Ok(target.clone()));
        assert_eq!(
            plan(&ok, on(), true, false, "hi"),
            Plan::Insert(target.clone())
        );
        assert_eq!(plan(&ok, on(), true, false, "  "), Plan::Skip);
        // A verifiable target is checked by the backend itself, so a
        // detour through Starling's window is not a reason to refuse.
        assert_eq!(
            plan(&ok, on(), true, true, "hi"),
            Plan::Insert(target.clone())
        );
        assert_eq!(
            plan(
                &captured(Err(InsertError::TargetIsStarling)),
                on(),
                true,
                false,
                "hi"
            ),
            Plan::Skip
        );
        let unavailable = InsertError::Unavailable {
            reason: "no display".into(),
        };
        assert_eq!(
            plan(&captured(Err(unavailable.clone())), on(), true, false, "hi"),
            Plan::Fail(Failure::Insert(unavailable))
        );

        assert_eq!(
            plan(&ok, on(), false, false, "hi"),
            Plan::Fail(Failure::UnverifiedOff)
        );
        let opted_in = InsertionSettings {
            allow_unverified: true,
            ..on()
        };
        assert_eq!(
            plan(&ok, opted_in, false, false, "hi"),
            Plan::Insert(target)
        );
        assert_eq!(
            plan(&ok, opted_in, false, true, "hi"),
            Plan::Fail(Failure::FocusMovedThroughStarling)
        );
    }

    #[gpui::test]
    fn a_take_types_its_text_into_the_target_it_started_in(cx: &mut gpui::TestAppContext) {
        let (app, fake) = app_with(cx, on());
        fake.focus(FakeTarget::named("Editor", "notes.txt"));
        let target = fake.capture().unwrap();
        take(&app, cx, "take-1", "Hello there.", |_| {});
        assert_eq!(
            fake.insertions(),
            vec![(target.target_ref, "Hello there.".to_string())]
        );
        assert_eq!(failure(&app, cx), None);

        // The capture is consumed: a retried transcript never types again.
        app.update(cx, |app, cx| app.deliver_finished_take("take-1", cx));
        cx.run_until_parked();
        assert_eq!(fake.insertions().len(), 1);
    }

    #[gpui::test]
    fn a_changed_target_keeps_the_text_for_recovery(cx: &mut gpui::TestAppContext) {
        let (app, fake) = app_with(cx, on());
        fake.focus(FakeTarget::named("Editor", "notes.txt"));
        take(&app, cx, "take-1", "Hello there.", |_| {
            fake.focus(FakeTarget::named("Browser", "A tab"));
        });
        assert!(fake.insertions().is_empty());
        assert!(matches!(
            failure(&app, cx),
            Some(Failure::Insert(InsertError::TargetChanged { .. }))
        ));
        app.read_with(cx, |app, _| {
            let recovery = app.delivery.recovery.as_ref().unwrap();
            assert_eq!(recovery.text, "Hello there.");
            assert_eq!(recovery.take_id, "take-1");
            assert_eq!(recovery.title(), "Not inserted: focus moved");
        });

        // A closed target names that instead.
        fake.focus(FakeTarget::named("Editor", "notes.txt"));
        take(&app, cx, "take-2", "Second.", |_| fake.destroy_target());
        assert_eq!(
            failure(&app, cx),
            Some(Failure::Insert(InsertError::TargetGone))
        );
    }

    #[gpui::test]
    fn partial_delivery_and_held_modifiers_are_explained(cx: &mut gpui::TestAppContext) {
        let (app, fake) = app_with(cx, on());
        fake.focus(FakeTarget::named("Editor", "notes.txt"));
        fake.set_insert_behavior(InsertBehavior::FailWith(InsertError::PartialDelivery {
            delivered_chars: 16,
            total_chars: 40,
            cause: Box::new(InsertError::TargetGone),
        }));
        take(&app, cx, "take-1", &"x".repeat(40), |_| {});
        app.read_with(cx, |app, _| {
            let recovery = app.delivery.recovery.as_ref().unwrap();
            assert_eq!(recovery.title(), "Inserted only in part");
            assert!(
                recovery.explanation().contains("Up to 16 of 40 characters"),
                "{}",
                recovery.explanation()
            );
        });

        fake.set_insert_behavior(InsertBehavior::FailWith(InsertError::ModifiersHeld {
            held: vec!["Control".into()],
        }));
        take(&app, cx, "take-2", "Hello.", |_| {});
        app.read_with(cx, |app, _| {
            let recovery = app.delivery.recovery.as_ref().unwrap();
            assert_eq!(recovery.take_id, "take-2");
            assert_eq!(recovery.title(), "Not inserted: keys were held");
            assert!(recovery.explanation().contains("Control"));
        });
    }

    #[gpui::test]
    fn typing_stops_when_starlings_window_takes_focus(cx: &mut gpui::TestAppContext) {
        let opted_in = InsertionSettings {
            allow_unverified: true,
            ..on()
        };
        let (app, fake) = app_with(cx, opted_in);
        fake.set_verifies_target(false);
        fake.focus(FakeTarget::named("Editor", "notes.txt"));
        // Focus comes to Starling while the insert is still queued.
        app.update(cx, |app, cx| {
            app.delivery_take_started();
            let capture = app.delivery_take_stopped();
            app.bind_delivery(capture, "take-1");
            app.sessions.push(session("take-1", "Hello there."));
            app.deliver_finished_take("take-1", cx);
            app.window_focus.push((Instant::now(), true));
            app.delivery_window_activation(true, cx);
        });
        cx.run_until_parked();
        assert!(fake.insertions().is_empty());
        assert_eq!(
            failure(&app, cx),
            Some(Failure::Insert(InsertError::TargetIsStarling))
        );

        // And mid-typing: what went out is reported as a partial delivery.
        let (app, fake) = app_with(cx, opted_in);
        fake.set_verifies_target(false);
        fake.focus(FakeTarget::named("Editor", "notes.txt"));
        let own_focus = app.read_with(cx, |app, _| app.delivery.own_focus.clone());
        fake.on_key(move |index| {
            if index == 5 {
                own_focus.store(true, Ordering::SeqCst);
            }
        });
        take(&app, cx, "take-1", "Hello there.", |_| {});
        assert!(fake.insertions().is_empty());
        app.read_with(cx, |app, _| {
            let recovery = app.delivery.recovery.as_ref().unwrap();
            assert_eq!(
                recovery.failure,
                Failure::Insert(InsertError::PartialDelivery {
                    delivered_chars: 5,
                    total_chars: 12,
                    cause: Box::new(InsertError::TargetIsStarling),
                })
            );
            assert_eq!(recovery.title(), "Inserted only in part");
        });
    }

    #[gpui::test]
    fn dictating_into_starling_or_with_insertion_off_types_nothing_quietly(
        cx: &mut gpui::TestAppContext,
    ) {
        let (app, fake) = app_with(cx, on());
        fake.focus(FakeTarget::owned_by_this_process());
        take(&app, cx, "take-1", "Into Starling.", |_| {});
        assert!(fake.insertions().is_empty());
        assert_eq!(failure(&app, cx), None);

        // Starling's own window focused at the start (Wayland cannot see
        // the pid; the window can).
        fake.focus(FakeTarget::named("Editor", "notes.txt"));
        app.update(cx, |app, _| app.window_focus.push((Instant::now(), true)));
        take(&app, cx, "take-2", "Into Starling.", |app| {
            app.window_focus.push((Instant::now(), false));
        });
        assert!(fake.insertions().is_empty());
        assert_eq!(failure(&app, cx), None);

        let (app, fake) = app_with(
            cx,
            InsertionSettings {
                auto_insert: false,
                ..on()
            },
        );
        fake.focus(FakeTarget::named("Editor", "notes.txt"));
        take(&app, cx, "take-1", "Copy only.", |_| {});
        assert!(fake.insertions().is_empty());
        assert_eq!(failure(&app, cx), None);
    }

    #[gpui::test]
    fn unverified_targets_need_the_opt_in_and_an_unbroken_focus(cx: &mut gpui::TestAppContext) {
        let (app, fake) = app_with(cx, on());
        fake.set_verifies_target(false);
        fake.focus(FakeTarget::named("Editor", "notes.txt"));
        take(&app, cx, "take-1", "Hello.", |_| {});
        assert!(fake.insertions().is_empty());
        assert_eq!(failure(&app, cx), Some(Failure::UnverifiedOff));
        app.read_with(cx, |app, _| {
            assert!(app.delivery.recovery.as_ref().unwrap().offers_settings())
        });

        let opted_in = InsertionSettings {
            allow_unverified: true,
            ..on()
        };
        let (app, fake) = app_with(cx, opted_in);
        fake.set_verifies_target(false);
        fake.focus(FakeTarget::named("Editor", "notes.txt"));
        take(&app, cx, "take-1", "Hello.", |app| {
            // The user clicked into Starling mid-take and left again.
            app.window_focus.push((Instant::now(), true));
            app.window_focus.push((Instant::now(), false));
        });
        assert!(fake.insertions().is_empty());
        assert_eq!(failure(&app, cx), Some(Failure::FocusMovedThroughStarling));

        take(&app, cx, "take-2", "Typed.", |_| {});
        assert_eq!(fake.insertions().len(), 1);
    }

    #[gpui::test]
    fn paste_last_captures_a_new_target_and_never_the_old_one(cx: &mut gpui::TestAppContext) {
        let (app, fake) = app_with(cx, on());
        fake.focus(FakeTarget::named("Editor", "notes.txt"));
        let original = fake.capture().unwrap();
        take(&app, cx, "take-1", "Hello there.", |_| {
            fake.focus(FakeTarget::named("Browser", "A tab"));
        });
        assert!(failure(&app, cx).is_some());

        // Armed from Starling's window: nothing happens until focus
        // leaves it, then the new field is captured after the settle.
        app.update(cx, |app, cx| {
            app.window_focus.push((Instant::now(), true));
            app.toggle_paste_last(cx);
        });
        cx.executor().advance_clock(PASTE_SETTLE * 2);
        cx.run_until_parked();
        assert!(fake.insertions().is_empty());

        fake.focus(FakeTarget::named("Chat", "Message"));
        let chosen = fake.capture().unwrap();
        app.update(cx, |app, cx| {
            app.window_focus.push((Instant::now(), false));
            app.delivery_window_activation(false, cx);
        });
        cx.executor().advance_clock(PASTE_SETTLE);
        cx.run_until_parked();
        assert_eq!(
            fake.insertions(),
            vec![(chosen.target_ref.clone(), "Hello there.".to_string())]
        );
        assert_ne!(chosen.target_ref, original.target_ref);
        assert_eq!(
            failure(&app, cx),
            None,
            "a successful paste resolves the notice"
        );
    }

    #[gpui::test]
    fn paste_last_waits_out_the_settle_of_the_latest_focus_loss(cx: &mut gpui::TestAppContext) {
        let (app, fake) = app_with(cx, on());
        fake.focus(FakeTarget::named("Editor", "notes.txt"));
        take(&app, cx, "take-1", "Hello there.", |_| {
            fake.destroy_target()
        });
        let focus = |app: &gpui::Entity<StarlingApp>, cx: &mut gpui::TestAppContext, active| {
            app.update(cx, |app, cx| {
                app.window_focus.push((Instant::now(), active));
                app.delivery_window_activation(active, cx);
            })
        };
        app.update(cx, |app, cx| {
            app.window_focus.push((Instant::now(), true));
            app.toggle_paste_last(cx);
        });
        // Out to a window switcher, back, and out again: the first loss's
        // timer must not capture the switcher.
        fake.focus(FakeTarget::named("Switcher", "Overview"));
        focus(&app, cx, false);
        cx.executor().advance_clock(PASTE_SETTLE / 4);
        focus(&app, cx, true);
        cx.executor().advance_clock(PASTE_SETTLE / 2);
        focus(&app, cx, false);
        cx.executor().advance_clock(PASTE_SETTLE / 2);
        cx.run_until_parked();
        assert!(fake.insertions().is_empty(), "{:?}", fake.insertions());

        fake.focus(FakeTarget::named("Chat", "Message"));
        let chosen = fake.capture().unwrap();
        cx.executor().advance_clock(PASTE_SETTLE / 2);
        cx.run_until_parked();
        assert_eq!(
            fake.insertions(),
            vec![(chosen.target_ref, "Hello there.".to_string())]
        );
    }

    #[gpui::test]
    fn paste_last_refuses_a_target_that_changes_before_typing(cx: &mut gpui::TestAppContext) {
        let (app, fake) = app_with(cx, on());
        fake.focus(FakeTarget::named("Editor", "notes.txt"));
        take(&app, cx, "take-1", "Hello there.", |_| {
            fake.destroy_target()
        });

        // The settled capture is revalidated inside the insert: a focus
        // change between capture and typing refuses.
        fake.focus(FakeTarget::named("Chat", "Message"));
        fake.fail_revalidate(Some(InsertError::TargetGone));
        app.update(cx, |app, cx| app.toggle_paste_last(cx));
        cx.executor().advance_clock(PASTE_SETTLE);
        cx.run_until_parked();
        assert!(fake.insertions().is_empty());
        app.read_with(cx, |app, _| {
            let recovery = app.delivery.recovery.as_ref().unwrap();
            assert_eq!(recovery.failure, Failure::Insert(InsertError::TargetGone));
            assert!(!recovery.pasting);
            assert_eq!(recovery.armed, None, "one press types at most once");
        });

        // Paste last back into Starling's own window is refused too.
        fake.fail_revalidate(None);
        fake.focus(FakeTarget::owned_by_this_process());
        app.update(cx, |app, cx| app.toggle_paste_last(cx));
        cx.executor().advance_clock(PASTE_SETTLE);
        cx.run_until_parked();
        assert!(fake.insertions().is_empty());
        assert_eq!(
            failure(&app, cx),
            Some(Failure::Insert(InsertError::TargetIsStarling))
        );
    }

    #[gpui::test]
    fn an_armed_paste_expires_and_a_new_take_disarms_it(cx: &mut gpui::TestAppContext) {
        let (app, fake) = app_with(cx, on());
        fake.focus(FakeTarget::named("Editor", "notes.txt"));
        take(&app, cx, "take-1", "Hello there.", |_| {
            fake.destroy_target()
        });
        let armed = |app: &gpui::Entity<StarlingApp>, cx: &mut gpui::TestAppContext| {
            app.read_with(cx, |app, _| {
                app.delivery.recovery.as_ref().unwrap().armed.is_some()
            })
        };

        app.update(cx, |app, cx| {
            app.window_focus.push((Instant::now(), true));
            app.toggle_paste_last(cx);
        });
        assert!(armed(&app, cx));
        cx.executor().advance_clock(PASTE_ARM_TIMEOUT);
        cx.run_until_parked();
        assert!(!armed(&app, cx));

        app.update(cx, |app, cx| app.toggle_paste_last(cx));
        assert!(armed(&app, cx));
        app.update(cx, |app, _| app.delivery_take_started());
        assert!(!armed(&app, cx));
        // Focus leaving now (the user went back to dictate) types nothing.
        fake.focus(FakeTarget::named("Chat", "Message"));
        app.update(cx, |app, cx| {
            app.window_focus.push((Instant::now(), false));
            app.delivery_window_activation(false, cx);
        });
        cx.executor().advance_clock(PASTE_SETTLE);
        cx.run_until_parked();
        assert!(fake.insertions().is_empty());
    }

    #[gpui::test]
    fn a_dismissed_notice_ignores_a_late_paste_result(cx: &mut gpui::TestAppContext) {
        let (app, fake) = app_with(cx, on());
        fake.focus(FakeTarget::named("Editor", "notes.txt"));
        take(&app, cx, "take-1", "Hello there.", |_| {
            fake.destroy_target()
        });
        fake.focus(FakeTarget::named("Chat", "Message"));
        fake.set_insert_behavior(InsertBehavior::FailWith(InsertError::TargetGone));
        app.update(cx, |app, cx| {
            let generation = app.delivery.generation;
            app.dismiss_recovery(cx);
            app.insert_finished(
                "take-1".into(),
                "Hello there.".into(),
                Some(generation),
                Err(InsertError::TargetGone),
                cx,
            );
            assert!(app.delivery.recovery.is_none());
        });
    }

    #[gpui::test]
    fn copy_writes_the_text_only_when_asked(cx: &mut gpui::TestAppContext) {
        let (app, fake) = app_with(cx, on());
        cx.write_to_clipboard(ClipboardItem::new_string("the user's own copy".into()));
        fake.focus(FakeTarget::named("Editor", "notes.txt"));
        take(&app, cx, "take-1", "Hello there.", |_| {
            fake.destroy_target()
        });
        // A failed delivery leaves the user's clipboard alone.
        assert_eq!(
            cx.read_from_clipboard().and_then(|item| item.text()),
            Some("the user's own copy".to_string())
        );
        app.update(cx, |app, cx| app.copy_recovery(cx));
        assert_eq!(
            cx.read_from_clipboard().and_then(|item| item.text()),
            Some("Hello there.".to_string())
        );
        app.read_with(cx, |app, _| {
            assert!(app.delivery.recovery.as_ref().unwrap().copied)
        });
    }

    #[gpui::test]
    fn a_take_without_a_transcript_job_never_delivers(cx: &mut gpui::TestAppContext) {
        let (app, fake) = app_with(cx, on());
        fake.focus(FakeTarget::named("Editor", "notes.txt"));
        app.update(cx, |app, cx| {
            app.delivery_take_started();
            let capture = app.delivery_take_stopped();
            app.bind_delivery(capture, "take-1");
            app.forget_delivery("take-1");
            app.sessions.push(session("take-1", "Retried later."));
            app.deliver_finished_take("take-1", cx);
        });
        cx.run_until_parked();
        assert!(fake.insertions().is_empty());
        assert_eq!(failure(&app, cx), None);
    }
}
