//! Focus-safe delivery of finished takes (#221): the text goes into the
//! window that had focus when the take started, without Starling's window
//! ever coming forward.
//!
//! - **Capture at start.** `start_recording` captures the focused target
//!   before anything else can move focus. The capture is a display round
//!   trip: it runs on a worker thread, and the UI thread waits at most
//!   [`CAPTURE_BUDGET`] for it (a display that does not answer by then
//!   counts as unavailable; a later answer could name a later window). The
//!   capture travels with the take (`stop_recording` → the save → its store
//!   id).
//! - **Deliver once.** A direct take's raw transcript is typed into the
//!   captured target when it lands. A staged take (#297) waits in its
//!   panel: the user edits it, maybe runs a mode over it, and presses
//!   Insert; the draft as it stands then is typed once focus leaves
//!   Starling's window (the user switching back). The backend
//!   revalidates the target immediately before typing and before every
//!   chunk; nothing is ever submitted (control characters are refused, so
//!   no Enter). A capture is consumed by its first delivery attempt: a
//!   retried transcription or a second Insert never types again.
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
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, mpsc};
use std::time::{Duration, Instant};

use gpui::{AppContext, ClipboardItem, Context, Task};
use starling_dictation::settings::InsertionSettings;
use starling_insertion::{InsertError, Inserter, TargetSnapshot};

use crate::app::StarlingApp;

/// How long an armed Paste last waits for the window to lose focus.
pub(crate) const PASTE_ARM_TIMEOUT: Duration = Duration::from_secs(15);
/// After Starling's window loses focus, the new focus settles this long
/// before it is captured (a window switcher may pass through first).
pub(crate) const PASTE_SETTLE: Duration = Duration::from_millis(350);

/// How long the UI thread waits for a capture's display round trip
/// (normally well under a millisecond).
pub(crate) const CAPTURE_BUDGET: Duration = Duration::from_millis(100);

/// How long an insert may take before it is given up as stalled (a
/// display that stopped answering), so the inserts after it still run.
fn typing_budget(text: &str) -> Duration {
    Duration::from_secs(10) + Duration::from_millis(20) * text.chars().count() as u32
}

/// What a take captured when it started.
#[derive(Clone, Debug)]
pub(crate) struct Capture {
    pub target: Result<TargetSnapshot, InsertError>,
    pub at: Instant,
}

/// Clears a worker's "out" flag when the worker ends, also by a panic:
/// a flag left set would refuse every later worker.
struct OutGuard(Arc<AtomicBool>);

impl Drop for OutGuard {
    fn drop(&mut self) {
        self.0.store(false, Ordering::SeqCst);
    }
}

/// The inserter's capture on a worker thread, waited for at most
/// [`CAPTURE_BUDGET`]: a stuck display cannot freeze the UI thread, and an
/// answer that comes later is not taken (focus may have moved by then).
/// `out` is set while a worker is out; one the display never answered
/// is not joined by more. The cutoff is absolute: an answer the worker
/// had after it is refused even when the UI thread, itself late, finds it
/// already waiting.
fn bounded_capture(
    inserter: Arc<Inserter>,
    out: &Arc<AtomicBool>,
) -> Result<TargetSnapshot, InsertError> {
    bounded_capture_after(inserter, out, || {})
}

/// [`bounded_capture`], running `stall` between starting the worker and
/// waiting for it: tests stand in a descheduled UI thread with it.
fn bounded_capture_after(
    inserter: Arc<Inserter>,
    out: &Arc<AtomicBool>,
    stall: impl FnOnce(),
) -> Result<TargetSnapshot, InsertError> {
    let deadline = Instant::now() + CAPTURE_BUDGET;
    if out.swap(true, Ordering::SeqCst) {
        return Err(InsertError::Unavailable {
            reason: "the display has not answered an earlier focus check".to_string(),
        });
    }
    let (sender, receiver) = mpsc::channel();
    let worker_out = out.clone();
    let spawned = std::thread::Builder::new()
        .name("starling-capture".into())
        .spawn(move || {
            let worker = OutGuard(worker_out);
            let target = inserter.capture();
            let answered = Instant::now();
            // Cleared before the answer, so the next capture never sees
            // an answered worker as out.
            drop(worker);
            let _ = sender.send((target, answered));
        });
    if let Err(error) = spawned {
        out.store(false, Ordering::SeqCst);
        return Err(InsertError::Unavailable {
            reason: format!("cannot start the focus check: {error}"),
        });
    }
    stall();
    // A channel hands over a waiting answer before it looks at the
    // timeout, so the answer's own time decides.
    match receiver.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
        Ok((target, answered)) if answered <= deadline => target,
        _ => Err(InsertError::Unavailable {
            reason: "the display did not say in time which window has focus".to_string(),
        }),
    }
}

/// Whether the Paste last or Insert armed at `armed_at` (on the executor's
/// clock) is past [`PASTE_ARM_TIMEOUT`], whether or not its expiry timer
/// has run.
fn arming_expired(armed_at: Instant, cx: &Context<StarlingApp>) -> bool {
    cx.background_executor()
        .now()
        .saturating_duration_since(armed_at)
        >= PASTE_ARM_TIMEOUT
}

/// The stop check of one insert: Starling's window took focus, or the
/// insert overran its budget and was given up.
fn insert_stop(
    had_focus: impl Fn() -> bool + Send + 'static,
    abandoned: Arc<AtomicBool>,
) -> impl Fn() -> Option<InsertError> + Send + 'static {
    move || {
        if abandoned.load(Ordering::SeqCst) {
            Some(InsertError::Unavailable {
                reason: "typing stalled and was given up".to_string(),
            })
        } else {
            had_focus().then_some(InsertError::TargetIsStarling)
        }
    }
}

/// Why text did not land (completely).
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Failure {
    Insert(InsertError),
    /// Typing did not finish within its budget and was given up.
    Stalled,
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
            Failure::Stalled => "Not inserted: typing stalled",
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
            Failure::Stalled => {
                "Typing did not finish in time: the display stopped answering. Some of the text \
                 may already be in the window; check it before pasting again."
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

/// Where a staged take's Insert stands, for its panel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum StagedInsert {
    Ready,
    /// Pressed; types once focus leaves Starling's window.
    Waiting,
    Typing,
    Inserted,
}

/// A pressed Insert waiting for focus to leave Starling's window.
struct ArmedInsert {
    take_id: String,
    capture: Capture,
    armed_at: Instant,
}

/// Starling's window focus as typing off the UI thread sees it (Wayland
/// cannot tell that Starling's window took focus; Starling can).
#[derive(Default)]
pub(crate) struct OwnFocus {
    focused: AtomicBool,
    /// Bumped on every focus gain, so a trip through Starling's window
    /// and back out still counts.
    gains: AtomicU64,
}

impl OwnFocus {
    fn set(&self, active: bool) {
        if active {
            self.gains.fetch_add(1, Ordering::SeqCst);
        }
        self.focused.store(active, Ordering::SeqCst);
    }

    /// The stop check for one insert: Starling's window has had focus
    /// at some point since this call.
    fn had_focus_since_now(self: &Arc<Self>) -> impl Fn() -> bool + Send + 'static {
        let start = self.gains.load(Ordering::SeqCst);
        let focus = self.clone();
        move || focus.focused.load(Ordering::SeqCst) || focus.gains.load(Ordering::SeqCst) != start
    }
}

/// The app's delivery state.
pub(crate) struct DeliveryState {
    pub(crate) inserter: Arc<Inserter>,
    pub(crate) settings: InsertionSettings,
    /// The running take's capture.
    live: Option<Capture>,
    /// Saved takes waiting for their transcript (direct) or for Insert
    /// (staged), by store id.
    by_take: HashMap<String, Capture>,
    /// A staged take's pressed Insert.
    staged_armed: Option<ArmedInsert>,
    /// Staged takes whose Insert is typing (`false`) or typed (`true`).
    staged_inserts: HashMap<String, bool>,
    pub(crate) recovery: Option<Recovery>,
    /// Bumped whenever the recovery is replaced or dismissed, so a late
    /// Paste last result lands only on the notice it was started from.
    generation: u64,
    /// Bumped on every focus change of Starling's window, so only the
    /// settle timer of the latest focus loss fires Paste last.
    focus_changes: u64,
    /// An insert stops before its next key once Starling's window has
    /// had focus since it was started.
    own_focus: Arc<OwnFocus>,
    /// The latest insert. The next one waits for it, so inserts type in
    /// the order they were started, one at a time.
    last_insert: Option<Task<()>>,
    /// A capture worker is out (see [`bounded_capture`]).
    capture_out: Arc<AtomicBool>,
    /// A settings session check is out (see
    /// [`StarlingApp::check_session_verifies`]).
    session_check_out: Arc<AtomicBool>,
    /// An insert is out on the background executor. One given up as
    /// stalled may never return; later inserts are refused rather than
    /// queued behind it on the insertion lock, so at most one worker is
    /// ever stuck.
    insert_out: Arc<AtomicBool>,
}

impl DeliveryState {
    pub(crate) fn new(inserter: Arc<Inserter>, settings: InsertionSettings) -> DeliveryState {
        DeliveryState {
            inserter,
            settings,
            live: None,
            by_take: HashMap::new(),
            staged_armed: None,
            staged_inserts: HashMap::new(),
            recovery: None,
            generation: 0,
            focus_changes: 0,
            own_focus: Arc::default(),
            last_insert: None,
            capture_out: Arc::default(),
            session_check_out: Arc::default(),
            insert_out: Arc::default(),
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

    /// Captures the focused target now. Starling's own focus is decided
    /// here: dictating into Starling itself. On Wayland only the app can
    /// tell; elsewhere the backend refuses Starling's pid too.
    fn capture_now(&self) -> Capture {
        let at = Instant::now();
        let target = if self.starling_focused() {
            Err(InsertError::TargetIsStarling)
        } else {
            bounded_capture(self.delivery.inserter.clone(), &self.delivery.capture_out)
        };
        Capture { target, at }
    }

    /// New insertion settings, from the next delivery on. Copy only
    /// disarms a pressed Insert, which would otherwise type on the next
    /// focus loss.
    pub(crate) fn set_insertion_settings(&mut self, settings: InsertionSettings) {
        self.delivery.settings = settings;
        if !settings.auto_insert {
            self.disarm_staged_insert();
        }
    }

    /// Asks off the UI thread whether this session types where the target
    /// cannot be verified, for the settings note. At most one check is
    /// out, on its own thread: one a backend never answers is not joined
    /// by more, and the note keeps the last answer meanwhile.
    pub(crate) fn check_session_verifies(&mut self, cx: &mut Context<Self>) {
        let out = self.delivery.session_check_out.clone();
        if out.swap(true, Ordering::SeqCst) {
            return;
        }
        let inserter = self.delivery.inserter.clone();
        let (sender, answer) = tokio::sync::oneshot::channel();
        let worker_out = out.clone();
        let spawned = std::thread::Builder::new()
            .name("starling-session-check".into())
            .spawn(move || {
                let worker = OutGuard(worker_out);
                let verifies = inserter.session_verifies();
                drop(worker);
                let _ = sender.send(verifies);
            });
        if spawned.is_err() {
            out.store(false, Ordering::SeqCst);
            return;
        }
        cx.spawn(async move |this, cx| {
            let Ok(verifies) = answer.await else {
                return;
            };
            this.update(cx, |app, cx| {
                app.insertion_unverifiable = verifies == Some(false);
                cx.notify();
            })
            .ok();
        })
        .detach();
    }

    /// A take started: capture where its text goes, before anything can
    /// move focus. An armed Paste last is disarmed: the user is dictating
    /// again, not choosing a field.
    pub(crate) fn delivery_take_started(&mut self) {
        if let Some(recovery) = self.delivery.recovery.as_mut() {
            recovery.armed = None;
        }
        self.disarm_staged_insert();
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
        // A pressed Insert goes with its panel.
        if self
            .delivery
            .staged_armed
            .as_ref()
            .is_some_and(|armed| armed.take_id == id)
        {
            self.delivery.staged_armed = None;
        }
    }

    /// A direct take's transcript landed: type its text into the
    /// captured target, once.
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

    /// Types `text` into `target` off the UI thread, after every insert
    /// started before it finished. `paste` is the recovery generation a
    /// Paste last runs for; `None` for the take's own delivery. An insert
    /// that overruns its [`typing_budget`] is reported as stalled and
    /// stops before its next key should it ever resume, so one stuck
    /// insert cannot hold up the ones after it.
    fn spawn_insert(
        &mut self,
        id: String,
        target: TargetSnapshot,
        text: String,
        paste: Option<u64>,
        cx: &mut Context<Self>,
    ) {
        let inserter = self.delivery.inserter.clone();
        let had_focus = self.delivery.own_focus.had_focus_since_now();
        let typed_text = text.clone();
        let previous = self.delivery.last_insert.take();
        let budget = typing_budget(&text);
        let out = self.delivery.insert_out.clone();
        let insert = cx.spawn(async move |this, cx| {
            if let Some(previous) = previous {
                previous.await;
            }
            let result = if out.swap(true, Ordering::SeqCst) {
                Err(Failure::Insert(InsertError::Unavailable {
                    reason: "an earlier insert is still waiting for the display".to_string(),
                }))
            } else {
                let abandoned = Arc::new(AtomicBool::new(false));
                let stop = insert_stop(had_focus, abandoned.clone());
                let typing = cx.background_spawn(async move {
                    let _worker = OutGuard(out);
                    inserter.insert(&target, &typed_text, &stop)
                });
                let timer = cx.background_executor().timer(budget);
                match futures_util::future::select(typing, timer).await {
                    futures_util::future::Either::Left((result, _)) => {
                        result.map_err(Failure::Insert)
                    }
                    futures_util::future::Either::Right(((), typing)) => {
                        abandoned.store(true, Ordering::SeqCst);
                        typing.detach();
                        Err(Failure::Stalled)
                    }
                }
            };
            this.update(cx, |app, cx| {
                app.insert_finished(id, text, paste, result, cx)
            })
            .ok();
        });
        self.delivery.last_insert = Some(insert);
    }

    fn insert_finished(
        &mut self,
        id: String,
        text: String,
        paste: Option<u64>,
        result: Result<starling_insertion::InsertReceipt, Failure>,
        cx: &mut Context<Self>,
    ) {
        // A staged take's Insert shows "Inserted"; after a failure, the
        // notice's Paste last is the retry.
        if result.is_ok() && self.staged_text_for(&id).is_some() {
            self.delivery.staged_inserts.insert(id.clone(), true);
        } else if paste.is_none() {
            self.delivery.staged_inserts.remove(&id);
        }
        match paste {
            None => {
                if let Err(failure) = result {
                    self.delivery
                        .replace_recovery(Some(Recovery::new(&id, text, failure)));
                }
            }
            Some(generation) => {
                if generation != self.delivery.generation {
                    // The notice was dismissed or replaced meanwhile.
                    return;
                }
                match result {
                    Ok(_) => self.delivery.replace_recovery(None),
                    Err(failure) => {
                        if let Some(recovery) = self.delivery.recovery.as_mut() {
                            recovery.pasting = false;
                            recovery.failure = failure;
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
        let armed_at = cx.background_executor().now();
        recovery.armed = Some(armed_at);
        // One text waits for the next window at a time.
        self.disarm_staged_insert();
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
            self.after_focus_settles(armed_at, Self::fire_paste_last, cx);
        }
        cx.notify();
    }

    /// Where the staged take `id`'s Insert stands; `None` when there is
    /// nothing to insert into (copy only, dictated into Starling, a
    /// failure handed to the recovery notice, or a closed panel).
    pub(crate) fn staged_insert(&self, id: &str) -> Option<StagedInsert> {
        if let Some(&typed) = self.delivery.staged_inserts.get(id) {
            return Some(if typed {
                StagedInsert::Inserted
            } else {
                StagedInsert::Typing
            });
        }
        if self
            .delivery
            .staged_armed
            .as_ref()
            .is_some_and(|armed| armed.take_id == id)
        {
            return Some(StagedInsert::Waiting);
        }
        let capture = self.delivery.by_take.get(id)?;
        (self.delivery.settings.auto_insert
            && !matches!(capture.target, Err(InsertError::TargetIsStarling)))
        .then_some(StagedInsert::Ready)
    }

    /// "Insert" for the staged take `id` (#297): its draft goes into the
    /// window the take started in, once. The panel is in Starling's
    /// window, so the insert waits for focus to leave it (the user
    /// switching back) and its settle; the target is then revalidated as
    /// for a direct take. Pressing it again while it waits cancels; it
    /// expires like Paste last.
    pub(crate) fn insert_staged(&mut self, id: &str, cx: &mut Context<Self>) {
        match self.staged_insert(id) {
            Some(StagedInsert::Ready) => {}
            Some(StagedInsert::Waiting) => {
                self.disarm_staged_insert();
                cx.notify();
                return;
            }
            _ => return,
        }
        let Some(capture) = self.delivery.by_take.remove(id) else {
            return;
        };
        if let Some(recovery) = self.delivery.recovery.as_mut() {
            recovery.armed = None;
        }
        self.disarm_staged_insert();
        let armed_at = cx.background_executor().now();
        self.delivery.staged_armed = Some(ArmedInsert {
            take_id: id.to_string(),
            capture,
            armed_at,
        });
        cx.spawn(async move |this, cx| {
            cx.background_executor().timer(PASTE_ARM_TIMEOUT).await;
            this.update(cx, |app, cx| {
                if app
                    .delivery
                    .staged_armed
                    .as_ref()
                    .is_some_and(|armed| armed.armed_at == armed_at)
                {
                    app.disarm_staged_insert();
                    cx.notify();
                }
            })
            .ok();
        })
        .detach();
        if !self.starling_focused() {
            self.after_focus_settles(armed_at, Self::fire_staged_insert, cx);
        }
        cx.notify();
    }

    /// A pressed Insert that has not typed goes back to waiting for a
    /// press, if its panel is still there to press it.
    fn disarm_staged_insert(&mut self) {
        if let Some(armed) = self.delivery.staged_armed.take() {
            if self.staging_shows(&armed.take_id) {
                self.delivery.by_take.insert(armed.take_id, armed.capture);
            }
        }
    }

    /// The pressed Insert fires: the draft as it stands now goes into the
    /// take's captured target.
    fn fire_staged_insert(&mut self, armed_at: Instant, cx: &mut Context<Self>) {
        let armed = self
            .delivery
            .staged_armed
            .as_ref()
            .is_some_and(|armed| armed.armed_at == armed_at);
        // Back in Starling before the focus settled: stay armed.
        if !armed || self.starling_focused() {
            return;
        }
        // The expiry timer may not have run yet, and copy only may have
        // been chosen since the press.
        if arming_expired(armed_at, cx) || !self.delivery.settings.auto_insert {
            self.disarm_staged_insert();
            cx.notify();
            return;
        }
        let Some(ArmedInsert {
            take_id: id,
            capture,
            ..
        }) = self.delivery.staged_armed.take()
        else {
            return;
        };
        let Some(text) = self.staged_text_for(&id).or_else(|| self.head_text(&id)) else {
            cx.notify();
            return;
        };
        // Starling's window had focus: the Insert was pressed there. A
        // verifiable target must still be the captured one. A target that
        // was captured but cannot be checked (Wayland) says nothing about
        // which window that was, so the window the user switched to after
        // pressing Insert is captured and receives the text, as with Paste
        // last. A failed capture is reported, never replaced.
        let unverifiable = capture
            .target
            .as_ref()
            .is_ok_and(|target| !self.delivery.inserter.verifies(target));
        let capture = if unverifiable {
            self.capture_now()
        } else {
            capture
        };
        let verified = self.verifiable(&capture);
        match plan(&capture, self.delivery.settings, verified, false, &text) {
            Plan::Skip => {}
            Plan::Fail(failure) => {
                self.delivery
                    .replace_recovery(Some(Recovery::new(&id, text, failure)));
            }
            Plan::Insert(target) => {
                self.delivery.staged_inserts.insert(id.clone(), false);
                self.spawn_insert(id, target, text, None, cx);
            }
        }
        cx.notify();
    }

    fn verifiable(&self, capture: &Capture) -> bool {
        capture
            .target
            .as_ref()
            .is_ok_and(|target| self.delivery.inserter.verifies(target))
    }

    /// Starling's window gained or lost focus.
    pub(crate) fn delivery_window_activation(&mut self, active: bool, cx: &mut Context<Self>) {
        self.delivery.focus_changes += 1;
        self.delivery.own_focus.set(active);
        if active {
            return;
        }
        if let Some(armed_at) = self
            .delivery
            .recovery
            .as_ref()
            .and_then(|recovery| recovery.armed)
        {
            self.after_focus_settles(armed_at, Self::fire_paste_last, cx);
        }
        if let Some(armed_at) = self
            .delivery
            .staged_armed
            .as_ref()
            .map(|armed| armed.armed_at)
        {
            self.after_focus_settles(armed_at, Self::fire_staged_insert, cx);
        }
    }

    /// Runs `fire` for the arming at `armed_at` once the current focus
    /// settled, unless focus moved again meanwhile.
    fn after_focus_settles(
        &mut self,
        armed_at: Instant,
        fire: fn(&mut Self, Instant, &mut Context<Self>),
        cx: &mut Context<Self>,
    ) {
        let focus_changes = self.delivery.focus_changes;
        cx.spawn(async move |this, cx| {
            cx.background_executor().timer(PASTE_SETTLE).await;
            this.update(cx, |app, cx| {
                // Focus moved again meanwhile: that move's own timer
                // decides, after its full settle.
                if app.delivery.focus_changes == focus_changes {
                    fire(app, armed_at, cx);
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
        if arming_expired(armed_at, cx) {
            if let Some(recovery) = self.delivery.recovery.as_mut() {
                recovery.armed = None;
            }
            cx.notify();
            return;
        }
        let capture = self.capture_now();
        let generation = self.delivery.generation;
        let Some(recovery) = self.delivery.recovery.as_mut() else {
            return;
        };
        recovery.armed = None;
        let (id, text) = (recovery.take_id.clone(), recovery.text.clone());
        let verified = self.verifiable(&capture);
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
impl StarlingApp {
    /// Moves the armed Paste last and Insert `by` into the past, as if
    /// the UI thread had been too busy to run their expiry timers: those
    /// no longer match them, so only the fire-time check is left.
    pub(crate) fn backdate_armings(&mut self, by: Duration) {
        let back = |at: &mut Instant| *at = at.checked_sub(by).expect("an earlier instant");
        if let Some(armed) = self.delivery.staged_armed.as_mut() {
            back(&mut armed.armed_at);
        }
        if let Some(at) = self
            .delivery
            .recovery
            .as_mut()
            .and_then(|recovery| recovery.armed.as_mut())
        {
            back(at);
        }
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

    /// Takes finishing together type one after another, in the order
    /// they were delivered. The test executor runs background work in a
    /// random order per seed, so without the chain some seeds type the
    /// later take first.
    #[gpui::test(iterations = 30)]
    fn overlapping_deliveries_type_whole_texts_in_take_order(cx: &mut gpui::TestAppContext) {
        let (app, fake) = app_with(cx, on());
        fake.focus(FakeTarget::named("Editor", "notes.txt"));
        let target = fake.capture().unwrap();
        app.update(cx, |app, cx| {
            for (id, text) in [
                ("take-1", "first take."),
                ("take-2", "second take."),
                ("take-3", "third take."),
            ] {
                app.delivery_take_started();
                let capture = app.delivery_take_stopped();
                app.bind_delivery(capture, id);
                app.sessions.push(session(id, text));
            }
            for id in ["take-1", "take-2", "take-3"] {
                app.deliver_finished_take(id, cx);
            }
        });
        cx.run_until_parked();
        assert_eq!(fake.field(), "first take.second take.third take.");
        assert_eq!(
            fake.insertions(),
            ["first take.", "second take.", "third take."]
                .map(|text| (target.target_ref.clone(), text.to_string()))
                .to_vec()
        );
        assert_eq!(failure(&app, cx), None);
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
        // Focus passes through Starling while the insert is still queued:
        // the window it lands in afterwards is not the take's.
        app.update(cx, |app, cx| {
            app.delivery_take_started();
            let capture = app.delivery_take_stopped();
            app.bind_delivery(capture, "take-1");
            app.sessions.push(session("take-1", "Hello there."));
            app.deliver_finished_take("take-1", cx);
            for active in [true, false] {
                app.window_focus.push((Instant::now(), active));
                app.delivery_window_activation(active, cx);
            }
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
                own_focus.set(true);
                own_focus.set(false);
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

    /// A settle that runs after an armed Paste last expired, before its
    /// expiry timer got to run, types nothing.
    #[gpui::test]
    fn an_expired_paste_never_fires_ahead_of_its_timer(cx: &mut gpui::TestAppContext) {
        let (app, fake) = app_with(cx, on());
        fake.focus(FakeTarget::named("Editor", "notes.txt"));
        take(&app, cx, "take-1", "Hello there.", |_| {
            fake.destroy_target()
        });
        app.update(cx, |app, cx| {
            app.window_focus.push((Instant::now(), true));
            app.toggle_paste_last(cx);
            app.backdate_armings(PASTE_ARM_TIMEOUT);
        });
        fake.focus(FakeTarget::named("Chat", "Message"));
        app.update(cx, |app, cx| {
            app.window_focus.push((Instant::now(), false));
            app.delivery_window_activation(false, cx);
        });
        cx.executor().advance_clock(PASTE_SETTLE);
        cx.run_until_parked();
        assert!(fake.insertions().is_empty());
        app.read_with(cx, |app, _| {
            assert_eq!(app.delivery.recovery.as_ref().unwrap().armed, None);
        });
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
                Err(Failure::Insert(InsertError::TargetGone)),
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

    /// An insert that overran its budget is given up: should the display
    /// ever answer again, it stops before its next key.
    #[test]
    fn a_given_up_insert_stops_before_its_next_key() {
        let fake = FakeBackend::new();
        fake.focus(FakeTarget::named("Editor", "notes.txt"));
        let target = fake.capture().unwrap();
        let abandoned = Arc::new(AtomicBool::new(false));
        let stop = insert_stop(|| false, abandoned.clone());
        fake.on_key(move |index| {
            if index == 3 {
                abandoned.store(true, Ordering::SeqCst);
            }
        });
        let result = fake.insert_guarded(&target, "Hello there.", &stop);
        assert!(
            matches!(
                result,
                Err(InsertError::PartialDelivery {
                    delivered_chars: 3,
                    ..
                })
            ),
            "{result:?}"
        );
        assert_eq!(fake.field(), "Hel");
        let notice = Recovery::new("take-1", "Hello there.".into(), Failure::Stalled);
        assert_eq!(notice.title(), "Not inserted: typing stalled");
        assert!(
            notice
                .explanation()
                .contains("check it before pasting again")
        );
    }

    /// A display that answers the capture late counts as unavailable:
    /// the UI thread does not wait for it, and the late answer could name
    /// a window focused after the take started.
    #[gpui::test]
    fn a_capture_the_display_answers_late_is_unavailable(cx: &mut gpui::TestAppContext) {
        let (app, fake) = app_with(cx, on());
        fake.focus(FakeTarget::named("Editor", "notes.txt"));
        fake.set_capture_delay(CAPTURE_BUDGET * 3);
        take(&app, cx, "take-1", "Hello there.", |_| {});
        assert!(fake.insertions().is_empty());
        assert!(matches!(
            failure(&app, cx),
            Some(Failure::Insert(InsertError::Unavailable { .. }))
        ));
    }

    /// A capture the display never answered is not joined by more
    /// workers: the next one is refused until it returns.
    #[gpui::test]
    fn a_capture_still_out_refuses_the_next(cx: &mut gpui::TestAppContext) {
        let (app, fake) = app_with(cx, on());
        fake.focus(FakeTarget::named("Editor", "notes.txt"));
        fake.set_capture_delay(CAPTURE_BUDGET * 3);
        take(&app, cx, "take-1", "First.", |_| {});
        fake.set_capture_delay(Duration::ZERO);
        take(&app, cx, "take-2", "Second.", |_| {});
        assert!(fake.insertions().is_empty());
        app.read_with(cx, |app, _| {
            let recovery = app.delivery.recovery.as_ref().unwrap();
            assert_eq!(recovery.take_id, "take-2");
            assert!(
                recovery.explanation().contains("earlier focus check"),
                "{}",
                recovery.explanation()
            );
        });
        // Once the late worker returns, captures work again.
        std::thread::sleep(CAPTURE_BUDGET * 4);
        take(&app, cx, "take-3", "Third.", |_| {});
        assert_eq!(fake.field(), "Third.");
    }

    /// The cutoff is absolute: an answer had in time is taken even when
    /// the UI thread comes for it late, and one had after the cutoff is
    /// refused even though it is already waiting.
    #[test]
    fn a_capture_answered_after_the_cutoff_is_refused_though_waiting() {
        let (fake, inserter) = fake_session();
        fake.focus(FakeTarget::named("Editor", "notes.txt"));
        let out = Arc::new(AtomicBool::new(false));
        let descheduled = || std::thread::sleep(CAPTURE_BUDGET * 3);
        let prompt = bounded_capture_after(inserter.clone(), &out, descheduled);
        assert!(prompt.is_ok(), "{prompt:?}");
        fake.set_capture_delay(CAPTURE_BUDGET * 3 / 2);
        let late = bounded_capture_after(inserter, &out, descheduled);
        assert!(
            matches!(late, Err(InsertError::Unavailable { .. })),
            "{late:?}"
        );
    }

    /// A capture worker that panics fails its capture and does not keep
    /// refusing the ones after it.
    #[test]
    fn a_panicked_capture_worker_does_not_block_later_captures() {
        let (fake, inserter) = fake_session();
        fake.focus(FakeTarget::named("Editor", "notes.txt"));
        let out = Arc::new(AtomicBool::new(false));
        fake.set_capture_panics(true);
        let panicked = bounded_capture(inserter.clone(), &out);
        assert!(
            matches!(panicked, Err(InsertError::Unavailable { .. })),
            "{panicked:?}"
        );
        fake.set_capture_panics(false);
        let next = bounded_capture(inserter, &out);
        assert!(next.is_ok(), "{next:?}");
    }

    /// Opening Settings again while a session check is unanswered starts
    /// no second one; the note keeps the last answer until it returns.
    #[gpui::test]
    fn an_unanswered_session_check_is_not_joined_by_more(cx: &mut gpui::TestAppContext) {
        let (app, fake) = app_with(cx, on());
        let unverifiable = |app: &gpui::Entity<StarlingApp>, cx: &mut gpui::TestAppContext| {
            app.read_with(cx, |app, _| app.insertion_unverifiable)
        };
        fake.set_verifies_target(false);
        fake.set_availability_delay(Duration::from_millis(200));
        app.update(cx, |app, cx| {
            app.check_session_verifies(cx);
            app.check_session_verifies(cx);
            app.check_session_verifies(cx);
        });
        cx.run_until_parked();
        assert!(!unverifiable(&app, cx));
        std::thread::sleep(Duration::from_millis(600));
        cx.run_until_parked();
        assert_eq!(fake.availability_checks(), 1);
        assert!(unverifiable(&app, cx));

        // Once answered, the next opening asks again.
        fake.set_availability_delay(Duration::ZERO);
        fake.set_verifies_target(true);
        app.update(cx, |app, cx| app.check_session_verifies(cx));
        std::thread::sleep(Duration::from_millis(200));
        cx.run_until_parked();
        assert_eq!(fake.availability_checks(), 2);
        assert!(!unverifiable(&app, cx));
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
