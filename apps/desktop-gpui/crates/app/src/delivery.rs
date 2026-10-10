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
//! - **Insertion boundary (#341).** When a take starts, the field that has
//!   focus is located too (AT-SPI on Linux), off the UI thread. Right
//!   before typing, the text before that field's caret is read, if it is
//!   still the focused field, and the insertion-boundary rules adjust the
//!   leading space and the first letter's case. A password field is never
//!   read, a verbatim mode (the take's, or one a spoken override routes it
//!   to) reads nothing, and a field that cannot be read gets the text as
//!   dictated. An adjusted delivery is recorded as a revision derived
//!   from the take's text; the head stays the unadjusted text, so Copy and
//!   history give it as dictated. The field's text is used for this
//!   decision only and kept nowhere.
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
use std::sync::{Arc, Mutex, PoisonError, mpsc};
use std::time::{Duration, Instant};

use gpui::{AppContext, ClipboardItem, Context, Task};
use starling_dictation::settings::InsertionSettings;
use starling_insertion::{FieldAnchor, InsertError, Inserter, Surrounding, TargetSnapshot};
use starling_processing::boundary::{self, BoundaryChange, BoundaryContext, BoundaryOptions};
use starling_processing::contract::{Behavior, ModeEntry};

use crate::app::StarlingApp;
use crate::overlay::DeliveryStatus;

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
    /// The target's focused field, once the capture worker located it
    /// (after the capture's answer, so never within its budget).
    pub field: FieldSlot,
    /// The mode the take was dictated in.
    pub mode: &'static ModeEntry,
}

/// Where a capture's focused field is, set by the capture worker.
pub(crate) type FieldSlot = Arc<std::sync::OnceLock<FieldAnchor>>;

/// What the insertion-boundary rules made of a delivered text (#341).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Adjusted {
    pub text: String,
    /// The contract kinds of the rules that fired.
    pub changes: Vec<String>,
}

/// The text to type for `text` at `target`: adjusted by the
/// insertion-boundary rules where the field located at capture reports
/// its text before the caret. Nothing is read for a verbatim delivery.
/// Blocking (a field read is an accessibility-bus round trip).
fn boundary_text(
    inserter: &Inserter,
    target: &TargetSnapshot,
    field: &FieldSlot,
    verbatim: bool,
    text: &str,
) -> Option<Adjusted> {
    if verbatim {
        return None;
    }
    let Surrounding::Text(surrounding) = inserter.surrounding_text(target, field.get()) else {
        return None;
    };
    let context = BoundaryContext {
        before: &surrounding.before,
        after: &surrounding.after,
        showing_hint: false,
    };
    let adjustment = boundary::adjust(text, &context, &BoundaryOptions::default());
    (!adjustment.is_unchanged()).then(|| Adjusted {
        text: adjustment.text,
        changes: adjustment
            .changes
            .iter()
            .map(|change| match change {
                BoundaryChange::LeadingSpace => "leading_space".to_string(),
                BoundaryChange::FirstLetterCase => "first_letter_case".to_string(),
            })
            .collect(),
    })
}

/// Clears a worker's "out" flag when the worker ends, also by a panic:
/// a flag left set would refuse every later worker.
struct OutGuard(Arc<AtomicBool>);

impl Drop for OutGuard {
    fn drop(&mut self) {
        self.0.store(false, Ordering::SeqCst);
    }
}

/// How long a capture worker the display has not answered keeps later
/// captures waiting. Past it the worker is abandoned (its answer would be
/// refused by the cutoff anyway) and the next capture tries a fresh
/// connection.
const STUCK_CAPTURE: Duration = Duration::from_secs(5);
/// At most this many abandoned capture workers are ever out: a display
/// that answers no connection at all does not cost a thread per take.
const MAX_ABANDONED_CAPTURES: usize = 2;

/// The capture workers out (see [`bounded_capture`]).
#[derive(Default)]
pub(crate) struct CaptureWorkers {
    state: Mutex<Workers>,
}

#[derive(Default)]
struct Workers {
    next: u64,
    /// The worker the latest capture waited for, and when it started.
    current: Option<(u64, Instant)>,
    abandoned: Vec<u64>,
}

impl CaptureWorkers {
    /// Registers a new worker, or says why none may start: the current
    /// one is younger than `stuck_after`, or too many were abandoned.
    fn start(&self, stuck_after: Duration) -> Option<u64> {
        let mut workers = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some((id, started)) = workers.current {
            if started.elapsed() < stuck_after || workers.abandoned.len() >= MAX_ABANDONED_CAPTURES
            {
                return None;
            }
            workers.abandoned.push(id);
        }
        workers.next += 1;
        let id = workers.next;
        workers.current = Some((id, Instant::now()));
        Some(id)
    }

    fn finish(&self, id: u64) {
        let mut workers = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        if workers.current.is_some_and(|(current, _)| current == id) {
            workers.current = None;
        }
        workers.abandoned.retain(|&abandoned| abandoned != id);
    }
}

/// Ends a capture worker's registration when it ends, also by a panic: a
/// worker left registered would refuse later ones.
struct CaptureGuard(Arc<CaptureWorkers>, u64);

impl Drop for CaptureGuard {
    fn drop(&mut self) {
        self.0.finish(self.1);
    }
}

/// The inserter's capture on a worker thread, waited for at most
/// [`CAPTURE_BUDGET`]: a stuck display cannot freeze the UI thread, and an
/// answer that comes later is not taken (focus may have moved by then).
/// One the display never answered is not joined by more until it is
/// [`STUCK_CAPTURE`] old. The cutoff is absolute: an answer the worker
/// had after it is refused even when the UI thread, itself late, finds it
/// already waiting.
///
/// With `locate`, the worker then locates the target's focused field
/// into the returned slot, after its answer: a slow accessibility bus
/// never delays or fails the capture.
fn bounded_capture(
    inserter: Arc<Inserter>,
    workers: &Arc<CaptureWorkers>,
    locate: bool,
) -> (Result<TargetSnapshot, InsertError>, FieldSlot) {
    let field = FieldSlot::default();
    let slot = locate.then(|| field.clone());
    let target = bounded_capture_after(inserter, workers, STUCK_CAPTURE, || {}, slot);
    (target, field)
}

/// [`bounded_capture`] with the stuck-worker limit given, running `stall`
/// between starting the worker and waiting for it: tests stand in a
/// descheduled UI thread with it.
fn bounded_capture_after(
    inserter: Arc<Inserter>,
    workers: &Arc<CaptureWorkers>,
    stuck_after: Duration,
    stall: impl FnOnce(),
    locate: Option<FieldSlot>,
) -> Result<TargetSnapshot, InsertError> {
    let deadline = Instant::now() + CAPTURE_BUDGET;
    let Some(id) = workers.start(stuck_after) else {
        return Err(InsertError::Unavailable {
            reason: "the display has not answered an earlier focus check".to_string(),
        });
    };
    let (sender, receiver) = mpsc::channel();
    let guard = CaptureGuard(workers.clone(), id);
    let spawned = std::thread::Builder::new()
        .name("starling-capture".into())
        .spawn(move || {
            let target = inserter.capture();
            let answered = Instant::now();
            // Finished before the answer, so the next capture never sees
            // an answered worker as out.
            drop(guard);
            let located = target.as_ref().ok().cloned();
            let _ = sender.send((target, answered));
            if let (Some(slot), Some(target)) = (locate, located) {
                if let Some(field) = inserter.locate_field(&target) {
                    let _ = slot.set(field);
                }
            }
        });
    if let Err(error) = spawned {
        // The closure, and the guard in it, are dropped with the error.
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
    /// An earlier insert given up as stalled has still not returned:
    /// nothing more is typed until it does.
    EarlierTypingStuck,
    /// The target cannot be verified here and the user has not opted in.
    UnverifiedOff,
    /// The target cannot be verified, and Starling's own window had focus
    /// since the capture: the original target cannot be assumed.
    FocusMovedThroughStarling,
    /// The text is a retried take's (#356): it is never typed by itself
    /// — the editor the take was dictated into may have moved on.
    Retried,
}

impl Failure {
    /// The notice's title, and the overlay's line under its failure.
    pub(crate) fn title(&self) -> &'static str {
        match self {
            Failure::Insert(InsertError::PartialDelivery { .. }) => "Inserted only in part",
            Failure::Insert(InsertError::TargetChanged { .. })
            | Failure::FocusMovedThroughStarling => "Not inserted: focus moved",
            Failure::Insert(InsertError::TargetGone) => "Not inserted: the window closed",
            Failure::Insert(InsertError::ModifiersHeld { .. }) => "Not inserted: keys were held",
            Failure::Insert(InsertError::Unavailable { .. }) => "Insertion unavailable",
            Failure::Stalled => "Not inserted: typing stalled",
            Failure::EarlierTypingStuck => "Not inserted: earlier typing is stuck",
            Failure::UnverifiedOff => "Not inserted: the target cannot be checked",
            Failure::Retried => "Retried: not typed anywhere",
            Failure::Insert(_) => "Not inserted",
        }
    }
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
    /// The mode the take was dictated in: Paste last follows its
    /// verbatim flag too.
    pub mode: &'static ModeEntry,
    pub failure: Failure,
    /// Paste last is armed: the next focus away from Starling receives
    /// the text. The instant tells one arming from the next.
    pub armed: Option<Instant>,
    /// A Paste last insert is running.
    pub pasting: bool,
    pub copied: bool,
}

impl Recovery {
    fn new(take_id: &str, text: String, mode: &'static ModeEntry, failure: Failure) -> Recovery {
        Recovery {
            take_id: take_id.to_string(),
            text,
            mode,
            failure,
            armed: None,
            pasting: false,
            copied: false,
        }
    }

    pub(crate) fn title(&self) -> &'static str {
        self.failure.title()
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
            Failure::Retried => {
                "This transcript comes from a retry, so it was not typed anywhere: the window \
                 you dictated into may hold other text by now. Copy it, or press Paste last \
                 and click into the field it belongs in."
                    .to_string()
            }
            Failure::EarlierTypingStuck => {
                "Typing from an earlier insert is still stuck waiting for the display, so \
                 nothing was typed. Copy the text; if this keeps happening, restart Starling."
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
    /// The capture workers out (see [`bounded_capture`]).
    capture_out: Arc<CaptureWorkers>,
    /// The overlay window has focus. It never should, but on Wayland the
    /// compositor decides (see `overlay.rs`), so delivery counts it as
    /// Starling's own window. `overlay_gained` is when it last took focus.
    overlay_active: bool,
    overlay_gained: Option<Instant>,
    /// Inserts started and not finished, the first failure among them,
    /// and whether one landed: the overlay says "Inserting…" until the
    /// last one ends.
    inserts_pending: usize,
    batch_failure: Option<&'static str>,
    batch_delivered: bool,
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
            overlay_active: false,
            overlay_gained: None,
            inserts_pending: 0,
            batch_failure: None,
            batch_delivered: false,
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
    /// Whether Starling's main window or its overlay has focus now.
    fn starling_focused(&self) -> bool {
        self.window_focus.last().is_some_and(|(_, active)| *active) || self.delivery.overlay_active
    }

    /// Whether Starling's main window or its overlay had focus at any
    /// point since `at`.
    fn starling_focused_since(&self, at: Instant) -> bool {
        self.starling_focused()
            || self
                .window_focus
                .iter()
                .any(|(changed, active)| *active && *changed >= at)
            || self
                .delivery
                .overlay_gained
                .is_some_and(|gained| gained >= at)
    }

    /// Captures the focused target now, for a take dictated in `mode`.
    /// Starling's own focus is decided here: dictating into Starling
    /// itself. On Wayland only the app can tell; elsewhere the backend
    /// refuses Starling's pid too. The focused field is located only
    /// where the boundary rules can apply (not in a verbatim mode).
    fn capture_now(&self, mode: &'static ModeEntry) -> Capture {
        let at = Instant::now();
        let (target, field) = if self.starling_focused() {
            (Err(InsertError::TargetIsStarling), FieldSlot::default())
        } else {
            let locate = mode.behavior != Behavior::Verbatim;
            bounded_capture(self.delivery.inserter.clone(), &self.delivery.capture_out, locate)
        };
        Capture {
            target,
            at,
            field,
            mode,
        }
    }

    /// An insert started: the overlay says so.
    fn overlay_insert_started(&mut self, cx: &mut Context<Self>) {
        self.delivery.inserts_pending += 1;
        self.set_delivery_status(DeliveryStatus::Delivering, cx);
    }

    /// An insert ended (`started`), or a delivery failed before typing.
    /// The overlay shows the outcome once no insert is left running; a
    /// failure among them outranks the successes. `outcome` is `None` for
    /// a Paste last whose notice was dismissed or replaced meanwhile: it
    /// shows nowhere, so with nothing else to show the overlay goes idle.
    fn overlay_delivery_ended(
        &mut self,
        started: bool,
        outcome: Option<Result<(), &Failure>>,
        cx: &mut Context<Self>,
    ) {
        if started {
            self.delivery.inserts_pending = self.delivery.inserts_pending.saturating_sub(1);
        }
        match outcome {
            Some(Ok(())) => self.delivery.batch_delivered = true,
            Some(Err(failure)) => {
                self.delivery.batch_failure.get_or_insert(failure.title());
            }
            None => {}
        }
        if self.delivery.inserts_pending > 0 {
            return;
        }
        let failure = self.delivery.batch_failure.take();
        let delivered = std::mem::take(&mut self.delivery.batch_delivered);
        // A pressed Insert still waits: the switch it asks for comes first.
        let status = if self.delivery.staged_armed.is_some() {
            DeliveryStatus::Waiting
        } else if let Some(reason) = failure {
            DeliveryStatus::Failed(reason.to_string())
        } else if delivered {
            DeliveryStatus::Delivered
        } else {
            DeliveryStatus::Idle
        };
        self.set_delivery_status(status, cx);
    }

    /// A pressed Insert stopped waiting: the overlay no longer asks for
    /// the switch (the next activation poll shows it).
    fn overlay_insert_unarmed(&mut self) {
        if *self.overlay.model.delivery() == DeliveryStatus::Waiting {
            self.overlay
                .model
                .set_delivery(DeliveryStatus::Idle, Instant::now());
        }
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
        let mode = self.active_mode();
        self.delivery.live = self
            .delivery
            .settings
            .auto_insert
            .then(|| self.capture_now(mode));
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
            self.overlay_insert_unarmed();
        }
    }

    /// A retried take's transcript landed (#356): it is never typed by
    /// itself. With typing on, the notice offers Copy and an explicit
    /// Paste last; with copy-only delivery the drawer's Copy is the way.
    /// A take whose staging panel is open keeps its panel's own Insert.
    /// `text` is exactly the retry's result: the take's history head may
    /// already be a later retry another window asked for.
    pub(crate) fn offer_retried_text(&mut self, id: &str, text: &str, cx: &mut Context<Self>) {
        if !self.delivery.settings.auto_insert || self.staging_shows(id) || text.trim().is_empty() {
            return;
        }
        let mode = self.active_mode();
        self.delivery.replace_recovery(Some(Recovery::new(
            id,
            text.to_string(),
            mode,
            Failure::Retried,
        )));
        cx.notify();
    }

    /// Whether take `id`'s text still goes somewhere when it is ready.
    pub(crate) fn delivers(&self, id: &str) -> bool {
        self.delivery.by_take.contains_key(id)
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
        let Some(text) = self
            .staged_text_for(id)
            .or_else(|| self.own_result(id))
            .or_else(|| self.head_text(id))
        else {
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
                self.overlay_delivery_ended(false, Some(Err(&failure)), cx);
                self.delivery.replace_recovery(Some(Recovery::new(
                    id,
                    text,
                    capture.mode,
                    failure,
                )));
                cx.notify();
            }
            Plan::Insert(target) => {
                self.spawn_insert(id.to_string(), target, text, &capture, None, cx)
            }
        }
    }

    /// Types `text` into `target` off the UI thread, after every insert
    /// started before it finished, adjusted by the insertion-boundary
    /// rules at `capture`'s field unless its mode delivers `text`
    /// verbatim. `paste` is the recovery generation a Paste last runs for;
    /// `None` for the take's own delivery. An insert that overruns its
    /// [`typing_budget`] is reported as stalled and stops before its next
    /// key should it ever resume, so one stuck insert cannot hold up the
    /// ones after it.
    fn spawn_insert(
        &mut self,
        id: String,
        target: TargetSnapshot,
        text: String,
        capture: &Capture,
        paste: Option<u64>,
        cx: &mut Context<Self>,
    ) {
        let inserter = self.delivery.inserter.clone();
        let had_focus = self.delivery.own_focus.had_focus_since_now();
        let typed_text = text.clone();
        let mode = capture.mode;
        let field = capture.field.clone();
        let verbatim = crate::processing::delivers_verbatim(mode, &text);
        let previous = self.delivery.last_insert.take();
        // The adjusted text is at most a space longer.
        let budget = typing_budget(&text) + Duration::from_millis(20);
        let out = self.delivery.insert_out.clone();
        self.overlay_insert_started(cx);
        let insert = cx.spawn(async move |this, cx| {
            if let Some(previous) = previous {
                previous.await;
            }
            // A stalled insert still out holds the insertion lock; typing
            // beside it could interleave keys, so nothing more is typed.
            let result = if out.swap(true, Ordering::SeqCst) {
                Err(Failure::EarlierTypingStuck)
            } else {
                let abandoned = Arc::new(AtomicBool::new(false));
                let stop = insert_stop(had_focus, abandoned.clone());
                let typing = cx.background_spawn(async move {
                    let _worker = OutGuard(out);
                    // Read right before typing: the text before the caret
                    // as it is now, never as it was at the take's start.
                    let adjusted = boundary_text(&inserter, &target, &field, verbatim, &typed_text);
                    let typed = adjusted.as_ref().map_or(typed_text.as_str(), |a| a.text.as_str());
                    inserter
                        .insert(&target, typed, &stop)
                        .map(|receipt| (receipt, adjusted))
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
                app.insert_finished(id, text, mode, paste, result, cx)
            })
            .ok();
        });
        self.delivery.last_insert = Some(insert);
    }

    fn insert_finished(
        &mut self,
        id: String,
        text: String,
        mode: &'static ModeEntry,
        paste: Option<u64>,
        result: Result<(starling_insertion::InsertReceipt, Option<Adjusted>), Failure>,
        cx: &mut Context<Self>,
    ) {
        if let Ok((_, Some(adjusted))) = &result {
            self.record_boundary_revision(&id, &text, adjusted.clone(), cx);
        }
        // A Paste last for a notice dismissed or replaced meanwhile
        // reports nowhere, the overlay included.
        let superseded = paste.is_some_and(|generation| generation != self.delivery.generation);
        let outcome = (!superseded).then(|| result.as_ref().map(drop));
        self.overlay_delivery_ended(true, outcome, cx);
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
                        .replace_recovery(Some(Recovery::new(&id, text, mode, failure)));
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

    /// Stores an adjusted delivery as a revision derived from `source`
    /// (#341). The text is in the field either way; a failed write only
    /// leaves history without the derived revision, and says so.
    fn record_boundary_revision(
        &mut self,
        id: &str,
        source: &str,
        adjusted: Adjusted,
        cx: &mut Context<Self>,
    ) {
        let Some(store) = self.store.clone() else {
            return;
        };
        let (id, source) = (id.to_string(), source.to_string());
        cx.spawn(async move |this, cx| {
            let saved = cx
                .background_spawn(async move {
                    store.record_boundary_revision(&id, &source, &adjusted.text, &adjusted.changes)
                })
                .await;
            match saved {
                // The take was deleted meanwhile: nothing to keep.
                Ok(()) | Err(starling_dictation::storage::StorageError::NotFound(_)) => {}
                Err(err) => {
                    this.update(cx, |app, cx| {
                        app.error = Some(format!(
                            "The text was inserted with its boundary adjusted, but history \
                             could not keep the adjusted version: {err}"
                        ));
                        cx.notify();
                    })
                    .ok();
                }
            }
        })
        .detach();
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
        self.set_delivery_status(DeliveryStatus::Waiting, cx);
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
            self.overlay_insert_unarmed();
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
        self.overlay_insert_unarmed();
        // The draft, else exactly the take's own result: never whatever a
        // later retry put in history.
        let Some(text) = self
            .staged_text_for(&id)
            .or_else(|| self.own_result(&id))
            .or_else(|| self.head_text(&id))
        else {
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
            self.capture_now(capture.mode)
        } else {
            capture
        };
        let verified = self.verifiable(&capture);
        match plan(&capture, self.delivery.settings, verified, false, &text) {
            Plan::Skip => {}
            Plan::Fail(failure) => {
                self.overlay_delivery_ended(false, Some(Err(&failure)), cx);
                self.delivery.replace_recovery(Some(Recovery::new(
                    &id,
                    text,
                    capture.mode,
                    failure,
                )));
            }
            Plan::Insert(target) => {
                self.delivery.staged_inserts.insert(id.clone(), false);
                self.spawn_insert(id, target, text, &capture, None, cx);
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
        self.delivery_focus_changed(active || self.delivery.overlay_active, cx);
    }

    /// The overlay window gained or lost focus (a compositor may give it
    /// focus on Wayland); it counts as Starling's own window, so focus
    /// moving there never fires an armed Insert or Paste last into it.
    pub(crate) fn delivery_overlay_activation(&mut self, active: bool, cx: &mut Context<Self>) {
        if active == self.delivery.overlay_active {
            return;
        }
        self.delivery.overlay_active = active;
        if active {
            self.delivery.overlay_gained = Some(Instant::now());
        }
        let main = self.window_focus.last().is_some_and(|(_, active)| *active);
        self.delivery_focus_changed(active || main, cx);
    }

    /// Starling's windows together gained (`active`) or lost focus.
    fn delivery_focus_changed(&mut self, active: bool, cx: &mut Context<Self>) {
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
        let Some(mode) = self.delivery.recovery.as_ref().map(|recovery| recovery.mode) else {
            return;
        };
        let capture = self.capture_now(mode);
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
                self.spawn_insert(id, target, text, &capture, Some(generation), cx);
            }
            Plan::Skip => {
                let failure = Failure::Insert(InsertError::TargetIsStarling);
                self.overlay_delivery_ended(false, Some(Err(&failure)), cx);
                if let Some(recovery) = self.delivery.recovery.as_mut() {
                    recovery.failure = failure;
                }
            }
            Plan::Fail(failure) => {
                self.overlay_delivery_ended(false, Some(Err(&failure)), cx);
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
    use starling_insertion::testing::{FakeBackend, FakeFields, FakeTarget, InsertBehavior};

    fn on() -> InsertionSettings {
        InsertionSettings::default()
    }

    fn captured(target: Result<TargetSnapshot, InsertError>) -> Capture {
        Capture {
            target,
            at: Instant::now(),
            field: FieldSlot::default(),
            mode: verbatim(),
        }
    }

    fn verbatim() -> &'static ModeEntry {
        crate::processing::mode("verbatim")
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
            interrupted: false,
            confirmed_ms: None,
            results: Vec::new(),
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

    /// The overlay follows delivery: "Inserting…" until the last of
    /// overlapping inserts ends, then the outcome; a failure says why.
    #[gpui::test]
    fn the_overlay_follows_delivery(cx: &mut gpui::TestAppContext) {
        let (app, fake) = app_with(cx, on());
        fake.focus(FakeTarget::named("Editor", "notes.txt"));
        let status = |app: &gpui::Entity<StarlingApp>, cx: &mut gpui::TestAppContext| {
            app.read_with(cx, |app, _| app.overlay.model.delivery().clone())
        };
        app.update(cx, |app, cx| {
            for (id, text) in [("take-1", "First."), ("take-2", "Second.")] {
                app.delivery_take_started();
                let capture = app.delivery_take_stopped();
                app.bind_delivery(capture, id);
                app.sessions.push(session(id, text));
            }
            app.deliver_finished_take("take-1", cx);
            app.deliver_finished_take("take-2", cx);
        });
        assert_eq!(status(&app, cx), DeliveryStatus::Delivering);
        cx.run_until_parked();
        assert_eq!(fake.field(), "First.Second.");
        assert_eq!(status(&app, cx), DeliveryStatus::Delivered);

        take(&app, cx, "take-3", "Third.", |_| {
            fake.focus(FakeTarget::named("Browser", "A tab"));
        });
        assert_eq!(
            status(&app, cx),
            DeliveryStatus::Failed("Not inserted: focus moved".into())
        );
    }

    /// A pressed Insert keeps the overlay asking for the switch when an
    /// earlier insert ends meanwhile.
    #[gpui::test]
    fn a_waiting_insert_outlasts_an_earlier_inserts_outcome(cx: &mut gpui::TestAppContext) {
        let (app, fake) = app_with(cx, on());
        fake.focus(FakeTarget::named("Editor", "notes.txt"));
        app.update(cx, |app, cx| {
            app.delivery_take_started();
            let capture = app.delivery_take_stopped();
            app.bind_delivery(capture, "take-1");
            app.sessions.push(session("take-1", "First."));
            app.deliver_finished_take("take-1", cx);
            app.delivery.staged_armed = Some(ArmedInsert {
                take_id: "take-2".into(),
                capture: captured(Err(InsertError::TargetGone)),
                armed_at: Instant::now(),
            });
            app.set_delivery_status(DeliveryStatus::Waiting, cx);
        });
        cx.run_until_parked();
        assert_eq!(fake.field(), "First.");
        app.read_with(cx, |app, _| {
            assert_eq!(app.overlay.model.delivery(), &DeliveryStatus::Waiting);
        });
    }

    /// Should a compositor give the overlay focus, it counts as
    /// Starling's own window: a Wayland take is not typed after it, and
    /// an armed Paste last waits for focus to leave it too.
    #[gpui::test]
    fn focus_on_the_overlay_is_starlings_own(cx: &mut gpui::TestAppContext) {
        let opted_in = InsertionSettings {
            allow_unverified: true,
            ..on()
        };
        let (app, fake) = app_with(cx, opted_in);
        fake.set_verifies_target(false);
        fake.focus(FakeTarget::named("Editor", "notes.txt"));
        app.update(cx, |app, cx| {
            app.delivery_take_started();
            let capture = app.delivery_take_stopped();
            app.bind_delivery(capture, "take-1");
            app.delivery_overlay_activation(true, cx);
            app.delivery_overlay_activation(false, cx);
            app.sessions.push(session("take-1", "Hello there."));
            app.deliver_finished_take("take-1", cx);
        });
        cx.run_until_parked();
        assert!(fake.insertions().is_empty());
        assert_eq!(failure(&app, cx), Some(Failure::FocusMovedThroughStarling));

        // Paste last, armed in the main window, which hands focus to the
        // overlay: nothing is typed into the overlay.
        app.update(cx, |app, cx| {
            app.window_focus.push((Instant::now(), true));
            app.toggle_paste_last(cx);
            app.window_focus.push((Instant::now(), false));
            app.delivery_window_activation(false, cx);
            app.delivery_overlay_activation(true, cx);
        });
        cx.executor().advance_clock(PASTE_SETTLE * 2);
        cx.run_until_parked();
        assert!(fake.insertions().is_empty());
        app.read_with(cx, |app, _| {
            assert!(app.delivery.recovery.as_ref().unwrap().armed.is_some());
        });

        fake.focus(FakeTarget::named("Chat", "Message"));
        app.update(cx, |app, cx| app.delivery_overlay_activation(false, cx));
        cx.executor().advance_clock(PASTE_SETTLE);
        cx.run_until_parked();
        assert_eq!(fake.field(), "Hello there.");

        // An overlay focus before a take's capture says nothing about it.
        take(&app, cx, "take-2", " Again.", |_| {});
        assert_eq!(fake.field(), "Hello there. Again.");
    }

    /// An overlay window that goes away without reporting its focus loss
    /// (a compositor closed it) no longer counts as Starling's focus.
    #[gpui::test]
    fn a_closed_overlay_stops_counting_as_focus(cx: &mut gpui::TestAppContext) {
        let (app, _fake) = app_with(cx, on());
        let weak = app.downgrade();
        let overlay =
            cx.add_window(|window, cx| crate::views::overlay::OverlayView::new(weak, window, cx));
        app.update(cx, |app, cx| app.delivery_overlay_activation(true, cx));
        app.read_with(cx, |app, _| assert!(app.starling_focused()));
        overlay
            .update(cx, |_, window, _| window.remove_window())
            .unwrap();
        cx.run_until_parked();
        app.read_with(cx, |app, _| assert!(!app.starling_focused()));
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

    /// A test app with a store, a fake backend focused on an editor, a
    /// scripted field reader, and the active mode `mode`.
    fn app_with_fields(
        cx: &mut gpui::TestAppContext,
        store: Option<crate::store::Store>,
        mode: &str,
    ) -> (gpui::Entity<StarlingApp>, Arc<FakeBackend>, Arc<FakeFields>) {
        let fake = Arc::new(FakeBackend::new());
        fake.focus(FakeTarget::named("Editor", "notes.txt"));
        let fields = Arc::new(FakeFields::new());
        let inserter = Inserter::with_backends(vec![Box::new(fake.clone())])
            .with_field_reader(Box::new(fields.clone()));
        let mode = mode.to_string();
        let app = cx.new(|cx| {
            let mut app = StarlingApp::for_test(store, cx);
            app.delivery = DeliveryState::new(Arc::new(inserter), on());
            app.processing_settings.mode = mode;
            app
        });
        (app, fake, fields)
    }

    /// A stored take whose transcript is `text`; its id.
    fn stored_take(store: &crate::store::Store, text: &str) -> String {
        use starling_dictation::{audio, storage::TranscriptionResult};
        let wav = audio::encode_wav_16k(&audio::PcmAudio {
            samples: vec![0.; 160],
            sample_rate: 16_000,
            channels: 1,
        })
        .unwrap();
        let id = store.save_capture(Arc::new(wav)).unwrap().id;
        store.mark_attempt(&id, "test").unwrap();
        store
            .save_transcript(
                &id,
                TranscriptionResult {
                    text: text.into(),
                    segments: vec![],
                    duration_seconds: None,
                    request_id: None,
                },
            )
            .unwrap();
        id
    }

    /// Waits for the capture worker of take `id` to have located its
    /// field (it does so after the capture's answer, off the UI thread).
    fn field_located(app: &gpui::Entity<StarlingApp>, cx: &mut gpui::TestAppContext, id: &str) {
        let slot = app.read_with(cx, |app, _| app.delivery.by_take[id].field.clone());
        let deadline = Instant::now() + Duration::from_secs(5);
        while slot.get().is_none() {
            assert!(Instant::now() < deadline, "the field was never located");
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    /// One take through the app's hooks, delivered once its field is
    /// located (`located`) or right away.
    fn located_take(
        app: &gpui::Entity<StarlingApp>,
        cx: &mut gpui::TestAppContext,
        id: &str,
        text: &str,
        located: bool,
        between: impl FnOnce(),
    ) {
        app.update(cx, |app, _| {
            app.delivery_take_started();
            let capture = app.delivery_take_stopped();
            app.bind_delivery(capture, id);
            app.sessions.push(session(id, text));
        });
        if located {
            field_located(app, cx, id);
        }
        between();
        app.update(cx, |app, cx| app.deliver_finished_take(id, cx));
        cx.run_until_parked();
    }

    /// #341 end to end: the field the take started in is read right
    /// before typing, the boundary rules adjust the text typed, and the
    /// adjustment is stored as a revision derived from the transcript,
    /// which stays the take's text.
    #[gpui::test]
    fn the_boundary_rules_adjust_the_typed_text_and_history_keeps_both(
        cx: &mut gpui::TestAppContext,
    ) {
        let root = tempfile::tempdir().unwrap();
        let store = crate::store::Store::at_test_root(root.path());
        let id = stored_take(&store, "Fox jumps.");
        let (app, fake, fields) = app_with_fields(cx, Some(store.clone()), "clean-local");
        let target = fake.capture().unwrap();
        fields.focus_text("The quick");
        // The user typed on while the take was transcribed: the text as
        // it is right before typing counts.
        located_take(&app, cx, &id, "Fox jumps.", true, || {
            fields.set_before("The quick brown")
        });
        assert_eq!(
            fake.insertions(),
            vec![(target.target_ref, " fox jumps.".to_string())]
        );
        assert_eq!(failure(&app, cx), None);

        let doc = store.processing_doc(&id).unwrap().expect("a processing document");
        assert_eq!(doc.head_text, "Fox jumps.", "the transcript stays the head");
        assert_eq!(
            doc.boundary,
            vec![starling_runtime_host::history::BoundaryRevision {
                text: " fox jumps.".to_string(),
                source_text: "Fox jumps.".to_string(),
                derived_from: Some(1),
                changes: vec!["leading_space".into(), "first_letter_case".into()],
            }]
        );
        assert_eq!(app.read_with(cx, |app, _| app.head_text(&id)).as_deref(), Some("Fox jumps."));
    }

    /// A verbatim mode, a spoken override to one, a password field, a
    /// field that lost focus, and a field nothing could locate: the text
    /// goes in as dictated and nothing is recorded.
    #[gpui::test]
    fn the_text_goes_in_as_dictated_where_the_rules_do_not_apply(cx: &mut gpui::TestAppContext) {
        let root = tempfile::tempdir().unwrap();
        let store = crate::store::Store::at_test_root(root.path());

        // The verbatim mode never looks for the field.
        let (app, fake, fields) = app_with_fields(cx, Some(store.clone()), "verbatim");
        fields.focus_text("The quick brown");
        let id = stored_take(&store, "Fox jumps.");
        located_take(&app, cx, &id, "Fox jumps.", false, || {});
        assert_eq!(fields.locates(), 0);
        assert_eq!(fields.text_reads(), 0);

        // A spoken override routing the take to the verbatim mode.
        let (app, fake2, fields) = app_with_fields(cx, Some(store.clone()), "clean-local");
        fields.focus_text("The quick brown");
        let spoken = stored_take(&store, "literal Fox jumps.");
        located_take(&app, cx, &spoken, "literal Fox jumps.", true, || {});
        assert_eq!(fields.text_reads(), 0);

        // A password field is located but never read.
        let (app, fake3, fields) = app_with_fields(cx, Some(store.clone()), "clean-local");
        fields.focus_password();
        let password = stored_take(&store, "Fox jumps.");
        located_take(&app, cx, &password, "Fox jumps.", true, || {});
        assert_eq!(fields.text_reads(), 0);

        // Another field took focus since the take started.
        let (app, fake4, fields) = app_with_fields(cx, Some(store.clone()), "clean-local");
        fields.focus_text("The quick brown");
        let moved = stored_take(&store, "Fox jumps.");
        located_take(&app, cx, &moved, "Fox jumps.", true, || {
            fields.focus_text("Elsewhere")
        });
        assert_eq!(fields.text_reads(), 0);

        // Nothing focused to locate at the take's start.
        let (app, fake5, fields) = app_with_fields(cx, Some(store.clone()), "clean-local");
        let unlocated = stored_take(&store, "Fox jumps.");
        located_take(&app, cx, &unlocated, "Fox jumps.", false, || {
            fields.focus_text("The quick brown")
        });
        assert_eq!(fields.text_reads(), 0);

        for (fake, text) in [
            (fake, "Fox jumps."),
            (fake2, "literal Fox jumps."),
            (fake3, "Fox jumps."),
            (fake4, "Fox jumps."),
            (fake5, "Fox jumps."),
        ] {
            assert_eq!(fake.insertions().len(), 1);
            assert_eq!(fake.insertions()[0].1, text);
        }
        for id in [id, spoken, password, moved, unlocated] {
            let boundary = store
                .processing_doc(&id)
                .unwrap()
                .map(|doc| doc.boundary)
                .unwrap_or_default();
            assert!(boundary.is_empty(), "{id}: {boundary:?}");
        }
        let _ = app;
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

    /// While an insert given up as stalled is still out, later ones type
    /// nothing and say why, with the way out.
    #[gpui::test]
    fn a_stuck_insert_still_out_is_explained(cx: &mut gpui::TestAppContext) {
        let (app, fake) = app_with(cx, on());
        fake.focus(FakeTarget::named("Editor", "notes.txt"));
        app.update(cx, |app, _| {
            app.delivery.insert_out.store(true, Ordering::SeqCst)
        });
        take(&app, cx, "take-1", "Hello there.", |_| {});
        assert!(fake.insertions().is_empty());
        app.read_with(cx, |app, _| {
            let recovery = app.delivery.recovery.as_ref().unwrap();
            assert_eq!(recovery.failure, Failure::EarlierTypingStuck);
            assert_eq!(recovery.title(), "Not inserted: earlier typing is stuck");
            assert!(recovery.explanation().contains("restart Starling"));
            assert!(!recovery.offers_settings());
        });
    }

    /// A repeated focus-loss report does not strand an armed Paste last:
    /// it fires one full settle after the last report.
    #[gpui::test]
    fn a_repeated_focus_loss_still_fires_paste_last(cx: &mut gpui::TestAppContext) {
        let (app, fake) = app_with(cx, on());
        fake.focus(FakeTarget::named("Editor", "notes.txt"));
        take(&app, cx, "take-1", "Hello there.", |_| {
            fake.destroy_target()
        });
        fake.focus(FakeTarget::named("Chat", "Message"));
        app.update(cx, |app, cx| {
            app.window_focus.push((Instant::now(), true));
            app.toggle_paste_last(cx);
            app.window_focus.push((Instant::now(), false));
            app.delivery_window_activation(false, cx);
        });
        cx.executor().advance_clock(PASTE_SETTLE / 2);
        app.update(cx, |app, cx| app.delivery_window_activation(false, cx));
        cx.executor().advance_clock(PASTE_SETTLE);
        cx.run_until_parked();
        assert_eq!(fake.field(), "Hello there.");
    }

    /// A Paste last whose notice was dismissed while it typed shows its
    /// outcome nowhere: the overlay goes idle rather than flash it.
    #[gpui::test]
    fn a_dismissed_paste_shows_no_outcome_in_the_overlay(cx: &mut gpui::TestAppContext) {
        let (app, fake) = app_with(cx, on());
        fake.focus(FakeTarget::named("Editor", "notes.txt"));
        take(&app, cx, "take-1", "Hello there.", |_| {
            fake.destroy_target()
        });
        app.update(cx, |app, cx| {
            // A Paste last is typing when its notice is dismissed.
            let generation = app.delivery.generation;
            app.overlay_insert_started(cx);
            app.dismiss_recovery(cx);
            app.insert_finished(
                "take-1".into(),
                "Hello there.".into(),
                verbatim(),
                Some(generation),
                Err(Failure::Insert(InsertError::TargetGone)),
                cx,
            );
            assert_eq!(app.overlay.model.delivery(), &DeliveryStatus::Idle);
        });
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
                verbatim(),
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
        let notice = Recovery::new("take-1", "Hello there.".into(), verbatim(), Failure::Stalled);
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
        let out = Arc::default();
        let descheduled = || std::thread::sleep(CAPTURE_BUDGET * 3);
        let prompt =
            bounded_capture_after(inserter.clone(), &out, STUCK_CAPTURE, descheduled, None);
        assert!(prompt.is_ok(), "{prompt:?}");
        fake.set_capture_delay(CAPTURE_BUDGET * 3 / 2);
        let late = bounded_capture_after(inserter, &out, STUCK_CAPTURE, descheduled, None);
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
        let out = Arc::default();
        fake.set_capture_panics(true);
        let (panicked, _) = bounded_capture(inserter.clone(), &out, false);
        assert!(
            matches!(panicked, Err(InsertError::Unavailable { .. })),
            "{panicked:?}"
        );
        fake.set_capture_panics(false);
        let (next, _) = bounded_capture(inserter, &out, false);
        assert!(next.is_ok(), "{next:?}");
    }

    /// A capture worker the display never answers holds later captures
    /// off only until it is stuck: then a fresh one may try, while the
    /// number of abandoned workers stays bounded.
    #[test]
    fn a_stuck_capture_worker_is_abandoned_within_a_bound() {
        let (fake, inserter) = fake_session();
        fake.focus(FakeTarget::named("Editor", "notes.txt"));
        let stuck = CAPTURE_BUDGET * 2;
        let capture = |out: &Arc<CaptureWorkers>| {
            bounded_capture_after(inserter.clone(), out, stuck, || {}, None)
        };
        let refused = |result: Result<TargetSnapshot, InsertError>| {
            matches!(&result, Err(InsertError::Unavailable { reason })
                if reason.contains("earlier focus check"))
        };

        let out = Arc::default();
        fake.set_capture_delay(Duration::from_secs(2));
        assert!(capture(&out).is_err());
        // Still young: the next capture waits for it.
        assert!(refused(capture(&out)));
        // Stuck: abandoned, and a fresh worker answers.
        std::thread::sleep(stuck);
        fake.set_capture_delay(Duration::ZERO);
        assert!(capture(&out).is_ok());

        // A display that answers nothing costs a bounded number of threads.
        let out = Arc::default();
        fake.set_capture_delay(Duration::from_secs(2));
        for _ in 0..=MAX_ABANDONED_CAPTURES {
            assert!(!refused(capture(&out)));
            std::thread::sleep(stuck);
        }
        assert!(refused(capture(&out)));
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
