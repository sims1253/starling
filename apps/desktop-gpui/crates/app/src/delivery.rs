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
//!   to) reads nothing, and a field that cannot be read (or not within
//!   [`READ_BUDGET`], which is over before typing starts) gets the text
//!   as dictated. An adjusted delivery is recorded as a revision derived
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
use std::sync::{Arc, Condvar, Mutex, PoisonError, mpsc};
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
    /// The target's focused field, once the capture worker looked for it
    /// (after the capture's answer, so never within its budget).
    pub field: FieldSlot,
    /// The mode the take was dictated in.
    pub mode: &'static ModeEntry,
}

/// Where a capture's focused field is: set by the capture worker once it
/// looked, `None` when it found none (or not in time).
#[derive(Clone)]
pub(crate) struct FieldSlot(Arc<FieldCell>);

struct FieldCell {
    /// `None` while the lookup runs.
    field: Mutex<Option<Option<FieldAnchor>>>,
    set: Condvar,
    /// After it, no lookup result can still be taken: the capture's
    /// answer came within [`CAPTURE_BUDGET`] of the slot, and a field
    /// within [`LOCATE_BUDGET`] of that answer.
    deadline: Instant,
}

impl Default for FieldSlot {
    fn default() -> FieldSlot {
        FieldSlot(Arc::new(FieldCell {
            field: Mutex::new(None),
            set: Condvar::new(),
            deadline: Instant::now() + CAPTURE_BUDGET + LOCATE_BUDGET,
        }))
    }
}

impl std::fmt::Debug for FieldSlot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("FieldSlot").field(&self.lock()).finish()
    }
}

impl FieldSlot {
    fn lock(&self) -> std::sync::MutexGuard<'_, Option<Option<FieldAnchor>>> {
        self.0.field.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The lookup ended; the first answer stands.
    fn set(&self, field: Option<FieldAnchor>) {
        let mut slot = self.lock();
        if slot.is_none() {
            *slot = Some(field);
            self.0.set.notify_all();
        }
    }

    /// The field located, waiting for a lookup still running until its
    /// result could no longer be taken (a Paste last types right after
    /// its capture). Blocking.
    fn wait(&self) -> Option<FieldAnchor> {
        let mut slot = self.lock();
        while slot.is_none() {
            let left = self.0.deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return None;
            }
            slot = self
                .0
                .set
                .wait_timeout(slot, left)
                .unwrap_or_else(PoisonError::into_inner)
                .0;
        }
        slot.clone().flatten()
    }

}

/// How long after the capture's answer the field may be located: focus
/// moving to another field of the same window in between could not be
/// told apart, so a later answer is not taken.
const LOCATE_BUDGET: Duration = Duration::from_millis(250);

/// How long typing waits for the field's text before the caret. A field
/// that does not answer by then gets the text as dictated; the read is
/// over before the typing budget starts.
const READ_BUDGET: Duration = Duration::from_millis(200);

/// What the insertion-boundary rules made of a delivered text (#341).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Adjusted {
    pub text: String,
    /// The contract kinds of the rules that fired.
    pub changes: Vec<String>,
}

/// The text to type for `text` at `target`: adjusted by the
/// insertion-boundary rules where the field located at capture reports
/// its text before the caret. Only that field is read: without one,
/// nothing is. Blocking: waits for a lookup still running (see
/// [`FieldSlot::wait`]), then at most [`READ_BUDGET`] for the read on
/// its own thread; one read out at a time (`reading`), so a field that
/// stopped answering costs one thread and later reads are skipped
/// meanwhile.
fn boundary_text(
    inserter: Arc<Inserter>,
    target: TargetSnapshot,
    field: &FieldSlot,
    reading: Arc<AtomicBool>,
    text: String,
) -> Option<Adjusted> {
    let anchor = field.wait()?;
    if reading.swap(true, Ordering::SeqCst) {
        return None;
    }
    // Made before the thread, so a thread that never starts clears it too.
    let read = OutGuard(reading);
    let (sender, receiver) = mpsc::channel();
    let spawned = std::thread::Builder::new()
        .name("starling-boundary".into())
        .spawn(move || {
            let _read = read;
            let _ = sender.send(adjust_at(&inserter, &target, &anchor, &text));
        });
    if spawned.is_err() {
        return None;
    }
    receiver.recv_timeout(READ_BUDGET).ok().flatten()
}

/// The boundary rules for `text` at the field `anchor`, as it is now.
fn adjust_at(
    inserter: &Inserter,
    target: &TargetSnapshot,
    anchor: &FieldAnchor,
    text: &str,
) -> Option<Adjusted> {
    let Surrounding::Text(surrounding) = inserter.read_field(target, anchor) else {
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
    /// A field lookup is running: at most one is ever out, so an
    /// accessibility bus that stopped answering costs one thread.
    locating: Arc<AtomicBool>,
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
    if !locate {
        field.set(None);
    }
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
    let locating = workers.locating.clone();
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
            if let Some(slot) = locate {
                let field = located.and_then(|target| {
                    if locating.swap(true, Ordering::SeqCst) {
                        return None;
                    }
                    let _lookup = OutGuard(locating);
                    let field = inserter.locate_field(&target)?;
                    (answered.elapsed() <= LOCATE_BUDGET).then_some(field)
                });
                slot.set(field);
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
    /// The mode the take was dictated in, and whether its text goes in
    /// verbatim (decided when the text was ready): Paste last follows it.
    pub mode: &'static ModeEntry,
    pub verbatim: bool,
    pub failure: Failure,
    /// Paste last is armed: the next focus away from Starling receives
    /// the text. The instant tells one arming from the next.
    pub armed: Option<Instant>,
    /// A Paste last insert is running.
    pub pasting: bool,
    pub copied: bool,
}

impl Recovery {
    fn new(
        take_id: &str,
        text: String,
        mode: &'static ModeEntry,
        verbatim: bool,
        failure: Failure,
    ) -> Recovery {
        Recovery {
            take_id: take_id.to_string(),
            text,
            mode,
            verbatim,
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
    /// A field read for the boundary rules is out (see [`boundary_text`]).
    boundary_out: Arc<AtomicBool>,
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
            boundary_out: Arc::default(),
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
    /// where the boundary rules can apply: not in a verbatim mode, nor
    /// for a text already known to go in `verbatim`.
    fn capture_now(&self, mode: &'static ModeEntry, verbatim: bool) -> Capture {
        let at = Instant::now();
        let (target, field) = if self.starling_focused() {
            (Err(InsertError::TargetIsStarling), FieldSlot::default())
        } else {
            let locate = !verbatim && mode.behavior != Behavior::Verbatim;
            bounded_capture(
                self.delivery.inserter.clone(),
                &self.delivery.capture_out,
                locate,
            )
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
            .then(|| self.capture_now(mode, false));
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
        // Routed on the retry's own text.
        let mode = self.active_mode();
        let verbatim = crate::processing::delivers_verbatim(mode, text);
        self.delivery.replace_recovery(Some(Recovery::new(
            id,
            text.to_string(),
            mode,
            verbatim,
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
        let verbatim = self.take_verbatim(id, capture.mode, &text);
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
                    verbatim,
                    failure,
                )));
                cx.notify();
            }
            Plan::Insert(target) => {
                self.spawn_insert(id.to_string(), target, text, &capture, verbatim, None, cx)
            }
        }
    }

    /// Whether take `id`, dictated in `mode`, goes in verbatim: routed on
    /// its own transcript, not on `text`, as processing may have dropped
    /// the spoken override that made it verbatim.
    fn take_verbatim(&self, id: &str, mode: &'static ModeEntry, text: &str) -> bool {
        let transcript = self.transcript_for(id);
        crate::processing::delivers_verbatim(mode, transcript.as_deref().unwrap_or(text))
    }

    /// Types `text` into `target` off the UI thread, after every insert
    /// started before it finished, adjusted by the insertion-boundary
    /// rules at `capture`'s field unless it goes in `verbatim`. `paste`
    /// is the recovery generation a Paste last runs for;
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
        verbatim: bool,
        paste: Option<u64>,
        cx: &mut Context<Self>,
    ) {
        let inserter = self.delivery.inserter.clone();
        let had_focus = self.delivery.own_focus.had_focus_since_now();
        let typed_text = text.clone();
        let mode = capture.mode;
        let field = capture.field.clone();
        let previous = self.delivery.last_insert.take();
        let out = self.delivery.insert_out.clone();
        let reading = self.delivery.boundary_out.clone();
        self.overlay_insert_started(cx);
        let insert = cx.spawn(async move |this, cx| {
            if let Some(previous) = previous {
                previous.await;
            }
            // A stalled insert still out holds the insertion lock; typing
            // beside it could interleave keys, so nothing more is typed.
            let result = if out.load(Ordering::SeqCst) {
                Err(Failure::EarlierTypingStuck)
            } else {
                // Read right before typing: the text before the caret as
                // it is now, never as it was at the take's start. Bounded
                // on its own, before the typing budget starts.
                let adjusted = if verbatim {
                    None
                } else {
                    let (inserter, target) = (inserter.clone(), target.clone());
                    cx.background_spawn(async move {
                        boundary_text(inserter, target, &field, reading, typed_text)
                    })
                    .await
                };
                let typed = adjusted.as_ref().map_or(text.clone(), |a| a.text.clone());
                let budget = typing_budget(&typed);
                let abandoned = Arc::new(AtomicBool::new(false));
                let stop = insert_stop(had_focus, abandoned.clone());
                // Inserts run one after another: none set it meanwhile.
                out.store(true, Ordering::SeqCst);
                let typing = cx.background_spawn(async move {
                    let _worker = OutGuard(out);
                    inserter
                        .insert(&target, &typed, &stop)
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
                app.insert_finished(id, text, mode, verbatim, paste, result, cx)
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
        verbatim: bool,
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
                        .replace_recovery(Some(Recovery::new(&id, text, mode, verbatim, failure)));
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

    /// Take `id`'s own transcript (a retry's own result first), before
    /// any processing.
    fn transcript_for(&self, id: &str) -> Option<String> {
        self.own_result(id).or_else(|| {
            self.sessions
                .iter()
                .find(|session| session.id == id)?
                .transcript
                .as_ref()
                .map(|transcript| transcript.text.clone())
        })
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
        let verbatim = self.take_verbatim(&id, capture.mode, &text);
        let capture = if unverifiable {
            self.capture_now(capture.mode, verbatim)
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
                    verbatim,
                    failure,
                )));
            }
            Plan::Insert(target) => {
                self.delivery.staged_inserts.insert(id.clone(), false);
                self.spawn_insert(id, target, text, &capture, verbatim, None, cx);
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
        let Some((mode, verbatim)) = self
            .delivery
            .recovery
            .as_ref()
            .map(|recovery| (recovery.mode, recovery.verbatim))
        else {
            return;
        };
        let capture = self.capture_now(mode, verbatim);
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
                self.spawn_insert(id, target, text, &capture, verbatim, Some(generation), cx);
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
