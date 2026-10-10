//! The recording shortcut's activation state machine (#221).
//!
//! One pure machine turns shortcut presses and releases, the record
//! button, Escape, timer ticks, and the recorder's own callbacks into
//! take-level effects. It owns the single-take guarantees, so the app
//! never decides them ad hoc:
//!
//! - **One take at a time.** Every take gets a fresh [`TakeId`]; a
//!   callback carrying an older id (a late "first samples" report, a
//!   start failure from a take that was already cancelled) is ignored.
//! - **Key repeat never acts.** The machine tracks whether the shortcut
//!   is down; a press while it is down is auto-repeat (Windows sends
//!   `WM_HOTKEY` repeats, gpui sends held key-downs) and does nothing.
//!   A press long after the last key event while the key still reads as
//!   down means a release was lost (focus moved mid-hold): it recovers
//!   instead of leaving a stuck recording.
//! - **Readiness tracks real audio.** A take starts in
//!   [`Readiness::Starting`] and only becomes [`Readiness::Listening`]
//!   when the app reports captured samples — the announcement effect
//!   ([`Effect::Listening`]) fires once, and never after the take
//!   failed, was cancelled, or was superseded. A take that never
//!   produces samples is cancelled after [`START_STALL`] instead of
//!   recording silence forever.
//! - **Stopping before any audio arrived cancels** ([`CancelReason::NoAudioYet`])
//!   rather than persisting an empty take.
//! - **Escape cancels** the active take and clears any latch. The app
//!   keeps whatever audio was captured (recoverable from history) and
//!   delivers nothing.
//!
//! Activation modes (`settings::ActivationMode`): `Toggle` (press starts,
//! next press stops), `Hold` (push-to-talk; with the optional double tap
//! a quick tap-tap latches the take hands-free), and `HoldOrToggle`
//! (a quick tap latches, a long hold records until release).

use std::time::{Duration, Instant};

use gpui::Context;
use starling_dictation::settings::{ActivationMode, DictationSettings};

use crate::app::StarlingApp;
use crate::shortcut::{GlobalEvent, GlobalShortcuts};

/// How often the app feeds system-wide shortcut events, timers, and the
/// recorder's sample count into the machine while one is running (a
/// take is active, or the shortcut key is down and its repeats stream
/// in). Event timestamps are taken where the events arrive, so this
/// only bounds reaction latency.
const POLL_INTERVAL: Duration = Duration::from_millis(20);

/// The cadence the same loop backs off to once the machine idles: no
/// take runs and the shortcut key is up, so the poll's only job is to
/// notice the next event or the next take — and a system-wide event
/// carries the timestamp of when it was received, so tap and hold
/// durations stay honest; only how long an idle press takes to start a
/// take grows, by at most the difference between the two cadences.
const IDLE_POLL_INTERVAL: Duration = Duration::from_millis(50);

/// Identifies one take for the lifetime of the app.
pub(crate) type TakeId = u64;

/// A release this soon after its press is a tap, not a hold.
pub(crate) const TAP_MAX: Duration = Duration::from_millis(300);

/// Hold mode with double tap: how long after a tap's release the second
/// press may come to latch the take hands-free.
pub(crate) const DOUBLE_TAP_WINDOW: Duration = Duration::from_millis(400);

/// A press while the key reads as down is auto-repeat if it comes within
/// this long of the previous key event. Repeats arrive every 25–50 ms
/// after an initial delay of at most one second on every desktop the app
/// runs on, so a longer silence means the release was lost.
pub(crate) const REPEAT_GAP: Duration = Duration::from_millis(1200);

/// What the record button showed when it was clicked: the state a click
/// must still match for it to mean anything.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RecordButton {
    Start,
    Stop,
}

/// Whether a record-button click still means what the button showed when
/// it was clicked. A click aimed at Stop after an Escape or release already
/// ended the take (or at Start after the shortcut began one) is stale and
/// does nothing, instead of toggling the other way.
pub(crate) fn click_matches(showed: RecordButton, recording: bool) -> bool {
    match showed {
        RecordButton::Stop => recording,
        RecordButton::Start => !recording,
    }
}

/// Whether the Starling window was focused at `at`, from its recorded
/// focus changes (oldest first). Before the first record: not focused.
pub(crate) fn focused_at(changes: &[(Instant, bool)], at: Instant) -> bool {
    changes
        .iter()
        .rev()
        .find(|(when, _)| *when <= at)
        .is_some_and(|(_, focused)| *focused)
}

/// How many focus changes are remembered for [`focused_at`].
const FOCUS_HISTORY: usize = 16;

/// A take whose microphone has delivered no samples this long after the
/// start is cancelled: the device is not producing audio, and recording
/// nothing indefinitely would be a stuck take.
pub(crate) const START_STALL: Duration = Duration::from_secs(5);

/// The machine's configuration, taken from the dictation settings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ActivationConfig {
    pub mode: ActivationMode,
    pub double_tap_hands_free: bool,
}

impl ActivationConfig {
    pub(crate) fn from_settings(settings: &DictationSettings) -> Self {
        Self {
            mode: settings.activation,
            double_tap_hands_free: settings.double_tap_hands_free,
        }
    }
}

impl Default for ActivationConfig {
    fn default() -> Self {
        Self {
            mode: ActivationMode::default(),
            double_tap_hands_free: false,
        }
    }
}

/// Whether the take's microphone has delivered audio yet.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Readiness {
    /// The take started; no captured samples have arrived.
    Starting,
    /// Real samples arrived: the user is being heard.
    Listening,
}

/// What ends the active take.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Latch {
    /// The shortcut is held: releasing it finishes the take.
    Held,
    /// Latched by a tap, the toggle mode, or the record button: the next
    /// press finishes the take.
    Latched,
    /// Latched hands-free by a double tap (hold mode): the next press
    /// finishes the take.
    HandsFree,
}

/// Why a take was cancelled rather than finished.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CancelReason {
    /// The user pressed Escape.
    Escape,
    /// The take was stopped before its microphone delivered any audio.
    NoAudioYet,
    /// The microphone delivered no audio within [`START_STALL`].
    MicStalled,
    /// The microphone failed or stopped delivering mid-take: the audio
    /// captured before that is kept as an interrupted take.
    InputLost,
}

/// What the app must do in response to an input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Effect {
    /// Start recording this take. The app reports a failure back through
    /// [`Activation::start_failed`].
    Start(TakeId),
    /// Stop recording and process the take as usual.
    Finish(TakeId),
    /// Stop recording, keep any captured audio, deliver nothing.
    Cancel(TakeId, CancelReason),
    /// The take's first real samples arrived: announce listening.
    Listening(TakeId),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Idle,
    Active {
        take: TakeId,
        readiness: Readiness,
        latch: Latch,
        /// The configuration this take started with: a mode saved
        /// mid-gesture applies to the next take, never to the one the
        /// user is holding.
        config: ActivationConfig,
        started_at: Instant,
        pressed_at: Instant,
        /// Hold mode with double tap: the take was tapped and waits this
        /// long for the second press before finishing.
        tap_deadline: Option<Instant>,
    },
}

/// The activation state machine. Pure: time comes in as arguments.
#[derive(Debug)]
pub(crate) struct Activation {
    config: ActivationConfig,
    phase: Phase,
    key_down: bool,
    /// Whether the window reported the press that holds the key down. A
    /// window press while the key is down is always auto-repeat (gpui
    /// forwards held repeats, after a delay the user may set above
    /// `REPEAT_GAP`); that path's lost release is the window losing focus
    /// instead (see [`Activation::window_lost_focus`]).
    key_from_window: bool,
    last_key_event: Option<Instant>,
    /// The most recently issued take id; takes are numbered from 1.
    last_take: TakeId,
}

impl Activation {
    pub(crate) fn new(config: ActivationConfig) -> Self {
        Self {
            config,
            phase: Phase::Idle,
            key_down: false,
            key_from_window: false,
            last_key_event: None,
            last_take: 0,
        }
    }

    /// A settings change applies to the next gesture; an active take
    /// keeps running.
    pub(crate) fn set_config(&mut self, config: ActivationConfig) {
        self.config = config;
    }

    /// The most recently started take (0 before the first).
    pub(crate) fn last_started(&self) -> TakeId {
        self.last_take
    }

    pub(crate) fn active_take(&self) -> Option<TakeId> {
        match self.phase {
            Phase::Active { take, .. } => Some(take),
            Phase::Idle => None,
        }
    }

    pub(crate) fn is_active(&self) -> bool {
        self.active_take().is_some()
    }

    pub(crate) fn readiness(&self) -> Option<Readiness> {
        match self.phase {
            Phase::Active { readiness, .. } => Some(readiness),
            Phase::Idle => None,
        }
    }

    /// What ends the active take — the latch the UI shows.
    pub(crate) fn latch(&self) -> Option<Latch> {
        match self.phase {
            Phase::Active {
                latch,
                tap_deadline,
                ..
            } => Some(if tap_deadline.is_some() {
                // Between the tap and the double-tap deadline the take
                // ends on its own unless the second press comes; to the
                // user it is not held any more.
                Latch::Latched
            } else {
                latch
            }),
            Phase::Idle => None,
        }
    }

    /// The shortcut changed: forget the old key's state.
    pub(crate) fn reset_key(&mut self) {
        self.key_down = false;
        self.last_key_event = None;
    }

    /// Whether the shortcut key is down from a press the machine took.
    pub(crate) fn key_is_down(&self) -> bool {
        self.key_down
    }

    /// The shortcut went down. `may_start` is false while the app must
    /// not begin a take (a modal is open): such a press can still stop
    /// the active take, but never starts one.
    pub(crate) fn press(&mut self, now: Instant, may_start: bool) -> Vec<Effect> {
        self.press_from(now, may_start, false)
    }

    /// A press the Starling window reported (see `key_from_window`).
    pub(crate) fn press_in_window(&mut self, now: Instant, may_start: bool) -> Vec<Effect> {
        self.press_from(now, may_start, true)
    }

    /// The window lost focus. A key it saw go down can no longer be seen
    /// going up, so its release is taken as happening now: a held take
    /// finishes instead of recording until a release that never comes.
    pub(crate) fn window_lost_focus(&mut self, now: Instant) -> Vec<Effect> {
        if self.key_down && self.key_from_window {
            self.release(now)
        } else {
            Vec::new()
        }
    }

    fn press_from(&mut self, now: Instant, may_start: bool, window: bool) -> Vec<Effect> {
        if self.key_down && window && self.key_from_window {
            self.last_key_event = Some(now);
            return Vec::new();
        }
        if self.key_down {
            let repeat = self
                .last_key_event
                .is_some_and(|last| now.saturating_duration_since(last) < REPEAT_GAP);
            self.last_key_event = Some(now);
            if repeat {
                return Vec::new();
            }
            // The release was lost. A held take would otherwise record
            // until the next release that may never come: this press is
            // the user's way out, so it finishes the take and does
            // nothing else.
            if let Phase::Active {
                latch: Latch::Held,
                tap_deadline: None,
                ..
            } = self.phase
            {
                return self.finish();
            }
        }
        self.key_down = true;
        self.key_from_window = window;
        self.last_key_event = Some(now);

        match self.phase {
            Phase::Idle => self.start(now, may_start, self.initial_latch()),
            Phase::Active {
                latch: Latch::Held,
                tap_deadline: Some(deadline),
                ..
            } => {
                if now <= deadline {
                    self.set_latch(Latch::HandsFree);
                    Vec::new()
                } else {
                    // The tick that should have finished the tapped take
                    // has not run yet: finish it now, and this press
                    // starts the next take.
                    let mut effects = self.finish();
                    effects.extend(self.start(now, may_start, self.initial_latch()));
                    effects
                }
            }
            Phase::Active { .. } => self.finish(),
        }
    }

    /// The shortcut went up. The take decides with the configuration it
    /// started with — a mode saved mid-gesture would otherwise change
    /// what a pending release means.
    pub(crate) fn release(&mut self, now: Instant) -> Vec<Effect> {
        if !self.key_down {
            return Vec::new();
        }
        self.key_down = false;
        self.last_key_event = Some(now);
        let Phase::Active {
            latch: Latch::Held,
            pressed_at,
            tap_deadline: None,
            config,
            ..
        } = self.phase
        else {
            return Vec::new();
        };
        let tap = now.saturating_duration_since(pressed_at) < TAP_MAX;
        match config.mode {
            ActivationMode::HoldOrToggle if tap => {
                self.set_latch(Latch::Latched);
                Vec::new()
            }
            ActivationMode::Hold if tap && config.double_tap_hands_free => {
                if let Phase::Active { tap_deadline, .. } = &mut self.phase {
                    *tap_deadline = Some(now + DOUBLE_TAP_WINDOW);
                }
                Vec::new()
            }
            // A held take never started in Toggle mode (its takes are
            // latched from the start), so every other release finishes it.
            _ => self.finish(),
        }
    }

    /// The on-screen record button: a toggle in every mode (a click
    /// cannot be held, and it never auto-repeats).
    pub(crate) fn click(&mut self, now: Instant) -> Vec<Effect> {
        match self.phase {
            Phase::Idle => self.start(now, true, Latch::Latched),
            Phase::Active { .. } => self.finish(),
        }
    }

    /// Escape: cancel the active take and clear any latch.
    pub(crate) fn escape(&mut self) -> Vec<Effect> {
        match self.phase {
            Phase::Active { take, .. } => {
                self.phase = Phase::Idle;
                vec![Effect::Cancel(take, CancelReason::Escape)]
            }
            Phase::Idle => Vec::new(),
        }
    }

    /// Time passed: the double-tap window and the start stall expire here.
    pub(crate) fn tick(&mut self, now: Instant) -> Vec<Effect> {
        let Phase::Active {
            take,
            readiness,
            started_at,
            tap_deadline,
            ..
        } = self.phase
        else {
            return Vec::new();
        };
        if readiness == Readiness::Starting
            && now.saturating_duration_since(started_at) >= START_STALL
        {
            self.phase = Phase::Idle;
            return vec![Effect::Cancel(take, CancelReason::MicStalled)];
        }
        if tap_deadline.is_some_and(|deadline| now > deadline) {
            return self.finish();
        }
        Vec::new()
    }

    /// `take`'s microphone failed or stopped delivering mid-take: cancel it
    /// as interrupted. A stale id (an older take) does nothing.
    pub(crate) fn input_lost(&mut self, take: TakeId) -> Vec<Effect> {
        match self.phase {
            Phase::Active { take: active, .. } if active == take => {
                self.phase = Phase::Idle;
                vec![Effect::Cancel(take, CancelReason::InputLost)]
            }
            _ => Vec::new(),
        }
    }

    /// The disk under `take`'s journal is nearly full (#342): finish it
    /// now, exactly like a stop the user asked for, so it is saved while
    /// there is room. A stale id does nothing.
    pub(crate) fn storage_full(&mut self, take: TakeId) -> Vec<Effect> {
        match self.phase {
            Phase::Active { take: active, .. } if active == take => self.finish(),
            _ => Vec::new(),
        }
    }

    /// The recorder reported captured samples for `take`.
    pub(crate) fn samples_arrived(&mut self, take: TakeId) -> Vec<Effect> {
        match &mut self.phase {
            Phase::Active {
                take: active,
                readiness: readiness @ Readiness::Starting,
                ..
            } if *active == take => {
                *readiness = Readiness::Listening;
                vec![Effect::Listening(take)]
            }
            _ => Vec::new(),
        }
    }

    /// Starting `take` failed: nothing is recording, nothing is announced.
    pub(crate) fn start_failed(&mut self, take: TakeId) {
        self.ended(take);
    }

    /// `take` ended outside the machine (the recorder stopped on its own).
    pub(crate) fn ended(&mut self, take: TakeId) {
        if self.active_take() == Some(take) {
            self.phase = Phase::Idle;
        }
    }

    fn initial_latch(&self) -> Latch {
        match self.config.mode {
            ActivationMode::Toggle => Latch::Latched,
            ActivationMode::Hold | ActivationMode::HoldOrToggle => Latch::Held,
        }
    }

    fn start(&mut self, now: Instant, may_start: bool, latch: Latch) -> Vec<Effect> {
        if !may_start {
            return Vec::new();
        }
        self.last_take += 1;
        let take = self.last_take;
        self.phase = Phase::Active {
            take,
            readiness: Readiness::Starting,
            latch,
            config: self.config,
            started_at: now,
            pressed_at: now,
            tap_deadline: None,
        };
        vec![Effect::Start(take)]
    }

    fn set_latch(&mut self, new: Latch) {
        if let Phase::Active {
            latch,
            tap_deadline,
            ..
        } = &mut self.phase
        {
            *latch = new;
            *tap_deadline = None;
        }
    }

    /// End the active take the normal way: finished when it has audio,
    /// cancelled when its microphone never delivered any.
    fn finish(&mut self) -> Vec<Effect> {
        let Phase::Active {
            take, readiness, ..
        } = self.phase
        else {
            return Vec::new();
        };
        self.phase = Phase::Idle;
        match readiness {
            Readiness::Listening => vec![Effect::Finish(take)],
            Readiness::Starting => vec![Effect::Cancel(take, CancelReason::NoAudioYet)],
        }
    }
}

/// The capture pane's headline for the active take: listening is only
/// claimed once real samples arrived.
pub(crate) fn readiness_headline(readiness: Option<Readiness>) -> &'static str {
    match readiness {
        Some(Readiness::Starting) => "Starting the microphone…",
        Some(Readiness::Listening) => "Listening closely.",
        None => "Say it as you mean it.",
    }
}

/// How the active take ends, in the words the capture pane shows.
pub(crate) fn finish_hint(latch: Option<Latch>, shortcut: &str) -> Option<String> {
    Some(match latch? {
        Latch::Held => "Release to finish · Esc cancels".to_string(),
        Latch::Latched => format!("Press {shortcut} to finish · Esc cancels"),
        Latch::HandsFree => format!("Hands-free · press {shortcut} to finish · Esc cancels"),
    })
}

// ---- App glue -------------------------------------------------------------

impl StarlingApp {
    /// Takes ownership of the system-wide shortcut registrations (#221),
    /// registers the configured shortcut, and starts the loop that feeds
    /// the machine.
    pub(crate) fn install_global_shortcuts(
        &mut self,
        shortcuts: Result<GlobalShortcuts, String>,
        cx: &mut Context<Self>,
    ) {
        match shortcuts {
            Ok(shortcuts) => {
                self.global_shortcuts = Some(shortcuts);
                let current = self.shortcut.clone();
                self.register_shortcut(&current);
            }
            Err(reason) => self.shortcut_registration = Err(reason),
        }
        if let Err(reason) = &self.shortcut_registration {
            eprintln!("Global shortcut unavailable: {reason}");
        }
        self.portal_shortcuts = crate::portal::PortalShortcuts::start_for_session(&self.shortcut);
        self.install_key_interceptor(cx);
        cx.spawn(async move |this, cx| {
            // 20 ms while a take runs or the key is held; the idle
            // backoff otherwise — `poll_activation` says which, turn by
            // turn.
            let mut wait = POLL_INTERVAL;
            loop {
                gpui::Timer::after(wait).await;
                match this.update(cx, |app, cx| app.poll_activation(cx)) {
                    Ok(next) => wait = next,
                    Err(_) => break,
                }
            }
        })
        .detach();
    }

    /// (Re-)register a shortcut system-wide, recording whether it took.
    pub(crate) fn register_shortcut(&mut self, shortcut: &crate::shortcut::Shortcut) {
        if let Some(shortcuts) = self.global_shortcuts.as_mut() {
            self.shortcut_registration = shortcuts.set_record(shortcut);
        }
    }

    /// Record the window's focus changes, so system-wide events can be
    /// matched to whether Starling had focus when they happened.
    pub(crate) fn track_window_focus(&mut self, window: &mut gpui::Window, cx: &mut Context<Self>) {
        self.window_focus.push((Instant::now(), window.is_window_active()));
        let subscription = cx.observe_window_activation(window, |app, window, cx| {
            let active = window.is_window_active();
            app.window_focus.push((Instant::now(), active));
            app.delivery_window_activation(active, cx);
            if !active {
                // Like every UI-thread input: system-wide events that came
                // first (an Escape) are processed before the synthetic
                // release, so a focus change never overtakes a cancel.
                app.flush_system_events(cx);
                app.activation_input(|machine| machine.window_lost_focus(Instant::now()), cx);
            }
            if app.window_focus.len() > FOCUS_HISTORY {
                app.window_focus.remove(0);
            }
        });
        self.focus_observer = Some(subscription);
    }

    /// Whether a system-wide event belongs to the machine. On Wayland a
    /// compositor may forward the focused native window's keys to XWayland
    /// too, so an event that happened while Starling had focus is the
    /// window's own key event seen twice: the window already handled it.
    /// Only keys the window can match itself are dropped: a shortcut the
    /// window has no name for (Pause) still comes through.
    fn system_event_is_ours(&self, event: GlobalEvent) -> bool {
        let (at, window_can_match) = match event {
            GlobalEvent::Pressed(at) | GlobalEvent::Released(at) => {
                (at, self.shortcut.works_in_window())
            }
            GlobalEvent::Escape(at) => (at, true),
        };
        !(window_can_match
            && crate::shortcut::wayland_session()
            && focused_at(&self.window_focus, at))
    }

    /// Feed every system-wide event received so far to the machine, in
    /// arrival order. Runs on the poll and before every UI-thread input
    /// (window keys, the record button, opening or closing Settings), so a
    /// UI input never overtakes a system-wide one that happened first.
    pub(crate) fn flush_system_events(&mut self, cx: &mut Context<Self>) {
        // One physical press from one source: while the desktop's portal
        // holds a binding it is the system-wide shortcut, and the X11
        // grab's presses (an XWayland app focused) are dropped. Its Escape
        // grabs still cancel.
        let portal_bound = self
            .portal_shortcuts
            .as_ref()
            .is_some_and(|portal| portal.is_bound());
        while let Some(shortcuts) = self.global_shortcuts.as_mut() {
            let Some(raw) = shortcuts.next_raw() else {
                break;
            };
            let Some(event) = shortcuts.classify(raw) else {
                continue;
            };
            if !self.system_event_is_ours(event)
                || (portal_bound && !matches!(event, GlobalEvent::Escape(_)))
            {
                continue;
            }
            self.system_event(event, cx);
        }
        // The portal's edges are never the focused window's own keys seen
        // twice (the desktop consumes a bound shortcut), so no focus
        // filter applies; a desktop that also forwards them to the
        // Starling window is covered by the machine's repeat rule.
        while let Some(event) = self.portal_shortcuts.as_mut().and_then(|p| p.next_event()) {
            self.system_event(event, cx);
        }
        if self
            .portal_shortcuts
            .as_mut()
            .is_some_and(|portal| portal.take_status_changed())
        {
            cx.notify();
        }
    }

    /// One system-wide event into the machine.
    fn system_event(&mut self, event: GlobalEvent, cx: &mut Context<Self>) {
        if self.settings_open && matches!(event, GlobalEvent::Pressed(_)) {
            self.note_shortcut_in_dialog(cx);
        }
        let may_start = !self.settings_open;
        self.activation_input(
            |machine| match event {
                GlobalEvent::Pressed(at) => machine.press(at, may_start),
                GlobalEvent::Released(at) => machine.release(at),
                GlobalEvent::Escape(_) => machine.escape(),
            },
            cx,
        );
    }

    /// In-window presses go through a keystroke interceptor: it runs
    /// before any key binding, so the shortcut also works while the
    /// staging editor or a settings field has focus, and a matched press
    /// never reaches them.
    fn install_key_interceptor(&mut self, cx: &mut Context<Self>) {
        let app = cx.entity().downgrade();
        let subscription = cx.intercept_keystrokes(move |event, _window, cx| {
            let handled = app
                .update(cx, |app, cx| app.shortcut_key_down(&event.keystroke, cx))
                .unwrap_or(false);
            if handled {
                cx.stop_propagation();
            }
        });
        self.key_interceptor = Some(subscription);
    }

    /// An in-window key-down. Returns whether it was the shortcut or the
    /// Escape that cancels the active take (and so must not propagate).
    pub(crate) fn shortcut_key_down(
        &mut self,
        keystroke: &gpui::Keystroke,
        cx: &mut Context<Self>,
    ) -> bool {
        self.flush_system_events(cx);
        if self.activation.is_active() && crate::shortcut::is_escape(keystroke) {
            self.activation_input(|machine| machine.escape(), cx);
            return true;
        }
        if !self.shortcut.matches_key_down(keystroke) {
            return false;
        }
        if self.settings_open {
            self.note_shortcut_in_dialog(cx);
        }
        let may_start = !self.settings_open;
        // A window press while the key is down is a repeat — unless the
        // window can never see this shortcut's release (a Cmd chord on
        // macOS), where it keeps the system path's lost-release recovery.
        if self.shortcut.window_reports_release() {
            self.activation_input(|machine| machine.press_in_window(Instant::now(), may_start), cx);
        } else {
            self.activation_input(|machine| machine.press(Instant::now(), may_start), cx);
        }
        true
    }

    /// An in-window key-up: the release of a held shortcut. Returns
    /// whether it was consumed — the release matched and the machine had
    /// the key down from a press it took — so an unrelated chord that
    /// happens to end on this key still reaches the focused editor.
    pub(crate) fn shortcut_key_up(
        &mut self,
        keystroke: &gpui::Keystroke,
        cx: &mut Context<Self>,
    ) -> bool {
        self.flush_system_events(cx);
        let consumed = self.shortcut.matches_key_up(keystroke) && self.activation.key_is_down();
        if consumed {
            self.activation_input(|machine| machine.release(Instant::now()), cx);
        }
        consumed
    }

    /// One loop turn: system-wide events in arrival order, then timers.
    /// Returns how long the loop waits before its next turn: the fast
    /// [`POLL_INTERVAL`] while a take is active, the shortcut key is
    /// down (the press/release pair and the repeats arrive as a stream
    /// then), or the settings dialog's microphone check is recording
    /// (its time limit is enforced here), and the slower
    /// [`IDLE_POLL_INTERVAL`] once the machine idles — which costs
    /// nothing but idle start latency, since the timers act on the
    /// events' receive timestamps, not on the poll's.
    pub(crate) fn poll_activation(&mut self, cx: &mut Context<Self>) -> Duration {
        self.flush_system_events(cx);
        self.activation_input(|machine| machine.tick(Instant::now()), cx);
        self.poll_microphone(cx);
        self.sync_overlay(cx);
        if self.activation.is_active()
            || self.activation.key_is_down()
            || self.mic_check_recording()
        {
            POLL_INTERVAL
        } else {
            IDLE_POLL_INTERVAL
        }
    }

    /// Feed one input to the machine and perform its effects. Readiness
    /// is refreshed first, so a stop or stall decision never acts on a
    /// sample count older than the input itself: audio that already
    /// arrived is never cancelled as "no audio".
    pub(crate) fn activation_input(
        &mut self,
        input: impl FnOnce(&mut Activation) -> Vec<Effect>,
        cx: &mut Context<Self>,
    ) {
        self.check_readiness(cx);
        let effects = input(&mut self.activation);
        self.apply_activation(effects, cx);
    }

    /// Readiness tracks real audio: listening is announced once the
    /// running take's recorder has captured samples, not when the
    /// shortcut fired.
    fn check_readiness(&mut self, cx: &mut Context<Self>) {
        if self.activation.readiness() != Some(Readiness::Starting) {
            return;
        }
        let (Some(take), Some(recorder)) = (self.recording_take, self.recorder.as_ref()) else {
            return;
        };
        if recorder.captured_sample_count() > 0 {
            let effects = self.activation.samples_arrived(take);
            self.apply_activation(effects, cx);
        }
    }

    /// Perform the machine's effects, in order.
    pub(crate) fn apply_activation(&mut self, effects: Vec<Effect>, cx: &mut Context<Self>) {
        if effects.is_empty() {
            return;
        }
        for effect in effects {
            match effect {
                Effect::Start(take) => {
                    self.cue_take_starting();
                    if self.start_recording(cx) {
                        self.recording_take = Some(take);
                        self.overlay_take_started();
                    } else {
                        self.activation.start_failed(take);
                    }
                }
                Effect::Finish(take) => {
                    if self.recording_take == Some(take) {
                        self.recording_take = None;
                        // A take whose microphone died is never presented
                        // as complete, even when the stop came before the
                        // watchdog noticed.
                        if self.note_live_interruption() {
                            self.cancel_recording(take, CancelReason::InputLost, cx);
                        } else {
                            self.stop_recording(take, cx);
                        }
                        // A stop that did not lead to a save (the device
                        // failed, the stop itself failed) ends like a cancel.
                        if !self.overlay.model.is_saving() {
                            self.overlay.model.take_cancelled(Instant::now());
                        }
                        self.cue_take_ended(take, cx);
                    }
                }
                Effect::Cancel(take, reason) => {
                    if self.recording_take == Some(take) {
                        self.recording_take = None;
                        self.cancel_recording(take, reason, cx);
                        self.overlay.model.take_cancelled(Instant::now());
                        self.cue_take_ended(take, cx);
                    }
                }
                Effect::Listening(take) => self.cue_listening(take, cx),
            }
        }
        // Escape cleanup is queued first, so a deferred swap below never
        // holds it back.
        self.sync_escape_grab();
        if !self.activation.is_active() {
            // A shortcut saved mid-take takes over once the take ended, so
            // the held key's release still finishes the take it started.
            // (On Linux this waits for the worker once, only after a
            // shortcut was changed mid-take.) A platform refusal keeps the
            // previous shortcut registered, writes it back to the settings
            // file the save already wrote, and says why.
            if let Some(shortcut) = self.pending_shortcut.take() {
                if let Err(reason) = self.apply_shortcut(shortcut) {
                    self.dictation_settings.shortcut = self.shortcut.text().to_string();
                    self.error = Some(format!(
                        "The new dictation shortcut could not be registered ({reason}); the \
                         previous one is still active."
                    ));
                    self.persist_committed_settings(cx);
                }
            }
        }
        self.sync_overlay(cx);
        cx.notify();
    }

    /// Make `shortcut` the recording shortcut, now. A caller with a take
    /// running defers instead (`pending_shortcut`), so the held key's
    /// release still finishes the take it started.
    ///
    /// `set_record` takes the new grab before releasing the old one, so a
    /// platform refusal returns here with the previous shortcut still
    /// registered; only a successful swap changes the app's shortcut.
    pub(crate) fn apply_shortcut(
        &mut self,
        shortcut: crate::shortcut::Shortcut,
    ) -> Result<(), String> {
        if shortcut == self.shortcut {
            return Ok(());
        }
        let outcome = self
            .global_shortcuts
            .as_mut()
            .map(|shortcuts| shortcuts.set_record(&shortcut));
        let applied = match outcome {
            Some(Err(reason)) => Err(reason),
            Some(Ok(())) => {
                self.shortcut_registration = Ok(());
                self.shortcut = shortcut;
                self.activation.reset_key();
                Ok(())
            }
            // No system-wide registrations at all (no X display): the
            // in-window matcher is the shortcut.
            None => {
                self.shortcut = shortcut;
                self.activation.reset_key();
                Ok(())
            }
        };
        // A desktop portal binding is offered the new keys too (the
        // desktop may keep the ones the user picked in its dialog).
        if let (Ok(()), Some(portal)) = (&applied, self.portal_shortcuts.as_ref()) {
            portal.rebind(&self.shortcut);
        }
        applied
    }

    /// Escape is grabbed system-wide exactly while a take is active.
    fn sync_escape_grab(&mut self) {
        let active = self.activation.is_active();
        let Some(shortcuts) = self.global_shortcuts.as_mut() else {
            return;
        };
        if shortcuts.escape_armed() != active {
            if let Err(reason) = shortcuts.arm_escape(active, &self.shortcut) {
                // In-window Escape still cancels; only the system-wide
                // grab is missing.
                eprintln!("Escape could not be grabbed system-wide: {reason}");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn machine(mode: ActivationMode, double_tap: bool) -> Activation {
        Activation::new(ActivationConfig {
            mode,
            double_tap_hands_free: double_tap,
        })
    }

    fn ms(n: u64) -> Duration {
        Duration::from_millis(n)
    }

    /// Start a take at `t0` and report its first samples.
    fn start_listening(machine: &mut Activation, t0: Instant) -> TakeId {
        let effects = machine.press(t0, true);
        let [Effect::Start(take)] = effects[..] else {
            panic!("expected a start, got {effects:?}");
        };
        assert_eq!(machine.samples_arrived(take), vec![Effect::Listening(take)]);
        take
    }

    #[test]
    fn toggle_starts_on_press_and_stops_on_the_next_press() {
        let mut m = machine(ActivationMode::Toggle, false);
        let t0 = Instant::now();
        let take = start_listening(&mut m, t0);
        assert_eq!(m.latch(), Some(Latch::Latched));
        assert!(m.release(t0 + ms(80)).is_empty());
        assert!(m.is_active());
        assert_eq!(m.press(t0 + ms(2000), true), vec![Effect::Finish(take)]);
        assert!(m.release(t0 + ms(2080)).is_empty());
        assert!(!m.is_active());
    }

    #[test]
    fn hold_records_while_held_and_finishes_on_release() {
        let mut m = machine(ActivationMode::Hold, false);
        let t0 = Instant::now();
        let take = start_listening(&mut m, t0);
        assert_eq!(m.latch(), Some(Latch::Held));
        assert_eq!(m.release(t0 + ms(1500)), vec![Effect::Finish(take)]);
        assert!(!m.is_active());
    }

    #[test]
    fn hold_without_double_tap_finishes_even_a_quick_tap() {
        let mut m = machine(ActivationMode::Hold, false);
        let t0 = Instant::now();
        let take = start_listening(&mut m, t0);
        assert_eq!(m.release(t0 + ms(100)), vec![Effect::Finish(take)]);
    }

    #[test]
    fn hold_or_toggle_latches_a_tap_and_finishes_a_hold() {
        let mut m = machine(ActivationMode::HoldOrToggle, false);
        let t0 = Instant::now();
        let take = start_listening(&mut m, t0);
        assert!(m.release(t0 + ms(150)).is_empty());
        assert_eq!(m.latch(), Some(Latch::Latched));
        assert_eq!(m.press(t0 + ms(3000), true), vec![Effect::Finish(take)]);
        // The stopping press's release does nothing.
        assert!(m.release(t0 + ms(3100)).is_empty());

        let t1 = t0 + ms(5000);
        let take = start_listening(&mut m, t1);
        assert_eq!(m.release(t1 + ms(900)), vec![Effect::Finish(take)]);
    }

    #[test]
    fn double_tap_latches_hold_mode_hands_free() {
        let mut m = machine(ActivationMode::Hold, true);
        let t0 = Instant::now();
        let take = start_listening(&mut m, t0);
        assert!(m.release(t0 + ms(120)).is_empty());
        // Waiting for the second tap: not held any more.
        assert_eq!(m.latch(), Some(Latch::Latched));
        assert!(m.press(t0 + ms(300), true).is_empty());
        assert_eq!(m.latch(), Some(Latch::HandsFree));
        assert!(m.release(t0 + ms(380)).is_empty());
        assert!(m.tick(t0 + ms(5000)).is_empty());
        assert!(m.is_active());
        assert_eq!(m.press(t0 + ms(9000), true), vec![Effect::Finish(take)]);
    }

    #[test]
    fn a_single_tap_in_double_tap_hold_mode_finishes_when_the_window_closes() {
        let mut m = machine(ActivationMode::Hold, true);
        let t0 = Instant::now();
        let take = start_listening(&mut m, t0);
        assert!(m.release(t0 + ms(120)).is_empty());
        assert!(m.tick(t0 + ms(120) + DOUBLE_TAP_WINDOW).is_empty());
        assert_eq!(
            m.tick(t0 + ms(121) + DOUBLE_TAP_WINDOW),
            vec![Effect::Finish(take)]
        );
    }

    #[test]
    fn a_late_second_tap_finishes_the_tapped_take_and_starts_the_next() {
        let mut m = machine(ActivationMode::Hold, true);
        let t0 = Instant::now();
        let take = start_listening(&mut m, t0);
        assert!(m.release(t0 + ms(100)).is_empty());
        // No tick ran in between: the press itself sees the expired window.
        let effects = m.press(t0 + ms(100) + DOUBLE_TAP_WINDOW + ms(50), true);
        assert_eq!(effects, vec![Effect::Finish(take), Effect::Start(take + 1)]);
        assert_eq!(m.latch(), Some(Latch::Held));
    }

    #[test]
    fn a_held_double_tap_hold_records_until_release() {
        let mut m = machine(ActivationMode::Hold, true);
        let t0 = Instant::now();
        let take = start_listening(&mut m, t0);
        assert_eq!(m.release(t0 + ms(800)), vec![Effect::Finish(take)]);
    }

    #[test]
    fn key_repeat_never_starts_or_stops_a_take() {
        for mode in [
            ActivationMode::Toggle,
            ActivationMode::Hold,
            ActivationMode::HoldOrToggle,
        ] {
            let mut m = machine(mode, true);
            let t0 = Instant::now();
            let take = start_listening(&mut m, t0);
            // Windows-style repeats: an initial delay, then every 33 ms.
            let mut at = t0 + ms(500);
            for _ in 0..200 {
                assert!(m.press(at, true).is_empty(), "{mode:?}");
                at += ms(33);
            }
            assert_eq!(m.active_take(), Some(take), "{mode:?}");
            // The long hold ends: hold-like modes finish, toggle keeps going.
            let released = m.release(at);
            match mode {
                ActivationMode::Toggle => assert!(released.is_empty()),
                _ => assert_eq!(released, vec![Effect::Finish(take)], "{mode:?}"),
            }
        }
    }

    #[test]
    fn rapid_presses_never_overlap_takes() {
        let mut m = machine(ActivationMode::Toggle, false);
        let t0 = Instant::now();
        let first = start_listening(&mut m, t0);
        assert!(m.release(t0 + ms(30)).is_empty());
        assert_eq!(m.press(t0 + ms(60), true), vec![Effect::Finish(first)]);
        assert!(m.release(t0 + ms(90)).is_empty());
        let effects = m.press(t0 + ms(120), true);
        assert_eq!(effects, vec![Effect::Start(first + 1)]);
        assert!(m.release(t0 + ms(150)).is_empty());
        // Stopped before its microphone delivered anything: cancelled,
        // never persisted as an empty take.
        assert_eq!(
            m.press(t0 + ms(180), true),
            vec![Effect::Cancel(first + 1, CancelReason::NoAudioYet)]
        );
    }

    #[test]
    fn a_lost_release_does_not_leave_a_stuck_hold() {
        let mut m = machine(ActivationMode::Hold, false);
        let t0 = Instant::now();
        let take = start_listening(&mut m, t0);
        // The release never arrived (focus moved). A press long after the
        // last key event finishes the take and starts nothing.
        assert_eq!(m.press(t0 + ms(4000), true), vec![Effect::Finish(take)]);
        assert!(!m.is_active());
        // Its own release is ignored, and the next press starts normally.
        assert!(m.release(t0 + ms(4100)).is_empty());
        assert_eq!(m.press(t0 + ms(6000), true), vec![Effect::Start(take + 1)]);
    }

    #[test]
    fn a_slow_window_repeat_never_ends_a_held_take() {
        // Native Wayland: gpui forwards held repeats through the window,
        // after a delay the user may set well above REPEAT_GAP.
        let mut m = machine(ActivationMode::Hold, false);
        let t0 = Instant::now();
        let [Effect::Start(take)] = m.press_in_window(t0, true)[..] else {
            panic!("expected a start");
        };
        m.samples_arrived(take);
        assert!(m.press_in_window(t0 + ms(1500), true).is_empty());
        assert!(m.press_in_window(t0 + ms(1530), true).is_empty());
        assert_eq!(m.active_take(), Some(take));
        assert_eq!(m.release(t0 + ms(4000)), vec![Effect::Finish(take)]);
    }

    #[test]
    fn losing_window_focus_releases_a_key_the_window_held() {
        let mut m = machine(ActivationMode::Hold, false);
        let t0 = Instant::now();
        let [Effect::Start(take)] = m.press_in_window(t0, true)[..] else {
            panic!("expected a start");
        };
        m.samples_arrived(take);
        assert_eq!(m.window_lost_focus(t0 + ms(800)), vec![Effect::Finish(take)]);
        assert!(m.window_lost_focus(t0 + ms(900)).is_empty());
        // A key held through the system-wide grab is not the window's.
        let take = start_listening(&mut m, t0 + ms(2000));
        assert!(m.window_lost_focus(t0 + ms(2500)).is_empty());
        assert_eq!(m.active_take(), Some(take));
    }

    #[test]
    fn a_lost_release_while_idle_does_not_swallow_the_next_press() {
        let mut m = machine(ActivationMode::Toggle, false);
        let t0 = Instant::now();
        let take = start_listening(&mut m, t0);
        assert_eq!(m.press(t0 + ms(3000), true), vec![Effect::Finish(take)]);
        // That press's release was lost; the next real press still works.
        assert_eq!(m.press(t0 + ms(8000), true), vec![Effect::Start(take + 1)]);
    }

    #[test]
    fn escape_cancels_and_clears_the_latch() {
        let mut m = machine(ActivationMode::Hold, true);
        let t0 = Instant::now();
        let take = start_listening(&mut m, t0);
        assert!(m.release(t0 + ms(100)).is_empty());
        assert!(m.press(t0 + ms(250), true).is_empty());
        assert_eq!(m.latch(), Some(Latch::HandsFree));
        assert_eq!(m.escape(), vec![Effect::Cancel(take, CancelReason::Escape)]);
        assert_eq!(m.latch(), None);
        assert!(m.escape().is_empty());
        // The second tap's release after Escape does nothing.
        assert!(m.release(t0 + ms(300)).is_empty());
        assert!(!m.is_active());
    }

    #[test]
    fn escape_while_held_ignores_the_later_release() {
        let mut m = machine(ActivationMode::Hold, false);
        let t0 = Instant::now();
        let take = start_listening(&mut m, t0);
        assert_eq!(m.escape(), vec![Effect::Cancel(take, CancelReason::Escape)]);
        assert!(m.release(t0 + ms(2000)).is_empty());
        assert_eq!(m.press(t0 + ms(3000), true), vec![Effect::Start(take + 1)]);
    }

    #[test]
    fn listening_is_announced_once_and_only_for_the_live_take() {
        let mut m = machine(ActivationMode::Toggle, false);
        let t0 = Instant::now();
        let [Effect::Start(take)] = m.press(t0, true)[..] else {
            panic!("expected a start");
        };
        assert_eq!(m.readiness(), Some(Readiness::Starting));
        assert_eq!(m.samples_arrived(take), vec![Effect::Listening(take)]);
        assert_eq!(m.readiness(), Some(Readiness::Listening));
        assert!(m.samples_arrived(take).is_empty());
        // A late report for a superseded take is ignored.
        assert!(m.samples_arrived(take - 1).is_empty());
    }

    #[test]
    fn no_listening_after_a_failed_start_or_a_cancel() {
        let mut m = machine(ActivationMode::Toggle, false);
        let t0 = Instant::now();
        let [Effect::Start(take)] = m.press(t0, true)[..] else {
            panic!("expected a start");
        };
        m.start_failed(take);
        assert!(!m.is_active());
        assert!(m.samples_arrived(take).is_empty());
        assert!(m.release(t0 + ms(50)).is_empty());

        let [Effect::Start(next)] = m.press(t0 + ms(2000), true)[..] else {
            panic!("expected a start");
        };
        assert_eq!(m.escape(), vec![Effect::Cancel(next, CancelReason::Escape)]);
        assert!(m.samples_arrived(next).is_empty());
        // A stale failure report cannot end a newer take.
        let [Effect::Start(third)] = m.press(t0 + ms(4000), true)[..] else {
            panic!("expected a start");
        };
        m.start_failed(next);
        assert_eq!(m.active_take(), Some(third));
    }

    #[test]
    fn a_microphone_that_never_delivers_is_cancelled_not_stuck() {
        let mut m = machine(ActivationMode::Toggle, false);
        let t0 = Instant::now();
        let [Effect::Start(take)] = m.press(t0, true)[..] else {
            panic!("expected a start");
        };
        assert!(m.tick(t0 + START_STALL - ms(1)).is_empty());
        assert_eq!(
            m.tick(t0 + START_STALL),
            vec![Effect::Cancel(take, CancelReason::MicStalled)]
        );
        assert!(!m.is_active());
        assert!(m.samples_arrived(take).is_empty());
    }

    #[test]
    fn a_listening_take_never_stalls() {
        let mut m = machine(ActivationMode::Toggle, false);
        let t0 = Instant::now();
        start_listening(&mut m, t0);
        assert!(m.tick(t0 + START_STALL * 10).is_empty());
        assert!(m.is_active());
    }

    #[test]
    fn a_lost_input_cancels_only_its_own_active_take() {
        let t0 = Instant::now();
        let mut machine = machine(ActivationMode::Toggle, false);
        let take = start_listening(&mut machine, t0);
        assert_eq!(machine.input_lost(take + 1), Vec::new(), "a stale id does nothing");
        assert!(machine.is_active());
        assert_eq!(
            machine.input_lost(take),
            vec![Effect::Cancel(take, CancelReason::InputLost)]
        );
        assert!(!machine.is_active());
        assert_eq!(machine.input_lost(take), Vec::new(), "already ended");
    }

    #[test]
    fn a_full_disk_finishes_only_its_own_take_like_a_normal_stop() {
        // #342: the take is saved and transcribed, not cancelled.
        let t0 = Instant::now();
        let mut machine = machine(ActivationMode::Toggle, false);
        let take = start_listening(&mut machine, t0);
        assert_eq!(machine.storage_full(take + 1), Vec::new(), "a stale id does nothing");
        assert!(machine.is_active());
        assert_eq!(machine.storage_full(take), vec![Effect::Finish(take)]);
        assert!(!machine.is_active());
        assert_eq!(machine.storage_full(take), Vec::new(), "already ended");

        // Before any audio arrived there is nothing to save.
        let effects = machine.click(t0);
        let Some(Effect::Start(starting)) = effects.first().cloned() else {
            panic!("no start: {effects:?}");
        };
        assert_eq!(
            machine.storage_full(starting),
            vec![Effect::Cancel(starting, CancelReason::NoAudioYet)]
        );
    }

    #[test]
    fn a_modal_blocks_starting_but_not_stopping() {
        let mut m = machine(ActivationMode::Toggle, false);
        let t0 = Instant::now();
        assert!(m.press(t0, false).is_empty());
        assert!(m.release(t0 + ms(50)).is_empty());
        let take = start_listening(&mut m, t0 + ms(1000));
        assert!(m.release(t0 + ms(1050)).is_empty());
        assert_eq!(m.press(t0 + ms(2000), false), vec![Effect::Finish(take)]);
    }

    #[test]
    fn a_hold_through_an_opening_modal_does_not_start_on_release() {
        let mut m = machine(ActivationMode::HoldOrToggle, false);
        let t0 = Instant::now();
        assert!(m.press(t0, false).is_empty());
        assert!(m.release(t0 + ms(100)).is_empty());
        assert!(!m.is_active());
    }

    #[test]
    fn the_record_button_toggles_in_every_mode() {
        for mode in [
            ActivationMode::Toggle,
            ActivationMode::Hold,
            ActivationMode::HoldOrToggle,
        ] {
            let mut m = machine(mode, true);
            let t0 = Instant::now();
            let [Effect::Start(take)] = m.click(t0)[..] else {
                panic!("expected a start");
            };
            assert_eq!(m.latch(), Some(Latch::Latched));
            m.samples_arrived(take);
            assert!(m.tick(t0 + ms(2000)).is_empty());
            assert_eq!(m.click(t0 + ms(2100)), vec![Effect::Finish(take)]);
        }
    }

    #[test]
    fn a_config_change_mid_hold_keeps_the_take_running() {
        let mut m = machine(ActivationMode::Hold, false);
        let t0 = Instant::now();
        let take = start_listening(&mut m, t0);
        m.set_config(ActivationConfig {
            mode: ActivationMode::Toggle,
            double_tap_hands_free: false,
        });
        // The take keeps the config it started with: its release still
        // finishes it, as hold does.
        assert_eq!(m.release(t0 + ms(2000)), vec![Effect::Finish(take)]);
        assert!(!m.is_active());
        // The new mode applies to the next take.
        let next = start_listening(&mut m, t0 + ms(4000));
        assert_eq!(m.latch(), Some(Latch::Latched));
        assert!(m.release(t0 + ms(4050)).is_empty());
        assert_eq!(m.press(t0 + ms(6000), true), vec![Effect::Finish(next)]);
    }

    #[test]
    fn the_pane_claims_listening_only_after_samples_and_names_the_latch() {
        assert_eq!(readiness_headline(None), "Say it as you mean it.");
        assert_eq!(readiness_headline(Some(Readiness::Starting)), "Starting the microphone…");
        assert_eq!(readiness_headline(Some(Readiness::Listening)), "Listening closely.");
        assert_eq!(finish_hint(None, "F9"), None);
        assert_eq!(
            finish_hint(Some(Latch::Held), "F9").as_deref(),
            Some("Release to finish · Esc cancels")
        );
        assert_eq!(
            finish_hint(Some(Latch::HandsFree), "F9").as_deref(),
            Some("Hands-free · press F9 to finish · Esc cancels")
        );
    }

    #[test]
    fn a_click_acts_only_on_the_state_its_button_showed() {
        assert!(click_matches(RecordButton::Start, false), "Start while idle starts");
        assert!(click_matches(RecordButton::Stop, true), "Stop while recording stops");
        assert!(!click_matches(RecordButton::Stop, false), "Stop after Escape ended the take");
        assert!(!click_matches(RecordButton::Start, true), "Start after the shortcut began one");
    }

    #[test]
    fn focus_at_an_instant_follows_the_recorded_changes() {
        let t0 = Instant::now();
        let changes = [(t0, false), (t0 + ms(100), true), (t0 + ms(500), false)];
        assert!(!focused_at(&changes, t0 + ms(50)));
        assert!(focused_at(&changes, t0 + ms(100)));
        assert!(focused_at(&changes, t0 + ms(499)));
        assert!(!focused_at(&changes, t0 + ms(600)));
        assert!(!focused_at(&[], t0));
    }

    #[test]
    fn a_new_shortcut_starts_with_a_clean_key_state() {
        let mut m = machine(ActivationMode::Toggle, false);
        let t0 = Instant::now();
        let take = start_listening(&mut m, t0);
        assert!(m.release(t0 + ms(50)).is_empty());
        assert_eq!(m.press(t0 + ms(1000), true), vec![Effect::Finish(take)]);
        // The old key is still down when the shortcut changes.
        m.reset_key();
        assert_eq!(
            m.press(t0 + ms(1100), true),
            vec![Effect::Start(take + 1)]
        );
    }

    #[test]
    fn an_external_end_returns_to_idle_only_for_the_live_take() {
        let mut m = machine(ActivationMode::Toggle, false);
        let t0 = Instant::now();
        let take = start_listening(&mut m, t0);
        m.ended(take + 7);
        assert!(m.is_active());
        m.ended(take);
        assert!(!m.is_active());
    }
}
