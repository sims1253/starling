//! The dictation overlay (#221): a small always-on-top window that shows
//! what the active take is doing while the user dictates into another
//! app.
//!
//! The overlay only ever reads app state. Its phase is derived from the
//! activation machine's readiness and the take's pipeline (save,
//! transcription, processing, delivery) by the pure [`OverlayModel`], so
//! hiding it, or changing its mode mid-take, never changes a recording
//! or a transcript; the live text it shows is a read-only copy of the
//! staging draft (#297), which keeps ownership of every edit.
//!
//! # Platform reach (gpui 0.2.2)
//!
//! The window is a `WindowKind::PopUp` opened with `focus: false`:
//!
//! - **X11**: a `_NET_WM_WINDOW_TYPE_NOTIFICATION` window. Window managers
//!   keep those above normal windows and do not give them focus. It opens
//!   bottom-centre on the monitor under the pointer (RandR monitors, so
//!   one X screen spanning several outputs still picks the right one).
//! - **Wayland**: gpui only creates `xdg_toplevel`s (no layer-shell), so
//!   the compositor decides where the overlay goes, whether it floats,
//!   and whether it takes focus. The window's app id is
//!   [`OVERLAY_APP_ID`] so a compositor rule can float it unfocused.
//! - **Windows**: gpui makes a `WS_EX_TOOLWINDOW` pop-up; the overlay adds
//!   `WS_EX_TOPMOST` and `WS_EX_NOACTIVATE` itself
//!   ([`keep_on_top_without_focus`]) so it stays above other windows and a
//!   click on Cancel does not activate it. **macOS**: a pop-up-level
//!   window. Both open on the primary display (no pointer lookup here);
//!   neither was run for this change (the Windows part is compile-checked
//!   only).

use std::time::{Duration, Instant};

use gpui::{
    point, px, size, AnyWindowHandle, AppContext, Bounds, Context, Pixels, Point, Size,
    WindowBackgroundAppearance, WindowBounds, WindowDecorations, WindowHandle, WindowKind,
    WindowOptions,
};
use starling_dictation::settings::OverlayMode;

use crate::activation::Readiness;
use crate::app::StarlingApp;
use crate::views::overlay::OverlayView;

/// The overlay window's app id (Wayland) / WM class.
pub(crate) const OVERLAY_APP_ID: &str = "starling-overlay";

/// How long a finished, cancelled or delivered take stays on the overlay.
pub(crate) const TERMINAL_LINGER: Duration = Duration::from_millis(1500);

/// How long a failed insertion stays on the overlay.
pub(crate) const FAILURE_LINGER: Duration = Duration::from_secs(6);

/// Distance from the bottom edge of the monitor, in logical pixels.
const BOTTOM_MARGIN: f32 = 72.;

/// What the overlay says.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OverlayPhase {
    /// The take started; no captured samples yet.
    StartingMic,
    /// Real samples arrived.
    Listening,
    /// Stopped: the take is being saved and recognized.
    Finishing,
    /// The transcript is being processed (#295).
    Processing,
    /// The text is being inserted into the target (#220).
    Delivering,
    /// The take's transcript is in Starling.
    Ready,
    /// The text landed in the target.
    Delivered,
    /// Inserting failed; the main window has the recovery.
    DeliveryFailed,
    /// The take ended without a transcript: cancelled, or its save or
    /// recognition failed. The main window says which.
    Stopped,
}

impl OverlayPhase {
    /// The take is still recording: the level meter and Cancel apply.
    pub(crate) fn is_recording(self) -> bool {
        matches!(self, OverlayPhase::StartingMic | OverlayPhase::Listening)
    }

    pub(crate) fn label(self) -> &'static str {
        match self {
            OverlayPhase::StartingMic => "Starting the microphone…",
            OverlayPhase::Listening => "Listening",
            OverlayPhase::Finishing => "Finishing recognition…",
            OverlayPhase::Processing => "Processing…",
            OverlayPhase::Delivering => "Inserting…",
            OverlayPhase::Ready => "Transcript ready in Starling",
            OverlayPhase::Delivered => "Inserted",
            OverlayPhase::DeliveryFailed => "Insert failed — your text is in Starling",
            OverlayPhase::Stopped => "Stopped — see Starling",
        }
    }
}

/// How delivering the take's text is going. Fed by the insertion path
/// (`StarlingApp::set_delivery_status`); the overlay only displays it.
/// Nothing feeds it yet: the insertion path (#220) lands separately.
#[allow(dead_code)]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) enum DeliveryStatus {
    #[default]
    Idle,
    Delivering,
    Delivered,
    /// Why it failed, in the user's words.
    Failed(String),
}

/// Which take the overlay follows once it stopped recording.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Follow {
    None,
    /// Finished at this instant (the take's identity until it has a
    /// session id); its save has not landed yet.
    Saving(Instant),
    /// Saved under this session id.
    Take(String),
}

/// Where the followed take is, as the app reports it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Pipeline {
    /// Recognition is running, or its result has not been handed to
    /// processing yet.
    pub recognizing: bool,
    pub processing: bool,
}

/// The overlay's pure state: take lifecycle events and pipeline
/// snapshots in, the phase to show out. Time comes in as arguments.
#[derive(Debug)]
pub(crate) struct OverlayModel {
    follow: Follow,
    /// A terminal phase and when it was entered.
    terminal: Option<(OverlayPhase, Instant)>,
    delivery: DeliveryStatus,
    delivery_since: Instant,
}

impl OverlayModel {
    pub(crate) fn new(now: Instant) -> Self {
        Self {
            follow: Follow::None,
            terminal: None,
            delivery: DeliveryStatus::Idle,
            delivery_since: now,
        }
    }

    /// A new take started: whatever the overlay showed is replaced.
    pub(crate) fn take_started(&mut self, now: Instant) {
        self.follow = Follow::None;
        self.terminal = None;
        self.delivery = DeliveryStatus::Idle;
        self.delivery_since = now;
    }

    /// The take that stopped at `stopped_at` is being saved for
    /// recognition.
    pub(crate) fn take_finished(&mut self, stopped_at: Instant) {
        self.follow = Follow::Saving(stopped_at);
        self.terminal = None;
    }

    /// Whether a finished take's save is in flight.
    pub(crate) fn is_saving(&self) -> bool {
        matches!(self.follow, Follow::Saving(_))
    }

    /// The active take ended without being transcribed.
    pub(crate) fn take_cancelled(&mut self, now: Instant) {
        self.follow = Follow::None;
        self.terminal = Some((OverlayPhase::Stopped, now));
    }

    /// The take that stopped at `stopped_at` was saved as `id`. A save
    /// for any other take (an older one, an import) changes nothing.
    pub(crate) fn take_saved(&mut self, stopped_at: Instant, id: &str) {
        if self.follow == Follow::Saving(stopped_at) {
            self.follow = Follow::Take(id.to_string());
        }
    }

    /// The save of the take that stopped at `stopped_at` failed.
    pub(crate) fn save_failed(&mut self, stopped_at: Instant, now: Instant) {
        if self.follow == Follow::Saving(stopped_at) {
            self.follow = Follow::None;
            self.terminal = Some((OverlayPhase::Stopped, now));
        }
    }

    /// The session the overlay follows, if its save landed.
    pub(crate) fn followed_take(&self) -> Option<&str> {
        match &self.follow {
            Follow::Take(id) => Some(id),
            _ => None,
        }
    }

    pub(crate) fn set_delivery(&mut self, status: DeliveryStatus, now: Instant) {
        if self.delivery != status {
            self.delivery = status;
            self.delivery_since = now;
        }
    }

    pub(crate) fn delivery(&self) -> &DeliveryStatus {
        &self.delivery
    }

    /// The phase to show, or `None` when the overlay has nothing to say.
    /// `pipeline` describes the followed take (see
    /// [`OverlayModel::followed_take`]); `failed` says whether its
    /// recognition ended in an error.
    pub(crate) fn phase(
        &mut self,
        readiness: Option<Readiness>,
        pipeline: Pipeline,
        failed: bool,
        now: Instant,
    ) -> Option<OverlayPhase> {
        match readiness {
            Some(Readiness::Starting) => return Some(OverlayPhase::StartingMic),
            Some(Readiness::Listening) => return Some(OverlayPhase::Listening),
            None => {}
        }
        if self.delivery == DeliveryStatus::Delivering {
            return Some(OverlayPhase::Delivering);
        }
        match &self.follow {
            Follow::Saving(_) => return Some(OverlayPhase::Finishing),
            Follow::Take(_) if pipeline.processing => return Some(OverlayPhase::Processing),
            Follow::Take(_) if pipeline.recognizing => return Some(OverlayPhase::Finishing),
            Follow::Take(_) => {
                self.follow = Follow::None;
                let done = if failed {
                    OverlayPhase::Stopped
                } else {
                    OverlayPhase::Ready
                };
                self.terminal = Some((done, now));
            }
            Follow::None => {}
        }
        let since_delivery = now.saturating_duration_since(self.delivery_since);
        match self.delivery {
            DeliveryStatus::Failed(_) if since_delivery < FAILURE_LINGER => {
                return Some(OverlayPhase::DeliveryFailed);
            }
            DeliveryStatus::Delivered if since_delivery < TERMINAL_LINGER => {
                return Some(OverlayPhase::Delivered);
            }
            _ => {}
        }
        match self.terminal {
            Some((phase, since)) if now.saturating_duration_since(since) < TERMINAL_LINGER => {
                Some(phase)
            }
            _ => {
                self.terminal = None;
                None
            }
        }
    }
}

/// Whether the overlay window should be open.
pub(crate) fn overlay_visible(mode: OverlayMode, phase: Option<OverlayPhase>) -> bool {
    mode != OverlayMode::Hidden && phase.is_some()
}

/// The overlay's size for `mode`, in logical pixels.
pub(crate) fn overlay_size(mode: OverlayMode) -> Size<Pixels> {
    match mode {
        OverlayMode::LiveText => size(px(440.), px(132.)),
        OverlayMode::Minimal | OverlayMode::Hidden => size(px(340.), px(56.)),
    }
}

/// The monitors (primary first) and the pointer, in device pixels.
type MonitorsAndPointer = (Vec<MonitorRect>, Option<(f32, f32)>);

/// A monitor's rectangle, in the units of the pointer that comes with it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct MonitorRect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl MonitorRect {
    fn contains(&self, (x, y): (f32, f32)) -> bool {
        x >= self.x && x < self.x + self.width && y >= self.y && y < self.y + self.height
    }
}

/// Where the overlay opens: bottom-centre of the monitor under the
/// pointer, else of the first monitor (callers list the primary first).
/// `monitors` and `pointer` are in device pixels; `scale` converts them
/// to the logical pixels the result (and `overlay`) are in. The overlay
/// is kept inside the monitor even when it is larger than the margin
/// allows.
pub(crate) fn place_overlay(
    monitors: &[MonitorRect],
    pointer: Option<(f32, f32)>,
    overlay: Size<Pixels>,
    scale: f32,
) -> Option<Point<Pixels>> {
    let scale = if scale.is_finite() && scale > 0. {
        scale
    } else {
        1.
    };
    let monitor = pointer
        .and_then(|pointer| monitors.iter().find(|monitor| monitor.contains(pointer)))
        .or_else(|| monitors.first())?;
    let (left, top) = (monitor.x / scale, monitor.y / scale);
    let (width, height) = (monitor.width / scale, monitor.height / scale);
    let (w, h) = (f32::from(overlay.width), f32::from(overlay.height));
    let x = left + ((width - w) / 2.).max(0.);
    let y = top + (height - h - BOTTOM_MARGIN).max(0.);
    Some(point(px(x.round()), px(y.round())))
}

/// The monitors (primary first) and the pointer of an X11 session, in
/// device pixels. `None` when this is not an X11 session (gpui uses
/// Wayland whenever `WAYLAND_DISPLAY` is set) or the server can't say.
#[cfg(target_os = "linux")]
fn x11_monitors_and_pointer() -> Option<MonitorsAndPointer> {
    use x11rb::connection::Connection;
    use x11rb::protocol::randr::ConnectionExt as _;
    use x11rb::protocol::xproto::ConnectionExt as _;

    if std::env::var_os("WAYLAND_DISPLAY").is_some() || std::env::var_os("DISPLAY").is_none() {
        return None;
    }
    let (conn, screen) = x11rb::connect(None).ok()?;
    let root = conn.setup().roots.get(screen)?;
    let pointer = conn
        .query_pointer(root.root)
        .ok()
        .and_then(|cookie| cookie.reply().ok())
        .map(|reply| (f32::from(reply.root_x), f32::from(reply.root_y)));
    let mut monitors: Vec<(bool, MonitorRect)> = conn
        .randr_get_monitors(root.root, true)
        .ok()
        .and_then(|cookie| cookie.reply().ok())
        .map(|reply| {
            reply
                .monitors
                .iter()
                .map(|monitor| {
                    (
                        monitor.primary,
                        MonitorRect {
                            x: f32::from(monitor.x),
                            y: f32::from(monitor.y),
                            width: f32::from(monitor.width),
                            height: f32::from(monitor.height),
                        },
                    )
                })
                .collect()
        })
        .unwrap_or_default();
    // Primary first, so a pointer outside every monitor lands there.
    monitors.sort_by_key(|(primary, _)| !primary);
    let mut monitors: Vec<MonitorRect> = monitors.into_iter().map(|(_, rect)| rect).collect();
    if monitors.is_empty() {
        monitors.push(MonitorRect {
            x: 0.,
            y: 0.,
            width: f32::from(root.width_in_pixels),
            height: f32::from(root.height_in_pixels),
        });
    }
    Some((monitors, pointer))
}

#[cfg(not(target_os = "linux"))]
fn x11_monitors_and_pointer() -> Option<MonitorsAndPointer> {
    None
}

/// The overlay window and the model behind it.
pub(crate) struct Overlay {
    pub(crate) model: OverlayModel,
    /// The phase the window shows, refreshed by [`StarlingApp::sync_overlay`].
    pub(crate) phase: Option<OverlayPhase>,
    window: Option<WindowHandle<OverlayView>>,
    /// The mode the open window was sized for.
    window_mode: OverlayMode,
    /// An open in flight for this mode (the placement lookup runs off the
    /// UI thread).
    opening: Option<OverlayMode>,
    /// Bumped whenever the window should go away, so an open that was
    /// still in flight never shows a window.
    generation: u64,
    /// The main window's scale factor, for placing on X11.
    pub(crate) scale: f32,
    /// The staging draft of the take on the overlay (staged dictation):
    /// the live text shows that draft and no other.
    pub(crate) staging_token: Option<u64>,
}

impl Overlay {
    pub(crate) fn new(now: Instant) -> Self {
        Self {
            model: OverlayModel::new(now),
            phase: None,
            window: None,
            window_mode: OverlayMode::Minimal,
            opening: None,
            generation: 0,
            scale: 1.,
            staging_token: None,
        }
    }

    /// Whether `handle` is the overlay window.
    pub(crate) fn is_overlay_window(handle: AnyWindowHandle) -> bool {
        handle.downcast::<OverlayView>().is_some()
    }
}

// ---- App glue -------------------------------------------------------------

impl StarlingApp {
    /// What the insertion path reports about delivering the take's text
    /// (#220); the overlay shows it.
    #[allow(dead_code)]
    pub(crate) fn set_delivery_status(&mut self, status: DeliveryStatus, cx: &mut Context<Self>) {
        self.overlay.model.set_delivery(status, Instant::now());
        self.sync_overlay(cx);
    }

    /// A take started recording: the overlay follows it, and its staging
    /// draft when it has one.
    pub(crate) fn overlay_take_started(&mut self) {
        self.overlay.model.take_started(Instant::now());
        self.overlay.staging_token = self
            .staging
            .as_ref()
            .filter(|staging| staging.phase == crate::staging::StagingPhase::Recording)
            .map(|staging| staging.token);
    }

    /// The followed take's pipeline, as far as the app knows it.
    fn overlay_pipeline(&self) -> (Pipeline, bool) {
        let Some(id) = self.overlay.model.followed_take() else {
            return (Pipeline::default(), false);
        };
        // A staged draft that could not be rebased keeps its take's stop
        // instant; it is not recognizing any more.
        let staging_failed = self
            .staging
            .iter()
            .chain(&self.background_stagings)
            .any(|staging| {
                staging.take_id.as_deref() == Some(id)
                    && staging.phase == crate::staging::StagingPhase::Failed
            });
        let pipeline = Pipeline {
            // `stop_instants` holds a saved take until its transcript is
            // handed to processing (or processing is skipped).
            recognizing: self.active_ids.contains(id)
                || (self.stop_instants.contains_key(id) && !staging_failed),
            processing: self.processing_jobs.contains_key(id),
        };
        // A transcribed take is in the listing with its transcript by the
        // time it leaves recognition (the listing refresh comes first).
        let failed = staging_failed
            || !self
                .sessions
                .iter()
                .any(|session| session.id == id && session.transcript.is_some());
        (pipeline, failed)
    }

    /// Re-derives the overlay's phase and opens or closes its window.
    /// Called from the activation loop and after every take event; it
    /// never touches recording or transcript state.
    pub(crate) fn sync_overlay(&mut self, cx: &mut Context<Self>) {
        let (pipeline, failed) = self.overlay_pipeline();
        let phase = self.overlay.model.phase(
            self.activation.readiness(),
            pipeline,
            failed,
            Instant::now(),
        );
        let changed = phase != self.overlay.phase;
        self.overlay.phase = phase;
        let mode = self.feedback.overlay;
        let visible = overlay_visible(mode, phase);
        let stale_window =
            self.overlay.window.is_some() && (!visible || self.overlay.window_mode != mode);
        let stale_open = self
            .overlay
            .opening
            .is_some_and(|opening| !visible || opening != mode);
        if stale_window || stale_open {
            self.close_overlay(cx);
        }
        if visible && self.overlay.window.is_none() && self.overlay.opening.is_none() {
            self.open_overlay(mode, cx);
        }
        if changed {
            if let Some(window) = self.overlay.window {
                window.update(cx, |_, _, cx| cx.notify()).ok();
            }
        }
    }

    fn close_overlay(&mut self, cx: &mut Context<Self>) {
        self.overlay.generation += 1;
        self.overlay.opening = None;
        if let Some(window) = self.overlay.window.take() {
            window
                .update(cx, |_, window, _| window.remove_window())
                .ok();
        }
    }

    fn open_overlay(&mut self, mode: OverlayMode, cx: &mut Context<Self>) {
        self.overlay.opening = Some(mode);
        let generation = self.overlay.generation;
        let scale = self.overlay.scale;
        let app = cx.entity().downgrade();
        cx.spawn(async move |this, cx| {
            let screen = cx
                .background_spawn(async move { x11_monitors_and_pointer() })
                .await;
            let overlay_size = overlay_size(mode);
            let origin = match screen {
                Some((monitors, pointer)) => place_overlay(&monitors, pointer, overlay_size, scale),
                None => cx
                    .update(|cx| {
                        cx.primary_display().and_then(|display| {
                            let bounds = display.bounds();
                            let monitor = MonitorRect {
                                x: f32::from(bounds.origin.x),
                                y: f32::from(bounds.origin.y),
                                width: f32::from(bounds.size.width),
                                height: f32::from(bounds.size.height),
                            };
                            place_overlay(&[monitor], None, overlay_size, 1.)
                        })
                    })
                    .ok()
                    .flatten(),
            };
            let display_id = cx
                .update(|cx| cx.primary_display().map(|display| display.id()))
                .ok()
                .flatten();
            let options = WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(Bounds::new(
                    origin.unwrap_or_else(|| point(px(0.), px(0.))),
                    overlay_size,
                ))),
                titlebar: None,
                focus: false,
                show: true,
                kind: WindowKind::PopUp,
                is_movable: false,
                is_resizable: false,
                is_minimizable: false,
                display_id: if origin.is_some() && screen_is_x11() {
                    None
                } else {
                    display_id
                },
                window_background: WindowBackgroundAppearance::Opaque,
                app_id: Some(OVERLAY_APP_ID.to_string()),
                window_min_size: None,
                window_decorations: Some(WindowDecorations::Client),
                tabbing_identifier: None,
            };
            // Hidden, re-moded or done while the placement was looked up:
            // never map a window nobody wants.
            let current = this
                .update(cx, |this, _| this.overlay.generation == generation)
                .unwrap_or(false);
            if !current {
                return;
            }
            let opened = cx.open_window(options, |_window, cx| {
                cx.new(|cx| OverlayView::new(app, cx))
            });
            this.update(cx, |this, cx| {
                let current = this.overlay.generation == generation;
                if current {
                    this.overlay.opening = None;
                }
                match opened {
                    Ok(window) if current => {
                        #[cfg(target_os = "windows")]
                        window
                            .update(cx, |_, window, _| keep_on_top_without_focus(window))
                            .ok();
                        this.overlay.window = Some(window);
                        this.overlay.window_mode = mode;
                    }
                    // Closed (or re-moded) while the open was in flight.
                    Ok(window) => {
                        window
                            .update(cx, |_, window, _| window.remove_window())
                            .ok();
                    }
                    Err(err) => eprintln!("Could not open the dictation overlay: {err}"),
                }
                // The phase may have moved on while the window opened.
                this.sync_overlay(cx);
            })
            .ok();
        })
        .detach();
    }
}

/// gpui's Windows pop-up is neither topmost nor non-activating: adds both,
/// so the overlay stays above the app being dictated into and clicking it
/// never takes that app's focus.
#[cfg(target_os = "windows")]
fn keep_on_top_without_focus(window: &gpui::Window) {
    use raw_window_handle::{HasWindowHandle, RawWindowHandle};
    use windows::Win32::Foundation::HWND;
    use windows::Win32::UI::WindowsAndMessaging::{
        GetWindowLongPtrW, SetWindowLongPtrW, SetWindowPos, GWL_EXSTYLE, HWND_TOPMOST,
        SWP_NOACTIVATE, SWP_NOMOVE, SWP_NOSIZE, WS_EX_NOACTIVATE, WS_EX_TOPMOST,
    };
    let Ok(handle) = window.window_handle() else {
        return;
    };
    let RawWindowHandle::Win32(handle) = handle.as_raw() else {
        return;
    };
    let hwnd = HWND(handle.hwnd.get() as *mut core::ffi::c_void);
    // SAFETY: `hwnd` is the live overlay window, owned by this thread.
    unsafe {
        let style = GetWindowLongPtrW(hwnd, GWL_EXSTYLE);
        SetWindowLongPtrW(
            hwnd,
            GWL_EXSTYLE,
            style | (WS_EX_NOACTIVATE.0 | WS_EX_TOPMOST.0) as isize,
        );
        let _ = SetWindowPos(
            hwnd,
            Some(HWND_TOPMOST),
            0,
            0,
            0,
            0,
            SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE,
        );
    }
}

/// Whether gpui runs on X11 here (its Linux backend choice).
fn screen_is_x11() -> bool {
    cfg!(target_os = "linux")
        && std::env::var_os("WAYLAND_DISPLAY").is_none()
        && std::env::var_os("DISPLAY").is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ms(n: u64) -> Duration {
        Duration::from_millis(n)
    }

    fn idle() -> Pipeline {
        Pipeline::default()
    }

    const RECOGNIZING: Pipeline = Pipeline {
        recognizing: true,
        processing: false,
    };

    const PROCESSING: Pipeline = Pipeline {
        recognizing: false,
        processing: true,
    };

    #[test]
    fn readiness_decides_the_recording_phases() {
        let t0 = Instant::now();
        let mut model = OverlayModel::new(t0);
        model.take_started(t0);
        assert_eq!(
            model.phase(Some(Readiness::Starting), idle(), false, t0),
            Some(OverlayPhase::StartingMic)
        );
        assert_eq!(
            model.phase(Some(Readiness::Listening), idle(), false, t0),
            Some(OverlayPhase::Listening)
        );
        assert_eq!(
            model.phase(None, idle(), false, t0),
            None,
            "nothing to follow"
        );
    }

    #[test]
    fn a_finished_take_is_followed_through_save_recognition_and_processing() {
        let t0 = Instant::now();
        let mut model = OverlayModel::new(t0);
        model.take_started(t0);
        model.take_finished(t0);
        assert!(model.is_saving());
        // Saving: no id yet, the pipeline says nothing about this take.
        assert_eq!(
            model.phase(None, idle(), false, t0),
            Some(OverlayPhase::Finishing)
        );
        // An import's save landing meanwhile is not this take.
        model.take_saved(t0 - ms(1), "import");
        assert_eq!(model.followed_take(), None);
        model.take_saved(t0, "take-1");
        assert_eq!(model.followed_take(), Some("take-1"));
        assert_eq!(
            model.phase(None, RECOGNIZING, false, t0),
            Some(OverlayPhase::Finishing)
        );
        assert_eq!(
            model.phase(None, PROCESSING, false, t0),
            Some(OverlayPhase::Processing)
        );
        // Done: shown for the linger, then gone.
        let t1 = t0 + ms(10);
        assert_eq!(
            model.phase(None, idle(), false, t1),
            Some(OverlayPhase::Ready)
        );
        assert_eq!(model.followed_take(), None);
        assert_eq!(
            model.phase(None, idle(), false, t1 + TERMINAL_LINGER - ms(1)),
            Some(OverlayPhase::Ready)
        );
        assert_eq!(model.phase(None, idle(), false, t1 + TERMINAL_LINGER), None);
    }

    #[test]
    fn a_failed_recognition_or_save_ends_as_stopped_not_ready() {
        let t0 = Instant::now();
        let mut model = OverlayModel::new(t0);
        model.take_finished(t0);
        model.take_saved(t0, "take-1");
        assert_eq!(
            model.phase(None, idle(), true, t0),
            Some(OverlayPhase::Stopped)
        );

        let mut model = OverlayModel::new(t0);
        model.take_finished(t0);
        // Another take's failure is not this one's.
        model.save_failed(t0 + ms(1), t0);
        assert!(model.is_saving());
        model.save_failed(t0, t0);
        assert!(!model.is_saving());
        assert_eq!(
            model.phase(None, idle(), false, t0),
            Some(OverlayPhase::Stopped)
        );
    }

    #[test]
    fn a_cancel_shows_stopped_briefly_and_follows_nothing() {
        let t0 = Instant::now();
        let mut model = OverlayModel::new(t0);
        model.take_started(t0);
        model.take_cancelled(t0);
        assert_eq!(
            model.phase(None, RECOGNIZING, false, t0),
            Some(OverlayPhase::Stopped)
        );
        assert_eq!(model.phase(None, idle(), false, t0 + TERMINAL_LINGER), None);
        // A late save for the cancelled take's predecessor is not adopted.
        model.take_saved(t0, "old");
        assert_eq!(model.followed_take(), None);
    }

    #[test]
    fn a_new_take_replaces_the_previous_one_on_the_overlay() {
        let t0 = Instant::now();
        let mut model = OverlayModel::new(t0);
        model.take_finished(t0);
        model.take_saved(t0, "take-1");
        model.take_started(t0);
        assert_eq!(model.followed_take(), None);
        // Take 1 still recognizing in the background does not show.
        assert_eq!(
            model.phase(Some(Readiness::Starting), RECOGNIZING, false, t0),
            Some(OverlayPhase::StartingMic)
        );
        model.take_saved(t0, "late");
        assert_eq!(
            model.followed_take(),
            None,
            "a save the overlay did not wait for"
        );
    }

    #[test]
    fn delivery_states_come_from_the_insertion_path() {
        let t0 = Instant::now();
        let mut model = OverlayModel::new(t0);
        model.take_finished(t0);
        model.take_saved(t0, "take-1");
        model.set_delivery(DeliveryStatus::Delivering, t0);
        assert_eq!(
            model.phase(None, idle(), false, t0),
            Some(OverlayPhase::Delivering)
        );
        let t1 = t0 + ms(5);
        model.set_delivery(DeliveryStatus::Failed("target closed".into()), t1);
        // Failure outranks the plain "ready" terminal and lingers longer.
        assert_eq!(
            model.phase(None, idle(), false, t1),
            Some(OverlayPhase::DeliveryFailed)
        );
        assert_eq!(
            model.phase(None, idle(), false, t1 + FAILURE_LINGER - ms(1)),
            Some(OverlayPhase::DeliveryFailed)
        );
        assert_eq!(model.phase(None, idle(), false, t1 + FAILURE_LINGER), None);

        model.set_delivery(DeliveryStatus::Delivered, t1);
        assert_eq!(
            model.phase(None, idle(), false, t1),
            Some(OverlayPhase::Delivered)
        );
        // Recording outranks any delivery state.
        assert_eq!(
            model.phase(Some(Readiness::Starting), idle(), false, t1),
            Some(OverlayPhase::StartingMic)
        );
        // A new take resets the delivery state.
        model.take_started(t1);
        assert_eq!(model.delivery(), &DeliveryStatus::Idle);
    }

    #[test]
    fn the_hidden_mode_never_opens_a_window() {
        for phase in [
            None,
            Some(OverlayPhase::Listening),
            Some(OverlayPhase::Ready),
        ] {
            assert!(!overlay_visible(OverlayMode::Hidden, phase));
        }
        assert!(overlay_visible(
            OverlayMode::Minimal,
            Some(OverlayPhase::StartingMic)
        ));
        assert!(overlay_visible(
            OverlayMode::LiveText,
            Some(OverlayPhase::Finishing)
        ));
        assert!(!overlay_visible(OverlayMode::Minimal, None));
    }

    fn monitor(x: f32, y: f32, width: f32, height: f32) -> MonitorRect {
        MonitorRect {
            x,
            y,
            width,
            height,
        }
    }

    #[test]
    fn the_overlay_opens_bottom_centre_on_the_pointer_monitor() {
        let monitors = [
            monitor(0., 0., 1920., 1080.),
            monitor(1920., 0., 2560., 1440.),
        ];
        let overlay = size(px(340.), px(56.));
        let on_second = place_overlay(&monitors, Some((2000., 500.)), overlay, 1.).unwrap();
        assert_eq!(
            on_second,
            point(px(1920. + 1110.), px(1440. - 56. - BOTTOM_MARGIN))
        );
        let on_first = place_overlay(&monitors, Some((10., 10.)), overlay, 1.).unwrap();
        assert_eq!(on_first, point(px(790.), px(1080. - 56. - BOTTOM_MARGIN)));
        // No pointer, or a pointer outside every monitor: the first one.
        assert_eq!(place_overlay(&monitors, None, overlay, 1.), Some(on_first));
        assert_eq!(
            place_overlay(&monitors, Some((-5., 9000.)), overlay, 1.),
            Some(on_first)
        );
        assert_eq!(place_overlay(&[], Some((1., 1.)), overlay, 1.), None);
    }

    #[test]
    fn placement_converts_device_pixels_with_the_scale_factor() {
        // A 4K monitor right of a 1080p one, at scale 2: logical 1920x1080
        // starting at logical x 960.
        let monitors = [
            monitor(0., 0., 1920., 1080.),
            monitor(1920., 0., 3840., 2160.),
        ];
        let overlay = size(px(340.), px(56.));
        let placed = place_overlay(&monitors, Some((3000., 100.)), overlay, 2.).unwrap();
        assert_eq!(
            placed,
            point(px(960. + 790.), px(1080. - 56. - BOTTOM_MARGIN))
        );
        // A nonsense scale is treated as 1.
        assert_eq!(
            place_overlay(&monitors[..1], None, overlay, 0.),
            place_overlay(&monitors[..1], None, overlay, 1.)
        );
    }

    #[test]
    fn a_monitor_smaller_than_the_overlay_keeps_it_on_screen() {
        let tiny = [monitor(100., 50., 200., 40.)];
        let placed = place_overlay(&tiny, None, size(px(340.), px(56.)), 1.).unwrap();
        assert_eq!(placed, point(px(100.), px(50.)));
    }
}
