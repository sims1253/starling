//! The dictation overlay window's view (#221): phase, input level, the
//! active microphone and Cancel, plus the live transcript in the
//! live-text mode. Everything here is read from the app on each frame;
//! the only thing the overlay can do is Cancel, which goes through the
//! activation machine like Escape.

use std::cell::RefCell;
use std::time::Duration;

use gpui::{
    div, prelude::*, pulsating_between, px, Animation, AnimationExt, App, Context, FontWeight,
    Subscription, WeakEntity, Window,
};
use starling_dictation::settings::OverlayMode;

use crate::app::StarlingApp;
use crate::overlay::OverlayPhase;
use crate::theme;

/// How much of the live transcript the overlay shows: its tail.
const LIVE_TEXT_CHARS: usize = 180;

/// Bars in the level meter.
const METER_BARS: usize = 12;

/// What one overlay frame shows.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct OverlayState {
    pub phase: OverlayPhase,
    pub mode: OverlayMode,
    /// The microphone the take records from.
    pub microphone: Option<String>,
    /// The input level, 0..=1.
    pub level: f32,
    /// The live transcript's tail (live-text mode only).
    pub text: Option<String>,
    /// Why delivery failed.
    pub failure: Option<String>,
}

/// An RMS level in 0..=1 on a -60..0 dBFS scale.
pub(crate) fn meter_level(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.;
    }
    let mean_square = samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32;
    let db = 10. * mean_square.max(1e-12).log10();
    ((db + 60.) / 60.).clamp(0., 1.)
}

/// The last `max` characters of `text`, starting at a word when one
/// starts close enough, with an ellipsis when anything was cut.
pub(crate) fn text_tail(text: &str, max: usize) -> String {
    let text = text.trim();
    let count = text.chars().count();
    if count <= max {
        return text.to_string();
    }
    let tail: String = text.chars().skip(count - max).collect();
    let tail = match tail.find(char::is_whitespace) {
        Some(space) if space < 24 => tail[space..].trim_start().to_string(),
        _ => tail,
    };
    format!("…{tail}")
}

/// The live text to show: the followed take's own text while it can still
/// be read (`current`), the text last shown for it once it cannot.
pub(crate) fn live_text(current: Option<String>, shown: &RefCell<String>) -> String {
    match current {
        Some(text) => {
            shown.replace(text.clone());
            text
        }
        None => shown.borrow().clone(),
    }
}

impl StarlingApp {
    /// The overlay's current frame, or `None` when it has nothing to show.
    pub(crate) fn overlay_state(&self, cx: &App) -> Option<OverlayState> {
        let phase = self.overlay.phase?;
        let recording = phase.is_recording();
        let microphone = match self.recorder.as_ref() {
            Some(handle) if recording => Some(crate::mic::device_name(handle)),
            _ if recording => Some(
                self.microphone_settings
                    .preferred_device
                    .clone()
                    .unwrap_or_else(|| "System default microphone".to_string()),
            ),
            _ => None,
        };
        let level = match self.recorder.as_ref() {
            Some(handle) if phase == OverlayPhase::Listening => {
                meter_level(&handle.latest_window(1024))
            }
            _ => 0.,
        };
        let text = (self.feedback.overlay == OverlayMode::LiveText).then(|| {
            let current = match self.overlay.staging_token {
                // The take's own staging draft (read-only here: the
                // staging editor owns edits) wherever the main window put
                // it, until it is dismissed.
                Some(token) => self
                    .staging
                    .iter()
                    .chain(&self.background_stagings)
                    .find(|staging| staging.token == token)
                    .map(|staging| {
                        text_tail(&staging.editor.read(cx).buffer.text, LIVE_TEXT_CHARS)
                    }),
                // The direct-mode partial, cleared when the take stops.
                None => self
                    .recorder
                    .is_some()
                    .then(|| text_tail(&self.live_partial, LIVE_TEXT_CHARS)),
            };
            live_text(current, &self.overlay.live_text)
        });
        let failure = match self.overlay.model.delivery() {
            crate::overlay::DeliveryStatus::Failed(reason) => Some(reason.clone()),
            _ => None,
        };
        Some(OverlayState {
            phase,
            mode: self.feedback.overlay,
            microphone,
            level,
            text,
            failure,
        })
    }

    /// Cancel from the overlay: the same path as Escape.
    fn overlay_cancel(&mut self, cx: &mut Context<Self>) {
        self.flush_system_events(cx);
        self.activation_input(|machine| machine.escape(), cx);
    }
}

pub(crate) struct OverlayView {
    app: WeakEntity<StarlingApp>,
    _observe: Option<Subscription>,
    _activation: Subscription,
    _release: Subscription,
}

impl OverlayView {
    pub(crate) fn new(
        app: WeakEntity<StarlingApp>,
        window: &mut gpui::Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let observe = app
            .upgrade()
            .map(|app| cx.observe(&app, |_, _, cx| cx.notify()));
        // Delivery counts the overlay as Starling's own window should a
        // compositor give it focus (#221).
        let activation = cx.observe_window_activation(window, |view, window, cx| {
            let active = window.is_window_active();
            view.app
                .update(cx, |app, cx| app.delivery_overlay_activation(active, cx))
                .ok();
        });
        // A window the compositor closes reports no focus loss.
        let handle = window.window_handle();
        let release = cx.on_release(move |view, cx| {
            view.app
                .update(cx, |app, cx| app.overlay_window_released(handle, cx))
                .ok();
        });
        Self {
            app,
            _observe: observe,
            _activation: activation,
            _release: release,
        }
    }
}

fn phase_color(phase: OverlayPhase) -> gpui::Rgba {
    match phase {
        OverlayPhase::Listening => theme::LIME,
        OverlayPhase::StartingMic
        | OverlayPhase::Finishing
        | OverlayPhase::Processing
        | OverlayPhase::Delivering
        | OverlayPhase::InsertWaiting => theme::AMBER,
        OverlayPhase::Ready | OverlayPhase::Delivered => theme::LIME,
        OverlayPhase::DeliveryFailed | OverlayPhase::Stopped => theme::CORAL,
    }
}

impl Render for OverlayView {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let root = div()
            .id("overlay-root")
            .size_full()
            .bg(theme::BG)
            .border_1()
            .border_color(theme::RECORD_BORDER)
            .text_color(theme::INK)
            .px(px(14.))
            .py(px(9.))
            .flex()
            .flex_col()
            .gap(px(6.));
        let Some(app) = self.app.upgrade() else {
            return root;
        };
        let Some(state) = app.read(cx).overlay_state(cx) else {
            return root;
        };
        if state.phase.is_recording() {
            // The level meter moves with the audio.
            window.request_animation_frame();
        }
        let color = phase_color(state.phase);
        let busy = !matches!(
            state.phase,
            OverlayPhase::Listening
                | OverlayPhase::InsertWaiting
                | OverlayPhase::Ready
                | OverlayPhase::Delivered
                | OverlayPhase::DeliveryFailed
                | OverlayPhase::Stopped
        );
        let dot = div().size(px(9.)).flex_none().rounded(px(5.)).bg(color);
        let dot = if busy {
            dot.with_animation(
                "overlay-dot",
                Animation::new(Duration::from_millis(900))
                    .repeat()
                    .with_easing(pulsating_between(0.3, 1.0)),
                |dot, delta| dot.opacity(delta),
            )
            .into_any_element()
        } else {
            dot.into_any_element()
        };

        let lit = (state.level * METER_BARS as f32).round() as usize;
        let meter = div()
            .id("overlay-meter")
            .flex()
            .flex_row()
            .items_end()
            .gap(px(2.))
            .h(px(14.))
            .children((0..METER_BARS).map(|bar| {
                let height = 4. + 10. * (bar as f32 + 1.) / METER_BARS as f32;
                div()
                    .w(px(3.))
                    .h(px(height))
                    .rounded(px(1.))
                    .bg(if bar < lit { theme::LIME } else { theme::LINE })
            }));

        let weak = self.app.clone();
        let cancel = div()
            .id("overlay-cancel")
            .flex_none()
            .px(px(8.))
            .py(px(3.))
            .rounded(px(3.))
            .border_1()
            .border_color(theme::RECORD_BORDER)
            .text_size(px(10.))
            .text_color(theme::MUTED)
            .cursor_pointer()
            .hover(|style| style.bg(theme::GEAR_HOVER_BG).text_color(theme::INK))
            .child("Cancel · Esc")
            .on_click(move |_, _window, cx| {
                if let Some(app) = weak.upgrade() {
                    app.update(cx, |app, cx| app.overlay_cancel(cx));
                }
            });

        let headline = div()
            .flex()
            .flex_row()
            .items_center()
            .gap(px(9.))
            .child(dot)
            .child(
                div()
                    .flex_1()
                    .min_w_0()
                    .flex()
                    .flex_col()
                    .child(
                        div()
                            .text_size(px(12.))
                            .font_weight(FontWeight::SEMIBOLD)
                            .truncate()
                            .child(state.phase.label()),
                    )
                    .when_some(
                        state.microphone.clone().or(state.failure.clone()),
                        |column, detail| {
                            column.child(
                                div()
                                    .text_size(px(10.))
                                    .text_color(theme::MUTED)
                                    .truncate()
                                    .child(detail),
                            )
                        },
                    ),
            )
            .when(state.phase == OverlayPhase::Listening, |row| {
                row.child(meter)
            })
            .when(state.phase.is_recording(), |row| row.child(cancel));

        root.child(headline).when_some(state.text, |root, text| {
            root.child(
                div()
                    .id("overlay-live-text")
                    .flex_1()
                    .min_h_0()
                    .overflow_hidden()
                    .pt(px(6.))
                    .border_t_1()
                    .border_color(theme::LINE)
                    .font(theme::serif_font())
                    .text_size(px(14.))
                    .line_height(px(14. * 1.4))
                    .text_color(if text.is_empty() {
                        theme::DIM
                    } else {
                        theme::INK
                    })
                    .child(if text.is_empty() {
                        "Your words appear here as you speak.".to_string()
                    } else {
                        text
                    }),
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_meter_maps_rms_onto_sixty_decibels() {
        assert_eq!(meter_level(&[]), 0.);
        assert_eq!(meter_level(&[0.; 64]), 0.);
        assert!((meter_level(&[1.; 64]) - 1.).abs() < 1e-6);
        // -20 dBFS sits two thirds up the scale.
        let quiet = vec![0.1f32; 64];
        assert!((meter_level(&quiet) - 2. / 3.).abs() < 1e-3);
    }

    #[test]
    fn the_live_text_keeps_the_newest_words() {
        assert_eq!(text_tail("  short text ", 50), "short text");
        let long = "alpha beta gamma delta epsilon zeta eta theta";
        let tail = text_tail(long, 20);
        assert!(tail.starts_with('…'), "{tail}");
        assert!(long.ends_with(tail.trim_start_matches('…')), "{tail}");
        assert!(tail.chars().count() <= 21, "{tail}");
        // Starts at a word boundary when one is near.
        assert!(!tail.trim_start_matches('…').starts_with(' '), "{tail}");
        // Multi-byte text is cut on characters.
        assert_eq!(text_tail("äöüäöü", 3), "…äöü");
    }

    #[test]
    fn the_live_text_stays_once_its_source_is_gone() {
        let shown = RefCell::default();
        assert_eq!(live_text(Some("all nouns".into()), &shown), "all nouns");
        // The take stopped (partial cleared) or its draft was dismissed.
        assert_eq!(live_text(None, &shown), "all nouns");
        // A source that is still there is shown as it is, even emptied.
        assert_eq!(live_text(Some(String::new()), &shown), "");
        assert_eq!(live_text(None, &shown), "");
    }
}
