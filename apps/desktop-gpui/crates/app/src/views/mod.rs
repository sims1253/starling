//! Per-region view builders. Each takes the app entity state plus its
//! context and returns one region of the layout, mirroring the JSX in
//! `apps/desktop/src/App.tsx` and the metrics in `styles.css`.

mod capture;
mod drawer;
mod history;
mod settings;
mod topbar;

pub use capture::render_capture;
pub use drawer::render_drawer;
pub use history::render_history;
pub use settings::render_settings_modal;
pub use topbar::render_topbar;

use std::time::Duration;

use gpui::{
    Animation, AnimationExt, Context, ElementId, Rgba, SharedString, Svg, Transformation, Window,
    div, point, prelude::*, pulsating_between, px, radians, svg,
};
use starling_dictation::settings as dictation_settings;

use crate::app::{Connection, StarlingApp};
use crate::theme;

pub fn render_workspace(
    app: &mut StarlingApp,
    window: &mut Window,
    cx: &mut Context<StarlingApp>,
) -> impl IntoElement {
    div()
        .id("workspace")
        .flex()
        .flex_1()
        .min_h_0()
        .overflow_hidden()
        .child(render_capture(app, window, cx))
        .child(render_history(app, window, cx))
}

pub(crate) fn icon(path: &'static str, size: f32, color: Rgba) -> Svg {
    svg().path(path).size(px(size)).text_color(color)
}

pub(crate) fn spinner(
    id: impl Into<SharedString>,
    size: f32,
    color: Rgba,
) -> gpui::AnimationElement<Svg> {
    icon("icons/loader.svg", size, color).with_animation(
        ElementId::Name(id.into()),
        Animation::new(Duration::from_millis(1000)).repeat(),
        |el, delta| {
            el.with_transformation(Transformation::rotate(radians(
                delta * std::f32::consts::TAU,
            )))
        },
    )
}

/// The history rows' spinner id (R04): every transcribing row animates, so a
/// shared `"history-spinner"` name would put several elements under one id
/// in a single frame. Deriving it from the session id keeps the ids unique
/// and stable across re-renders — a row keeps its animation even when list
/// order changes around it.
pub(crate) fn history_spinner_id(session_id: &str) -> SharedString {
    SharedString::from(format!("history-spinner-{session_id}"))
}

/// The status dots' per-call-site ids (R04): the topbar dot and the settings
/// callout dot are on screen together while the modal is open, and both
/// blink when the connection is Busy/Checking, so they may never share the
/// `"dot-blink"` name.
pub(crate) const TOPBAR_STATUS_DOT_ID: &str = "topbar-status-dot";
pub(crate) const SETTINGS_CALLOUT_DOT_ID: &str = "settings-callout-dot";

/// The 7px status dot with its 4px halo ring (styles.css box-shadow): lime
/// (optionally with glow) when ready, blinking amber when busy/checking, coral
/// when offline. `id` names the call site (R04) — only the blinking branch
/// animates, but the id must be stable and unique per site regardless of
/// connection state.
pub(crate) fn status_dot(
    id: impl Into<SharedString>,
    connection: Connection,
    glow: bool,
) -> gpui::AnyElement {
    let halo = |color: u32| {
        vec![gpui::BoxShadow {
            color: gpui::rgba(color).into(),
            offset: point(px(0.), px(0.)),
            blur_radius: px(0.),
            spread_radius: px(4.),
        }]
    };
    let base = div().size(px(7.)).rounded_full();
    match connection {
        Connection::Ready => {
            let mut dot = base.bg(theme::LIME).shadow(halo(0xD9FF6A14));
            if glow {
                dot = dot.shadow(vec![
                    gpui::BoxShadow {
                        color: gpui::rgba(0xD9FF6A14).into(),
                        offset: point(px(0.), px(0.)),
                        blur_radius: px(0.),
                        spread_radius: px(4.),
                    },
                    gpui::BoxShadow {
                        color: gpui::rgba(0xD9FF6A4D).into(),
                        offset: point(px(0.), px(0.)),
                        blur_radius: px(13.),
                        spread_radius: px(0.),
                    },
                ]);
            }
            dot.into_any_element()
        }
        Connection::Busy | Connection::Checking => base
            .bg(theme::AMBER)
            .shadow(halo(0xEFC26B14))
            .with_animation(
                ElementId::Name(id.into()),
                Animation::new(Duration::from_millis(1200))
                    .repeat()
                    .with_easing(pulsating_between(0.35, 1.0)),
                |el, delta| el.opacity(delta),
            )
            .into_any_element(),
        Connection::Offline => base
            .bg(theme::CORAL)
            .shadow(halo(0xFF745B14))
            .into_any_element(),
    }
}

/// The settings modal's protocol-toggle option ids (R04): keyed by the
/// option, never by selection state. The old `"protocol-selected"` /
/// `"protocol-option"` pair flipped an option's element identity the moment
/// it was clicked, re-seating element state; a stable per-option id keeps
/// each half of the toggle itself across re-renders.
pub(crate) fn protocol_option_id(protocol: dictation_settings::Protocol) -> SharedString {
    SharedString::from(format!("protocol-option-{}", protocol_slug(protocol)))
}

fn protocol_slug(protocol: dictation_settings::Protocol) -> &'static str {
    match protocol {
        dictation_settings::Protocol::Starling => "starling",
        dictation_settings::Protocol::OpenAI => "openai",
    }
}

pub(crate) fn connection_label(app: &StarlingApp) -> String {
    match app.connection {
        Connection::Ready => format!("{} ready", app.server_model),
        Connection::Busy => format!("{} working", app.server_model),
        Connection::Checking => "checking".to_string(),
        Connection::Offline => "offline".to_string(),
    }
    .to_uppercase()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn history_spinner_ids_are_unique_per_session_and_stable_across_renders() {
        // R04: two transcribing rows must never animate under one shared
        // "history-spinner" id; each row's id derives from its session id,
        // so it also survives list reordering.
        let first = history_spinner_id("session-a");
        let second = history_spinner_id("session-b");
        assert_ne!(first, second);
        assert_eq!(first, history_spinner_id("session-a"));
        assert_eq!(second, history_spinner_id("session-b"));
    }

    #[test]
    fn the_two_status_dot_call_sites_never_share_an_id() {
        // R04: the topbar dot and the settings callout dot are both visible
        // while the modal is open, and both blink when Busy/Checking.
        assert_ne!(TOPBAR_STATUS_DOT_ID, SETTINGS_CALLOUT_DOT_ID);
    }

    #[test]
    fn protocol_option_ids_are_keyed_by_option_not_selection_state() {
        // R04: clicking an option must not re-seat either half of the
        // toggle — the ids depend only on which option they name, so they
        // are equal before and after the selection flips.
        let starling = protocol_option_id(dictation_settings::Protocol::Starling);
        let openai = protocol_option_id(dictation_settings::Protocol::OpenAI);
        assert_ne!(starling, openai);
        assert_eq!(
            starling,
            protocol_option_id(dictation_settings::Protocol::Starling)
        );
        assert_eq!(openai, protocol_option_id(dictation_settings::Protocol::OpenAI));
    }
}
