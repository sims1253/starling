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
    Animation, AnimationExt, Context, ElementId, Rgba, Svg, Transformation, Window, div, point,
    prelude::*, pulsating_between, px, radians, svg,
};

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

pub(crate) fn spinner(id: &'static str, size: f32, color: Rgba) -> gpui::AnimationElement<Svg> {
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

/// The 7px status dot: lime (optionally with glow) when ready, blinking amber
/// when busy/checking, coral when offline.
pub(crate) fn status_dot(connection: Connection, glow: bool) -> gpui::AnyElement {
    let base = div().size(px(7.)).rounded_full();
    match connection {
        Connection::Ready => {
            let mut dot = base.bg(theme::LIME);
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
            .with_animation(
                ElementId::Name("dot-blink".into()),
                Animation::new(Duration::from_millis(1200))
                    .repeat()
                    .with_easing(pulsating_between(0.35, 1.0)),
                |el, delta| el.opacity(delta),
            )
            .into_any_element(),
        Connection::Offline => base.bg(theme::CORAL).into_any_element(),
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
