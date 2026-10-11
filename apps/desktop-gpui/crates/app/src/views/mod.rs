//! Per-region view builders. Each takes the app entity state plus its
//! context and returns one region of the layout, mirroring the JSX in
//! `apps/desktop/src/App.tsx` and the metrics in `styles.css`.

mod capture;
mod delivery;
pub(crate) mod drawer;
mod feedback;
mod history;
mod live_preview;
pub(crate) mod microphone;
pub(crate) mod overlay;
mod settings;
pub(crate) mod staging;
mod storage;
mod system_check;
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

use starling_dictation::engine::{EnginePhase, EngineSnapshot};
use starling_dictation::settings::EngineMode;

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

/// What the connection indicator shows for the built-in engine (#362):
/// the dot's state plus the label beside it, derived purely from the
/// engine snapshot so the mapping is testable without a manager. The
/// label is the raw sentence — `connection_label` applies the topbar's
/// mono uppercase styling like every other status.
pub(crate) struct EngineStatusView {
    pub connection: Connection,
    pub label: String,
}

/// Builtin-mode indicator from the engine snapshot (#362): Ready shows
/// the model label and the device the engine actually runs on; the
/// bring-up phases show their stage word; Restarting names the attempt;
/// NoModel asks for a model (steady amber — nothing is broken, but
/// nothing can transcribe); Failed shows the failure's own sentence —
/// each variant carries what to do next.
pub(crate) fn engine_status_view(snapshot: &EngineSnapshot) -> EngineStatusView {
    let checking = |label: String| EngineStatusView {
        connection: Connection::Checking,
        label,
    };
    match &snapshot.phase {
        EnginePhase::Ready => {
            let label = snapshot.active.as_ref().map(|active| {
                let model = snapshot
                    .models
                    .iter()
                    .find(|model| model.id == active.model_id)
                    .map(|model| model.label.clone())
                    .unwrap_or_else(|| active.model_id.clone());
                match active.device.as_deref() {
                    Some(device) => format!("{model} · {device}"),
                    None => model,
                }
            });
            EngineStatusView {
                connection: Connection::Ready,
                label: label.unwrap_or_else(|| "engine ready".to_string()),
            }
        }
        EnginePhase::SelectingBackend => checking("selecting engine".to_string()),
        EnginePhase::Starting => checking("starting".to_string()),
        EnginePhase::Loading => checking("loading".to_string()),
        EnginePhase::Warming => checking("warming".to_string()),
        EnginePhase::Restarting { attempt, .. } => {
            checking(format!("Engine restarting (attempt {attempt})"))
        }
        EnginePhase::NoModel => EngineStatusView {
            connection: Connection::Busy,
            label: "Pick a model in Settings".to_string(),
        },
        EnginePhase::Failed(failure) => EngineStatusView {
            connection: Connection::Offline,
            label: failure.to_string(),
        },
    }
}

pub(crate) fn connection_label(app: &StarlingApp) -> String {
    match app.engine_settings.mode {
        // #362: in builtin mode the indicator is the engine's own state —
        // the manual health probe never runs here, so it never writes
        // this dot either. With no engine at all, why is the sentence (it
        // already says what to do); before the recording service reported
        // the engine, it is still starting — unless the connection to it
        // was lost, which took its report with it (the dot says Offline).
        EngineMode::Builtin => match app.engine_snapshot() {
            Some(snapshot) => engine_status_view(&snapshot).label,
            None if app.engine_status.is_none() && app.connection == Connection::Offline => {
                "offline".to_string()
            }
            None if app.engine_status.is_none() => "starting".to_string(),
            None => app.engine_unavailable(),
        }
        .to_uppercase(),
        EngineMode::Manual => match app.connection {
            Connection::Ready => format!("{} ready", app.server_model),
            Connection::Busy => format!("{} working", app.server_model),
            Connection::Checking => "checking".to_string(),
            Connection::Offline => "offline".to_string(),
        }
        .to_uppercase(),
    }
}

