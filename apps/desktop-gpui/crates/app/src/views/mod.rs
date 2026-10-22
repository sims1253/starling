//! Per-region view builders. Each takes the app entity state plus its
//! context and returns one region of the layout: the bar across the top,
//! the takes column on the left, and the stage on the right with the
//! transcript sheet under it.

mod capture;
mod delivery;
pub(crate) mod drawer;
mod feedback;
mod history;
pub(crate) mod kit;
pub(crate) mod licenses;
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
pub use settings::{SettingsPage, render_settings_modal};
pub use topbar::render_topbar;

use std::time::Duration;

use gpui::{
    Animation, AnimationExt, Context, ElementId, Rgba, SharedString, Svg, Transformation, Window,
    div, prelude::*, pulsating_between, px, radians, svg,
};

use starling_dictation::engine::{EnginePhase, EngineSnapshot};
use starling_dictation::settings::EngineMode;

use crate::app::{Connection, StarlingApp};
use crate::theme;

/// Everything under the bar: the takes column, then the stage with the
/// selected take's transcript sheet under it. The sheet belongs to the
/// stage, not the window, so the list of takes keeps its full height.
pub fn render_workspace(
    app: &mut StarlingApp,
    window: &mut Window,
    cx: &mut Context<StarlingApp>,
) -> impl IntoElement {
    let has_transcript = app.selected().is_some();
    div()
        .id("workspace")
        .flex()
        .flex_row()
        .flex_1()
        .min_h_0()
        .overflow_hidden()
        .child(render_history(app, window, cx))
        .child(
            div()
                .flex()
                .flex_col()
                .flex_1()
                .min_w_0()
                .min_h_0()
                .child(render_capture(app, window, cx))
                .when(has_transcript, |stage| {
                    stage.child(render_drawer(app, window, cx))
                }),
        )
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

/// The status dot: teal when ready, a slow gold pulse while busy or
/// checking, rose when offline. `id` names the call site (R04) — only the
/// pulsing branch animates, but the id must be stable and unique per site
/// regardless of connection state.
pub(crate) fn status_dot(id: impl Into<SharedString>, connection: Connection) -> gpui::AnyElement {
    let base = div().size(px(8.)).flex_none().rounded_full();
    match connection {
        Connection::Ready => base.bg(theme::SHEEN).into_any_element(),
        Connection::Busy | Connection::Checking => base
            .bg(theme::BEAK)
            .with_animation(
                ElementId::Name(id.into()),
                Animation::new(Duration::from_millis(1400))
                    .repeat()
                    .with_easing(pulsating_between(0.35, 1.0)),
                |el, delta| el.opacity(delta),
            )
            .into_any_element(),
        Connection::Offline => base.bg(theme::ROSE).into_any_element(),
    }
}

/// What the connection indicator shows for the built-in engine (#362):
/// the dot's state plus the label beside it, derived purely from the
/// engine snapshot so the mapping is testable without a manager. The
/// label is the raw sentence — `connection_label` capitalizes it for the
/// bar like every other status.
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
                    Some(device) => format!("{model} on {device}"),
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
    let label = match app.engine_settings.mode {
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
        },
        EngineMode::Manual => match app.connection {
            Connection::Ready => format!("{} ready", app.server_model),
            Connection::Busy => format!("{} working", app.server_model),
            Connection::Checking => "checking".to_string(),
            Connection::Offline => "offline".to_string(),
        },
    };
    sentence_case(&label)
}

/// The first letter capitalized, the rest left as written: model names
/// and device names keep their own casing.
pub(crate) fn sentence_case(text: &str) -> String {
    let mut chars = text.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().chain(chars).collect(),
        None => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::sentence_case;

    #[test]
    fn a_status_reads_as_a_sentence_without_recasing_names() {
        assert_eq!(sentence_case("parakeet ready"), "Parakeet ready");
        assert_eq!(sentence_case("Parakeet TDT on Vulkan"), "Parakeet TDT on Vulkan");
        assert_eq!(sentence_case(""), "");
    }
}

