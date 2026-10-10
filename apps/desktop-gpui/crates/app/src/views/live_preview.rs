//! The settings dialog's LIVE PREVIEW section (#357): how early and how
//! often a take asks the server for live text.

use gpui::{div, prelude::*, px, Context, Div, FontWeight, SharedString};
use starling_dictation::settings::LivePreviewSettings;

use crate::app::StarlingApp;
use crate::theme;
use crate::views::settings::{choice_row, helper};

const CADENCE_CHOICES: [(LivePreviewSettings, &str, &str); 2] = [
    (
        LivePreviewSettings::FAST,
        "Early",
        "First words after about a second of speech, then updated every half second. Uses \
         about half a second of engine time per second spoken on a current Starling server.",
    ),
    (
        LivePreviewSettings::SERVER,
        "Server default",
        "Whatever the server is configured with.",
    ),
];

pub(crate) fn render_live_preview_section(
    app: &mut StarlingApp,
    cx: &mut Context<StarlingApp>,
) -> Div {
    let mut rows = div().flex().flex_col().gap(px(6.));
    let draft = app.draft_live_preview;
    for (cadence, name, description) in CADENCE_CHOICES {
        rows = rows.child(
            choice_row(
                SharedString::from(format!("live-preview-{name}")),
                draft == cadence,
                name,
                description,
                true,
            )
            .on_click(cx.listener(move |this, _, _window, cx| {
                this.draft_live_preview = cadence;
                cx.notify();
            })),
        );
    }
    // A hand-edited cadence stays as it is until a choice above is made.
    let custom = CADENCE_CHOICES.iter().all(|(cadence, ..)| *cadence != draft);
    let note = if custom {
        let (min, interval) = draft.effective();
        let seconds = |value: Option<f64>| {
            value.map_or_else(|| "server default".to_string(), |value| format!("{value} s"))
        };
        format!(
            "From the settings file: first words after {}, updates every {}. Servers that \
             predate this setting use their own.",
            seconds(min),
            seconds(interval)
        )
    } else {
        "Applies from the next take. Servers that predate this setting use their own.".to_string()
    };

    div()
        .mt(px(29.))
        .pt(px(21.))
        .border_t_1()
        .border_color(theme::SETTINGS_LINE)
        .child(
            div()
                .mb(px(13.))
                .font(theme::mono_font())
                .text_size(px(10.))
                .font_weight(FontWeight::MEDIUM)
                .text_color(theme::SETTINGS_EYEBROW)
                .child("LIVE PREVIEW"),
        )
        .child(rows)
        .child(div().mt(px(8.)).child(helper(note)))
}
