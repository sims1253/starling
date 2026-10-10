//! The settings dialog's FEEDBACK section (#221): the dictation overlay
//! mode and the optional start/stop cues with their volume and preview.

use gpui::{div, prelude::*, px, Context, Div, FontWeight, SharedString};
use starling_dictation::settings::OverlayMode;

use crate::app::StarlingApp;
use crate::theme;
use crate::views::settings::{choice_row, helper};

const OVERLAY_CHOICES: [(OverlayMode, &str, &str); 3] = [
    (
        OverlayMode::Hidden,
        "Hidden",
        "No overlay. The take shows in this window only; Escape still cancels.",
    ),
    (
        OverlayMode::Minimal,
        "Minimal status",
        "A small strip while you dictate: what the take is doing, the input level, the \
         microphone, and Cancel.",
    ),
    (
        OverlayMode::LiveText,
        "Live text",
        "The status strip plus your words as they are recognized. Edit them in this window; \
         the overlay only shows them.",
    ),
];

/// What the overlay can promise on this desktop.
fn overlay_reach_note() -> &'static str {
    if cfg!(target_os = "linux") && std::env::var_os("WAYLAND_DISPLAY").is_some() {
        "On Wayland your compositor places the overlay and decides whether it takes focus. Add a \
         rule for the app id \"starling-overlay\" that opens it floating and unfocused (niri: \
         open-floating true, open-focused false); without one, the overlay may take focus from \
         the app you dictate into."
    } else if cfg!(target_os = "linux") {
        "The overlay opens at the bottom of the monitor under the pointer and never takes \
         keyboard focus."
    } else {
        "The overlay opens at the bottom of the primary display and never takes keyboard focus."
    }
}

pub(crate) fn render_feedback_section(app: &mut StarlingApp, cx: &mut Context<StarlingApp>) -> Div {
    let mut rows = div().flex().flex_col().gap(px(6.));
    for (mode, name, description) in OVERLAY_CHOICES {
        let selected = app.draft_overlay_mode == mode;
        rows = rows.child(
            choice_row(
                SharedString::from(format!("overlay-{name}")),
                selected,
                name,
                description,
                true,
            )
            .on_click(cx.listener(move |this, _, _window, cx| {
                this.draft_overlay_mode = mode;
                cx.notify();
            })),
        );
    }

    let cues = app.draft_cues;
    let cue_row = choice_row(
        SharedString::from("start-stop-cues"),
        cues,
        "Start and stop sounds",
        "A short rising tone once the microphone is really listening, a falling one when it has \
         stopped. Never after a start that failed or a take cancelled before it was heard.",
        true,
    )
    .on_click(cx.listener(|this, _, _window, cx| {
        this.draft_cues = !this.draft_cues;
        cx.notify();
    }));

    let volume = app.draft_cue_volume.read(cx).value();
    let volume_row = div()
        .id("cue-volume")
        .flex()
        .flex_col()
        .gap(px(6.))
        .p(px(10.))
        .rounded(px(3.))
        .border_1()
        .border_color(if cues {
            theme::SETTINGS_INK
        } else {
            theme::SETTINGS_LINE
        })
        .when(!cues, |row| row.opacity(0.5))
        .child(
            div()
                .flex()
                .flex_row()
                .justify_between()
                .items_center()
                .text_size(px(11.))
                .font_weight(FontWeight::SEMIBOLD)
                .child("Sound volume")
                .child(
                    div()
                        .flex()
                        .flex_row()
                        .items_center()
                        .gap(px(9.))
                        .child(
                            div()
                                .font(theme::mono_font())
                                .font_weight(FontWeight::NORMAL)
                                .child(format!("{volume}%")),
                        )
                        .child(
                            div()
                                .id("cue-preview")
                                .px(px(8.))
                                .py(px(3.))
                                .rounded(px(3.))
                                .border_1()
                                .border_color(theme::PAPER_BUTTON_LINE)
                                .font_weight(FontWeight::MEDIUM)
                                .text_size(px(10.))
                                .cursor_pointer()
                                .hover(|style| style.bg(theme::PAPER_HOVER))
                                .child("Preview")
                                .on_click(cx.listener(|this, _, _window, cx| {
                                    this.preview_cues(cx);
                                })),
                        ),
                ),
        )
        .child(app.draft_cue_volume.clone())
        .child(helper(
            "The sounds stay audible while playback is lowered or muted during recording.",
        ));

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
                .child("FEEDBACK"),
        )
        .child(rows)
        .child(
            div()
                .id("overlay-reach")
                .mt(px(12.))
                .bg(theme::SETTINGS_CALLOUT)
                .p(px(12.))
                .text_size(px(10.))
                .line_height(px(10. * 1.55))
                .child(overlay_reach_note()),
        )
        .child(div().mt(px(12.)).child(cue_row))
        .child(div().mt(px(6.)).child(volume_row))
}
