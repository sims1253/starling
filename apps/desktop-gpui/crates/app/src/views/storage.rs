//! The settings dialog's history-audio section (#342): what finished
//! recordings are kept as, and the opt-in retention limits.

use gpui::{Context, Div, FontWeight, SharedString, div, prelude::*, px};
use starling_dictation::settings::RetentionLimits;

use crate::app::StarlingApp;
use crate::theme;

/// The age limits offered, in days (`None` keeps audio forever).
const AGE_CHOICES: [(Option<u32>, &str); 4] = [
    (None, "Keep"),
    (Some(30), "30 days"),
    (Some(90), "90 days"),
    (Some(365), "1 year"),
];

/// The size limits offered, in MiB.
const SIZE_CHOICES: [(Option<u64>, &str); 4] = [
    (None, "No limit"),
    (Some(1024), "1 GB"),
    (Some(5 * 1024), "5 GB"),
    (Some(20 * 1024), "20 GB"),
];

pub(crate) fn render_storage_section(app: &mut StarlingApp, cx: &mut Context<StarlingApp>) -> Div {
    let limits = app.audio_upkeep.draft.standard;
    let mut age_row = div().flex().flex_row().flex_wrap().gap(px(6.));
    for (days, label) in AGE_CHOICES {
        age_row = age_row.child(
            chip(
                SharedString::from(format!("retention-age-{label}")),
                limits.max_age_days == days,
                label,
            )
            .on_click(cx.listener(move |this, _, _window, cx| {
                this.audio_upkeep.draft.standard = RetentionLimits {
                    max_age_days: days,
                    ..this.audio_upkeep.draft.standard
                };
                cx.notify();
            })),
        );
    }
    let mut size_row = div().flex().flex_row().flex_wrap().gap(px(6.));
    for (mb, label) in SIZE_CHOICES {
        size_row = size_row.child(
            chip(
                SharedString::from(format!("retention-size-{label}")),
                limits.max_total_mb == mb,
                label,
            )
            .on_click(cx.listener(move |this, _, _window, cx| {
                this.audio_upkeep.draft.standard = RetentionLimits {
                    max_total_mb: mb,
                    ..this.audio_upkeep.draft.standard
                };
                cx.notify();
            })),
        );
    }
    let limited = limits.max_age_days.is_some() || limits.max_total_mb.is_some();
    let include_referenced = app.audio_upkeep.draft.include_referenced;
    let referenced_row = super::settings::choice_row(
        SharedString::from("retention-include-referenced"),
        include_referenced,
        "Also remove audio that documents or corrections use",
        "Off: recordings a document revision or a saved correction refers to keep their audio, \
         and the cleanup below says how many it kept.",
        limited,
    )
    .when(!limited, |row| row.opacity(0.5))
    .when(limited, |row| {
        row.on_click(cx.listener(|this, _, _window, cx| {
            this.audio_upkeep.draft.include_referenced =
                !this.audio_upkeep.draft.include_referenced;
            cx.notify();
        }))
    });

    let mut section = div()
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
                .child("HISTORY AUDIO"),
        )
        .child(
            div()
                .mb(px(13.))
                .text_size(px(11.))
                .line_height(px(11. * 1.65))
                .text_color(theme::SETTINGS_MUTED)
                .child(
                    "Finished recordings are kept losslessly compressed (FLAC): retries send \
                     exactly the audio the first transcription did. Nothing is removed unless \
                     you set a limit; a limit removes audio only and keeps transcripts, never \
                     touches recordings from the last day, ones still transcribing, or ones \
                     that never got a transcript.",
                ),
        )
        .child(row_label("Remove audio older than"))
        .child(age_row)
        .child(row_label("Keep at most").mt(px(12.)))
        .child(size_row)
        .child(div().mt(px(12.)).child(referenced_row));
    if let Some(report) = app.audio_upkeep.last_report.clone() {
        section = section.child(
            div()
                .id("history-audio-report")
                .mt(px(12.))
                .bg(theme::SETTINGS_CALLOUT)
                .p(px(12.))
                .text_size(px(10.))
                .line_height(px(10. * 1.55))
                .child(format!("Last cleanup: {report}")),
        );
    }
    section
}

fn row_label(text: &'static str) -> Div {
    div()
        .mb(px(6.))
        .text_size(px(11.))
        .font_weight(FontWeight::SEMIBOLD)
        .child(text)
}

fn chip(id: SharedString, selected: bool, label: &'static str) -> gpui::Stateful<Div> {
    div()
        .id(id)
        .px(px(10.))
        .py(px(6.))
        .rounded(px(3.))
        .border_1()
        .border_color(if selected {
            theme::SETTINGS_INK
        } else {
            theme::SETTINGS_LINE
        })
        .when(selected, |chip| chip.font_weight(FontWeight::SEMIBOLD))
        .cursor_pointer()
        .hover(|style| style.bg(theme::PAPER_HOVER))
        .text_size(px(10.))
        .child(label)
}
