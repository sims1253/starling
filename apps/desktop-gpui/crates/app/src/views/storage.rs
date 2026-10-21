//! The settings dialog's history-audio section (#342): what finished
//! recordings are kept as, and the opt-in retention limits.

use gpui::{Context, Div, FontWeight, SharedString, div, prelude::*, px};
use starling_dictation::settings::{RetentionLimits, StorageSettings};

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
    let draft = app.audio_upkeep.draft;
    let standard = limit_rows(
        "standard",
        draft.standard,
        |settings| &mut settings.standard,
        cx,
    );
    let archival = limit_rows(
        "archival",
        draft.archival,
        |settings| &mut settings.archival,
        cx,
    );
    let limited = [draft.standard, draft.archival]
        .iter()
        .any(|limits| limits.max_age_days.is_some() || limits.max_total_mb.is_some());
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
        .flex()
        .flex_col()
        .child(
            div()
                .mb(px(13.))
                .text_size(px(theme::TEXT_MD))
                .line_height(px(theme::TEXT_MD * 1.65))
                .text_color(theme::SHEET_QUILL)
                .child(
                    "Finished recordings are kept losslessly compressed (FLAC): retries send \
                     exactly the audio the first transcription did. Nothing is removed unless \
                     you set a limit; a limit removes audio only and keeps transcripts, never \
                     touches recordings from the last day, ones still transcribing, or ones \
                     that never got a transcript.",
                ),
        )
        .child(standard)
        .child(
            div()
                .mt(px(16.))
                .mb(px(10.))
                .text_size(px(theme::TEXT_MD))
                .line_height(px(theme::TEXT_MD * 1.65))
                .text_color(theme::SHEET_QUILL)
                .child(
                    "Archived recordings (Archive in a take's drawer) follow their own limits. \
                     They are kept as FLAC too; a smaller lossy archival format is not available \
                     yet.",
                ),
        )
        .child(archival)
        .child(div().mt(px(12.)).child(referenced_row));
    if let Some(report) = app.audio_upkeep.last_report.clone() {
        section = section.child(
            div()
                .id("history-audio-report")
                .mt(px(12.))
                .bg(theme::SHEET_CALLOUT)
                .p(px(12.))
                .text_size(px(theme::TEXT_SM))
                .line_height(px(theme::TEXT_SM * 1.55))
                .child(format!("Last cleanup: {report}")),
        );
    }
    section
}

/// The age and size chip rows for one class; `field` picks the class's
/// limits out of the draft.
fn limit_rows(
    class: &'static str,
    limits: RetentionLimits,
    field: fn(&mut StorageSettings) -> &mut RetentionLimits,
    cx: &mut Context<StarlingApp>,
) -> Div {
    let mut age_row = div().flex().flex_row().flex_wrap().gap(px(6.));
    for (days, label) in AGE_CHOICES {
        age_row = age_row.child(
            chip(
                SharedString::from(format!("retention-{class}-age-{label}")),
                limits.max_age_days == days,
                label,
            )
            .on_click(cx.listener(move |this, _, _window, cx| {
                field(&mut this.audio_upkeep.draft).max_age_days = days;
                cx.notify();
            })),
        );
    }
    let mut size_row = div().flex().flex_row().flex_wrap().gap(px(6.));
    for (mb, label) in SIZE_CHOICES {
        size_row = size_row.child(
            chip(
                SharedString::from(format!("retention-{class}-size-{label}")),
                limits.max_total_mb == mb,
                label,
            )
            .on_click(cx.listener(move |this, _, _window, cx| {
                field(&mut this.audio_upkeep.draft).max_total_mb = mb;
                cx.notify();
            })),
        );
    }
    let (age_label, size_label) = if class == "archival" {
        ("Remove archived audio older than", "Keep at most, archived")
    } else {
        ("Remove audio older than", "Keep at most")
    };
    div()
        .child(row_label(age_label))
        .child(age_row)
        .child(row_label(size_label).mt(px(12.)))
        .child(size_row)
}

fn row_label(text: &'static str) -> Div {
    div()
        .mb(px(6.))
        .text_size(px(theme::TEXT_MD))
        .font_weight(FontWeight::SEMIBOLD)
        .child(text)
}

fn chip(id: SharedString, selected: bool, label: &'static str) -> gpui::Stateful<Div> {
    div()
        .id(id)
        .px(px(10.))
        .py(px(6.))
        .rounded(px(6.))
        .border_1()
        .border_color(if selected {
            theme::SHEET_BEAK
        } else {
            theme::SHEET_LINE
        })
        .when(selected, |chip| chip.font_weight(FontWeight::SEMIBOLD))
        .cursor_pointer()
        .hover(|style| style.bg(theme::SHEET_HOVER))
        .text_size(px(theme::TEXT_SM))
        .child(label)
}
