//! The settings modal: dark scrim over the workspace, light centered card
//! with endpoint/model/terms fields, protocol toggle, callout and footer.

use std::time::Duration;

use gpui::{
    Animation, AnimationExt, ClickEvent, Context, Div, FontWeight, MouseButton, Stateful, Window,
    div, prelude::*, px, rgba,
};
use starling_dictation::settings;
use starling_dictation::store_v2::MigrationReport;

use crate::app::{Connection, StarlingApp};
use crate::migration::MigrationPhase;
use crate::theme;
use crate::views::{
    SETTINGS_CALLOUT_DOT_ID, icon, protocol_option_id, status_dot,
};

pub fn render_settings_modal(
    app: &mut StarlingApp,
    _window: &mut Window,
    cx: &mut Context<StarlingApp>,
) -> impl IntoElement {
    let endpoint_for_callout = app.endpoint.clone();
    let connection = app.connection;
    let draft_endpoint = app.draft_endpoint.clone();
    let draft_model = app.draft_model.clone();
    let draft_terms = app.draft_terms.clone();
    let selected_protocol = app.settings_protocol;

    let card = div()
        .id("settings-card")
        .on_mouse_down(
            MouseButton::Left,
            cx.listener(|_, _, _window, cx| {
                cx.stop_propagation();
            }),
        )
        .relative()
        .w(px(510.))
        .max_w_full()
        .max_h(px(680.))
        .overflow_y_scroll()
        .bg(theme::PAPER_LIGHT)
        .text_color(theme::SETTINGS_INK)
        .p(px(33.))
        .shadow(vec![gpui::BoxShadow {
            color: rgba(0x00000099).into(),
            offset: gpui::point(px(0.), px(30.)),
            blur_radius: px(100.),
            spread_radius: px(0.),
        }])
        .child(
            div()
                .id("settings-close")
                .absolute()
                .top(px(18.))
                .right(px(18.))
                .cursor_pointer()
                .text_color(theme::SETTINGS_MUTED)
                .on_click(cx.listener(|this, _, _window, cx| {
                    this.close_settings(cx);
                }))
                .child(icon("icons/x.svg", 18., theme::SETTINGS_MUTED)),
        )
        .child(
            div()
                .mb(px(13.))
                .font(theme::mono_font())
                .text_size(px(10.))
                .font_weight(FontWeight::MEDIUM)
                .text_color(theme::SETTINGS_EYEBROW)
                .child("CONNECTION"),
        )
        .child(
            div()
                .mb(px(13.))
                .font(theme::serif_font())
                .font_weight(FontWeight::MEDIUM)
                .text_size(px(33.))
                .line_height(px(33.))
                .child("Transcription server"),
        )
        .child(
            div()
                .mb(px(25.))
                .text_size(px(11.))
                .line_height(px(11. * 1.65))
                .text_color(theme::SETTINGS_MUTED)
                .child(
                    "Choose a starling-serve endpoint or an OpenAI-compatible transcription \
                     endpoint.",
                ),
        )
        .child(field_label("Server endpoint").child(draft_endpoint))
        .child(
            div()
                .flex()
                .flex_row()
                .gap(px(12.))
                .child(
                    field_label("API format")
                        .flex_1()
                        .min_w_0()
                        .mt(px(0.))
                        .child(protocol_toggle(selected_protocol, cx)),
                )
                .child(
                    field_label("Model")
                        .flex_1()
                        .min_w_0()
                        .mt(px(0.))
                        .child(draft_model),
                ),
        )
        .child(
            field_label("Words to watch").child(draft_terms).child(
                div()
                    .mt(px(2.))
                    .font_weight(FontWeight::NORMAL)
                    .line_height(px(10. * 1.5))
                    .text_color(theme::SETTINGS_HELPER)
                    .child(
                        "Comma-separated terms are checked after transcription. They are never \
                         inserted or substituted.",
                    ),
            ),
        )
        .child(storage_section(app, cx))
        .child(
            div()
                .flex()
                .flex_row()
                .items_center()
                .gap(px(12.))
                .bg(theme::SETTINGS_CALLOUT)
                .p(px(12.))
                .mt(px(21.))
                .child(status_dot(SETTINGS_CALLOUT_DOT_ID, connection, false))
                .child(
                    div()
                        .flex()
                        .flex_col()
                        .gap(px(2.))
                        .text_size(px(10.))
                        .child(div().font_weight(FontWeight::SEMIBOLD).child(
                            if connection == Connection::Ready {
                                "Server connected"
                            } else {
                                "Server needs attention"
                            },
                        ))
                        .child(
                            div()
                                .font(theme::mono_font())
                                .text_size(px(9.))
                                .text_color(theme::SETTINGS_CALLOUT_ENDPOINT)
                                .child(endpoint_for_callout),
                        ),
                ),
        )
        .child(
            div()
                .flex()
                .flex_row()
                .justify_end()
                .gap(px(9.))
                .mt(px(25.))
                .child(
                    div()
                        .id("test-connection")
                        .px(px(13.))
                        .py(px(9.))
                        .rounded(px(3.))
                        .border_1()
                        .border_color(theme::SETTINGS_FOOT_LINE)
                        .text_size(px(10.))
                        .cursor_pointer()
                        .hover(|style| style.bg(theme::PAPER_HOVER))
                        .on_click(cx.listener(|this, _, _window, cx| {
                            this.test_connection(cx);
                        }))
                        .child("Test connection"),
                )
                .child(
                    div()
                        .id("save-settings")
                        .px(px(13.))
                        .py(px(9.))
                        .rounded(px(3.))
                        .border_1()
                        .border_color(theme::SETTINGS_INK)
                        .bg(theme::SETTINGS_INK)
                        .text_color(theme::SETTINGS_PRIMARY_TEXT)
                        .text_size(px(10.))
                        .cursor_pointer()
                        .hover(|style| style.opacity(0.9))
                        .on_click(cx.listener(|this, _, _window, cx| {
                            this.save_settings(cx);
                        }))
                        .child("Save settings"),
                ),
        );

    let animated_card = card.with_animation(
        "modal-fade",
        Animation::new(Duration::from_millis(200)),
        |el, delta| el.opacity(delta),
    );

    div()
        .id("settings-layer")
        .absolute()
        .top(px(0.))
        .bottom(px(0.))
        .left(px(0.))
        .right(px(0.))
        .bg(theme::SCRIM)
        .flex()
        .items_center()
        .justify_center()
        .p(px(20.))
        .on_mouse_down(
            MouseButton::Left,
            cx.listener(|this, _, _window, cx| {
                this.close_settings(cx);
            }),
        )
        .child(animated_card)
}

fn field_label(label: &str) -> Div {
    div()
        .flex()
        .flex_col()
        .gap(px(7.))
        .my(px(16.))
        .text_size(px(10.))
        .font_weight(FontWeight::SEMIBOLD)
        .text_color(theme::SETTINGS_INK)
        .child(label.to_string())
}

// ---------------------------------------------------------------------------
// Storage (E02 cutover phase 1): the honest backend status line plus the
// reviewed migration flow — preview (dry run) → confirmed import →
// cutover or rollback. The v1 originals are never touched by any path.
// ---------------------------------------------------------------------------

fn storage_section(app: &mut StarlingApp, cx: &mut Context<StarlingApp>) -> Div {
    let kind = app.storage_kind;
    let reason = app.storage_reason;
    let phase = app.migration.phase;
    let busy = app.migration.busy;
    let report = app.migration.report.as_ref();
    let cutover_persisted = app.migration.cutover_persisted;
    let notice = app.migration.notice.clone();
    let cutover_ready = app.migration.cutover_ready();
    let rollback_ready = app.migration.rollback_ready();

    let mut section = div()
        .flex()
        .flex_col()
        .gap(px(7.))
        .my(px(16.))
        .text_size(px(10.))
        .font_weight(FontWeight::SEMIBOLD)
        .text_color(theme::SETTINGS_INK)
        .child("Recording storage")
        .child(
            div()
                .font(theme::mono_font())
                .text_size(px(9.))
                .font_weight(FontWeight::NORMAL)
                .text_color(theme::SETTINGS_MUTED)
                .child(format!(
                    "Active this session: {} — {}.",
                    kind.label(),
                    reason
                ))
                .when(kind == crate::store::StorageKind::V2, |line| {
                    line.child(format!(
                        " To fall back to v1, restart without {V2} set.",
                        V2 = starling_dictation::store_v2::STORAGE_V2_FLAG_ENV
                    ))
                }),
        );

    // The notice (cutover landed / rollback done) rides above the phase's
    // own screen.
    if let Some(notice) = notice {
        section = section.child(
            div()
                .text_size(px(9.))
                .font_weight(FontWeight::NORMAL)
                .line_height(px(9. * 1.5))
                .text_color(theme::SETTINGS_MUTED)
                .child(notice),
        );
    }

    if busy {
        section = section.child(
            div()
                .text_size(px(9.))
                .font_weight(FontWeight::NORMAL)
                .text_color(theme::SETTINGS_MUTED)
                .child(match phase {
                    MigrationPhase::ConfirmApply => "Importing into v2…",
                    MigrationPhase::ConfirmCutover => "Saving the storage choice…",
                    MigrationPhase::ConfirmRollback => "Rolling back the import…",
                    _ => "Preparing the preview…",
                }),
        );
        return section;
    }

    match phase {
        MigrationPhase::Idle => section
            .child(storage_help(
                "Import your v1 recordings into storage v2 with a preview first: counts, \
                 verified hashes, and anything too damaged to read — before anything is \
                 copied. The v1 originals are never modified or deleted.",
            ))
            .child(
                div()
                    .flex()
                    .flex_row()
                    .gap(px(9.))
                    .child(migration_button(cx, "migration-dry-run", "Preview import (dry run)", true, |this, _, _, cx| {
                        this.migration_start_dry_run(cx);
                    })),
            ),
        MigrationPhase::DryRunReady => {
            let mut block = section.child(
                div()
                    .text_size(px(9.))
                    .font_weight(FontWeight::NORMAL)
                    .line_height(px(9. * 1.6))
                    .text_color(theme::SETTINGS_INK)
                    .child(report_preview(report)),
            );
            if let Some(report) = report {
                let skipped = skipped_damaged_lines(report);
                if !skipped.is_empty() {
                    block = block.child(
                        div()
                            .text_size(px(9.))
                            .font_weight(FontWeight::NORMAL)
                            .line_height(px(9. * 1.5))
                            .text_color(theme::SETTINGS_MUTED)
                            .child(skipped),
                    );
                }
            }
            block.child(
                div()
                    .flex()
                    .flex_row()
                    .gap(px(9.))
                    .child(migration_button(cx, "migration-apply", "Import into v2", true, |this, _, _, cx| {
                        this.migration_arm_apply(cx);
                    }))
                    .child(migration_button(cx, "migration-redry", "Re-run preview", false, |this, _, _, cx| {
                        this.migration_start_dry_run(cx);
                    })),
            )
        }
        MigrationPhase::ConfirmApply => section
            .child(storage_help(&format!(
                "Import {} into storage v2? Journals and transcripts are copied and every \
                 recording is re-verified by content hash before the result is shown. The v1 \
                 originals are never modified; you can roll the import back.",
                import_scope(report)
            )))
            .child(
                div()
                    .flex()
                    .flex_row()
                    .gap(px(9.))
                    .child(migration_button(cx, "migration-apply-confirm", "Confirm import", true, |this, _, _, cx| {
                        this.migration_confirm_apply(cx);
                    }))
                    .child(migration_button(cx, "migration-apply-cancel", "Cancel", false, |this, _, _, cx| {
                        this.migration_cancel_apply(cx);
                    })),
            ),
        MigrationPhase::Applied => {
            let mut block = section.child(
                div()
                    .text_size(px(9.))
                    .font_weight(FontWeight::NORMAL)
                    .line_height(px(9. * 1.6))
                    .text_color(theme::SETTINGS_INK)
                    .child(report_result(report, cutover_persisted)),
            );
            if let Some(report) = report {
                let failures = failure_lines(report);
                if !failures.is_empty() {
                    block = block.child(
                        div()
                            .text_size(px(9.))
                            .font_weight(FontWeight::NORMAL)
                            .line_height(px(9. * 1.5))
                            .text_color(theme::SETTINGS_MUTED)
                            .child(failures),
                    );
                }
            }
            let mut buttons = div().flex().flex_row().gap(px(9.));
            if cutover_ready {
                buttons = buttons.child(migration_button(
                    cx,
                    "migration-cutover",
                    "Use v2 from now on",
                    true,
                    |this, _, _, cx| this.migration_arm_cutover(cx),
                ));
            }
            if rollback_ready {
                buttons = buttons.child(migration_button(
                    cx,
                    "migration-rollback",
                    "Roll back import",
                    false,
                    |this, _, _, cx| this.migration_arm_rollback(cx),
                ));
            }
            if cutover_persisted || rollback_ready {
                block = block.child(buttons);
            } else {
                block = block.child(
                    div()
                        .flex()
                        .flex_row()
                        .gap(px(9.))
                        .child(migration_button(cx, "migration-reset", "Start over", false, |this, _, _, cx| {
                            this.migration_reset(cx);
                        })),
                );
            }
            block
        }
        MigrationPhase::ConfirmCutover => section
            .child(storage_help(
                "Make storage v2 the store of record? The choice is saved and applies from \
                 the next start; this session keeps running as it is. The v1 originals stay \
                 on disk untouched, and rolling the import back later also switches you \
                 back.",
            ))
            .child(
                div()
                    .flex()
                    .flex_row()
                    .gap(px(9.))
                    .child(migration_button(cx, "migration-cutover-confirm", "Use v2 from now on", true, |this, _, _, cx| {
                        this.migration_confirm_cutover(cx);
                    }))
                    .child(migration_button(cx, "migration-cutover-cancel", "Cancel", false, |this, _, _, cx| {
                        this.migration_cancel_cutover(cx);
                    })),
            ),
        MigrationPhase::ConfirmRollback => section
            .child(storage_help(
                "Discard the imported recordings? Their rows are removed and their journals \
                 are quarantined (never deleted). The v1 originals were never touched and \
                 become the only copies again. A saved v2 choice is switched back to v1.",
            ))
            .child(
                div()
                    .flex()
                    .flex_row()
                    .gap(px(9.))
                    .child(migration_button(cx, "migration-rollback-confirm", "Confirm rollback", true, |this, _, _, cx| {
                        this.migration_confirm_rollback(cx);
                    }))
                    .child(migration_button(cx, "migration-rollback-cancel", "Cancel", false, |this, _, _, cx| {
                        this.migration_cancel_rollback(cx);
                    })),
            ),
    }
}

fn storage_help(text: &str) -> Div {
    div()
        .text_size(px(9.))
        .font_weight(FontWeight::NORMAL)
        .line_height(px(9. * 1.6))
        .text_color(theme::SETTINGS_MUTED)
        .child(text.to_string())
}

fn migration_button<F>(
    cx: &mut Context<StarlingApp>,
    id: &'static str,
    label: &str,
    primary: bool,
    on_click: F,
) -> Stateful<Div>
where
    F: Fn(&mut StarlingApp, &ClickEvent, &mut Window, &mut Context<StarlingApp>)
        + 'static
        + Clone,
{
    div()
        .id(id)
        .px(px(13.))
        .py(px(9.))
        .rounded(px(3.))
        .border_1()
        .border_color(if primary {
            theme::SETTINGS_INK
        } else {
            theme::SETTINGS_FOOT_LINE
        })
        .when(primary, |button| {
            button
                .bg(theme::SETTINGS_INK)
                .text_color(theme::SETTINGS_PRIMARY_TEXT)
        })
        .text_size(px(10.))
        .cursor_pointer()
        .hover(|style| {
            if primary {
                style.opacity(0.9)
            } else {
                style.bg(theme::PAPER_HOVER)
            }
        })
        .on_click(cx.listener(move |this, event: &ClickEvent, window, cx| {
            on_click(this, event, window, cx);
        }))
        .child(label.to_string())
}

/// The read-only dry-run report: counts and the verification basis.
fn report_preview(report: Option<&MigrationReport>) -> String {
    let Some(report) = report else {
        return "No preview yet.".to_string();
    };
    let counts = &report.counts;
    format!(
        "Preview (nothing copied yet): {} recording{} ready to import, each verified by \
         content hash after the copy. {} too damaged to read would be skipped.",
        counts.source_sessions,
        if counts.source_sessions == 1 { "" } else { "s" },
        counts.source_damaged,
    )
}

/// How many recordings the confirmed apply will copy (the dry-run scope).
fn import_scope(report: Option<&MigrationReport>) -> String {
    match report {
        Some(report) => format!(
            "{} recording{}",
            report.counts.source_sessions,
            if report.counts.source_sessions == 1 { "" } else { "s" }
        ),
        None => "the previewed recordings".to_string(),
    }
}

/// The applied report: what landed and what verified.
fn report_result(report: Option<&MigrationReport>, cutover_persisted: bool) -> String {
    let Some(report) = report else {
        return "No import result.".to_string();
    };
    let counts = &report.counts;
    let mut text = format!(
        "Import finished: {} of {} recordings copied, {} verified by content hash.",
        counts.imported, counts.source_sessions, counts.verified,
    );
    if counts.verified < counts.imported {
        text.push_str(
            " Some recordings did not verify — cutover stays disabled until a fully \
             verified import; rolling back is safe either way.",
        );
    }
    if cutover_persisted {
        text.push_str(" v2 is saved as your storage choice and takes over on the next start.");
    }
    text
}

/// Skipped-damaged entries, capped to keep the modal readable.
fn skipped_damaged_lines(report: &MigrationReport) -> String {
    const CAP: usize = 5;
    let mut lines: Vec<String> = report
        .skipped_damaged
        .iter()
        .take(CAP)
        .map(|skipped| format!("Skipped (kept as-is): {} — {}", skipped.id, skipped.reason))
        .collect();
    let rest = report.skipped_damaged.len().saturating_sub(CAP);
    if rest > 0 {
        lines.push(format!("… and {rest} more, listed in migration_report.json"));
    }
    lines.join("\n")
}

/// Per-record import failures, capped the same way.
fn failure_lines(report: &MigrationReport) -> String {
    const CAP: usize = 5;
    let failed: Vec<&starling_dictation::store_v2::MigratedRecord> = report
        .records
        .iter()
        .filter(|record| record.error.is_some() || record.verified == Some(false))
        .take(CAP)
        .collect();
    let mut lines: Vec<String> = failed
        .into_iter()
        .map(|record| {
            format!(
                "Not verified: {} — {}",
                record.id,
                record.error.as_deref().unwrap_or("content hash mismatch")
            )
        })
        .collect();
    let rest = report
        .records
        .iter()
        .filter(|record| record.error.is_some() || record.verified == Some(false))
        .count()
        .saturating_sub(CAP);
    if rest > 0 {
        lines.push(format!("… and {rest} more, listed in migration_report.json"));
    }
    lines.join("\n")
}

fn protocol_toggle(selected: settings::Protocol, cx: &mut Context<StarlingApp>) -> Div {
    let options = [
        (settings::Protocol::Starling, "Starling native"),
        (settings::Protocol::OpenAi, "OpenAI compatible"),
    ];
    let mut toggle = div()
        .flex()
        .flex_row()
        .w_full()
        .rounded(px(3.))
        .border_1()
        .border_color(theme::SETTINGS_LINE)
        .overflow_hidden();
    for (protocol, label) in options {
        let is_selected = selected == protocol;
        toggle = toggle.child(
            div()
                // R04: the id names the option and nothing else — the old
                // "protocol-selected"/"protocol-option" pair flipped both
                // options' identities on every click.
                .id(protocol_option_id(protocol))
                .flex()
                .flex_1()
                .items_center()
                .justify_center()
                .py(px(11.))
                .font(theme::mono_font())
                .text_size(px(11.))
                .text_color(if is_selected {
                    theme::SETTINGS_PRIMARY_TEXT
                } else {
                    theme::SETTINGS_INK
                })
                .bg(if is_selected {
                    theme::SETTINGS_INK
                } else {
                    theme::SETTINGS_FIELD_BG
                })
                .cursor_pointer()
                .on_click(cx.listener(move |this, _, _window, cx| {
                    this.settings_protocol = protocol;
                    cx.notify();
                }))
                .child(label),
        );
    }
    toggle
}
