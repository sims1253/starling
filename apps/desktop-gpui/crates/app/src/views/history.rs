//! The 350px history column: archive head, the session row list, and the
//! damaged-record rows G02 appends after them.

use gpui::{Context, Div, FontWeight, SharedString, Window, div, prelude::*, px};
use starling_dictation::storage::{DamagedRecord, SessionStatus, SessionSummary};

use crate::app::StarlingApp;
use crate::theme;
use crate::views::{history_spinner_id, icon, spinner};

pub fn render_history(
    app: &mut StarlingApp,
    _window: &mut Window,
    cx: &mut Context<StarlingApp>,
) -> impl IntoElement {
    let rows: Vec<_> = app
        .sessions
        .iter()
        .map(|session| row_data(app, session))
        .collect();
    let rows: Vec<_> = rows.iter().map(|data| render_row(data, cx)).collect();
    // G02: quarantined records stay visible with their reason, sorted after
    // every readable take.
    let damaged_rows: Vec<_> = app
        .damaged
        .iter()
        .map(|damaged| render_damaged_row(damaged, cx))
        .collect();

    div()
        .id("history-pane")
        .w(px(350.))
        .min_w_0()
        .flex()
        .flex_col()
        .min_h_0()
        .border_l_1()
        .border_color(theme::LINE)
        .bg(theme::PANEL_SOFT)
        .child(
            div()
                .h(px(92.))
                .flex()
                .flex_row()
                .items_center()
                .justify_between()
                .px(px(23.))
                .border_b_1()
                .border_color(theme::LINE)
                .text_color(theme::DIM)
                .child(
                    div()
                        .flex()
                        .flex_col()
                        .child(
                            div()
                                .mb(px(5.))
                                .font(theme::mono_font())
                                .text_size(px(10.))
                                .font_weight(FontWeight::MEDIUM)
                                .text_color(theme::EYEBROW)
                                .child("ARCHIVE"),
                        )
                        .child(
                            div()
                                .font(theme::serif_font())
                                .font_weight(FontWeight::MEDIUM)
                                .text_size(px(21.))
                                .text_color(theme::INK)
                                .child("Recent takes"),
                        ),
                )
                .child(icon("icons/clock.svg", 18., theme::DIM)),
        )
        .child(
            div()
                .id("history-list")
                .flex()
                .flex_col()
                .flex_1()
                .min_h_0()
                .overflow_y_scroll()
                .when(app.sessions.is_empty(), |list| {
                    list.child(
                        div()
                            .max_w(px(280.))
                            .px(px(24.))
                            .py(px(28.))
                            .text_size(px(12.))
                            .line_height(px(12. * 1.65))
                            .text_color(theme::DIM)
                            .child("Your recordings will collect here, ready to retry or export."),
                    )
                })
                .children(rows)
                .children(damaged_rows),
        )
}

struct RowData {
    id: String,
    active: bool,
    title: String,
    meta: String,
    status: SessionStatus,
}

fn row_data(app: &StarlingApp, session: &SessionSummary) -> RowData {
    let title = session
        .transcript
        .as_ref()
        .map(|transcript| transcript.text.lines().next().unwrap_or("").to_string())
        .filter(|line| !line.trim().is_empty())
        .unwrap_or_else(|| {
            if session.status == SessionStatus::Failed {
                "Saved. Retry available".to_string()
            } else if session.status == SessionStatus::Interrupted {
                // I1 phase 2: recovered from a journal / salvaged after a
                // quiesce timeout — the audio is here and retryable.
                "Recovered. Retry available".to_string()
            } else {
                "Sending to server…".to_string()
            }
        });

    let mut meta = format!(
        "{} · {}",
        theme::fmt_when(&session.created_at),
        theme::fmt_duration(session.duration_ms)
    );
    if session.attempt_count > 1 {
        meta.push_str(&format!(" · {} attempts", session.attempt_count));
    }

    RowData {
        id: session.id.clone(),
        active: app.selected_id.as_deref() == Some(session.id.as_str()),
        title,
        meta,
        status: session.status,
    }
}

fn render_row(data: &RowData, cx: &mut Context<StarlingApp>) -> gpui::Stateful<Div> {
    let id = data.id.clone();
    let active = data.active;

    let state = div()
        .size(px(16.))
        .flex()
        .items_center()
        .justify_center()
        .flex_none()
        .child(match data.status {
            SessionStatus::Transcribing => {
                // R04: one id per transcribing row, not a shared
                // "history-spinner" shared by all of them.
                spinner(history_spinner_id(&data.id), 14., theme::DIM).into_any_element()
            }
            SessionStatus::Transcribed => div()
                .size(px(6.))
                .rounded_full()
                .bg(theme::LIME)
                .into_any_element(),
            SessionStatus::Failed => div()
                .size(px(6.))
                .rounded_full()
                .bg(theme::CORAL)
                .into_any_element(),
            SessionStatus::Interrupted => div()
                .size(px(6.))
                .rounded_full()
                .bg(theme::AMBER)
                .into_any_element(),
            SessionStatus::Captured => div()
                .size(px(6.))
                .rounded_full()
                .bg(theme::DOT_GREY)
                .into_any_element(),
        });

    div()
        .id(SharedString::from(format!("history-row-{}", data.id)))
        .relative()
        .flex()
        .flex_row()
        .items_center()
        .gap(px(10.))
        .w_full()
        .min_h(px(78.))
        .px(px(18.))
        .py(px(15.))
        .border_b_1()
        .border_color(theme::LINE)
        .cursor_pointer()
        .hover(|style| style.bg(theme::HOVER_BG))
        .when(active, |row| row.bg(theme::ACTIVE_ROW_BG))
        .on_click(cx.listener(move |this, _, _window, cx| {
            this.select_session(id.clone(), cx);
        }))
        .when(active, |row| {
            row.child(
                div()
                    .absolute()
                    .top(px(0.))
                    .bottom(px(0.))
                    .left(px(0.))
                    .w(px(2.))
                    .bg(theme::LIME),
            )
        })
        .child(state)
        .child(
            div()
                .flex()
                .flex_col()
                .gap(px(6.))
                .flex_1()
                .min_w_0()
                .overflow_hidden()
                .child(
                    div()
                        .text_size(px(11.))
                        .text_color(theme::TAKE_TITLE)
                        .truncate()
                        .child(data.title.clone()),
                )
                .child(
                    div()
                        .font(theme::mono_font())
                        .text_size(px(9.))
                        .text_color(theme::DIM)
                        .child(data.meta.clone()),
                ),
        )
        .child(icon(
            "icons/chevron-right.svg",
            16.,
            if active { theme::LIME } else { theme::DIM },
        ))
}

/// A quarantined record (G02): visible, explained, never deletable by the
/// app itself. Clicking surfaces the recorded reason; the underlying files
/// stay untouched for manual recovery.
fn render_damaged_row(
    damaged: &DamagedRecord,
    cx: &mut Context<StarlingApp>,
) -> gpui::Stateful<Div> {
    let id = damaged.id.clone();

    div()
        .id(SharedString::from(format!("history-row-{}", damaged.id)))
        .relative()
        .flex()
        .flex_row()
        .items_center()
        .gap(px(10.))
        .w_full()
        .min_h(px(78.))
        .px(px(18.))
        .py(px(15.))
        .border_b_1()
        .border_color(theme::LINE)
        .opacity(0.8)
        .cursor_pointer()
        .hover(|style| style.bg(theme::HOVER_BG))
        .on_click(cx.listener(move |this, _, _window, cx| {
            this.surface_damage(&id, cx);
        }))
        .child(
            div()
                .size(px(16.))
                .flex()
                .items_center()
                .justify_center()
                .flex_none()
                .child(icon("icons/alert-circle.svg", 12., theme::CORAL)),
        )
        .child(
            div()
                .flex()
                .flex_col()
                .gap(px(6.))
                .flex_1()
                .min_w_0()
                .overflow_hidden()
                .child(
                    div()
                        .text_size(px(11.))
                        .text_color(theme::DIM)
                        .truncate()
                        .child("Damaged recording — kept as-is"),
                )
                .child(
                    div()
                        .font(theme::mono_font())
                        .text_size(px(9.))
                        .text_color(theme::DIM)
                        .truncate()
                        .child(damaged.reason.clone()),
                ),
        )
}
