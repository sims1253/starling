//! The 300px takes column: its head, the take rows, and the
//! damaged-record rows G02 appends after them.

use gpui::{Context, Div, SharedString, Window, div, prelude::*, px};
use starling_dictation::storage::{DamagedRecord, SessionStatus, SessionSummary};

use crate::app::StarlingApp;
use crate::theme;
use crate::views::{history_spinner_id, icon, spinner};

pub fn render_history(
    app: &mut StarlingApp,
    _window: &mut Window,
    cx: &mut Context<StarlingApp>,
) -> impl IntoElement {
    // R10: one pass — build each row's data and render it immediately,
    // instead of collecting a RowData vec and then re-walking it with
    // fresh clones for the elements.
    let rows: Vec<_> = app
        .sessions
        .iter()
        .map(|session| render_row(row_data(app, session), cx))
        .collect();
    // G02: quarantined records stay visible with their reason, sorted after
    // every readable take.
    let damaged_rows: Vec<_> = app
        .damaged
        .iter()
        .map(|damaged| render_damaged_row(damaged, cx))
        .collect();
    let count = app.sessions.len() + app.damaged.len();

    div()
        .id("history-pane")
        .w(px(300.))
        .flex_none()
        .flex()
        .flex_col()
        .min_h_0()
        .border_r_1()
        .border_color(theme::LINE)
        .bg(theme::ROOST)
        .child(
            div()
                .flex()
                .flex_row()
                .items_baseline()
                .justify_between()
                .px(px(20.))
                .pt(px(20.))
                .pb(px(14.))
                .child(
                    div()
                        .font(theme::display_font())
                        .text_size(px(theme::HEADING))
                        .text_color(theme::SPECK)
                        .child("Takes"),
                )
                .when(count > 0, |head| {
                    head.child(
                        div()
                            .text_size(px(theme::TEXT_SM))
                            .text_color(theme::DIM)
                            .child(count.to_string()),
                    )
                }),
        )
        .child(
            div()
                .id("history-list")
                .flex()
                .flex_col()
                .flex_1()
                .min_h_0()
                .overflow_y_scroll()
                .border_t_1()
                .border_color(theme::LINE)
                .when(app.sessions.is_empty() && app.damaged.is_empty(), |list| {
                    list.child(
                        div()
                            .px(px(20.))
                            .py(px(20.))
                            .text_size(px(theme::TEXT_SM))
                            .line_height(px(theme::TEXT_SM * 1.6))
                            .text_color(theme::QUILL)
                            .child(
                                "Every take you record or import lands here. Select one to read, \
                                 retry, or export it.",
                            ),
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
    /// When it was recorded: a time today, a date before.
    when: String,
    /// The duration, then anything unusual about the take, in words.
    detail: String,
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
            } else if session.status == SessionStatus::Transcribed {
                // #356: an empty answer is a result, not a pending send.
                "Empty transcript. Retry available".to_string()
            } else {
                "Sending to server…".to_string()
            }
        });

    let mut detail = vec![theme::fmt_duration(session.duration_ms)];
    if session.attempt_count > 1 {
        detail.push(format!("{} attempts", session.attempt_count));
    }
    if session.archival {
        detail.push("archived".to_string());
    }
    // #356: the take's own interruption stays visible after a retry.
    if session.interrupted && session.status != SessionStatus::Interrupted {
        detail.push("interrupted".to_string());
    }

    RowData {
        id: session.id.clone(),
        active: app.selected_id.as_deref() == Some(session.id.as_str()),
        title,
        when: theme::fmt_when(&session.created_at),
        detail: detail.join(", "),
        status: session.status,
    }
}

/// The state of a take as a small mark beside its duration: a spinner
/// while it transcribes, otherwise a dot in the state's color.
fn state_mark(id: &str, status: SessionStatus) -> gpui::AnyElement {
    let dot = |color| div().size(px(6.)).rounded_full().bg(color).into_any_element();
    match status {
        // R04: one id per transcribing row, not a shared
        // "history-spinner" shared by all of them.
        SessionStatus::Transcribing => {
            spinner(history_spinner_id(id), 12., theme::QUILL).into_any_element()
        }
        SessionStatus::Transcribed => dot(theme::SHEEN),
        SessionStatus::Failed => dot(theme::ROSE),
        SessionStatus::Interrupted => dot(theme::BEAK),
        SessionStatus::Captured => dot(theme::DIM),
    }
}

/// R10: takes the row data by value — the strings move straight into the
/// element instead of being cloned a second time.
fn render_row(data: RowData, cx: &mut Context<StarlingApp>) -> gpui::Stateful<Div> {
    let RowData {
        id,
        active,
        title,
        when,
        detail,
        status,
    } = data;
    let click_id = id.clone();
    let mark = state_mark(&id, status);

    take_row(SharedString::from(format!("history-row-{id}")), active)
        .on_click(cx.listener(move |this, _, _window, cx| {
            this.select_session(click_id.clone(), cx);
        }))
        .child(when_column(when))
        .child(
            div()
                .flex()
                .flex_col()
                .gap(px(4.))
                .flex_1()
                .min_w_0()
                .overflow_hidden()
                .child(
                    div()
                        .text_size(px(theme::TEXT_MD))
                        .text_color(theme::SPECK)
                        .truncate()
                        .child(title),
                )
                .child(
                    div()
                        .flex()
                        .flex_row()
                        .items_center()
                        .gap(px(7.))
                        .text_size(px(theme::TEXT_XS))
                        .text_color(theme::DIM)
                        .child(div().size(px(12.)).flex().items_center().justify_center().child(mark))
                        .child(div().min_w_0().truncate().child(detail)),
                ),
        )
}

/// A row in the takes column: the time on the left, the take beside it.
/// The selected row is lifted and marked with the beak on its edge.
fn take_row(id: SharedString, active: bool) -> gpui::Stateful<Div> {
    div()
        .id(id)
        .relative()
        .flex()
        .flex_row()
        .items_start()
        .gap(px(12.))
        .w_full()
        .px(px(20.))
        .py(px(12.))
        .border_b_1()
        .border_color(theme::LINE)
        .cursor_pointer()
        .hover(|style| style.bg(theme::HOVER))
        .when(active, |row| {
            row.bg(theme::ROOST_SELECTED).child(
                div()
                    .absolute()
                    .top(px(0.))
                    .bottom(px(0.))
                    .left(px(0.))
                    .w(px(3.))
                    .bg(theme::BEAK),
            )
        })
}

fn when_column(when: String) -> Div {
    div()
        .w(px(58.))
        .flex_none()
        .pt(px(2.))
        .text_size(px(theme::TEXT_XS))
        .text_color(theme::QUILL)
        .child(when)
}

/// A quarantined record (G02): visible, explained, never deletable by the
/// app itself. Clicking surfaces the recorded reason; the underlying files
/// stay untouched for manual recovery.
fn render_damaged_row(
    damaged: &DamagedRecord,
    cx: &mut Context<StarlingApp>,
) -> gpui::Stateful<Div> {
    let id = damaged.id.clone();

    take_row(SharedString::from(format!("history-row-{}", damaged.id)), false)
        .on_click(cx.listener(move |this, _, _window, cx| {
            this.surface_damage(&id, cx);
        }))
        .child(
            div()
                .w(px(58.))
                .flex_none()
                .pt(px(2.))
                .child(icon("icons/alert-circle.svg", 14., theme::ROSE)),
        )
        .child(
            div()
                .flex()
                .flex_col()
                .gap(px(4.))
                .flex_1()
                .min_w_0()
                .overflow_hidden()
                .child(
                    div()
                        .text_size(px(theme::TEXT_MD))
                        .text_color(theme::QUILL)
                        .truncate()
                        .child("Damaged recording, kept as it is"),
                )
                .child(
                    div()
                        .text_size(px(theme::TEXT_XS))
                        .text_color(theme::DIM)
                        .truncate()
                        .child(damaged.reason.clone()),
                ),
        )
}
