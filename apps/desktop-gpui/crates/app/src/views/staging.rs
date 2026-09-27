//! The staging panel (#297): the take's editable draft under the recorder,
//! with the processing proposal next to it and the delivery actions.

use gpui::{Context, Div, FontWeight, SharedString, Stateful, Window, div, prelude::*, px};

use crate::app::StarlingApp;
use crate::processing::ProcessingState;
use crate::staging::StagingPhase;
use crate::theme;
use crate::views::spinner;

fn panel_button(
    id: &'static str,
    disabled: bool,
    on_click: impl Fn(&gpui::ClickEvent, &mut Window, &mut gpui::App) + 'static,
) -> Stateful<Div> {
    div()
        .id(id)
        .flex()
        .flex_row()
        .items_center()
        .gap(px(6.))
        .px(px(10.))
        .py(px(6.))
        .rounded(px(5.))
        .border_1()
        .border_color(theme::LINE)
        .text_size(px(11.))
        .text_color(theme::INK)
        .when(!disabled, |button| {
            button
                .cursor_pointer()
                .hover(|style| style.bg(theme::GEAR_HOVER_BG).border_color(theme::LIME))
        })
        .when(disabled, |button| button.opacity(0.35))
        .on_click(move |event, window, cx| {
            if !disabled {
                on_click(event, window, cx);
            }
        })
}

fn eyebrow(text: impl Into<SharedString>) -> Div {
    div()
        .font(theme::mono_font())
        .text_size(px(10.))
        .font_weight(FontWeight::MEDIUM)
        .text_color(theme::EYEBROW)
        .child(text.into())
}

fn note(text: impl Into<SharedString>, color: gpui::Rgba) -> Div {
    div()
        .text_size(px(11.))
        .line_height(px(11. * 1.5))
        .text_color(color)
        .child(text.into())
}

fn seconds(ms: Option<f64>) -> Option<String> {
    ms.map(|ms| format!("{:.1} s", ms / 1000.))
}

pub fn render_staging_panel(
    app: &mut StarlingApp,
    _window: &mut Window,
    cx: &mut Context<StarlingApp>,
) -> Option<Div> {
    let staging = app.staging.as_ref()?;
    let phase = staging.phase;
    let editor = staging.editor.clone();
    let notice = staging
        .save_error
        .clone()
        .or_else(|| staging.notice.clone());
    let copied = staging.copied;
    let (text, raw) = app
        .visible_staging_draft()
        .map(|draft| (draft.text(), draft.raw_text()))
        .unwrap_or_default();
    let processing = app.staging_processing();
    let running = matches!(processing, Some((_, ProcessingState::Running)));

    let status = match phase {
        StagingPhase::Recording => {
            "Live. Grey words may still change; anything you type stays yours."
        }
        StagingPhase::Finishing => "Finishing the transcript…",
        StagingPhase::Ready if staging.save_error.is_some() => "Edits not saved.",
        StagingPhase::Ready if app.staging_has_unsaved_edits() => "Saving edits…",
        StagingPhase::Ready if text == raw => "Saved with the take, as recognized.",
        StagingPhase::Ready => "Saved with the take. Copy and Export use this text.",
        StagingPhase::Failed => "Not saved.",
    };

    let mut panel = div()
        .w_full()
        .max_w(px(620.))
        .mt(px(6.))
        .mb(px(14.))
        .flex()
        .flex_col()
        .gap(px(10.))
        .p(px(14.))
        .rounded(px(7.))
        .border_1()
        .border_color(theme::LINE)
        .bg(theme::STAGING_BG)
        .child(
            div()
                .flex()
                .flex_row()
                .items_center()
                .justify_between()
                .gap(px(12.))
                .child(eyebrow("DRAFT"))
                .child(
                    div()
                        .text_size(px(11.))
                        .text_color(theme::MUTED)
                        .child(status),
                ),
        )
        .child(
            div()
                .h(px(210.))
                .font(theme::serif_font())
                .text_size(px(18.))
                .line_height(px(27.))
                .text_color(theme::INK)
                .child(editor),
        );

    if let Some(notice) = notice {
        panel = panel.child(note(notice, theme::AMBER));
    }

    if let Some((label, state)) = processing {
        let mut block = div()
            .flex()
            .flex_col()
            .gap(px(8.))
            .pt(px(10.))
            .border_t_1()
            .border_color(theme::LINE);
        let mut buttons = div().flex().flex_row().items_center().gap(px(7.));
        let idle = matches!(state, ProcessingState::Idle);
        match state {
            ProcessingState::Running => {
                block = block.child(
                    div()
                        .flex()
                        .flex_row()
                        .items_center()
                        .gap(px(8.))
                        .text_size(px(11.))
                        .text_color(theme::MUTED)
                        .child(spinner("staging-processing-spinner", 14., theme::MUTED))
                        .child(format!(
                            "Processing with {label}… your text stays as it is."
                        )),
                );
                buttons = buttons.child(
                    panel_button(
                        "staging-cancel",
                        false,
                        cx.listener(|this, _, _window, cx| {
                            if let Some(id) = this.staging.as_ref().and_then(|s| s.take_id.clone())
                            {
                                this.cancel_processing(&id, cx);
                            }
                        }),
                    )
                    .child("Cancel"),
                );
            }
            ProcessingState::Proposal { row, current } => {
                let heading = [
                    Some("PROPOSAL".to_string()),
                    Some(label.clone()).filter(|l| !l.is_empty()),
                    seconds(row.stop_to_result_ms),
                ]
                .into_iter()
                .flatten()
                .collect::<Vec<_>>()
                .join(" · ");
                block = block.child(eyebrow(heading));
                if !current {
                    block = block.child(note(
                        "Stale: the text changed after this was requested. Use it anyway only if \
                         you want to replace your edits.",
                        theme::AMBER,
                    ));
                }
                block = block.child(
                    div()
                        .font(theme::serif_font())
                        .text_size(px(16.))
                        .line_height(px(16. * 1.45))
                        .text_color(theme::INK)
                        .when(!current, |text| text.opacity(0.6))
                        .child(if row.text.trim().is_empty() {
                            "(nothing left after cleanup)".to_string()
                        } else {
                            row.text.clone()
                        }),
                );
                buttons = buttons
                    .child(
                        panel_button(
                            "staging-accept",
                            false,
                            cx.listener(move |this, _, _window, cx| {
                                this.accept_staging(!current, cx)
                            }),
                        )
                        .child(if current {
                            "Use processed"
                        } else {
                            "Use anyway"
                        }),
                    )
                    .child(
                        panel_button(
                            "staging-dismiss",
                            false,
                            cx.listener(|this, _, _window, cx| {
                                if let Some(id) =
                                    this.staging.as_ref().and_then(|s| s.take_id.clone())
                                {
                                    this.dismiss_processed(&id, cx);
                                }
                            }),
                        )
                        .child("Dismiss"),
                    );
            }
            ProcessingState::Failed { message } | ProcessingState::Blocked { message } => {
                block = block.child(note(message, theme::ERROR_TEXT));
            }
            ProcessingState::Cancelled => {
                block = block.child(note("Cancelled. Your text is unchanged.", theme::MUTED));
            }
            ProcessingState::Idle => {}
        }
        if !idle {
            panel = panel.child(block.child(buttons));
        }
    }

    let ready = phase == StagingPhase::Ready;
    let mut actions = div()
        .flex()
        .flex_row()
        .flex_wrap()
        .items_center()
        .gap(px(7.));
    if app.mode_processes() && ready && !running {
        actions = actions.child(
            panel_button(
                "staging-run",
                false,
                cx.listener(|this, _, _window, cx| this.process_staging(cx)),
            )
            .child(format!("Run {}", app.active_mode().name)),
        );
    }
    if ready && text != raw {
        actions = actions.child(
            panel_button(
                "staging-revert",
                false,
                cx.listener(|this, _, _window, cx| this.revert_staging(cx)),
            )
            .child("Back to raw"),
        );
    }
    actions = actions
        .child(
            panel_button(
                "staging-copy",
                text.is_empty(),
                cx.listener(|this, _, _window, cx| this.copy_staging(cx)),
            )
            .child(if copied { "Copied" } else { "Copy" }),
        )
        .child(
            panel_button(
                "staging-done",
                phase == StagingPhase::Recording,
                cx.listener(|this, _, _window, cx| this.finish_staging(cx)),
            )
            .child(if phase == StagingPhase::Failed {
                "Close"
            } else {
                "Done"
            }),
        );
    let hints = if cfg!(target_os = "macos") {
        "⌥⌫ word · ⌘⇧K visual line · ⌥←/→ jump · ⌘Z undo · ⌘↩ done · Esc leave"
    } else {
        "Ctrl+⌫ word · Ctrl+Shift+K visual line · Ctrl+←/→ jump · Ctrl+Z undo · Ctrl+Enter done · Esc leave"
    };
    panel = panel.child(
        div()
            .flex()
            .flex_row()
            .flex_wrap()
            .items_center()
            .justify_between()
            .gap(px(10.))
            .child(actions)
            .child(
                div()
                    .font(theme::mono_font())
                    .text_size(px(9.))
                    .text_color(theme::DIM)
                    .child(hints),
            ),
    );
    Some(panel)
}
