//! The transcript sheet under the stage: what produced the text and the
//! take's actions, then playback and the transcript itself, then the
//! fidelity warnings strip.

use std::time::Duration;

use gpui::{
    Animation, AnimationExt, Context, Div, FontWeight, Window, div, ease_out_quint, prelude::*,
    px,
};
use starling_dictation::storage::SessionStatus;

use crate::app::StarlingApp;
use crate::processing::ProcessingState;
use crate::theme;
use crate::views::kit::{self, Tone};
use crate::views::{icon, spinner};

/// The built-in catalog, computed once and shared for the entity's
/// lifetime (#366): `default_catalog()` builds fresh `String`s for every
/// entry, and the drawer renders one label per row per frame.
fn catalog_entries() -> &'static [starling_dictation::engine::CatalogEntry] {
    static CATALOG: std::sync::OnceLock<Vec<starling_dictation::engine::CatalogEntry>> =
        std::sync::OnceLock::new();
    CATALOG.get_or_init(starling_dictation::engine::default_catalog)
}

/// How the drawer words a take's transcription provenance (#363):
/// `engine:<id>` maps to the catalog label (falling back to the raw id
/// when this build's catalog no longer knows it), `openai:<m>` names the
/// model and that it came from a server, and any older label shape shows
/// verbatim rather than being guessed at. Parsing and rendering go
/// through `BackendLabel`, the typed form of the persisted shape.
pub(crate) fn provenance_label(label: &str) -> String {
    match starling_dictation::storage::BackendLabel::parse(label) {
        Some(starling_dictation::storage::BackendLabel::Engine { model_id }) => {
            catalog_entries()
                .iter()
                .find(|entry| entry.id == model_id)
                .map(|entry| entry.label.clone())
                .unwrap_or(model_id)
        }
        Some(starling_dictation::storage::BackendLabel::OpenAi { model }) => {
            format!("{model} (server)")
        }
        None => label.to_string(),
    }
}

pub fn render_drawer(
    app: &mut StarlingApp,
    window: &mut Window,
    cx: &mut Context<StarlingApp>,
) -> Div {
    let Some(session) = app.selected().cloned() else {
        return div();
    };

    let active = app.is_active(&session.id);
    let playing = app.playing_id.as_deref() == Some(session.id.as_str());
    // #214.2: the flags name the take they were earned on, so switching
    // selections can never show "Copied"/"Saved" for a take nothing
    // happened on.
    let copied = app.copied.as_deref() == Some(session.id.as_str());
    let wav_saved = app.wav_saved.as_deref() == Some(session.id.as_str());
    let transcript_scale = app.transcript_scale(window.viewport_size().width);
    let transcript = session.transcript.clone();
    let warnings = app.fidelity_warnings();
    let has_player = app.player.is_some();

    let mut actions = div()
        .flex()
        .flex_row()
        .flex_wrap()
        .justify_end()
        .items_center()
        .gap(px(6.));

    // #356: any saved take can be transcribed again, with the same or
    // another model, from its stored audio; a success adds a result.
    let pending_switch = app
        .pending_retry
        .as_ref()
        .filter(|pending| pending.take_id == session.id)
        .map(|pending| provenance_label(&format!("engine:{}", pending.model_id)));
    let menu_open = app.retry_menu.as_deref() == Some(session.id.as_str()) && !active;
    // The choices snapshot the engine: read only for an open menu, or at
    // the click that decides whether one opens.
    let retry_choices = if menu_open {
        app.retry_choices()
    } else {
        Vec::new()
    };
    if !active {
        let menu_id = session.id.clone();
        actions = actions.child(
            action_button(
                "drawer-retry",
                false,
                cx.listener(move |this, _, _window, cx| {
                    if this.retry_choices().is_empty() {
                        this.retry_selected(cx);
                    } else {
                        this.toggle_retry_menu(&menu_id, cx);
                    }
                }),
            )
            .child(if session.status == SessionStatus::Transcribed {
                "Transcribe again"
            } else {
                "Retry"
            }),
        );
    }

    actions = actions.child(
        action_button(
            "drawer-wav",
            false,
            cx.listener(|this, _, _window, cx| {
                this.export_audio(cx);
            }),
        )
        .child(if wav_saved { "Saved" } else { "WAV" }),
    );

    // #356: the same samples, losslessly compressed.
    actions = actions.child(
        action_button(
            "drawer-flac",
            false,
            cx.listener(|this, _, _window, cx| {
                this.export_audio_as(true, cx);
            }),
        )
        .child("FLAC"),
    );

    let copy_disabled = transcript.is_none();
    actions = actions.child(
        action_button(
            "drawer-copy",
            copy_disabled,
            cx.listener(|this, _, _window, cx| {
                this.copy_transcript(cx);
            }),
        )
        .child(if copied { "Copied" } else { "Copy" }),
    );

    actions = actions.child(
        action_button(
            "drawer-export",
            copy_disabled,
            cx.listener(|this, _, _window, cx| {
                this.export_transcript(cx);
            }),
        )
        .child("Export"),
    );

    // #342: the archival class keeps its own retention limits.
    actions = actions.child(
        action_button(
            "drawer-archive",
            false,
            cx.listener(|this, _, _window, cx| {
                this.toggle_archival_selected(cx);
            }),
        )
        .child(if session.archival { "Archived" } else { "Archive" }),
    );

    // B05, #208: the trash button never deletes on its first click. It
    // arms a "Confirm delete?" state for exactly the selected take; only a
    // second click confirms. Also disabled while a confirmed delete is in
    // flight (#214.3), like the transcription-active case above.
    let delete_disabled = active || app.is_deleting(&session.id);
    let delete_armed = app.delete_armed();
    let delete_id = session.id.clone();
    actions = actions.child(
        kit::button("drawer-delete", Tone::SheetDanger, delete_disabled)
            .on_click(cx.listener(move |this, _, _window, cx| {
                if !delete_disabled {
                    let id = delete_id.clone();
                    this.request_delete_session(id, cx);
                }
            }))
            .child(icon("icons/trash.svg", 14., theme::SHEET_ROSE))
            .child(if delete_armed { "Confirm delete" } else { "Delete" }),
    );

    let mut body = div()
        .id("transcript-body")
        .flex()
        .flex_col()
        .flex_1()
        .min_h_0()
        .overflow_y_scroll()
        .px(px(28.))
        .pt(px(16.))
        .pb(px(20.));

    if menu_open {
        body = body.child(retry_menu(&retry_choices, cx));
    }

    if let Some(model) = pending_switch {
        body = body.child(
            div()
                .flex()
                .flex_row()
                .items_center()
                .gap(px(8.))
                .mb(px(10.))
                .text_size(px(theme::TEXT_SM))
                .text_color(theme::SHEET_QUILL)
                .child(spinner("drawer-switch-spinner", 14., theme::SHEET_QUILL))
                .child(format!(
                    "Loading {model} to transcribe this recording again… The recording and its \
                     earlier results are unchanged."
                )),
        );
    }

    // #356: a take that ended without a clean stop says so, with how much
    // came back and how much of that the journal had confirmed on disk —
    // whatever its transcription attempts did since.
    if session.interrupted {
        let seconds = |ms: Option<f64>| ms.map(|ms| format!("{:.1} s", ms / 1000.0));
        let mut parts = Vec::new();
        if let Some(kept) = seconds(session.duration_ms) {
            parts.push(format!("{kept} kept"));
        }
        if let Some(confirmed) = seconds(session.confirmed_ms) {
            parts.push(format!("{confirmed} confirmed on disk"));
        }
        let line = if parts.is_empty() {
            "Interrupted recording".to_string()
        } else {
            format!("Interrupted recording: {}", parts.join(", "))
        };
        body = body.child(
            div()
                .mb(px(6.))
                .text_size(px(theme::TEXT_SM))
                .font_weight(FontWeight::SEMIBOLD)
                .text_color(theme::SHEET_WARN)
                .child(line),
        );
        if session.status == SessionStatus::Interrupted {
            if let Some(note) = session.last_error.clone() {
                body = body.child(
                    div()
                        .mb(px(10.))
                        .text_size(px(theme::TEXT_SM))
                        .text_color(theme::SHEET_QUILL)
                        .child(note),
                );
            }
        }
    }

    if has_player {
        let play_id = session.id.clone();
        body = body.child(
            div()
                .id("audio-review")
                .flex()
                .flex_row()
                .items_center()
                .gap(px(7.))
                .mb(px(12.))
                .text_size(px(theme::TEXT_SM))
                .text_color(theme::SHEET_QUILL)
                .cursor_pointer()
                .hover(|style| style.text_color(theme::SHEET_INK))
                .on_click(cx.listener(move |this, _, _window, cx| {
                    let id = play_id.clone();
                    this.toggle_play(&id, cx);
                }))
                .child(if playing {
                    icon("icons/stop.svg", 12., theme::SHEET_QUILL).into_any_element()
                } else {
                    icon("icons/play.svg", 12., theme::SHEET_QUILL).into_any_element()
                })
                .child(if playing { "Stop" } else { "Play recording" }),
        );
    }

    if session.status == SessionStatus::Transcribing {
        body = body.child(
            div()
                .flex()
                .flex_row()
                .items_center()
                .gap(px(10.))
                .mb(px(10.))
                .font(theme::reading_font())
                .text_size(transcript_scale)
                .text_color(theme::SHEET_QUILL)
                .child(spinner("drawer-spinner", 18., theme::SHEET_QUILL))
                .child("The server is transcribing this recording…"),
        );
    }

    if session.status == SessionStatus::Failed {
        body = body.child(
            div()
                .mb(px(10.))
                .text_size(px(theme::TEXT_MD))
                .line_height(px(theme::TEXT_MD * 1.5))
                .text_color(theme::SHEET_ROSE)
                .child(session.last_error.clone().unwrap_or_else(|| {
                    "The server could not transcribe this take. The original audio is safe here."
                        .to_string()
                })),
        );
    }

    if let Some(transcript) = transcript {
        if transcript.text.trim().is_empty() {
            body = body.child(
                div()
                    .font(theme::reading_font())
                    .text_size(transcript_scale)
                    .line_height(transcript_scale * 1.5)
                    .text_color(theme::SHEET_QUILL)
                    .italic()
                    .child("The model returned an empty transcript."),
            );
        } else {
            // R10: one text element for the whole transcript. gpui shapes
            // '\n' into lines internally, so the previous one-div-per-line
            // build (an unbounded element count on every render) bought
            // nothing. `line_height` keeps the rhythm, `min_h` the empty-
            // line height, exactly as the per-line divs did.
            body = body.child(
                div()
                    .max_w(px(760.))
                    .font(theme::reading_font())
                    .text_size(transcript_scale)
                    .line_height(transcript_scale * 1.5)
                    .text_color(theme::SHEET_INK)
                    .min_h(transcript_scale * 1.5)
                    // The scrolling body is a flex column: without this the
                    // text shrinks to one line and paints over what follows.
                    .flex_shrink_0()
                    .child(transcript.text),
            );
        }
    }

    if session.transcript.is_some() && !active {
        if let Some(block) = processing_block(app, &session.id, transcript_scale, cx) {
            body = body.child(block);
        }
    }

    if let Some(block) = earlier_results(&session.results, cx) {
        body = body.child(block);
    }

    let mut drawer = div()
        .id("transcript-drawer")
        .relative()
        .flex()
        .flex_col()
        .h_full()
        .bg(theme::SHEET)
        .text_color(theme::SHEET_INK)
        .child(
            div()
                .flex()
                .flex_row()
                .items_start()
                .justify_between()
                .gap(px(16.))
                .px(px(28.))
                .pt(px(14.))
                .pb(px(12.))
                .border_b_1()
                .border_color(theme::SHEET_LINE)
                .child(
                    div()
                        .flex_none()
                        .flex()
                        .flex_col()
                        .gap(px(2.))
                        .pt(px(4.))
                        .child(
                            div()
                                .text_size(px(theme::TEXT_MD))
                                .font_weight(FontWeight::SEMIBOLD)
                                .child("Transcript"),
                        )
                        // #363: which engine produced this transcript —
                        // per take, so a take transcribed by the built-in
                        // engine keeps its label even after a later take
                        // used the manual server.
                        .child(
                            div()
                                .text_size(px(theme::TEXT_SM))
                                .text_color(theme::SHEET_QUILL)
                                .child(match session.model_label.as_deref() {
                                    Some(label) => format!(
                                        "Exactly as {} returned it",
                                        provenance_label(label)
                                    ),
                                    None => "Exactly as the model returned it".to_string(),
                                }),
                        ),
                )
                .child(actions),
        )
        .child(body);

    if !warnings.is_empty() {
        drawer = drawer.child(
            div()
                .flex()
                .flex_row()
                .items_center()
                .gap(px(7.))
                .min_h(px(34.))
                .px(px(28.))
                .py(px(8.))
                .border_t_1()
                .border_color(theme::SHEET_LINE)
                .text_size(px(theme::TEXT_SM))
                .line_height(px(theme::TEXT_SM * 1.45))
                .text_color(theme::SHEET_WARN)
                .child(icon("icons/alert-circle.svg", 15., theme::SHEET_WARN))
                .child(div().flex_1().min_w_0().child(warnings.join(" "))),
        );
    }

    div()
        .flex_none()
        .h(px(270.))
        .min_h(px(210.))
        .child(drawer.with_animation(
            "drawer-in",
            Animation::new(Duration::from_millis(250)).with_easing(ease_out_quint()),
            |el, delta| el.opacity(0.3 + 0.7 * delta),
        ))
}

/// The processed side of a take (#295): a proposal next to the raw text,
/// never in place of it. "Use processed" makes it the take's head (what
/// Copy and Export use) only while it is current; "Back to raw" is always
/// one click away.
fn processing_block(
    app: &StarlingApp,
    id: &str,
    scale: gpui::Pixels,
    cx: &mut Context<StarlingApp>,
) -> Option<gpui::Stateful<Div>> {
    // The staging panel shows this take's processing next to its text.
    if app.staging_shows(id) {
        return None;
    }
    let take = app.processing.get(id).cloned();
    let processes = app.mode_processes();
    if take.is_none() && !processes {
        return None;
    }
    let label = take.as_ref().map(|take| take.label.clone()).unwrap_or_default();
    let state = take
        .as_ref()
        .map(|take| take.state.clone())
        .unwrap_or(ProcessingState::Idle);
    let processed_head = take.as_ref().and_then(|take| take.processed_head.clone());

    let heading = if label.is_empty() {
        "Processed".to_string()
    } else {
        format!("Processed with {label}")
    };
    let mut block = div()
        .id("processing-block")
        .flex_shrink_0()
        .flex()
        .flex_col()
        .gap(px(8.))
        .mt(px(16.))
        .pt(px(12.))
        .border_t_1()
        .border_color(theme::SHEET_LINE)
        .child(kit::sheet_subheading(heading));
    let note = |text: String, color| {
        div()
            .text_size(px(theme::TEXT_SM))
            .line_height(px(theme::TEXT_SM * 1.5))
            .text_color(color)
            .child(text)
    };
    let mut buttons = div().flex().flex_row().items_center().gap(px(7.));
    let take_id = id.to_string();
    let run_label = format!("Run {}", app.active_mode().name);

    match state {
        ProcessingState::Running => {
            block = block.child(
                div()
                    .flex()
                    .flex_row()
                    .items_center()
                    .gap(px(8.))
                    .text_size(px(theme::TEXT_SM))
                    .text_color(theme::SHEET_QUILL)
                    .child(spinner("processing-spinner", 14., theme::SHEET_QUILL))
                    .child("Processing… the raw transcript above is already saved."),
            );
            let cancel_id = take_id.clone();
            buttons = buttons.child(
                action_button(
                    "processing-cancel",
                    false,
                    cx.listener(move |this, _, _window, cx| this.cancel_processing(&cancel_id, cx)),
                )
                .child("Cancel"),
            );
        }
        ProcessingState::Proposal { row, current } => {
            if !current {
                block = block.child(note(
                    "Stale: the take changed after this was requested, so it is only a \
                     suggestion."
                        .to_string(),
                    theme::SHEET_WARN,
                ));
            }
            block = block.child(
                div()
                    .font(theme::reading_font())
                    .text_size(scale * 0.85)
                    .line_height(scale * 0.85 * 1.4)
                    .text_color(theme::SHEET_INK)
                    .when(!current, |text| text.opacity(0.6))
                    .child(if row.text.trim().is_empty() {
                        "(nothing left after cleanup)".to_string()
                    } else {
                        row.text.clone()
                    }),
            );
            let accept_id = take_id.clone();
            buttons = buttons.child(
                action_button(
                    "processing-accept",
                    false,
                    cx.listener(move |this, _, _window, cx| this.accept_processed(&accept_id, !current, cx)),
                )
                .child(if current { "Use processed" } else { "Use anyway" }),
            );
            let dismiss_id = take_id.clone();
            buttons = buttons.child(
                action_button(
                    "processing-dismiss",
                    false,
                    cx.listener(move |this, _, _window, cx| this.dismiss_processed(&dismiss_id, cx)),
                )
                .child("Dismiss"),
            );
        }
        ProcessingState::Failed { message } => {
            block = block.child(note(message, theme::SHEET_ROSE));
        }
        ProcessingState::Cancelled => {
            block = block.child(note(
                "Cancelled. The raw transcript is unchanged.".to_string(),
                theme::SHEET_QUILL,
            ));
        }
        ProcessingState::Blocked { message } => {
            block = block.child(note(message, theme::SHEET_ROSE));
        }
        ProcessingState::Idle => {}
    }

    if let Some(head) = processed_head {
        if !matches!(take.as_ref().map(|t| &t.state), Some(ProcessingState::Proposal { .. })) {
            block = block
                .child(note(
                    "Copy and Export use this text, not the raw transcript.".to_string(),
                    theme::SHEET_QUILL,
                ))
                .child(
                    div()
                        .font(theme::reading_font())
                        .text_size(scale * 0.85)
                        .line_height(scale * 0.85 * 1.4)
                        .text_color(theme::SHEET_INK)
                        .child(head),
                );
        }
        let revert_id = take_id.clone();
        buttons = buttons.child(
            action_button(
                "processing-revert",
                false,
                cx.listener(move |this, _, _window, cx| this.revert_to_raw(&revert_id, cx)),
            )
            .child("Back to raw"),
        );
    }

    let running = app.processing_jobs.contains_key(id);
    if processes && !running {
        let run_id = take_id.clone();
        buttons = buttons.child(
            action_button(
                "processing-run",
                false,
                cx.listener(move |this, _, _window, cx| this.process_take(run_id.clone(), cx)),
            )
            .child(run_label),
        );
    }
    Some(block.child(buttons))
}

/// The "Retry with" choices (#356): one button per installed model and
/// the server from Settings. Choosing another built-in model switches
/// the engine to it, the way Settings would.
fn retry_menu(
    choices: &[crate::upload::RetryChoice],
    cx: &mut Context<StarlingApp>,
) -> gpui::Stateful<Div> {
    let mut buttons = div().flex().flex_row().flex_wrap().items_center().gap(px(7.));
    for (index, choice) in choices.iter().enumerate() {
        let with = choice.with.clone();
        buttons = buttons.child(
            action_button(
                ("retry-choice", index),
                false,
                cx.listener(move |this, _, _window, cx| {
                    this.retry_selected_with(with.clone(), cx);
                }),
            )
            .child(choice.label.clone()),
        );
    }
    let mut menu = div()
        .id("retry-menu")
        .flex_shrink_0()
        .flex()
        .flex_col()
        .gap(px(7.))
        .mb(px(14.))
        .pb(px(12.))
        .border_b_1()
        .border_color(theme::SHEET_LINE)
        .child(kit::sheet_subheading("Transcribe again with"))
        .child(buttons);
    let mut note = "Uses the saved audio; earlier results are kept beside the new one, and \
                    nothing is typed into another app."
        .to_string();
    if choices.iter().any(|choice| choice.switches_engine) {
        note.push_str(" Another built-in model becomes the engine's active model.");
    }
    menu = menu.child(
        div()
            .text_size(px(theme::TEXT_SM))
            .line_height(px(theme::TEXT_SM * 1.5))
            .text_color(theme::SHEET_QUILL)
            .child(note),
    );
    menu
}

/// Every transcript the take has besides the one shown (#356), newest
/// first, each with what produced it and its own Copy.
fn earlier_results(
    results: &[starling_dictation::storage::TakeResult],
    cx: &mut Context<StarlingApp>,
) -> Option<gpui::Stateful<Div>> {
    let others: Vec<_> = results.iter().rev().filter(|result| !result.shown).collect();
    if others.is_empty() {
        return None;
    }
    let mut block = div()
        .id("earlier-results")
        .flex_shrink_0()
        .flex()
        .flex_col()
        .gap(px(9.))
        .mt(px(16.))
        .pt(px(12.))
        .border_t_1()
        .border_color(theme::SHEET_LINE)
        .child(kit::sheet_subheading("Other results for this recording"));
    for (index, result) in others.into_iter().enumerate() {
        let when = result
            .created_at
            .as_deref()
            .and_then(|utc| utc.get(..16))
            .map(|utc| format!(", {}", utc.replace('T', " ")))
            .unwrap_or_default();
        let text = result.text.clone();
        block = block.child(
            div()
                .flex()
                .flex_col()
                .gap(px(4.))
                .child(
                    div()
                        .flex()
                        .flex_row()
                        .items_center()
                        .justify_between()
                        .child(
                            div()
                                .text_size(px(theme::TEXT_SM))
                                .text_color(theme::SHEET_QUILL)
                                .child(format!("{}{when}", provenance_label(&result.backend))),
                        )
                        .child(
                            action_button(
                                ("earlier-copy", index),
                                text.trim().is_empty(),
                                cx.listener(move |_this, _, _window, cx| {
                                    cx.write_to_clipboard(gpui::ClipboardItem::new_string(
                                        text.clone(),
                                    ));
                                }),
                            )
                            .child("Copy"),
                        ),
                )
                .child(
                    div()
                        .font(theme::reading_font())
                        .text_size(px(theme::TEXT_LG))
                        .line_height(px(theme::TEXT_LG * 1.5))
                        .text_color(theme::SHEET_QUILL)
                        .child(if result.text.trim().is_empty() {
                            "(empty transcript)".to_string()
                        } else {
                            result.text.clone()
                        }),
                ),
        );
    }
    Some(block)
}

fn action_button(
    id: impl Into<gpui::ElementId>,
    disabled: bool,
    on_click: impl Fn(&gpui::ClickEvent, &mut Window, &mut gpui::App) + 'static,
) -> gpui::Stateful<Div> {
    kit::button(id, Tone::SheetQuiet, disabled).on_click(move |event, window, cx| {
        if !disabled {
            on_click(event, window, cx);
        }
    })
}
