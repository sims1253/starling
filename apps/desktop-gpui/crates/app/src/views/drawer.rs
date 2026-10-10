//! The transcript drawer: grip, raw transcript head with actions, body with
//! playback toggle and transcript text, fidelity warnings strip.

use std::time::Duration;

use gpui::{
    Animation, AnimationExt, Context, Div, FontWeight, Window, div, ease_out_quint, point,
    prelude::*, px, rgba,
};
use starling_dictation::storage::SessionStatus;

use crate::app::StarlingApp;
use crate::processing::ProcessingState;
use crate::theme;
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

    let mut actions = div().flex().flex_row().items_center().gap(px(7.));

    if session.status != SessionStatus::Transcribed && !active {
        actions = actions.child(
            action_button(
                "drawer-retry",
                false,
                cx.listener(|this, _, _window, cx| {
                    this.retry_selected(cx);
                }),
            )
            .child(icon("icons/refresh.svg", 16., theme::PAPER_INK))
            .child("Retry"),
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
        .child(icon("icons/file-audio.svg", 16., theme::PAPER_INK))
        .child(if wav_saved { "Saved" } else { "WAV" }),
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
        .child(if copied {
            icon("icons/check.svg", 16., theme::PAPER_INK)
        } else {
            icon("icons/clipboard.svg", 16., theme::PAPER_INK)
        })
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
        .child(icon("icons/download.svg", 16., theme::PAPER_INK))
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
        .child(icon("icons/archive.svg", 16., theme::PAPER_INK))
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
        action_button(
            "drawer-delete",
            delete_disabled,
            cx.listener(move |this, _, _window, cx| {
                let id = delete_id.clone();
                this.request_delete_session(id, cx);
            }),
        )
        .px(px(7.))
        .text_color(theme::DANGER)
        .child(icon("icons/trash.svg", 16., theme::DANGER))
        .when(delete_armed, |button| button.child("Confirm delete?")),
    );

    let mut body = div()
        .id("transcript-body")
        .flex()
        .flex_col()
        .flex_1()
        .min_h_0()
        .overflow_y_scroll()
        .px(px(27.))
        .pt(px(19.))
        .pb(px(20.));

    if has_player {
        let play_id = session.id.clone();
        body = body.child(
            div()
                .id("audio-review")
                .flex()
                .flex_row()
                .items_center()
                .gap(px(8.))
                .mb(px(15.))
                .opacity(0.72)
                .cursor_pointer()
                .on_click(cx.listener(move |this, _, _window, cx| {
                    let id = play_id.clone();
                    this.toggle_play(&id, cx);
                }))
                .child(if playing {
                    icon("icons/stop.svg", 11., theme::PAPER_SUBTLE).into_any_element()
                } else {
                    icon("icons/play.svg", 11., theme::PAPER_SUBTLE).into_any_element()
                })
                .child(
                    div()
                        .font(theme::mono_font())
                        .text_size(px(9.))
                        .text_color(theme::PAPER_SUBTLE)
                        .child(if playing { "Stop" } else { "Play recording" }),
                ),
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
                .font(theme::serif_font())
                .text_size(transcript_scale)
                .text_color(theme::PROCESSING)
                .child(spinner("drawer-spinner", 20., theme::PROCESSING))
                .child("The server is transcribing this recording…"),
        );
    }

    if session.status == SessionStatus::Failed {
        body = body.child(
            div()
                .mb(px(10.))
                .text_size(px(12.))
                .text_color(theme::FAILED_COPY)
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
                    .font(theme::serif_font())
                    .text_size(transcript_scale)
                    .line_height(transcript_scale * 1.4)
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
                    .font(theme::serif_font())
                    .font_weight(FontWeight::MEDIUM)
                    .text_size(transcript_scale)
                    .line_height(transcript_scale * 1.4)
                    .text_color(theme::PAPER_INK)
                    .min_h(transcript_scale * 1.4)
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

    let mut drawer = div()
        .id("transcript-drawer")
        .relative()
        .flex()
        .flex_col()
        .h_full()
        .bg(theme::PAPER)
        .text_color(theme::PAPER_INK)
        .shadow(vec![gpui::BoxShadow {
            color: rgba(0x00000040).into(),
            offset: point(px(0.), px(-24.)),
            blur_radius: px(70.),
            spread_radius: px(0.),
        }])
        .child(
            div()
                .absolute()
                .top(px(9.))
                .left(px(0.))
                .right(px(0.))
                .flex()
                .justify_center()
                .child(div().w(px(42.)).h(px(3.)).rounded(px(2.)).bg(theme::GRIP)),
        )
        .child(
            div()
                .min_h(px(64.))
                .flex()
                .flex_row()
                .items_center()
                .justify_between()
                .px(px(26.))
                .pt(px(18.))
                .pb(px(12.))
                .border_b_1()
                .border_color(theme::PAPER_LINE)
                .child(
                    div()
                        .flex()
                        .flex_col()
                        .child(
                            div()
                                .mb(px(3.))
                                .font(theme::mono_font())
                                .text_size(px(10.))
                                .font_weight(FontWeight::MEDIUM)
                                .text_color(theme::PAPER_EYEBROW)
                                .child("RAW TRANSCRIPT"),
                        )
                        .child(
                            div()
                                .text_size(px(10.))
                                .text_color(theme::PAPER_SUBTLE)
                                .child("No cleanup or silent rewriting"),
                        )
                        // #363: which engine produced this transcript —
                        // per take, so a take transcribed by the built-in
                        // engine keeps its label even after a later take
                        // used the manual server.
                        .when_some(session.model_label.clone(), |column, label| {
                            column.child(
                                div()
                                    .text_size(px(10.))
                                    .text_color(theme::PAPER_SUBTLE)
                                    .child(format!(
                                        "Transcribed with {}",
                                        provenance_label(&label)
                                    )),
                            )
                        }),
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
                .min_h(px(31.))
                .px(px(27.))
                .pt(px(7.))
                .pb(px(9.))
                .border_t_1()
                .border_color(theme::PAPER_LINE_SOFT)
                .text_size(px(11.))
                .line_height(px(11. * 1.45))
                .text_color(theme::AMBER_DEEP)
                .child(icon("icons/alert-circle.svg", 15., theme::AMBER_DEEP))
                .child(div().flex_1().min_w_0().child(warnings.join(" "))),
        );
    }

    div()
        .flex_none()
        .h(px(250.))
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

    let eyebrow = if label.is_empty() {
        "PROCESSED".to_string()
    } else {
        format!("PROCESSED · {label}")
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
        .border_color(theme::PAPER_LINE_SOFT)
        .child(
            div()
                .font(theme::mono_font())
                .text_size(px(10.))
                .font_weight(FontWeight::MEDIUM)
                .text_color(theme::PAPER_EYEBROW)
                .child(eyebrow),
        );
    let note = |text: String, color| div().text_size(px(11.)).text_color(color).child(text);
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
                    .text_size(px(11.))
                    .text_color(theme::PROCESSING)
                    .child(spinner("processing-spinner", 14., theme::PROCESSING))
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
                    theme::AMBER_DEEP,
                ));
            }
            block = block.child(
                div()
                    .font(theme::serif_font())
                    .text_size(scale * 0.85)
                    .line_height(scale * 0.85 * 1.4)
                    .text_color(theme::PAPER_INK)
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
            block = block.child(note(message, theme::FAILED_COPY));
        }
        ProcessingState::Cancelled => {
            block = block.child(note(
                "Cancelled. The raw transcript is unchanged.".to_string(),
                theme::PAPER_SUBTLE,
            ));
        }
        ProcessingState::Blocked { message } => {
            block = block.child(note(message, theme::FAILED_COPY));
        }
        ProcessingState::Idle => {}
    }

    if let Some(head) = processed_head {
        if !matches!(take.as_ref().map(|t| &t.state), Some(ProcessingState::Proposal { .. })) {
            block = block
                .child(note(
                    "Copy and Export use this text, not the raw transcript.".to_string(),
                    theme::PAPER_SUBTLE,
                ))
                .child(
                    div()
                        .font(theme::serif_font())
                        .text_size(scale * 0.85)
                        .line_height(scale * 0.85 * 1.4)
                        .text_color(theme::PAPER_INK)
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

fn action_button(
    id: &'static str,
    disabled: bool,
    on_click: impl Fn(&gpui::ClickEvent, &mut Window, &mut gpui::App) + 'static,
) -> gpui::Stateful<Div> {
    div()
        .id(id)
        .flex()
        .flex_row()
        .items_center()
        .gap(px(6.))
        .px(px(10.))
        .py(px(7.))
        .rounded(px(5.))
        .border_1()
        .border_color(theme::PAPER_BUTTON_LINE)
        .text_size(px(10.))
        .text_color(theme::PAPER_INK)
        .when(!disabled, |button| {
            button
                .cursor_pointer()
                .hover(|style| style.bg(theme::PAPER_HOVER))
        })
        .when(disabled, |button| button.opacity(0.35))
        .on_click(move |event, window, cx| {
            if !disabled {
                on_click(event, window, cx);
            }
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_labels_map_to_the_catalog_and_fall_back_to_the_id() {
        // #363: the catalog label is what the user chose in Settings, so
        // that is what history shows; an id this build no longer ships
        // still names itself.
        assert_eq!(
            provenance_label("engine:parakeet-v3-q4km-s16"),
            "Parakeet TDT 0.6B v3 (q4_k_m)"
        );
        assert_eq!(provenance_label("engine:some-future-model"), "some-future-model");
    }

    #[test]
    fn openai_labels_name_the_model_and_the_server() {
        assert_eq!(
            provenance_label("openai:whisper-large-v3"),
            "whisper-large-v3 (server)"
        );
    }

    #[test]
    fn other_labels_show_verbatim() {
        // Older attempt shapes (e.g. `starling:parakeet`) are history's
        // truth; guessing at them would be worse than showing them.
        assert_eq!(provenance_label("starling:parakeet"), "starling:parakeet");
    }
}
