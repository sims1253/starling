//! The settings modal: dark scrim over the workspace, light centered card
//! with the Engine section (mode, models, switches — #362/#363), the
//! manual server fields, and the after-transcription section.

use std::time::Duration;

use gpui::{
    Animation, AnimationExt, Context, Div, FontWeight, MouseButton, SharedString, Window, div,
    prelude::*, px, rgba,
};
use starling_dictation::engine::{
    Backend, EngineFailure, EngineSnapshot, InstallState, SwapDecision, SwitchReport, SwitchStage,
};
use starling_dictation::settings::{ActivationMode, EngineMode, PlaybackMode};

use crate::app::{ConnectionProbe, StarlingApp, settings_callout_view};
use crate::processing;
use crate::theme;
use crate::views::{SETTINGS_CALLOUT_DOT_ID, icon, status_dot};

pub fn render_settings_modal(
    app: &mut StarlingApp,
    _window: &mut Window,
    cx: &mut Context<StarlingApp>,
) -> impl IntoElement {
    // B06 (#207): the status line belongs to the probe once one ran — a
    // draft's failure shows here without touching the live connection —
    // and without a probe the committed status shows. Manual mode only:
    // builtin mode has no manual endpoint, and its engine state fills the
    // callout's place instead.
    let manual = app.draft_engine_mode == EngineMode::Manual;
    let callout = settings_callout_view(app.probe.as_ref(), app.connection, &app.endpoint);
    let probing = matches!(app.probe, Some(ConnectionProbe::Testing { .. }));
    let draft_endpoint = app.draft_endpoint.clone();
    let draft_model = app.draft_model.clone();
    let draft_terms = app.draft_terms.clone();
    let engine_section = render_engine_section(app, cx);
    let processing_section = render_processing_section(app, cx);
    let microphone_section = crate::views::microphone::render_microphone_section(app, cx);
    let dictation_section = render_dictation_section(app, cx);
    let playback_section = render_playback_section(app, cx);
    let feedback_section = crate::views::feedback::render_feedback_section(app, cx);
    let live_preview_section = crate::views::live_preview::render_live_preview_section(app, cx);
    let storage_section = crate::views::storage::render_storage_section(app, cx);
    let insertion_section = crate::views::delivery::render_insertion_section(app, cx);

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
        // ---- Engine (#362, #363): the section at the top ------------
        .child(
            div()
                .mb(px(13.))
                .font(theme::mono_font())
                .text_size(px(10.))
                .font_weight(FontWeight::MEDIUM)
                .text_color(theme::SETTINGS_EYEBROW)
                .child("ENGINE"),
        )
        .child(
            div()
                .mb(px(13.))
                .font(theme::serif_font())
                .font_weight(FontWeight::MEDIUM)
                .text_size(px(33.))
                .line_height(px(33.))
                .child("Transcription engine"),
        )
        .child(
            div()
                .mb(px(25.))
                .text_size(px(11.))
                .line_height(px(11. * 1.65))
                .text_color(theme::SETTINGS_MUTED)
                .child(if manual {
                    "Your own starling-serve or OpenAI-compatible transcription endpoint."
                } else {
                    "Dictation runs on the bundled engine on this machine; models are \
                     downloaded and switched here."
                }),
        )
        .child(mode_choice(app, cx))
        .children((!manual).then_some(engine_section))
        // ---- Manual server fields (manual mode only) ----------------
        .when(manual, |card| {
            card.child(
                div()
                    .mt(px(10.))
                    .mb(px(13.))
                    .font(theme::mono_font())
                    .text_size(px(10.))
                    .font_weight(FontWeight::MEDIUM)
                    .text_color(theme::SETTINGS_EYEBROW)
                    .child("CONNECTION"),
            )
        })
        .when(manual, |card| {
            card.child(field_label("Server endpoint").child(draft_endpoint))
                .child(field_label("Model").child(draft_model))
        })
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
        .when(manual, |card| {
            card.child(
                div()
                    .flex()
                    .flex_row()
                    .items_center()
                    .gap(px(12.))
                    .bg(theme::SETTINGS_CALLOUT)
                    .p(px(12.))
                    .mt(px(21.))
                    .child(status_dot(SETTINGS_CALLOUT_DOT_ID, callout.dot, false))
                    .child(
                        div()
                            .flex()
                            .flex_col()
                            .gap(px(2.))
                            .text_size(px(10.))
                            .child(div().font_weight(FontWeight::SEMIBOLD).child(callout.title))
                            .child(
                                div()
                                    .font(theme::mono_font())
                                    .text_size(px(9.))
                                    .text_color(theme::SETTINGS_CALLOUT_ENDPOINT)
                                    .child(callout.detail),
                            ),
                    ),
            )
        })
        .child(dictation_section)
        .child(microphone_section)
        .child(playback_section)
        .child(feedback_section)
        .child(live_preview_section)
        .child(storage_section)
        .child(insertion_section)
        .child(processing_section)
        .child(
            div()
                .flex()
                .flex_row()
                .justify_end()
                .gap(px(9.))
                .mt(px(25.))
                .when(manual, |footer| {
                    footer.child(
                        div()
                            .id("test-connection")
                            .px(px(13.))
                            .py(px(9.))
                            .rounded(px(3.))
                            .border_1()
                            .border_color(theme::SETTINGS_FOOT_LINE)
                            .text_size(px(10.))
                            // B06 (#207): one probe at a time — while the
                            // newest press is in flight the button carries no
                            // click handler at all (and `test_connection`
                            // drops re-entrant presses as the state-layer
                            // twin of the same guard).
                            .when(!probing, |button| {
                                button
                                    .cursor_pointer()
                                    .hover(|style| style.bg(theme::PAPER_HOVER))
                                    .on_click(cx.listener(|this, _, _window, cx| {
                                        this.test_connection(cx);
                                    }))
                            })
                            .child(if probing {
                                "Testing…"
                            } else {
                                "Test connection"
                            }),
                    )
                })
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

/// One step of a model switch's progress line (#363): the word and
/// whether it is the current one.
pub(crate) struct SwitchStep {
    pub word: String,
    pub current: bool,
}

/// The switch progress line (#363): the stage words exactly
/// "Downloading n% → Verifying → Loading → Warming → Active" with the
/// current one highlighted, plus a leading note for the stages that are
/// not part of that ladder (waiting for the open take, stopping the old
/// engine). Pure so the wording is pinned by tests.
pub(crate) fn switch_progress(stage: &SwitchStage) -> (Option<String>, Vec<SwitchStep>) {
    let ladder = |current: usize, downloading: Option<u64>| {
        let words = [
            match downloading {
                Some(percent) => format!("Downloading {percent}%"),
                None => "Downloading".to_string(),
            },
            "Verifying".to_string(),
            "Loading".to_string(),
            "Warming".to_string(),
            "Active".to_string(),
        ];
        words
            .into_iter()
            .enumerate()
            .map(|(index, word)| SwitchStep {
                word,
                current: index == current,
            })
            .collect()
    };
    match stage {
        SwitchStage::Downloading { done, total } => {
            (None, ladder(0, Some(download_percent(*done, *total))))
        }
        SwitchStage::Verifying => (None, ladder(1, None)),
        SwitchStage::Loading => (None, ladder(2, None)),
        SwitchStage::Warming => (None, ladder(3, None)),
        SwitchStage::CuttingOver => (None, ladder(4, None)),
        SwitchStage::WaitingForTake => (
            Some("Waiting for the current take to finish…".to_string()),
            ladder(usize::MAX, None),
        ),
        SwitchStage::Draining => (
            Some("Stopping the previous model…".to_string()),
            ladder(usize::MAX, None),
        ),
    }
}

/// The download's percent, whole numbers, `0` for an unknown total.
pub(crate) fn download_percent(done: u64, total: u64) -> u64 {
    // `checked_div`: a zero total is the unknown-size case, not a panic.
    done.min(total)
        .saturating_mul(100)
        .checked_div(total)
        .unwrap_or(0)
}

/// A byte size as the settings rows show it (#363): MB under a GB, GB
/// above, one decimal.
pub(crate) fn fmt_size(bytes: u64) -> String {
    const MB: f64 = 1_000_000.;
    let mb = bytes as f64 / MB;
    if mb >= 1000. {
        format!("{:.1} GB", mb / 1000.)
    } else {
        format!("{:.1} MB", mb)
    }
}

/// The last switch report line (#363): "Switched in 4.2 s · peak engine
/// memory 1.9 GB" — the memory half only when a reading exists.
pub(crate) fn switch_report_line(report: &SwitchReport) -> String {
    match report.peak_rss_bytes {
        Some(peak) => format!(
            "Switched in {:.1} s · peak engine memory {}",
            report.duration.as_secs_f64(),
            fmt_size(peak)
        ),
        None => format!("Switched in {:.1} s", report.duration.as_secs_f64()),
    }
}

/// The sentence for a refused swap (#363): nothing changed, and the
/// numbers say why.
pub(crate) fn refused_decision_sentence(needed: u64, available: u64) -> String {
    format!(
        "Not enough free memory for this model (needs {}, {} available), even after stopping \
         the current one. Nothing changed; the current model keeps serving.",
        fmt_size(needed),
        fmt_size(available)
    )
}

/// The fixed NeedsDrain question (#363): verbatim, because it is the
/// contract the two buttons answer.
pub(crate) const NEEDS_DRAIN_SENTENCE: &str =
    "Not enough free memory to keep both models loaded. Switch after the current take \
     finishes? The engine is briefly unavailable.";

/// Which failure actions apply to one engine failure (#362): Retry always
/// (it re-runs the last model), the CPU engine exactly when Vulkan is the
/// failing or chosen family, and the manual escape always. Pure.
pub(crate) struct FailureActions {
    pub use_cpu: bool,
}

pub(crate) fn failure_actions(failure: &EngineFailure, backend: Option<Backend>) -> FailureActions {
    let vulkan_failed = match failure {
        EngineFailure::MissingLibrary {
            backend: Backend::Vulkan,
            ..
        } => true,
        EngineFailure::NoUsableEngine { rejected } => {
            rejected.iter().any(|(backend, _)| *backend == Backend::Vulkan)
        }
        _ => false,
    };
    FailureActions {
        use_cpu: backend == Some(Backend::Vulkan) || vulkan_failed,
    }
}

/// The mode radio rows (#362): built-in engine vs the user's own server.
fn mode_choice(app: &mut StarlingApp, cx: &mut Context<StarlingApp>) -> Div {
    let mut rows = div().flex().flex_col().gap(px(6.));
    for (mode, name, description) in [
        (
            EngineMode::Builtin,
            "Built-in engine",
            "The bundled engine runs and downloads models on this machine. Nothing leaves it.",
        ),
        (
            EngineMode::Manual,
            "My own server (advanced)",
            "A starling-serve or OpenAI-compatible endpoint you run yourself.",
        ),
    ] {
        let selected = app.draft_engine_mode == mode;
        rows = rows.child(
            div()
                .id(SharedString::from(format!("engine-mode-{}", mode_label(mode))))
                .flex()
                .flex_row()
                .gap(px(10.))
                .p(px(10.))
                .rounded(px(3.))
                .border_1()
                .border_color(if selected {
                    theme::SETTINGS_INK
                } else {
                    theme::SETTINGS_LINE
                })
                .cursor_pointer()
                .hover(|style| style.bg(theme::PAPER_HOVER))
                .on_click(cx.listener(move |this, _, _window, cx| {
                    this.pick_draft_engine_mode(mode, cx);
                }))
                .child(
                    div()
                        .mt(px(2.))
                        .size(px(10.))
                        .flex_none()
                        .rounded(px(5.))
                        .border_1()
                        .border_color(theme::SETTINGS_INK)
                        .when(selected, |dot| dot.bg(theme::SETTINGS_INK)),
                )
                .child(
                    div()
                        .flex()
                        .flex_col()
                        .gap(px(3.))
                        .child(
                            div()
                                .text_size(px(11.))
                                .font_weight(FontWeight::SEMIBOLD)
                                .child(name),
                        )
                        .child(
                            div()
                                .text_size(px(10.))
                                .line_height(px(10. * 1.5))
                                .text_color(theme::SETTINGS_HELPER)
                                .child(description),
                        ),
                ),
        );
    }
    rows
}

fn mode_label(mode: EngineMode) -> &'static str {
    match mode {
        EngineMode::Builtin => "builtin",
        EngineMode::Manual => "manual",
    }
}

/// The builtin-mode engine panel (#362, #363): backend line with every
/// notice, the CPU/auto toggle, failure sentence + actions, the model
/// list, the switch progress line, and pending memory decisions. All of
/// it is live manager state — the actions apply immediately, nothing here
/// waits for Save.
fn render_engine_section(app: &mut StarlingApp, cx: &mut Context<StarlingApp>) -> Div {
    let mut section = div().mt(px(21.)).flex().flex_col().gap(px(10.));

    // No manager at all: the startup failure is the whole story.
    let Some(snapshot) = app.engine_snapshot() else {
        let sentence = app.engine_unavailable();
        return section.child(engine_note(&sentence)).child(
            div()
                .flex()
                .flex_row()
                .gap(px(7.))
                .child(engine_action("engine-switch-manual", cx.listener(|this, _, _window, cx| {
                    this.engine_switch_to_manual(cx);
                }))
                .child("Switch to my own server")),
        );
    };

    // Backend line: the chosen family, the runtime device, every notice.
    let mut backend_line = String::new();
    if let Some(backend) = &snapshot.backend {
        backend_line = format!(
            "Engine: {} {}",
            backend.backend, backend.version
        );
        if let Some(device) = &backend.device {
            backend_line.push_str(&format!(" · device {device}"));
        }
    }
    section = section
        .when(!backend_line.is_empty(), |section| {
            section.child(
                div()
                    .text_size(px(10.))
                    .line_height(px(10. * 1.5))
                    .text_color(theme::SETTINGS_MUTED)
                    .child(backend_line),
            )
        })
        .children(
            snapshot
                .notices
                .iter()
                .map(|notice| engine_note(notice).into_any_element()),
        );

    // The CPU/auto toggle: shows what a click will do.
    let pinned_cpu = app.draft_backend_override.as_deref() == Some("cpu");
    section = section.child(
        div()
            .flex()
            .flex_row()
            .gap(px(7.))
            .child(
                engine_action(
                    "engine-backend-toggle",
                    cx.listener(|this, _, _window, cx| {
                        this.engine_toggle_cpu(cx);
                    }),
                )
                .child(if pinned_cpu {
                    "Use automatic engine"
                } else {
                    "Use CPU engine"
                }),
            )
            .child(
                div()
                    .text_size(px(10.))
                    .line_height(px(10. * 1.5))
                    .text_color(theme::SETTINGS_HELPER)
                    .child("Automatic prefers Vulkan and falls back to the CPU engine."),
            ),
    );

    // Failure sentence + actions.
    if let starling_dictation::engine::EnginePhase::Failed(failure) = &snapshot.phase {
        let actions = failure_actions(failure, snapshot.backend.as_ref().map(|b| b.backend));
        let mut row = div().flex().flex_row().gap(px(7.));
        row = row.child(
            engine_action("engine-retry", cx.listener(|this, _, _window, cx| {
                this.engine_retry(cx);
            }))
            .child("Retry"),
        );
        if actions.use_cpu && !pinned_cpu {
            row = row.child(
                engine_action(
                    "engine-failure-cpu",
                    cx.listener(|this, _, _window, cx| {
                        this.engine_toggle_cpu(cx);
                    }),
                )
                .child("Use CPU engine"),
            );
        }
        row = row.child(
            engine_action("engine-failure-manual", cx.listener(|this, _, _window, cx| {
                this.engine_switch_to_manual(cx);
            }))
            .child("Switch to my own server"),
        );
        section = section.child(engine_note(&failure.to_string())).child(row);
    }

    // The model list.
    section = section.child(model_list(app, &snapshot, cx));

    // The switch progress line and its report.
    if let Some(switch) = &snapshot.switch {
        let (note, steps) = switch_progress(&switch.stage);
        let mut line = div()
            .flex()
            .flex_wrap()
            .items_center()
            .gap(px(5.))
            .text_size(px(10.))
            .font(theme::mono_font());
        if let Some(note) = note {
            line = line
                .child(div().text_color(theme::SETTINGS_MUTED).child(note))
                .child(div().child("·"));
        }
        for (index, step) in steps.iter().enumerate() {
            if index > 0 {
                line = line.child(div().text_color(theme::SETTINGS_MUTED).child("→"));
            }
            line = line.child(
                div()
                    .text_color(if step.current {
                        theme::SETTINGS_INK
                    } else {
                        theme::SETTINGS_MUTED
                    })
                    .font_weight(if step.current {
                        FontWeight::SEMIBOLD
                    } else {
                        FontWeight::NORMAL
                    })
                    .child(step.word.clone()),
            );
        }
        section = section
            .child(line)
            .child(
                div()
                    .text_size(px(10.))
                    .line_height(px(10. * 1.5))
                    .text_color(theme::SETTINGS_HELPER)
                    .child("The current model keeps serving until the new one is ready."),
            );
    }
    if let Some(report) = &snapshot.last_switch {
        section = section.child(
            div()
                .text_size(px(10.))
                .line_height(px(10. * 1.5))
                .text_color(theme::SETTINGS_MUTED)
                .child(switch_report_line(report)),
        );
    }
    if let Some(error) = &snapshot.last_error {
        section = section.child(engine_note(&format!(
            "{error} The previous model is still active."
        )));
    }

    // Pending memory decisions (#363).
    match &snapshot.pending_decision {
        Some(SwapDecision::NeedsDrain { .. }) => {
            section = section
                .child(engine_note(NEEDS_DRAIN_SENTENCE))
                .child(
                    div()
                        .flex()
                        .flex_row()
                        .gap(px(7.))
                        .child(
                            engine_action(
                                "engine-drain-confirm",
                                cx.listener(|this, _, _window, cx| {
                                    this.engine_confirm_drain_swap(cx);
                                }),
                            )
                            .child("Switch after take"),
                        )
                        .child(
                            engine_action(
                                "engine-drain-cancel",
                                cx.listener(|this, _, _window, cx| {
                                    this.engine_cancel_switch(cx);
                                }),
                            )
                            .child("Cancel"),
                        ),
                );
        }
        Some(SwapDecision::Refused { needed, available }) => {
            section = section
                .child(engine_note(&refused_decision_sentence(*needed, *available)))
                .child(
                    engine_action(
                        "engine-refused-ok",
                        cx.listener(|this, _, _window, cx| {
                            // No switch is running to cancel; this only
                            // clears the decision so the panel moves on.
                            this.engine_cancel_switch(cx);
                        }),
                    )
                    .child("OK"),
                );
        }
        None => {}
    }

    section
}

/// A small emphasized note line: engine failures and notices carry their
/// own what-to-do sentences, so they are shown verbatim (#362).
fn engine_note(text: &str) -> Div {
    div()
        .flex()
        .flex_row()
        .items_start()
        .gap(px(6.))
        .p(px(9.))
        .bg(theme::SETTINGS_CALLOUT)
        .rounded(px(3.))
        .text_size(px(10.))
        .line_height(px(10. * 1.55))
        .text_color(theme::SETTINGS_MUTED)
        .child(icon("icons/alert-circle.svg", 13., theme::SETTINGS_MUTED))
        .child(div().flex_1().min_w_0().child(text.to_string()))
}

fn engine_action(
    id: &'static str,
    on_click: impl Fn(&gpui::ClickEvent, &mut Window, &mut gpui::App) + 'static,
) -> gpui::Stateful<Div> {
    div()
        .id(id)
        .flex_none()
        .px(px(10.))
        .py(px(7.))
        .rounded(px(3.))
        .border_1()
        .border_color(theme::SETTINGS_FOOT_LINE)
        .text_size(px(10.))
        .text_color(theme::SETTINGS_INK)
        .cursor_pointer()
        .hover(|style| style.bg(theme::PAPER_HOVER))
        .on_click(move |event, window, cx| on_click(event, window, cx))
}

/// One catalog row (#363): label (recommended marked), size, note,
/// install state — and the button that state allows. Delete is offered
/// only for an installed, inactive model with no switch running.
fn model_list(app: &StarlingApp, snapshot: &EngineSnapshot, cx: &mut Context<StarlingApp>) -> Div {
    let switching = snapshot.switch.is_some();
    // One check at a time: the host refuses a second while one runs. Only
    // the row being measured is held — disabling every installed row
    // while nothing on it says why reads as a glitch, and a row the user
    // presses anyway gets a precise refusal.
    let measuring = app.engine_speedcheck_status().and_then(|status| match status {
        starling_runtime_host::engine::SpeedCheckStatus::Running { model_id } => {
            Some(model_id.clone())
        }
        _ => None,
    });
    let mut rows = div().flex().flex_col().gap(px(6.));
    for model in &snapshot.models {
        let mut row = div()
            .id(SharedString::from(format!("engine-model-{}", model.id)))
            .flex()
            .flex_row()
            .items_center()
            .gap(px(10.))
            .p(px(10.))
            .rounded(px(3.))
            .border_1()
            .border_color(if model.active {
                theme::SETTINGS_INK
            } else {
                theme::SETTINGS_LINE
            });

        let mut info = div().flex().flex_col().gap(px(3.)).flex_1().min_w_0();
        info = info.child(
            div()
                .flex()
                .flex_row()
                .items_center()
                .gap(px(6.))
                .child(
                    div()
                        .text_size(px(11.))
                        .font_weight(FontWeight::SEMIBOLD)
                        .child(model.label.clone()),
                )
                .when(model.recommended, |label| {
                    label.child(
                        div()
                            .px(px(5.))
                            .py(px(1.))
                            .rounded(px(2.))
                            .border_1()
                            .border_color(theme::SETTINGS_LINE)
                            .font(theme::mono_font())
                            .text_size(px(8.))
                            .text_color(theme::SETTINGS_MUTED)
                            .child("RECOMMENDED"),
                    )
                }),
        );
        info = info.child(
            div()
                .text_size(px(10.))
                .line_height(px(10. * 1.5))
                .text_color(theme::SETTINGS_HELPER)
                .child(format!("{} · {}", fmt_size(model.size_bytes), model.note)),
        );

        // Buttons by install state.
        let mut buttons = div().flex().flex_row().items_center().gap(px(6.));
        match &model.install {
            InstallState::NotInstalled => {
                buttons = buttons.child(
                    engine_action(
                        "model-download",
                        {
                            let id = model.id.clone();
                            cx.listener(move |this, _, _window, cx| {
                                this.engine_download(&id, cx);
                            })
                        },
                    )
                    .child("Download"),
                );
            }
            InstallState::Downloading { done, total } => {
                buttons = buttons
                    .child(
                        div()
                            .font(theme::mono_font())
                            .text_size(px(9.))
                            .text_color(theme::SETTINGS_MUTED)
                            .child(format!("{}%", download_percent(*done, *total))),
                    )
                    .child(
                        engine_action(
                            "model-cancel-download",
                            {
                                let id = model.id.clone();
                                cx.listener(move |this, _, _window, cx| {
                                    this.engine_cancel_download(&id, cx);
                                })
                            },
                        )
                        .child("Cancel"),
                    );
            }
            InstallState::Verifying => {
                buttons = buttons.child(
                    div()
                        .font(theme::mono_font())
                        .text_size(px(9.))
                        .text_color(theme::SETTINGS_MUTED)
                        .child("Verifying…"),
                );
            }
            InstallState::NeedsVerification | InstallState::Installed => {
                if model.active {
                    buttons = buttons.child(
                        div()
                            .px(px(8.))
                            .py(px(4.))
                            .rounded(px(3.))
                            .bg(theme::SETTINGS_INK)
                            .text_size(px(9.))
                            .text_color(theme::SETTINGS_PRIMARY_TEXT)
                            .child("Active"),
                    );
                } else {
                    buttons = buttons.child(
                        engine_action(
                            "model-activate",
                            {
                                let id = model.id.clone();
                                cx.listener(move |this, _, _window, cx| {
                                    this.engine_activate(&id, cx);
                                })
                            },
                        )
                        .child("Activate"),
                    );
                }
                // The speed check measures the model, not the choice:
                // offered for the installed, active or not (#475).
                if measuring.as_deref() != Some(model.id.as_str()) {
                    buttons = buttons.child(
                        div()
                            .id(SharedString::from(format!(
                                "model-speedcheck-{}",
                                model.id
                            )))
                            .flex_none()
                            .px(px(10.))
                            .py(px(7.))
                            .rounded(px(3.))
                            .border_1()
                            .border_color(theme::SETTINGS_FOOT_LINE)
                            .text_size(px(10.))
                            .text_color(theme::SETTINGS_INK)
                            .cursor_pointer()
                            .hover(|style| style.bg(theme::PAPER_HOVER))
                            .on_click({
                                let id = model.id.clone();
                                cx.listener(move |this, _, _window, cx| {
                                    this.engine_speedcheck(&id, cx);
                                })
                            })
                            .child("Speed check"),
                    );
                }
            }
            InstallState::Failed(message) => {
                info = info.child(
                    div()
                        .text_size(px(10.))
                        .line_height(px(10. * 1.5))
                        .text_color(theme::FAILED_COPY)
                        .child(message.clone()),
                );
                buttons = buttons.child(
                    engine_action(
                        "model-download",
                        {
                            let id = model.id.clone();
                            cx.listener(move |this, _, _window, cx| {
                                this.engine_download(&id, cx);
                            })
                        },
                    )
                    .child("Download"),
                );
            }
        }
        if matches!(
            model.install,
            InstallState::Installed | InstallState::NeedsVerification
        ) && !model.active
            && !switching
        {
            buttons = buttons.child(
                div()
                    .id(SharedString::from(format!("model-delete-{}", model.id)))
                    .flex_none()
                    .px(px(8.))
                    .py(px(6.))
                    .rounded(px(3.))
                    .border_1()
                    .border_color(theme::SETTINGS_FOOT_LINE)
                    .text_size(px(10.))
                    .text_color(theme::DANGER)
                    .cursor_pointer()
                    .hover(|style| style.bg(theme::PAPER_HOVER))
                    .on_click({
                        let id = model.id.clone();
                        cx.listener(move |this, _, _window, cx| {
                            this.engine_delete_model(&id, cx);
                        })
                    })
                    .child("Delete"),
            );
        }
        // The check's outcome lives under the row it measured:
        // one line per probe, plus the failure sentence.
        if let Some(status) = app.engine_speedcheck_status() {
            if status.model_id() == model.id {
                for line in speedcheck_lines(status) {
                    info = info.child(
                        div()
                            .mt(px(4.))
                            .font(theme::mono_font())
                            .text_size(px(9.))
                            .line_height(px(13.))
                            .text_color(theme::SETTINGS_HELPER)
                            .child(line),
                    );
                }
            }
        }
        row = row.child(info).child(buttons);
        rows = rows.child(row);
    }
    rows
}

/// The lines a model row shows for the host's per-model speed check
/// (#475), in the order the probes ran.
///
/// The arithmetic is the shared one (`starling_dictation::speedcheck`):
/// this only renders it. What it must not do is invent a number — an
/// unmeasurable probe reports unknown, and the synthetic one is always
/// labelled as the upper bound it is (a tone decodes faster than speech).
fn speedcheck_lines(status: &starling_runtime_host::engine::SpeedCheckStatus) -> Vec<String> {
    use starling_runtime_host::engine::SpeedCheckStatus;
    match status {
        SpeedCheckStatus::Running { .. } => vec!["Measuring…".to_string()],
        SpeedCheckStatus::Failed { message, .. } => vec![format!("Speed check: {message}")],
        // The wording — including the sentence that keeps a synthetic
        // probe from being read as real-speech speed — comes from the
        // type that measured it, not from a second copy of it here.
        SpeedCheckStatus::Reported { results, .. } => {
            results.iter().map(|result| result.describe()).collect()
        }
    }
}

/// Processing after transcription (#295): one row per built-in mode, the
/// draft mode's destination and context fields (shown before the mode is
/// saved), and the fields its provider needs.
fn render_processing_section(app: &mut StarlingApp, cx: &mut Context<StarlingApp>) -> Div {
    let draft = app.draft_processing_settings(cx);
    let disclosure = processing::disclosure(&draft.mode, &draft);
    let mut rows = div().flex().flex_col().gap(px(6.));
    for mode in &processing::modes().profiles {
        let selected = mode.id == draft.mode;
        let id = mode.id.clone();
        rows = rows.child(
            div()
                .id(gpui::SharedString::from(format!("mode-{}", mode.id)))
                .flex()
                .flex_row()
                .gap(px(10.))
                .p(px(10.))
                .rounded(px(3.))
                .border_1()
                .border_color(if selected {
                    theme::SETTINGS_INK
                } else {
                    theme::SETTINGS_LINE
                })
                .cursor_pointer()
                .hover(|style| style.bg(theme::PAPER_HOVER))
                .on_click(cx.listener(move |this, _, _window, cx| {
                    this.pick_draft_mode(&id, cx);
                }))
                .child(
                    div()
                        .mt(px(2.))
                        .size(px(10.))
                        .flex_none()
                        .rounded(px(5.))
                        .border_1()
                        .border_color(theme::SETTINGS_INK)
                        .when(selected, |dot| dot.bg(theme::SETTINGS_INK)),
                )
                .child(
                    div()
                        .flex()
                        .flex_col()
                        .gap(px(3.))
                        .child(
                            div()
                                .text_size(px(11.))
                                .font_weight(FontWeight::SEMIBOLD)
                                .child(mode.name.clone()),
                        )
                        .child(
                            div()
                                .text_size(px(10.))
                                .line_height(px(10. * 1.5))
                                .text_color(theme::SETTINGS_HELPER)
                                .child(mode.description.clone()),
                        ),
                ),
        );
    }

    let route = processing::mode(&draft.mode).authoring_route.clone();
    let mut fields = div().flex().flex_col();
    if route.as_deref() == Some(processing::S1_ROUTE) {
        fields = fields.child(
            field_label("S1-mini server (on this computer)")
                .child(app.draft_s1_endpoint.clone())
                .child(helper(
                    "A second starling-serve running the S1-mini GGUF. It cannot share the \
                     transcription server, which keeps one model loaded.",
                )),
        );
    }
    if route.as_deref() == Some(processing::API_ROUTE) {
        fields = fields
            .child(field_label("API endpoint (OpenAI-compatible)").child(app.draft_api_endpoint.clone()))
            .child(field_label("API model").child(app.draft_api_model.clone()))
            .child(
                field_label("API key environment variable")
                    .child(app.draft_api_key_env.clone())
                    .child(helper("The key is read from this variable; it is never written to settings.")),
            );
    }

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
                .child("AFTER TRANSCRIPTION"),
        )
        .child(rows)
        .child(
            div()
                .id("processing-disclosure")
                .mt(px(12.))
                .bg(theme::SETTINGS_CALLOUT)
                .p(px(12.))
                .text_size(px(10.))
                .line_height(px(10. * 1.55))
                .child(disclosure),
        )
        .child(fields)
}

/// The activation choices (#221), in the order the dialog lists them.
pub(crate) const ACTIVATION_CHOICES: [(ActivationMode, &str, &str); 3] = [
    (
        ActivationMode::HoldOrToggle,
        "Hold or tap",
        "Hold the shortcut to talk and let go to finish, or tap it once to keep recording \
         until you press it again.",
    ),
    (
        ActivationMode::Hold,
        "Hold to talk",
        "Recording runs while the shortcut is held and finishes when you let go.",
    ),
    (
        ActivationMode::Toggle,
        "Toggle",
        "Press once to start, press again to finish.",
    ),
];

fn render_dictation_section(app: &mut StarlingApp, cx: &mut Context<StarlingApp>) -> Div {
    let mut rows = div().flex().flex_col().gap(px(6.));
    for (mode, name, description) in ACTIVATION_CHOICES {
        let selected = app.draft_activation == mode;
        rows = rows.child(
            choice_row(
                SharedString::from(format!("activation-{name}")),
                selected,
                name,
                description,
                true,
            )
            .on_click(cx.listener(move |this, _, _window, cx| {
                this.draft_activation = mode;
                cx.notify();
            })),
        );
    }
    let hold = app.draft_activation == ActivationMode::Hold;
    let double_tap = app.draft_double_tap;
    let double_tap_row = choice_row(
        SharedString::from("double-tap-hands-free"),
        double_tap,
        "Double tap for hands-free",
        if hold {
            "Tap the shortcut twice quickly to keep recording without holding it; press it \
             again to finish."
        } else {
            "Only for Hold to talk: the other modes already keep recording after a tap."
        },
        hold,
    )
    .when(!hold, |row| row.opacity(0.5))
    .when(hold, |row| {
        row.on_click(cx.listener(|this, _, _window, cx| {
            this.draft_double_tap = !this.draft_double_tap;
            cx.notify();
        }))
    });

    let portal_bound = app.portal_shortcuts.as_ref().is_some_and(|portal| portal.is_bound());
    let reach = crate::shortcut::reach_note(&app.shortcut_registration, &app.shortcut, portal_bound);
    let mut field = field_label("Recording shortcut")
        .child(app.draft_shortcut.clone())
        .child(helper(
            "Modifiers and one key, for example Ctrl+Shift+Space, Alt+D, or a single F9. \
             Escape cancels a take; recording never brings this window forward.",
        ));
    if let Some(reason) = app.dictation_draft_error.clone() {
        field = field.child(
            div()
                .id("shortcut-error")
                .mt(px(2.))
                .font_weight(FontWeight::NORMAL)
                .line_height(px(10. * 1.5))
                .text_color(theme::CORAL)
                .child(reason),
        );
    }

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
                .child("DICTATION"),
        )
        .child(field)
        .child(rows)
        .child(div().mt(px(6.)).child(double_tap_row))
        .child(
            div()
                .id("shortcut-reach")
                .mt(px(12.))
                .bg(theme::SETTINGS_CALLOUT)
                .p(px(12.))
                .text_size(px(10.))
                .line_height(px(10. * 1.55))
                .child(reach),
        )
        .children(crate::views::system_check::render_desktop_shortcut(app, cx))
        .children(crate::views::system_check::render_system_check(app, cx))
}

/// "During recording" (#361).
const PLAYBACK_CHOICES: [(PlaybackMode, &str, &str); 3] = [
    (
        PlaybackMode::Off,
        "Off",
        "Playback keeps playing while you record.",
    ),
    (
        PlaybackMode::Lower,
        "Lower volume",
        "Playback is lowered to the level below while you record and restored when recording \
         stops.",
    ),
    (
        PlaybackMode::Mute,
        "Mute",
        "Playback is muted while you record and unmuted when recording stops.",
    ),
];

fn render_playback_section(app: &mut StarlingApp, cx: &mut Context<StarlingApp>) -> Div {
    let mut rows = div().flex().flex_col().gap(px(6.));
    for (mode, name, description) in PLAYBACK_CHOICES {
        let selected = app.draft_playback_mode == mode;
        rows = rows.child(
            choice_row(
                SharedString::from(format!("playback-{name}")),
                selected,
                name,
                description,
                true,
            )
            .on_click(cx.listener(move |this, _, _window, cx| {
                this.draft_playback_mode = mode;
                cx.notify();
            })),
        );
    }

    // Dimmed outside Lower mode, like the double-tap row.
    let lowering = app.draft_playback_mode == PlaybackMode::Lower;
    let level = app.draft_lower_level.read(cx).value();
    let slider_row = div()
        .id("playback-lower-level")
        .flex()
        .flex_col()
        .gap(px(6.))
        .p(px(10.))
        .rounded(px(3.))
        .border_1()
        .border_color(if lowering {
            theme::SETTINGS_INK
        } else {
            theme::SETTINGS_LINE
        })
        .when(!lowering, |row| row.opacity(0.5))
        .child(
            div()
                .flex()
                .flex_row()
                .justify_between()
                .text_size(px(11.))
                .font_weight(FontWeight::SEMIBOLD)
                .child("Lower playback to")
                .child(
                    div()
                        .font(theme::mono_font())
                        .font_weight(FontWeight::NORMAL)
                        .child(format!("{level}%")),
                ),
        )
        .child(app.draft_lower_level.clone())
        .child(helper("A level above your current volume leaves it unchanged."));

    // Known only once a take has tried.
    let unsupported = app.playback.handle().unsupported_reason();
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
                .child("PLAYBACK"),
        )
        .child(
            div()
                .mb(px(13.))
                .text_size(px(11.))
                .line_height(px(11. * 1.65))
                .text_color(theme::SETTINGS_MUTED)
                .child(
                    "What happens to other audio while you dictate. Your volume and mute settings \
                     are restored when each recording ends.",
                ),
        )
        .child(rows)
        .child(div().mt(px(6.)).child(slider_row));
    if let Some(reason) = unsupported {
        section = section.child(
            div()
                .id("playback-unsupported")
                .mt(px(12.))
                .bg(theme::SETTINGS_CALLOUT)
                .p(px(12.))
                .text_size(px(10.))
                .line_height(px(10. * 1.55))
                .child(format!(
                    "Playback cannot be adjusted on this system ({reason}); recordings leave it \
                     as it is."
                )),
        );
    }
    section
}

/// One selectable row: a radio dot, a name, and a description. A
/// disabled row carries no pointer or hover affordance — there is
/// nothing to click.
pub(super) fn choice_row(
    id: SharedString,
    selected: bool,
    name: &'static str,
    description: &'static str,
    enabled: bool,
) -> gpui::Stateful<Div> {
    div()
        .id(id)
        .flex()
        .flex_row()
        .gap(px(10.))
        .p(px(10.))
        .rounded(px(3.))
        .border_1()
        .border_color(if selected {
            theme::SETTINGS_INK
        } else {
            theme::SETTINGS_LINE
        })
        .when(enabled, |row| {
            row.cursor_pointer()
                .hover(|style| style.bg(theme::PAPER_HOVER))
        })
        .child(
            div()
                .mt(px(2.))
                .size(px(10.))
                .flex_none()
                .rounded(px(5.))
                .border_1()
                .border_color(theme::SETTINGS_INK)
                .when(selected, |dot| dot.bg(theme::SETTINGS_INK)),
        )
        .child(
            // Shrinks to the row so a long description wraps.
            div()
                .flex_1()
                .min_w_0()
                .flex()
                .flex_col()
                .gap(px(3.))
                .child(
                    div()
                        .text_size(px(11.))
                        .font_weight(FontWeight::SEMIBOLD)
                        .child(name),
                )
                .child(
                    div()
                        .text_size(px(10.))
                        .line_height(px(10. * 1.5))
                        .text_color(theme::SETTINGS_HELPER)
                        .child(description),
                ),
        )
}

pub(super) fn helper(text: impl Into<SharedString>) -> Div {
    div()
        .mt(px(2.))
        .text_size(px(10.))
        .font_weight(FontWeight::NORMAL)
        .line_height(px(10. * 1.5))
        .text_color(theme::SETTINGS_HELPER)
        .child(text.into())
}

