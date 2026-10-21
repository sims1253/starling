//! The settings sheet: a scrim over the window, the page list on the
//! left, the chosen page on the right, and Save always in reach at the
//! foot. Pages: the Engine (mode, models, switches — #362/#363, or the
//! manual server's fields), then dictation, input, and what happens to a
//! take afterwards.

use std::time::Duration;

use gpui::{
    Animation, AnimationExt, Context, Div, FontWeight, MouseButton, SharedString, Window, div,
    prelude::*, px,
};
use starling_dictation::engine::{
    Backend, EngineFailure, EngineSnapshot, InstallState, SwapDecision, SwitchReport, SwitchStage,
};
use starling_dictation::settings::{ActivationMode, EngineMode, PlaybackMode};

use crate::app::{ConnectionProbe, StarlingApp, settings_callout_view};
use crate::processing;
use crate::theme;
use crate::views::kit::{self, Tone};
use crate::views::{SETTINGS_CALLOUT_DOT_ID, icon, status_dot};

/// The pages of the settings sheet, in the order the list shows them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SettingsPage {
    Engine,
    Dictation,
    Microphone,
    Playback,
    Feedback,
    LivePreview,
    HistoryAudio,
    AfterTake,
    AfterTranscription,
}

impl SettingsPage {
    pub const ALL: [SettingsPage; 9] = [
        SettingsPage::Engine,
        SettingsPage::Dictation,
        SettingsPage::Microphone,
        SettingsPage::Playback,
        SettingsPage::Feedback,
        SettingsPage::LivePreview,
        SettingsPage::HistoryAudio,
        SettingsPage::AfterTake,
        SettingsPage::AfterTranscription,
    ];

    pub fn label(self) -> &'static str {
        match self {
            SettingsPage::Engine => "Engine",
            SettingsPage::Dictation => "Dictation",
            SettingsPage::Microphone => "Microphone",
            SettingsPage::Playback => "Playback",
            SettingsPage::Feedback => "Overlay and sounds",
            SettingsPage::LivePreview => "Live preview",
            SettingsPage::HistoryAudio => "History audio",
            SettingsPage::AfterTake => "After a take",
            SettingsPage::AfterTranscription => "After transcription",
        }
    }

    fn id(self) -> &'static str {
        match self {
            SettingsPage::Engine => "settings-page-engine",
            SettingsPage::Dictation => "settings-page-dictation",
            SettingsPage::Microphone => "settings-page-microphone",
            SettingsPage::Playback => "settings-page-playback",
            SettingsPage::Feedback => "settings-page-feedback",
            SettingsPage::LivePreview => "settings-page-live-preview",
            SettingsPage::HistoryAudio => "settings-page-history-audio",
            SettingsPage::AfterTake => "settings-page-after-take",
            SettingsPage::AfterTranscription => "settings-page-after-transcription",
        }
    }
}

pub fn render_settings_modal(
    app: &mut StarlingApp,
    window: &mut Window,
    cx: &mut Context<StarlingApp>,
) -> impl IntoElement {
    let manual = app.draft_engine_mode == EngineMode::Manual;
    let probing = matches!(app.probe, Some(ConnectionProbe::Testing { .. }));
    let page = app.settings_page;
    let viewport = window.viewport_size();
    let height = (viewport.height - px(48.)).min(px(720.));

    let content: Div = match page {
        SettingsPage::Engine => render_engine_page(app, cx),
        SettingsPage::Dictation => render_dictation_section(app, cx),
        SettingsPage::Microphone => crate::views::microphone::render_microphone_section(app, cx),
        SettingsPage::Playback => render_playback_section(app, cx),
        SettingsPage::Feedback => crate::views::feedback::render_feedback_section(app, cx),
        SettingsPage::LivePreview => {
            crate::views::live_preview::render_live_preview_section(app, cx)
        }
        SettingsPage::HistoryAudio => crate::views::storage::render_storage_section(app, cx),
        SettingsPage::AfterTake => crate::views::delivery::render_insertion_section(app, cx),
        SettingsPage::AfterTranscription => render_processing_section(app, cx),
    };

    let mut nav = div()
        .id("settings-nav")
        .w(px(210.))
        .flex_none()
        .flex()
        .flex_col()
        .gap(px(2.))
        .px(px(12.))
        .py(px(20.))
        .bg(theme::SHEET)
        .border_r_1()
        .border_color(theme::SHEET_LINE)
        .child(
            div()
                .px(px(10.))
                .pb(px(14.))
                .font(theme::display_font())
                .text_size(px(theme::HEADING))
                .text_color(theme::SHEET_INK)
                .child("Settings"),
        );
    for entry in SettingsPage::ALL {
        let current = entry == page;
        nav = nav.child(
            div()
                .id(entry.id())
                .relative()
                .px(px(10.))
                .py(px(7.))
                .rounded(px(6.))
                .text_size(px(theme::TEXT_MD))
                .text_color(if current {
                    theme::SHEET_INK
                } else {
                    theme::SHEET_QUILL
                })
                .when(current, |item| {
                    item.bg(theme::SHEET_CARD).font_weight(FontWeight::SEMIBOLD)
                })
                .when(!current, |item| {
                    item.cursor_pointer()
                        .hover(|style| style.bg(theme::SHEET_HOVER).text_color(theme::SHEET_INK))
                        .on_click(cx.listener(move |this, _, _window, cx| {
                            this.settings_page = entry;
                            cx.notify();
                        }))
                })
                .child(entry.label()),
        );
    }

    let footer = div()
        .flex_none()
        .flex()
        .flex_row()
        .items_center()
        .justify_end()
        .gap(px(8.))
        .px(px(28.))
        .py(px(14.))
        .border_t_1()
        .border_color(theme::SHEET_LINE)
        .child(
            div()
                .flex_1()
                .text_size(px(theme::TEXT_SM))
                .text_color(theme::SHEET_QUILL)
                .child("Engine actions apply at once; everything else applies on Save."),
        )
        .when(manual, |footer| {
            // B06 (#207): one probe at a time — while the newest press is
            // in flight the button carries no click handler at all (and
            // `test_connection` drops re-entrant presses as the
            // state-layer twin of the same guard).
            footer.child(
                kit::button("test-connection", Tone::SheetQuiet, probing)
                    .when(!probing, |button| {
                        button.on_click(cx.listener(|this, _, _window, cx| {
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
            kit::button("save-settings", Tone::SheetPrimary, false)
                .on_click(cx.listener(|this, _, _window, cx| {
                    this.save_settings(cx);
                }))
                .child("Save settings"),
        );

    let card = div()
        .id("settings-card")
        .on_mouse_down(
            MouseButton::Left,
            cx.listener(|_, _, _window, cx| {
                cx.stop_propagation();
            }),
        )
        .w(px(880.))
        .max_w_full()
        .h(height)
        .flex()
        .flex_row()
        .overflow_hidden()
        .rounded(px(10.))
        .bg(theme::SHEET_CARD)
        .text_color(theme::SHEET_INK)
        .child(nav)
        .child(
            div()
                .flex_1()
                .min_w_0()
                .flex()
                .flex_col()
                .child(
                    div()
                        .flex_none()
                        .flex()
                        .flex_row()
                        .items_center()
                        .justify_between()
                        .pl(px(28.))
                        .pr(px(14.))
                        .pt(px(18.))
                        .pb(px(12.))
                        .child(
                            div()
                                .font(theme::display_font())
                                .text_size(px(30.))
                                .text_color(theme::SHEET_INK)
                                .child(page.label()),
                        )
                        .child(
                            div()
                                .id("settings-close")
                                .size(px(34.))
                                .flex()
                                .items_center()
                                .justify_center()
                                .rounded(px(6.))
                                .cursor_pointer()
                                .hover(|style| style.bg(theme::SHEET_HOVER))
                                .on_click(cx.listener(|this, _, _window, cx| {
                                    this.close_settings(cx);
                                }))
                                .child(icon("icons/x.svg", 18., theme::SHEET_QUILL)),
                        ),
                )
                .child(
                    div()
                        .id(SharedString::from(format!("{}-body", page.id())))
                        .flex_1()
                        .min_h_0()
                        .overflow_y_scroll()
                        .px(px(28.))
                        .pb(px(28.))
                        .text_size(px(theme::TEXT_MD))
                        .child(content),
                )
                .child(footer),
        );

    let animated_card = card.with_animation(
        "modal-fade",
        Animation::new(Duration::from_millis(160)),
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

/// The Engine page: which engine transcribes (#362, #363), its models or
/// the manual server's fields, and the words to watch for.
fn render_engine_page(app: &mut StarlingApp, cx: &mut Context<StarlingApp>) -> Div {
    // B06 (#207): the status line belongs to the probe once one ran — a
    // draft's failure shows here without touching the live connection —
    // and without a probe the committed status shows. Manual mode only:
    // builtin mode has no manual endpoint, and its engine state fills the
    // callout's place instead.
    let manual = app.draft_engine_mode == EngineMode::Manual;
    let callout = settings_callout_view(app.probe.as_ref(), app.connection, &app.endpoint);
    let draft_endpoint = app.draft_endpoint.clone();
    let draft_model = app.draft_model.clone();
    let draft_terms = app.draft_terms.clone();
    let engine_section = render_engine_section(app, cx);

    div()
        .flex()
        .flex_col()
        .child(kit::sheet_intro(if manual {
            "Your own starling-serve or OpenAI-compatible transcription endpoint."
        } else {
            "Dictation runs on the bundled engine on this machine; models are downloaded and \
             switched here."
        }))
        .child(div().mt(px(16.)).child(mode_choice(app, cx)))
        .children((!manual).then_some(engine_section))
        // ---- Manual server fields (manual mode only) ----------------
        .when(manual, |page| {
            page.child(kit::sheet_subheading("Connection").mt(px(22.)))
                .child(field_label("Server endpoint").child(draft_endpoint))
                .child(field_label("Model").child(draft_model))
        })
        .child(
            field_label("Words to watch").child(draft_terms).child(helper(
                "Comma-separated terms are checked after transcription. They are never inserted \
                 or substituted.",
            )),
        )
        .when(manual, |page| {
            page.child(
                div()
                    .flex()
                    .flex_row()
                    .items_center()
                    .gap(px(12.))
                    .p(px(12.))
                    .mt(px(8.))
                    .rounded(px(6.))
                    .bg(theme::SHEET_CALLOUT)
                    .child(status_dot(SETTINGS_CALLOUT_DOT_ID, callout.dot))
                    .child(
                        div()
                            .flex()
                            .flex_col()
                            .gap(px(2.))
                            .text_size(px(theme::TEXT_SM))
                            .child(div().font_weight(FontWeight::SEMIBOLD).child(callout.title))
                            .child(div().text_color(theme::SHEET_QUILL).child(callout.detail)),
                    ),
            )
        })
}

fn field_label(label: &str) -> Div {
    div()
        .flex()
        .flex_col()
        .gap(px(7.))
        .mt(px(18.))
        .text_size(px(theme::TEXT_MD))
        .font_weight(FontWeight::SEMIBOLD)
        .text_color(theme::SHEET_INK)
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

/// A byte size as the settings rows show it (#363) — the one shared
/// spelling next to the memory policy, so the speed check's refusal
/// and this screen's sentences cannot drift apart.
pub(crate) fn fmt_size(bytes: u64) -> String {
    starling_dictation::engine::fmt_size(bytes)
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
                .gap(px(12.))
                .p(px(12.))
                .rounded(px(6.))
                .border_1()
                .border_color(if selected {
                    theme::SHEET_BEAK
                } else {
                    theme::SHEET_LINE
                })
                .cursor_pointer()
                .hover(|style| style.bg(theme::SHEET_HOVER))
                .on_click(cx.listener(move |this, _, _window, cx| {
                    this.pick_draft_engine_mode(mode, cx);
                }))
                .child(
                    kit::radio(selected),
                )
                .child(
                    div()
                        .flex()
                        .flex_col()
                        .flex_1()
                        .min_w_0()
                        .gap(px(3.))
                        .child(
                            div()
                                .text_size(px(theme::TEXT_MD))
                                .font_weight(FontWeight::SEMIBOLD)
                                .child(name),
                        )
                        .child(
                            div()
                                .text_size(px(theme::TEXT_SM))
                                .line_height(px(theme::TEXT_SM * 1.5))
                                .text_color(theme::SHEET_QUILL)
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
                    .text_size(px(theme::TEXT_SM))
                    .line_height(px(theme::TEXT_SM * 1.5))
                    .text_color(theme::SHEET_QUILL)
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
            .items_center()
            .gap(px(10.))
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
                    .text_size(px(theme::TEXT_SM))
                    .line_height(px(theme::TEXT_SM * 1.5))
                    .text_color(theme::SHEET_QUILL)
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
            .gap(px(6.))
            .text_size(px(theme::TEXT_SM));
        if let Some(note) = note {
            line = line.child(div().text_color(theme::SHEET_QUILL).child(format!("{note} ")));
        }
        for (index, step) in steps.iter().enumerate() {
            if index > 0 {
                line = line.child(div().text_color(theme::SHEET_QUILL).child("→"));
            }
            line = line.child(
                div()
                    .text_color(if step.current {
                        theme::SHEET_INK
                    } else {
                        theme::SHEET_QUILL
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
                    .text_size(px(theme::TEXT_SM))
                    .line_height(px(theme::TEXT_SM * 1.5))
                    .text_color(theme::SHEET_QUILL)
                    .child("The current model keeps serving until the new one is ready."),
            );
    }
    if let Some(report) = &snapshot.last_switch {
        section = section.child(
            div()
                .text_size(px(theme::TEXT_SM))
                .line_height(px(theme::TEXT_SM * 1.5))
                .text_color(theme::SHEET_QUILL)
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
        .p(px(12.))
        .bg(theme::SHEET_CALLOUT)
        .rounded(px(6.))
        .text_size(px(theme::TEXT_SM))
        .line_height(px(theme::TEXT_SM * 1.55))
        .text_color(theme::SHEET_INK)
        .child(div().pt(px(2.)).child(icon("icons/alert-circle.svg", 14., theme::SHEET_QUILL)))
        .child(div().flex_1().min_w_0().child(text.to_string()))
}

fn engine_action(
    id: &'static str,
    on_click: impl Fn(&gpui::ClickEvent, &mut Window, &mut gpui::App) + 'static,
) -> gpui::Stateful<Div> {
    kit::button(id, Tone::SheetQuiet, false)
        .on_click(move |event, window, cx| on_click(event, window, cx))
}

/// One catalog row (#363): label (recommended marked), size, note,
/// install state — and the button that state allows. Delete is offered
/// only for an installed, inactive model with no switch running.
fn model_list(app: &StarlingApp, snapshot: &EngineSnapshot, cx: &mut Context<StarlingApp>) -> Div {
    let switching = snapshot.switch.is_some();
    // The speed-check status is read once for the whole list: the
    // per-row "Measuring…" gate and the per-row result lines share the
    // same borrow, so the render does not walk it twice per row.
    let status = app.engine_speedcheck_status();
    // One check at a time: the host refuses a second while one runs. Only
    // the row being measured is held — disabling every installed row
    // while nothing on it says why reads as a glitch, and a row the user
    // presses anyway gets a precise refusal.
    let measuring = status.and_then(|status| match status {
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
            .rounded(px(6.))
            .border_1()
            .border_color(if model.active {
                theme::SHEET_BEAK
            } else {
                theme::SHEET_LINE
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
                        .text_size(px(theme::TEXT_MD))
                        .font_weight(FontWeight::SEMIBOLD)
                        .child(model.label.clone()),
                )
                .when(model.recommended, |label| {
                    label.child(
                        div()
                            .px(px(7.))
                            .py(px(1.))
                            .rounded_full()
                            .bg(theme::SHEET_CALLOUT)
                            .text_size(px(theme::TEXT_XS))
                            .text_color(theme::SHEET_BEAK)
                            .child("Recommended"),
                    )
                }),
        );
        info = info.child(
            div()
                .text_size(px(theme::TEXT_SM))
                .line_height(px(theme::TEXT_SM * 1.5))
                .text_color(theme::SHEET_QUILL)
                .child(format!("{}. {}", fmt_size(model.size_bytes), model.note)),
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
                            .text_size(px(theme::TEXT_SM))
                            .text_color(theme::SHEET_QUILL)
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
                        .text_size(px(theme::TEXT_SM))
                        .text_color(theme::SHEET_QUILL)
                        .child("Verifying…"),
                );
            }
            InstallState::NeedsVerification | InstallState::Installed => {
                if model.active {
                    buttons = buttons.child(
                        div()
                            .flex()
                            .flex_row()
                            .items_center()
                            .gap(px(6.))
                            .h(px(30.))
                            .px(px(4.))
                            .text_size(px(theme::TEXT_SM))
                            .font_weight(FontWeight::SEMIBOLD)
                            .text_color(theme::SHEET_INK)
                            .child(icon("icons/check.svg", 14., theme::SHEET_BEAK))
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
                        kit::button(
                            SharedString::from(format!("model-speedcheck-{}", model.id)),
                            Tone::SheetQuiet,
                            false,
                        )
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
                        .text_size(px(theme::TEXT_SM))
                        .line_height(px(theme::TEXT_SM * 1.5))
                        .text_color(theme::SHEET_ROSE)
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
                kit::button(
                    SharedString::from(format!("model-delete-{}", model.id)),
                    Tone::SheetDanger,
                    false,
                )
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
        for line in speedcheck_status_lines(status, &model.id) {
            info = info.child(
                div()
                    .mt(px(4.))
                    .text_size(px(theme::TEXT_XS))
                    .line_height(px(theme::TEXT_XS * 1.5))
                    .text_color(theme::SHEET_QUILL)
                    .child(line),
            );
        }
        row = row.child(info).child(buttons);
        rows = rows.child(row);
    }
    rows
}

/// The lines `model_id`'s row shows for the host's per-model speed
/// check (#475), in the order the probes ran. `status` is the whole
/// list's status, shared across the rows: only the row it belongs to
/// returns anything.
///
/// The arithmetic is the shared one (`starling_dictation::speedcheck`):
/// this only renders it. What it must not do is invent a number — an
/// unmeasurable probe reports unknown, and the synthetic one is always
/// labelled as the upper bound it is (a tone decodes faster than speech).
fn speedcheck_status_lines(
    status: Option<&starling_runtime_host::engine::SpeedCheckStatus>,
    model_id: &str,
) -> Vec<String> {
    let Some(status) = status else { return Vec::new() };
    if status.model_id() != model_id {
        return Vec::new();
    }
    match status {
        starling_runtime_host::engine::SpeedCheckStatus::Running { .. } => {
            vec!["Measuring…".to_string()]
        }
        starling_runtime_host::engine::SpeedCheckStatus::Failed { message, .. } => {
            vec![format!("Speed check: {message}")]
        }
        // The wording — including the sentence that keeps a synthetic
        // probe from being read as real-speech speed — comes from the
        // type that measured it, not from a second copy of it here.
        starling_runtime_host::engine::SpeedCheckStatus::Reported { results, .. } => {
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
                .gap(px(12.))
                .p(px(12.))
                .rounded(px(6.))
                .border_1()
                .border_color(if selected {
                    theme::SHEET_BEAK
                } else {
                    theme::SHEET_LINE
                })
                .cursor_pointer()
                .hover(|style| style.bg(theme::SHEET_HOVER))
                .on_click(cx.listener(move |this, _, _window, cx| {
                    this.pick_draft_mode(&id, cx);
                }))
                .child(
                    kit::radio(selected),
                )
                .child(
                    div()
                        .flex()
                        .flex_col()
                        .flex_1()
                        .min_w_0()
                        .gap(px(3.))
                        .child(
                            div()
                                .text_size(px(theme::TEXT_MD))
                                .font_weight(FontWeight::SEMIBOLD)
                                .child(mode.name.clone()),
                        )
                        .child(
                            div()
                                .text_size(px(theme::TEXT_SM))
                                .line_height(px(theme::TEXT_SM * 1.5))
                                .text_color(theme::SHEET_QUILL)
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
        .flex()
        .flex_col()
        .child(rows)
        .child(
            div()
                .id("processing-disclosure")
                .mt(px(12.))
                .bg(theme::SHEET_CALLOUT)
                .p(px(12.))
                .text_size(px(theme::TEXT_SM))
                .line_height(px(theme::TEXT_SM * 1.55))
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
                .line_height(px(theme::TEXT_SM * 1.5))
                .text_color(theme::SHEET_ROSE)
                .child(reason),
        );
    }

    div()
        .flex()
        .flex_col()
        .child(field)
        .child(rows)
        .child(div().mt(px(6.)).child(double_tap_row))
        .child(
            div()
                .id("shortcut-reach")
                .mt(px(12.))
                .bg(theme::SHEET_CALLOUT)
                .p(px(12.))
                .text_size(px(theme::TEXT_SM))
                .line_height(px(theme::TEXT_SM * 1.55))
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
        .rounded(px(6.))
        .border_1()
        .border_color(if lowering {
            theme::SHEET_BEAK
        } else {
            theme::SHEET_LINE
        })
        .when(!lowering, |row| row.opacity(0.5))
        .child(
            div()
                .flex()
                .flex_row()
                .justify_between()
                .text_size(px(theme::TEXT_MD))
                .font_weight(FontWeight::SEMIBOLD)
                .child("Lower playback to")
                .child(
                    div()
                        .font_weight(FontWeight::NORMAL)
                        .text_color(theme::SHEET_QUILL)
                        .child(format!("{level}%")),
                ),
        )
        .child(app.draft_lower_level.clone())
        .child(helper("A level above your current volume leaves it unchanged."));

    // Known only once a take has tried.
    let unsupported = app.playback.handle().unsupported_reason();
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
                .bg(theme::SHEET_CALLOUT)
                .p(px(12.))
                .text_size(px(theme::TEXT_SM))
                .line_height(px(theme::TEXT_SM * 1.55))
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
        .gap(px(12.))
        .p(px(12.))
        .rounded(px(6.))
        .border_1()
        .border_color(if selected {
            theme::SHEET_BEAK
        } else {
            theme::SHEET_LINE
        })
        .when(enabled, |row| {
            row.cursor_pointer()
                .hover(|style| style.bg(theme::SHEET_HOVER))
        })
        .child(
            kit::radio(selected),
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
                        .text_size(px(theme::TEXT_MD))
                        .font_weight(FontWeight::SEMIBOLD)
                        .child(name),
                )
                .child(
                    div()
                        .text_size(px(theme::TEXT_SM))
                        .line_height(px(theme::TEXT_SM * 1.5))
                        .text_color(theme::SHEET_QUILL)
                        .child(description),
                ),
        )
}

pub(super) fn helper(text: impl Into<SharedString>) -> Div {
    div()
        .mt(px(2.))
        .text_size(px(theme::TEXT_SM))
        .font_weight(FontWeight::NORMAL)
        .line_height(px(theme::TEXT_SM * 1.5))
        .text_color(theme::SHEET_QUILL)
        .child(text.into())
}

