//! The capture pane: headline copy, waveform recorder block, import button,
//! and the pinned error/recovery banner.

use std::time::Duration;

use gpui::{
    Animation, AnimationExt, Context, Div, ElementId, FontWeight, Stateful, Window, div,
    ease_in_out, point, prelude::*, px, rgba,
};

use crate::app::StarlingApp;
use crate::theme;
use crate::views::icon;

const BAR_COUNT: usize = 52;
const BAR_STRIDE: f32 = 7.0; // 2px bar + 5px gap
const WAVEFORM_WIDTH: f32 = BAR_STRIDE * BAR_COUNT as f32 - 5.0;

/// Approximation of the CSS `mask-image` edge fade: linear ramp over the
/// first and last 18% of the waveform width.
fn edge_fade(index: usize) -> f32 {
    let center = index as f32 * BAR_STRIDE + 1.0;
    let fade_zone = WAVEFORM_WIDTH * 0.18;
    let left = (center / fade_zone).min(1.0);
    let right = ((WAVEFORM_WIDTH - center) / fade_zone).min(1.0);
    left.min(right)
}

pub fn render_capture(
    app: &mut StarlingApp,
    window: &mut Window,
    cx: &mut Context<StarlingApp>,
) -> impl IntoElement {
    let recording = app.recorder.is_some();
    let has_transcript = app.selected().is_some();
    let viewport = window.viewport_size();
    let pad_top = if has_transcript {
        px(18.)
    } else {
        theme::clamp_px(viewport.height * 0.04, 25., 55.)
    };
    let pad_x = theme::clamp_px(viewport.width * 0.07, 46., 110.);
    let pad_bottom = if has_transcript { px(14.) } else { px(24.) };
    let h1_size = if has_transcript {
        theme::clamp_px(viewport.width * 0.042, 40., 60.)
    } else {
        theme::clamp_px(viewport.width * 0.052, 45., 76.)
    };

    div()
        .id("capture-pane")
        .overflow_y_scroll()
        .relative()
        .flex()
        .flex_col()
        .items_center()
        .flex_1()
        .min_h_0()
        .min_w(px(560.))
        .pt(pad_top)
        .px(pad_x)
        .pb(pad_bottom)
        .child(
            div()
                .w_full()
                .max_w(px(570.))
                .flex()
                .flex_col()
                .child(
                    div()
                        .mb(px(13.))
                        .font(theme::mono_font())
                        .text_size(px(10.))
                        .font_weight(FontWeight::MEDIUM)
                        .text_color(theme::EYEBROW)
                        .child("LOCAL DICTATION"),
                )
                .child(
                    div()
                        .max_w(px(640.))
                        .font(theme::serif_font())
                        .font_weight(FontWeight::MEDIUM)
                        .text_size(h1_size)
                        .line_height(h1_size * 0.96)
                        .text_color(theme::INK)
                        .child(if recording {
                            "Listening closely."
                        } else {
                            "Say it as you mean it."
                        }),
                )
                .child(
                    div()
                        .max_w(px(480.))
                        .mt(px(8.))
                        .mb(if has_transcript { px(4.) } else { px(0.) })
                        .text_size(px(13.))
                        .line_height(px(13. * 1.65))
                        .text_color(theme::MUTED)
                        .child(
                            "Your recording is saved locally, sent only to your selected server, \
                             and shown exactly as the model returned it.",
                        ),
                ),
        )
        .child(render_recorder(app, cx, recording, has_transcript))
        .child(render_import_button(cx))
        .children(render_banner(app, cx))
}

fn render_recorder(
    app: &mut StarlingApp,
    cx: &mut Context<StarlingApp>,
    recording: bool,
    has_transcript: bool,
) -> Div {
    let button_size = if has_transcript { 90. } else { 116. };

    let recorder = if has_transcript {
        div().h(px(150.)).flex_none()
    } else {
        div().flex_1().min_h(px(255.))
    };

    recorder
        .relative()
        .w_full()
        .max_w(px(620.))
        .flex()
        .flex_col()
        .items_center()
        .justify_center()
        .child(
            div()
                .absolute()
                .top(px(0.))
                .bottom(px(0.))
                .left(px(0.))
                .right(px(0.))
                .flex()
                .items_center()
                .justify_center()
                .opacity(if recording { 0.84 } else { 0.45 })
                .child(
                    div()
                        .id("waveform")
                        .flex()
                        .flex_row()
                        .items_center()
                        .gap(px(5.))
                        .h(px(120.))
                        .w(gpui::relative(0.9))
                        .max_w(px(620.))
                        .children(
                            app.levels
                                .iter()
                                .enumerate()
                                .map(|(index, level)| {
                                    let height = (level * 106.0).max(5.0);
                                    div()
                                        .w(px(2.))
                                        .h(px(height))
                                        .rounded(px(99.))
                                        .bg(if recording {
                                            theme::LIME
                                        } else {
                                            theme::WAVE_IDLE
                                        })
                                        .opacity(edge_fade(index))
                                })
                                .collect::<Vec<_>>(),
                        ),
                ),
        )
        .child(
            div()
                .relative()
                .flex_none()
                .when(recording, |wrapper| {
                    wrapper.child(
                        div()
                            .absolute()
                            .top(px(-24.))
                            .left(px(-24.))
                            .size(px(button_size + 48.))
                            .rounded_full()
                            .bg(rgba(0xFF745B0F))
                            .with_animation(
                                ElementId::Name("record-pulse".into()),
                                Animation::new(Duration::from_millis(2000))
                                    .repeat()
                                    .with_easing(ease_in_out),
                                |el, delta| el.opacity(0.15 + 0.85 * delta),
                            ),
                    )
                })
                .child(
                    div()
                        .id("record-button")
                        .size(px(button_size))
                        .rounded_full()
                        .border_1()
                        .border_color(theme::RECORD_BORDER)
                        .p(px(9.))
                        .flex()
                        .items_center()
                        .justify_center()
                        .flex_none()
                        .bg(theme::RECORD_BG)
                        .shadow(vec![gpui::BoxShadow {
                            color: rgba(0x0000008C).into(),
                            offset: point(px(0.), px(16.)),
                            blur_radius: px(70.),
                            spread_radius: px(0.),
                        }])
                        .cursor_pointer()
                        .hover(|style| {
                            style.shadow(vec![
                                gpui::BoxShadow {
                                    color: rgba(0x000000A6).into(),
                                    offset: point(px(0.), px(18.)),
                                    blur_radius: px(75.),
                                    spread_radius: px(0.),
                                },
                                gpui::BoxShadow {
                                    color: rgba(0xD9FF6A07).into(),
                                    offset: point(px(0.), px(0.)),
                                    blur_radius: px(0.),
                                    spread_radius: px(9.),
                                },
                            ])
                        })
                        .on_click(cx.listener(|this, _, _window, cx| {
                            this.toggle_recording(cx);
                        }))
                        .child(
                            div()
                                .size_full()
                                .rounded_full()
                                .flex()
                                .items_center()
                                .justify_center()
                                .bg(if recording { theme::CORAL } else { theme::LIME })
                                .child(if recording {
                                    div()
                                        .size(px(24.))
                                        .rounded(px(5.))
                                        .bg(theme::PAPER_LIGHT)
                                        .into_any_element()
                                } else {
                                    icon("icons/mic.svg", 34., theme::MIC_FG).into_any_element()
                                }),
                        ),
                ),
        )
        .child(
            div()
                .mt(px(16.))
                .flex()
                .flex_col()
                .items_center()
                .gap(px(8.))
                .text_size(px(11.))
                .text_color(theme::MUTED)
                .child(div().child(if recording {
                    theme::fmt_duration(Some(app.elapsed_ms))
                } else if app.busy() {
                    "Transcribing…".to_string()
                } else {
                    "Tap to record".to_string()
                }))
                .when(!has_transcript, |meta| {
                    meta.child(
                        div()
                            .px(px(7.))
                            .py(px(4.))
                            .rounded(px(4.))
                            .border_1()
                            .border_color(theme::LINE)
                            .font(theme::mono_font())
                            .text_size(px(9.))
                            .text_color(theme::DIM)
                            .child(if cfg!(target_os = "macos") {
                                "⌘ Shift Space"
                            } else {
                                "Ctrl Shift Space"
                            }),
                    )
                }),
        )
}

fn render_import_button(cx: &mut Context<StarlingApp>) -> impl IntoElement {
    div()
        .id("import-audio")
        .flex()
        .flex_row()
        .items_center()
        .gap(px(8.))
        .px(px(1.))
        .py(px(6.))
        .mb(px(2.))
        .border_b_1()
        .border_color(theme::IMPORT_LINE)
        .text_size(px(11.))
        .text_color(theme::MUTED)
        .cursor_pointer()
        .hover(|style| style.text_color(theme::INK).border_color(theme::LIME))
        .on_click(cx.listener(|this, _, _window, cx| {
            this.import_audio(cx);
        }))
        .child(icon("icons/file-audio.svg", 17., theme::MUTED))
        .child("Import an audio file")
        .child(
            div()
                .font(theme::mono_font())
                .text_size(px(8.))
                .text_color(theme::DIM)
                .child(".wav"),
        )
}

fn render_banner(app: &mut StarlingApp, cx: &mut Context<StarlingApp>) -> Option<impl IntoElement> {
    if app.error.is_none() && app.unsaved.is_empty() {
        let message = app.capture_warning.clone()?;
        return Some(quality_banner(message, cx));
    }
    let error = app.error.clone();
    let unsaved_count = app.unsaved.len();

    let mut banner = div()
        .id("error-banner")
        .absolute()
        .left(px(28.))
        .right(px(28.))
        .bottom(px(28.))
        .flex()
        .flex_row()
        .items_start()
        .gap(px(12.))
        .p(px(14.))
        .bg(theme::ERROR_BG)
        .border_1()
        .border_color(theme::ERROR_LINE)
        .rounded(px(7.))
        .text_size(px(11.))
        .text_color(theme::ERROR_TEXT)
        .child(icon("icons/alert-circle.svg", 18., theme::ERROR_TEXT))
        .child(
            div()
                .flex()
                .flex_col()
                .gap(px(3.))
                .flex_1()
                .child(
                    div().text_color(theme::ERROR_TITLE).child(if error.is_some() {
                        "Action failed"
                    } else {
                        "Recording not saved"
                    }),
                )
                .children(error.clone().map(|message| div().child(message)))
                .child(div().text_color(theme::ERROR_SUBTLE).child(if unsaved_count > 0 {
                    format!(
                        "{unsaved_count} recording{} could not be saved. Download {} before you close Starling.",
                        if unsaved_count == 1 { "" } else { "s" },
                        if unsaved_count == 1 { "it" } else { "them" },
                    )
                } else {
                    "Starling keeps audio in history after a successful local save.".to_string()
                })),
        );

    if unsaved_count > 0 {
        let mut actions = div().flex().flex_wrap().items_center().gap(px(6.));
        for (index, capture) in app.unsaved.iter().enumerate() {
            let id = capture.id.clone();
            actions = actions.child(
                div()
                    .id(gpui::SharedString::from(format!("recover-{index}")))
                    .px(px(9.))
                    .py(px(7.))
                    .rounded(px(4.))
                    .border_1()
                    .border_color(theme::RECOVERY_LINE)
                    .text_size(px(9.))
                    .text_color(theme::RECOVERY_TEXT)
                    .cursor_pointer()
                    .on_click(cx.listener(move |this, _, _window, cx| {
                        this.export_unsaved_audio(&id, cx);
                    }))
                    .child(format!("Download WAV {}", index + 1)),
            );
        }
        let confirm = app.confirm_discard;
        actions = actions.child(
            div()
                .id("discard-unsaved")
                .px(px(9.))
                .py(px(7.))
                .rounded(px(4.))
                .border_1()
                .border_color(theme::RECOVERY_LINE)
                .text_size(px(9.))
                .text_color(theme::RECOVERY_TEXT)
                .cursor_pointer()
                .on_click(cx.listener(|this, _, _window, cx| {
                    this.discard_unsaved(cx);
                }))
                .child(if confirm {
                    "Confirm discard"
                } else {
                    "Discard unsaved"
                }),
        );
        banner = banner.child(actions);
    } else {
        banner = banner.child(
            div()
                .id("dismiss-error")
                .cursor_pointer()
                .on_click(cx.listener(|this, _, _window, cx| {
                    this.error = None;
                    cx.notify();
                }))
                .child(icon("icons/x.svg", 16., theme::ERROR_TEXT)),
        );
    }

    Some(banner)
}

/// A non-fatal notice (shown instead of the error banner when nothing failed):
/// e.g. a take that was saved and sent but arrived heavily clipped.
fn quality_banner(message: String, cx: &mut Context<StarlingApp>) -> Stateful<Div> {
    div()
        .id("quality-banner")
        .absolute()
        .left(px(28.))
        .right(px(28.))
        .bottom(px(28.))
        .flex()
        .flex_row()
        .items_start()
        .gap(px(12.))
        .p(px(14.))
        .bg(theme::ERROR_BG)
        .border_1()
        .border_color(theme::ERROR_LINE)
        .rounded(px(7.))
        .text_size(px(11.))
        .text_color(theme::ERROR_TEXT)
        .child(icon("icons/alert-circle.svg", 18., theme::ERROR_TEXT))
        .child(
            div()
                .flex()
                .flex_col()
                .gap(px(3.))
                .flex_1()
                .child(
                    div()
                        .text_color(theme::ERROR_TITLE)
                        .child("Recording clipped"),
                )
                .child(message)
                .child(div().text_color(theme::ERROR_SUBTLE).child(
                    "The take was still saved and sent; heavily clipped audio transcribes poorly.",
                )),
        )
        .child(
            div()
                .id("dismiss-quality")
                .cursor_pointer()
                .on_click(cx.listener(|this, _, _window, cx| {
                    this.capture_warning = None;
                    cx.notify();
                }))
                .child(icon("icons/x.svg", 16., theme::ERROR_TEXT)),
        )
}
