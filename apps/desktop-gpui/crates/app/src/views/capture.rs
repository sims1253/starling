//! The stage: the state sentence, the level field around the hex record
//! control, the import action, and the pinned error/recovery banner.

use std::time::Duration;

use gpui::{
    Animation, AnimationExt, Context, Div, ElementId, FontWeight, Stateful, Window, div,
    ease_in_out, prelude::*, px,
};
use starling_dictation::engine::{EnginePhase, InstallState, SwitchStage};
use starling_dictation::playback::NoticeKind;
use starling_dictation::settings::EngineMode;

use crate::app::StarlingApp;
use crate::theme;
use crate::views::kit::{self, Tone};
use crate::views::{icon, spinner};

/// Whether the first-run model card shows (#363): builtin mode, no model
/// serving yet, and the engine itself usable — the states where picking
/// a model is THE next step. A failed engine (or a manual server) never
/// offers a download as the way forward; recording stays possible in
/// every one of these states either way. Pure so the policy is testable.
pub(crate) fn first_run_card_visible(phase: &EnginePhase, has_active: bool) -> bool {
    !has_active
        && matches!(
            phase,
            EnginePhase::NoModel | EnginePhase::Starting | EnginePhase::Loading | EnginePhase::Warming
        )
}

/// The first-run card's progress line (#363, #366): `None` (the button row
/// offers the download) while nothing is in flight; a stage line while the
/// recommended model downloads or verifies — including a hand-placed file
/// being verified by an activation switch — and while the engine brings it
/// up. Pure so the policy is testable.
pub(crate) fn first_run_progress(
    phase: &EnginePhase,
    install: &InstallState,
    switch: Option<&SwitchStage>,
) -> Option<String> {
    match install {
        InstallState::Downloading { done, total } => Some(format!(
            "Downloading {}% — you can record meanwhile; transcription starts once the model \
             is ready.",
            crate::views::settings::download_percent(*done, *total),
        )),
        // A finished download being verified is still installing: the
        // button must not re-offer "Download and use" over it.
        InstallState::Verifying => Some("Verifying the download…".to_string()),
        // A NeedsVerification file only verifies through an activation
        // switch; while that switch is verifying, the same rule holds.
        InstallState::NeedsVerification
            if matches!(switch, Some(SwitchStage::Verifying)) =>
        {
            Some("Verifying the download…".to_string())
        }
        _ => match phase {
            EnginePhase::Starting => Some("Starting the engine…".to_string()),
            EnginePhase::Loading => Some("Loading the model…".to_string()),
            EnginePhase::Warming => Some("Warming up…".to_string()),
            _ => None,
        },
    }
}

/// The level field: one column of specks per level, like the pale spots
/// on a starling's coat. A quiet room is a dotted line; speech raises the
/// columns symmetrically around it.
const COLUMN_COUNT: usize = 52;
const SPECK_SIZE: f32 = 3.0;
const SPECK_GAP: f32 = 4.0;
const COLUMN_GAP: f32 = 6.0;
const COLUMN_STRIDE: f32 = SPECK_SIZE + COLUMN_GAP;
const FIELD_WIDTH: f32 = COLUMN_STRIDE * COLUMN_COUNT as f32 - COLUMN_GAP;
/// Specks either side of the centre line at full level.
const MAX_REACH: f32 = 7.0;

/// The field's edges thin out over the first and last 18% of its width,
/// so it reads as a sound rising out of silence rather than a box.
fn edge_fade(index: usize) -> f32 {
    let center = index as f32 * COLUMN_STRIDE + SPECK_SIZE / 2.0;
    let fade_zone = FIELD_WIDTH * 0.18;
    let left = (center / fade_zone).min(1.0);
    let right = ((FIELD_WIDTH - center) / fade_zone).min(1.0);
    left.min(right)
}

/// How many specks a column shows either side of its centre speck.
pub(crate) fn speck_reach(level: f32) -> usize {
    (level.clamp(0.0, 1.0) * MAX_REACH).round() as usize
}

pub fn render_capture(
    app: &mut StarlingApp,
    window: &mut Window,
    cx: &mut Context<StarlingApp>,
) -> impl IntoElement {
    let recording = app.recorder.is_some();
    // What the record button shows, captured for its click handler: a
    // click only acts while the state it targeted still holds.
    let record_button = if recording {
        crate::activation::RecordButton::Stop
    } else {
        crate::activation::RecordButton::Start
    };
    // The staging panel needs the room a transcript would: the recorder
    // and headline go compact either way.
    let has_transcript = app.selected().is_some() || app.staging.is_some();
    let staging_panel = crate::views::staging::render_staging_panel(app, window, cx);
    let viewport = window.viewport_size();
    let pad_top = if has_transcript {
        px(20.)
    } else {
        theme::clamp_px(viewport.height * 0.05, 28., 56.)
    };
    let pad_x = theme::clamp_px(viewport.width * 0.045, 36., 72.);
    let h1_size = if has_transcript {
        theme::clamp_px(viewport.width * 0.03, 32., 40.)
    } else {
        theme::clamp_px(viewport.width * 0.045, 42., 64.)
    };

    div()
        .id("capture-pane")
        .overflow_y_scroll()
        .relative()
        .flex()
        .flex_col()
        .items_start()
        .flex_1()
        .min_h_0()
        .min_w(px(520.))
        .pt(pad_top)
        .px(pad_x)
        .pb(px(20.))
        .child(
            div()
                .w_full()
                .flex()
                .flex_col()
                .gap(px(10.))
                .child(
                    div()
                        .max_w(px(680.))
                        .font(theme::display_font())
                        .text_size(h1_size)
                        .line_height(h1_size * 1.04)
                        .text_color(theme::SPECK)
                        // Never "listening" to a silent or dead input.
                        .child(
                            app.live_input_headline()
                                .filter(|_| {
                                    app.activation.readiness()
                                        == Some(crate::activation::Readiness::Listening)
                                })
                                .unwrap_or_else(|| {
                                    crate::activation::readiness_headline(
                                        app.activation.readiness().filter(|_| recording),
                                    )
                                }),
                        ),
                )
                .when(!has_transcript, |head| {
                    head.child(
                        div()
                            .max_w(px(470.))
                            .text_size(px(theme::TEXT_MD))
                            .line_height(px(theme::TEXT_MD * 1.6))
                            .text_color(theme::QUILL)
                            // #220: takes record in Starling's recording
                            // service; while it is not reachable, say so.
                            .child(
                                app.host_unavailable()
                                    .filter(|_| !recording)
                                    .unwrap_or_else(|| {
                                        "Your recording is saved on this computer, sent only to \
                                         the engine you chose, and shown exactly as the model \
                                         returned it."
                                            .to_string()
                                    }),
                            ),
                    )
                }),
        )
        .child(render_recorder(app, cx, record_button, has_transcript))
        .children(render_engine_card(app, cx))
        .children(staging_panel)
        .when(recording && app.staging.is_none() && !app.live_partial.is_empty(), |pane| {
            pane.child(
                div()
                    .max_w(px(640.))
                    .mb(px(12.))
                    .font(theme::reading_font())
                    .text_size(px(theme::TEXT_XL))
                    .line_height(px(theme::TEXT_XL * 1.5))
                    .text_color(theme::SPECK)
                    .child(app.live_partial.clone()),
            )
        })
        // Live text going quiet mid-take is explained while it is quiet,
        // not only beside the saved take.
        .when_some(app.stream_degradation.clone().filter(|_| recording), |pane, reason| {
            pane.child(
                div()
                    .max_w(px(640.))
                    .mb(px(12.))
                    .text_size(px(theme::TEXT_SM))
                    .text_color(theme::QUILL)
                    .child(reason),
            )
        })
        .child(render_import_button(cx))
        .children(render_banner(app, cx))
        // The error banner holds the bottom: words that did not reach
        // their window (#221) wait at the top instead of behind it.
        .children(if app.error.is_some() || !app.unsaved.is_empty() {
            crate::views::delivery::render_recovery(app, true, cx)
        } else {
            None
        })
}

/// The specks of one column, centre out: `reach` either side of a centre
/// speck, each a little fainter the further it sits from the centre line.
fn speck_column(index: usize, level: f32, recording: bool) -> Div {
    let reach = speck_reach(level);
    let fade = edge_fade(index);
    let color = if recording { theme::SPECK } else { theme::QUILL };
    let base = if recording { 1.0 } else { 0.55 };
    div()
        .flex()
        .flex_col()
        .items_center()
        .gap(px(SPECK_GAP))
        .children((0..reach * 2 + 1).map(move |row| {
            let distance = (row as f32 - reach as f32).abs();
            let falloff = 1.0 - 0.6 * distance / (MAX_REACH + 1.0);
            div()
                .size(px(SPECK_SIZE))
                .rounded_full()
                .bg(color)
                .opacity(base * fade * falloff)
        }))
}

fn render_recorder(
    app: &mut StarlingApp,
    cx: &mut Context<StarlingApp>,
    button: crate::activation::RecordButton,
    has_transcript: bool,
) -> Div {
    let hex_size = if has_transcript { 76. } else { 104. };
    let recording = button == crate::activation::RecordButton::Stop;

    // Compact, the recorder is as tall as what it holds; otherwise it
    // takes the stage's free height and centres in it.
    let recorder = if has_transcript {
        div().flex_none().pt(px(20.)).pb(px(12.))
    } else {
        div().flex_1().min_h(px(270.))
    };
    let ring = 14.;

    recorder
        .w_full()
        .flex()
        .flex_col()
        .items_center()
        .justify_center()
        .child(
            // The hex and its ring, with the level field running through
            // them at the hex's centre line.
            div()
                .relative()
                .w_full()
                .h(px(hex_size + ring * 2.))
                .flex()
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
                        .child(
                            div()
                                .id("level-field")
                                .flex()
                                .flex_row()
                                .items_center()
                                .gap(px(COLUMN_GAP))
                                .children(
                                    app.levels
                                        .iter()
                                        .enumerate()
                                        .map(|(index, level)| {
                                            speck_column(index, *level, recording)
                                        })
                                        .collect::<Vec<_>>(),
                                ),
                        ),
                )
                .child(
                    div()
                        .id("record-button")
                        .relative()
                        .flex_none()
                        .size(px(hex_size))
                        .flex()
                        .items_center()
                        .justify_center()
                        .cursor_pointer()
                        .on_click(cx.listener(move |this, _, _window, cx| {
                            this.toggle_recording(button, cx);
                        }))
                        // A hex ring breathes around the control while it
                        // records.
                        .when(recording, |hex| {
                            hex.child(
                                div()
                                    .absolute()
                                    .top(px(-ring))
                                    .left(px(-ring))
                                    .child(icon(
                                        "icons/hex-outline.svg",
                                        hex_size + ring * 2.,
                                        theme::ROSE,
                                    ))
                                    .with_animation(
                                        ElementId::Name("record-pulse".into()),
                                        Animation::new(Duration::from_millis(1800))
                                            .repeat()
                                            .with_easing(ease_in_out),
                                        |el, delta| el.opacity(0.15 + 0.55 * delta),
                                    ),
                            )
                        })
                        .child(
                            div()
                                .absolute()
                                .top(px(0.))
                                .left(px(0.))
                                .child(icon(
                                    "icons/hex.svg",
                                    hex_size,
                                    if recording { theme::ROSE } else { theme::BEAK },
                                )),
                        )
                        .child(if recording {
                            div()
                                .size(px(hex_size * 0.22))
                                .rounded(px(4.))
                                .bg(theme::SPECK)
                                .into_any_element()
                        } else {
                            icon("icons/mic.svg", hex_size * 0.32, theme::ON_BEAK)
                                .into_any_element()
                        }),
                ),
        )
        .child(
            div()
                .mt(px(4.))
                .flex()
                .flex_col()
                .items_center()
                .gap(px(8.))
                .text_size(px(theme::TEXT_SM))
                .text_color(theme::QUILL)
                .child(if recording {
                    div()
                        .font(theme::mono_font())
                        .text_size(px(theme::TEXT_LG))
                        .text_color(theme::SPECK)
                        .child(theme::fmt_duration(Some(app.elapsed_ms)))
                } else if app.busy() {
                    div().child("Transcribing…")
                } else {
                    div().child("Click to record, or press")
                })
                // #221: how this take ends (held, latched, hands-free).
                .children(
                    crate::activation::finish_hint(
                        app.activation.latch().filter(|_| recording),
                        &app.shortcut.label(),
                    )
                    .map(|hint| div().id("finish-hint").child(hint)),
                )
                .children(crate::views::microphone::live_input_line(app))
                .when(!recording && !app.busy(), |meta| {
                    meta.child(kit::keycaps(&app.shortcut.label(), false))
                }),
        )
}

/// The first-run model card (#363): builtin mode with no model serving
/// offers the recommended catalog entry — its label, size, and one
/// "Download and use" button — plus "More models" for the full list in
/// Settings. While the recommended model downloads (or the engine loads
/// it) the card shows the progress stage instead of the button row going
/// silent. Recording is never blocked by this card.
fn render_engine_card(
    app: &mut StarlingApp,
    cx: &mut Context<StarlingApp>,
) -> Option<impl IntoElement> {
    if app.engine_settings.mode != EngineMode::Builtin {
        return None;
    }
    let snapshot = app.engine_snapshot()?;
    if !first_run_card_visible(&snapshot.phase, snapshot.active.is_some()) {
        return None;
    }
    let recommended = snapshot.models.iter().find(|model| model.recommended)?;
    let id = recommended.id.clone();

    // While downloading, verifying, or loading, the card shows the progress
    // stage instead of the download button.
    let progress = first_run_progress(
        &snapshot.phase,
        &recommended.install,
        snapshot.switch.as_ref().map(|switch| &switch.stage),
    );
    let installing = progress.is_some();

    Some(
        div()
            .id("engine-first-run")
            .w_full()
            .max_w(px(600.))
            .flex()
            .flex_col()
            .gap(px(8.))
            .mb(px(18.))
            .p(px(18.))
            .bg(theme::ROOST)
            .border_1()
            .border_color(theme::LINE)
            .rounded(px(8.))
            .child(
                div()
                    .text_size(px(theme::TEXT_LG))
                    .font_weight(FontWeight::SEMIBOLD)
                    .text_color(theme::SPECK)
                    .child("Choose a speech model to start dictating offline"),
            )
            .child(
                div()
                    .text_size(px(theme::TEXT_SM))
                    .line_height(px(theme::TEXT_SM * 1.55))
                    .text_color(theme::QUILL)
                    .child(format!(
                        "{}, {}. {}",
                        recommended.label,
                        crate::views::settings::fmt_size(recommended.size_bytes),
                        recommended.note
                    )),
            )
            .children(progress.map(|line| {
                div()
                    .flex()
                    .flex_row()
                    .items_center()
                    .gap(px(8.))
                    .text_size(px(theme::TEXT_SM))
                    .text_color(theme::QUILL)
                    .child(spinner("engine-first-run-spinner", 14., theme::QUILL))
                    .child(line)
            }))
            .child(
                div()
                    .mt(px(4.))
                    .flex()
                    .flex_row()
                    .items_center()
                    .gap(px(8.))
                    .when(!installing, |row| {
                        row.child(
                            kit::button("engine-first-run-download", Tone::Primary, false)
                                .on_click(cx.listener(move |this, _, _window, cx| {
                                    this.engine_activate(&id, cx);
                                }))
                                .child("Download and use"),
                        )
                    })
                    .child(
                        kit::button("engine-first-run-more", Tone::Quiet, false)
                            .on_click(cx.listener(|this, _, _window, cx| {
                                this.open_settings_at(crate::views::SettingsPage::Engine, cx);
                            }))
                            .child("More models"),
                    ),
            ),
    )
}

fn render_import_button(cx: &mut Context<StarlingApp>) -> impl IntoElement {
    div()
        .id("import-audio")
        .flex_none()
        .flex()
        .flex_row()
        .items_center()
        .gap(px(8.))
        .py(px(6.))
        .text_size(px(theme::TEXT_SM))
        .text_color(theme::QUILL)
        .cursor_pointer()
        .hover(|style| style.text_color(theme::SPECK))
        .on_click(cx.listener(|this, _, _window, cx| {
            this.import_audio(cx);
        }))
        .child(icon("icons/file-audio.svg", 16., theme::QUILL))
        .child("Import a WAV file")
}

/// What the pinned error/recovery banner shows (R08): the footer line and
/// whether the X that clears `app.error` is offered. Pure so the policy is
/// testable without building elements.
pub(crate) struct ErrorBannerSurface {
    /// The unsaved-count line under the message. `None` when there are no
    /// unsaved recordings — an error-only banner shows no footer at all,
    /// never the success-ish "Starling keeps audio in history…" line that
    /// used to ride it.
    pub footer: Option<String>,
    /// Whether the dismiss X is rendered. True whenever there is an error
    /// to clear — including alongside unsaved recordings, which used to
    /// leave an error with no dismiss affordance at all. False for an
    /// unsaved-only banner, whose exits are download and discard, not
    /// dismiss.
    pub dismissible: bool,
}

/// Decide the error/recovery banner's surface (R08).
pub(crate) fn error_banner_surface(has_error: bool, unsaved_count: usize) -> ErrorBannerSurface {
    ErrorBannerSurface {
        footer: (unsaved_count > 0).then(|| unsaved_footer_line(unsaved_count)),
        dismissible: has_error,
    }
}

fn unsaved_footer_line(count: usize) -> String {
    format!(
        "{count} recording{} could not be saved. Download {} before you close Starling.",
        if count == 1 { "" } else { "s" },
        if count == 1 { "it" } else { "them" },
    )
}

fn render_banner(app: &mut StarlingApp, cx: &mut Context<StarlingApp>) -> Option<impl IntoElement> {
    if app.error.is_none() && app.unsaved.is_empty() {
        // Words that did not reach their window (#221) outrank every
        // other notice.
        if let Some(recovery) = crate::views::delivery::render_recovery(app, false, cx) {
            return Some(recovery);
        }
        // Two independent ephemeral notices share this slot; the capture
        // warning outranks a one-off export rename (banner_notice in
        // `app.rs` is the tested form of that priority).
        if let Some(message) = app.capture_warning.clone() {
            return Some(quality_banner(
                "Recording clipped",
                message,
                "The take was still saved and sent; heavily clipped audio transcribes poorly.",
                |app: &mut StarlingApp| app.capture_warning = None,
                cx,
            ));
        }
        if let Some(notice) = app.playback_notice.clone() {
            let title = match notice.kind {
                NoticeKind::AdjustFailed => "Playback not adjusted",
                NoticeKind::RestoreFailed => "Playback not restored",
                NoticeKind::OutputRemoved => "Playback device removed",
            };
            return Some(quality_banner(
                title,
                notice.message,
                "The recording itself is unaffected.",
                |app: &mut StarlingApp| app.playback_notice = None,
                cx,
            ));
        }
        if let Some(message) = app.recovery_notice.clone() {
            return Some(quality_banner(
                "Recordings recovered",
                message,
                "Audio is kept up to the last point the capture journal confirmed on disk.",
                |app: &mut StarlingApp| app.recovery_notice = None,
                cx,
            ));
        }
        if let Some(message) = app.service_notice.clone() {
            return Some(quality_banner(
                "Recording kept",
                message,
                "Starling's recording service keeps a take going while this window is closed \
                 or restarting.",
                |app: &mut StarlingApp| app.service_notice = None,
                cx,
            ));
        }
        if let Some(message) = app.take_notice.clone() {
            return Some(quality_banner(
                "Take cancelled",
                message,
                "No transcript was kept and nothing was inserted anywhere.",
                |app: &mut StarlingApp| app.take_notice = None,
                cx,
            ));
        }
        if let Some(message) = app.export_notice.clone() {
            return Some(quality_banner(
                "Export renamed",
                message,
                "Nothing was overwritten; both files are in your downloads directory.",
                |app: &mut StarlingApp| app.export_notice = None,
                cx,
            ));
        }
        return None;
    }
    let error = app.error.clone();
    let unsaved_count = app.unsaved.len();
    // R08: the footer and dismiss affordance come from one tested decision
    // — an error state never shows the reassurance footer, and there is
    // always a dismiss path for the error itself.
    let surface = error_banner_surface(error.is_some(), unsaved_count);

    let mut banner = notice_box("error-banner")
        .bottom(px(20.))
        .child(icon("icons/alert-circle.svg", 18., theme::NOTICE_TEXT))
        .child(
            div()
                .flex()
                .flex_col()
                .gap(px(3.))
                .flex_1()
                .min_w_0()
                .child(
                    div()
                        .font_weight(FontWeight::SEMIBOLD)
                        .text_color(theme::NOTICE_TITLE)
                        .child(if error.is_some() {
                            "Action failed"
                        } else {
                            "Recording not saved"
                        }),
                )
                .children(error.clone().map(|message| div().child(message)))
                .children(crate::views::microphone::input_problem_actions(app, cx))
                .children(
                    surface
                        .footer
                        .map(|line| div().text_color(theme::NOTICE_SUBTLE).child(line)),
                ),
        );

    if unsaved_count > 0 {
        let mut actions = div().flex().flex_wrap().items_center().gap(px(6.));
        for (index, capture) in app.unsaved.iter().enumerate() {
            let id = capture.id.clone();
            actions = actions.child(
                notice_action(gpui::SharedString::from(format!("recover-{index}")))
                    .on_click(cx.listener(move |this, _, _window, cx| {
                        this.export_unsaved_audio(&id, cx);
                    }))
                    .child(format!("Download WAV {}", index + 1)),
            );
        }
        // #214.4: the arm only counts while it was armed for exactly the
        // batch now in the banner — a take stashed since disarms it.
        let confirm = app.discard_armed();
        actions = actions.child(
            notice_action("discard-unsaved")
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
    }
    if surface.dismissible {
        // R08: an error alongside unsaved recordings used to have no
        // dismiss at all; the X now always clears the error, while the
        // unsaved list keeps its own download/discard exits.
        banner = banner.child(
            div()
                .id("dismiss-error")
                .cursor_pointer()
                .on_click(cx.listener(|this, _, _window, cx| {
                    this.error = None;
                    cx.notify();
                }))
                .child(icon("icons/x.svg", 16., theme::NOTICE_TEXT)),
        );
    }

    Some(banner)
}

/// A non-fatal notice (shown instead of the error banner when nothing
/// failed): e.g. a take that was saved and sent but arrived heavily
/// clipped, or an export that had to land on a `-N` name. `clear` dismisses
/// whichever app field the notice came from.
fn quality_banner(
    title: &'static str,
    message: String,
    footer: &'static str,
    clear: fn(&mut StarlingApp),
    cx: &mut Context<StarlingApp>,
) -> Stateful<Div> {
    notice_box("quality-banner")
        .bottom(px(20.))
        .child(icon("icons/alert-circle.svg", 18., theme::NOTICE_TEXT))
        .child(
            // The text column may shrink below its words' width (a flex
            // item's default minimum is its content), so long notices
            // wrap inside the box instead of running past its edge.
            div()
                .flex()
                .flex_col()
                .gap(px(3.))
                .flex_1()
                .min_w_0()
                .child(
                    div()
                        .font_weight(FontWeight::SEMIBOLD)
                        .text_color(theme::NOTICE_TITLE)
                        .child(title),
                )
                .child(div().w_full().child(message))
                .child(div().w_full().text_color(theme::NOTICE_SUBTLE).child(footer)),
        )
        .child(
            div()
                .id("dismiss-quality")
                .cursor_pointer()
                .on_click(cx.listener(move |this, _, _window, cx| {
                    clear(this);
                    cx.notify();
                }))
                .child(icon("icons/x.svg", 16., theme::NOTICE_TEXT)),
        )
}

/// The box every notice and error shares, floating across the stage;
/// the caller pins it to the stage's foot or head.
pub(crate) fn notice_box(id: &'static str) -> Stateful<Div> {
    div()
        .id(id)
        .absolute()
        .left(px(24.))
        .right(px(24.))
        .flex()
        .flex_row()
        .items_start()
        .gap(px(12.))
        .p(px(14.))
        .bg(theme::NOTICE_BG)
        .border_1()
        .border_color(theme::NOTICE_LINE)
        .rounded(px(8.))
        .text_size(px(theme::TEXT_SM))
        .line_height(px(theme::TEXT_SM * 1.5))
        .text_color(theme::NOTICE_TEXT)
}

pub(crate) fn notice_action(id: impl Into<gpui::ElementId>) -> Stateful<Div> {
    div()
        .id(id)
        .px(px(10.))
        .py(px(5.))
        .rounded(px(6.))
        .border_1()
        .border_color(theme::NOTICE_LINE)
        .text_size(px(theme::TEXT_XS))
        .text_color(theme::NOTICE_TITLE)
        .cursor_pointer()
        .hover(|style| style.bg(theme::HOVER))
}

#[cfg(test)]
mod tests {
    use super::speck_reach;

    #[test]
    fn a_quiet_room_is_one_speck_per_column_and_speech_reaches_out() {
        // The idle level the app starts with draws the dotted line.
        assert_eq!(speck_reach(0.06), 0);
        assert_eq!(speck_reach(0.5), 4);
        assert_eq!(speck_reach(1.0), 7);
        // A level past full never grows the column past the field.
        assert_eq!(speck_reach(3.0), 7);
        assert_eq!(speck_reach(-1.0), 0);
    }
}
