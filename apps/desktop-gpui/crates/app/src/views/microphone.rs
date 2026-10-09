//! Microphone views: the settings dialog's Microphone section —
//! picker, preferred vs. in-use input, the repeatable microphone and
//! shortcut check — and the capture pane's input line and recovery
//! actions.

use gpui::{div, prelude::*, px, relative, Context, Div, FontWeight, SharedString};
use starling_dictation::microphone::{InputProblem, InputRoute};

use crate::app::StarlingApp;
use crate::mic::{CheckOutcome, DeviceList, MicCheck, page_label, picker_rows, platform_pages};
use crate::theme;

fn button(id: impl Into<SharedString>, label: &'static str) -> gpui::Stateful<Div> {
    div()
        .id(id.into())
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
        .child(label)
}

fn note(text: impl Into<SharedString>) -> Div {
    div()
        .text_size(px(10.))
        .line_height(px(10. * 1.55))
        .text_color(theme::SETTINGS_HELPER)
        .child(text.into())
}

fn eyebrow(text: &'static str) -> Div {
    div()
        .mt(px(10.))
        .mb(px(13.))
        .font(theme::mono_font())
        .text_size(px(10.))
        .font_weight(FontWeight::MEDIUM)
        .text_color(theme::SETTINGS_EYEBROW)
        .child(text)
}

/// "In use" wording for a route: the device, and the fallback notice when
/// the take did not get what the user asked for.
pub(crate) fn route_line(route: &InputRoute) -> String {
    match route.notice() {
        Some(notice) => notice,
        // Not one microphone: the sound server picks (and may switch) the
        // physical source behind it — say so rather than name a device.
        None if route.follows_sound_server() => format!(
            "Recording from the system default input ({}) — PulseAudio/PipeWire picks the \
             microphone and may switch it if that device disconnects.",
            route.device
        ),
        None => format!("Recording from {}.", route.device),
    }
}

/// What the platform can and cannot promise about input routing.
fn platform_note() -> &'static str {
    if cfg!(target_os = "linux") {
        "On Linux, “default” is whatever PulseAudio/PipeWire selects; pick a named device to pin \
         one. The sound server may move a running “default” stream to another input when its \
         device disappears — Starling cannot see that move, so pin a device if it matters."
    } else if cfg!(target_os = "macos") {
        "Microphones are listed by name. Without microphone permission macOS records silence \
         instead of reporting an error, so when the test below hears only silence, check the \
         permission as well as the device."
    } else {
        "Microphones are listed by name. Windows privacy settings can block microphone access \
         for desktop apps; the test below reports an access error or silence when they do."
    }
}

pub(crate) fn render_microphone_section(
    app: &mut StarlingApp,
    cx: &mut Context<StarlingApp>,
) -> Div {
    let rows = picker_rows(&app.mic.devices, app.draft_microphone.as_deref());
    let mut list = div().flex().flex_col().gap(px(6.));
    for (index, row) in rows.into_iter().enumerate() {
        let device = row.device.clone();
        list = list.child(
            div()
                .id(SharedString::from(format!("mic-row-{index}")))
                .flex()
                .flex_row()
                .gap(px(10.))
                .p(px(10.))
                .rounded(px(3.))
                .border_1()
                .border_color(if row.selected {
                    theme::SETTINGS_INK
                } else {
                    theme::SETTINGS_LINE
                })
                .cursor_pointer()
                .hover(|style| style.bg(theme::PAPER_HOVER))
                .on_click(cx.listener(move |this, _, _window, cx| {
                    this.pick_draft_microphone(device.clone(), cx);
                }))
                .child(
                    div()
                        .mt(px(2.))
                        .size(px(10.))
                        .flex_none()
                        .rounded(px(5.))
                        .border_1()
                        .border_color(theme::SETTINGS_INK)
                        .when(row.selected, |dot| dot.bg(theme::SETTINGS_INK)),
                )
                .child(
                    div()
                        .flex()
                        .flex_col()
                        .gap(px(3.))
                        .min_w_0()
                        .child(
                            div()
                                .text_size(px(11.))
                                .font_weight(FontWeight::SEMIBOLD)
                                .child(row.label),
                        )
                        .children(row.detail.map(note)),
                ),
        );
    }

    let listing_line = match &app.mic.devices {
        DeviceList::Loading => Some(note("Looking for microphones…")),
        DeviceList::Failed(error) => Some(note(format!(
            "Could not list microphones ({error}). Your saved choice is kept; recordings use the \
             system default until listing works again."
        ))),
        DeviceList::Listed(devices) if devices.is_empty() => Some(note(
            "No microphones were found. Plug one in (or enable it in your sound settings), then \
             Refresh.",
        )),
        DeviceList::Listed(_) => None,
    };

    // Preferred vs. actually in use: the live take's route while one
    // records, otherwise the last take's.
    let live_route = app
        .recorder
        .as_ref()
        .and_then(|handle| handle.input_route());
    let in_use = match (live_route, app.mic.last_route.as_ref()) {
        (Some(route), _) => Some(format!("Now: {}", route_line(route))),
        (None, Some(route)) => Some(format!("Last recording: {}", route_line(route))),
        (None, None) => None,
    };
    let saved = match app.microphone_settings.preferred_device.as_deref() {
        Some(name) => format!("Saved choice: {name}."),
        None => "Saved choice: follow the system default.".to_string(),
    };
    let unsaved = app.draft_microphone != app.microphone_settings.preferred_device;

    div()
        .flex()
        .flex_col()
        .child(eyebrow("MICROPHONE"))
        .child(
            div()
                .mb(px(10.))
                .text_size(px(11.))
                .line_height(px(11. * 1.65))
                .text_color(theme::SETTINGS_MUTED)
                .child(
                    "The input Starling records from. Playback keeps using your system output, \
                     so headphones plus a laptop or USB microphone work as expected.",
                ),
        )
        .child(list)
        .child(
            div()
                .mt(px(8.))
                .flex()
                .flex_col()
                .gap(px(4.))
                .children(listing_line)
                .child(note(if unsaved {
                    format!("{saved} The selection above applies after Save.")
                } else {
                    saved
                }))
                .children(in_use.map(note))
                .child(note(platform_note())),
        )
        .child(div().mt(px(8.)).flex().flex_row().gap(px(6.)).child(
            button("mic-refresh", "Refresh").on_click(cx.listener(|this, _, _window, cx| {
                this.refresh_input_devices(cx);
            })),
        ))
        .child(render_check(app, cx))
}

fn meter(fill: f32) -> Div {
    div()
        .h(px(6.))
        .w_full()
        .rounded(px(3.))
        .bg(theme::SETTINGS_LINE)
        .child(
            div()
                .h_full()
                .w(relative(fill.clamp(0.0, 1.0)))
                .rounded(px(3.))
                .bg(theme::SETTINGS_INK),
        )
}

/// One button per OS settings page the problem offers.
fn settings_buttons(
    problem: &InputProblem,
    prefix: &'static str,
    cx: &mut Context<StarlingApp>,
) -> Vec<gpui::Stateful<Div>> {
    platform_pages(problem)
        .into_iter()
        .map(|page| {
            button(format!("{prefix}-{page:?}"), page_label(page)).on_click(
                cx.listener(move |this, _, _window, cx| this.open_input_settings(page, cx)),
            )
        })
        .collect()
}

fn problem_block(problem: &InputProblem, cx: &mut Context<StarlingApp>) -> Div {
    let settings = settings_buttons(problem, "mic-check-settings", cx);
    div()
        .flex()
        .flex_col()
        .gap(px(6.))
        .p(px(9.))
        .bg(theme::SETTINGS_CALLOUT)
        .rounded(px(3.))
        .child(
            div()
                .text_size(px(10.))
                .font_weight(FontWeight::SEMIBOLD)
                .child(problem.message()),
        )
        .child(note(problem.recovery()))
        .child(
            div()
                .flex()
                .flex_row()
                .gap(px(6.))
                .child(
                    button("mic-check-retry", "Retry")
                        .on_click(cx.listener(|this, _, _window, cx| this.start_mic_check(cx))),
                )
                .children(settings),
        )
}

/// The microphone & shortcut check: optional, repeatable, never saved,
/// never inserted anywhere.
fn render_check(app: &mut StarlingApp, cx: &mut Context<StarlingApp>) -> Div {
    let shortcut = app.shortcut.label();
    let shortcut_line = match &app.shortcut_registration {
        Ok(()) => format!("Shortcut: {shortcut} (registered system-wide)."),
        Err(reason) => format!(
            "Shortcut: {shortcut} — not available system-wide ({reason}); it still works while \
             this window has focus."
        ),
    };
    let heard = if app.mic.shortcut_heard.is_some() {
        format!("✓ {shortcut} was received while this dialog was open.")
    } else {
        format!(
            "Press {shortcut} now to check it arrives (it does not record while Settings is open)."
        )
    };

    let body: Div = match app.mic.check.as_ref() {
        None => note(
            "Records a few seconds from the selection above (before saving), shows the level, \
             and transcribes it here. Like a normal take, non-silent audio is sent to your \
             transcription engine (the bundled engine on this machine, or your own server in \
             manual mode) for this preview. Nothing is saved to history or typed into another \
             app. Optional — you can record normally without it.",
        ),
        Some(MicCheck::Blocked(reason)) => note(*reason),
        Some(MicCheck::Recording {
            handle,
            meter: fill,
        }) => {
            let device = handle.input_route().map(route_line).unwrap_or_default();
            div()
                .flex()
                .flex_col()
                .gap(px(6.))
                .child(note(format!(
                    "Recording {:.1} s — speak a sentence. {device}",
                    handle.elapsed().as_secs_f32()
                )))
                .child(meter(*fill))
        }
        Some(MicCheck::Transcribing {
            route,
            level,
            seconds,
        }) => note(format!(
            "Transcribing {seconds:.1} s from {} (level {:.0} dBFS)…",
            route
                .as_ref()
                .map(|route| route.device.as_str())
                .unwrap_or("the microphone"),
            level.rms_dbfs()
        )),
        Some(MicCheck::Failed { problem, route }) => {
            let block = problem_block(problem, cx);
            div()
                .flex()
                .flex_col()
                .gap(px(6.))
                .children(route.as_ref().and_then(|route| route.notice()).map(note))
                .child(block)
        }
        Some(MicCheck::Done {
            route,
            level,
            seconds,
            outcome,
        }) => {
            let summary = format!(
                "{seconds:.1} s from {} · peak {:.0} dBFS, average {:.0} dBFS.",
                route
                    .as_ref()
                    .map(|route| route.device.as_str())
                    .unwrap_or("the microphone"),
                20.0 * level.peak.max(1e-9).log10(),
                level.rms_dbfs()
            );
            let result: Div = match outcome {
                CheckOutcome::Transcript(text) => div().child(
                    div()
                        .id("mic-check-preview")
                        .p(px(9.))
                        .rounded(px(3.))
                        .border_1()
                        .border_color(theme::SETTINGS_LINE)
                        .text_size(px(11.))
                        .child(if text.trim().is_empty() {
                            "(The engine heard no words.)".to_string()
                        } else {
                            text.clone()
                        }),
                ),
                CheckOutcome::Silent(problem) => problem_block(problem, cx),
                CheckOutcome::NoEngine => note(
                    "The microphone works. Transcription was skipped: no transcription engine \
                     is ready yet.",
                ),
                CheckOutcome::TranscriptionFailed(message) => note(format!(
                    "The microphone works, but transcription failed: {message}"
                )),
                CheckOutcome::Interrupted(message) => {
                    note(format!("{message}. Check the device, then test again."))
                }
            };
            div()
                .flex()
                .flex_col()
                .gap(px(6.))
                .children(route.as_ref().and_then(|route| route.notice()).map(note))
                .child(note(summary))
                .child(result)
        }
    };

    let recording = matches!(app.mic.check, Some(MicCheck::Recording { .. }));
    let busy = matches!(app.mic.check, Some(MicCheck::Transcribing { .. }));
    let action = if recording {
        Some(
            button("mic-check-stop", "Stop and transcribe")
                .on_click(cx.listener(|this, _, _window, cx| this.stop_mic_check(cx))),
        )
    } else if busy {
        None
    } else {
        Some(
            button(
                "mic-check-start",
                if app.mic.check.is_some() {
                    "Test again"
                } else {
                    "Test microphone"
                },
            )
            .on_click(cx.listener(|this, _, _window, cx| this.start_mic_check(cx))),
        )
    };

    div()
        .flex()
        .flex_col()
        .gap(px(8.))
        .child(eyebrow("TEST MICROPHONE & SHORTCUT"))
        .child(body)
        .child(div().flex().flex_row().gap(px(6.)).children(action))
        .children(app.mic.settings_launch_error.clone().map(note))
        .child(note(shortcut_line))
        .child(note(heard))
}

/// The capture pane's input line while recording: the device in use, and
/// the fallback notice when it is not the one the user chose.
pub(crate) fn live_input_line(app: &StarlingApp) -> Option<Div> {
    let route = app.recorder.as_ref()?.input_route()?;
    let fallback = route.is_fallback();
    Some(
        div().child(
            div()
                .id("live-input")
                .max_w(px(460.))
                .text_size(px(10.))
                .text_color(if fallback { theme::CORAL } else { theme::DIM })
                .child(if fallback {
                    route_line(route)
                } else if route.follows_sound_server() {
                    "Mic: system default (chosen by the sound server)".to_string()
                } else {
                    format!("Mic: {}", route.device)
                }),
        ),
    )
}

/// Recovery actions under the error banner while it explains an input
/// problem: retry, the right OS settings page, or the microphone picker.
pub(crate) fn input_problem_actions(
    app: &StarlingApp,
    cx: &mut Context<StarlingApp>,
) -> Option<Div> {
    let settings = settings_buttons(app.shown_input_problem()?, "input-settings", cx);
    let actions = div()
        .mt(px(6.))
        .flex()
        .flex_row()
        .flex_wrap()
        .gap(px(6.))
        .child(
            button("input-retry", "Retry").on_click(cx.listener(|this, _, _window, cx| {
                this.toggle_recording(crate::activation::RecordButton::Start, cx)
            })),
        )
        .children(settings)
        .child(
            button("input-choose", "Choose microphone")
                .on_click(cx.listener(|this, _, _window, cx| this.open_settings(cx))),
        );
    Some(
        div().child(actions).children(
            app.mic
                .settings_launch_error
                .clone()
                .map(|error| div().mt(px(6.)).child(error)),
        ),
    )
}
