//! The settings modal: dark scrim over the workspace, light centered card
//! with endpoint/model/terms fields, callout and footer.

use std::time::Duration;

use gpui::{
    Animation, AnimationExt, Context, Div, FontWeight, MouseButton, Window, div, prelude::*, px,
    rgba,
};
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
    // and without a probe the committed status shows.
    let callout = settings_callout_view(app.probe.as_ref(), app.connection, &app.endpoint);
    let probing = matches!(app.probe, Some(ConnectionProbe::Testing { .. }));
    let draft_endpoint = app.draft_endpoint.clone();
    let draft_model = app.draft_model.clone();
    let draft_terms = app.draft_terms.clone();
    let processing_section = render_processing_section(app, cx);

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
        .child(
            div()
                .mb(px(13.))
                .font(theme::mono_font())
                .text_size(px(10.))
                .font_weight(FontWeight::MEDIUM)
                .text_color(theme::SETTINGS_EYEBROW)
                .child("CONNECTION"),
        )
        .child(
            div()
                .mb(px(13.))
                .font(theme::serif_font())
                .font_weight(FontWeight::MEDIUM)
                .text_size(px(33.))
                .line_height(px(33.))
                .child("Transcription server"),
        )
        .child(
            div()
                .mb(px(25.))
                .text_size(px(11.))
                .line_height(px(11. * 1.65))
                .text_color(theme::SETTINGS_MUTED)
                .child(
                    "Choose a starling-serve endpoint or an OpenAI-compatible transcription \
                     endpoint.",
                ),
        )
        .child(field_label("Server endpoint").child(draft_endpoint))
        .child(field_label("Model").child(draft_model))
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
        .child(
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
        .child(processing_section)
        .child(
            div()
                .flex()
                .flex_row()
                .justify_end()
                .gap(px(9.))
                .mt(px(25.))
                .child(
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

fn helper(text: &'static str) -> Div {
    div()
        .mt(px(2.))
        .font_weight(FontWeight::NORMAL)
        .line_height(px(10. * 1.5))
        .text_color(theme::SETTINGS_HELPER)
        .child(text)
}
