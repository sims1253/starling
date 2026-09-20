//! The settings modal: dark scrim over the workspace, light centered card
//! with endpoint/model/terms fields, protocol toggle, callout and footer.

use std::time::Duration;

use gpui::{
    Animation, AnimationExt, Context, Div, FontWeight, MouseButton, Window, div, prelude::*, px,
    rgba,
};
use starling_dictation::settings;

use crate::app::{Connection, StarlingApp};
use crate::theme;
use crate::views::{
    SETTINGS_CALLOUT_DOT_ID, icon, protocol_option_id, status_dot,
};

pub fn render_settings_modal(
    app: &mut StarlingApp,
    _window: &mut Window,
    cx: &mut Context<StarlingApp>,
) -> impl IntoElement {
    let endpoint_for_callout = app.endpoint.clone();
    let connection = app.connection;
    let draft_endpoint = app.draft_endpoint.clone();
    let draft_model = app.draft_model.clone();
    let draft_terms = app.draft_terms.clone();
    let selected_protocol = app.settings_protocol;

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
        .child(
            div()
                .flex()
                .flex_row()
                .gap(px(12.))
                .child(
                    field_label("API format")
                        .flex_1()
                        .min_w_0()
                        .mt(px(0.))
                        .child(protocol_toggle(selected_protocol, cx)),
                )
                .child(
                    field_label("Model")
                        .flex_1()
                        .min_w_0()
                        .mt(px(0.))
                        .child(draft_model),
                ),
        )
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
                .child(status_dot(SETTINGS_CALLOUT_DOT_ID, connection, false))
                .child(
                    div()
                        .flex()
                        .flex_col()
                        .gap(px(2.))
                        .text_size(px(10.))
                        .child(div().font_weight(FontWeight::SEMIBOLD).child(
                            if connection == Connection::Ready {
                                "Server connected"
                            } else {
                                "Server needs attention"
                            },
                        ))
                        .child(
                            div()
                                .font(theme::mono_font())
                                .text_size(px(9.))
                                .text_color(theme::SETTINGS_CALLOUT_ENDPOINT)
                                .child(endpoint_for_callout),
                        ),
                ),
        )
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
                        .cursor_pointer()
                        .hover(|style| style.bg(theme::PAPER_HOVER))
                        .on_click(cx.listener(|this, _, _window, cx| {
                            this.test_connection(cx);
                        }))
                        .child("Test connection"),
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

fn protocol_toggle(selected: settings::Protocol, cx: &mut Context<StarlingApp>) -> Div {
    let options = [
        (settings::Protocol::Starling, "Starling native"),
        (settings::Protocol::OpenAI, "OpenAI compatible"),
    ];
    let mut toggle = div()
        .flex()
        .flex_row()
        .w_full()
        .rounded(px(3.))
        .border_1()
        .border_color(theme::SETTINGS_LINE)
        .overflow_hidden();
    for (protocol, label) in options {
        let is_selected = selected == protocol;
        toggle = toggle.child(
            div()
                // R04: the id names the option and nothing else — the old
                // "protocol-selected"/"protocol-option" pair flipped both
                // options' identities on every click.
                .id(protocol_option_id(protocol))
                .flex()
                .flex_1()
                .items_center()
                .justify_center()
                .py(px(11.))
                .font(theme::mono_font())
                .text_size(px(11.))
                .text_color(if is_selected {
                    theme::SETTINGS_PRIMARY_TEXT
                } else {
                    theme::SETTINGS_INK
                })
                .bg(if is_selected {
                    theme::SETTINGS_INK
                } else {
                    theme::SETTINGS_FIELD_BG
                })
                .cursor_pointer()
                .on_click(cx.listener(move |this, _, _window, cx| {
                    this.settings_protocol = protocol;
                    cx.notify();
                }))
                .child(label),
        );
    }
    toggle
}
