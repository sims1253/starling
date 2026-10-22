//! The 48px bar: the brand on the left, then what the engine is doing and
//! the settings button on the right.

use gpui::{Context, Div, Window, div, prelude::*, px};

use crate::app::StarlingApp;
use crate::theme;
use crate::views::{TOPBAR_STATUS_DOT_ID, connection_label, icon, status_dot};

pub fn render_topbar(
    app: &mut StarlingApp,
    _window: &mut Window,
    cx: &mut Context<StarlingApp>,
) -> gpui::Stateful<Div> {
    div()
        .id("topbar")
        .h(px(48.))
        .flex_none()
        .flex()
        .flex_row()
        .items_center()
        .gap(px(16.))
        .pl(px(20.))
        .pr(px(12.))
        .border_b_1()
        .border_color(theme::LINE)
        .child(
            div()
                .flex()
                .flex_row()
                .items_center()
                .gap(px(9.))
                .child(icon("icons/brand.svg", 20., theme::BEAK))
                .child(
                    div()
                        .font(theme::display_font())
                        .text_size(px(20.))
                        .text_color(theme::SPECK)
                        .child("Starling"),
                ),
        )
        .child(
            div()
                .id("engine-status")
                .flex_1()
                .min_w_0()
                .flex()
                .flex_row()
                .justify_end()
                .items_center()
                .gap(px(8.))
                .text_size(px(theme::TEXT_SM))
                .text_color(theme::QUILL)
                .child(status_dot(TOPBAR_STATUS_DOT_ID, app.connection))
                .child(div().min_w_0().truncate().child(connection_label(app))),
        )
        .child(
            div()
                .id("open-settings")
                .size(px(34.))
                .flex_none()
                .rounded(px(6.))
                .flex()
                .items_center()
                .justify_center()
                .cursor_pointer()
                .hover(|style| style.bg(theme::HOVER))
                .on_click(cx.listener(|this, _, _window, cx| {
                    this.open_settings(cx);
                }))
                .child(icon("icons/settings.svg", 18., theme::QUILL)),
        )
}
