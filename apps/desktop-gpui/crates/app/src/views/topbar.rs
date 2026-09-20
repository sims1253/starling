//! The 68px topbar: brand mark, centered connection indicator, settings gear.

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
        .h(px(68.))
        .flex()
        .flex_row()
        .items_center()
        .px(px(28.))
        .border_b_1()
        .border_color(theme::LINE)
        .child(
            div()
                .flex()
                .flex_row()
                .items_center()
                .gap(px(11.))
                .font(theme::serif_font())
                .text_size(px(22.))
                .text_color(theme::INK)
                .child(
                    div()
                        .size(px(25.))
                        .rounded_full()
                        .border_1()
                        .border_color(theme::BRAND_RING)
                        .flex()
                        .items_center()
                        .justify_center()
                        .gap(px(2.))
                        .child(div().w(px(2.)).h(px(8.)).rounded(px(2.)).bg(theme::LIME))
                        .child(div().w(px(2.)).h(px(15.)).rounded(px(2.)).bg(theme::LIME))
                        .child(div().w(px(2.)).h(px(5.)).rounded(px(2.)).bg(theme::LIME)),
                )
                .child("starling"),
        )
        .child(
            div().flex_1().flex().flex_row().justify_center().child(
                div()
                    .flex()
                    .flex_row()
                    .items_center()
                    .gap(px(8.))
                    .font(theme::mono_font())
                    .text_size(px(10.))
                    .text_color(theme::MUTED)
                    .child(status_dot(TOPBAR_STATUS_DOT_ID, app.connection, true))
                    .child(connection_label(app)),
            ),
        )
        .child(
            div().flex_1().flex().flex_row().justify_end().child(
                div()
                    .id("open-settings")
                    .size(px(38.))
                    .rounded_full()
                    .flex()
                    .items_center()
                    .justify_center()
                    .text_color(theme::MUTED)
                    .cursor_pointer()
                    .hover(|style| style.bg(theme::GEAR_HOVER_BG).text_color(theme::INK))
                    .on_click(cx.listener(|this, _, _window, cx| {
                        this.open_settings(cx);
                    }))
                    .child(icon("icons/settings.svg", 19., theme::MUTED)),
            ),
        )
}
