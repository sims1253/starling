//! Settings → Dictation, Linux additions (#221): the desktop shortcut
//! (GlobalShortcuts portal) row and the "System check" panel.

use gpui::{div, prelude::*, px, Context, Div, FontWeight, SharedString};

use crate::app::StarlingApp;
use crate::system_check::Verdict;
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

/// The desktop shortcut row: what the portal binding is, and the action
/// that fits (set it up, or open the desktop's own settings for it).
/// Nothing outside a native Wayland session.
pub(crate) fn render_desktop_shortcut(app: &mut StarlingApp, cx: &mut Context<StarlingApp>) -> Option<gpui::Stateful<Div>> {
    let status = app.portal_shortcuts.as_ref()?.status().clone();
    let mut actions = div().flex().flex_row().gap(px(8.)).mt(px(8.));
    let mut any_action = false;
    if status.can_set_up() {
        any_action = true;
        actions = actions.child(button("portal-set-up", "Set up desktop shortcut").on_click(
            cx.listener(|this, _, _window, _cx| {
                if let Some(portal) = this.portal_shortcuts.as_ref() {
                    portal.set_up(&this.shortcut);
                }
            }),
        ));
    }
    if status.configurable() {
        any_action = true;
        actions = actions.child(
            button("portal-configure", "Change in desktop settings").on_click(cx.listener(
                |this, _, _window, _cx| {
                    if let Some(portal) = this.portal_shortcuts.as_ref() {
                        portal.configure();
                    }
                },
            )),
        );
    }
    Some(
        div()
            .id("desktop-shortcut")
            .mt(px(12.))
            .child(
                div()
                    .text_size(px(10.))
                    .font_weight(FontWeight::SEMIBOLD)
                    .text_color(theme::SETTINGS_INK)
                    .child("Desktop shortcut (Wayland)"),
            )
            .child(note(status.describe()))
            .when(any_action, |row| row.child(actions)),
    )
}

/// The setup check: a button, then one line per topic with its fix.
/// Linux only; elsewhere nothing.
pub(crate) fn render_system_check(app: &mut StarlingApp, cx: &mut Context<StarlingApp>) -> Option<gpui::Stateful<Div>> {
    if !cfg!(target_os = "linux") {
        return None;
    }
    let running = app.system_check.running;
    let lines = app.system_check_lines();
    let label = match (running, lines.is_some()) {
        (true, _) => "Checking…",
        (false, true) => "Check again",
        (false, false) => "Run system check",
    };
    let mut run = button("system-check-run", label);
    if !running {
        run = run.on_click(cx.listener(|this, _, _window, cx| this.run_system_check(cx)));
    }
    let mut body = div().flex().flex_col().gap(px(8.)).mt(px(10.));
    match lines {
        None => {
            body = body.child(note(
                "Checks what this desktop session gives Starling: the session type, the \
                 desktop's shortcut portal, typing into other apps, the accessibility bus, \
                 input methods, the sound server and the microphones. Read-only; it changes \
                 nothing.",
            ));
        }
        Some(lines) => {
            for (index, line) in lines.into_iter().enumerate() {
                let (mark, color) = match line.verdict {
                    Verdict::Ok => ("✓", theme::SETTINGS_INK),
                    Verdict::Limited => ("◐", theme::SETTINGS_INK),
                    Verdict::Missing => ("✗", theme::DANGER),
                    Verdict::Info => ("·", theme::SETTINGS_HELPER),
                };
                body = body.child(
                    div()
                        .id(SharedString::from(format!("system-check-{index}")))
                        .flex()
                        .flex_row()
                        .gap(px(8.))
                        .child(
                            div()
                                .w(px(12.))
                                .flex_none()
                                .text_size(px(10.))
                                .text_color(color)
                                .child(mark),
                        )
                        .child(
                            div()
                                .flex()
                                .flex_col()
                                .flex_1()
                                .child(
                                    div()
                                        .text_size(px(10.))
                                        .line_height(px(10. * 1.55))
                                        .text_color(theme::SETTINGS_INK)
                                        .child(format!("{}: {}", line.topic, line.summary)),
                                )
                                .children(line.fix.map(|fix| note(format!("Fix: {fix}")))),
                        ),
                );
            }
        }
    }
    Some(
        div()
            .id("system-check")
            .mt(px(16.))
            .child(
                div()
                    .flex()
                    .flex_row()
                    .items_center()
                    .justify_between()
                    .child(
                        div()
                            .text_size(px(10.))
                            .font_weight(FontWeight::SEMIBOLD)
                            .text_color(theme::SETTINGS_INK)
                            .child("System check"),
                    )
                    .child(run),
            )
            .child(body),
    )
}
