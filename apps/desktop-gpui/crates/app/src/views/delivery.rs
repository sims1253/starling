//! The failed-insertion notice and the insertion settings (#221).

use gpui::{Context, Div, FontWeight, SharedString, Stateful, div, prelude::*, px};

use crate::app::StarlingApp;
use crate::theme;
use crate::views::icon;

/// The longest transcript excerpt the notice shows; Copy and Paste last
/// always use the whole text.
const EXCERPT_CHARS: usize = 240;

fn excerpt(text: &str) -> String {
    let mut chars = text.chars();
    let head: String = chars.by_ref().take(EXCERPT_CHARS).collect();
    if chars.next().is_some() {
        format!("“{head}…”")
    } else {
        format!("“{head}”")
    }
}

fn action(id: &'static str, label: impl Into<SharedString>) -> Stateful<Div> {
    div()
        .id(id)
        .px(px(9.))
        .py(px(7.))
        .rounded(px(4.))
        .border_1()
        .border_color(theme::RECOVERY_LINE)
        .text_size(px(9.))
        .text_color(theme::RECOVERY_TEXT)
        .cursor_pointer()
        .child(label.into())
}

/// The notice for a take whose text did not land, in the capture pane's
/// banner slot, or at the pane's top while an error banner holds it.
pub(crate) fn render_recovery(
    app: &StarlingApp,
    at_top: bool,
    cx: &mut Context<StarlingApp>,
) -> Option<Stateful<Div>> {
    let recovery = app.delivery.recovery.as_ref()?;
    let paste_label = if recovery.pasting {
        "Pasting…".to_string()
    } else if recovery.armed.is_some() {
        "Cancel paste".to_string()
    } else {
        "Paste last".to_string()
    };
    let hint = if recovery.armed.is_some() {
        "Now switch to the field it belongs in: Starling types it there once its window loses \
         focus. Nothing is submitted."
    } else {
        "Paste last types it into the field you switch to next. The take stays in history with \
         its audio; dismissing this only hides the notice."
    };
    let mut actions = div()
        .flex()
        .flex_wrap()
        .items_center()
        .gap(px(6.))
        .child(
            action(
                "recovery-copy",
                if recovery.copied { "Copied" } else { "Copy" },
            )
            .on_click(cx.listener(|this, _, _window, cx| this.copy_recovery(cx))),
        )
        .child(
            action("recovery-paste-last", paste_label)
                .on_click(cx.listener(|this, _, _window, cx| this.toggle_paste_last(cx))),
        );
    if recovery.offers_settings() {
        actions = actions.child(
            action("recovery-settings", "Open settings")
                .on_click(cx.listener(|this, _, _window, cx| this.open_settings(cx))),
        );
    }
    Some(
        div()
            .id("delivery-recovery")
            .absolute()
            .left(px(28.))
            .right(px(28.))
            .map(|card| {
                if at_top {
                    card.top(px(28.))
                } else {
                    card.bottom(px(28.))
                }
            })
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
                    .gap(px(5.))
                    .flex_1()
                    // Without it the column sizes to its longest line, and
                    // the text runs past the card instead of wrapping.
                    .min_w(px(0.))
                    .child(div().text_color(theme::ERROR_TITLE).child(recovery.title()))
                    .child(div().child(recovery.explanation()))
                    .child(
                        div()
                            .id("recovery-text")
                            .text_color(theme::RECOVERY_TEXT)
                            .child(excerpt(&recovery.text)),
                    )
                    .child(actions)
                    .child(div().text_color(theme::ERROR_SUBTLE).child(hint)),
            )
            .child(
                div()
                    .id("dismiss-recovery")
                    .cursor_pointer()
                    .on_click(cx.listener(|this, _, _window, cx| this.dismiss_recovery(cx)))
                    .child(icon("icons/x.svg", 16., theme::ERROR_TEXT)),
            ),
    )
}

/// "After a take" in the settings dialog.
pub(crate) fn render_insertion_section(app: &StarlingApp, cx: &mut Context<StarlingApp>) -> Div {
    let draft = app.draft_insertion;
    let insert = super::settings::choice_row(
        SharedString::from("insertion-auto"),
        draft.auto_insert,
        "Insert into the app you dictated into",
        "Typed into the window that had focus when the take started, without pressing Enter. \
         Direct takes are inserted as soon as their text is final. Staged takes wait in the \
         draft panel: press Insert when you are ready and switch back. If that window changed, \
         the text waits here.",
        true,
    )
    .on_click(cx.listener(|this, _, _window, cx| {
        this.draft_insertion.auto_insert = true;
        cx.notify();
    }));
    let copy_only = super::settings::choice_row(
        SharedString::from("insertion-copy-only"),
        !draft.auto_insert,
        "Copy only",
        "Takes stay in Starling; copy them from the take or the panel.",
        true,
    )
    .on_click(cx.listener(|this, _, _window, cx| {
        this.draft_insertion.auto_insert = false;
        cx.notify();
    }));
    let enabled = draft.auto_insert;
    let unverified = super::settings::choice_row(
        SharedString::from("insertion-unverified"),
        draft.allow_unverified,
        "Also type where the window cannot be checked",
        "On Wayland, Starling cannot tell which window has focus: a direct take goes to whatever \
         is focused when it is ready, and a staged take to the window you switch to after \
         Insert. Best effort; a direct take during which Starling's window was focused is never \
         typed.",
        enabled,
    )
    .when(!enabled, |row| row.opacity(0.5))
    .when(enabled, |row| {
        row.on_click(cx.listener(|this, _, _window, cx| {
            this.draft_insertion.allow_unverified = !this.draft_insertion.allow_unverified;
            cx.notify();
        }))
    });
    let note = if app.insertion_unverifiable {
        "Typing here uses the Wayland compositor's virtual keyboard (wlroots, niri, COSMIC; \
         not GNOME or KDE), and the target window cannot be verified."
    } else {
        "The target window is checked again right before every part of the text is typed."
    };

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
                .child("AFTER A TAKE"),
        )
        .child(
            div()
                .flex()
                .flex_col()
                .gap(px(6.))
                .child(insert)
                .child(copy_only),
        )
        .child(div().mt(px(6.)).child(unverified))
        .child(
            div()
                .id("insertion-note")
                .mt(px(12.))
                .bg(theme::SETTINGS_CALLOUT)
                .p(px(12.))
                .text_size(px(10.))
                .line_height(px(10. * 1.55))
                .child(note),
        )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_excerpt_quotes_short_text_whole_and_marks_a_cut() {
        assert_eq!(excerpt("Hello."), "“Hello.”");
        let long = "a".repeat(EXCERPT_CHARS + 5);
        let shown = excerpt(&long);
        assert!(shown.ends_with("…”"));
        assert_eq!(shown.chars().count(), EXCERPT_CHARS + 3);
    }
}
