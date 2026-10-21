//! The few controls every region shares, so a button on the night and a
//! button on the sheet are the same button in two inks.

use gpui::{Div, FontWeight, Rgba, SharedString, Stateful, div, prelude::*, px};

use crate::theme;

/// Which surface a control sits on, and how loudly it speaks there.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Tone {
    /// The one thing to do next, on the night: filled with the beak.
    Primary,
    /// Everything else on the night.
    Quiet,
    /// The one thing to do next on a sheet: filled with sheet ink.
    SheetPrimary,
    /// Everything else on a sheet.
    SheetQuiet,
    /// Destructive, on a sheet: rose text, no fill.
    SheetDanger,
}

/// A text button. `disabled` drops the pointer and hover and dims it; the
/// caller still decides whether its click does anything.
pub(crate) fn button(id: impl Into<gpui::ElementId>, tone: Tone, disabled: bool) -> Stateful<Div> {
    let (fg, bg, line, hover): (Rgba, Option<Rgba>, Rgba, Rgba) = match tone {
        Tone::Primary => (theme::ON_BEAK, Some(theme::BEAK), theme::BEAK, theme::BEAK),
        Tone::Quiet => (theme::SPECK, None, theme::LINE, theme::HOVER),
        Tone::SheetPrimary => (
            theme::SHEET_CARD,
            Some(theme::SHEET_INK),
            theme::SHEET_INK,
            theme::SHEET_INK,
        ),
        Tone::SheetQuiet => (
            theme::SHEET_INK,
            None,
            theme::SHEET_LINE,
            theme::SHEET_HOVER,
        ),
        Tone::SheetDanger => (
            theme::SHEET_ROSE,
            None,
            theme::SHEET_LINE,
            theme::SHEET_HOVER,
        ),
    };
    let filled = bg.is_some();
    div()
        .id(id)
        .flex()
        .flex_none()
        .flex_row()
        .items_center()
        .gap(px(6.))
        .h(px(30.))
        .px(px(11.))
        .rounded(px(6.))
        .border_1()
        .border_color(line)
        .when_some(bg, |button, bg| button.bg(bg))
        .text_size(px(theme::TEXT_SM))
        .text_color(fg)
        .when(filled, |button| button.font_weight(FontWeight::SEMIBOLD))
        .when(!disabled, |button| {
            button.cursor_pointer().hover(move |style| {
                if filled {
                    style.opacity(0.88)
                } else {
                    style.bg(hover)
                }
            })
        })
        .when(disabled, |button| button.opacity(0.4))
}

/// The mark of a choice row on a sheet: an ink ring, filled with the
/// deepened beak when chosen.
pub(crate) fn radio(selected: bool) -> Div {
    div()
        .mt(px(2.))
        .size(px(14.))
        .flex_none()
        .flex()
        .items_center()
        .justify_center()
        .rounded_full()
        .border_1()
        .border_color(if selected {
            theme::SHEET_BEAK
        } else {
            theme::SHEET_QUILL
        })
        .when(selected, |ring| {
            ring.child(div().size(px(8.)).rounded_full().bg(theme::SHEET_BEAK))
        })
}

/// A shortcut drawn as keycaps: "Ctrl Shift Space" becomes three caps.
pub(crate) fn keycaps(label: &str, on_sheet: bool) -> Div {
    let (fg, line) = if on_sheet {
        (theme::SHEET_INK, theme::SHEET_LINE)
    } else {
        (theme::SPECK, theme::LINE)
    };
    div().flex().flex_row().items_center().gap(px(4.)).children(
        shortcut_keys(label).into_iter().map(move |key| {
            div()
                .px(px(6.))
                .pt(px(1.))
                .pb(px(2.))
                .rounded(px(4.))
                .border_1()
                .border_b_2()
                .border_color(line)
                .font(theme::mono_font())
                .text_size(px(theme::TEXT_XS))
                .text_color(fg)
                .child(key)
        }),
    )
}

/// The keys of a shortcut label (`Shortcut::label`), in order. Every key
/// name it produces is one word, so the spaces are the separators.
pub(crate) fn shortcut_keys(label: &str) -> Vec<SharedString> {
    label
        .split_whitespace()
        .map(|key| SharedString::from(key.to_string()))
        .collect()
}

/// A paragraph that explains, on a sheet.
pub(crate) fn sheet_intro(text: impl Into<SharedString>) -> Div {
    div()
        .max_w(px(560.))
        .text_size(px(theme::TEXT_MD))
        .line_height(px(theme::TEXT_MD * 1.55))
        .text_color(theme::SHEET_QUILL)
        .child(text.into())
}

/// A tinted box for what the system can or cannot do, on a sheet.
pub(crate) fn sheet_callout(id: impl Into<gpui::ElementId>) -> Stateful<Div> {
    div()
        .id(id)
        .p(px(12.))
        .rounded(px(6.))
        .bg(theme::SHEET_CALLOUT)
        .text_size(px(theme::TEXT_SM))
        .line_height(px(theme::TEXT_SM * 1.55))
        .text_color(theme::SHEET_INK)
}

/// A group's own heading inside a settings page.
pub(crate) fn sheet_subheading(text: impl Into<SharedString>) -> Div {
    div()
        .text_size(px(theme::TEXT_MD))
        .font_weight(FontWeight::SEMIBOLD)
        .text_color(theme::SHEET_INK)
        .child(text.into())
}

#[cfg(test)]
mod tests {
    use super::shortcut_keys;

    fn keys(label: &str) -> Vec<String> {
        shortcut_keys(label)
            .into_iter()
            .map(|key| key.to_string())
            .collect()
    }

    #[test]
    fn a_shortcut_label_splits_into_its_keys() {
        assert_eq!(keys("Ctrl Shift Space"), ["Ctrl", "Shift", "Space"]);
        assert_eq!(keys("F9"), ["F9"]);
        assert_eq!(keys("⌘ Shift ="), ["⌘", "Shift", "="]);
    }
}
