//! The Licenses page: Starling's own license, then every open-source
//! component the app ships with its license; a row opens its full text.

use gpui::{Context, Div, FontWeight, SharedString, div, prelude::*, px};

use crate::app::StarlingApp;
use crate::licenses::{self, Component, Notices};
use crate::theme;
use crate::views::kit;

/// Which license text is open on the page.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpenLicense {
    Starling,
    /// An index into the notices' components.
    Component(usize),
}

pub(super) fn render_licenses_section(app: &mut StarlingApp, cx: &mut Context<StarlingApp>) -> Div {
    let open = app.license_open;
    let notices = licenses::notices();
    let mut page = div()
        .flex()
        .flex_col()
        .child(kit::sheet_intro(
            "Starling is free software under the MIT license, built on the open-source \
             components below. Select one to read its license.",
        ))
        .child(div().mt(px(16.)).child(row(
            "license-starling",
            "Starling",
            env!("CARGO_PKG_VERSION"),
            "MIT",
            open == Some(OpenLicense::Starling),
            cx,
            OpenLicense::Starling,
        )))
        .children((open == Some(OpenLicense::Starling)).then(|| {
            text_panel(
                "https://github.com/sims1253/starling",
                [licenses::STARLING_LICENSE],
            )
        }));

    match notices {
        None => {
            page = page.child(kit::sheet_callout("licenses-missing").mt(px(16.)).child(
                "This development build carries no third-party notices; packaged builds \
                 include them. Run python3 scripts/licenses/notices.py from the repository \
                 root and rebuild to list them here.",
            ));
        }
        Some(notices) => {
            page = page.child(
                kit::sheet_subheading(format!(
                    "Open-source components ({})",
                    notices.components.len()
                ))
                .mt(px(22.))
                .mb(px(8.)),
            );
            for (index, component) in notices.components.iter().enumerate() {
                let key = OpenLicense::Component(index);
                page = page
                    .child(row(
                        SharedString::from(format!("license-{index}")),
                        &component.name,
                        &component.version,
                        &component.license,
                        open == Some(key),
                        cx,
                        key,
                    ))
                    .children((open == Some(key)).then(|| component_panel(notices, component)));
            }
        }
    }
    page
}

fn row(
    id: impl Into<SharedString>,
    name: &str,
    version: &str,
    license: &str,
    is_open: bool,
    cx: &mut Context<StarlingApp>,
    key: OpenLicense,
) -> impl IntoElement {
    let id: SharedString = id.into();
    let selector = id.to_string();
    div()
        .id(id)
        .debug_selector(move || selector)
        .flex()
        .flex_row()
        .items_baseline()
        .gap(px(8.))
        .px(px(10.))
        .py(px(6.))
        .rounded(px(6.))
        .cursor_pointer()
        .when(is_open, |row| row.bg(theme::SHEET_HOVER))
        .hover(|style| style.bg(theme::SHEET_HOVER))
        .on_click(cx.listener(move |this, _, _window, cx| {
            this.license_open = if this.license_open == Some(key) {
                None
            } else {
                Some(key)
            };
            cx.notify();
        }))
        .child(
            div()
                .font_weight(FontWeight::SEMIBOLD)
                .text_color(theme::SHEET_INK)
                .child(SharedString::from(name.to_owned())),
        )
        .child(
            div()
                .flex_1()
                .text_size(px(theme::TEXT_SM))
                .text_color(theme::SHEET_QUILL)
                .child(SharedString::from(version.to_owned())),
        )
        .child(
            div()
                .text_size(px(theme::TEXT_SM))
                .text_color(theme::SHEET_QUILL)
                .child(SharedString::from(license.to_owned())),
        )
}

fn component_panel(notices: &Notices, component: &Component) -> Div {
    text_panel(&component.url, notices.texts(component))
}

fn text_panel<'a>(url: &str, texts: impl IntoIterator<Item = &'a str>) -> Div {
    let mut panel = div()
        .flex()
        .flex_col()
        .gap(px(12.))
        .mx(px(10.))
        .mt(px(4.))
        .mb(px(10.))
        .p(px(12.))
        .rounded(px(6.))
        .bg(theme::SHEET_CALLOUT)
        .child(
            div()
                .text_size(px(theme::TEXT_SM))
                .text_color(theme::SHEET_QUILL)
                .child(SharedString::from(url.to_owned())),
        );
    for text in texts {
        panel = panel.child(
            div()
                .font(theme::mono_font())
                .text_size(px(theme::TEXT_XS))
                .line_height(px(theme::TEXT_XS * 1.45))
                .text_color(theme::SHEET_INK)
                .child(SharedString::from(text.to_owned())),
        );
    }
    panel
}

#[cfg(test)]
mod tests {
    use gpui::{Modifiers, VisualTestContext};

    use super::OpenLicense;
    use crate::app::StarlingApp;
    use crate::views::SettingsPage;

    /// The page renders inside the real settings sheet, and a row opens
    /// (and closes again) its license text.
    #[gpui::test]
    fn a_row_opens_and_closes_its_license(cx: &mut gpui::TestAppContext) {
        let (app, cx): (_, &mut VisualTestContext) =
            cx.add_window_view(|_window, cx| StarlingApp::for_test(None, cx));
        app.update(cx, |app, cx| {
            app.open_settings_at(SettingsPage::Licenses, cx)
        });
        cx.run_until_parked();

        let click = |selector: &'static str, cx: &mut VisualTestContext| {
            let bounds = cx
                .debug_bounds(selector)
                .unwrap_or_else(|| panic!("{selector} rendered"));
            cx.simulate_click(bounds.center(), Modifiers::none());
            cx.run_until_parked();
        };
        click("license-starling", cx);
        assert_eq!(
            app.read_with(cx, |app, _| app.license_open),
            Some(OpenLicense::Starling)
        );
        click("license-starling", cx);
        assert_eq!(app.read_with(cx, |app, _| app.license_open), None);

        // With generated notices (packaging, or notices.py before the
        // build) every component has a row of its own.
        if crate::licenses::notices().is_some_and(|n| !n.components.is_empty()) {
            click("license-0", cx);
            assert_eq!(
                app.read_with(cx, |app, _| app.license_open),
                Some(OpenLicense::Component(0))
            );
        }
    }
}
