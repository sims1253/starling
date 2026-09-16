//! Embedded monochrome SVG icons (lucide 24x24 stroke paths). gpui renders
//! SVGs as alpha masks tinted by the element's text color.

use std::borrow::Cow;

use gpui::{AssetSource, Result, SharedString};

const HEAD: &str = r##"<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#000000" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">"##;
const TAIL: &str = "</svg>";

fn icon(body: &str) -> Cow<'static, [u8]> {
    Cow::Owned(format!("{HEAD}{body}{TAIL}").into_bytes())
}

pub struct Assets;

impl AssetSource for Assets {
    fn load(&self, path: &str) -> Result<Option<Cow<'static, [u8]>>> {
        let body = match path {
            "icons/settings.svg" => {
                r#"<path d="M20 7h-9"/><path d="M14 17H5"/><circle cx="17" cy="17" r="3"/><circle cx="7" cy="7" r="3"/>"#
            }
            "icons/mic.svg" => {
                r#"<path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" x2="12" y1="19" y2="22"/>"#
            }
            "icons/file-audio.svg" => {
                r#"<path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9Z"/><path d="M13 2v7h7"/><path d="M9 18v-6"/><path d="M15 14.5c0 1-1.34 1.5-3 1.5s-3-.5-3-1.5 1.34-1.5 3-1.5 3 .5 3 1.5Z"/><path d="M15 14.5V18c0 1-1.34 2-3 2s-3-1-3-2v-3.5"/>"#
            }
            "icons/alert-circle.svg" => {
                r#"<circle cx="12" cy="12" r="10"/><line x1="12" x2="12" y1="8" y2="12"/><line x1="12" x2="12.01" y1="16" y2="16"/>"#
            }
            "icons/x.svg" => r#"<path d="M18 6 6 18"/><path d="m6 6 12 12"/>"#,
            "icons/clock.svg" => {
                r#"<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>"#
            }
            "icons/loader.svg" => r#"<path d="M21 12a9 9 0 1 1-6.219-8.56"/>"#,
            "icons/chevron-right.svg" => r#"<path d="m9 18 6-6-6-6"/>"#,
            "icons/refresh.svg" => {
                r#"<path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"/><path d="M21 3v5h-5"/><path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"/><path d="M8 16H3v5"/>"#
            }
            "icons/check.svg" => r#"<path d="M20 6 9 17l-5-5"/>"#,
            "icons/clipboard.svg" => {
                r#"<rect width="8" height="4" x="8" y="2" rx="1" ry="1"/><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/>"#
            }
            "icons/download.svg" => {
                r#"<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" x2="12" y1="15" y2="3"/>"#
            }
            "icons/trash.svg" => {
                r#"<path d="M3 6h18"/><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"/><path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"/><line x1="10" x2="10" y1="11" y2="17"/><line x1="14" x2="14" y1="11" y2="17"/>"#
            }
            "icons/play.svg" => r#"<polygon points="6 3 20 12 6 21 6 3"/>"#,
            "icons/stop.svg" => r#"<rect width="14" height="14" x="5" y="5" rx="2"/>"#,
            _ => return Ok(None),
        };
        Ok(Some(icon(body)))
    }

    fn list(&self, path: &str) -> Result<Vec<SharedString>> {
        if path.starts_with("icons") {
            Ok([
                "settings",
                "mic",
                "file-audio",
                "alert-circle",
                "x",
                "clock",
                "loader",
                "chevron-right",
                "refresh",
                "check",
                "clipboard",
                "download",
                "trash",
                "play",
                "stop",
            ]
            .iter()
            .map(|name| SharedString::from(format!("icons/{name}.svg")))
            .collect())
        } else {
            Ok(vec![])
        }
    }
}
