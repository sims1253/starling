//! Design tokens: the "plumage" palette, the type families, and small
//! formatting helpers.
//!
//! The palette is taken from the bird the app is named after: the night
//! indigo of a starling's coat, the pale specks that give it its name, the
//! gold of its beak, and the teal and rose sheen its feathers throw in the
//! light. Gold is the one action color; teal says "done", rose says
//! "recording" or "something went wrong". Light "sheet" surfaces are for
//! reading: the transcript and the settings.

use std::borrow::Cow;

use chrono::{DateTime, Datelike, Local, Timelike};
use gpui::{App, Font, FontFallbacks, FontFeatures, FontStyle, FontWeight, Pixels, Rgba, px};

const fn rgb(hex: u32) -> Rgba {
    Rgba {
        r: ((hex >> 16) & 0xFF) as f32 / 255.0,
        g: ((hex >> 8) & 0xFF) as f32 / 255.0,
        b: (hex & 0xFF) as f32 / 255.0,
        a: 1.0,
    }
}

/// `0xRRGGBBAA` with alpha as the low byte, matching gpui's `rgba()`.
const fn rgba(hex: u32) -> Rgba {
    Rgba {
        r: ((hex >> 24) & 0xFF) as f32 / 255.0,
        g: ((hex >> 16) & 0xFF) as f32 / 255.0,
        b: ((hex >> 8) & 0xFF) as f32 / 255.0,
        a: (hex & 0xFF) as f32 / 255.0,
    }
}

// ---- Night: the window itself --------------------------------------------
pub const NIGHT: Rgba = rgb(0x191830);
/// The takes column and other surfaces that sit on the night.
pub const ROOST: Rgba = rgb(0x211F3B);
/// A selected row on the roost.
pub const ROOST_SELECTED: Rgba = rgb(0x2C2A50);
pub const HOVER: Rgba = rgba(0xEFE9DC0D);
pub const LINE: Rgba = rgb(0x34325A);
pub const SCRIM: Rgba = rgba(0x0D0C1ACC);

/// Text on the night, and the specks of the level field.
pub const SPECK: Rgba = rgb(0xEFE9DC);
/// Secondary text: what something means, not what it is.
pub const QUILL: Rgba = rgb(0xA6A2C4);
/// Meta text that can be skipped. Still 4.5:1 on the night.
pub const DIM: Rgba = rgb(0x8985AC);

// ---- Accents ----------------------------------------------------------------
/// The action color: the record control, primary buttons, the selection.
pub const BEAK: Rgba = rgb(0xE3AC3B);
/// Text and icons drawn on the beak.
pub const ON_BEAK: Rgba = rgb(0x191830);
/// Ready and done.
pub const SHEEN: Rgba = rgb(0x5FB3A6);
/// Recording, and anything that failed.
pub const ROSE: Rgba = rgb(0xE0607E);

// ---- Notices on the night -----------------------------------------------
pub const NOTICE_BG: Rgba = rgb(0x2E2244);
pub const NOTICE_LINE: Rgba = rgba(0xE0607E59);
pub const NOTICE_TITLE: Rgba = rgb(0xFFE3EA);
pub const NOTICE_TEXT: Rgba = rgb(0xF3B3C2);
pub const NOTICE_SUBTLE: Rgba = rgb(0xBE9AB5);

// ---- Sheet: reading surfaces (transcript, settings) ---------------------
pub const SHEET: Rgba = rgb(0xE7E5EE);
/// The settings card: a step lighter than the transcript sheet.
pub const SHEET_CARD: Rgba = rgb(0xF2F1F6);
pub const SHEET_INK: Rgba = rgb(0x1E1C36);
pub const SHEET_QUILL: Rgba = rgb(0x55527A);
pub const SHEET_LINE: Rgba = rgb(0xCAC7DA);
pub const SHEET_HOVER: Rgba = rgba(0x1E1C3612);
pub const SHEET_FIELD: Rgba = rgba(0xFFFFFFB3);
pub const SHEET_CALLOUT: Rgba = rgb(0xE2E0EC);
/// The beak, deepened until it reads on the sheet: selection and focus.
pub const SHEET_BEAK: Rgba = rgb(0x8F620A);
/// A warning on the sheet (stale proposals, interrupted takes).
pub const SHEET_WARN: Rgba = rgb(0x7A5512);
/// Failed and destructive on the sheet.
pub const SHEET_ROSE: Rgba = rgb(0xAE2F50);

/// The staging editor's selection, on the night.
pub const SELECTION: Rgba = rgba(0xE3AC3B4D);

// ---- Type -----------------------------------------------------------------
//
// Two families. Literata, a serif drawn for long reading on screens, sets
// what the user said: the transcript, in its text cut, and the state
// sentence above the recorder, in its 72pt display cut. Atkinson
// Hyperlegible Next, drawn for low-vision readers, sets everything the
// interface says. Its mono sibling is kept for keycaps and the take timer,
// where figures must not jump. All are embedded (OFL, see
// `assets/fonts`), so the design does not depend on what is installed.

pub const DISPLAY_FAMILY: &str = "Literata 72pt";
pub const READING_FAMILY: &str = "Literata";
pub const UI_FAMILY: &str = "Atkinson Hyperlegible Next";
pub const MONO_FAMILY: &str = "Atkinson Hyperlegible Mono";

/// A modular scale on a 14px base.
pub const TEXT_XS: f32 = 12.;
pub const TEXT_SM: f32 = 13.;
pub const TEXT_MD: f32 = 14.;
pub const TEXT_LG: f32 = 16.;
pub const TEXT_XL: f32 = 18.;
pub const HEADING: f32 = 24.;

const FONTS: [&[u8]; 6] = [
    include_bytes!("../assets/fonts/Literata72pt-Regular.ttf"),
    include_bytes!("../assets/fonts/Literata-Regular.ttf"),
    include_bytes!("../assets/fonts/Literata-Italic.ttf"),
    include_bytes!("../assets/fonts/AtkinsonHyperlegibleNext-Regular.ttf"),
    include_bytes!("../assets/fonts/AtkinsonHyperlegibleNext-SemiBold.ttf"),
    include_bytes!("../assets/fonts/AtkinsonHyperlegibleMono-Regular.ttf"),
];

/// Hand the embedded families to gpui's text system. A failure leaves the
/// fallback chains below to pick installed faces.
pub fn register_fonts(cx: &mut App) {
    let fonts = FONTS.iter().map(|bytes| Cow::Borrowed(*bytes)).collect();
    if let Err(err) = cx.text_system().add_fonts(fonts) {
        eprintln!("Could not load Starling's fonts: {err}");
    }
}

// gpui's font-kit fork does not consult fontconfig substitution: a missing
// family silently falls back to the default sans. Every family carries an
// explicit chain so the design degrades to the nearest installed face.
fn font(family: &'static str, fallbacks: &[&'static str]) -> Font {
    Font {
        family: family.into(),
        features: FontFeatures::default(),
        fallbacks: Some(FontFallbacks::from_fonts(
            fallbacks.iter().map(|name| name.to_string()).collect(),
        )),
        weight: FontWeight::default(),
        style: FontStyle::default(),
    }
}

const SERIF_FALLBACKS: [&str; 5] = [
    "Literata",
    "Georgia",
    "Noto Serif",
    "DejaVu Serif",
    "serif",
];

/// The state sentence above the recorder.
pub fn display_font() -> Font {
    font(DISPLAY_FAMILY, &SERIF_FALLBACKS)
}

/// What the user said: transcripts, drafts, proposals.
pub fn reading_font() -> Font {
    font(READING_FAMILY, &SERIF_FALLBACKS[1..])
}

/// Everything the interface says. Set once on each window's root.
pub fn ui_font() -> Font {
    font(
        UI_FAMILY,
        &[
            "Atkinson Hyperlegible",
            "Segoe UI",
            "Noto Sans",
            "DejaVu Sans",
            "sans-serif",
        ],
    )
}

/// Keycaps and the take timer.
pub fn mono_font() -> Font {
    font(
        MONO_FAMILY,
        &[
            "SFMono-Regular",
            "Noto Sans Mono",
            "DejaVu Sans Mono",
            "monospace",
        ],
    )
}

/// CSS `clamp(min, value, max)` for pixel lengths.
pub fn clamp_px(value: Pixels, min: f32, max: f32) -> Pixels {
    value.max(px(min)).min(px(max))
}

/// `formatDuration` from App.tsx: `m:ss`, `0:00` when missing.
pub fn fmt_duration(ms: Option<f64>) -> String {
    let Some(ms) = ms else {
        return "0:00".to_string();
    };
    if ms <= 0.0 {
        return "0:00".to_string();
    }
    let total = (ms / 1000.0).round() as i64;
    format!("{}:{:02}", total / 60, total % 60)
}

/// `h:mm AM/PM` with explicit padding (R11): chrono's `%-I`/`%-d` are
/// glibc-isms that format as literal text on other platforms; build the
/// string from components instead.
fn twelve_hour(local: &DateTime<Local>) -> String {
    let (is_pm, hour) = local.hour12();
    format!(
        "{hour}:{:02} {}",
        local.minute(),
        if is_pm { "PM" } else { "AM" }
    )
}

/// `formatWhen` from App.tsx: local `h:mm AM/PM` today, otherwise `Mon D`.
pub fn fmt_when(iso: &str) -> String {
    let Ok(parsed) = DateTime::parse_from_rfc3339(iso) else {
        return iso.to_string();
    };
    let local = parsed.with_timezone(&Local);
    let now = Local::now();
    if local.date_naive() == now.date_naive() {
        twelve_hour(&local)
    } else {
        // `%b` is locale-independent in chrono; the day comes from `day()`
        // (no leading zero) rather than a platform-dependent `%-d`.
        format!("{} {}", local.format("%b"), local.day())
    }
}

