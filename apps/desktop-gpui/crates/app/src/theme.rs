//! Design tokens and small formatting helpers, mirroring
//! `apps/desktop/src/styles.css`.

use chrono::{DateTime, Local};
use gpui::{Font, FontFallbacks, FontFeatures, FontStyle, FontWeight, Pixels, Rgba, px};

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

// Root surfaces
pub const BG: Rgba = rgb(0x171813);
pub const PANEL_SOFT: Rgba = rgba(0x1E1F1AB3); // rgba(30,31,26,0.7)
pub const SCRIM: Rgba = rgba(0x060705B8); // rgba(6,7,5,0.72)
pub const PAPER: Rgba = rgb(0xE8E4D9);
pub const PAPER_LIGHT: Rgba = rgb(0xEFEBE0);

// Text colors
pub const INK: Rgba = rgb(0xE8E5DC);
pub const MUTED: Rgba = rgb(0x98998E);
pub const DIM: Rgba = rgb(0x6E7067);
pub const EYEBROW: Rgba = rgb(0xA8AA8E);
pub const TAKE_TITLE: Rgba = rgb(0xC9C7BF);

// Accents
pub const LIME: Rgba = rgb(0xD9FF6A);
pub const CORAL: Rgba = rgb(0xFF745B);
pub const AMBER: Rgba = rgb(0xEFC26B);
pub const AMBER_DEEP: Rgba = rgb(0x705A2B);
pub const DOT_GREY: Rgba = rgb(0x6F7165);
pub const WAVE_IDLE: Rgba = rgb(0x80836E);
pub const BRAND_RING: Rgba = rgb(0x8B8D6E);

// Lines and translucent fills
pub const LINE: Rgba = rgba(0xE8E5DC1B); // rgba(232,229,220,0.105)
pub const HOVER_BG: Rgba = rgba(0xFFFFFF06); // rgba(255,255,255,0.025)
pub const GEAR_HOVER_BG: Rgba = rgba(0xFFFFFF0E); // rgba(255,255,255,0.055)
pub const ACTIVE_ROW_BG: Rgba = rgba(0xD9FF6A0E); // rgba(217,255,106,0.055)
pub const RECORD_BORDER: Rgba = rgba(0xFFFFFF1F); // rgba(255,255,255,0.12)
pub const RECORD_BG: Rgba = rgba(0x171813CC); // rgba(23,24,19,0.8)
pub const IMPORT_LINE: Rgba = rgb(0x5B5C52);
pub const MIC_FG: Rgba = rgb(0x1C2011);

// Error banner
pub const ERROR_BG: Rgba = rgb(0x39241F);
pub const ERROR_LINE: Rgba = rgba(0xFF745B47); // rgba(255,116,91,0.28)
pub const ERROR_TEXT: Rgba = rgb(0xFFAC99);
pub const ERROR_TITLE: Rgba = rgb(0xFFD8CF);
pub const ERROR_SUBTLE: Rgba = rgb(0xAD8279);
pub const RECOVERY_TEXT: Rgba = rgb(0xFFE3DC);
pub const RECOVERY_LINE: Rgba = rgba(0xFFE3DC59); // rgba(255,227,220,0.35)

// Transcript drawer (paper)
pub const PAPER_INK: Rgba = rgb(0x24251E);
pub const PAPER_SUBTLE: Rgba = rgb(0x7D7D71);
pub const PAPER_EYEBROW: Rgba = rgb(0x747568);
pub const PAPER_LINE: Rgba = rgba(0x24251E1F); // rgba(36,37,30,0.12)
pub const PAPER_LINE_SOFT: Rgba = rgba(0x24251E1A); // rgba(36,37,30,0.1)
pub const PAPER_HOVER: Rgba = rgba(0x24251E12); // rgba(36,37,30,0.07)
pub const PAPER_BUTTON_LINE: Rgba = rgba(0x24251E2E); // rgba(36,37,30,0.18)
pub const FAILED_COPY: Rgba = rgb(0x93483B);
pub const PROCESSING: Rgba = rgb(0x68695E);
pub const GRIP: Rgba = rgb(0xA6A399);
pub const DANGER: Rgba = rgb(0xA04435);

// Settings modal
pub const SETTINGS_INK: Rgba = rgb(0x25261F);
pub const SETTINGS_MUTED: Rgba = rgb(0x68695E);
pub const SETTINGS_EYEBROW: Rgba = rgb(0x77796C);
pub const SETTINGS_LINE: Rgba = rgb(0xC4C1B8);
pub const SETTINGS_FIELD_BG: Rgba = rgba(0xFFFFFF7A); // rgba(255,255,255,0.48)
pub const SETTINGS_CALLOUT: Rgba = rgb(0xDEDBCF);
pub const SETTINGS_CALLOUT_ENDPOINT: Rgba = rgb(0x737468);
pub const SETTINGS_FOOT_LINE: Rgba = rgb(0xA9A79D);
pub const SETTINGS_PRIMARY_TEXT: Rgba = rgb(0xEEEBDF);
pub const SETTINGS_HELPER: Rgba = rgb(0x858579);
pub const FOCUS_RING: Rgba = rgb(0x747E3D);

// gpui's font-kit fork does not consult fontconfig substitution: a missing
// family silently falls back to the default sans. Resolve through an explicit
// fallback chain so the serif/mono design survives on Linux.
pub fn serif_font() -> Font {
    Font {
        family: "Georgia".into(),
        features: FontFeatures::default(),
        fallbacks: Some(FontFallbacks::from_fonts(vec![
            "Iowan Old Style".into(),
            "Palatino Linotype".into(),
            "Noto Serif".into(),
            "DejaVu Serif".into(),
            "Liberation Serif".into(),
            "serif".into(),
        ])),
        weight: FontWeight::default(),
        style: FontStyle::default(),
    }
}

pub fn mono_font() -> Font {
    Font {
        family: "SFMono-Regular".into(),
        features: FontFeatures::default(),
        fallbacks: Some(FontFallbacks::from_fonts(vec![
            "Noto Sans Mono".into(),
            "DejaVu Sans Mono".into(),
            "Liberation Mono".into(),
            "monospace".into(),
        ])),
        weight: FontWeight::default(),
        style: FontStyle::default(),
    }
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

/// `formatWhen` from App.tsx: local `h:mm AM/PM` today, otherwise `Mon D`.
pub fn fmt_when(iso: &str) -> String {
    let Ok(parsed) = DateTime::parse_from_rfc3339(iso) else {
        return iso.to_string();
    };
    let local = parsed.with_timezone(&Local);
    let now = Local::now();
    if local.date_naive() == now.date_naive() {
        local.format("%-I:%M %p").to_string()
    } else {
        local.format("%b %-d").to_string()
    }
}
