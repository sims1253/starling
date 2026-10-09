//! Insertion-boundary formatting: when dictated text lands in the middle of
//! existing text, fix the leading space and the case of the first letter,
//! and nothing else.
//!
//! The rules are frozen in `packages/contracts/insertion-boundary/` (the
//! README is authoritative). `tests/boundary_conformance.rs` replays the
//! contract's fixture table through this port; the Python oracle
//! (`tests/insertion_boundary.py`) replays the same file.

use serde::Deserialize;

/// The text around the insertion point, as reported by the delivery
/// adapter. No v1 rule reads `after`: the end of the dictated text is
/// never touched.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BoundaryContext<'a> {
    /// Text immediately before the insertion point ("" at a field start).
    pub before: &'a str,
    /// Text immediately after the insertion point.
    pub after: &'a str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BoundaryOptions {
    /// Disables every rule (verbatim modes).
    pub verbatim: bool,
}

/// A rule that fired; the contract's `kind`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BoundaryChange {
    /// One U+0020 prepended.
    LeadingSpace,
    /// The first cased character was lowercased.
    FirstLetterCase,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct BoundaryAdjustment {
    /// The adjusted text; byte-for-byte the input when no rule fired.
    pub text: String,
    /// The rules that fired, in rule order.
    pub changes: Vec<BoundaryChange>,
}

impl BoundaryAdjustment {
    pub fn is_unchanged(&self) -> bool {
        self.changes.is_empty()
    }
}

/// After these, dictated content is starting (no space, case kept).
/// Straight `"` and `'` are ambiguous at an insertion point; the contract
/// pins them as opening.
const OPENING: [char; 13] = [
    '(', '[', '{', '"', '\'', '“', '‘', '„', '«', '「', '『', '【', '（',
];

/// Punctuation that continues a sentence (besides alphanumerics). Sentence
/// enders are deliberately absent: the case is kept after them.
const CONTINUING: [char; 14] = [
    ',', ';', ':', ')', ']', '}', '”', '’', '»', '」', '』', '】', '）', '》',
];

/// Han, kana and CJK punctuation: no space between two of them. Hangul is
/// absent on purpose (Korean separates words with spaces).
const NO_SPACE_SCRIPTS: [(char, char); 9] = [
    ('\u{3000}', '\u{30FF}'), // CJK symbols and punctuation, Hiragana, Katakana
    ('\u{31F0}', '\u{31FF}'), // Katakana phonetic extensions
    ('\u{3400}', '\u{4DBF}'), // CJK Extension A
    ('\u{4E00}', '\u{9FFF}'), // CJK Unified Ideographs
    ('\u{F900}', '\u{FAFF}'), // CJK Compatibility Ideographs
    ('\u{FF01}', '\u{FF0F}'), // fullwidth punctuation
    ('\u{FF1A}', '\u{FF20}'), // fullwidth punctuation
    ('\u{FF5B}', '\u{FF9F}'), // fullwidth punctuation, halfwidth Katakana
    ('\u{20000}', '\u{3FFFF}'), // CJK Extensions B and later
];

fn no_space_boundary(ch: char) -> bool {
    NO_SPACE_SCRIPTS
        .iter()
        .any(|&(low, high)| (low..=high).contains(&ch))
}

pub fn adjust(raw: &str, ctx: &BoundaryContext<'_>, opts: &BoundaryOptions) -> BoundaryAdjustment {
    let mut text = raw.to_string();
    let mut changes = Vec::new();
    if opts.verbatim || raw.is_empty() {
        return BoundaryAdjustment { text, changes };
    }

    if let (Some(last_before), Some(first_raw)) = (ctx.before.chars().last(), raw.chars().next()) {
        if !first_raw.is_whitespace()
            && !last_before.is_whitespace()
            && !OPENING.contains(&last_before)
            && !(no_space_boundary(last_before) && no_space_boundary(first_raw))
        {
            text.insert(0, ' ');
            changes.push(BoundaryChange::LeadingSpace);
        }
    }

    if continues_sentence(ctx.before) && !first_token_is_protected(raw) {
        // Only the first cased character is considered: if it is already
        // lowercase, later capitals are left alone.
        if let Some((index, ch)) = text.char_indices().find(|&(_, ch)| is_cased(ch)) {
            if lowercase_differs(ch) {
                // The full mapping: 'İ' lowercases to two code points.
                let lowered = ch.to_lowercase().to_string();
                text.replace_range(index..index + ch.len_utf8(), &lowered);
                changes.push(BoundaryChange::FirstLetterCase);
            }
        }
    }

    BoundaryAdjustment { text, changes }
}

/// Whether `before` ends mid-sentence: its last non-whitespace character
/// is alphanumeric or continuing punctuation, and the trailing whitespace
/// holds no line break (a new line behaves like a field start).
fn continues_sentence(before: &str) -> bool {
    let trimmed = before.trim_end();
    let Some(last) = trimmed.chars().next_back() else {
        return false;
    };
    !before[trimmed.len()..].contains(['\n', '\r'])
        && (last.is_alphanumeric() || CONTINUING.contains(&last))
}

/// Code-looking first tokens keep their case: paths, URLs, emails,
/// hashtags, snake_case, camel humps, ALL-CAPS (two or more letters), and
/// the pronoun `I` (also `I,` and `I'm`).
fn first_token_is_protected(raw: &str) -> bool {
    let Some(token) = raw.split_whitespace().next() else {
        return false;
    };
    if token.contains(['_', '/', '\\', '@', '#']) || token.starts_with("www.") {
        return true;
    }
    let mut chars = token.chars();
    if chars.next() == Some('I') && !chars.next().is_some_and(char::is_alphanumeric) {
        return true;
    }
    let has_hump = token
        .chars()
        .zip(token.chars().skip(1))
        .any(|(a, b)| a.is_lowercase() && b.is_uppercase());
    let mut letters = token.chars().filter(|c| c.is_alphabetic());
    let all_caps = letters.clone().count() >= 2 && letters.all(char::is_uppercase);
    has_hump || all_caps
}

fn lowercase_differs(ch: char) -> bool {
    !ch.to_lowercase().eq([ch])
}

/// Python's notion of cased: some case mapping changes the character.
fn is_cased(ch: char) -> bool {
    lowercase_differs(ch) || !ch.to_uppercase().eq([ch])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn adjust_after(before: &str, raw: &str) -> BoundaryAdjustment {
        adjust(
            raw,
            &BoundaryContext { before, after: "" },
            &BoundaryOptions::default(),
        )
    }

    #[test]
    fn mid_sentence_adds_space_and_lowercases() {
        let adj = adjust_after("The quick brown", "Fox jumps");
        assert_eq!(adj.text, " fox jumps");
        assert_eq!(
            adj.changes,
            [
                BoundaryChange::LeadingSpace,
                BoundaryChange::FirstLetterCase
            ]
        );
    }

    #[test]
    fn protected_tokens_keep_their_case() {
        for raw in [
            "MyClass here",
            "MY_CONST value",
            "https://Example.com",
            "I said",
        ] {
            let adj = adjust_after("Use", raw);
            assert_eq!(adj.text, format!(" {raw}"));
            assert_eq!(adj.changes, [BoundaryChange::LeadingSpace]);
        }
    }

    #[test]
    fn multi_character_lowercase_mapping_is_applied_in_full() {
        assert_eq!(adjust_after("text", "İstanbul").text, " i̇stanbul");
    }
}
