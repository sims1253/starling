//! Insertion-boundary formatting (#341): the deterministic delivery-time
//! step that fixes the boundary when dictated text lands in the middle of
//! existing text — a leading space and the case of the first letter, and
//! nothing else.
//!
//! The rules are frozen in `packages/contracts/insertion-boundary/` (README
//! is authoritative); the fixture table there is replayed by the Python
//! oracle (`tests/insertion_boundary.py`) and by this crate's conformance
//! test, so neither implementation can drift from the contract.
//!
//! Boundary rules only: words are never changed, the end of the dictated
//! text is never touched, code-looking first tokens keep their case, and a
//! verbatim mode disables every rule. The caller (the runtime's delivery
//! machine) records the adjustment as a revision — the raw recognition
//! text stays unchanged and one action away.
//!
//! Security: the surrounding text is read by adapters only where the
//! platform exposes it without extra permissions, never for secure or
//! incognito fields (an adapter-side duty), used for this decision only,
//! and neither stored nor sent to any processing provider.

/// The text around the insertion point, as reported by the delivery
/// adapter. `after` is carried for validation and future rules; no v1 rule
/// touches the end of the dictated text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BoundaryContext<'a> {
    /// Text immediately before the insertion point ("" at a field start).
    pub before: &'a str,
    /// Text immediately after the insertion point.
    pub after: &'a str,
}

/// Boundary options. `verbatim` disables every rule (verbatim modes).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BoundaryOptions {
    pub verbatim: bool,
}

impl BoundaryOptions {
    pub const fn new() -> Self {
        BoundaryOptions { verbatim: false }
    }
}

/// Which boundary rule fired.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryChangeKind {
    /// One U+0020 prepended.
    LeadingSpace,
    /// The first cased character was lowercased.
    FirstLetterCase,
}

impl BoundaryChangeKind {
    /// The contract's wire name (`boundary.schema.json`'s `kind` enum).
    pub const fn as_str(self) -> &'static str {
        match self {
            BoundaryChangeKind::LeadingSpace => "leading_space",
            BoundaryChangeKind::FirstLetterCase => "first_letter_case",
        }
    }
}

/// One fired rule, with the mechanical reason (mirrored in the fixtures).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundaryChange {
    pub kind: BoundaryChangeKind,
    pub detail: String,
}

/// The result of the boundary step.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct BoundaryAdjustment {
    /// The adjusted text; byte-for-byte the input when no rule fired.
    pub text: String,
    /// The changes that fired, in rule order.
    pub changes: Vec<BoundaryChange>,
}

impl BoundaryAdjustment {
    /// True when nothing fired (the common case at a field start).
    pub fn is_unchanged(&self) -> bool {
        self.changes.is_empty()
    }
}

/// Characters after which dictated content is *starting* (no space, case
/// kept). Straight `"` and `'` are ambiguous at an insertion point; the
/// contract pins them as opening.
const OPENING: [char; 12] = [
    '(', '[', '{', '"', '\'', '“', '‘', '„', '「', '『', '【', '（',
];

/// Mid-sentence punctuation (continues the sentence) beyond alphanumerics.
/// Sentence enders (`. ! ? … 。 ！ ？`) are deliberately absent: the case is
/// kept as recognized after them.
const CONTINUING: [char; 14] = [
    ',', ';', ':', ')', ']', '}', '”', '’', '»', '」', '』', '】', '）', '》',
];

/// Apply the frozen rules. Never more than the boundary.
pub fn adjust(raw: &str, ctx: &BoundaryContext<'_>, opts: &BoundaryOptions) -> BoundaryAdjustment {
    let mut changes = Vec::new();
    if opts.verbatim || raw.is_empty() {
        return BoundaryAdjustment {
            text: raw.to_string(),
            changes,
        };
    }

    let mut text = raw.to_string();

    // Rule 2: leading space.
    if let (Some(last_before), Some(first_raw)) = (ctx.before.chars().last(), text.chars().next()) {
        if !first_raw.is_whitespace()
            && !last_before.is_whitespace()
            && !OPENING.contains(&last_before)
        {
            text.insert(0, ' ');
            changes.push(BoundaryChange {
                kind: BoundaryChangeKind::LeadingSpace,
                detail: format!(
                    "previous character {last_before:?} is not whitespace or an opening \
                     bracket/quote"
                ),
            });
        }
    }

    // Rule 3: first-letter case.
    if continues_sentence(ctx.before) && !first_token_is_protected(&text) {
        if let Some(index) = first_uppercase_char_index(&text) {
            let lowered: String = text
                .chars()
                .enumerate()
                .map(|(i, ch)| {
                    if i == index {
                        ch.to_lowercase().next().unwrap_or(ch)
                    } else {
                        ch
                    }
                })
                .collect();
            let last = last_non_whitespace(ctx.before).expect("continues_sentence implies one");
            text = lowered;
            changes.push(BoundaryChange {
                kind: BoundaryChangeKind::FirstLetterCase,
                detail: format!("previous non-whitespace {last:?} continues the sentence"),
            });
        }
    }

    BoundaryAdjustment { text, changes }
}

/// The trailing whitespace run of `before` must contain no newline, and the
/// last non-whitespace character must be mid-sentence (alphanumeric or
/// continuing punctuation). A newline boundary behaves like the start of a
/// field: the case is kept as recognized.
fn continues_sentence(before: &str) -> bool {
    let trimmed = before.trim_end();
    let Some(last) = trimmed.chars().next_back() else {
        return false;
    };
    // A newline in the trailing whitespace run behaves like the start of a
    // field: the case is kept as recognized.
    if before[trimmed.len()..].contains(['\n', '\r']) {
        return false;
    }
    last.is_alphanumeric() || CONTINUING.contains(&last)
}

fn last_non_whitespace(before: &str) -> Option<char> {
    before.trim_end().chars().next_back()
}

/// The first whitespace-delimited token.
fn first_token(raw: &str) -> &str {
    raw.split_whitespace().next().unwrap_or("")
}

/// Protected first tokens (no case change; the space rule still applies):
/// paths, URLs, emails, hashtags, snake_case (`_ / \ @ #`, `://`, `www.`);
/// camel humps (a lowercase letter immediately followed by an uppercase
/// one); ALL-CAPS tokens (at least two alphabetic characters); and the
/// English pronoun `I`.
fn first_token_is_protected(raw: &str) -> bool {
    let token = first_token(raw);
    if token.is_empty() {
        return false;
    }
    if token.contains(['_', '/', '\\', '@', '#']) {
        return true;
    }
    if token.contains("://") || token.starts_with("www.") {
        return true;
    }
    let chars: Vec<char> = token.chars().collect();
    let has_hump = chars
        .windows(2)
        .any(|pair| pair[0].is_lowercase() && pair[1].is_uppercase());
    if has_hump {
        return true;
    }
    let alphabetic: Vec<char> = chars
        .iter()
        .copied()
        .filter(|c| c.is_alphabetic())
        .collect();
    if alphabetic.len() >= 2 && alphabetic.iter().all(|c| c.is_uppercase()) {
        return true;
    }
    token == "I"
}

/// Char-index of the first alphabetic character whose lowercase differs
/// (the first cased character, in the contract's wording).
fn first_uppercase_char_index(text: &str) -> Option<usize> {
    text.char_indices()
        .position(|(_, ch)| ch.is_alphabetic() && ch.to_lowercase().next() != Some(ch))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx<'a>(before: &'a str, after: &'a str) -> BoundaryContext<'a> {
        BoundaryContext { before, after }
    }

    #[test]
    fn field_start_is_untouched() {
        let adj = adjust("Hello", &ctx("", " world"), &BoundaryOptions::new());
        assert_eq!(adj.text, "Hello");
        assert!(adj.is_unchanged());
    }

    #[test]
    fn mid_sentence_adds_space_and_lowercases() {
        let adj = adjust(
            "Fox jumps",
            &ctx("The quick brown", ""),
            &BoundaryOptions::new(),
        );
        assert_eq!(adj.text, " fox jumps");
        assert_eq!(adj.changes.len(), 2);
        assert_eq!(adj.changes[0].kind.as_str(), "leading_space");
        assert_eq!(adj.changes[1].kind.as_str(), "first_letter_case");
    }

    #[test]
    fn verbatim_disables_everything() {
        let opts = BoundaryOptions { verbatim: true };
        let adj = adjust("Next", &ctx("Done.", ""), &opts);
        assert_eq!(adj.text, "Next");
        assert!(adj.is_unchanged());
    }

    #[test]
    fn protected_tokens_keep_their_case() {
        for raw in [
            "MyClass here",
            "MY_CONST value",
            "https://Example.com",
            "I said",
        ] {
            let adj = adjust(raw, &ctx("Use", ""), &BoundaryOptions::new());
            assert!(adj.text.ends_with(raw), "{raw}: case was changed");
            assert_eq!(adj.changes.len(), 1, "{raw}");
            assert_eq!(adj.changes[0].kind, BoundaryChangeKind::LeadingSpace);
        }
    }

    #[test]
    fn the_tail_is_never_touched() {
        let adj = adjust("this", &ctx("Note", ", please"), &BoundaryOptions::new());
        assert_eq!(adj.text, " this");
        assert_eq!(adj.changes.len(), 1);
    }
}
