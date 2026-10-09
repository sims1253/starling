//! The trailing spoken instruction grammar, ported from the oracle
//! `tests/spoken_instructions.py`. The delimiter table is the contract
//! file `packages/contracts/mode-routing/spoken-instructions.json`,
//! compiled in.
//!
//! A delimiter spoken near the end of a finalized take splits it: the
//! text after it is the instruction (it travels as the transform
//! request's `instruction`, never as input text), the text before it is
//! the payload. Tokens and cores are the deterministic step's
//! ([`crate::transforms`]), which is also why a quoted mention never
//! matches: the opening quote stays attached to the token.

use std::sync::OnceLock;

use serde::Deserialize;

use crate::transforms::{is_ws, literal_word, token_core, token_spans};

const TABLE_JSON: &str =
    include_str!("../../../../../packages/contracts/mode-routing/spoken-instructions.json");

#[derive(Debug, Deserialize)]
struct Delimiter {
    match_tokens: Vec<String>,
    window_words: usize,
}

#[derive(Debug, Deserialize)]
struct Table {
    delimiter: Delimiter,
}

fn delimiter() -> &'static Delimiter {
    static TABLE: OnceLock<Table> = OnceLock::new();
    &TABLE
        .get_or_init(|| serde_json::from_str(TABLE_JSON).expect("spoken-instructions.json parses"))
        .delimiter
}

fn is_delimiter(token: &str) -> bool {
    let core = token_core(token);
    delimiter().match_tokens.iter().any(|token| *token == core)
}

/// The split of one finalized take; spans are Unicode code points. When
/// nothing fired the payload is the whole text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Split {
    pub matched: bool,
    pub payload: String,
    pub instruction: String,
    pub delimiter_span: Option<(usize, usize)>,
    pub instruction_span: Option<(usize, usize)>,
}

/// Splits a finalized take at the last delimiter within the last
/// `window_words` tokens that is not escaped by the literal word and is
/// followed by an instruction with a letter or digit.
pub fn split(text: &str, language: Option<&str>) -> Split {
    let spans = token_spans(text);
    let literal = literal_word(language);
    let floor = spans.len().saturating_sub(delimiter().window_words);
    let found = (floor..spans.len()).rev().find(|&i| {
        let (start, end) = spans[i];
        is_delimiter(&text[start..end])
            && !(i > 0 && token_core(&text[spans[i - 1].0..spans[i - 1].1]) == literal)
            && text[end..].chars().any(char::is_alphanumeric)
    });
    let Some(i) = found else {
        return Split {
            matched: false,
            payload: text.to_string(),
            instruction: String::new(),
            delimiter_span: None,
            instruction_span: None,
        };
    };
    let (start, end) = spans[i];
    let char_start = text[..start].chars().count();
    let char_end = char_start + text[start..end].chars().count();
    Split {
        matched: true,
        payload: text[..start].to_string(),
        instruction: text[end..].trim_matches(is_ws).to_string(),
        delimiter_span: Some((char_start, char_end)),
        instruction_span: Some((char_end, text.chars().count())),
    }
}

/// Removes a leading delimiter token from an instruction region's text:
/// the draft's command region carries the delimiter, the model sees only
/// the instruction.
pub fn strip_delimiter(instruction: &str) -> String {
    match token_spans(instruction).first() {
        Some(&(start, end)) if is_delimiter(&instruction[start..end]) => {
            instruction[end..].trim_matches(is_ws).to_string()
        }
        _ => instruction.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stripping_only_removes_the_leading_delimiter() {
        assert_eq!(
            strip_delimiter("Starling, make it formal"),
            "make it formal"
        );
        assert_eq!(strip_delimiter("Sterling translate this"), "translate this");
        assert_eq!(strip_delimiter("make it formal"), "make it formal");
    }

    #[test]
    fn a_quoted_mention_never_matches_because_the_core_changes() {
        assert!(!split(r#"The wake word "Starling" ends this take now"#, Some("en")).matched);
    }
}
