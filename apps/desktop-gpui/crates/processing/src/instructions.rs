//! The trailing spoken instruction grammar (#298), ported from the
//! executable oracle `tests/spoken_instructions.py`. The delimiter
//! configuration is the contract file
//! `packages/contracts/mode-routing/spoken-instructions.json`, compiled
//! in so both implementations read one table.
//!
//! A delimiter phrase spoken near the END of a *finalized* take splits
//! it: everything after the delimiter is the instruction (it travels as
//! the transform request's [`crate::contract::TransformRequest`]
//! `instruction`, never as input text), everything before it is the
//! payload. Tokens and token cores are exactly the deterministic step's
//! ([`crate::transforms`]), so both grammars split the same way.
//!
//! Nothing here interprets the transcript beyond the closed rules in
//! the contract file: the last-window rule, last-occurrence-wins, the
//! literal escape, the whole-token core (which is also why a quoted
//! mention never matches — an opening quote stays attached to the
//! token) and the non-empty instruction tail.

use std::sync::OnceLock;

use serde::Deserialize;

use crate::transforms::{is_ws, literal_word, token_core};

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

fn table() -> &'static Table {
    static TABLE: OnceLock<Table> = OnceLock::new();
    TABLE.get_or_init(|| serde_json::from_str(TABLE_JSON).expect("spoken-instructions.json parses"))
}

/// One whitespace-separated token: byte span, code-point span and core.
struct Token {
    byte_start: usize,
    byte_end: usize,
    char_start: usize,
    char_end: usize,
    core: String,
}

fn tokens(text: &str) -> Vec<Token> {
    let mut out = Vec::new();
    let mut char_pos = 0;
    let mut start: Option<(usize, usize)> = None;
    for (index, c) in text.char_indices() {
        if is_ws(c) {
            if let Some((byte_start, char_start)) = start.take() {
                out.push(make_token(text, byte_start, index, char_start, char_pos));
            }
        } else if start.is_none() {
            start = Some((index, char_pos));
        }
        char_pos += 1;
    }
    if let Some((byte_start, char_start)) = start {
        out.push(make_token(text, byte_start, text.len(), char_start, char_pos));
    }
    out
}

fn make_token(
    text: &str,
    byte_start: usize,
    byte_end: usize,
    char_start: usize,
    char_end: usize,
) -> Token {
    Token {
        byte_start,
        byte_end,
        char_start,
        char_end,
        core: token_core(&text[byte_start..byte_end]),
    }
}

fn trim_ws(text: &str) -> &str {
    text.trim_matches(is_ws)
}

fn is_match_token(core: &str) -> bool {
    table().delimiter.match_tokens.iter().any(|token| token == core)
}

/// The split of one finalized take. Offsets are Unicode code points,
/// like every other span in this contract; `matched == false` means no
/// delimiter fired and the payload is the whole text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Split {
    pub matched: bool,
    pub payload: String,
    pub instruction: String,
    pub delimiter_span: Option<(usize, usize)>,
    pub instruction_span: Option<(usize, usize)>,
}

/// Splits a FINALIZED take at the last delimiter occurrence inside the
/// configured window. See the module docs for the closed rules.
pub fn split(text: &str, language: Option<&str>) -> Split {
    let no_match = || Split {
        matched: false,
        payload: text.to_string(),
        instruction: String::new(),
        delimiter_span: None,
        instruction_span: None,
    };
    let tokens = tokens(text);
    let literal = literal_word(language);
    let window = table().delimiter.window_words;
    let floor = tokens.len().saturating_sub(window);
    let mut found = None;
    for (i, token) in tokens.iter().enumerate() {
        if i < floor {
            continue; // outside the last window words: mid-take text
        }
        if !is_match_token(&token.core) {
            continue;
        }
        if i > 0 && tokens[i - 1].core == literal {
            continue; // dictated as text, like every literal escape
        }
        let tail = trim_ws(&text[token.byte_end..]);
        if !tail.chars().any(|c| c.is_alphanumeric()) {
            continue; // no instruction after it: not an occurrence
        }
        found = Some(i);
    }
    let Some(index) = found else {
        return no_match();
    };
    let token = &tokens[index];
    let total_chars = text.chars().count();
    Split {
        matched: true,
        payload: text[..token.byte_start].to_string(),
        instruction: trim_ws(&text[token.byte_end..]).to_string(),
        delimiter_span: Some((token.char_start, token.char_end)),
        instruction_span: Some((token.char_end, total_chars)),
    }
}

/// Removes a leading delimiter token from an instruction region's text:
/// the draft's command region carries the delimiter (and everything
/// after it), the request the model receives carries only the
/// instruction. A no-op when the text does not start with a delimiter.
pub fn strip_delimiter(instruction: &str) -> String {
    let tokens = tokens(instruction);
    let Some(token) = tokens.first() else {
        return instruction.to_string();
    };
    if is_match_token(&token.core) {
        trim_ws(&instruction[token.byte_end..]).to_string()
    } else {
        instruction.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_table_compiles_in() {
        assert!(table().delimiter.window_words >= 1);
        assert!(!table().delimiter.match_tokens.is_empty());
    }

    #[test]
    fn stripping_only_removes_the_leading_delimiter() {
        assert_eq!(strip_delimiter("Starling, make it formal"), "make it formal");
        assert_eq!(strip_delimiter("Sterling translate this"), "translate this");
        assert_eq!(strip_delimiter("make it formal"), "make it formal");
    }

    #[test]
    fn a_quoted_mention_never_matches_because_the_core_changes() {
        // The opening quote is not recognizer punctuation, so it stays
        // attached to the token and the core is no longer a delimiter.
        assert!(!split(r#"The wake word "Starling" ends this take now"#, Some("en")).matched);
    }
}
