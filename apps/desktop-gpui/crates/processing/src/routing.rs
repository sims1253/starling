//! A line-faithful Rust port of the routing oracle `tests/mode_routing.py`
//! (`validate_config`, `prefix_match`, `resolve`, `conflicts_for`), so the
//! desktop app routes a leading spoken phrase through the same rules
//! instead of a second router. The Python file is the contract;
//! `tests/routing_conformance.rs` replays the fixtures both read.
//!
//! Precedence (as encoded by `resolve`): policy blocks > a locked manual
//! choice > the literal escape > an unambiguous longest leading alias >
//! the scoped rule > the default profile; ties across different modes are
//! `needs_resolution`, never a guess. The routed payload is a view of the
//! raw text with the removed prefix span recorded in Unicode code points.

use std::collections::HashSet;
use std::fmt;

use crate::contract::{Delivery, ProfilesDocument, Rule, SelectedText};

/// Whitespace as the oracle's `re` patterns and `str.strip()` treat it:
/// Python's `str.isspace()`, which is [`char::is_whitespace`] plus the
/// information separators U+001C..U+001F.
fn is_ws(c: char) -> bool {
    matches!(c, '\x1c'..='\x1f') || c.is_whitespace()
}

fn skip_ws(text: &str) -> &str {
    text.trim_start_matches(is_ws)
}

/// The oracle's `phrase.strip().split()`.
fn words(phrase: &str) -> impl Iterator<Item = &str> {
    phrase.split(is_ws).filter(|word| !word.is_empty())
}

/// A character's class under Python's `re.IGNORECASE`: the simple
/// (single-character) lowercase, merged by sre's extra equivalences
/// (`_equivalences` in Python's regex compiler).
fn fold(c: char) -> char {
    let mut lower = c.to_lowercase();
    let lower = match (lower.next(), lower.next()) {
        (Some(single), None) => single,
        // U+0130 is the only character with a multi-character lowercase;
        // its simple lowercase is 'i'.
        _ => 'i',
    };
    match lower {
        '\u{131}' => 'i',
        '\u{17f}' => 's',
        '\u{b5}' => '\u{3bc}',
        '\u{345}' | '\u{1fbe}' => '\u{3b9}',
        '\u{1fd3}' => '\u{390}',
        '\u{1fe3}' => '\u{3b0}',
        '\u{3d0}' => '\u{3b2}',
        '\u{3f5}' => '\u{3b5}',
        '\u{3d1}' => '\u{3b8}',
        '\u{3f0}' => '\u{3ba}',
        '\u{3d6}' => '\u{3c0}',
        '\u{3f1}' => '\u{3c1}',
        '\u{3c2}' => '\u{3c3}',
        '\u{3d5}' => '\u{3c6}',
        '\u{1e9b}' => '\u{1e61}',
        '\u{fb05}' => '\u{fb06}',
        '\u{1c80}' => '\u{432}',
        '\u{1c81}' => '\u{434}',
        '\u{1c82}' => '\u{43e}',
        '\u{1c83}' => '\u{441}',
        '\u{1c84}' | '\u{1c85}' => '\u{442}',
        '\u{1c86}' => '\u{44a}',
        '\u{1c87}' => '\u{463}',
        '\u{1c88}' => '\u{a64b}',
        other => other,
    }
}

/// Strips `word` from the front of `text`, case-insensitively.
fn strip_word_ci<'a>(text: &'a str, word: &str) -> Option<&'a str> {
    let mut rest = text.chars();
    for expected in word.chars() {
        if fold(rest.next()?) != fold(expected) {
            return None;
        }
    }
    Some(rest.as_str())
}

/// The oracle's `prefix_match`: a leading phrase match, never quoted or
/// mid-sentence. Returns the byte offset of `match.end()`.
fn prefix_match(text: &str, phrase: &str) -> Option<usize> {
    let mut rest = skip_ws(text);
    for (index, word) in words(phrase).enumerate() {
        if index > 0 {
            // `\s+` between words.
            let after = skip_ws(rest);
            if after.len() == rest.len() {
                return None;
            }
            rest = after;
        }
        rest = strip_word_ci(rest, word)?;
    }
    // `(?=$|[\s:])`: the phrase ends at a token boundary.
    if rest.starts_with(|c: char| !is_ws(c) && c != ':') {
        return None;
    }
    // `(?:\s*:\s*|\s+)?`: one separator is consumed with the match.
    let tail = match skip_ws(rest).strip_prefix(':') {
        Some(after) => skip_ws(after),
        None => skip_ws(rest),
    };
    Some(text.len() - tail.len())
}

/// The oracle's `validate_config`.
pub fn validate_config(doc: &ProfilesDocument) -> Result<(), String> {
    let ids: Vec<&str> = doc.profiles.iter().map(|p| p.id.as_str()).collect();
    if distinct(ids.iter().copied()) != ids.len() || !ids.contains(&doc.default_profile.as_str()) {
        return Err("Unique profiles and a valid default are required".to_string());
    }
    if !ids.contains(&"verbatim") {
        return Err("The literal escape requires a verbatim profile".to_string());
    }
    if doc
        .rules
        .iter()
        .any(|rule| !ids.contains(&rule.profile_id.as_str()))
    {
        return Err("Rule references an unknown profile".to_string());
    }
    if distinct(doc.rules.iter().map(|r| r.id.as_str())) != doc.rules.len() {
        return Err("Duplicate rule ID".to_string());
    }
    for profile in &doc.profiles {
        if profile.local_only
            && (profile.asr_route.starts_with("remote-")
                || profile
                    .authoring_route
                    .as_deref()
                    .is_some_and(|route| route.starts_with("remote-")))
        {
            return Err("A local-only profile cannot use a remote route".to_string());
        }
        if profile.selection_required && profile.selected_text == SelectedText::Off {
            return Err("Required selection cannot be disabled".to_string());
        }
        if profile.delivery == Delivery::ReplaceSelection
            && profile.selected_text != SelectedText::EditTarget
        {
            return Err("Replacement requires edit-target authority".to_string());
        }
    }
    Ok(())
}

fn distinct<'a>(ids: impl Iterator<Item = &'a str>) -> usize {
    ids.collect::<HashSet<_>>().len()
}

/// One routing request (the oracle's `request` dict).
#[derive(Clone, Debug, Default)]
pub struct RouteRequest {
    pub raw_text: String,
    pub manual_mode: Option<String>,
    pub manual_locked: bool,
    pub session_allows_aliases: bool,
    pub secure_field: bool,
    pub project_id: Option<String>,
    pub site: Option<String>,
    pub app_id: Option<String>,
    pub selection_available: bool,
    pub selection_granted: bool,
}

impl RouteRequest {
    /// A request with the oracle's defaults: a locked manual choice, a
    /// session that allows aliases.
    pub fn new(raw_text: impl Into<String>) -> RouteRequest {
        RouteRequest {
            raw_text: raw_text.into(),
            manual_locked: true,
            session_allows_aliases: true,
            ..RouteRequest::default()
        }
    }
}

/// Why a take routed where it did; displays as the oracle's `source`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Source {
    Manual,
    Default,
    EscapeLiteral,
    Phrase(String),
    Rule(String),
    RuleConflict,
    PhraseConflict,
    PolicySecureField,
}

impl fmt::Display for Source {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Source::Manual => f.write_str("manual"),
            Source::Default => f.write_str("default"),
            Source::EscapeLiteral => f.write_str("escape:literal"),
            Source::Phrase(alias) => write!(f, "phrase:{alias}"),
            Source::Rule(id) => write!(f, "rule:{id}"),
            Source::RuleConflict => f.write_str("rule_conflict"),
            Source::PhraseConflict => f.write_str("phrase_conflict"),
            Source::PolicySecureField => f.write_str("policy:secure-field"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Ready,
    NeedsInput,
    Blocked,
    NeedsResolution,
}

impl Status {
    pub fn as_str(self) -> &'static str {
        match self {
            Status::Ready => "ready",
            Status::NeedsInput => "needs_input",
            Status::Blocked => "blocked",
            Status::NeedsResolution => "needs_resolution",
        }
    }
}

/// A tied candidate of a `needs_resolution` result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Candidate {
    pub mode_id: String,
    /// The rule id or alias it came via.
    pub via: String,
}

/// The routing fields of `resolve`'s answer. Blocked and
/// needs_resolution results carry no mode.
#[derive(Clone, Debug)]
pub struct Resolution {
    pub mode: Option<String>,
    pub source: Source,
    pub payload: String,
    pub status: Status,
    pub prefix_span_codepoints: Option<(usize, usize)>,
}

/// A rule's (specificity, priority) when every scope key it carries
/// matches the request; a rule without scope keys never applies.
fn rule_rank(rule: &Rule, request: &RouteRequest) -> Option<(u8, i64)> {
    let mut specificity = None;
    for (rank, want, got) in [
        (3, &rule.project_id, &request.project_id),
        (2, &rule.site, &request.site),
        (1, &rule.app_id, &request.app_id),
    ] {
        let Some(want) = want else { continue };
        if got.as_ref() != Some(want) {
            return None;
        }
        specificity = specificity.or(Some(rank));
    }
    specificity.map(|specificity| (specificity, rule.priority))
}

/// Equal-best-rank rules, sorted by id (the oracle's `_rule_winners`).
fn rule_winners<'a>(doc: &'a ProfilesDocument, request: &RouteRequest) -> Vec<&'a Rule> {
    let ranked: Vec<_> = doc
        .rules
        .iter()
        .filter_map(|rule| Some((rule_rank(rule, request)?, rule)))
        .collect();
    let Some(best) = ranked.iter().map(|(rank, _)| *rank).max() else {
        return Vec::new();
    };
    let mut winners: Vec<&Rule> = ranked
        .into_iter()
        .filter(|(rank, _)| *rank == best)
        .map(|(_, rule)| rule)
        .collect();
    winners.sort_by(|a, b| a.id.cmp(&b.id));
    winners
}

struct PhraseMatch<'a> {
    alias: &'a str,
    mode_id: &'a str,
    /// Byte offset where the payload view starts.
    end: usize,
}

/// Equal-best-rank leading-alias matches (most words, then most
/// characters), sorted by (alias, mode) (the oracle's `_phrase_winners`).
fn phrase_winners<'a>(doc: &'a ProfilesDocument, raw: &str) -> Vec<PhraseMatch<'a>> {
    let mut matches = Vec::new();
    for profile in &doc.profiles {
        for alias in &profile.aliases {
            if let Some(end) = prefix_match(raw, alias) {
                let rank = (
                    words(alias).count(),
                    alias.trim_matches(is_ws).chars().count(),
                );
                let found = PhraseMatch {
                    alias,
                    mode_id: &profile.id,
                    end,
                };
                matches.push((rank, found));
            }
        }
    }
    let Some(best) = matches.iter().map(|(rank, _)| *rank).max() else {
        return Vec::new();
    };
    let mut winners: Vec<PhraseMatch> = matches
        .into_iter()
        .filter(|(rank, _)| *rank == best)
        .map(|(_, found)| found)
        .collect();
    winners.sort_by(|a, b| (a.alias, a.mode_id).cmp(&(b.alias, b.mode_id)));
    winners
}

/// The oracle's `resolve`. Routing is frozen: this port must never grow
/// semantics the Python oracle does not have.
pub fn resolve(doc: &ProfilesDocument, request: &RouteRequest) -> Result<Resolution, String> {
    validate_config(doc)?;
    let raw = request.raw_text.as_str();
    let unresolved = |source, status| Resolution {
        mode: None,
        source,
        payload: raw.to_string(),
        status,
        prefix_span_codepoints: None,
    };
    if request.secure_field {
        return Ok(unresolved(Source::PolicySecureField, Status::Blocked));
    }
    let profile = |id: &str| doc.profile(id).expect("validate_config checked the id");
    let (mut chosen, mut source) = match request.manual_mode.as_deref() {
        Some(manual) if doc.profile(manual).is_none() => {
            return Err("Unknown manual mode".to_string());
        }
        Some(manual) => (manual, Source::Manual),
        None => {
            let winners = rule_winners(doc, request);
            if distinct(winners.iter().map(|rule| rule.profile_id.as_str())) > 1 {
                return Ok(unresolved(Source::RuleConflict, Status::NeedsResolution));
            }
            match winners.first() {
                Some(rule) => (rule.profile_id.as_str(), Source::Rule(rule.id.clone())),
                None => (doc.default_profile.as_str(), Source::Default),
            }
        }
    };
    let mut prefix_end = None;
    let locked = request.manual_mode.is_some() && request.manual_locked;
    if !locked && profile(chosen).allow_spoken_overrides && request.session_allows_aliases {
        if let Some(end) = prefix_match(raw, "literal") {
            (chosen, source, prefix_end) = ("verbatim", Source::EscapeLiteral, Some(end));
        } else {
            let winners = phrase_winners(doc, raw);
            if distinct(winners.iter().map(|found| found.mode_id)) > 1 {
                return Ok(unresolved(Source::PhraseConflict, Status::NeedsResolution));
            }
            if let Some(found) = winners.first() {
                chosen = found.mode_id;
                source = Source::Phrase(found.alias.to_string());
                prefix_end = Some(found.end);
            }
        }
    }
    let payload = &raw[prefix_end.unwrap_or(0)..];
    let profile = profile(chosen);
    let needs_selection =
        profile.selection_required && !(request.selection_available && request.selection_granted);
    let status = if payload.trim_matches(is_ws).is_empty() || needs_selection {
        Status::NeedsInput
    } else {
        Status::Ready
    };
    Ok(Resolution {
        mode: Some(chosen.to_string()),
        source,
        payload: payload.to_string(),
        status,
        prefix_span_codepoints: prefix_end.map(|end| (0, raw[..end].chars().count())),
    })
}

/// The oracle's `conflicts_for`: the tied candidates of a
/// `needs_resolution` result (empty for every other result).
pub fn conflicts_for(
    doc: &ProfilesDocument,
    request: &RouteRequest,
    result: &Resolution,
) -> Vec<Candidate> {
    if result.status != Status::NeedsResolution {
        return Vec::new();
    }
    let candidates: Vec<Candidate> = match result.source {
        Source::RuleConflict => rule_winners(doc, request)
            .into_iter()
            .map(|rule| Candidate {
                mode_id: rule.profile_id.clone(),
                via: rule.id.clone(),
            })
            .collect(),
        Source::PhraseConflict => phrase_winners(doc, &request.raw_text)
            .into_iter()
            .map(|found| Candidate {
                mode_id: found.mode_id.to_string(),
                via: found.alias.to_string(),
            })
            .collect(),
        _ => return Vec::new(),
    };
    if distinct(candidates.iter().map(|c| c.mode_id.as_str())) > 1 {
        candidates
    } else {
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_prefix_grammar_is_whole_token_and_boundary_aware() {
        // Leading, case-insensitive, separator consumed.
        assert_eq!(prefix_match("Code this: add logs", "code this"), Some(11));
        assert_eq!(prefix_match("  code this add logs", "code this"), Some(12));
        assert_eq!(prefix_match("code this", "code this"), Some(9));
        // Python's re.IGNORECASE classes: dotless ı, long ſ, Kelvin K, İ.
        assert_eq!(prefix_match("İ thıſ \u{212a}", "i this k"), Some(13));
        // Whole token: a longer word is not the alias.
        assert_eq!(prefix_match("code thisway stays", "code this"), None);
        // A comma directly after the alias is not a permitted separator.
        assert_eq!(prefix_match("code this, add logs", "code this"), None);
        // Quoted or mid-sentence mentions never match.
        assert_eq!(prefix_match(r#""code this" now"#, "code this"), None);
        assert_eq!(prefix_match(r#"say "code this" now"#, "code this"), None);
        assert_eq!(prefix_match("please code this now", "code this"), None);
    }
}
