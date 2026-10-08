//! The routing oracle (#298): a line-faithful Rust port of the frozen
//! executable oracle `tests/mode_routing.py` (`validate_config`,
//! `prefix_match`, `resolve`, `conflicts_for`, `explain`), which is the
//! port of the review package's reference router. The Python file is
//! the contract; this module exists so the desktop app routes a leading
//! spoken phrase through the SAME oracle instead of growing a second
//! router. `tests/routing_conformance.rs` replays the routing fixtures
//! both implementations read.
//!
//! Frozen precedence (as encoded by `resolve`): policy blocks > a locked
//! manual choice > the literal escape > an unambiguous longest leading
//! alias > the scoped rule > the default profile; ties across different
//! modes are `needs_resolution`, never a guess. The prefix grammar is
//! leading position after permitted whitespace, whole-token boundaries,
//! a finite separator set (space or colon), case-insensitive; quoted or
//! mid-sentence mentions never match. The routed payload is a view: the
//! raw recognition text is preserved and the removed prefix span is
//! recorded in Unicode code points.

use std::collections::{HashMap, HashSet};

use crate::contract::{Delivery, ModeEntry, ProfilesDocument, Rule, SelectedText};

/// Whitespace exactly as the oracle's `re` patterns treat it: Python's
/// `str.isspace()` set, which is [`char::is_whitespace`] plus the four
/// information separators U+001C..U+001F.
fn is_ws(c: char) -> bool {
    matches!(c, '\x1c'..='\x1f') || c.is_whitespace()
}

fn skip_ws(text: &str) -> &str {
    text.trim_start_matches(is_ws)
}

/// Case-insensitive single-character equality (the oracle's
/// `re.IGNORECASE`). Compared per character: aliases are configuration
/// written by hand, and the fixtures are ASCII.
fn eq_ignore_case(a: char, b: char) -> bool {
    a.to_lowercase().eq(b.to_lowercase())
}

/// Strips `word` from the front of `text`, case-insensitively, or
/// returns `None` when the front does not spell the word.
fn strip_word_ci<'a>(text: &'a str, word: &str) -> Option<&'a str> {
    let mut rest = text;
    let mut word = word;
    loop {
        let mut word_chars = word.chars();
        let Some(expected) = word_chars.next() else {
            return Some(rest);
        };
        let Some(found) = rest.chars().next() else {
            return None;
        };
        if !eq_ignore_case(found, expected) {
            return None;
        }
        rest = &rest[found.len_utf8()..];
        word = &word[expected.len_utf8()..];
    }
}

/// The oracle's `prefix_match`: a leading phrase match, never quoted or
/// mid-sentence. Returns the code-point length of the match (where the
/// payload view starts), mirroring the oracle's `match.end()`.
fn prefix_match(text: &str, phrase: &str) -> Option<usize> {
    prefix_match_bytes(text, phrase).map(|end| text[..end].chars().count())
}

/// [`prefix_match`] as a byte offset, for slicing the payload view.
fn prefix_match_bytes(text: &str, phrase: &str) -> Option<usize> {
    let words: Vec<&str> = phrase.split_whitespace().collect();
    let mut rest = skip_ws(text);
    for (index, word) in words.iter().enumerate() {
        if index > 0 {
            // `\s+`: at least one whitespace character between words.
            let after = skip_ws(rest);
            if after.len() == rest.len() {
                return None;
            }
            rest = after;
        }
        rest = strip_word_ci(rest, word)?;
    }
    // `(?=$|[\s:])`: the phrase must end at a token boundary.
    match rest.chars().next() {
        None => {}
        Some(c) if is_ws(c) || c == ':' => {}
        _ => return None,
    }
    // `(?:\s*:\s*|\s+)?`: one separator (a colon with its spaces, or a
    // whitespace run) is consumed with the match.
    let after_colon = skip_ws(rest).strip_prefix(':');
    let tail = match after_colon {
        Some(after) => skip_ws(after),
        None => skip_ws(rest),
    };
    let byte_end = text.len() - tail.len();
    Some(byte_end)
}

/// The oracle's `validate_config`: the structural rules a profiles
/// document must satisfy before anything routes through it.
pub fn validate_config(doc: &ProfilesDocument) -> Result<(), String> {
    let ids: Vec<&str> = doc.profiles.iter().map(|p| p.id.as_str()).collect();
    let unique = ids.iter().collect::<HashSet<_>>().len() == ids.len();
    if !unique || !ids.contains(&doc.default_profile.as_str()) {
        return Err("Unique profiles and a valid default are required".to_string());
    }
    if !ids.contains(&"verbatim") {
        return Err("The literal escape requires a verbatim profile".to_string());
    }
    for rule in &doc.rules {
        if !ids.contains(&rule.profile_id.as_str()) {
            return Err("Rule references an unknown profile".to_string());
        }
    }
    let rule_ids: Vec<&str> = doc.rules.iter().map(|r| r.id.as_str()).collect();
    if rule_ids.iter().collect::<HashSet<_>>().len() != rule_ids.len() {
        return Err("Duplicate rule ID".to_string());
    }
    for profile in &doc.profiles {
        if profile.local_only
            && (profile.asr_route.starts_with("remote-")
                || profile
                    .authoring_route
                    .as_deref()
                    .unwrap_or_default()
                    .starts_with("remote-"))
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

/// One routing request (the oracle's `request` dict). `manual_locked`
/// and `session_allows_aliases` default to true in the oracle; callers
/// set them explicitly here.
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

/// Why a take routed where it did (the oracle's `source` strings).
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

impl Source {
    /// The oracle's exact spelling, so fixtures compare textually.
    pub fn as_str(&self) -> String {
        match self {
            Source::Manual => "manual".to_string(),
            Source::Default => "default".to_string(),
            Source::EscapeLiteral => "escape:literal".to_string(),
            Source::Phrase(alias) => format!("phrase:{alias}"),
            Source::Rule(id) => format!("rule:{id}"),
            Source::RuleConflict => "rule_conflict".to_string(),
            Source::PhraseConflict => "phrase_conflict".to_string(),
            Source::PolicySecureField => "policy:secure-field".to_string(),
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
    /// The rule id or alias phrase it came via.
    pub via: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConflictKind {
    Rule,
    Phrase,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Conflict {
    pub kind: ConflictKind,
    pub candidates: Vec<Candidate>,
}

/// `resolve`'s answer. The early returns (blocked, needs_resolution)
/// carry no mode and no delivery fields, exactly like the oracle's.
#[derive(Clone, Debug)]
pub struct Resolution {
    pub mode: Option<String>,
    pub source: Source,
    pub payload: String,
    pub status: Status,
    pub raw_text: String,
    pub prefix_span_codepoints: Option<(usize, usize)>,
    pub delivery: Option<Delivery>,
    pub selected_text_role: Option<SelectedText>,
    pub local_only: Option<bool>,
}

impl Resolution {
    fn early(mode: Option<String>, source: Source, raw: &str, status: Status) -> Resolution {
        Resolution {
            mode,
            source,
            payload: raw.to_string(),
            status,
            raw_text: raw.to_string(),
            prefix_span_codepoints: None,
            delivery: None,
            selected_text_role: None,
            local_only: None,
        }
    }
}

fn rule_applies(rule: &Rule, request: &RouteRequest) -> Option<u8> {
    // The oracle: criteria are the scope keys the rule carries; a rule
    // with no scope key never applies, and every key must match.
    let mut specificity = None;
    for (key, value) in [
        ("project_id", &request.project_id),
        ("site", &request.site),
        ("app_id", &request.app_id),
    ] {
        let carries = match key {
            "project_id" => rule.project_id.is_some(),
            "site" => rule.site.is_some(),
            _ => rule.app_id.is_some(),
        };
        if !carries {
            continue;
        }
        let rule_value = match key {
            "project_id" => rule.project_id.as_deref(),
            "site" => rule.site.as_deref(),
            _ => rule.app_id.as_deref(),
        };
        if value.as_deref() != rule_value {
            return None;
        }
        let rank = match key {
            "project_id" => 3,
            "site" => 2,
            _ => 1,
        };
        specificity = Some(specificity.map_or(rank, |seen: u8| seen.max(rank)));
    }
    specificity
}

/// The oracle's `resolve`. Routing itself is frozen; this port must
/// never grow semantics the Python oracle does not have.
pub fn resolve(doc: &ProfilesDocument, request: &RouteRequest) -> Result<Resolution, String> {
    validate_config(doc)?;
    let raw = request.raw_text.clone();
    if request.secure_field {
        return Ok(Resolution::early(
            None,
            Source::PolicySecureField,
            &raw,
            Status::Blocked,
        ));
    }
    let profiles: HashMap<&str, &ModeEntry> =
        doc.profiles.iter().map(|p| (p.id.as_str(), p)).collect();
    if let Some(manual) = &request.manual_mode {
        if !profiles.contains_key(manual.as_str()) {
            return Err("Unknown manual mode".to_string());
        }
    }
    let mut chosen = request
        .manual_mode
        .clone()
        .unwrap_or_else(|| doc.default_profile.clone());
    let mut source = if request.manual_mode.is_some() {
        Source::Manual
    } else {
        Source::Default
    };
    if request.manual_mode.is_none() {
        let mut matched: Vec<((u8, i64), &Rule)> = Vec::new();
        for rule in &doc.rules {
            if let Some(specificity) = rule_applies(rule, request) {
                matched.push(((specificity, rule.priority), rule));
            }
        }
        if !matched.is_empty() {
            let best = matched.iter().map(|(rank, _)| *rank).max().unwrap();
            let mut winners: Vec<&Rule> = matched
                .iter()
                .filter(|(rank, _)| *rank == best)
                .map(|(_, rule)| *rule)
                .collect();
            if winners
                .iter()
                .map(|rule| rule.profile_id.as_str())
                .collect::<HashSet<_>>()
                .len()
                > 1
            {
                return Ok(Resolution::early(
                    None,
                    Source::RuleConflict,
                    &raw,
                    Status::NeedsResolution,
                ));
            }
            winners.sort_by(|a, b| a.id.cmp(&b.id));
            let winner = winners[0];
            chosen = winner.profile_id.clone();
            source = Source::Rule(winner.id.clone());
        }
    }
    let mut payload: &str = &raw;
    let mut prefix_span = None;
    let locked = request.manual_mode.is_some() && request.manual_locked;
    let chosen_profile = *profiles
        .get(chosen.as_str())
        .expect("validate_config checked the chosen id");
    let aliases_allowed = !locked
        && chosen_profile.allow_spoken_overrides
        && request.session_allows_aliases;
    if aliases_allowed {
        // The escape word is the frozen reference's English "literal".
        if let Some(end) = prefix_match_bytes(&raw, "literal") {
            let chars = raw[..end].chars().count();
            chosen = "verbatim".to_string();
            source = Source::EscapeLiteral;
            payload = &raw[end..];
            prefix_span = Some((0, chars));
        } else {
            let mut matches: Vec<(usize, usize, &str, &str, usize)> = Vec::new();
            for profile in &doc.profiles {
                for alias in &profile.aliases {
                    if let Some(end) = prefix_match_bytes(&raw, alias) {
                        matches.push((
                            alias.split_whitespace().count(),
                            alias.chars().count(),
                            alias.as_str(),
                            profile.id.as_str(),
                            end,
                        ));
                    }
                }
            }
            if !matches.is_empty() {
                let rank = matches.iter().map(|m| (m.0, m.1)).max().unwrap();
                let mut winners: Vec<&(usize, usize, &str, &str, usize)> =
                    matches.iter().filter(|m| (m.0, m.1) == rank).collect();
                if winners
                    .iter()
                    .map(|m| m.3)
                    .collect::<HashSet<_>>()
                    .len()
                    > 1
                {
                    return Ok(Resolution::early(
                        None,
                        Source::PhraseConflict,
                        &raw,
                        Status::NeedsResolution,
                    ));
                }
                winners.sort_by(|a, b| (a.2, a.3).cmp(&(b.2, b.3)));
                let winner = winners[0];
                let chars = raw[..winner.4].chars().count();
                chosen = winner.3.to_string();
                source = Source::Phrase(winner.2.to_string());
                payload = &raw[winner.4..];
                prefix_span = Some((0, chars));
            }
        }
    }
    let profile = *profiles
        .get(chosen.as_str())
        .expect("validate_config checked the chosen id");
    let needs_selection = profile.selection_required
        && !(request.selection_available && request.selection_granted);
    let status = if payload.trim().is_empty() || needs_selection {
        Status::NeedsInput
    } else {
        Status::Ready
    };
    Ok(Resolution {
        mode: Some(chosen),
        source,
        payload: payload.to_string(),
        status,
        raw_text: raw.clone(),
        prefix_span_codepoints: prefix_span,
        delivery: Some(profile.delivery),
        selected_text_role: Some(profile.selected_text),
        local_only: Some(profile.local_only),
    })
}

/// The oracle's `conflicts_for`: the tied candidates of a
/// `needs_resolution` result (empty for every other result).
pub fn conflicts_for(
    doc: &ProfilesDocument,
    request: &RouteRequest,
    result: &Resolution,
) -> Vec<Conflict> {
    if result.status != Status::NeedsResolution {
        return Vec::new();
    }
    match &result.source {
        Source::RuleConflict => {
            let mut matched: Vec<((u8, i64), &Rule)> = Vec::new();
            for rule in &doc.rules {
                if let Some(specificity) = rule_applies(rule, request) {
                    matched.push(((specificity, rule.priority), rule));
                }
            }
            if matched.is_empty() {
                return Vec::new();
            }
            let best = matched.iter().map(|(rank, _)| *rank).max().unwrap();
            let mut winners: Vec<&Rule> = matched
                .iter()
                .filter(|(rank, _)| *rank == best)
                .map(|(_, rule)| *rule)
                .collect();
            winners.sort_by(|a, b| a.id.cmp(&b.id));
            if winners
                .iter()
                .map(|rule| rule.profile_id.as_str())
                .collect::<HashSet<_>>()
                .len()
                > 1
            {
                vec![Conflict {
                    kind: ConflictKind::Rule,
                    candidates: winners
                        .iter()
                        .map(|rule| Candidate {
                            mode_id: rule.profile_id.clone(),
                            via: rule.id.clone(),
                        })
                        .collect(),
                }]
            } else {
                Vec::new()
            }
        }
        Source::PhraseConflict => {
            let raw = request.raw_text.as_str();
            let mut matches: Vec<(usize, usize, String, String, usize)> = Vec::new();
            for profile in &doc.profiles {
                for alias in &profile.aliases {
                    if let Some(end) = prefix_match(raw, alias) {
                        matches.push((
                            alias.split_whitespace().count(),
                            alias.chars().count(),
                            alias.clone(),
                            profile.id.clone(),
                            end,
                        ));
                    }
                }
            }
            if matches.is_empty() {
                return Vec::new();
            }
            let rank = matches.iter().map(|m| (m.0, m.1)).max().unwrap();
            let mut winners: Vec<&(usize, usize, String, String, usize)> =
                matches.iter().filter(|m| (m.0, m.1) == rank).collect();
            winners.sort_by(|a, b| (&a.2, &a.3).cmp(&(&b.2, &b.3)));
            if winners.iter().map(|m| m.3.as_str()).collect::<HashSet<_>>().len() > 1 {
                vec![Conflict {
                    kind: ConflictKind::Phrase,
                    candidates: winners
                        .iter()
                        .map(|winner| Candidate {
                            mode_id: winner.3.clone(),
                            via: winner.2.clone(),
                        })
                        .collect(),
                }]
            } else {
                Vec::new()
            }
        }
        _ => Vec::new(),
    }
}

/// The oracle's `explain`: a deterministic, content-free route
/// explanation that names modes, rules and configured aliases only —
/// never payload or selection contents.
pub fn explain(result: &Resolution, doc: Option<&ProfilesDocument>) -> String {
    let source = &result.source;
    let mode = result.mode.as_deref().unwrap_or("none");
    match source {
        Source::Manual => {
            format!("manual mode {mode} for this take; text not parsed for other modes")
        }
        Source::Default => format!("no rule or leading phrase matched; default mode {mode}"),
        Source::Rule(id) => {
            let rule = doc
                .and_then(|doc| doc.rules.iter().find(|rule| rule.id == *id))
                .map(|rule| {
                    if rule.project_id.is_some() {
                        "project rule"
                    } else if rule.site.is_some() {
                        "site rule"
                    } else {
                        "app rule"
                    }
                    .to_string()
                })
                .unwrap_or_else(|| "rule".to_string());
            format!("{rule} {id} selected mode {mode}")
        }
        Source::Phrase(alias) => format!("leading phrase \"{alias}\" selected mode {mode}"),
        Source::EscapeLiteral => "leading literal escape; payload handled verbatim".to_string(),
        Source::PolicySecureField => "secure field: routing suppressed by policy".to_string(),
        Source::RuleConflict => "equal-rank rules conflict; resolution required".to_string(),
        Source::PhraseConflict => {
            "equal-rank leading phrases conflict; resolution required".to_string()
        }
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
        // Whole token: a longer word is not the alias.
        assert_eq!(prefix_match("code thisway stays", "code this"), None);
        // A comma directly after the alias is not a permitted separator.
        assert_eq!(prefix_match("code this, add logs", "code this"), None);
        // Quoted or mid-sentence mentions never match.
        assert_eq!(prefix_match(r#""code this" now"#, "code this"), None);
        assert_eq!(prefix_match(r#"say "code this" now"#, "code this"), None);
        assert_eq!(prefix_match("please code this now", "code this"), None);
    }

    #[test]
    fn spans_are_code_points() {
        // "äh" is two code points, three bytes.
        assert_eq!(prefix_match("äh code this now", "äh"), Some(3));
    }
}
