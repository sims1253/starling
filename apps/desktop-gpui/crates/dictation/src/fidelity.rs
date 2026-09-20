//! Fidelity-sensitive transcript analysis.
//!
//! Faithful port of `packages/dictation/src/fidelity.ts`. The warnings flag
//! spans worth preserving or reviewing; they never guess replacements. The
//! exact warning message strings from the TS source are kept (tests assert
//! them).
//!
//! Documented deviations from the TS source:
//! - `range` on warnings and the `suggestedEdits` passthrough are not ported;
//!   the gpui UI consumes code/severity/message/term only.
//! - Term matching lowercases both sides instead of NFKC-normalizing first.
//!   Full NFKC needs the `unicode-normalization` crate and this task may only
//!   touch this file, so it is skipped; for the ASCII domain terms fed in by
//!   the "Words to watch" setting the result is identical.
//! - The TS list-start regex uses the lookahead `(?=\s)`, which the `regex`
//!   crate does not support, so the character following `[.)]` is verified by
//!   hand in [`first_list_item_number`].

use std::sync::OnceLock;

use regex::Regex;

/// Why a warning was raised. Mirrors the TS `FidelityWarningCode` union.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FidelityWarningCode {
    NonOneListStart,
    PossibleSelfCorrection,
    NegationPresent,
    ExpectedTermMissing,
    ShortAnswer,
    DiscourseWordPreserved,
    PossibleAudioGap,
}

/// How urgently a warning deserves attention.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Severity {
    Info,
    Review,
}

/// One fidelity observation about the raw transcript.
#[derive(Clone, Debug)]
pub struct FidelityWarning {
    pub code: FidelityWarningCode,
    pub severity: Severity,
    pub message: String,
    /// Set only for [`FidelityWarningCode::ExpectedTermMissing`], holding the
    /// term exactly as the user wrote it.
    pub term: Option<String>,
}

#[derive(Clone, Debug, Default)]
pub struct TranscriptAnalysisOptions {
    /// Domain words that should have appeared in this utterance.
    pub expected_terms: Vec<String>,
    /// Optional coverage data from timestamp-bearing server responses.
    pub recording_duration_seconds: Option<f64>,
    pub covered_duration_seconds: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct TranscriptAnalysis {
    /// The exact string supplied to [`analyze_transcript`], including whitespace.
    pub raw_text: String,
    pub warnings: Vec<FidelityWarning>,
}

struct Patterns {
    list_item: Regex,
    filler: Regex,
    negation: Regex,
    like: Regex,
    short_answer: Regex,
}

static PATTERNS: OnceLock<Patterns> = OnceLock::new();

/// All regexes are Unicode-aware like the TS `/u` source and built once.
fn patterns() -> &'static Patterns {
    PATTERNS.get_or_init(|| Patterns {
        // TS: `/(?:^|\n)[ \t]*(\d+)[.)](?=\s)/m` — see module docs.
        list_item: Regex::new(r"(?m)(?:^|\n)[ \t]*(\d+)[.)]").unwrap(),
        filler: Regex::new(r"(?i)\b(?:er+|uh+|um+)\b").unwrap(),
        negation: Regex::new(
            r"(?i)\b(?:not|never|no|neither|nor|cannot|can't|won't|wouldn't|shouldn't|isn't|aren't|haven't|hasn't|hadn't|don't|doesn't|didn't)\b",
        )
        .unwrap(),
        like: Regex::new(r"(?i)\blike\b").unwrap(),
        short_answer: Regex::new(r"(?i)^\s*(?:[A-Za-z]|agreed|yes|no|okay|ok)\s*[.!]?\s*$").unwrap(),
    })
}

/// Analyze fidelity-sensitive text without changing it.
///
/// The warnings identify spans worth preserving or reviewing. They cannot tell
/// whether the recognizer heard the audio correctly, and they deliberately do
/// not guess replacements for uncommon words, negations, or corrections.
pub fn analyze_transcript(
    raw_text: &str,
    options: &TranscriptAnalysisOptions,
) -> TranscriptAnalysis {
    let patterns = patterns();
    let mut warnings: Vec<FidelityWarning> = Vec::new();

    if let Some(number) = first_list_item_number(raw_text, &patterns.list_item) {
        // JS compares `Number(digits) !== 1`; digit runs too large for u64 are
        // astronomically large numbers, i.e. also "not one".
        if !number.parse::<u64>().is_ok_and(|value| value == 1) {
            warnings.push(FidelityWarning {
                code: FidelityWarningCode::NonOneListStart,
                severity: Severity::Info,
                message: format!("List starts at {number}. Keep that number."),
                term: None,
            });
        }
    }

    push_matches(
        raw_text,
        &patterns.filler,
        FidelityWarningCode::PossibleSelfCorrection,
        Severity::Review,
        "Possible spoken correction. Check the nearby words before editing.",
        &mut warnings,
    );
    push_matches(
        raw_text,
        &patterns.negation,
        FidelityWarningCode::NegationPresent,
        Severity::Info,
        "Keep negations when editing; removing them can change the meaning.",
        &mut warnings,
    );
    push_matches(
        raw_text,
        &patterns.like,
        FidelityWarningCode::DiscourseWordPreserved,
        Severity::Info,
        "The transcript keeps the word \"like\".",
        &mut warnings,
    );

    if patterns.short_answer.is_match(raw_text) {
        warnings.push(FidelityWarning {
            code: FidelityWarningCode::ShortAnswer,
            severity: Severity::Info,
            message: "Keep short answers, including a single letter.".to_string(),
            term: None,
        });
    }

    let normalized_text = normalize_term(raw_text);
    for term in &options.expected_terms {
        if term.is_empty() || normalized_text.contains(&normalize_term(term)) {
            continue;
        }
        warnings.push(FidelityWarning {
            code: FidelityWarningCode::ExpectedTermMissing,
            severity: Severity::Review,
            message: format!("Expected term \"{term}\" is missing. Check the saved audio."),
            term: Some(term.clone()),
        });
    }

    if let (Some(recording), Some(covered)) = (
        options.recording_duration_seconds,
        options.covered_duration_seconds,
    ) {
        if recording.is_finite()
            && covered.is_finite()
            && recording > 1.0
            && covered >= 0.0
            && covered < recording * 0.8
        {
            warnings.push(FidelityWarning {
                code: FidelityWarningCode::PossibleAudioGap,
                severity: Severity::Review,
                message:
                    "Timed segments cover less than 80% of the recording. Check the saved audio."
                        .to_string(),
                term: None,
            });
        }
    }

    TranscriptAnalysis {
        raw_text: raw_text.to_string(),
        warnings,
    }
}

/// Finds the first `[.)]`-terminated list number followed by whitespace,
/// standing in for the TS lookahead `(?=\s)`.
fn first_list_item_number<'text>(text: &'text str, list_item: &Regex) -> Option<&'text str> {
    list_item.captures_iter(text).find_map(|captures| {
        let number = captures.get(1)?.as_str();
        let end = captures.get(0)?.end();
        let followed_by_whitespace = text
            .get(end..)
            .and_then(|rest| rest.chars().next())
            .is_some_and(char::is_whitespace);
        followed_by_whitespace.then_some(number)
    })
}

fn push_matches(
    text: &str,
    expression: &Regex,
    code: FidelityWarningCode,
    severity: Severity,
    message: &str,
    target: &mut Vec<FidelityWarning>,
) {
    for _ in expression.find_iter(text) {
        target.push(FidelityWarning {
            code,
            severity,
            message: message.to_string(),
            term: None,
        });
    }
}

/// TS: `value.normalize("NFKC").toLocaleLowerCase()` — NFKC is skipped, see the
/// module docs.
fn normalize_term(value: &str) -> String {
    value.to_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn codes_of(analysis: &TranscriptAnalysis) -> Vec<FidelityWarningCode> {
        analysis
            .warnings
            .iter()
            .map(|warning| warning.code)
            .collect()
    }

    fn count(analysis: &TranscriptAnalysis, code: FidelityWarningCode) -> usize {
        analysis
            .warnings
            .iter()
            .filter(|warning| warning.code == code)
            .count()
    }

    #[test]
    fn numbered_list_starting_after_one_is_flagged() {
        let raw = "3. capture audio\n4. transcribe it";
        let analysis = analyze_transcript(raw, &TranscriptAnalysisOptions::default());
        assert_eq!(analysis.raw_text, raw);
        assert_eq!(
            analysis.warnings[0].code,
            FidelityWarningCode::NonOneListStart
        );
        assert_eq!(analysis.warnings[0].severity, Severity::Info);
        assert_eq!(
            analysis.warnings[0].message,
            "List starts at 3. Keep that number."
        );
    }

    #[test]
    fn list_starting_at_one_is_not_flagged() {
        let analysis = analyze_transcript(
            "1. capture audio\n2. transcribe it",
            &TranscriptAnalysisOptions::default(),
        );
        assert!(!codes_of(&analysis).contains(&FidelityWarningCode::NonOneListStart));
    }

    #[test]
    fn correction_sounds_are_flagged_without_guessing_edits() {
        let raw = "I want the color to be orange, err, yellow.";
        let analysis = analyze_transcript(raw, &TranscriptAnalysisOptions::default());
        assert_eq!(analysis.raw_text, raw);
        assert_eq!(
            count(&analysis, FidelityWarningCode::PossibleSelfCorrection),
            1
        );
        let warning = analysis
            .warnings
            .iter()
            .find(|warning| warning.code == FidelityWarningCode::PossibleSelfCorrection)
            .unwrap();
        assert_eq!(warning.severity, Severity::Review);
    }

    #[test]
    fn um_and_uh_fillers_are_flagged() {
        let analysis = analyze_transcript("um, wait, uh", &TranscriptAnalysisOptions::default());
        assert_eq!(
            count(&analysis, FidelityWarningCode::PossibleSelfCorrection),
            2
        );
    }

    #[test]
    fn every_meaning_bearing_negation_is_counted() {
        let raw = "I'd prefer to never merge this, and I haven't approved it.";
        let analysis = analyze_transcript(raw, &TranscriptAnalysisOptions::default());
        assert_eq!(analysis.raw_text, raw);
        assert_eq!(count(&analysis, FidelityWarningCode::NegationPresent), 2);
    }

    #[test]
    fn discourse_word_like_is_reported() {
        let raw = "This was, like, much easier to review.";
        let analysis = analyze_transcript(raw, &TranscriptAnalysisOptions::default());
        assert_eq!(analysis.raw_text, raw);
        let warning = analysis
            .warnings
            .iter()
            .find(|warning| warning.code == FidelityWarningCode::DiscourseWordPreserved)
            .expect("like warning");
        assert_eq!(warning.message, "The transcript keeps the word \"like\".");
        assert_eq!(warning.severity, Severity::Info);
    }

    #[test]
    fn short_answer_no_is_kept() {
        let analysis = analyze_transcript("No.", &TranscriptAnalysisOptions::default());
        let warning = analysis
            .warnings
            .iter()
            .find(|warning| warning.code == FidelityWarningCode::ShortAnswer)
            .expect("short answer warning");
        assert_eq!(
            warning.message,
            "Keep short answers, including a single letter."
        );
    }

    #[test]
    fn single_letter_and_agreed_are_kept_but_words_are_not() {
        for raw in ["A.", "agreed."] {
            let analysis = analyze_transcript(raw, &TranscriptAnalysisOptions::default());
            assert!(
                codes_of(&analysis).contains(&FidelityWarningCode::ShortAnswer),
                "{raw} should be a short answer"
            );
        }
        let analysis = analyze_transcript("Nope.", &TranscriptAnalysisOptions::default());
        assert!(!codes_of(&analysis).contains(&FidelityWarningCode::ShortAnswer));
    }

    #[test]
    fn expected_term_missing_is_case_insensitive_on_both_sides() {
        let present = analyze_transcript(
            "Add AUTH to the proxy.",
            &TranscriptAnalysisOptions {
                expected_terms: vec!["auth".to_string()],
                ..Default::default()
            },
        );
        assert!(!codes_of(&present).contains(&FidelityWarningCode::ExpectedTermMissing));

        let missing = analyze_transcript(
            "Add off to the proxy.",
            &TranscriptAnalysisOptions {
                expected_terms: vec!["auth".to_string()],
                ..Default::default()
            },
        );
        let warning = missing
            .warnings
            .iter()
            .find(|warning| warning.code == FidelityWarningCode::ExpectedTermMissing)
            .expect("missing auth warning");
        assert_eq!(warning.term.as_deref(), Some("auth"));
        assert_eq!(warning.severity, Severity::Review);
        assert_eq!(
            warning.message,
            "Expected term \"auth\" is missing. Check the saved audio."
        );
    }

    #[test]
    fn empty_expected_terms_are_ignored() {
        let options = TranscriptAnalysisOptions {
            expected_terms: vec![String::new(), "auth".to_string()],
            ..Default::default()
        };
        let analysis = analyze_transcript("nothing relevant here", &options);
        assert_eq!(
            count(&analysis, FidelityWarningCode::ExpectedTermMissing),
            1
        );
    }

    #[test]
    fn coverage_gap_below_80_percent_is_flagged() {
        let gap = TranscriptAnalysisOptions {
            recording_duration_seconds: Some(600.0),
            covered_duration_seconds: Some(240.0),
            ..Default::default()
        };
        let analysis = analyze_transcript("First answer. Second answer.", &gap);
        assert!(codes_of(&analysis).contains(&FidelityWarningCode::PossibleAudioGap));

        let covered = TranscriptAnalysisOptions {
            recording_duration_seconds: Some(600.0),
            covered_duration_seconds: Some(500.0),
            ..Default::default()
        };
        let analysis = analyze_transcript("First answer. Second answer.", &covered);
        assert!(!codes_of(&analysis).contains(&FidelityWarningCode::PossibleAudioGap));
    }
}
