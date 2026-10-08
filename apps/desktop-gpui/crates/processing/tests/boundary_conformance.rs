//! Replays `packages/contracts/insertion-boundary/fixtures/boundary-cases.json`
//! (the #341 boundary cases) through [`starling_processing::boundary`]. The
//! Python oracle (`tests/insertion_boundary.py`, `tests/test_insertion_boundary.py`)
//! replays the same file, so neither implementation can test against a
//! different copy.

use std::path::PathBuf;

use serde::Deserialize;
use serde_json::Value;
use starling_processing::boundary::{self, BoundaryContext, BoundaryOptions};

fn contract_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../../packages/contracts/insertion-boundary")
}

fn cases() -> Vec<Value> {
    let path = contract_dir().join("fixtures").join("boundary-cases.json");
    let text =
        std::fs::read_to_string(&path).unwrap_or_else(|err| panic!("{}: {err}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|err| panic!("{}: {err}", path.display()))
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ChangeKind {
    LeadingSpace,
    FirstLetterCase,
}

#[derive(Debug, Deserialize)]
struct Change {
    kind: ChangeKind,
    #[allow(dead_code)]
    detail: String,
}

#[derive(Debug, Deserialize)]
struct Case {
    #[allow(dead_code)]
    case_id: String,
    before: String,
    #[allow(dead_code)]
    after: String,
    raw: String,
    verbatim: bool,
    expected_text: String,
    expected_changes: Vec<Change>,
}

#[test]
fn replays_every_pinned_case() {
    let docs = cases();
    assert!(
        docs.len() >= 30,
        "the acceptance fixture classes must stay pinned"
    );

    for doc in docs {
        let case: Case = serde_json::from_value(doc.clone())
            .unwrap_or_else(|err| panic!("{}: {err}", doc["case_id"].as_str().unwrap()));

        let got = boundary::adjust(
            &case.raw,
            &BoundaryContext {
                before: &case.before,
                after: &case.after,
            },
            &BoundaryOptions {
                verbatim: case.verbatim,
            },
        );

        assert_eq!(
            got.text,
            case.expected_text,
            "{}: adjusted text differs",
            doc["case_id"].as_str().unwrap()
        );
        assert_eq!(
            got.changes.len(),
            case.expected_changes.len(),
            "{}: change count differs: {:?}",
            doc["case_id"].as_str().unwrap(),
            got.changes
        );
        for (got, expected) in got.changes.iter().zip(&case.expected_changes) {
            let expected_kind = match expected.kind {
                ChangeKind::LeadingSpace => boundary::BoundaryChangeKind::LeadingSpace,
                ChangeKind::FirstLetterCase => boundary::BoundaryChangeKind::FirstLetterCase,
            };
            assert_eq!(
                got.kind,
                expected_kind,
                "{}: change kind differs",
                doc["case_id"].as_str().unwrap()
            );
        }
    }
}

/// The same invariant the Python suite pins: the adjustment may only
/// prepend one space and lowercase the first cased character (its full
/// lowercase mapping) — nothing beyond the boundary ever changes.
#[test]
fn raw_is_never_modified_beyond_the_boundary() {
    for doc in cases() {
        let case: Case = serde_json::from_value(doc.clone()).unwrap();
        let got = boundary::adjust(
            &case.raw,
            &BoundaryContext {
                before: &case.before,
                after: &case.after,
            },
            &BoundaryOptions {
                verbatim: case.verbatim,
            },
        );

        let stripped: String = if got.text.starts_with(' ') && !case.raw.starts_with(' ') {
            got.text[1..].to_string()
        } else {
            got.text.clone()
        };
        if stripped == case.raw {
            continue;
        }

        // `stripped` equals the raw text with exactly the first cased
        // character replaced by its full lowercase mapping.
        let raw_chars: Vec<char> = case.raw.chars().collect();
        let first_cased = raw_chars
            .iter()
            .position(|ch| ch.is_alphabetic() && ch.to_lowercase().next() != Some(*ch))
            .expect("an unexpected difference appeared");
        let lowered: String = raw_chars[first_cased].to_lowercase().collect();
        let mut expected = String::new();
        expected.extend(&raw_chars[..first_cased]);
        expected.push_str(&lowered);
        expected.extend(&raw_chars[first_cased + 1..]);
        assert_eq!(
            stripped, expected,
            "{}: changes beyond the boundary",
            doc["case_id"].as_str().unwrap()
        );
    }
}
