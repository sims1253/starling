//! Replays `packages/contracts/insertion-boundary/fixtures/boundary-cases.json`
//! through [`starling_processing::boundary`]. The Python oracle
//! (`tests/insertion_boundary.py`) replays the same file.

use std::path::PathBuf;

use serde::Deserialize;
use starling_processing::boundary::{self, BoundaryChange, BoundaryContext, BoundaryOptions};

#[derive(Deserialize)]
struct Change {
    kind: BoundaryChange,
}

#[derive(Deserialize)]
struct Case {
    case_id: String,
    before: String,
    after: String,
    raw: String,
    verbatim: bool,
    expected_text: String,
    expected_changes: Vec<Change>,
}

#[test]
fn replays_every_pinned_case() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../../packages/contracts/insertion-boundary/fixtures/boundary-cases.json");
    let text =
        std::fs::read_to_string(&path).unwrap_or_else(|err| panic!("{}: {err}", path.display()));
    let cases: Vec<Case> =
        serde_json::from_str(&text).unwrap_or_else(|err| panic!("{}: {err}", path.display()));
    assert!(
        cases.len() >= 30,
        "the acceptance fixture classes must stay pinned"
    );

    for case in cases {
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
        let expected_changes: Vec<BoundaryChange> = case
            .expected_changes
            .iter()
            .map(|change| change.kind)
            .collect();
        assert_eq!(got.text, case.expected_text, "{}", case.case_id);
        assert_eq!(got.changes, expected_changes, "{}", case.case_id);
    }
}
