//! Replays `packages/contracts/mode-routing/fixtures/staging.json` (the
//! #293 staging cases) through [`starling_processing::staging`], and checks
//! the other contract fixtures parse into the typed records and agree with
//! the ported cross-field rules. The Python oracle (`tests/staging.py`,
//! `tests/test_staging.py`) replays the same files.

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde_json::Value;
use starling_processing::contract::{
    self, ProcessingRoute, ProfilesDocument, ProviderDecl, TransformRequest, TransformResult,
};
use starling_processing::staging::{Draft, Op};

fn contract_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../../packages/contracts/mode-routing")
}

fn fixture(name: &str) -> Value {
    let path = contract_dir().join("fixtures").join(name);
    let text =
        std::fs::read_to_string(&path).unwrap_or_else(|err| panic!("{}: {err}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|err| panic!("{}: {err}", path.display()))
}

/// Every (attempt id -> (segment, text)) the case fed in, first occurrence.
fn fed_finals(ops: &[Value], upto: usize) -> Vec<(String, u32, String)> {
    ops[..=upto]
        .iter()
        .filter(|op| op["op"] == "final")
        .map(|op| {
            (
                op["attempt_id"].as_str().unwrap().to_string(),
                op["segment"].as_u64().unwrap() as u32,
                op["text"].as_str().unwrap().to_string(),
            )
        })
        .collect()
}

/// The oracle's `invariant_violations`.
fn invariant_violations(draft: &Draft, finals: &[(String, u32, String)]) -> Vec<String> {
    let mut found = Vec::new();
    let snap = draft.snapshot();
    let mut pos = 0;
    for region in &snap.regions {
        if region.span[0] != pos || region.span[1] <= region.span[0] {
            found.push(format!("regions do not tile the text at {region:?}"));
        }
        pos = region.span[1];
    }
    if pos != snap.text.chars().count() {
        found.push("regions do not cover the text".to_string());
    }
    let mut seen: BTreeMap<&str, (u32, &str)> = BTreeMap::new();
    for (id, segment, text) in finals {
        seen.entry(id.as_str()).or_insert((*segment, text.as_str()));
    }
    for attempt in draft.attempts() {
        if seen.get(attempt.attempt_id.as_str()) != Some(&(attempt.segment, attempt.text.as_str()))
        {
            found.push(format!(
                "attempt {} is not the recognition it came from",
                attempt.attempt_id
            ));
        }
    }
    let recorded: Vec<&str> = draft
        .attempts()
        .iter()
        .map(|a| a.attempt_id.as_str())
        .collect();
    let mut latest: BTreeMap<u32, &str> = BTreeMap::new();
    let mut order: Vec<&str> = Vec::new();
    for (id, _, _) in finals {
        if !order.contains(&id.as_str()) {
            order.push(id);
        }
    }
    for id in order {
        if recorded.contains(&id) {
            let (segment, text) = seen[id];
            latest.insert(segment, text);
        }
    }
    let expected: String = latest.values().copied().collect();
    if draft.raw_text().as_bytes() != expected.as_bytes() {
        found.push("raw text does not round-trip byte for byte".to_string());
    }
    for proposal in &snap.proposals {
        if proposal.status == starling_processing::staging::ProposalStatus::Current
            && proposal.base_revision != snap.revision
        {
            found.push(format!(
                "proposal {} current on an old base",
                proposal.request_id
            ));
        }
    }
    found
}

fn expectation_mismatches(draft: &Draft, expect: &Value) -> Vec<String> {
    let snap = serde_json::to_value(draft.snapshot()).unwrap();
    let mut found = Vec::new();
    for key in ["text", "revision", "deleted", "pinned_segments"] {
        if let Some(want) = expect.get(key) {
            if &snap[key] != want {
                found.push(format!("{key}: expected {want}, got {}", snap[key]));
            }
        }
    }
    if let Some(want) = expect.get("raw_text") {
        if want.as_str() != Some(draft.raw_text().as_str()) {
            found.push(format!(
                "raw_text: expected {want}, got {:?}",
                draft.raw_text()
            ));
        }
    }
    let text: Vec<char> = draft.text().chars().collect();
    if let Some(want) = expect.get("regions") {
        let got: Vec<Value> = draft
            .snapshot()
            .regions
            .iter()
            .map(|region| {
                serde_json::json!([
                    serde_json::to_value(region.kind).unwrap(),
                    text[region.span[0]..region.span[1]]
                        .iter()
                        .collect::<String>()
                ])
            })
            .collect();
        if &Value::Array(got.clone()) != want {
            found.push(format!("regions: expected {want}, got {got:?}"));
        }
    }
    let statuses = |list: &Value| -> Value {
        Value::Object(
            list.as_array()
                .unwrap()
                .iter()
                .map(|item| {
                    (
                        item["request_id"].as_str().unwrap().to_string(),
                        item["status"].clone(),
                    )
                })
                .collect(),
        )
    };
    for key in ["requests", "proposals"] {
        if let Some(want) = expect.get(key) {
            let got = statuses(&snap[key]);
            if &got != want {
                found.push(format!("{key}: expected {want}, got {got}"));
            }
        }
    }
    if let Some(want) = expect.get("request_inputs") {
        for (id, pair) in want.as_object().unwrap() {
            let request = draft.request(id).expect("request recorded");
            let got = serde_json::json!([request.input, request.instruction]);
            if &got != pair {
                found.push(format!("request_inputs[{id}]: expected {pair}, got {got}"));
            }
        }
    }
    found
}

#[test]
fn every_staging_case_replays_like_the_oracle() {
    let cases = fixture("staging.json");
    let cases = cases.as_array().expect("case list");
    assert!(cases.len() >= 20, "the staging corpus is loaded");
    let mut failures = Vec::new();
    'cases: for case in cases {
        let name = case["name"].as_str().unwrap();
        let ops = case["ops"].as_array().unwrap();
        let mut draft = Draft::new(
            case.get("draft_id")
                .and_then(Value::as_str)
                .unwrap_or("draft-1"),
            case.get("capture_id")
                .and_then(Value::as_str)
                .unwrap_or("cap-1"),
        );
        for (step, raw_op) in ops.iter().enumerate() {
            let op: Op = match serde_json::from_value(raw_op.clone()) {
                Ok(op) => op,
                Err(err) => {
                    failures.push(format!("{name} step {step}: {err}"));
                    continue 'cases;
                }
            };
            let outcome = draft.apply(&op);
            if let Some(want) = raw_op.get("expect").and_then(Value::as_str) {
                if want != outcome.as_str() {
                    failures.push(format!(
                        "{name} step {step}: expected {want}, got {}",
                        outcome.as_str()
                    ));
                }
            }
            if let Some(want) = raw_op.get("expect_text").and_then(Value::as_str) {
                if want != draft.text() {
                    failures.push(format!(
                        "{name} step {step}: expected text {want:?}, got {:?}",
                        draft.text()
                    ));
                }
            }
            for violation in invariant_violations(&draft, &fed_finals(ops, step)) {
                failures.push(format!("{name} step {step}: {violation}"));
            }
        }
        for mismatch in expectation_mismatches(&draft, &case["expect"]) {
            failures.push(format!("{name}: {mismatch}"));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

#[test]
fn snapshots_carry_exactly_the_schema_fields() {
    let schema: Value = serde_json::from_str(
        &std::fs::read_to_string(contract_dir().join("draft.schema.json")).unwrap(),
    )
    .unwrap();
    let mut draft = Draft::new("draft-1", "cap-1");
    draft.final_attempt(0, "att-1", "hello");
    draft.request_transform("r1", None);
    draft.result(
        "r1",
        starling_processing::staging::ResultKind::Completed,
        Some("Hello."),
    );
    draft.deliver("d1", "t1");
    let snap = serde_json::to_value(draft.snapshot()).unwrap();
    let mut want: Vec<&str> = schema["required"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    let mut got: Vec<&str> = snap
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    want.sort_unstable();
    got.sort_unstable();
    assert_eq!(got, want);
    let region_fields: Vec<&str> = schema["properties"]["regions"]["items"]["required"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    let mut got: Vec<&str> = snap["regions"][0]
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    let mut want = region_fields;
    got.sort_unstable();
    want.sort_unstable();
    assert_eq!(got, want);
}

#[test]
fn raw_round_trips_through_the_wire_byte_for_byte() {
    let raw = "na\u{ef}ve \u{1F469}\u{200D}\u{1F4BB} cafe\u{301} \u{5e9}\u{5dc}\u{5d5}\u{5dd}\r\n\ttab\u{0}";
    let mut draft = Draft::new("draft-1", "cap-1");
    draft.final_attempt(0, "att-1", raw);
    draft.insert(3, "X");
    draft.delete(0, 2);
    let wire = serde_json::to_vec(&draft.snapshot()).unwrap();
    let back: starling_processing::staging::DraftSnapshot = serde_json::from_slice(&wire).unwrap();
    assert_eq!(back.attempts[0].text.as_bytes(), raw.as_bytes());
    assert_eq!(draft.raw_text().as_bytes(), raw.as_bytes());
}

#[test]
fn profiles_documents_parse_and_pass_the_processing_rules() {
    let dir = contract_dir().join("fixtures");
    let mut seen = 0;
    for entry in std::fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        if !(name.starts_with("profiles") && name.ends_with(".json")) {
            continue;
        }
        let doc: ProfilesDocument = serde_json::from_str(&std::fs::read_to_string(&path).unwrap())
            .unwrap_or_else(|err| panic!("{name}: {err}"));
        contract::validate_processing(&doc).unwrap_or_else(|err| panic!("{name}: {err}"));
        seen += 1;
    }
    assert!(seen >= 5);
}

#[test]
fn processing_route_cases_match_the_oracle() {
    let providers: Vec<ProviderDecl> = serde_json::from_value(fixture("providers.json")).unwrap();
    for provider in &providers {
        contract::validate_provider(provider).unwrap();
    }
    let doc: ProfilesDocument = serde_json::from_value(fixture("profiles.json")).unwrap();
    for case in fixture("processing-routes.json").as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let mut profile =
            serde_json::to_value(doc.profile(case["profile"].as_str().unwrap()).unwrap()).unwrap();
        if let Some(patch) = case.get("patch") {
            for (key, value) in patch.as_object().unwrap() {
                profile[key] = value.clone();
            }
        }
        let profile = serde_json::from_value(profile).unwrap();
        let expect = &case["expect"];
        let got = match contract::processing_route(&profile, &providers) {
            ProcessingRoute::None => serde_json::json!({
                "status": "none", "provider": null, "reason": null, "context_fields": []}),
            ProcessingRoute::Ready {
                provider,
                context_fields,
            } => serde_json::json!({
                "status": "ready", "provider": provider.id, "reason": null,
                "context_fields": context_fields}),
            ProcessingRoute::Blocked(block) => serde_json::json!({
                "status": "blocked", "provider": null, "reason": block.as_str(), "context_fields": []}),
        };
        assert_eq!(&got, expect, "{name}");
    }
}

#[test]
fn request_and_result_fixtures_round_trip_through_the_typed_records() {
    for name in ["transform-requests.json", "transform-results.json"] {
        for item in fixture(name).as_array().unwrap() {
            let back = if name.starts_with("transform-requests") {
                let request: TransformRequest = serde_json::from_value(item.clone()).unwrap();
                serde_json::to_value(request).unwrap()
            } else {
                let result: TransformResult = serde_json::from_value(item.clone()).unwrap();
                serde_json::to_value(result).unwrap()
            };
            assert_eq!(&back, item, "{name}");
        }
    }
    let mut poisoned = fixture("transform-requests.json")[0].clone();
    poisoned["execute"] = Value::Bool(true);
    assert!(serde_json::from_value::<TransformRequest>(poisoned).is_err());
}
