//! Replays `packages/contracts/mode-routing/fixtures/routing.json` and
//! `routing-variants.json` (the frozen routing corpus the Python oracle
//! `tests/mode_routing.py` replays in `tests/test_mode_routing.py`)
//! through the Rust port of the oracle, so both implementations are
//! held to one fixture set. Also checks the profiles fixtures parse and
//! pass the oracle's structural rules.

use std::collections::HashMap;
use std::path::PathBuf;

use serde_json::Value;
use starling_processing::contract::ProfilesDocument;
use starling_processing::routing::{self, RouteRequest, Status as RouteStatus};

fn contract_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../../packages/contracts/mode-routing")
}

fn fixture(name: &str) -> Value {
    let path = contract_dir().join("fixtures").join(name);
    let text =
        std::fs::read_to_string(&path).unwrap_or_else(|err| panic!("{}: {err}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|err| panic!("{}: {err}", path.display()))
}

fn profiles(name: &str) -> ProfilesDocument {
    serde_json::from_value(fixture(name)).unwrap_or_else(|err| panic!("{name}: {err}"))
}

fn request(case: &Value) -> RouteRequest {
    let raw = case["request"].clone();
    let field = |name: &str| raw[name].as_str().map(str::to_string);
    RouteRequest {
        raw_text: raw["raw_text"].as_str().expect("raw_text").to_string(),
        manual_mode: field("manual_mode"),
        manual_locked: raw
            .get("manual_locked")
            .and_then(Value::as_bool)
            .unwrap_or(true),
        session_allows_aliases: raw
            .get("session_allows_aliases")
            .and_then(Value::as_bool)
            .unwrap_or(true),
        secure_field: raw.get("secure_field").and_then(Value::as_bool).unwrap_or(false),
        project_id: field("project_id"),
        site: field("site"),
        app_id: field("app_id"),
        selection_available: raw
            .get("selection_available")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        selection_granted: raw
            .get("selection_granted")
            .and_then(Value::as_bool)
            .unwrap_or(false),
    }
}

#[test]
fn every_routing_fixture_replays_like_the_oracle() {
    let mut configs: HashMap<String, ProfilesDocument> = HashMap::new();
    let mut cases = Vec::new();
    for filename in ["routing.json", "routing-variants.json"] {
        for case in fixture(filename).as_array().unwrap() {
            let config_name = case["profiles"].as_str().unwrap_or("profiles.json").to_string();
            cases.push((config_name, case.clone()));
        }
    }
    assert_eq!(cases.len(), 33, "25 reference cases plus 8 variants");

    let mut failures = Vec::new();
    for (config_name, case) in &cases {
        let doc = configs
            .entry(config_name.clone())
            .or_insert_with(|| profiles(config_name));
        let request = request(case);
        let result = routing::resolve(doc, &request).expect("resolve");
        let expected = &case["expected"];
        let want_mode = expected["mode"].as_str().map(str::to_string);
        if result.mode != want_mode {
            failures.push(format!(
                "{config_name}/{}: mode {:?} != {:?}",
                case["name"], result.mode, want_mode
            ));
        }
        if result.source.as_str() != expected["source"].as_str().unwrap() {
            failures.push(format!(
                "{config_name}/{}: source {} != {}",
                case["name"],
                result.source.as_str(),
                expected["source"]
            ));
        }
        if result.payload != expected["payload"].as_str().unwrap() {
            failures.push(format!(
                "{config_name}/{}: payload {:?} != {:?}",
                case["name"], result.payload, expected["payload"]
            ));
        }
        if result.status.as_str() != expected["status"].as_str().unwrap() {
            failures.push(format!(
                "{config_name}/{}: status {} != {}",
                case["name"],
                result.status.as_str(),
                expected["status"]
            ));
        }
        // The view invariant: with a recorded removed prefix span the
        // payload is exactly the raw text from the span's end (spans
        // start at 0), so the raw recognition reconstructs.
        if let Some((start, end)) = result.prefix_span_codepoints {
            let view: String = result.raw_text.chars().skip(end).collect();
            let prefix: String = result.raw_text.chars().take(start).collect();
            if view != result.payload || !prefix.is_empty() {
                failures.push(format!(
                    "{config_name}/{}: payload is not a prefix-removed view",
                    case["name"]
                ));
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

#[test]
fn ambiguous_phrases_surface_their_candidates_not_a_guess() {
    let doc = profiles("profiles-alias-collision.json");
    let request = RouteRequest::new("code this: now");
    let result = routing::resolve(&doc, &request).unwrap();
    assert_eq!(result.status, RouteStatus::NeedsResolution);
    assert_eq!(result.mode, None);
    let conflicts = routing::conflicts_for(&doc, &request, &result);
    let [conflict] = conflicts.as_slice() else {
        panic!("a phrase conflict lists its candidates: {conflicts:?}");
    };
    assert_eq!(conflict.candidates.len(), 2);
    assert!(conflict
        .candidates
        .iter()
        .all(|candidate| candidate.via == "code this"));
    assert!(routing::explain(&result, Some(&doc)).contains("conflict"));
}

#[test]
fn every_profiles_fixture_parses_and_passes_the_structural_rules() {
    for name in [
        "profiles.json",
        "profiles-alias-collision.json",
        "profiles-longest-alias.json",
        "profiles-project-priority.json",
        "profiles-rule-conflict.json",
    ] {
        let doc = profiles(name);
        routing::validate_config(&doc).unwrap_or_else(|err| panic!("{name}: {err}"));
    }
}

#[test]
fn an_unknown_manual_mode_is_an_error_never_a_guess() {
    let doc = profiles("profiles.json");
    let request = RouteRequest {
        manual_mode: Some("missing".to_string()),
        ..RouteRequest::new("hello")
    };
    assert!(routing::resolve(&doc, &request).is_err());
}

#[test]
fn verbatim_ignores_aliases_even_when_a_session_allows_them() {
    // The frozen invariant: a verbatim session routes by its own rules;
    // leading phrases are not parsed for it.
    let doc = profiles("profiles.json");
    let request = RouteRequest {
        manual_mode: Some("verbatim".to_string()),
        manual_locked: false,
        session_allows_aliases: true,
        ..RouteRequest::new("code this add logs")
    };
    let result = routing::resolve(&doc, &request).unwrap();
    assert_eq!(result.source.as_str(), "manual");
    assert_eq!(result.mode.as_deref(), Some("verbatim"));
    assert_eq!(result.payload, "code this add logs");
}
