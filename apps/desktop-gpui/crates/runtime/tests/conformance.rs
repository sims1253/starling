//! Conformance suite — the port of `tests/test_runtime_protocol.py`
//! (E17-I0): every valid fixture trace replays green through this
//! crate's oracle port, every invalid fixture is rejected with the
//! oracle's reason class, and the §2 invariants hold.
//!
//! The fixtures are vendored under `tests/fixtures/` byte-identical to
//! `packages/contracts/runtime-protocol/fixtures/` (the frozen I0
//! contract data), so this suite runs wherever the crate runs — the
//! "independent test fixtures" acceptance criterion.

use std::collections::HashSet;
use std::path::PathBuf;

use starling_runtime::protocol::replay::{
    self, is_directive, route_freeze_violations, seq_violations, MachineReplay, Violation,
};
use starling_runtime::protocol::tables::{self, spec_for};
use starling_runtime::protocol::{nack_for, Command, Envelope};

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn fixture_paths() -> Vec<PathBuf> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(fixture_dir())
        .expect("fixtures directory")
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().is_some_and(|ext| ext == "json"))
        .collect();
    let mut invalid: Vec<PathBuf> = std::fs::read_dir(fixture_dir().join("invalid"))
        .expect("invalid fixtures directory")
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().is_some_and(|ext| ext == "json"))
        .collect();
    paths.append(&mut invalid);
    paths.sort();
    paths
}

fn load_fixtures() -> (
    Vec<(PathBuf, serde_json::Value)>,
    Vec<(PathBuf, serde_json::Value)>,
) {
    replay::split_fixtures(&fixture_paths())
}

fn iter_envelopes(trace: &serde_json::Value) -> Vec<serde_json::Value> {
    trace
        .get("messages")
        .and_then(serde_json::Value::as_array)
        .map(|records| {
            records
                .iter()
                .filter(|record| !is_directive(record))
                .cloned()
                .collect()
        })
        .unwrap_or_default()
}

fn name_of(path: &PathBuf) -> String {
    path.file_name().unwrap().to_string_lossy().to_string()
}

// ------------------------------------------------------------------------- //
// Fixture corpus shape
// ------------------------------------------------------------------------- //

#[test]
fn fixture_corpus_covers_all_machines() {
    let (valid, invalid) = load_fixtures();
    let machines: HashSet<&str> = valid
        .iter()
        .map(|(_, trace)| trace.get("machine").and_then(|m| m.as_str()).unwrap())
        .collect();
    assert_eq!(
        machines,
        HashSet::from(["capture", "jobs", "context", "docs", "delivery"])
    );
    // By name, so a new fixture shows up as exactly the file it is.
    let names = |set: &[(PathBuf, serde_json::Value)]| -> Vec<String> {
        set.iter().map(|(path, _)| name_of(path)).collect()
    };
    assert_eq!(
        names(&valid),
        [
            "capture-abort.json",
            "capture-happy.json",
            "capture-interrupted.json",
            "capture-recovery.json",
            "context-freeze.json",
            "delivery-failure-conflict.json",
            "delivery-lifecycle.json",
            "docs-cas.json",
            "jobs-failure.json",
            "jobs-lifecycle.json",
            "jobs-transform.json",
        ]
    );
    assert_eq!(
        names(&invalid),
        [
            "invalid-capture-seq.json",
            "invalid-capture-stop-idle.json",
            "invalid-capture-stopped-not-draining.json",
            "invalid-context-mode-set-observing.json",
            "invalid-delivery-apply-unprepared.json",
            "invalid-docs-append-turn-validating.json",
            "invalid-jobs-cancel-after-completed.json",
            "invalid-jobs-completed-from-transforming.json",
            "invalid-jobs-transformed-from-recognizing.json",
            "invalid-jobs-unfrozen-route.json",
            "invalid-runtime-unknown-version.json",
        ]
    );
}

// ------------------------------------------------------------------------- //
// Schema conformance (structural): the Rust spelling of the schemas'
// additionalProperties: false + envelope structure.
// ------------------------------------------------------------------------- //

#[test]
fn valid_fixture_messages_conform() {
    let (valid, _) = load_fixtures();
    for (path, trace) in &valid {
        for message in iter_envelopes(trace) {
            let envelope = Envelope::from_value(&message)
                .unwrap_or_else(|| panic!("{}: envelope did not parse", name_of(path)));
            let errors = starling_runtime::protocol::envelope_errors(&envelope.to_value());
            assert!(errors.is_empty(), "{}: {errors:?}", name_of(path));
            // The typed payload parses (deny_unknown_fields implements
            // additionalProperties: false).
            let payload = envelope.payload.clone();
            if replay::message_kind(&envelope.type_)
                == Some(starling_runtime::protocol::Kind::Command)
            {
                Command::from_parts(&envelope.type_, &payload)
                    .unwrap_or_else(|err| panic!("{}: {}: {err}", name_of(path), envelope.type_));
            }
        }
    }
}

#[test]
fn unknown_payload_field_rejected() {
    let (valid, _) = load_fixtures();
    let (_, trace) = valid
        .iter()
        .find(|(path, _)| name_of(path) == "capture-happy.json")
        .expect("capture-happy fixture");
    let mut poisoned = iter_envelopes(trace)
        .into_iter()
        .next()
        .expect("first message");
    poisoned
        .as_object_mut()
        .unwrap()
        .get_mut("payload")
        .unwrap()
        .as_object_mut()
        .unwrap()
        .insert("extra".into(), "no".into());
    let envelope = Envelope::from_value(&poisoned).expect("parses");
    assert!(Command::from_parts(&envelope.type_, &envelope.payload).is_err());
}

#[test]
fn unknown_envelope_field_rejected() {
    let (valid, _) = load_fixtures();
    let (_, trace) = valid
        .iter()
        .find(|(path, _)| name_of(path) == "capture-happy.json")
        .expect("capture-happy fixture");
    let mut poisoned = iter_envelopes(trace)
        .into_iter()
        .next()
        .expect("first message");
    poisoned
        .as_object_mut()
        .unwrap()
        .insert("trace_id".into(), "no".into());
    let errors = starling_runtime::protocol::envelope_errors(&poisoned);
    assert!(errors.iter().any(|e| e.contains("trace_id")));
}

#[test]
fn missing_required_envelope_field_rejected() {
    let (valid, _) = load_fixtures();
    let (_, trace) = valid
        .iter()
        .find(|(path, _)| name_of(path) == "capture-happy.json")
        .expect("capture-happy fixture");
    let mut poisoned = iter_envelopes(trace)
        .into_iter()
        .next()
        .expect("first message");
    poisoned.as_object_mut().unwrap().remove("ts");
    let errors = starling_runtime::protocol::envelope_errors(&poisoned);
    assert!(errors.iter().any(|e| e.contains("ts")));
}

// ------------------------------------------------------------------------- //
// Replay: valid traces green, invalid traces rejected with the right class
// ------------------------------------------------------------------------- //

#[test]
fn valid_traces_replay_green() {
    let (valid, _) = load_fixtures();
    for (path, trace) in &valid {
        let replayed = replay_trace_of(trace)
            .unwrap_or_else(|violation| panic!("{}: {violation}", name_of(path)));
        assert!(
            !replayed.transitions.is_empty(),
            "{}: no transitions applied",
            name_of(path)
        );
    }
}

fn replay_trace_of(trace: &serde_json::Value) -> Result<MachineReplay, Violation> {
    replay::replay_trace(trace)
}

#[test]
fn invalid_fixtures_fail_with_expected_violation() {
    let (_, invalid) = load_fixtures();
    let (valid, _) = load_fixtures();
    for (path, trace) in &invalid {
        let expected = trace
            .get("replay")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default();
        if expected == "route_not_frozen" {
            // Corpus-level ordering rule, not a single-machine rule: the
            // machine replay alone is green; appended to the valid corpus,
            // the submit's route is flagged.
            assert!(replay_trace_of(trace).is_ok(), "{}", name_of(path));
            let mut corpus = replay::valid_corpus_messages(&valid);
            for message in iter_envelopes(trace) {
                corpus.push(message);
            }
            let violations = route_freeze_violations(&corpus);
            assert!(!violations.is_empty(), "{}", name_of(path));
            assert!(
                violations.iter().all(|v| v.route == "route-never-frozen"),
                "{:?}",
                violations
            );
            continue;
        }
        let violation = replay_trace_of(trace)
            .err()
            .unwrap_or_else(|| panic!("{}: expected rejection", name_of(path)));
        assert_eq!(
            violation.code(),
            expected,
            "{}: expected {expected}, got {}",
            name_of(path),
            violation.code()
        );
    }
}

#[test]
fn valid_traces_seq_monotonic() {
    let (valid, _) = load_fixtures();
    for (path, trace) in &valid {
        let messages = trace
            .get("messages")
            .and_then(serde_json::Value::as_array)
            .map(|records| records.clone())
            .unwrap_or_default();
        assert!(seq_violations(&messages).is_empty(), "{}", name_of(path));
    }
}

#[test]
fn every_named_state_visited() {
    let (valid, _) = load_fixtures();
    let mut visited: std::collections::HashMap<&str, HashSet<String>> = Default::default();
    for (_, trace) in &valid {
        let machine = trace.get("machine").and_then(|m| m.as_str()).unwrap();
        let replayed = replay_trace_of(trace).expect("valid trace");
        visited
            .entry(machine)
            .or_default()
            .extend(replayed.visited.iter().cloned());
    }
    for spec in tables::all_specs() {
        let seen = visited.get(spec.name).cloned().unwrap_or_default();
        let missing: Vec<&str> = spec
            .states
            .iter()
            .filter(|state| !seen.contains(**state))
            .copied()
            .collect();
        assert!(
            missing.is_empty(),
            "{}: states never visited {missing:?}",
            spec.name
        );
    }
}

// ------------------------------------------------------------------------- //
// Schema <-> oracle agreement (the tables are the schemas' Rust spelling)
// ------------------------------------------------------------------------- //

#[test]
fn machine_tables_match_schema_kinds() {
    // The union of the tables equals the typed v1 command/event sets —
    // exhaustively checked in the crate's own unit tests; here the
    // fixtures' message types must all be table members.
    let (valid, invalid) = load_fixtures();
    for (_, trace) in valid.iter().chain(invalid.iter()) {
        for message in iter_envelopes(trace) {
            let type_ = message.get("type").and_then(|t| t.as_str()).unwrap();
            if type_ == "runtime.nack" {
                continue;
            }
            assert!(
                tables::machine_of_type(type_).is_some(),
                "{type_} is not a v1 message type"
            );
        }
    }
}

#[test]
fn internal_edges_stay_inside_declared_states() {
    for spec in tables::all_specs() {
        for (from, to) in spec.internal {
            assert!(spec.states.contains(from));
            assert!(spec.states.contains(to));
        }
    }
}

// ------------------------------------------------------------------------- //
// Invariant spot-tests (E17 section 2)
// ------------------------------------------------------------------------- //

#[test]
fn capture_never_blocks_draining_on_jobs() {
    let capture = spec_for("capture").unwrap();
    for state in capture.states {
        let leaving = capture.exits(state);
        assert!(
            !leaving.iter().any(|t| t.starts_with("jobs.")),
            "{state}: {leaving:?}"
        );
    }
    let draining_exits = capture.exits("Draining");
    assert!(draining_exits.contains(&"capture.stopped"));
    assert!(!draining_exits.iter().any(|t| t.starts_with("jobs.")));
}

#[test]
fn docs_head_update_is_cas_with_candidate_retained() {
    let (valid, _) = load_fixtures();
    let (_, trace) = valid
        .iter()
        .find(|(path, _)| name_of(path) == "docs-cas.json")
        .expect("docs-cas fixture");
    let conflict = iter_envelopes(trace)
        .into_iter()
        .find(|message| message.get("type").and_then(|t| t.as_str()) == Some("docs.headConflict"))
        .expect("conflict event");
    let payload = &conflict["payload"];
    assert_ne!(payload["expected"], payload["actual"]);
    assert_eq!(payload["candidatePreserved"], serde_json::json!(true));

    let replayed = replay_trace_of(trace).expect("green replay");
    let conflict_edges: Vec<_> = replayed
        .transitions
        .iter()
        .filter(|t| t.type_ == Some("docs.headConflict"))
        .collect();
    assert_eq!(conflict_edges.len(), 1);
    assert_eq!(conflict_edges[0].from, "Validating");
    assert_eq!(conflict_edges[0].to, "Conflicted");
    assert!(replayed
        .transitions
        .iter()
        .any(|t| { t.type_ == Some("docs.headUpdated") && t.to == "Committed" }));
}

#[test]
fn delivery_has_no_auto_enter_or_auto_send_command() {
    let delivery_commands: HashSet<&str> = spec_for("delivery")
        .unwrap()
        .commands
        .iter()
        .map(|(name, _)| *name)
        .collect();
    assert_eq!(
        delivery_commands,
        HashSet::from([
            "delivery.prepare",
            "delivery.apply",
            "delivery.cancel",
            "delivery.copyFallback",
        ])
    );
    for type_ in &delivery_commands {
        for word in ["send", "enter", "auto", "inject", "press", "confirm"] {
            assert!(!type_.contains(word), "{type_} contains {word}");
        }
    }
    // apply is user-initiated and names the delivery it applies.
    let apply = Command::DeliveryApply {
        delivery_id: "dlv-1a".into(),
    };
    let payload = apply.payload_value();
    assert!(payload.get("deliveryId").is_some());
}

#[test]
fn route_freeze_precedes_any_audio_leave() {
    let (valid, _) = load_fixtures();
    let corpus = replay::valid_corpus_messages(&valid);
    assert!(route_freeze_violations(&corpus).is_empty());

    // mode.routeFrozen is only legal from ModeDecided — a decision must
    // exist before audio can leave on the frozen route.
    let rule = spec_for("context")
        .unwrap()
        .event_rule("mode.routeFrozen")
        .expect("routeFrozen rule");
    assert_eq!(rule.from, &["ModeDecided"]);

    // The frozen route in the context fixture precedes the take that uses it.
    let (_, context_trace) = valid
        .iter()
        .find(|(path, _)| name_of(path) == "context-freeze.json")
        .expect("context fixture");
    let frozen_at: std::collections::HashMap<String, String> = iter_envelopes(context_trace)
        .into_iter()
        .filter(|message| message.get("type").and_then(|t| t.as_str()) == Some("mode.routeFrozen"))
        .map(|message| {
            (
                message["payload"]["route"].as_str().unwrap().to_string(),
                message["ts"].as_str().unwrap().to_string(),
            )
        })
        .collect();
    assert_eq!(
        frozen_at.get("local-default").map(String::as_str),
        Some("2026-09-20T10:00:04Z")
    );
    let first_submit = valid
        .iter()
        .filter(|(_, trace)| trace.get("machine").and_then(|m| m.as_str()) == Some("jobs"))
        .flat_map(|(_, trace)| iter_envelopes(trace))
        .find(|message| message.get("type").and_then(|t| t.as_str()) == Some("jobs.submit"))
        .expect("first submit");
    let submit_ts = first_submit["ts"].as_str().unwrap();
    let route = first_submit["payload"]["route"].as_str().unwrap();
    let frozen_ts = frozen_at[route].as_str();
    assert!(
        submit_ts > frozen_ts,
        "submit {submit_ts} must follow freeze {frozen_ts}"
    );
}

#[test]
fn unknown_version_answers_nack() {
    let (_, invalid) = load_fixtures();
    let (_, trace) = invalid
        .iter()
        .find(|(path, _)| name_of(path) == "invalid-runtime-unknown-version.json")
        .expect("unknown-version fixture");
    let bad = iter_envelopes(trace)
        .into_iter()
        .next()
        .expect("the bad message");
    assert_eq!(bad["v"], serde_json::json!(2));
    assert!(!starling_runtime::protocol::envelope_errors(&bad).is_empty());

    let nack = nack_for(&bad).expect("v2 must be nacked");
    assert_eq!(nack["type"], "runtime.nack");
    assert_eq!(nack["payload"]["reason"], "unsupported_version");
    assert_eq!(nack["corr"], bad["id"]);
    assert!(
        starling_runtime::protocol::envelope_errors(&nack).is_empty(),
        "the nack itself must be schema-valid"
    );

    let ok = serde_json::json!({
        "v": 1, "id": "x", "ts": "2026-09-20T10:00:00Z",
        "type": "capture.stop", "payload": {}
    });
    assert!(nack_for(&ok).is_none());
}

#[test]
fn pending_command_requires_correlated_outcome() {
    let (valid, _) = load_fixtures();
    let (_, trace) = valid
        .iter()
        .find(|(path, _)| name_of(path) == "jobs-lifecycle.json")
        .expect("jobs fixture");
    let mut poisoned = trace.clone();
    for message in poisoned
        .get_mut("messages")
        .and_then(serde_json::Value::as_array_mut)
        .unwrap()
    {
        if message.get("type").and_then(|t| t.as_str()) == Some("jobs.queued") {
            message
                .as_object_mut()
                .unwrap()
                .insert("corr".into(), "wrong-stream".into());
            break;
        }
    }
    let violation = match replay_trace_of(&poisoned) {
        Err(violation) => violation,
        Ok(_) => panic!("corr mismatch must be flagged"),
    };
    assert_eq!(violation.code(), "corr_mismatch");
}

#[test]
fn directives_only_advance_internal_edges() {
    let (valid, _) = load_fixtures();
    for (_, trace) in &valid {
        for record in trace
            .get("messages")
            .and_then(serde_json::Value::as_array)
            .unwrap()
        {
            if is_directive(record) {
                let keys: Vec<&str> = record
                    .as_object()
                    .unwrap()
                    .keys()
                    .map(String::as_str)
                    .collect();
                assert_eq!(keys, vec!["$advance"]);
            }
        }
    }
    let (_, trace) = valid
        .iter()
        .find(|(path, _)| name_of(path) == "jobs-lifecycle.json")
        .expect("jobs fixture");
    let replayed = replay_trace_of(trace).expect("green");
    let advanced: Vec<&str> = replayed
        .transitions
        .iter()
        .filter(|t| t.kind == replay::TransitionKind::Internal)
        .map(|t| t.to.as_str())
        .collect();
    assert_eq!(
        advanced,
        vec!["Dispatched", "Loading", "Recognizing", "Dispatched"]
    );
}
