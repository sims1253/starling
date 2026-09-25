"""Contract tests for staging and processing (#293).

Two layers, like tests/test_mode_routing.py:

* Semantics - tests/staging.py is the executable oracle for the staged
  draft (partials, user edits, command spans, proposals tied to a
  revision, delivery idempotency); every case in fixtures/staging.json
  replays green with the recorded outcome of every op, and the invariants
  (regions tile the text, attempts immutable, raw round-trips byte for
  byte, no current proposal on an old base) hold after every op. The Rust
  port (apps/desktop-gpui/crates/processing, tests/staging_conformance.rs)
  replays the same file.
* Schema conformance - draft snapshots after every op, provider
  declarations, transform requests and results validate against their
  schemas (jsonschema when installed, always minischema), and the
  cross-field rules in tests/mode_routing.py (validate_processing,
  validate_provider, processing_route, check_request, check_result) hold
  for the fixtures and reject poisoned copies.
"""

from __future__ import annotations

import copy
import json

import pytest

import minischema
import mode_routing as mr
import staging as st

try:
    import jsonschema
except ImportError:  # pragma: no cover - exercised only without the package
    jsonschema = None

FIXTURES = mr.FIXTURE_DIR

DRAFT_SCHEMA = mr.load_schema("draft.schema.json")
PROVIDER_SCHEMA = mr.load_schema("provider.schema.json")
REQUEST_SCHEMA = mr.load_schema("transform-request.schema.json")
RESULT_SCHEMA = mr.load_schema("transform-result.schema.json")
MODE_SCHEMA = mr.load_schema("mode.schema.json")

CASES = st.staging_cases()
PROVIDERS = mr.load_json(FIXTURES / "providers.json")
REQUESTS = mr.load_json(FIXTURES / "transform-requests.json")
RESULTS = mr.load_json(FIXTURES / "transform-results.json")
ROUTES = mr.load_json(FIXTURES / "processing-routes.json")
CONFIG = mr.load_profiles()
PROFILES = {p["id"]: p for p in CONFIG["profiles"]}
PROVIDERS_BY_ID = {p["id"]: p for p in PROVIDERS}

# The scenarios #293's acceptance names, each frozen by at least one case.
REQUIRED_SCENARIOS = {
    "revised partials": "revised_partials",
    "typing during a partial replacement": "typing_during_partial_replacement",
    "stale proposal": "stale_proposal",
    "concurrent retry": "concurrent_retry",
    "delete during processing": "delete_during_processing",
    "crash/reconnect": "crash_reconnect_result_still_current",
    "duplicate final delivery": "duplicate_final_delivery",
}


def schema_errors(instance, schema) -> list[str]:
    errors = list(minischema.errors(instance, schema))
    if jsonschema is not None:  # pragma: no branch
        validator = jsonschema.Draft202012Validator(schema)
        errors += [f"jsonschema: {e.message}" for e in validator.iter_errors(instance)]
    return errors


def assert_valid(instance, schema) -> None:
    problems = schema_errors(instance, schema)
    assert not problems, problems


def assert_invalid(instance, schema) -> None:
    assert schema_errors(instance, schema), "expected schema rejection"


def api_profile() -> dict:
    """The remote twin of the fixture's capture mode (req-2 names it)."""
    profile = copy.deepcopy(PROFILES["capture"])
    profile.update(id="capture-api", authoring_route="remote-authoring-openai",
                   local_only=False, context_fields=["vocabulary"])
    return profile


# --------------------------------------------------------------------------- #
# Staging semantics
# --------------------------------------------------------------------------- #
def test_required_scenarios_are_frozen() -> None:
    names = {case["name"] for case in CASES}
    for scenario, name in REQUIRED_SCENARIOS.items():
        assert name in names, f"no fixture case for {scenario}"
    assert len(names) == len(CASES), "case names are unique"


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_staging_case_replays(case) -> None:
    draft, outcomes, violations = st.run_case(case)
    assert not violations, violations
    assert all("expect" in op for op in case["ops"]), "every op records its outcome"
    mismatches = st.expectation_mismatches(draft, case["expect"])
    assert not mismatches, mismatches


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_every_snapshot_conforms(case) -> None:
    draft = st.Draft("draft-1", "cap-1")
    assert_valid(draft.snapshot(), DRAFT_SCHEMA)
    for op in case["ops"]:
        draft.apply(op)
        assert_valid(draft.snapshot(), DRAFT_SCHEMA)


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_raw_round_trips_through_serialization(case) -> None:
    draft, _, _ = st.run_case(case)
    wire = json.dumps(draft.snapshot(), ensure_ascii=False).encode("utf-8")
    back = json.loads(wire.decode("utf-8"))
    finals = {op["attempt_id"]: op["text"] for op in reversed(case["ops"]) if op["op"] == "final"}
    for attempt in back["attempts"]:
        assert attempt["text"].encode("utf-8") == finals[attempt["attempt_id"]].encode("utf-8")
    assert back["text"] == draft.text()


def test_a_result_never_changes_the_draft_by_itself() -> None:
    for case in CASES:
        draft = st.Draft("draft-1", "cap-1")
        for op in case["ops"]:
            before = (draft.text(), draft.revision)
            draft.apply(op)
            if op["op"] in ("result", "request", "cancel", "reject", "deliver", "crash") and not draft.deleted:
                if op["op"] == "crash":
                    continue  # a crash may drop live partials, never add text
                assert (draft.text(), draft.revision) == before, (case["name"], op)


def test_unknown_op_is_an_error() -> None:
    with pytest.raises(ValueError):
        st.Draft("d", "c").apply({"op": "execute"})


# --------------------------------------------------------------------------- #
# Mode entries: processing fields
# --------------------------------------------------------------------------- #
def test_profiles_documents_pass_processing_rules() -> None:
    for path in sorted(FIXTURES.glob("profiles*.json")):
        mr.validate_processing(mr.load_json(path))


@pytest.mark.parametrize("patch, message", [
    ({"id": "faithful", "delivery": "insert_enter"}, "default"),
    ({"id": "verbatim", "transform_kinds": ["clean"], "authoring_route": "local-authoring-default"}, "verbatim"),
    ({"id": "verbatim", "spoken_commands": True}, "verbatim"),
    ({"id": "capture", "authoring_route": None}, "authoring route"),
    ({"id": "code-guidance", "style": {"formality": "formal", "structure": "prose", "context": "general"}}, "style"),
    ({"id": "edit-selection", "delivery": "insert_enter"}, "selection"),
    ({"id": "capture", "transform_kinds": ["clean", "clean"]}, "duplicate"),
])
def test_processing_rules_reject(patch, message) -> None:
    config = copy.deepcopy(CONFIG)
    profile = next(p for p in config["profiles"] if p["id"] == patch["id"])
    profile.update(patch)
    with pytest.raises(ValueError, match=message):
        mr.validate_processing(config)


def test_insert_enter_is_a_valid_opt_in_delivery() -> None:
    config = copy.deepcopy(CONFIG)
    profile = next(p for p in config["profiles"] if p["id"] == "code-guidance")
    profile["delivery"] = "insert_enter"
    mr.validate_processing(config)
    assert_valid(profile, MODE_SCHEMA)


def test_bad_style_axis_rejected() -> None:
    profile = copy.deepcopy(PROFILES["capture"])
    profile["style"]["formality"] = "shouting"
    assert_invalid(profile, MODE_SCHEMA)


def test_bad_transform_kind_rejected() -> None:
    profile = copy.deepcopy(PROFILES["capture"])
    profile["transform_kinds"] = ["execute"]
    assert_invalid(profile, MODE_SCHEMA)


# --------------------------------------------------------------------------- #
# Providers and the processing route
# --------------------------------------------------------------------------- #
def test_providers_conform() -> None:
    for provider in PROVIDERS:
        assert_valid(provider, PROVIDER_SCHEMA)
        mr.validate_provider(provider)


def test_provider_locality_must_match_route() -> None:
    provider = copy.deepcopy(PROVIDERS_BY_ID["remote-openai"])
    provider["locality"] = "local"
    with pytest.raises(ValueError, match="locality"):
        mr.validate_provider(provider)


def test_rewrite_provider_needs_instructions() -> None:
    provider = copy.deepcopy(PROVIDERS_BY_ID["local-s1"])
    provider["transform_kinds"] = ["clean", "rewrite"]
    with pytest.raises(ValueError, match="instructions"):
        mr.validate_provider(provider)


@pytest.mark.parametrize("case", ROUTES, ids=[c["name"] for c in ROUTES])
def test_processing_route_cases(case) -> None:
    profile = copy.deepcopy(PROFILES[case["profile"]])
    profile.update(case.get("patch", {}))
    assert_valid(profile, MODE_SCHEMA)
    assert mr.processing_route(profile, PROVIDERS) == case["expect"]


def test_local_only_never_falls_back() -> None:
    # Even with a local provider available for another route, a local_only
    # mode pointed at a remote route is blocked, not rerouted.
    profile = copy.deepcopy(PROFILES["capture"])
    profile["authoring_route"] = "remote-authoring-openai"
    route = mr.processing_route(profile, PROVIDERS)
    assert route["status"] == "blocked" and route["provider"] is None


# --------------------------------------------------------------------------- #
# Requests and results
# --------------------------------------------------------------------------- #
def request_profile(request) -> dict:
    return PROFILES["capture"] if request["mode_id"] == "capture" else api_profile()


def test_requests_conform() -> None:
    for request in REQUESTS:
        assert_valid(request, REQUEST_SCHEMA)
        provider = PROVIDERS_BY_ID[request["provider"]["id"]]
        assert mr.check_request(request, request_profile(request), provider) == []


@pytest.mark.parametrize("patch, message", [
    ({"local_only": True}, "local_only"),
    ({"context": {"personal_context": "I am Max"}}, "context field personal_context"),
    ({"instruction": "make it formal"}, "instruction"),
    ({"language": "de"}, "language"),
])
def test_request_cross_field_rules(patch, message) -> None:
    request = copy.deepcopy(REQUESTS[1])
    request.update(patch)
    problems = mr.check_request(request, api_profile(), PROVIDERS_BY_ID["remote-openai"])
    assert any(message in p for p in problems), problems


def test_unknown_request_field_rejected() -> None:
    request = copy.deepcopy(REQUESTS[0])
    request["execute"] = True
    assert_invalid(request, REQUEST_SCHEMA)


def test_request_input_is_data_not_a_channel() -> None:
    # The fixture's API request carries a prompt injection as its input:
    # the contract has no field through which input could name a tool, an
    # endpoint or a provider, so there is nothing for it to reach.
    names = set(REQUEST_SCHEMA["properties"])
    assert not {n for n in names if any(f in n for f in ("tool", "url", "endpoint", "command", "script"))}
    assert "ignore previous instructions" in REQUESTS[1]["input"]


def test_results_conform() -> None:
    for result in RESULTS:
        assert_valid(result, RESULT_SCHEMA)
        assert mr.check_result(result) == []


def test_empty_completed_output_is_valid() -> None:
    assert any(r["status"] == "completed" and r["text"] == "" for r in RESULTS)


@pytest.mark.parametrize("patch", [
    {"status": "completed", "text": None},
    {"status": "failed", "text": "partial words"},
    {"status": "cancelled", "failure": {"reason": "timeout", "retryable": True, "detail": ""}},
])
def test_result_cross_field_rules(patch) -> None:
    result = copy.deepcopy(RESULTS[2])
    result.update(patch)
    assert mr.check_result(result)


def test_bad_failure_reason_rejected() -> None:
    result = copy.deepcopy(RESULTS[2])
    result["failure"]["reason"] = "reasoning_stripped"
    assert_invalid(result, RESULT_SCHEMA)


def test_result_timing_carries_no_text() -> None:
    timing = RESULT_SCHEMA["properties"]["timing"]
    assert timing["additionalProperties"] is False
    assert set(timing["properties"]) == {"queued_ms", "processing_ms", "stop_to_result_ms"}
