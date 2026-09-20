"""Conformance tests for the E17-I0 runtime protocol contract.

Two layers, mirroring tests/test_insight_events.py:

* Schema conformance — every fixture message validates against
  envelope.schema.json plus commands/events.schema.json (jsonschema when
  installed, minischema fallback otherwise; no new dependencies).
* Machine semantics — tests/runtime_protocol.py is the executable oracle the
  I3 runtime crate and the I4 IPC host must reproduce: replay every valid
  fixture trace green, reject every invalid fixture with its violation class,
  and hold the section-2 invariants (Draining independence from jobs, docs
  CAS, no delivery auto-send, route freeze before audio leaves, NACK on
  unknown envelope version).
"""

from __future__ import annotations

import copy
import json

import pytest

import minischema
import runtime_protocol as rp

try:
    import jsonschema
except ImportError:  # pragma: no cover - exercised only without the package
    jsonschema = None

COMMANDS = rp.COMMAND_KINDS
EVENTS = rp.EVENT_KINDS

VALID_FIXTURES, INVALID_FIXTURES = rp.load_fixtures()
VALID_BY_NAME = {path.name: trace for path, trace in VALID_FIXTURES}
INVALID_BY_NAME = {path.name: trace for path, trace in INVALID_FIXTURES}


# --------------------------------------------------------------------------- #
# Schema conformance helpers
# --------------------------------------------------------------------------- #
def schema_errors(msg: dict) -> list[str]:
    errors = minischema.errors(msg, rp.ENVELOPE_SCHEMA)
    kind_schema = rp.COMMAND_SCHEMA if msg["type"] in COMMANDS else rp.EVENT_SCHEMA
    errors += minischema.errors(msg, kind_schema)
    if jsonschema is not None:  # pragma: no branch - both paths run when present
        for schema in (rp.ENVELOPE_SCHEMA, kind_schema):
            validator = jsonschema.Draft202012Validator(schema)
            errors += [
                f"jsonschema: {e.message}" for e in validator.iter_errors(msg)
            ]
    return errors


def assert_valid(msg: dict) -> None:
    problems = schema_errors(msg)
    assert not problems, problems


def assert_invalid(msg: dict) -> None:
    assert schema_errors(msg), "expected schema rejection"


# --------------------------------------------------------------------------- #
# Fixture corpus shape
# --------------------------------------------------------------------------- #
def test_fixture_corpus_covers_all_machines() -> None:
    assert {trace["machine"] for _, trace in VALID_FIXTURES} == {
        "capture",
        "jobs",
        "context",
        "docs",
        "delivery",
    }
    assert INVALID_FIXTURES, "no invalid fixtures found"


# --------------------------------------------------------------------------- #
# Schema conformance
# --------------------------------------------------------------------------- #
def test_valid_fixture_messages_conform() -> None:
    for path, trace in VALID_FIXTURES:
        for msg in rp.iter_envelopes(trace):
            assert_valid(msg)


def test_invalid_fixtures_flagged_schema_invalid_where_claimed() -> None:
    for path, trace in INVALID_FIXTURES:
        if trace["schema_valid"]:
            for msg in rp.iter_envelopes(trace):
                assert_valid(msg)
        else:
            rejected = [
                msg for msg in rp.iter_envelopes(trace) if schema_errors(msg)
            ]
            assert rejected, f"{path.name}: expected a schema rejection"


def test_unknown_payload_field_rejected() -> None:
    msg = next(rp.iter_envelopes(VALID_BY_NAME["capture-happy.json"]))
    poisoned = copy.deepcopy(msg)
    poisoned["payload"]["extra"] = "no"
    assert_invalid(poisoned)


def test_unknown_envelope_field_rejected() -> None:
    msg = next(rp.iter_envelopes(VALID_BY_NAME["capture-happy.json"]))
    poisoned = copy.deepcopy(msg)
    poisoned["trace_id"] = "no"
    assert_invalid(poisoned)


def test_missing_required_envelope_field_rejected() -> None:
    msg = next(rp.iter_envelopes(VALID_BY_NAME["capture-happy.json"]))
    poisoned = copy.deepcopy(msg)
    del poisoned["ts"]
    assert_invalid(poisoned)
    assert rp.envelope_errors(poisoned)


# --------------------------------------------------------------------------- #
# Replay: valid traces green, invalid traces rejected with the right class
# --------------------------------------------------------------------------- #
def test_valid_traces_replay_green() -> None:
    for path, trace in VALID_FIXTURES:
        replay = rp.replay_trace(trace)
        assert replay.transitions, f"{path.name}: no transitions applied"


@pytest.mark.parametrize(
    "name",
    sorted(INVALID_BY_NAME),
)
def test_invalid_fixtures_fail_with_expected_violation(name: str) -> None:
    trace = INVALID_BY_NAME[name]
    expected = trace["replay"]
    if expected == "route_not_frozen":
        # Corpus-level ordering rule, not a single-machine rule.
        rp.replay_trace(trace)  # machine replay alone is green
        corpus = rp.valid_corpus_messages()
        for msg in rp.iter_envelopes(trace):
            corpus.append(msg)
        violations = rp.route_freeze_violations(corpus)
        assert violations and all(
            v["route"] == "route-never-frozen" for v in violations
        )
        return
    with pytest.raises(rp.ProtocolViolation) as excinfo:
        rp.replay_trace(trace)
    assert excinfo.value.code == expected, (
        f"{name}: expected {expected}, got {excinfo.value.code}"
    )


def test_valid_traces_seq_monotonic() -> None:
    for path, trace in VALID_FIXTURES:
        assert not rp.seq_violations(trace["messages"]), path.name


def test_every_named_state_visited() -> None:
    visited: dict[str, set[str]] = {}
    for _, trace in VALID_FIXTURES:
        replay = rp.replay_trace(trace)
        visited.setdefault(trace["machine"], set()).update(replay.visited)
    for machine, spec in rp.MACHINES.items():
        assert visited[machine] == set(spec["states"]), (
            f"{machine}: states never visited "
            f"{sorted(set(spec['states']) - visited[machine])}"
        )


# --------------------------------------------------------------------------- #
# Schema <-> oracle agreement
# --------------------------------------------------------------------------- #
def test_machine_tables_match_schema_kinds() -> None:
    schema_commands = {t for t in COMMANDS if not t.startswith("runtime.")}
    schema_events = {t for t in EVENTS if not t.startswith("runtime.")}
    oracle_commands: set[str] = set()
    oracle_events: set[str] = set()
    for spec in rp.MACHINES.values():
        oracle_commands |= set(spec["commands"])
        oracle_events |= set(spec["events"])
    assert oracle_commands == schema_commands
    assert oracle_events == schema_events


def test_internal_edges_stay_inside_declared_states() -> None:
    for machine, spec in rp.MACHINES.items():
        states = set(spec["states"])
        for from_, to in spec["internal"]:
            assert from_ in states and to in states, (machine, from_, to)


# --------------------------------------------------------------------------- #
# Invariant spot-tests (E17 section 2)
# --------------------------------------------------------------------------- #
def test_capture_never_blocks_draining_on_jobs() -> None:
    # No jobs.* message may be required to leave Draining (E17 AC): the exits
    # of every capture state are capture-owned only.
    for state in rp.MACHINES["capture"]["states"]:
        leaving = rp.exits("capture", state)
        assert not any(t.startswith("jobs.") for t in leaving), (state, leaving)
    draining_exits = rp.exits("capture", "Draining")
    assert "capture.stopped" in draining_exits
    assert not any(t.startswith("jobs.") for t in draining_exits)


def test_docs_head_update_is_cas_with_candidate_retained() -> None:
    trace = VALID_BY_NAME["docs-cas.json"]
    conflict = next(
        msg
        for msg in rp.iter_envelopes(trace)
        if msg["type"] == "docs.headConflict"
    )
    payload = conflict["payload"]
    assert payload["expected"] != payload["actual"]
    assert payload["candidatePreserved"] is True

    replay = rp.replay_trace(trace)
    conflict_edges = [
        t for t in replay.transitions if t["type"] == "docs.headConflict"
    ]
    assert conflict_edges == [
        {"kind": "event", "type": "docs.headConflict", "from": "Validating", "to": "Conflicted"}
    ]
    # And the retry against the actual base commits.
    assert any(
        t["type"] == "docs.headUpdated" and t["to"] == "Committed"
        for t in replay.transitions
    )


def test_delivery_has_no_auto_enter_or_auto_send_command() -> None:
    delivery_commands = {t for t in COMMANDS if t.startswith("delivery.")}
    assert delivery_commands == {
        "delivery.prepare",
        "delivery.apply",
        "delivery.cancel",
        "delivery.copyFallback",
    }
    forbidden = ("send", "enter", "auto", "inject", "press", "confirm")
    for type_ in delivery_commands:
        assert not any(word in type_ for word in forbidden), type_
    # apply is user-initiated and names the delivery it applies.
    apply_payload = rp.COMMAND_SCHEMA["$defs"]["payloads"]["delivery.apply"]
    assert apply_payload["required"] == ["deliveryId"]


def test_route_freeze_precedes_any_audio_leave() -> None:
    corpus = rp.valid_corpus_messages()
    assert not rp.route_freeze_violations(corpus)

    # mode.routeFrozen is only legal from ModeDecided — a decision must exist
    # before audio can leave on the frozen route.
    rule = rp.MACHINES["context"]["events"]["mode.routeFrozen"]
    assert rule["from"] == ["ModeDecided"]

    # The frozen route in the context fixture precedes the take that uses it.
    context_trace = VALID_BY_NAME["context-freeze.json"]
    frozen_at = {
        msg["payload"]["route"]: msg["ts"]
        for msg in rp.iter_envelopes(context_trace)
        if msg["type"] == "mode.routeFrozen"
    }
    assert frozen_at["local-default"] == "2026-09-20T10:00:04Z"
    first_submit = next(
        msg
        for _, trace in VALID_FIXTURES
        if trace["machine"] == "jobs"
        for msg in rp.iter_envelopes(trace)
        if msg["type"] == "jobs.submit"
    )
    assert first_submit["ts"] > frozen_at[first_submit["payload"]["route"]]


def test_unknown_version_answers_nack() -> None:
    bad = next(
        rp.iter_envelopes(INVALID_BY_NAME["invalid-runtime-unknown-version.json"])
    )
    assert bad["v"] == 2
    assert_invalid(bad)

    nack = rp.nack_for(bad)
    assert nack is not None
    assert nack["type"] == "runtime.nack"
    assert nack["payload"] == {"reason": "unsupported_version"}
    assert nack["corr"] == bad["id"]
    assert_valid(nack)  # the NACK itself is a v1 event

    assert rp.nack_for({"v": 1, "id": "x", "ts": "2026-09-20T10:00:00Z",
                        "type": "capture.stop", "payload": {}}) is None


def test_pending_command_requires_correlated_outcome() -> None:
    trace = copy.deepcopy(VALID_BY_NAME["jobs-lifecycle.json"])
    queued = next(
        m for m in trace["messages"] if m.get("type") == "jobs.queued"
    )
    queued["corr"] = "wrong-stream"
    with pytest.raises(rp.ProtocolViolation) as excinfo:
        rp.replay_trace(trace)
    assert excinfo.value.code == "corr_mismatch"


def test_directives_only_advance_internal_edges() -> None:
    for _, trace in VALID_FIXTURES:
        for record in trace["messages"]:
            if rp.is_directive(record):
                assert set(record) == {"$advance"}
    replay = rp.replay_trace(VALID_BY_NAME["jobs-lifecycle.json"])
    advanced = [t for t in replay.transitions if t["kind"] == "internal"]
    assert [t["to"] for t in advanced] == [
        "Dispatched",
        "Loading",
        "Recognizing",
        "Dispatched",
    ]
