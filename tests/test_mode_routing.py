"""Contract tests for the Starling mode-routing contract (E18/E19/E26).

Two layers, mirroring tests/test_runtime_protocol.py:

* Machine semantics - tests/mode_routing.py is the executable oracle (a
  line-faithful port of the review package's reference/router.py) that the
  native runtime must reproduce: every routing fixture replays green
  (leading-phrase matching, locked one-take choices, literal escape,
  quoted/mid-sentence stays literal, longest-alias and scope-specificity
  precedence, ambiguity conflicts, selection lease decisions).
* Schema conformance - every profiles document validates against
  profiles.schema.json, every mode entry against mode.schema.json, every
  context snapshot against context-snapshot.schema.json and every decision
  record (static fixtures AND every decision the oracle emits for every
  routing case) against decision.schema.json. Validation uses jsonschema
  when installed and always also the self-contained minischema fallback;
  no new dependencies.

Frozen invariants under test:

- raw preservation: the routed payload is a view; raw text equals the
  payload with the recorded removed prefix span reattached, and the raw
  attempt is referenced, never mutated;
- the decision output is structurally INCAPABLE of expressing a provider,
  route or permission change (a phrase recognized after capture cannot
  broaden anything);
- verbatim modes disable spoken aliases by default;
- the deterministic precedence order of MODES_AND_CONTEXT.md.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import minischema
import mode_routing as mr

try:
    import jsonschema
except ImportError:  # pragma: no cover - exercised only without the package
    jsonschema = None

REPO = Path(__file__).resolve().parents[1]
CONTRACT = REPO / "packages" / "contracts" / "mode-routing"
FIXTURES = CONTRACT / "fixtures"

MODE_SCHEMA = mr.load_schema("mode.schema.json")
PROFILES_SCHEMA = mr.load_schema("profiles.schema.json")
DECISION_SCHEMA = mr.load_schema("decision.schema.json")
SNAPSHOT_SCHEMA = mr.load_schema("context-snapshot.schema.json")

CONFIG = mr.load_profiles()
ROUTING_CASES = mr.routing_cases()
SELECTION_CASES = mr.selection_cases()
DECISIONS = mr.load_json(FIXTURES / "decisions.json")
SNAPSHOTS = [
    mr.load_json(FIXTURES / "context-snapshot.json"),
    mr.load_json(FIXTURES / "context-snapshot-secure.json"),
]
PROFILES_DOCS = sorted(FIXTURES.glob("profiles*.json"))

# The 25 reference routing cases, in reference order (routing.json is the
# byte-faithful port; this freezes the port).
REFERENCE_ROUTING_NAMES = [
    "default",
    "vscode_rule",
    "notes_rule_is_literal",
    "leading_alias",
    "leading_whitespace",
    "mid_sentence",
    "quoted_prefix",
    "boundary",
    "literal_escape",
    "locked_manual",
    "locked_faithful",
    "manual_unlocked",
    "required_selection_missing",
    "required_selection_ready",
    "selection_not_granted",
    "ask_no_replace",
    "alias_only",
    "empty",
    "private_target",
    "site_rule",
    "lookalike_site",
    "session_aliases_off",
    "capture",
    "verbatim_manual_unlocked_still_literal",
    "numeric_list",
]


# --------------------------------------------------------------------------- #
# Schema conformance helpers (both validators, like test_runtime_protocol)
# --------------------------------------------------------------------------- #
def schema_errors(instance: dict, schema: dict) -> list[str]:
    errors = list(minischema.errors(instance, schema))
    if jsonschema is not None:  # pragma: no branch - both paths run when present
        validator = jsonschema.Draft202012Validator(schema)
        errors += [
            f"jsonschema: {e.message}" for e in validator.iter_errors(instance)
        ]
    return errors


def assert_valid(instance: dict, schema: dict) -> None:
    problems = schema_errors(instance, schema)
    assert not problems, problems


def assert_invalid(instance: dict, schema: dict) -> None:
    assert schema_errors(instance, schema), "expected schema rejection"


def iter_property_names(node) -> "iter[str]":
    """Every property name in a schema, recursing into properties/items."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "properties" and isinstance(value, dict):
                yield from value
            yield from iter_property_names(value)
    elif isinstance(node, list):
        for item in node:
            yield from iter_property_names(item)


# --------------------------------------------------------------------------- #
# Routing behavior (port of reference/test_behavior.py Routing)
# --------------------------------------------------------------------------- #
def test_reference_routing_fixture_names_frozen() -> None:
    ported = mr.load_json(FIXTURES / "routing.json")
    assert [case["name"] for case in ported] == REFERENCE_ROUTING_NAMES


@pytest.mark.parametrize(
    "name", [case["name"] for _, _, case in ROUTING_CASES]
)
def test_reference_routing_fixture_cases(name: str) -> None:
    for config_name, config, case in ROUTING_CASES:
        if case["name"] == name:
            result = mr.resolve(config, case["request"])
            assert {k: result[k] for k in case["expected"]} == case["expected"], (
                f"{config_name}/{name}"
            )
            return
    pytest.fail(f"case {name} not found")  # pragma: no cover


def test_collision_is_not_a_guess() -> None:
    config = copy.deepcopy(CONFIG)
    config["profiles"][0]["aliases"] = ["code this"]
    assert mr.resolve(config, {"raw_text": "code this: now"})["status"] == "needs_resolution"


def test_longest_alias() -> None:
    config = copy.deepcopy(CONFIG)
    config["profiles"][4]["aliases"] = ["code this carefully"]
    result = mr.resolve(config, {"raw_text": "code this carefully: explain",
                                 "selection_available": True, "selection_granted": True})
    assert result["mode"] == "ask-selection"
    assert result["payload"] == "explain"


def test_equal_rule_conflict() -> None:
    config = copy.deepcopy(CONFIG)
    config["rules"].append({"id": "conflict", "profile_id": "faithful",
                            "priority": 20, "app_id": "vscode"})
    assert mr.resolve(config, {"raw_text": "hello", "app_id": "vscode"})["source"] == "rule_conflict"


def test_project_overrides_app() -> None:
    config = copy.deepcopy(CONFIG)
    config["rules"].append({"id": "project", "profile_id": "verbatim",
                            "priority": 0, "project_id": "notes-project"})
    assert mr.resolve(config, {"raw_text": "hello", "app_id": "vscode",
                               "project_id": "notes-project"})["mode"] == "verbatim"


def test_unknown_mode_errors() -> None:
    with pytest.raises(ValueError):
        mr.resolve(CONFIG, {"raw_text": "hello", "manual_mode": "missing"})


def test_raw_is_immutable() -> None:
    text = "  CODE THIS: never drop auth."
    result = mr.resolve(CONFIG, {"raw_text": text})
    assert result["raw_text"] == text
    assert text[result["prefix_span_codepoints"][1]:] == result["payload"]


def test_local_remote_conflict() -> None:
    config = copy.deepcopy(CONFIG)
    config["profiles"][0]["asr_route"] = "remote-asr"
    with pytest.raises(ValueError):
        mr.validate_config(config)


def test_reference_cannot_replace() -> None:
    config = copy.deepcopy(CONFIG)
    config["profiles"][2]["delivery"] = "replace_selection"
    with pytest.raises(ValueError):
        mr.validate_config(config)


# --------------------------------------------------------------------------- #
# Selection lease behavior (port of reference/test_behavior.py Selection)
# --------------------------------------------------------------------------- #
def test_reference_selection_fixture_cases() -> None:
    assert len(SELECTION_CASES) == 4
    for item in SELECTION_CASES:
        assert mr.selection_decision(
            item["snapshot"], item["current"], item["role"], item["conditional_apply"]
        ) == item["expected"]


def test_app_change_conflict() -> None:
    item = copy.deepcopy(SELECTION_CASES[0])
    item["current"]["app"] = "mail"
    assert mr.selection_decision(item["snapshot"], item["current"],
                                 "edit_target", True) == "conflict"


def test_unknown_version_needs_preview() -> None:
    item = copy.deepcopy(SELECTION_CASES[0])
    item["current"]["version"] = None
    assert mr.selection_decision(item["snapshot"], item["current"],
                                 "edit_target", True) == "preview"


def test_secure_target_blocked() -> None:
    item = copy.deepcopy(SELECTION_CASES[0])
    item["current"]["secure_field"] = True
    assert mr.selection_decision(item["snapshot"], item["current"],
                                 "edit_target", True) == "blocked"


def test_offset_encoding_mismatch_conflict() -> None:
    item = copy.deepcopy(SELECTION_CASES[0])
    item["current"]["offset_encoding"] = "utf8_bytes"
    assert mr.selection_decision(item["snapshot"], item["current"],
                                 "edit_target", True) == "conflict"


# --------------------------------------------------------------------------- #
# Schema conformance: profiles documents and mode entries
# --------------------------------------------------------------------------- #
def test_profiles_documents_conform() -> None:
    assert len(PROFILES_DOCS) == 5  # canonical + 4 variants
    for path in PROFILES_DOCS:
        assert_valid(mr.load_json(path), PROFILES_SCHEMA)


def test_mode_entries_conform_to_mode_schema() -> None:
    for path in PROFILES_DOCS:
        for entry in mr.load_json(path)["profiles"]:
            assert_valid(entry, MODE_SCHEMA)


def test_mode_schema_root_matches_profiles_items() -> None:
    # Anti-drift: the per-entry model is one definition, embedded in both.
    mode_core = {k: v for k, v in MODE_SCHEMA.items()
                 if k not in ("$schema", "$id", "$comment")}
    items = PROFILES_SCHEMA["properties"]["profiles"]["items"]
    assert mode_core == items


def test_unknown_mode_entry_field_rejected() -> None:
    poisoned = copy.deepcopy(CONFIG["profiles"][0])
    poisoned["elevate"] = "no"
    assert_invalid(poisoned, MODE_SCHEMA)


def test_bad_behavior_enum_rejected() -> None:
    poisoned = copy.deepcopy(CONFIG["profiles"][0])
    poisoned["behavior"] = "execute_shell"
    assert_invalid(poisoned, MODE_SCHEMA)


def test_missing_mode_version_rejected() -> None:
    poisoned = copy.deepcopy(CONFIG["profiles"][0])
    del poisoned["version"]
    assert_invalid(poisoned, MODE_SCHEMA)


def test_rule_without_scope_rejected() -> None:
    poisoned = copy.deepcopy(CONFIG)
    poisoned["rules"].append({"id": "scopeless", "profile_id": "faithful",
                              "priority": 1})
    assert_invalid(poisoned, PROFILES_SCHEMA)


# --------------------------------------------------------------------------- #
# Schema conformance: context snapshots
# --------------------------------------------------------------------------- #
def test_context_snapshots_conform() -> None:
    for snapshot in SNAPSHOTS:
        assert_valid(snapshot, SNAPSHOT_SCHEMA)


def test_secure_snapshot_cannot_carry_selected_text() -> None:
    poisoned = copy.deepcopy(SNAPSHOTS[1])
    poisoned["selected_text"] = "secret"
    assert_invalid(poisoned, SNAPSHOT_SCHEMA)


def test_snapshot_missing_expiry_rejected() -> None:
    poisoned = copy.deepcopy(SNAPSHOTS[0])
    del poisoned["expires_at"]
    assert_invalid(poisoned, SNAPSHOT_SCHEMA)


def test_snapshot_digest_shape_enforced() -> None:
    poisoned = copy.deepcopy(SNAPSHOTS[0])
    poisoned["target"]["content_sha256"] = "not-a-sha"
    assert_invalid(poisoned, SNAPSHOT_SCHEMA)


# --------------------------------------------------------------------------- #
# Schema conformance: decision records
# --------------------------------------------------------------------------- #
def test_decision_fixtures_conform() -> None:
    assert len(DECISIONS) == 12
    for record in DECISIONS:
        assert_valid(record, DECISION_SCHEMA)


def test_unknown_decision_field_rejected() -> None:
    poisoned = copy.deepcopy(DECISIONS[0])
    poisoned["provider"] = "remote-asr"
    assert_invalid(poisoned, DECISION_SCHEMA)


def test_conflict_candidate_minimum_enforced() -> None:
    poisoned = copy.deepcopy(next(r for r in DECISIONS if r["conflicts"]))
    poisoned["conflicts"][0]["candidates"] = poisoned["conflicts"][0]["candidates"][:1]
    assert_invalid(poisoned, DECISION_SCHEMA)


@pytest.mark.parametrize(
    "name", [case["name"] for _, _, case in ROUTING_CASES]
)
def test_every_routing_decision_output_conforms(name: str) -> None:
    for config_name, config, case in ROUTING_CASES:
        if case["name"] == name:
            result = mr.resolve(config, case["request"])
            decision = mr.decision_from_resolve(
                config, case["request"], result,
                decision_id=f"decision-{name}", capture_id=f"take-{name}",
                raw_attempt_id=f"attempt-{name}",
            )
            assert_valid(decision, DECISION_SCHEMA)
            # conflicts appear exactly for ambiguous equal-rank outcomes
            if result["status"] == "needs_resolution":
                assert decision["source"] == "conflict"
                assert decision["conflicts"], f"{name}: conflict must list candidates"
                assert all(len(c["candidates"]) >= 2 for c in decision["conflicts"])
            else:
                assert decision["conflicts"] == []
                # mode is chosen exactly when a mode was routed to; blocked
                # decisions carry mode null by contract
                if result["status"] == "blocked":
                    assert decision["mode_id"] is None
                    assert decision["source"] == "policy"
                else:
                    assert decision["mode_id"] is not None
            return
    pytest.fail(f"case {name} not found")  # pragma: no cover


# --------------------------------------------------------------------------- #
# Frozen invariant: raw preserved, payload is a prefix-removed view
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "name", [case["name"] for _, _, case in ROUTING_CASES]
)
def test_raw_preserved_and_payload_view_prefix_removal(name: str) -> None:
    for config_name, config, case in ROUTING_CASES:
        if case["name"] == name:
            raw = case["request"]["raw_text"]
            result = mr.resolve(config, case["request"])
            # The blocked/needs_resolution early returns (reference shape)
            # carry no raw_text echo; every other path echoes it unchanged.
            if "raw_text" in result:
                assert result["raw_text"] == raw
            span = result.get("prefix_span_codepoints")
            if span is None:
                assert result["payload"] == raw
            else:
                assert span[0] == 0
                assert raw == raw[: span[1]] + result["payload"]
            decision = mr.decision_from_resolve(
                config, case["request"], result,
                decision_id=f"decision-{name}", capture_id=f"take-{name}",
                raw_attempt_id=f"attempt-{name}",
            )
            view = decision["payload_view"]
            assert view["raw_preserved"] is True
            assert view["text"] == result["payload"]
            assert view["removed_prefix_span"] == decision["matched_prefix_span"]
            if span is None:
                assert view["text"] == raw
            else:
                assert view["text"] == raw[span[1]:]
            return
    pytest.fail(f"case {name} not found")  # pragma: no cover


# --------------------------------------------------------------------------- #
# Frozen invariant: a decision cannot broaden providers/permissions
# --------------------------------------------------------------------------- #
FORBIDDEN_FRAGMENTS = (
    "provider", "permission", "permiss", "grant", "allow", "route", "scope",
    "network", "remote", "upload", "credential", "secret", "key",
    "entitlement", "consent", "authoriz", "elevat", "access", "trust",
)

DECISION_ROOT_FIELDS = {
    "schema_version", "decision_id", "capture_id", "raw_attempt_id",
    "mode_id", "mode_version", "source", "status", "matched_prefix_span",
    "span_encoding", "explanation", "payload_view", "conflicts",
    "context_snapshot_id",
}
PAYLOAD_VIEW_FIELDS = {"raw_preserved", "text", "removed_prefix_span"}
CONFLICT_FIELDS = {"kind", "candidates"}
CANDIDATE_FIELDS = {"mode_id", "via"}


def test_decision_field_vocabulary_is_whitelisted() -> None:
    props = DECISION_SCHEMA["properties"]
    assert set(props) == DECISION_ROOT_FIELDS
    assert set(props["payload_view"]["properties"]) == PAYLOAD_VIEW_FIELDS
    conflict_items = props["conflicts"]["items"]
    assert set(conflict_items["properties"]) == CONFLICT_FIELDS
    candidates = conflict_items["properties"]["candidates"]["items"]
    assert set(candidates["properties"]) == CANDIDATE_FIELDS
    # closed records: no silently ignored extras anywhere
    assert DECISION_SCHEMA["additionalProperties"] is False
    assert props["payload_view"]["additionalProperties"] is False
    assert conflict_items["additionalProperties"] is False
    assert candidates["additionalProperties"] is False


def test_decision_schema_cannot_express_provider_or_permission_change() -> None:
    # E26: a recognized prefix cannot retroactively authorize additional
    # context or providers. Expressed structurally: no field name in the
    # decision schema is capable of carrying such a change, and the record
    # vocabulary is closed (additionalProperties false everywhere, checked
    # above), so no such field can be added silently either.
    names = list(iter_property_names(DECISION_SCHEMA))
    assert names
    for name in names:
        lowered = name.lower()
        for fragment in FORBIDDEN_FRAGMENTS:
            assert fragment not in lowered, (
                f"decision field {name!r} can express a provider/permission "
                f"concern ({fragment!r})"
            )


def test_snapshot_cannot_encode_permission_elevation() -> None:
    # The snapshot is a frozen data-only record of activation-time facts.
    # The only grant-shaped field is granted_sources, and it is an
    # enum-constrained list of source KINDS - it cannot name, add or expand
    # grant targets.
    granted = SNAPSHOT_SCHEMA["$defs"]["grantedSources"]
    assert "enum" in granted["items"]
    for name in iter_property_names(SNAPSHOT_SCHEMA):
        lowered = name.lower()
        if "grant" in lowered:
            assert name == "granted_sources"
        assert not lowered.startswith(("additional", "extra", "new_")), name


# --------------------------------------------------------------------------- #
# Frozen invariant: verbatim modes disable spoken aliases by default
# --------------------------------------------------------------------------- #
def test_verbatim_modes_disable_aliases_by_default() -> None:
    for path in PROFILES_DOCS:
        for entry in mr.load_json(path)["profiles"]:
            if entry["behavior"] == "verbatim":
                assert entry["allow_spoken_overrides"] is False, (
                    f"{path.name}/{entry['id']}: verbatim must ship alias-locked"
                )
                assert entry["aliases"] == [], (
                    f"{path.name}/{entry['id']}: verbatim ships no aliases"
                )


# --------------------------------------------------------------------------- #
# Frozen invariant: deterministic precedence order
# --------------------------------------------------------------------------- #
def test_frozen_precedence_order() -> None:
    # 1. policy beats even a locked manual choice
    assert mr.resolve(CONFIG, {"raw_text": "hello", "secure_field": True,
                               "manual_mode": "verbatim"})["status"] == "blocked"
    # 2. a locked manual choice beats escape and aliases (text unparsed)
    assert mr.resolve(CONFIG, {"raw_text": "literal code this stays",
                               "manual_mode": "verbatim"})["source"] == "manual"
    # 3. in an alias-enabled session the literal escape beats aliases
    assert mr.resolve(CONFIG, {"raw_text": "literal code this is an example"}
                      )["source"] == "escape:literal"
    # 4. an unambiguous leading alias beats the frozen app rule
    assert mr.resolve(CONFIG, {"raw_text": "code this: add logs",
                               "app_id": "vscode"})["source"] == "phrase:code this"
    # 5. a frozen rule beats the default
    assert mr.resolve(CONFIG, {"raw_text": "hello",
                               "app_id": "vscode"})["source"] == "rule:vscode-guidance"
    # 6. scope specificity: site beats app despite lower priority...
    assert mr.resolve(CONFIG, {"raw_text": "hello", "site": "github.com",
                               "app_id": "vscode"})["source"] == "rule:github-guidance"
    # ...and project beats both
    config = copy.deepcopy(CONFIG)
    config["rules"].append({"id": "project", "profile_id": "verbatim",
                            "priority": 0, "project_id": "notes-project"})
    assert mr.resolve(config, {"raw_text": "hello", "site": "github.com",
                               "app_id": "vscode",
                               "project_id": "notes-project"})["source"] == "rule:project"


# --------------------------------------------------------------------------- #
# Snapshot expiry
# --------------------------------------------------------------------------- #
def test_snapshot_expires_after_capture_and_expires() -> None:
    for snapshot in SNAPSHOTS:
        assert snapshot["expires_at"] > snapshot["captured_at"]
        assert mr.snapshot_expired(snapshot, snapshot["captured_at"]) is False
        assert mr.snapshot_expired(snapshot, snapshot["expires_at"]) is False
        assert mr.snapshot_expired(snapshot, "2999-01-01T00:00:00Z") is True
