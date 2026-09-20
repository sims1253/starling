"""Contract tests for the Starling Insight event schema and metric oracle (E28).

Two layers:

* Schema conformance — every fixture event validates against
  ``packages/contracts/insight-events/schema.json``, and events carrying
  raw-text/selection/path/window-title/secret-shaped extra fields are
  REJECTED (the privacy contract is structural, not advisory). Validation
  uses ``jsonschema`` when installed and falls back to the self-contained
  ``minischema`` structural validator otherwise; no new dependencies.
* Metric semantics — ``insight_metrics.aggregate`` is the executable
  contract the E17 native runtime must reproduce: dedupe across retries and
  sync replays, speech-vs-generated separation, deletion propagation,
  timezone grouping, reset/export and negative time-saved proxy cases.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import minischema
from insight_metrics import activity_by_day, aggregate, local_wall_time

try:
    import jsonschema
except ImportError:  # pragma: no cover - exercised only without the package
    jsonschema = None

REPO = Path(__file__).resolve().parents[1]
CONTRACT = REPO / "packages" / "contracts" / "insight-events"
SCHEMA = json.loads((CONTRACT / "schema.json").read_text())
FIXTURES = sorted((CONTRACT / "fixtures").glob("*.json"))


def assert_valid(instance):
    if jsonschema is not None:
        jsonschema.Draft202012Validator(SCHEMA).validate(instance)
    else:
        problems = minischema.errors(instance, SCHEMA)
        assert not problems, problems


def assert_invalid(instance):
    if jsonschema is not None:
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.Draft202012Validator(SCHEMA).validate(instance)
    else:
        assert minischema.errors(instance, SCHEMA), "expected schema rejection"


def load(name: str) -> list[dict]:
    return json.loads((CONTRACT / "fixtures" / name).read_text())


BASIC = load("basic-session.json")


# --------------------------------------------------------------------------- #
# Schema conformance
# --------------------------------------------------------------------------- #
def test_every_fixture_event_conforms() -> None:
    assert FIXTURES, "fixture directory unexpectedly empty"
    for fixture in FIXTURES:
        events = json.loads(fixture.read_text())
        for event in events:
            assert_valid(event)


def test_missing_required_field_rejected() -> None:
    event = copy.deepcopy(BASIC[0])
    del event["sample_rate"]
    assert_invalid(event)


def test_unknown_event_type_rejected() -> None:
    event = copy.deepcopy(BASIC[0])
    event["type"] = "telemetry_uploaded"
    assert_invalid(event)


def test_unknown_delivery_status_rejected() -> None:
    event = copy.deepcopy(BASIC[4])  # delivery_recorded
    event["status"] = "probably_inserted"
    assert_invalid(event)


@pytest.mark.parametrize(
    "field",
    [
        "transcript",
        "raw_text",
        "selected_text",
        "clipboard_content",
        "file_path",
        "window_title",
        "app_title",
        "api_key",
        "oauth_token",
        "url",
    ],
)
def test_privacy_forbidden_extra_fields_rejected(field: str) -> None:
    # One representative event per kind: no event type may grow a free-form
    # string field carrying raw text, selections, paths, titles or secrets.
    samples = {
        e["type"]: e for e in BASIC if e["type"] != "recognition_selected"
    }
    selected = next(e for e in BASIC if e["type"] == "recognition_selected")
    samples["recognition_selected"] = selected
    for event in samples.values():
        poisoned = copy.deepcopy(event)
        poisoned[field] = "C:/Users/me/secret-diary.txt — never share this"
        assert_invalid(poisoned)


# --------------------------------------------------------------------------- #
# Frozen metric semantics
# --------------------------------------------------------------------------- #
def test_basic_session_counts() -> None:
    result = aggregate(BASIC, 50)
    assert result["formula_version"] == 1
    assert result["unique_takes"] == 2
    assert result["selected_recognitions"] == 2
    assert result["recognized_words"] == 110  # latest selection only
    assert result["raw_recognized_words"] == 114
    assert result["captured_seconds"] == pytest.approx(50.0)
    assert result["eligible_capture_seconds"] == pytest.approx(50.0)
    assert result["recognized_words_per_captured_minute"] == pytest.approx(132.0)
    assert result["typing_time_comparison_seconds"] == pytest.approx(75.0)
    assert result["delivery_counts"]["confirmed"] == 1
    assert result["delivery_counts"]["submitted_unconfirmed"] == 1
    assert result["output_words_by_status"]["confirmed"] == 120
    assert result["generated_words_by_status"]["confirmed"] == 20
    assert result["output_words_by_status"]["submitted_unconfirmed"] == 200
    assert result["generated_words_by_status"]["submitted_unconfirmed"] == 190


def test_wpm_is_weighted_total_not_per_take_average() -> None:
    # take-1: 100 words / 40s = 150 wpm; take-2: 10 / 10s = 60 wpm.
    # The contract reports the weighted total 110*60/50 = 132, never the
    # mean of per-take rates (105).
    result = aggregate(BASIC)
    assert result["recognized_words_per_captured_minute"] == pytest.approx(132.0)


def test_sync_replay_and_reordering_do_not_double_count() -> None:
    replay = load("sync-replay.json")
    assert aggregate(replay) == aggregate(BASIC)
    assert aggregate(list(reversed(BASIC))) == aggregate(BASIC)


def test_retry_replaces_rather_than_adds() -> None:
    # take-1 has two selections (90 then 100 lexical words); only 100 counts.
    assert aggregate(BASIC)["recognized_words"] == 110


def test_deletion_removes_attributable_events_and_beats_stale_replay() -> None:
    deleted = load("deletion-propagation.json")
    # take-3 (30 words, 5s, one confirmed delivery) is tombstoned; the file
    # also replays take-3's recognition AFTER the tombstone. Totals must
    # equal the untouched basic session.
    assert aggregate(deleted) == aggregate(BASIC)
    tombstone_first = [
        {"schema_version": 1, "event_id": "del-first", "capture_id": "take-1",
         "occurred_at": "2026-09-20T13:00:00Z", "type": "capture_deleted"},
        *BASIC,
    ]
    result = aggregate(tombstone_first)
    assert result["unique_takes"] == 1
    assert result["recognized_words"] == 10


def test_conflicting_event_id_is_an_error() -> None:
    conflict = copy.deepcopy(BASIC[1])
    conflict["lexical_words"] = 20
    with pytest.raises(ValueError, match="conflicting payload"):
        aggregate(BASIC + [conflict])


def test_conflicting_selection_sequence_is_an_error() -> None:
    conflict = copy.deepcopy(BASIC[1])
    conflict["event_id"] = "conflict"
    conflict["lexical_words"] = 20
    with pytest.raises(ValueError, match="sequence"):
        aggregate(BASIC + [conflict])


def test_structurally_impossible_counts_refused() -> None:
    lexical_over_raw = copy.deepcopy(BASIC[1])
    lexical_over_raw["event_id"] = "impossible-1"
    lexical_over_raw["attempt_id"] = "attempt-x"
    lexical_over_raw["selection_seq"] = 3
    lexical_over_raw["lexical_words"] = 500
    with pytest.raises(ValueError, match="Lexical"):
        aggregate(BASIC + [lexical_over_raw])
    generated_over_output = copy.deepcopy(BASIC[4])
    generated_over_output["event_id"] = "impossible-2"
    generated_over_output["delivery_id"] = "delivery-x"
    generated_over_output["generated_words"] = 999
    with pytest.raises(ValueError, match="Generated"):
        aggregate(BASIC + [generated_over_output])
    zero_rate = copy.deepcopy(BASIC[0])
    zero_rate["event_id"] = "impossible-3"
    zero_rate["capture_id"] = "take-x"
    zero_rate["sample_rate"] = 0
    with pytest.raises(ValueError):
        aggregate(BASIC + [zero_rate])


def test_nonfinite_event_fields_never_degrade_into_numbers() -> None:
    # json.loads parses NaN/Infinity literals, so a corrupt sync file can
    # carry them into aggregate(); the contract refuses them (finite check,
    # not just `value < 0`, which NaN survives) instead of computing
    # plausible-looking totals from them.
    nan_words = copy.deepcopy(BASIC[1])
    nan_words["event_id"] = "nan-words"
    nan_words["attempt_id"] = "attempt-nan"
    nan_words["selection_seq"] = 3
    nan_words["lexical_words"] = float("nan")
    with pytest.raises(ValueError, match="lexical_words must be finite"):
        aggregate(BASIC + [nan_words])
    inf_wait = copy.deepcopy(BASIC[1])
    inf_wait["event_id"] = "inf-wait"
    inf_wait["attempt_id"] = "attempt-inf"
    inf_wait["selection_seq"] = 4
    inf_wait["post_stop_ready_ms"] = float("inf")
    with pytest.raises(ValueError, match="post_stop_ready_ms must be finite"):
        aggregate(BASIC + [inf_wait])
    nan_rate = copy.deepcopy(BASIC[0])
    nan_rate["event_id"] = "nan-rate"
    nan_rate["capture_id"] = "take-nan"
    nan_rate["sample_rate"] = float("nan")
    with pytest.raises(ValueError, match="sample_rate must be finite"):
        aggregate(BASIC + [nan_rate])
    nan_count = copy.deepcopy(BASIC[0])
    nan_count["event_id"] = "nan-count"
    nan_count["capture_id"] = "take-nan-2"
    nan_count["sample_count"] = float("nan")
    with pytest.raises(ValueError, match="sample_count must be finite"):
        aggregate(BASIC + [nan_count])


def test_bad_typing_baseline_refused() -> None:
    for value in (0, -1, float("nan")):
        with pytest.raises(ValueError):
            aggregate(BASIC, value)


def test_incomparable_tokenizers_disable_rate_not_corrupt_it() -> None:
    mixed = [
        copy.deepcopy(e)
        for e in BASIC
    ]
    for event in mixed:
        if event["event_id"] == "a3":
            event["tokenizer"] = "uax29-de-v1"
    result = aggregate(mixed)
    assert result["recognized_words_per_captured_minute"] is None
    assert result["recognized_words"] == 110


def test_no_typing_baseline_means_no_proxy() -> None:
    assert aggregate(BASIC)["typing_time_comparison_seconds"] is None


def test_unknown_wait_suppresses_proxy_rather_than_guessing() -> None:
    events = [
        dict(e, post_stop_ready_ms=None) if e["type"] == "recognition_selected" else e
        for e in BASIC
    ]
    assert aggregate(events, 50)["typing_time_comparison_seconds"] is None


def test_negative_time_saved_is_reported_not_clamped() -> None:
    # 100 words at 50 typing-wpm = 120s estimate; 40s capture + 1000s
    # post-stop wait => -920s. The fixture freezes that this is displayed.
    result = aggregate(load("negative-proxy.json"), 50)
    assert result["typing_time_comparison_seconds"] == pytest.approx(-920.0)
    assert result["incomplete_captures"] == 1


def test_generated_and_snippet_output_separate_from_speech() -> None:
    result = aggregate(load("generated-output.json"), 50)
    assert result["recognized_words"] == 110  # speech only, never 255
    assert result["generated_words_by_status"]["confirmed"] == 145  # 60 model + 85 snippet
    assert result["output_words_by_status"]["confirmed"] == 240
    assert result["recognized_words_per_captured_minute"] == pytest.approx(132.0)
    # rev-g1 was refined twice: seq2 {style: 2} replaces seq1 {style: 5};
    # the snippet pass adds {snippet: 1}.
    assert result["change_counts"] == {
        "structural": 0, "user": 0, "dictionary": 0, "snippet": 1, "style": 2,
    }
    assert result["transformation_counts"]["model_authoring"] == 1
    assert result["transformation_counts"]["snippet_expansion"] == 1
    # take-g2 has an unknown wait => no proxy despite a baseline.
    assert result["typing_time_comparison_seconds"] is None


def test_submission_is_not_confirmed_delivery() -> None:
    result = aggregate(BASIC)
    assert result["delivery_counts"]["confirmed"] == 1
    assert result["delivery_counts"]["submitted_unconfirmed"] == 1
    confirmed_words = result["output_words_by_status"]["confirmed"]
    submitted_words = result["output_words_by_status"]["submitted_unconfirmed"]
    assert (confirmed_words, submitted_words) == (120, 200)


def test_orphan_events_are_not_takes() -> None:
    orphan = {
        "schema_version": 1, "event_id": "orphan", "capture_id": "ghost",
        "occurred_at": "2026-09-20T12:00:00Z", "type": "recognition_selected",
        "attempt_id": "ghost-1", "selection_seq": 1, "lexical_words": 99,
        "raw_words": 99, "tokenizer": "uax29-en-v1", "post_stop_ready_ms": 10,
    }
    result = aggregate([orphan])
    assert result["unique_takes"] == 0
    assert result["recognized_words"] == 0


def test_duplicate_capture_finalization_is_an_error() -> None:
    twin = copy.deepcopy(BASIC[0])
    twin["event_id"] = "c1-twin"
    with pytest.raises(ValueError, match="canonical"):
        aggregate(BASIC + [twin])


# --------------------------------------------------------------------------- #
# Timezone, reset, export
# --------------------------------------------------------------------------- #
def test_activity_calendar_grouping_is_timezone_dependent() -> None:
    events = load("timezone-shift.json")
    berlin = activity_by_day(events, "Europe/Berlin")
    utc = activity_by_day(events, "UTC")
    # 23:30Z on Oct 24 is already Oct 25 in Berlin.
    assert berlin == {"2026-10-25": {"takes": 3, "captured_seconds": pytest.approx(15.0)}}
    assert utc["2026-10-24"]["takes"] == 1
    assert utc["2026-10-25"]["takes"] == 2


def test_dst_fallback_is_not_naive_wall_clock() -> None:
    # During the Oct 25 2026 Berlin fallback, 00:30Z and 01:30Z both read
    # 02:30 local wall time but carry different UTC offsets (+2 then +1).
    pre = local_wall_time("2026-10-25T00:30:00Z", "Europe/Berlin")
    post = local_wall_time("2026-10-25T01:30:00Z", "Europe/Berlin")
    assert pre[:3] == post[:3] == (2, 30, 0)
    assert pre[3] == pytest.approx(2.0)
    assert post[3] == pytest.approx(1.0)


def test_deletion_propagates_to_activity_calendar() -> None:
    events = load("deletion-propagation.json")
    days = activity_by_day(events, "Europe/Berlin")
    assert sum(d["takes"] for d in days.values()) == 2  # take-3 removed


def test_unknown_timezone_refused() -> None:
    with pytest.raises(ValueError, match="timezone"):
        activity_by_day(load("timezone-shift.json"), "Mars/Olympus_Mons")


def test_reset_yields_zero_state_and_export_round_trips() -> None:
    empty = aggregate([])
    assert empty["unique_takes"] == 0
    assert empty["recognized_words"] == 0
    assert empty["recognized_words_per_captured_minute"] is None
    assert empty["typing_time_comparison_seconds"] is None
    # Export contract: every aggregate is plain JSON.
    for payload in (empty, aggregate(BASIC, 50), aggregate(load("generated-output.json"))):
        round_tripped = json.loads(json.dumps(payload))
        assert round_tripped == payload
