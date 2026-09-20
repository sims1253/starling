"""Executable metric contract for Starling local Insight events (E28).

This module is the *executable contract*, not the production implementation:
the production aggregator is the native runtime owned by E17 and must reproduce
the semantics frozen here. Semantics ported (not vendored) from the review
package's reference oracle, extended with transformation change-counts,
timezone day grouping and formula versioning.

Frozen semantics (see packages/contracts/insight-events/README.md):

* Dedupe key: ``event_id``. Same ID + identical payload = idempotent replay
  (retry/sync). Same ID + different payload = conflict (ValueError).
* Recognition selection: latest ``selection_seq`` per capture wins; a retry
  *replaces* the word count, it never adds.
* Tombstones: any ``capture_deleted`` for a capture_id removes every
  attributable event of that capture, regardless of arrival order. Stale sync
  replays cannot resurrect deleted totals.
* WPM denominator: silence-inclusive captured seconds of takes that have a
  selected recognition ("recognized words per captured minute"), summed over
  the population (weighted totals, never an average of per-take rates).
* Generated/snippet words are counted from delivery events, separately from
  recognized speech words, and never enter the WPM or time proxy numerators.
* Time-saved proxy: typing estimate at a user baseline minus eligible capture
  seconds minus non-overlapping post-Stop wait. ``None`` when the baseline is
  unset or any wait is unknown; may be negative and is reported as such.

Pure stdlib; no tokenizer (word counts must be supplied precomputed by a
declared tokenizer), no storage, no network.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

FORMULA_VERSION = 1
SUPPORTED_SCHEMA_VERSION = 1
DELIVERY_STATUSES = ("confirmed", "submitted_unconfirmed", "failed", "conflict", "cancelled")
CHANGE_KINDS = ("structural", "user", "dictionary", "snippet", "style")
TRANSFORMATION_KINDS = ("model_authoring", "snippet_expansion", "user_edit", "dictionary_substitution")


def nonnegative(value: Any, label: str) -> float:
    """Reject bools, non-numbers, non-finite and negative values."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return float(value)


def latest_by_sequence(items: list[dict[str, Any]], field: str) -> dict[str, Any] | None:
    """Latest event by a monotone sequence field; equal seq + different payload = conflict."""
    seen: dict[int, dict[str, Any]] = {}
    for item in items:
        seq = item[field]
        if type(seq) is not int or seq < 0:
            raise ValueError(f"Invalid {field} sequence")
        if seq in seen and seen[seq] != item:
            raise ValueError(f"Conflicting {field} sequence values")
        seen[seq] = item
    return seen[max(seen)] if seen else None


def _dedupe_and_tombstone(events: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Idempotency dedupe by event_id, then drop tombstoned captures entirely."""
    unique: dict[str, dict[str, Any]] = {}
    for event in events:
        if event.get("schema_version") != SUPPORTED_SCHEMA_VERSION:
            raise ValueError("Unsupported event schema_version")
        id_ = event["event_id"]
        if id_ in unique and unique[id_] != event:
            raise ValueError("Same event ID has conflicting payloads")
        unique[id_] = event
    # Tombstones dominate stale replays; explicit restore is not part of v1.
    deleted = {e["capture_id"] for e in unique.values() if e["type"] == "capture_deleted"}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for event in unique.values():
        if event["capture_id"] not in deleted:
            grouped.setdefault(event["capture_id"], []).append(event)
    return grouped


def _parse_utc(occurred_at: str) -> datetime:
    if occurred_at.endswith("Z"):
        occurred_at = occurred_at[:-1] + "+00:00"
    stamp = datetime.fromisoformat(occurred_at)
    if stamp.tzinfo is None:
        raise ValueError("occurred_at must carry an explicit UTC offset")
    return stamp.astimezone(ZoneInfo("UTC"))


def aggregate(events: list[dict[str, Any]], typing_wpm: float | None = None) -> dict[str, Any]:
    """Aggregate a local event log into the v1 metric contract.

    Raises ValueError on structurally impossible input (conflicting event IDs
    or sequences, impossible counts, non-positive sample rates) so bugs can
    never silently degrade into plausible numbers.
    """
    if typing_wpm is not None and nonnegative(typing_wpm, "typing_wpm") == 0:
        raise ValueError("typing_wpm must be positive")
    grouped = _dedupe_and_tombstone(events)

    count = words = raw_words = selected_count = incomplete = 0
    captured_seconds = eligible_seconds = wait_seconds = 0.0
    waits_known = True
    tokenizer_set: set[str] = set()
    delivery_counts: dict[str, int] = {k: 0 for k in DELIVERY_STATUSES}
    delivered_words = {k: 0 for k in DELIVERY_STATUSES}
    generated_words = {k: 0 for k in DELIVERY_STATUSES}
    change_counts = {k: 0 for k in CHANGE_KINDS}
    transformation_counts = {k: 0 for k in TRANSFORMATION_KINDS}

    for capture_id in sorted(grouped):
        items = grouped[capture_id]
        captures = [e for e in items if e["type"] == "capture_finalized"]
        if not captures:
            continue  # orphan metadata is not evidence of a captured take
        if len(captures) != 1:
            raise ValueError("Multiple canonical capture finalizations")
        capture = captures[0]
        sample_rate = nonnegative(capture["sample_rate"], "sample_rate")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        duration = nonnegative(capture["sample_count"], "sample_count") / sample_rate
        captured_seconds += duration
        count += 1
        incomplete += int(not capture["complete_audio"])

        selected = latest_by_sequence(
            [e for e in items if e["type"] == "recognition_selected"], "selection_seq"
        )
        if selected is not None:
            selected_count += 1
            lexical = nonnegative(selected["lexical_words"], "lexical_words")
            raw = nonnegative(selected["raw_words"], "raw_words")
            if lexical > raw:
                raise ValueError("Lexical count cannot exceed raw ASR count")
            words += int(lexical)
            raw_words += int(raw)
            eligible_seconds += duration
            tokenizer_set.add(selected["tokenizer"])
            wait = selected["post_stop_ready_ms"]
            if wait is None:
                waits_known = False
            else:
                wait_seconds += nonnegative(wait, "post_stop_ready_ms") / 1000

        # Transformation revisions: latest revision_seq per revision_id wins
        # (a refinement retry replaces, it does not add); distinct revision
        # IDs are distinct passes and both are counted. Change counts are
        # labelled "changes", never "corrected errors".
        revisions: dict[str, list[dict[str, Any]]] = {}
        for item in items:
            if item["type"] == "transformation_completed":
                revisions.setdefault(item["revision_id"], []).append(item)
        for revision_items in revisions.values():
            revision = latest_by_sequence(revision_items, "revision_seq")
            assert revision is not None
            transformation_counts[revision["transformation_kind"]] += 1
            for kind in CHANGE_KINDS:
                change_counts[kind] += int(
                    nonnegative(revision["change_counts"][kind], f"change_counts.{kind}")
                )

        deliveries: dict[str, list[dict[str, Any]]] = {}
        for item in items:
            if item["type"] == "delivery_recorded":
                deliveries.setdefault(item["delivery_id"], []).append(item)
        for same_delivery in deliveries.values():
            delivery = latest_by_sequence(same_delivery, "delivery_seq")
            assert delivery is not None
            status = delivery["status"]
            if status not in delivery_counts:
                raise ValueError("Unknown delivery acknowledgement")
            output = nonnegative(delivery["output_words"], "output_words")
            generated = nonnegative(delivery["generated_words"], "generated_words")
            if generated > output:
                raise ValueError("Generated words cannot exceed output words")
            delivery_counts[status] += 1
            delivered_words[status] += int(output)
            generated_words[status] += int(generated)

    comparable = len(tokenizer_set) == 1
    wpm = words * 60 / eligible_seconds if eligible_seconds > 0 and comparable else None
    proxy = (
        words * 60 / typing_wpm - eligible_seconds - wait_seconds
        if typing_wpm and waits_known and selected_count and comparable
        else None
    )
    return {
        "formula_version": FORMULA_VERSION,
        "unique_takes": count,
        "selected_recognitions": selected_count,
        "incomplete_captures": incomplete,
        "recognized_words": words,
        "raw_recognized_words": raw_words,
        "captured_seconds": captured_seconds,
        "eligible_capture_seconds": eligible_seconds,
        # Silence-inclusive capture-normalized rate; VAD speech-rate is a
        # distinct metric (E17 runtime) and is never silently substituted.
        "recognized_words_per_captured_minute": wpm,
        "tokenizers": sorted(tokenizer_set),
        "delivery_counts": delivery_counts,
        "output_words_by_status": delivered_words,
        "generated_words_by_status": generated_words,
        # Changes by type; NOT certified error corrections.
        "change_counts": change_counts,
        "transformation_counts": transformation_counts,
        "typing_time_comparison_seconds": proxy,
        "time_comparison_caveat": (
            "Estimate excludes unobserved correction time; generated words are "
            "not speech and never enter this estimate."
        ),
    }


def activity_by_day(
    events: list[dict[str, Any]], timezone_name: str
) -> dict[str, dict[str, Any]]:
    """Group attributable takes into local calendar days of a declared timezone.

    DST-aware via zoneinfo (a fallback wall-clock conversion would be a
    silent bug, so the function refuses to run without a tz database).
    """
    try:
        tz = ZoneInfo(timezone_name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError(f"Unknown reporting timezone: {timezone_name}") from exc
    grouped = _dedupe_and_tombstone(events)
    days: dict[str, dict[str, Any]] = {}
    for capture_id in sorted(grouped):
        items = grouped[capture_id]
        captures = [e for e in items if e["type"] == "capture_finalized"]
        if len(captures) != 1:
            if not captures:
                continue
            raise ValueError("Multiple canonical capture finalizations")
        capture = captures[0]
        sample_rate = nonnegative(capture["sample_rate"], "sample_rate")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        duration = nonnegative(capture["sample_count"], "sample_count") / sample_rate
        local_day = _parse_utc(capture["occurred_at"]).astimezone(tz).date().isoformat()
        bucket = days.setdefault(
            local_day, {"takes": 0, "captured_seconds": 0.0}
        )
        bucket["takes"] += 1
        bucket["captured_seconds"] += duration
    return days


def local_wall_time(occurred_at: str, timezone_name: str) -> tuple[int, int, int, float]:
    """(local hour, minute, second, utc_offset_hours) for DST-ambiguity proofs."""
    try:
        tz = ZoneInfo(timezone_name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError(f"Unknown reporting timezone: {timezone_name}") from exc
    local = _parse_utc(occurred_at).astimezone(tz)
    offset = local.utcoffset()
    assert offset is not None
    return local.hour, local.minute, local.second, offset.total_seconds() / 3600
