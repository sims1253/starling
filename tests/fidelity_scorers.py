"""Fidelity gate scorers for the Starling fidelity corpus (E06).

Executable contract, stdlib + math only: no ML, no audio decoding, no model
inference. The production gate implementation (native runtime / pipeline
harness) must reproduce these semantics; nothing here runs ASR, so scores
computed on synthetic reference outputs are gate calibration only.

Design rules frozen by tests/test_fidelity_corpus.py:

* Silence is never an omission: coverage denominators count speech regions
  only.
* Timestamp presence is not coverage: capture scoring uses real segment
  spans only and ignores any declared timestamps or timing claims.
* Gates run per stage (capture / asr / authored / delivered) exactly when the
  fixture's expected contract for that stage declares the corresponding key,
  and stages never merge into one average score.
"""

from __future__ import annotations

import copy
from typing import Any

DEFAULT_MIN_COVERAGE = 0.99
DEFAULT_MIN_RECALL = 1.0
DUPLICATE_MIN_REPEAT = 6

STAGE_THRESHOLD_KEYS = {
    "capture": "capture_completeness",
    "asr": "asr_content",
    "authored": "authored_intent",
    "delivered": "delivered_text",
}


def normalize_token(token: str) -> str:
    """Lowercase and strip token-edge punctuation; digits keep commas dropped."""
    stripped = token.strip().strip(".,;:!?\"'()[]").lower()
    return stripped.replace(",", "")


def _norm_tokens(tokens: list[str]) -> list[str]:
    return [normalize_token(t) for t in tokens]


def phrase_present(phrase: list[str], tokens: list[str]) -> bool:
    """Contiguous normalized subsequence match."""
    hay, needle = _norm_tokens(tokens), [normalize_token(t) for t in phrase]
    if not needle or len(needle) > len(hay):
        return False
    return any(hay[i:i + len(needle)] == needle for i in range(len(hay) - len(needle) + 1))


# --------------------------------------------------------------------------- #
# Content gates
# --------------------------------------------------------------------------- #
def critical_token_recall(expected: list[str], tokens: list[str]) -> dict[str, Any]:
    present = {normalize_token(t) for t in tokens}
    missing = [t for t in expected if normalize_token(t) not in present]
    recall = (len(expected) - len(missing)) / len(expected) if expected else 1.0
    return {"missing": missing, "recall": recall}


def negation_retention(spans: list[dict[str, Any]], tokens: list[str]) -> dict[str, Any]:
    present = {normalize_token(t) for t in tokens}
    dropped, kept = [], 0
    for span in spans:
        negator = normalize_token(span["negator"])
        governed = [normalize_token(g) for g in span["governed"]]
        if negator in present and all(g in present for g in governed):
            kept += 1
        else:
            dropped.append(span)
    total = len(spans)
    return {"dropped": dropped, "kept": kept, "total": total,
            "retention": kept / total if total else 1.0}


def entity_accuracy(entities: list[str], tokens: list[str]) -> dict[str, Any]:
    present = {normalize_token(t) for t in tokens}
    missing = [e for e in entities if normalize_token(e) not in present]
    accuracy = (len(entities) - len(missing)) / len(entities) if entities else 1.0
    return {"missing": missing, "accuracy": accuracy}


def numeric_accuracy(values: list[str], tokens: list[str]) -> dict[str, Any]:
    return entity_accuracy(values, tokens)


def list_continuation(contract: dict[str, Any], tokens: list[str]) -> dict[str, Any]:
    """Markers must continue prior_index, appear in order, and not restart."""
    reasons: list[str] = []
    prior = contract["prior_index"]
    markers = [normalize_token(m) for m in contract["markers"]]
    if markers and [int(m) for m in markers] != list(
        range(prior + 1, prior + 1 + len(markers))
    ):
        reasons.append(f"contract markers {markers} do not continue prior_index {prior}")
    normalized = _norm_tokens(tokens)
    cursor = -1
    for marker in markers:
        try:
            cursor = normalized.index(marker, cursor + 1)
        except ValueError:
            reasons.append(f"marker {marker!r} missing or out of order")
            break
    if prior >= 1 and "1" not in markers and "1" in normalized:
        reasons.append("list restarts at 1 where continuation was intended")
    return {"reasons": reasons, "continued": not reasons}


def filler_retention(fillers: list[str], tokens: list[str]) -> dict[str, Any]:
    present = {normalize_token(t) for t in tokens}
    missing = [f for f in fillers if normalize_token(f) not in present]
    retained = len(fillers) - len(missing)
    return {"missing": missing, "retained": retained, "total": len(fillers),
            "retention": retained / len(fillers) if fillers else 1.0}


def short_answer_retention(answers: list[list[str]], tokens: list[str]) -> dict[str, Any]:
    missing = [a for a in answers if not phrase_present(a, tokens)]
    retained = len(answers) - len(missing)
    return {"missing": missing, "retained": retained, "total": len(answers),
            "retention": retained / len(answers) if answers else 1.0}


def correction_resolution(corrections: list[dict[str, Any]], tokens: list[str]) -> dict[str, Any]:
    """replacement: final kept, retracted gone. disjunction: all alternatives kept."""
    problems: list[str] = []
    for correction in corrections:
        if correction["mode"] == "replacement":
            if not phrase_present(correction["final"], tokens):
                problems.append(f"final {correction['final']} missing")
            if phrase_present(correction["retracted"], tokens):
                problems.append(f"retracted {correction['retracted']} survived replacement")
        elif correction["mode"] == "disjunction":
            for alternative in correction["alternatives"]:
                if normalize_token(alternative) not in {normalize_token(t) for t in tokens}:
                    problems.append(f"disjunction alternative {alternative!r} dropped")
        else:
            problems.append(f"unknown correction mode {correction['mode']!r}")
    return {"problems": problems, "resolved": not problems}


def revision_freshness(
    final_phrases: list[list[str]], superseded_phrases: list[list[str]], tokens: list[str]
) -> dict[str, Any]:
    problems: list[str] = []
    for phrase in final_phrases:
        if not phrase_present(phrase, tokens):
            problems.append(f"final phrase {phrase} missing from delivered text")
    for phrase in superseded_phrases:
        if phrase_present(phrase, tokens):
            problems.append(f"superseded phrase {phrase} delivered")
    return {"problems": problems, "fresh": not problems}


def duplicate_segments(tokens: list[str], min_repeat: int = DUPLICATE_MIN_REPEAT) -> dict[str, Any]:
    """Any contiguous run of >= min_repeat tokens appearing twice (non-overlapping)."""
    normalized = _norm_tokens(tokens)
    duplicates: list[list[str]] = []
    for run in range(min_repeat, len(normalized) // 2 + 1):
        seen: dict[tuple[str, ...], int] = {}
        for i in range(len(normalized) - run + 1):
            key = tuple(normalized[i:i + run])
            if key in seen and i - seen[key] >= run:
                duplicates.append(list(key))
                break
            seen.setdefault(key, i)
        if duplicates:
            break
    return {"duplicates": duplicates, "clean": not duplicates}


# --------------------------------------------------------------------------- #
# Capture-stage gate
# --------------------------------------------------------------------------- #
def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def speech_region_coverage(
    speech_regions: list[list[float]],
    captured_segments: list[list[float]],
    min_region_coverage: float = DEFAULT_MIN_COVERAGE,
) -> dict[str, Any]:
    """Coverage of expected SPEECH regions by real captured spans.

    Silence is excluded from the denominator by construction: only
    speech_regions are scored. Declared timestamps/word timings are not an
    input to this function at all, so a system cannot earn coverage by
    asserting timestamps over audio it never captured.
    """
    speech_total = sum(end - start for start, end in speech_regions)
    if speech_total <= 0:
        return {"coverage": 1.0, "uncovered_regions": []}
    covered = 0.0
    uncovered: list[list[float]] = []
    for start, end in speech_regions:
        region_covered = sum(
            _overlap(start, end, seg_start, seg_end) for seg_start, seg_end in captured_segments
        )
        covered += region_covered
        if region_covered < (end - start) * min_region_coverage:
            uncovered.append([start, end])
    coverage = covered / speech_total
    return {"coverage": coverage, "uncovered_regions": uncovered}


# --------------------------------------------------------------------------- #
# Stage orchestration
# --------------------------------------------------------------------------- #
def score_stage(fixture: dict[str, Any], stage: str, output: dict[str, Any]) -> dict[str, Any]:
    """Run every gate declared for `stage`; report per-gate results + overall."""
    expected = fixture["expected"][stage]
    thresholds = fixture["stages"].get(STAGE_THRESHOLD_KEYS[stage], {})
    gates: dict[str, dict[str, Any]] = {}

    if stage == "capture":
        result = speech_region_coverage(
            expected["speech_regions"],
            output.get("segments", []),
            thresholds.get("min_speech_region_coverage", DEFAULT_MIN_COVERAGE),
        )
        gates["speech_region_coverage"] = {
            "pass": result["coverage"] >= thresholds.get(
                "min_speech_region_coverage", DEFAULT_MIN_COVERAGE
            ) and not result["uncovered_regions"],
            "detail": result,
        }
        return {"gates": gates, "pass": all(g["pass"] for g in gates.values())}

    tokens = output.get("tokens", [])
    min_recall = thresholds.get("min_critical_token_recall", DEFAULT_MIN_RECALL)

    if "critical_tokens" in expected:
        result = critical_token_recall(expected["critical_tokens"], tokens)
        gates["critical_token_recall"] = {
            "pass": result["recall"] >= min_recall, "detail": result}
    if "negation_spans" in expected:
        result = negation_retention(expected["negation_spans"], tokens)
        gates["negation_retention"] = {"pass": not result["dropped"], "detail": result}
    if "entities" in expected:
        result = entity_accuracy(expected["entities"], tokens)
        gates["entity_accuracy"] = {
            "pass": result["accuracy"] >= thresholds.get("min_entity_accuracy", DEFAULT_MIN_RECALL),
            "detail": result,
        }
    if "numeric_values" in expected:
        result = numeric_accuracy(expected["numeric_values"], tokens)
        gates["numeric_accuracy"] = {
            "pass": result["accuracy"] >= thresholds.get("min_numeric_accuracy", DEFAULT_MIN_RECALL),
            "detail": result,
        }
    if "list_continuation" in expected:
        result = list_continuation(expected["list_continuation"], tokens)
        gates["list_continuation"] = {"pass": result["continued"], "detail": result}
    if "fillers" in expected:
        result = filler_retention(expected["fillers"], tokens)
        gates["filler_retention"] = {"pass": not result["missing"], "detail": result}
    if "short_answers" in expected:
        result = short_answer_retention(expected["short_answers"], tokens)
        gates["short_answer_retention"] = {"pass": not result["missing"], "detail": result}
    if "corrections" in expected:
        result = correction_resolution(expected["corrections"], tokens)
        gates["correction_resolution"] = {"pass": result["resolved"], "detail": result}

    if stage == "delivered":
        if "final_phrases" in expected or "superseded_phrases" in expected:
            result = revision_freshness(
                expected.get("final_phrases", []), expected.get("superseded_phrases", []), tokens
            )
            required = thresholds.get("require_fresh_revision", True)
            gates["revision_freshness"] = {"pass": (result["fresh"] or not required),
                                           "detail": result}
        if thresholds.get("require_no_duplicates", True):
            result = duplicate_segments(tokens)
            gates["no_duplicate_segments"] = {"pass": result["clean"], "detail": result}

    assert gates, f"no gates declared for fixture {fixture['id']} stage {stage}"
    return {"gates": gates, "pass": all(g["pass"] for g in gates.values())}


# --------------------------------------------------------------------------- #
# Seeded mutants
# --------------------------------------------------------------------------- #
def apply_mutant(fixture: dict[str, Any], mutant: dict[str, Any]) -> dict[str, Any]:
    """Apply a seeded defect to the fixture's clean reference stage output."""
    stage, defect, params = mutant["stage"], mutant["defect"], mutant["params"]
    output = copy.deepcopy(fixture["reference_outputs"][stage])

    if stage == "capture":
        if defect != "dropped-tail":
            raise ValueError(f"capture-stage mutant {defect!r} not defined")
        segments = output["segments"]
        drop = params.get("drop_last", 1)
        if drop > len(segments):
            raise ValueError("dropped-tail exceeds captured segments")
        return {"segments": segments[: len(segments) - drop]}

    tokens = output["tokens"]
    if defect == "dropped-negation":
        negator = normalize_token(params["negator"])
        for i, token in enumerate(tokens):
            if normalize_token(token) == negator:
                del tokens[i]
                break
        else:
            raise ValueError(f"negator {params['negator']!r} not in reference output")
    elif defect == "wrong-renumbering":
        markers = {normalize_token(m) for m in params["markers"]}
        counter = 0
        renumbered = []
        for token in tokens:
            if normalize_token(token) in markers:
                counter += 1
                renumbered.append(str(counter))
            else:
                renumbered.append(token)
        tokens = renumbered
    elif defect == "false-overlap":
        del tokens[params["start"]:params["end"]]
    elif defect == "truncated-refinement":
        tokens = tokens[: params["keep_tokens"]]
    elif defect == "duplicate-insertion":
        span = tokens[params["start"]:params["end"]]
        tokens = tokens[: params["end"]] + span + tokens[params["end"]:]
    elif defect == "stale-thread-update":
        find = params["find"]
        if not phrase_present(find, tokens):
            raise ValueError(f"phrase {find} not found for stale-thread-update")
        replace = params["replace"]
        hay = _norm_tokens(tokens)
        needle = [normalize_token(t) for t in find]
        start = next(
            i for i in range(len(hay) - len(needle) + 1) if hay[i:i + len(needle)] == needle
        )
        tokens = tokens[:start] + replace + tokens[start + len(needle):]
    else:
        raise ValueError(f"unknown defect {defect!r}")
    return {"tokens": tokens}
