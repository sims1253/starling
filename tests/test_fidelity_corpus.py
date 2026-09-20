"""Contract tests for the Starling fidelity corpus and gates (E06).

Proves three things against packages/contracts/fidelity-corpus/:

1. The corpus manifest and mutant definitions validate against their schemas.
2. Every seeded mutant is caught by its declared gate (critical-token
   deletion, negation retention, entity/number accuracy, list continuation,
   short-answer retention, speech-region coverage, duplicate and stale-thread
   gates) — while silence is never declared an omission and timestamp
   presence alone never counts as coverage.
3. Clean reference outputs pass every gate (false-alarm contract), so the
   gates do not simply reject everything.

Scoring functions live in tests/fidelity_scorers.py (stdlib + math only);
synthetic audio in tests/fidelity_audio.py is transport/stitch evidence
only, never ASR-quality evidence.
"""

from __future__ import annotations

import copy
import io
import json
from pathlib import Path

import pytest

import minischema
from fidelity_audio import (
    read_wav,
    speech_frame_bounds,
    split_at_silences,
    synthesize_samples,
    write_wav,
)
from fidelity_scorers import (
    apply_mutant,
    correction_resolution,
    duplicate_segments,
    entity_accuracy,
    list_continuation,
    negation_retention,
    normalize_token,
    numeric_accuracy,
    score_stage,
    short_answer_retention,
    speech_region_coverage,
)

try:
    import jsonschema
except ImportError:  # pragma: no cover - exercised only without the package
    jsonschema = None

REPO = Path(__file__).resolve().parents[1]
CONTRACT = REPO / "packages" / "contracts" / "fidelity-corpus"
CORPUS = json.loads((CONTRACT / "corpus.json").read_text())
CORPUS_SCHEMA = json.loads((CONTRACT / "corpus.schema.json").read_text())
MUTANTS = json.loads((CONTRACT / "mutants.json").read_text())["mutants"]
MUTANTS_SCHEMA = json.loads((CONTRACT / "mutants.schema.json").read_text())
FIXTURES = {f["id"]: f for f in CORPUS["fixtures"]}
STAGES = ("capture", "asr", "authored", "delivered")


def assert_valid(instance, schema):
    if jsonschema is not None:
        jsonschema.Draft202012Validator(schema).validate(instance)
    else:
        problems = minischema.errors(instance, schema)
        assert not problems, problems


# --------------------------------------------------------------------------- #
# (a) fixtures validate against the schema
# --------------------------------------------------------------------------- #
def test_corpus_validates_against_schema() -> None:
    assert_valid(CORPUS, CORPUS_SCHEMA)


def test_mutants_validate_against_schema() -> None:
    assert_valid({"schema_version": 1, "mutants": MUTANTS}, MUTANTS_SCHEMA)


def test_all_eight_categories_present() -> None:
    categories = {f["category"] for f in CORPUS["fixtures"]}
    assert categories == {
        "resumed-numbering", "technical-terms", "self-correction", "negation",
        "intentional-filler", "short-answers", "long-passage",
        "multilingual-code-switching",
    }


def test_provenance_is_synthetic_only() -> None:
    assert all(f["provenance"] == "synthetic" for f in CORPUS["fixtures"])


def test_region_annotations_partition_the_synthesis_timeline() -> None:
    for fixture in CORPUS["fixtures"]:
        speech, silence, clock = [], [], 0.0
        for segment in fixture["synthesis"]["segments"]:
            start = round(clock, 3)
            clock = round(clock + segment["duration_seconds"], 3)
            (speech if segment["kind"] == "speech" else silence).append([start, clock])
        expected = fixture["expected"]["capture"]
        assert expected["speech_regions"] == speech
        assert expected["silence_regions"] == silence
        assert fixture["duration_seconds"] == pytest.approx(clock)
        # Clean capture reference covers exactly the speech regions.
        assert fixture["reference_outputs"]["capture"]["segments"] == speech


# --------------------------------------------------------------------------- #
# (c) false-alarm behavior: clean outputs pass
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fixture_id", sorted(FIXTURES))
@pytest.mark.parametrize("stage", STAGES)
def test_clean_reference_output_passes_every_stage(fixture_id: str, stage: str) -> None:
    fixture = FIXTURES[fixture_id]
    result = score_stage(fixture, stage, fixture["reference_outputs"][stage])
    failing = {name for name, gate in result["gates"].items() if not gate["pass"]}
    assert not failing, f"{fixture_id}/{stage}: gates failed on clean output: {failing}"


def test_intentional_short_repeats_are_not_duplicates() -> None:
    # fx-long-01 deliberately repeats "checkpoint saved" in the ASR output;
    # the duplicate gate must not flag short intentional repetitions.
    tokens = FIXTURES["fx-long-01"]["reference_outputs"]["asr"]["tokens"]
    assert normalize_token("checkpoint") in tokens
    assert duplicate_segments(tokens)["clean"]


# --------------------------------------------------------------------------- #
# (b) every mutant is caught by its declared gate
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mutant", MUTANTS, ids=lambda m: m["id"])
def test_mutant_caught_by_declared_gate(mutant: dict) -> None:
    fixture = FIXTURES[mutant["fixture"]]
    mutated = apply_mutant(fixture, mutant)
    result = score_stage(fixture, mutant["stage"], mutated)
    assert result["pass"] is False, f"{mutant['id']} slipped through every gate"
    gate = result["gates"][mutant["expected_gate"]]
    assert gate["pass"] is False, (
        f"{mutant['id']}: declared gate {mutant['expected_gate']} did not catch it"
    )
    # The clean counterpart must pass that same gate (no blanket rejection).
    clean = score_stage(fixture, mutant["stage"], fixture["reference_outputs"][mutant["stage"]])
    assert clean["gates"][mutant["expected_gate"]]["pass"] is True


def test_mutant_fixture_and_stage_references_resolve() -> None:
    for mutant in MUTANTS:
        assert mutant["fixture"] in FIXTURES
        assert mutant["stage"] in STAGES


def test_all_seven_defect_kinds_seeded() -> None:
    assert {m["defect"] for m in MUTANTS} == {
        "dropped-tail", "dropped-negation", "wrong-renumbering", "false-overlap",
        "truncated-refinement", "duplicate-insertion", "stale-thread-update",
    }


# --------------------------------------------------------------------------- #
# Speech-region coverage semantics
# --------------------------------------------------------------------------- #
def test_silence_is_not_an_omission() -> None:
    fixture = FIXTURES["fx-short-01"]
    speech = fixture["expected"]["capture"]["speech_regions"]
    # Captured exactly the speech, dropped every silence gap: full coverage.
    result = speech_region_coverage(speech, speech)
    assert result["coverage"] == pytest.approx(1.0)
    assert result["uncovered_regions"] == []
    assert score_stage(fixture, "capture", {"segments": speech})["pass"]


def test_dropped_tail_speech_region_is_an_omission() -> None:
    fixture = FIXTURES["fx-short-01"]
    speech = fixture["expected"]["capture"]["speech_regions"]
    result = speech_region_coverage(speech, speech[:-1])
    assert result["coverage"] < 1.0
    assert result["uncovered_regions"] == [speech[-1]]


def test_timestamp_presence_alone_is_not_coverage() -> None:
    fixture = FIXTURES["fx-short-01"]
    speech = fixture["expected"]["capture"]["speech_regions"]
    duration = fixture["duration_seconds"]
    # A system that declares word timestamps spanning the whole session but
    # actually captured only two of the four speech regions.
    gamed = {
        "segments": speech[:2],
        "timestamps": [[0.0, duration]],  # decoy: ignored by the scorer
        "word_timings_present": True,
    }
    result = score_stage(fixture, "capture", gamed)
    assert result["pass"] is False
    detail = result["gates"]["speech_region_coverage"]["detail"]
    assert detail["coverage"] < 1.0
    assert len(detail["uncovered_regions"]) == 2
    # Negative control: the same decoy fields change nothing when segments
    # genuinely cover the speech.
    honest = dict(gamed, segments=speech)
    assert score_stage(fixture, "capture", honest)["pass"]


# --------------------------------------------------------------------------- #
# Individual gate semantics
# --------------------------------------------------------------------------- #
def test_negation_gate_requires_negator_and_governed_content() -> None:
    fixture = FIXTURES["fx-neg-01"]
    spans = fixture["expected"]["asr"]["negation_spans"]
    tokens = fixture["reference_outputs"]["asr"]["tokens"]
    assert not negation_retention(spans, tokens)["dropped"]
    # Dropping only the negator while the governed text survives must fail.
    without_never = [t for t in tokens if normalize_token(t) != "never"]
    dropped = negation_retention(spans, without_never)["dropped"]
    assert dropped and dropped[0]["negator"] == "never"


def test_entity_and_numeric_gates_are_exact() -> None:
    fixture = FIXTURES["fx-long-01"]
    tokens = list(fixture["reference_outputs"]["asr"]["tokens"])
    assert entity_accuracy(["whisper", "parakeet"], tokens)["accuracy"] == 1.0
    assert numeric_accuracy(["1", "2", "3", "4", "5"], tokens)["accuracy"] == 1.0
    assert entity_accuracy(["whisper", "parakeet"], tokens + ["parakeet-2"])["missing"] == []
    damaged = [t for t in tokens if normalize_token(t) != "parakeet"]
    assert entity_accuracy(["whisper", "parakeet"], damaged)["missing"] == ["parakeet"]
    assert numeric_accuracy(["5"], ["4"])["accuracy"] == 0.0


def test_list_continuation_distinguishes_resume_from_restart() -> None:
    contract = {"prior_index": 4, "markers": ["5", "6", "7"]}
    resumed = ["ok", "5", "buy", "oat", "milk", "6", "buy", "eggs", "7", "stop"]
    assert list_continuation(contract, resumed)["continued"]
    restarted = ["ok", "1", "buy", "oat", "milk", "2", "buy", "eggs", "3", "stop"]
    assert not list_continuation(contract, restarted)["continued"]
    skipped = ["ok", "5", "buy", "oat", "milk", "7", "buy", "eggs"]
    assert not list_continuation(contract, skipped)["continued"]


def test_short_answer_retention_down_to_single_tokens() -> None:
    answers = [["yes"], ["no"], ["b"], ["forty", "two"]]
    assert not short_answer_retention(answers, ["yes", "no", "b", "forty", "two"])["missing"]
    assert short_answer_retention(answers, ["yes", "no", "b", "forty"])["missing"] == [["forty", "two"]]
    assert short_answer_retention(answers, ["yes", "no", "x", "forty", "two"])["missing"] == [["b"]]


def test_self_correction_replacement_is_not_disjunction() -> None:
    replacement = {"mode": "replacement",
                   "retracted": ["deploy", "on", "friday"],
                   "final": ["deploy", "on", "monday"]}
    disjunction = {"mode": "disjunction", "alternatives": ["friday", "saturday"]}
    authored = ["deploy", "on", "monday", "use", "friday", "or", "saturday"]
    assert correction_resolution([replacement, disjunction], authored)["resolved"]
    # Conjoining the retracted alternative is a resolution failure, not a
    # stylistic choice: the authored text must pick the corrected final.
    conjoined = ["deploy", "on", "friday", "and", "monday"]
    assert not correction_resolution([replacement], conjoined)["resolved"]
    # A disjunction that dropped one alternative is also a failure.
    assert not correction_resolution([disjunction], ["use", "friday"])["resolved"]


def test_duplicate_gate_minimum_run_length() -> None:
    sentence = ["1", "review", "the", "latency", "budget", "2", "check"]
    assert duplicate_segments(sentence + sentence)["duplicates"]
    assert duplicate_segments(["checkpoint", "saved", "x", "checkpoint", "saved"])["clean"]


# --------------------------------------------------------------------------- #
# Synthetic audio: transport/stitch evidence only
# --------------------------------------------------------------------------- #
def test_wav_synth_round_trip_and_region_bounds() -> None:
    fixture = FIXTURES["fx-short-01"]
    segments = fixture["synthesis"]["segments"]
    buffer = io.BytesIO()
    frames = write_wav(buffer, segments)
    expected_frames = int(round(fixture["duration_seconds"] * 16000))
    assert frames == expected_frames
    buffer.seek(0)
    read_frames, samples = read_wav(buffer)
    assert read_frames == expected_frames
    # Region annotations map to frame ranges inside the rendered stream.
    bounds = speech_frame_bounds(segments)
    assert len(bounds) == len(fixture["expected"]["capture"]["speech_regions"])
    for (start, end), region in zip(bounds, fixture["expected"]["capture"]["speech_regions"]):
        assert start / 16000 == pytest.approx(region[0], abs=1e-3)
        assert end / 16000 == pytest.approx(region[1], abs=1e-3)


def test_chunk_split_and_reassembly_is_lossless() -> None:
    fixture = FIXTURES["fx-short-01"]
    segments = fixture["synthesis"]["segments"]
    samples = synthesize_samples(segments)
    chunks = split_at_silences(samples, segments)
    assert len(chunks) == len(fixture["expected"]["capture"]["speech_regions"])
    reassembled = [frame for chunk in chunks for frame in chunk]
    speech_frames = sum(end - start for start, end in speech_frame_bounds(segments))
    assert len(reassembled) == speech_frames
    assert len(reassembled) < len(samples)  # silence dropped, speech intact


def test_mutation_of_fixture_data_never_touches_shared_state() -> None:
    fixture = FIXTURES["fx-neg-01"]
    mutant = next(m for m in MUTANTS if m["defect"] == "dropped-negation")
    before = copy.deepcopy(fixture)
    apply_mutant(fixture, mutant)
    assert fixture == before


# --------------------------------------------------------------------------- #
# OCR review fixes (PR #195): type-aware const/enum equality, empty combiners,
# overlap clamp in region coverage
# --------------------------------------------------------------------------- #
def test_minischema_bool_is_not_integer_in_const_and_enum() -> None:
    assert minischema.errors(1, {"const": True}) == [
        "$: expected const True, got 1"
    ]
    assert minischema.errors(True, {"const": 1}) == [
        "$: expected const 1, got True"
    ]
    assert minischema.errors(True, {"enum": [1, 2]}) == [
        "$: True not in enum [1, 2]"
    ]
    assert minischema.errors(1, {"enum": [True]}) == [
        "$: 1 not in enum [True]"
    ]
    assert minischema.errors(True, {"const": True}) == []
    assert minischema.errors(1, {"enum": [1, True]}) == []


def test_minischema_empty_combiner_is_a_validation_error_not_a_crash() -> None:
    for combiner in ("oneOf", "anyOf"):
        problems = minischema.errors("x", {combiner: []})
        assert problems and "no branches" in problems[0]


def test_region_coverage_clamps_overlap_double_counting() -> None:
    from fidelity_scorers import speech_region_coverage

    # Two fully-overlapping captured segments over the same speech region must
    # not report coverage 2.0.
    result = speech_region_coverage(
        speech_regions=[(0.0, 2.0)],
        captured_segments=[(0.0, 2.0), (0.5, 1.5)],
    )
    assert result["coverage"] <= 1.0
    assert result["uncovered_regions"] == []
