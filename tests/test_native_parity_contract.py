"""The native parity CONTRACT tests — executable without any model/binary.

These tests make the numerical-correctness contract of issue #167 self-checking:

* the manifest (``tests/native_parity_manifest.json``) is schema-valid and
  every tolerance names its metric, aggregation, normalization and margin;
* the manifest's test selection matches the real tests (parsed with ``ast``,
  no heavy imports) — fixture lists included;
* the rendered contract table in ``docs/ggml-engine.md`` equals the manifest;
* the comparators used by the engine parity gates REJECT seeded truncation, a
  wrong token, and a wrong EOS boundary, and compare the ENTIRE output (a
  matching prefix never passes);
* required-target coverage is accounted: a required target whose assets are
  present but which executed zero tests FAILS; genuinely unavailable coverage
  is reported distinctly (and only fails under ``STARLING_PARITY_STRICT=1``).

Everything here runs on a bare CPU CI runner (pytest only, no downloads), so
the contract is executable everywhere while the measurement gates stay gated
on real assets.
"""

from __future__ import annotations

import ast
import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import parity_contract as pc  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def manifest() -> dict:
    return pc.load_manifest()


# --------------------------------------------------------------------------- #
# Manifest validation
# --------------------------------------------------------------------------- #
def test_manifest_loads_and_validates(manifest: dict) -> None:
    assert manifest["schema_version"] == pc.SCHEMA_VERSION
    assert manifest["targets"], "manifest must cover at least the initial set"
    ids = [t["id"] for t in manifest["targets"]]
    assert len(ids) == len(set(ids))


def test_manifest_covers_required_initial_models(manifest: dict) -> None:
    """Issue #167 initial scope: MOSS, Parakeet and the shared-Qwen decoder."""
    ids = {t["id"] for t in manifest["targets"]}
    for needed in ("moss.intree.text", "parakeet.intree.text",
                   "qwen_decode.shared_fixture"):
        assert needed in ids, f"manifest must cover {needed}"


def test_manifest_rejects_bad_payloads() -> None:
    base = pc.load_manifest()

    def first_corpus_tolerance() -> tuple[str, list[str]]:
        """(target-id, key-path) of the first corpus tolerance in the
        manifest — base block or per-fixture override."""
        for t in base["targets"]:
            if t["contract"]["class"] == "corpus_quality_acceptance":
                return t["id"], ["contract", "tolerance"]
            for fname, o in (t["contract"].get("fixture_contracts") or {}).items():
                if o.get("class") == "corpus_quality_acceptance":
                    return t["id"], ["contract", "fixture_contracts", fname, "tolerance"]
        raise AssertionError("manifest has no corpus tolerance to mutate")

    tid, path = first_corpus_tolerance()
    # A tolerance without a margin/normalization is an unreviewed escape hatch.
    broken = copy.deepcopy(base)
    block: dict = next(t for t in broken["targets"] if t["id"] == tid)
    for key in path[:-1]:
        block = block[key]
    block[path[-1]] = {"metric": "cer", "aggregation": "per-fixture"}
    with pytest.raises(pc.ManifestError, match="tolerance block missing"):
        pc.validate_manifest(broken)
    # Unknown contract class.
    broken = copy.deepcopy(base)
    broken["targets"][0]["contract"]["class"] = "pretty_close"
    with pytest.raises(pc.ManifestError, match="unknown contract class"):
        pc.validate_manifest(broken)
    # Wrong schema version.
    broken = copy.deepcopy(base)
    broken["schema_version"] = 99
    with pytest.raises(pc.ManifestError, match="schema_version"):
        pc.validate_manifest(broken)
    # Missing stop policy.
    broken = copy.deepcopy(base)
    del broken["targets"][0]["policies"]["stop"]
    with pytest.raises(pc.ManifestError, match="policies.stop"):
        pc.validate_manifest(broken)


def test_manifest_rejects_bad_fixture_overrides() -> None:
    """A per-fixture override must name a declared fixture and carry a valid
    tolerance — a weaker gate can never silently widen or dangle."""
    base = pc.load_manifest()
    with_overrides = next(
        t for t in base["targets"] if t["contract"].get("fixture_contracts"))
    # Override for an undeclared fixture.
    broken = copy.deepcopy(base)
    bt = next(t for t in broken["targets"] if t["id"] == with_overrides["id"])
    fc = bt["contract"]["fixture_contracts"]
    fc["nonexistent"] = dict(next(iter(fc.values())))
    with pytest.raises(pc.ManifestError, match="not in"):
        pc.validate_manifest(broken)
    # Corpus override without a tolerance block.
    broken = copy.deepcopy(base)
    bt = next(t for t in broken["targets"] if t["id"] == with_overrides["id"])
    key, o = next(iter(bt["contract"]["fixture_contracts"].items()))
    if o.get("class") != "corpus_quality_acceptance":
        o["class"] = "corpus_quality_acceptance"
        o.pop("tolerance", None)
    else:
        del o["tolerance"]
    with pytest.raises(pc.ManifestError, match="corpus_quality_acceptance"):
        pc.validate_manifest(broken)


def test_manifest_changelog_records_this_change(manifest: dict) -> None:
    revs = [e["revision"] for e in manifest["changelog"]]
    assert manifest["manifest_revision"] in revs
    for entry in manifest["changelog"]:
        assert entry["change"].strip() and entry["justification"].strip()


# --------------------------------------------------------------------------- #
# Test-selection agreement (manifest <-> tests/test_ggml_parity.py)
# --------------------------------------------------------------------------- #
PARITY_FILE = REPO_ROOT / "tests" / "test_ggml_parity.py"

#: Module-level tests in test_ggml_parity.py that are NOT model gates and so
#: are claimed by no manifest target: the coverage-accounting summary itself.
#: Anything else defined there must belong to a target — an unclaimed gate is
#: an accuracy claim nobody reviewed.
NON_GATE_TESTS = {"test_parity_coverage_accounting"}


def test_manifest_tests_exist_and_fixtures_agree(manifest: dict) -> None:
    collected = pc.collect_parametrized_tests([PARITY_FILE])
    source = PARITY_FILE.read_text()
    for t in manifest["targets"]:
        parametrized: list[str] = []
        for node in t["tests"]:
            name = node.split("::")[-1].split("[", 1)[0]
            assert f"def {name}(" in source, (
                f"{t['id']} references {node} which does not exist in "
                f"{PARITY_FILE}")
            if name in collected:
                parametrized.extend(collected[name])
        if parametrized:
            # The union of the entry's parametrize lists must be exactly the
            # manifest fixtures (one entry may split them across tests, e.g.
            # exact short + approximate medium/long).
            assert sorted(set(parametrized)) == sorted(set(t["contract"]["fixtures"])), (
                f"{t['id']}: manifest fixtures {t['contract']['fixtures']} != "
                f"test parametrization {sorted(set(parametrized))}")
        # entries whose tests are not parametrized (single-shot gates like the
        # C++ component binaries) carry their fixtures internally


def test_every_parity_test_is_claimed_by_the_manifest(manifest: dict) -> None:
    """No unclaimed gates: a test outside the manifest is an accuracy claim
    nobody reviewed (and a removed manifest entry must take its test with it)."""
    claimed = pc.manifest_test_names(manifest)
    collected = pc.collect_parametrized_tests([PARITY_FILE])
    tree = ast.parse(PARITY_FILE.read_text())
    defined = {n.name for n in tree.body
               if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")}
    unclaimed = defined - claimed - NON_GATE_TESTS
    assert not unclaimed, (
        f"tests defined but not claimed by any manifest target: {sorted(unclaimed)}")
    assert collected  # the extractor actually found parametrized tests


# --------------------------------------------------------------------------- #
# Rendered documentation agreement
# --------------------------------------------------------------------------- #
def test_docs_manifest_table_agrees(manifest: dict) -> None:
    rendered = pc.render_manifest_markdown(manifest).strip("\n")
    in_docs = pc.docs_block()
    assert in_docs == rendered, (
        "docs/ggml-engine.md parity-manifest block and "
        "tests/native_parity_manifest.json disagree — run "
        "`uv run python tests/parity_contract.py --write-docs`")


# --------------------------------------------------------------------------- #
# Comparator negative tests (issue #167 acceptance criteria)
# --------------------------------------------------------------------------- #
REF_TEXT = ("the quick brown fox jumps over the lazy dog.  "
            "it jumps again, and again, until it stops.")
REF_TOKENS = [5, 7, 9, 11, 13, 15, 17, 19]
EOS = 151645


def _ok(fn, *args, **kwargs) -> None:
    fn(*args, **kwargs)  # must not raise


def test_exact_text_rejects_seeded_truncation() -> None:
    for cut in (1, 10, len(REF_TEXT) - 1):
        with pytest.raises(pc.ContractFailure, match="truncated"):
            pc.assert_exact_text(REF_TEXT[:cut], REF_TEXT, target="t", fixture="f")


def test_exact_text_rejects_matching_prefix_with_tail_garbage() -> None:
    with pytest.raises(pc.ContractFailure, match="trailing garbage"):
        pc.assert_exact_text(REF_TEXT + " extra", REF_TEXT, target="t", fixture="f")
    with pytest.raises(pc.ContractFailure, match="mismatch"):
        pc.assert_exact_text("x" + REF_TEXT, REF_TEXT, target="t", fixture="f")


def test_exact_text_rejects_single_character_flip() -> None:
    flipped = REF_TEXT.replace("fox", "box", 1)
    with pytest.raises(pc.ContractFailure, match="mismatch"):
        pc.assert_exact_text(flipped, REF_TEXT, target="t", fixture="f")


def test_exact_text_accepts_only_recorded_normalization() -> None:
    # The recorded policy is trailing-whitespace rstrip: a trailing newline on
    # the candidate is the file-format artifact, interior bytes are compared.
    _ok(pc.assert_exact_text, REF_TEXT + "\n", REF_TEXT, target="t", fixture="f")
    with pytest.raises(pc.ContractFailure):
        pc.assert_exact_text(REF_TEXT + "\n", REF_TEXT, target="t", fixture="f",
                             rstrip=False)
    # Interior whitespace differences are NOT normalized away.
    with pytest.raises(pc.ContractFailure):
        pc.assert_exact_text(REF_TEXT.replace("  ", " "), REF_TEXT,
                             target="t", fixture="f")


def test_exact_tokens_rejects_incorrect_token() -> None:
    for pos in (0, 3, len(REF_TOKENS) - 1):
        bad = list(REF_TOKENS)
        bad[pos] += 1
        with pytest.raises(pc.ContractFailure, match="differing token"):
            pc.assert_exact_token_stream(bad, REF_TOKENS, target="t", fixture="f")


def test_exact_tokens_rejects_seeded_truncation() -> None:
    for cut in (1, 4, len(REF_TOKENS) - 1):
        with pytest.raises(pc.ContractFailure, match="truncated"):
            pc.assert_exact_token_stream(REF_TOKENS[:cut], REF_TOKENS,
                                         target="t", fixture="f")
    with pytest.raises(pc.ContractFailure, match="overlong"):
        pc.assert_exact_token_stream(REF_TOKENS + [21, 22], REF_TOKENS,
                                     target="t", fixture="f")


def test_exact_tokens_rejects_incorrect_eos_boundary() -> None:
    ref = REF_TOKENS + [EOS]          # the reference stream ends at its EOS
    # EOS omitted entirely (decode never stopped):
    with pytest.raises(pc.ContractFailure):
        pc.assert_exact_token_stream(REF_TOKENS, ref, target="t", fixture="f",
                                     eos_ids=(EOS,))
    # tokens emitted PAST the EOS boundary:
    with pytest.raises(pc.ContractFailure):
        pc.assert_exact_token_stream(REF_TOKENS + [EOS, 42, 43], ref,
                                     target="t", fixture="f", eos_ids=(EOS,))
    # EOS one position early (stop boundary shifted):
    early = REF_TOKENS[:-1] + [EOS, REF_TOKENS[-1]]
    with pytest.raises(pc.ContractFailure):
        pc.assert_exact_token_stream(early, ref, target="t", fixture="f",
                                     eos_ids=(EOS,))
    # sanity: the true stream passes
    _ok(pc.assert_exact_token_stream, ref, ref, target="t", fixture="f",
        eos_ids=(EOS,))


def test_cer_gate_rejects_above_margin_and_passes_below() -> None:
    ref = "the quick brown fox jumps over the lazy dog"
    ok = "the quick brown fox jumps over the lazy dog."
    assert pc.assert_cer_below(ok, ref, target="t", fixture="f", margin=0.05) == 0.0
    with pytest.raises(pc.ContractFailure, match="margin"):
        pc.assert_cer_below("completely unrelated words", ref,
                            target="t", fixture="f", margin=0.10)
    # margin is exclusive; a CER exactly at the margin fails
    cer = pc.character_error_rate(ref, "x" + ref[:-1])
    with pytest.raises(pc.ContractFailure):
        pc.assert_cer_below("x" + ref[:-1], ref, target="t", fixture="f",
                            margin=cer)


def test_token_match_rate_punishes_truncation() -> None:
    ref = [1, 2, 3, 4]
    assert pc.token_match_rate(ref, ref) == 1.0
    assert pc.token_match_rate(ref[:2], ref) == 0.5   # truncation cannot score 1.0
    assert pc.token_match_rate(ref + [9], ref) == 4 / 5


def test_cer_normalization_is_stable() -> None:
    assert pc.normalize_for_cer("Eyes.  It, jumps!") == "eyes it jumps"
    assert pc.character_error_rate("same text", "same text") == 0.0


# --------------------------------------------------------------------------- #
# Required-coverage accounting
# --------------------------------------------------------------------------- #
def test_required_targets_never_silently_pass(manifest: dict, request) -> None:
    """Required coverage is accounted, not assumed.

    * required + assets present + ZERO tests executed -> FAIL (skip-vacuum)
    * required + assets absent -> distinct UNAVAILABLE report (fails only
      under STARLING_PARITY_STRICT=1)
    * asset hash mismatch -> FAIL (never measure the wrong artifact)

    An explicitly deselected run (-k/-m/--deselect) suspends the zero-executed
    enforcement: scoping is a deliberate user decision, not a silent skip.
    """
    report, failures = pc.coverage_lines(
        manifest, deselection=pc.pytest_deselection(request.config))
    if report:
        print("\n[native-parity coverage report — distinct, not silent]")
        for line in report:
            print(f"  {line}")
    if pc.strict_mode():
        assert not failures, f"strict-mode coverage failures:\n{failures}"
    else:
        hard = [f for f in failures if "UNAVAILABLE" not in f]
        assert not hard, (
            "required-target coverage failures (assets present but zero tests "
            f"executed, or asset hash mismatches):\n{hard}")


def test_probe_detects_hash_mismatch_as_problem(tmp_path, monkeypatch) -> None:
    """A present-but-wrong GGUF/golden is a FAILURE, not a skip."""
    entry = {
        "id": "x.test", "engine": "cpp-test", "model": "x", "needs_lib": False,
        "gguf": {"env": "X_GGUF", "default_path": "models/x.gguf",
                 "sha256": "0" * 64},
    }
    gguf = tmp_path / "x.gguf"
    gguf.write_bytes(b"payload")
    monkeypatch.setenv("X_GGUF", str(gguf))
    probe = pc.probe_target_assets(entry, repo_root=tmp_path)
    assert probe.missing == []
    assert probe.problems and "sha256" in probe.problems[0]
    assert not probe.present

    entry["gguf"]["sha256"] = pc.sha256_file(gguf)
    probe = pc.probe_target_assets(entry, repo_root=tmp_path)
    assert probe.present and not probe.problems and not probe.missing
