"""The executable native numerical-correctness contract (issue #167).

This module makes the parity contract DATA (``tests/native_parity_manifest.json``)
the single source of truth for three consumers and provides the comparators the
parity tests assert with:

* **Validation** — :func:`validate_manifest` checks the manifest is internally
  consistent (unique ids, known contract classes, complete tolerance blocks,
  referenced tests really exist in the collected suite).
* **Test selection** — :func:`collect_parametrized_tests` reads the pytest test
  files with ``ast`` (no import, no dependencies) so
  ``test_manifest_test_selection_agrees`` can prove every manifest entry maps to
  a real test with the same fixture list, and every parity test is claimed by
  at least one manifest entry (a gate may legitimately feed two targets, e.g.
  the moss_llm_test components + token-stream split).
* **Rendered documentation** — :func:`render_manifest_markdown`` produces the
  contract table embedded between markers in ``docs/ggml-engine.md``;
  ``test_manifest_docs_agree`` fails when the two drift apart.
* **Comparators** — the exact-text / exact-token / tolerance assertions the
  engine parity tests call. They compare the ENTIRE applicable output (a
  matching prefix is a failure) and encode the stop/truncation and suppression
  policies recorded in the manifest. The negative tests in
  ``tests/test_native_parity_contract.py`` prove they reject seeded truncation,
  a wrong token, and a wrong EOS boundary.
* **Coverage accounting** — the registry (:func:`record_executed`,
  :func:`record_unavailable`) lets ``test_required_targets_not_silent`` fail a
  REQUIRED target whose assets are present but which executed zero tests, while
  genuinely unavailable coverage is reported distinctly (printed summary; only
  ``STARLING_PARITY_STRICT=1`` fails on it).

The module deliberately imports nothing beyond the standard library at module
scope, so the contract tests run in the CPU CI environment (pytest only).
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = Path(__file__).resolve().parent / "native_parity_manifest.json"
DOCS_PATH = REPO_ROOT / "docs" / "ggml-engine.md"

SCHEMA_VERSION = 1

#: The four contract classes every correctness target must declare (issue
#: #167), plus the explicit NON-certifying class for inventory entries that
#: are deliberately loose (smoke checks, regression guards). A target carrying
#: ``smoke_non_certifying`` renders as "smoke" in the docs table so it can
#: never be mistaken for accuracy certification.
CONTRACT_CLASSES = (
    "exact_token_equality",
    "exact_text_equality",
    "component_numerical_tolerance",
    "corpus_quality_acceptance",
    "smoke_non_certifying",
)

#: Status values a backend may carry inside one target.
BACKEND_STATUSES = ("validated", "unvalidated")

DOCS_MARKERS = ("<!-- parity-manifest:v{} begin -->".format(SCHEMA_VERSION),
                "<!-- parity-manifest:v{} end -->".format(SCHEMA_VERSION))


class ManifestError(Exception):
    """The manifest violates its own schema."""


class ContractFailure(AssertionError):
    """A comparator rejected a candidate output."""


# --------------------------------------------------------------------------- #
# Manifest loading + validation
# --------------------------------------------------------------------------- #
_MANIFEST_CACHE: dict | None = None


def load_manifest(path: Path = MANIFEST_PATH) -> dict:
    """Load and validate the manifest (cached)."""
    global _MANIFEST_CACHE
    if _MANIFEST_CACHE is None:
        try:
            raw = json.loads(path.read_text())
        except FileNotFoundError as e:
            raise ManifestError(f"manifest missing: {path}") from e
        except json.JSONDecodeError as e:
            raise ManifestError(f"manifest is not valid JSON: {e}") from e
        validate_manifest(raw)
        _MANIFEST_CACHE = raw
    return _MANIFEST_CACHE


def _validate_tolerance(t: dict, target_id: str) -> None:
    """A corpus/component tolerance must name its metric BEFORE it is used."""
    required = ("metric", "aggregation", "normalization", "margin", "justification")
    missing = [k for k in required if not str(t.get(k, "")).strip()]
    if missing:
        raise ManifestError(
            f"target {target_id!r}: tolerance block missing {missing}; a tolerance "
            "without metric/aggregation/normalization/margin/justification is an "
            "unreviewed escape hatch (issue #167)")


def _validate_contract_block(contract: dict, target_id: str, where: str) -> None:
    cls = contract.get("class")
    if cls not in CONTRACT_CLASSES:
        raise ManifestError(
            f"target {target_id!r}{where}: unknown contract class {cls!r}")
    if not contract.get("fixtures"):
        raise ManifestError(f"target {target_id!r}{where}: fixtures are required")
    if cls == "corpus_quality_acceptance":
        tol = contract.get("tolerance")
        if not isinstance(tol, dict):
            raise ManifestError(
                f"target {target_id!r}{where}: corpus_quality_acceptance "
                "requires a tolerance block")
        _validate_tolerance(tol, target_id)
    if cls == "component_numerical_tolerance":
        tols = contract.get("tolerances")
        if not isinstance(tols, list) or not tols:
            raise ManifestError(
                f"target {target_id!r}{where}: component_numerical_tolerance "
                "requires tolerances[] with per-component bounds")
        for tol in tols:
            _validate_tolerance(tol, target_id)
            if not tol.get("component"):
                raise ManifestError(
                    f"target {target_id!r}{where}: tolerance missing 'component'")


def validate_manifest(m: dict) -> None:
    if m.get("schema_version") != SCHEMA_VERSION:
        raise ManifestError(
            f"schema_version {m.get('schema_version')!r} != supported {SCHEMA_VERSION}; "
            "bump SCHEMA_VERSION and re-render docs in the same reviewed change")
    if not m.get("manifest_revision"):
        raise ManifestError("manifest_revision (a human-readable change id) is required")
    if not m.get("changelog") or not isinstance(m["changelog"], list):
        raise ManifestError(
            "changelog is required: every tolerance/reference change must be a "
            "visible, justified entry (issue #167)")
    for entry in m["changelog"]:
        for k in ("revision", "change", "justification"):
            if not str(entry.get(k, "")).strip():
                raise ManifestError(f"changelog entry missing {k!r}: {entry}")

    targets = m.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ManifestError("targets list is required and non-empty")
    seen: set[str] = set()
    for t in targets:
        tid = t.get("id")
        if not tid or tid in seen:
            raise ManifestError(f"target id missing or duplicated: {tid!r}")
        seen.add(tid)

        contract = t.get("contract") or {}
        _validate_contract_block(contract, tid, "")
        fixtures = contract["fixtures"]
        # Per-fixture overrides: a target may mix classes across its fixtures
        # (e.g. exact on short/medium, corpus-quality on long). Every override
        # is validated with the same rules as a base contract and must name a
        # declared fixture — a weaker gate can never silently widen.
        for fname, override in (contract.get("fixture_contracts") or {}).items():
            if fname not in fixtures:
                raise ManifestError(
                    f"target {tid!r}: fixture_contracts key {fname!r} is not in "
                    f"contract.fixtures {fixtures}")
            if not isinstance(override, dict):
                raise ManifestError(
                    f"target {tid!r}: fixture_contracts[{fname!r}] must be an object")
            _validate_contract_block(override, tid, f" fixture {fname!r}")
        if not t.get("tests"):
            raise ManifestError(f"target {tid!r}: tests (pytest node fragments) required")
        backends = t.get("backends")
        if not isinstance(backends, dict) or not backends:
            raise ManifestError(f"target {tid!r}: backends map is required")
        for backend, status in backends.items():
            if status not in BACKEND_STATUSES:
                raise ManifestError(
                    f"target {tid!r}: backend {backend!r} status {status!r} not in "
                    f"{BACKEND_STATUSES}")
        for field_name in ("policies", "reference", "engine", "model"):
            if not t.get(field_name):
                raise ManifestError(f"target {tid!r}: {field_name} is required")
        policies = t["policies"]
        for pol in ("normalization", "tie_break", "stop", "truncation", "suppression"):
            if not str(policies.get(pol, "")).strip():
                raise ManifestError(
                    f"target {tid!r}: policies.{pol} must be recorded explicitly "
                    "(even 'none'/'n/a')")


# --------------------------------------------------------------------------- #
# Test-selection agreement (pure ast — no imports of the test modules)
# --------------------------------------------------------------------------- #
def collect_parametrized_tests(test_files: list[Path]) -> dict[str, list[str]]:
    """Map ``test_name -> fixture-parameter list`` for parametrized tests.

    Only ``@pytest.mark.parametrize`` decorators whose parameter list is a
    literal list of strings are collected (that is the shape used by the parity
    suite). Works by parsing the source, so it needs neither pytest nor the
    test modules' heavy imports (soundfile/torch).
    """
    out: dict[str, list[str]] = {}
    for tf in test_files:
        tree = ast.parse(tf.read_text())
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            params: list[str] = []
            for dec in node.decorator_list:
                if not isinstance(dec, ast.Call):
                    continue
                fn = dec.func
                is_parametrize = (
                    (isinstance(fn, ast.Attribute) and fn.attr == "parametrize")
                    or (isinstance(fn, ast.Name) and fn.id == "parametrize"))
                if not is_parametrize:
                    continue
                for arg in dec.args[1:]:
                    if isinstance(arg, ast.List) and all(
                            isinstance(e, ast.Constant) and isinstance(e.value, str)
                            for e in arg.elts):
                        params.extend(e.value for e in arg.elts)
            if params or any("parametrize" in ast.unparse(d) for d in node.decorator_list):
                out.setdefault(node.name, []).extend(params)
    return out


def manifest_test_names(m: dict) -> set[str]:
    """The bare test function names referenced by the manifest."""
    names: set[str] = set()
    for t in m["targets"]:
        for node in t["tests"]:
            names.add(node.split("::")[-1].split("[", 1)[0])
    return names


# --------------------------------------------------------------------------- #
# Rendered documentation agreement
# --------------------------------------------------------------------------- #
def _status_badge(backends: dict) -> str:
    return ", ".join(f"{b}:{'V' if s == 'validated' else '-'}"
                     for b, s in sorted(backends.items()))


_CLASS_SHORT = {
    "exact_token_equality": "exact-tokens",
    "exact_text_equality": "exact-text",
    "component_numerical_tolerance": "component-tol",
    "corpus_quality_acceptance": "corpus-quality",
    "smoke_non_certifying": "smoke (non-certifying)",
}


def _class_cell(contract: dict) -> str:
    """The class column: the base class plus any per-fixture overrides."""
    cell = _CLASS_SHORT.get(contract["class"], contract["class"])
    overrides = contract.get("fixture_contracts") or {}
    if overrides:
        parts = [cell] + [
            f"{fname}:{_CLASS_SHORT.get(o.get('class'), o.get('class'))}"
            for fname, o in overrides.items()]
        cell = " + ".join(parts)
    return cell


def render_manifest_markdown(m: dict) -> str:
    """Render the manifest as the markdown block embedded in the docs."""
    lines = [
        f"This table is GENERATED from `tests/native_parity_manifest.json` "
        f"(schema v{SCHEMA_VERSION}, revision "
        f"`{m['manifest_revision']}`) — edit the manifest, then run "
        f"`uv run python tests/parity_contract.py --write-docs`. "
        f"`tests/test_native_parity_contract.py` fails if this table and the "
        f"manifest disagree. V = validated backend, - = unvalidated.",
        "",
        "| Target | Model | Engine | Contract class | Fixtures | Backends (V/-) | Required |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for t in m["targets"]:
        lines.append(
            f"| `{t['id']}` | {t['model']} | {t['engine']} "
            f"| {_class_cell(t['contract'])} "
            f"| {', '.join(t['contract']['fixtures'])} "
            f"| {_status_badge(t['backends'])} "
            f"| {'yes' if t.get('required') else 'no'} |")
    return "\n".join(lines)


def docs_block(docs_path: Path = DOCS_PATH) -> str:
    text = docs_path.read_text()
    begin, end = DOCS_MARKERS
    i = text.find(begin)
    j = text.find(end)
    if i < 0 or j < 0 or j < i:
        raise ManifestError(
            f"{docs_path}: parity-manifest markers missing/malformed; expected "
            f"{begin!r} ... {end!r}")
    return text[i + len(begin):j].strip("\n")


def write_docs(docs_path: Path = DOCS_PATH, manifest_path: Path = MANIFEST_PATH) -> None:
    m = load_manifest(manifest_path)
    text = docs_path.read_text()
    begin, end = DOCS_MARKERS
    i, j = text.find(begin), text.find(end)
    if i < 0 or j < 0:
        raise ManifestError(f"{docs_path}: markers missing; add them first")
    text = text[:i + len(begin)] + "\n" + render_manifest_markdown(m) + "\n" + text[j:]
    docs_path.write_text(text)


# --------------------------------------------------------------------------- #
# Comparators (the machinery the parity gates assert with)
# --------------------------------------------------------------------------- #
_WS = re.compile(r"\s+")


def normalize_for_cer(s: str) -> str:
    """The corpus-quality normalization: lowercase, strip punctuation, collapse
    whitespace. This exact function is the ``normalization`` field of every
    CER tolerance in the manifest — do not change it without a manifest
    revision (its identity is part of the contract)."""
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    return _WS.sub(" ", s).strip()


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def character_error_rate(reference: str, candidate: str) -> float:
    n, r = normalize_for_cer(reference), normalize_for_cer(candidate)
    return levenshtein(n, r) / max(1, len(n))


def assert_exact_text(candidate: str, reference: str, *, target: str,
                      fixture: str, rstrip: bool = True) -> None:
    """Exact-text contract: the ENTIRE output must equal the entire reference.

    Rejects a matching prefix in either direction (truncation and trailing
    garbage are both failures), after the manifest's recorded normalization
    (default: trailing-whitespace rstrip only — interior bytes are compared
    verbatim).
    """
    c = candidate.rstrip() if rstrip else candidate
    r = reference.rstrip() if rstrip else reference
    if c != r:
        kind = "prefix-of-reference (truncated output)" if r.startswith(c) else \
               "reference-is-prefix (trailing garbage)" if c.startswith(r) else "mismatch"
        point = next((i for i, (a, b) in enumerate(zip(c, r)) if a != b),
                     min(len(c), len(r)))
        raise ContractFailure(
            f"[{target}] {fixture}: exact-text contract violated ({kind}); first "
            f"difference at char {point} of {len(r)}:\n"
            f"  reference[{point}:{point + 48}]: {r[point:point + 48]!r}\n"
            f"  candidate [{point}:{point + 48}]: {c[point:point + 48]!r}\n"
            f"  lengths: reference={len(r)} candidate={len(c)}")


def assert_exact_token_stream(candidate: list[int], reference: list[int], *,
                              target: str, fixture: str,
                              eos_ids: tuple[int, ...] = ()) -> None:
    """Exact-token contract on the FULL emitted stream (including the stop
    token when the reference contains one).

    Rejects truncation (prefix), trailing tokens past the reference EOS, and
    any single differing token. When ``eos_ids`` is given, additionally checks
    the EOS boundary itself: the reference must end with its first EOS and the
    candidate's first-EOS position must coincide — a stream that ends early or
    runs past EOS is a stop-policy violation even if the shared prefix matches.
    """
    if candidate != reference:
        n = min(len(candidate), len(reference))
        first = next((i for i in range(n) if candidate[i] != reference[i]), n)
        kind = ("truncated (candidate is a prefix; missing "
                f"{len(reference) - len(candidate)} trailing tokens)") if \
            candidate == reference[:len(candidate)] else \
            ("overlong (candidate continues past the reference stream)") if \
            reference == candidate[:len(reference)] else f"first differing token at {first}"
        raise ContractFailure(
            f"[{target}] {fixture}: exact-token contract violated ({kind}); "
            f"lengths reference={len(reference)} candidate={len(candidate)}"
            + (f"; at {first}: reference={reference[first:first + 8]} "
               f"candidate={candidate[first:first + 8]}" if first < n else ""))

    if eos_ids:
        ref_positions = [i for i, t in enumerate(reference) if t in eos_ids]
        cand_positions = [i for i, t in enumerate(candidate) if t in eos_ids]
        if ref_positions and cand_positions != ref_positions:
            raise ContractFailure(
                f"[{target}] {fixture}: EOS boundary violated: reference EOS at "
                f"{ref_positions} candidate at {cand_positions}")


def assert_cer_below(candidate: str, reference: str, *, target: str, fixture: str,
                     margin: float) -> float:
    """Corpus-quality gate: normalized CER must stay below ``margin``.

    Returns the measured CER so callers can record it. The normalization is
    :func:`normalize_for_cer` (the manifest's recorded normalization).
    """
    cer = character_error_rate(reference, candidate)
    if not cer < margin:
        raise ContractFailure(
            f"[{target}] {fixture}: normalized CER {cer:.4f} not below the "
            f"accepted margin {margin} (normalization: lowercase, punctuation "
            f"stripped, whitespace collapsed):\n"
            f"  reference: {normalize_for_cer(reference)[:120]!r}\n"
            f"  candidate: {normalize_for_cer(candidate)[:120]!r}")
    return cer


def token_match_rate(candidate: list[int], reference: list[int]) -> float:
    """Positional token match rate, divided by the LONGER stream so unmatched
    trailing tokens count against the score (a decode that just truncates
    cannot score 1.0)."""
    n = min(len(candidate), len(reference))
    matches = sum(1 for i in range(n) if candidate[i] == reference[i])
    return matches / max(1, max(len(candidate), len(reference)))


# --------------------------------------------------------------------------- #
# Asset probing (is a target runnable HERE, and is it the pinned asset?)
# --------------------------------------------------------------------------- #
_SHA_CACHE: dict[tuple[str, int, int], str] = {}


def sha256_file(path: Path) -> str:
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    hit = _SHA_CACHE.get(key)
    if hit:
        return hit
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    _SHA_CACHE[key] = h.hexdigest()
    return h.hexdigest()


@dataclass
class AssetProbe:
    present: bool                     # everything needed to execute exists here
    problems: list[str] = field(default_factory=list)   # hard problems (fail, not skip)
    missing: list[str] = field(default_factory=list)    # genuinely absent (distinct report)
    detail: dict = field(default_factory=dict)


def _check_gguf_pin(t: dict, path: Path, pin: str, problems: list[str]) -> None:
    """A present GGUF whose sha256 differs from the manifest pin is a PROBLEM
    (fail loudly), never a skip: the validated measurement belongs to the
    pinned artifact."""
    if not pin:
        return
    got = sha256_file(path)
    if got != pin:
        problems.append(
            f"{t['id']}: GGUF {path} sha256 {got[:12]}... != pinned "
            f"{pin[:12]}...")


def probe_target_assets(t: dict, repo_root: Path = REPO_ROOT) -> AssetProbe:
    """Probe whether a manifest target's assets exist in THIS environment.

    * A missing GGUF/lib/golden makes the target *unavailable* (distinct
      report; a required-but-absent target only fails under
      ``STARLING_PARITY_STRICT=1``).
    * A present GGUF/golden whose sha256 does NOT match the manifest pin is a
      *problem* (wrong asset = fail loudly, never silently measure something
      else).
    """
    problems: list[str] = []
    missing: list[str] = []

    if t.get("needs_lib", True):
        from_ok = True
        if t.get("engine", "").startswith("starling-ggml"):
            sys.path.insert(0, str(repo_root / "src"))
            try:
                from starling._ggml import _native  # noqa: PLC0415
                from_ok = _native.available()
            except Exception:
                from_ok = False
            finally:
                sys.path.remove(str(repo_root / "src"))
        if not from_ok:
            missing.append("libstarling_ggml")

    gguf = t.get("gguf") or {}
    if gguf:
        if gguf.get("hardcoded"):
            # The consuming binary hardcodes the repo-relative model path and
            # IGNORES the engine env override (cpp/tests/* load
            # root + "/models/<name>.gguf") — probing any env-provided file
            # would pin a GGUF the binary never reads (pullfrog review).
            path = repo_root / gguf["default_path"]
            _check_gguf_pin(t, path, gguf.get("sha256", ""), problems)
        else:
            env, default = gguf.get("env", ""), gguf.get("default_path", "")
            raw = os.environ.get(env, "") if env else ""
            if raw:
                path = Path(raw).expanduser()
            elif gguf.get("default_path_env_root"):
                # External engines resolve their default under a root that is
                # itself env-configurable (benchmarks/engines.py:
                # ASR_BENCH_ROOT, default ~/asr-bench) — mirror it so the
                # probe does not report a runnable engine as unavailable.
                spec = gguf["default_path_env_root"]
                root = os.environ.get(spec["root_env"]) or spec["root_default"]
                path = Path(os.path.expanduser(root)) / spec["relative"]
            elif default:
                # A default may be repo-relative or absolute/`~` (external
                # engines keep models outside the repo, e.g. ~/asr-bench).
                dpath = Path(os.path.expanduser(default))
                path = dpath if dpath.is_absolute() else (repo_root / default)
            else:
                path = None
            if path is None:
                # env-only probe with the variable unset: the asset lives
                # outside the repo (external engines); report it missing
                # rather than treating the repository root as "the model".
                missing.append(f"gguf:{env or '(unspecified)'} (env unset)")
            elif path.exists():
                _check_gguf_pin(t, path, gguf.get("sha256", ""), problems)
            else:
                missing.append(f"gguf:{env or default}")

    for g in t.get("goldens", ()):  # ["moss_short_text.txt", ...] under golden/
        p = repo_root / "golden" / g
        if not p.exists():
            missing.append(f"golden:{g}")
    for f in t.get("assets_files", ()):  # repo-relative inputs (fixtures, ...)
        if not (repo_root / f).exists():
            missing.append(f"file:{f}")
    pins = t.get("golden_sha256") or {}
    for name, pin in pins.items():
        p = repo_root / "golden" / name
        if p.exists():
            got = sha256_file(p)
            if got != pin:
                problems.append(
                    f"{t['id']}: golden {name} sha256 {got[:12]}... != pinned "
                    f"{pin[:12]}... — goldens were regenerated? A reference update "
                    f"must be a separately reviewed manifest change (issue #167)")

    if t.get("binary"):
        if not (repo_root / "build" / t["binary"]).exists():
            missing.append(f"binary:build/{t['binary']}")

    return AssetProbe(present=not missing and not problems, problems=problems,
                      missing=missing)


def strict_mode() -> bool:
    return os.environ.get("STARLING_PARITY_STRICT", "").lower() in ("1", "true", "yes")


def pytest_deselection(config) -> bool:
    """True when the user explicitly deselected tests for this run (-k/-m/...).

    An explicit deselection is a deliberate scoping decision by the person
    running pytest, not a silent skip: the zero-executed enforcement is
    suspended so `pytest -k moss` does not fail merely because the parakeet
    gates were deselected.
    """
    if config is None:
        return False
    opt = getattr(config, "option", None)
    if opt is None:
        return False
    return bool(getattr(opt, "keyword", None)          # -k
                or getattr(opt, "markerexpr", None)     # -m
                or getattr(opt, "deselect", None))      # --deselect


# --------------------------------------------------------------------------- #
# Coverage registry: executed vs unavailable, per target id
# --------------------------------------------------------------------------- #
@dataclass
class CoverageRecord:
    executed: int = 0
    unavailable_reasons: list[str] = field(default_factory=list)


_RECORDS: dict[str, CoverageRecord] = {}
_PARITY_MODULE_LOADED = False


def mark_parity_module_loaded() -> None:
    """Called at import time by tests/test_ggml_parity.py.

    Distinguishes "the parity module was collected in this run" (its zero-
    executed assertions are binding) from "only the contract file ran" (a
    targeted run must not fail just because the engine tests were deselected).
    """
    global _PARITY_MODULE_LOADED
    _PARITY_MODULE_LOADED = True


def parity_module_loaded() -> bool:
    return _PARITY_MODULE_LOADED


def record_executed(target_id: str) -> None:
    _RECORDS.setdefault(target_id, CoverageRecord()).executed += 1


def record_unavailable(target_id: str, reason: str) -> None:
    _RECORDS.setdefault(target_id, CoverageRecord()).unavailable_reasons.append(reason)


def coverage_lines(m: dict, repo_root: Path = REPO_ROOT,
                   deselection: bool = False) -> tuple[list[str], list[str]]:
    """Summarize coverage; returns (report lines, hard-failure lines).

    Hard failures: a REQUIRED target whose assets are present but which
    executed zero tests (a skip-vacuum: everything silently skipped is a broken
    gate, not a green one), and any asset problems (wrong GGUF/golden hash).
    Unavailable required targets are reported distinctly and only fail in
    strict mode. ``deselection`` (from :func:`pytest_deselection`) suspends the
    zero-executed enforcement for explicitly scoped runs (``-k``/``-m``).
    """
    report: list[str] = []
    failures: list[str] = []
    for t in m["targets"]:
        probe = probe_target_assets(t, repo_root)
        rec = _RECORDS.get(t["id"], CoverageRecord())
        if t.get("required"):
            if probe.problems:
                failures.extend(probe.problems)
            if (probe.present and rec.executed == 0
                    and parity_module_loaded() and not deselection):
                failures.append(
                    f"{t['id']}: REQUIRED target has all assets present but executed "
                    f"ZERO tests in this run (silent skip = broken gate)")
            if probe.missing:
                line = (f"{t['id']}: required coverage UNAVAILABLE here "
                        f"(missing: {', '.join(probe.missing)})")
                if strict_mode():
                    failures.append(line + " [STARLING_PARITY_STRICT=1]")
                else:
                    report.append(line)
        elif probe.missing and not probe.problems:
            report.append(f"{t['id']}: optional target unavailable (missing: "
                          f"{', '.join(probe.missing)})")
        elif probe.problems:
            report.append(f"{t['id']}: optional target asset problem: "
                          f"{'; '.join(probe.problems)}")
    return report, failures


if __name__ == "__main__":  # pragma: no cover - maintainer CLI
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write-docs", action="store_true",
                    help="rewrite the generated manifest table in docs/ggml-engine.md")
    ap.add_argument("--check-docs", action="store_true",
                    help="exit non-zero if docs and manifest disagree")
    args = ap.parse_args()
    m = load_manifest()
    if args.write_docs:
        write_docs()
        print(f"docs updated: {DOCS_PATH}")
    rendered = render_manifest_markdown(m)
    if args.check_docs:
        if docs_block() != rendered.strip("\n"):
            print("docs and manifest DISAGREE — run "
                  "`uv run python tests/parity_contract.py --write-docs`")
            raise SystemExit(1)
        print("docs and manifest agree")
    else:
        print(rendered)
