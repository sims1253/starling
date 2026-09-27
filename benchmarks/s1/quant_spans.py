"""Pilot protected-span audit for S1-mini quantization (#316).

Run each model in a fresh process, then compare the records. The annotated
cases are deliberately small; this reports candidate-induced span losses and
cannot certify a recipe for release without the #310 workload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASES = ROOT / "tests/fixtures/s1_quant_spans.json"
sys.path.insert(0, str(ROOT / "src"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_cases(cases: list[dict]) -> None:
    ids = set()
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases must be a nonempty list")
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("id"), str) or case["id"] in ids:
            raise ValueError("case IDs must be unique strings")
        ids.add(case["id"])
        if not isinstance(case.get("transcript"), str) or not case["transcript"].strip():
            raise ValueError(f"{case['id']}: missing transcript")
        spans = case.get("spans")
        if not isinstance(spans, dict) or not spans:
            raise ValueError(f"{case['id']}: missing spans")
        for label, pattern in spans.items():
            if not isinstance(label, str) or not isinstance(pattern, str) or not pattern:
                raise ValueError(f"{case['id']}: invalid span")
            re.compile(pattern)


def audit_output(case: dict, output: str) -> dict[str, bool]:
    return {label: re.search(pattern, output, flags=re.IGNORECASE) is not None
            for label, pattern in case["spans"].items()}


def verified_library_path(requested: Path) -> Path:
    """Resolve the library selected by ctypes and require the requested file."""
    from starling._ggml import _native

    loaded = _native._load_lib()
    name = getattr(loaded, "_name", None)
    actual = Path(name).resolve() if name else None
    if actual != requested.resolve():
        raise RuntimeError(f"requested native library {requested}, but loaded {actual}")
    return actual


def run(model: Path, source: Path, cases_path: Path, device: str, library: Path) -> dict:
    from starling._ggml import GgmlModel, S1
    from starling._ggml._native import backend_name

    cases = json.loads(cases_path.read_text())
    check_cases(cases)
    os.environ["STARLING_GGML_DEVICE"] = device
    os.environ["STARLING_GGML_LIB"] = str(library.resolve())
    loaded_library = verified_library_path(library)
    engine = GgmlModel(S1, str(model.resolve()))
    try:
        actual = backend_name()
        if actual.lower() != device.lower():
            raise RuntimeError(f"requested {device}, engine selected {actual}")
        outputs = []
        for case in cases:
            text = engine.normalize_text(case["transcript"])
            outputs.append({"id": case["id"], "output": text,
                            "spans_present": audit_output(case, text)})
    finally:
        engine.close()
    return {
        "schema": "s1-quant-spans-v1",
        "model_sha256": sha256_file(model),
        "model_bytes": model.stat().st_size,
        "source_sha256": sha256_file(source),
        "engine_sha256": sha256_file(loaded_library),
        "cases_sha256": sha256_file(cases_path),
        "device": actual,
        "results": outputs,
    }


def compare(baseline: dict, candidate: dict, cases_path: Path | None = None) -> dict:
    if baseline.get("schema") != "s1-quant-spans-v1" or candidate.get("schema") != baseline["schema"]:
        raise ValueError("incompatible record schema")
    for key in ("source_sha256", "engine_sha256", "cases_sha256", "device"):
        if not baseline.get(key) or baseline[key] != candidate.get(key):
            raise ValueError(f"{key} differs between arms")
    for arm, record in (("baseline", baseline), ("candidate", candidate)):
        model_hash = record.get("model_sha256")
        if not isinstance(model_hash, str) or re.fullmatch(r"[0-9a-f]{64}", model_hash) is None:
            raise ValueError(f"{arm}: invalid model_sha256")
        size = record.get("model_bytes")
        if type(size) is not int or size <= 0:
            raise ValueError(f"{arm}: invalid model_bytes")
    if baseline["model_sha256"] != baseline["source_sha256"]:
        raise ValueError("baseline model_sha256 must match source_sha256")
    if candidate["model_sha256"] == baseline["model_sha256"]:
        raise ValueError("candidate model_sha256 must differ from baseline")
    b = {row["id"]: row for row in baseline["results"]}
    c = {row["id"]: row for row in candidate["results"]}
    if len(b) != len(baseline["results"]) or len(c) != len(candidate["results"]) or set(b) != set(c):
        raise ValueError("case IDs are missing, duplicated, or changed")
    if cases_path is not None:
        cases = json.loads(cases_path.read_text())
        check_cases(cases)
        if baseline["cases_sha256"] != sha256_file(cases_path):
            raise ValueError("case file differs from recorded cases_sha256")
        expected = {case["id"]: case for case in cases}
        if set(b) != set(expected):
            raise ValueError("case IDs differ from the supplied case file")
        for arm in (b, c):
            for case_id, row in arm.items():
                if row["spans_present"] != audit_output(expected[case_id], row["output"]):
                    raise ValueError(f"{case_id}: stored span status does not match output")
    new_violations = []
    baseline_violations = []
    changed_outputs = []
    for case_id in sorted(b):
        if set(b[case_id]["spans_present"]) != set(c[case_id]["spans_present"]):
            raise ValueError(f"{case_id}: span labels differ")
        if b[case_id]["output"] != c[case_id]["output"]:
            changed_outputs.append(case_id)
        for label, before in b[case_id]["spans_present"].items():
            after = c[case_id]["spans_present"][label]
            if not before:
                baseline_violations.append(f"{case_id}:{label}")
            if before and not after:
                new_violations.append(f"{case_id}:{label}")
    return {"schema": "s1-quant-span-comparison-v1", "cases": len(b),
            "source_sha256": baseline["source_sha256"],
            "baseline_model_sha256": baseline["model_sha256"],
            "candidate_model_sha256": candidate["model_sha256"],
            "changed_outputs": changed_outputs,
            "baseline_violations": baseline_violations,
            "new_violations": new_violations,
            "baseline_bytes": baseline["model_bytes"],
            "candidate_bytes": candidate["model_bytes"],
            "bytes_saved": baseline["model_bytes"] - candidate["model_bytes"],
            "device": baseline["device"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    runner = sub.add_parser("run")
    runner.add_argument("--model", type=Path, required=True)
    runner.add_argument("--source", type=Path, required=True)
    runner.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    runner.add_argument("--device", default="cpu")
    runner.add_argument("--library", type=Path, required=True)
    runner.add_argument("--json", type=Path, required=True)
    comparator = sub.add_parser("compare")
    comparator.add_argument("--baseline", type=Path, required=True)
    comparator.add_argument("--candidate", type=Path, required=True)
    comparator.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    comparator.add_argument("--json", type=Path)
    args = ap.parse_args()
    try:
        if args.command == "run":
            for path in (args.model, args.source, args.cases, args.library):
                if not path.is_file():
                    raise ValueError(f"missing input: {path}")
            result = run(args.model, args.source, args.cases, args.device, args.library)
            args.json.write_text(json.dumps(result, indent=2) + "\n")
        else:
            result = compare(json.loads(args.baseline.read_text()),
                             json.loads(args.candidate.read_text()), args.cases)
            if args.json:
                args.json.write_text(json.dumps(result, indent=2) + "\n")
    except (OSError, ValueError, RuntimeError) as error:
        ap.error(str(error))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
