"""Gate a compact quantization candidate on paired WER and stored bytes.

Inputs are single-model ``wer_quant.py --include-clips --protocol`` JSON files
from separate processes. The protocol file is hashed into each run before
evaluation, so a later comparison cannot silently change its acceptance bar.
The verdict applies only to the exact recorded corpus and backend.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

from wer import wer_pct


class InvalidComparison(ValueError):
    """The runs cannot support a paired comparison."""


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise InvalidComparison(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_run(path: Path) -> dict:
    rows = json.loads(path.read_text())
    _check(isinstance(rows, list) and len(rows) == 1,
           f"{path}: expected exactly one model row from a fresh process")
    row = rows[0]
    _check(isinstance(row, dict) and isinstance(row.get("clips"), dict),
           f"{path}: missing per-clip records; run with --include-clips")
    return row


def _percentile(sorted_values: list[float], pct: float) -> float:
    pos = (len(sorted_values) - 1) * pct
    lo = math.floor(pos)
    fraction = pos - lo
    return sorted_values[lo] * (1 - fraction) + sorted_values[min(lo + 1, len(sorted_values) - 1)] * fraction


def _paired_interval(differences: list[float], draws: int, seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(differences)
    means = sorted(sum(differences[rng.randrange(n)] for _ in range(n)) / n
                   for _ in range(draws))
    return _percentile(means, 0.025), _percentile(means, 0.975)


def _clip_map(clips: list[dict], cohort: str, arm: str) -> dict[str, dict]:
    _check(isinstance(clips, list), f"{arm} {cohort}: clips must be a list")
    by_id = {}
    for clip in clips:
        _check(isinstance(clip, dict) and isinstance(clip.get("id"), str)
               and clip["id"] and clip["id"] not in by_id,
               f"{arm} {cohort}: missing or duplicate clip ID")
        for field in ("audio_sha256", "reference", "hypothesis", "wer"):
            _check(field in clip, f"{arm} {cohort} {clip['id']}: missing {field}")
        _check(isinstance(clip["audio_sha256"], str) and len(clip["audio_sha256"]) == 64,
               f"{arm} {cohort} {clip['id']}: invalid audio hash")
        measured = wer_pct(clip["reference"], clip["hypothesis"])
        _check(math.isfinite(float(clip["wer"])) and
               math.isclose(measured, float(clip["wer"]), abs_tol=1e-8),
               f"{arm} {cohort} {clip['id']}: stored WER does not match scorer")
        by_id[clip["id"]] = clip
    return by_id


def compare(protocol_path: Path, baseline_path: Path, candidate_path: Path) -> dict:
    protocol = json.loads(protocol_path.read_text())
    _check(isinstance(protocol, dict), "protocol must be a JSON object")
    _check(protocol.get("schema") == "quant-wer-noninferiority-v1", "unsupported protocol schema")
    cohorts = protocol.get("cohorts")
    _check(isinstance(cohorts, dict) and cohorts,
           "protocol needs a nonempty map of cohorts to exact clip counts")
    _check(all(isinstance(k, str) and type(v) is int and v >= 5
               for k, v in cohorts.items()), "each cohort needs at least five clips")
    margin = protocol.get("max_wer_delta_pp")
    saving = protocol.get("min_bytes_saved")
    draws = protocol.get("bootstrap_resamples")
    seed = protocol.get("bootstrap_seed")
    _check(type(margin) in (int, float) and math.isfinite(margin) and 0 <= margin < 100,
           "max_wer_delta_pp must be a nonnegative percentage-point margin")
    _check(type(saving) is int and saving > 0, "min_bytes_saved must be positive")
    _check(type(draws) is int and draws >= 1000, "bootstrap_resamples must be >= 1000")
    _check(type(seed) is int, "bootstrap_seed must be an integer")

    baseline, candidate = _load_run(baseline_path), _load_run(candidate_path)
    seal = _sha256_file(protocol_path)
    for arm, row in (("baseline", baseline), ("candidate", candidate)):
        provenance = row.get("provenance")
        _check(isinstance(provenance, dict), f"{arm}: missing provenance")
        _check(provenance.get("protocol_sha256") == seal,
               f"{arm}: protocol seal differs from the supplied protocol")
        for key in ("model_sha256", "source_sha256", "imatrix_sha256",
                    "engine_sha256", "scorer_sha256", "device"):
            _check(isinstance(provenance.get(key), str) and provenance[key],
                   f"{arm}: missing {key}; supply STARLING_GGML_LIB when evaluating")
        _check(isinstance(provenance.get("model_bytes"), int)
               and provenance["model_bytes"] > 0,
               f"{arm}: missing exact model byte count")
        _check(set(row["clips"]) == set(cohorts),
               f"{arm}: corpus cohorts differ from protocol")

    bp, cp = baseline["provenance"], candidate["provenance"]
    for key in ("source_sha256", "imatrix_sha256", "engine_sha256",
                "scorer_sha256", "device"):
        _check(bp[key] == cp[key], f"{key} differs between arms")
    _check(bp["scorer_sha256"] == _sha256_file(Path(__file__).with_name("wer.py")),
           "recorded scorer differs from this comparator's WER implementation")
    _check(bp["model_sha256"] != cp["model_sha256"],
           "baseline and candidate model hashes are identical")
    saved = bp["model_bytes"] - cp["model_bytes"]
    result = {
        "protocol_sha256": seal,
        "device": bp["device"],
        "baseline_sha256": bp["model_sha256"],
        "candidate_sha256": cp["model_sha256"],
        "source_sha256": bp["source_sha256"],
        "imatrix_sha256": bp["imatrix_sha256"],
        "bytes_saved": saved,
        "min_bytes_saved": saving,
        "max_wer_delta_pp": margin,
        "cohorts": {},
    }
    for index, (cohort, minimum) in enumerate(cohorts.items()):
        b = _clip_map(baseline["clips"][cohort], cohort, "baseline")
        c = _clip_map(candidate["clips"][cohort], cohort, "candidate")
        _check(set(b) == set(c), f"{cohort}: clip IDs differ between arms")
        _check(len(b) == minimum,
               f"{cohort}: found {len(b)} clips; protocol requires exactly {minimum}")
        differences = []
        for clip_id in sorted(b):
            _check((b[clip_id]["audio_sha256"], b[clip_id]["reference"]) ==
                   (c[clip_id]["audio_sha256"], c[clip_id]["reference"]),
                   f"{cohort} {clip_id}: audio or reference differs between arms")
            differences.append(float(c[clip_id]["wer"]) - float(b[clip_id]["wer"]))
        low, high = _paired_interval(differences, draws, seed + index)
        result["cohorts"][cohort] = {
            "clips": len(differences),
            "delta_wer_pp": sum(differences) / len(differences),
            "paired_95_ci_pp": [low, high],
            "noninferior": high < margin,
            "regression_established": low > margin,
        }
    if saved < saving or any(row["regression_established"] for row in result["cohorts"].values()):
        result["verdict"] = "fail"
    elif all(row["noninferior"] for row in result["cohorts"].values()):
        result["verdict"] = "pass"
    else:
        result["verdict"] = "inconclusive"
    if saved < saving:
        result["reason"] = "candidate did not meet the predeclared storage saving"
    elif result["verdict"] == "fail":
        result["reason"] = "paired WER interval establishes a regression beyond the margin"
    elif result["verdict"] == "inconclusive":
        result["reason"] = "at least one paired WER interval reaches the predeclared margin"
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", required=True, type=Path)
    ap.add_argument("--baseline", required=True, type=Path)
    ap.add_argument("--candidate", required=True, type=Path)
    ap.add_argument("--json", type=Path, help="write the comparison result")
    args = ap.parse_args()
    try:
        result = compare(args.protocol, args.baseline, args.candidate)
    except (InvalidComparison, OSError, ValueError) as exc:
        ap.error(str(exc))
    if args.json:
        args.json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
