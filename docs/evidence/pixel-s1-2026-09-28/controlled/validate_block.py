#!/usr/bin/env python3
"""Fail-closed structural validation for the Pixel controlled quant pilot."""

import argparse
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[3]
CASES = json.loads((PROJECT / "tests/fixtures/s1_quant_spans.json").read_text())
GPU = "PowerVR D-Series DXT-48-1536 MC1"
BENCH = re.compile(r"^.+/(short|medium)\.wav run=(\d+) audio=([0-9.]+)s time=([0-9.]+)ms rtf=([0-9.]+)$")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def check_sample(sample):
    for key in ("charge_uah", "voltage_mv", "battery_status",
                "battery_temp_deci_c", "ac_powered", "usb_powered", "wireless_powered",
                "mono_start_ns", "mono_end_ns", "mono_mid_ns", "phase"):
        require(sample.get(key) is not None, f"sample missing {key}")
    require(sample["mono_start_ns"] <= sample["mono_mid_ns"] <= sample["mono_end_ns"],
            "sample timestamp order")
    require(sample["battery_status"] == 3 and all(sample[k] == "false" for k in
            ("ac_powered", "usb_powered", "wireless_powered")), "phone charged or plugged")
    require(sample["battery_temp_deci_c"] < 420, "battery temperature >=42C")
    if "thermal_status" in sample or "screen_state" in sample or "wakefulness" in sample:
        require(sample.get("thermal_status") is not None, "full sample missing thermal status")
        require(sample.get("screen_state") is not None, "full sample missing screen state")
        require(sample["thermal_status"] < 2, "thermal status >=2")
        require(sample["screen_state"] == "OFF", "screen not off")


def check_samples(rows):
    require(len(rows) >= 22, "insufficient battery/status samples")
    for row in rows:
        check_sample(row)
    for phase, minimum in (("pre_idle", 9), ("post_idle", 9)):
        require(sum(row["phase"] == phase for row in rows) >= minimum,
                f"insufficient {phase} samples")
        phase_rows = [row for row in rows if row["phase"] == phase]
        require(phase_rows[-1]["mono_mid_ns"] - phase_rows[0]["mono_mid_ns"] >= 40e9,
                f"{phase} shorter than 40 s")
    for phase in ("active_boundary_before_launch", "active_boundary_after_exit"):
        require(sum(row["phase"] == phase for row in rows) == 1,
                f"expected one {phase} sample")
    require(all(a["mono_mid_ns"] <= b["mono_mid_ns"] for a, b in zip(rows, rows[1:])),
            "sample times reversed")
    require(any(row.get("thermal_status") is not None for row in rows if row["phase"] == "active"),
            "no active full-state sample")


def parse_fast(stdout, stderr, runs=72):
    lines = stdout.splitlines()
    require(lines and re.fullmatch(r"load [0-9.]+ ms\s+backend=\S+", lines[0]),
            "missing fast load header")
    require(f"[fast] device '{GPU}'" in stderr and
            f"parakeet engine: fast/vulkan '{GPU}'" in stderr,
            "selected engine is not actual fast/Vulkan PowerVR")
    require("weights 718.6 MiB" in stderr, "unexpected fast weight allocation")
    require(len(lines) == 1 + 2 * runs * 2, "wrong number of fast result lines")
    result = {"short": [], "medium": []}
    for i in range(runs * 2):
        line = lines[1 + 2 * i]
        hypothesis = lines[2 + 2 * i]
        match = BENCH.fullmatch(line)
        require(match is not None, f"malformed fast result {i}")
        fixture, index = match.group(1), int(match.group(2))
        expected = "short" if i < runs else "medium"
        require(fixture == expected and index == i % runs,
                f"out-of-order/duplicate fast result {i}: {fixture} {index}")
        require(hypothesis.startswith("  ") and hypothesis[2:].strip(),
                f"missing fast hypothesis {i}")
        ms = float(match.group(4))
        require(math.isfinite(ms) and ms > 0, "invalid fast latency")
        result[fixture].append({"run": index, "ms": ms, "hypothesis": hypothesis[2:]})
    for fixture, rows in result.items():
        require(len({row["hypothesis"] for row in rows}) == 1,
                f"nondeterministic {fixture} transcript")
    return result


def parse_s1(result, log, arm):
    require("model loaded in" in log and "warmup complete" in log and
            "starting on 127.0.0.1:8181 (model=s1, backend=CPU" in log,
            "S1 load/warmup/actual CPU backend absent")
    rows = result.get("responses")
    require(isinstance(rows, list) and len(rows) == len(CASES),
            "missing or duplicate S1 responses")
    expected_prior = json.loads((ROOT.parent / f"s1-{arm}-pixel.json").read_text())
    prior = {row["id"]: row["output"] for row in expected_prior["results"]}
    for case, row in zip(CASES, rows):
        require(row.get("id") == case["id"], "S1 case order changed")
        require(row.get("output") == prior[case["id"]],
                f"S1 output changed from validated functional pilot: {case['id']}")
        require(isinstance(row.get("latency_ms"), (float, int)) and
                math.isfinite(row["latency_ms"]) and row["latency_ms"] > 0,
                "invalid S1 latency")
        for label, pattern in case["spans"].items():
            require(re.search(pattern, row["output"], flags=re.IGNORECASE),
                    f"protected span missing: {case['id']}:{label}")
    return rows


def validate_block(block_dir, protocol_sha, expected_family=None, expected_arm=None,
                   expected_index=None):
    block_dir = Path(block_dir)
    result = json.loads((block_dir / "result.json").read_text())
    require(result.get("completed") is True, "block incomplete")
    require(result.get("protocol_sha256") == protocol_sha, "protocol hash differs")
    for key, expected in (("family", expected_family), ("arm", expected_arm),
                          ("index", expected_index)):
        if expected is not None:
            require(result.get(key) == expected, f"block {key} differs from ABBA schedule")
    require(result.get("cooled_before_block") is True, "cooling rule breached")
    rows = [json.loads(line) for line in (block_dir / "battery-state.jsonl").read_text().splitlines()]
    check_samples(rows)
    require(result["active_boundary_start"] == next(row for row in rows if
            row["phase"] == "active_boundary_before_launch"), "start boundary differs")
    require(result["active_boundary_end"] == next(row for row in rows if
            row["phase"] == "active_boundary_after_exit"), "end boundary differs")
    require(result["active_boundary_start"]["mono_end_ns"] <= result["process_launch_ns"] and
            result["process_exit_ns"] <= result["active_boundary_end"]["mono_start_ns"],
            "charge boundary does not bracket process")
    family = result["family"]
    if family == "parakeet_fast":
        payload = parse_fast((block_dir / "bench.stdout").read_text(),
                             (block_dir / "bench.stderr").read_text())
    elif family == "s1_cpu":
        payload = parse_s1(result, (block_dir / "serve.stderr").read_text(), result["arm"])
    else:
        raise ValueError("unknown family")
    return {"family": family, "arm": result["arm"], "index": result["index"],
            "samples": len(rows), "results": {k: len(v) for k, v in payload.items()}
            if family == "parakeet_fast" else len(payload)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("block", type=Path)
    parser.add_argument("--protocol", type=Path)
    args = parser.parse_args()
    import hashlib
    digest = (ROOT / "protocol.sha256").read_text().strip() if args.protocol is None else \
        hashlib.sha256(args.protocol.read_bytes()).hexdigest()
    print(json.dumps(validate_block(args.block, digest), indent=2))


if __name__ == "__main__":
    main()
