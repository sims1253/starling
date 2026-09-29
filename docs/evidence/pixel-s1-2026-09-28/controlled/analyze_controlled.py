#!/usr/bin/env python3
"""Analyze only structurally valid Pixel ABBA blocks under the sealed protocol."""

import argparse
import json
import math
import statistics
from pathlib import Path

from validate_block import parse_fast, validate_block

ROOT = Path(__file__).resolve().parent
ARMS = {"parakeet_fast": ["baseline", "embedding", "embedding", "baseline"],
        "s1_cpu": ["bf16", "q4-k-m", "q4-k-m", "bf16"]}


def timestamp(sample):
    return sample["mono_mid_ns"] / 1e9


def slope(rows):
    elapsed = timestamp(rows[-1]) - timestamp(rows[0])
    if elapsed <= 0:
        raise ValueError("idle window has no duration")
    return (rows[0]["charge_uah"] - rows[-1]["charge_uah"]) / elapsed


def slope_interval(rows, step):
    duration = timestamp(rows[-1]) - timestamp(rows[0])
    measured = slope(rows)
    return [measured - 2 * step / duration, measured + 2 * step / duration]


def energy_record(result, samples, step):
    pre = [row for row in samples if row["phase"] == "pre_idle"]
    post = [row for row in samples if row["phase"] == "post_idle"]
    start = result["active_boundary_start"]
    end = result["active_boundary_end"]
    active = [row for row in samples if row["phase"] == "active"]
    duration = timestamp(end) - timestamp(start)
    if duration <= 0 or step <= 0:
        raise ValueError("invalid gauge time or resolution")
    pre_rate, post_rate = slope(pre), slope(post)
    pre_bounds, post_bounds = slope_interval(pre, step), slope_interval(post, step)
    observed_drop = start["charge_uah"] - end["charge_uah"]
    idle_low, idle_high = min(pre_bounds[0], post_bounds[0]), max(pre_bounds[1], post_bounds[1])
    mean_idle = (pre_rate + post_rate) / 2
    corrected = observed_drop - mean_idle * duration
    low = observed_drop - idle_high * duration - 2 * step
    high = observed_drop - idle_low * duration + 2 * step
    voltage_mv = statistics.mean(row["voltage_mv"] for row in [start, *active, end])
    increases = [(a["phase"], b["phase"], b["charge_uah"] - a["charge_uah"])
                 for a, b in zip(samples, samples[1:]) if b["charge_uah"] > a["charge_uah"]]
    active_rows = [start, *active, end]
    plateau_start = active_rows[0]
    plateaus = []
    for row in active_rows[1:]:
        if row["charge_uah"] != plateau_start["charge_uah"]:
            plateaus.append(timestamp(row) - timestamp(plateau_start))
            plateau_start = row
    plateaus.append(timestamp(active_rows[-1]) - timestamp(plateau_start))
    max_plateau_s = max(plateaus)
    return {"active_boundary_duration_s": duration,
            "launch_after_start_sample_s": (result["process_launch_ns"] - start["mono_end_ns"]) / 1e9,
            "end_sample_after_exit_s": (end["mono_start_ns"] - result["process_exit_ns"]) / 1e9,
            "observed_charge_drop_uah": observed_drop,
            "pre_idle_rate_uah_per_s": pre_rate,
            "post_idle_rate_uah_per_s": post_rate,
            "pre_idle_rate_interval_uah_per_s": pre_bounds,
            "post_idle_rate_interval_uah_per_s": post_bounds,
            "mean_idle_rate_uah_per_s": mean_idle,
            "illustrative_idle_corrected_uah": corrected,
            "illustrative_interval_uah": [low, high],
            "mean_active_voltage_mv": voltage_mv,
            "gauge_increases": increases,
            "valid_gauge_direction": all(inc <= step for _, _, inc in increases),
            "max_active_counter_plateau_s": max_plateau_s,
            "gauge_freshness_confirmed": False,
            "freshness_reason": "counter update latency was not independently bounded; arithmetic is diagnostic only",
            "resolution_step_uah": step}


def analyze(family):
    # The original protocol hash is pinned; the published copy redacts the
    # private phone address and host-specific ADB path.
    protocol_sha = (ROOT / "protocol.sha256").read_text().strip()
    blocks = [ROOT / f"{family}-{i + 1}-{arm}" for i, arm in enumerate(ARMS[family])]
    for index, (block, arm) in enumerate(zip(blocks, ARMS[family])):
        validate_block(block, protocol_sha, family, arm, index)
    samples = [[json.loads(line) for line in (block / "battery-state.jsonl").read_text().splitlines()]
               for block in blocks]
    increments = [abs(b["charge_uah"] - a["charge_uah"])
                  for group in samples for a, b in zip(group, group[1:])
                  if b["charge_uah"] != a["charge_uah"]]
    if not increments:
        raise ValueError("gauge never changed; no measured resolution")
    step = min(increments)
    records = []
    all_text = {"short": set(), "medium": set()}
    for block, arm, group in zip(blocks, ARMS[family], samples):
        result = json.loads((block / "result.json").read_text())
        item = {"block": block.name, "arm": arm,
                "energy": energy_record(result, group, step),
                "process_duration_ms": result["process_duration_ms"],
                "cooling_rule_met": result["cooled_before_block"],
                "temperatures_deci_c": [min(row["battery_temp_deci_c"] for row in group),
                                         max(row["battery_temp_deci_c"] for row in group)],
                "thermal_statuses": sorted({row["thermal_status"] for row in group
                                            if row.get("thermal_status") is not None})}
        if family == "parakeet_fast":
            parsed = parse_fast((block / "bench.stdout").read_text(),
                                (block / "bench.stderr").read_text())
            item["load_header_ms"] = float((block / "bench.stdout").read_text().splitlines()[0]
                                           .split()[1])
            item["latency_ms"] = {}
            for fixture, rows in parsed.items():
                all_text[fixture].add(rows[0]["hypothesis"])
                eligible = [row["ms"] for row in rows if row["run"] >= 3]
                item["latency_ms"][fixture] = {"eligible_runs": len(eligible),
                                                "median": statistics.median(eligible),
                                                "mean": statistics.mean(eligible),
                                                "p95": sorted(eligible)[math.ceil(.95 * len(eligible)) - 1]}
        else:
            item["launch_to_ready_ms"] = result["launch_to_ready_ms"]
            item["latency_ms"] = {row["id"]: row["latency_ms"] for row in result["responses"]}
            item["case_median_ms"] = statistics.median(item["latency_ms"].values())
        records.append(item)
    if family == "parakeet_fast" and any(len(texts) != 1 for texts in all_text.values()):
        raise ValueError("fast hypotheses differ across arms")
    pairs = []
    for bi, ci in ((0, 1), (3, 2)):
        b, c = records[bi], records[ci]
        be, ce = b["energy"], c["energy"]
        central = be["illustrative_idle_corrected_uah"] - ce["illustrative_idle_corrected_uah"]
        interval = [be["illustrative_interval_uah"][0] - ce["illustrative_interval_uah"][1],
                    be["illustrative_interval_uah"][1] - ce["illustrative_interval_uah"][0]]
        # The protocol did not independently bound counter update latency.
        # An apparent charge contrast can never authorize an energy claim here.
        pairs.append({"baseline_block": b["block"], "candidate_block": c["block"],
                      "illustrative_counter_contrast_uah": central,
                      "illustrative_interval_uah": interval,
                      "candidate_energy_win_resolved": False})
    result = {"schema": "pixel-controlled-quant-analysis-v2", "family": family,
              "protocol_sha256": protocol_sha, "gauge_step_uah": step,
              "blocks": records, "adjacent_pairs": pairs,
              "candidate_energy_win_resolved": all(pair["candidate_energy_win_resolved"]
                                                   for pair in pairs),
              "energy_scope": "whole-device charge counter; no validated energy estimate or interval",
              "energy_inconclusive_reason": "gauge update latency was not independently bounded"}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("family", choices=list(ARMS))
    args = parser.parse_args()
    result = analyze(args.family)
    path = ROOT / f"{args.family}-analysis.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"family": args.family, "step_uah": result["gauge_step_uah"],
                      "energy_win": result["candidate_energy_win_resolved"],
                      "pairs": result["adjacent_pairs"]}, indent=2))


if __name__ == "__main__":
    main()
