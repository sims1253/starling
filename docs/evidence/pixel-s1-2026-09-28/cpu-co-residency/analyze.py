#!/usr/bin/env python3
"""Validate and summarize the staged Pixel CPU residency observation."""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORDER = ("idle", "asr_only", "asr_plus_optional_s1", "recovery")
EXPECTED_PROCESSES = ((), ("asr",), ("asr", "s1"), ())
PUBLIC_PROTOCOL_SHA256 = "03d204739f8faed3a5e53d67101628ba84802c16d750498b325c5b7815352827"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    protocol = HERE / "protocol.json"
    record = json.loads((HERE / "record.json").read_text())
    rows = [json.loads(line) for line in (HERE / "samples.jsonl").read_text().splitlines()]
    # The record pins the original sealed protocol. The published copy removes
    # the private device address, so its own hash differs by design.
    original_hash = (HERE / "protocol.sha256").read_text().strip()
    require(record["protocol_sha256"] == original_hash,
            "record/original protocol hash mismatch")
    require(hashlib.sha256(protocol.read_bytes()).hexdigest() == PUBLIC_PROTOCOL_SHA256,
            "published protocol hash mismatch")
    public_protocol = json.loads(protocol.read_text())
    require(public_protocol.get("schema") == "pixel-cpu-asr-optional-s1-residency-v1"
            and public_protocol.get("phone") == "Pixel 10 Pro Android 17, serial [redacted]",
            "published protocol schema or redaction differs")
    require(record["completed"] is True, "run incomplete")
    require(record["remote_pids_after_cleanup"] == [], "server survived cleanup")
    require(len(rows) == 12, "wrong sample count")
    require([row["stage"] for row in rows] == [stage for stage in ORDER for _ in range(3)],
            "stage order/count mismatch")
    for name in ("asr", "s1"):
        log = (HERE / f"{name}.log").read_text()
        model = "parakeet" if name == "asr" else "s1"
        require(f"model={model}, backend=CPU" in log, f"{name} CPU backend log missing")
        require("warmup complete" in log and "warmup error:" not in log,
                f"{name} warmup incomplete")
        require(record["servers"][name]["health"]["phase"] == "ready",
                f"{name} health not ready")
    summary = {"schema": "pixel-cpu-residency-summary-v1",
               "protocol_sha256": record["protocol_sha256"],
               "server_ready_ms": {name: record["servers"][name]["ready_ms"]
                                   for name in ("asr", "s1")},
               "stages": {}}
    for stage, names in zip(ORDER, EXPECTED_PROCESSES):
        group = [row for row in rows if row["stage"] == stage]
        for row in group:
            require(tuple(sorted(row["processes"])) == tuple(sorted(names)),
                    f"{stage} process set mismatch")
            phone = row["phone"]
            require(phone["battery_status"] == 3, "phone not discharging")
            require(all(phone[k] == "false" for k in
                        ("ac_powered", "usb_powered", "wireless_powered")),
                    "phone charging")
            require(phone["screen_state"] == "OFF", "screen on")
            require(phone["thermal_status"] < 2, "thermal status too high")
            require(phone["battery_temp_deci_c"] < 420, "battery too hot")
            require(row["meminfo_kb"]["MemAvailable"] > 0, "MemAvailable invalid")
            for name in names:
                require(row["processes"][name]["pid"] == record["servers"][name]["pid"],
                        f"{name} PID mismatch")
                require(row["processes"][name]["pss_kb"] > 0, f"{name} PSS invalid")
        def extent(values):
            return {"min": min(values), "max": max(values)}
        summary["stages"][stage] = {
            "samples": len(group),
            "mem_available_kb": extent([r["meminfo_kb"]["MemAvailable"] for r in group]),
            "combined_process_pss_kb": extent([
                sum(p["pss_kb"] for p in r["processes"].values()) for r in group]),
            "combined_process_rss_kb": extent([
                sum(p["rss_kb"] for p in r["processes"].values()) for r in group]),
            "process_pss_kb": {name: extent([r["processes"][name]["pss_kb"]
                                             for r in group]) for name in names},
            "process_vmhwm_kb": {name: extent([r["processes"][name]["vmhwm_kb"]
                                                for r in group]) for name in names},
        }
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("Validated 12 samples, CPU backend, ready/warmup logs, state and cleanup")


if __name__ == "__main__":
    main()
