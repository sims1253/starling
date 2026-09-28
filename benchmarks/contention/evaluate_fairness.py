"""Apply the preregistered #178 pilot rule to app-ready serial/fair results."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = Path(__file__).with_name("fairness_comparison_spec.json")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def nonnegative_number(value: object) -> bool:
    return (type(value) in (int, float) and math.isfinite(value)
            and value >= 0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", type=Path, required=True)
    parser.add_argument("--fair", type=Path, required=True)
    parser.add_argument("--candidate-spec", type=Path,
                        help="candidate spec used for this run; defaults to the current final gate")
    args = parser.parse_args()
    spec = json.loads(SPEC.read_text())
    serial_spec = ROOT / spec["baseline_spec"]
    fair_spec = args.candidate_spec or ROOT / spec["candidate_spec"]
    serial = json.loads(args.serial.read_text())
    fair = json.loads(args.fair.read_text())
    reasons: list[str] = []
    if serial.get("spec_sha256") != sha256(serial_spec):
        reasons.append("serial run does not match committed app-ready spec")
    if fair.get("spec_sha256") != sha256(fair_spec):
        reasons.append("fair run does not match committed app-ready spec")
    if serial.get("cpu_affinity") != fair.get("cpu_affinity"):
        reasons.append("CPU affinity differs between arms")
    if serial.get("model") != fair.get("model"):
        reasons.append("model path differs between arms")
    serial_trials, fair_trials = serial.get("trials", []), fair.get("trials", [])
    if len(serial_trials) != 6 or len(fair_trials) != 6:
        reasons.append("both arms require all six scored trials")
    else:
        rule = spec["candidate_rule"]
        for i in range(3):
            si, sm = serial_trials[2 * i:2 * i + 2]
            fi, fm = fair_trials[2 * i:2 * i + 2]
            tag = f"pair {i}"
            if [t.get("scenario") for t in (si, sm, fi, fm)] != [
                "idle", "mixed", "idle", "mixed"
            ]:
                reasons.append(f"{tag}: scenario order changed")
                continue
            if any(not nonnegative_number(t.get("ws_app_ready_ms"))
                   for t in (si, sm, fi, fm)):
                reasons.append(f"{tag}: application readiness barrier missing")
                continue
            s_idle, s_mix = si.get("short", {}), sm.get("short", {})
            f_idle, f_mix = fi.get("short", {}), fm.get("short", {})
            s_long, f_long = sm.get("long", {}), fm.get("long", {})
            shorts = (s_idle, s_mix, f_idle, f_mix)
            hashes = [s.get("text_sha256") for s in shorts]
            if any(s.get("status") != "final" for s in shorts) or None in hashes:
                reasons.append(f"{tag}: missing short final")
                continue
            if len(set(hashes)) != 1:
                reasons.append(f"{tag}: short final hashes differ")
                continue
            if s_long.get("status") != 200 or f_long.get("status") != 200:
                reasons.append(f"{tag}: long HTTP final missing")
                continue
            if not s_long.get("text_sha256") or (
                s_long.get("text_sha256") != f_long.get("text_sha256")
            ):
                reasons.append(f"{tag}: long final hashes differ")
                continue
            st = sm.get("long_trace", {})
            ft = fm.get("long_trace", {})
            if (not isinstance(st.get("chunks"), int) or st["chunks"] < 2
                    or not isinstance(st.get("service_ms"), list)
                    or len(st["service_ms"]) != 1):
                reasons.append(f"{tag}: serial long trace is incomplete")
                continue
            if (ft.get("chunks") != rule["long_chunks"]
                    or not isinstance(ft.get("service_ms"), list)
                    or len(ft["service_ms"]) != rule["long_chunks"]):
                reasons.append(f"{tag}: fair long trace is incomplete")
                continue
            if not s_mix.get("first_commit_before_long_done") or not f_mix.get(
                "first_commit_before_long_done"
            ):
                reasons.append(f"{tag}: short did not overlap long work")
                continue
            idle_ms = s_idle.get("stop_to_final_ms")
            serial_ms = s_mix.get("stop_to_final_ms")
            fair_ms = f_mix.get("stop_to_final_ms")
            serial_wall_ms = s_long.get("wall_ms")
            fair_wall_ms = f_long.get("wall_ms")
            values = (idle_ms, serial_ms, fair_ms, serial_wall_ms, fair_wall_ms)
            if not all(nonnegative_number(value) for value in values):
                reasons.append(f"{tag}: missing or invalid timing")
                continue
            serial_busy = s_mix.get("busy_responses")
            fair_busy = f_mix.get("busy_responses")
            if not all(type(value) is int and value >= 0
                       for value in (serial_busy, fair_busy)):
                reasons.append(f"{tag}: missing or invalid busy-response count")
                continue
            if serial_ms - idle_ms < 2000 or serial_ms < 2 * idle_ms or serial_busy < 1:
                reasons.append(f"{tag}: app-ready serial arm did not reproduce contention")
            if fair_ms > rule["short_latency_max_ratio_to_serial_mixed"] * serial_ms:
                reasons.append(f"{tag}: fair short ratio missed")
            if serial_ms - fair_ms < rule["short_latency_min_saved_ms"]:
                reasons.append(f"{tag}: fair short absolute saving missed")
            if fair_busy > rule["candidate_busy_responses_max"]:
                reasons.append(f"{tag}: fair stream still received busy responses")
            if fair_wall_ms > rule["long_wall_max_ratio_to_serial_mixed"] * serial_wall_ms:
                reasons.append(f"{tag}: fair long throughput bound missed")
    result = {"status": "pilot_pass" if not reasons else "no_go_or_inconclusive",
              "reasons": reasons, "rule": spec["candidate_rule"]}
    print(json.dumps(result, indent=2))
    return 0 if not reasons else 1


if __name__ == "__main__":
    raise SystemExit(main())
