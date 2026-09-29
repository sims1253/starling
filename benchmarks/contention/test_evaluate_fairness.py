"""The measured contention decision must fail closed on incomplete trials."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


class FairnessDecisionTest(unittest.TestCase):
    def evaluate(self, serial: dict, fair: dict,
                 candidate_spec: Path | None = None) -> tuple[int, dict]:
        with tempfile.TemporaryDirectory() as directory:
            serial_path = Path(directory) / "serial.json"
            fair_path = Path(directory) / "fair.json"
            serial_path.write_text(json.dumps(serial))
            fair_path.write_text(json.dumps(fair))
            command = [sys.executable, str(HERE / "evaluate_fairness.py"),
                       "--serial", str(serial_path), "--fair", str(fair_path)]
            if candidate_spec is not None:
                command += ["--candidate-spec", str(candidate_spec)]
            run = subprocess.run(
                command,
                text=True, capture_output=True, check=False,
            )
        self.assertFalse(run.stderr, run.stderr)
        return run.returncode, json.loads(run.stdout)

    def test_saved_complete_run_passes_and_missing_timing_fails_closed(self) -> None:
        serial = json.loads((RESULTS / "serial-app-ready-cpu-2026-09-27.json").read_text())
        fair = json.loads((RESULTS / "fair-app-ready-cpu-2026-09-28.json").read_text())
        # The saved candidate predates the post-review binary amendment.
        # Its historical spec is explicit; the default must fail closed.
        code, result = self.evaluate(serial, fair)
        self.assertEqual((code, result["status"]), (1, "no_go_or_inconclusive"))
        self.assertIn("fair run does not match committed app-ready spec", result["reasons"])

        initial = HERE / "fair_ready_spec_initial.json"
        code, result = self.evaluate(serial, fair, initial)
        self.assertEqual((code, result["status"]), (0, "pilot_pass"))

        fair["trials"][1]["ws_app_ready_ms"] = True
        code, result = self.evaluate(serial, fair, initial)
        self.assertEqual((code, result["status"]), (1, "no_go_or_inconclusive"))
        self.assertIn("pair 0: application readiness barrier missing", result["reasons"])
        fair["trials"][1]["ws_app_ready_ms"] = 0

        del fair["trials"][1]["long"]["wall_ms"]
        code, result = self.evaluate(serial, fair, initial)
        self.assertEqual((code, result["status"]), (1, "no_go_or_inconclusive"))
        self.assertIn("pair 0: missing or invalid timing", result["reasons"])


    def test_missing_candidate_spec_fails_closed(self) -> None:
        code, result = self.evaluate({}, {}, HERE / "missing_spec.json")
        self.assertEqual((code, result["status"]), (1, "no_go_or_inconclusive"))
        self.assertIn("comparison inputs invalid", result["reasons"][0])


if __name__ == "__main__":
    unittest.main()
