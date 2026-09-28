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
    def evaluate(self, serial: dict, fair: dict) -> tuple[int, dict]:
        with tempfile.TemporaryDirectory() as directory:
            serial_path = Path(directory) / "serial.json"
            fair_path = Path(directory) / "fair.json"
            serial_path.write_text(json.dumps(serial))
            fair_path.write_text(json.dumps(fair))
            run = subprocess.run(
                [sys.executable, str(HERE / "evaluate_fairness.py"),
                 "--serial", str(serial_path), "--fair", str(fair_path)],
                text=True, capture_output=True, check=False,
            )
        self.assertFalse(run.stderr, run.stderr)
        return run.returncode, json.loads(run.stdout)

    def test_saved_complete_run_passes_and_missing_timing_fails_closed(self) -> None:
        serial = json.loads((RESULTS / "serial-app-ready-cpu-2026-09-27.json").read_text())
        fair = json.loads((RESULTS / "fair-app-ready-cpu-2026-09-28.json").read_text())
        code, result = self.evaluate(serial, fair)
        self.assertEqual((code, result["status"]), (0, "pilot_pass"))

        del fair["trials"][1]["long"]["wall_ms"]
        code, result = self.evaluate(serial, fair)
        self.assertEqual((code, result["status"]), (1, "no_go_or_inconclusive"))
        self.assertIn("pair 0: missing or invalid timing", result["reasons"])


if __name__ == "__main__":
    unittest.main()
