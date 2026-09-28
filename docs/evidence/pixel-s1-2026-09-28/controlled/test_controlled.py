"""Mutation checks for the published Pixel S1 ABBA evidence."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from analyze_controlled import ROOT, analyze, energy_record
from validate_block import validate_block


class ControlledS1Tests(unittest.TestCase):
    def setUp(self):
        self.protocol_sha = (ROOT / "protocol.sha256").read_text().strip()
        self.source = ROOT / "s1_cpu-1-bf16"
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.block = Path(self.temp.name) / self.source.name
        shutil.copytree(self.source, self.block)

    def update_result(self, edit):
        path = self.block / "result.json"
        result = json.loads(path.read_text())
        edit(result)
        path.write_text(json.dumps(result))

    def validate(self):
        return validate_block(self.block, self.protocol_sha, "s1_cpu", "bf16", 0)

    def test_published_blocks_and_energy_remain_inconclusive(self):
        analysis = analyze("s1_cpu")
        self.assertEqual(analysis, json.loads((ROOT / "s1_cpu-analysis.json").read_text()))
        self.assertEqual([b["arm"] for b in analysis["blocks"]],
                         ["bf16", "q4-k-m", "q4-k-m", "bf16"])
        self.assertFalse(analysis["candidate_energy_win_resolved"])
        self.assertTrue(all(not b["energy"]["gauge_freshness_confirmed"]
                            for b in analysis["blocks"]))
        self.assertTrue(all(pair["illustrative_interval_uah"][0] < 0 <
                            pair["illustrative_interval_uah"][1]
                            for pair in analysis["adjacent_pairs"]))

    def test_schedule_and_protocol_mutants_fail(self):
        self.assertEqual(self.validate()["results"], 8)
        for field, wrong, message in (("protocol_sha256", "0" * 64, "protocol hash"),
                                      ("arm", "q4-k-m", "ABBA schedule"),
                                      ("index", 2, "ABBA schedule")):
            with self.subTest(field=field):
                shutil.copy2(self.source / "result.json", self.block / "result.json")
                self.update_result(lambda r: r.__setitem__(field, wrong))
                with self.assertRaisesRegex(ValueError, message):
                    self.validate()

    def test_response_count_order_output_and_backend_mutants_fail(self):
        for edit, message in (
            (lambda r: r["responses"].pop(), "missing or duplicate"),
            (lambda r: r["responses"][1].__setitem__("id", r["responses"][0]["id"]),
             "case order"),
            (lambda r: r["responses"][0].__setitem__("output", ""), "output changed"),
        ):
            with self.subTest(message=message):
                shutil.copy2(self.source / "result.json", self.block / "result.json")
                self.update_result(edit)
                with self.assertRaisesRegex(ValueError, message):
                    self.validate()
        shutil.copy2(self.source / "result.json", self.block / "result.json")
        path = self.block / "serve.stderr"
        path.write_text(path.read_text().replace("backend=CPU", "backend=GPU"))
        with self.assertRaisesRegex(ValueError, "actual CPU backend"):
            self.validate()

    def test_missing_telemetry_and_stale_gauge_fail_closed(self):
        path = self.block / "battery-state.jsonl"
        samples = [json.loads(line) for line in path.read_text().splitlines()]
        missing = [row.copy() for row in samples]
        missing[0].pop("charge_uah")
        path.write_text("".join(json.dumps(row) + "\n" for row in missing))
        with self.assertRaisesRegex(ValueError, "sample missing charge_uah"):
            self.validate()

        shutil.copy2(self.source / "battery-state.jsonl", path)
        result = json.loads((self.source / "result.json").read_text())
        record = energy_record(result, samples, 5000)
        self.assertFalse(record["gauge_freshness_confirmed"])
        self.assertGreater(record["max_active_counter_plateau_s"], 60)


if __name__ == "__main__":
    unittest.main()
