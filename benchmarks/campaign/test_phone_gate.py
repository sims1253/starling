"""Hermetic host-side tests for gates/phone_bench_ab.sh (issue #176).

The phone A/B gate is the one substantive component CI never executes (no
adb, no device), yet every pixel campaign's keep/revert decision flows
through it. A scripted fake `adb` on PATH exercises the verdict logic
hermetically — the same trick test_gates.py uses for gate scripts:

- exit-status propagation: a crashed bench must be a LOUD fail (diagnostic
  printed), never a silent zero;
- truncated output: fewer `time=` lines than --runs must be rejected, not
  turned into a fast-biased median and a wrong pass;
- empty transcripts stay inconclusive (exit 3), not fail;
- values interpolated into the remote shell command are validated.

`sleep` is shimmed to return instantly: the wall-clock pacing between rounds
is protocol, not logic under test.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
GATE = HERE / "gates" / "phone_bench_ab.sh"
REPO = HERE.parents[1]

FAKE_ADB = '''#!/usr/bin/env python3
"""Scripted adb for phone-gate tests; behavior comes from FAKE_ADB_CONFIG."""
import json
import os
import sys


def main():
    args = sys.argv[1:]
    cfg = {}
    path = os.environ.get("FAKE_ADB_CONFIG")
    if path:
        with open(path, encoding="utf-8") as fh:
            cfg = json.load(fh)
    if not args:
        return 0
    if args[0] == "push":
        log = os.environ.get("FAKE_ADB_PUSH_LOG")
        if log:
            with open(log, "a", encoding="utf-8") as fh:
                fh.write(" ".join(args[1:]) + "\\n")
        return 0
    if args[0] == "get-state":
        return 0
    if args[0] == "shell":
        cmd = " ".join(args[1:])
        if "dumpsys power" in cmd:
            print("mWakefulness=Asleep")
            return 0
        if "input keyevent" in cmd:
            return 0
        if cmd.startswith("[ -f "):
            # have(): is the model/wav already on the device?
            name = cmd[len("[ -f "):].rstrip("]").rstrip().split("/")[-1]
            return 0 if name in cfg.get("device_files", ["model.gguf", "in.wav"]) else 1
        if "pidof starling-bench-" in cmd:
            print("4242")
            return 0
        if "cat /proc/" in cmd:
            print("Name:\\tstarling-bench-\\nVmHWM:\\t  4096 kB")
            return 0
        if "./starling-bench-base" in cmd or "./starling-bench-cand" in cmd:
            side = "base" if "./starling-bench-base" in cmd else "cand"
            spec = cfg.get(side) or {}
            for t in spec.get("times_ms", []):
                print(f"time={t}ms")
            if spec.get("transcript"):
                print(f"  {spec['transcript']}")
            for line in spec.get("extra_lines", []):
                print(line)
            return int(spec.get("exit", 0))
        # pidof / cat /proc / mkdir / kill / [ -f ... ] probes: quiet success
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''

BASE_OK = {"exit": 0, "times_ms": [12.0, 12.5, 12.2, 12.8],
           "transcript": "hello world"}
CAND_OK = {"exit": 0, "times_ms": [10.0, 10.5, 10.2, 10.8],
           "transcript": "hello world"}


class PhoneBenchAbTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="phone-gate-test-"))
        self.addCleanup(lambda: shutil.rmtree(self.tmp, ignore_errors=True))
        bindir = self.tmp / "bin"
        bindir.mkdir()
        for name, text in (("adb", FAKE_ADB), ("sleep", "#!/bin/sh\nexit 0\n")):
            shim = bindir / name
            shim.write_text(text, encoding="utf-8")
            shim.chmod(0o755)
        for name in ("base.bin", "cand.bin", "model.gguf", "in.wav"):
            (self.tmp / name).write_bytes(b"x")
        self.cfg_path = self.tmp / "adb.json"

    def run_gate(self, config, *extra_args):
        self.cfg_path.write_text(json.dumps(config), encoding="utf-8")
        self.push_log = self.tmp / "push.log"
        self.push_log.unlink(missing_ok=True)
        env = dict(os.environ)
        env["PATH"] = f"{self.tmp / 'bin'}:{env['PATH']}"
        env["TRUSTED"] = str(REPO)
        env["FAKE_ADB_CONFIG"] = str(self.cfg_path)
        env["FAKE_ADB_PUSH_LOG"] = str(self.push_log)
        cmd = ["bash", str(GATE),
               "--base-bin", str(self.tmp / "base.bin"),
               "--cand-bin", str(self.tmp / "cand.bin"),
               "--gguf", str(self.tmp / "model.gguf"),
               "--wav", str(self.tmp / "in.wav"),
               "--rounds", "2", "--runs", "4", *extra_args]
        return subprocess.run(cmd, capture_output=True, text=True,
                              env=env, timeout=180)

    def test_clean_run_reports_metrics_and_passes(self):
        r = self.run_gate({"base": BASE_OK, "cand": CAND_OK})
        self.assertEqual(r.returncode, 0, r.stderr)
        for line in ("METRIC total_ms=", "METRIC total_ms_base=",
                     "METRIC total_ms_delta_pct=", "METRIC transcripts_match=1",
                     "METRIC peak_rss_kb=4096"):
            self.assertIn(line, r.stdout, r.stderr)

    def test_cand_bench_crash_is_a_loud_fail(self):
        cand = dict(CAND_OK, exit=7, times_ms=[], transcript="",
                    extra_lines=["ERROR: fake cand crash"])
        r = self.run_gate({"base": BASE_OK, "cand": cand})
        self.assertEqual(r.returncode, 1)
        self.assertIn("cand bench failed (round 1)", r.stderr)
        self.assertIn("ERROR: fake cand crash", r.stderr)  # evidence survives
        self.assertNotIn("METRIC total_ms=", r.stdout)

    def test_base_bench_crash_is_a_loud_fail(self):
        base = dict(BASE_OK, exit=3, times_ms=[], transcript="",
                    extra_lines=["ERROR: fake base crash"])
        r = self.run_gate({"base": base, "cand": CAND_OK})
        self.assertEqual(r.returncode, 1)
        self.assertIn("base bench failed (round 1)", r.stderr)
        self.assertIn("ERROR: fake base crash", r.stderr)

    def test_truncated_bench_output_is_rejected(self):
        # exit 0 but only 2 of 4 runs produced time= lines: a fast-biased
        # median over the survivors must never become a pass.
        cand = dict(CAND_OK, times_ms=[10.0, 10.2])
        r = self.run_gate({"base": BASE_OK, "cand": cand})
        self.assertEqual(r.returncode, 1)
        self.assertIn("truncated", r.stderr)
        self.assertIn("2/4 time= lines", r.stderr)
        self.assertNotIn("METRIC transcripts_match=1", r.stdout)

    def test_empty_transcripts_stay_inconclusive(self):
        r = self.run_gate({"base": dict(BASE_OK, transcript=""),
                           "cand": dict(CAND_OK, transcript="")})
        self.assertEqual(r.returncode, 3)
        self.assertIn("METRIC transcripts_match=0", r.stdout)
        self.assertIn("transcripts empty", r.stderr)

    def test_unsafe_extra_env_is_a_usage_error(self):
        r = self.run_gate({"base": BASE_OK, "cand": CAND_OK},
                          "--extra-env-cand", "FOO=bar baz")
        self.assertEqual(r.returncode, 2)
        self.assertIn("extra-env-cand", r.stderr)

    def test_missing_device_files_are_pushed(self):
        r = self.run_gate({"base": BASE_OK, "cand": CAND_OK, "device_files": []})
        self.assertEqual(r.returncode, 0, r.stderr)
        pushes = self.push_log.read_text(encoding="utf-8")
        self.assertIn("model.gguf", pushes)
        self.assertIn("in.wav", pushes)

    def test_non_numeric_rounds_is_a_usage_error(self):
        r = self.run_gate({"base": BASE_OK, "cand": CAND_OK}, "--rounds", "zero")
        self.assertEqual(r.returncode, 2)
        self.assertIn("positive integers", r.stderr)


if __name__ == "__main__":
    unittest.main()
