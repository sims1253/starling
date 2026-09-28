"""Device-free checks for the sealed Pixel native-copy runner."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

RUNNER = Path(__file__).with_name("run_copy.py")
SPEC = importlib.util.spec_from_file_location("pixel_native_copy_runner", RUNNER)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)

BATTERY = """AC powered: false
USB powered: false
Wireless powered: false
status: 3
level: 58
voltage: 3944
temperature: 274
Charge counter: 2695000
"""
TELEMETRY = {
    "dumpsys battery": BATTERY,
    "dumpsys thermalservice": "Thermal Status: 0\n",
    "dumpsys display": "  mScreenState=OFF\n",
    "dumpsys power": "  mWakefulness=Dozing\n",
}


class RunnerStateTests(unittest.TestCase):
    def test_state_has_saved_schema_and_enforces_protocol_bounds(self):
        with patch.object(runner, "adb", side_effect=lambda _shell, command: TELEMETRY[command]):
            state = runner.sample_state()
        saved = json.loads((RUNNER.parent / "preflight-1-greedy/states.jsonl").read_text().splitlines()[0])
        self.assertEqual(set(state), set(saved) - {"label"})
        self.assertEqual(runner.bad_state(state), None)
        for change, expected in (
            ({"battery_status": 2}, "discharging"),
            ({"usb_powered": "true"}, "unplugged"),
            ({"battery_temp_deci_c": 420}, ">=42C"),
            ({"thermal_status": 2}, "thermal status"),
            ({"screen_state": "ON"}, "screen turned on"),
            ({"thermal_status": None}, "missing phone state"),
        ):
            self.assertIn(expected, runner.bad_state({**state, **change}))

    def test_rejected_state_is_written_before_aborting(self):
        with tempfile.TemporaryDirectory() as directory:
            state = {key: None for key in (
                "charge_uah", "voltage_mv", "battery_status", "battery_percent",
                "battery_temp_deci_c", "ac_powered", "usb_powered",
                "wireless_powered", "thermal_status", "screen_state", "wakefulness")}
            with patch.object(runner, "sample_state", return_value=state):
                with self.assertRaisesRegex(RuntimeError, "missing phone state"):
                    runner.checked_state(Path(directory), "before_launch")
            row = json.loads((Path(directory) / "states.jsonl").read_text())
            self.assertEqual(row["label"], "before_launch")

    def test_configured_adb_and_serial_are_used_for_commands(self):
        with patch.object(runner.shutil, "which", return_value="/tools/adb"):
            runner.configure_device("adb", "phone-2")
        with patch.object(runner.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, b"ok")) as call:
            self.assertEqual(runner.adb("shell", "dumpsys battery"), "ok")
        self.assertEqual(call.call_args.args[0],
                         ["/tools/adb", "-s", "phone-2", "shell", "dumpsys battery"])
        with patch.object(runner.shutil, "which", return_value=None):
            with self.assertRaisesRegex(ValueError, "ADB executable not found"):
                runner.configure_device("missing-adb", "phone-2")

    def test_cli_rejects_missing_adb_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new-output"
            with patch.object(sys, "argv", ["run_copy.py", "preflight", "--adb", "missing-adb",
                                            "--output-dir", str(output)]), \
                 patch.object(runner.shutil, "which", return_value=None):
                with self.assertRaises(SystemExit) as exit_code:
                    runner.main()
            self.assertEqual(exit_code.exception.code, 2)
            self.assertFalse(output.exists())

    def test_existing_process_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "preflight-1-greedy"
            output.mkdir()
            marker = output / "record.json"
            marker.write_text("sealed")
            with patch.object(runner, "OUTPUT_ROOT", Path(directory)):
                with self.assertRaises(FileExistsError):
                    runner.run_process("preflight", 1, "greedy", {}, 0)
            self.assertEqual(marker.read_text(), "sealed")


if __name__ == "__main__":
    unittest.main()
