"""Monitor probes: safety stops and cooldowns, all injectable (issue #176 §4)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import monitors  # noqa: E402
import toy as toy_mod  # noqa: E402

MEMINFO_OK = "MemTotal:       16000000 kB\nMemAvailable:    8000000 kB\n"
MEMINFO_LOW = "MemTotal:       16000000 kB\nMemAvailable:      100000 kB\n"


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, s):
        self.now += s


class ScriptedProbe(monitors.Probe):
    """Probe whose adb/filesystem reads come from a scripted table."""

    def __init__(self, *, adb_ok=True, battery=None, severity="NONE",
                 marker=False, meminfo=MEMINFO_OK, thermal_zones=None,
                 cpu="Fake CPU 5650U"):
        self.calls = {"shell": [], "read": {}}
        self._adb_ok = adb_ok
        self._battery = battery or {"level_pct": 80, "status": 3, "temperature_c": 35.0}
        self._battery_reads = 0
        self._severity = severity
        self._marker = marker
        self._meminfo = meminfo
        self._thermal = thermal_zones or []
        reads = {
            "/proc/cpuinfo": f"model name\t: {cpu}\n",
            "/proc/version": "Linux fake",
            "/proc/meminfo": meminfo,
        }
        for i, t in enumerate(self._thermal):
            reads[f"/sys/class/thermal/thermal_zone{i}/temp"] = f"{int(t * 1000)}\n"
        super().__init__(run=self._run, read=self._read, exists=self._exists)
        self._reads = reads

    def _run(self, cmd):
        if cmd[0] != "adb":
            return 127, ""
        args = list(cmd[1:])
        if args[:1] == ["-s"]:
            args = args[2:]
        if args[:1] == ["get-state"]:
            return (0, "device") if self._adb_ok else (1, "error: no devices")
        if args[:1] == ["shell"]:
            shell_cmd = args[1]
            self.calls["shell"].append(shell_cmd)
            if shell_cmd.startswith("getprop"):
                return 0, "Fake Pixel 10 Pro\n"
            if shell_cmd == "dumpsys battery":
                self._battery_reads += 1
                b = self._battery() if callable(self._battery) else self._battery
                return 0, (f"  level: {b['level_pct']}\n  status: {b['status']}\n"
                           f"  temperature: {int(b['temperature_c'] * 10)}\n")
            if shell_cmd == "dumpsys thermalservice":
                return 0, f"Status: 0 Severity: {self._severity}\n"
            if shell_cmd == "cat /proc/meminfo":
                return 0, self._meminfo
            if shell_cmd.startswith("test -f"):
                return (0, "") if self._marker else (1, "")
        return 1, ""

    def _read(self, path):
        return self._reads.get(path)

    def _exists(self, path):
        return str(path).endswith("starling-fast-gpu-wedged") and self._marker


def pixel_profile(**over):
    profile = toy_mod.toy_profile(Path("."))
    profile.update({"device": "pixel", "adb_serial": "FAKE123",
                    "thresholds": {"max_temp_c": 43.0, "min_battery_pct": 40.0,
                                   "min_mem_available_mb": 1500.0}})
    profile.update(over)
    return profile


def notebook_profile(**over):
    profile = toy_mod.toy_profile(Path("."))
    profile["thresholds"] = {"max_temp_c": 85.0, "min_mem_available_mb": 2048.0}
    profile.update(over)
    return profile


class PixelProbeTests(unittest.TestCase):
    def test_disconnected_adb_stops(self):
        result = monitors.probe(pixel_profile(), probe_obj=ScriptedProbe(adb_ok=False))
        self.assertEqual((result.status, result.reason), ("stop", "adb_disconnected"))

    def test_wedge_marker_stops_without_retry(self):
        probe = ScriptedProbe(marker=True)
        result = monitors.probe(pixel_profile(), probe_obj=probe)
        self.assertEqual((result.status, result.reason), ("stop", "wedge"))
        self.assertIn("never", result.readings["message"])
        # a wedge stop is terminal: a second probe also refuses (no retry storm)
        result2 = monitors.probe(pixel_profile(), probe_obj=probe)
        self.assertEqual(result2.status, "stop")

    def test_wedge_pattern_in_last_gate_log_stops(self):
        result = monitors.probe(
            pixel_profile(), last_gate_log="vkWaitForFences failed: -4\n...",
            probe_obj=ScriptedProbe())
        self.assertEqual((result.status, result.reason), ("stop", "driver_failure"))

    def test_low_battery_while_discharging_stops(self):
        result = monitors.probe(
            pixel_profile(), probe_obj=ScriptedProbe(
                battery={"level_pct": 15, "status": 3, "temperature_c": 35.0}))
        self.assertEqual((result.status, result.reason), ("stop", "battery"))

    def test_low_battery_while_charging_is_only_recorded(self):
        result = monitors.probe(
            pixel_profile(), probe_obj=ScriptedProbe(
                battery={"level_pct": 15, "status": 2, "temperature_c": 35.0}))
        self.assertEqual(result.status, "ok")
        self.assertTrue(result.readings["charging"])

    def test_hot_battery_cools_down_and_continues(self):
        clock = FakeClock()
        # battery reads report hot on the first probe, cool afterwards
        battery = {"level_pct": 80, "status": 3, "temperature_c": 45.0}
        probe = ScriptedProbe()

        def cooling_battery():
            temp = 45.0 if probe._battery_reads <= 1 else 40.0
            return {"level_pct": 80, "status": 3, "temperature_c": temp}

        probe._battery = cooling_battery
        result = monitors.wait_cooldown(pixel_profile(), 30, probe_obj=probe,
                                        sleep=clock.sleep, monotonic=clock.monotonic)
        self.assertEqual(result.status, "ok")

    def test_hot_battery_stays_hot_stops_thermal(self):
        clock = FakeClock()
        probe = ScriptedProbe(battery={"level_pct": 80, "status": 3,
                                       "temperature_c": 47.0})
        result = monitors.wait_cooldown(pixel_profile(), 10, probe_obj=probe,
                                        sleep=clock.sleep, monotonic=clock.monotonic)
        self.assertEqual((result.status, result.reason), ("stop", "thermal"))

    def test_moderate_thermal_severity_cooldowns(self):
        probe = ScriptedProbe(severity="MODERATE")
        result = monitors.probe(pixel_profile(), probe_obj=probe)
        self.assertEqual((result.status, result.reason), ("cooldown", "thermal"))


class NotebookProbeTests(unittest.TestCase):
    def test_thermal_zone_hot_cools(self):
        clock = FakeClock()
        probe = ScriptedProbe(thermal_zones=[90.0])
        result = monitors.wait_cooldown(notebook_profile(), 5, probe_obj=probe,
                                        sleep=clock.sleep, monotonic=clock.monotonic)
        # zones stay hot -> stop thermal
        self.assertEqual((result.status, result.reason), ("stop", "thermal"))

    def test_memory_pressure_after_cooldown_stops(self):
        clock = FakeClock()
        probe = ScriptedProbe(meminfo=MEMINFO_LOW)
        result = monitors.probe(notebook_profile(), probe_obj=probe)
        self.assertEqual((result.status, result.reason), ("cooldown", "memory_pressure"))
        result = monitors.wait_cooldown(notebook_profile(), 5, probe_obj=probe,
                                        sleep=clock.sleep, monotonic=clock.monotonic)
        self.assertEqual((result.status, result.reason), ("stop", "memory_pressure"))

    def test_healthy_notebook_ok(self):
        result = monitors.probe(notebook_profile(), probe_obj=ScriptedProbe())
        self.assertEqual(result.status, "ok")


class DiscoveryTests(unittest.TestCase):
    def test_notebook_discovery_fields(self):
        probe = ScriptedProbe()
        identity = monitors.discover(notebook_profile(), probe)
        self.assertEqual(identity["cpu"], "Fake CPU 5650U")
        self.assertEqual(identity["vulkan"]["status"], "unavailable")
        self.assertEqual(identity["mem"]["MemAvailable_kb"], 8000000)

    def test_pixel_discovery_fields(self):
        identity = monitors.discover(pixel_profile(), ScriptedProbe())
        self.assertEqual(identity["model"], "Fake Pixel 10 Pro")
        self.assertEqual(identity["battery"]["level_pct"], 80)

    def test_device_expect_mismatch_detected(self):
        identity = monitors.discover(pixel_profile(), ScriptedProbe())
        expect_ok = pixel_profile(device_expect="Pixel 10 Pro")
        self.assertTrue(monitors.device_matches(expect_ok, identity))
        bad = pixel_profile(device_expect="Pixel 8")
        self.assertFalse(monitors.device_matches(bad, identity))
        self.assertIsNone(monitors.device_matches(pixel_profile(), identity))


if __name__ == "__main__":
    unittest.main()
