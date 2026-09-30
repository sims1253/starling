"""Device discovery + safety probes (issue #176 §4).

Before launch the runner records the actual device/backend/driver
(`discover`); before each attempt and between gates it probes for states
that invalidate measurements or endanger hardware (`probe`):

- stop: adb disconnected, the #325 GPU-wedge marker present, wedge/driver
  failure patterns in the last gate's log, battery below the floor while
  discharging, memory pressure that survives a cooldown wait.
- cooldown: temperature above the profile's limit (phone battery/thermal
  severity, notebook thermal zones) -> wait, re-probe, stop if still hot.

Binding rules:

- NEVER reboot the phone, NEVER delete or modify the wedge marker, NEVER
  retry a load after a wedge. A wedge stop is terminal for the run;
  `resume` re-probes and refuses while the marker exists (the message tells
  the human to inspect and clear it manually).
- Charging is recorded; energy gates refuse charging themselves
  (phone_energy.sh exits non-zero) -> inconclusive, never zero.
- Every adb/sysfs read is injectable: a Probe object takes overridable
  `run`/`read`/`exists` callables so tests can simulate disconnected adb,
  wedge markers, hot -> cools, hot -> stays hot, low battery and memory
  pressure without hardware.

Stdlib only, Python 3.10+.
"""

from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

UNAVAILABLE = "unavailable"

# The Pixel fast-engine wedge marker (#325): written by cpp/fast/vk_runtime.cpp
# on VK_ERROR_OUT_OF_DEVICE_MEMORY / VK_ERROR_DEVICE_LOST / fence timeout.
WEDGE_MARKER_DEVICE = "/data/local/tmp/starling/starling-fast-gpu-wedged"

# Log patterns that mean the GPU driver failed or wedged (one constant, from
# vk_runtime.cpp's error strings plus the amdgpu timeout). Regexes.
WEDGE_PATTERNS = (
    "vkWaitForFences failed",
    r"VK_ERROR_OUT_OF_DEVICE_MEMORY",
    r"VK_ERROR_DEVICE_LOST",
    "device lost",
    "GPU driver is likely wedged",
    "a previous process observed GPU driver",
    r"amdgpu: .*timeout",
)

# BatteryManager status codes (see phone_energy.sh).
BATTERY_CHARGING = (2, 5)

DEFAULT_THRESHOLDS = {
    "notebook": {"max_temp_c": 85.0, "min_mem_available_mb": 2048.0},
    "pixel": {"max_temp_c": 43.0, "min_battery_pct": 40.0, "min_mem_available_mb": 1500.0},
}
COOLDOWN_INTERVAL_S = 5.0
ADB_TIMEOUT_S = 15.0


@dataclass
class ProbeResult:
    status: str  # "ok" | "cooldown" | "stop"
    reason: str = ""
    readings: dict = field(default_factory=dict)

    def __str__(self) -> str:
        if self.status == "ok":
            return "ok"
        return f"{self.status}: {self.reason}"


class Probe:
    """Injectable wrapper around every external read the monitors do."""

    def __init__(
        self,
        run: Callable[[list[str]], tuple[int, str]] | None = None,
        read: Callable[[str], str | None] | None = None,
        exists: Callable[[str], bool] | None = None,
    ):
        self._run = run or self._default_run
        self._read = read or self._default_read
        self._exists = exists or self._default_exists

    # -- overridable primitives ------------------------------------------
    def _default_run(self, cmd: list[str]) -> tuple[int, str]:
        try:
            out = subprocess.run(
                cmd, capture_output=True, text=True, timeout=ADB_TIMEOUT_S
            )
            return out.returncode, out.stdout
        except (OSError, subprocess.TimeoutExpired) as e:
            return 127, str(e)

    def _default_read(self, path: str) -> str | None:
        try:
            return Path(path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None

    def _default_exists(self, path: str) -> bool:
        return Path(path).exists()

    def run(self, cmd: list[str]) -> tuple[int, str]:
        return self._run(cmd)

    def read(self, path: str) -> str | None:
        return self._read(path)

    def exists(self, path: str) -> bool:
        return self._exists(path)

    # -- adb --------------------------------------------------------------
    def adb(self, serial: str, *args: str) -> tuple[int, str]:
        cmd = ["adb"]
        if serial:
            cmd += ["-s", serial]
        cmd += list(args)
        return self.run(cmd)

    def adb_state(self, serial: str) -> str:
        rc, out = self.adb(serial, "get-state")
        out = out.strip().splitlines()[0] if out.strip() else ""
        return out if rc == 0 else (out or UNAVAILABLE)

    def getprop(self, serial: str, prop: str) -> str:
        rc, out = self.adb(serial, "shell", f"getprop {prop}")
        return out.strip() if rc == 0 and out.strip() else UNAVAILABLE

    def shell(self, serial: str, command: str) -> tuple[int, str]:
        return self.adb(serial, "shell", command)

    # -- structured reads ---------------------------------------------------
    def battery(self, serial: str) -> dict:
        rc, out = self.shell(serial, "dumpsys battery")
        fields = {}
        if rc == 0:
            for line in out.splitlines():
                m = re.match(r"\s*(level|status|temperature|Charge counter):\s*(\S+)", line)
                if m:
                    fields[m.group(1)] = m.group(2)
        def _int(key):
            try:
                return int(str(fields.get(key, "")).strip())
            except (TypeError, ValueError):
                return None
        temp = _int("temperature")
        return {
            "level_pct": _int("level"),
            "status": _int("status"),
            "temperature_c": round(temp / 10.0, 1) if temp is not None else UNAVAILABLE,
        }

    def thermal_severity(self, serial: str) -> str:
        rc, out = self.shell(serial, "dumpsys thermalservice")
        if rc != 0:
            return UNAVAILABLE
        m = re.search(r"Severity:\s*(\w+)", out)
        return m.group(1).upper() if m else UNAVAILABLE

    def meminfo(self, serial: str | None = None) -> dict:
        if serial is None:
            text = self.read("/proc/meminfo")
        else:
            rc, out = self.shell(serial, "cat /proc/meminfo")
            text = out if rc == 0 else None
        if not text:
            return {"MemAvailable_kb": None, "MemTotal_kb": None}
        values = {}
        for key in ("MemTotal", "MemAvailable"):
            m = re.search(rf"^{key}:\s+(\d+) kB", text, re.MULTILINE)
            values[key + "_kb"] = int(m.group(1)) if m else None
        return values

    def thermal_zones_notebook(self) -> list[float]:
        temps = []
        for i in range(24):
            text = self.read(f"/sys/class/thermal/thermal_zone{i}/temp")
            if text is None:
                continue
            try:
                t = float(text.strip())
            except ValueError:
                continue
            temps.append(t / 1000.0 if t > 1000 else t)
        return temps

    def wedge_marker_present(self, profile: dict, serial: str | None) -> bool | str:
        if profile.get("device") == "pixel":
            rc, _ = self.shell(serial or "", f"test -f {WEDGE_MARKER_DEVICE}")
            return rc == 0
        cache_dir = _env_fast_cache_dir()
        if cache_dir is None:
            return "unchecked"  # no cache dir configured: nothing to check
        return self.exists(str(Path(cache_dir) / "starling-fast-gpu-wedged"))


def _env_fast_cache_dir() -> str | None:
    import os

    return os.environ.get("STARLING_FAST_CACHE_DIR")


def _thresholds(profile: dict) -> dict:
    merged = dict(DEFAULT_THRESHOLDS.get(profile.get("device", "notebook"), {}))
    merged.update(profile.get("thresholds") or {})
    return merged


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover(profile: dict, probe: Probe | None = None) -> dict:
    """Record the actual device/backend/driver at launch. Missing -> unavailable."""
    probe = probe or Probe()
    if profile.get("device") == "pixel":
        serial = profile.get("adb_serial") or ""
        battery = probe.battery(serial)
        return {
            "device": "pixel",
            "adb_state": probe.adb_state(serial),
            "model": probe.getprop(serial, "ro.product.model"),
            "soc_model": probe.getprop(serial, "ro.soc.model"),
            "build_fingerprint": probe.getprop(serial, "ro.build.fingerprint"),
            "egl": probe.getprop(serial, "ro.hardware.egl"),
            "vulkan": probe.getprop(serial, "ro.hardware.vulkan"),
            "battery": battery,
            "thermal_severity": probe.thermal_severity(serial),
            "mem": probe.meminfo(serial),
        }
    cpu = UNAVAILABLE
    text = probe.read("/proc/cpuinfo")
    if text:
        m = re.search(r"^model name\s*:\s*(.+)$", text, re.MULTILINE)
        cpu = m.group(1).strip() if m else UNAVAILABLE
    kernel = UNAVAILABLE
    text = probe.read("/proc/version")
    if text:
        kernel = text.strip()
    vulkan = _vulkaninfo(probe)
    gpus = []
    for card in ("card0", "card1", "card2"):
        vendor = probe.read(f"/sys/class/drm/{card}/device/vendor")
        device = probe.read(f"/sys/class/drm/{card}/device/device")
        if vendor or device:
            gpus.append({
                "card": card,
                "vendor": (vendor or "").strip() or UNAVAILABLE,
                "device": (device or "").strip() or UNAVAILABLE,
            })
    return {
        "device": "notebook",
        "cpu": cpu,
        "kernel": kernel,
        "vulkan": vulkan,
        "drm_devices": gpus,
        "mem": probe.meminfo(None),
    }


def _vulkaninfo(probe: Probe) -> dict:
    rc, out = probe.run(["vulkaninfo", "--summary"])
    if rc != 0:
        return {"status": UNAVAILABLE}
    info = {"status": "present"}
    for key, field_name in (
        ("deviceName", "device_name"), ("driverName", "driver_name"),
        ("driverInfo", "driver_info"), ("apiVersion", "api_version"),
    ):
        m = re.search(rf"{key}\s*=\s*(.+)", out)
        info[field_name] = m.group(1).strip() if m else UNAVAILABLE
    return info


def device_matches(profile: dict, identity: dict) -> bool | None:
    """Check the profile's device_expect against the discovered identity.

    Returns None when no expectation is set or the identity is unavailable
    (nothing to compare), True on match, False on mismatch.
    """
    expect = profile.get("device_expect")
    if not expect or not isinstance(expect, str):
        return None
    if profile.get("device") == "pixel":
        model = identity.get("model", UNAVAILABLE)
    else:
        model = identity.get("cpu", UNAVAILABLE)
    if model in (None, UNAVAILABLE):
        return None
    return expect.lower() in str(model).lower()


# ---------------------------------------------------------------------------
# Probing
# ---------------------------------------------------------------------------

def _hot_severity(sev: str) -> bool:
    return sev.upper() in ("MODERATE", "SEVERE", "CRITICAL", "EMERGENCY", "SHUTDOWN")


def probe(
    profile: dict,
    last_gate_log: str | None = None,
    probe_obj: Probe | None = None,
) -> ProbeResult:
    """One safety probe. ok / cooldown / stop (with reason and readings)."""
    p = probe_obj or Probe()
    thresholds = _thresholds(profile)
    device = profile.get("device", "notebook")
    readings: dict = {}
    serial = profile.get("adb_serial") or ""

    # Wedge/driver-failure patterns in the last gate's log: terminal.
    if last_gate_log:
        for pattern in WEDGE_PATTERNS:
            if re.search(pattern, last_gate_log):
                return ProbeResult(
                    "stop", "driver_failure",
                    {"pattern": pattern,
                     "message": "GPU driver failure pattern in the last gate log; "
                                "not retrying (issue #325)"},
                )
        readings["log_scanned_bytes"] = len(last_gate_log)

    marker = p.wedge_marker_present(profile, serial)
    readings["wedge_marker"] = marker if isinstance(marker, (bool, str)) else UNAVAILABLE
    if marker is True:
        return ProbeResult(
            "stop", "wedge",
            {**readings, "marker": WEDGE_MARKER_DEVICE if device == "pixel" else "STARLING_FAST_CACHE_DIR/starling-fast-gpu-wedged",
             "message": "GPU wedge marker present. Reboot the phone and clear the "
                        "marker manually after inspecting it; the campaign never "
                        "reboots, deletes the marker, or retries a load after a wedge."},
        )

    if device == "pixel":
        state = p.adb_state(serial)
        readings["adb_state"] = state
        if state != "device":
            return ProbeResult("stop", "adb_disconnected", readings)
        battery = p.battery(serial)
        readings["battery"] = battery
        charging = battery["status"] in BATTERY_CHARGING
        readings["charging"] = charging
        min_pct = thresholds.get("min_battery_pct")
        if (
            min_pct is not None
            and battery["level_pct"] is not None
            and battery["level_pct"] < min_pct
            and not charging
        ):
            return ProbeResult(
                "stop", "battery",
                {**readings, "message": f"battery {battery['level_pct']}% below floor {min_pct}% while discharging"},
            )
        temp = battery["temperature_c"]
        hot = isinstance(temp, (int, float)) and temp > thresholds.get("max_temp_c", 1e9)
        severity = p.thermal_severity(serial)
        readings["thermal_severity"] = severity
        if hot or _hot_severity(str(severity)):
            return ProbeResult(
                "cooldown", "thermal",
                {**readings, "message": f"phone too hot (battery {temp} C, severity {severity})"},
            )
    else:
        temps = p.thermal_zones_notebook()
        readings["thermal_zone_max_c"] = max(temps) if temps else UNAVAILABLE
        if temps and max(temps) > thresholds.get("max_temp_c", 1e9):
            return ProbeResult(
                "cooldown", "thermal",
                {**readings, "message": f"thermal zone max {max(temps):.1f} C above limit"},
            )

    mem = p.meminfo(serial if device == "pixel" else None)
    readings["mem"] = mem
    avail = mem.get("MemAvailable_kb")
    min_mb = thresholds.get("min_mem_available_mb")
    if avail is not None and min_mb is not None and avail < min_mb * 1024:
        return ProbeResult(
            "cooldown", "memory_pressure",
            {**readings, "message": f"MemAvailable {avail // 1024} MB below floor {min_mb:.0f} MB; waiting out a cooldown before stopping"},
        )

    return ProbeResult("ok", "", readings)


def wait_cooldown(
    profile: dict,
    max_s: float,
    last_gate_log: str | None = None,
    probe_obj: Probe | None = None,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> ProbeResult:
    """Re-probe in intervals up to `max_s`; still bad -> stop with the reason."""
    deadline = monotonic() + max(max_s, 0.0)
    result = probe(profile, last_gate_log, probe_obj)
    while result.status == "cooldown" and monotonic() < deadline:
        sleep(min(COOLDOWN_INTERVAL_S, max(deadline - monotonic(), 0.05)))
        result = probe(profile, last_gate_log, probe_obj)
    if result.status == "cooldown":
        return ProbeResult(
            "stop", result.reason,
            {**result.readings, "message": f"still {result.reason} after {max_s:.0f}s cooldown"},
        )
    return result
