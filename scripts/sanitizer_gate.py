"""Run the CUDA C++ kernel boundary corpus under NVIDIA Compute Sanitizer.

Run with ``uv run python scripts/sanitizer_gate.py`` on an NVIDIA GPU host.
Each tool gets a fresh process and a separate log. A missing GPU/tool, a test
failure, a sanitizer finding, or a timeout makes the command fail closed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

TOOLS = ("memcheck", "initcheck", "racecheck", "synccheck")
ROOT = Path(__file__).resolve().parent.parent
TEST = ROOT / "tests" / "test_cuda_kernel_boundaries.py"
EXPECTED_TESTS = 47


def _report_result(report: Path, expected_tests: int) -> tuple[int, str | None]:
    if not report.exists():
        return 0, "pytest did not write a JUnit report"
    try:
        cases = ET.parse(report).findall(".//testcase")
    except ET.ParseError:
        return 0, "pytest wrote an invalid JUnit report"
    if len(cases) != expected_tests:
        return len(cases), f"expected {expected_tests} executed tests, found {len(cases)}"
    for case in cases:
        if any(case.find(tag) is not None for tag in ("skipped", "failure", "error")):
            return len(cases), "JUnit report contains skipped or failed tests"
    return len(cases), None


def _sanitizer_result(log: Path, tool: str) -> str | None:
    output = log.read_text(errors="replace")
    if tool == "racecheck":
        if re.search(r"RACECHECK SUMMARY:\s*0 hazards displayed \(0 errors, 0 warnings\)", output):
            return None
    elif re.search(r"ERROR SUMMARY:\s*0 errors", output):
        return None
    return f"missing clean {tool} summary"


def run_tool(
    sanitizer: str, tool: str, output_dir: Path, timeout_seconds: int,
    *, test: Path = TEST, expected_tests: int = EXPECTED_TESTS, cwd: Path = ROOT,
) -> dict[str, object]:
    """Require both executed pytest cases and a clean sanitizer summary."""
    log = output_dir / f"{tool}.log"
    report = output_dir / f"{tool}.junit.xml"
    report.unlink(missing_ok=True)  # a previous run must never certify this one
    command = [
        sanitizer, "--tool", tool, "--error-exitcode", "86", "--check-exit-code", "yes",
        sys.executable, "-m", "pytest", "-o", "addopts=", "-q",
        "--junitxml", str(report), str(test),
    ]
    env = os.environ.copy()
    env.pop("PYTEST_ADDOPTS", None)
    env.pop("PYTEST_PLUGINS", None)
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    try:
        with log.open("w") as output:
            process = subprocess.run(
                command, cwd=cwd, env=env, stdout=output, stderr=subprocess.STDOUT,
                timeout=timeout_seconds, check=False,
            )
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "log": str(log), "report": str(report)}

    executed, report_error = _report_result(report, expected_tests)
    sanitizer_error = _sanitizer_result(log, tool)
    errors = [error for error in (report_error, sanitizer_error) if error]
    if process.returncode:
        errors.insert(0, f"process exit code {process.returncode}")
    return {
        "status": "fail" if errors else "pass", "exit_code": process.returncode,
        "executed_tests": executed, "log": str(log), "report": str(report),
        "reason": "; ".join(errors) if errors else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tools", nargs="+", choices=TOOLS, default=list(TOOLS))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "sanitizer")
    parser.add_argument("--timeout-seconds", type=int, default=900)
    args = parser.parse_args()

    sanitizer = shutil.which("compute-sanitizer")
    if sanitizer is None:
        parser.error("compute-sanitizer is required on PATH")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")

    import torch

    if not torch.cuda.is_available():
        parser.error("a CUDA device is required; skipped tests are not a passing gate")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    result = {
        "schema_version": 1,
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "commit": commit,
        "gpu": torch.cuda.get_device_name(0),
        "cuda_runtime": torch.version.cuda,
        "torch": torch.__version__,
        "test": str(TEST.relative_to(ROOT)),
        "tools": {},
    }

    for tool in args.tools:
        log = args.output_dir / f"{tool}.log"
        print(f"[{tool}] running {TEST.name}; log: {log}", flush=True)
        result["tools"][tool] = run_tool(sanitizer, tool, args.output_dir, args.timeout_seconds)
        print(f"[{tool}] {result['tools'][tool]['status']}", flush=True)
        (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")

    return 0 if all(item["status"] == "pass" for item in result["tools"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
