"""Run the CUDA C++ kernel boundary corpus under NVIDIA Compute Sanitizer.

Run with ``uv run python scripts/sanitizer_gate.py`` on an NVIDIA GPU host.
Each tool gets a fresh process and a separate log. A missing GPU/tool, a test
failure, a sanitizer finding, or a timeout makes the command fail closed.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

TOOLS = ("memcheck", "initcheck", "racecheck", "synccheck")
ROOT = Path(__file__).resolve().parent.parent
TEST = ROOT / "tests" / "test_cuda_kernel_boundaries.py"


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
        command = [
            sanitizer,
            "--tool", tool,
            "--error-exitcode", "86",
            "--check-exit-code", "yes",
            sys.executable, "-m", "pytest", "-q", str(TEST),
        ]
        print(f"[{tool}] running {TEST.name}; log: {log}", flush=True)
        try:
            with log.open("w") as output:
                process = subprocess.run(
                    command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT,
                    timeout=args.timeout_seconds, check=False,
                )
            status = "pass" if process.returncode == 0 else "fail"
            result["tools"][tool] = {"status": status, "exit_code": process.returncode, "log": str(log)}
        except subprocess.TimeoutExpired:
            result["tools"][tool] = {"status": "timeout", "log": str(log)}
        print(f"[{tool}] {result['tools'][tool]['status']}", flush=True)
        (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")

    return 0 if all(item["status"] == "pass" for item in result["tools"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
