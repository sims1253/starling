"""Reproduce the byte-exactness gap in a real PTX equivalence check.

This is a research probe, not an optimization gate. It needs CUDA, an NVIDIA
GPU, and a locally built Volta CLI. It does not modify project dependencies.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess
import tempfile


SOURCE = Path(__file__).with_name("ptx_rounding_witness.cu")


def run(*argv: str) -> str:
    result = subprocess.run(argv, text=True, capture_output=True, check=False)
    output = result.stdout + result.stderr
    if result.returncode:
        raise RuntimeError(f"{' '.join(argv)} exited {result.returncode}:\n{output}")
    return output


def normalize_pointer_annotations(ptx: str) -> str:
    """Remove syntax Volta cannot parse; this does not preserve a proof claim."""
    normalized, count = re.subn(r"\.ptr\s+\.align\s+\d+\s+", "", ptx)
    if count == 0:
        raise RuntimeError("nvcc PTX has no expected .ptr .align annotation to normalize")
    return normalized


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volta", type=Path, required=True, help="built Volta CLI")
    parser.add_argument("--nvcc", type=Path, default=Path("/usr/local/cuda/bin/nvcc"))
    parser.add_argument("--arch", default="sm_120")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="starling-ptx-rounding-") as tmp:
        directory = Path(tmp)
        raw_ptx = directory / "witness.ptx"
        normalized_ptx = directory / "witness-normalized.ptx"
        executable = directory / "witness"
        run(str(args.nvcc), "-std=c++17", f"-arch={args.arch}", "-ptx",
            str(SOURCE), "-o", str(raw_ptx))
        run(str(args.nvcc), "-std=c++17", f"-arch={args.arch}", "-DWITNESS_RUN",
            str(SOURCE), "-o", str(executable))
        gpu_result = run(str(executable)).strip()
        if gpu_result != "staged=0x3f80 direct=0x3f81":
            raise RuntimeError(f"unexpected GPU result: {gpu_result}")

        normalized_ptx.write_text(normalize_pointer_annotations(raw_ptx.read_text()))
        run(str(args.volta), "--no-log-file", "parse", str(normalized_ptx))
        verdict = run(
            str(args.volta), "--no-log-file", "compare", str(normalized_ptx),
            str(normalized_ptx), "--kernel1", "staged_sum", "--kernel2", "direct_sum",
            "-b", "1", "-g", "1", "--array", "x:0x10000:2:1:in",
            "--array", "y:0x20000:2:1:in", "--array", "z:0x30000:2:1:in",
            "--array", "out:0x40000:2:1:out", "--param", "ptr:x",
            "--param", "ptr:y", "--param", "ptr:z", "--param", "ptr:out",
            "--check-array", "out", "--no-profile",
        )
        if "EQUIVALENT" not in verdict or "NOT EQUIVALENT" in verdict:
            raise RuntimeError(f"Volta did not reproduce the expected result:\n{verdict}")
        print(gpu_result)
        print(verdict.strip())
        print("byte_exact_gate=unsupported (real-arithmetic equivalence misses bf16 rounding)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
