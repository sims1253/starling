"""Compare standalone direct Parakeet with the CPU ggml reference on one WAV."""

from __future__ import annotations

import argparse
import os
import re
import statistics
import subprocess
from pathlib import Path


def run(command: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, text=True, capture_output=True, env=env, check=True)
    return result


def value(output: str, key: str) -> str:
    for line in output.splitlines():
        if line.startswith(key + "="):
            return line[len(key) + 1 :]
    raise RuntimeError(f"missing {key} in benchmark output")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gguf", type=Path, required=True)
    p.add_argument("--wav", type=Path, required=True)
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--direct", type=Path, required=True)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--strict-ids", action="store_true", help="fail on differing blank timing")
    args = p.parse_args()
    if args.threads < 1 or args.runs < 1:
        p.error("threads and runs must be positive")
    env = os.environ.copy()
    env["STARLING_GGML_DEVICE"] = "cpu"
    env["STARLING_GGML_THREADS"] = str(args.threads)
    env["OMP_PROC_BIND"] = "true"
    root = Path(__file__).resolve().parent
    reference = run(
        ["python3", str(root / "bench_end_to_end.py"), "--library", str(args.library),
         "--model", "parakeet", "--gguf", str(args.gguf), "--wav", str(args.wav),
         "--threads", str(args.threads), "--iterations", str(args.runs), "--ids"], env
    )
    candidate = run(
        [str(args.direct), str(args.gguf), str(args.wav), str(args.threads),
         "--iterations", str(args.runs + 1)], env
    )
    ref_ids = value(reference.stdout, "ids")
    direct_ids = value(candidate.stdout, "ids")
    ref_text = value(reference.stdout, "transcript")
    direct_text = repr(value(candidate.stdout, "transcript"))
    blank = "8192"
    ref_nonblank = [token for token in ref_ids.split(",") if token != blank]
    direct_nonblank = [token for token in direct_ids.split(",") if token != blank]
    print(f"tokens_match={ref_ids == direct_ids} "
          f"nonblank_tokens_match={ref_nonblank == direct_nonblank} "
          f"transcript_match={ref_text == direct_text}")
    if ref_ids != direct_ids:
        lhs = ref_ids.split(",")
        rhs = direct_ids.split(",")
        first = next((i for i, pair in enumerate(zip(lhs, rhs)) if pair[0] != pair[1]),
                     min(len(lhs), len(rhs)))
        print(f"first_token_difference={first} ggml_ids={len(lhs)} direct_ids={len(rhs)}")
    times = [float(x) for x in re.findall(r"total_s=([0-9.]+)", candidate.stderr)]
    if len(times) != args.runs + 1:
        raise RuntimeError("could not parse direct iteration times")
    ref_time = float(re.search(r"warm_median_s=([0-9.]+)", reference.stdout).group(1))
    direct_time = statistics.median(times[1:])
    print(f"ggml_cpu_median_s={ref_time:.4f} direct_cpu_median_s={direct_time:.4f} "
          f"direct_over_ggml={ref_time / direct_time:.3f}x")
    print(f"ggml_backend={re.search(r'backend=([^ ]+)', reference.stdout).group(1)} "
          f"direct_backend=direct-cpu threads={args.threads}")
    print(f"ggml_transcript={ref_text}")
    print(f"direct_transcript={direct_text}")
    if ref_text != direct_text or (args.strict_ids and ref_ids != direct_ids):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
