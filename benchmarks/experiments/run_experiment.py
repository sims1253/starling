"""Command line for the shared experiment record + comparison (issue #168).

Single documented reproduction of a paired CPU experiment (uses the real
HTTP serving stack via the contract fixture binary — build it first):

    git submodule update --init --recursive
    cmake -B build-exp -DSTARLING_SERVE=ON -DSTARLING_GGML_TESTS=ON \
      -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF -DGGML_LLAMAFILE=OFF
    cmake --build build-exp -j --target starling-serve-contract-fixture
    python benchmarks/experiments/run_experiment.py demo \
        --binary build-exp/starling-serve-contract-fixture

The demo runs the SAME binary as both arms, so its honest verdict is
never "pass" (normally "inconclusive") — a live demonstration that the
comparator refuses to manufacture a win from noise (issue #168's core
requirement). The negative control is sized for shared CI runners
(issue #256): 12 fresh-process repeats and a 12% improvement bar, so
scheduler jitter cannot push every repeat's paired improvement past the
bar at once.

Real experiments: pin a workload, write a spec, run both arms, compare:

    python benchmarks/experiments/run_experiment.py pin-workload --audio clips/
    # -> paste the printed block into the spec's "workload"
    python benchmarks/experiments/run_experiment.py run --spec spec.json --run-dir runs/exp1
    python benchmarks/experiments/run_experiment.py compare --spec spec.json --run-dir runs/exp1

Stdlib only; production server installations gain no Python dependency.
"""

from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path

import compare as compare_mod
import runner as runner_mod
from record import (
    RecordError,
    load_spec,
    spec_sha256,
    validate_spec,
    workload_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

# The demo's negative control (issue #256). A false "pass" on identical arms
# requires the paired-improvement bootstrap CI to clear min_improvement_pct
# ENTIRELY, and the bootstrap resamples whole fresh-process repeats: with only
# 4 repeats, shared-runner scheduler jitter (per-process cluster effects of
# ±10% are routine on sub-millisecond HTTP round trips) occasionally tips
# every repeat the same way and the CI manufactures a win. 12 repeats give the
# cluster bootstrap enough independent processes to concentrate the CI on the
# true 0% effect, and the 12% bar demands a coordinated one-sided noise level
# that interleaved arms cannot plausibly produce (measured on the comparator:
# ~3-4% false-pass at 4 repeats / 5% bar under calibrated noise, ~0.01% here).
# tolerated_failures rides along because 24 fresh processes x 8 requests must
# tolerate a single stray transport hiccup without the demo degrading to
# "unavailable".
DEMO_PROTOCOL = {
    "repeats": 12,
    "requests_per_repeat": 6,
    "warmup_requests": 1,
    "order": "interleaved_random",
    "seed": 20260919,
    "timeout_s": 60,
}
DEMO_ACCEPTANCE = {
    "min_improvement_pct": 12.0,
    "max_ci_halfwidth_pct": 15.0,
    "max_regression_pct": 0.0,
}
DEMO_TOLERATED_FAILURES = 2


def _cmd_pin_workload(args: argparse.Namespace) -> int:
    audio = args.audio
    if audio.is_dir():
        files = sorted(p for p in audio.iterdir() if p.is_file())
        audio_dir = audio
    elif audio.is_file():
        # A single clip pins the same way a directory does.
        files = [audio]
        audio_dir = audio.parent
    else:
        print(f"no such file or directory: {audio}", file=sys.stderr)
        return 2
    if not files:
        print(f"no files under {audio}", file=sys.stderr)
        return 2
    manifest = workload_manifest(files)
    block = {
        "workload": {
            "audio": str(audio_dir),
            "files": [f.name for f in files],
            "sha256": manifest["sha256"],
        }
    }
    print(json.dumps(block, indent=2))
    return 0


def _run_interleaved(spec: dict, out_dir: Path) -> None:
    """Drive the interleaved protocol: repeat-by-repeat, both arms, seeded order.

    run_arm executes exactly one (repeat, arm) slot in a fresh process; this
    loop is what makes baseline and candidate repeats interleave (issue
    #168) instead of each arm running its repeats back-to-back.
    """
    protocol = spec["protocol"]
    for repeat in range(protocol["repeats"]):
        records = {}
        for arm in runner_mod.arm_order(spec, repeat):
            records[arm] = runner_mod.run_arm(spec, arm, out_dir, REPO_ROOT, repeat)
        if any(r["status"] == "failed" for r in records.values()):
            print("a run exceeded its tolerated failures; the comparison will be "
                  "unavailable — stopping early", file=sys.stderr)
            return


def _cmd_run(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec)
    run_dir = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    # Seal the spec before any arm runs: the records embed this hash and the
    # comparator refuses pairs whose seals disagree with the compared spec.
    sealed = run_dir / "spec.sealed.json"
    sealed.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (run_dir / "spec.sha256").write_text(spec_sha256(spec) + "\n", encoding="utf-8")

    _run_interleaved(spec, run_dir)
    print(f"records written under {run_dir}/{{baseline,candidate}}/record.json")
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec)
    try:
        result = compare_mod.compare_directories(
            args.spec, args.run_dir / "baseline", args.run_dir / "candidate",
            out_path=args.out,
        )
    except RecordError as e:
        print(f"comparison rejected: {e}")
        return 2
    print(compare_mod.render_summary(result))
    return 0 if result["verdict"] in ("pass", "fail", "inconclusive") else 3


def _demo_spec(demo_dir: Path, binary: Path) -> dict:
    # Deterministic tiny WAV (0.5 s of 16 kHz mono silence). The fixture
    # engine ignores audio content; the timing exercise is the HTTP path.
    wav = demo_dir / "audio" / "silence.wav"
    wav.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 8000)
    manifest = workload_manifest([wav])
    return {
        "schema": "starling-experiment-spec/1",
        "experiment_id": "fixture-paired-cpu-demo",
        "objective": "Demonstrate the paired experiment loop on identical arms; "
                     "the expected honest verdict is inconclusive.",
        "metric": "http_transcribe_wall_ms",
        "direction": "lower",
        "normalizer_identity": "none-raw-wall-time",
        "arms": {
            "baseline": {"binary": str(binary), "model_slug": "parakeet",
                         "model": None, "env": {}},
            "candidate": {"binary": str(binary), "model_slug": "parakeet",
                          "model": None, "env": {}},
        },
        "workload": {
            "audio": str(wav.parent),
            "files": [wav.name],
            "sha256": manifest["sha256"],
        },
        "protocol": dict(DEMO_PROTOCOL),
        "acceptance": dict(DEMO_ACCEPTANCE),
        "tolerated_failures": DEMO_TOLERATED_FAILURES,
    }


def _cmd_demo(args: argparse.Namespace) -> int:
    binary = Path(args.binary).resolve()
    if not binary.is_file():
        print(f"fixture binary not found: {binary}\n"
              "Build it first: cmake --build build-exp --target "
              "starling-serve-contract-fixture", file=sys.stderr)
        return 2
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = _demo_spec(out_dir, binary)
    problems = validate_spec(spec)
    if problems:
        print("demo spec invalid: " + "; ".join(problems), file=sys.stderr)
        return 2
    spec_path = out_dir / "demo-spec.json"
    spec_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    print(f"[demo] spec sealed at {spec_path} ({spec_sha256(spec)[:12]}…)")
    _run_interleaved(spec, out_dir)
    try:
        result = compare_mod.compare_directories(
            spec_path, out_dir / "baseline", out_dir / "candidate",
            out_path=out_dir / "comparison.json",
        )
    except RecordError as e:
        # e.g. an arm record is missing entirely after an early stop.
        print(f"[demo] comparison rejected: {e}", file=sys.stderr)
        return 1
    print(json.dumps({k: result[k] for k in ("verdict", "effect") if k in result},
                     indent=2, default=str))
    if result["verdict"] == "pass":
        # The demo's arms are the SAME binary: a "pass" here is exactly the
        # false win this harness exists to catch.
        print("[demo] ERROR: identical arms produced a 'pass' — the comparator "
              "manufactured a win from noise; see comparison.json", file=sys.stderr)
        return 1
    if result["verdict"] == "unavailable":
        # The demo spec tolerates 2 failures per arm, so an unusable run
        # means an arm accumulated 3+ failed requests — a defect signal,
        # not runner noise. Green only on completed-run verdicts
        # (inconclusive, or a noise-tipped fail).
        print(f"[demo] run unusable: {result.get('reason')} — the demo must "
              "complete both arms to demonstrate the no-false-win contract",
              file=sys.stderr)
        return 1
    print(f"[demo] identical arms -> verdict {result['verdict']!r}: no false win; "
          "full details in comparison.json")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run_experiment",
        description="Shared baseline-versus-candidate experiment records (#168)",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("pin-workload", help="print the workload manifest block for a spec")
    p.add_argument("--audio", required=True, type=Path)
    p.set_defaults(fn=_cmd_pin_workload)

    p = sub.add_parser("run", help="execute both arms of a sealed spec")
    p.add_argument("--spec", required=True, type=Path)
    p.add_argument("--run-dir", required=True, type=Path,
                   help="directory the records are written to and compare reads back")
    p.set_defaults(fn=_cmd_run)

    p = sub.add_parser("compare", help="apply the preregistered rules to two run records")
    p.add_argument("--spec", required=True, type=Path)
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--out", type=Path, help="write the comparison JSON here")
    p.set_defaults(fn=_cmd_compare)

    p = sub.add_parser("demo", help="reproduce a paired CPU experiment (fixture binary)")
    p.add_argument("--binary", required=True, type=Path,
                   help="path to starling-serve-contract-fixture")
    p.add_argument("--out-dir", required=True, type=Path)
    p.set_defaults(fn=_cmd_demo)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
