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
"inconclusive" — a live demonstration that the comparator refuses to
manufacture a win from noise (issue #168's core requirement).

Real experiments: pin a workload, write a spec, run both arms, compare:

    python benchmarks/experiments/run_experiment.py pin-workload --audio clips/
    # -> paste the printed block into the spec's "workload"
    python benchmarks/experiments/run_experiment.py run --spec spec.json --out-dir runs/exp1
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


def _cmd_pin_workload(args: argparse.Namespace) -> int:
    files = sorted(p for p in args.audio.iterdir() if p.is_file()) if args.audio.is_dir() \
        else sorted(args.audio)
    if not files:
        print(f"no files under {args.audio}", file=sys.stderr)
        return 2
    manifest = workload_manifest(files)
    block = {
        "workload": {
            "audio": str(args.audio),
            "files": [f.name for f in files],
            "sha256": manifest["sha256"],
        }
    }
    print(json.dumps(block, indent=2))
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    # Seal the spec before any arm runs: the records embed this hash and the
    # comparator refuses pairs whose seals disagree with the compared spec.
    sealed = out_dir / "spec.sealed.json"
    sealed.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "spec.sha256").write_text(spec_sha256(spec) + "\n", encoding="utf-8")

    # Interleaved repeats: both arms execute repeat-by-repeat in the seeded
    # order (runner.arm_order), each in fresh processes.
    protocol = spec["protocol"]
    for repeat in range(protocol["repeats"]):
        for arm in runner_mod.arm_order(spec, repeat):
            runner_mod.run_arm(spec, arm, out_dir, REPO_ROOT)
    print(f"records written under {out_dir}/{{baseline,candidate}}/record.json")
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
    print(compare_mod.render_summary(result) if hasattr(compare_mod, "render_summary")
          else json.dumps(result, indent=2))
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
        "protocol": {
            "repeats": 4,
            "requests_per_repeat": 6,
            "warmup_requests": 1,
            "order": "interleaved_random",
            "seed": 20260919,
            "timeout_s": 60,
        },
        "acceptance": {
            "min_improvement_pct": 5.0,
            "max_ci_halfwidth_pct": 15.0,
            "max_regression_pct": 0.0,
        },
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
    for repeat in range(spec["protocol"]["repeats"]):
        for arm in runner_mod.arm_order(spec, repeat):
            runner_mod.run_arm(spec, arm, out_dir, REPO_ROOT)
    result = compare_mod.compare_directories(
        spec_path, out_dir / "baseline", out_dir / "candidate",
        out_path=out_dir / "comparison.json",
    )
    print(json.dumps({k: result[k] for k in ("verdict", "effect") if k in result},
                     indent=2, default=str))
    print("[demo] identical arms: an honest 'inconclusive' here demonstrates the "
          "no-false-win contract; full details in comparison.json")
    return 0 if result["verdict"] == "inconclusive" else 1


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
    p.add_argument("--out-dir", required=True, type=Path)
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
