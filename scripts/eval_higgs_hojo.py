"""Turnkey eval driver for the higgs / hojo GGML engines on this branch.

Prepares everything needed to validate the branch so the GPU session is a
single command. Two modes:

``preflight`` (CPU-only — safe to run while the GPU is busy):
    uv run python scripts/eval_higgs_hojo.py preflight

  * libstarling_ggml loads via ctypes (no backend/GPU init) + ABI + backend
  * GGUFs, fixtures, and golden JSONs are present and parse
  * loader metadata-guard checks on both real GGUFs (build/loader_guard_test)
  * host conv stack bit-parity + speedup (build/hojo_conv_bench), if present

``run`` (GPU free):
    uv run python scripts/eval_higgs_hojo.py run [--build] [--repeats 3]
        [--skip-parity]

  1. optional rebuild:            cmake --build build -j
  2. byte-exact text parity:      pytest tests/test_ggml_parity.py -k "higgs or hojo"
  3. per-fixture phase timings:   STARLING_HIGGS/HOJO_TIMING=1 (stderr captured)
  4. RTFx cold + warm best-of-N:  per fixture per engine
  5. writes outputs/higgs_hojo_eval.json

The CPU guard binaries (build/loader_guard_test, build/hojo_conv_bench) are
dev scratch harnesses; the script uses them when present and skips otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import xml.etree.ElementTree as ET
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "benchmarks"))

FIXTURE_NAMES = ("short", "medium", "long")
GOLDENS = {
    "higgs": REPO / "golden" / "higgs_golden.json",
    "hojo": REPO / "golden" / "hojo_reference.json",
}
MODEL_ENV = {
    "higgs": "STARLING_GGML_HIGGS_MODEL",
    "hojo": "STARLING_GGML_HOJO_MODEL",
}


def _ok(cond: bool, what: str, detail: str = "") -> bool:
    print(f"  [{'PASS' if cond else 'FAIL'}] {what}{(' — ' + detail) if detail and not cond else ''}")
    return bool(cond)


@contextmanager
def _capture_fd2():
    """Capture the C library's stderr (phase timings) for one call."""
    sys.stderr.flush()
    with tempfile.TemporaryFile(mode="w+") as tmp:
        saved = os.dup(2)
        try:
            os.dup2(tmp.fileno(), 2)
            yield tmp
        finally:
            sys.stderr.flush()
            os.dup2(saved, 2)
            os.close(saved)
            tmp.seek(0)


def _fixtures() -> dict[str, tuple[Path, float]]:
    import soundfile as sf

    out = {}
    for name in FIXTURE_NAMES:
        path = REPO / "tests" / "fixtures" / f"{name}.wav"
        info = sf.info(str(path))
        out[name] = (path, info.frames / info.samplerate)
    return out


# --------------------------------------------------------------------------- #
# preflight (CPU-only)
# --------------------------------------------------------------------------- #
def cmd_preflight(args: argparse.Namespace) -> int:
    print("== higgs/hojo eval preflight (CPU-only, no GPU init) ==")
    good = True

    print("python ggml binding:")
    try:
        from starling._ggml import available, backend_name

        good &= _ok(available(), "libstarling_ggml loads + ABI matches")
        if available():
            print(f"      backend-for-build: {backend_name()}")
    except Exception as e:  # noqa: BLE001
        good &= _ok(False, "starling._ggml import", repr(e))

    print("models + fixtures + goldens:")
    for mdl, golden_path in GOLDENS.items():
        model = Path(os.environ.get(MODEL_ENV[mdl], REPO / "models" /
                                    ("higgs-audio-v3-bf16-exact.gguf" if mdl == "higgs"
                                     else "hojo-asr-v1.gguf")))
        good &= _ok(model.exists(), f"{mdl} GGUF present", str(model))
        try:
            fixtures = json.loads(golden_path.read_text())
            keys = sorted(fixtures.get("fixtures", fixtures).keys())
            good &= _ok(sorted(keys) == sorted(FIXTURE_NAMES), f"{mdl} golden parses",
                        f"keys={keys}")
        except Exception as e:  # noqa: BLE001
            good &= _ok(False, f"{mdl} golden parses ({golden_path})", repr(e))
    try:
        fx = _fixtures()
        for name, (path, dur) in fx.items():
            good &= _ok(path.exists(), f"fixture {name}.wav ({dur:.1f}s)")
    except Exception as e:  # noqa: BLE001
        good &= _ok(False, "fixtures readable", repr(e))

    print("engines selectable:")
    try:
        from engines import StarlingGgmlHiggs, StarlingGgmlHojo

        good &= _ok(StarlingGgmlHiggs().available, "StarlingGgmlHiggs available")
        good &= _ok(StarlingGgmlHojo().available, "StarlingGgmlHojo available")
    except Exception as e:  # noqa: BLE001
        good &= _ok(False, "engines import", repr(e))

    # Optional dev harnesses (build/loader_guard_test, build/hojo_conv_bench).
    guard = REPO / "build" / "loader_guard_test"
    if guard.exists():
        print("loader metadata guards (CPU-only, real GGUFs):")
        hojo_gguf = os.environ.get(MODEL_ENV["hojo"], str(REPO / "models" / "hojo-asr-v1.gguf"))
        higgs_gguf = os.environ.get(MODEL_ENV["higgs"],
                                    str(REPO / "models" / "higgs-audio-v3-bf16-exact.gguf"))
        for mode, path in (("higgs", higgs_gguf), ("hojo", hojo_gguf), ("hojo-width", hojo_gguf)):
            r = subprocess.run([str(guard), mode, path], capture_output=True, text=True)
            good &= _ok(r.returncode == 0, f"loader_guard_test {mode}", r.stdout.strip())
        r = subprocess.run([str(guard), "hojo-beam"], capture_output=True, text=True)
        good &= _ok(r.returncode == 0 and "B=4 -> ACCEPT" in r.stdout and
                    "B=5 -> REJECT" in r.stdout, "beam cap predicate", r.stdout.strip())
    else:
        print("loader metadata guards: skipped (build/loader_guard_test absent)")

    bench = REPO / "build" / "hojo_conv_bench"
    if bench.exists():
        print("host conv stack parity/perf (CPU-only, real weights, ~2 min):")
        r = subprocess.run([str(bench)], capture_output=True, text=True)
        good &= _ok(r.returncode == 0 and "ALL PARITY OK" in r.stdout,
                    "conv stack bit-identical + threaded", r.stdout.strip()[-200:])
        for line in r.stdout.splitlines():
            if "time:" in line:
                print(f"      {line.strip()}")
    else:
        print("host conv parity: skipped (build/hojo_conv_bench absent)")

    mel = REPO / "build" / "mel_parity_test"
    if mel.exists():
        print("whisper-mel unification parity (CPU-only, all 4 models):")
        r = subprocess.run([str(mel)], capture_output=True, text=True)
        good &= _ok(r.returncode == 0 and "ALL MEL PARITY OK" in r.stdout,
                    "lib whisper-mel bit-identical per model", r.stdout.strip()[-200:])
    else:
        print("whisper-mel parity: skipped (build/mel_parity_test absent)")

    print("PREFLIGHT:", "READY" if good else "NOT READY")
    return 0 if good else 1


# --------------------------------------------------------------------------- #
# run (GPU)
# --------------------------------------------------------------------------- #
def cmd_run(args: argparse.Namespace) -> int:
    print("== higgs/hojo eval run (GPU) ==")
    if args.build:
        print("-- rebuilding libstarling_ggml")
        subprocess.run(["cmake", "--build", str(REPO / "build"), "-j"], check=True)

    if not args.skip_parity:
        # One pytest process per model token in the expression: loading
        # moss+ark+higgs+hojo (28 GB of weights) in ONE process overflows a
        # 32 GB GPU because ggml's process-global backend does not return
        # realized weights + replay caches to the pool between modules.
        # Process exit is the only reliable VRAM fence.
        models = [t for t in re.split(r"\s+or\s+", args.parity_k) if t]
        failed = []
        for mdl in models:
            print(f"-- byte-exact parity: {mdl} (isolated process)")
            # Judge by test verdicts, not the process exit code: the moss/ark
            # ctypes processes hit a pre-existing ggml teardown double-free at
            # interpreter shutdown (RC 134) AFTER pytest has reported results.
            # The printed summary line always precedes the abort; junitxml
            # (finalized around the same point) is the backup verdict.
            with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as tf:
                xml_path = tf.name
            # Full test-name selector: bare short tokens like "ark" do not
            # substring-filter in pytest -k and would run (and load) every
            # model in one process.
            selector = f"starling_ggml_{mdl}_text_parity"
            r = subprocess.run(
                [sys.executable, "-m", "pytest", str(REPO / "tests" / "test_ggml_parity.py"),
                 "-k", selector, "-v", "--junitxml", xml_path],
                cwd=str(REPO), capture_output=True, text=True)
            tail = "\n".join((r.stdout + r.stderr).splitlines()[-12:])
            print("\n".join("    " + line for line in tail.splitlines()[-6:]))
            summaries = [line for line in r.stdout.splitlines()
                         if re.fullmatch(r"=+ .* in .*=?=*", line) and "passed" in line]
            ok = bool(summaries) and not any(
                re.search(r"\b\d+ (failed|error|errors)\b", line) for line in summaries)
            if not ok:
                try:
                    root = ET.parse(xml_path).getroot()
                    suites = root.findall(".//testsuite") or [root]
                    n_bad = sum(int(s.attrib.get("failures", "0")) +
                                int(s.attrib.get("errors", "0")) for s in suites)
                    n_run = sum(int(s.attrib.get("tests", "0")) for s in suites)
                    ok = n_run > 0 and n_bad == 0
                except ET.ParseError:
                    pass
            os.unlink(xml_path)
            if not ok:
                failed.append(mdl)
        if failed:
            print(f"PARITY FAILED for {failed} — aborting eval run")
            return 1

    import numpy as np
    import soundfile as sf
    from engines import StarlingGgmlHiggs, StarlingGgmlHojo

    fixtures = _fixtures()
    results: dict[str, dict] = {"engines": {}}
    wanted = [m.strip() for m in args.models.split(",") if m.strip()]
    for label, cls, timing_env in (
        ("higgs", StarlingGgmlHiggs, "STARLING_HIGGS_TIMING"),
        ("hojo", StarlingGgmlHojo, "STARLING_HOJO_TIMING"),
    ):
        if label not in wanted:
            continue
        if not cls().available:
            print(f"-- {label}: engine unavailable, skipping")
            continue
        print(f"-- {label}: loading")
        engine = cls()
        engine.load()
        per_fixture = {}
        old = os.environ.get(timing_env)
        os.environ[timing_env] = "1"
        try:
            for name, (path, audio_s) in fixtures.items():
                audio, sr = sf.read(str(path))
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                audio = np.ascontiguousarray(audio, dtype=np.float32)
                # Cold call (graph capture / first-touch) with phase timings.
                with _capture_fd2() as cap:
                    t0 = time.perf_counter()
                    text_cold = engine._run_one(audio)
                    cold = time.perf_counter() - t0
                    phases = cap.read()
                # Warm best-of-N.
                warm = float("inf")
                for _ in range(args.repeats):
                    t0 = time.perf_counter()
                    text = engine._run_one(audio)
                    warm = min(warm, time.perf_counter() - t0)
                assert text == text_cold, f"{label}/{name}: nondeterministic output"
                per_fixture[name] = {
                    "audio_s": round(audio_s, 3),
                    "cold_s": round(cold, 3),
                    "warm_s": round(warm, 3),
                    "rtfx_cold": round(audio_s / cold, 2),
                    "rtfx_warm": round(audio_s / warm, 2),
                    "chars": len(text),
                    "phases_cold": [line.strip() for line in phases.splitlines()],
                }
                print(f"   {name:6s} {audio_s:6.1f}s audio | cold {cold:7.2f}s "
                      f"(RTFx {audio_s / cold:6.2f}) | warm {warm:7.2f}s "
                      f"(RTFx {audio_s / warm:6.2f})")
                for line in per_fixture[name]["phases_cold"]:
                    print(f"          {line}")
        finally:
            if old is None:
                os.environ.pop(timing_env, None)
            else:
                os.environ[timing_env] = old
            engine.close()
        results["engines"][label] = per_fixture

    out = REPO / "outputs" / "higgs_hojo_eval.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"-- wrote {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    sub.add_parser("preflight", help="CPU-only readiness checks (no GPU init)")
    run_p = sub.add_parser("run", help="full GPU eval: parity + timings + RTFx")
    run_p.add_argument("--build", action="store_true", help="cmake --build build first")
    run_p.add_argument("--repeats", type=int, default=3, help="warm best-of-N repeats")
    run_p.add_argument("--skip-parity", action="store_true")
    run_p.add_argument("--parity-k", default="higgs or hojo",
                       help='pytest -k expression split on "or" into per-model '
                            "subprocesses (VRAM isolation); e.g. "
                            '"higgs or hojo or moss or ark"')
    run_p.add_argument("--models", default="higgs,hojo",
                       help="comma list limiting the timing/RTFx phase, e.g. --models hojo")
    args = ap.parse_args()
    return cmd_preflight(args) if args.mode == "preflight" else cmd_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
