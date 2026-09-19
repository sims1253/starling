"""The experiment runner: fresh processes, interleaved arms, cold/warm split.

Executes one arm of a sealed spec (record.py) and writes its record.json.
Protocol (issue #168):

- Per repeat, each arm runs in a FRESH server process (no warm state leaks
  across repeats, and arm order within a repeat is seeded-shuffled so
  systematic drift hits both arms equally).
- The first request per process is the cold sample (model load + graph
  capture): recorded, reported, and excluded from the gated estimate.
- protocol.warmup_requests further untimed requests follow, then the timed
  warm samples.
- Timeouts kill the whole process group and are recorded as failures; a run
  with more failures than tolerated flips to status=failed, which the
  comparator turns into an unavailable verdict.
- The workload manifest is verified against the spec pin BEFORE anything
  runs — a changed corpus is a refusal, never a silent baseline shift.

GPU serialization follows the SONAR harness contract: arms never run
concurrently (the runner is strictly sequential), and all STARLING_* GPU
lock environment variables are passed through to the server processes so an
externally held lock is honored.

Stdlib + requests only. Production server installs gain no dependency.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import requests

from record import (
    RECORD_SCHEMA,
    UNAVAILABLE,
    spec_sha256,
    validate_spec,
    workload_manifest,
    sha256_file,
)

TOOL_IDENTITY = {"tool": "starling-experiments", "version": 1}


class RunnerError(RuntimeError):
    pass


def _git(repo: Path, *args: str) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.TimeoutExpired):
        return UNAVAILABLE
    return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else UNAVAILABLE


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _hardware_identity() -> dict:
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            out = subprocess.run(
                [smi, "--query-gpu=name,driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=30,
            )
            if out.returncode == 0 and out.stdout.strip():
                name, driver = [x.strip() for x in out.stdout.strip().split(",", 1)]
                return {"hardware": name, "driver": driver}
        except (OSError, subprocess.TimeoutExpired, ValueError):
            pass
    return {"hardware": "cpu-only-host", "driver": UNAVAILABLE}


def collect_provenance(spec: dict, arm: dict, repo_root: Path, manifest: dict) -> dict:
    binary = Path(os.path.expandvars(arm["binary"]))
    model = arm.get("model")
    version_output = UNAVAILABLE
    if binary.is_file() and os.access(binary, os.X_OK):
        try:
            out = subprocess.run(
                [str(binary), "--version"], capture_output=True, text=True, timeout=30
            )
            version_output = out.stdout.strip() or UNAVAILABLE
        except (OSError, subprocess.TimeoutExpired):
            version_output = UNAVAILABLE
    return {
        "repo_revision": _git(repo_root, "rev-parse", "HEAD"),
        "ggml_revision": _git(repo_root / "third_party" / "ggml", "rev-parse", "HEAD"),
        "binary": str(binary),
        "binary_sha256": sha256_file(binary) if binary.is_file() else UNAVAILABLE,
        "binary_version": version_output,
        "build_flags": arm.get("build_flags", UNAVAILABLE),
        "runtime": _hardware_identity(),
        "optimization_env": {k: v for k, v in arm.get("env", {}).items()},
        "commands": [f"run_experiment.py run --spec <sealed spec> (arm recorded)"],
        "warmup_policy": (
            f"1 cold sample + {spec['protocol']['warmup_requests']} untimed warmup "
            f"request(s) before {spec['protocol']['requests_per_repeat']} timed samples "
            f"x {spec['protocol']['repeats']} fresh-process repeats"
        ),
        "seed": spec["protocol"]["seed"],
        "workload_sha256": manifest["sha256"],
        "normalizer": spec.get("normalizer_identity", "none-raw-wall-time"),
        "model_claim": {
            "model": model,
            "model_sha256": sha256_file(Path(model)) if model and Path(model).is_file() else None,
        },
        "metric_identity": {
            "tool": TOOL_IDENTITY["tool"],
            "metric": spec["metric"],
            "normalizer": spec.get("normalizer_identity", "none-raw-wall-time"),
        },
    }


class ArmServer:
    """One fresh server process: start -> health -> requests -> stop."""

    def __init__(self, arm: dict, port: int, log_path: Path):
        self.port = port
        env = dict(os.environ)
        env.update({k: str(v) for k, v in arm.get("env", {}).items()})
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(log_path, "wb")
        # Test doubles may declare arm["stub_args"] (see stub_serve.py);
        # real arms run the starling-serve CLI.
        if arm.get("stub_args"):
            cmd = [sys.executable, *arm["stub_args"], str(port)]
        else:
            cmd = [
                os.path.expandvars(arm["binary"]),
                "--model", arm.get("model_slug") or "parakeet",
                "--gguf", os.path.expandvars(arm.get("gguf") or "/dev/null"),
                "--port", str(port),
            ]
        self.proc = subprocess.Popen(
            cmd,
            stdout=self._log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        self.base = f"http://127.0.0.1:{port}"

    def wait_healthy(self, timeout_s: float) -> float:
        t0 = time.monotonic()
        deadline = t0 + timeout_s
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RunnerError(
                    f"server exited with {self.proc.returncode} during startup"
                )
            try:
                r = requests.get(f"{self.base}/health", timeout=2.0)
                if r.status_code == 200 and r.json().get("loaded"):
                    return time.monotonic() - t0
            except requests.RequestException:
                pass
            time.sleep(0.1)
        raise RunnerError(f"server did not become healthy within {timeout_s}s")

    def request(self, audio: Path, timeout_s: float) -> tuple[float, str | None]:
        t0 = time.monotonic()
        error = None
        try:
            with open(audio, "rb") as f:
                r = requests.post(
                    f"{self.base}/inference",
                    files={"file": (audio.name, f, "audio/wav")},
                    timeout=timeout_s,
                )
            if r.status_code != 200:
                error = f"HTTP {r.status_code}: {r.text[:200]}"
            else:
                r.json()["text"]
        except requests.RequestException as e:
            error = f"transport: {e}"
        except (ValueError, KeyError) as e:
            error = f"response shape: {e}"
        return (time.monotonic() - t0) * 1000.0, error

    def stop(self) -> None:
        if self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
            except OSError:
                self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except OSError:
                    self.proc.kill()
                self.proc.wait(timeout=10)
        self._log.close()


def run_arm(spec: dict, arm_name: str, run_dir: Path, repo_root: Path,
            stdout=sys.stdout) -> dict:
    problems = validate_spec(spec)
    if problems:
        raise RunnerError("spec invalid: " + "; ".join(problems))

    arm = spec["arms"][arm_name]
    protocol = spec["protocol"]
    audio_dir = Path(spec["workload"]["audio"])
    try:
        files = [audio_dir / name for name in spec["workload"]["files"]]
    except (KeyError, TypeError):
        files = []
    files = [f for f in files if f.is_file()]
    if not files:
        raise RunnerError(f"no workload files under {audio_dir}")
    manifest = workload_manifest(files)
    pin = spec["workload"].get("sha256")
    if pin is not None and manifest["sha256"] != pin:
        raise RunnerError(
            "workload manifest does not match the preregistered pin "
            f"({manifest['sha256']} != {pin}); "
            "re-pin deliberately (run_experiment.py pin-workload), never silently"
        )

    provenance = collect_provenance(spec, arm, repo_root, manifest)
    record = {
        "schema": RECORD_SCHEMA,
        "role": arm_name,
        "experiment_id": spec["experiment_id"],
        "spec_sha256": spec_sha256(spec),
        "provenance": provenance,
        "samples": [],
        "failures": [],
        "status": "ok",
    }

    arm_dir = run_dir / arm_name
    arm_dir.mkdir(parents=True, exist_ok=True)
    max_failures = int(spec.get("tolerated_failures", 0))
    timeout_s = float(protocol["timeout_s"])

    for repeat in range(protocol["repeats"]):
        rng = random.Random(spec["protocol"]["seed"] * 1000003 + repeat * 7919
                            + (0 if arm_name == "baseline" else 104729))
        server = ArmServer(arm, _free_port(), arm_dir / f"repeat{repeat}.server.log")
        try:
            server.wait_healthy(timeout_s)
            cold_done = False
            for i in range(protocol["warmup_requests"] + protocol["requests_per_repeat"]):
                audio = files[rng.randrange(len(files))]
                ms, error = server.request(audio, timeout_s)
                sample = {
                    "arm": arm_name,
                    "repeat": repeat,
                    "request": i,
                    "cold": not cold_done,
                    "warmup": i < protocol["warmup_requests"],
                    "wall_ms": round(ms, 3),
                }
                if error:
                    sample["error"] = error
                    record["failures"].append(
                        {"repeat": repeat, "request": i, "error": error}
                    )
                cold_done = True
                record["samples"].append(sample)
        except RunnerError as e:
            record["failures"].append({"repeat": repeat, "error": str(e)})
        finally:
            server.stop()
        if len(record["failures"]) > max_failures:
            record["status"] = "failed"
            break
        print(f"[experiment] {arm_name} repeat {repeat + 1}/{protocol['repeats']} done",
              file=stdout)

    (arm_dir / "record.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    return record


def arm_order(spec: dict, repeat: int) -> list[str]:
    """Interleaved arm order for one repeat (seeded, recorded, reproducible)."""
    if spec["protocol"]["order"] == "fixed":
        return ["baseline", "candidate"]
    rng = random.Random(spec["protocol"]["seed"] * 1000003 + repeat * 31)
    order = ["baseline", "candidate"]
    rng.shuffle(order)
    return order
