"""The experiment runner: fresh processes, interleaved arms, cold/warm split.

Executes one (repeat, arm) slot of a sealed spec (record.py) and merges it
into the arm's record.json. The CLI (run_experiment.py) is the single driver
of interleaving: for each repeat it calls run_arm once per arm in the seeded
arm_order, so baseline and candidate repeats alternate instead of running
back-to-back. Protocol (issue #168):

- Per repeat, each arm runs in a FRESH server process (no warm state leaks
  across repeats, and arm order within a repeat is seeded-shuffled so
  systematic drift hits both arms equally).
- The first request per process is the cold sample (model load + graph
  capture): recorded, reported, and excluded from the gated estimate.
- protocol.warmup_requests further untimed requests follow, then the timed
  warm samples — 1 cold + W warmup + T timed requests per process exactly.
- Both arms derive the per-request workload file from (seed, repeat,
  request) only, so a (repeat, request) pair always measures the same clip
  and the paired difference isolates the binary.
- A timed-out request is recorded as a failure with its diagnostics and the
  repeat moves on; the server's whole process group is torn down when the
  repeat ends. A run with more failures than tolerated flips to
  status=failed, which the comparator turns into an unavailable verdict.
- The workload manifest is verified against the spec pin BEFORE anything
  runs — a changed corpus is a refusal, never a silent baseline shift.

GPU serialization follows the SONAR harness contract: arms never run
concurrently (the runner is strictly sequential), all STARLING_* GPU lock
environment variables are passed through to the server processes so an
externally held lock is honored, and concurrent experiment processes on one
machine serialize through an advisory lock (see _experiment_lock).

Stdlib only: this module is importable without the repository's Python
project installed (production server installs gain no dependency).
"""

from __future__ import annotations

import contextlib
import json
import os
import random
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

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
    # No NVIDIA discovery: identify the CPU model so the comparator can
    # still tell machines apart (all "cpu-only-host" records would compare
    # as the same hardware otherwise).
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return {"hardware": line.split(":", 1)[1].strip(), "driver": UNAVAILABLE}
    except OSError:
        pass
    return {"hardware": "cpu-only-host", "driver": UNAVAILABLE}


@contextlib.contextmanager
def _experiment_lock():
    """Serialize concurrent experiment processes on one machine.

    The SONAR harness serializes GPU work through starling.gpu.session; this
    runner must stay importable without the repository project (stdlib-only
    contract), so it takes its own advisory flock while a server process is
    live: two `run`/`demo` commands on one host take turns instead of
    contaminating each other's timings or exhausting GPU memory.
    STARLING_GPU_LOCK_DISABLE=1 opts out, matching the SONAR harness
    convention.
    """
    if os.environ.get("STARLING_GPU_LOCK_DISABLE") == "1":
        yield
        return
    try:
        import fcntl
    except ImportError:  # non-POSIX: no advisory locks available
        yield
        return
    path = Path(os.environ.get(
        "STARLING_EXPERIMENT_LOCK",
        Path(tempfile.gettempdir()) / "starling-experiments.lock",
    ))
    fd = open(path, "w")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fd.close()  # releases the flock


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
        # real arms run the starling-serve CLI. The served weights are the
        # validated, provenance-hashed arms.<name>.model — there is no
        # separate unhashed "gguf" field that could silently differ from
        # the model the record claims.
        model = arm.get("model")
        if arm.get("stub_args"):
            cmd = [sys.executable, *arm["stub_args"], str(port)]
        else:
            cmd = [
                os.path.expandvars(arm["binary"]),
                "--model", arm.get("model_slug") or "parakeet",
                "--gguf", os.path.expandvars(model) if model else "/dev/null",
                "--port", str(port),
            ]
        try:
            self.proc = subprocess.Popen(
                cmd,
                stdout=self._log,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
        except OSError:
            self._log.close()  # recorded as a failure downstream; no fd leak
            raise
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
                with urllib.request.urlopen(f"{self.base}/health", timeout=2.0) as r:
                    if r.status == 200 and json.loads(r.read()).get("loaded"):
                        return time.monotonic() - t0
            except (OSError, ValueError):
                pass
            time.sleep(0.1)
        raise RunnerError(f"server did not become healthy within {timeout_s}s")

    def request(self, audio: Path, timeout_s: float) -> tuple[float, str | None]:
        t0 = time.monotonic()
        error = None
        boundary = uuid.uuid4().hex
        try:
            with open(audio, "rb") as f:
                data = f.read()
            body = b"".join([
                f"--{boundary}\r\n".encode("ascii"),
                (f'Content-Disposition: form-data; name="file"; '
                 f'filename="{audio.name}"\r\n').encode("utf-8"),
                b"Content-Type: audio/wav\r\n\r\n",
                data,
                f"\r\n--{boundary}--\r\n".encode("ascii"),
            ])
            req = urllib.request.Request(
                f"{self.base}/inference",
                data=body,
                method="POST",
                headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            )
            with urllib.request.urlopen(req, timeout=timeout_s) as r:
                if r.status != 200:
                    error = f"HTTP {r.status}"
                else:
                    json.loads(r.read())["text"]
        except urllib.error.HTTPError as e:
            try:
                detail = e.read(200).decode("utf-8", "replace")
            except OSError:
                detail = ""
            error = f"HTTP {e.code}: {detail}" if detail else f"HTTP {e.code}"
        except (OSError, ValueError, KeyError) as e:
            error = f"transport: {e}"
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
            repeat: int, stdout=sys.stdout) -> dict:
    """Execute ONE (repeat, arm) slot and merge it into the arm's record.

    The CLI interleaves: for each repeat it calls this once per arm in
    runner.arm_order's seeded order, so baseline and candidate repeats
    alternate (issue #168's drift-mitigation). Each call starts a FRESH
    server process, appends its samples to run_dir/<arm>/record.json, and
    returns the merged record — samples accumulate across the repeat calls.
    """
    problems = validate_spec(spec)
    if problems:
        raise RunnerError("spec invalid: " + "; ".join(problems))
    repeats = spec["protocol"]["repeats"]
    if not 0 <= repeat < repeats:
        raise RunnerError(f"repeat {repeat} outside the sealed protocol (0..{repeats - 1})")

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
    pin = spec["workload"]["sha256"]
    if manifest["sha256"] != pin:
        raise RunnerError(
            "workload manifest does not match the preregistered pin "
            f"({manifest['sha256']} != {pin}); "
            "re-pin deliberately (run_experiment.py pin-workload), never silently"
        )

    arm_dir = run_dir / arm_name
    arm_dir.mkdir(parents=True, exist_ok=True)
    record_path = arm_dir / "record.json"
    seal = spec_sha256(spec)
    if record_path.exists():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if (record.get("spec_sha256") != seal
                or record.get("role") != arm_name
                or record.get("experiment_id") != spec["experiment_id"]):
            raise RunnerError(
                f"{record_path} was produced by a different spec/arm; "
                "use a fresh run directory for a new experiment"
            )
        if any(s.get("repeat") == repeat for s in record.get("samples", [])):
            # e.g. re-running after _run_interleaved's early stop: appending
            # would duplicate (repeat, request) keys and only blow up later
            # in record validation. Refuse here, at run time, with guidance.
            raise RunnerError(
                f"{record_path} already contains repeat {repeat}; re-running "
                "into a partially complete run directory would duplicate "
                "samples — use a fresh run directory (the partial records "
                "can still be compared via the compare command)"
            )
    else:
        record = {
            "schema": RECORD_SCHEMA,
            "role": arm_name,
            "experiment_id": spec["experiment_id"],
            "spec_sha256": seal,
            "provenance": collect_provenance(spec, arm, repo_root, manifest),
            "samples": [],
            "failures": [],
            "status": "ok",
        }

    max_failures = int(spec.get("tolerated_failures", 0))
    timeout_s = float(protocol["timeout_s"])
    # Both arms share this per-repeat file order (no arm term in the seed):
    # a (repeat, request) pair must measure the SAME clip for the paired
    # difference to isolate the binary, not clip difficulty.
    rng = random.Random(protocol["seed"] * 1000003 + repeat * 7919)

    with _experiment_lock():
        try:
            server = ArmServer(arm, _free_port(), arm_dir / f"repeat{repeat}.server.log")
        except OSError as e:
            # A missing or non-executable binary is the most likely spec
            # error; record it like any other failure instead of escaping
            # as a traceback.
            record["failures"].append({"repeat": repeat, "error": f"cannot start server: {e}"})
        else:
            try:
                server.wait_healthy(timeout_s)
                # Exactly 1 cold + warmup_requests untimed + requests_per_repeat
                # timed requests per fresh process, in that order.
                for i in range(1 + protocol["warmup_requests"]
                               + protocol["requests_per_repeat"]):
                    audio = files[rng.randrange(len(files))]
                    ms, error = server.request(audio, timeout_s)
                    sample = {
                        "arm": arm_name,
                        "repeat": repeat,
                        "request": i,
                        "cold": i == 0,
                        "warmup": 0 < i <= protocol["warmup_requests"],
                        "audio": audio.name,
                        "wall_ms": round(ms, 3),
                    }
                    if error:
                        sample["error"] = error
                        record["failures"].append(
                            {"repeat": repeat, "request": i, "error": error}
                        )
                    record["samples"].append(sample)
            except RunnerError as e:
                record["failures"].append({"repeat": repeat, "error": str(e)})
            finally:
                server.stop()

    if len(record["failures"]) > max_failures:
        record["status"] = "failed"
    record_path.write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    if record["status"] != "failed":
        print(f"[experiment] {arm_name} repeat {repeat + 1}/{repeats} done",
              file=stdout)
    return record


def arm_order(spec: dict, repeat: int) -> list[str]:
    """Interleaved arm order for one repeat (seeded, recorded, reproducible)."""
    if spec["protocol"]["order"] == "fixed":
        return ["baseline", "candidate"]
    rng = random.Random(spec["protocol"]["seed"] * 1000003 + repeat * 31)
    order = ["baseline", "candidate"]
    rng.shuffle(order)
    return order
