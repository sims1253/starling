#!/usr/bin/env python3
"""serve_contract_smoke.py — trusted correctness gate: start a serve binary
(starling-serve or the contract fixture) on a free port, POST each input WAV
(default: one generated WAV), and require HTTP 200 with a JSON `text` field.

With --baseline-binary the same inputs go through the ORIGINAL baseline
binary first (fresh process, never concurrent) and the candidate's
transcripts must match it exactly — the exact numerical contract for
same-model kernel changes. A truncated or altered transcript is a failure
with a first-divergence line, however fast it was.

Prints `METRIC contract_ok=1`, `METRIC transcribe_ms=…` and, with a
baseline, `METRIC transcripts_match=0|1`, `METRIC output_chars=…`,
`METRIC baseline_chars=…`. Exit codes: 0 ok, 1 fail (wrong response or
divergent transcript), 3 inconclusive (cannot decide: server did not come
up, transport trouble).

Stdlib only; no models needed for the contract-fixture binary.

Usage:
    python benchmarks/experiments/serve_contract_smoke.py \
        --binary build-candidate/starling-serve \
        --baseline-binary build-baseline/starling-serve \
        --model parakeet [--gguf ${STARLING_MODELS_DIR}/model.gguf]
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
import wave
from pathlib import Path


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def make_wav(path: Path) -> None:
    # 0.5 s of 16 kHz mono silence; the fixture engine ignores audio content.
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 8000)


def wait_healthy(base: str, proc: subprocess.Popen, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with {proc.returncode} during startup")
        try:
            with urllib.request.urlopen(f"{base}/health", timeout=2.0) as r:
                if r.status == 200 and json.loads(r.read()).get("loaded"):
                    return
        except (OSError, ValueError):
            pass
        time.sleep(0.2)
    raise RuntimeError(f"server did not become healthy within {timeout_s}s")


def post_wav(base: str, wav: Path, model: str,
             timeout_s: float) -> tuple[float, str | None, str | None]:
    """-> (wall ms, error or None, transcript text or None)."""
    boundary = uuid.uuid4().hex
    data = wav.read_bytes()
    body = b"".join([
        f"--{boundary}\r\n".encode(),
        b'Content-Disposition: form-data; name="model"\r\n\r\n',
        model.encode(),
        f"\r\n--{boundary}\r\n".encode(),
        (f'Content-Disposition: form-data; name="file"; filename="{wav.name}"\r\n').encode(),
        b"Content-Type: audio/wav\r\n\r\n",
        data,
        f"\r\n--{boundary}--\r\n".encode(),
    ])
    req = urllib.request.Request(
        f"{base}/v1/audio/transcriptions",
        data=body,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            payload = json.loads(r.read())
            ms = (time.monotonic() - t0) * 1000
            # non-2xx arrives as HTTPError below; a 2xx response is open here
            if not isinstance(payload.get("text"), str):
                return ms, "response JSON has no string 'text' field", None
            return ms, None, payload["text"]
    except urllib.error.HTTPError as e:
        return (time.monotonic() - t0) * 1000, f"HTTP {e.code}", None
    except (OSError, ValueError) as e:
        return (time.monotonic() - t0) * 1000, f"transport: {e}", None


class Unavailable(RuntimeError):
    """The gate cannot decide (server never came up, transport trouble)."""


class Violation(RuntimeError):
    """The serve contract was violated (HTTP error, missing text)."""


def stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except OSError:
            proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                proc.kill()
            proc.wait(timeout=10)


def transcribe_all(binary: Path, args, wavs: list[Path], log_path: Path,
                   ) -> tuple[list[str], list[float]]:
    """One fresh server process for `binary`; every wav in order."""
    port = args.port or free_port()
    cmd = [str(binary), "--model", args.model,
           "--gguf", args.gguf or "/dev/null", "--port", str(port)]
    base = f"http://127.0.0.1:{port}"
    with open(log_path, "wb") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            try:
                wait_healthy(base, proc, args.startup_timeout)
            except RuntimeError as e:
                raise Unavailable(f"{binary.name}: {e}") from e
            texts, times = [], []
            for wav in wavs:
                ms, error, text = post_wav(base, wav, args.model, args.request_timeout)
                if error and error.startswith("transport"):
                    raise Unavailable(f"{binary.name} on {wav.name}: {error}")
                if error:
                    raise Violation(f"{binary.name} on {wav.name}: {error}")
                texts.append(text)
                times.append(ms)
            return texts, times
        finally:
            stop_server(proc)


def first_divergence(expected: str, got: str) -> str:
    ew, gw = expected.split(), got.split()
    for i, (a, b) in enumerate(zip(ew, gw)):
        if a != b:
            return f"first differing word at {i}: expected {a!r} got {b!r}"
    if len(ew) != len(gw):
        return f"word counts differ: expected {len(ew)} got {len(gw)}"
    return f"whitespace differs (lengths {len(expected)} vs {len(got)})"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True, help="candidate serve binary")
    ap.add_argument("--baseline-binary", default=None,
                    help="original baseline binary; enables the exact transcript check")
    ap.add_argument("--model", default="parakeet")
    ap.add_argument("--gguf", default=None)
    ap.add_argument("--wav", action="append", default=None,
                    help="input wav (repeatable; default: generated silence)")
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--startup-timeout", type=float, default=120.0)
    ap.add_argument("--request-timeout", type=float, default=120.0)
    args = ap.parse_args()

    binary = Path(args.binary)
    baseline = Path(args.baseline_binary) if args.baseline_binary else None
    for b in [binary] + ([baseline] if baseline else []):
        if not b.is_file():
            print("METRIC contract_ok=unavailable")
            print(f"binary missing: {b}", file=sys.stderr)
            return 3
    # Gate-owned artifacts (generated wav, server logs) go to a scratch dir —
    # never into the candidate's build tree, which must stay hashable and may
    # be read-only.
    scratch = Path(tempfile.mkdtemp(prefix="starling-smoke-"))
    if args.wav:
        wavs = [Path(w) for w in args.wav]
        missing = [str(w) for w in wavs if not w.is_file()]
        if missing:
            print("METRIC contract_ok=unavailable")
            print(f"input wav(s) missing: {', '.join(missing)}", file=sys.stderr)
            return 3
    else:
        wav = scratch / "smoke-input.wav"
        make_wav(wav)
        wavs = [wav]

    try:
        expected = None
        if baseline is not None:
            expected, _ = transcribe_all(baseline, args, wavs,
                                         scratch / "serve-baseline.log")
        texts, times = transcribe_all(binary, args, wavs, scratch / "serve.log")
    except Unavailable as e:
        print("METRIC contract_ok=unavailable")
        print(f"cannot decide: {e}", file=sys.stderr)
        return 3
    except Violation as e:
        print("METRIC contract_ok=0")
        print(f"DIVERGENCE: serve contract violated: {e}")
        return 1

    print("METRIC contract_ok=1")
    print(f"METRIC transcribe_ms={sum(times):.1f}")
    if expected is None:
        return 0
    print(f"METRIC output_chars={sum(len(t) for t in texts)}")
    print(f"METRIC baseline_chars={sum(len(t) for t in expected)}")
    if not any(t.strip() for t in expected):
        # Empty == empty would "match" vacuously: cannot confirm equality.
        print("METRIC transcripts_match=unavailable")
        print("baseline transcripts are empty; cannot confirm equality", file=sys.stderr)
        return 3
    for wav, want, got in zip(wavs, expected, texts):
        if want != got:
            print("METRIC transcripts_match=0")
            print(f"DIVERGENCE: {wav.name}: {first_divergence(want, got)}")
            return 1
    print("METRIC transcripts_match=1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
