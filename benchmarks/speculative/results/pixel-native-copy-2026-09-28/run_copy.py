#!/usr/bin/env python3
"""Execute the sealed Pixel S1 native-copy parity and bounded latency pilot."""

import argparse
import datetime as dt
import hashlib
import json
import re
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / "protocol.json"
FIXTURES = ROOT / "fixtures.json"
ADB = "/home/m0hawk/android-sdk/platform-tools/adb"
SERIAL = "192.168.178.59:34113"
DEVICE = "/data/local/tmp/starling-issue-batch"
EXE = f"{DEVICE}/native-copy"
PORT = 18182
REMOTE_PORT = 8182
TIMING = re.compile(
    r"^S1_TIMING prompt=(\d+)tok gen=(\d+)tok total=([0-9.]+)ms "
    r"copy=(\d+) max_k=(\d+) accepted=(\d+)/(\d+) verify_calls=(\d+) "
    r"fallback_steps=(\d+) proposal=([0-9.]+)ms verify=([0-9.]+)ms "
    r"fallback=([0-9.]+)ms$", re.MULTILINE)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def adb(*args, timeout=35):
    return subprocess.run([ADB, "-s", SERIAL, *args], check=True,
                          capture_output=True, timeout=timeout).stdout.decode()


def remote_server_pids():
    result = subprocess.run([ADB, "-s", SERIAL, "shell", "pidof starling-serve"],
                            capture_output=True, text=True, timeout=15)
    if result.returncode == 1 and not result.stdout.strip() and not result.stderr.strip():
        return []
    if result.returncode != 0 or result.stderr.strip():
        raise RuntimeError(f"pidof failed: {result.stderr}")
    raw = result.stdout.strip()
    if raw and not all(part.isdecimal() for part in raw.split()):
        raise RuntimeError(f"unexpected server PID list: {raw!r}")
    return raw.split()


def state():
    from sys import path
    old = list(path)
    try:
        path.insert(0, str(ROOT.parent / "controlled-20260928"))
        from run_controlled import sample_state, bad_state
        s = sample_state(full=True)
        reason = bad_state(s)
        if reason:
            raise RuntimeError(reason)
        return s
    finally:
        path[:] = old


def checked_state(out, label):
    s = state()
    with (out / "states.jsonl").open("a") as file:
        file.write(json.dumps({"label": label, **s}) + "\n")
    return s


def post(case, transcript, deadline):
    data = json.dumps({"transcript": transcript}).encode()
    request = urllib.request.Request(f"http://127.0.0.1:{PORT}/normalize",
                                     data=data, headers={"Content-Type": "application/json"})
    start = time.monotonic_ns()
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("stage duration bound reached before request")
    with urllib.request.urlopen(request, timeout=min(180, remaining)) as response:
        answer = json.load(response)
    end = time.monotonic_ns()
    if not isinstance(answer.get("text"), str):
        raise RuntimeError(f"{case}: missing HTTP text")
    return answer["text"], (end - start) / 1e6


def run_process(stage, index, arm, fixtures, deadline, reference=None):
    out = ROOT / f"{stage}-{index}-{arm}"
    out.mkdir(exist_ok=False)
    cfg = json.loads(PROTOCOL.read_text())
    record = {"stage": stage, "index": index, "arm": arm,
              "protocol_sha256": sha(PROTOCOL), "fixture_sha256": sha(FIXTURES),
              "binary_sha256": cfg["binary_sha256"], "rows": [],
              "completed": False}
    def save():
        (out / "record.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    save()
    # The preceding process has exited. Cooling is measured before launch.
    cooling = checked_state(out, "pre_cool")
    if cooling["battery_temp_deci_c"] >= 370:
        for _ in range(6):
            time.sleep(30)
            cooling = checked_state(out, "cooling")
            if cooling["battery_temp_deci_c"] < 370:
                break
    record["directional_latency_allowed"] = cooling["battery_temp_deci_c"] < 370
    checked_state(out, "before_launch")
    if remote_server_pids():
        raise RuntimeError("starling-serve already running before launch")
    adb("forward", f"tcp:{PORT}", f"tcp:{REMOTE_PORT}")
    dump = f"{EXE}/ids.bin"
    env = (f"LD_LIBRARY_PATH={EXE} STARLING_ENGINE=ggml "
           f"STARLING_GGML_DEVICE=CPU STARLING_GGML_THREADS=6 "
           f"STARLING_S1_TIMING=1 STARLING_S1_DUMP_IDS={dump} ")
    if arm == "copy":
        env += "STARLING_S1_COPY_DRAFT=1 STARLING_S1_COPY_MAX_K=2 "
    remote = (f"cd {EXE} && env {env}./starling-serve --model s1 "
              f"--gguf {DEVICE}/s1-bf16.gguf --warmup --host 127.0.0.1 "
              f"--port {REMOTE_PORT}")
    pid = None
    launched = time.monotonic_ns()
    with (out / "serve.log").open("wb") as log:
        process = subprocess.Popen([ADB, "-s", SERIAL, "shell", remote],
                                   stdout=log, stderr=subprocess.STDOUT)
        try:
            ready_deadline = min(deadline, time.monotonic() + 180)
            while time.monotonic() < ready_deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"server exited during startup: {process.returncode}")
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as response:
                        health = json.load(response)
                    if health.get("phase") == "ready" and health.get("model") == "s1":
                        break
                except (urllib.error.URLError, TimeoutError, OSError):
                    pass
                time.sleep(1)
            else:
                raise RuntimeError("server did not become ready")
            record["launch_to_ready_ms"] = (time.monotonic_ns() - launched) / 1e6
            record["health"] = health
            pids = remote_server_pids()
            if len(pids) != 1:
                raise RuntimeError(f"ambiguous server PIDs: {pids!r}")
            pid = pids[0]
            checked_state(out, "ready")
            order = (["short", "medium"] * 2 if stage == "performance"
                     else ["short", "medium"])
            if stage == "performance":
                requests = [(name, "same_case_warmup") for name in ("short", "medium")]
                requests += [(name, "measured") for name in order]
            else:
                requests = []
                for name in order:
                    requests.extend(((name, "same_case_warmup"), (name, "diagnostic")))
            cases = {row["case"]: row for row in fixtures["cases"]}
            for n, (name, kind) in enumerate(requests):
                checked_state(out, f"before_{n}")
                adb("shell", f"rm -f {dump}")
                result, wall_ms = post(name, cases[name]["transcript"], deadline)
                local_ids = out / f"ids-{n}-{name}-{kind}.bin"
                adb("pull", dump, str(local_ids))
                if local_ids.stat().st_size == 0 or local_ids.stat().st_size % 4:
                    raise RuntimeError(f"invalid int32 ID dump {local_ids}")
                row = {"request_index": n, "case": name, "kind": kind,
                       "text": result, "host_http_ms": wall_ms,
                       "ids_file": local_ids.name, "ids_sha256": sha(local_ids),
                       "ids_bytes": local_ids.stat().st_size}
                if reference is not None and kind == "measured":
                    want = reference[name]
                    if row["text"] != want["text"] or row["ids_sha256"] != want["ids_sha256"]:
                        raise RuntimeError(f"{name} measured ID/text differs from preflight greedy")
                record["rows"].append(row)
                checked_state(out, f"after_{n}")
                save()
                print(f"{stage} {index} {arm} {n} {name} {kind}: {wall_ms:.1f} ms ID {row['ids_sha256'][:12]}", flush=True)
            checked_state(out, "after_requests")
        except BaseException as error:
            record["failure"] = repr(error)
            save()
            raise
        finally:
            # This also covers a startup failure before readiness supplied a PID.
            try:
                pids = [pid] if pid else remote_server_pids()
                if len(pids) > 1:
                    raise RuntimeError(f"cannot identify owned server on cleanup: {pids!r}")
                if pids:
                    adb("shell", f"kill -TERM {pids[0]}")
            except Exception as cleanup_error:
                record["cleanup_error"] = repr(cleanup_error)
                save()
                try:
                    process.terminate()
                except Exception:
                    pass
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            leftovers = remote_server_pids()
            try:
                adb("forward", "--remove", f"tcp:{PORT}")
            finally:
                if leftovers:
                    record["cleanup_error"] = f"remote server survived TERM: {leftovers!r}"
                    save()
                    raise RuntimeError(record["cleanup_error"])
    checked_state(out, "after_exit")
    record["process_ms"] = (time.monotonic_ns() - launched) / 1e6
    log = (out / "serve.log").read_text(errors="replace")
    if "backend=CPU" not in log or "model=s1" not in log:
        raise RuntimeError("actual server CPU backend/model log missing")
    timings = list(TIMING.finditer(log))
    if len(timings) != len(record["rows"]) + 1:
        raise RuntimeError(f"timing count {len(timings)} != startup + requests {len(record['rows'])+1}")
    record["startup_native_timing"] = timings[0].group(0)
    expected_mode = (1, 2) if arm == "copy" else (0, 0)
    if (int(timings[0].group(4)), int(timings[0].group(5))) != expected_mode:
        raise RuntimeError("startup native mode fields differ from arm")
    for row, match in zip(record["rows"], timings[1:]):
        native = {"prompt_tokens": int(match.group(1)), "generated_tokens": int(match.group(2)),
                  "native_total_ms": float(match.group(3)), "copy": int(match.group(4)),
                  "max_k": int(match.group(5)), "accepted": int(match.group(6)),
                  "proposed": int(match.group(7)), "verify_calls": int(match.group(8)),
                  "fallback_steps": int(match.group(9)), "proposal_ms": float(match.group(10)),
                  "verify_ms": float(match.group(11)), "fallback_ms": float(match.group(12))}
        if (native["copy"], native["max_k"]) != expected_mode:
            raise RuntimeError(f"bad native mode fields: {native}")
        if row["ids_bytes"] != 4 * native["generated_tokens"]:
            raise RuntimeError(f"partial or extra ID dump for request {row['request_index']}")
        row.update(native)
    record["completed"] = True
    save()
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("preflight", "performance"))
    args = parser.parse_args()
    if sha(FIXTURES) != json.loads(PROTOCOL.read_text())["fixture_sha256"]:
        raise RuntimeError("fixture hash changed")
    fixtures = json.loads(FIXTURES.read_text())
    preflight_file = ROOT / "preflight-reference.json"
    if args.stage == "preflight":
        if preflight_file.exists():
            raise RuntimeError("preflight already exists")
        deadline = time.monotonic() + 25 * 60
        greedy = run_process("preflight", 1, "greedy", fixtures, deadline)
        time.sleep(30)
        copy = run_process("preflight", 2, "copy", fixtures, deadline)
        ref = {r["case"]: {"text": r["text"], "ids_sha256": r["ids_sha256"]}
               for r in greedy["rows"] if r["kind"] == "diagnostic"}
        for row in copy["rows"]:
            if row["kind"] == "diagnostic" and (row["text"] != ref[row["case"]]["text"] or
                  row["ids_sha256"] != ref[row["case"]]["ids_sha256"]):
                raise RuntimeError(f"preflight parity failed: {row['case']}")
        preflight_file.write_text(json.dumps(ref, indent=2, ensure_ascii=False) + "\n")
        print("PREFLIGHT EXACT ID/TEXT PARITY PASS", flush=True)
    else:
        if not preflight_file.exists():
            raise RuntimeError("preflight parity gate missing")
        ref = json.loads(preflight_file.read_text())
        deadline = time.monotonic() + 45 * 60
        records = []
        for index, arm in enumerate(("greedy", "copy", "copy", "greedy"), 1):
            if index > 1:
                time.sleep(30)
            records.append(run_process("performance", index, arm, fixtures, deadline, ref))
        (ROOT / "performance-raw-summary.json").write_text(json.dumps(records, indent=2,
                                                                     ensure_ascii=False) + "\n")
        print("PERFORMANCE BLOCKS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
