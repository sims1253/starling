"""Reproduce: does the parakeet-server return empty text on SAME-shape replay?

Hits one fixture 4x in a row to expose the per-shape replay-graph bug.
"""
import io, json, signal, subprocess, sys, time, urllib.request, uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests" / "fixtures"))
import make_fixtures as mkfx  # noqa: E402
import soundfile as sf  # noqa: E402

SERVER = "/home/m0hawk/Documents/parakeet.cpp/build-cuda/examples/server/parakeet-server"
MODEL = "/home/m0hawk/asr-bench/models/tdt-0.6b-v3-f16.gguf"
PORT = 8768


def post(audio, host="127.0.0.1", port=PORT):
    buf = io.BytesIO()
    sf.write(buf, audio, 16000, format="WAV", subtype="PCM_16")
    wb = buf.getvalue()
    b = "----pk" + uuid.uuid4().hex
    body = (b"--" + b.encode() + b"\r\nContent-Disposition: form-data; name=\"file\";"
            b" filename=\"a.wav\"\r\nContent-Type: audio/wav\r\n\r\n" + wb +
            b"\r\n--" + b.encode() + b"--\r\n")
    req = urllib.request.Request(
        f"http://{host}:{port}/v1/audio/transcriptions", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={b}"})
    return json.loads(urllib.request.urlopen(req, timeout=60).read().decode())["text"]


def main():
    subprocess.run(["pkill", "-f", "parakeet-server"],
                   capture_output=True); time.sleep(1)
    logf = open("/tmp/pks_replay.log", "w")
    p = subprocess.Popen([SERVER, "--model", MODEL, "--host", "127.0.0.1",
                          "--port", str(PORT)], stdout=logf, stderr=subprocess.STDOUT)
    try:
        for _ in range(40):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2); break
            except Exception:
                time.sleep(0.5)
        fx = mkfx.load_fixtures()
        for shape in ("short",):  # same shape repeated
            wav = fx[shape]
            print(f"=== {shape} x4 (same shape) ===")
            for i in range(4):
                t0 = time.perf_counter()
                txt = post(wav)
                print(f"  call {i}: {(time.perf_counter()-t0)*1000:.0f}ms "
                      f"len={len(txt)} head={txt[:35]!r}")
    finally:
        p.send_signal(signal.SIGTERM)
        try:
            p.wait(timeout=5)
        except Exception:
            p.kill()
        logf.close()
        print("--- server log (tail) ---")
        print(open("/tmp/pks_replay.log").read()[-1800:])


if __name__ == "__main__":
    main()
