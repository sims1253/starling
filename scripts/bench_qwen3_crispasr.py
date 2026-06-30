#!/usr/bin/env python3
"""CrispASR qwen3-1.7b benchmark across the short/medium/long tiers.

Same audio as the starling bench. Reports transcribe_ms + RTFx + transcript.
CrispASR uses the cstr/qwen3-asr-1.7b-GGUF f16 conversion (same weights).
"""
import json, re, subprocess, time
from pathlib import Path

ASR_BENCH = Path("/home/m0hawk/asr-bench")
BIN = ASR_BENCH / "bin/crispasr-linux-x86_64-cuda13/crispasr"
GGUF = ASR_BENCH / "models/qwen3-asr-1.7b-f16.gguf"
FIX = Path("/home/m0hawk/Documents/starling/tests/fixtures")
ENV = {
    "PATH": "/usr/bin:/bin",
    "LD_LIBRARY_PATH": f"{ASR_BENCH}/libs/usr/lib/x86_64-linux-gnu/openblas-pthread:{ASR_BENCH}/bin/parakeet-v0.3.2-bin-linux-cuda-x64",
    "CRISPASR_N_GPU_LAYERS": "999",
    "HOME": str(Path.home()),
}


def run(wav: str, reps=3):
    times = []
    text = ""
    for _ in range(reps):
        t0 = time.perf_counter()
        p = subprocess.run(
            [str(BIN), "--backend", "qwen3-1.7b", "-m", str(GGUF), "-f", wav,
             "-n", "400", "--gpu-backend", "cuda"],
            capture_output=True, text=True, env=ENV, cwd=str(ASR_BENCH), timeout=180,
        )
        times.append((time.perf_counter() - t0) * 1000.0)
        # transcript on stdout; timing line on stderr/stdout
        if not text:
            text = p.stdout.strip()
    return min(times), text


def main():
    # import durations from starling bench
    import sys
    sys.path.insert(0, str(Path("/home/m0hawk/Documents/starling-qwen3/src")))
    from starling.qwen3.audio import load_wav
    results = {}
    for label, fname in [("short", "short.wav"), ("medium", "medium.wav"), ("long", "long.wav")]:
        wavp = str(FIX / fname)
        wav, sr = load_wav(wavp)
        audio_s = wav.shape[1] / sr
        ms, text = run(wavp)
        rtfx = audio_s / (ms / 1000.0)
        results[label] = {"audio_s": round(audio_s, 2), "crispasr_ms": round(ms, 1), "crispasr_rtfx": round(rtfx, 1),
                          "transcript": text[:120]}
        print(f"[crispasr] {label:7s} {audio_s:5.1f}s: {ms:8.1f}ms ({rtfx:5.1f}x)  text={text[:60]!r}")
    out = Path("/home/m0hawk/Documents/starling-qwen3/golden/qwen3/bench_crispasr.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
