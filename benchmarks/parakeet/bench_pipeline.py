"""Integrated end-to-end pipeline benchmark for nvidia/parakeet-tdt-0.6b-v3.

Times the fully integrated ``MegaParakeetPipeline.transcribe`` (audio -> text,
no CPU roundtrip) for uniform-medium batches [1, 4, 8, 16] and reports the
integrated realtime factor (RTF) plus a per-stage breakdown.

Method (under the shared-GPU lock, per comms.md §P1):
  * ``total_ms`` -- end-to-end ``pipeline.transcribe(audio_list)`` timed with a
    single cuda-event pair (mel + encoder + decode + batch_decode). This is the
    authoritative RTF number.
  * per-stage breakdown (``mel_ms`` / ``encoder_ms`` / ``decode_ms``) -- from
    ``pipeline.transcribe_with_timing`` (each stage bracketed by its own
    cuda-event pair + synchronize; decode includes ``batch_decode``).
  * cuda.Event + synchronize, warmup>=8, median + p90 over >=15 samples (8s cap).
  * The graphed-decoder graph is captured ONCE per (B, T_enc) shape during
    warmup; timed calls reuse it (capture is amortised, matching the
    production-realistic shape).

GPU-contention guard: samples ``nvidia-smi`` before and inside the lock; REFUSES
to run if util > 30% (comms.md §P1) and exits non-zero so the orchestrator knows
the bench was deferred.

Writes ``outputs/parakeet/pipeline_bench.json`` and prints a summary table.
Headline metric: integrated RTF at batch=8 uniform-medium (projection ~1600x:
mel 3.6ms + enc 56ms + dec 51ms ~= 111ms for 178s).

Run:  uv run python benchmarks/parakeet/bench_pipeline.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from tabulate import tabulate

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "tests" / "fixtures"))
sys.path.insert(0, str(_REPO_ROOT / "src"))

import make_fixtures as mkfx  # noqa: E402
from starling.parakeet.gpu_lock import with_gpu_lock  # noqa: E402
from starling.parakeet.pipeline import MegaParakeetPipeline  # noqa: E402

MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"
SAMPLE_RATE = 16000
OUTPUTS = _REPO_ROOT / "outputs" / "parakeet"
ORACLE = _REPO_ROOT / "outputs" / "oracle.json"
BASELINE = _REPO_ROOT / "outputs" / "baseline_bench.json"

WARMUP = 8
REPEATS = 15
MAX_SECONDS = 8.0
GPU_UTIL_THRESHOLD_PCT = 30   # comms.md: defer if util > 30%
BATCH_SIZES = [1, 4, 8, 16]


def _suppress() -> None:
    for mod in (
        "transformers.generation.utils",
        "transformers.models.parakeet.generation_parakeet",
        "torch.nn.modules.rnn",
    ):
        try:
            warnings.filterwarnings("ignore", module=mod)
        except Exception:
            pass
    warnings.filterwarnings("ignore", message=".*max_length.*")
    warnings.filterwarnings("ignore", message=".*RNN module weights.*")


def gpu_utilization_pct() -> int | None:
    """Sample GPU utilization via nvidia-smi (None if unavailable)."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.STDOUT,
            text=True,
            timeout=10,
        ).strip()
        return int(out.splitlines()[0].strip())
    except Exception:
        return None


def assert_gpu_idle(*, where: str) -> None:
    util = gpu_utilization_pct()
    if util is not None and util > GPU_UTIL_THRESHOLD_PCT:
        raise SystemExit(
            f"[bench_pipeline] GPU util={util}% (> {GPU_UTIL_THRESHOLD_PCT}% "
            f"threshold) at {where}; deferring benchmark per comms.md §P1. "
            f"Re-run when the GPU is idle."
        )


def time_end_to_end(fn, *, warmup=WARMUP, repeats=REPEATS, max_s=MAX_SECONDS):
    """Median + p90 end-to-end GPU time (ms) for ``fn`` via one cuda-event pair."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples: list[float] = []
    wall0 = time.perf_counter()
    for _ in range(repeats):
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        samples.append(s.elapsed_time(e))
        if time.perf_counter() - wall0 > max_s:
            break
    med = float(np.median(samples))
    p90 = float(np.percentile(samples, 90))
    return med, p90, len(samples)


def time_stages(pipe, audio_list, *, warmup=WARMUP, repeats=REPEATS, max_s=MAX_SECONDS):
    """Median per-stage ms (mel / encoder / decode) via ``transcribe_with_timing``.

    Warmup first (this also captures the graph for the shape if not yet done),
    then collect the per-stage medians.
    """
    for _ in range(warmup):
        pipe.transcribe_with_timing(audio_list)
    torch.cuda.synchronize()
    mel, enc, dec = [], [], []
    wall0 = time.perf_counter()
    for _ in range(repeats):
        _, t = pipe.transcribe_with_timing(audio_list)
        mel.append(t["mel_ms"])
        enc.append(t["encoder_ms"])
        dec.append(t["decode_ms"])
        if time.perf_counter() - wall0 > max_s:
            break
    return float(np.median(mel)), float(np.median(enc)), float(np.median(dec))


def main() -> int:
    _suppress()
    assert_gpu_idle(where="startup")

    print("[bench_pipeline] loading MegaParakeetPipeline ...")
    pipe = MegaParakeetPipeline(model_id=MODEL_ID, device="cuda", dtype=torch.bfloat16)

    fixtures = mkfx.load_fixtures()
    for name in ("short", "medium", "long"):
        if not fixtures[name].any():
            raise RuntimeError(f"fixture {name} empty -- regenerate via make_fixtures")
    medium = fixtures["medium"]

    oracle = {e["name"]: e["text"] for e in json.loads(ORACLE.read_text())}
    expected_medium = oracle["medium"]

    # baseline headline (for the "above baseline 295x" callout)
    baseline_rtf = None
    if BASELINE.exists():
        try:
            baseline_rtf = float(json.loads(BASELINE.read_text())["headline"]["rtf_median"])
        except (KeyError, ValueError):
            pass

    results = []
    print("[bench_pipeline] acquiring GPU lock ...")
    with with_gpu_lock(
        session="parakeet-mega", model=MODEL_ID,
        eta_min=5, note="pipeline bench",
    ):
        assert_gpu_idle(where="inside GPU lock")
        free, total = torch.cuda.mem_get_info()
        print(f"[bench_pipeline] lock held; GPU free={free/1e9:.1f}GB / "
              f"{total/1e9:.1f}GB")

        for B in BATCH_SIZES:
            audio_list = mkfx.build_uniform_batch(medium, B)
            audio_seconds = sum(len(a) / SAMPLE_RATE for a in audio_list)

            # correctness gate: integrated transcript must match the oracle
            texts = pipe.transcribe(audio_list)
            ok = all(t == expected_medium for t in texts)
            print(f"[bench_pipeline] B={B:2d} correctness: all==oracle ? {ok}")
            if not ok:
                raise SystemExit(
                    f"[bench_pipeline] B={B} transcript drift; aborting bench"
                )

            # (1) end-to-end transcribe (authoritative RTF number)
            total_ms, total_p90, n_t = time_end_to_end(
                lambda: pipe.transcribe(audio_list)
            )
            rtf = audio_seconds / (total_ms / 1000.0) if total_ms > 0 else 0.0

            # (2) per-stage breakdown (graph already captured above)
            mel_ms, enc_ms, dec_ms = time_stages(pipe, audio_list)

            print(
                f"  B={B:2d}  total={total_ms:7.2f}ms (p90 {total_p90:6.1f}, n={n_t}) "
                f" rtf={rfm(rtf):>6}x  | mel={mel_ms:5.2f} enc={enc_ms:6.2f} "
                f"dec={dec_ms:6.2f}"
            )
            results.append({
                "batch_size": B,
                "audio_seconds": round(audio_seconds, 4),
                "total_ms": round(total_ms, 4),
                "total_p90_ms": round(total_p90, 4),
                "mel_ms": round(mel_ms, 4),
                "encoder_ms": round(enc_ms, 4),
                "decode_ms": round(dec_ms, 4),
                "rtf": round(rtf, 4),
                "n_total_samples": n_t,
            })

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_id": MODEL_ID,
        "dtype": "bfloat16",
        "device": torch.cuda.get_device_name(0),
        "method": (
            "cuda.Event, warmup>=8, >=15 samples (8s cap), median+p90; "
            "total_ms = end-to-end pipeline.transcribe (mel+encoder+decode+"
            "batch_decode); per-stage mel/encoder/decode via transcribe_with_timing; "
            "graphed-decoder graph captured once per (B,T_enc) and amortised"
        ),
        "fixture": "uniform medium (22.305s each)",
        "baseline_rtf_b8_medium": baseline_rtf,
        "results": results,
    }
    out_path = OUTPUTS / "pipeline_bench.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\n[bench_pipeline] wrote {out_path}")

    rows = [
        [
            r["batch_size"], r["audio_seconds"],
            f"{r['total_ms']:.2f}", f"{r['total_p90_ms']:.2f}",
            f"{r['mel_ms']:.2f}", f"{r['encoder_ms']:.2f}", f"{r['decode_ms']:.2f}",
            f"{r['rtf']:.1f}x",
        ]
        for r in results
    ]
    print("\n" + tabulate(
        rows,
        headers=["B", "audio_s", "total_ms", "total_p90",
                 "mel_ms", "enc_ms", "dec_ms", "RTF"],
        tablefmt="github",
    ))
    headline = next((r for r in results if r["batch_size"] == 8), None)
    if headline is not None:
        base_str = (
            f" (baseline was {baseline_rtf:.0f}x -> "
            f"{headline['rtf'] / baseline_rtf:.1f}x over baseline)"
            if baseline_rtf else ""
        )
        print(
            f"\n*** HEADLINE batch=8 uniform-medium: integrated RTF = "
            f"{headline['rtf']:.1f}x  (total {headline['total_ms']:.1f}ms for "
            f"{headline['audio_seconds']:.1f}s audio | mel {headline['mel_ms']:.1f}"
            f" + enc {headline['encoder_ms']:.1f} + dec {headline['decode_ms']:.1f})"
            f"{base_str} ***"
        )
    return 0


def rfm(x: float) -> str:
    return f"{x:,.0f}" if x >= 1000 else f"{x:.1f}"


if __name__ == "__main__":
    raise SystemExit(main())
