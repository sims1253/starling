"""Benchmark: stock CPU ``processor(audio_list)`` vs GPU ``GpuMelExtractor``.

The stock feature extractor runs the entire 8-step mel pipeline on CPU and
returns CPU tensors, which is why ``feat_ms`` scales superlinearly with batch
size (68 ms at B=8 -> ~1 s at B=16 -- the throughput cliff). This benchmark
measures the win from moving the pipeline to GPU torch ops.

For each batch size in [1, 4, 8, 16] (uniform medium fixtures, regenerated from
``tests/fixtures/make_fixtures.py``), under the shared-GPU lock, with CUDA
events (warmup>=8, >=15 samples, 5s cap, median + p90):

  * ``stock_ms`` -- ``processor(audio_list)`` PLUS the H2D transfer of the
    resulting ``input_features`` / ``attention_mask`` to cuda (so the
    comparison is apples-to-apples with the GPU extractor, which returns cuda
    tensors). ``stock_to_gpu_h2d_included`` is set to True in the JSON.
  * ``gpu_ms`` -- ``GpuMelExtractor(audio_list)`` end-to-end (audio scatter
    to GPU, all 8 steps, no CPU roundtrip).

Derived: ``speedup = stock_ms / gpu_ms``.

Writes ``outputs/parakeet/mel_bench.json`` and prints a table.

GPU-contention guard: the script samples ``nvidia-smi`` GPU utilization before
and inside the lock; if util > 30% it REFUSES to run and
exits non-zero so the caller knows the bench was deferred. Re-run when
the GPU is idle.

Run:  uv run python benchmarks/parakeet/bench_mel.py
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
from transformers import AutoProcessor

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "tests" / "fixtures"))
sys.path.insert(0, str(_REPO_ROOT / "src"))

import make_fixtures as mkfx  # noqa: E402
from starling.parakeet.gpu_lock import with_gpu_lock  # noqa: E402
from starling.parakeet.mel_gpu import GpuMelExtractor  # noqa: E402

MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"
SAMPLE_RATE = 16000
OUTPUTS = _REPO_ROOT / "outputs" / "parakeet"

WARMUP = 8
REPEATS = 15
MAX_SECONDS = 5.0
GPU_UTIL_THRESHOLD_PCT = 30   # defer if util > 30%
BATCH_SIZES = [1, 4, 8, 16]


def _suppress() -> None:
    warnings.filterwarnings("ignore", message=".*max_length.*")
    warnings.filterwarnings("ignore", message=".*RNN module weights.*")


def gpu_utilization_pct() -> int | None:
    """Sample GPU utilization via nvidia-smi (returns None if unavailable)."""
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
        # take the first GPU's reading
        return int(out.splitlines()[0].strip())
    except Exception:
        return None


def assert_gpu_idle(*, where: str) -> None:
    util = gpu_utilization_pct()
    if util is not None and util > GPU_UTIL_THRESHOLD_PCT:
        raise SystemExit(
            f"[bench_mel] GPU util={util}% (> {GPU_UTIL_THRESHOLD_PCT}% threshold) "
            f"at {where}; deferring benchmark. Re-run when the "
            f"GPU is idle."
        )


def time_cuda(fn, *, warmup=WARMUP, repeats=REPEATS, max_s=MAX_SECONDS) -> tuple[float, float, int]:
    """Median + p90 GPU time (ms) for ``fn`` via cuda events."""
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


def main() -> int:
    _suppress()

    # refuse to run if the GPU is contended (re-check inside the lock too)
    assert_gpu_idle(where="startup")

    print("[bench_mel] loading processor + building GPU extractor ...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    extractor = GpuMelExtractor(processor, device="cuda")

    fixtures = mkfx.load_fixtures()
    for name in ("short", "medium", "long"):
        if not fixtures[name].any():
            raise RuntimeError(f"fixture {name} empty -- regenerate via make_fixtures")
    medium = fixtures["medium"]

    def stock_to_gpu(audio_list):
        """Stock processor + H2D of features/mask to cuda (apples-to-apples)."""
        inputs = processor(audio_list, sampling_rate=SAMPLE_RATE).to("cuda")
        return inputs["input_features"], inputs["attention_mask"]

    def gpu_extract(audio_list):
        return extractor(audio_list)

    # quick correctness check before timing (so a broken extractor doesn't
    # produce a "fast but wrong" benchmark)
    probe = [medium, medium, medium]
    s_feats, s_mask = stock_to_gpu(probe)
    g_feats, g_mask = gpu_extract(probe)
    max_abs = (g_feats.float() - s_feats.float()).abs().max().item()
    mask_match = torch.equal(g_mask, s_mask.bool())
    print(f"[bench_mel] correctness probe: max_abs={max_abs:.3e} mask_match={mask_match}")
    if max_abs >= 1e-3 or not mask_match:
        raise SystemExit(
            f"[bench_mel] extractor drift (max_abs={max_abs:.3e}, "
            f"mask_match={mask_match}); aborting bench"
        )

    results = []
    print("[bench_mel] acquiring GPU lock ...")
    with with_gpu_lock(session="parakeet", model=MODEL_ID,
                       eta_min=3, note="mel bench"):
        assert_gpu_idle(where="inside GPU lock")
        free, total = torch.cuda.mem_get_info()
        print(f"[bench_mel] lock held; GPU free={free/1e9:.1f}GB / {total/1e9:.1f}GB")

        for B in BATCH_SIZES:
            audio_list = mkfx.build_uniform_batch(medium, B)
            audio_seconds = sum(len(a) / SAMPLE_RATE for a in audio_list)

            stock_ms, stock_p90, n_s = time_cuda(lambda: stock_to_gpu(audio_list))
            gpu_ms, gpu_p90, n_g = time_cuda(lambda: gpu_extract(audio_list))
            speedup = stock_ms / gpu_ms if gpu_ms > 0 else float("inf")

            print(
                f"  B={B:2d}  stock={stock_ms:7.2f}ms (p90 {stock_p90:7.2f}, n={n_s}) "
                f" gpu={gpu_ms:7.2f}ms (p90 {gpu_p90:7.2f}, n={n_g}) "
                f" speedup={speedup:5.2f}x"
            )
            results.append({
                "batch_size": B,
                "audio_seconds": round(audio_seconds, 4),
                "stock_ms": round(stock_ms, 4),
                "stock_p90_ms": round(stock_p90, 4),
                "gpu_ms": round(gpu_ms, 4),
                "gpu_p90_ms": round(gpu_p90, 4),
                "speedup": round(speedup, 4),
                "stock_to_gpu_h2d_included": True,
                "n_stock_samples": n_s,
                "n_gpu_samples": n_g,
            })

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_id": MODEL_ID,
        "dtype": "float32",
        "device": torch.cuda.get_device_name(0),
        "method": (
            "cuda.Event, warmup>=8, >=15 samples (5s cap), median+p90; "
            "stock = processor(audio)+H2D to cuda (apples-to-apples vs GPU); "
            "gpu = GpuMelExtractor end-to-end"
        ),
        "fixture": "uniform medium (22.305s each)",
        "results": results,
    }
    out_path = OUTPUTS / "mel_bench.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\n[bench_mel] wrote {out_path}")

    rows = [
        [
            r["batch_size"], r["audio_seconds"],
            f"{r['stock_ms']:.2f}", f"{r['stock_p90_ms']:.2f}",
            f"{r['gpu_ms']:.2f}", f"{r['gpu_p90_ms']:.2f}",
            f"{r['speedup']:.2f}x",
        ]
        for r in results
    ]
    print("\n" + tabulate(
        rows,
        headers=["B", "audio_s", "stock_ms", "stock_p90", "gpu_ms", "gpu_p90", "speedup"],
        tablefmt="github",
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
