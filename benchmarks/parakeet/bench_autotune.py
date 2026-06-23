"""Autotuner benchmark for the parakeet megakernel pipeline.

Runs the full ``steps_per_replay`` (K) sweep on the current GPU UNDER the shared
GPU lock, prints the K-vs-ms results table, writes the cached config, shows the
cache file path + contents, and confirms the pipeline picks up the cached config
on a fresh construction (instant, no re-sweep).

What this measures
------------------
For each ``K in {1, 4, 8, 16, 32, 64}`` it captures a fresh
:class:`GraphedDecoder(K=K)` on a representative B=8 medium batch (22.3 s each),
warms up, and times the full K-step decode loop (``_run_loop``) via cuda events
(median of 10). It then picks the best K with the noise-robust
:func:`starling.parakeet.autotune.pick_best_k` (prefer the GPU-tier default within
10% of the fastest), computes ``chunk_batch_size`` from live free VRAM, and
writes the result to ``~/.cache/starling/autotune_<gpu>.json``.

GPU-contention guard: samples GPU util before/inside the lock; defers if util >
30%.

Writes ``outputs/parakeet/autotune_bench.json`` and prints tables.

Run:  uv run python benchmarks/parakeet/bench_autotune.py
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
from transformers import AutoModelForTDT, AutoProcessor

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "tests" / "fixtures"))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from starling.parakeet import autotune as at  # noqa: E402
from starling.parakeet.gpu_lock import with_gpu_lock  # noqa: E402

MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"
SAMPLE_RATE = 16000
OUTPUTS = _REPO_ROOT / "outputs" / "parakeet"
K_VALUES = list(at.DEFAULT_K_VALUES)
WARMUP = 3
REPEATS = 10
GPU_UTIL_THRESHOLD_PCT = 30  # defer if util > 30%


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
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.STDOUT, text=True, timeout=10,
        ).strip()
        return int(out.splitlines()[0].strip())
    except Exception:
        return None


def assert_gpu_idle(*, where: str) -> None:
    util = gpu_utilization_pct()
    if util is not None and util > GPU_UTIL_THRESHOLD_PCT:
        raise SystemExit(
            f"[bench_autotune] GPU util={util}% (> {GPU_UTIL_THRESHOLD_PCT}% "
            f"threshold) at {where}; deferring benchmark. "
            f"Re-run when the GPU is idle."
        )


def _fallback_table() -> list[list[str]]:
    """Per-GPU-tier fallback (K, B) defaults table for the report."""
    rows = []
    for cc, vram, label in [
        ((12, 0), 34.0, "RTX 5090 (sm_120)"),
        ((8, 9), 24.0, "RTX 4090 (sm_89)"),
        ((8, 0), 80.0, "A100 80GB (sm_80)"),
        ((9, 0), 80.0, "H100 80GB (sm_90)"),
        ((8, 6), 24.0, "RTX 3090 (sm_86)"),
        ((7, 5), 8.0, "generic >=8GB"),
        ((7, 5), 6.0, "<8GB"),
    ]:
        tier, k, b = at._classify_tier(cc, vram)
        rows.append([label, f"sm_{cc[0]*10+cc[1]}", f"{vram:g}", tier,
                     str(k), str(b)])
    return rows


def main() -> int:
    _suppress()
    assert_gpu_idle(where="startup")

    print("[bench_autotune] loading model + processor ...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForTDT.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()

    base = at.detect_gpu()
    print(f"[bench_autotune] GPU: {base.gpu_name} | cc={base.compute_capability} "
          f"| vram={base.gpu_vram_gb:.2f} GB | fallback K={base.steps_per_replay} "
          f"B={base.chunk_batch_size}")

    cache_path = at.cache_path(base.gpu_name)
    print(f"[bench_autotune] cache path: {cache_path}")

    print("[bench_autotune] acquiring GPU lock ...")
    with with_gpu_lock(
        session="parakeet", model=MODEL_ID,
        eta_min=8, note="autotune sweep",
    ):
        assert_gpu_idle(where="inside GPU lock")
        free, total = torch.cuda.mem_get_info()
        print(f"[bench_autotune] lock held; GPU free={free/1e9:.1f}GB / "
              f"{total/1e9:.1f}GB")

        t0 = time.perf_counter()
        # force=True -> always sweep (the whole point of this bench). The bench
        # already holds the lock, so the sweep must NOT re-acquire it.
        cfg = at.autotune(
            model, processor, force=True, k_values=tuple(K_VALUES),
            warmup=WARMUP, repeats=REPEATS, acquire_lock=False,
        )
        sweep_seconds = time.perf_counter() - t0
        print(f"[bench_autotune] sweep done in {sweep_seconds:.1f}s")

    # ---- K-vs-ms results table ----
    sweep_rows = []
    for k in K_VALUES:
        ms = cfg.sweep_results.get(str(k))
        sweep_rows.append([k, f"{ms:.3f}" if ms is not None else "-"])
    print("\n=== K sweep (B=8 medium, median decode-loop ms) ===")
    print(tabulate(
        sweep_rows, headers=["K (steps_per_replay)", "decode_ms"],
        tablefmt="github",
    ))
    print(f"\n  -> chosen K={cfg.steps_per_replay}  "
          f"(fallback hint was K={base.steps_per_replay})")

    # ---- cache file contents ----
    cache_text = cache_path.read_text() if cache_path.exists() else "(missing)"
    print(f"\n=== cache file ({cache_path}) ===")
    print(cache_text)

    # ---- second run: instant (loads cache) ----
    t1 = time.perf_counter()
    cfg2 = at.autotune(model, processor)  # default force=False -> cache hit
    cache_load_seconds = time.perf_counter() - t1
    print(f"[bench_autotune] second run (cache load): {cache_load_seconds*1000:.2f} ms")
    assert cfg2.steps_per_replay == cfg.steps_per_replay, (
        "cache reload did not reproduce the swept config"
    )

    # ---- pipeline picks up the cached config ----
    from starling.parakeet.pipeline import MegaParakeetPipeline  # noqa: WPS433

    print("[bench_autotune] constructing MegaParakeetPipeline(autotune=True) "
          "to confirm it loads the cache ...")
    t2 = time.perf_counter()
    pipe = MegaParakeetPipeline(autotune=True, encoder_mode="graphed")
    pipe_load_seconds = time.perf_counter() - t2
    print(f"[bench_autotune] pipeline built in {pipe_load_seconds:.1f}s "
          f"(incl. model load); config K={pipe.config.steps_per_replay} "
          f"B={pipe.config.chunk_batch_size} autotuned={pipe.config.autotuned}")
    ok = (
        pipe.config.steps_per_replay == cfg.steps_per_replay
        and pipe.config.autotuned is True
    )
    print(f"[bench_autotune] pipeline picked up cached config? {ok}")
    del pipe, model
    torch.cuda.empty_cache()

    # ---- write the bench output ----
    payload = {
        "model_id": MODEL_ID,
        "device": base.gpu_name,
        "compute_capability": list(base.compute_capability),
        "gpu_vram_gb_total": round(base.gpu_vram_gb, 4),
        "method": (
            "cuda.Event, warmup=3, repeats=10, median; decode-loop time = "
            "GraphedDecoder._run_loop (excl. capture + batch_decode); B=8 medium "
            "(22.3s each); K selection = pick_best_k (prefer GPU-tier hint within "
            "10% of fastest)"
        ),
        "fallback_hint_k": base.steps_per_replay,
        "fallback_hint_b": base.chunk_batch_size,
        "sweep_seconds": round(sweep_seconds, 3),
        "cache_path": str(cache_path),
        "cache_load_ms": round(cache_load_seconds * 1000, 3),
        "sweep_results": cfg.sweep_results,
        "chosen_steps_per_replay": cfg.steps_per_replay,
        "chosen_chunk_batch_size": cfg.chunk_batch_size,
        "sweep_date": cfg.sweep_date,
        "cache_contents": json.loads(cache_text) if cache_path.exists() else None,
        "fallback_tier_table": _fallback_table(),
    }
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS / "autotune_bench.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\n[bench_autotune] wrote {out_path}")

    # ---- fallback defaults table ----
    print("\n=== fallback defaults by GPU tier ===")
    print(tabulate(
        _fallback_table(),
        headers=["GPU", "sm", "vram(GB)", "tier", "K", "B"],
        tablefmt="github",
    ))

    print(
        f"\n*** HEADLINE: {base.gpu_name} -> K={cfg.steps_per_replay} "
        f"B={cfg.chunk_batch_size} (autotuned in {sweep_seconds:.1f}s; "
        f"cache reload {cache_load_seconds*1000:.2f}ms) ***"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
