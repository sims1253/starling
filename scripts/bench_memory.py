#!/usr/bin/env python3
"""VRAM / memory consumption breakdown benchmark for Granite-Speech-4.1-2b.

Measures, on the 24.9 s sample audio (comparable across configs) and on the
long-audio chunked path:

* **Peak VRAM** (allocated + reserved) per config: stock generate, mega
  non-spec, mega spec -- via ``torch.cuda.max_memory_allocated``.
* **Breakdown**: model weights, KV-cache (StaticCache, analytic + measured),
  captured-graph static buffers, and peak activations during transcribe.
* **CPU/RAM**: peak RSS via ``resource.getrusage``.
* **VRAM vs chunk size** for the long-audio path (should be roughly constant).

Results are printed as tables and saved to ``outputs/memory_bench.json``.

Usage:
    .venv/bin/python scripts/bench_memory.py
"""

from __future__ import annotations

import json
import resource
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from starling.audio import build_inputs, load_sample_audio  # noqa: E402
from starling.config import (  # noqa: E402
    LLM_HEAD_DIM,
    LLM_NUM_KV_HEADS,
    LLM_NUM_LAYERS,
)
from starling.long_audio import (  # noqa: E402
    synthesize_long_audio,
    transcribe_long,
)
from starling.loader import load_model_and_processor  # noqa: E402
from starling.parakeet.gpu_lock import with_gpu_lock  # noqa: E402
from starling.pipeline import MegaPipeline  # noqa: E402

N_TOKENS = 100  # generation budget on the 24.9s sample (comparable to bench_clean)
MAX_CACHE_LEN = 640  # hardcoded StaticCache length


def _b2mb(x: float) -> float:
    return x / (1024.0 * 1024.0)


def _alloc_mb() -> float:
    return _b2mb(torch.cuda.memory_allocated())


def _reserved_mb() -> float:
    return _b2mb(torch.cuda.memory_reserved())


def _peak_alloc_mb() -> float:
    return _b2mb(torch.cuda.max_memory_allocated())


def _peak_reserved_mb() -> float:
    return _b2mb(torch.cuda.max_memory_reserved())


def _rss_mb() -> float:
    """Peak RSS of this process in MB (Linux: ru_maxrss is in KB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _analytic_cache_bytes(dtype_bytes: int = 2) -> int:
    """StaticCache footprint: num_layers * 2(K,V) * n_kv_heads * head_dim * max_cache_len."""
    return (
        LLM_NUM_LAYERS
        * 2
        * LLM_NUM_KV_HEADS
        * LLM_HEAD_DIM
        * MAX_CACHE_LEN
        * dtype_bytes
    )


def main() -> int:
    out_path = _REPO_ROOT / "outputs" / "memory_bench.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with with_gpu_lock(
        session="granite-mega",
        model="granite-speech-4.1-2b",
        eta_min=6,
        note="memory breakdown benchmark",
    ):
        rss_before = _rss_mb()
        print("[bench_memory] loading model + processor ...", flush=True)
        torch.cuda.reset_peak_memory_stats()
        model, proc = load_model_and_processor("eager")
        torch.cuda.synchronize()
        weights_mb = _alloc_mb()  # current allocated = model weights only
        print(f"[bench_memory] model weights = {weights_mb:.1f} MB allocated", flush=True)

        wav, sr = load_sample_audio()
        sample_dur = wav.shape[1] / sr
        inputs = build_inputs(proc, wav)
        feats = inputs["input_features"].bfloat16()
        ids = inputs["input_ids"]
        mask = inputs.get("input_features_mask")

        pipe = MegaPipeline(
            model, proc, encoder_mode="cudagraph", use_fused_llm=True
        )

        # ---- warmup so all CUDA graphs are captured (counts toward baseline) ----
        print("[bench_memory] warmup (capture graphs) ...", flush=True)
        pipe.transcribe(feats, ids, mask, max_new_tokens=N_TOKENS, speculative=False)
        pipe.transcribe(feats, ids, mask, max_new_tokens=N_TOKENS, speculative=True)
        with torch.inference_mode():
            model.generate(
                input_ids=ids, input_features=feats,
                attention_mask=inputs["attention_mask"],
                input_features_mask=mask,
                max_new_tokens=N_TOKENS, do_sample=False, num_beams=1,
            )
        torch.cuda.synchronize()
        pipe_baseline_mb = _alloc_mb()  # weights + StaticCache + graph static buffers
        cache_buffers_mb = pipe_baseline_mb - weights_mb
        print(
            f"[bench_memory] after warmup baseline = {pipe_baseline_mb:.1f} MB "
            f"(cache+buffers = {cache_buffers_mb:.1f} MB)\n",
            flush=True,
        )

        cache_analytic_mb = _b2mb(_analytic_cache_bytes())

        # ---- per-config peak VRAM on the 24.9s sample ----
        configs: list[dict] = []

        def _measure(name, fn, *, speculative=False):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            fn()
            torch.cuda.synchronize()
            peak_a = _peak_alloc_mb()
            peak_r = _peak_reserved_mb()
            act = peak_a - pipe_baseline_mb  # activations above the warm baseline
            row = {
                "config": name,
                "peak_allocated_mb": round(peak_a, 1),
                "peak_reserved_mb": round(peak_r, 1),
                "activations_delta_mb": round(max(act, 0.0), 1),
                "speculative": speculative,
            }
            configs.append(row)
            print(
                f"  {name:<22} peak_alloc={peak_a:7.1f}MB  "
                f"peak_reserved={peak_r:7.1f}MB  "
                f"activations+={max(act,0.0):6.1f}MB",
                flush=True,
            )

        print("[bench_memory] per-config peak VRAM (24.9s sample):", flush=True)

        def _stock():
            with torch.inference_mode():
                model.generate(
                    input_ids=ids, input_features=feats,
                    attention_mask=inputs["attention_mask"],
                    input_features_mask=mask,
                    max_new_tokens=N_TOKENS, do_sample=False, num_beams=1,
                )
        _measure("stock transformers", _stock)

        def _mega_nonspec():
            pipe.transcribe(feats, ids, mask, max_new_tokens=N_TOKENS, speculative=False)
        _measure("mega (non-spec)", _mega_nonspec, speculative=False)

        def _mega_spec():
            pipe.transcribe(feats, ids, mask, max_new_tokens=N_TOKENS, speculative=True)
        _measure("mega (speculative)", _mega_spec, speculative=True)

        # ---- VRAM vs chunk size (long-audio path; should be ~constant) ----
        print("\n[bench_memory] VRAM vs chunk size (long-audio, spec, 90s audio):", flush=True)
        chunk_rows = []
        wav90, _ = synthesize_long_audio(90, base_wav=wav, sr=sr)
        for cs in [15.0, 30.0, 45.0]:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            res = transcribe_long(
                pipe, proc, wav90, sr,
                chunk_seconds=cs, max_new_tokens=N_TOKENS, speculative=True,
            )
            torch.cuda.synchronize()
            pa = _peak_alloc_mb()
            pr = _peak_reserved_mb()
            chunk_rows.append({
                "chunk_seconds": cs,
                "n_chunks": res.n_chunks,
                "peak_allocated_mb": round(pa, 1),
                "peak_reserved_mb": round(pr, 1),
                "total_ms": round(res.total_ms, 1),
            })
            print(
                f"  chunk={cs:4.0f}s  n_chunks={res.n_chunks:>3}  "
                f"peak_alloc={pa:7.1f}MB  peak_reserved={pr:7.1f}MB",
                flush=True,
            )
        del wav90

        rss_after = _rss_mb()

        # ---- print breakdown table ----
        print("\n" + "=" * 70)
        print("MEMORY BREAKDOWN  (bf16, 24.9s sample, batch=1)")
        print("=" * 70)
        print(f"{'component':<32}{'MB':>12}{'notes':>24}")
        print("-" * 70)
        print(f"{'model weights':<32}{weights_mb:>12.1f}{'2.3B params bf16':>24}")
        print(f"{'KV cache (analytic)':<32}{cache_analytic_mb:>12.1f}"
              f"{f'{LLM_NUM_LAYERS}L*2*{LLM_NUM_KV_HEADS}KV*{LLM_HEAD_DIM}d*{MAX_CACHE_LEN}':>24}")
        print(f"{'cache + graph buffers (meas)':<32}{cache_buffers_mb:>12.1f}"
              f"{'after warmup':>24}")
        print("-" * 70)
        for c in configs:
            act_str = f"act+{c['activations_delta_mb']:.0f}"
            print(f"  {c['config']:<30}{c['peak_allocated_mb']:>12.1f}"
                  f"{act_str:>24}")
        print("-" * 70)
        print(f"{'CPU RSS (peak)':<32}{max(rss_before, rss_after):>12.1f}"
              f"{'ru_maxrss (KB->MB)':>24}")
        print("=" * 70)

        # ---- save JSON ----
        payload = {
            "config": {
                "max_cache_len": MAX_CACHE_LEN,
                "n_tokens": N_TOKENS,
                "sample_audio_seconds": round(sample_dur, 3),
                "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(0),
            },
            "weights_mb": round(weights_mb, 1),
            "cache_analytic_mb": round(cache_analytic_mb, 1),
            "cache_buffers_measured_mb": round(cache_buffers_mb, 1),
            "pipe_baseline_mb": round(pipe_baseline_mb, 1),
            "per_config_peak": configs,
            "vram_vs_chunk_size": chunk_rows,
            "cpu_rss_peak_mb": round(max(rss_before, rss_after), 1),
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\n[bench_memory] saved -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
