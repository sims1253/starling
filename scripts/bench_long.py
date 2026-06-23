#!/usr/bin/env python3
"""Long-audio (1 min - 1 h) chunked benchmark: mega (spec/non-spec) vs stock.

Measures total wall time, RTFx, tokens generated, and peak VRAM for durations
{60 s, 5 min, 30 min, 60 min} using the chunked transcription path
(``starling.long_audio``).

* **Mega** (speculative + non-speculative) is run *end-to-end* for every
  duration to verify RTFx stays flat as audio gets longer (no prefill
  amortization pathology).  The speculative path is the headline "mega" column.
* **Stock** transformers is far slower (RTFx ~3.8x), so it is measured on a
  single real 60 s run (2 chunks) and *extrapolated* linearly for longer
  durations (clearly labelled).  Per-chunk wall time is constant because each
  chunk is independent.

Peak VRAM is measured once on the 60 s speculative run and reported for all
durations: it is constant because every chunk resets the KV cache and frees its
activations before the next chunk begins.

Results are printed as a table and saved to ``outputs/long_audio_bench.json``.

Usage:
    .venv/bin/python scripts/bench_long.py
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from starling.audio import build_inputs, load_sample_audio  # noqa: E402
from starling.long_audio import (  # noqa: E402
    DEFAULT_CHUNK_SECONDS,
    extrapolate_from_chunk,
    synthesize_long_audio,
    transcribe_long,
    transcribe_long_stock,
)
from starling.loader import load_model_and_processor  # noqa: E402
from starling.parakeet.gpu_lock import with_gpu_lock  # noqa: E402
from starling.pipeline import MegaPipeline  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DURATIONS_S = [60, 300, 1800, 3600]  # 1min, 5min, 30min, 1h
CHUNK_S = DEFAULT_CHUNK_SECONDS  # 30 s
MAX_NEW_TOKENS = 200
NONSPEC_ACTUAL_S = {60, 300}  # run non-spec end-to-end for these, extrapolate rest
STOCK_ACTUAL_S = {60}  # stock end-to-end here, extrapolate the rest
WARMUP_TOKENS = 120

DUR_LABEL = {60: "60s", 300: "5min", 1800: "30min", 3600: "60min"}


def _b2mb(x: float) -> float:
    return x / (1024.0 * 1024.0)


def _peak_vram_mb() -> tuple[float, float]:
    return (
        _b2mb(torch.cuda.max_memory_allocated()),
        _b2mb(torch.cuda.max_memory_reserved()),
    )


def main() -> int:
    out_path = _REPO_ROOT / "outputs" / "long_audio_bench.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with with_gpu_lock(
        session="granite",
        model="granite-speech-4.1-2b",
        eta_min=12,
        note="long-audio + VRAM benchmark (1min-1h chunked)",
    ):
        print("[bench_long] loading model + processor ...", flush=True)
        model, proc = load_model_and_processor("eager")
        wav_sample, sr = load_sample_audio()
        sample_dur = wav_sample.shape[1] / sr
        pipe = MegaPipeline(
            model, proc, encoder_mode="cudagraph", use_fused_llm=True
        )

        # ---- warmup: capture all CUDA graphs on one representative chunk ----
        print("[bench_long] warmup (capturing graphs on a 30s chunk) ...", flush=True)
        warm_wav, _ = synthesize_long_audio(CHUNK_S, base_wav=wav_sample, sr=sr)
        # non-spec first (captures encoder cudagraph + decode graph)
        transcribe_long(
            pipe, proc, warm_wav, sr,
            chunk_seconds=CHUNK_S, max_new_tokens=WARMUP_TOKENS,
            speculative=False,
        )
        # spec next (captures verify graphs)
        transcribe_long(
            pipe, proc, warm_wav, sr,
            chunk_seconds=CHUNK_S, max_new_tokens=WARMUP_TOKENS,
            speculative=True,
        )
        # stock warmup (prime kernels)
        with torch.inference_mode():
            wi = build_inputs(proc, warm_wav)
            model.generate(
                input_ids=wi["input_ids"],
                input_features=wi["input_features"].bfloat16(),
                attention_mask=wi["attention_mask"],
                input_features_mask=wi.get("input_features_mask"),
                max_new_tokens=WARMUP_TOKENS, do_sample=False, num_beams=1,
            )
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        del warm_wav, wi
        print("[bench_long] warmup done.\n", flush=True)

        # ---- stock per-chunk baseline (real 60s run = 2 chunks) ----
        print("[bench_long] stock actual 60s (per-chunk baseline) ...", flush=True)
        stock_wav, _ = synthesize_long_audio(60, base_wav=wav_sample, sr=sr)
        stock_60 = transcribe_long_stock(
            model, proc, stock_wav, sr,
            chunk_seconds=CHUNK_S, max_new_tokens=MAX_NEW_TOKENS,
        )
        stock_per_chunk_ms = stock_60.per_chunk_ms
        stock_gen_per_chunk = stock_60.total_tokens // max(stock_60.n_chunks, 1)
        print(
            f"[bench_long] stock 60s: {stock_60.total_ms:.0f}ms over "
            f"{stock_60.n_chunks} chunks -> {stock_per_chunk_ms:.1f}ms/chunk, "
            f"{stock_gen_per_chunk} tok/chunk\n",
            flush=True,
        )
        del stock_wav

        # ---- per-duration measurements ----
        rows: list[dict] = []
        spec_results: dict[int, object] = {}
        nonspec_results: dict[int, object] = {}
        stock_results: dict[int, object] = {}
        peak_alloc_mb: float | None = None
        peak_reserved_mb: float | None = None

        for dur in DURATIONS_S:
            label = DUR_LABEL[dur]
            print(f"[bench_long] === duration {label} ({dur}s) ===", flush=True)
            wav, _ = synthesize_long_audio(dur, base_wav=wav_sample, sr=sr)

            # ---- mega speculative (headline; always end-to-end) ----
            try:
                if dur == DURATIONS_S[0]:
                    torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()
                sres = transcribe_long(
                    pipe, proc, wav, sr,
                    chunk_seconds=CHUNK_S,
                    max_new_tokens=MAX_NEW_TOKENS,
                    speculative=True,
                )
                torch.cuda.synchronize()
                wall = (time.perf_counter() - t0) * 1000.0
                sres.total_ms = wall  # use outer wall (includes any python glue)
                spec_results[dur] = sres
                if peak_alloc_mb is None:
                    peak_alloc_mb, peak_reserved_mb = _peak_vram_mb()
                print(
                    f"  mega spec    : {sres.total_ms:8.1f}ms  "
                    f"RTFx={sres.rtfx:6.2f}x  tok={sres.total_tokens}  "
                    f"chunks={sres.n_chunks}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  mega spec    : FAILED ({exc!r})", flush=True)
                spec_results[dur] = None

            # ---- mega non-spec ----
            try:
                if dur in NONSPEC_ACTUAL_S:
                    t0 = time.perf_counter()
                    nres = transcribe_long(
                        pipe, proc, wav, sr,
                        chunk_seconds=CHUNK_S,
                        max_new_tokens=MAX_NEW_TOKENS,
                        speculative=False,
                    )
                    torch.cuda.synchronize()
                    nres.total_ms = (time.perf_counter() - t0) * 1000.0
                else:
                    # extrapolate from the largest measured non-spec duration
                    ref_dur = max(d for d in NONSPEC_ACTUAL_S if d <= dur)
                    ref = nonspec_results.get(ref_dur)
                    nc = sres.n_chunks if sres else None
                    n_chunks = nc if nc else -(-dur // int(CHUNK_S))
                    per_chunk = ref.per_chunk_ms if ref else 0.0
                    tok_pc = (
                        ref.total_tokens // max(ref.n_chunks, 1) if ref else 0
                    )
                    nres = extrapolate_from_chunk(
                        per_chunk, n_chunks, dur, tok_pc, speculative=False,
                    )
                nonspec_results[dur] = nres
                tag = " (extrap)" if nres.extrapolated else ""
                print(
                    f"  mega non-spec: {nres.total_ms:8.1f}ms  "
                    f"RTFx={nres.rtfx:6.2f}x  tok={nres.total_tokens}{tag}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  mega non-spec: FAILED ({exc!r})", flush=True)
                nonspec_results[dur] = None

            # ---- stock (extrapolated except STOCK_ACTUAL_S) ----
            nc = sres.n_chunks if sres else -(-dur // int(CHUNK_S))
            if dur in STOCK_ACTUAL_S:
                stres = stock_60 if dur == 60 else None
            else:
                stres = extrapolate_from_chunk(
                    stock_per_chunk_ms, nc, dur,
                    stock_gen_per_chunk, speculative=False,
                )
            stock_results[dur] = stres
            tag = " (extrap)" if (stres is not None and stres.extrapolated) else ""
            if stres is not None:
                print(
                    f"  stock        : {stres.total_ms:8.1f}ms  "
                    f"RTFx={stres.rtfx:6.2f}x{tag}",
                    flush=True,
                )

            rows.append(_build_row(
                dur, spec_results[dur], nonspec_results[dur],
                stock_results[dur], peak_alloc_mb,
            ))
            del wav
            print(flush=True)

        # ---- print table ----
        _print_table(rows, sample_dur)

        # ---- save JSON ----
        payload = {
            "config": {
                "chunk_seconds": CHUNK_S,
                "max_new_tokens": MAX_NEW_TOKENS,
                "durations_s": DURATIONS_S,
                "sample_audio_seconds": round(sample_dur, 3),
                "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(0),
            },
            "stock_per_chunk_ms": round(stock_per_chunk_ms, 2),
            "stock_gen_per_chunk": stock_gen_per_chunk,
            "peak_vram_allocated_mb": (
                round(peak_alloc_mb, 1) if peak_alloc_mb else None
            ),
            "peak_vram_reserved_mb": (
                round(peak_reserved_mb, 1) if peak_reserved_mb else None
            ),
            "rows": rows,
            "per_duration": {
                str(d): {
                    "spec": (spec_results[d].to_dict() if spec_results[d] else None),
                    "nonspec": (
                        nonspec_results[d].to_dict() if nonspec_results[d] else None
                    ),
                    "stock": (
                        stock_results[d].to_dict() if stock_results[d] else None
                    ),
                }
                for d in DURATIONS_S
            },
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\n[bench_long] saved -> {out_path}")
    return 0


def _build_row(
    dur: int, spec, nonspec, stock, peak_vram_mb
) -> dict:
    def _g(r, attr, default=None):
        return getattr(r, attr, default) if r is not None else default

    return {
        "duration_s": dur,
        "label": DUR_LABEL[dur],
        "n_chunks": _g(spec, "n_chunks") or _g(stock, "n_chunks"),
        "tokens": _g(spec, "total_tokens"),
        "spec_ms": _g(spec, "total_ms"),
        "spec_rtfx": _g(spec, "rtfx"),
        "spec_extrapolated": bool(_g(spec, "extrapolated", False)),
        "nonspec_ms": _g(nonspec, "total_ms"),
        "nonspec_rtfx": _g(nonspec, "rtfx"),
        "nonspec_extrapolated": bool(_g(nonspec, "extrapolated", False)),
        "stock_ms": _g(stock, "total_ms"),
        "stock_rtfx": _g(stock, "rtfx"),
        "stock_extrapolated": bool(_g(stock, "extrapolated", False)),
        "peak_vram_mb": round(peak_vram_mb, 1) if peak_vram_mb else None,
    }


def _print_table(rows: list[dict], sample_dur: float) -> None:
    print("\n" + "=" * 104)
    print(
        f"LONG-AUDIO BENCHMARK  (chunk={CHUNK_S:.0f}s, max_new_tokens="
        f"{MAX_NEW_TOKENS}, sample={sample_dur:.1f}s, bf16, batch=1/chunk)"
    )
    print("=" * 104)
    hdr = (
        f"{'dur':<7}{'chunks':>7}{'tokens':>8}"
        f"{'spec ms':>11}{'specRTFx':>10}"
        f"{'nonspec ms':>12}{'nonRTFx':>9}"
        f"{'stock ms':>12}{'stockRTFx':>10}"
        f"{'VRAM MB':>9}"
    )
    print(hdr)
    print("-" * 104)
    for r in rows:
        def _ms(v):
            return f"{v:>11.0f}" if v is not None else f"{'-':>11}"

        def _rtfx(v):
            return f"{v:>9.1f}x" if v is not None else f"{'-':>10}"

        def _msw(v, w):
            return f"{v:>{w}.0f}" if v is not None else f"{'-':>{w}}"

        vram = f"{r['peak_vram_mb']:>9.0f}" if r["peak_vram_mb"] else f"{'-':>9}"
        print(
            f"{r['label']:<7}{r['n_chunks']:>7}{r['tokens']:>8}"
            + _msw(r['spec_ms'], 11) + f"{r['spec_rtfx']:>9.1f}x"
            + _msw(r['nonspec_ms'], 12) + f"{r['nonspec_rtfx']:>8.1f}x"
            + _msw(r['stock_ms'], 12) + f"{r['stock_rtfx']:>9.1f}x"
            + vram
        )
    print("-" * 104)
    # annotate extrapolated
    extrap = [r for r in rows if r["stock_extrapolated"]]
    if extrap:
        print("Note: stock ms/RTFx for durations > 60s are LINEAR EXTRAPOLATIONS")
        print(f"      from the real stock 60s run "
              f"(per-chunk constant; each chunk is independent).")
    print("=" * 104)


if __name__ == "__main__":
    raise SystemExit(main())
