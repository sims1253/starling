"""RTF + memory benchmark for the BATCHED chunked (long-audio) parakeet path.

Sister of ``bench_chunked.py``: identical workload (5/15/30/60 min synthetic
audio tiled from the canonical fixture sample) but through
``ChunkedTranscriber(chunk_batch_size=8)``, which groups chunks into mini-batches
of 8 and runs each through one set of batched mel+encoder+decode forwards.

Runs under the benchmark GPU lock with the adaptive memory guard
(ChunkedTranscriber shrinks B from live free VRAM and aborts only on the hard
floor). The HEADLINE result this prints is:

    **batched chunked RTF at 1 h** (+ peak VRAM, + num batches), compared to the
    sequential chunked numbers in ``chunked_bench.json`` (1 h: 12.3 s / 293x /
    1.65 GB). Batching 8 chunks at a time collapses ~121 sequential B=1
    iterations to ~16 B=8 iterations, recovering most of the megakernel
    pipeline's batched throughput (3109x RTF @ B8 medium).

Writes ``outputs/parakeet/batched_chunked_bench.json`` and prints tables.

Usage:  ``uv run python benchmarks/parakeet/bench_batched_chunked.py``
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "tests" / "fixtures"))

from starling.parakeet.chunking import ChunkedTranscriber  # noqa: E402
from starling.parakeet.gpu_lock import with_gpu_lock  # noqa: E402
from starling.parakeet.pipeline import MegaParakeetPipeline  # noqa: E402
import make_fixtures as mkfx  # noqa: E402

OUT_DIR = _REPO_ROOT / "outputs" / "parakeet"
SEQ_PATH = OUT_DIR / "chunked_bench.json"          # sequential chunked comparison
OUT_PATH = OUT_DIR / "batched_chunked_bench.json"  # this bench's output

CHUNK_BATCH_SIZE = 8
LENGTHS_MIN = [5, 15, 30, 60]   # 5 min, 15 min, 30 min, 1 h
SR = 16000


def gpu_util_pct() -> float:
    """Best-effort nvidia-smi GPU utilization read (0.0 on failure)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, text=True, timeout=10,
        ).strip()
        return float(out.splitlines()[0])
    except Exception:
        return 0.0


def tile_audio(base: np.ndarray, target_seconds: float) -> np.ndarray:
    need = int(target_seconds * SR)
    reps = (need + base.shape[0] - 1) // base.shape[0]
    return np.ascontiguousarray(np.tile(base, reps), dtype=np.float32)


def load_sequential_comparison() -> dict:
    """Pull the sequential (chunk_batch_size=1) numbers from chunked_bench.json."""
    if not SEQ_PATH.exists():
        return {}
    try:
        data = json.loads(SEQ_PATH.read_text())
    except Exception:
        return {}
    out = {}
    for e in data.get("results", []):
        if e.get("status") == "ok":
            out[e["length_min"]] = {
                "total_ms": e.get("total_ms"),
                "rtf": e.get("rtf"),
                "peak_vram_gb": e.get("peak_vram_gb"),
                "n_chunks": e.get("n_chunks"),
                "mean_chunk_ms": e.get("mean_chunk_ms"),
            }
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Defer if the GPU is busy: a contended card corrupts timings.
    util = gpu_util_pct()
    if util > 30.0:
        print(f"[bench] GPU util {util:.0f}% > 30%; deferring 30s...", flush=True)
        time.sleep(30)
        util = gpu_util_pct()
        if util > 30.0:
            print(f"[bench] GPU still busy ({util:.0f}%); proceeding under lock anyway.",
                  flush=True)

    print(f"[bench] loading pipeline + batched chunker (chunk_batch_size="
          f"{CHUNK_BATCH_SIZE}) ...", flush=True)
    pipe = MegaParakeetPipeline(use_graphed_encoder=True)
    chunker = ChunkedTranscriber(
        pipe, chunk_seconds=30.0, overlap_seconds=2.0,
        chunk_batch_size=CHUNK_BATCH_SIZE,
    )

    base = mkfx.load_sample()

    # Warmup: capture the shapes the timed runs will hit so capture cost is
    # amortised OUT of the timed region.
    #   * 32 s clip -> 1 chunk -> B=1 graph (the single-chunk / final-partial shape)
    #   * 250 s clip -> 9 chunks -> first batch is a full B=8 -> (8, T_enc_32s)
    #     graph, which every timed run's full batches reuse (dict hit). The 250 s
    #     run also captures its own remainder (B=1) batch.
    print("[bench] warmup (capture B=1 + B=8 graphs) ...", flush=True)
    _ = chunker.transcribe(tile_audio(base, 32.0))            # B=1 full-chunk graph
    _ = chunker.transcribe(tile_audio(base, 250.0))           # forces the B=8 graph

    comparison = load_sequential_comparison()

    results = []
    for minutes in LENGTHS_MIN:
        label = f"{minutes}min"
        target_s = minutes * 60
        audio = tile_audio(base, target_s)
        try:
            text, summary = chunker.transcribe_with_timing(audio)
        except MemoryError as e:
            print(f"[bench] {label}: memory guard tripped -> {e}", flush=True)
            results.append({
                "length_min": label, "target_s": target_s,
                "status": "aborted_low_vram", "error": str(e),
            })
            continue

        batch_sizes = [b["batch_size"] for b in summary["per_batch"]]
        batch_totals = [b["total_ms"] for b in summary["per_batch"]]
        total_ms = summary["total_ms"]
        audio_s = summary["audio_seconds"]
        rtf = audio_s / (total_ms / 1000.0) if total_ms > 0 else 0.0
        entry = {
            "length_min": label,
            "target_s": target_s,
            "audio_seconds": audio_s,
            "status": "ok",
            "total_ms": total_ms,
            "rtf": rtf,
            "peak_vram_gb": summary["peak_vram_gb"],
            "n_chunks": summary["n_chunks"],
            "n_batches": summary["n_batches"],
            "chunk_batch_size": summary["chunk_batch_size"],
            "batch_sizes": batch_sizes,
            "mean_batch_ms": float(np.mean(batch_totals)) if batch_totals else 0.0,
            "max_batch_ms": float(np.max(batch_totals)) if batch_totals else 0.0,
            "min_batch_ms": float(np.min(batch_totals)) if batch_totals else 0.0,
            "max_batch_size": int(max(batch_sizes)) if batch_sizes else 0,
            "n_tokens": summary["n_tokens_surviving"],
            "n_stitches": summary["n_stitches"],
            "chunk_seconds": summary["chunk_seconds"],
            "overlap_seconds": summary["overlap_seconds"],
            "text_preview": text[:200],
            "text_len": len(text),
        }
        results.append(entry)
        print(
            f"[bench] {label:5s}: total={total_ms:8.1f}ms  RTF={rtf:7.1f}x  "
            f"peakVRAM={summary['peak_vram_gb']:.3f}GB  "
            f"batches={summary['n_batches']:3d} (sizes {batch_sizes})  "
            f"mean_batch={entry['mean_batch_ms']:6.1f}ms  "
            f"tokens={summary['n_tokens_surviving']}  stitches={summary['n_stitches']}",
            flush=True,
        )
        torch.cuda.empty_cache()

    payload = {
        "model_id": "nvidia/parakeet-tdt-0.6b-v3",
        "dtype": "bfloat16",
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "?",
        "method": (
            "ChunkedTranscriber (chunk_seconds=30, overlap_seconds=2, "
            "chunk_batch_size=8); chunks grouped into mini-batches of up to 8, "
            "each run through one batched mel+graphed encoder+graphed TDT "
            "decode; left-biased frame-aligned stitching; adaptive batch-size "
            "guard shrinks B from free VRAM; empty_cache between batches; "
            "peak_vram = torch.cuda.max_memory_allocated reset per config; GPU "
            "lock held, deferred if nvidia-smi util>30%"
        ),
        "chunk_batch_size": CHUNK_BATCH_SIZE,
        "chunk_geometry": {
            "chunk_seconds": 30.0,
            "overlap_seconds": 2.0,
            "window_seconds": 32.0,
            "step_seconds": 30.0,
            "samples_per_enc_frame": chunker.samples_per_enc_frame,
        },
        "sequential_comparison_b1": comparison,
        "results": results,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"\n[bench] wrote {OUT_PATH}")

    # ---- batched vs sequential comparison table ----
    print("\n=== Batched (chunk_batch_size=8) vs sequential (chunk_batch_size=1) ===")
    print(f"{'length':>7} | {'batched total':>13} | {'batched RTF':>11} | "
          f"{'seq total':>10} | {'seq RTF':>8} | {'speedup':>7} | "
          f"{'batched VRAM':>12} | {'seq VRAM':>9} | {'batches':>7}")
    print("-" * 110)
    for e in results:
        if e.get("status") != "ok":
            print(f"{e['length_min']:>7} | {'ABORTED':>13} |")
            continue
        seq = comparison.get(e["length_min"], {})
        seq_ms = seq.get("total_ms")
        seq_rtf = seq.get("rtf")
        seq_vram = seq.get("peak_vram_gb")
        speedup = (seq_ms / e["total_ms"]) if seq_ms and e["total_ms"] else None
        seq_ms_s = f"{seq_ms:.1f}ms" if seq_ms is not None else "-"
        seq_rtf_s = f"{seq_rtf:.1f}x" if seq_rtf else "-"
        speedup_s = f"{speedup:.2f}x" if speedup else "-"
        seq_vram_s = f"{seq_vram:.3f}GB" if seq_vram else "-"
        print(
            f"{e['length_min']:>7} | {e['total_ms']:>10.1f}ms | "
            f"{e['rtf']:>9.1f}x | {seq_ms_s:>9} | {seq_rtf_s:>8} | "
            f"{speedup_s:>7} | {e['peak_vram_gb']:>10.3f}GB | "
            f"{seq_vram_s:>9} | {e['n_batches']:>4} (maxB={e['max_batch_size']})"
        )

    # ---- VRAM-bounded confirmation ----
    ok = [e for e in results if e.get("status") == "ok"]
    if ok:
        peaks = [e["peak_vram_gb"] for e in ok]
        print("\n=== VRAM-bounded confirmation (peak should be ~flat across lengths) ===")
        for e in ok:
            print(f"  {e['length_min']:>5}: peak VRAM = {e['peak_vram_gb']:.3f} GB  "
                  f"(batches={e['n_batches']}, max_batch_size={e['max_batch_size']})")
        print(f"  range: {min(peaks):.3f} - {max(peaks):.3f} GB  "
              f"(spread {max(peaks) - min(peaks):.3f} GB)")

        # HEADLINE
        h1h = next((e for e in ok if e["length_min"] == "60min"), None)
        seq1h = comparison.get("60min", {})
        if h1h:
            print("\n=== HEADLINE: batched chunked @ 1 h ===")
            sp = (seq1h.get("total_ms", 0) / h1h["total_ms"]
                  if seq1h.get("total_ms") else None)
            print(
                f"  1h batched: total={h1h['total_ms']:.1f}ms  "
                f"RTF={h1h['rtf']:.1f}x  peak VRAM={h1h['peak_vram_gb']:.3f}GB  "
                f"batches={h1h['n_batches']} (max batch size {h1h['max_batch_size']})"
            )
            if seq1h:
                print(
                    f"  1h sequential: total={seq1h.get('total_ms'):.1f}ms  "
                    f"RTF={seq1h.get('rtf'):.1f}x  "
                    f"peak VRAM={seq1h.get('peak_vram_gb'):.3f}GB  "
                    f"chunks={seq1h.get('n_chunks')}"
                )
            if sp:
                print(f"  speedup over sequential: {sp:.2f}x "
                      f"(RTF {h1h['rtf']:.1f}x vs {seq1h.get('rtf', 0):.1f}x)")

    return 0


if __name__ == "__main__":
    with with_gpu_lock(
        session="parakeet",
        model="parakeet-tdt-0.6b-v3",
        eta_min=5,
        note="batched chunked bench",
    ):
        sys.exit(main())
