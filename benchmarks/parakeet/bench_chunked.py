"""Memory + RTF benchmark for the chunked (long-audio) parakeet transcriber.

Runs under the shared GPU lock (comms.md P1) with a hard memory-safety guard
(abort a config if free VRAM < ``MIN_FREE_VRAM_GB``). The HEADLINE result this
prints is: **chunked RTF at 1 h + peak VRAM at 1 h**, proving VRAM is bounded by
chunk size and not by total length (vs the unchunked ``vram_cliff`` at 7 min in
``robust_bench.json``).

Configs: 5 min, 15 min, 30 min, 1 h (synthetic audio tiled from the canonical
fixture sample). For each: total_ms, mean per-chunk ms, RTF, peak VRAM, num
chunks, num tokens, stitches (overlap tokens dropped). Loads the unchunked
1/3/5 min B1 numbers from ``robust_bench.json`` for the comparison table.

Writes ``outputs/parakeet/chunked_bench.json`` and prints tables.

Usage:  ``uv run python benchmarks/parakeet/bench_chunked.py``
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
ROBUST_PATH = OUT_DIR / "robust_bench.json"
OUT_PATH = OUT_DIR / "chunked_bench.json"

# Memory-safety guard (comms.md / task): 32 GB card is shared; cap our own use
# at ~8 GB and abort a config if free VRAM drops below this.
MIN_FREE_VRAM_GB = 24.0

LENGTHS_MIN = [5, 15, 30, 60]   # 5 min, 15 min, 30 min, 1 h
SR = 16000


def gpu_util_pct() -> float:
    """Best-effort nvidia-smi GPU utilization read (0.0 on failure)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, text=True, timeout=10,
        ).strip()
        return float(out.splitlines()[0])
    except Exception:
        return 0.0


def tile_audio(base: np.ndarray, target_seconds: float) -> np.ndarray:
    need = int(target_seconds * SR)
    reps = (need + base.shape[0] - 1) // base.shape[0]
    return np.ascontiguousarray(np.tile(base, reps), dtype=np.float32)


def load_unchunked_comparison() -> dict:
    """Pull unchunked B1 1/3/5 min numbers from robust_bench.json for the table."""
    if not ROBUST_PATH.exists():
        return {}
    try:
        data = json.loads(ROBUST_PATH.read_text())
    except Exception:
        return {}
    sweep = data.get("results", {}).get("length_sweep", [])
    out = {}
    for e in sweep:
        if e.get("batch_size") == 1 and e.get("status") == "ok" and "total_ms" in e:
            key = e.get("length_min")
            out[key] = {
                "total_ms": e.get("total_ms"),
                "rtf": e.get("rtf"),
                "peak_vram_gb": e.get("peak_vram_gb"),
                "audio_seconds": e.get("audio_seconds"),
            }
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Defer if the GPU is busy (comms.md): a contended card corrupts timings.
    util = gpu_util_pct()
    if util > 30.0:
        print(f"[bench] GPU util {util:.0f}% > 30%; deferring 30s...", flush=True)
        time.sleep(30)
        util = gpu_util_pct()
        if util > 30.0:
            print(f"[bench] GPU still busy ({util:.0f}%); proceeding under lock anyway.",
                  flush=True)

    print("[bench] loading pipeline + chunker ...", flush=True)
    pipe = MegaParakeetPipeline(use_graphed_encoder=True)
    chunker = ChunkedTranscriber(
        pipe, chunk_seconds=30.0, overlap_seconds=2.0, min_free_vram_gb=MIN_FREE_VRAM_GB,
    )

    base = mkfx.load_sample()

    # Global warmup: one ~32 s clip captures the full-chunk encoder+decoder
    # graphs so the timed runs are capture-free for full chunks (only the final
    # partial chunk of each length pays a one-off capture).
    print("[bench] warmup (capture graphs) ...", flush=True)
    _ = chunker.transcribe(tile_audio(base, 32.0))

    comparison = load_unchunked_comparison()

    results = []
    for minutes in LENGTHS_MIN:
        label = f"{minutes}min"
        target_s = minutes * 60
        audio = tile_audio(base, target_s)
        free_gb = chunker._free_vram_gb()
        if free_gb < MIN_FREE_VRAM_GB:
            print(f"[bench] {label}: free VRAM {free_gb:.2f} GB < guard; ABORTING config",
                  flush=True)
            results.append({
                "length_min": label, "target_s": target_s, "status": "aborted_low_vram",
                "free_vram_gb": free_gb,
            })
            continue
        try:
            text, summary = chunker.transcribe_with_timing(audio)
        except MemoryError as e:
            print(f"[bench] {label}: memory guard tripped -> {e}", flush=True)
            results.append({
                "length_min": label, "target_s": target_s, "status": "aborted_low_vram",
                "error": str(e),
            })
            continue

        chunk_ms = [c["total_ms"] for c in summary["per_chunk"]]
        mean_chunk_ms = float(np.mean(chunk_ms)) if chunk_ms else 0.0
        total_ms = summary["total_ms"]
        audio_s = summary["audio_seconds"]
        rtf = audio_s / (total_ms / 1000.0) if total_ms > 0 else 0.0
        entry = {
            "length_min": label,
            "target_s": target_s,
            "audio_seconds": audio_s,
            "status": "ok",
            "total_ms": total_ms,
            "mean_chunk_ms": mean_chunk_ms,
            "min_chunk_ms": float(np.min(chunk_ms)) if chunk_ms else 0.0,
            "max_chunk_ms": float(np.max(chunk_ms)) if chunk_ms else 0.0,
            "rtf": rtf,
            "peak_vram_gb": summary["peak_vram_gb"],
            "n_chunks": summary["n_chunks"],
            "n_tokens": summary["n_tokens_surviving"],
            "n_stitches": summary["n_stitches"],
            "chunk_seconds": summary["chunk_seconds"],
            "overlap_seconds": summary["overlap_seconds"],
            "overlap_waste_fraction": (
                summary["overlap_seconds"] / summary["chunk_seconds"]
            ),
            "text_preview": text[:200],
            "text_len": len(text),
        }
        results.append(entry)
        print(
            f"[bench] {label:5s}: total={total_ms:8.1f}ms  RTF={rtf:7.1f}x  "
            f"peakVRAM={summary['peak_vram_gb']:.3f}GB  chunks={summary['n_chunks']:3d}  "
            f"mean_chunk={mean_chunk_ms:6.1f}ms  tokens={summary['n_tokens_surviving']}  "
            f"stitches={summary['n_stitches']}",
            flush=True,
        )
        torch.cuda.empty_cache()

    payload = {
        "model_id": "nvidia/parakeet-tdt-0.6b-v3",
        "dtype": "bfloat16",
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "?",
        "method": (
            "ChunkedTranscriber (chunk_seconds=30, overlap_seconds=2); per-chunk "
            "B=1 mel+graphed encoder+graphed TDT decode; left-biased frame-aligned "
            "stitching; empty_cache between chunks; peak_vram = "
            "torch.cuda.max_memory_allocated reset per config; GPU lock held, "
            "deferred if nvidia-smi util>30%; memory guard aborts if free<24GB"
        ),
        "min_free_vram_gb_guard": MIN_FREE_VRAM_GB,
        "chunk_geometry": {
            "chunk_seconds": 30.0,
            "overlap_seconds": 2.0,
            "window_seconds": 32.0,
            "step_seconds": 30.0,
            "samples_per_enc_frame": chunker.samples_per_enc_frame,
        },
        "unchunked_comparison_b1": comparison,
        "results": results,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"\n[bench] wrote {OUT_PATH}")

    # ---- comparison table ----
    print("\n=== Chunked vs unchunked (batch=1) ===")
    print(f"{'length':>7} | {'chunked total':>13} | {'chunked RTF':>11} | "
          f"{'chunked peakVRAM':>16} | {'unchunked RTF':>13} | {'unchunked VRAM':>13} | {'chunks':>6}")
    print("-" * 100)
    for e in results:
        if e.get("status") != "ok":
            print(f"{e['length_min']:>7} | {'ABORTED':>13} |")
            continue
        comp = comparison.get(e["length_min"], {})
        print(
            f"{e['length_min']:>7} | {e['total_ms']:>10.1f}ms | "
            f"{e['rtf']:>9.1f}x | {e['peak_vram_gb']:>13.3f}GB | "
            f"{(comp.get('rtf') or '-'):>13} | "
            f"{(comp.get('peak_vram_gb') or '-'):>13} | {e['n_chunks']:>6}"
        )

    # ---- VRAM-bounded confirmation ----
    ok = [e for e in results if e.get("status") == "ok"]
    if ok:
        peaks = [e["peak_vram_gb"] for e in ok]
        print("\n=== VRAM-bounded confirmation (peak should be ~flat across lengths) ===")
        for e in ok:
            print(f"  {e['length_min']:>5}: peak VRAM = {e['peak_vram_gb']:.3f} GB  "
                  f"(n_chunks={e['n_chunks']})")
        print(f"  range: {min(peaks):.3f} - {max(peaks):.3f} GB  "
              f"(spread {max(peaks)-min(peaks):.3f} GB)")
        # HEADLINE
        h1h = next((e for e in ok if e["length_min"] == "60min"), None)
        if h1h:
            print("\n=== HEADLINE ===")
            print(f"  1h chunked: total={h1h['total_ms']:.1f}ms  "
                  f"RTF={h1h['rtf']:.1f}x  peak VRAM={h1h['peak_vram_gb']:.3f}GB  "
                  f"chunks={h1h['n_chunks']}")

    return 0


if __name__ == "__main__":
    with with_gpu_lock(
        session="parakeet-mega",
        model="parakeet-tdt-0.6b-v3",
        eta_min=10,
        note="chunked bench",
    ):
        sys.exit(main())
