"""Comparable long-audio benchmark: granite vs parakeet on the same audio.

Both models transcribe the SAME 10-minute audio clip using their best chunked +
batched path. Metrics are directly comparable:

  * RTFx (throughput): audio_seconds / wall_seconds. Higher is faster.
  * peak VRAM: torch.cuda.max_memory_allocated (GB).

Granite uses transcribe_long_batched (B=16, 30s chunks, BatchedPipeline).
Parakeet uses ChunkedTranscriber (B=32, 30s+2s overlap, frame-aligned stitch).

Both run on the same tiled audio so the workload is identical. The only
difference is the model architecture and chunking strategy.

Run:  cd /home/m0hawk/Documents/megapar && .venv/bin/python benchmarks/bench_long_audio_comparable.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tests" / "fixtures"))

from starling.parakeet.gpu_lock import with_gpu_lock  # noqa: E402

AUDIO_SECONDS = 600  # 10 minutes
SR = 16000
OUTPUTS = REPO / "outputs"


def _suppress():
    for mod in ("transformers",):
        try:
            warnings.filterwarnings("ignore", module=mod)
        except Exception:
            pass
    warnings.filterwarnings("ignore", message=".*max_length.*")
    warnings.filterwarnings("ignore", message=".*RNN module weights.*")


def tile_granite_audio(seconds: int) -> torch.Tensor:
    """Tile the granite sample audio to the target length."""
    from starling.granite.audio import load_sample_audio
    wav, sr = load_sample_audio()
    reps = max(1, (seconds * sr + wav.shape[1] - 1) // wav.shape[1])
    return wav.repeat(1, reps)[:, :seconds * sr].contiguous()


def tile_parakeet_audio(seconds: int) -> np.ndarray:
    """Tile the parakeet fixture to the target length."""
    import make_fixtures as mkfx
    base = mkfx.load_sample()
    reps = max(1, (seconds * SR + len(base) - 1) // len(base))
    return np.concatenate([base] * reps)[:seconds * SR].astype(np.float32)


def main() -> int:
    _suppress()
    results = {
        "audio_seconds": AUDIO_SECONDS,
        "device": torch.cuda.get_device_name(0),
        "models": {},
    }

    print(f"[bench] tiling {AUDIO_SECONDS}s ({AUDIO_SECONDS//60} min) audio ...")

    with with_gpu_lock(
        session="long-audio-compare", model="both",
        eta_min=10, note="comparable long-audio bench",
    ):
        # ---- granite ---- #
        print("\n========== granite-speech (B=16, 30s chunks) ==========")
        from starling.granite.audio import load_sample_audio as _lsa
        from starling.granite.batched import BatchedPipeline
        from starling.granite.long_audio import transcribe_long_batched
        from starling.granite.loader import load_model_and_processor

        gran_wav = tile_granite_audio(AUDIO_SECONDS)
        model, proc = load_model_and_processor(attn_impl="eager")
        pipe = BatchedPipeline(model, proc, max_batch_size=16, encoder_mode="cudagraph")

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        import time
        t0 = time.perf_counter()
        gran_res = transcribe_long_batched(pipe, proc, gran_wav, SR, chunk_seconds=30.0)
        torch.cuda.synchronize()
        gran_wall = time.perf_counter() - t0
        gran_vram = torch.cuda.max_memory_allocated() / 1e9

        gran_rtfx = AUDIO_SECONDS / gran_wall
        print(f"  wall={gran_wall:.1f}s  RTFx={gran_rtfx:.1f}x  "
              f"chunks={gran_res.n_chunks}  tokens={gran_res.total_tokens}  "
              f"VRAM={gran_vram:.2f}GB")
        results["models"]["granite_speech"] = {
            "mode": "batched B=16, 30s chunks",
            "wall_s": round(gran_wall, 2),
            "rtfx": round(gran_rtfx, 1),
            "n_chunks": gran_res.n_chunks,
            "total_tokens": gran_res.total_tokens,
            "vram_gb": round(gran_vram, 2),
        }
        del pipe, model
        torch.cuda.empty_cache()

        # ---- parakeet ---- #
        print("\n========== parakeet (B=32, 30s+2s overlap) ==========")
        from starling.parakeet.pipeline import MegaParakeetPipeline
        from starling.parakeet.chunking import ChunkedTranscriber

        para_audio = tile_parakeet_audio(AUDIO_SECONDS)
        para_pipe = MegaParakeetPipeline(dtype=torch.bfloat16)
        chunker = ChunkedTranscriber(para_pipe, chunk_seconds=30.0, overlap_seconds=2.0,
                                     chunk_batch_size=32)

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        para_text, para_summary = chunker.transcribe_with_timing(para_audio, sr=SR)
        torch.cuda.synchronize()
        para_wall = time.perf_counter() - t0
        para_vram = torch.cuda.max_memory_allocated() / 1e9

        para_rtfx = AUDIO_SECONDS / para_wall
        print(f"  wall={para_wall:.1f}s  RTFx={para_rtfx:.1f}x  "
              f"chunks={para_summary['n_chunks']}  "
              f"tokens={para_summary['n_tokens_surviving']}  "
              f"VRAM={para_vram:.2f}GB")
        results["models"]["parakeet"] = {
            "mode": "batched B=32, 30s+2s overlap chunks",
            "wall_s": round(para_wall, 2),
            "rtfx": round(para_rtfx, 1),
            "n_chunks": para_summary["n_chunks"],
            "total_tokens": para_summary["n_tokens_surviving"],
            "vram_gb": round(para_vram, 2),
        }
        del chunker, para_pipe
        torch.cuda.empty_cache()

    # ---- write ---- #
    OUTPUTS.mkdir(exist_ok=True)
    out_path = OUTPUTS / "long_audio_comparable.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[bench] wrote {out_path}")

    # ---- table ---- #
    from tabulate import tabulate
    rows = []
    for name, d in results["models"].items():
        rows.append([name, d["mode"], f"{d['wall_s']:.1f}s", f"{d['rtfx']:.0f}x",
                     d["n_chunks"], d["total_tokens"], f"{d['vram_gb']:.2f}"])
    print(f"\n=== Comparable long-audio ({AUDIO_SECONDS}s / {AUDIO_SECONDS//60} min) ===")
    print(tabulate(rows, headers=["model", "mode", "wall", "RTFx", "chunks",
                                  "tokens", "VRAM(GB)"],
                   tablefmt="github"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
