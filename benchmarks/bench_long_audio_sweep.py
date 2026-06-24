"""Final optimization sweep: batch size + chunk size for granite long-audio.

Sweeps the two main levers for batched long-audio throughput:

  * Batch size ``B`` in {4, 8, 16} -- higher B saturates the tensor cores
    better (reading LLM weights once for B tokens), at the cost of KV-cache
    VRAM (B * 40 layers * 4 KV heads * 640 * 128 * 2 * 2 bytes).
  * Chunk seconds in {15, 30} -- smaller chunks produce more chunks to batch
    (better GPU utilization) but more chat-template overhead per second of
    audio. 30s is the existing default.

Compares every (B, chunk_s) config against the sequential B=1 baseline at
matching chunk_s, at 300s (5 min) and 600s (10 min) audio lengths.

Writes ``outputs/long_audio_sweep.json`` and prints tables.

Run:  uv run python benchmarks/bench_long_audio_sweep.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import torch
from tabulate import tabulate

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from starling.batched import BatchedPipeline  # noqa: E402
from starling.loader import load_model_and_processor  # noqa: E402
from starling.long_audio import (  # noqa: E402
    synthesize_long_audio,
    transcribe_long,
    transcribe_long_batched,
)
from starling.parakeet.gpu_lock import with_gpu_lock  # noqa: E402
from starling.pipeline import MegaPipeline  # noqa: E402

AUDIO_LENGTHS = [300, 600]
CHUNK_SECONDS_SET = [15.0, 30.0]
BATCH_SIZES = [4, 8, 16]
MAX_NEW_TOKENS = 200


def _suppress() -> None:
    for mod in ("transformers",):
        try:
            warnings.filterwarnings("ignore", module=mod)
        except Exception:
            pass
    warnings.filterwarnings("ignore", message=".*max_length.*")
    warnings.filterwarnings("ignore", message=".*RNN module weights.*")


def main() -> int:
    _suppress()

    print("[bench] loading model ...")
    model, proc = load_model_and_processor(attn_impl="eager")

    OUTPUTS = _REPO_ROOT / "outputs"
    OUTPUTS.mkdir(exist_ok=True)
    results = []

    print("\n[bench] acquiring GPU lock ...")
    with with_gpu_lock(
        session="granite-sweep", model="granite-speech-4.1-2b",
        eta_min=15, note="long-audio optimization sweep",
    ):
        free, total = torch.cuda.mem_get_info()
        print(f"[bench] lock held; GPU free={free/1e9:.1f}GB / {total/1e9:.1f}GB")

        # --- sequential baselines (fresh pipeline per chunk_seconds) ------ #
        for chunk_s in CHUNK_SECONDS_SET:
            seq_pipe = MegaPipeline(model, proc, encoder_mode="cudagraph")
            for seconds in AUDIO_LENGTHS:
                wav, sr = synthesize_long_audio(seconds)
                res = transcribe_long(
                    seq_pipe, proc, wav, sr,
                    chunk_seconds=chunk_s,
                    max_new_tokens=MAX_NEW_TOKENS,
                    speculative=False,
                )
                print(f"[seq]     {seconds}s chunk={chunk_s:.0f}s  "
                      f"RTFx={res.rtfx:.1f}x  wall={res.total_ms:.0f}ms")
                results.append({
                    "mode": "sequential", "batch_size": 1,
                    "chunk_seconds": chunk_s, "audio_seconds": seconds,
                    "total_ms": round(res.total_ms, 1),
                    "rtfx": round(res.rtfx, 1),
                    "total_tokens": res.total_tokens,
                    "n_chunks": res.n_chunks,
                })
            del seq_pipe
            torch.cuda.empty_cache()

        # --- batched sweep ------------------------------------------------- #
        for chunk_s in CHUNK_SECONDS_SET:
            for B in BATCH_SIZES:
                print(f"\n[bench] B={B}, chunk={chunk_s:.0f}s ...")
                batched_pipe = BatchedPipeline(
                    model, proc, max_batch_size=B, encoder_mode="cudagraph",
                )
                for seconds in AUDIO_LENGTHS:
                    wav, sr = synthesize_long_audio(seconds)
                    res = transcribe_long_batched(
                        batched_pipe, proc, wav, sr,
                        chunk_seconds=chunk_s,
                        max_new_tokens=MAX_NEW_TOKENS,
                    )
                    torch.cuda.synchronize()
                    print(f"[bat B={B:<2}] {seconds}s chunk={chunk_s:.0f}s  "
                          f"RTFx={res.rtfx:.1f}x  wall={res.total_ms:.0f}ms  "
                          f"tok={res.total_tokens}")
                    results.append({
                        "mode": "batched", "batch_size": B,
                        "chunk_seconds": chunk_s, "audio_seconds": seconds,
                        "total_ms": round(res.total_ms, 1),
                        "rtfx": round(res.rtfx, 1),
                        "total_tokens": res.total_tokens,
                        "n_chunks": res.n_chunks,
                    })
                del batched_pipe
                torch.cuda.empty_cache()

    payload = {
        "model": "granite-speech-4.1-2b",
        "dtype": "bfloat16",
        "device": torch.cuda.get_device_name(0),
        "max_new_tokens": MAX_NEW_TOKENS,
        "results": results,
    }
    out_path = OUTPUTS / "long_audio_sweep.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\n[bench] wrote {out_path}")

    # ---- per-audio-length tables ---- #
    for seconds in AUDIO_LENGTHS:
        subset = [r for r in results if r["audio_seconds"] == seconds]
        print(f"\n=== {seconds}s ({seconds//60}min) audio ===")
        rows = []
        for r in subset:
            rows.append([
                r["mode"], r["batch_size"], f"{r['chunk_seconds']:.0f}s",
                f"{r['total_ms']:.0f}", f"{r['rtfx']:.1f}",
                r["n_chunks"],
            ])
        print(tabulate(
            rows,
            headers=["mode", "B", "chunk", "wall_ms", "RTFx", "n_chunks"],
            tablefmt="github",
        ))
        best = max(subset, key=lambda r: r["rtfx"])
        seq = max(
            (r for r in subset if r["mode"] == "sequential"), key=lambda r: r["rtfx"]
        )
        speedup = best["rtfx"] / seq["rtfx"]
        print(f"  best: {best['mode']} B={best['batch_size']} "
              f"chunk={best['chunk_seconds']:.0f}s -> {best['rtfx']:.1f}x RTFx "
              f"({speedup:.2f}x vs sequential)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
