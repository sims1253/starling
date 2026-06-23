"""Benchmark: sequential vs batched long-audio transcription for granite-speech.

Compares :func:`starling.long_audio.transcribe_long` (sequential, B=1 per chunk)
against :func:`starling.long_audio.transcribe_long_batched` (B chunks decoded in
lock-step via :class:`starling.batched.BatchedPipeline`) across audio lengths
(1 min, 5 min, 10 min) and batch sizes (4, 8).

The headline metric is aggregate RTFx (``audio_seconds / wall_seconds``) --
higher is faster.  Batched decode should dominate because it turns the
launch-bound batch=1 GEMVs into saturating B-wide GEMMs, reading the 4.4 GB of
LLM weights once for B tokens instead of once per token.

Writes ``outputs/long_audio_batched_bench.json`` and prints tables.

Run:  uv run python benchmarks/bench_long_audio_batched.py
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import torch
from tabulate import tabulate

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from starling.batched import BatchedPipeline  # noqa: E402
from starling.config import DEFAULT_TASK_PROMPT  # noqa: E402
from starling.loader import load_model_and_processor  # noqa: E402
from starling.long_audio import (  # noqa: E402
    synthesize_long_audio,
    transcribe_long,
    transcribe_long_batched,
)
from starling.parakeet.gpu_lock import with_gpu_lock  # noqa: E402
from starling.pipeline import MegaPipeline  # noqa: E402

AUDIO_LENGTHS = [60, 300, 600]  # 1 min, 5 min, 10 min
BATCH_SIZES = [4, 8]
CHUNK_SECONDS = 30.0
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

    all_results = []

    print("\n[bench] acquiring GPU lock ...")
    with with_gpu_lock(
        session="granite-long-batched", model="granite-speech-4.1-2b",
        eta_min=10, note="long-audio batched bench",
    ):
        free, total = torch.cuda.mem_get_info()
        print(f"[bench] lock held; GPU free={free/1e9:.1f}GB / {total/1e9:.1f}GB")

        # --- sequential baseline (B=1 MegaPipeline) ----------------------- #
        print("[bench] building sequential MegaPipeline (B=1) ...")
        seq_pipe = MegaPipeline(model, proc, encoder_mode="cudagraph")

        for seconds in AUDIO_LENGTHS:
            wav, sr = synthesize_long_audio(seconds)
            print(f"\n[bench] === sequential: {seconds}s audio ({seconds//CHUNK_SECONDS:.0f} chunks) ===")
            res = transcribe_long(
                seq_pipe, proc, wav, sr,
                chunk_seconds=CHUNK_SECONDS,
                max_new_tokens=MAX_NEW_TOKENS,
                speculative=False,
            )
            print(f"  wall={res.total_ms:.0f}ms  RTFx={res.rtfx:.1f}x  "
                  f"tok={res.total_tokens}  chunks={res.n_chunks}")
            all_results.append({
                "mode": "sequential",
                "batch_size": 1,
                "audio_seconds": seconds,
                "total_ms": round(res.total_ms, 1),
                "rtfx": round(res.rtfx, 1),
                "total_tokens": res.total_tokens,
                "n_chunks": res.n_chunks,
                "per_chunk_ms": round(res.per_chunk_ms, 1),
            })

        # --- batched (B>1 BatchedPipeline) --------------------------------- #
        for B in BATCH_SIZES:
            print(f"\n[bench] building BatchedPipeline (B={B}) ...")
            batched_pipe = BatchedPipeline(
                model, proc, max_batch_size=B, encoder_mode="cudagraph",
            )
            for seconds in AUDIO_LENGTHS:
                wav, sr = synthesize_long_audio(seconds)
                print(f"\n[bench] === batched B={B}: {seconds}s audio "
                      f"({seconds//CHUNK_SECONDS:.0f} chunks, "
                      f"{(seconds//CHUNK_SECONDS + B - 1)//B} batches) ===")
                res = transcribe_long_batched(
                    batched_pipe, proc, wav, sr,
                    chunk_seconds=CHUNK_SECONDS,
                    max_new_tokens=MAX_NEW_TOKENS,
                )
                print(f"  wall={res.total_ms:.0f}ms  RTFx={res.rtfx:.1f}x  "
                      f"tok={res.total_tokens}  chunks={res.n_chunks}")
                all_results.append({
                    "mode": "batched",
                    "batch_size": B,
                    "audio_seconds": seconds,
                    "total_ms": round(res.total_ms, 1),
                    "rtfx": round(res.rtfx, 1),
                    "total_tokens": res.total_tokens,
                    "n_chunks": res.n_chunks,
                    "per_chunk_ms": round(res.per_chunk_ms, 1),
                })
            del batched_pipe
            torch.cuda.empty_cache()

    # ---- write JSON ---- #
    payload = {
        "model": "granite-speech-4.1-2b",
        "dtype": "bfloat16",
        "device": torch.cuda.get_device_name(0),
        "chunk_seconds": CHUNK_SECONDS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "results": all_results,
    }
    out_path = OUTPUTS / "long_audio_batched_bench.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\n[bench] wrote {out_path}")

    # ---- print table ---- #
    print("\n=== granite-speech long-audio: sequential vs batched ===")
    rows = []
    for r in all_results:
        rows.append([
            f"{r['audio_seconds']}s",
            r["mode"],
            r["batch_size"],
            f"{r['total_ms']:.0f}",
            f"{r['rtfx']:.1f}",
            r["n_chunks"],
            f"{r['per_chunk_ms']:.0f}",
        ])
    print(tabulate(
        rows,
        headers=["audio", "mode", "B", "wall_ms", "RTFx", "chunks", "ms/chunk"],
        tablefmt="github",
    ))

    # headline: best batched vs sequential at 5 min
    seq_5m = next(r for r in all_results
                  if r["mode"] == "sequential" and r["audio_seconds"] == 300)
    best_batched_5m = min(
        (r for r in all_results
         if r["mode"] == "batched" and r["audio_seconds"] == 300),
        key=lambda r: r["total_ms"],
    )
    speedup = seq_5m["total_ms"] / best_batched_5m["total_ms"]
    print(f"\n*** 5min: sequential {seq_5m['rtfx']:.1f}x RTFx -> "
          f"batched B={best_batched_5m['batch_size']} "
          f"{best_batched_5m['rtfx']:.1f}x RTFx "
          f"({speedup:.2f}x faster) ***")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
