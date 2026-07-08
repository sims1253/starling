"""Open-ASR-Leaderboard WER: whole-shot vs chunked(12/3) MOSS transcription.

The 7 leaderboard English datasets here are all long-form (every clip > 12s,
mean ~24s), and the MOSS leaderboard engine transcribes each clip whole-shot.
MOSS is short-form-trained, so a single long transcribe degrades; finalizing the
clip in fixed 12s / 3s-overlap windows (the production /stream path) keeps each
transcribe in-distribution and stitches the pieces.

This reproduces the leaderboard's accuracy methodology (Whisper normalization +
kaldialign WER; composite = unweighted mean of per-dataset WER) on the cached
``__n50`` shards, scoring both transcription modes side by side + RTFx.

Run:  ``uv run python -m benchmarks.moss.bench_leaderboard_chunked``
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "benchmarks"))
sys.path.insert(0, str(REPO / "tests" / "fixtures"))

CHUNK_S, OVERLAP_S, N = 12.0, 3.0, 50
import os
DO_CHUNKED = os.environ.get("CHUNKED", "1") == "1"
ENCODER_MODE = os.environ.get("ENCODER_MODE", "cudagraph")


def main():
    import leaderboard_corpus as lc
    from wer_leaderboard import wer_pct
    from starling.flags import OptFlags, set_default_flags
    from starling.moss.loader import load_model_and_processor
    from starling.moss.pipeline import MossMegaPipeline
    from starling.stream_chunk import ChunkStreamer

    set_default_flags(OptFlags())  # bf16 decode
    print(f"[bench] loading MOSS (encoder={ENCODER_MODE}, chunked={DO_CHUNKED}) ...")
    model, proc = load_model_and_processor()
    pipe = MossMegaPipeline(model, proc, max_cache_len=2048, encoder_mode=ENCODER_MODE)

    def transcribe(audio: np.ndarray) -> str:
        inp = proc(np.ascontiguousarray(audio, dtype="float32"))
        inp = {k: (v.cuda() if isinstance(v, torch.Tensor) else v) for k, v in inp.items()}
        with torch.inference_mode():
            text, _ = pipe.transcribe(
                inp["audio_data"], inp["audio_data_seqlens"], inp["input_ids"],
                inp["audio_input_mask"], max_new_tokens=400,
            )
        return text

    # timed wrappers (accumulate compute seconds for RTFx)
    class Timed:
        def __init__(self): self.compute = 0.0
        def __call__(self, audio):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            out = transcribe(audio)
            torch.cuda.synchronize(); self.compute += time.perf_counter() - t0
            return out

    def chunked_transcribe(audio, tx):
        cs = ChunkStreamer(sample_rate=lc.SAMPLE_RATE, chunk_seconds=CHUNK_S,
                           overlap_seconds=OVERLAP_S, min_seconds=4.0,
                           partial_interval_seconds=3.0)
        return cs.flush(audio, tx)

    keys = [k for k, _, _ in lc.DATASETS]
    rows = []
    print(f"\n{'dataset':18} {'clips':>5} {'audio_s':>8} "
          f"{'WER_whole':>9} {'WER_chunk':>9} {'RTFx_whole':>10} {'RTFx_chunk':>10}")
    ws_wers, ck_wers = [], []
    for key in keys:
        clips = lc.load_dataset_split(key, num_samples=N)
        refs = [c.reference for c in clips]
        audio_s = sum(len(c.audio) / lc.SAMPLE_RATE for c in clips)

        tw = Timed(); hyps_w = [tw(c.audio) for c in clips]
        if DO_CHUNKED:
            tc = Timed(); hyps_c = [chunked_transcribe(c.audio, tc) for c in clips]
        else:
            tc = Timed(); hyps_c = hyps_w

        wer_w = wer_pct(refs, hyps_w)
        wer_c = wer_pct(refs, hyps_c)
        ws_wers.append(wer_w); ck_wers.append(wer_c)
        rtfx_w = audio_s / tw.compute
        rtfx_c = audio_s / tc.compute
        rows.append({"dataset": key, "n": len(clips), "audio_s": round(audio_s, 1),
                     "wer_wholeshot": wer_w, "wer_chunked": wer_c,
                     "rtfx_wholeshot": round(rtfx_w, 1), "rtfx_chunked": round(rtfx_c, 1)})
        print(f"{key:18} {len(clips):>5} {audio_s:>8.0f} "
              f"{wer_w:>8.2f}% {wer_c:>8.2f}% {rtfx_w:>10.1f} {rtfx_c:>10.1f}")

    comp_w = sum(ws_wers) / len(ws_wers)
    comp_c = sum(ck_wers) / len(ck_wers)
    print(f"\n{'COMPOSITE (mean)':18} {'':>5} {'':>8} {comp_w:>8.2f}% {comp_c:>8.2f}%   "
          f"(chunked {'-' if comp_c < comp_w else '+'}{abs(comp_w - comp_c):.2f} pts)")

    out = REPO / "outputs" / "moss_leaderboard_chunked.json"
    out.write_text(json.dumps({"rows": rows, "composite_wholeshot": round(comp_w, 2),
                               "composite_chunked": round(comp_c, 2),
                               "chunk_s": CHUNK_S, "overlap_s": OVERLAP_S}, indent=2))
    print(f"[bench] saved -> {out}")


if __name__ == "__main__":
    raise SystemExit(main())
