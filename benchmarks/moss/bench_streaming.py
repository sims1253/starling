"""Streaming chunk/overlap sweep for MOSS-Transcribe.

Simulates the /stream fixed-window overlapping-chunk path (starling.stream_chunk)
over real concatenated LibriSpeech-clean audio at several total stream lengths,
sweeping ``chunk_seconds`` x ``overlap_seconds``.  For each cell it reports:

* **RTFx**        stream_seconds / total_compute_seconds (headroom over real time;
                  must be >> 1 to keep up while dictating).
* **commit_ms**   wall time of the final ``flush`` after you stop talking -- the
                  felt latency from speech-end to committed text.
* **redund.**     seconds of audio transcribed / stream_seconds (re-work factor).
* **WER%**        stitched transcript vs the human reference (accuracy cost of
                  chunking), via the Open-ASR-Leaderboard scorer.

Uses the same fp8 + adaptive-cudagraph pipeline the MOSS server runs.

Run:  ``uv run python -m benchmarks.moss.bench_streaming``
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "benchmarks"))
SR = 16000
CORPUS = REPO / "tests" / "fixtures" / "leaderboard_corpus" / "librispeech_clean__n50"


def load_clips():
    import soundfile as sf

    ref = json.loads((CORPUS / "reference.json").read_text())
    clips = []
    for i in sorted(ref, key=lambda x: int(x)):
        wav, sr = sf.read(str(CORPUS / f"clip_{int(i):05d}.wav"))
        if wav.ndim > 1:
            wav = wav.mean(1)
        clips.append((wav.astype("float32"), ref[i]))
    return clips


def build_stream(clips, target_s):
    """Concatenate whole clips until >= target_s. Returns (audio, ref, actual_s)."""
    audio, texts, got = [], [], 0.0
    ci = 0
    while got < target_s:
        wav, txt = clips[ci % len(clips)]
        ci += 1
        audio.append(wav)
        texts.append(txt)
        got += len(wav) / SR
    return np.concatenate(audio), " ".join(texts), got


class Transcriber:
    """MOSS transcribe callback that accumulates compute time / audio seconds."""

    def __init__(self, pipe, proc, max_new_tokens=256):
        self.pipe, self.proc, self.mnt = pipe, proc, max_new_tokens
        self.reset()

    def reset(self):
        self.compute_s = 0.0
        self.calls = 0
        self.audio_s = 0.0

    def __call__(self, window):
        w = np.ascontiguousarray(window, dtype=np.float32)
        inp = self.proc(w)
        inp = {k: (v.cuda() if isinstance(v, torch.Tensor) else v) for k, v in inp.items()}
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            text, _ = self.pipe.transcribe(
                inp["audio_data"], inp["audio_data_seqlens"], inp["input_ids"],
                inp["audio_input_mask"], max_new_tokens=self.mnt,
            )
        torch.cuda.synchronize()
        self.compute_s += time.perf_counter() - t0
        self.calls += 1
        self.audio_s += len(w) / SR
        return text


def stream_run(audio, chunk, overlap, partial_interval, tx):
    from starling.stream_chunk import ChunkStreamer

    cs = ChunkStreamer(
        sample_rate=SR, chunk_seconds=chunk, overlap_seconds=overlap,
        min_seconds=min(4.0, chunk * 0.4), partial_interval_seconds=partial_interval,
    )
    tx.reset()
    n = len(audio)
    now = 0.0
    for end in range(SR, n + 1, SR):  # feed 1s at a time
        now += 1.0
        cs.step(audio[:end], now, tx)
    if n % SR:
        now += 1.0
        cs.step(audio, now, tx)
    stream_compute = tx.compute_s
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    final = cs.flush(audio, tx)
    torch.cuda.synchronize()
    commit_s = time.perf_counter() - t0
    return {
        "final": final,
        "stream_compute_s": stream_compute,
        "commit_s": commit_s,
        "audio_transcribed_s": tx.audio_s,
    }


def whole_shot(audio, tx):
    """Single transcribe of the whole stream (accuracy ceiling; may overflow)."""
    tx.reset()
    try:
        text = tx(audio)
    except Exception as exc:  # cache overflow on long streams
        return None, str(exc)[:40]
    return text, None


def main():
    from starling.flags import OptFlags, set_default_flags
    from starling.moss.loader import load_model_and_processor
    from starling.moss.pipeline import MossMegaPipeline
    from wer_leaderboard import wer_pct

    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=float, nargs="+", default=[8, 12, 20])
    ap.add_argument("--overlaps", type=float, nargs="+", default=[1, 2, 4])
    ap.add_argument("--lengths", type=float, nargs="+", default=[30, 90, 180])
    ap.add_argument("--partial-interval", type=float, default=3.0)
    ap.add_argument("--fp8", action="store_true",
                    help="enable fp8 decode (batch-only; unstable under streaming churn)")
    args = ap.parse_args()

    if args.fp8:
        set_default_flags(OptFlags(tolerance_mode=True, fp8_weights=True))
    else:
        set_default_flags(OptFlags())
    print(f"[bench] loading MOSS (decode={'fp8' if args.fp8 else 'bf16'} + adaptive cudagraph) ...")
    model, proc = load_model_and_processor()
    pipe = MossMegaPipeline(model, proc, max_cache_len=2048, encoder_mode="cudagraph")
    tx = Transcriber(pipe, proc)

    clips = load_clips()
    streams = {T: build_stream(clips, T) for T in args.lengths}

    # warmup: exercise a couple of chunk lengths so torch.compile + captures are paid
    print("[bench] warmup ...")
    for c in {args.chunks[0], args.chunks[-1]}:
        stream_run(streams[args.lengths[0]][0], c, args.overlaps[0], args.partial_interval, tx)

    results = []
    print(f"\n{'len_s':>6} {'chunk':>5} {'ovlp':>4} {'RTFx':>6} {'commit_ms':>9} "
          f"{'redund':>7} {'WER%':>6} {'n_win':>5}")
    for T in args.lengths:
        audio, ref, actual = streams[T]
        # accuracy ceiling: one-shot over the whole stream (skip if it overflows)
        ws_text, ws_err = whole_shot(audio, tx)
        ws_wer = wer_pct([ref], [ws_text]) if ws_text is not None else float("nan")
        for chunk in args.chunks:
            for overlap in args.overlaps:
                if overlap >= chunk:
                    continue
                r = stream_run(audio, chunk, overlap, args.partial_interval, tx)
                total_compute = r["stream_compute_s"] + r["commit_s"]
                rtfx = actual / total_compute
                redund = r["audio_transcribed_s"] / actual
                wer = wer_pct([ref], [r["final"]])
                n_win = int((len(audio) / SR - overlap) // max(1e-9, (chunk - overlap))) + 1
                row = {
                    "len_s": round(actual, 1), "chunk": chunk, "overlap": overlap,
                    "rtfx": round(rtfx, 1), "commit_ms": round(r["commit_s"] * 1000, 1),
                    "redundancy": round(redund, 2), "wer_pct": wer, "n_windows": n_win,
                    "whole_shot_wer": ws_wer,
                }
                results.append(row)
                print(f"{actual:>6.0f} {chunk:>5.0f} {overlap:>4.0f} {rtfx:>6.1f} "
                      f"{row['commit_ms']:>9.1f} {redund:>7.2f} {wer:>6.2f} {n_win:>5}")
        print(f"   (len {actual:.0f}s whole-shot WER = "
              f"{ws_wer if ws_text is not None else 'overflow('+str(ws_err)+')'})")

    out = REPO / "outputs"
    out.mkdir(exist_ok=True)
    (out / "moss_streaming_bench.json").write_text(json.dumps(results, indent=2))
    print(f"\n[bench] saved -> {out}/moss_streaming_bench.json")


if __name__ == "__main__":
    raise SystemExit(main())
