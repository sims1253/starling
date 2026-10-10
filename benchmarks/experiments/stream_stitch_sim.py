"""Offline stitch replay: streaming finals without real-time pacing (issue #357).

The streaming final of a chunked take depends only on the raw text the
engine returns for each committed window (12 s windows every 9 s, their
re-decodes, the tail at stop) and on the stitcher. This runs the Python
``ChunkStreamer`` over a workload take exactly as a commit would and answers
its transcribe calls from a cache of window texts; a missing window is
transcribed through a server's batch endpoint (the same samples the stream
path decodes) and added to the cache. Each take is also replayed from later
starting offsets (``--offsets``), which moves every window boundary, so a
stitcher is judged on many boundary placements instead of one.

    python benchmarks/experiments/stream_stitch_sim.py \\
        --binary build/starling-serve --model models/parakeet.gguf \\
        --workload build/stream-workload --cache runs/windows.json \\
        --offsets 0,1.5,3,4.5,6,7.5

Reports per take and offset the final-vs-batch WER and the omissions and
duplications located in the take (``stream_replay.locate_errors``). The
native server stitches identically (tests/test_stream_chunk.py parity).
"""

from __future__ import annotations

import argparse
import io
import json
import statistics
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
# --src replays another checkout's stitcher (e.g. the base revision).
_SRC = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--src=")),
            str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, _SRC)

from runner import ArmServer, _free_port  # noqa: E402
from stream_replay import SAMPLE_RATE, _batch_bytes, _read_pcm, locate_errors, wer  # noqa: E402


def _wav(pcm: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm)
    return buf.getvalue()


class Transcriber:
    """Window texts by (take, first sample, length), from the cache or a
    lazily started server."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.cache: dict = json.loads(args.cache.read_text()) if args.cache.exists() else {}
        self.server = None
        self.misses = 0

    def text(self, take: str, pcm: bytes, start: int, length: int) -> str:
        key = f"{take}:{start}:{length}"
        if key not in self.cache:
            if self.server is None:
                if not self.args.binary:
                    raise SystemExit(f"window {key} not cached; pass --binary/--model")
                arm = {"binary": str(self.args.binary), "model": str(self.args.model),
                       "model_slug": self.args.model_slug, "args": []}
                self.server = ArmServer(arm, self.args.port or _free_port(),
                                        self.args.cache.with_suffix(".log"))
                self.server.wait_healthy(300)
            self.cache[key], _ = _batch_bytes(self.server.base,
                                              _wav(pcm[2 * start:2 * (start + length)]),
                                              self.args.model_slug, 300)
            self.misses += 1
        return self.cache[key]

    def close(self) -> None:
        if self.server is not None:
            self.server.stop()
        self.args.cache.parent.mkdir(parents=True, exist_ok=True)
        self.args.cache.write_text(json.dumps(self.cache, indent=0, sort_keys=True) + "\n")


def replay_take(name: str, take: dict, pcm: bytes, offset: int, tr: Transcriber,
                args: argparse.Namespace) -> dict:
    from starling.stream_chunk import ChunkStreamer

    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    samples = x[offset:]
    base = samples.__array_interface__["data"][0]
    calls = []
    cs = ChunkStreamer(sample_rate=SAMPLE_RATE, chunk_seconds=args.chunk_seconds,
                       overlap_seconds=args.overlap_seconds, min_seconds=0.0,
                       partial_interval_seconds=0.0)

    def tx(window: np.ndarray) -> str:
        start = (window.__array_interface__["data"][0] - base) // 4
        calls.append({"kind": cs.call_kind, "start_s": start / SAMPLE_RATE,
                      "end_s": (start + len(window)) / SAMPLE_RATE})
        return tr.text(name, pcm, offset + start, len(window))

    final = cs.flush(samples, tx)
    batch = tr.text(name, pcm, offset, len(samples))
    # Utterance spans move with the offset; words cut off at the start are
    # in neither text.
    shift = offset / SAMPLE_RATE
    utts = [{**u, "start_s": u["start_s"] - shift, "end_s": u["end_s"] - shift}
            for u in take["utterances"] if u["end_s"] > shift]
    ref = " ".join(u["text"] for u in utts)
    errs = locate_errors(batch, final, utts, ref, calls)
    return {"take": name, "offset_s": shift, "final": final, "batch": batch,
            "wer_final_vs_batch": round(wer(batch, final), 4),
            "wer_final_vs_ref": round(wer(ref, final), 4),
            "wer_batch_vs_ref": round(wer(ref, batch), 4),
            "redecodes": sum(1 for c in calls if c["kind"] == "redecode"),
            "windows": sum(1 for c in calls if c["kind"] != "redecode"),
            **{k: v for k, v in errs.items()}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--workload", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--binary", type=Path)
    ap.add_argument("--model", type=Path)
    ap.add_argument("--model-slug", default="parakeet")
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--takes", default="short,medium,long")
    ap.add_argument("--offsets", default="0", help="comma-separated start offsets (s)")
    ap.add_argument("--chunk-seconds", type=float, default=12.0)
    ap.add_argument("--overlap-seconds", type=float, default=3.0)
    ap.add_argument("--out", type=Path, help="write every replay as JSON")
    ap.add_argument("--src", help="src/ directory whose starling.stream_chunk to replay "
                    "(--src=PATH; default: this checkout)")
    ap.add_argument("-v", "--verbose", action="store_true", help="print each error span")
    args = ap.parse_args(argv)
    manifest = json.loads((args.workload / "manifest.json").read_text())
    tr = Transcriber(args)
    rows = []
    try:
        for name in args.takes.split(","):
            take = manifest["takes"][name]
            pcm = _read_pcm(args.workload / take["wav"])
            for off in args.offsets.split(","):
                r = replay_take(name, take, pcm, int(float(off) * SAMPLE_RATE), tr, args)
                rows.append(r)
                print(f"{name:>6} +{r['offset_s']:4.1f}s wer_vs_batch={r['wer_final_vs_batch']:.4f} "
                      f"vs_ref={r['wer_final_vs_ref']:.4f} (batch {r['wer_batch_vs_ref']:.4f}) "
                      f"omitted={r['omitted_words']} inserted={r['inserted_words']} "
                      f"duplicated={r['duplicated_words']} substituted={r['substituted_words']} "
                      f"redecodes={r['redecodes']}/{r['windows']}", flush=True)
                if args.verbose:
                    for e in r["spans"]:
                        print("       ", json.dumps(e))
    finally:
        tr.close()
    for name in args.takes.split(","):
        ws = [r["wer_final_vs_batch"] for r in rows if r["take"] == name]
        om = [r["omitted_words"] for r in rows if r["take"] == name]
        wr = [r["wer_final_vs_ref"] for r in rows if r["take"] == name]
        br = [r["wer_batch_vs_ref"] for r in rows if r["take"] == name]
        print(f"{name:>6} median vs_ref={statistics.median(wr):.4f} max={max(wr):.4f} "
              f"(batch median {statistics.median(br):.4f} max {max(br):.4f}) "
              f"median wer_vs_batch={statistics.median(ws):.4f} "
              f"max={max(ws):.4f} omitted total={sum(om)} (cache misses {tr.misses})")
    if args.out:
        args.out.write_text(json.dumps(rows, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
