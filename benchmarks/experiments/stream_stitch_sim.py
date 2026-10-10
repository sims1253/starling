"""Offline stitch replay: streaming finals without real-time pacing (issue #357).

The streaming final of a chunked take depends only on the raw text the
engine returns for each committed window (12 s windows every 9 s, their
re-decodes, the tail at stop) and on the stitcher. This runs the Python
``ChunkStreamer`` over a workload take exactly as a commit would and answers
its transcribe calls from a cache of window texts; a missing window is
transcribed through a server's batch endpoint (the same samples the stream
path decodes) and added to the cache. With ``--words`` the windows carry the
engine's word timestamps (issue #357): a missing window is streamed to the
server's ``/stream?trace=1`` and its words are read from the trace, and the
stitcher deduplicates by time; ``--words`` with ``--no-times`` replays the
same cached windows without their times (the text-only stitcher). Each take is also replayed from later
starting offsets (``--offsets``), which moves every window boundary, so a
stitcher is judged on many boundary placements instead of one.

    python benchmarks/experiments/stream_stitch_sim.py \\
        --binary build/starling-serve --model models/parakeet.gguf \\
        --workload build/stream-workload --cache runs/windows.json \\
        --offsets 0,1.5,3,4.5,6,7.5

Reports per take and offset the final-vs-batch WER and the omissions and
duplications located in the take (``stream_replay.locate_errors``). The
native server stitches identically (tests/test_stream_chunk.py parity).
At an offset inside an utterance, that utterance's whole text stays in the
reference, so reference WER there counts the cut-off words; batch WER of the
same cropped audio is the like-for-like comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import statistics
import sys
import wave
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
_DEFAULT_SRC = Path(__file__).resolve().parents[2] / "src"

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


def _file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


def use_stitcher(src: Path) -> None:
    """Import starling.stream_chunk from ``src`` (e.g. the base revision's
    checkout), refusing to run with another copy already imported."""
    src = src.resolve()
    sys.path.insert(0, str(src))
    import starling.stream_chunk as stream_chunk

    if not Path(stream_chunk.__file__).resolve().is_relative_to(src):
        raise SystemExit(f"--src {src}: starling.stream_chunk already imported "
                         f"from {stream_chunk.__file__}")


class Transcriber:
    """Window texts by (engine, audio), from the cache or a lazily started
    server. A cache entry is keyed on everything its text depends on: the
    server binary, the model file, the model slug and server arguments
    (their content hashes, not paths) and the window's samples, so a cache
    reused with another build, model or workload misses instead of
    returning another engine's or another recording's text."""

    SERVER_ARGS: list[str] = []
    # --words: one stream window covers any replayed window, so the stream's
    # first decode is the window's raw text (never a re-decode of it).
    WORDS_SERVER_ARGS: list[str] = ["--stream-chunk-seconds", "60"]

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.cache: dict = json.loads(args.cache.read_text()) if args.cache.exists() else {}
        self.server = None
        self.misses = 0
        self.words = getattr(args, "words", False)
        self.no_times = getattr(args, "no_times", False)
        self.server_args = self.WORDS_SERVER_ARGS if self.words else self.SERVER_ARGS
        self.engine = hashlib.sha256(json.dumps({
            "binary": _file_digest(args.binary), "model": _file_digest(args.model),
            "model_slug": args.model_slug, "args": self.server_args,
            **({"words": True} if self.words else {}),
        }, sort_keys=True).encode()).hexdigest()[:16]

    def key(self, pcm: bytes, start: int, length: int) -> str:
        span = pcm[2 * start:2 * (start + length)]
        return f"{self.engine}:{hashlib.sha256(span).hexdigest()[:24]}:{length}"

    def text(self, pcm: bytes, start: int, length: int) -> str:
        key = self.key(pcm, start, length)
        if key not in self.cache:
            self.cache[key] = self.transcribe(_wav(pcm[2 * start:2 * (start + length)]))
            self.misses += 1
        got = self.cache[key]
        return got if isinstance(got, str) else got["text"]

    def window(self, pcm: bytes, start: int, length: int):
        """The window's text, with its word timestamps under --words (a
        ``(text, ((word, start, end), ...))`` pair, made a ``Transcript`` by
        replay_take) unless --no-times."""
        if not self.words:
            return self.text(pcm, start, length)
        key = "words:" + self.key(pcm, start, length)
        if key not in self.cache:
            self.cache[key] = self.stream_words(pcm[2 * start:2 * (start + length)])
            self.misses += 1
        got = self.cache[key]
        if self.no_times or got["words"] is None:
            return got["text"]
        return got["text"], tuple((w["w"], w["start"], w["end"]) for w in got["words"])

    def _server(self):
        if self.server is None:
            arm = {"binary": str(self.args.binary), "model": str(self.args.model),
                   "model_slug": self.args.model_slug, "args": self.server_args}
            self.server = ArmServer(arm, self.args.port or _free_port(),
                                    self.args.cache.with_suffix(".log"))
            self.server.wait_healthy(300)
        return self.server

    def transcribe(self, wav: bytes) -> str:
        text, _ = _batch_bytes(self._server().base, wav, self.args.model_slug, 300)
        return text

    def stream_words(self, pcm: bytes) -> dict:
        """Text and words of the first decode of ``pcm`` streamed as one take
        (the trace's first committed call over the whole window)."""
        from websockets.sync.client import connect

        url = self._server().base.replace("http://", "ws://") + "/stream?trace=1"
        n = len(pcm) // 2
        for _ in range(50):
            msg = {}
            with connect(url, max_size=None, ping_interval=None, open_timeout=30) as ws:
                ws.send(pcm)
                ws.send(json.dumps({"type": "commit"}))
                for raw in ws:
                    msg = json.loads(raw)
                    if msg.get("type") in ("final", "error"):
                        break
            if msg.get("type") == "final":
                break
            if msg.get("message") != "server busy":
                raise RuntimeError(f"stream window failed: {msg}")
        else:
            raise RuntimeError("stream window: server stayed busy")
        for call in msg.get("trace", {}).get("calls", []):
            if (call["kind"] in ("flush_tail", "flush_window") and call["start_s"] == 0
                    and abs(call["end_s"] - n / SAMPLE_RATE) < 0.001
                    and call["result"] in ("ok", "reused")):
                words = call.get("words")
                return {"text": " ".join(w["w"] for w in words) if words is not None
                        else msg["text"], "words": words}
        raise RuntimeError(f"stream window: no decode of the whole window in {msg.get('trace')}")

    def close(self) -> None:
        if self.server is not None:
            self.server.stop()
        self.args.cache.parent.mkdir(parents=True, exist_ok=True)
        self.args.cache.write_text(json.dumps(self.cache, indent=0, sort_keys=True) + "\n")


def replay_take(name: str, take: dict, pcm: bytes, offset: int, tr: Transcriber,
                args: argparse.Namespace) -> dict:
    # numpy (and the stitcher) only for a real replay: the CI harness
    # imports this module without them.
    import numpy as np
    from starling.stream_chunk import ChunkStreamer

    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    samples = x[offset:]
    base = samples.__array_interface__["data"][0]
    calls = []
    cs = ChunkStreamer(sample_rate=SAMPLE_RATE, chunk_seconds=args.chunk_seconds,
                       overlap_seconds=args.overlap_seconds, min_seconds=0.0,
                       partial_interval_seconds=0.0)

    def tx(window: "np.ndarray"):
        start = (window.__array_interface__["data"][0] - base) // 4
        calls.append({"kind": cs.call_kind, "start_s": start / SAMPLE_RATE,
                      "end_s": (start + len(window)) / SAMPLE_RATE})
        got = tr.window(pcm, offset + start, len(window))
        if isinstance(got, str):
            return got
        # Only a timed window needs the types: --src stitchers from before
        # #357 have neither.
        from starling.stream_chunk import TimedWord, Transcript

        return Transcript(got[0], tuple(TimedWord(*w) for w in got[1]))

    final = cs.flush(samples, tx)
    batch = tr.text(pcm, offset, len(samples))
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
    ap.add_argument("--binary", type=Path, required=True,
                    help="server for cache misses; its hash is part of every cache key")
    ap.add_argument("--model", type=Path, required=True,
                    help="model file; its hash is part of every cache key")
    ap.add_argument("--model-slug", default="parakeet")
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--takes", default="short,medium,long")
    ap.add_argument("--offsets", default="0", help="comma-separated start offsets (s)")
    ap.add_argument("--chunk-seconds", type=float, default=12.0)
    ap.add_argument("--overlap-seconds", type=float, default=3.0)
    ap.add_argument("--out", type=Path, help="write every replay as JSON")
    ap.add_argument("--src", type=Path, default=_DEFAULT_SRC,
                    help="src/ directory whose starling.stream_chunk to replay "
                    "(default: this checkout)")
    ap.add_argument("--words", action="store_true",
                    help="windows with word timestamps, read from the server's stream trace")
    ap.add_argument("--no-times", action="store_true",
                    help="with --words: replay the same windows without their timestamps")
    ap.add_argument("-v", "--verbose", action="store_true", help="print each error span")
    args = ap.parse_args(argv)
    use_stitcher(args.src)
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
