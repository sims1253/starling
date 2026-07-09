"""Open ASR Leaderboard accuracy + RTFx benchmark.

Sweeps (model x engine) over the 7 English short-form Open ASR Leaderboard
datasets, scoring each with the leaderboard's normalization + kaldialign WER
(see :mod:`wer_leaderboard`) and timing RTFx per clip. The point is a real
quality metric on diverse real audio -- the synthetic-tile WER in
:mod:`bench_all` only checks starling-vs-stock byte-exact drift.

  uv run python benchmarks/bench_leaderboard.py                 # capped, fast
  uv run python benchmarks/bench_leaderboard.py --num-samples 0  # full splits
  uv run python benchmarks/bench_leaderboard.py --models granite --engines starling,stock

The corpus is cached under tests/fixtures/leaderboard_corpus/ on first run
(needs network + a HF token for the gated ``hf-audio/open-asr-leaderboard``
repo; set ``HF_TOKEN``). Later runs read the cache and need no network.

Output:
  outputs/leaderboard.json           full structured results
  outputs/leaderboard.md             markdown tables
  README <!-- BENCH:WER:START/END --> (with --update-readme)
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
import warnings
from pathlib import Path

import torch
from tabulate import tabulate

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tests" / "fixtures"))
sys.path.insert(0, str(REPO_ROOT / "benchmarks"))

import leaderboard_corpus as lc  # noqa: E402
from engines import Engine, SkipCell, build_engines  # noqa: E402
from wer_leaderboard import score_dataset  # noqa: E402

OUTPUTS = REPO_ROOT / "outputs"
README = REPO_ROOT / "README.md"
START, END = "<!-- BENCH:WER:START -->", "<!-- BENCH:WER:END -->"

MODEL_LABELS = {
    "granite": "granite-speech-4.1-2b",
    "parakeet": "parakeet-tdt-0.6b-v3",
    "parakeet_unified": "parakeet-unified-en-0.6b",
    "moss": "moss-transcribe-preview-2b",
    "qwen3": "qwen3-asr-1.7b",
    "ark": "ark-asr-3b",
    "cohere": "cohere-transcribe-03-2026",
    "higgs": "higgs-audio-v3-stt",
}


def _gpu_name() -> str:
    try:
        return torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        return "n/a"


def _run_engine_on_dataset(
    engine: Engine, clips: list[lc.LeaderboardClip], *, warmup: int,
) -> dict:
    """Transcribe every clip once (timed), score WER + RTFx.

    warmup: run the first clip this many extra (untimed) times so CUDA graphs
    are captured before the timed loop (matches bench_all's warmup discipline).
    """
    if not clips:
        return {"n": 0, "wer_pct": float("nan"), "rtfx": float("nan")}

    refs: list[str] = []
    hyps: list[str] = []
    times_s: list[float] = []
    durs_s: list[float] = []

    # warmup on the first clip (captures graphs, pays one-off costs)
    try:
        for _ in range(warmup):
            engine.transcribe(clips[0].audio, B=1)
    except SkipCell as sc:
        return {"n": len(clips), "skipped": sc.reason,
                "wer_pct": float("nan"), "rtfx": float("nan")}

    for clip in clips:
        try:
            # one timed call produces both the hypothesis and the latency
            t0 = time.perf_counter()
            torch.cuda.synchronize()
            text = engine.transcribe(clip.audio, B=1)[0]
            torch.cuda.synchronize()
            t = time.perf_counter() - t0
        except SkipCell:
            refs.append(clip.reference)
            hyps.append("")
            times_s.append(float("nan"))
            durs_s.append(clip.duration_s)
            continue
        refs.append(clip.reference)
        hyps.append(text)
        times_s.append(t)
        durs_s.append(clip.duration_s)

    # filter out any nan-timed (skipped) clips for WER if ALL skipped
    if all(t != t for t in times_s):  # all nan
        return {"n": len(clips), "skipped": "all clips skipped",
                "wer_pct": float("nan"), "rtfx": float("nan")}

    valid = [(r, h, t, d) for r, h, t, d in zip(refs, hyps, times_s, durs_s)
             if t == t]
    vrefs, vhyps, vtimes, vdurs = zip(*valid)
    scored = score_dataset(list(vrefs), list(vhyps),
                           times_s=list(vtimes), durations_s=list(vdurs))
    latency_ms = [t * 1000.0 for t in vtimes]
    scored.update({
        "latency_ms_median": round(statistics.median(latency_ms), 1),
        "latency_ms_stdev": round(statistics.stdev(latency_ms), 1)
        if len(latency_ms) > 1 else 0.0,
        "latency_ms_min": round(min(latency_ms), 1),
        "latency_ms_max": round(max(latency_ms), 1),
    })
    return scored


def run_grid(
    models: list[str],
    engine_map: dict[str, list[Engine]],
    corpus: dict[str, list[lc.LeaderboardClip]],
    *,
    warmup: int,
) -> dict:
    records: list[dict] = []
    for model in models:
        engines = engine_map.get(model, [])
        for engine in engines:
            print(f"\n=== {engine.name}-{model} ===  loading ...", flush=True)
            engine.load()
            for key, clips in corpus.items():
                rec = _run_engine_on_dataset(engine, clips, warmup=warmup)
                rec.update({"model": model, "engine": engine.name, "dataset": key})
                records.append(rec)
                wer = rec.get("wer_pct")
                rtfx = rec.get("rtfx")
                wer_s = f"{wer:.2f}%" if wer == wer else "skip"
                rtfx_s = f"{rtfx:.0f}x" if rtfx == rtfx else "-"
                spread_s = (
                    f"σ={rec['latency_ms_stdev']:.0f}ms"
                    if rec.get("latency_ms_stdev") is not None else ""
                )
                print(f"  {key:20s} n={rec.get('n',0):4d}  WER {wer_s:>8s}  "
                      f"RTFx {rtfx_s:>7s}  {spread_s}", flush=True)
            engine.close()
    return {
        "title": "starling open-asr-leaderboard benchmark",
        "date": time.strftime("%Y-%m-%d"),
        "hardware": _gpu_name(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "method": (
            "Open ASR Leaderboard English short-form: 7 datasets "
            "(voxpopuli/ami/earnings22/gigaspeech/librispeech clean+other/spgispeech) "
            "from hf-audio/open-asr-leaderboard. WER = kaldialign "
            "batch_error_rate(merge_compounds=True) on Whisper EnglishTextNormalizer-"
            "normalized text (%). Composite = unweighted mean of per-dataset WERs. "
            "RTFx = total audio_s / total inference_s."),
        "records": records,
    }


# ---------------------------------------------------------------------- #
# markdown
# ---------------------------------------------------------------------- #
def _wer_table(records: list[dict], models: list[str], engines: list[str],
               datasets: list[str]) -> str:
    header = ["model", "engine"] + datasets + ["avg"]
    rows = []
    for model in models:
        for engine in engines:
            cells = []
            vals = []
            for ds in datasets:
                rec = next((r for r in records
                            if r["model"] == model and r["engine"] == engine
                            and r["dataset"] == ds), None)
                if rec is None or rec.get("skipped"):
                    cells.append("—")
                else:
                    w = rec["wer_pct"]
                    cells.append(f"{w:.2f}%" if w == w else "—")
                    if w == w:
                        vals.append(w)
            avg = f"{sum(vals)/len(vals):.2f}%" if vals else "—"
            if any(c != "—" for c in cells) or vals:
                rows.append([MODEL_LABELS.get(model, model), engine, *cells, avg])
    out = ["**Open ASR Leaderboard — WER %** (per dataset, unweighted mean avg)\n"]
    out.append(tabulate(rows, headers=header, tablefmt="github"))
    return "\n".join(out)


def _rtfx_table(records: list[dict], models: list[str], engines: list[str],
                datasets: list[str]) -> str:
    header = ["model", "engine"] + datasets
    rows = []
    for model in models:
        for engine in engines:
            cells = []
            has = False
            for ds in datasets:
                rec = next((r for r in records
                            if r["model"] == model and r["engine"] == engine
                            and r["dataset"] == ds), None)
                if rec is None or rec.get("skipped") or rec.get("rtfx") is None:
                    cells.append("—")
                else:
                    r = rec["rtfx"]
                    spread = rec.get("latency_ms_stdev")
                    cells.append(
                        f"{r:.0f}x (σ {spread:.0f}ms)"
                        if r == r and spread is not None
                        else (f"{r:.0f}x" if r == r else "—")
                    )
                    if r == r:
                        has = True
            if has:
                rows.append([MODEL_LABELS.get(model, model), engine, *cells])
    out = ["\n**Open ASR Leaderboard — RTFx** (real audio_s / inference_s)\n"]
    out.append(tabulate(rows, headers=header, tablefmt="github"))
    return "\n".join(out)


def build_markdown(results: dict, *, models, engines) -> str:
    records = results["records"]
    datasets = [k for k, _, _ in lc.DATASETS]
    parts = [_wer_table(records, models, engines, datasets),
             _rtfx_table(records, models, engines, datasets)]
    return "\n".join(parts).rstrip() + "\n"


def _aggregate_duplicate_records(results: dict) -> dict:
    """Merge duplicate model/engine/dataset shards exactly from edit counts.

    Per-dataset sharding is useful for CUDA-graph-heavy models whose shape cache
    would otherwise accumulate across many clips. The shard JSONs carry
    kaldialign edit counts plus total audio/inference seconds, so WER and RTFx
    can be reconstructed exactly for duplicate cells.
    """
    buckets: dict[tuple[str, str, str], dict] = {}
    passthrough: list[dict] = []
    for rec in results["records"]:
        key = (rec["model"], rec["engine"], rec["dataset"])
        if {"ins", "dele", "sub", "ref_len", "audio_s", "infer_s"}.issubset(rec):
            acc = buckets.setdefault(
                key,
                {
                    "model": rec["model"],
                    "engine": rec["engine"],
                    "dataset": rec["dataset"],
                    "n": 0,
                    "ins": 0,
                    "dele": 0,
                    "sub": 0,
                    "ref_len": 0,
                    "audio_s": 0.0,
                    "infer_s": 0.0,
                },
            )
            acc["n"] += int(rec.get("n", 0))
            acc["ins"] += int(rec["ins"])
            acc["dele"] += int(rec["dele"])
            acc["sub"] += int(rec["sub"])
            acc["ref_len"] += int(rec["ref_len"])
            acc["audio_s"] += float(rec["audio_s"])
            acc["infer_s"] += float(rec["infer_s"])
        else:
            passthrough.append(rec)

    merged_records = []
    for rec in buckets.values():
        edits = rec["ins"] + rec["dele"] + rec["sub"]
        rec["wer_pct"] = round(100.0 * edits / rec["ref_len"], 2) if rec["ref_len"] else float("nan")
        rec["rtfx"] = round(rec["audio_s"] / rec["infer_s"], 2) if rec["infer_s"] > 0 else float("inf")
        rec["audio_s"] = round(rec["audio_s"], 1)
        rec["infer_s"] = round(rec["infer_s"], 1)
        merged_records.append(rec)
    results = dict(results)
    results["records"] = merged_records + passthrough
    return results


def splice_readme(md_body: str) -> bool:
    text = README.read_text()
    block = f"{START}\n{md_body}{END}"
    if START in text and END in text:
        pre = text[: text.index(START)]
        post = text[text.index(END) + len(END):]
        new = pre + block + post
    else:
        # append after the existing latency BENCH block, else at end
        anchor = "<!-- BENCH:END -->"
        if anchor in text:
            idx = text.index(anchor) + len(anchor)
            new = text[:idx] + "\n\n" + block + text[idx:]
        else:
            new = text.rstrip() + "\n\n" + block + "\n"
    if new != text:
        README.write_text(new)
        return True
    return False


def _parse_csv(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", default="granite,parakeet,moss",
                    help="comma list of model slugs")
    ap.add_argument("--engines", default="starling,stock",
                    help="comma list of engine families")
    ap.add_argument("--datasets",
                    default=",".join(k for k, _, _ in lc.DATASETS),
                    help="comma list of dataset keys")
    ap.add_argument("--num-samples", type=int, default=50,
                    help="first-N clips per dataset (0 = full split)")
    ap.add_argument("--sample-offset", type=int, default=0,
                    help="skip this many raw clips before applying --num-samples")
    ap.add_argument("--warmup", type=int, default=2,
                    help="untimed warmup runs (graph capture) on the first clip")
    ap.add_argument("--update-readme", action="store_true",
                    help="splice the WER/RTFx tables into README.md")
    ap.add_argument("--from-json", nargs="*", default=None,
                    help="skip the run; rebuild tables from one or more JSON "
                         "files (merges records across them). With no args, "
                         "reads outputs/leaderboard.json. Useful to combine "
                         "per-model runs into one table.")
    ap.add_argument("--out", default=None,
                    help="output JSON path (default outputs/leaderboard.json). "
                         "Set per-model to avoid overwriting during isolated runs.")
    args = ap.parse_args(argv)

    warnings.filterwarnings("ignore")
    models = _parse_csv(args.models)
    engines = _parse_csv(args.engines)
    dataset_keys = _parse_csv(args.datasets)
    n = args.num_samples or None

    if args.from_json is not None:
        files = args.from_json or [str(OUTPUTS / "leaderboard.json")]
        merged = None
        skipped = []
        for f in files:
            try:
                data = json.loads(Path(f).read_text())
            except (OSError, json.JSONDecodeError) as e:
                # a per-model run that crashed may have left a missing/corrupt
                # JSON; skip it so the merge still produces a table from the
                # models that succeeded.
                skipped.append(f"{f} ({type(e).__name__})")
                continue
            if merged is None:
                merged = data
            else:
                merged["records"].extend(data["records"])
        if merged is None:
            print("[leaderboard] no valid JSON to merge; nothing to do.")
            return 1
        merged = _aggregate_duplicate_records(merged)
        if skipped:
            print(f"[leaderboard] skipped {len(skipped)} unparseable/missing JSON(s):")
            for s in skipped:
                print(f"          {s}")
        results = merged
        models = _ordered_models(results)
        engines = _ordered_engines(results)
        md = build_markdown(results, models=models, engines=engines)
        (OUTPUTS / "leaderboard.md").write_text(md)
        print(md)
        if args.update_readme:
            changed = splice_readme(md)
            print(f"\n[leaderboard] README {'updated' if changed else 'unchanged'}")
        return 0

    token = os.environ.get("HF_TOKEN")
    sample_offset = max(0, int(args.sample_offset))
    print(
        f"[leaderboard] loading corpus (num_samples={n}, sample_offset={sample_offset}) ...",
        flush=True,
    )
    corpus = lc.load_all(n, keys=dataset_keys, token=token, sample_offset=sample_offset)
    for k, clips in corpus.items():
        print(f"  {k:20s} {len(clips)} clips")

    engine_map = build_engines(models, engines)
    if not engine_map:
        print("[leaderboard] no engines resolved.")
        return 1
    for m in list(engine_map):
        print(f"[leaderboard] {m}: " + ", ".join(e.name for e in engine_map[m]))

    from starling.parakeet.gpu_lock import with_gpu_lock

    with with_gpu_lock(
        session="leaderboard", model="+".join(engine_map),
        eta_min=60, note="open-asr-leaderboard sweep",
    ):
        results = run_grid(list(engine_map.keys()), engine_map, corpus,
                           warmup=args.warmup)

    OUTPUTS.mkdir(exist_ok=True)
    out_path = Path(args.out) if args.out else (OUTPUTS / "leaderboard.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write (temp + rename) so a crash mid-run never leaves a corrupt
    # half-written JSON that would break the final --from-json merge.
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(results, indent=2))
    tmp.replace(out_path)
    print(f"\n[leaderboard] wrote {out_path}")

    all_engines = sorted({r["engine"] for r in results["records"]})
    md = build_markdown(results, models=list(engine_map.keys()), engines=all_engines)
    (OUTPUTS / "leaderboard.md").write_text(md)
    print("\n[leaderboard] markdown:\n")
    print(md)

    if args.update_readme:
        changed = splice_readme(md)
        print(f"\n[leaderboard] README {'updated' if changed else 'unchanged'}")
    return 0


def _ordered_models(results: dict) -> list[str]:
    return list(dict.fromkeys(r["model"] for r in results["records"]))


def _ordered_engines(results: dict) -> list[str]:
    return list(dict.fromkeys(r["engine"] for r in results["records"]))


if __name__ == "__main__":
    raise SystemExit(main())
