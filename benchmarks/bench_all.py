"""Unified, re-runnable benchmark across all supported models and engines.

One command sweeps every (model x engine x length x batch) cell of the grid on
the SAME fixtures and the SAME ground-truth transcript, so RTFx and WER are
directly comparable across engines. Writes a JSON dump and two markdown tables
to ``outputs/``, and (with ``--update-readme``) splices the tables into
``README.md`` between sentinel comments.

  uv run python benchmarks/bench_all.py                  # run + print + JSON
  uv run python benchmarks/bench_all.py --update-readme  # also refresh README

Engines
-------
  starling          the fused megakernel pipeline (this repo)
  stock             the unmodified HuggingFace ``generate`` reference
  crispasr          the external ggml binary (granite + qwen3 + parakeet backends)
  parakeet.cpp      mudler's C++/ggml parakeet-cli (parakeet only)
  starling-batched  starling with batched LLM decode (granite + qwen3 only)
  starling-spec     starling with self-speculative decode (granite only)
  starling-compiled starling with non-byte-exact compiled encoder (parakeet only)

Cells
-----
  --lengths   short,medium,long   (the deterministic LibriSpeech fixtures)
  --batches   1,8                 (batch sizes; non-batched engines loop Bx1)
  --models    granite,parakeet,moss,ark,cohere[,qwen3,higgs]
  --engines   starling,stock[,crispasr,parakeet.cpp,starling-batched,starling-spec,starling-compiled]

Timing
------
Each cell is warmed up then timed with CUDA events (median of ``--reps`` runs).
Model load + CUDA-graph capture happen during warmup, so steady-state numbers
exclude one-off costs (matching every other bench in this repo). A single
``with_gpu_lock`` session wraps the whole run for isolation.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import warnings
from pathlib import Path
from statistics import median, stdev

import numpy as np
import torch
from tabulate import tabulate

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tests" / "fixtures"))

import make_fixtures as mkfx  # noqa: E402
from engines import Engine, SkipCell, build_engines  # noqa: E402
from wer import REFERENCE_TRANSCRIPTS, cer_pct, wer_pct  # noqa: E402

OUTPUTS = REPO_ROOT / "outputs"
README = REPO_ROOT / "README.md"
START, END = "<!-- BENCH:START -->", "<!-- BENCH:END -->"

MODEL_LABELS = {
    "granite": "granite-speech-4.1-2b",
    "parakeet": "parakeet-tdt-0.6b-v3",
    "parakeet_unified": "parakeet-unified-en-0.6b",
    "moss": "moss-transcribe-preview-2b",
    "qwen3": "qwen3-asr-1.7b",
    "ark": "ark-asr-3b",
    "ark06": "ark-asr-0.6b",
    "cohere": "cohere-transcribe-03-2026",
    "higgs": "higgs-audio-v3-stt",
    "audex": "nemotron-labs-audex-2b",
    "voxtral": "voxtral-mini-4b-realtime",
    "s1": "s1-mini (text normalizer)",
}


# ---------------------------------------------------------------------- #
# timing
# ---------------------------------------------------------------------- #
def _cuda_sync() -> None:
    """Synchronize the GPU when one is present.

    bench_all must also run on CUDA-less hosts (CPU/Vulkan ggml-only sweeps),
    where torch.cuda.synchronize() raises "Found no NVIDIA driver".
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _cuda_reset_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _time_samples_ms(fn, *, warmup: int, reps: int) -> list[float]:
    """Wall-time samples for ``fn`` after warmup, synchronized around each run.

    Bracketed by ``torch.cuda.synchronize()`` so GPU work is fully counted.
    CrispASR (no GPU tensors of ours) is timed the same way; the synchronize is
    a no-op-ish cost relative to its ~seconds-scale runtime.
    """
    for _ in range(warmup):
        fn()
    _cuda_sync()
    samples = []
    for _ in range(reps):
        _cuda_sync()
        t0 = time.perf_counter()
        fn()
        _cuda_sync()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return samples


def _time_ms(fn, *, warmup: int, reps: int) -> float:
    """Backward-compatible median timing helper."""
    return float(median(_time_samples_ms(fn, warmup=warmup, reps=reps)))


def _vram_gb() -> float:
    return torch.cuda.max_memory_allocated() / 1e9


# ---------------------------------------------------------------------- #
# the grid
# ---------------------------------------------------------------------- #
def run_grid(
    models: list[str],
    engine_map: dict[str, list[Engine]],
    lengths: list[str],
    batches: list[int],
    *,
    reps: int,
    warmup: int,
) -> dict:
    """Run every cell; return a structured results dict (also JSON-serializable).

    Batching policy: starling (which has true fused batching) is run at every
    ``batch``; stock transformers and CrispASR have no real fused batch path, so
    they run at B=1 only (their B>1 cells are recorded as "—" -- there is nothing
    meaningful to time). This keeps the sweep tractable and the comparison
    honest: a "B=8" number only appears where 8 clips are decoded together.
    """
    fixtures = mkfx.load_fixtures()
    records: list[dict] = []

    for model in models:
        engines = engine_map.get(model, [])
        for engine in engines:
            tag = f"{engine.name}-{model}"
            print(f"\n=== {tag} ===  loading ...", flush=True)
            engine.load()
            for length in lengths:
                audio = fixtures[length]
                audio_s = len(audio) / mkfx.SAMPLE_RATE
                ref = REFERENCE_TRANSCRIPTS[length]
                if model == "s1":
                    # Text normalizer: no audio semantics. The fixture tier
                    # selects the transcript; "audio_s" becomes the INPUT WORD
                    # count so RTFx reads normalized words/s, and there is no
                    # tiled-LibriSpeech reference to compute WER against.
                    from engines import _s1_transcripts

                    n_words = len(_s1_transcripts()[length].split())
                    audio_s = float(n_words)
                    ref = None
                engine_batches = batches if engine.supports_batch else [1]
                for B in engine_batches:
                    rec = _run_cell(
                        engine, model, length, B, audio, audio_s, ref,
                        reps=reps, warmup=warmup,
                    )
                    records.append(rec)
                    _print_cell(rec)
            engine.close()

    return {
        "title": "starling unified benchmark",
        "date": time.strftime("%Y-%m-%d"),
        "hardware": _gpu_name(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "method": (
            f"B=size, model preloaded, median of {reps} runs (warmup={warmup}), "
            "load + graph capture excluded; RTFx = audio_s / (ms/1000); "
            "WER/CER vs the LibriSpeech 2086-149220-0033 transcript tiled per tier. "
            "stock transformers + CrispASR run at B=1 only (no fused batch path)."
        ),
        "models": {m: MODEL_LABELS.get(m, m) for m in models},
        "records": records,
    }


def _run_cell(
    engine: Engine, model: str, length: str, B: int,
    audio: np.ndarray, audio_s: float, ref: str,
    *, reps: int, warmup: int,
) -> dict:
    # Probe once first: pays graph-capture for starling paths and surfaces
    # SkipCell (e.g. granite single-shot on audio longer than its static KV
    # cache) before we commit to a timed region.
    try:
        probe = engine.transcribe(audio, B=B)
    except SkipCell as sc:
        rec = {
            "model": model, "engine": engine.name, "length": length, "batch": B,
            "audio_s": round(audio_s, 3), "ms": None, "rtfx": None,
            "vram_gb": None, "wer_pct": None, "cer_pct": None,
            "sequential": B > 1 and not engine.supports_batch,
            "reps": reps, "transcript_head": "",
            "skipped": sc.reason,
        }
        return rec

    _cuda_reset_peak()
    timing_samples = _time_samples_ms(
        lambda: engine.transcribe(audio, B=B), warmup=warmup, reps=reps,
    )
    ms = float(median(timing_samples))
    ms_stdev = float(stdev(timing_samples)) if len(timing_samples) > 1 else 0.0
    vram = _vram_gb()
    # one transcript for WER (B copies are identical; take the first)
    hyp = probe[0] if engine.supports_batch and B > 1 else engine.transcribe(audio, B=1)[0]
    wer = wer_pct(ref, hyp) if ref is not None else None
    cer = cer_pct(ref, hyp) if ref is not None else None
    rtfx = audio_s / (ms / 1000.0) if ms > 0 else float("inf")
    sequential = B > 1 and not engine.supports_batch
    return {
        "model": model,
        "engine": engine.name,
        "length": length,
        "batch": B,
        "audio_s": round(audio_s, 3),
        "ms": round(ms, 1),
        "ms_stdev": round(ms_stdev, 1),
        "ms_min": round(min(timing_samples), 1),
        "ms_max": round(max(timing_samples), 1),
        "rtfx": round(rtfx, 1),
        "vram_gb": round(vram, 2),
        "wer_pct": round(wer, 2) if wer is not None else None,
        "cer_pct": round(cer, 2) if cer is not None else None,
        "sequential": sequential,
        "reps": reps,
        "transcript_head": hyp[:80],
    }


def _print_cell(rec: dict) -> None:
    if rec.get("skipped"):
        note = " (Bx1 sequential)" if rec["sequential"] else ""
        print(f"  {rec['length']:6s} B={rec['batch']:<2d}{note}: SKIPPED ({rec['skipped']})")
        return
    note = " (Bx1 sequential)" if rec["sequential"] else ""
    wer = rec["wer_pct"]
    wer_s = f"{wer:5.2f}%" if wer is not None else "  n/a"  # text models: no ref
    print(
        f"  {rec['length']:6s} B={rec['batch']:<2d}{note}: "
        f"{rec['ms']:7.0f}ms ± {rec.get('ms_stdev', 0):5.1f}  RTFx {rec['rtfx']:5.1f}x  "
        f"WER {wer_s}  VRAM {rec['vram_gb']:4.1f}GB"
    )


def _gpu_name() -> str:
    try:
        return torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        return "n/a"


# ---------------------------------------------------------------------- #
# markdown tables
# ---------------------------------------------------------------------- #
def _cells(results: dict) -> list[dict]:
    return results["records"]


def _by_key(records, model, length, batch, engine):
    for r in records:
        if (r["model"] == model and r["length"] == length
                and r["batch"] == batch and r["engine"] == engine):
            return r
    return None


def _engine_cols(model_engines: list[Engine]) -> list[str]:
    # preserve order, dedupe by name
    seen, cols = set(), []
    for e in model_engines:
        if e.name not in seen:
            seen.add(e.name)
            cols.append(e.name)
    return cols


def _row_engines_with_data(records, model, length, B, cols):
    """Engine names that actually have a (non-skipped) cell at this row."""
    out = []
    for eng in cols:
        rec = _by_key(records, model, length, B, eng)
        if rec is not None and not rec.get("skipped"):
            out.append(eng)
    return out


def latency_table(model: str, engines: list[Engine], records, *,
                  lengths, batches) -> str:
    cols = _engine_cols(engines)
    header = ["length", "batch"] + [f"{c}" for c in cols]
    rows, footnotes = [], []
    for length in lengths:
        for B in batches:
            # Skip rows where no engine has data (e.g. B=8 when no engine in
            # this model supports fused batching) -- a row of em-dashes adds
            # noise without information.
            if not _row_engines_with_data(records, model, length, B, cols):
                continue
            row = [length, B]
            for eng in cols:
                rec = _by_key(records, model, length, B, eng)
                if rec is None:
                    row.append("—")
                elif rec.get("skipped"):
                    row.append(f"†{len(footnotes)+1}")
                    footnotes.append(f"{length}/B{B} {eng}: {rec['skipped']}")
                else:
                    spread = rec.get("ms_stdev")
                    timing = (
                        f"{rec['ms']:.0f}±{spread:.0f}ms"
                        if spread is not None
                        else f"{rec['ms']:.0f}ms"
                    )
                    row.append(f"{timing} ({rec['rtfx']:.0f}x)")
            rows.append(row)
    out = [f"**{MODEL_LABELS.get(model, model)}** — latency / RTFx (ms, RTFx×)\n"]
    out.append(tabulate(rows, headers=header, tablefmt="github"))
    if footnotes:
        out.append("")
        for i, fn in enumerate(footnotes, 1):
            out.append(f"†{i} {fn}")
    return "\n".join(out)


def wer_table(model: str, engines: list[Engine], records, *,
              lengths, batches) -> str:
    cols = _engine_cols(engines)
    header = ["length", "batch"] + cols
    rows = []
    for length in lengths:
        for B in batches:
            if not _row_engines_with_data(records, model, length, B, cols):
                continue
            row = [length, B]
            for eng in cols:
                rec = _by_key(records, model, length, B, eng)
                if rec is None or rec.get("skipped"):
                    row.append("—")
                else:
                    row.append(f"{rec['wer_pct']:.2f}%")
            rows.append(row)
    out = [f"**{MODEL_LABELS.get(model, model)}** — WER % vs LibriSpeech reference\n"]
    out.append(tabulate(rows, headers=header, tablefmt="github"))
    return "\n".join(out)


def _engine_map_from_records(records: list[dict]) -> dict[str, list[Engine]]:
    """Reconstruct a minimal engine_map (engine-name-only stubs) from records.

    Used by ``--from-json`` so the table builders can render without re-running
    any model. Only ``Engine.name`` is read by the table code.
    """
    order: dict[str, list[str]] = {}
    for r in records:
        order.setdefault(r["model"], [])
        if r["engine"] not in order[r["model"]]:
            order[r["model"]].append(r["engine"])
    return {
        m: [_EngineStub(name) for name in names] for m, names in order.items()
    }


class _EngineStub:
    """Name-only stand-in for :class:`Engine` (table rendering only)."""

    def __init__(self, name: str) -> None:
        self.name = name


def build_markdown(results: dict, engine_map: dict[str, list[Engine]], *,
                   lengths, batches) -> str:
    """Build the latency/RTFx markdown for the README sentinel block.

    Emits ONLY the latency/RTFx tables. The tiled-fixture WER is meaningless
    (one utterance repeated N times -- the models don't emit it back exactly N
    times, so WER reflects tiling artifacts, not recognition quality) and is
    omitted. Real recognition-quality WER on diverse Open-ASR-Leaderboard audio
    lives in the separate ``BENCH:WER`` block produced by ``bench_leaderboard.py``.
    """
    records = _cells(results)
    models = list(results["models"].keys())
    parts = []
    for model in models:
        engines = engine_map.get(model, [])
        if not engines:
            continue
        parts.append(latency_table(model, engines, records,
                                   lengths=lengths, batches=batches))
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


# ---------------------------------------------------------------------- #
# README splice
# ---------------------------------------------------------------------- #
def splice_readme(md_body: str) -> bool:
    """Replace the sentinel-wrapped region in README.md with ``md_body``.

    Returns True if the file changed. Adds the sentinels after the
    ``## Benchmark`` heading if absent.
    """
    text = README.read_text()
    block = f"{START}\n{md_body}{END}"
    if START in text and END in text:
        pre = text[: text.index(START)]
        post = text[text.index(END) + len(END):]
        new = pre + block + post
    elif "## Benchmark" in text:
        idx = text.index("## Benchmark")
        # insert right after the heading line
        eol = text.index("\n", idx) + 1
        pre, post = text[:eol], text[eol:]
        new = pre + "\n" + block + "\n" + post
    else:
        new = text.rstrip() + "\n\n## Benchmark\n\n" + block + "\n"
    if new != text:
        README.write_text(new)
        return True
    return False


# ---------------------------------------------------------------------- #
# entrypoint
# ---------------------------------------------------------------------- #
def _parse_csv(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _parse_csv_int(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", default="granite,parakeet,moss,ark,cohere",
                    help="comma list of model slugs "
                         "(granite,parakeet,moss,ark,ark06,cohere,qwen3,higgs,audex,voxtral,s1; "
                         "qwen3/higgs/audex/s1 are auto-gated on availability; "
                         "s1 is text-in/text-out: fixture tiers select transcripts "
                         "and RTFx reads normalized words/s)")
    ap.add_argument("--engines", default="starling,stock",
                    help="comma list of engine families: starling,stock,crispasr,"
                         "parakeet.cpp,starling-batched,starling-spec,"
                         "starling-compiled")
    ap.add_argument("--lengths", default="short,medium,long",
                    help="comma list of fixture tiers")
    ap.add_argument("--batches", default="1,8", type=str,
                    help="comma list of batch sizes")
    ap.add_argument("--reps", default=5, type=int, help="timed runs per cell")
    ap.add_argument("--warmup", default=2, type=int, help="untimed warmup runs")
    ap.add_argument("--update-readme", action="store_true",
                    help="splice the tables into README.md (sentinel-wrapped)")
    ap.add_argument("--from-json", action="store_true",
                    help="skip the run; rebuild tables + splice README from "
                         "outputs/bench_all.json (no GPU needed -- useful to "
                         "reformat or refresh the README after a table change)")
    args = ap.parse_args(argv)

    warnings.filterwarnings("ignore")

    lengths = _parse_csv(args.lengths)
    batches = _parse_csv_int(args.batches)

    if args.from_json:
        results = json.loads((OUTPUTS / "bench_all.json").read_text())
        lengths = lengths or list(dict.fromkeys(r["length"] for r in results["records"]))
        batches = batches or sorted({r["batch"] for r in results["records"]})
        engine_map = _engine_map_from_records(results["records"])
        md = build_markdown(results, engine_map, lengths=lengths, batches=batches)
        (OUTPUTS / "bench_all.md").write_text(md)
        print(md)
        if args.update_readme:
            changed = splice_readme(md)
            print(f"\n[bench] README {'updated' if changed else 'unchanged'}")
        return 0

    models = _parse_csv(args.models)
    engines = _parse_csv(args.engines)

    engine_map = build_engines(models, engines)
    if not engine_map:
        print("[bench] no engines resolved for the given --models/--engines.")
        print(f"        available: {', '.join(sorted(__import__('engines').available_keys()))}")
        return 1
    for m in list(engine_map):
        print(f"[bench] {m}: " + ", ".join(e.name for e in engine_map[m]))

    from starling.parakeet.gpu_lock import with_gpu_lock

    with with_gpu_lock(
        session="bench-all", model="+".join(engine_map),
        eta_min=60, note="unified benchmark sweep",
    ):
        results = run_grid(
            list(engine_map.keys()), engine_map, lengths, batches,
            reps=args.reps, warmup=args.warmup,
        )

    OUTPUTS.mkdir(exist_ok=True)
    (OUTPUTS / "bench_all.json").write_text(json.dumps(results, indent=2))
    print(f"\n[bench] wrote {OUTPUTS}/bench_all.json")

    md = build_markdown(results, engine_map, lengths=lengths, batches=batches)
    (OUTPUTS / "bench_all.md").write_text(md)
    print("\n[bench] markdown tables:\n")
    print(md)

    if args.update_readme:
        changed = splice_readme(md)
        print(f"\n[bench] README {'updated' if changed else 'unchanged'}")

    # sanity gate: starling is byte-exact with stock, so per-cell WER should
    # match. Flag any cell where they diverge by more than 1 point (a real bug,
    # not the model's intrinsic WER on the repeated-audio fixtures).
    by_cell = {}
    for r in results["records"]:
        by_cell.setdefault((r["model"], r["length"], r["batch"]), {})[r["engine"]] = r
    drift = []
    for (model, length, B), cells in by_cell.items():
        s, k = cells.get("starling"), cells.get("stock transformers")
        if s and k and not s.get("skipped") and not k.get("skipped"):
            if s["wer_pct"] is None or k["wer_pct"] is None:
                continue  # text models (s1): no shared WER reference
            if abs(s["wer_pct"] - k["wer_pct"]) > 1.0:
                drift.append((model, length, B, s["wer_pct"], k["wer_pct"]))
    if drift:
        print("\n[bench] ERROR: starling vs stock WER drift > 1 percentage point.")
        print("        The threshold tolerates fixture-normalization noise; larger drift")
        print("        violates the benchmark's stock-equivalence sanity gate.")
        for model, length, B, sw, kw in drift:
            print(f"          {model}/{length}/B{B}: starling {sw}% vs stock {kw}%")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
