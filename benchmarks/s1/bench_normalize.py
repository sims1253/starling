"""S1-mini normalization benchmark: latency, throughput, accuracy.

The single source of truth for s1-mini numbers (the shared ``bench_all.py``
grid is audio-shaped; s1 is text-in/text-out, so it gets this dedicated
harness plus its own README sentinel block).

Three modes (combinable):

  default           latency/throughput grid: engine x tier (short/medium/long
                    transcripts), median +/- stdev of --reps after warmup.
                    Engines: starling (CUDA-graph pipeline), stock
                    transformers, starling-ggml (in-tree ggml engine).
  --accuracy        correctness: byte-exact parity of starling and
                    starling-ggml vs stock on every tier, the curated
                    QUALITY_CASES (expected outputs re-verified against the
                    shipped weights), and the full 16-combination control
                    matrix (styling x structure x context) vs stock.
  --sweep-k         replay-chunk-size sweep for the CUDA pipeline (tunes
                    NormalizePipeline._steps_for_prompt).
  --update-readme   splice the tables into README.md between the
                    <!-- BENCH:S1:START/END --> sentinels.

Run:
  uv run python benchmarks/s1/bench_normalize.py
  uv run python benchmarks/s1/bench_normalize.py --accuracy
  uv run python benchmarks/s1/bench_normalize.py --update-readme --accuracy

The whole run holds the GPU lock (one GPU job at a time).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from statistics import median, stdev

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "benchmarks"))

OUTPUTS = REPO_ROOT / "outputs"
README = REPO_ROOT / "README.md"
START, END = "<!-- BENCH:S1:START -->", "<!-- BENCH:S1:END -->"


def _load_transcript_module():
    spec = importlib.util.spec_from_file_location(
        "s1_transcripts", REPO_ROOT / "tests" / "fixtures" / "s1_transcripts.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


FX = _load_transcript_module()


# ---------------------------------------------------------------------- #
# engine wrappers (load once, close after; sequential — one GPU job)
# ---------------------------------------------------------------------- #
class StarlingPath:
    name = "starling"

    def __init__(self, steps_per_replay: int | None = None):
        self._k = steps_per_replay
        self.pipe = None

    def load(self):
        from starling.s1.pipeline import NormalizePipeline

        self.pipe = NormalizePipeline.from_pretrained(
            steps_per_replay=self._k)

    def close(self):
        self.pipe = None
        torch.cuda.empty_cache()

    def normalize(self, transcript: str, **controls) -> str:
        text, _ = self.pipe.normalize(transcript, **controls)
        return text


class StockPath:
    name = "stock transformers"

    def __init__(self):
        self.model = None

    def load(self):
        from starling.s1.config import (
            SYSTEM_PROMPT, control_line, max_new_tokens_for)
        from starling.s1.loader import load_model_and_tokenizer

        self._system = SYSTEM_PROMPT
        self._control_line = control_line
        self._budget = max_new_tokens_for
        self.model, self.tok = load_model_and_tokenizer(attn_impl="eager")

    def close(self):
        self.model = self.tok = None
        torch.cuda.empty_cache()

    def normalize(self, transcript: str, *, styling="semi-formal",
                  structure="prose", context="general") -> str:
        messages = [
            {"role": "system", "content": self._system},
            {"role": "user", "content":
                f"{self._control_line(styling, structure, context)}\n{transcript}"},
        ]
        text = self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
        inputs = self.tok(text, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self._budget(inputs.input_ids.shape[1]),
                do_sample=False)
        ids = out[0][inputs.input_ids.shape[1]:]
        return self.tok.decode(ids, skip_special_tokens=True)


class GgmlPath:
    name = "starling-ggml"

    def __init__(self):
        self._model = None

    def load(self):
        from starling._ggml import S1, GgmlModel

        self._model = GgmlModel(S1, str(
            REPO_ROOT / "models" / "s1-mini-bf16-exact.gguf"))

    def close(self):
        if self._model is not None:
            self._model.close()
            self._model = None
        torch.cuda.empty_cache()

    def normalize(self, transcript: str, **controls) -> str:
        return self._model.normalize_text(
            transcript,
            controls.get("styling"), controls.get("structure"),
            controls.get("context"))


def _time_ms(fn, *, warmup: int, reps: int) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return median(samples), (stdev(samples) if len(samples) > 1 else 0.0)


def _n_words(s: str) -> int:
    return len(s.split())


# ---------------------------------------------------------------------- #
# modes
# ---------------------------------------------------------------------- #
def run_latency(reps: int, warmup: int) -> dict:
    tiers = FX.LENGTH_TIERS
    records = []
    for cls in (StarlingPath, StockPath, GgmlPath):
        try:
            eng = cls()
            eng.load()
        except Exception as e:  # noqa: BLE001 -- gated engine, report + skip
            print(f"[s1-bench] {cls.name} unavailable: {e}")
            continue
        for tier, transcript in tiers.items():
            eng.normalize(transcript)  # capture/warm once
            ms, sd = _time_ms(
                lambda t=transcript: eng.normalize(t),
                warmup=warmup, reps=reps)
            rec = {
                "engine": cls.name, "tier": tier,
                "in_words": _n_words(transcript),
                "ms": round(ms, 1), "ms_stdev": round(sd, 1),
                "words_per_s": round(_n_words(transcript) / (ms / 1000.0), 1),
            }
            records.append(rec)
            print(f"[s1-bench] {cls.name:18s} {tier:6s} "
                  f"{ms:8.1f}±{sd:5.1f} ms  ({rec['words_per_s']:7.1f} words/s)")
        eng.close()
    return {"records": records}


def run_accuracy() -> dict:
    """Byte-exact parity + curated quality + the full control matrix."""
    stock = StockPath()
    stock.load()
    summary: dict = {"tiers": {}, "quality": [], "control_matrix": {}}

    # (1) tier parity: starling + ggml vs stock, exact match required.
    fast_paths: list = []
    for cls in (StarlingPath, GgmlPath):
        try:
            eng = cls()
            eng.load()
            fast_paths.append(eng)
        except Exception as e:  # noqa: BLE001
            print(f"[s1-acc] {cls.name} unavailable: {e}")
            summary["tiers"][cls.name] = {"error": str(e)}

    try:
        for tier, transcript in FX.LENGTH_TIERS.items():
            ref = stock.normalize(transcript)
            row = {"stock": ref}
            for eng in fast_paths:
                got = eng.normalize(transcript)
                row[eng.name] = got
                row[f"{eng.name}_exact"] = got == ref
            summary["tiers"][tier] = row
            print(f"[s1-acc] {tier:6s} "
                  + " ".join(f"{e.name}={'EXACT' if row[f'{e.name}_exact'] else 'DIFF'}"
                             for e in fast_paths))
    finally:
        for eng in fast_paths:
            eng.close()

    # (2) curated quality cases (expected outputs verified against the
    # shipped weights — provenance in the fixtures module).
    for transcript, styling, structure, context, expected in FX.QUALITY_CASES:
        got = stock.normalize(
            transcript, styling=styling, structure=structure, context=context)
        ok = got == expected
        summary["quality"].append({
            "styling": styling, "structure": structure, "context": context,
            "transcript": transcript, "expected": expected, "got": got,
            "ok": ok,
        })
        print(f"[s1-acc] quality {styling}/{structure}/{context}: "
              f"{'OK' if ok else 'DRIFT'}")

    # (3) control matrix: every trained combination, starling+ggml vs stock.
    n_ok = {"total": 0}
    fast_paths = []
    for cls in (StarlingPath, GgmlPath):
        try:
            eng = cls()
            eng.load()
            fast_paths.append(eng)
        except Exception:
            continue
    try:
        for transcript, styling, structure, context in FX.CONTROL_MATRIX:
            ref = stock.normalize(
                transcript, styling=styling, structure=structure, context=context)
            for eng in fast_paths:
                got = eng.normalize(
                    transcript, styling=styling, structure=structure, context=context)
                key = f"{eng.name}_exact"
                n_ok[key] = n_ok.get(key, 0) + (1 if got == ref else 0)
                n_ok["total"] += 1
                if got != ref:
                    # A tie-flip diff (see the gate in main): record its CER so
                    # punctuation-level flips are distinguishable from real
                    # divergence. The semi-casual/lists/general cell has an
                    # EXACT bf16 tie between ' so' and ',' at one step
                    # (verified top-2 f32 gap 0.0000 on stock eager), so the
                    # winner is argmax tie-break order, not numerics.
                    sys.path.insert(0, str(REPO_ROOT / "benchmarks"))
                    from wer import cer_pct  # noqa: E402

                    summary["control_matrix"].setdefault(key, []).append({
                        "styling": styling, "structure": structure,
                        "context": context, "stock": ref, "got": got,
                        "cer_pct": round(cer_pct(ref, got), 2),
                    })
    finally:
        for eng in fast_paths:
            eng.close()
    summary["control_matrix"]["counts"] = {
        k: v for k, v in n_ok.items() if k != "total"}
    summary["control_matrix"]["n_cases_per_engine"] = len(FX.CONTROL_MATRIX)
    stock.close()
    print(f"[s1-acc] control matrix (16 combos x {len(fast_paths)} engines): "
          + ", ".join(f"{k}={v}/{len(FX.CONTROL_MATRIX)}"
                      for k, v in summary["control_matrix"]["counts"].items()))
    return summary


def run_sweep_k() -> dict:
    from starling.s1.pipeline import NormalizePipeline

    pipe = NormalizePipeline.from_pretrained()
    out = []
    for K in (1, 2, 4, 8, 16, 32):
        for tier, transcript in FX.LENGTH_TIERS.items():
            llm = pipe._get_multistep_llm(K)
            ids = pipe.build_prompt_ids(transcript)
            T = int(ids.shape[1])
            emb = pipe.embed_tokens(ids)
            mnt = min(int(T * 1.3) + 32, 4096 - T + 1)
            llm.generate(emb, max_new_tokens=mnt)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            res = llm.generate(emb, max_new_tokens=mnt)
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1000.0
            out.append({"K": K, "tier": tier, "prompt_tok": T,
                        "gen_tok": res.n_tokens, "ms": round(ms, 1)})
            print(f"[s1-k] K={K:2d} {tier:6s} T={T:4d} {ms:8.1f} ms "
                  f"({res.n_tokens} tok)")
    pipe = None
    torch.cuda.empty_cache()
    return {"records": out}


# ---------------------------------------------------------------------- #
# markdown
# ---------------------------------------------------------------------- #
def build_markdown(results: dict) -> str:
    lines = ["**s1-mini** — normalization latency / throughput (ms, words/s)",
             "", "Text-in/text-out: fixture tiers are raw transcripts;",
             "words/s = input words normalized per second (higher is faster).",
             "bf16, model load + graph capture excluded, single RTX 5090.", ""]
    by_tier: dict[str, list[dict]] = {}
    for r in results["latency"]["records"]:
        by_tier.setdefault(r["tier"], []).append(r)
    header = ["tier", "engine", "ms", "words/s"]
    rows = []
    for tier in ("short", "medium", "long"):
        for r in by_tier.get(tier, []):
            rows.append([tier, r["engine"],
                         f"{r['ms']:.0f}±{r['ms_stdev']:.0f}ms",
                         f"{r['words_per_s']:.0f}"])
    if rows:
        from tabulate import tabulate
        lines.append(tabulate(rows, headers=header, tablefmt="github"))
    acc = results.get("accuracy")
    if acc:
        lines += ["", "**s1-mini** — accuracy (vs stock transformers greedy)", ""]
        tiers = acc.get("tiers", {})
        # engine columns present in the tier rows (a gated engine may be absent)
        cols = [k for k in ("starling_exact", "starling-ggml_exact")
                if any(k in tiers.get(t, {}) for t in ("short", "medium", "long"))]
        arows = []
        for tier in ("short", "medium", "long"):
            row = tiers.get(tier)
            if not row:
                continue
            arows.append([tier] + [("byte-exact" if row.get(k) else "DIFF")
                                   for k in cols])
        if arows:
            from tabulate import tabulate

            lines.append(tabulate(
                arows, headers=["tier"] + [k.removesuffix("_exact") for k in cols],
                tablefmt="github"))
        cm = acc.get("control_matrix", {}).get("counts", {})
        if cm:
            counts = ", ".join(f"{k}: {v}/16" for k, v in cm.items())
            lines += ["", f"Control matrix (4 styling x 2 structure x 2 context): {counts}"]
        nq = sum(1 for q in acc.get("quality", []) if q["ok"])
        lines.append(f"Curated quality cases: {nq}/{len(acc.get('quality', []))} expected outputs matched")
    return "\n".join(lines)


def splice_readme(md: str) -> bool:
    body = README.read_text()
    if START in body:
        pre, rest = body.split(START, 1)
        _, post = rest.split(END, 1)
        new = pre + START + "\n" + md + "\n" + END + post
    else:
        anchor = "<!-- BENCH:WER:END -->"
        if anchor not in body:
            return False
        pre, post = body.split(anchor, 1)
        new = (pre + anchor + "\n\n" + START + "\n" + md + "\n" + END + post)
    if new != body:
        README.write_text(new)
        return True
    return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="S1-mini normalization benchmark")
    ap.add_argument("--reps", default=5, type=int)
    ap.add_argument("--warmup", default=2, type=int)
    ap.add_argument("--accuracy", action="store_true",
                    help="run the parity + quality + control-matrix suite")
    ap.add_argument("--sweep-k", action="store_true",
                    help="sweep the CUDA pipeline K-step replay size")
    ap.add_argument("--update-readme", action="store_true")
    args = ap.parse_args(argv)

    results: dict = {}

    from starling.parakeet.gpu_lock import with_gpu_lock

    with with_gpu_lock(session="s1-bench", model="s1", eta_min=30,
                       note="s1 normalization benchmark"):
        results["latency"] = run_latency(args.reps, args.warmup)
        if args.accuracy:
            results["accuracy"] = run_accuracy()
        if args.sweep_k:
            results["sweep_k"] = run_sweep_k()

    OUTPUTS.mkdir(exist_ok=True)
    (OUTPUTS / "s1_normalize.json").write_text(json.dumps(results, indent=2))
    print(f"\n[s1-bench] wrote {OUTPUTS}/s1_normalize.json")

    md = build_markdown(results)
    (OUTPUTS / "s1_normalize.md").write_text(md)
    print("\n[s1-bench] markdown:\n")
    print(md)

    if args.update_readme:
        changed = splice_readme(md)
        print(f"\n[s1-bench] README {'updated' if changed else 'unchanged'}")

    ok = True
    if args.accuracy:
        acc = results["accuracy"]
        for tier, row in acc.get("tiers", {}).items():
            for k in ("starling_exact", "starling-ggml_exact"):
                if k in row and not row[k]:
                    ok = False  # canonical fixtures: byte-exact, no tolerance
        if any(not q["ok"] for q in acc.get("quality", [])):
            ok = False
        counts = acc.get("control_matrix", {}).get("counts", {})
        diffs = [d for v in acc.get("control_matrix", {}).values()
                 if isinstance(v, list) for d in v]
        # Control matrix gate: byte-exact except documented greedy tie-flips.
        # One cell (semi-casual/lists/general) hits an EXACT bf16 tie between
        # ' so' and ',' (stock-eager top-2 f32 gap 0.0000); the winner is the
        # implementation's argmax tie-break order. Accept at most one such
        # flip per engine, and only at punctuation CER (<= 2%).
        n_cases = len(FX.CONTROL_MATRIX)
        for k, v in counts.items():
            if v < n_cases - 1 or any(d["cer_pct"] > 2.0 for d in diffs):
                ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
