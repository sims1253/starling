"""Benchmark utilities and entry point for Granite-Speech-4.1-2b staging.

This module provides:

* :func:`cuda_timer` — robust GPU timer using ``torch.cuda.Event``.
* :class:`Benchmark` — accumulates {name, ms, notes} results, prints a table,
  and serialises to JSON.
* :func:`profile` — chrome-trace exporter via ``torch.profiler``.
* ``__main__`` block — captures golden if missing, then times each pipeline
  stage on (a) the eager GOLDEN model and (b) the "stock-optimized" model
  where the LLM uses SDPA. Prints a comparison table.

Run: ``python -m starling.bench``.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import torch

from .audio import build_inputs, load_sample_audio
from .config import GOLDEN_DIR, MODEL_ID, TRACES_DIR
from .golden import capture_golden
from .loader import get_components, load_model_and_processor, set_llm_attn_implementation


# ---------------------------------------------------------------------------
# cuda_timer
# ---------------------------------------------------------------------------
def cuda_timer(
    fn: Callable[[], Any],
    warmup: int = 3,
    iters: int = 20,
) -> float:
    """Time ``fn`` on the GPU and return the median time in milliseconds.

    Each call is bracketed by ``torch.cuda.synchronize`` and timed with a pair
    of CUDA events so we measure real GPU work, not Python overhead. The whole
    loop runs under ``torch.inference_mode``.
    """
    torch.cuda.synchronize()
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        times_ms: list[float] = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            times_ms.append(start.elapsed_time(end))
    return statistics.median(times_ms)


# ---------------------------------------------------------------------------
# Benchmark result container
# ---------------------------------------------------------------------------
@dataclass
class BenchResult:
    name: str
    ms: float
    notes: str = ""


@dataclass
class Benchmark:
    """Accumulates :class:`BenchResult` rows, prints tables, saves JSON."""

    results: list[BenchResult] = field(default_factory=list)

    def add(self, name: str, ms: float, notes: str = "") -> None:
        self.results.append(BenchResult(name=name, ms=float(ms), notes=notes))

    def print(self, title: Optional[str] = None) -> None:
        try:
            from tabulate import tabulate
        except Exception:  # noqa: BLE001
            tabulate = None  # type: ignore

        rows = [(r.name, f"{r.ms:.4f}", r.notes) for r in self.results]
        headers = ["stage", "ms", "notes"]
        if title:
            print(f"\n=== {title} ===")
        if tabulate is not None:
            print(tabulate(rows, headers=headers, tablefmt="github"))
        else:
            print(" | ".join(headers))
            for r in self.results:
                print(f"{r.name} | {r.ms:.4f} | {r.notes}")

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "results": [asdict(r) for r in self.results],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[bench] saved -> {path}")


# ---------------------------------------------------------------------------
# torch.profiler wrapper
# ---------------------------------------------------------------------------
def profile(fn: Callable[[], Any], trace_path: str | Path) -> Any:
    """Run ``fn`` under ``torch.profiler`` and export a chrome trace.

    The trace can be loaded into ``chrome://tracing`` or Perfetto.
    """
    trace_path = Path(trace_path)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    torch.cuda.synchronize()
    with torch.inference_mode():
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
        ) as prof:
            # Warmup + a few timed iters inside the profile window.
            for _ in range(3):
                fn()
            torch.cuda.synchronize()
    prof.export_chrome_trace(str(trace_path))
    print(f"[bench] chrome trace -> {trace_path}")
    return prof


# ---------------------------------------------------------------------------
# Stage timing helpers
# ---------------------------------------------------------------------------
def _time_stages(model: Any, inputs: dict[str, torch.Tensor], bench: Benchmark, tag: str) -> None:
    """Time the five pipeline stages on the given model and append to ``bench``."""
    components = get_components(model)
    encoder = components["encoder"]
    projector = components["projector"]
    dtype = model.dtype

    input_ids = inputs["input_ids"]
    input_features = inputs["input_features"]
    attention_mask = inputs["attention_mask"]
    input_features_mask = inputs.get("input_features_mask")

    feats_bf = input_features.to(dtype)

    # Prime audio outputs for the standalone encoder/projector stages.
    with torch.inference_mode():
        enc_lhs = encoder(feats_bf, return_dict=True).last_hidden_state
    with torch.inference_mode():
        audio_embeds = model.get_audio_features(feats_bf, return_dict=True).pooler_output

    # (a) encoder forward only
    def _enc():
        encoder(feats_bf, return_dict=True)
    ms_enc = cuda_timer(_enc)
    bench.add(f"{tag}:encoder", ms_enc, "encoder(input_features)")

    # (b) projector forward only
    def _proj():
        projector(enc_lhs)
    ms_proj = cuda_timer(_proj)
    bench.add(f"{tag}:projector", ms_proj, "projector(encoder_lhs)")

    # (c) get_audio_features (encoder + projector)
    def _gaf():
        model.get_audio_features(feats_bf, return_dict=True)
    ms_gaf = cuda_timer(_gaf)
    bench.add(f"{tag}:get_audio_features", ms_gaf, "encoder + projector")

    # (d) LLM prefill forward (single forward, use_cache=True)
    def _llm_prefill():
        model(
            input_ids=input_ids,
            input_features=input_features,
            attention_mask=attention_mask,
            input_features_mask=input_features_mask,
            use_cache=True,
            logits_to_keep=1,
        )
    ms_llm = cuda_timer(_llm_prefill, warmup=2, iters=10)
    bench.add(f"{tag}:llm_prefill", ms_llm, "single forward, use_cache=True, logits_to_keep=1")

    # (e) full generate greedy end-to-end + tokens/s
    n_prompt = int(input_ids.shape[1])
    max_new_tokens = 200

    # Warmup once outside timer.
    with torch.inference_mode():
        _ = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
        )
        torch.cuda.synchronize()
        # Timed run (wall clock for tokens/s).
        t0 = time.perf_counter()
        gen = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
        )
        torch.cuda.synchronize()
        t1 = time.perf_counter()
    n_new = int(gen.shape[1]) - n_prompt
    wall_ms = (t1 - t0) * 1000.0
    tok_per_s = n_new / max(t1 - t0, 1e-9)
    bench.add(
        f"{tag}:generate_e2e",
        wall_ms,
        f"greedy max_new_tokens={max_new_tokens}; produced {n_new} new tokens @ {tok_per_s:.1f} tok/s",
    )


def _resolve_llm_attn_impl(model: Any) -> str:
    components = get_components(model)
    cfg = getattr(components["language_model"], "config", None)
    if cfg is None:
        return "<unknown>"
    return getattr(cfg, "_attn_implementation", "<unknown>")


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------
def main() -> int:
    # Make sure golden artefacts exist (uses eager model). The golden capture
    # also doubles as the correctness check on the sample transcript.
    if not GOLDEN_DIR.exists() or not any(GOLDEN_DIR.iterdir()):
        print("[bench] golden missing; capturing ...")
        capture_golden()

    print("[bench] loading sample audio ...")
    wav, sr = load_sample_audio()

    bench = Benchmark()

    # ----- (1) eager baseline (GOLDEN correctness path) -----
    print(f"[bench] loading EAGER model from {MODEL_ID} ...")
    model_eager, processor_eager = load_model_and_processor(attn_impl="eager")
    inputs_eager = build_inputs(processor_eager, wav)
    print(f"[bench] eager LLM attn_impl = {_resolve_llm_attn_impl(model_eager)}")
    _time_stages(model_eager, inputs_eager, bench, tag="eager")

    # Free eager model before loading the next one to keep peak VRAM bounded.
    del model_eager
    torch.cuda.empty_cache()

    # ----- (2) stock-optimized (LLM on SDPA, projector stays eager) -----
    print(f"[bench] loading STOCK-OPTIMIZED model from {MODEL_ID} ...")
    model_opt, processor_opt = load_model_and_processor(attn_impl="eager")
    set_llm_attn_implementation(model_opt, "sdpa")
    inputs_opt = build_inputs(processor_opt, wav)
    resolved = _resolve_llm_attn_impl(model_opt)
    print(f"[bench] stock-optimized LLM attn_impl = {resolved}")

    try:
        _time_stages(model_opt, inputs_opt, bench, tag="stock-optimized")
    except Exception as exc:  # noqa: BLE001
        # If SDPA on the LLM errors, fall back and note it.
        print(f"[bench] SDPA LLM forward failed ({exc!r}); falling back to eager for stock-optimized")
        set_llm_attn_implementation(model_opt, "eager")
        resolved = _resolve_llm_attn_impl(model_opt)
        print(f"[bench] stock-optimized LLM attn_impl rolled back to {resolved}")
        _time_stages(model_opt, inputs_opt, bench, tag="stock-optimized(sdma-fail-eager)")

    # Persist results.
    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    bench.save(TRACES_DIR / "bench_results.json")
    bench.print(title="starling stage benchmarks (median ms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
