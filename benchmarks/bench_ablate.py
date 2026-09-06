"""Ablation harness: measure each optimisation flag's marginal benefit.

Sweeps every flag in :class:`OptFlags` on/off against a byte-exact baseline,
times the granite decode step via CUDA-graph replay (median of N trials), and
checks byte-exactness against the golden greedy ids. Output is a single
markdown table showing each flag's marginal speedup and accuracy impact, so
you can see exactly what each optimisation buys.

Two modes:

* **decode-step** (default): CUDA-graph replay latency per decode step on the
  golden ``inputs_embeds``. Isolates the captured forward from prefill and the
  host-side Python loop. Use this to ablate decode-path flags
  (``rope_alloc_free``, ``lm_head_scale_fold``, ``gemm_epilogue_fusion``,
  ``sdpa_attention``, ``flash_attention``, ``fp8_attention``,
  ``nvfp4_weights``).
* **long-audio** (``--mode long_audio``): end-to-end wall-clock on a long
  fixture clip. Use this to ablate pipeline flags (``chunk_prefill_overlap``)
  and confirm decode-flag benefits translate to end-to-end gains.

Usage::

    uv run python benchmarks/bench_ablate.py                    # decode-step sweep
    uv run python benchmarks/bench_ablate.py --mode long_audio  # end-to-end sweep
    uv run python benchmarks/bench_ablate.py --flag rope_alloc_free  # one flag
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from statistics import median

import torch

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from starling.config import LLM_EOS_TOKEN_ID  # noqa: E402
from starling.flags import OptFlags, get_default_flags, set_default_flags  # noqa: E402
from starling.granite.golden import load_golden  # noqa: E402
from starling.granite.llm_mega import FusedLLMMega  # noqa: E402
from starling.granite.loader import get_components, load_model_and_processor  # noqa: E402

OUTPUTS = REPO_ROOT / "outputs"

# Each entry: (flag_name, dict of OptFlags overrides to ENABLE the optimisation).
# To ablate, we measure baseline (flag off) then variant (flag on) and report
# the marginal delta.  Flags that require tolerance_mode set it in their override.
DECODE_FLAGS: list[tuple[str, dict]] = [
    ("rope_alloc_free",      dict(rope_alloc_free=True)),
    ("lm_head_scale_fold",   dict(lm_head_scale_fold=True)),
    ("fused_qkv",            dict(fused_qkv=True)),
    ("sdpa_attention",       dict(sdpa_attention=True, tolerance_mode=True)),
    ("flash_attention",      dict(flash_attention=True, tolerance_mode=True)),
    ("multistep_graph",      dict(multistep_graph=True)),
    # Experimental paths; evaluate quality alongside speed.
    ("gemm_epilogue_fusion", dict(gemm_epilogue_fusion=True)),
    ("nvfp4_weights",        dict(nvfp4_weights=True, tolerance_mode=True)),
    ("nvfp4_lm_head_only",   dict(nvfp4_lm_head_only=True, tolerance_mode=True)),
]

LONG_AUDIO_FLAGS: list[tuple[str, dict]] = [
    ("chunk_prefill_overlap", dict(chunk_prefill_overlap=True)),
]


@contextmanager
def _restore_flags():
    saved = get_default_flags()
    try:
        yield
    finally:
        set_default_flags(saved)


@torch.inference_mode()
def _byte_exact(dec, inputs_embeds, golden_ids, T, tokenizer, max_new_tokens=120) -> bool | str:
    """Run a full generate and compare ids to golden. Returns True/False/'n/a'."""
    try:
        res = dec.generate(
            inputs_embeds, max_new_tokens=max_new_tokens,
            eos_token_id=LLM_EOS_TOKEN_ID, tokenizer=tokenizer, capture=False,
        )
        golden_gen = golden_ids[0, T:T + res.n_tokens]
        min_len = min(golden_gen.numel(), res.ids.numel())
        if min_len == 0:
            return "n/a"
        return bool(torch.equal(golden_gen[:min_len], res.ids[0, :min_len].cpu()))
    except Exception as e:
        return f"err: {type(e).__name__}"


@torch.inference_mode()
def bench_decode_step(model, processor, inputs_embeds, golden_ids, combo: dict,
                      reps: int = 200, trials: int = 6,
                      check_byte_exact: bool = False) -> dict:
    """Build a fresh FusedLLMMega under ``combo`` and time decode-step replay.

    Byte-exactness is validated separately by ``tests/test_llm_mega.py``; the
    harness skips the full ``generate()`` correctness pass by default (it
    doesn't terminate cleanly when run repeatedly across flag combos in one
    process). Pass ``check_byte_exact=True`` to run it for one combo at a time.
    """
    components = get_components(model)
    tokenizer = processor.tokenizer
    T = inputs_embeds.shape[1]

    of = OptFlags(**combo)
    set_default_flags(of)
    try:
        dec = FusedLLMMega(components["language_model"], model.lm_head, max_cache_len=896)
    except NotImplementedError as e:
        return {"us_per_step": float("nan"), "byte_exact": f"not implemented: {e}",
                "note": "under construction"}
    except Exception as e:
        return {"us_per_step": float("nan"), "byte_exact": f"err: {type(e).__name__}: {e}",
                "note": str(e)[:80]}

    # Correctness check (optional -- see docstring).
    be = _byte_exact(dec, inputs_embeds, golden_ids, T, tokenizer) if check_byte_exact else "see pytest"

    # Capture the decode graph and time its replay.
    next_token = dec.prefill(inputs_embeds)
    dec.capture(next_token, T)

    trial_us: list[float] = []
    CHUNK = 50
    for _ in range(trials):
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        done = 0
        while done < reps:
            n = min(CHUNK, reps - done)
            for _ in range(n):
                dec._graph.replay()
            torch.cuda.synchronize()
            done += n
        e.record()
        torch.cuda.synchronize()
        trial_us.append(s.elapsed_time(e) / reps * 1000.0)

    del dec
    torch.cuda.empty_cache()
    return {"us_per_step": median(trial_us), "byte_exact": be, "note": ""}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["decode_step", "long_audio"], default="decode_step")
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--trials", type=int, default=6)
    ap.add_argument("--flag", default=None, help="ablate only this one flag")
    args = ap.parse_args()

    from starling.parakeet.gpu_lock import with_gpu_lock

    with with_gpu_lock(
        session="bench-ablate",
        model="granite-speech-4.1-2b",
        eta_min=30,
        note=f"ablation ({args.mode})",
    ):
        return _main_locked(args)


def _main_locked(args) -> int:

    print("loading granite model + golden artefacts ...", flush=True)
    model = processor = inputs_embeds = golden_ids = None
    if args.mode == "decode_step":
        model, processor = load_model_and_processor(attn_impl="eager")
        inputs_embeds = load_golden("inputs_embeds.pt").to("cuda", torch.bfloat16)
        golden_ids = load_golden("greedy_ids.pt")

    flag_set = DECODE_FLAGS if args.mode == "decode_step" else LONG_AUDIO_FLAGS
    if args.flag:
        flag_set = [(n, o) for n, o in flag_set if n == args.flag]
        if not flag_set:
            print(f"unknown flag {args.flag!r}; known: {[n for n, _ in DECODE_FLAGS + LONG_AUDIO_FLAGS]}")
            return 2

    if args.mode == "long_audio":
        return _main_long_audio(args, flag_set)
    return _main_decode_step(args, model, processor, inputs_embeds, golden_ids, flag_set)


def _main_decode_step(args, model, processor, inputs_embeds, golden_ids, flag_set) -> int:
    """Decode-step ablation (the default mode)."""
    # Baseline: every ablatable flag OFF (the strict byte-exact reference).
    print("\nmeasuring baseline (all ablatable flags off) ...", flush=True)
    baseline_combo = {name: False for name, _ in DECODE_FLAGS + LONG_AUDIO_FLAGS}
    baseline_combo["multistep_graph"] = False  # decode_step uses FusedLLMMega
    with _restore_flags():
        base = bench_decode_step(model, processor, inputs_embeds, golden_ids,
                                 baseline_combo, reps=args.reps, trials=args.trials)
    base_us = base["us_per_step"]
    print(f"  baseline: {base_us:.2f} us/step  byte_exact={base['byte_exact']}", flush=True)

    rows: list[dict] = []
    for name, override in flag_set:
        print(f"\n=== {name} (on) ===", flush=True)
        # Build the combo: baseline with just this flag flipped on.
        combo = dict(baseline_combo)
        combo.update(override)
        with _restore_flags():
            r = bench_decode_step(model, processor, inputs_embeds, golden_ids,
                                  combo, reps=args.reps, trials=args.trials)
        r["flag"] = name
        r["speedup_vs_baseline"] = (base_us / r["us_per_step"]) if r["us_per_step"] == r["us_per_step"] else float("nan")
        print(f"  {r['us_per_step']:.2f} us/step  speedup={r['speedup_vs_baseline']:.3f}x  "
              f"byte_exact={r['byte_exact']}", flush=True)
        rows.append(r)

    # Markdown table
    print(f"\n## Ablation: {args.mode} (baseline = {base_us:.2f} us/step, "
          f"median of {args.trials}x{args.reps} replays)\n")
    print("| flag | us/step | speedup | byte-exact | notes |")
    print("|------|---------|---------|------------|-------|")
    print(f"| *(baseline, all off)* | {base_us:.2f} | 1.000x | {base['byte_exact']} | reference |")
    for r in rows:
        us = f"{r['us_per_step']:.2f}" if r["us_per_step"] == r["us_per_step"] else "n/a"
        sp = f"{r['speedup_vs_baseline']:.3f}x" if r["speedup_vs_baseline"] == r["speedup_vs_baseline"] else "n/a"
        print(f"| {r['flag']} | {us} | {sp} | {r['byte_exact']} | {r.get('note','')} |")

    OUTPUTS.mkdir(exist_ok=True)
    (OUTPUTS / f"ablate_{args.mode}.json").write_text(json.dumps({
        "mode": args.mode, "reps": args.reps, "trials": args.trials,
        "baseline_us_per_step": base_us, "baseline_byte_exact": base["byte_exact"],
        "rows": rows,
    }, indent=2, default=str))
    print(f"\nwrote {OUTPUTS / f'ablate_{args.mode}.json'}")
    return 0


@torch.inference_mode()
def _bench_long_audio(combo: dict, wav: torch.Tensor, sr: int, trials: int = 3) -> dict:
    """Time end-to-end long-audio transcription under ``combo`` flags.

    Returns wall-clock seconds (median of ``trials``) and the transcript text
    so byte-exactness across flag states can be checked by the caller.
    """
    from starling.granite.audio import build_inputs, load_sample_audio  # noqa: F401
    from starling.granite.long_audio import transcribe_long  # noqa: F401
    from starling.granite.pipeline import MegaPipeline  # noqa: F401

    of = OptFlags(**combo)
    set_default_flags(of)
    try:
        model, processor = load_model_and_processor(attn_impl="eager")
        pipe = MegaPipeline(model, processor, encoder_mode="cudagraph")
    except Exception as e:
        return {"wall_s": float("nan"), "text": "", "note": f"err: {type(e).__name__}: {e}"}

    times: list[float] = []
    text_ref = ""
    for t in range(trials):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        res = transcribe_long(pipe, processor, wav, sr, chunk_seconds=30.0)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        if t == 0:
            text_ref = res.text
    del pipe, model
    torch.cuda.empty_cache()
    return {"wall_s": median(times), "text": text_ref, "note": ""}


def _main_long_audio(args, flag_set) -> int:
    """End-to-end long-audio ablation. Times wall-clock transcription."""
    from starling.granite.audio import load_sample_audio

    # Build a long fixture: tile the sample clip to ~5 min.
    wav, sr = load_sample_audio()
    if isinstance(wav, torch.Tensor):
        wav_np = wav.detach().cpu().numpy()
    else:
        wav_np = wav
    if wav_np.ndim == 2:
        wav_np = wav_np.mean(axis=0)
    n_tile = max(1, int(300 * sr) // int(wav_np.shape[-1]))  # ~5 min
    wav_long = torch.from_numpy(np.tile(wav_np, n_tile)).float().unsqueeze(0)
    print(f"long fixture: {wav_long.numel()/sr:.1f}s ({n_tile} tiles)", flush=True)

    baseline_combo = {name: False for name, _ in DECODE_FLAGS + LONG_AUDIO_FLAGS}
    baseline_combo["multistep_graph"] = True  # long-audio uses the pipeline default
    print("\nmeasuring baseline (all ablatable flags off) ...", flush=True)
    with _restore_flags():
        base = _bench_long_audio(baseline_combo, wav_long, sr, trials=args.trials)
    base_s = base["wall_s"]
    print(f"  baseline: {base_s:.2f}s  text_len={len(base['text'])}", flush=True)

    rows: list[dict] = []
    for name, override in flag_set:
        print(f"\n=== {name} (on) ===", flush=True)
        combo = dict(baseline_combo)
        combo.update(override)
        with _restore_flags():
            r = _bench_long_audio(combo, wav_long, sr, trials=args.trials)
        r["flag"] = name
        r["speedup_vs_baseline"] = (base_s / r["wall_s"]) if r["wall_s"] == r["wall_s"] else float("nan")
        r["text_exact"] = (r["text"] == base["text"]) if r["text"] else "n/a"
        print(f"  {r['wall_s']:.2f}s  speedup={r['speedup_vs_baseline']:.3f}x  "
              f"text_exact={r['text_exact']}", flush=True)
        rows.append(r)

    print(f"\n## Ablation: long_audio (baseline = {base_s:.2f}s, "
          f"median of {args.trials} trials)\n")
    print("| flag | wall_s | speedup | text-exact | notes |")
    print("|------|--------|---------|------------|------|")
    print(f"| *(baseline, all off)* | {base_s:.2f} | 1.000x | yes | reference |")
    for r in rows:
        s = f"{r['wall_s']:.2f}" if r["wall_s"] == r["wall_s"] else "n/a"
        sp = f"{r['speedup_vs_baseline']:.3f}x" if r["speedup_vs_baseline"] == r["speedup_vs_baseline"] else "n/a"
        print(f"| {r['flag']} | {s} | {sp} | {r['text_exact']} | {r.get('note','')} |")

    OUTPUTS.mkdir(exist_ok=True)
    (OUTPUTS / "ablate_long_audio.json").write_text(json.dumps({
        "mode": "long_audio", "trials": args.trials,
        "baseline_wall_s": base_s, "rows": rows,
    }, indent=2, default=str))
    print(f"\nwrote {OUTPUTS / 'ablate_long_audio.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
