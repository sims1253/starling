#!/usr/bin/env python3
"""Benchmark: speculative vs non-speculative vs stock transformers.

Produces a comparison table for Granite-Speech-4.1-2b ASR on the sample audio:

    | path           | full (ms) | decode tok/s | RTFx  | speedup vs stock |

The speculative path also reports draft token count and acceptance rate.

Usage:
    .venv/bin/python scripts/bench_speculative.py
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from megapar.audio import build_inputs, load_sample_audio  # noqa: E402
from megapar.config import LLM_EOS_TOKEN_ID  # noqa: E402
from megapar.loader import get_components, load_model_and_processor  # noqa: E402
from megapar.pipeline import MegaPipeline  # noqa: E402
from megapar.speculative import CTCBPEDraft, SpeculativeDecoder, load_out_llm  # noqa: E402


def wall_time_ms(fn, *, warmup: int = 2, iters: int = 5) -> float:
    """Median wall-clock time (ms) for ``fn``."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times)


def main() -> int:
    print("=" * 78)
    print(" megapar speculative decoding benchmark")
    print("=" * 78)

    print("\n[load] model + processor ...")
    model, processor = load_model_and_processor(attn_impl="eager")
    pipe = MegaPipeline(model, processor, encoder_mode="cudagraph", use_fused_llm=True)

    wav, sr = load_sample_audio()
    inputs = build_inputs(processor, wav)
    audio_seconds = wav.shape[1] / sr
    feats = inputs["input_features"]
    input_ids = inputs["input_ids"]
    ifm = inputs.get("input_features_mask")
    n_gen = 100
    print(f"[load] audio = {audio_seconds:.2f}s, prompt = {input_ids.shape[1]} tokens")

    with torch.inference_mode():
        # ---------------------------------------------------------------- #
        # Non-speculative mega pipeline
        # ---------------------------------------------------------------- #
        print("\n[bench] non-speculative mega pipeline ...")
        nonspec_ms = wall_time_ms(
            lambda: pipe.transcribe(feats, input_ids, ifm, max_new_tokens=n_gen),
            warmup=2, iters=5,
        )
        nonspec_rtfx = audio_seconds / (nonspec_ms / 1000.0)
        _, nonspec_ids = pipe.transcribe(feats, input_ids, ifm, max_new_tokens=n_gen)
        nonspec_tps = n_gen / (nonspec_ms / 1000.0)
        print(f"  full={nonspec_ms:.1f}ms  RTFx={nonspec_rtfx:.2f}x  "
              f"effective={nonspec_tps:.1f} tok/s")

        # ---------------------------------------------------------------- #
        # Speculative mega pipeline
        # ---------------------------------------------------------------- #
        print("\n[bench] speculative mega pipeline ...")
        # Warmup the verify graphs (first call captures them).
        _ = pipe.transcribe(feats, input_ids, ifm, max_new_tokens=n_gen, speculative=True)

        spec_ms = wall_time_ms(
            lambda: pipe.transcribe(feats, input_ids, ifm, max_new_tokens=n_gen, speculative=True),
            warmup=2, iters=5,
        )
        spec_rtfx = audio_seconds / (spec_ms / 1000.0)

        # Get acceptance stats via direct call.
        out_llm = load_out_llm(device="cuda", dtype=torch.bfloat16)
        ctc = CTCBPEDraft(pipe.fused_encoder, out_llm)
        mid_h, enc_hidden = ctc.encode_with_mid(feats)
        draft = ctc.draft(enc_hidden, mid_h)
        audio_embeds = pipe.projector(enc_hidden)
        ie = pipe.build_inputs_embeds(input_ids, audio_embeds, ifm)
        sd = pipe._spec_decoder
        res = sd.generate(ie, draft, max_new_tokens=n_gen, eos_token_id=LLM_EOS_TOKEN_ID)
        spec_text, spec_ids = pipe.transcribe(
            feats, input_ids, ifm, max_new_tokens=n_gen, speculative=True
        )
        spec_tps = n_gen / (spec_ms / 1000.0)
        print(f"  full={spec_ms:.1f}ms  RTFx={spec_rtfx:.2f}x  "
              f"effective={spec_tps:.1f} tok/s")
        print(f"  draft={res.draft_count} tokens  accepted={res.accepted} "
              f"({res.acceptance_rate:.1%})  verify_forwards={res.verify_forwards}")

        # ---------------------------------------------------------------- #
        # Stock transformers baseline (from comms.md: 6594ms)
        # ---------------------------------------------------------------- #
        print("\n[bench] stock transformers (estimated from prior run) ...")
        stock_ms = 6594.0  # from bench_pipeline.py / comms.md
        stock_rtfx = audio_seconds / (stock_ms / 1000.0)
        print(f"  full={stock_ms:.0f}ms  RTFx={stock_rtfx:.2f}x")

        # ---------------------------------------------------------------- #
        # Correctness check
        # ---------------------------------------------------------------- #
        from megapar.golden import load_golden, load_golden_text
        golden_gen = load_golden("greedy_ids.pt")[0, 271:]
        golden_text = load_golden_text().strip().split("ASSISTANT:", 1)[1].strip()
        tok_match = (spec_ids[0] == nonspec_ids[0]).all().item()
        text_match = spec_text.strip() == golden_text
        print(f"\n[verify] speculative == non-speculative: {'YES' if tok_match else 'NO'}")
        print(f"[verify] speculative == golden text: {'YES' if text_match else 'NO'}")

        # ---------------------------------------------------------------- #
        # Comparison table
        # ---------------------------------------------------------------- #
        spec_vs_nonspec = nonspec_ms / spec_ms
        spec_vs_stock = stock_ms / spec_ms

        print("\n" + "=" * 78)
        print(f" {'path':<28}{'full (ms)':>12}{'tok/s':>10}{'RTFx':>8}{'vs stock':>10}")
        print("-" * 78)
        print(f" {'stock transformers':<28}{stock_ms:>12.0f}{n_gen/(stock_ms/1000):>9.1f} {stock_rtfx:>7.2f}x{'1.00x':>10}")
        print(f" {'mega (non-speculative)':<28}{nonspec_ms:>12.1f}{nonspec_tps:>9.1f} {nonspec_rtfx:>7.2f}x"
              f"{stock_ms/nonspec_ms:>9.2f}x")
        print(f" {'mega (speculative)':<28}{spec_ms:>12.1f}{spec_tps:>9.1f} {spec_rtfx:>7.2f}x"
              f"{spec_vs_stock:>9.2f}x")
        print("-" * 78)
        print(f" {'speedup spec vs non-spec':<28}{spec_vs_nonspec:>11.2f}x")
        print(f" {'acceptance rate':<28}{res.acceptance_rate:>11.1%}")
        print(f" {'draft tokens':<28}{res.draft_count:>12}")
        print(f" {'verify forwards':<28}{res.verify_forwards:>12}")
        print("=" * 78)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
