#!/usr/bin/env python3
"""Benchmark: v2 pure-verify speculative decoding vs non-spec vs stock.

Produces a comparison table for Granite-Speech-4.1-2b ASR on the sample audio
plus the v2 counters (acceptance, verify_forwards, decode_steps, decode_probes).

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

from starling.audio import build_inputs, load_sample_audio  # noqa: E402
from starling.config import LLM_EOS_TOKEN_ID  # noqa: E402
from starling.loader import load_model_and_processor  # noqa: E402
from starling.parakeet.gpu_lock import with_gpu_lock  # noqa: E402
from starling.pipeline import MegaPipeline  # noqa: E402
from starling.speculative import CTCBPEDraft, load_out_llm  # noqa: E402

# Reference numbers from comms.md (uncontended, byte-exact, 24.9s audio):
V1_TOK_S = 269.0       # v1 spec mega
V1_MS = 372.0
NONSPEC_TOK_S = 182.0  # non-spec mega
NONSPEC_MS = 549.0


def wall_ms(fn, *, warm: int = 3, iters: int = 8):
    """Median + min wall-clock ms for fn (with cuda sync)."""
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(ts), min(ts)


def main() -> int:
    with with_gpu_lock(
        session="granite-mega",
        model="granite-speech-4.1-2b",
        eta_min=10,
        note="v2 pure-verify speculative benchmark",
    ):
        print("loading model + processor ...", flush=True)
        model, proc = load_model_and_processor("eager")
        wav, sr = load_sample_audio()
        inputs = build_inputs(proc, wav)
        feats = inputs["input_features"].bfloat16()
        ids = inputs["input_ids"]
        mask = inputs.get("input_features_mask")
        dur = wav.shape[1] / sr
        n_tok = 100
        pipe = MegaPipeline(model, proc, encoder_mode="cudagraph", use_fused_llm=True)
        print(f"audio {dur:.1f}s, prompt {ids.shape[1]} tokens, gen {n_tok}\n", flush=True)

        with torch.inference_mode():
            # ---- stock transformers ----
            def stock():
                model.generate(
                    input_ids=ids, input_features=feats,
                    attention_mask=inputs["attention_mask"],
                    input_features_mask=mask, max_new_tokens=n_tok,
                    do_sample=False, num_beams=1,
                )
            smed, smin = wall_ms(stock, warm=2, iters=4)

            # ---- mega non-spec ----
            def nonspec():
                pipe.transcribe(feats, ids, mask, max_new_tokens=n_tok, speculative=False)
            nmed, nmin = wall_ms(nonspec)

            # ---- mega spec v2 (warmup captures verify graphs first) ----
            _ = pipe.transcribe(feats, ids, mask, max_new_tokens=n_tok, speculative=True)
            def spec():
                pipe.transcribe(feats, ids, mask, max_new_tokens=n_tok, speculative=True)
            emed, emin = wall_ms(spec)

            # ---- v2 counters via direct decoder call ----
            out_llm = load_out_llm(device="cuda", dtype=torch.bfloat16)
            ctc = CTCBPEDraft(pipe.fused_encoder, out_llm)
            mid_h, enc_hidden = ctc.encode_with_mid(feats)
            draft = ctc.draft(enc_hidden, mid_h)
            audio_embeds = pipe.projector(enc_hidden)
            ie = pipe.build_inputs_embeds(ids, audio_embeds, mask)
            _ctc, sd = pipe._get_spec_components()
            res = sd.generate(ie, draft, max_new_tokens=n_tok, eos_token_id=LLM_EOS_TOKEN_ID)
            spec_text, spec_ids = pipe.transcribe(
                feats, ids, mask, max_new_tokens=n_tok, speculative=True
            )

            # ---- correctness ----
            from starling.golden import load_golden, load_golden_text
            golden_gen = load_golden("greedy_ids.pt")[0, 271:]
            golden_text = load_golden_text().strip().split("ASSISTANT:", 1)[1].strip()
            nn = min(spec_ids.shape[1], golden_gen.shape[0])
            exact = bool((spec_ids[0, :nn] == golden_gen[:nn]).all().item())
            text_exact = spec_text.strip() == golden_text

        # ---- comparison table ----
        hdr = f"{'path':<26}{'median ms':>11}{'min ms':>10}{'tok/s':>9}{'RTFx':>8}{'vs stock':>10}"
        print("=" * 84)
        print(hdr)
        print("-" * 84)
        print(f"{'stock transformers':<26}{smed:>11.1f}{smin:>10.1f}"
              f"{n_tok/(smed/1000):>9.1f}{dur/(smed/1000):>8.2f}x{'1.00x':>10}")
        print(f"{'mega (non-spec)':<26}{nmed:>11.1f}{nmin:>10.1f}"
              f"{n_tok/(nmed/1000):>9.1f}{dur/(nmed/1000):>8.2f}x{smed/nmed:>9.2f}x")
        print(f"{'mega (spec v2 pure-verify)':<26}{emed:>11.1f}{emin:>10.1f}"
              f"{n_tok/(emed/1000):>9.1f}{dur/(emed/1000):>8.2f}x{smed/emed:>9.2f}x")
        print("-" * 84)
        print(f"spec v2 vs non-spec : {nmed/emed:.2f}x   spec v2 vs v1 : {V1_MS/emed:.2f}x "
              f"({V1_TOK_S:.0f} -> {n_tok/(emed/1000):.0f} tok/s)")
        print(f"spec v2 vs stock     : {smed/emed:.2f}x")
        print()
        print("v2 counters (single direct decode call):")
        print(f"  byte-exact vs golden : {'YES' if exact else 'NO'}  (text: {'YES' if text_exact else 'NO'})")
        print(f"  acceptance_rate      : {res.acceptance_rate:.2%}  "
              f"(accepted={res.accepted}/{res.draft_count})")
        print(f"  verify_forwards      : {res.verify_forwards}")
        print(f"  decode_steps         : {res.decode_steps}  (fallback after draft)")
        print(f"  decode_probes        : {res.decode_probes}  (MUST be 0: pure verify)")
        print(f"  total forwards       : {res.verify_forwards + res.decode_steps}")
        print(f"  draft portion ms     : {res.total_ms:.1f}  ({res.tok_per_s:.0f} tok/s)")
        print("=" * 84)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
