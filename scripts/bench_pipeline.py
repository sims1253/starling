#!/usr/bin/env python3
"""End-to-end benchmark: stock transformers vs the fused MegaPipeline.

Produces a stage-by-stage comparison table for Granite-Speech-4.1-2b ASR on the
sample audio (24.94s):

    | stage              | stock baseline | mega pipeline | speedup |

Stages: encoder, projector, audio-features (enc+proj), LLM prefill, LLM decode
(tok/s), and FULL transcribe end-to-end (ms + RTFx).  CUDA events are used for
per-stage timing; wall-clock is used for the full transcribe / generate.

RTFx = audio_seconds / total_seconds  (real-time factor; higher is better).

Usage:
    .venv/bin/python scripts/bench_pipeline.py
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from starling.granite.audio import build_inputs, load_sample_audio  # noqa: E402
from starling.config import LLM_EOS_TOKEN_ID  # noqa: E402
from starling.granite.loader import get_components, load_model_and_processor  # noqa: E402
from starling.granite.pipeline import MegaPipeline  # noqa: E402


def cuda_time_ms(fn, *, warmup: int = 3, iters: int = 10) -> float:
    """Median GPU time (ms) for ``fn`` using CUDA events."""
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return statistics.median(times)


def wall_time_ms(fn, *, warmup: int = 1, iters: int = 3) -> float:
    """Median wall-clock time (ms) for ``fn`` (includes Python overhead)."""
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
    print(" starling end-to-end benchmark: stock transformers vs MegaPipeline")
    print("=" * 78)

    print("\n[load] model + processor (eager) ...")
    model, processor = load_model_and_processor(attn_impl="eager")
    comps = get_components(model)
    encoder = comps["encoder"]
    projector = comps["projector"]
    language_model = comps["language_model"]

    wav, sr = load_sample_audio()
    inputs = build_inputs(processor, wav)
    audio_seconds = wav.shape[1] / sr
    input_ids = inputs["input_ids"]
    feats = inputs["input_features"]
    feats_bf = feats.to(torch.bfloat16)
    ifm = inputs.get("input_features_mask")
    n_gen = 100

    print(f"[load] audio = {audio_seconds:.2f}s, mel = {tuple(feats.shape)}, "
          f"prompt = {input_ids.shape[1]} tokens")

    # ------------------------------------------------------------------ #
    # STOCK BASELINE stage timing
    # ------------------------------------------------------------------ #
    print("\n[stock] timing stages ...")
    with torch.inference_mode():
        # encoder (stock conformer)
        stock_encoder_ms = cuda_time_ms(
            lambda: encoder(feats_bf, return_dict=True), warmup=3, iters=10
        )
        enc_lhs = encoder(feats_bf, return_dict=True).last_hidden_state

        # projector (stock BLIP2 q-former, eager)
        stock_projector_ms = cuda_time_ms(
            lambda: projector(enc_lhs), warmup=3, iters=20
        )

        # audio-features = encoder + projector (stock get_audio_features)
        stock_audio_features_ms = cuda_time_ms(
            lambda: model.get_audio_features(feats_bf, return_dict=True),
            warmup=3, iters=10,
        )

        # merged inputs_embeds (stock) + LLM prefill (single forward)
        audio_embeds = model.get_audio_features(
            feats_bf, return_dict=True
        ).pooler_output
        stock_inputs_embeds = model.model.get_merged_audio_embeddings(
            input_ids=input_ids,
            audio_features=audio_embeds,
            input_features_mask=ifm,
        )
        T = stock_inputs_embeds.shape[1]
        pos_ids = torch.arange(T, device="cuda").unsqueeze(0)

        def _stock_prefill():
            language_model(
                inputs_embeds=stock_inputs_embeds,
                position_ids=pos_ids,
                use_cache=True,
            )

        stock_prefill_ms = cuda_time_ms(_stock_prefill, warmup=3, iters=10)

        # full generate (wall-clock). This is the slow path (~7s); few iters.
        def _stock_generate():
            model.generate(
                input_ids=input_ids,
                input_features=feats,
                attention_mask=inputs["attention_mask"],
                input_features_mask=ifm,
                max_new_tokens=n_gen,
                do_sample=False,
                num_beams=1,
            )

        stock_full_ms = wall_time_ms(_stock_generate, warmup=1, iters=2)
        # decode = generate - audio_features - prefill ; tok/s over decode region
        stock_decode_ms = max(
            stock_full_ms - stock_audio_features_ms - stock_prefill_ms, 1e-6
        )
        stock_decode_tps = n_gen / (stock_decode_ms / 1000.0)
        stock_rtfx = audio_seconds / (stock_full_ms / 1000.0)

    print(f"  encoder={stock_encoder_ms:.2f}ms  projector={stock_projector_ms:.2f}ms")
    print(f"  audio-features={stock_audio_features_ms:.2f}ms  prefill={stock_prefill_ms:.2f}ms")
    print(f"  full-generate={stock_full_ms:.1f}ms  decode~{stock_decode_tps:.1f}tok/s  RTFx={stock_rtfx:.2f}x")

    # ------------------------------------------------------------------ #
    # MEGA PIPELINE stage timing
    # ------------------------------------------------------------------ #
    print("\n[mega] building MegaPipeline ...")
    pipe = MegaPipeline(model, processor, encoder_mode="cudagraph", use_fused_llm=True)

    print("[mega] timing stages ...")
    with torch.inference_mode():
        # fused encoder (cudagraph) -- first call captures
        mega_encoder_ms = cuda_time_ms(
            lambda: pipe.fused_encoder(feats_bf), warmup=3, iters=20
        )
        mega_enc_lhs = pipe.fused_encoder(feats_bf)

        # projector (same eager q-former as stock)
        mega_projector_ms = cuda_time_ms(
            lambda: pipe.projector(mega_enc_lhs), warmup=3, iters=20
        )

        # audio-features = fused encoder + projector
        mega_audio_features_ms = cuda_time_ms(
            lambda: pipe.encode_audio(feats_bf), warmup=3, iters=20
        )

        # LLM prefill + per-token decode + total (from FusedLLMMega.bench)
        bench_inputs_embeds = pipe.build_inputs_embeds(
            input_ids,
            pipe.encode_audio(feats_bf)[1],
            ifm,
        )
        rep = pipe.llm.bench(
            bench_inputs_embeds, max_new_tokens=n_gen, decode_iters=20
        )
        mega_prefill_ms = rep.prefill_ms
        mega_decode_tps = rep.decode_tok_per_s
        mega_decode_ms_per_tok = rep.decode_ms_per_token

        # full transcribe (wall-clock end-to-end)
        def _mega_transcribe():
            pipe.transcribe(feats, input_ids, ifm, max_new_tokens=n_gen)

        mega_full_ms = wall_time_ms(_mega_transcribe, warmup=2, iters=5)
        mega_rtfx = audio_seconds / (mega_full_ms / 1000.0)

    print(f"  encoder={mega_encoder_ms:.2f}ms  projector={mega_projector_ms:.2f}ms")
    print(f"  audio-features={mega_audio_features_ms:.2f}ms  prefill={mega_prefill_ms:.2f}ms")
    print(f"  decode~{mega_decode_tps:.1f}tok/s  full-transcribe={mega_full_ms:.1f}ms  RTFx={mega_rtfx:.2f}x")

    # ------------------------------------------------------------------ #
    # comparison table
    # ------------------------------------------------------------------ #
    full_speedup = stock_full_ms / mega_full_ms
    enc_speedup = stock_encoder_ms / mega_encoder_ms
    af_speedup = stock_audio_features_ms / mega_audio_features_ms
    prefill_speedup = stock_prefill_ms / mega_prefill_ms
    decode_speedup = mega_decode_tps / stock_decode_tps if stock_decode_tps > 0 else float("nan")

    print("\n" + "=" * 78)
    print(f" {'stage':<24}{'stock baseline':>18}{'mega pipeline':>18}{'speedup':>12}")
    print("-" * 78)
    print(f" {'encoder (ms)':<24}{stock_encoder_ms:>18.2f}{mega_encoder_ms:>18.2f}{enc_speedup:>11.2f}x")
    print(f" {'projector (ms)':<24}{stock_projector_ms:>18.2f}{mega_projector_ms:>18.2f}{stock_projector_ms / mega_projector_ms:>11.2f}x")
    print(f" {'audio-features (ms)':<24}{stock_audio_features_ms:>18.2f}{mega_audio_features_ms:>18.2f}{af_speedup:>11.2f}x")
    print(f" {'LLM prefill (ms)':<24}{stock_prefill_ms:>18.2f}{mega_prefill_ms:>18.2f}{prefill_speedup:>11.2f}x")
    print(f" {'LLM decode (tok/s)':<24}{stock_decode_tps:>17.1f} {mega_decode_tps:>17.1f} {decode_speedup:>11.2f}x")
    print("-" * 78)
    print(f" {'FULL transcribe (ms)':<24}{stock_full_ms:>18.1f}{mega_full_ms:>18.1f}{full_speedup:>11.2f}x")
    print(f" {'RTFx':<24}{stock_rtfx:>17.2f}x{mega_rtfx:>17.2f}x{'':>12}")
    print(f" {'tokens generated':<24}{n_gen:>18}{n_gen:>18}")
    print(f" {'audio seconds':<24}{audio_seconds:>18.2f}{audio_seconds:>18.2f}")
    print("=" * 78)
    print(f" FULL end-to-end speedup: {full_speedup:.2f}x  "
          f"(stock {stock_full_ms:.0f}ms -> mega {mega_full_ms:.0f}ms)")
    print(f" RTFx: stock {stock_rtfx:.2f}x -> mega {mega_rtfx:.2f}x  "
          f"(decode {mega_decode_tps:.0f} tok/s @ {mega_decode_ms_per_tok:.3f} ms/tok)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
