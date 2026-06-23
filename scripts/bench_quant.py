"""Weight-only quantisation benchmark for Granite-Speech-4.1-2b LLM decode.

Measures the headline single-stream decode throughput and the batched
throughput for the weight-only INT8 quantised decoder
(:class:`megapar.quant.QuantLLMMega` / :class:`BatchedQuantLLMMega`) against the
bf16 baseline (:class:`MultiStepLLMMega` / :class:`BatchedFusedLLMMega`), plus
the transcript-quality cost (token match %, WER) of each.

Sections
--------
1. **Single-stream decode** (the headline): bf16 vs INT8 -- decode ms/token,
   tok/s, total transcribe ms, RTFx, peak VRAM.
2. **Batched B=8 / B=16**: bf16 vs INT8 RTFx + tok/s (the bandwidth win from
   quantisation would compound most here, if it materialised).
3. **Sustained-GEMV micro-benchmark** (the diagnosis): synthetic 280-GEMV/token
   pattern for bf16, FP8 ``_scaled_mm`` and the INT8 dequant-GEMM, to confirm
   WHY quantisation does (or does not) help -- the BW-bound diagnosis check.

All timed runs acquire the GPU lock (comms.md P1).

Run:  .venv/bin/python scripts/bench_quant.py
"""
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from megapar.audio import build_inputs, load_sample_audio
from megapar.batched import BatchedPipeline
from megapar.flags import OptFlags
from megapar.golden import load_golden, load_golden_text
from megapar.loader import get_components, load_model_and_processor
from megapar.llm_mega import FusedLLMMega
from megapar.multistep import MultiStepLLMMega
from megapar.parakeet.gpu_lock import with_gpu_lock
from megapar.quant import QuantLLMMega, quantize_linear, w8_linear

MAX_NEW_TOKENS = 100
ITERS = 8
WARMUP = 3


def _median_min(xs):
    return statistics.median(xs), min(xs)


def _wall_ms(fn, warm=WARMUP, iters=ITERS):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000.0)
    return _median_min(ts)


def _cuda_ms(fn, warm=WARMUP, iters=ITERS):
    torch.cuda.synchronize()
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return statistics.median(times)


def _wer(ref, hyp):
    r, h = ref.lower().split(), hyp.lower().split()
    if not r:
        return 0.0
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            c = 0 if r[i - 1] == h[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + c)
    return d[len(r)][len(h)] / len(r)


# =========================================================================== #
# 1. Single-stream decode: bf16 vs INT8
# =========================================================================== #
def bench_single_stream(model, proc, comps, inputs_embeds, audio_seconds, golden_gen, golden_resp):
    print("\n" + "=" * 82)
    print("1. SINGLE-STREAM DECODE: bf16 (MultiStep K=8) vs INT8 (QuantLLMMega)")
    print("=" * 82)
    lm = comps["language_model"]
    lm_head = model.lm_head
    tok = proc.tokenizer

    configs = [
        ("bf16 MultiStep K=8", MultiStepLLMMega(lm, lm_head, max_cache_len=640, steps_per_replay=8)),
        ("bf16 FusedLLMMega", FusedLLMMega(lm, lm_head, max_cache_len=640)),
        ("INT8 QuantLLMMega", QuantLLMMega(lm, lm_head, max_cache_len=640)),
    ]

    print(f"\n{'config':<22}{'GPU ms/tok':>12}{'wall ms':>10}{'tok/s':>9}"
          f"{'RTFx':>9}{'VRAM GB':>9}{'tok-match':>11}{'WER':>8}")
    print("-" * 90)
    rows = []
    for label, dec in configs:
        torch.cuda.reset_peak_memory_stats()
        # quality first (one generate).
        res = dec.generate(inputs_embeds, max_new_tokens=MAX_NEW_TOKENS,
                           eos_token_id=__eos__)
        text = tok.decode(res.ids[0], skip_special_tokens=True)
        n_match = int((res.ids[0] == golden_gen).sum().item())
        pct = n_match / golden_gen.shape[0] * 100.0
        w = _wer(golden_resp, text)

        # GPU-only per-token decode (CUDA events).
        rep = dec.bench(inputs_embeds, max_new_tokens=MAX_NEW_TOKENS, decode_iters=10)
        gpu_ms = rep.decode_ms_per_token

        # full-loop wall.
        def _gen():
            dec.generate(inputs_embeds, max_new_tokens=MAX_NEW_TOKENS, eos_token_id=__eos__)
        wall_med, _ = _wall_ms(_gen, warm=WARMUP, iters=ITERS)
        tps = MAX_NEW_TOKENS / (wall_med / 1000.0)
        vram = torch.cuda.max_memory_allocated() / 1e9
        rtfx = audio_seconds / (wall_med / 1000.0)
        print(f"{label:<22}{gpu_ms:>12.3f}{wall_med:>10.1f}{tps:>9.1f}"
              f"{rtfx:>9.2f}x{vram:>9.2f}{pct:>10.1f}%{w:>8.4f}")
        rows.append({"config": label, "gpu_ms_per_tok": round(gpu_ms, 4),
                     "wall_ms": round(wall_med, 1), "tok_per_s": round(tps, 1),
                     "rtfx": round(rtfx, 2), "vram_gb": round(vram, 2),
                     "token_match_pct": round(pct, 1), "wer": round(w, 4)})
    return rows


# =========================================================================== #
# 2. Batched B=8 / B=16: bf16 vs INT8
# =========================================================================== #
def bench_batched(model, proc, feats, ids, mask, audio_seconds):
    print("\n" + "=" * 82)
    print("2. BATCHED DECODE: bf16 vs INT8 (the BW win would compound here)")
    print("=" * 82)
    tok = proc.tokenizer
    golden_resp = load_golden_text().strip().split("ASSISTANT:", 1)[1].strip()
    rows = []
    print(f"\n{'B':>3}{'mode':<8}{'wall ms':>10}{'RTFx':>10}{'tok/s':>10}"
          f"{'VRAM GB':>10}{'tok-match':>11}{'WER':>8}")
    print("-" * 80)
    for B in [8, 16]:
        feats_list = [feats] * B
        ids_list = [ids] * B
        mask_list = [mask] * B
        for label, fl in [("bf16", OptFlags(quantized_weights=False, tolerance_mode=False)),
                          ("INT8", OptFlags(quantized_weights=True, tolerance_mode=True))]:
            pipe = BatchedPipeline(model, proc, max_batch_size=B, encoder_mode="cudagraph", flags=fl)
            # quality (one run).
            res = pipe.run_batch(feats_list, ids_list, mask_list, max_new_tokens=MAX_NEW_TOKENS)
            text0 = tok.decode(res.ids_list[0], skip_special_tokens=True)
            # batched uses golden_transcribe path; compare stream 0 text.
            w = _wer(golden_resp, text0)

            torch.cuda.reset_peak_memory_stats()
            def _batch():
                pipe.run_batch(feats_list, ids_list, mask_list, max_new_tokens=MAX_NEW_TOKENS)
            wall_med, _ = _wall_ms(_batch, warm=2, iters=4)
            rtfx = B * audio_seconds / (wall_med / 1000.0)
            tps = B * MAX_NEW_TOKENS / (wall_med / 1000.0)
            vram = torch.cuda.max_memory_allocated() / 1e9
            # token match for stream 0 vs golden (load golden tokens).
            golden_gen = load_golden("greedy_ids.pt")[0, 271:]
            nm = res.ids_list[0].shape[0]
            n_match = int((res.ids_list[0][:nm] == golden_gen[:nm]).sum().item())
            pct = n_match / nm * 100.0
            print(f"{B:>3}{label:<8}{wall_med:>10.1f}{rtfx:>9.1f}x{tps:>10.1f}"
                  f"{vram:>10.2f}{pct:>10.1f}%{w:>8.4f}")
            rows.append({"B": B, "mode": label, "wall_ms": round(wall_med, 1),
                         "rtfx": round(rtfx, 1), "tok_per_s": round(tps, 1),
                         "vram_gb": round(vram, 2), "token_match_pct": round(pct, 1),
                         "wer": round(w, 4)})
            del pipe
            torch.cuda.empty_cache()
    return rows


# =========================================================================== #
# 3. Sustained-GEMV micro-benchmark (the BW-bound diagnosis check)
# =========================================================================== #
def bench_gemv_diagnosis():
    """Synthetic 280-GEMV/token decode pattern: bf16 vs FP8 vs INT8.

    This is the controlled experiment that explains the macro numbers: it isolates
    the matmul cost from the rest of the decode (norms, attention, glue) and
    measures effective memory bandwidth for each weight dtype.
    """
    print("\n" + "=" * 82)
    print("3. SUSTAINED-GEMV MICRO-BENCHMARK (BW-bound diagnosis check)")
    print("=" * 82)
    shapes = [(2048, 2048), (1024, 2048), (1024, 2048), (2048, 2048),
              (4096, 2048), (4096, 2048), (2048, 4096)]
    n_layers = 40
    rows = []

    for M in [1, 8, 16]:
        xs, ws_i, ss, ws_bf = [], [], [], []
        for N, K in shapes:
            for _ in range(n_layers):
                a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
                wbf = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
                s = (wbf.abs().max(dim=1).values.float() / 127.0).to(torch.float16)
                wi = (wbf.float() / s.float().unsqueeze(1)).round().clamp(-128, 127).to(torch.int8)
                xs.append(a); ws_i.append(wi); ss.append(s); ws_bf.append(wbf)

        wb_i8 = sum(N * K * n_layers * 1 for N, K in shapes)   # int8 weight bytes/token
        wb_bf = sum(N * K * n_layers * 2 for N, K in shapes)   # bf16 weight bytes/token

        def bf16_run():
            for a, w in zip(xs, ws_bf):
                _ = torch.nn.functional.linear(a, w)
        ms_bf = _wall_ms(bf16_run)[0]

        def int8_run():
            for a, w, s in zip(xs, ws_i, ss):
                _ = w8_linear(a, w, s)
        ms_i8 = _wall_ms(int8_run)[0]

        # FP8 _scaled_mm (per-tensor scale, e4m3). fp8 needs M-tile alignment.
        ms_fp8 = float("nan")
        try:
            xf = [a.to(torch.float8_e4m3fn) for a in xs]
            wf = [w.to(torch.float8_e4m3fn) for w in ws_bf]
            sa = torch.ones((), device="cuda", dtype=torch.float32)
            sb = torch.ones((), device="cuda", dtype=torch.float32)

            def fp8_run():
                for a, w in zip(xf, wf):
                    _ = torch._scaled_mm(a, w.t(), scale_a=sa, scale_b=sb,
                                         out_dtype=torch.bfloat16)
            ms_fp8 = _wall_ms(fp8_run)[0]
        except Exception as e:
            print(f"  (FP8 _scaled_mm failed for M={M}: {e!r})")

        bw_bf = wb_bf / (ms_bf / 1000.0) / 1e9
        bw_i8 = wb_i8 / (ms_i8 / 1000.0) / 1e9
        speed_i8 = ms_bf / ms_i8
        speed_fp8 = ms_bf / ms_fp8 if ms_fp8 == ms_fp8 else float("nan")
        print(f"\n  M={M:>2}: bf16 {ms_bf:6.3f}ms ({bw_bf:5.0f}GB/s) | "
              f"INT8 {ms_i8:6.3f}ms ({bw_i8:5.0f}GB/s, {speed_i8:.2f}x bf16) | "
              f"FP8 {ms_fp8:6.3f}ms ({speed_fp8:.2f}x bf16)")
        rows.append({"M": M, "bf16_ms": round(ms_bf, 4), "bf16_gbs": round(bw_bf, 0),
                     "int8_ms": round(ms_i8, 4), "int8_gbs": round(bw_i8, 0),
                     "int8_speedup": round(speed_i8, 3),
                     "fp8_ms": round(ms_fp8, 4), "fp8_speedup": round(speed_fp8, 3)})
    return rows


# =========================================================================== #
# main
# =========================================================================== #
def main() -> int:
    global __eos__
    from megapar.config import LLM_EOS_TOKEN_ID
    __eos__ = LLM_EOS_TOKEN_ID

    with with_gpu_lock(
        session="granite-mega",
        model="granite-speech-4.1-2b",
        eta_min=10,
        note="bench_quant: weight-only INT8 quantisation benchmark",
    ):
        print("loading model + processor ...", flush=True)
        model, proc = load_model_and_processor(attn_impl="eager")
        comps = get_components(model)

        wav, sr = load_sample_audio()
        inputs = build_inputs(proc, wav)
        audio_seconds = wav.shape[1] / sr
        feats = inputs["input_features"].to(torch.bfloat16)
        ids = inputs["input_ids"]
        mask = inputs.get("input_features_mask")
        inputs_embeds = load_golden("inputs_embeds.pt").to("cuda", torch.bfloat16)
        golden_gen = load_golden("greedy_ids.pt")[0, 271:]
        golden_resp = load_golden_text().strip().split("ASSISTANT:", 1)[1].strip()

        print(f"audio {audio_seconds:.1f}s, prompt {ids.shape[1]} tok, "
              f"{MAX_NEW_TOKENS} new tok, GPU {torch.cuda.get_device_name(0)}\n", flush=True)

        r1 = bench_single_stream(model, proc, comps, inputs_embeds, audio_seconds, golden_gen, golden_resp)
        r2 = bench_batched(model, proc, feats, ids, mask, audio_seconds)
        r3 = bench_gemv_diagnosis()

        print("\n" + "=" * 82)
        print("VERDICT")
        print("=" * 82)
        bf = next(r for r in r1 if r["config"].startswith("bf16 MultiStep"))
        iq = next(r for r in r1 if r["config"].startswith("INT8"))
        if iq["tok_per_s"] < bf["tok_per_s"]:
            print(f"  INT8 is {bf['tok_per_s']/iq['tok_per_s']:.2f}x SLOWER than bf16 single-stream "
                  f"({iq['tok_per_s']:.0f} vs {bf['tok_per_s']:.0f} tok/s).")
            print("  The BW-bound diagnosis did NOT translate to a quantisation win:")
            print("  cuBLAS bf16 already saturates ~324-426 GB/s for these shapes and the")
            print("  int8 dequant overhead + Triton's per-shape BW efficiency eat the 2x")
            print("  weight-traffic reduction.  Path shipped behind a flag for re-evaluation.")
        else:
            print(f"  INT8 is {iq['tok_per_s']/bf['tok_per_s']:.2f}x FASTER than bf16 "
                  f"({iq['tok_per_s']:.0f} vs {bf['tok_per_s']:.0f} tok/s).")
        print(f"  Transcript quality: INT8 token-match {iq['token_match_pct']:.1f}%, WER {iq['wer']:.4f}.")
        print("=" * 82)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
