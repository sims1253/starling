"""RTF + per-stage benchmark for the cohere-transcribe-03-2026 megakernel pipeline.

Reports wall-clock transcribe time, RTFx, per-token decode throughput, and a
per-stage breakdown (encoder / prefill / decode) for the short/medium/long
fixtures, with graph capture + warmup excluded from the timed region. Optional
``--compare-stock`` re-runs the stock eager reference + ``model.generate()`` for
a head-to-head.

Run:  ``uv run python -m benchmarks.cohere.bench_pipeline [--compare-stock] [--K N]``
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import median

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))


def _load_wav(name: str):
    import soundfile as sf

    wav, sr = sf.read(str(REPO / "tests" / "fixtures" / f"{name}.wav"))
    if wav.ndim > 1:
        wav = wav.mean(1)
    return wav, sr


def _cuda_timer(fn, warmup: int = 3, iters: int = 7) -> float:
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
    return median(times)


def main() -> int:
    from starling.cohere.config import EOS_TOKEN_ID
    from starling.cohere.decode_mega import GraphedDecoder
    from starling.cohere.encoder_graph import GraphedEncoder
    from starling.cohere.loader import load_model_and_processor
    from starling.cohere.reference import encode, greedy_generate

    ap = argparse.ArgumentParser()
    ap.add_argument("--compare-stock", action="store_true")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--iters", type=int, default=7)
    ap.add_argument("--no-gpu-lock", action="store_true")
    args = ap.parse_args()

    print("[bench] loading model ...")
    model, proc = load_model_and_processor()
    ge = GraphedEncoder(model.model.encoder, warmup_iters=3)

    use_lock = not args.no_gpu_lock
    if use_lock:
        from starling.parakeet.gpu_lock import with_gpu_lock
    results = {}

    def _run():
        for name in ("short", "medium", "long"):
            wav, sr = _load_wav(name)
            seconds = len(wav) / sr
            inp = proc(wav, sampling_rate=sr, language="en", return_tensors="pt")
            feat = inp["input_features"].to(torch.bfloat16).cuda()
            amask = inp["attention_mask"].cuda()
            dec_in = inp["decoder_input_ids"].cuda()
            B = feat.shape[0]

            # capture graphs for this shape (excluded from timing)
            enc_h = model.model.encoder(
                input_features=feat, attention_mask=amask
            ).last_hidden_state
            S = enc_h.shape[1]
            neg = torch.finfo(torch.bfloat16).min
            enc_mask = torch.zeros(B, 1, 1, S, device="cuda", dtype=torch.bfloat16)
            gd = GraphedDecoder(model, steps_per_replay=args.K, warmup_iters=4)
            gd.capture(dec_in, enc_h, enc_mask)
            # warm the encoder graph
            _ = ge(feat, amask)

            # ---- starling timed (capture excluded) ----
            def _starling():
                eh = ge(feat, amask)
                return gd.decode(dec_in, eh, enc_mask, max_new_tokens=300)

            star_ms = _cuda_timer(_starling, warmup=3, iters=args.iters)
            n_tok = _starling().shape[1]
            rtfx = seconds / (star_ms / 1000.0)
            tok_s = n_tok / (star_ms / 1000.0)

            entry = {
                "seconds": round(seconds, 2),
                "B": B,
                "n_gen_tokens": int(n_tok),
                "starling_ms": round(star_ms, 1),
                "starling_rtfx": round(rtfx, 1),
                "starling_tok_s": round(tok_s, 1),
            }

            # ---- per-stage (encoder / prefill / decode) ----
            def _enc_only():
                return ge(feat, amask)
            enc_ms = _cuda_timer(_enc_only, warmup=2, iters=args.iters)
            entry["encoder_ms"] = round(enc_ms, 1)

            if args.compare_stock:
                def _stock():
                    with torch.inference_mode():
                        eh, em = encode(model, feat, amask)
                        return greedy_generate(
                            model, eh, em, dec_in, max_new_tokens=300
                        )
                stock_ms = _cuda_timer(_stock, warmup=2, iters=args.iters)
                entry["stock_ms"] = round(stock_ms, 1)
                entry["stock_rtfx"] = round(seconds / (stock_ms / 1000.0), 1)
                entry["speedup"] = round(stock_ms / star_ms, 2)

                # also HF generate() for reference
                def _hf():
                    with torch.inference_mode():
                        model.generate(
                            input_features=feat, attention_mask=amask,
                            decoder_input_ids=dec_in, max_length=dec_in.shape[1] + 300,
                        )
                hf_ms = _cuda_timer(_hf, warmup=1, iters=3)
                entry["hf_generate_ms"] = round(hf_ms, 1)

            results[name] = entry
            print(f"\n[{name}] {seconds:.1f}s B={B} n_tok={n_tok}")
            print(f"  starling: {star_ms:.1f}ms  RTFx {rtfx:.1f}x  {tok_s:.1f} tok/s")
            print(f"  encoder:  {enc_ms:.1f}ms   decode+prefill: {star_ms-enc_ms:.1f}ms")
            if args.compare_stock:
                print(f"  stock:    {stock_ms:.1f}ms  RTFx {entry['stock_rtfx']:.1f}x"
                      f"  -> {entry['speedup']:.2f}x speedup")
                print(f"  HF gen:   {entry['hf_generate_ms']:.1f}ms")

    if use_lock:
        with with_gpu_lock(
            session="cohere-bench", model="cohere-transcribe-03-2026",
            eta_min=5, note="cohere pipeline benchmark",
        ):
            _run()
    else:
        _run()

    out = REPO / "outputs" / "cohere" / "bench_pipeline.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\n[bench] saved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
