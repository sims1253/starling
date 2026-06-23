"""Encoder optimisation benchmark: graphed-eager vs torch.compile (+ BN fold).

The 24-block Conformer encoder is the dominant wall-time cost at B8 medium
(~32ms graphed-eager, 56% of the integrated pipeline). The profiler attributes
~36ms of glue to elementwise (15%) + memops (5%) that ``torch.compile`` could
fuse, but the conv module's BatchNorm1d (tiny ``running_var``) amplifies bf16
rounding differences ~316x (granite sibling finding), so we fold the BN into
the depthwise conv (a deterministic affine fold, exact in real arithmetic)
before compiling.

This script, UNDER the shared-GPU lock, measures at B=8 uniform-medium:

  * ``graphed_ms``     -- the existing GraphedEncoder (CUDA-graph capture of the
                          stock ``get_audio_features``; byte-exact, ~32ms).
  * ``compiled_ms``    -- CompiledEncoder (BN fold + torch.compile
                          ``reduce-overhead``; NOT guaranteed byte-exact).
  * ``max_abs``        -- ``max|graphed_pooler - compiled_pooler|`` (the
                          compiled-vs-reference deviation; 0.0 == byte-exact).
  * ``transcript_ok``  -- compiled integrated transcript == oracle (text-level).
  * BN ``running_var`` distribution (to answer: is parakeet's BN as unstable as
    granite's ``~4e-10``?).

Writes ``outputs/parakeet/encoder_opt_bench.json`` and prints a summary table.

Run:  uv run python benchmarks/parakeet/bench_encoder_opt.py
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from tabulate import tabulate

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "tests" / "fixtures"))
sys.path.insert(0, str(_REPO_ROOT / "src"))

import make_fixtures as mkfx  # noqa: E402
from starling.parakeet.encoder_graph import (  # noqa: E402
    CompiledEncoder,
    GraphedEncoder,
)
from starling.parakeet.gpu_lock import with_gpu_lock  # noqa: E402

MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"
SAMPLE_RATE = 16000
OUTPUTS = _REPO_ROOT / "outputs" / "parakeet"
ORACLE = _REPO_ROOT / "outputs" / "oracle.json"

WARMUP = 10
REPEATS = 25
MAX_SECONDS = 12.0
UTIL_GATE_PCT = 30  # defer the timed sweep while the shared GPU is busy
BATCH_SIZE = 8


def _suppress() -> None:
    for mod in (
        "transformers.generation.utils",
        "transformers.models.parakeet.generation_parakeet",
        "torch.nn.modules.rnn",
    ):
        try:
            warnings.filterwarnings("ignore", module=mod)
        except Exception:
            pass
    warnings.filterwarnings("ignore", message=".*max_length.*")
    warnings.filterwarnings("ignore", message=".*RNN module weights.*")


def _gpu_util_pct() -> float:
    try:
        return float(torch.cuda.utilization())
    except Exception:
        return 0.0


def _time_cuda(fn, *, warmup=WARMUP, repeats=REPEATS, max_s=MAX_SECONDS):
    """Median + p90 GPU/wall time (ms) for ``fn`` via cuda events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples: list[float] = []
    wall0 = time.perf_counter()
    for _ in range(repeats):
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        samples.append(s.elapsed_time(e))
        if time.perf_counter() - wall0 > max_s:
            break
    return float(np.median(samples)), float(np.percentile(samples, 90)), len(samples)


def main() -> int:
    _suppress()

    oracle = {}
    if ORACLE.exists():
        oracle = {e["name"]: e["text"] for e in json.loads(ORACLE.read_text())}
    else:
        print(f"[bench] WARN: {ORACLE} missing -- transcript gate disabled")
    fixtures = mkfx.load_fixtures()
    medium = fixtures["medium"]
    audio_list = mkfx.build_uniform_batch(medium, BATCH_SIZE)
    audio_seconds = sum(len(a) / SAMPLE_RATE for a in audio_list)

    # load model + processor + build mel features once (shared by both encoders)
    print("[bench] loading model ...")
    from transformers import AutoModelForTDT, AutoProcessor

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForTDT.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    pad_id = processor.tokenizer.pad_token_id

    # precompute mel features (bf16) for the encoder timing (isolates encoder ms)
    inp = processor(audio_list, sampling_rate=SAMPLE_RATE)
    input_features = inp["input_features"].to(torch.bfloat16).cuda()
    attention_mask = inp["attention_mask"].to("cuda")
    print(f"[bench] input_features={tuple(input_features.shape)} "
          f"dtype={input_features.dtype}")

    result: dict = {
        "model_id": MODEL_ID,
        "dtype": "bfloat16",
        "device": torch.cuda.get_device_name(0),
        "batch_size": BATCH_SIZE,
        "audio_seconds": round(audio_seconds, 4),
        "input_features_shape": list(input_features.shape),
    }

    print("\n[bench] acquiring GPU lock ...")
    with with_gpu_lock(
        session="parakeet-mega", model=MODEL_ID,
        eta_min=5, note="encoder opt bench",
    ):
        free, total = torch.cuda.mem_get_info()
        print(f"[bench] lock held; GPU free={free/1e9:.1f}GB / {total/1e9:.1f}GB")
        while True:
            u = _gpu_util_pct()
            if u <= UTIL_GATE_PCT:
                break
            print(f"[bench] GPU util {u:.0f}% > {UTIL_GATE_PCT}% -- deferring 5s")
            time.sleep(5.0)
        print(f"[bench] GPU util {_gpu_util_pct():.0f}% <= {UTIL_GATE_PCT}% -- proceeding")

        with torch.inference_mode():
            # --- reference: graphed-eager encoder (byte-exact baseline ~32ms) ---
            print("[bench] building GraphedEncoder (reference) ...")
            genc = GraphedEncoder(model)
            ref_out = genc(input_features, attention_mask)
            ref_pooler = ref_out.pooler_output
            print(f"[bench] reference pooler={tuple(ref_pooler.shape)} "
                  f"dtype={ref_pooler.dtype}")
            graphed_ms, graphed_p90, n_g = _time_cuda(
                lambda: genc(input_features, attention_mask)
            )
            print(f"[bench] graphed-eager: {graphed_ms:.3f}ms "
                  f"(p90 {graphed_p90:.3f}, n={n_g})")

            # --- compiled encoder (BN fold + torch.compile reduce-overhead) ---
            print("[bench] building CompiledEncoder (BN fold + torch.compile) ...")
            cenc = CompiledEncoder(model, fold_bn=True, compile_mode="reduce-overhead")
            if cenc.bn_stats is not None:
                print(f"[bench] BN fold stats: {cenc.bn_stats}")
            result["bn_stats"] = cenc.bn_stats
            # the compiled encoder's first call(s) pay tracing + autotune +
            # cudagraph capture (warmup happens inside _warmup); measure after.
            comp_out = cenc(input_features, attention_mask)
            comp_pooler = comp_out.pooler_output
            compiled_ms, compiled_p90, n_c = _time_cuda(
                lambda: cenc(input_features, attention_mask)
            )
            print(f"[bench] compiled: {compiled_ms:.3f}ms "
                  f"(p90 {compiled_p90:.3f}, n={n_c})")

            # --- accuracy: compiled-vs-reference pooler max_abs ---
            max_abs = (
                (ref_pooler.float() - comp_pooler.float()).abs().max().item()
            )
            mean_abs = (
                (ref_pooler.float() - comp_pooler.float()).abs().mean().item()
            )
            byte_exact = (max_abs == 0.0)
            print(f"[bench] pooler max_abs={max_abs:.3e} "
                  f"mean_abs={mean_abs:.3e} byte_exact={byte_exact}")

            # --- integrated transcript gate (text-level) ---
            transcript_ok = None
            compiled_text = None
            if "medium" in oracle:
                # run the compiled encoder -> graphed decode for a text check.
                # (encoder is the only thing that differs; the decode graph is
                # shape-cached and reuses the existing GraphedDecoder.)
                from starling.parakeet.decode_mega import GraphedDecoder

                valid_lengths = comp_out.attention_mask.to(torch.long).sum(-1).contiguous()
                dec = GraphedDecoder(model).capture(
                    comp_pooler.contiguous(), valid_lengths, pad_id
                )
                texts = dec.decode(comp_pooler.contiguous(), valid_lengths, processor)
                compiled_text = texts[0]
                transcript_ok = all(t == oracle["medium"] for t in texts)
                print(f"[bench] compiled transcript == oracle? {transcript_ok}")
                if not transcript_ok:
                    print(f"   oracle: {oracle['medium']!r}")
                    print(f"   compiled[0]: {compiled_text!r}")

            speedup = graphed_ms / compiled_ms if compiled_ms > 0 else 0.0

            result.update({
                "graphed_ms": round(graphed_ms, 4),
                "graphed_p90_ms": round(graphed_p90, 4),
                "compiled_ms": round(compiled_ms, 4),
                "compiled_p90_ms": round(compiled_p90, 4),
                "speedup": round(speedup, 3),
                "pooler_max_abs": float(max_abs),
                "pooler_mean_abs": float(mean_abs),
                "byte_exact": bool(byte_exact),
                "transcript_ok": transcript_ok,
                "compiled_text": compiled_text,
                "n_graphed_samples": n_g,
                "n_compiled_samples": n_c,
            })

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS / "encoder_opt_bench.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\n[bench] wrote {out_path}")

    # ---- summary table ----
    rows = [[
        "graphed-eager",
        f"{graphed_ms:.2f}", f"{graphed_p90:.2f}", "1.00x",
        "0.0 (ref)", "byte-exact",
    ], [
        "compiled (fold+compile)",
        f"{compiled_ms:.2f}", f"{compiled_p90:.2f}", f"{speedup:.2f}x",
        f"{max_abs:.2e}",
        ("byte-exact" if byte_exact else
         ("transcript ok" if transcript_ok else "DRIFT")),
    ]]
    print("\n" + tabulate(
        rows,
        headers=["encoder", "ms", "p90_ms", "speedup", "max_abs", "accuracy"],
        tablefmt="github",
    ))
    print(f"\n*** B8 medium encoder: graphed {graphed_ms:.1f}ms -> "
          f"compiled {compiled_ms:.1f}ms ({speedup:.2f}x); "
          f"max_abs={max_abs:.2e} byte_exact={byte_exact} "
          f"transcript_ok={transcript_ok} ***")

    # RTF projection: if the encoder drops, recompute the integrated RTF using
    # the last pipeline_bench mel/decode breakdown (if present).
    pipe_bench = OUTPUTS / "pipeline_bench.json"
    if pipe_bench.exists():
        try:
            pb = json.loads(pipe_bench.read_text())
            b8 = next((r for r in pb["results"] if r["batch_size"] == 8), None)
            if b8 is not None:
                new_total = b8["mel_ms"] + compiled_ms + b8["decode_ms"]
                new_rtf = audio_seconds / (new_total / 1000.0)
                print(f"*** RTF projection (B8 medium): "
                      f"mel {b8['mel_ms']:.1f} + enc {compiled_ms:.1f} + "
                      f"dec {b8['decode_ms']:.1f} = {new_total:.1f}ms -> "
                      f"{new_rtf:,.0f}x (was {b8['rtf']:,.0f}x with enc "
                      f"{b8['encoder_ms']:.1f}ms) ***")
                result["rtf_projection"] = {
                    "mel_ms": round(b8["mel_ms"], 4),
                    "decode_ms": round(b8["decode_ms"], 4),
                    "new_encoder_ms": round(compiled_ms, 4),
                    "new_total_ms": round(new_total, 4),
                    "new_rtf": round(new_rtf, 4),
                    "prior_rtf": round(b8["rtf"], 4),
                    "prior_encoder_ms": round(b8["encoder_ms"], 4),
                }
                out_path.write_text(json.dumps(result, indent=2))
        except (KeyError, ValueError, ZeroDivisionError) as exc:
            print(f"[bench] RTF projection skipped: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
