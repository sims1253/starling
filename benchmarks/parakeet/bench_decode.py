"""Benchmark: stock ``model.generate`` decode vs CUDA-graph-captured decode.

Under the shared-GPU lock, for each config (B=1 short/medium/long + B=8
uniform-medium) this script measures, with CUDA events (warmup>=8, >=15 samples,
8s cap, median + p90):

  * ``encoder_ms``  -- ``model.get_audio_features`` (Conformer + projector)
  * ``feat_ms``     -- ``processor`` + H2D + bf16 cast
  * ``stock_gen_ms``-- full ``model.generate`` (encoder + TDT decode loop)
  * ``graphed_decode_ms`` -- :class:`GraphedDecoder._run_loop` (capture amortised:
    graph captured ONCE, then the decode loop timed; excludes ``batch_decode``)

Derived (per config):
  * ``stock_decode_ms``   = ``stock_gen_ms - encoder_ms``
  * ``speedup``           = ``stock_decode_ms / graphed_decode_ms``
  * ``new_total_ms``      = ``headline.feat_ms + headline.encoder_ms + graphed_decode_ms``
                           (feat/encoder reused from baseline_bench.json headline,
                            as specified)
  * ``new_rtf``           = ``audio_seconds / (new_total_ms / 1000)``

A per-config honest ``new_total_measured_ms`` / ``new_rtf_measured`` (using that
config's own measured feat+encoder) is also reported.

Writes ``outputs/parakeet/decode_bench.json`` and prints a table.
Run:  uv run python benchmarks/parakeet/bench_decode.py
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
from transformers import AutoModelForTDT, AutoProcessor

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "tests" / "fixtures"))
sys.path.insert(0, str(_REPO_ROOT / "src"))

import make_fixtures as mkfx  # noqa: E402
from starling.parakeet.decode_mega import GraphedDecoder  # noqa: E402
from starling.parakeet.gpu_lock import with_gpu_lock  # noqa: E402

MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"
SAMPLE_RATE = 16000
OUTPUTS = _REPO_ROOT / "outputs" / "parakeet"
BASELINE = _REPO_ROOT / "outputs" / "baseline_bench.json"
ORACLE = _REPO_ROOT / "outputs" / "oracle.json"

WARMUP = 8
REPEATS = 15
MAX_SECONDS = 8.0


def _suppress() -> None:
    for mod in (
        "transformers.generation.utils",
        "transformers.models.parakeet.generation_parakeet",
    ):
        try:
            warnings.filterwarnings("ignore", module=mod)
        except Exception:
            pass
    warnings.filterwarnings("ignore", message=".*max_length.*")
    warnings.filterwarnings("ignore", message=".*RNN module weights.*")


def time_cuda(fn, *, warmup=WARMUP, repeats=REPEATS, max_s=MAX_SECONDS) -> tuple[float, float, int]:
    """Median + p90 GPU time (ms) for ``fn`` via cuda events."""
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
    med = float(np.median(samples))
    p90 = float(np.percentile(samples, 90))
    return med, p90, len(samples)


def main() -> int:
    _suppress()

    # ---- baseline headline (feat/encoder to reuse) ----
    headline = json.loads(BASELINE.read_text())["headline"]
    h_feat = float(headline["feat_ms"])
    h_enc = float(headline["encoder_ms"])
    print(f"[bench] baseline headline: feat={h_feat:.2f}ms encoder={h_enc:.2f}ms "
          f"decode(stock)={headline['decode_ms']:.2f}ms rtf={headline['rtf_median']:.1f}x")

    oracle = {e["name"]: e["text"] for e in json.loads(ORACLE.read_text())}
    fixtures = mkfx.load_fixtures()
    for name in ("short", "medium", "long"):
        if not fixtures[name].any():
            raise RuntimeError(f"fixture {name} empty -- regenerate via make_fixtures")

    print("[bench] loading model ...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForTDT.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    pad_id = processor.tokenizer.pad_token_id
    max_symbols = int(model.config.max_symbols_per_step)

    def prepare(audio_list):
        inp = processor(audio_list, sampling_rate=SAMPLE_RATE).to("cuda")
        inp["input_features"] = inp["input_features"].to(torch.bfloat16)
        return inp

    # configs: (label, batch_list)
    configs = [
        ("short", [fixtures["short"]]),
        ("medium", [fixtures["medium"]]),
        ("long", [fixtures["long"]]),
        ("batch8_medium", mkfx.build_uniform_batch(fixtures["medium"], 8)),
    ]

    results = []
    print("\n[bench] acquiring GPU lock ...")
    with with_gpu_lock(session="parakeet", model=MODEL_ID,
                       eta_min=5, note="decode bench (stock vs graphed)"):
        # re-check GPU state inside the lock
        free, total = torch.cuda.mem_get_info()
        print(f"[bench] lock held; GPU free={free/1e9:.1f}GB / {total/1e9:.1f}GB")

        with torch.inference_mode():
            for label, audio_list in configs:
                B = len(audio_list)
                audio_seconds = sum(len(a) / SAMPLE_RATE for a in audio_list)
                inputs = prepare(audio_list)
                input_features = inputs["input_features"]
                attention_mask = inputs["attention_mask"]

                # encoder features (precomputed once for this config)
                enc = model.get_audio_features(
                    input_features=input_features, attention_mask=attention_mask
                )
                pooler = enc.pooler_output.contiguous()
                valid_lengths = enc.attention_mask.to(torch.long).sum(-1).contiguous()
                T_enc = pooler.shape[1]

                print(f"\n[bench] === {label} (B={B}, T_enc={T_enc}, "
                      f"{audio_seconds:.1f}s) ===")

                # feat + encoder timing
                feat_ms, feat_p90, _ = time_cuda(lambda: prepare(audio_list))
                enc_ms, enc_p90, _ = time_cuda(
                    lambda: model.get_audio_features(
                        input_features=input_features, attention_mask=attention_mask)
                )
                print(f"  feat={feat_ms:.2f}ms  encoder={enc_ms:.2f}ms")

                # stock generate (full: encoder + decode)
                max_new = max_symbols * T_enc
                gen_ms, gen_p90, n_gen = time_cuda(
                    lambda: model.generate(
                        input_features=input_features,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new,
                    )
                )
                stock_decode_ms = gen_ms - enc_ms
                print(f"  stock generate={gen_ms:.2f}ms (n={n_gen}) -> "
                      f"stock_decode={stock_decode_ms:.2f}ms")

                # graphed decode (capture once, time the loop)
                gd = GraphedDecoder(model)
                gd.capture(pooler, valid_lengths, pad_id)
                # correctness check vs oracle for B=1 configs
                if B == 1 and label in oracle:
                    texts = gd.decode(pooler, valid_lengths, processor)
                    ok = texts[0] == oracle[label]
                    print(f"  correctness: graphed == oracle ? {ok}")
                    if not ok:
                        print(f"    WARN drift: {texts[0][:60]!r} vs {oracle[label][:60]!r}")

                dec_ms, dec_p90, n_dec = time_cuda(
                    lambda: gd._run_loop(pooler, valid_lengths)
                )
                print(f"  graphed_decode={dec_ms:.2f}ms (n={n_dec}, p90={dec_p90:.2f}ms)")

                speedup = stock_decode_ms / dec_ms if dec_ms > 0 else float("inf")
                # task-defined new_total (headline feat/encoder) + honest measured
                new_total_ms = h_feat + h_enc + dec_ms
                new_rtf = audio_seconds / (new_total_ms / 1000.0) if new_total_ms > 0 else 0.0
                new_total_meas = feat_ms + enc_ms + dec_ms
                new_rtf_meas = audio_seconds / (new_total_meas / 1000.0) if new_total_meas > 0 else 0.0

                results.append({
                    "config": label,
                    "batch_size": B,
                    "T_enc": T_enc,
                    "audio_seconds": round(audio_seconds, 4),
                    "feat_ms": round(feat_ms, 4),
                    "feat_p90_ms": round(feat_p90, 4),
                    "encoder_ms": round(enc_ms, 4),
                    "encoder_p90_ms": round(enc_p90, 4),
                    "stock_gen_ms": round(gen_ms, 4),
                    "stock_gen_p90_ms": round(gen_p90, 4),
                    "stock_decode_ms": round(stock_decode_ms, 4),
                    "graphed_decode_ms": round(dec_ms, 4),
                    "graphed_decode_p90_ms": round(dec_p90, 4),
                    "speedup": round(speedup, 4),
                    "new_total_ms": round(new_total_ms, 4),
                    "new_rtf": round(new_rtf, 4),
                    "new_total_measured_ms": round(new_total_meas, 4),
                    "new_rtf_measured": round(new_rtf_meas, 4),
                    "headline_feat_ms_used": h_feat,
                    "headline_encoder_ms_used": h_enc,
                    "n_gen_samples": n_gen,
                    "n_dec_samples": n_dec,
                })

    # ---- write JSON ----
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_id": MODEL_ID,
        "dtype": "bfloat16",
        "device": torch.cuda.get_device_name(0),
        "method": "cuda.Event, warmup>=8, >=15 samples (8s cap), median+p90; "
                  "graphed_decode excludes capture (amortised) and batch_decode",
        "baseline_headline": {
            "feat_ms": h_feat, "encoder_ms": h_enc,
            "stock_decode_ms": headline["decode_ms"],
            "rtf_median": headline["rtf_median"],
        },
        "configs": results,
    }
    out_path = OUTPUTS / "decode_bench.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\n[bench] wrote {out_path}")

    # ---- print table ----
    rows = []
    for r in results:
        rows.append([
            r["config"], r["batch_size"], r["audio_seconds"],
            f"{r['stock_decode_ms']:.1f}", f"{r['graphed_decode_ms']:.1f}",
            f"{r['speedup']:.2f}x", f"{r['new_total_ms']:.1f}",
            f"{r['new_rtf']:.1f}", f"{r['new_rtf_measured']:.1f}",
        ])
    print("\n" + tabulate(
        rows,
        headers=["config", "B", "audio_s", "stock_dec", "graph_dec",
                 "speedup", "new_total", "new_rtf", "new_rtf_meas"],
        tablefmt="github",
    ))
    # headline callout
    h = next((r for r in results if r["config"] == "batch8_medium"), None)
    if h is not None:
        print(f"\n*** batch8_medium: stock_decode={h['stock_decode_ms']:.1f}ms -> "
              f"graphed_decode={h['graphed_decode_ms']:.1f}ms ({h['speedup']:.2f}x); "
              f"new RTF={h['new_rtf']:.1f}x (was {headline['rtf_median']:.1f}x) ***")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
