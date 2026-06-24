"""Benchmark + tune ``steps_per_replay`` (K) for the multi-step graphed TDT decode.

The :class:`GraphedDecoder` captures ``K`` consecutive decode steps into ONE
CUDA graph and replays it ``ceil(max_out / K)`` times, syncing the host ONCE per
K steps instead of once per step. With K=1 the per-step ``.cpu()`` host sync
dominates wall time (~38ms of host overhead on top of ~12ms of pure CUDA work at
B8 medium); larger K collapses those serial syncs.

This script, UNDER the shared-GPU lock, times ``GraphedDecoder._run_loop`` for
K in {1, 2, 4, 8, 16, 32, 64} on B1-medium and B8-medium, and reports for each K:

  * ``replay_ms``         -- median wall time of the full decode loop (cuda-event
                             bracketed; this is the headline, comparable to the
                             prior K=1 ~50ms at B8 medium), excl. capture +
                             ``batch_decode`` (capture is amortised in production).
  * ``single_replay_gpu_ms`` -- pure GPU time of ONE K-step graph replay (no host
                             sync) -- the irreducible CUDA compute floor.
  * ``n_replays``         -- number of K-step replays the loop actually performs.
  * ``pure_gpu_ms``       -- ``single_replay_gpu_ms * n_replays`` (total CUDA work;
                             ~constant across K, the compute floor).
  * ``speedup_vs_k1``     -- ``replay_ms(K=1) / replay_ms(K)``.
  * ``cuda_wall_ratio``   -- ``pure_gpu_ms / replay_ms`` (fraction of wall that is
                             real GPU compute; rises with K as host-sync tax falls).

Writes ``outputs/parakeet/multistep_bench.json`` and prints tables.

Run:  uv run python benchmarks/parakeet/bench_multistep.py
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
ORACLE = _REPO_ROOT / "outputs" / "oracle.json"

K_VALUES = [1, 2, 4, 8, 16, 32, 64]
WARMUP = 8
REPEATS = 20
MAX_SECONDS = 10.0
UTIL_GATE_PCT = 30  # defer the timed sweep if GPU util > this (shared card)


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


def _gpu_util_pct() -> float:
    try:
        out = torch.cuda.utilization()  # 0-100 integer on recent torch
        return float(out)
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


def _time_single_replay(gd, *, warmup=6, repeats=30):
    """Pure GPU time (ms, median) of ONE K-step ``graph.replay()`` (no host sync)."""
    g = gd.graph
    for _ in range(warmup):
        g.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        g.replay()
        e.record()
        torch.cuda.synchronize()
        samples.append(s.elapsed_time(e))
    return float(np.median(samples))


def _prepare(processor, audio_list):
    inp = processor(audio_list, sampling_rate=SAMPLE_RATE).to("cuda")
    inp["input_features"] = inp["input_features"].to(torch.bfloat16)
    return inp


def main() -> int:
    _suppress()

    oracle = {}
    if ORACLE.exists():
        oracle = {e["name"]: e["text"] for e in json.loads(ORACLE.read_text())}
    fixtures = mkfx.load_fixtures()
    for name in ("short", "medium", "long"):
        assert fixtures[name].any(), f"fixture {name} empty -- regenerate make_fixtures"

    print("[bench] loading model ...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForTDT.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    pad_id = processor.tokenizer.pad_token_id

    # configs: (label, batch_list)
    configs = [
        ("b1_medium", [fixtures["medium"]]),
        ("b8_medium", mkfx.build_uniform_batch(fixtures["medium"], 8)),
    ]

    all_configs = []
    print("\n[bench] acquiring GPU lock ...")
    with with_gpu_lock(
        session="parakeet", model=MODEL_ID,
        eta_min=3, note="multistep bench",
    ):
        free, total = torch.cuda.mem_get_info()
        print(f"[bench] lock held; GPU free={free/1e9:.1f}GB / {total/1e9:.1f}GB")

        # defer the timed sweep while the shared GPU is busy (>UTIL_GATE_PCT)
        while True:
            u = _gpu_util_pct()
            if u <= UTIL_GATE_PCT:
                break
            print(f"[bench] GPU util {u:.0f}% > {UTIL_GATE_PCT}% -- deferring 5s")
            time.sleep(5.0)
        print(f"[bench] GPU util {_gpu_util_pct():.0f}% <= {UTIL_GATE_PCT}% -- proceeding")

        with torch.inference_mode():
            for label, audio_list in configs:
                B = len(audio_list)
                inputs = _prepare(processor, audio_list)
                enc = model.get_audio_features(
                    input_features=inputs["input_features"],
                    attention_mask=inputs["attention_mask"],
                )
                pooler = enc.pooler_output.contiguous()
                valid_lengths = enc.attention_mask.to(torch.long).sum(-1).contiguous()
                T_enc = pooler.shape[1]
                audio_seconds = sum(len(a) / SAMPLE_RATE for a in audio_list)

                print(f"\n[bench] === {label} (B={B}, T_enc={T_enc}, "
                      f"{audio_seconds:.1f}s) ===")

                config_results = []
                baseline_replay_ms = None
                for K in K_VALUES:
                    gd = GraphedDecoder(model, steps_per_replay=K)
                    gd.capture(pooler, valid_lengths, pad_id, steps_per_replay=K)

                    # correctness vs oracle (B1: single text; B8: all == medium)
                    if "medium" in oracle:
                        texts = gd.decode(pooler, valid_lengths, processor)
                        if B == 1:
                            ok = texts[0] == oracle["medium"]
                        else:
                            ok = all(t == oracle["medium"] for t in texts)
                        if not ok:
                            print(f"  [K={K}] WARN: decode != oracle medium")

                    # one untimed run to obtain out_step (deterministic) + warmup
                    out_step = gd._run_loop(pooler, valid_lengths)
                    n_replays = max(1, -(-(out_step - 2) // K))  # ceil((out_step-2)/K)

                    # full decode-loop wall time (the headline replay_ms)
                    replay_ms, replay_p90, _ = _time_cuda(
                        lambda: gd._run_loop(pooler, valid_lengths)
                    )
                    # pure GPU time of one K-step replay (no host sync)
                    single_gpu = _time_single_replay(gd)
                    pure_gpu_ms = single_gpu * n_replays

                    if baseline_replay_ms is None:
                        baseline_replay_ms = replay_ms
                    speedup = baseline_replay_ms / replay_ms if replay_ms > 0 else 0.0
                    ratio = pure_gpu_ms / replay_ms if replay_ms > 0 else 0.0

                    config_results.append({
                        "K": K,
                        "replay_ms": round(replay_ms, 4),
                        "replay_p90_ms": round(replay_p90, 4),
                        "single_replay_gpu_ms": round(single_gpu, 4),
                        "n_replays": int(n_replays),
                        "pure_gpu_ms": round(pure_gpu_ms, 4),
                        "speedup_vs_k1": round(speedup, 3),
                        "cuda_wall_ratio": round(ratio, 3),
                    })
                    print(
                        f"  K={K:<3} replay={replay_ms:6.2f}ms (p90 {replay_p90:6.2f}) "
                        f"1rep_gpu={single_gpu:6.3f}ms n_rep={n_replays:<4} "
                        f"pure_gpu={pure_gpu_ms:6.2f}ms "
                        f"speedup={speedup:4.2f}x cuda/wall={ratio:4.2f}"
                    )

                    # free this K's graph + buffers before capturing the next K
                    del gd
                    torch.cuda.empty_cache()

                best = min(config_results, key=lambda r: r["replay_ms"])
                print(f"  -> best K={best['K']} replay={best['replay_ms']:.2f}ms "
                      f"({best['speedup_vs_k1']:.2f}x vs K=1)")

                all_configs.append({
                    "config": label,
                    "batch_size": B,
                    "T_enc": T_enc,
                    "audio_seconds": round(audio_seconds, 4),
                    "out_step": int(out_step),
                    "best_K": int(best["K"]),
                    "best_replay_ms": round(best["replay_ms"], 4),
                    "best_speedup_vs_k1": round(best["speedup_vs_k1"], 3),
                    "results": config_results,
                })

    payload = {
        "model_id": MODEL_ID,
        "dtype": "bfloat16",
        "device": torch.cuda.get_device_name(0),
        "method": (
            "cuda.Event, warmup>=8, >=20 samples (10s cap), median+p90; "
            "replay_ms = full _run_loop wall (excl. capture + batch_decode); "
            "single_replay_gpu_ms = one K-step graph.replay() GPU time (no host sync); "
            "pure_gpu_ms = single_replay_gpu_ms * n_replays; "
            "cuda_wall_ratio = pure_gpu_ms / replay_ms"
        ),
        "util_gate_pct": UTIL_GATE_PCT,
        "configs": all_configs,
    }
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS / "multistep_bench.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\n[bench] wrote {out_path}")

    # ---- print tables ----
    for cfg in all_configs:
        print(f"\n=== {cfg['config']} (B={cfg['batch_size']}, T_enc={cfg['T_enc']}, "
              f"out_step={cfg['out_step']}) ===")
        rows = []
        for r in cfg["results"]:
            rows.append([
                r["K"], f"{r['replay_ms']:.2f}", f"{r['replay_p90_ms']:.2f}",
                f"{r['single_replay_gpu_ms']:.3f}", r["n_replays"],
                f"{r['pure_gpu_ms']:.2f}", f"{r['speedup_vs_k1']:.2f}x",
                f"{r['cuda_wall_ratio']:.2f}",
            ])
        print(tabulate(
            rows,
            headers=["K", "replay_ms", "p90_ms", "1rep_gpu_ms", "n_rep",
                     "pure_gpu_ms", "speedup", "cuda/wall"],
            tablefmt="github",
        ))
        print(f"  best: K={cfg['best_K']} replay={cfg['best_replay_ms']:.2f}ms "
              f"({cfg['best_speedup_vs_k1']:.2f}x vs K=1)")

    # headline callout
    h = next((c for c in all_configs if c["config"] == "b8_medium"), None)
    if h is not None:
        k1 = next(r for r in h["results"] if r["K"] == 1)
        print(f"\n*** b8_medium: replay {k1['replay_ms']:.1f}ms (K=1) -> "
              f"{h['best_replay_ms']:.1f}ms (K={h['best_K']}); "
              f"{h['best_speedup_vs_k1']:.2f}x; "
              f"cuda/wall {k1['cuda_wall_ratio']:.2f} -> "
              f"{next(r for r in h['results'] if r['K']==h['best_K'])['cuda_wall_ratio']:.2f} ***")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
