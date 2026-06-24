"""Benchmark + tune ``steps_per_replay`` (K) for the multi-step graphed Granite LLM.

Mirrors ``benchmarks/parakeet/bench_multistep.py`` but for the granite-speech
LLM decoder (:class:`starling.multistep.MultiStepLLMMega`). The decoder captures
``K`` consecutive greedy decode steps into ONE CUDA graph and replays it
``ceil(n_decode / K)`` times, syncing the host ONCE per K tokens instead of once
per token. With K=1 the per-token ``.item()`` host sync dominates wall time;
larger K collapses those serial syncs. The default is K=16.

This script, UNDER the shared-GPU lock, times the 99-token decode loop (prefill
excluded) for K in {1, 2, 4, 8, 16, 32, 48, 64} and reports for each K:

  * ``decode_loop_ms``     -- median wall time of the full chunked decode loop
                             (cuda-event bracketed; the headline metric).
  * ``per_token_ms``       -- ``decode_loop_ms / n_decode``.
  * ``tok_per_s``          -- ``n_decode / (decode_loop_ms / 1000)``.
  * ``single_replay_gpu_ms`` -- pure GPU time of ONE K-step graph replay (no host
                             sync) -- the irreducible CUDA compute floor.
  * ``n_replays``          -- number of K-step replays (``ceil(n_decode / K)``).
  * ``pure_gpu_ms``        -- ``single_replay_gpu_ms * n_replays``.
  * ``speedup_vs_k1``      -- ``decode_loop_ms(K=1) / decode_loop_ms(K)``.
  * ``cuda_wall_ratio``    -- ``pure_gpu_ms / decode_loop_ms``.

Also verifies byte-exactness: every K must reproduce
``golden/greedy_ids.pt[:, 271:]`` (the 100 greedy tokens).

Writes ``outputs/llm_multistep_bench.json`` and prints tables.

Run:  uv run python benchmarks/bench_multistep_llm.py
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

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from starling.config import LLM_EOS_TOKEN_ID  # noqa: E402
from starling.golden import load_golden  # noqa: E402
from starling.loader import get_components, load_model_and_processor  # noqa: E402
from starling.multistep import MultiStepLLMMega  # noqa: E402
from starling.parakeet.gpu_lock import with_gpu_lock  # noqa: E402

OUTPUTS = _REPO_ROOT / "outputs"

K_VALUES = [1, 2, 4, 8, 16, 32, 48, 64]
WARMUP = 8
REPEATS = 20
MAX_SECONDS = 15.0
MAX_NEW_TOKENS = 100
UTIL_GATE_PCT = 30


def _suppress() -> None:
    for mod in ("transformers", "transformers.generation.utils"):
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


def _time_single_replay(decoder, *, warmup=6, repeats=30):
    """Pure GPU time (ms, median) of ONE K-step ``_ms_graph.replay()`` (no sync)."""
    g = decoder._ms_graph
    T = decoder._prefill_len
    first_tok = decoder._first_token
    for _ in range(warmup):
        g.replay()
        decoder._reset_to_chunk_start(T, first_tok)
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
        decoder._reset_to_chunk_start(T, first_tok)
    return float(np.median(samples))


def main() -> int:
    _suppress()

    print("[bench] loading model + golden artefacts ...")
    model, proc = load_model_and_processor(attn_impl="eager")
    comps = get_components(model)
    inputs_embeds = load_golden("inputs_embeds.pt").to("cuda", torch.bfloat16)
    golden_gen = load_golden("greedy_ids.pt")[0][271:]
    T = inputs_embeds.shape[1]
    n_decode = MAX_NEW_TOKENS - 1  # 99 decode steps after the prefill token

    print("\n[bench] acquiring GPU lock ...")
    with with_gpu_lock(
        session="llm-multistep", model="granite-speech-4.1-2b",
        eta_min=5, note="K sweep",
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
            config_results = []
            baseline_ms = None
            for K in K_VALUES:
                # guard: K steps must fit in the cache
                n_chunks = (n_decode + K - 1) // K  # ceil(n_decode / K)
                total_steps = n_chunks * K
                if T - 1 + total_steps >= 640:
                    print(f"  K={K:<3} SKIP (cache overflow: T={T} + "
                          f"{total_steps} >= 640)")
                    continue

                decoder = MultiStepLLMMega(
                    comps["language_model"], model.lm_head,
                    max_cache_len=640, steps_per_replay=K,
                )
                # prefill (outside timing)
                first_tok = decoder.prefill(inputs_embeds)
                decoder._prefill_len = T
                decoder._first_token = first_tok
                decoder.capture(first_tok, T)
                decoder._reset_to_chunk_start(T, first_tok)

                # correctness: byte-exact vs golden
                res = decoder.generate(
                    inputs_embeds, max_new_tokens=MAX_NEW_TOKENS,
                    eos_token_id=LLM_EOS_TOKEN_ID,
                )
                exact = bool((res.ids[0] == golden_gen).all().item())
                if not exact:
                    diff = (res.ids[0] != golden_gen).nonzero()
                    pos = int(diff[0].item()) if diff.numel() else -1
                    print(f"  K={K:<3} WARN: token mismatch at {pos}")

                # time the full decode loop (prefill excluded): all n_chunks
                # K-step replays, each with one host sync (.tolist()).
                def _full_loop(d=decoder, T_=T, ft=first_tok, nc=n_chunks):
                    d._reset_to_chunk_start(T_, ft)
                    for _c in range(nc):
                        d._ms_graph.replay()
                        _ = d.output_ids.tolist()

                loop_ms, loop_p90, _ = _time_cuda(_full_loop)
                single_gpu = _time_single_replay(decoder)
                pure_gpu_ms = single_gpu * n_chunks

                if baseline_ms is None:
                    baseline_ms = loop_ms
                speedup = baseline_ms / loop_ms if loop_ms > 0 else 0.0
                ratio = pure_gpu_ms / loop_ms if loop_ms > 0 else 0.0
                per_tok = loop_ms / n_decode
                tps = n_decode / (loop_ms / 1000.0) if loop_ms > 0 else 0.0

                config_results.append({
                    "K": K,
                    "exact": exact,
                    "decode_loop_ms": round(loop_ms, 4),
                    "decode_loop_p90_ms": round(loop_p90, 4),
                    "per_token_ms": round(per_tok, 4),
                    "tok_per_s": round(tps, 1),
                    "single_replay_gpu_ms": round(single_gpu, 4),
                    "n_replays": int(n_chunks),
                    "pure_gpu_ms": round(pure_gpu_ms, 4),
                    "speedup_vs_k1": round(speedup, 3),
                    "cuda_wall_ratio": round(ratio, 3),
                })
                print(
                    f"  K={K:<3} loop={loop_ms:7.2f}ms (p90 {loop_p90:7.2f}) "
                    f"1rep_gpu={single_gpu:7.3f}ms n_rep={n_chunks:<3} "
                    f"pure_gpu={pure_gpu_ms:7.2f}ms "
                    f"per_tok={per_tok:5.3f}ms tps={tps:7.1f} "
                    f"speedup={speedup:4.2f}x cuda/wall={ratio:4.2f}"
                    f" {'OK' if exact else 'MISMATCH'}"
                )

                del decoder
                torch.cuda.empty_cache()

            best = min(config_results, key=lambda r: r["decode_loop_ms"])
            print(f"\n  -> best K={best['K']} loop={best['decode_loop_ms']:.2f}ms "
                  f"({best['speedup_vs_k1']:.2f}x vs K=1, "
                  f"{best['tok_per_s']:.1f} tok/s)")

    payload = {
        "model": "granite-speech-4.1-2b (LLM decoder)",
        "dtype": "bfloat16",
        "device": torch.cuda.get_device_name(0),
        "max_new_tokens": MAX_NEW_TOKENS,
        "n_decode": n_decode,
        "prefill_len": T,
        "method": (
            "cuda.Event, warmup>=8, >=20 samples (15s cap), median+p90; "
            "decode_loop_ms = full chunked K-step replay loop (prefill excluded); "
            "single_replay_gpu_ms = one K-step _ms_graph.replay() GPU time (no sync); "
            "pure_gpu_ms = single_replay_gpu_ms * n_replays; "
            "cuda_wall_ratio = pure_gpu_ms / decode_loop_ms"
        ),
        "util_gate_pct": UTIL_GATE_PCT,
        "results": config_results,
        "best_K": int(best["K"]),
        "best_decode_loop_ms": round(best["decode_loop_ms"], 4),
        "best_speedup_vs_k1": round(best["speedup_vs_k1"], 3),
    }
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS / "llm_multistep_bench.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\n[bench] wrote {out_path}")

    # ---- print table ----
    print(f"\n=== granite-speech LLM decode ({n_decode} tokens, prefill T={T}) ===")
    rows = []
    for r in config_results:
        rows.append([
            r["K"], f"{r['decode_loop_ms']:.2f}", f"{r['decode_loop_p90_ms']:.2f}",
            f"{r['per_token_ms']:.3f}", f"{r['tok_per_s']:.1f}",
            f"{r['single_replay_gpu_ms']:.3f}", r["n_replays"],
            f"{r['pure_gpu_ms']:.2f}", f"{r['speedup_vs_k1']:.2f}x",
            f"{r['cuda_wall_ratio']:.2f}",
        ])
    print(tabulate(
        rows,
        headers=["K", "loop_ms", "p90_ms", "per_tok", "tok/s",
                 "1rep_gpu", "n_rep", "pure_gpu", "speedup", "cuda/wall"],
        tablefmt="github",
    ))
    print(f"\n  best: K={best['K']} loop={best['decode_loop_ms']:.2f}ms "
          f"({best['speedup_vs_k1']:.2f}x vs K=1, {best['tok_per_s']:.1f} tok/s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
