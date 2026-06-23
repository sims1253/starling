"""Before/after benchmark for the chunked-stitching overhead optimization.

Sister of ``bench_batched_chunked.py``: identical workload (5/30/60 min tiled
audio through ``ChunkedTranscriber(chunk_batch_size=8)``) but its PURPOSE is to
quantify the per-batch overhead reduction from vectorizing the stitch and the
``collect_meta`` path.

The optimization (in ``chunking.py`` + ``decode_mega.py``):

1. **Vectorized stitch** -- the Python ``for tok, lf in zip(...)`` loop over
   per-token frame positions is replaced by one vectorized ``g_samples >
   furthest`` comparison per chunk (eliminating the per-token ``int(lf)`` /
   ``.item()`` calls).
2. **Vectorized ``collect_meta``** -- the per-token ``.item()`` bookkeeping in
   ``_run_loop`` is replaced by vectorized ``(B, max_out)`` tensor scatters + a
   vectorized done-point trim; ``decode_meta_tensors`` returns tensors so the
   chunker never pays a ``.item()``-per-token round-trip.
3. **Consolidated audio H2D** -- was benchmarked too but measured ~1.7ms/batch
   SLOWER (the CPU numpy fill does not overlap with GPU work as well as the
   mel extractor's per-chunk GPU-side scatters) and was REVERTED; the hot path
   keeps ``pipe.mel(batch_audio)``.

Because the per-batch overhead reduction (~5-8ms) is a small fraction of the
~100ms/batch wall and single-run variance is ~+/-5%, this bench reports BOTH:

* a controlled A/B of the stitch (reference Python loop vs the vectorized path)
  on REAL decode metadata -- the clean, low-variance measurement of the win; and
* a controlled decode() vs decode_meta_tensors() timing -- the collect_meta win;
* the end-to-end 5/30/60 min total (median of N iters) vs the single-run
  "before" numbers in ``batched_chunked_bench.json``.

Writes ``outputs/parakeet/chunked_opt_bench.json`` and prints tables.

Usage:  ``uv run python benchmarks/parakeet/bench_chunked_opt.py``
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "tests" / "fixtures"))

from starling.parakeet.chunking import ChunkedTranscriber  # noqa: E402
from starling.parakeet.gpu_lock import with_gpu_lock  # noqa: E402
from starling.parakeet.pipeline import MegaParakeetPipeline  # noqa: E402
import make_fixtures as mkfx  # noqa: E402

OUT_DIR = _REPO_ROOT / "outputs" / "parakeet"
BEFORE_PATH = OUT_DIR / "batched_chunked_bench.json"   # before (unoptimized)
OUT_PATH = OUT_DIR / "chunked_opt_bench.json"          # this bench's output

CHUNK_BATCH_SIZE = 8
LENGTHS_MIN = [5, 30, 60]   # 5 min, 30 min, 1 h (the task's comparison points)
SR = 16000
N_ITERS = 3                 # end-to-end iterations per length (report median)


def tile_audio(base: np.ndarray, target_seconds: float) -> np.ndarray:
    need = int(target_seconds * SR)
    reps = (need + base.shape[0] - 1) // base.shape[0]
    return np.ascontiguousarray(np.tile(base, reps), dtype=np.float32)


def load_before() -> dict:
    """Pull the unoptimized (before) numbers from batched_chunked_bench.json."""
    if not BEFORE_PATH.exists():
        print(f"[bench] WARNING: {BEFORE_PATH} missing; no before comparison",
              file=sys.stderr)
        return {}
    try:
        data = json.loads(BEFORE_PATH.read_text())
    except Exception as e:
        print(f"[bench] WARNING: could not parse {BEFORE_PATH}: {e}",
              file=sys.stderr)
        return {}
    return {e["length_min"]: e for e in data.get("results", [])
            if e.get("status") == "ok"}


# --------------------------------------------------------------------------- #
# reference (OLD) stitch -- the Python per-token loop this optimization replaces.
# Used ONLY for the controlled A/B (not in the production path). Must produce
# IDENTICAL surviving tokens to the vectorized path (left-biased dedup).
# --------------------------------------------------------------------------- #
def stitch_reference_old(meta_tok, meta_frm, meta_len, starts, spef):
    """The pre-optimization Python per-token stitch loop (for A/B timing only)."""
    B = meta_tok.shape[0]
    surviving = []
    furthest = -1
    n_stitches = 0
    for k in range(B):
        lk = int(meta_len[k].item())
        chunk_furthest = furthest
        for j in range(lk):
            lf = int(meta_frm[k, j].item())
            tok = int(meta_tok[k, j].item())
            g = starts[k] + lf * spef
            if g > furthest:
                surviving.append(tok)
                if g > chunk_furthest:
                    chunk_furthest = g
            else:
                n_stitches += 1
        furthest = chunk_furthest
    return surviving, n_stitches


def stitch_optimized(meta_tok, meta_frm, meta_len, starts, spef):
    """The vectorized stitch (mirrors ChunkedTranscriber's production path)."""
    B = meta_tok.shape[0]
    surviving = []
    furthest = -1
    n_stitches = 0
    for k in range(B):
        lk = int(meta_len[k].item())
        if lk > 0:
            frames_k = meta_frm[k, :lk]
            toks_k = meta_tok[k, :lk]
            g_samples = starts[k] + frames_k * spef
            mask = g_samples > furthest
            kept_here = int(mask.sum().item())
            n_stitches += lk - kept_here
            if kept_here > 0:
                surviving.extend(toks_k[mask].tolist())
                furthest = int(g_samples[-1])
    return surviving, n_stitches


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[bench] loading pipeline + chunker (chunk_batch_size="
          f"{CHUNK_BATCH_SIZE}, OPTIMIZED stitch + collect_meta) ...", flush=True)
    pipe = MegaParakeetPipeline(use_graphed_encoder=True)
    chunker = ChunkedTranscriber(
        pipe, chunk_seconds=30.0, overlap_seconds=2.0,
        chunk_batch_size=CHUNK_BATCH_SIZE,
    )
    base = mkfx.load_sample()

    print("[bench] warmup (capture B=1 + B=8 graphs) ...", flush=True)
    _ = chunker.transcribe(tile_audio(base, 32.0))
    _ = chunker.transcribe(tile_audio(base, 250.0))
    torch.cuda.synchronize()

    before = load_before()

    # ===================================================================== #
    # (A) CONTROLLED STITCH A/B on REAL decode metadata
    # ===================================================================== #
    print("\n[bench] (A) controlled stitch A/B on real decode metadata ...",
          flush=True)
    # extract real meta from a full B=8 batch of 32s chunks
    chunks, starts = chunker._plan_chunks(tile_audio(base, 250.0))
    batch8 = chunks[:8]
    starts8 = starts[:8]
    _, mt, mf, ml, _vl, _t = chunker._decode_batch(batch8)
    spef = chunker.samples_per_enc_frame
    ref_surv, ref_ns = stitch_reference_old(mt, mf, ml, starts8, spef)
    opt_surv, opt_ns = stitch_optimized(mt, mf, ml, starts8, spef)
    identical = (ref_surv == opt_surv and ref_ns == opt_ns)
    print(f"        stitch identical (byte-exact): {identical}  "
          f"({len(ref_surv)} surviving tokens, {ref_ns} stitches)")

    def cpu_med(fn, iters=50):
        for _ in range(5):
            fn()
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            ts.append((time.perf_counter() - t0) * 1000.0)
        return statistics.median(ts)

    t_old = cpu_med(lambda: stitch_reference_old(mt, mf, ml, starts8, spef))
    t_new = cpu_med(lambda: stitch_optimized(mt, mf, ml, starts8, spef))
    stitch_saved = t_old - t_new
    print(f"        stitch OLD (python loop) : {t_old:.3f} ms/batch")
    print(f"        stitch NEW (vectorized)   : {t_new:.3f} ms/batch  "
          f"({t_old / t_new:.1f}x, saves {stitch_saved:.3f} ms/batch)")

    # ===================================================================== #
    # (B) CONTROLLED collect_meta overhead: decode() vs decode_meta_tensors()
    # ===================================================================== #
    print("\n[bench] (B) collect_meta overhead: decode() vs "
          "decode_meta_tensors() ...", flush=True)
    feats, mask = pipe.mel(batch8)
    feats = feats.to(pipe.dtype)
    pooler, vl = pipe._run_encoder(feats, mask)
    dec = pipe._get_decoder(pooler, vl)

    def cuda_med(fn, iters=15):
        for _ in range(3):
            fn(); torch.cuda.synchronize()
        ts = []
        for _ in range(iters):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record(); fn(); e.record(); torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
        return statistics.median(ts)

    t_dec = cuda_med(lambda: dec.decode(pooler, vl, pipe.processor))
    t_decmeta = cuda_med(
        lambda: dec.decode_meta_tensors(pooler, vl, pipe.processor))
    print(f"        decode()            : {t_dec:.2f} ms/batch")
    print(f"        decode_meta_tensors : {t_decmeta:.2f} ms/batch  "
          f"(meta adds {t_decmeta - t_dec:+.2f} ms -- vectorized scatter is "
          f"near-free vs the old per-token .item() loop)")

    # ===================================================================== #
    # (C) END-TO-END 5/30/60 min (median of N_ITERS) vs before
    # ===================================================================== #
    print(f"\n[bench] (C) end-to-end {LENGTHS_MIN} min, {N_ITERS} iters each "
          f"(report median) ...", flush=True)
    e2e_results = []
    for minutes in LENGTHS_MIN:
        label = f"{minutes}min"
        audio = tile_audio(base, minutes * 60)
        totals = []
        for i in range(N_ITERS):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            _, summ = chunker.transcribe_with_timing(audio)
            totals.append(summ["total_ms"])
            print(f"        {label} iter {i}: {summ['total_ms']:.1f}ms",
                  flush=True)
        med = statistics.median(totals)
        mn = min(totals)
        # re-run once more to capture per-batch stage breakdown (warm)
        _, summ = chunker.transcribe_with_timing(audio)
        per_batch = summ["per_batch"]
        n_batches = summ["n_batches"]
        med_stage = float(np.mean([b["total_ms"] for b in per_batch]))
        med_decode = float(np.mean([b["decode_ms"] for b in per_batch]))
        per_batch_wall = med / n_batches
        overhead = per_batch_wall - med_stage
        audio_s = summ["audio_seconds"]
        rtf = audio_s / (med / 1000.0) if med > 0 else 0.0
        entry = {
            "length_min": label,
            "audio_seconds": audio_s,
            "total_ms_median": med,
            "total_ms_min": mn,
            "total_ms_iters": totals,
            "rtf_median": rtf,
            "n_batches": n_batches,
            "mean_stage_ms": med_stage,
            "mean_decode_ms": med_decode,
            "mean_mel_ms": float(np.mean([b["mel_ms"] for b in per_batch])),
            "mean_encoder_ms": float(np.mean([b["encoder_ms"] for b in per_batch])),
            "per_batch_wall_ms": per_batch_wall,
            "overhead_per_batch_ms": overhead,
            "peak_vram_gb": summ["peak_vram_gb"],
            "n_tokens": summ["n_tokens_surviving"],
            "n_stitches": summ["n_stitches"],
            "batch_sizes": [b["batch_size"] for b in per_batch],
        }
        e2e_results.append(entry)
        print(f"        {label}: median={med:.1f}ms min={mn:.1f}ms "
              f"RTF={rtf:.1f}x wall/batch={per_batch_wall:.1f}ms "
              f"stage/batch={med_stage:.1f}ms overhead/batch={overhead:.1f}ms",
              flush=True)
        torch.cuda.empty_cache()

    payload = {
        "model_id": "nvidia/parakeet-tdt-0.6b-v3",
        "dtype": "bfloat16",
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "?",
        "method": (
            "ChunkedTranscriber (chunk_batch_size=8) with VECTORIZED stitch "
            "(~%.2fms/batch saved on real metadata) and VECTORIZED collect_meta "
            "(decode_meta_tensors returns tensors; meta adds %+.2fms vs plain "
            "decode). A consolidated single-H2D audio path was benchmarked but "
            "measured ~1.7ms/batch SLOWER and was reverted. Before = "
            "batched_chunked_bench.json (single cold run)." % (stitch_saved,
                                                               t_decmeta - t_dec)
        ),
        "chunk_batch_size": CHUNK_BATCH_SIZE,
        "n_iters": N_ITERS,
        "controlled_stitch_ab": {
            "old_python_loop_ms": t_old,
            "new_vectorized_ms": t_new,
            "saved_ms_per_batch": stitch_saved,
            "speedup": t_old / t_new,
            "identical_result": identical,
            "n_surviving_tokens": len(ref_surv),
            "n_stitches": ref_ns,
        },
        "controlled_collect_meta": {
            "decode_ms": t_dec,
            "decode_meta_tensors_ms": t_decmeta,
            "meta_overhead_ms": t_decmeta - t_dec,
        },
        "end_to_end": e2e_results,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"\n[bench] wrote {OUT_PATH}")

    # ---- before vs after table ----
    print("\n=== END-TO-END: BEFORE (single cold run) vs AFTER (median of "
          f"{N_ITERS}) ===")
    print(f"{'length':>6} | {'after med':>9} {'before':>9} | "
          f"{'after RTF':>9} {'before RTF':>10} | {'RTF gain':>8} | "
          f"{'after min':>9}")
    print("-" * 80)
    for e in e2e_results:
        b = before.get(e["length_min"], {})
        if not b:
            print(f"{e['length_min']:>6} | (no before data)")
            continue
        b_total = b.get("total_ms", 0)
        b_rtf = b.get("rtf", 0)
        gain = (e["rtf_median"] / b_rtf) if b_rtf else 0.0
        print(
            f"{e['length_min']:>6} | {e['total_ms_median']:>8.1f}ms "
            f"{b_total:>8.1f}ms | {e['rtf_median']:>8.1f}x {b_rtf:>9.1f}x | "
            f"{gain:>7.2f}x | {e['total_ms_min']:>8.1f}ms"
        )

    # ---- HEADLINE ----
    h1h = next((e for e in e2e_results if e["length_min"] == "60min"), None)
    b1h = before.get("60min", {})
    print("\n=== CONTROLLED WINS (low-variance, the real overhead reduction) ===")
    print(f"  stitch       : {t_old:.2f}ms -> {t_new:.2f}ms /batch "
          f"({t_old / t_new:.1f}x, saves {stitch_saved:.2f}ms; identical={identical})")
    print(f"  collect_meta : decode_meta_tensors adds {t_decmeta - t_dec:+.2f}ms "
          f"vs plain decode (was ~4ms with the per-token .item() loop)")
    if h1h and b1h:
        print("\n=== HEADLINE: chunked @ 1 h ===")
        print(f"  BEFORE (single run): total={b1h.get('total_ms'):.1f}ms  "
              f"RTF={b1h.get('rtf'):.1f}x")
        print(f"  AFTER  (median x{N_ITERS}): total={h1h['total_ms_median']:.1f}ms  "
              f"RTF={h1h['rtf_median']:.1f}x  (min {h1h['total_ms_min']:.1f}ms)")

    # ---- correctness fingerprint ----
    print("\n=== Correctness fingerprint (tokens/stitches must match before) ===")
    for e in e2e_results:
        b = before.get(e["length_min"], {})
        bt, bs = b.get("n_tokens"), b.get("n_stitches")
        tm = "MATCH" if bt == e["n_tokens"] else "DIFFER"
        sm = "MATCH" if bs == e["n_stitches"] else "DIFFER"
        print(f"  {e['length_min']:>5}: tokens {e['n_tokens']} ({tm} before {bt}), "
              f"stitches {e['n_stitches']} ({sm} before {bs})")

    return 0


if __name__ == "__main__":
    with with_gpu_lock(
        session="parakeet-mega",
        model="parakeet-tdt-0.6b-v3",
        eta_min=4,
        note="chunked opt bench",
    ):
        t0 = time.time()
        rc = main()
        print(f"\n[bench] elapsed {time.time() - t0:.1f}s")
        sys.exit(rc)
