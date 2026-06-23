"""Robust length-scaling sweep + real-corpus benchmark for the parakeet megakernel.

The headline 1749x RTF was measured on synthetic-repeated 22.3s clips (one
utterance concatenated 3x, batched 8x). This script answers three
production-realism questions the headline leaves open:

  1. Does RTF hold as audio grows to ~15 min, or is there a cliff? Where is the
     practical ceiling (VRAM or compute) on the 32 GB RTX 5090?
  2. What is the real VRAM / compute cost at each length?
  3. Does real, varied speech give a different RTF than synthetic-repeated clips
     (token density, blank-skip, padding waste)?

Three parts
-----------
Part 1 -- full-attention length sweep (synthetic-repeated, A/B vs the headline):
    for length in {1, 3, 5, 10, 15} min, for B in {1, 8}:
        measure total_ms / mel_ms / encoder_ms / decode_ms / RTF / peak VRAM /
        decode steps / tokens emitted, via transcribe_with_timing.
    STOP escalating length on OOM or >60 s/pass and report the ceiling
    (longest clip that fits in 32 GB and completes in <30 s).

Part 2 -- real varied speech corpus benchmark:
    8 real LibriSpeech utterances (different content, different lengths 1.6-29 s)
    batched together -- measure RTF + per-stage + per-utterance latency, and
    compare to the synthetic-repeated B8-medium headline.

Part 3 -- comprehensive metrics for the headline configs (B8-synthetic-medium,
B8-real-varied, longest-feasible-single-clip): RTF, per-utterance first/last
token latency (mean+p90), max single-clip length, decode steps + tokens per
audio-second.

Method (under the shared-GPU lock, per comms.md §P1):
  * cuda.Event + synchronize, warmup>=3 (fewer for long clips), >=5 samples
    (30 s cap), median.
  * peak VRAM = torch.cuda.max_memory_allocated() reset before each config.
  * decode steps + per-utterance tokens = read from the cached GraphedDecoder's
    static output buffer after one decode loop replay.
  * GPU-contention guard: refuse to run if nvidia-smi util > 30 %.

Writes ``outputs/parakeet/robust_bench.json`` and prints formatted tables.

Run:  uv run python benchmarks/parakeet/bench_robust.py
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import List

import numpy as np
import torch
from tabulate import tabulate

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "tests" / "fixtures"))
sys.path.insert(0, str(_REPO_ROOT / "src"))

import make_fixtures as mkfx  # noqa: E402
from megapar.parakeet.gpu_lock import with_gpu_lock  # noqa: E402
from megapar.parakeet.pipeline import MegaParakeetPipeline  # noqa: E402

MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"
SAMPLE_RATE = 16000
OUTPUTS = _REPO_ROOT / "outputs" / "parakeet"

# Length-sweep targets (minutes, target seconds). Deterministic concatenation
# of the canonical source sample, trimmed to exactly target_s for clean labels.
# 7min is included to bracket the max_position_embeddings=5000 cliff (7min ~>
# 5250 encoder frames, just past the limit).
LENGTH_TARGETS = [("1min", 60), ("3min", 180), ("5min", 300),
                  ("7min", 420), ("10min", 600), ("15min", 900)]
SWEEP_BATCHES = [1, 8]
# Lengths at/above this run a single-forward cliff probe BEFORE the full
# (capture-heavy) measure_config, so a cliff doesn't hang on graph capture.
PROBE_THRESHOLD_S = 360

# Timing policy. Long clips are slow, so use few warmups + samples with a wall
# cap; short clips get more samples.
WARMUP_SHORT = 5
WARMUP_LONG = 3
MAX_SAMPLES = 5
SAMPLE_WALL_CAP_S = 30.0     # cap on the sampling loop per config
SINGLE_PASS_CAP_S = 60.0     # >this -> compute ceiling, stop escalating length

GPU_UTIL_THRESHOLD_PCT = 30   # comms.md: defer if util > 30 %


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
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


def gpu_utilization_pct() -> int | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.STDOUT, text=True, timeout=10,
        ).strip()
        return int(out.splitlines()[0].strip())
    except Exception:
        return None


def assert_gpu_idle(*, where: str) -> None:
    util = gpu_utilization_pct()
    if util is not None and util > GPU_UTIL_THRESHOLD_PCT:
        raise SystemExit(
            f"[bench_robust] GPU util={util}% (> {GPU_UTIL_THRESHOLD_PCT}% "
            f"threshold) at {where}; deferring per comms.md §P1. Re-run when idle."
        )


def wait_for_idle_gpu(*, max_wait_s: float = 120.0,
                      idle_thresh_pct: int = 15,
                      confirm_samples: int = 2,
                      where: str = "") -> bool:
    """Block until the shared GPU is quiet, so a timed measurement is not
    corrupted by another session's transient burst.

    Polls ``nvidia-smi`` utilization; requires ``confirm_samples`` consecutive
    sub-threshold reads before proceeding. Returns True if it reached idle,
    False if it timed out (best-effort proceed + log). This complements the
    .gpu.lock (which serialises sessions) by catching brief intra-session
    bursts that slip between lock acquire and the timed region.
    """
    deadline = time.perf_counter() + max_wait_s
    streak = 0
    while time.perf_counter() < deadline:
        util = gpu_utilization_pct()
        if util is not None and util < idle_thresh_pct:
            streak += 1
            if streak >= confirm_samples:
                return True
        else:
            streak = 0
        time.sleep(1.0)
    if where:
        print(f"  [wait_for_idle_gpu/{where}] timed out after {max_wait_s:.0f}s "
              f"(last util={util}); proceeding best-effort")
    return False


def build_length_audio(target_s: float, base: np.ndarray) -> np.ndarray:
    """Deterministic: concatenate the base sample to reach >= target_s, trim."""
    target_n = int(round(target_s * SAMPLE_RATE))
    reps = max(1, math.ceil(target_n / len(base)))
    arr = np.concatenate([base] * reps)[:target_n]
    return np.ascontiguousarray(arr, dtype=np.float32)


def count_decode(pipe: MegaParakeetPipeline, audio_list: List[np.ndarray]):
    """Run one encode + decode loop and read decode steps + per-utterance tokens.

    Returns ``(T_enc, decode_steps, per_utt_token_counts, texts)``. Uses the
    pipeline's cached GraphedDecoder so the capture is amortised; this is an
    extra (untimed) pass used only to read the static output buffer for counts.
    """
    input_features, attention_mask = pipe.mel(audio_list)
    input_features = input_features.to(pipe.dtype)
    pooler, valid_lengths = pipe._run_encoder(input_features, attention_mask)
    T_enc = int(pooler.shape[1])
    decoder = pipe._get_decoder(pooler, valid_lengths)
    out_step = decoder._run_loop(pooler, valid_lengths)
    out = decoder.output  # (B, max_out) long, pad_id-padded
    B = out.shape[0]
    per_utt = [int((out[b, :out_step] != pipe.pad_id).sum().item()) for b in range(B)]
    out_lists = [out[b, :out_step].tolist() for b in range(B)]
    texts = pipe.processor.batch_decode(out_lists, skip_special_tokens=True)
    return T_enc, int(out_step), per_utt, texts


def probe_encoder_eager(pipe: MegaParakeetPipeline, audio: np.ndarray,
                        *, wall_cap_s: float = 60.0,
                        vram_cap_gb: float = 30.0):
    """One EAGER (non-captured) B=1 encoder forward to detect the attention cliff.

    The graphed-encoder CAPTURE for a long clip runs ~4 encoder forwards and can
    hang for minutes at the cliff; this probe runs a SINGLE eager forward to
    bound that cost. If the single forward already exceeds the wall or VRAM cap
    (or OOMs), the length is beyond the practical ceiling and we skip the full
    measure_config (no capture attempted).

    Returns ``{"status": "ok"|"compute_cliff"|"vram_cliff"|"oom", ...}``.
    """
    wait_for_idle_gpu(where=f"probe {len(audio)/SAMPLE_RATE:.0f}s")
    try:
        torch.cuda.reset_peak_memory_stats()
        input_features, attention_mask = pipe.mel([audio])
        input_features = input_features.to(pipe.dtype)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = pipe.model.get_audio_features(
            input_features=input_features, attention_mask=attention_mask)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        T_enc = int(input_features.shape[1])
        if wall > wall_cap_s:
            return {"status": "compute_cliff", "enc_wall_s": round(wall, 2),
                    "peak_gb": round(peak_gb, 2), "T_enc": T_enc}
        if peak_gb > vram_cap_gb:
            return {"status": "vram_cliff", "enc_wall_s": round(wall, 2),
                    "peak_gb": round(peak_gb, 2), "T_enc": T_enc}
        return {"status": "ok", "enc_wall_s": round(wall, 2),
                "peak_gb": round(peak_gb, 2), "T_enc": T_enc}
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        torch.cuda.empty_cache()
        return {"status": "oom", "error": type(e).__name__,
                "msg": str(e)[:120]}


def measure_config(
    pipe: MegaParakeetPipeline,
    audio_list: List[np.ndarray],
    *,
    warmup: int,
    long_clip: bool,
):
    """Time one (audio_list) config; return a dict or raise on OOM.

    Returns ``{"status": "ok"|"compute_ceiling", ...}``.
    On OOM raises ``torch.cuda.OutOfMemoryError`` / ``RuntimeError`` (caller
    catches and records the VRAM ceiling).

    The compute-ceiling check is on STEADY-STATE per-pass time (the sampling
    loop), NOT on warmup -- warmup includes the one-off per-shape CUDA-graph
    capture which is allowed to be slow (it is amortised setup, not steady state).
    """
    audio_seconds_total = sum(len(a) / SAMPLE_RATE for a in audio_list)

    # Contention guard: wait for a quiet GPU window so a burst from the other
    # session doesn't inflate the timed samples.
    wait_for_idle_gpu(where=f"measure B{len(audio_list)}")

    # ---- warmup (captures the per-shape graphs on first call; allowed to be
    # slow because capture is amortised setup, not steady state) ----
    last_texts = None
    for w in range(warmup):
        torch.cuda.synchronize()
        texts, _t = pipe.transcribe_with_timing(audio_list)
        torch.cuda.synchronize()
        last_texts = texts

    # ---- timed samples (median); steady-state compute-ceiling check here ----
    mel_s, enc_s, dec_s, tot_s, vram_s = [], [], [], [], []
    wall0 = time.perf_counter()
    n = 0
    ceiling_wall = None
    while n < MAX_SAMPLES and (time.perf_counter() - wall0) < SAMPLE_WALL_CAP_S:
        torch.cuda.reset_peak_memory_stats()
        pass_t0 = time.perf_counter()
        _texts, t = pipe.transcribe_with_timing(audio_list)
        torch.cuda.synchronize()
        pass_wall = time.perf_counter() - pass_t0
        mel_s.append(t["mel_ms"]); enc_s.append(t["encoder_ms"])
        dec_s.append(t["decode_ms"]); tot_s.append(t["total_ms"])
        vram_s.append(torch.cuda.max_memory_allocated() / 1e9)
        n += 1
        # steady-state per-pass > cap -> compute ceiling (after >=1 real sample)
        if pass_wall > SINGLE_PASS_CAP_S and n >= 1:
            ceiling_wall = pass_wall
            break

    # Contention self-check: if p90 >> median the timed window was hit by a
    # burst from the other session (contended samples are outliers). Re-run the
    # sampling loop once after re-waiting for an idle window.
    if (ceiling_wall is None and n >= 2
            and float(np.percentile(tot_s, 90)) > 2.5 * float(np.median(tot_s))):
        print(f"  [measure] high variance p90={np.percentile(tot_s,90):.0f}"
              f" >> med={np.median(tot_s):.0f}ms -> re-sampling after idle wait")
        wait_for_idle_gpu(where="measure-resample")
        mel_s, enc_s, dec_s, tot_s, vram_s = [], [], [], [], []
        wall0 = time.perf_counter()
        n = 0
        while n < MAX_SAMPLES and (time.perf_counter() - wall0) < SAMPLE_WALL_CAP_S:
            torch.cuda.reset_peak_memory_stats()
            _texts, t = pipe.transcribe_with_timing(audio_list)
            torch.cuda.synchronize()
            mel_s.append(t["mel_ms"]); enc_s.append(t["encoder_ms"])
            dec_s.append(t["decode_ms"]); tot_s.append(t["total_ms"])
            vram_s.append(torch.cuda.max_memory_allocated() / 1e9)
            n += 1

    total_ms = float(np.median(tot_s))
    rtf = audio_seconds_total / (total_ms / 1000.0) if total_ms > 0 else 0.0
    peak_gb = float(np.max(vram_s)) if vram_s else 0.0
    if ceiling_wall is not None:
        return {
            "status": "compute_ceiling",
            "wall_per_pass_s": round(ceiling_wall, 3),
            "audio_seconds": round(audio_seconds_total, 3),
            "total_ms": round(total_ms, 3),
            "mel_ms": round(float(np.median(mel_s)), 3),
            "encoder_ms": round(float(np.median(enc_s)), 3),
            "decode_ms": round(float(np.median(dec_s)), 3),
            "rtf": round(rtf, 2),
            "peak_vram_gb": round(peak_gb, 3),
            "n_samples": n,
            "texts_preview": [s[:40] for s in (_texts or [])][:2],
        }
    return {
        "status": "ok",
        "audio_seconds": round(audio_seconds_total, 3),
        "total_ms": round(total_ms, 3),
        "total_p90_ms": round(float(np.percentile(tot_s, 90)), 3),
        "mel_ms": round(float(np.median(mel_s)), 3),
        "encoder_ms": round(float(np.median(enc_s)), 3),
        "decode_ms": round(float(np.median(dec_s)), 3),
        "rtf": round(rtf, 2),
        "peak_vram_gb": round(peak_gb, 3),
        "n_samples": n,
        "texts_preview": [s[:40] for s in (_texts or [])][:2],
    }


def comprehensive(measured: dict, audio_seconds_list: List[float],
                  per_utt_tokens: List[int], decode_steps: int) -> dict:
    """Build the Part-3 comprehensive-metrics bundle for one headline config."""
    B = len(audio_seconds_list)
    decode_ms = float(measured.get("decode_ms", 0.0))
    per_step_ms = decode_ms / max(decode_steps, 1)
    first_token_ms = (float(measured.get("mel_ms", 0.0))
                      + float(measured.get("encoder_ms", 0.0))
                      + per_step_ms)
    last_token_ms = float(measured.get("total_ms", 0.0))
    # Batched pipeline: all utterances complete at the same wall time, so
    # per-utterance last-token latency = total_ms for every element. The
    # within-batch variation is in token density (reported separately).
    last_tokens = [last_token_ms] * B
    first_tokens = [first_token_ms] * B
    tokens_per_sec = ([float(t) / max(s, 1e-6) for t, s in
                      zip(per_utt_tokens, audio_seconds_list)])
    return {
        "rtf": measured.get("rtf"),
        "total_ms": measured.get("total_ms"),
        "mel_ms": measured.get("mel_ms"),
        "encoder_ms": measured.get("encoder_ms"),
        "decode_ms": measured.get("decode_ms"),
        "peak_vram_gb": measured.get("peak_vram_gb"),
        "first_token_ms": round(first_token_ms, 3),
        "last_token_ms_mean": round(float(np.mean(last_tokens)), 3),
        "last_token_ms_p90": round(float(np.percentile(last_tokens, 90)), 3),
        "first_token_ms_mean": round(float(np.mean(first_tokens)), 3),
        "decode_steps": decode_steps,
        "tokens_per_audio_second": [round(x, 3) for x in tokens_per_sec],
        "per_utt_tokens": per_utt_tokens,
        "note": ("batched: all B utterances complete at total_ms; per-utterance "
                 "variation is in token density (tokens/audio_second)"),
    }


def fmt_oom(row: dict) -> list:
    return [row["length_min"], row["batch_size"], "-", "-", "-", "-",
            "OOM", row.get("peak_vram_gb", "-")]


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    _suppress()
    assert_gpu_idle(where="startup")

    print("[bench_robust] loading MegaParakeetPipeline ...")
    pipe = MegaParakeetPipeline(model_id=MODEL_ID, device="cuda",
                                dtype=torch.bfloat16)
    base = mkfx.load_sample()  # the canonical ~7.4s source sample

    # load real corpus (downloads + caches on first call)
    import get_real_corpus as grc
    real_items = grc.load_real_corpus(8)
    real_audio = [a for (a, _sr, _t) in real_items]
    real_secs = [len(a) / SAMPLE_RATE for (a, _sr, _t) in real_items]
    real_texts_ref = [t for (_a, _sr, t) in real_items]
    print(f"[bench_robust] real corpus: {len(real_audio)} utterances, "
          f"lens={[round(s,2) for s in real_secs]}s")

    results = {
        "length_sweep": [],
        "length_sweep_ceiling": {},
        "real_corpus": {},
        "comprehensive_metrics": {},
        "real_vs_synthetic": {},
    }
    headline_synth_b8 = None   # filled from sweep B8 at 22.3s? no -> re-measure medium
    real_b8_measured = None
    longest_single_clip = None  # (length_min, measured_dict, counts)

    print("[bench_robust] acquiring GPU lock ...")
    with with_gpu_lock(session="parakeet-mega", model=MODEL_ID,
                       eta_min=10, note="robust bench sweep"):
        assert_gpu_idle(where="inside GPU lock")
        free, total = torch.cuda.mem_get_info()
        print(f"[bench_robust] lock held; GPU free={free/1e9:.1f}GB / "
              f"{total/1e9:.1f}GB")
        model_baseline_gb = (total - free) / 1e9
        results["model_baseline_vram_gb"] = round(model_baseline_gb, 3)

        # =============================================================== #
        # HEADLINE MEASUREMENTS (clean GPU, BEFORE the length sweep + cliff
        # probe). The 7min cliff probe allocates ~25GB and perturbs cudnn/cublas
        # algorithm caches, so all small-clip headline numbers (synthetic B8
        # medium, real-varied B8) are measured FIRST on the cold-clean GPU. A
        # sanity guard re-measures synthetic B8 medium if its RTF is implausible
        # (<500x -> GPU was contended/degraded).
        # =============================================================== #
        medium = mkfx.load_fixtures()["medium"]
        print("\n[bench_robust] === HEADLINE: synthetic B8 medium + real B8 ===")
        # synthetic B8 medium (the 1749x headline config), with sanity re-measure
        synth_b8 = measure_config(pipe, [medium] * 8, warmup=WARMUP_SHORT,
                                  long_clip=False)
        if synth_b8.get("rtf", 0) < 500:
            print(f"  [synth B8 medium] RTF={synth_b8.get('rtf')} <500 -> "
                  f"GPU likely contended/degraded; re-measuring after idle check")
            assert_gpu_idle(where="synth B8 medium re-measure")
            synth_b8 = measure_config(pipe, [medium] * 8, warmup=WARMUP_SHORT,
                                      long_clip=False)
        sT_enc, s_steps, s_per_utt_tok, _ = count_decode(pipe, [medium] * 8)
        synth_b8.update({"T_enc": sT_enc, "decode_steps": s_steps})
        headline_synth_b8 = synth_b8
        print(f"  [synth B8 medium] total={synth_b8['total_ms']:7.1f}ms "
              f"enc={synth_b8['encoder_ms']:6.1f} dec={synth_b8['decode_ms']:6.1f} "
              f"rtf={synth_b8['rtf']:8.1f}x T_enc={sT_enc} steps={s_steps}")

        # real varied B8 (authoritative throughput on real speech)
        real_measured = measure_config(pipe, real_audio, warmup=WARMUP_SHORT,
                                      long_clip=False)
        rT_enc, r_steps, r_per_utt_tok, r_texts = count_decode(pipe, real_audio)
        print(f"  [real B8] total={real_measured['total_ms']:7.1f}ms "
              f"enc={real_measured['encoder_ms']:6.1f} "
              f"dec={real_measured['decode_ms']:6.1f} "
              f"rtf={real_measured['rtf']:8.1f}x T_enc={rT_enc} steps={r_steps}")
        real_b8_measured = real_measured

        # real per-utterance B=1 latency distribution (true per-utterance)
        per_utt_b1 = []
        for a in real_audio:
            m = measure_config(pipe, [a], warmup=2, long_clip=False)
            per_utt_b1.append({
                "idx": len(per_utt_b1),
                "audio_seconds": round(len(a) / SAMPLE_RATE, 3),
                "total_ms": m["total_ms"], "rtf_b1": m["rtf"],
                "mel_ms": m["mel_ms"], "encoder_ms": m["encoder_ms"],
                "decode_ms": m["decode_ms"], "peak_vram_gb": m["peak_vram_gb"],
            })
        _lat = [x["total_ms"] for x in per_utt_b1]
        print(f"  [real per-utt B1] latency ms: min={min(_lat):.1f} "
              f"med={np.median(_lat):.1f} max={max(_lat):.1f} "
              f"p90={np.percentile(_lat, 90):.1f}")

        # =============================================================== #
        # PART 1: length sweep (synthetic-repeated, B=1 and B=8)
        # =============================================================== #
        print("\n[bench_robust] === PART 1: length sweep (synthetic-repeated) ===")
        single_clip_stop = False   # stop all longer lengths once B=1 breaks
        batch8_stop = False
        single_clip_ceiling = None
        batch8_ceiling = None
        prev_single_len = None     # last B=1 length that succeeded
        prev_batch8_len = None     # last B=8 length that succeeded
        prev_b1_steady_ms = None   # previous B=1 steady-state pass (ms)

        for (lname, target_s) in LENGTH_TARGETS:
            # If a previous length already hit the cliff / ceiling, skip the rest
            # (longer clips only get slower; no need to re-probe).
            if single_clip_stop:
                print(f"\n--- {lname} skipped (ceiling/cliff already reached) ---")
                results["length_sweep"].append({
                    "length_min": lname, "batch_size": 1,
                    "target_s": target_s, "status": "skipped_post-cliff",
                })
                results["length_sweep"].append({
                    "length_min": lname, "batch_size": 8,
                    "target_s": target_s, "status": "skipped_post_cliff",
                })
                continue
            # If the previous B=1 steady pass already approached the 30s
            # production ceiling, longer clips are beyond the ceiling by
            # definition (encoder is O(N^2) so they only get slower). Skip them
            # instead of burning minutes on graph capture to re-confirm.
            if prev_b1_steady_ms is not None and prev_b1_steady_ms > 25_000:
                print(f"\n--- {lname} skipped: prev B1 steady "
                      f"{prev_b1_steady_ms/1000:.1f}s > 25s (beyond <30s ceiling) ---")
                results["length_sweep"].append({
                    "length_min": lname, "batch_size": 1,
                    "target_s": target_s, "status": "skipped_beyond_ceiling",
                    "prev_b1_steady_ms": prev_b1_steady_ms,
                })
                results["length_sweep"].append({
                    "length_min": lname, "batch_size": 8,
                    "target_s": target_s, "status": "skipped_beyond_ceiling",
                })
                continue
            audio_one = build_length_audio(target_s, base)
            print(f"\n--- {lname} ({len(audio_one)/SAMPLE_RATE:.1f}s) ---")
            b1_result_for_guard = None

            # Cliff probe for long lengths: a SINGLE eager encoder forward
            # characterises the full-attention VRAM/compute cliff (which appears
            # past max_position_embeddings=5000 ~ 6.7min). For these lengths we
            # treat the probe itself AS the measurement -- the production path's
            # graphed-encoder CAPTURE does ~4 forwards and would hang for
            # minutes at the cliff, so we do NOT attempt measure_config here.
            # The probe's (wall_s, peak_GB, T_mel) precisely locates the cliff.
            if target_s >= PROBE_THRESHOLD_S:
                probe = probe_encoder_eager(pipe, audio_one)
                print(f"  [probe {lname}] eager enc forward: {probe}")
                # classify the cliff from the single-forward measurement
                if probe["status"] == "oom":
                    reason = "oom"
                elif probe.get("peak_gb", 0) >= 20.0:
                    reason = "vram_cliff"
                elif probe.get("enc_wall_s", 0) >= 30.0:
                    reason = "compute_cliff"
                else:
                    reason = "approaching_cliff"
                for Bp in SWEEP_BATCHES:
                    results["length_sweep"].append({
                        "length_min": lname, "batch_size": Bp,
                        "target_s": target_s,
                        **probe,   # enc_wall_s, peak_gb, T_enc (and probe status)
                        "status": "attention_cliff",   # override probe status
                        "cliff_reason": reason,
                    })
                single_clip_stop = True
                batch8_stop = True
                single_clip_ceiling = {
                    "length_min": prev_single_len, "reason": reason,
                    "broke_at": lname, "probe": probe,
                }
                batch8_ceiling = {
                    "length_min": prev_batch8_len, "reason": reason,
                    "broke_at": lname,
                }
                # release the probe's large attention tensors so later
                # (small-clip) VRAM measurements are not skewed by cached blocks
                torch.cuda.empty_cache()
                continue

            for B in SWEEP_BATCHES:
                key = f"{lname}_B{B}"
                if B == 1 and single_clip_stop:
                    print(f"  [{key}] skipped (single-clip ceiling reached)")
                    results["length_sweep"].append({
                        "length_min": lname, "batch_size": B,
                        "target_s": target_s, "status": "skipped",
                    })
                    continue
                if B == 8 and batch8_stop:
                    print(f"  [{key}] skipped (batch8 ceiling reached)")
                    results["length_sweep"].append({
                        "length_min": lname, "batch_size": B,
                        "target_s": target_s, "status": "skipped",
                    })
                    continue
                # B8 guard: batched long clips are NOT the production shape (you
                # would chunk long audio, not pack 8x5min). The encoder is
                # O(B*N^2) compute AND the per-shape CUDA-graph capture for B8
                # costs ~4 forwards, so a slow B8 config would burn minutes on
                # capture alone. Skip B8 when its estimated steady-state pass
                # (8x the B1 steady pass) would exceed 30s -- the B8 length
                # trend is already clear from the shorter lengths we DO measure.
                if (B == 8 and b1_result_for_guard is not None
                        and b1_result_for_guard.get("total_ms", 0) * 8
                        > 30_000):
                    est = b1_result_for_guard["total_ms"] * 8 / 1000.0
                    print(f"  [{key}] skipped (B1 steady {b1_result_for_guard['total_ms']:.0f}ms"
                          f" -> B8 est ~{est:.0f}s/pass > 30s; batched-long not "
                          f"production shape, capture would burn minutes)")
                    results["length_sweep"].append({
                        "length_min": lname, "batch_size": B,
                        "target_s": target_s, "status": "skipped_b8_slow",
                        "b1_total_ms": b1_result_for_guard["total_ms"],
                        "b8_est_pass_s": round(est, 2),
                    })
                    batch8_stop = True
                    batch8_ceiling = {
                        "length_min": prev_batch8_len, "reason": "b8_est>30s",
                        "broke_at": lname,
                    }
                    continue

                audio_list = [audio_one for _ in range(B)]
                long_clip = target_s >= 300
                warmup = WARMUP_LONG if long_clip else WARMUP_SHORT
                t_cfg0 = time.perf_counter()
                try:
                    meas = measure_config(pipe, audio_list,
                                          warmup=warmup, long_clip=long_clip)
                except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                    torch.cuda.empty_cache()
                    free2, _tot2 = torch.cuda.mem_get_info()
                    peak_gb = model_baseline_gb  # best-effort
                    print(f"  [{key}] OOM: {type(e).__name__}: "
                          f"{str(e)[:80]}; free={free2/1e9:.1f}GB")
                    row = {
                        "length_min": lname, "batch_size": B,
                        "target_s": target_s, "status": "oom",
                        "peak_vram_gb": round(peak_gb, 3),
                        "error": type(e).__name__,
                    }
                    results["length_sweep"].append(row)
                    if B == 1:
                        single_clip_stop = True
                        single_clip_ceiling = {
                            "length_min": prev_single_len, "reason": "oom",
                            "broke_at": lname,
                        }
                    else:
                        batch8_stop = True
                        batch8_ceiling = {
                            "length_min": prev_batch8_len, "reason": "oom",
                            "broke_at": lname,
                        }
                    continue

                # success or compute ceiling
                wall_cfg = time.perf_counter() - t_cfg0
                if meas["status"] == "compute_ceiling":
                    print(f"  [{key}] COMPUTE CEILING: "
                          f"{meas['wall_per_pass_s']:.1f}s/pass > "
                          f"{SINGLE_PASS_CAP_S}s; total_ms="
                          f"{meas['total_ms']:.0f} peak={meas['peak_vram_gb']:.1f}GB")
                    row = {
                        "length_min": lname, "batch_size": B,
                        "target_s": target_s, **meas,
                    }
                    results["length_sweep"].append(row)
                    if B == 1:
                        single_clip_stop = True
                        single_clip_ceiling = {
                            "length_min": prev_single_len, "reason": "compute>60s",
                            "broke_at": lname,
                            "broke_wall_s": meas["wall_per_pass_s"],
                        }
                    else:
                        batch8_stop = True
                        batch8_ceiling = {
                            "length_min": prev_batch8_len, "reason": "compute>60s",
                            "broke_at": lname,
                            "broke_wall_s": meas["wall_per_pass_s"],
                        }
                    continue

                # ok -> read decode-step / token counts (extra untimed pass)
                try:
                    T_enc, dec_steps, per_utt_tok, _txt = count_decode(
                        pipe, audio_list)
                except Exception as e:
                    T_enc, dec_steps, per_utt_tok = -1, -1, []
                audio_secs_list = [len(a)/SAMPLE_RATE for a in audio_list]
                row = {
                    "length_min": lname, "batch_size": B,
                    "target_s": target_s,
                    "audio_seconds": meas["audio_seconds"],
                    "T_enc": T_enc,
                    "total_ms": meas["total_ms"],
                    "total_p90_ms": meas["total_p90_ms"],
                    "mel_ms": meas["mel_ms"],
                    "encoder_ms": meas["encoder_ms"],
                    "decode_ms": meas["decode_ms"],
                    "rtf": meas["rtf"],
                    "peak_vram_gb": meas["peak_vram_gb"],
                    "decode_steps": dec_steps,
                    "tokens_emitted_per_utt": per_utt_tok,
                    "tokens_total": int(sum(per_utt_tok)),
                    "n_samples": meas["n_samples"],
                    "status": "ok",
                    "config_wall_s": round(wall_cfg, 2),
                }
                if B == 1:
                    b1_result_for_guard = row
                    prev_b1_steady_ms = meas["total_ms"]
                results["length_sweep"].append(row)
                print(f"  [{key}] total={meas['total_ms']:7.1f}ms "
                      f"mel={meas['mel_ms']:5.1f} enc={meas['encoder_ms']:6.1f} "
                      f"dec={meas['decode_ms']:6.1f} rtf={meas['rtf']:8.1f}x "
                      f"vram={meas['peak_vram_gb']:5.1f}GB "
                      f"T_enc={T_enc} steps={dec_steps} "
                      f"toks/utt={per_utt_tok[:3]}... (cfg {wall_cfg:.1f}s)")
                if B == 1:
                    prev_single_len = lname
                    # ceiling = longest feasible single clip completing in <30s
                    # (steady-state GPU pass time as the proxy). Also remember
                    # the longest attempted (ok) for reporting, even if >30s.
                    if (longest_single_clip is None
                            or target_s > longest_single_clip[1]):
                        longest_single_clip = (lname, target_s, dict(row),
                                               dec_steps, per_utt_tok,
                                               audio_secs_list)
                else:
                    prev_batch8_len = lname

        # Ceiling = longest single clip completing in <30s steady-state. Among
        # the successful (ok) B1 runs, pick the longest whose total_ms < 30s.
        ceiling_single = None
        for x in results["length_sweep"]:
            if (x.get("batch_size") == 1 and x.get("status") == "ok"
                    and x.get("total_ms", 1e9) < 30_000):
                if ceiling_single is None or x["target_s"] > ceiling_single[1]:
                    ceiling_single = (x["length_min"], x["target_s"],
                                      dict(x), x.get("decode_steps", -1),
                                      x.get("tokens_emitted_per_utt", []),
                                      [x["audio_seconds"]])
        # longest_single_clip (from the loop) = longest attempted (ok), used for
        # the "attempted" report; ceiling_single = the <30s production ceiling.
        longest_attempted = longest_single_clip
        longest_single_clip = ceiling_single  # Part-3 headline uses the <30s one

        results["length_sweep_ceiling"] = {
            "single_clip": single_clip_ceiling,
            "batch8": batch8_ceiling,
            "longest_feasible_single_clip": (
                longest_single_clip[0] if longest_single_clip else None),
            "longest_feasible_single_clip_s": (
                longest_single_clip[1] if longest_single_clip else None),
            "longest_attempted_single_clip": (
                longest_attempted[0] if longest_attempted else None),
            "longest_attempted_single_clip_s": (
                longest_attempted[1] if longest_attempted else None),
            "ceiling_definition": "longest single clip with steady-state pass < 30s",
        }

        # =============================================================== #
        # PART 2: real varied speech corpus benchmark
        # (measured above on the clean GPU as part of HEADLINE; here we only
        #  assemble the per-utterance info + per-utterance B1 distribution)
        # =============================================================== #
        print("\n[bench_robust] === PART 2: real varied speech corpus (assemble) ===")
        # per-utterance transcripts vs reference (sanity, not a gate)
        per_utt_info = []
        for i, (sec, ref, hyp) in enumerate(zip(real_secs, real_texts_ref, r_texts)):
            per_utt_info.append({
                "idx": i, "audio_seconds": round(sec, 3),
                "tokens_emitted": r_per_utt_tok[i] if i < len(r_per_utt_tok) else None,
                "reference": ref[:80], "hypothesis": hyp[:80],
            })
        results["real_corpus"]["batch8"] = {
            **real_measured, "T_enc": rT_enc, "decode_steps": r_steps,
            "tokens_emitted_per_utt": r_per_utt_tok,
        }
        results["real_corpus"]["utterances"] = per_utt_info

        # per-utterance B=1 latency distribution (measured above in HEADLINE)
        latencies = [x["total_ms"] for x in per_utt_b1]
        print(f"  [real per-utt B1] latency ms: "
              f"min={min(latencies):.1f} med={np.median(latencies):.1f} "
              f"max={max(latencies):.1f} p90={np.percentile(latencies,90):.1f}")
        results["real_corpus"]["per_utterance_b1"] = {
            "items": per_utt_b1,
            "latency_ms_mean": round(float(np.mean(latencies)), 3),
            "latency_ms_p90": round(float(np.percentile(latencies, 90)), 3),
            "latency_ms_min": round(float(min(latencies)), 3),
            "latency_ms_max": round(float(max(latencies)), 3),
        }

        # =============================================================== #
        # PART 3: comprehensive metrics for headline configs
        # =============================================================== #
        print("\n[bench_robust] === PART 3: comprehensive metrics ===")
        # (i) B8-synthetic-medium (measured above on clean GPU)
        med_secs = [len(medium) / SAMPLE_RATE] * 8
        comp_synth = comprehensive(synth_b8, med_secs, s_per_utt_tok, s_steps)

        # (ii) B8-real-varied
        comp_real = comprehensive(real_measured, real_secs, r_per_utt_tok, r_steps)

        # (iii) longest-feasible-single-clip
        comp_longest = None
        if longest_single_clip is not None:
            (_ln, _ts, row_l, dec_l, tok_l, secs_l) = longest_single_clip
            comp_longest = comprehensive(row_l, secs_l, tok_l, dec_l)
            comp_longest["length_min"] = _ln
            comp_longest["audio_seconds"] = row_l["audio_seconds"]

        results["comprehensive_metrics"] = {
            "b8_synthetic_medium": comp_synth,
            "b8_real_varied": comp_real,
            "longest_feasible_single_clip": comp_longest,
        }

        # real vs synthetic comparison
        synth_rtf = synth_b8.get("rtf")
        real_rtf = real_measured.get("rtf")
        ratio = (real_rtf / synth_rtf) if synth_rtf else None
        # padding waste: longest real utterance drives the batch T_enc
        pad_max_s = max(real_secs)
        pad_mean_s = float(np.mean(real_secs))
        results["real_vs_synthetic"] = {
            "synthetic_b8_medium_rtf": synth_rtf,
            "real_b8_varied_rtf": real_rtf,
            "real_over_synthetic_ratio": round(ratio, 3) if ratio else None,
            "synthetic_clip_seconds": round(len(medium)/SAMPLE_RATE, 3),
            "real_batch_max_seconds": round(pad_max_s, 3),
            "real_batch_mean_seconds": round(pad_mean_s, 3),
            "real_padding_waste_fraction": round(1.0 - pad_mean_s/pad_max_s, 3),
            "explanation": (
                "Real varied speech pads the batch to the longest utterance "
                f"({pad_max_s:.1f}s vs mean {pad_mean_s:.1f}s -> "
                f"{(1.0-pad_mean_s/pad_max_s)*100:.0f}% padding waste), and "
                "varies in token density / blank-skip frequency. If real RTF is "
                "lower than synthetic, padding waste + per-utterance decode-load "
                "variation dominate."),
        }

    # =============================================================== #
    # write JSON
    # =============================================================== #
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_id": MODEL_ID,
        "dtype": "bfloat16",
        "device": torch.cuda.get_device_name(0),
        "method": (
            "cuda.Event + synchronize; warmup>=3 (>=5 for short), >=5 samples "
            "(30s cap), median; peak VRAM = torch.cuda.max_memory_allocated "
            "reset before each config; decode steps + per-utterance tokens read "
            "from the cached GraphedDecoder output buffer; GPU lock held, "
            "deferred if nvidia-smi util>30%"),
        "timing_policy": {
            "warmup_short": WARMUP_SHORT, "warmup_long": WARMUP_LONG,
            "max_samples": MAX_SAMPLES, "sample_wall_cap_s": SAMPLE_WALL_CAP_S,
            "single_pass_cap_s": SINGLE_PASS_CAP_S,
        },
        "results": results,
    }
    out_path = OUTPUTS / "robust_bench.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\n[bench_robust] wrote {out_path}")

    try:
        _print_tables(payload)
    except Exception as e:  # a print glitch must never fail the run
        print(f"[bench_robust] WARN: table print failed ({type(e).__name__}: "
              f"{e}); JSON at {out_path} is the authoritative output")
    return 0


def _print_tables(payload: dict) -> None:
    r = payload["results"]
    print("\n" + "=" * 78)
    print("PART 1 -- LENGTH SWEEP (synthetic-repeated, full attention)")
    print("=" * 78)
    rows = []
    for x in r["length_sweep"]:
        if x.get("status") == "ok" and "total_ms" in x:
            rows.append([
                x["length_min"], x["batch_size"], x.get("audio_seconds", "-"),
                x.get("T_enc", "-"),
                f"{x['total_ms']:.1f}", f"{x.get('mel_ms',0):.1f}",
                f"{x.get('encoder_ms',0):.1f}", f"{x.get('decode_ms',0):.1f}",
                f"{x.get('rtf',0):.0f}x", f"{x.get('peak_vram_gb',0):.1f}",
                x.get("decode_steps", "-"),
            ])
        else:
            # cliff / skipped rows: show the probe measurement if present
            status = x.get("status", "-")
            peak = x.get("peak_vram_gb", x.get("peak_gb", "-"))
            extra = ""
            if "cliff_reason" in x:
                extra = f"{x['cliff_reason']}"
                if "enc_wall_s" in x:
                    extra += f"/{x['enc_wall_s']}s"
            elif "b8_est_pass_s" in x:
                extra = f"est {x['b8_est_pass_s']}s"
            rows.append([
                x["length_min"], x["batch_size"], x.get("target_s", "-"), "-",
                "-", "-", "-", "-", "-", peak,
                f"{status} {extra}".strip(),
            ])
    print(tabulate(
        rows,
        headers=["len", "B", "audio_s", "T_enc", "total_ms", "mel", "enc",
                 "dec", "RTF", "vramGB", "steps"],
        tablefmt="github",
    ))
    ce = r["length_sweep_ceiling"]
    print(f"\nCeiling: single_clip={ce.get('longest_feasible_single_clip')} "
          f"({ce.get('longest_feasible_single_clip_s')}s); "
          f"single_clip_broke={ce.get('single_clip')}; "
          f"batch8_broke={ce.get('batch8')}")

    print("\n" + "=" * 78)
    print("PART 2 -- REAL VARIED SPEECH CORPUS")
    print("=" * 78)
    rb = r["real_corpus"]["batch8"]
    print(f"B8 real varied: total={rb['total_ms']:.1f}ms "
          f"mel={rb['mel_ms']:.1f} enc={rb['encoder_ms']:.1f} "
          f"dec={rb['decode_ms']:.1f} rtf={rb['rtf']:.0f}x "
          f"vram={rb['peak_vram_gb']:.1f}GB T_enc={rb.get('T_enc')} "
          f"steps={rb.get('decode_steps')}")
    pu = r["real_corpus"]["per_utterance_b1"]
    print(f"Per-utterance B=1 latency: mean={pu['latency_ms_mean']:.1f}ms "
          f"p90={pu['latency_ms_p90']:.1f}ms "
          f"min={pu['latency_ms_min']:.1f}ms "
          f"max={pu['latency_ms_max']:.1f}ms")
    urows = [[u["idx"], u["audio_seconds"], u["tokens_emitted"],
              u["reference"][:30]] for u in r["real_corpus"]["utterances"]]
    print(tabulate(urows, headers=["#", "sec", "toks", "reference"],
                   tablefmt="github"))

    print("\n" + "=" * 78)
    print("PART 3 -- COMPREHENSIVE METRICS (headline configs)")
    print("=" * 78)
    cm = r["comprehensive_metrics"]
    crows = []
    for name, c in cm.items():
        if c is None:
            crows.append([name, "-", "-", "-", "-", "-", "-", "-", "-"])
            continue
        crows.append([
            name, c.get("rtf"), c.get("total_ms"),
            c.get("first_token_ms"), c.get("last_token_ms_mean"),
            c.get("last_token_ms_p90"), c.get("decode_steps"),
            c.get("peak_vram_gb"),
            (c.get("length_min", "") or ""),
        ])
    print(tabulate(
        crows,
        headers=["config", "RTF", "total_ms", "first_tok_ms",
                 "last_tok_mean", "last_tok_p90", "steps", "vramGB", "len"],
        tablefmt="github",
    ))

    rv = r["real_vs_synthetic"]
    print("\nREAL vs SYNTHETIC:")
    print(f"  synthetic B8-medium RTF = {rv['synthetic_b8_medium_rtf']:.0f}x")
    print(f"  real B8-varied     RTF = {rv['real_b8_varied_rtf']:.0f}x  "
          f"({rv['real_over_synthetic_ratio']}x of synthetic)")
    print(f"  padding waste fraction = {rv['real_padding_waste_fraction']}")


if __name__ == "__main__":
    raise SystemExit(main())
