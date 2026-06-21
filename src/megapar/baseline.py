"""Reference baseline inference path + RTF benchmarking for parakeet-tdt-0.6b-v3.

This module wraps the STOCK HuggingFace path (no custom kernels) and produces the
two contracts every later optimization phase depends on:

  1. a CORRECTNESS ORACLE  -- deterministic greedy transcripts for the fixtures
  2. the RTF-TO-BEAT       -- a per-stage (feat / encoder / decode) timing
                              breakdown with median + p90 statistics

Timing rules (strict, so later speedup claims are meaningful):
  * every timed region is bracketed by ``torch.cuda.Event`` with an explicit
    ``torch.cuda.synchronize()`` on both sides;
  * we WARMUP then collect ``repeats`` samples (capped by a wall-clock budget);
  * we report MEDIAN and p90 -- never the mean (transducer decode loops are
    long-tailed).

Per-stage split method (explicit split, confirmed empirically):
  * feat_ms     = ``processor(audio)`` + H2D + bf16 cast
  * encoder_ms  = ``model.get_audio_features(input_features, attention_mask)``
                  (the Conformer encoder + the 1024->640 encoder_projector)
  * gen_ms      = full ``model.generate(**inputs)`` (encoder + TDT decode loop)
  * decode_ms   = gen_ms - encoder_ms            (the RNN-T/TDT decode machinery)
  * total_ms    = end-to-end (feat + generate) timed as a single region
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np
import torch
from transformers import AutoModelForTDT, AutoProcessor

MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"
SAMPLE_RATE = 16000

# Architecture dims (verified against the config + the loaded module tree).
# Kept here so downstream files can sanity-check kernel tile sizing without
# reloading the model.
ARCH_DIMS = {
    "encoder_hidden": 1024,
    "encoder_layers": 24,
    "encoder_heads": 8,
    "encoder_ffn_intermediate": 4096,
    "encoder_conv_kernel": 9,
    "subsampling_factor": 8,
    "subsampling_conv_channels": 256,
    "subsampling_conv_kernel": 3,
    "subsampling_conv_stride": 2,
    "num_mel_bins": 128,
    "max_pos_embeddings": 5000,
    "decoder_hidden": 640,
    "decoder_layers": 2,
    "vocab_size": 8193,
    "blank_id": 8192,
    "durations": [0, 1, 2, 3, 4],
    "max_symbols_per_step": 10,
}


@dataclass
class StageTiming:
    """Statistical summary of one timed region (milliseconds)."""

    median_ms: float
    p90_ms: float
    n_samples: int
    samples: list[float]

    @property
    def mean_ms(self) -> float:
        return float(np.mean(self.samples)) if self.samples else float("nan")


def _percentile(values: Iterable[float], q: float) -> float:
    return float(np.percentile(list(values), q))


def time_cuda(
    fn: Callable[[], Any],
    *,
    warmup: int = 8,
    repeats: int = 20,
    max_seconds: float = 12.0,
) -> StageTiming:
    """Time a callable on the GPU.

    Runs ``warmup`` untimed iterations, then up to ``repeats`` timed iterations
    (each bracketed by cuda events + synchronize), stopping early if the
    wall-clock budget is exceeded. Returns median / p90 over the collected
    samples. The callable's return value is discarded.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples: list[float] = []
    wall0 = time.perf_counter()
    for _ in range(repeats):
        torch.cuda.synchronize()
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()
        fn()
        end_evt.record()
        torch.cuda.synchronize()
        samples.append(start_evt.elapsed_time(end_evt))  # ms
        if time.perf_counter() - wall0 > max_seconds:
            break

    return StageTiming(
        median_ms=float(np.median(samples)),
        p90_ms=_percentile(samples, 90),
        n_samples=len(samples),
        samples=samples,
    )


class BaselineRunner:
    """Stock HF inference path for parakeet-tdt-0.6b-v3, instrumented for timing."""

    def __init__(
        self,
        model_id: str = MODEL_ID,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.dtype = dtype

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForTDT.from_pretrained(
            model_id, dtype=dtype, device_map=device
        )
        self.model.eval()

        self.pad_id = self.processor.tokenizer.pad_token_id
        # Sanity-print the real encoder path so callers (and humans) can confirm
        # we are timing the Conformer encoder, not a guess.
        self.encoder_module = self.model.encoder

    # ------------------------------------------------------------------ #
    # provenance helpers (used by the report / analysis)
    # ------------------------------------------------------------------ #
    @property
    def encoder_class_name(self) -> str:
        return type(self.encoder_module).__name__

    @property
    def encoder_attr_path(self) -> str:
        return "model.encoder"

    def param_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    # ------------------------------------------------------------------ #
    # inference
    # ------------------------------------------------------------------ #
    def prepare_inputs(self, audio_list: list[np.ndarray]) -> dict[str, torch.Tensor]:
        """processor(feature extraction) + H2D transfer + bf16 cast.

        This whole region is what we time as ``feat_ms``.
        """
        inputs = self.processor(audio_list, sampling_rate=SAMPLE_RATE)
        inputs = inputs.to(self.device)
        # attention_mask stays its native (bool) dtype; only features are cast.
        inputs["input_features"] = inputs["input_features"].to(self.dtype)
        return inputs

    @torch.inference_mode()
    def transcribe_batch(
        self, audio_list: list[np.ndarray], *, return_tokens: bool = False
    ):
        """End-to-end stock transcribe: feat -> generate -> batch_decode."""
        inputs = self.prepare_inputs(audio_list)
        out = self.model.generate(**inputs, return_dict_in_generate=True)
        texts = self.processor.batch_decode(out.sequences, skip_special_tokens=True)
        if return_tokens:
            ntok = [int((seq != self.pad_id).sum().item()) for seq in out.sequences]
            return texts, ntok
        return texts

    @torch.inference_mode()
    def oracle_transcribe(self, audio: np.ndarray) -> tuple[str, int]:
        """Single-utterance deterministic transcript (the correctness oracle)."""
        inputs = self.prepare_inputs([audio])
        out = self.model.generate(**inputs, return_dict_in_generate=True)
        text = self.processor.batch_decode(out.sequences, skip_special_tokens=True)[0]
        ntok = int((out.sequences[0] != self.pad_id).sum().item())
        return text, ntok

    # ------------------------------------------------------------------ #
    # timing harness
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def bench(
        self,
        audio_list: list[np.ndarray],
        *,
        warmup: int = 8,
        repeats: int = 20,
        max_seconds: float = 12.0,
        measure_peak: bool = True,
    ) -> dict[str, Any]:
        """Time one batch end-to-end and per-stage.

        Returns a dict with feat_ms / encoder_ms / gen_ms / decode_ms / total_ms
        (each with a p90 companion where measured), RTF (median + p90), peak GPU
        memory, and the audio duration. ``total_ms`` is the end-to-end median
        (feat+generate as a single region); the stage breakdown is measured
        separately and should sum to ~= total_ms (reported as a sanity field).
        """
        audio_seconds = sum(len(a) / SAMPLE_RATE for a in audio_list)

        inputs = self.prepare_inputs(audio_list)
        input_features = inputs["input_features"]
        attention_mask = inputs["attention_mask"]

        # ---- feature extraction (CPU mel + H2D + cast) ----
        feat_t = time_cuda(
            lambda: self.prepare_inputs(audio_list),
            warmup=warmup,
            repeats=repeats,
            max_seconds=max_seconds,
        )

        # ---- encoder forward only (Conformer + encoder_projector) ----
        def _encoder():
            return self.model.get_audio_features(
                input_features=input_features, attention_mask=attention_mask
            )

        enc_t = time_cuda(_encoder, warmup=warmup, repeats=repeats, max_seconds=max_seconds)

        # ---- full generate (encoder + TDT decode loop) ----
        def _generate():
            return self.model.generate(**inputs, return_dict_in_generate=True)

        gen_t = time_cuda(_generate, warmup=warmup, repeats=repeats, max_seconds=max_seconds)

        # ---- end-to-end (feat + generate) for a clean RTF ----
        def _e2e():
            local_inputs = self.prepare_inputs(audio_list)
            self.model.generate(**local_inputs, return_dict_in_generate=True)

        e2e_t = time_cuda(_e2e, warmup=warmup, repeats=repeats, max_seconds=max_seconds)

        # Per the task's canonical definition: decode_ms = total - feat - encoder.
        # This is internally consistent by construction: feat + encoder + decode == total.
        total_ms = e2e_t.median_ms
        total_p90 = e2e_t.p90_ms
        decode_ms = total_ms - feat_t.median_ms - enc_t.median_ms
        rtf_median = audio_seconds / (total_ms / 1000.0) if total_ms > 0 else float("inf")
        rtf_p90 = audio_seconds / (total_p90 / 1000.0) if total_p90 > 0 else float("inf")

        # ---- peak GPU memory (one full pass, reset before) ----
        peak_mem_gb: float | None = None
        if measure_peak:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            self.transcribe_batch(audio_list)
            torch.cuda.synchronize()
            peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9

        stage_sum = feat_t.median_ms + gen_t.median_ms  # diagnostic: feat + generate

        return {
            "batch_size": len(audio_list),
            "audio_seconds": round(audio_seconds, 4),
            "feat_ms": round(feat_t.median_ms, 4),
            "feat_p90_ms": round(feat_t.p90_ms, 4),
            "encoder_ms": round(enc_t.median_ms, 4),
            "encoder_p90_ms": round(enc_t.p90_ms, 4),
            "gen_ms": round(gen_t.median_ms, 4),
            "gen_p90_ms": round(gen_t.p90_ms, 4),
            "decode_ms": round(decode_ms, 4),
            "total_ms": round(total_ms, 4),
            "total_p90_ms": round(total_p90, 4),
            "feat_plus_gen_ms": round(stage_sum, 4),
            "rtf_median": round(rtf_median, 4),
            "rtf_p90": round(rtf_p90, 4),
            "peak_mem_gb": round(peak_mem_gb, 4) if peak_mem_gb is not None else None,
            "n_samples": e2e_t.n_samples,
        }

    # ------------------------------------------------------------------ #
    # profiler
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def profile(
        self,
        audio_list: list[np.ndarray],
        *,
        top_k: int = 40,
    ) -> dict[str, Any]:
        """Profile one forward with torch.profiler (CPU+CUDA, shapes, FLOPs).

        Returns:
            - ``table``: the key_averages table sorted by CUDA self-time (str).
            - ``buckets``: CUDA time aggregated into coarse stage buckets.
            - ``total_cuda_time_us``: denominator for bucket percentages.
        """
        from torch.profiler import ProfilerActivity, profile

        inputs = self.prepare_inputs(audio_list)

        # warmup (so JIT/cudnn autotune settle) then a short profiled window.
        for _ in range(3):
            self.model.generate(**inputs, return_dict_in_generate=True)
        torch.cuda.synchronize()

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_flops=True,
        ) as prof:
            for _ in range(3):
                self.model.generate(**inputs, return_dict_in_generate=True)
            torch.cuda.synchronize()

        key_averages = prof.key_averages()
        # sort_by self device (CUDA) self-time -- the hotspot "self-time" view.
        # (In torch 2.12 the cuda_time_total attribute was renamed to
        # device_time_total / self_device_time_total.)
        try:
            table = key_averages.table(
                sort_by="self_device_time_total", row_limit=top_k, max_name_column_width=70
            )
        except Exception:  # noqa: BLE001 -- fall back to total device time
            table = key_averages.table(
                sort_by="device_time_total", row_limit=top_k, max_name_column_width=70
            )

        buckets = _bucketize_cuda_time(key_averages)
        return {
            "table": table,
            "buckets": buckets["buckets"],
            "total_cuda_time_us": buckets["total_cuda_time_us"],
            "bucket_pct": buckets["bucket_pct"],
        }


# ---------------------------------------------------------------------- #
# profiler bucketization
# ---------------------------------------------------------------------- #
# Coarse stage buckets. Rules are matched (case-insensitive substring) in order;
# the first matching rule wins. The dominant CUDA consumers here are the cutlass
# GEMM kernels (the fused FFN matmuls, named "...s16816gemm_relu..."), so matmul
# is checked BEFORE activation to keep "gemm_relu" kernels out of the activation
# bucket. Kernel-level profiling cannot separate an attention GEMM from an FFN
# GEMM (both are mm/addmm/cutlass-gemm), so all GEMMs are reported under one
# "matmul/gemm" bucket; the analysis file notes the FFN is the bulk of it.
BUCKET_RULES: list[tuple[str, list[str]]] = [
    ("mel/feature", ["mel", "stft", "spectrogram", "fft", "feature_extract"]),
    ("conv", ["conv", "depthwise", "im2col", "col2im", "unfold"]),
    ("attention", ["softmax", "sdpa", "flash", "scaled_dot_product", "_efficient_attention", "fmha", "memeffattention"]),
    ("matmul/gemm", ["gemm", "cutlass", "tensorop", "wmma", "addmm", "_scaled_mm", "bmm", "::mm", "linear", "matmul"]),
    ("norm", ["layer_norm", "rmsnorm", "batch_norm", "group_norm"]),
    ("activation", ["silu", "swish", "gelu", "relu", "glu", "sigmoid", "tanh", "elu"]),
    ("decoder/rnnt-tdt", ["_cudnn_rnn", "cudnn_rnn", "::rnn", "lstm", "joint"]),
    ("memops", ["copy_", "memcpy", "::to", "::fill", "::empty", "clone", "contiguous", "zero_"]),
    ("control/reduction", ["argmax", "cumsum", "::index", "scatter", "gather", "where", "select", "slice_", "clamp", "masked_", "nonzero", "::any", "::eq", "::ne", "reduce_kernel"]),
    ("elementwise", ["aten::mul", "aten::add", "aten::sub", "aten::div", "aten::neg", "elementwise_kernel"]),
]


def _classify_op(key: str) -> str:
    lower = key.lower()
    for bucket, keywords in BUCKET_RULES:
        for kw in keywords:
            if kw in lower:
                return bucket
    return "other"


def _bucketize_cuda_time(key_averages) -> dict[str, Any]:
    """Aggregate per-op CUDA self-time into the coarse buckets."""
    bucket_us: dict[str, float] = {}
    per_op: dict[str, dict[str, float]] = {}
    total = 0.0
    for evt in key_averages:
        # torch 2.12: cuda_time_total -> self_device_time_total (CUDA self-time)
        cuda_us = getattr(evt, "self_device_time_total", 0) or 0
        if cuda_us <= 0:
            continue
        bucket = _classify_op(evt.key)
        bucket_us[bucket] = bucket_us.get(bucket, 0.0) + cuda_us
        total += cuda_us
        per_op.setdefault(bucket, {})
        per_op[bucket][evt.key] = per_op[bucket].get(evt.key, 0.0) + cuda_us

    # sort each bucket's ops by time desc for inspection
    for b in per_op:
        per_op[b] = dict(sorted(per_op[b].items(), key=lambda kv: kv[1], reverse=True))

    bucket_pct = {
        b: round(bucket_us.get(b, 0.0) / total * 100.0, 3) if total > 0 else 0.0
        for b in [r[0] for r in BUCKET_RULES] + ["other"]
    }
    return {
        "buckets": {
            b: {
                "cuda_time_us": round(bucket_us.get(b, 0.0), 3),
                "pct": bucket_pct[b],
                "top_ops": list(per_op.get(b, {}).items())[:6],
            }
            for b in ([r[0] for r in BUCKET_RULES] + ["other"])
        },
        "total_cuda_time_us": round(total, 3),
        "bucket_pct": bucket_pct,
    }
