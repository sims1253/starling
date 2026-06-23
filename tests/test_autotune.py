"""Tests for the runtime autotuner (:mod:`starling.parakeet.autotune`).

Covers:
  1. pure GPU-tier classification (no GPU needed) -- every documented tier.
  2. :func:`detect_gpu` on the real card (RTX 5090 -> K=16, B=16, no sweep).
  3. :func:`pick_best_k` noise-robust selection (prefer the hint within margin).
  4. cache write/read round-trip + miss handling.
  5. :func:`autotune` loads a cached config WITHOUT sweeping (no model/GPU work).
  6. a *tiny* real sweep ({1, 4}) that exercises the capture+time path end-to-end
     (kept small per the task: "Do NOT run the full sweep in tests").
  7. :class:`MegaParakeetPipeline(autotune=False)` uses fallback defaults.
  8. :class:`MegaParakeetPipeline(config=...)` respects a custom config.

The full K-sweep + the on-disk cache-population happen in
``benchmarks/parakeet/bench_autotune.py`` (under the GPU lock), NOT here.

Run with:  ``uv run pytest tests/test_autotune.py -q``
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "tests" / "fixtures"))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from starling.parakeet import autotune as at  # noqa: E402
from starling.parakeet.autotune import (  # noqa: E402
    KernelConfig,
    chunk_batch_size_from_vram,
    detect_gpu,
)

MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"
HAS_CUDA = torch.cuda.is_available()


# ====================================================================== #
# 1. pure GPU-tier classification (no GPU / no model)
# ====================================================================== #
@pytest.mark.parametrize(
    "cc,vram,exp_k,exp_b,tier",
    [
        ((12, 0), 34.0, 32, 32, "high_consumer"),   # RTX 5090 (Blackwell) - autotuned
        ((8, 9), 24.0, 32, 32, "high_consumer"),    # RTX 4090 - same tier
        ((8, 0), 40.0, 16, 32, "datacentre"),       # A100 40GB
        ((8, 0), 80.0, 16, 32, "datacentre"),       # A100 80GB
        ((9, 0), 80.0, 16, 32, "datacentre"),       # H100 80GB
        ((8, 6), 24.0, 8, 8, "ampere_consumer"),    # RTX 3090
        ((8, 6), 10.0, 8, 8, "ampere_consumer"),    # RTX 3080 10GB
        ((7, 5), 8.0, 8, 4, "mid"),                 # generic >=8GB
        ((7, 5), 6.0, 4, 2, "low"),                 # <8GB
    ],
)
def test_classify_tier(cc, vram, exp_k, exp_b, tier):
    """Each documented GPU tier maps to its (K, B) fallback."""
    got_tier, k, b = at._classify_tier(cc, vram)
    assert (k, b) == (exp_k, exp_b), f"cc={cc} vram={vram}: got ({k},{b})"
    assert got_tier == tier


def test_classify_tier_ordering_datacentre_before_consumer():
    """An H100 (sm_90, 80GB) must be datacentre (B=32), not the sm_89+ consumer
    tier (B=16) -- the >=40GB check must win over the sm>=89 check."""
    tier, _k, b = at._classify_tier((9, 0), 80.0)
    assert tier == "datacentre"
    assert b == 32


# ====================================================================== #
# 2. detect_gpu on the real card
# ====================================================================== #
@pytest.mark.skipif(not HAS_CUDA, reason="needs CUDA")
def test_detect_gpu_current():
    """detect_gpu() on this box (RTX 5090) -> K=16, B=16 fallback, no sweep."""
    cfg = detect_gpu()
    assert "5090" in cfg.gpu_name, f"unexpected gpu_name {cfg.gpu_name!r}"
    assert cfg.compute_capability == (12, 0)
    assert cfg.steps_per_replay == 32          # 5090 fallback = measured autotune winner
    assert cfg.chunk_batch_size == 32           # 5090 fallback = measured autotune winner
    assert cfg.autotuned is False               # detect_gpu never sweeps
    assert cfg.sweep_results == {}
    assert cfg.gpu_vram_gb > 20.0               # 5090 has ~32GB


# ====================================================================== #
# 3. pick_best_k (noise-robust: prefer the hint within margin)
# ====================================================================== #
def test_pick_best_k_prefers_hint_within_margin():
    """Measured RTX 5090 B8 band (K=8/16/32/64 within ~9%): hint=16 is within the
    10% margin of the fastest (K=32), so 16 is chosen (no noise-driven thrash)."""
    times = {1: 41.7, 4: 22.4, 8: 17.8, 16: 17.2, 32: 16.4, 64: 17.7}
    assert at.pick_best_k(times, hint_k=16, margin=0.10) == 16


def test_pick_best_k_picks_true_fastest_when_hint_bad():
    """If the hint is genuinely uncompetitive AND has no near-best competitors,
    fall back to the smallest near-best K (here K=32 is the only candidate within
    10% of itself, so it is chosen despite hint=16 being a 2.4x outlier)."""
    times = {1: 41.7, 4: 22.4, 8: 30.0, 16: 40.0, 32: 16.4, 64: 25.0}
    chosen = at.pick_best_k(times, hint_k=16, margin=0.10)
    assert chosen == 32


def test_pick_best_k_prefers_smallest_nearbest_when_hint_absent():
    """When the hint is not a candidate, pick the SMALLEST near-best K (lower
    capture cost + less wasted compute on finished sequences). K=8 (17.8ms) is
    within 10% of the fastest (K=32 16.4ms), so 8 is preferred over 32."""
    times = {1: 41.7, 4: 22.4, 8: 17.8, 16: 40.0, 32: 16.4, 64: 17.7}
    chosen = at.pick_best_k(times, hint_k=16, margin=0.10)
    assert chosen == 8


def test_pick_best_k_empty_returns_hint():
    assert at.pick_best_k({}, hint_k=16) == 16


def test_pick_best_k_hint_not_in_swept_set():
    """When the hint is not among the swept K, pick the smallest near-best."""
    times = {1: 40.0, 4: 22.0}
    assert at.pick_best_k(times, hint_k=16, margin=0.10) == 4


# ====================================================================== #
# 4. cache round-trip + miss (hermetic: tmp dir)
# ====================================================================== #
def test_cache_roundtrip(tmp_path):
    cfg = KernelConfig(
        gpu_name="NVIDIA GeForce RTX 5090", gpu_vram_gb=34.19,
        compute_capability=(12, 0), steps_per_replay=16, chunk_batch_size=64,
        autotuned=True, sweep_results={"16": 17.2, "32": 16.4},
        sweep_date="2026-06-23T00:00:00",
    )
    p = at.save_cache(cfg, cache_dir=tmp_path)
    assert p.exists()
    assert at.sanitize_gpu_name(cfg.gpu_name) in p.name

    loaded = at.load_cache(cfg.gpu_name, cache_dir=tmp_path)
    assert loaded is not None
    assert loaded.steps_per_replay == 16
    assert loaded.chunk_batch_size == 64
    assert loaded.compute_capability == (12, 0)   # tuple survives JSON list round-trip
    assert loaded.autotuned is True
    assert loaded.sweep_results["32"] == 16.4
    assert loaded.sweep_date == "2026-06-23T00:00:00"


def test_cache_miss_returns_none(tmp_path):
    assert at.load_cache("Nonexistent GPU XYZ", cache_dir=tmp_path) is None


def test_cache_corrupt_returns_none(tmp_path):
    """A corrupt cache file is treated as a miss (re-sweep on next autotune)."""
    p = at.cache_path("NVIDIA GeForce RTX 5090", cache_dir=tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{ not valid json")
    assert at.load_cache("NVIDIA GeForce RTX 5090", cache_dir=tmp_path) is None


# ====================================================================== #
# 5. autotune loads a cached config WITHOUT sweeping (no model / no GPU work)
# ====================================================================== #
@pytest.mark.skipif(not HAS_CUDA, reason="needs CUDA (detect_gpu reads device name)")
def test_autotune_loads_cache_no_sweep(tmp_path):
    """A pre-written cache short-circuits the sweep: autotune returns the cached
    config without ever touching the (dummy) model/processor."""
    cached = KernelConfig(
        gpu_name=detect_gpu().gpu_name,   # must match the live GPU's cache key
        gpu_vram_gb=34.0, compute_capability=(12, 0),
        steps_per_replay=16, chunk_batch_size=64, autotuned=True,
        sweep_results={"16": 17.2}, sweep_date="2026-06-23T00:00:00",
    )
    at.save_cache(cached, cache_dir=tmp_path)

    # dummy model/processor: the cache path returns before they are used
    cfg = at.autotune(object(), object(), cache_dir=tmp_path)
    assert cfg.steps_per_replay == 16
    assert cfg.chunk_batch_size == 64
    assert cfg.autotuned is True


# ====================================================================== #
# 6. tiny real sweep ({1, 4}) -- exercises capture + time end-to-end
# ====================================================================== #
@pytest.mark.skipif(not HAS_CUDA, reason="needs CUDA")
def test_autotune_tiny_sweep(tmp_path):
    """A real but tiny sweep (K in {1, 4}): captures GraphedDecoders, times the
    decode loop, writes the cache, returns an autotuned config. The full sweep
    lives in the benchmark, not the tests."""
    from transformers import AutoModelForTDT, AutoProcessor

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForTDT.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    try:
        cfg = at.autotune(
            model, processor, force=True, k_values=(1, 4), warmup=2, repeats=4,
            cache_dir=tmp_path, acquire_lock=False,
        )
    finally:
        del model
        torch.cuda.empty_cache()

    assert cfg.autotuned is True
    assert cfg.steps_per_replay in (1, 4)
    assert set(cfg.sweep_results.keys()) == {"1", "4"}
    assert all(v > 0 for v in cfg.sweep_results.values())
    assert cfg.chunk_batch_size == chunk_batch_size_from_vram()
    # cache was written + reloads identically
    loaded = at.load_cache(cfg.gpu_name, cache_dir=tmp_path)
    assert loaded is not None
    assert loaded.steps_per_replay == cfg.steps_per_replay


# ====================================================================== #
# 7 + 8. pipeline integration
# ====================================================================== #
# Building a pipeline loads the model (~25 s); cache ONE pipeline per resolution
# mode across these two tests so the suite is not needlessly slow.
_PIPES: dict[str, "object"] = {}


def _get_pipe(mode: str):
    """mode in {"fallback", "custom"} -> a cached MegaParakeetPipeline."""
    if mode not in _PIPES:
        from starling.parakeet.pipeline import MegaParakeetPipeline  # noqa: WPS433

        if mode == "fallback":
            _PIPES[mode] = MegaParakeetPipeline(autotune=False, encoder_mode="graphed")
        else:
            custom = KernelConfig(
                gpu_name="custom-test", gpu_vram_gb=34.0,
                compute_capability=(12, 0), steps_per_replay=8, chunk_batch_size=4,
                autotuned=True, sweep_results={"8": 10.0},
            )
            _PIPES[mode] = MegaParakeetPipeline(config=custom, encoder_mode="graphed")
    return _PIPES[mode]


@pytest.mark.skipif(not HAS_CUDA, reason="needs CUDA")
def test_pipeline_autotune_false_uses_fallback():
    """MegaParakeetPipeline(autotune=False) -> detect_gpu() fallback defaults,
    no sweep. On the RTX 5090 this is K=32, B=32 (measured autotune winners)."""
    pipe = _get_pipe("fallback")
    assert pipe.config.autotuned is False
    assert pipe.config.steps_per_replay == 32
    assert pipe.config.chunk_batch_size == 32
    # convenience aliases are wired
    assert pipe.steps_per_replay == 32
    assert pipe.chunk_batch_size == 32


@pytest.mark.skipif(not HAS_CUDA, reason="needs CUDA")
def test_pipeline_respects_custom_config():
    """MegaParakeetPipeline(config=custom) uses the custom config directly and
    plumbs steps_per_replay into the captured GraphedDecoder."""
    import make_fixtures as mkfx  # noqa: E402

    custom = KernelConfig(
        gpu_name="custom-test", gpu_vram_gb=34.0, compute_capability=(12, 0),
        steps_per_replay=8, chunk_batch_size=4, autotuned=True,
        sweep_results={"8": 10.0},
    )
    pipe = _get_pipe("custom")
    assert pipe.config is not None
    assert pipe.config.steps_per_replay == custom.steps_per_replay
    assert pipe.config.chunk_batch_size == custom.chunk_batch_size
    assert pipe.steps_per_replay == 8
    assert pipe.chunk_batch_size == 4

    # the captured decoder for a real shape must carry the custom K=8
    fixtures = mkfx.load_fixtures()
    pipe.transcribe([fixtures["short"]])
    assert len(pipe._decoders) == 1, "expected one captured decoder shape"
    dec = next(iter(pipe._decoders.values()))
    assert dec.steps_per_replay == 8, (
        f"custom K=8 not plumbed into GraphedDecoder; got "
        f"{dec.steps_per_replay}"
    )
