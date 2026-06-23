"""Runtime autotuner for the parakeet megakernel pipeline.

The pipeline's two main throughput knobs are tuned per-GPU
(``steps_per_replay`` for :class:`GraphedDecoder`, ``chunk_batch_size`` for
:class:`ChunkedTranscriber`). The 5090 defaults (K=32, B=32) are the **measured
autotune winners** (sweep 2026-06-23: K=32 = 14.93ms decode vs K=16 = 16.10ms).
This module makes the pipeline "focused on RTX 5090 but reasonably usable on
other cards" **with zero user configuration**:

* :func:`detect_gpu` -- instant (no sweep): reads the GPU name / VRAM /
  compute-capability and returns sensible per-GPU-tier fallback defaults. On the
  RTX 5090 this returns K=32, B=32 -- the measured autotune winners, so the
  ``autotune=False`` path already uses the GPU-optimal values.
* :func:`autotune` -- one-time (~30 s) sweep of ``steps_per_replay`` on a
  representative batch, picks the fastest, computes ``chunk_batch_size`` from the
  live free VRAM, and caches the result to
  ``~/.cache/megapar/autotune_<gpu>.json`` so every subsequent run is instant.

What is autotuned
-----------------
1. ``steps_per_replay`` (K) for :class:`GraphedDecoder` -- K consecutive TDT
   decode steps are captured into one CUDA graph replay. Too low = per-step
   host-sync overhead; too high = graph-capture cost + wasted compute on already-
   finished sequences. K is *swept* over ``{1, 4, 8, 16, 32, 64}`` on a medium
   fixture (B=8) and the fastest K is chosen.
2. ``chunk_batch_size`` (B) for :class:`ChunkedTranscriber` -- bounded by VRAM,
   not compute, so it is NOT swept: it is computed once from
   ``torch.cuda.mem_get_info()`` as ``max_B = int((free_gb - 4) / 0.15)`` capped
   at 64.

K selection policy
------------------
Pure ``argmin`` is fragile: on the RTX 5090 the K=8/16/32 band is flat within
~9% (host-sync is already amortised), so measurement noise would thrash the
choice between near-equal values. We instead pick the GPU-tier default K
(:func:`detect_gpu`) whenever it is within ``margin`` (default 10%) of the
measured fastest -- i.e. "trust the documented sweet spot unless the sweep finds
something *meaningfully* faster". This keeps a GPU on its tier default unless the
sweep finds a real win.

GPU lock
--------
The timed sweep is a GPU-exclusive operation. By default :func:`autotune`
acquires the shared ``.gpu.lock`` before sweeping,
unless it is already held by this session (so benchmarks that already hold the
lock can call it with ``acquire_lock=False`` without self-deadlock). Loading a
cached config never touches the GPU and never acquires the lock.
"""

from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from .decode_mega import GraphedDecoder
from .gpu_lock import LOCK_PATH, with_gpu_lock

# K values swept for the GraphedDecoder. The 5090 sweet spot (16) is in the set.
DEFAULT_K_VALUES: tuple[int, ...] = (1, 4, 8, 16, 32, 64)

# Each ~32 s chunk costs ~0.15 GB peak; reserve 4 GB for the resident model +
# other processes when sizing chunk_batch_size from live free VRAM.
_PER_CHUNK_VRAM_GB = 0.15
_VRAM_RESERVE_GB = 4.0
_CHUNK_BATCH_CAP = 64

# K selection: prefer the GPU-tier default K whenever it is within this fraction
# of the measured fastest (avoids noise-driven thrashing on the flat band).
_DEFAULT_MARGIN = 0.10

_REPRESENTATIVE_BATCH = 8        # B for the sweep (medium fixture, 22.3 s each)
_REPRESENTATIVE_DURATION_S = 22.3  # fallback synthetic length if no fixture
_AUTOTUNE_SESSION = "autotune"   # session label for the GPU lock during sweeps


@dataclass
class KernelConfig:
    """Auto-detected per-GPU configuration for the parakeet megakernel.

    Attributes:
        gpu_name: ``torch.cuda.get_device_name`` string (e.g.
            ``"NVIDIA GeForce RTX 5090"``).
        gpu_vram_gb: total VRAM in decimal GB (``total / 1e9``) -- a fixed
            property of the card, used for tier classification.
        compute_capability: ``(major, minor)`` tuple (e.g. ``(12, 0)``).
        steps_per_replay: K for :class:`GraphedDecoder` (decode steps per graph
            replay).
        chunk_batch_size: B for :class:`ChunkedTranscriber` (chunks per
            mini-batch).
        autotuned: ``True`` if produced by a sweep, ``False`` if a fallback
            default from :func:`detect_gpu`.
        sweep_results: per-K median decode-loop time (ms) if swept, keyed by the
            stringified K. Empty for fallback configs.
        sweep_date: ISO-8601 timestamp of the sweep, or ``None``.
    """

    gpu_name: str
    gpu_vram_gb: float
    compute_capability: tuple[int, int]
    steps_per_replay: int
    chunk_batch_size: int
    autotuned: bool
    sweep_results: dict = field(default_factory=dict)
    sweep_date: str | None = None

    # ------------------------------------------------------------------ #
    # JSON (de)serialisation for the on-disk cache
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        return {
            "gpu_name": self.gpu_name,
            "gpu_vram_gb": float(self.gpu_vram_gb),
            "compute_capability": [int(self.compute_capability[0]),
                                   int(self.compute_capability[1])],
            "steps_per_replay": int(self.steps_per_replay),
            "chunk_batch_size": int(self.chunk_batch_size),
            "autotuned": bool(self.autotuned),
            "sweep_date": self.sweep_date,
            "sweep_results": {str(k): float(v)
                              for k, v in self.sweep_results.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "KernelConfig":
        cc = d.get("compute_capability", [0, 0])
        return cls(
            gpu_name=d["gpu_name"],
            gpu_vram_gb=float(d["gpu_vram_gb"]),
            compute_capability=(int(cc[0]), int(cc[1])),
            steps_per_replay=int(d["steps_per_replay"]),
            chunk_batch_size=int(d["chunk_batch_size"]),
            autotuned=bool(d.get("autotuned", False)),
            sweep_results={str(k): float(v)
                           for k, v in d.get("sweep_results", {}).items()},
            sweep_date=d.get("sweep_date"),
        )


# ====================================================================== #
# GPU detection + tier classification
# ====================================================================== #
def _classify_tier(
    compute_capability: tuple[int, int], vram_gb: float
) -> tuple[str, int, int]:
    """Return ``(tier_name, steps_per_replay, chunk_batch_size)`` fallback.

    Pure (no torch) so it is unit-testable with fake specs. Order matters: the
    datacentre tier (>=40 GB) is checked before the high-end consumer tier so an
    H100 (sm_90, 80 GB) is classified as datacentre (B=32) rather than falling
    into the sm_89+ consumer tier (B=16).

    K/B values for the RTX 5090 tier are the **measured autotune winners**
    (swept 2026-06-23 on a clean 5090, B=8 medium fixture): K=32 gave 14.93ms
    decode vs K=16's 16.10ms (7.3% faster); K=64 regressed to 19.52ms (wasted
    compute on finished sequences). B=32 is the measured sweet spot for chunked
    long-audio throughput on 32GB (B=16 gave 2704x RTFx @ 30min, B=32 marginally
    better with headroom to spare; capped conservatively below the VRAM-formula
    max of 64).

    Tiers:
      * ``datacentre``    -- A100 / H100 (sm >= 80, >= 40 GB): K=16, B=32
      * ``high_consumer`` -- RTX 5090 / 4090 (sm >= 89, >= 20 GB): K=32, B=32
      * ``ampere_consumer``-- RTX 3090 / 3080 (sm_86, >= 10 GB): K=8, B=8
      * ``mid``           -- other (>= 8 GB): K=8, B=4
      * ``low``           -- < 8 GB: K=4, B=2
    """
    sm = compute_capability[0] * 10 + compute_capability[1]
    if sm >= 80 and vram_gb >= 40:
        return "datacentre", 16, 32
    if sm >= 89 and vram_gb >= 20:
        return "high_consumer", 32, 32
    if compute_capability == (8, 6) and vram_gb >= 10:
        return "ampere_consumer", 8, 8
    if vram_gb >= 8:
        return "mid", 8, 4
    return "low", 4, 2


def detect_gpu() -> KernelConfig:
    """Detect the GPU and return fallback defaults (no sweep, instant).

    Reads ``torch.cuda`` properties and maps them to a per-GPU-tier default via
    :func:`_classify_tier`. On the RTX 5090 this returns K=32, B=32 -- the
    **measured autotune winners** (swept 2026-06-23), so the ``autotune=False``
    path is a fast path that already uses the GPU-optimal values.
    """
    if not torch.cuda.is_available():
        # The pipeline requires CUDA, but expose a conservative fallback so the
        # tier logic is callable in GPU-less environments (e.g. CI unit tests).
        return KernelConfig(
            gpu_name="cpu",
            gpu_vram_gb=0.0,
            compute_capability=(0, 0),
            steps_per_replay=4,
            chunk_batch_size=2,
            autotuned=False,
        )
    gpu_name = torch.cuda.get_device_name(0)
    cc = torch.cuda.get_device_capability(0)
    _free, total = torch.cuda.mem_get_info()
    vram_gb = total / 1e9
    _tier, k, b = _classify_tier(cc, vram_gb)
    return KernelConfig(
        gpu_name=gpu_name,
        gpu_vram_gb=vram_gb,
        compute_capability=cc,
        steps_per_replay=k,
        chunk_batch_size=b,
        autotuned=False,
    )


# ====================================================================== #
# chunk_batch_size from live free VRAM
# ====================================================================== #
def chunk_batch_size_from_vram(
    *,
    reserve_gb: float = _VRAM_RESERVE_GB,
    per_chunk_gb: float = _PER_CHUNK_VRAM_GB,
    cap: int = _CHUNK_BATCH_CAP,
) -> int:
    """Largest safe ``chunk_batch_size`` for the current free VRAM.

    ``max_B = int((free_gb - reserve_gb) / per_chunk_gb)`` clamped to ``[1, cap]``.
    B is VRAM-bounded (not compute-bounded), so it is computed rather than swept.
    """
    free, _total = torch.cuda.mem_get_info()
    free_gb = free / 1e9
    max_b = int((free_gb - reserve_gb) / per_chunk_gb)
    return max(1, min(cap, max_b))


# ====================================================================== #
# K selection (noise-robust: prefer the tier default within a margin)
# ====================================================================== #
def pick_best_k(
    times: dict[int, float], *, hint_k: int, margin: float = _DEFAULT_MARGIN
) -> int:
    """Choose ``steps_per_replay`` from per-K median decode-loop times.

    Among all K whose median time is within ``margin`` (default 10%) of the
    measured fastest, prefer ``hint_k`` (the GPU-tier default from
    :func:`detect_gpu`); if ``hint_k`` is not competitive, pick the smallest
    near-best K (smaller K = lower capture cost + less wasted compute on
    finished sequences). This avoids noise-driven thrashing on the flat band
    (RTX 5090: K=8/16/32/64 all within ~9%) while still allowing a real win when
    the default is wrong.
    """
    if not times:
        return hint_k
    best = min(times.values())
    threshold = best * (1.0 + margin)
    candidates = [k for k, t in times.items() if t <= threshold]
    if not candidates:  # numerical safety; argmin always qualifies
        return min(times, key=times.get)
    if hint_k in candidates:
        return hint_k
    return min(candidates)


# ====================================================================== #
# on-disk cache
# ====================================================================== #
def default_cache_dir() -> Path:
    """Default cache location: ``~/.cache/megapar``."""
    return Path.home() / ".cache" / "megapar"


def sanitize_gpu_name(name: str) -> str:
    """Filesystem-safe form of a GPU name (``"NVIDIA GeForce RTX 5090"`` ->
    ``"NVIDIA_GeForce_RTX_5090"``)."""
    s = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    return s or "gpu"


def cache_path(gpu_name: str, cache_dir: Path | None = None) -> Path:
    d = cache_dir if cache_dir is not None else default_cache_dir()
    return d / f"autotune_{sanitize_gpu_name(gpu_name)}.json"


def save_cache(cfg: KernelConfig, *, cache_dir: Path | None = None) -> Path:
    p = cache_path(cfg.gpu_name, cache_dir=cache_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cfg.to_dict(), indent=2))
    return p


def load_cache(gpu_name: str, *, cache_dir: Path | None = None) -> KernelConfig | None:
    p = cache_path(gpu_name, cache_dir=cache_dir)
    if not p.exists():
        return None
    try:
        return KernelConfig.from_dict(json.loads(p.read_text()))
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


# ====================================================================== #
# the sweep
# ====================================================================== #
def _repo_root() -> Path:
    # src/megapar/parakeet/autotune.py -> parents[3] is the repo root (matches
    # gpu_lock.LOCK_PATH's parents[3] derivation).
    return Path(__file__).resolve().parents[3]


def _load_representative_audio(
    *, duration_s: float = _REPRESENTATIVE_DURATION_S, sr: int = 16000
) -> np.ndarray:
    """Load a ~22.3 s representative clip for the sweep.

    Prefers the repo's committed ``medium`` fixture (the exact workload the
    pipeline is tuned for); falls back to a deterministic summed-sine signal of
    the same length if the fixture is unavailable, so the autotuner is
    self-contained.
    """
    fixture = _repo_root() / "tests" / "fixtures" / "medium.wav"
    if fixture.exists():
        try:
            import soundfile as sf

            audio, s = sf.read(str(fixture))
            if s == sr and audio.ndim == 1:
                return np.ascontiguousarray(audio, dtype=np.float32)
        except Exception:
            pass
    n = int(duration_s * sr)
    t = np.arange(n, dtype=np.float32) / sr
    sig = 0.08 * (np.sin(2 * np.pi * 220.0 * t)
                  + 0.5 * np.sin(2 * np.pi * 553.0 * t))
    return np.ascontiguousarray(sig, dtype=np.float32)


def _suppress_warnings() -> None:
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


def _encode_representative(model, processor, audio_list, *, sr: int = 16000):
    """Run the processor mel + Conformer encoder once; return pooler + lengths.

    The decode-loop timing depends only on the ``(B, T_enc)`` shape and the
    decoder state trajectory, not on how the mel was produced, so the stock
    ``processor(...)`` path is sufficient for the sweep (matches the
    ``bench_multistep`` preparation).
    """
    inputs = processor(audio_list, sampling_rate=sr)
    feats = inputs["input_features"].to("cuda").to(torch.bfloat16)
    mask = inputs["attention_mask"].to("cuda")
    enc = model.get_audio_features(
        input_features=feats, attention_mask=mask
    )
    pooler = enc.pooler_output.contiguous()
    valid_lengths = enc.attention_mask.to(torch.long).sum(-1).contiguous()
    return pooler, valid_lengths


def _time_decode_loop(
    gd: GraphedDecoder, pooler: torch.Tensor, valid_lengths: torch.Tensor,
    *, warmup: int, repeats: int,
) -> float:
    """Median (ms) of the full K-step decode loop (``_run_loop``) via cuda events."""
    for _ in range(warmup):
        gd._run_loop(pooler, valid_lengths)
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(repeats):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        gd._run_loop(pooler, valid_lengths)
        e.record()
        torch.cuda.synchronize()
        samples.append(s.elapsed_time(e))
    return float(np.median(samples))


def _run_sweep(
    model, processor, *, k_values, warmup, repeats, batch_size, sr: int = 16000,
) -> dict[int, float]:
    """Sweep ``steps_per_replay`` over ``k_values``; return ``{K: median_ms}``.

    Encodes the representative batch ONCE (shape is K-independent), then for each
    K captures a fresh :class:`GraphedDecoder`, warms up, and times the decode
    loop. Each K's graph + buffers are freed before the next to keep VRAM flat.
    """
    audio = _load_representative_audio(sr=sr)
    audio_list = [audio for _ in range(batch_size)]
    pad_id = processor.tokenizer.pad_token_id
    with torch.inference_mode():
        pooler, valid_lengths = _encode_representative(
            model, processor, audio_list, sr=sr
        )
        times: dict[int, float] = {}
        for k in k_values:
            gd = GraphedDecoder(model, steps_per_replay=k)
            gd.capture(pooler, valid_lengths, pad_id, steps_per_replay=k)
            try:
                times[k] = _time_decode_loop(
                    gd, pooler, valid_lengths, warmup=warmup, repeats=repeats
                )
            finally:
                del gd
                torch.cuda.empty_cache()
        return times


def _autotune_holds_lock() -> bool:
    """True if the shared ``.gpu.lock`` is currently held by the autotune sweep.

    Prevents self-deadlock when a caller that already holds the lock invokes the
    sweep (benchmarks do this with ``acquire_lock=False``, but this is a safety
    net).
    """
    try:
        data = json.loads(LOCK_PATH.read_text())
        return data.get("session") == _AUTOTUNE_SESSION
    except Exception:
        return False


def autotune(
    model,
    processor,
    *,
    force: bool = False,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
    acquire_lock: bool = True,
    cache_dir: Path | None = None,
    warmup: int = 3,
    repeats: int = 10,
    batch_size: int = _REPRESENTATIVE_BATCH,
    margin: float = _DEFAULT_MARGIN,
) -> KernelConfig:
    """Sweep ``steps_per_replay`` to find the fastest, cache it, return the config.

    Subsequent runs are instant: a cached config for this GPU is loaded directly
    with no sweep and no GPU work (and no GPU-lock acquisition). The one-time
    sweep (~30 s) encodes a representative B=8 medium batch once, then for each K
    captures a :class:`GraphedDecoder` and times the decode loop, picking the
    best K via :func:`pick_best_k` (noise-robust) and computing
    ``chunk_batch_size`` from live free VRAM.

    Args:
        model: a loaded ``ParakeetForTDT`` on cuda (eval mode, bf16).
        processor: the matching ``AutoProcessor``.
        force: re-run the sweep even if a cache exists (overwrites the cache).
        k_values: K values to sweep (default the full set; pass a small tuple
            like ``(1, 4)`` for a fast test sweep).
        acquire_lock: if True (default) acquire the shared GPU lock before the
            timed sweep (unless this session already holds it). Set False when
            the caller already holds the lock (e.g. a benchmark).
        cache_dir: override the cache directory (default ``~/.cache/megapar``);
            tests pass a tmp dir for hermetic isolation.
        warmup / repeats: warmup iterations and timed samples per K for the
            median.
        batch_size: B for the representative sweep batch (default 8).
        margin: K-selection margin (see :func:`pick_best_k`).

    Returns:
        A :class:`KernelConfig` with ``autotuned=True``.
    """
    base = detect_gpu()
    if not force:
        cached = load_cache(base.gpu_name, cache_dir=cache_dir)
        if cached is not None:
            return cached

    def _do() -> KernelConfig:
        _suppress_warnings()
        times = _run_sweep(
            model, processor, k_values=k_values, warmup=warmup,
            repeats=repeats, batch_size=batch_size,
        )
        best_k = pick_best_k(times, hint_k=base.steps_per_replay, margin=margin)
        b = chunk_batch_size_from_vram()
        cfg = KernelConfig(
            gpu_name=base.gpu_name,
            gpu_vram_gb=base.gpu_vram_gb,
            compute_capability=base.compute_capability,
            steps_per_replay=best_k,
            chunk_batch_size=b,
            autotuned=True,
            sweep_results={str(k): times[k] for k in sorted(times)},
            sweep_date=datetime.now().isoformat(timespec="seconds"),
        )
        save_cache(cfg, cache_dir=cache_dir)
        return cfg

    need_lock = acquire_lock and not _autotune_holds_lock()
    if need_lock:
        with with_gpu_lock(
            session=_AUTOTUNE_SESSION, model="parakeet-tdt-0.6b-v3",
            eta_min=8, note="autotune sweep",
        ):
            return _do()
    return _do()


def resolve_config(
    model, processor, *, config: KernelConfig | None, do_autotune: bool,
    cache_dir: Path | None = None,
) -> KernelConfig:
    """Pipeline-facing resolver: explicit config > autotune > detect_gpu.

    * ``config`` is provided -> use it directly (no GPU work).
    * else ``do_autotune`` -> :func:`autotune` (cache -> sweep).
    * else -> :func:`detect_gpu` (instant fallback, no sweep).
    """
    if config is not None:
        return config
    if do_autotune:
        return autotune(model, processor, cache_dir=cache_dir)
    return detect_gpu()
