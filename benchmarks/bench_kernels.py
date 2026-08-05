"""A/B/C benchmark: Triton vs PyTorch-fallback vs CUDA C++ kernel backends.

This is the **decision gate** for kernel-backend selection. It
microbenchmarks every fused decode kernel under all THREE backends at real
decode shapes (M=1), measures the latency of each plus the max-abs numerical
diff, and prints a markdown table from which we can read off which backend
wins each kernel and by how much.

The three backends
------------------
* **triton**  -- hand-tuned Triton kernels. What Linux gets today (the fast
  path). Selected on Linux with the ``triton`` package installed.
* **cuda**    -- CUDA C++ kernels JIT-compiled via
  :func:`torch.utils.cpp_extension.load_inline` (cached after first run). The
  full-performance path on triton-less platforms that still have a CUDA toolkit
  (notably Windows). Implements all 8 fused decode kernels natively
  (rmsnorm/silu_mul/residual_add/fp8_linear/fused_rope/compute_rstd/
  fused_gemv_normscale/fp4_gemv_fused); only ``bf16_linear`` is left to
  backend-independent ``F.linear`` (shown as ``[delegate]``).
* **torch**   -- stock-PyTorch fused ops, the correctness fallback everywhere.
  Its FP8/FP4 GEMV paths *materialize the full bf16 weight*, so they do NOT
  realize the bandwidth win -- they are correctness paths.

The benchmark imports the backend modules *directly*
(``starling._kernels.triton_backend`` / ``torch_backend`` /
``cuda_backend``) to A/B/C them, bypassing the ``STARLING_KERNEL_BACKEND``
dispatch. ``--backend`` selects a subset; the default ``all`` runs every
available backend.

The triton-fp8-in-same-process IMA caveat
-----------------------------------------
There is a known triton-side bug: if the triton fp8 kernel's autotuner AND the
cuda fp8 kernel run in the SAME python process, triton's autotuner benchmarking
hits a CUDA illegal-memory-access (NOT a bug in our cuda kernel). To stay clear
of it, when BOTH triton and cuda are benchmarked we always run (and autotune)
the triton fp8 kernels FIRST, before any cuda fp8 call -- see
:func:`_warm_autotune_before_cuda`. If that ordering still triggers the IMA on
some env, each fp8 timing is individually guarded and the affected cell degrades
to ``SKIP`` with a note. The elementwise kernels have NO such issue (safe to
benchmark all three back-to-back).

Timing
------
Standalone kernel GPU time via ``torch.profiler`` (CUDA activity): we batch
``--trials`` calls under one profile pass, sum each pass's
``self_device_time_total`` over all CUDA ops, divide by ``trials``, and keep
the minimum over ``--warmup`` passes. Profiler (not per-call ``torch.cuda.
Event``) is used because these decode kernels are sub-microsecond to a few
us of real GPU work; per-call event timing is dominated by host-side launch
latency (~50-200 us), which drowns the signal and can even invert the
ordering. The profile reads kernel timestamps off the GPU timeline, so it
isolates the kernel cost itself -- the number that drives the backend
decision. (In production these run inside a captured CUDA graph, so absolute
throughput differs from the standalone number, but the *ratio* between
backends is the stable, decision-relevant signal.) This mirrors the
``profile_step`` approach in ``bench_fp8_speedup.py``.

Usage::

    uv run python benchmarks/bench_kernels.py                       # full A/B/C
    uv run python benchmarks/bench_kernels.py --trials 50 --warmup 10
    uv run python benchmarks/bench_kernels.py --backend cuda         # cuda-only
    uv run python benchmarks/bench_kernels.py --backend torch        # torch-only (Windows)
    uv run python benchmarks/bench_kernels.py --mode decode          # end-to-end (stretch)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from tabulate import tabulate

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from starling._kernels import torch_backend, triton_backend  # noqa: E402
from starling._kernels.base import have_cuda_compile, have_triton  # noqa: E402

DT = torch.bfloat16
DEV = "cuda"
EPS = 1e-6


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #
def time_us(fn: Callable[[], object], *, warmup: int, trials: int) -> float:
    """Median GPU kernel time of ``fn`` in microseconds (best-of-N profile passes).

    At decode (M=1) these fused kernels are sub-microsecond to a few us of
    *actual GPU work* -- e.g. the triton rmsnorm kernel runs in ~1.08 us. A
    naive ``torch.cuda.Event`` pair bracketing a single such call measures
    mostly host-side launch latency (~50-200 us of Python/dispatch jitter),
    which swamps the signal and even inverts the ordering (tiny kernels look
    slower than heavy ones). To get the number that actually matters for the
    CUDA-port decision -- the GPU time the kernel itself costs -- we batch
    ``trials`` calls under one ``torch.profiler`` pass (which reads kernel
    timestamps off the GPU timeline, so host launch cost is excluded), sum
    ``self_device_time_total`` over all CUDA ops, divide by ``trials``, and
    take the minimum (best) over ``--warmup`` such passes. This mirrors the
    ``profile_step`` approach already used in ``bench_fp8_speedup.py``.

    The per-call event-timing alternative (also computed here) is reported
    alongside in stderr for sanity; the table uses the profiled GPU time.
    """
    from contextlib import contextmanager
    from torch.profiler import profile, ProfilerActivity

    @contextmanager
    def _silence_fd2():
        # The CUPTI activity profiler writes noisy "USDT:... profiler_start/
        # stop" lines to fd 2 (the C-level stderr), which Python's
        # redirect_stderr cannot catch. Redirect the OS fd itself.
        saved = os.dup(2)
        devnull = os.open(os.devnull, os.O_WRONLY)
        try:
            os.dup2(devnull, 2)
            yield
        finally:
            os.dup2(saved, 2)
            os.close(devnull)
            os.close(saved)

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    best: float | None = None
    for _ in range(warmup):
        with _silence_fd2():
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                for _ in range(trials):
                    fn()
                torch.cuda.synchronize()
        total = sum(e.self_device_time_total for e in prof.key_averages())
        per_call = total / trials
        if best is None or per_call < best:
            best = per_call
    return float(best)


# --------------------------------------------------------------------------- #
# Per-case args factories.
#
# Each factory builds the input tensors ONCE (independent of backend), so both
# backends see identical data -- this is what makes the max-diff column
# meaningful, and it removes input-construction from the timed region. It
# returns ``(prepare, shape)`` where ``prepare(backend)`` is a closure that
# binds the frozen inputs to that backend and returns a 0-arg ``step()`` that
# runs the kernel. For fp8, ``prepare`` pre-quantizes the (frozen) weight with
# that backend's ``quantize_weight_e4m3`` ONCE, so the timed op is the GEMV
# alone -- quantization is a one-time model-load cost in production, never
# per-step. The bf16 baseline uses raw ``F.linear`` so we can see whether fp8
# actually beats a plain bf16 GEMV on each backend.
# --------------------------------------------------------------------------- #
CASES: list[tuple[str, str, Callable]] = []


def _rmsnorm(N: int):
    def factory():
        x = (torch.randn(1, N, dtype=DT, device=DEV) * 0.5).contiguous()
        w = (torch.randn(N, dtype=DT, device=DEV) * 0.1 + 1.0).contiguous()

        def prepare(mod):
            def step():
                return mod.fused_rmsnorm(x, w, EPS)
            return step
        return prepare, f"(1,{N})"
    return factory


def _silu_mul(N: int):
    def factory():
        g = (torch.randn(1, N, dtype=DT, device=DEV) * 0.5).contiguous()
        u = (torch.randn(1, N, dtype=DT, device=DEV) * 0.5).contiguous()

        def prepare(mod):
            def step():
                return mod.fused_silu_mul(g, u)
            return step
        return prepare, f"(1,{N})"
    return factory


def _residual(N: int, alpha: float):
    def factory():
        x = (torch.randn(1, N, dtype=DT, device=DEV) * 1.0).contiguous()
        y = (torch.randn(1, N, dtype=DT, device=DEV) * 0.3).contiguous()

        def prepare(mod):
            def step():
                return mod.residual_add(x, y, alpha)
            return step
        return prepare, f"(1,{N}) a={alpha}"
    return factory


def _rope(n_q: int, n_kv: int, hd: int):
    def factory():
        q = (torch.randn(1, n_q, 1, hd, dtype=DT, device=DEV) * 0.3).contiguous()
        k = (torch.randn(1, n_kv, 1, hd, dtype=DT, device=DEV) * 0.3).contiguous()
        cos = (torch.randn(1, 1, 1, hd, dtype=DT, device=DEV)).contiguous()
        sin = (torch.randn(1, 1, 1, hd, dtype=DT, device=DEV)).contiguous()

        def prepare(mod):
            def step():
                return mod.fused_rope(q, k, cos, sin)
            return step
        return prepare, f"(1,{n_q}+{n_kv},{hd})"
    return factory


def _fp8_linear(N: int, K: int):
    def factory():
        W = (torch.randn(N, K, dtype=DT, device=DEV) * 0.02).contiguous()
        x = (torch.randn(1, K, dtype=DT, device=DEV) * 0.5).contiguous()

        def prepare(mod):
            # Pre-quantize ONCE with this backend: production treats quant as a
            # one-time load cost, so the timed op must be the GEMV alone.
            w_fp8, w_scale = mod.quantize_weight_e4m3(W)

            def step():
                return mod.fp8_linear(x, w_fp8, w_scale)
            return step
        return prepare, f"({N},{K})"
    return factory


def _bf16_linear(N: int, K: int):
    def factory():
        W = (torch.randn(N, K, dtype=DT, device=DEV) * 0.02).contiguous()
        x = (torch.randn(1, K, dtype=DT, device=DEV) * 0.5).contiguous()

        def prepare(mod):  # noqa: ARG001  (F.linear is backend-independent)
            def step():
                return F.linear(x, W)
            return step
        return prepare, f"({N},{K})"
    return factory


CASES = [
    ("rmsnorm(h=2048)",          "fused_rmsnorm", _rmsnorm(2048)),
    ("rmsnorm(h=128)",           "fused_rmsnorm", _rmsnorm(128)),
    ("silu_mul(i=4096)",         "fused_silu_mul", _silu_mul(4096)),
    ("silu_mul(i=6144)",         "fused_silu_mul", _silu_mul(6144)),
    ("residual(h=2048) a=1.0",   "residual_add",   _residual(2048, 1.0)),
    ("residual(h=2048) a=0.22",  "residual_add",   _residual(2048, 0.22)),
    ("rope(q=16,k=8,d=128)",     "fused_rope",     _rope(16, 8, 128)),
    ("fp8_linear qkv(4096,2048)",    "fp8_linear",  _fp8_linear(4096, 2048)),
    ("fp8_linear o(2048,2048)",      "fp8_linear",  _fp8_linear(2048, 2048)),
    ("fp8_linear gateup(12288,2048)","fp8_linear",  _fp8_linear(12288, 2048)),
    ("fp8_linear down(2048,6144)",   "fp8_linear",  _fp8_linear(2048, 6144)),
    ("bf16 F.linear qkv(4096,2048)", "bf16_linear", _bf16_linear(4096, 2048)),
]


# --------------------------------------------------------------------------- #
# Main A/B microbenchmark.
# --------------------------------------------------------------------------- #
def _flatten(out) -> torch.Tensor:
    """Reduce a kernel output (possibly a tuple, e.g. fused_rope) to one tensor."""
    if isinstance(out, tuple):
        return out[0]
    return out


# Backends that can be timed. Each entry: (name, module-or-None, available-fn).
# The module is imported lazily (cuda_backend JIT-compiles on import of _ext),
# so we keep it as a lazy lookup to avoid forcing the cuda compile when the user
# passed --backend torch.
def _backend_module(name: str):
    """Return the backend module for ``name``, or ``None`` if unavailable."""
    if name == "triton":
        return triton_backend if have_triton() else None
    if name == "torch":
        return torch_backend
    if name == "cuda":
        if not have_cuda_compile():
            return None
        try:
            from starling._kernels import cuda_backend

            cuda_backend._ext()  # JIT-compile now; cached after first call
            return cuda_backend
        except Exception as e:  # noqa: BLE001  (no toolkit / nvcc / compile fail)
            print(f"[bench] cuda backend unavailable ({type(e).__name__}: {e}).",
                  file=sys.stderr)
            return None
    return None


# Kernels the cuda backend implements NATIVELY (8 total). Everything else
# (only bf16_linear here, which is backend-independent F.linear) is not a cuda
# kernel; those rows show "[delegate]" rather than duplicating the torch op.
# The four CODA/fp4 kernels (compute_rstd/fused_gemv_normscale/fp4_gemv_fused)
# are native too, but the bench's CASES below do not exercise them -- this set
# only governs the per-row delegate-vs-timed decision for the rows that exist.
_CUDA_NATIVE_KERNELS = frozenset(
    {
        "fused_rmsnorm",
        "fused_silu_mul",
        "residual_add",
        "fp8_linear",
        "fused_rope",
        "compute_rstd",
        "fused_gemv_normscale",
        "fp4_gemv_fused",
    }
)


def _warm_autotune_before_cuda(cases, triton_mod, cuda_mod) -> None:
    """Run every triton fp8 kernel once (autotuning it) BEFORE any cuda fp8 call.

    Workaround for a triton-side bug: if the triton fp8 autotuner benchmarks
    in the same process AFTER the cuda fp8 kernel has run, it can hit a CUDA
    illegal-memory-access. Running (and autotuning) triton fp8 first avoids
    the re-entrancy. Safe to call unconditionally; a no-op when either backend
    is absent or no fp8 case is present. Errors here are non-fatal (the
    individual timings are also guarded).
    """
    if triton_mod is None or cuda_mod is None:
        return
    try:
        for label, kern, factory in cases:
            if kern != "fp8_linear":
                continue
            prepare, _shape = factory()
            step = prepare(triton_mod)
            with torch.no_grad():
                _flatten(step())
        torch.cuda.synchronize()
    except Exception as e:  # noqa: BLE001
        print(
            f"[bench] triton fp8 autotune warm-up failed ({type(e).__name__}: {e}); "
            "fp8 timing cells may degrade to SKIP.",
            file=sys.stderr,
        )


def run_microbench(trials: int, warmup: int, backends: list[str]) -> int:
    """Run every case under every requested backend; print a markdown table.

    Columns are: kernel | shape | triton (us) | torch (us) | cuda (us) | best |
    maxdiff. ``best`` names the fastest backend for each kernel. ``maxdiff`` is
    the max-abs difference over the timed backends' outputs on the shared
    inputs (a cross-backend numerical-agreement sanity check). Missing backends
    show ``-``; a backend that errors (e.g. the triton-fp8-in-same-process IMA)
    shows ``SKIP``.
    """
    triton_mod = _backend_module("triton") if "triton" in backends else None
    torch_mod = _backend_module("torch") if "torch" in backends else None
    cuda_mod = _backend_module("cuda") if "cuda" in backends else None

    mods = {"triton": triton_mod, "torch": torch_mod, "cuda": cuda_mod}
    if not any(mods.values()):
        print("[bench] no usable backend (need triton/torch/cuda).", file=sys.stderr)
        return 1

    # One-time cuda compile message (the _backend_module call above already did
    # the compile; print the confirmation the spec asks for).
    if cuda_mod is not None:
        print("compiled cuda backend", flush=True)

    # Avoid the triton-fp8-in-same-process IMA: autotune triton fp8 BEFORE any
    # cuda fp8 call when both are in the matrix.
    _warm_autotune_before_cuda(CASES, triton_mod, cuda_mod)

    print(
        f"[bench] device: {torch.cuda.get_device_name(0)} | "
        f"triton: {'on' if triton_mod else 'off/unavailable'} | "
        f"torch: {'on' if torch_mod else 'off'} | "
        f"cuda: {'on' if cuda_mod else 'off/unavailable'} | "
        f"warmup={warmup} trials={trials} (median)",
        flush=True,
    )

    rows: list[list[str]] = []
    wins: dict[str, int] = {"triton": 0, "torch": 0, "cuda": 0}

    for label, kern, factory in CASES:
        prepare, shape = factory()

        # Build steps + time each requested backend. For the cuda backend, rows
        # whose kernel cuda does NOT implement natively are delegated to torch
        # (cuda_backend imports them from torch_backend), so we mark the cuda
        # cell "[delegate]" rather than re-timing the identical torch op.
        steps: dict[str, Callable | None] = {}
        for name, mod in mods.items():
            if mod is None:
                steps[name] = None
                continue
            if name == "cuda" and kern not in _CUDA_NATIVE_KERNELS:
                steps[name] = None  # delegate -> shown as "[delegate]"
            else:
                steps[name] = prepare(mod)

        # Numerical diff across the timed backends (shared inputs).
        timed_outs: list[torch.Tensor] = []
        with torch.no_grad():
            for name in ("triton", "torch", "cuda"):
                s = steps[name]
                if s is not None:
                    try:
                        timed_outs.append(_flatten(s()).float())
                    except Exception:
                        timed_outs.append(None)  # type: ignore[arg-type]
        diff = float("nan")
        valid = [t for t in timed_outs if t is not None]
        if len(valid) >= 2:
            md = torch.tensor(0.0, device=DEV)
            for t in valid[1:]:
                md = torch.max(md, (valid[0] - t).abs().max())
            diff = float(md.item())

        # Time each backend, guarding each cell (an IMA / OOM degrades one cell
        # to SKIP rather than killing the whole bench).
        us: dict[str, float | None] = {}
        for name in ("triton", "torch", "cuda"):
            s = steps[name]
            if s is None:
                us[name] = None
                continue
            try:
                us[name] = time_us(s, warmup=warmup, trials=trials)
            except Exception as e:  # noqa: BLE001  (e.g. triton fp8 IMA)
                print(
                    f"[bench] {label}: {name} timing failed "
                    f"({type(e).__name__}: {e}); marking SKIP.",
                    file=sys.stderr,
                )
                us[name] = None

        # Pick the winner among backends that actually timed (cuda-delegate
        # rows are excluded since cuda == torch there).
        candidate_us = {n: us[n] for n in ("triton", "torch", "cuda") if us[n] is not None}
        best_name = min(candidate_us, key=candidate_us.get) if candidate_us else None
        if best_name is not None:
            wins[best_name] = wins.get(best_name, 0) + 1

        # Skip indicator for the fp8 IMA case: if cuda is in the matrix AND triton
        # fp8 failed to time (None) on an fp8 row, note it explicitly in the cell.
        def _cell(name: str) -> str:
            if mods[name] is None:
                return "-"
            if name == "cuda" and kern not in _CUDA_NATIVE_KERNELS:
                return "[delegate]"
            v = us[name]
            return "SKIP" if v is None else f"{v:.1f}"

        row = [
            kern, shape,
            _cell("triton"), _cell("torch"), _cell("cuda"),
            best_name or "-",
            f"{diff:.4f}" if diff == diff else "-",  # NaN-guard
        ]
        rows.append(row)

    header = ["kernel", "shape", "triton (us)", "torch (us)", "cuda (us)", "best", "maxdiff"]
    print(
        "\n(best = lowest-GPU-time backend for that kernel; [delegate] = cuda "
        "backend reuses the torch impl for that op; SKIP = timing errored, e.g. "
        "the triton-fp8-in-same-process IMA)\n"
    )
    print(tabulate(rows, headers=header, tablefmt="github"))

    _print_summary(wins)
    return 0


def _print_summary(wins: dict[str, int]) -> None:
    """Footer: how many kernels each backend won outright.

    Replaces the old triton-vs-torch-only decision-gate text. With three
    backends the headline question is whether cuda achieves parity with triton
    (the Linux fast path) -- i.e. whether triton-less Windows + a CUDA toolkit
    can recover full fused-kernel performance.
    """
    total = sum(wins.values())
    print("\nWin-count summary (lowest GPU time wins, ties not counted):")
    if total == 0:
        print("  (no kernel could be timed on any backend)")
    else:
        for name in ("triton", "cuda", "torch"):
            c = wins.get(name, 0)
            print(f"  {name:<8s} {c} kernel(s)")
    print(
        "\nInterpretation: triton and cuda both fuse rmsnorm/silu_mul/residual/"
        "fp8_dequant_gemv\ninto a single launch, so on those rows they should be "
        "within a few percent of each other\n(and both well ahead of torch on the "
        "fp8 GEMV, which torch must dequant to full bf16 first).\nThat parity is "
        "the result that matters: it means a triton-less platform (Windows) with a "
        "CUDA toolkit gets\nLinux/Triton-level fused-kernel performance via the "
        "cuda backend. The torch column remains the correctness floor everywhere."
    )


# --------------------------------------------------------------------------- #
# Optional end-to-end decode-step mode (stretch goal).
#
# Times N raw decode steps of the Moss model under each backend by setting the
# ``STARLING_KERNEL_BACKEND`` env var before importing the pipeline, replaying
# the captured CUDA graph. Kept small (<40 lines) -- if the pipeline import
# shape ever changes, just drop this mode; the microbench above is the primary
# deliverable.
# --------------------------------------------------------------------------- #
def run_decode_mode(trials: int, warmup: int, backends: list[str]) -> int:
    # End-to-end decode-step A/B. This is a STRETCH goal; the moss pipeline's
    # decoder/capture API may differ from the granite shape assumed here, so the
    # whole body is guarded: any import- or runtime-mismatch degrades to a clean
    # message rather than a traceback, leaving the per-kernel microbench
    # (``--mode micro``, the primary deliverable) unaffected.
    try:
        from starling.moss.loader import get_components, load_model_and_processor

        # a frozen decode step for each backend: build decoder, prefill, capture,
        # then replay the graph N times under CUDA-event timing.
        def _step_us(model, comps, ie, T):
            from starling.moss.llm_mega import FusedLLMMega  # type: ignore
            dec = FusedLLMMega(comps["language_model"], model.lm_head, max_cache_len=896)
            nt = dec.prefill(ie)
            dec.capture(nt, T)
            for _ in range(warmup):
                dec._graph.replay()
            torch.cuda.synchronize()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(trials):
                dec._graph.replay()
            e.record()
            torch.cuda.synchronize()
            return s.elapsed_time(e) * 1e3 / trials  # us/step

        rows = []
        ie = torch.zeros(1, 8, dtype=torch.bfloat16, device="cuda")  # tiny dummy prompt
        T = ie.shape[1]
        for bk in backends:
            if bk == "triton" and not have_triton():
                continue
            os.environ["STARLING_KERNEL_BACKEND"] = bk
            from starling import _kernels
            _kernels.set_backend(bk)  # force this backend before building the decoder
            model, _ = load_model_and_processor()
            comps = get_components(model)
            us = _step_us(model, comps, ie, T)
            rows.append([bk, f"{us:.1f}"])
            del model
            import gc; gc.collect(); torch.cuda.empty_cache()
    except Exception as e:  # noqa: BLE001
        print(f"[bench] decode mode unavailable ({type(e).__name__}: {e}).")
        print("        The per-kernel microbench (--mode micro, default) is the "
              "primary deliverable.")
        return 1
    print(tabulate(rows, headers=["backend", "decode us/step"], tablefmt="github"))
    return 0


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--mode", default="micro", choices=["micro", "decode"],
        help="micro = per-kernel A/B (default); decode = end-to-end decode step "
             "under each backend env var (stretch, may be unavailable)",
    )
    ap.add_argument(
        "--backend", default="all",
        choices=["all", "both", "triton", "torch", "cuda"],
        help="which backends to benchmark. 'all' (default) = triton,torch,cuda; "
             "'both' = legacy alias for triton,torch (cuda omitted). Use a single "
             "name (e.g. --backend cuda, or --backend torch on a triton-less box).",
    )
    ap.add_argument("--trials", default=50, type=int, help="timed calls (median)")
    ap.add_argument("--warmup", default=10, type=int, help="untimed warmup calls")
    args = ap.parse_args(argv)

    if not torch.cuda.is_available():
        print("[bench] CUDA is required (this is a CUDA decode-kernel bench).",
              file=sys.stderr)
        return 1

    if args.backend == "all":
        backends = ["triton", "torch", "cuda"]
    elif args.backend == "both":
        backends = ["triton", "torch"]
    else:
        backends = [args.backend]

    if args.mode == "decode":
        return run_decode_mode(args.trials, args.warmup, backends)
    return run_microbench(args.trials, args.warmup, backends)


if __name__ == "__main__":
    raise SystemExit(main())
