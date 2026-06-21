"""Full RTF benchmark + correctness oracle + profiler for the parakeet-tdt baseline.

Running ``uv run python benchmarks/bench_rtf.py`` produces, under ``outputs/``:

  * oracle.json            -- deterministic transcripts (short/medium/long)
  * baseline_bench.json    -- per-batch stage timings + headline RTF + arch dims
  * profile_top_ops.txt    -- torch.profiler top-op table (CUDA self-time)
  * profile_stages.json    -- CUDA time bucketed into coarse stages
  * profile_analysis.md    -- short hotspot analysis driving the next phase

It also prints a formatted table (tabulate) to stdout.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from tabulate import tabulate

# Make the fixtures module importable (tests/fixtures is not an installed pkg).
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "tests" / "fixtures"))
sys.path.insert(0, str(_REPO_ROOT / "src"))

import make_fixtures as mkfx  # noqa: E402
from megapar.baseline import (  # noqa: E402
    ARCH_DIMS,
    MODEL_ID,
    SAMPLE_RATE,
    BaselineRunner,
)

OUTPUTS = _REPO_ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)

# Batch sweep (mixed: cycle short/medium/long) and the headline/profile config.
BATCH_SIZES = [1, 4, 8, 16]
HEADLINE_BATCH = 8
HEADLINE_LENGTH = "medium"


def _suppress_noisy_warnings() -> None:
    # max_length default + LSTM flatten_parameters are benign for inference.
    for mod in (
        "transformers.generation.utils",
        "transformers.models.parakeet.generation_parakeet",
    ):
        try:
            warnings.filterwarnings("ignore", module=mod)
        except Exception:  # noqa: BLE001
            pass
    warnings.filterwarnings("ignore", message=".*max_length.*")
    warnings.filterwarnings("ignore", message=".*RNN module weights.*")


def build_oracle(runner: BaselineRunner, fixtures: dict[str, np.ndarray]) -> list[dict]:
    """Run the stock path on short/medium/long and capture gold transcripts."""
    oracle: list[dict] = []
    for name in ("short", "medium", "long"):
        audio = fixtures[name]
        text, ntok = runner.oracle_transcribe(audio)
        audio_seconds = round(len(audio) / SAMPLE_RATE, 4)
        print(f"[oracle] {name:7s} ({audio_seconds:6.2f}s) -> {ntok:4d} tokens")
        print(f"         {text!r}")
        oracle.append(
            {
                "name": name,
                "audio_seconds": audio_seconds,
                "num_tokens": ntok,
                "text": text,
            }
        )
    return oracle


def _gpu_settle(runner: BaselineRunner, fixtures: dict[str, np.ndarray]) -> None:
    """Let GPU clocks recover between heavy configs (thermal stabilisation)."""
    import time

    torch.cuda.empty_cache()
    # a couple of tiny untimed warmup gens on the short fixture
    short = fixtures["short"]
    with torch.inference_mode():
        inputs = runner.prepare_inputs([short])
        for _ in range(3):
            runner.model.generate(**inputs, return_dict_in_generate=True)
    torch.cuda.synchronize()
    time.sleep(0.4)


def run_bench(
    runner: BaselineRunner, fixtures: dict[str, np.ndarray]
) -> tuple[list[dict], dict]:
    """Headline (batch=8 medium) FIRST for thermal stability, then the mixed sweep."""
    results: list[dict] = []

    # Headline: batch=8 all-medium (clean, matches the profile config). Run first
    # while the GPU is coolest so the headline number is the most stable.
    _gpu_settle(runner, fixtures)
    headline_batch = mkfx.build_uniform_batch(fixtures[HEADLINE_LENGTH], HEADLINE_BATCH)
    print(f"[bench] headline batch={HEADLINE_BATCH} ({HEADLINE_LENGTH}) ...")
    headline = runner.bench(headline_batch, warmup=8, repeats=20, max_seconds=12.0)
    headline["scenario"] = f"headline_{HEADLINE_LENGTH}"

    for bs in BATCH_SIZES:
        batch = mkfx.build_batch(fixtures, bs)  # mixed: cycle short/medium/long
        _gpu_settle(runner, fixtures)
        print(f"\n[bench] mixed batch={bs} ...")
        res = runner.bench(batch, warmup=8, repeats=20, max_seconds=12.0)
        res["scenario"] = "mixed"
        results.append(res)

    return results, headline


def run_profile(
    runner: BaselineRunner, fixtures: dict[str, np.ndarray]
) -> dict:
    """Profile one representative forward (batch=8 medium)."""
    batch = mkfx.build_uniform_batch(fixtures[HEADLINE_LENGTH], HEADLINE_BATCH)
    print(f"\n[profile] batch={HEADLINE_BATCH} ({HEADLINE_LENGTH}) ...")
    return runner.profile(batch, top_k=40)


def fmt_table(rows: list[dict]) -> str:
    header = [
        "scenario",
        "B",
        "feat_ms",
        "enc_ms",
        "dec_ms",
        "total_ms",
        "rtf_med",
        "rtf_p90",
        "peak_GB",
    ]
    data = []
    for r in rows:
        data.append(
            [
                r.get("scenario", "?"),
                r["batch_size"],
                f"{r['feat_ms']:.2f}",
                f"{r['encoder_ms']:.2f}",
                f"{r['decode_ms']:.2f}",
                f"{r['total_ms']:.2f}",
                f"{r['rtf_median']:.2f}",
                f"{r['rtf_p90']:.2f}",
                "n/a" if r["peak_mem_gb"] is None else f"{r['peak_mem_gb']:.2f}",
            ]
        )
    return tabulate(data, headers=header, tablefmt="github")


def write_analysis(profile: dict, headline: dict) -> str:
    """Write a short hotspot analysis to profile_analysis.md and return it."""
    pct = profile["bucket_pct"]
    buckets = profile["buckets"]
    ranked = sorted(pct.items(), key=lambda kv: kv[1], reverse=True)
    top3 = ranked[:3]
    total_cuda_ms = profile["total_cuda_time_us"] / 1000.0
    per_pass_cuda_ms = total_cuda_ms / 3.0  # profile ran 3 generate passes

    mm_pct = pct.get("matmul/gemm", 0.0)
    conv_pct = pct.get("conv", 0.0)
    attn_pct = pct.get("attention", 0.0)
    norm_pct = pct.get("norm", 0.0)
    mel_pct = pct.get("mel/feature", 0.0)
    decode_pct = pct.get("decoder/rnnt-tdt", 0.0)
    memops_pct = pct.get("memops", 0.0)
    ctrl_pct = pct.get("control/reduction", 0.0)
    act_pct = pct.get("activation", 0.0)

    # GPU-busy (CUDA utilisation) of the decode region. The profiled
    # per_pass_cuda_ms is the WHOLE generate (encoder + decode); the decode
    # wall is decode-only. To get a decode-only GPU-busy we subtract the
    # encoder's CUDA time (the encoder is compute-bound, so its CUDA time
    # ~= its wall time encoder_ms). We ALSO report the whole-generate
    # GPU-busy (whole/whole, apples-to-apples) so both scopes are visible.
    decode_wall_ms = headline["decode_ms"]
    gen_wall_ms = headline["gen_ms"]
    enc_wall_ms = headline["encoder_ms"]
    decode_cuda_ms = max(0.0, per_pass_cuda_ms - enc_wall_ms)
    decode_gpu_busy = (decode_cuda_ms / decode_wall_ms * 100.0) if decode_wall_ms > 0 else 0.0
    gen_gpu_busy = (per_pass_cuda_ms / gen_wall_ms * 100.0) if gen_wall_ms > 0 else 0.0

    lines: list[str] = []
    lines.append("# Profiler hotspot analysis (parakeet-tdt-0.6b-v3 baseline)\n")
    lines.append(
        f"Profiled config: **batch=8, medium-length audio** ({headline['audio_seconds']:.0f}s audio), "
        f"bf16, RTX 5090 (sm_120, cu130).  \n"
        f"Profiled CUDA self-time: {total_cuda_ms:.1f} ms over 3 generate passes "
        f"({per_pass_cuda_ms:.1f} ms/pass).\n"
    )
    lines.append("## Top-3 CUDA-time buckets (% of CUDA self-time)\n")
    for name, p in top3:
        lines.append(f"- **{name}**: {p:.1f}%")
    lines.append("")
    lines.append("## The headline finding: the decode loop is LAUNCH-BOUND\n")
    lines.append(
        f"The TDT decode region takes **{decode_wall_ms:.0f} ms** wall-clock per pass "
        f"but only **~{decode_cuda_ms:.0f} ms** of decode-only CUDA self-time "
        f"(~{decode_gpu_busy:.0f}% decode GPU-busy; whole-generate is ~{gen_gpu_busy:.0f}% "
        f"= {per_pass_cuda_ms:.0f} ms CUDA over {gen_wall_ms:.0f} ms generate wall). "
        f"The remaining ~{100-decode_gpu_busy:.0f}% of the decode region is CPU "
        f"launch / Python-loop / sync overhead -- the autoregressive transducer loop "
        f"dispatches thousands of tiny kernels (see `aten::copy_` x5958, "
        f"`aten::where` x2502, `aten::eq` x2631 in the op table) and the GPU sits "
        f"idle between them. **Reducing launch overhead (CUDA-graph the decode loop, "
        f"fuse the joint+decoder into a megakernel, batch decode steps) is the "
        f"single highest-leverage optimization**, not faster GEMMs."
    )
    lines.append("\n## Per-stage CUDA breakdown\n")
    lines.append(
        f"1. **GEMM/matmul ({mm_pct:.1f}%)** -- the largest CUDA consumer. Dominated "
        f"by the fused FFN matmuls in the 24 Conformer layers (`cutlass "
        f"s16816gemm_relu`, i.e. Linear+SiLU/ReLU fused) plus the encoder "
        f"projection/joint matmuls (`aten::mm`). On the ENCODER side this is "
        f"compute-bound, so FP8 `_scaled_mm` + FFN fusion are the win there."
    )
    lines.append(
        f"2. **Conv ({conv_pct:.1f}%)** -- the kernel-9 depthwise conv module in "
        f"every Conformer block (`cudnn_convolution` + `_conv_depthwise2d`). A "
        f"custom Triton conv / channels-last path is a solid secondary target."
    )
    lines.append(
        f"3. **Attention ({attn_pct:.1f}%)** -- already flash/MemEff-backed "
        f"(`_efficient_attention_forward`); the subsampled sequence is short (T/8), "
        f"so attention is cheap. Low priority."
    )
    lines.append(
        f"4. **Memops + control/reduction ({memops_pct:.1f}% + {ctrl_pct:.1f}%)** -- "
        f"`copy_`, `fill_`, `argmax`, `where`, `eq`, `index`, `any`: these are the "
        f"debris of the Python decode loop (masking, frame-pointer arithmetic, "
        f"greedy argmax). They are individually tiny but collectively large because "
        f"they run once per decode step. Fusing the decode step removes most of them."
    )
    lines.append(
        f"5. **Decoder LSTM ({decode_pct:.1f}%)** -- `_cudnn_rnn`; cheap on CUDA but "
        f"called 399x (per-step). Fuse with the joint into a single kernel."
    )
    lines.append(
        f"6. **Norm ({norm_pct:.1f}%) / Activation ({act_pct:.1f}%)** -- three norms "
        f"per block; a fused norm+activation kernel is a moderate win across 24 "
        f"layers."
    )
    lines.append(
        f"7. **Feature extraction ({mel_pct:.1f}% CUDA)** -- runs on CPU + H2D "
        f"(see feat_ms = {headline['feat_ms']:.0f} ms in the timing table); not a "
        f"CUDA hotspot but a latency one -- move mel onto GPU later."
    )
    lines.append(
        f"\n## Headline RTF (batch=8 medium)\n"
        f"RTF(median) = **{headline['rtf_median']:.1f}x** realtime | total "
        f"{headline['total_ms']:.0f} ms (feat {headline['feat_ms']:.0f} / encoder "
        f"{headline['encoder_ms']:.0f} / decode {headline['decode_ms']:.0f}). The "
        f"encoder is compute-bound (GEMM+conv); the decode is launch-bound. Next "
        f"phase should attack the decode launch overhead first (biggest wall-clock "
        f"piece) and the encoder GEMMs second."
    )
    text = "\n".join(lines) + "\n"
    (OUTPUTS / "profile_analysis.md").write_text(text)
    return text


def main() -> int:
    _suppress_noisy_warnings()
    print("=" * 72)
    print("megapar baseline: oracle + RTF bench + profiler")
    print("=" * 72)

    fixtures = mkfx.load_fixtures()
    for name, arr in fixtures.items():
        print(f"  fixture {name:7s}: {len(arr)/SAMPLE_RATE:6.2f}s")

    runner = BaselineRunner(model_id=MODEL_ID, device="cuda", dtype=torch.bfloat16)
    n_params = runner.param_count()
    print(
        f"\n  model: {MODEL_ID}  params={n_params} ({n_params/1e9:.2f}B)  "
        f"encoder={runner.encoder_attr_path} ({runner.encoder_class_name})  "
        f"dtype=bf16"
    )

    # ---- C. correctness oracle ----
    print("\n--- correctness oracle ---")
    oracle = build_oracle(runner, fixtures)
    (OUTPUTS / "oracle.json").write_text(json.dumps(oracle, indent=2, ensure_ascii=False))
    # quick sanity: transcripts must be non-empty English
    for entry in oracle:
        if not entry["text"].strip():
            raise RuntimeError(f"oracle transcript EMPTY for {entry['name']} -- aborting")

    # ---- D/E. timing harness + bench ----
    print("\n--- timing harness (median/p90, cuda events, warmup=8) ---")
    stage_results, headline = run_bench(runner, fixtures)

    print("\n==================== RTF table ====================")
    print(fmt_table(stage_results + [headline]))
    print("\n*** HEADLINE (batch=8 medium) ***")
    print(
        f"    RTF(median) = {headline['rtf_median']:.2f}x realtime  |  "
        f"RTF(p90) = {headline['rtf_p90']:.2f}x  |  total = {headline['total_ms']:.2f} ms\n"
        f"    feat = {headline['feat_ms']:.2f} ms | encoder = {headline['encoder_ms']:.2f} ms "
        f"| decode = {headline['decode_ms']:.2f} ms | peak_mem = {headline['peak_mem_gb']:.2f} GB"
    )

    bench_payload = {
        "model_id": MODEL_ID,
        "dtype": "bfloat16",
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "encoder_path": runner.encoder_attr_path,
        "encoder_class": runner.encoder_class_name,
        "param_count": n_params,
        "arch_dims": ARCH_DIMS,
        "timing": {
            "method": "explicit split (feat=processor+H2D, encoder=get_audio_features, "
                      "decode=generate-encoder, total=end-to-end)",
            "warmup": 8,
            "repeats": 20,
            "max_seconds": 12,
            "stat": "median + p90 (cuda.Event)",
        },
        "scenario_batches": stage_results,
        "headline": {**headline, "length": HEADLINE_LENGTH},
    }
    (OUTPUTS / "baseline_bench.json").write_text(json.dumps(bench_payload, indent=2))

    # ---- F. profiler ----
    print("\n--- torch.profiler hotspot report ---")
    profile = run_profile(runner, fixtures)
    (OUTPUTS / "profile_top_ops.txt").write_text(profile["table"])
    (OUTPUTS / "profile_stages.json").write_text(
        json.dumps(
            {
                "total_cuda_time_us": profile["total_cuda_time_us"],
                "bucket_pct": profile["bucket_pct"],
                "buckets": profile["buckets"],
                "config": {"batch": HEADLINE_BATCH, "length": HEADLINE_LENGTH, "dtype": "bf16"},
            },
            indent=2,
        )
    )

    # print bucket table
    print("\n--- CUDA-time buckets ---")
    btab = [[b, d["pct"], f"{d['cuda_time_us']/1000:.2f}"] for b, d in profile["buckets"].items()]
    print(tabulate(btab, headers=["bucket", "% cuda", "ms"], tablefmt="github"))

    analysis = write_analysis(profile, headline)
    print("\n--- profile_analysis.md (head) ---")
    print("\n".join(analysis.splitlines()[:14]))

    # report peak mem at batch=16
    peak16 = next(r for r in stage_results if r["batch_size"] == 16)["peak_mem_gb"]
    print(f"\npeak GPU mem @ batch=16 (mixed): {peak16:.2f} GB")

    print("\noutputs written:")
    for f in sorted(OUTPUTS.glob("*")):
        print(f"  {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
