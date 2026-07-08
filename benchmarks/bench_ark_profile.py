"""Deep per-stage profiler for the ARK-ASR-3B pipeline.

Breaks the end-to-end transcribe path into every stage and times each with
CUDA events (device) and perf_counter (host), over the short/medium/long
fixtures plus a synthetic 30s clip. Prints ms + the host/device gap to spot
CPU-bound stages (gap large => host is the bottleneck, not the GPU).

  TRUST_REMOTE_CODE=1 uv run python benchmarks/bench_ark_profile.py
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(REPO_ROOT / "src"))

from starling.ark.audio import build_inputs_embeds, build_prompt_ids, extract_mel
from starling.ark.config import EOS_TOKEN_ID
from starling.ark.pipeline import MegaPipeline

FIXTURES = REPO_ROOT / "tests" / "fixtures"


def _wav(name: str) -> np.ndarray:
    wav, sr = sf.read(str(FIXTURES / f"{name}.wav"))
    if wav.ndim > 1:
        wav = wav[:, 0]
    return np.ascontiguousarray(wav, dtype=np.float32)


def _evt_pair():
    return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


@contextmanager
def _timed_block():
    """Context manager yielding (dev_times, host_times) lists."""
    pass


def _timed(fn, warmup=3, iters=10):
    """Return (median_device_ms, median_host_ms, last_output)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dev_times, host_times = [], []
    out = None
    for _ in range(iters):
        s, e = _evt_pair()
        t0 = time.perf_counter()
        s.record()
        out = fn()
        e.record()
        torch.cuda.synchronize()
        host_times.append((time.perf_counter() - t0) * 1000.0)
        dev_times.append(s.elapsed_time(e))
    return float(np.median(dev_times)), float(np.median(host_times)), out


@torch.inference_mode()
def main() -> int:
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    print("building pipeline ...", flush=True)
    pipe = MegaPipeline.from_pretrained(max_cache_len=4096)
    pipe.prewarm()
    print("ready.\n", flush=True)

    fixtures = [
        ("short", _wav("short")),
        ("medium", _wav("medium")),
        ("long", _wav("long")),
    ]
    fixtures.append(("synth30", np.zeros(int(30 * 16000), dtype=np.float32)))

    for name, wav in fixtures:
        dur = len(wav) / 16000
        hop = int(getattr(pipe.processor.feature_extractor, "hop_length", 160))
        n_mel_frames = len(wav) // hop

        stages = {}

        # 1. mel extraction (CPU FFT in WhisperFeatureExtractor)
        dev, host, mel = _timed(lambda: extract_mel(pipe.processor, [wav]))
        stages["mel(extract)"] = (dev, host)
        mel_bf = mel.to(dtype=torch.bfloat16, device="cuda")
        dev, host, _ = _timed(lambda: mel.to(dtype=torch.bfloat16, device="cuda"))
        stages["mel(to cuda)"] = (dev, host)

        # 2. prompt id construction (tokenization)
        dev, host, input_ids = _timed(
            lambda: build_prompt_ids(
                pipe.processor.tokenizer, "Transcribe the audio to text.",
                n_mel_frames=n_mel_frames,
            ).to("cuda")
        )
        stages["prompt_ids(tok)"] = (dev, host)

        # 3. fused encoder
        dev, host, af = _timed(lambda: pipe.fused_encoder(mel_bf))
        stages["encoder(graph)"] = (dev, host)

        # 4. audio-embedding injection
        dev, host, ie = _timed(
            lambda: build_inputs_embeds(pipe.model, input_ids, af)
        )
        stages["inject_embeds"] = (dev, host)

        T = int(ie.shape[1])
        llm = pipe._get_llm(T)

        # 5. prefill
        dev, host, _ = _timed(lambda: llm.prefill(ie), warmup=2)
        stages["prefill"] = (dev, host)

        # 6. decode (K-step). Warm the captured graph first.
        llm._reset_cache_pos(0)
        first_tok = llm.prefill(ie)
        if not getattr(llm, "_ms_captured", False):
            llm.capture(first_tok, T)
        llm._reset_to_chunk_start(T, first_tok)
        K = llm.K

        def _k_replay():
            llm._ms_graph.replay()
            llm._reset_to_chunk_start(T, first_tok)

        dev, host, _ = _timed(_k_replay, warmup=3, iters=20)
        stages[f"decode(K={K} replay)"] = (dev, host)
        per_tok_ms = dev / K

        # 7. full generate (wall clock)
        dev, host, res = _timed(
            lambda: llm.generate(ie, max_new_tokens=200, eos_token_id=EOS_TOKEN_ID),
            warmup=1, iters=3,
        )
        stages[f"generate({res.n_tokens}tok)"] = (dev, host)

        print(f"=== {name}  ({dur:.1f}s, T={T}, decode/tok={per_tok_ms:.2f}ms,"
              f" {1000/per_tok_ms:.0f} tok/s) ===")
        print(f"  {'stage':<24s} {'device':>8s} {'host':>8s} {'gap':>7s}")
        for sname, (d, h) in stages.items():
            print(f"  {sname:<24s} {d:>7.2f}m {h:>7.2f}m {h-d:>6.2f}m")
        core = {k: v[0] for k, v in stages.items() if "generate" not in k}
        total_core = sum(core.values())
        print(f"  {'TOTAL(stages, device)':<24s} {total_core:>7.1f}m"
              f"   (RTFx {dur/(total_core/1000):.0f}x on stage sum)")
        print(flush=True)

    print(f"GPU peak memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
