"""Kernel-level profile of one K-step decode replay for ARK-ASR-3B.

Uses torch.profiler to attribute the ~6.5ms/token decode to specific kernel
groups, so we know whether to attack the GEMVs, the attention, or the
elementwise glue.

  TRUST_REMOTE_CODE=1 uv run python benchmarks/bench_ark_kernel_profile.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.profiler import ProfilerActivity, profile

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from starling.ark.audio import build_inputs_embeds, build_prompt_ids, extract_mel
from starling.ark.config import EOS_TOKEN_ID
from starling.ark.pipeline import MegaPipeline

FIXTURES = REPO_ROOT / "tests" / "fixtures"


@torch.inference_mode()
def main() -> int:
    pipe = MegaPipeline.from_pretrained(max_cache_len=4096)
    pipe.prewarm()

    wav, sr = sf.read(str(FIXTURES / "long.wav"))
    wav = np.ascontiguousarray(wav, dtype=np.float32)
    hop = int(pipe.processor.feature_extractor.hop_length)
    n_mel_frames = len(wav) // hop
    mel = extract_mel(pipe.processor, [wav]).to(dtype=torch.bfloat16, device="cuda")
    input_ids = build_prompt_ids(
        pipe.processor.tokenizer, "Transcribe the audio to text.",
        n_mel_frames=n_mel_frames,
    ).to("cuda")
    af = pipe.fused_encoder(mel)
    ie = build_inputs_embeds(pipe.model, input_ids, af)
    T = int(ie.shape[1])
    llm = pipe._get_llm(T)
    llm._reset_cache_pos(0)
    first_tok = llm.prefill(ie)
    llm.capture(first_tok, T)
    llm._reset_to_chunk_start(T, first_tok)
    K = llm.K

    # warmup the replay
    for _ in range(5):
        llm._ms_graph.replay()
        llm._reset_to_chunk_start(T, first_tok)
    torch.cuda.synchronize()

    print(f"=== one K={K} decode replay, grouped by kernel ===\n")
    with profile(
        activities=[ProfilerActivity.CUDA],
        record_shapes=False,
    ) as prof:
        for _ in range(20):
            llm._ms_graph.replay()
            llm._reset_to_chunk_start(T, first_tok)
        torch.cuda.synchronize()

    print(prof.key_averages(group_by_input_shape=False).table(
        sort_by="cuda_time_total", row_limit=25))
    print("\nNOTE: each row is per-K-step-replay; divide cuda_time by K for per-token.")

    # Prefill breakdown too
    print(f"\n=== one prefill (T={T}), grouped by kernel ===\n")
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(10):
            llm._reset_cache_pos(0)
            llm.prefill(ie)
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
