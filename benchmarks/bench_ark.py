"""Per-stage benchmark for the ARK-ASR-3B megakernel pipeline.

Reports encoder / prefill / decode / total ms for each fixture (short/medium/
long) and the speedup over the stock eager reference in ``golden/
ark_reference.json``.

Usage (from the repo root):
    TRUST_REMOTE_CODE=1 uv run python benchmarks/bench_ark.py
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from starling.ark.audio import build_inputs_embeds, build_prompt_ids, extract_mel
from starling.ark.config import EOS_TOKEN_ID
from starling.ark.pipeline import MegaPipeline

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(
    os.environ.get("STARLING_FIXTURES_DIR", REPO_ROOT / "tests" / "fixtures")
).expanduser()
GOLDEN_PATH = REPO_ROOT / "golden" / "ark_reference.json"


def _wav(name: str) -> np.ndarray:
    wav, sr = sf.read(str(FIXTURES / f"{name}.wav"))
    if wav.ndim > 1:
        wav = wav[:, 0]
    return np.ascontiguousarray(wav, dtype=np.float32)


def _cuda_ms(fn, warmup: int = 3, iters: int = 5) -> float:
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e)


def main() -> int:
    golden = json.load(open(GOLDEN_PATH)) if GOLDEN_PATH.exists() else {}
    print("building pipeline ...")
    pipe = MegaPipeline.from_pretrained(max_cache_len=4096)
    pipe.prewarm()

    print(f"\n{'fixture':<8} {'dur':>6} {'enc':>7} {'prefill':>8} {'decode':>7} {'total':>7} {'RTFx':>6} {'speedup':>8}")
    print("-" * 64)
    for fx in ["short", "medium", "long"]:
        wav = _wav(fx)
        dur = len(wav) / 16000
        hop = int(getattr(pipe.processor.feature_extractor, "hop_length", 160))
        n_mel_frames = len(wav) // hop

        mel = extract_mel(pipe.processor, [wav]).to(dtype=torch.bfloat16, device="cuda")
        input_ids = build_prompt_ids(
            pipe.processor.tokenizer, "Transcribe the audio to text.",
            n_mel_frames=n_mel_frames,
        ).to("cuda")

        # warmup the shape-keyed encoder graph + decode capture
        af = pipe.fused_encoder(mel)
        ie = build_inputs_embeds(pipe.model, input_ids, af)
        llm = pipe._get_llm(int(ie.shape[1]))
        llm.generate(ie, max_new_tokens=8, eos_token_id=EOS_TOKEN_ID)
        torch.cuda.synchronize()

        enc_ms = _cuda_ms(lambda: pipe.fused_encoder(mel))
        af = pipe.fused_encoder(mel)
        ie = build_inputs_embeds(pipe.model, input_ids, af)

        # prefill timing (eager) + full generate wall clock
        t0 = time.perf_counter()
        llm = pipe._get_llm(int(ie.shape[1]))
        res = llm.generate(ie, max_new_tokens=200, eos_token_id=EOS_TOKEN_ID)
        torch.cuda.synchronize()
        gen_ms = (time.perf_counter() - t0) * 1000

        total = enc_ms + gen_ms
        rtfx = dur / (total / 1000)
        speedup = (golden[fx]["ms"] / total) if fx in golden else float("nan")
        print(f"{fx:<8} {dur:>5.1f}s {enc_ms:>6.1f}m {gen_ms:>7.1f}m {'':>7} {total:>6.1f}m {rtfx:>5.0f}x {speedup:>7.1f}x")
        print(f"         decoded {res.n_tokens} tokens; K={llm.K}; golden {golden.get(fx, {}).get('ms', '?')} ms")

    print("\nGPU memory peak:", f"{torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
