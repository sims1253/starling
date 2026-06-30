"""Batched B>1 benchmark + byte-exactness check vs single-stream.

Verifies each batched stream matches the B=1 transcript, then reports
aggregate RTFx across a B-size sweep (B=1,2,4,8,16) on the medium fixture
tiled B times.
"""

from __future__ import annotations

import statistics
import time

import torch

from starling.qwen3.audio import build_inputs, load_wav
from starling.qwen3.batched import BatchedPipeline
from starling.qwen3.config import MODEL_ID, REPO_ROOT
from starling.qwen3.pipeline import MegaPipeline


def _fixture_path(name: str) -> str:
    p = REPO_ROOT / "tests" / "fixtures" / name
    if p.exists():
        return str(p)
    return str(REPO_ROOT.parent / "starling" / "tests" / "fixtures" / name)


def main() -> int:
    from starling.parakeet.gpu_lock import with_gpu_lock

    print(f"[bench-batch] loading {MODEL_ID} ...")
    model, processor = MegaPipeline.from_pretrained.__wrapped__ if False else (None, None)
    # load fresh for batched
    from starling.qwen3.loader import load_model_and_processor

    model, processor = load_model_and_processor()

    wav, sr = load_wav(_fixture_path("medium.wav"))
    inputs = build_inputs(processor, wav, sr=sr)
    audio_s = wav.shape[1] / sr

    with with_gpu_lock(session="qwen3-asr", model=MODEL_ID, eta_min=30, note="batched sweep"):
        # (1) byte-exactness: B=4 batched vs B=1 single-stream
        print("[bench-batch] --- byte-exactness: batched vs single ---")
        single = MegaPipeline(model=model, processor=processor, max_cache_len=4096)
        text1, ids1 = single.transcribe(
            inputs["input_features"], inputs["input_ids"], inputs["input_features_mask"], max_new_tokens=200
        )

        bpipe = BatchedPipeline(model=model, processor=processor, max_batch_size=4, max_cache_len=4096)
        B = 4
        feats = [inputs["input_features"]] * B
        iids = [inputs["input_ids"]] * B
        masks = [inputs["input_features_mask"]] * B
        texts = bpipe.transcribe_batch(feats, iids, masks, max_new_tokens=200)
        match = all(t.strip() == text1.strip() for t in texts)
        print(f"[bench-batch] B={B} batched streams all match single-stream transcript: {match}")
        print(f"[bench-batch]   single: {text1[:70]!r}")
        print(f"[bench-batch]   batched[0]: {texts[0][:70]!r}")

        # (2) RTFx sweep
        print("[bench-batch] --- aggregate RTFx sweep (medium, tiled B) ---")
        del single, bpipe
        torch.cuda.empty_cache()
        for B in [1, 2, 4, 8, 16]:
            try:
                bpipe = BatchedPipeline(model=model, processor=processor, max_batch_size=B, max_cache_len=4096)
            except Exception as e:
                print(f"[bench-batch] B={B}: skip ({e})")
                continue
            feats = [inputs["input_features"]] * B
            iids = [inputs["input_ids"]] * B
            masks = [inputs["input_features_mask"]] * B
            # warmup
            bpipe.transcribe_batch(feats, iids, masks, max_new_tokens=200)
            times = []
            for _ in range(4):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                bpipe.transcribe_batch(feats, iids, masks, max_new_tokens=200)
                torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1000.0)
            ms = statistics.median(times)
            total_audio = audio_s * B
            rtfx = total_audio / (ms / 1000.0)
            peak = torch.cuda.max_memory_allocated() / 1e9
            print(f"[bench-batch] B={B:2d}: {ms:7.1f}ms  aggregate {total_audio:5.1f}s  RTFx={rtfx:7.1f}x  VRAM~{peak:.1f}GB")
            del bpipe
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
