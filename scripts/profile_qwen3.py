"""Stage-level profiler for the Qwen3-ASR megakernel.

Splits wall time into: mel/processor, encoder, projector, merge, prefill, and
per-token decode, on short/medium/long tiers. Uses CUDA events for GPU time +
wall clock for host. The output tells us which stage to attack next.
"""

from __future__ import annotations

import statistics

import torch

from starling.qwen3.audio import build_inputs, load_wav
from starling.qwen3.config import MODEL_ID, REPO_ROOT
from starling.qwen3.loader import get_components
from starling.qwen3.pipeline import MegaPipeline


def _fixture_path(name: str) -> str:
    p = REPO_ROOT / "tests" / "fixtures" / name
    if p.exists():
        return str(p)
    return str(REPO_ROOT.parent / "starling" / "tests" / "fixtures" / name)


def _cuda_ms(fn, warmup=3, iters=10):
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts)


@torch.inference_mode()
def main() -> int:
    print(f"[profile] loading {MODEL_ID} ...")
    pipe = MegaPipeline.from_pretrained(encoder_mode="eager", max_cache_len=4096)
    comps = get_components(pipe.model)
    enc = comps["encoder"]
    proj = comps["projector"]

    for label, fname in [("short", "short.wav"), ("medium", "medium.wav"), ("long", "long.wav")]:
        wav, sr = load_wav(_fixture_path(fname))
        inputs = build_inputs(pipe.processor, wav, sr=sr)
        feats = inputs["input_features"]
        mask = inputs["input_features_mask"]
        input_ids = inputs["input_ids"]
        audio_s = wav.shape[1] / sr

        # encoder
        enc_ms = _cuda_ms(lambda: enc(input_features=feats, input_features_mask=mask, return_dict=True))
        enc_lhs = enc(input_features=feats, input_features_mask=mask, return_dict=True).last_hidden_state

        # projector
        proj_ms = _cuda_ms(lambda _lhs=enc_lhs: proj(_lhs.clone()))
        audio_embeds = proj(enc_lhs.clone())

        # merge
        def _merge(_emb=audio_embeds, _ids=input_ids) -> torch.Tensor:
            return pipe.build_inputs_embeds(_ids, _emb)

        merge_ms = _cuda_ms(_merge)
        inputs_embeds = _merge()

        # prefill
        def _prefill(_emb=inputs_embeds) -> torch.Tensor:
            pipe.llm._reset_cache_pos(0)
            return pipe.llm.prefill(_emb)

        prefill_ms = _cuda_ms(_prefill, warmup=2, iters=5)
        first_tok = pipe.llm.prefill(inputs_embeds)
        T = inputs_embeds.shape[1]

        # capture decode graph for steady-state per-token timing
        pipe.llm._captured = False
        pipe.llm.capture(first_tok, T)
        pipe.llm.static_input_ids.copy_(first_tok.reshape(1, 1))
        pipe.llm.static_position_ids.copy_(torch.tensor([[T]], device="cuda"))
        pipe.llm._set_mask(T + 1)

        def _one_decode():
            pipe.llm._graph.replay()
            pipe.llm._reset_cache_pos(T)

        decode_ms = _cuda_ms(_one_decode, warmup=3, iters=20)
        decode_tps = 1000.0 / decode_ms if decode_ms > 0 else 0.0

        print(
            f"[profile] {label:7s} {audio_s:5.1f}s prompt={T:4d} | "
            f"enc {enc_ms:6.1f}ms proj {proj_ms:5.1f} merge {merge_ms:5.1f} "
            f"prefill {prefill_ms:6.1f} | decode {decode_ms:5.2f}ms/tok ({decode_tps:6.1f} tok/s)"
        )
        # reset for next tier
        pipe.llm._captured = False
        pipe.llm._reset_cache_pos(0)
        del enc_lhs, audio_embeds, inputs_embeds
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
