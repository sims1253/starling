"""Capture staged Qwen3-ASR-1.7B reference tensors (``golden/qwen3_<fix>_<stage>.pt``).

Companion to ``scripts/make_qwen3_golden.py``: per fixture (short only, like
the granite components script -- long audio repeats the short/medium shape),
persist the intermediate tensors the C++ engine's staged probes compare
against, and RE-VERIFY the end-to-end ids against the reference JSON so the
stages are anchored to the recorded token stream.

Stages:
  mel          valid log-mel frames (1, 128, T) f32 (pre chunk-padding)
  mel_padded   the processor's input_features (1, 128, T_pad) bf16
               (zero mel-pad out to the 100-frame chunk multiple)
  enc_hidden   packed encoder output (L, 1024) bf16
  audio_embeds projector output (L, 2048) bf16
  prompt_ids   chat-templated ids (1, P) incl. the <|audio_pad|> expansion
  inputs_embeds merged multimodal embeds (1, P, 2048) bf16
  prefill_logits first-step logits (1, 1, 151936) f32

Run after make_qwen3_golden.py (needs golden/qwen3_reference.json), GPU:
    uv run python scripts/qwen3_golden_components.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "tests" / "fixtures"
GOLDEN_REF = REPO_ROOT / "golden" / "qwen3_reference.json"

STAGES = ("mel", "mel_padded", "enc_hidden", "audio_embeds", "prompt_ids",
          "inputs_embeds", "prefill_logits")


def main() -> int:
    from starling.qwen3.audio import build_inputs, load_wav
    from starling.qwen3.loader import get_components, load_model_and_processor
    from starling.parakeet.gpu_lock import with_gpu_lock

    if not GOLDEN_REF.exists():
        raise SystemExit(
            f"{GOLDEN_REF} missing -- run scripts/make_qwen3_golden.py first"
        )
    ref = json.loads(GOLDEN_REF.read_text())

    with with_gpu_lock(
        session="ggml-goldens",
        model="Qwen3-ASR-1.7B-hf",
        eta_min=15,
        note="capturing qwen3 staged component goldens",
    ):
        print("[qwen3-components] loading model (eager, bf16) ...")
        model, processor = load_model_and_processor(attn_impl="eager")
        comps = get_components(model)
        encoder, projector = comps["encoder"], comps["projector"]
        embed = comps["language_model"].get_input_embeddings()
        audio_token_id = int(model.config.audio_token_id)

        captured: dict[str, torch.Tensor] = {}

        def enc_hook(_m, _args, output):
            captured["enc_hidden"] = output.last_hidden_state.detach()

        handle = encoder.register_forward_hook(enc_hook)
        try:
            wav, sr = load_wav(str(FIXTURES / "short.wav"))
            wav = wav.to("cuda")
            inputs = build_inputs(processor, wav, sr=sr)
            input_ids = inputs["input_ids"]
            feats = inputs["input_features"]
            mask = inputs["input_features_mask"]
            T = int(mask.sum())
            captured["mel"] = feats[0, :, :T].float().cpu()
            captured["mel_padded"] = feats.detach().cpu()
            captured["prompt_ids"] = input_ids.detach().cpu()

            with torch.inference_mode():
                audio = model.get_audio_features(
                    input_features=feats,
                    input_features_mask=mask,
                    return_dict=True,
                )
                captured["audio_embeds"] = audio.pooler_output.detach().cpu()
                # merged embeds, replicating Qwen3ASRModel.forward
                inputs_embeds = embed(input_ids)
                special = (input_ids == audio_token_id)
                special = special.unsqueeze(-1).expand_as(inputs_embeds)
                inputs_embeds = inputs_embeds.masked_scatter(
                    special, audio.pooler_output.to(inputs_embeds.dtype)
                )
                captured["inputs_embeds"] = inputs_embeds.detach().cpu()
                out = model(
                    input_ids=input_ids,
                    input_features=feats,
                    input_features_mask=mask,
                    use_cache=True,
                    logits_to_keep=1,
                )
                captured["prefill_logits"] = out.logits.detach().float().cpu()
        finally:
            handle.remove()

        # id re-verification: rerun the end-to-end decode at the reference
        # chunk's budget and check ids + first-step argmax.
        from starling.qwen3.pipeline import MegaPipeline

        chunk_ref = ref["fixtures"]["short"]["chunks"][0]
        pipe = MegaPipeline(model, processor, max_cache_len=4096,
                            encoder_mode="eager", use_fused_llm=False)
        with torch.inference_mode():
            text, ids = pipe.transcribe(
                inputs["input_features"], inputs["input_ids"],
                inputs["input_features_mask"],
                max_new_tokens=chunk_ref["budget"],
            )
        assert ids[0].cpu().tolist() == chunk_ref["ids"], (
            "short: end-to-end ids drifted from golden/qwen3_reference.json"
        )
        assert captured["prefill_logits"].argmax(dim=-1).item() == chunk_ref["ids"][0], (
            "short: prefill argmax != first reference token"
        )
        print(f"[qwen3-components] ids re-verified ({len(chunk_ref['ids'])} tokens)")

        meta = {
            "model": ref["model"],
            "fixture": "tests/fixtures/short.wav",
            "n_valid_frames": int(inputs["input_features_mask"].sum()),
            "padded_frames": int(inputs["input_features"].shape[2]),
            "n_audio_tokens": int((inputs["input_ids"] == audio_token_id).sum()),
            "prompt_len": int(inputs["input_ids"].shape[1]),
            "budget": chunk_ref["budget"],
            "first_generated_token_id": chunk_ref["ids"][0],
            "reference_ids": chunk_ref["ids"],
            "reference_text": chunk_ref["text"],
            "stages": {k: {"shape": list(v.shape), "dtype": str(v.dtype)}
                       for k, v in captured.items()},
        }
        (REPO_ROOT / "golden" / "qwen3_short_meta.json").write_text(
            json.dumps(meta, indent=2, sort_keys=True) + "\n"
        )
        for name, tensor in captured.items():
            path = REPO_ROOT / "golden" / f"qwen3_short_{name}.pt"
            torch.save(tensor.contiguous().cpu(), path)
            print(f"[qwen3-components] {path.name}: {tuple(tensor.shape)} {tensor.dtype}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
