"""Capture staged Nemotron-Labs-Audex-2B reference tensors (``golden/audex_short_<stage>.pt``).

Companion to ``scripts/make_audex_golden.py``: per fixture (short only, like
the granite/qwen3 components scripts -- long audio repeats the same fixed
3000-frame shapes), persist the intermediate tensors the C++ engine's staged
probes compare against, and RE-VERIFY the end-to-end ids against the
reference JSON so the stages are anchored to the recorded token stream.

Stages (all FIXED-shape per 30 s clip):
  mel            extractor input_features (1, 128, 3000) bf16 (padded clip)
  enc_hidden     post avg-pool + ln_post encoder output (750, 1280) bf16
  audio_embeds   projector output (750, 2048) bf16
  prompt_ids     chat-templated ids (1, 773) incl. the 750 <so_embedding> slots
  inputs_embeds  merged multimodal embeds (1, 773, 2048) bf16
  prefill_logits first-step logits (1, 1, 205312) f32

Run after make_audex_golden.py (needs golden/audex_reference.json), GPU:
    uv run python scripts/audex_golden_components.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Purge preamble (see scripts/make_audex_golden.py).
for _k in [k for k in list(sys.modules) if k == "starling" or k.startswith("starling.")]:
    del sys.modules[_k]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "tests" / "fixtures"
GOLDEN_REF = REPO_ROOT / "golden" / "audex_reference.json"

import torch  # noqa: E402


def main() -> int:
    from starling.audex.audio import build_inputs, load_wav
    from starling.audex.config import SOUND_TOKEN_ID
    from starling.audex.loader import load_model_and_processor
    from starling.audex.pipeline import MegaPipeline
    from starling.parakeet.gpu_lock import with_gpu_lock

    if not GOLDEN_REF.exists():
        raise SystemExit(
            f"{GOLDEN_REF} missing -- run scripts/make_audex_golden.py first"
        )
    ref = json.loads(GOLDEN_REF.read_text())

    with with_gpu_lock(
        session="ggml-goldens",
        model="Nemotron-Labs-Audex-2B",
        eta_min=15,
        note="capturing audex staged component goldens",
    ):
        print("[audex-components] loading model (eager, bf16) ...")
        model, tokenizer, feature_extractor = load_model_and_processor(
            attn_impl="eager"
        )

        captured: dict[str, torch.Tensor] = {}

        def enc_hook(_m, _args, output):
            captured["enc_hidden"] = output.last_hidden_state.detach()

        handle = model.audio_encoder.register_forward_hook(enc_hook)
        try:
            wav, sr = load_wav(str(FIXTURES / "short.wav"))
            inputs = build_inputs(tokenizer, feature_extractor, wav)
            input_ids = inputs["input_ids"]
            feats = inputs["input_features"]
            captured["mel"] = feats.detach().cpu()
            captured["prompt_ids"] = input_ids.detach().cpu()

            with torch.inference_mode():
                # encode_audio: stock encoder + stock projector (the hook
                # grabs the encoder's last_hidden_state on the way through).
                audio_embeds = model.encode_audio(feats)
                captured["audio_embeds"] = audio_embeds.detach().cpu()
                # merged embeds, replicating prepare_inputs_embeds exactly:
                # embed ids, boolean-mask scatter the projected audio rows.
                embed = model.model.embed_tokens
                inputs_embeds = embed(input_ids).clone()
                mask = (input_ids[0].to(inputs_embeds.device) == SOUND_TOKEN_ID)
                inputs_embeds[0, mask] = audio_embeds.to(inputs_embeds.dtype)
                captured["inputs_embeds"] = inputs_embeds.detach().cpu()
                out = model(
                    input_ids=input_ids,
                    input_features=feats,
                    use_cache=True,
                )
                # The audex remote-code forward ignores logits_to_keep;
                # keep only the last position (the greedy next token).
                captured["prefill_logits"] = out.logits[:, -1:, :].detach().float().cpu()
        finally:
            handle.remove()

        # id re-verification: rerun the end-to-end eager decode at the
        # reference chunk's budget and check ids + first-step argmax.
        chunk_ref = ref["fixtures"]["short"]["chunks"][0]
        pipe = MegaPipeline(
            model, tokenizer, feature_extractor,
            max_cache_len=4096, use_fused_llm=False,
        )
        with torch.inference_mode():
            text, ids = pipe.transcribe(wav, max_new_tokens=chunk_ref["budget"])
        assert ids[0].cpu().tolist() == chunk_ref["ids"], (
            "short: end-to-end ids drifted from golden/audex_reference.json"
        )
        assert captured["prefill_logits"].argmax(dim=-1).item() == chunk_ref["ids"][0], (
            "short: prefill argmax != first reference token"
        )
        print(f"[audex-components] ids re-verified ({len(chunk_ref['ids'])} tokens)")

        meta = {
            "model": ref["model"],
            "fixture": "tests/fixtures/short.wav",
            "n_valid_frames": int(feats.shape[2]),
            "n_audio_tokens": int((input_ids == SOUND_TOKEN_ID).sum()),
            "prompt_len": int(input_ids.shape[1]),
            "budget": chunk_ref["budget"],
            "first_generated_token_id": chunk_ref["ids"][0],
            "reference_ids": chunk_ref["ids"],
            "reference_text": chunk_ref["text"],
            "stages": {k: {"shape": list(v.shape), "dtype": str(v.dtype)}
                       for k, v in captured.items()},
        }
        (REPO_ROOT / "golden" / "audex_short_meta.json").write_text(
            json.dumps(meta, indent=2, sort_keys=True) + "\n"
        )
        for name, tensor in captured.items():
            path = REPO_ROOT / "golden" / f"audex_short_{name}.pt"
            torch.save(tensor.contiguous().cpu(), path)
            print(f"[audex-components] {path.name}: {tuple(tensor.shape)} {tensor.dtype}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
