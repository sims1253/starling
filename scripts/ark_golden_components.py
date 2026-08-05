"""Capture staged, self-describing ARK-ASR-3B reference goldens.

Hooks the eager HF model to capture the intermediate tensors the ggml C++ port
must reproduce, one stage at a time, so divergence can be localized during
porting:

  mel              Whisper log-mel (128, mel_T) f32
  encoder_hidden   Whisper encoder output + ARK LayerNorm, before the adapter
                   merge/reshape (B, T_enc, 1280)
  audio_embeds     adapter output after merge-by-4 + MLP (B, N, 2048)
  prompt_ids       the chat-templated prompt token ids (1, T)
  inputs_embeds    multimodal embeddings with audio scattered in (1, T, 2048)
  prefill_logits   the first-step logits whose argmax seeds greedy decode
                   (1, 1, vocab)

The end-to-end emitted token ids are also re-captured and asserted equal to
golden/ark_reference.json so the staged tensors correspond to the exact decode
the ggml port must match.

Usage (from the repo root):
    TRUST_REMOTE_CODE=1 uv run python scripts/ark_golden_components.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

NAMES = ("short", "medium", "long")
STAGES = (
    "mel",
    "encoder_hidden",
    "audio_embeds",
    "prompt_ids",
    "inputs_embeds",
    "prefill_logits",
)
MODEL_ID = "AutoArk-AI/ARK-ASR-3B"
DEFAULT_INSTRUCTION = "Transcribe the audio to text."


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    wav, sr = sf.read(str(path))
    if wav.ndim > 1:
        wav = wav[:, 0]
    return wav.astype(np.float32), int(sr)


def cpu_float(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to(device="cpu", dtype=torch.float32).contiguous()


def shape_dtype(t: torch.Tensor) -> dict[str, Any]:
    return {"shape": list(t.shape), "dtype": str(t.dtype).removeprefix("torch.")}


def build_inputs(model: Any, wav: np.ndarray, proc: Any) -> dict:
    """Build the processor outputs for one utterance (the eager golden path)."""
    conv = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "array": wav},
                {"type": "text", "text": DEFAULT_INSTRUCTION},
            ],
        }
    ]
    data = proc.apply_chat_template(
        conv,
        audio_torch_dtype=torch.bfloat16,
        tokenize=True,
        return_tensors="pt",
        add_generation_prompt=True,
    )
    return {k: (v.to("cuda") if isinstance(v, torch.Tensor) else v) for k, v in data.items()}


def capture_stages(model: Any, proc: Any, wav: np.ndarray) -> tuple[dict, list[int]]:
    """Run one forward + greedy decode with hooks grabbing every stage tensor.

    Hooks:
      - encoder_hidden: capture the output of ``audio_encoder.layer_norm`` (the
        ARK post-encoder LayerNorm), i.e. the (B, T_enc, 1280) tensor just before
        the adapter merge/reshape.
      - audio_embeds: capture the output of ``audio_encoder.adapting`` (the MLP),
        i.e. the (B, N, 2048) adapter features.

    mel is taken from the processor's ``input_features``; prompt_ids and
    inputs_embeds from the forward; prefill_logits from the first logits.
    """
    data = build_inputs(model, wav, proc)
    mel = data["audios"]  # (1, 128, mel_T) bf16 -- the processor emits the mel here
    prompt_ids = data["input_ids"]  # (1, T)

    captured: dict[str, torch.Tensor] = {}

    def hook_adapting(_m: torch.nn.Module, _i, out):
        captured["audio_embeds"] = out

    def hook_layer_norm(_m: torch.nn.Module, _i, out):
        captured["encoder_hidden"] = out

    h1 = model.audio_encoder.adapting.register_forward_hook(hook_adapting)
    h2 = model.audio_encoder.layer_norm.register_forward_hook(hook_layer_norm)

    try:
        with torch.inference_mode():
            # First forward: prefill (past_len==0) injects audio and yields
            # logits over the last position; hook fires during injection.
            out = model(
                input_ids=prompt_ids,
                audios=data["audios"],
                attention_mask=None,
                use_cache=True,
                past_key_values=None,
            )
            prefill_logits = out.logits[:, -1:, :]  # (1, 1, vocab)

            # Reconstruct inputs_embeds the same way the model did: re-derive
            # the (audio-injected) ids. The model's forward set inputs_embeds
            # internally then called self.model(...); we replicate the inject
            # path manually using the captured audio_embeds.
            mask = prompt_ids == model.audio_token_id
            llm_ids = torch.where(mask, 0, prompt_ids)
            inputs_embeds = model.model.embed_tokens(llm_ids)
            n_slots = int(mask[0].sum())
            feat = captured["audio_embeds"][0]
            sa = int(feat.shape[0])
            if sa < n_slots:
                feat = torch.cat([feat, feat.new_zeros((n_slots - sa, feat.shape[-1]))], dim=0)
            elif sa > n_slots:
                feat = feat[:n_slots]
            pos = mask[0].nonzero(as_tuple=False).squeeze(-1)
            inputs_embeds[0, pos, :] = feat.to(inputs_embeds.dtype)

            # Greedy decode the rest using the model's own generate for the e2e
            # id check (matches make_ark_golden exactly).
            captured["prefill_logits"] = prefill_logits
            captured["mel"] = mel
            captured["prompt_ids"] = prompt_ids
            captured["inputs_embeds"] = inputs_embeds
    finally:
        h1.remove()
        h2.remove()

    # Full e2e greedy decode (separate call; matches make_ark_golden).
    with torch.inference_mode():
        T = prompt_ids.shape[1]
        out_full = model.generate(
            input_ids=prompt_ids,
            audios=data["audios"],
            attention_mask=None,
            max_new_tokens=200,
            do_sample=False,
        )
    gen_ids = out_full[0][T:].cpu().tolist()
    return captured, gen_ids


def verify_saved(gdir: Path) -> None:
    expected = [gdir / f"ark_{name}_{stage}.pt" for name in NAMES for stage in STAGES]
    expected += [gdir / f"ark_{name}_meta.json" for name in NAMES]
    missing = [str(p) for p in expected if not p.is_file()]
    assert not missing, f"missing goldens: {missing}"
    for p in expected:
        if p.suffix == ".pt":
            v = torch.load(p, map_location="cpu", weights_only=True)
            assert isinstance(v, torch.Tensor), f"{p} did not reload as a tensor"
        else:
            json.loads(p.read_text())


def main() -> int:
    from transformers import AutoModelForCausalLM, AutoProcessor
    from starling.parakeet.gpu_lock import with_gpu_lock

    gdir = REPO / "golden"
    gdir.mkdir(exist_ok=True)
    rows: list[tuple[str, str, str, str]] = []

    with with_gpu_lock(
        session="ggml-goldens",
        model="ARK-ASR-3B",
        eta_min=20,
        note="capturing staged ARK C++ reference goldens",
    ):
        print("[ark-components] loading model ...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            dtype=torch.bfloat16,
            device_map="cuda",
            trust_remote_code=True,
            attn_implementation="eager",
        )
        model.eval()
        proc = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

        ref = json.loads((gdir / "ark_reference.json").read_text())

        for name in NAMES:
            wav, sr = load_wav(REPO / "tests" / "fixtures" / f"{name}.wav")
            seconds = len(wav) / sr
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            captured, gen_ids = capture_stages(model, proc, wav)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            expected_ids = ref[name]["ids"]
            assert gen_ids == expected_ids, (
                f"{name}: captured decode {gen_ids[:12]}... diverged from "
                f"ark_reference.json {expected_ids[:12]}..."
            )
            first_token = int(captured["prefill_logits"].argmax(dim=-1).item())
            assert first_token == gen_ids[0], (
                f"{name}: prefill argmax {first_token} != first decoded token {gen_ids[0]}"
            )

            saved = {
                "mel": cpu_float(captured["mel"]),
                "encoder_hidden": cpu_float(captured["encoder_hidden"]),
                "audio_embeds": cpu_float(captured["audio_embeds"]),
                "prompt_ids": captured["prompt_ids"]
                .detach()
                .to(device="cpu", dtype=torch.int64)
                .contiguous(),
                "inputs_embeds": cpu_float(captured["inputs_embeds"]),
                "prefill_logits": cpu_float(captured["prefill_logits"]),
            }
            for stage, tensor in saved.items():
                torch.save(tensor, gdir / f"ark_{name}_{stage}.pt")
                rows.append(
                    (name, stage, str(tuple(tensor.shape)), str(tensor.dtype).removeprefix("torch."))
                )

            meta = {
                "model": MODEL_ID,
                "fixture": f"tests/fixtures/{name}.wav",
                "seconds": seconds,
                "mel_frontend": {
                    "sample_rate": 16000,
                    "mel_dim": 128,
                    "n_fft": 400,
                    "hop_length": 160,
                },
                "n_prompt_tokens": int(captured["prompt_ids"].shape[1]),
                "n_audio_tokens": int((captured["prompt_ids"][0] == 151663).sum()),
                "first_generated_token_id": first_token,
                "reference_ids": gen_ids,
                "reference_text": ref[name]["text"],
                "tensors": {stage: shape_dtype(t) for stage, t in saved.items()},
            }
            (gdir / f"ark_{name}_meta.json").write_text(
                json.dumps(meta, indent=2, sort_keys=True) + "\n"
            )
            print(
                f"[ark-components] {name}: ids verified ({len(gen_ids)} tok), {elapsed:.1f}s"
            )

    verify_saved(gdir)
    print("\nfixture  stage            shape                      dtype")
    print("-------  ---------------  --------------------------  -------")
    for name, stage, shape, dtype in rows:
        print(f"{name:<7}  {stage:<15}  {shape:<26}  {dtype}")
    print(f"\n[ark-components] verified {len(rows)} component files reload correctly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
