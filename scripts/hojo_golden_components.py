"""Capture staged, self-describing HojoAI/Hojo-ASR-V1 reference goldens.

Hooks the eager Hojo ASR model (the installed ``hojo_asr`` package, run under
``.venv-hojo``) to capture the intermediate tensors the ggml C++ port must
reproduce, one stage at a time, so divergence can be localized during porting:

  mel              Whisper log-mel (T, 128) f32  -- the feat_extractor output
  audio_tower      speech_encoder output (n_speech, 2048) -- post proj2
  bottleneck       bottleneck output (B, T, 2560) -- post after_norm, pre ln_speech
  speech_embeds    ln_speech(bottleneck) (B, T, 2560) -- the final audio embeds
  inputs_embeds    cat([embed(bos), speech_embeds]) (B, S, 2560) feeding the
                   Qwen3 decoder prefill
  prefill_logits   the first-step logits whose argmax seeds beam-4 decode
                   (1, 1, vocab)

The end-to-end emitted token ids are re-captured via ``model.generate`` (beam-4)
and asserted equal to ``golden/hojo_reference.json``'s ``gen_ids`` so the staged
tensors correspond to the exact decode the ggml port must match.

Run under the isolated venv (from the repo root):
    .venv-hojo/bin/python scripts/hojo_golden_components.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

REPO = Path(__file__).resolve().parents[1]

NAMES = ("short", "medium", "long")
STAGES = (
    "mel",
    "audio_tower",
    "bottleneck",
    "speech_embeds",
    "inputs_embeds",
    "prefill_logits",
)
MODEL_DIR = REPO / ".hf-cache" / "hojo-asr-v1"


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    wav, sr = sf.read(str(path))
    if wav.ndim > 1:
        # Match the runtime path (waveform[0:1, :] -> channel 0), not an average
        # across channels; the reference and tests select channel 0.
        wav = wav[:, 0]
    return np.asarray(wav, dtype=np.float32), int(sr)


def cpu_float(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to(device="cpu", dtype=torch.float32).contiguous()


def shape_dtype(t: torch.Tensor) -> dict[str, Any]:
    return {"shape": list(t.shape), "dtype": str(t.dtype).removeprefix("torch.")}


def capture_stages(model, audio_np: np.ndarray) -> tuple[dict, list[int]]:
    """Run encode_speech (with hooks) + a beam-4 generate for the goldens."""
    captured: dict[str, torch.Tensor] = {}

    def hook_tower(_m, _i, out):
        if hasattr(out, "last_hidden_state"):
            captured["audio_tower"] = out.last_hidden_state
        else:
            captured["audio_tower"] = out[0]

    def hook_bottleneck(_m, _i, out):
        # ConformerEncoder.forward returns (xs, masks). Capture xs (pre ln_speech).
        if isinstance(out, tuple):
            captured["bottleneck"] = out[0]
        else:
            captured["bottleneck"] = out

    def hook_ln_speech(_m, _i, out):
        captured["speech_embeds"] = out

    h1 = model.speech_encoder.register_forward_hook(hook_tower)
    h2 = model.bottleneck.register_forward_hook(hook_bottleneck)
    h3 = model.ln_speech.register_forward_hook(hook_ln_speech)

    try:
        # ---- Build the mel the same way the dataset does (single utterance). ----
        waveform = torch.from_numpy(audio_np)
        feats = model.feat_extractor(
            waveform.numpy(), sampling_rate=16000, return_tensors="pt",
            padding=False).input_features
        mel = feats.squeeze(0).transpose(0, 1)  # (T, 128)
        captured["mel"] = mel

        # encode_speech + the decoder generate must run under the model's
        # autocast_context (fp16) to match the reference path exactly (the
        # bf16 decoder weights are autocast-compatible under fp16 generate).
        with model.autocast_context(), torch.inference_mode():
            # ---- encode_speech (fills tower/bottleneck/ln_speech hooks). ----
            spectrogram = mel.unsqueeze(0).to(model.device)  # (1, T, 128)
            spectrogram_lengths = torch.tensor([mel.shape[0]], dtype=torch.int64)
            speech_embeddings, speech_attn = model.encode_speech(
                spectrogram, spectrogram_lengths.to(model.device))

            # ---- Build inputs_embeds (the cat of bos + speech embeds). ----
            batch_size = 1
            bos_ids = (torch.ones(batch_size, 1, dtype=torch.int32,
                                  device=speech_embeddings.device) * model.bos_token_id)
            bos_embeds = model.decoder_model.model.embed_tokens(bos_ids)
            inputs_embeds = torch.cat([bos_embeds, speech_embeddings], dim=1)
            attention_mask = torch.cat([speech_attn[:, :1], speech_attn], dim=1)
            # Capture inputs_embeds in f32 (the C++ builds it in f32 then casts
            # at the prefill boundary; the f32 form is the reference).
            captured["inputs_embeds"] = inputs_embeds.detach().float()

            # ---- beam-4 generate (matches the golden capture path). Capture
            # scores[0] as the prefill logits. ----
            from hojo_asr.hojo_asr_model import StopOnTokenSequences
            from transformers import StoppingCriteriaList
            stop = torch.tensor([-100], device=model.device)
            criteria = StoppingCriteriaList(
                [StopOnTokenSequences(stop_token_seqs=[stop])])
            feat_len = speech_embeddings.size(1)
            max_new_tokens = max(min(200, int(feat_len * 2) + 10), 10)
            outputs = model.decoder_model.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=max_new_tokens,
                eos_token_id=model.tokenizer.eos_token_id,
                stopping_criteria=criteria,
                num_beams=4,
                do_sample=False,
                min_length=1,
                temperature=1.0,
                top_p=0.9,
                repetition_penalty=2.0,
                length_penalty=1.0,
                attention_mask=attention_mask,
                output_scores=True,
                return_dict_in_generate=True,
            )
            # scores[0] is the prefill logits over the vocab (shape
            # (num_beams*batch, vocab) at beam step 0). Reshape to (1,1,vocab).
            captured["prefill_logits"] = outputs.scores[0][:1].unsqueeze(1).detach()
            gen_ids = outputs.sequences[0].cpu().tolist()
    finally:
        h1.remove()
        h2.remove()
        h3.remove()

    return captured, gen_ids


def verify_saved(gdir: Path) -> None:
    expected = [gdir / f"hojo_{name}_{stage}.pt" for name in NAMES for stage in STAGES]
    expected += [gdir / f"hojo_{name}_meta.json" for name in NAMES]
    missing = [str(p) for p in expected if not p.is_file()]
    assert not missing, f"missing goldens: {missing}"
    for p in expected:
        if p.suffix == ".pt":
            v = torch.load(p, map_location="cpu", weights_only=True)
            assert isinstance(v, torch.Tensor), f"{p} did not reload as a tensor"
        else:
            json.loads(p.read_text())


def main() -> int:
    gdir = REPO / "golden"
    gdir.mkdir(exist_ok=True)
    rows: list[tuple[str, str, str, str]] = []

    print("[hojo-components] loading model ...", flush=True)
    from hojo_asr.hojo_asr_model import HOJO_ASR
    model = HOJO_ASR.load_model(str(MODEL_DIR), device="cuda:0")
    model.eval()

    ref_path = gdir / "hojo_reference.json"
    ref = json.loads(ref_path.read_text())

    for name in NAMES:
        wav, sr = load_wav(REPO / "tests" / "fixtures" / f"{name}.wav")
        seconds = len(wav) / sr
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        captured, gen_ids = capture_stages(model, wav)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        expected_ids = ref["fixtures"][name]["gen_ids"]
        assert len(gen_ids) == len(expected_ids), (
            f"{name}: captured decode length {len(gen_ids)} != golden length "
            f"{len(expected_ids)} (got {gen_ids[:12]}... expected {expected_ids[:12]}...)"
        )
        n = len(gen_ids)
        assert gen_ids == expected_ids, (
            f"{name}: captured decode diverged from hojo_reference.json at "
            f"index {next((i for i in range(n) if gen_ids[i] != expected_ids[i]), n)} "
            f"(got {gen_ids[:12]}... expected {expected_ids[:12]}...)"
        )

        saved = {
            "mel": cpu_float(captured["mel"]),
            "audio_tower": cpu_float(captured["audio_tower"]),
            "bottleneck": cpu_float(captured["bottleneck"]),
            "speech_embeds": cpu_float(captured["speech_embeds"]),
            "inputs_embeds": cpu_float(captured["inputs_embeds"]),
            "prefill_logits": cpu_float(captured["prefill_logits"]),
        }
        for stage, tensor in saved.items():
            torch.save(tensor, gdir / f"hojo_{name}_{stage}.pt")
            rows.append(
                (name, stage, str(tuple(tensor.shape)),
                 str(tensor.dtype).removeprefix("torch.")))

        meta = {
            "model": "HojoAI/Hojo-ASR-V1",
            "fixture": f"tests/fixtures/{name}.wav",
            "seconds": seconds,
            "mel_frontend": {
                "sample_rate": 16000, "mel_dim": 128,
                "n_fft": 400, "hop_length": 160,
            },
            "n_speech_embeds": int(saved["speech_embeds"].shape[1]),
            "n_inputs_embeds_tokens": int(saved["inputs_embeds"].shape[1]),
            "first_generated_token_id": int(
                captured["prefill_logits"].argmax(dim=-1).item()),
            "reference_ids": gen_ids,
            "reference_text": ref["fixtures"][name]["text"],
            "tensors": {stage: shape_dtype(t) for stage, t in saved.items()},
        }
        (gdir / f"hojo_{name}_meta.json").write_text(
            json.dumps(meta, indent=2, sort_keys=True) + "\n")
        print(f"[hojo-components] {name}: ids verified ({len(gen_ids)} tok), "
              f"{elapsed:.1f}s", flush=True)

    verify_saved(gdir)
    print("\nfixture  stage            shape                          dtype")
    print("-------  ---------------  -----------------------------  -------")
    for name, stage, shape, dtype in rows:
        print(f"{name:<7}  {stage:<15}  {shape:<29}  {dtype}")
    print(f"\n[hojo-components] verified {len(rows)} component files reload correctly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
