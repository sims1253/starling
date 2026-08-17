"""Capture staged, self-describing granite-speech-4.1-2b reference goldens.

Hooks the STOCK-numerics path (eager encoder + the model's own decoder layers)
to capture the intermediate tensors the ggml C++ port must reproduce, one stage
at a time, so divergence can be localized during porting:

  mel               torchaudio log-mel AFTER normalize, BEFORE the odd-frame
                    drop / pair-stack (1, T_mel, 80) f32
  mel_stacked       the processor's input_features (1, T', 160) f32 -- the
                    encoder's actual input
  encoder_hidden    GraniteSpeechCTCEncoder last_hidden_state (1, T', 1024) bf16
  audio_embeds      projector output (1, N, 2048) bf16
  prompt_ids        the chat-templated prompt ids with the <|audio|> expansion
  inputs_embeds     merged multimodal embeddings pre-multiplier (1, T, 2048)
  prefill_logits    first-step logits (lm_head / logits_scaling) (1, 1, 100353)

For medium/long the stages are captured on the FIRST 30 s chunk of the server
chunk policy (zero-padded to the full chunk length, exactly what the engine
sees); the end-to-end ids of that chunk are re-captured and asserted equal to
``golden/granite_reference.json`` so the staged tensors correspond to the exact
decode the ggml port must match.

Usage (from the repo root, GPU):
    uv run python scripts/granite_golden_components.py
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
    "mel_stacked",
    "encoder_hidden",
    "audio_embeds",
    "prompt_ids",
    "inputs_embeds",
    "prefill_logits",
)
AUDIO_TOKEN_ID = 100352
LOGITS_SCALING = 8.0
SAMPLE_RATE = 16000
CHUNK_SECONDS = 30.0


def load_wav(path: Path) -> np.ndarray:
    wav, sr = sf.read(str(path))
    if wav.ndim > 1:
        wav = wav[:, 0]
    assert sr == SAMPLE_RATE
    return wav.astype(np.float32)


def cpu_float(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to(device="cpu", dtype=torch.float32).contiguous()


def shape_dtype(t: torch.Tensor) -> dict[str, Any]:
    return {"shape": list(t.shape), "dtype": str(t.dtype).removeprefix("torch.")}


def pre_stack_mel(fe: Any, wav: torch.Tensor) -> torch.Tensor:
    """Replicate GraniteSpeechFeatureExtractor up to (excluding) the stack.

    mel = melspec(wav); logmel = log10(clip(mel, 1e-10));
    logmel = maximum(logmel, amax - 8) / 4 + 1.  Returns (1, T_mel, 80) f32.
    """
    with torch.no_grad():
        mel = fe.mel_filters.to(wav.device)(wav.float())
        logmel = mel.transpose(-1, -2).clip_(min=1e-10).log10_()
        mx = logmel.amax(dim=(-2, -1), keepdim=True)
        logmel = torch.maximum(logmel, mx - 8.0).div_(4).add_(1)
    return logmel


def capture_first_chunk(wav: np.ndarray, chunk_samples: int) -> np.ndarray:
    """The first server-policy chunk (zero-padded to the full chunk length)."""
    chunk = wav[:chunk_samples]
    if len(chunk) < chunk_samples:
        chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
    return chunk


def main() -> int:
    from starling.flags import OptFlags
    from starling.granite.audio import build_inputs
    from starling.granite.loader import load_model_and_processor
    from starling.granite.pipeline import MegaPipeline
    from starling.parakeet.gpu_lock import with_gpu_lock

    gdir = REPO / "golden"
    gdir.mkdir(exist_ok=True)
    rows: list[tuple[str, str, str, str]] = []
    chunk_samples = int(round(CHUNK_SECONDS * SAMPLE_RATE))

    with with_gpu_lock(
        session="ggml-goldens",
        model="granite-speech-4.1-2b",
        eta_min=20,
        note="capturing staged granite C++ reference goldens",
    ):
        print("[granite-components] loading model ...")
        model, processor = load_model_and_processor(attn_impl="eager")
        pipe = MegaPipeline(
            model,
            processor,
            encoder_mode="eager",
            use_fused_llm=False,
            flags=OptFlags(multistep_graph=False),
        )

        ref_path = gdir / "granite_reference.json"
        if not ref_path.is_file():
            print("[granite-components] run scripts/make_granite_golden.py first")
            return 1
        ref = json.loads(ref_path.read_text())

        captured: dict[str, torch.Tensor] = {}

        def hook_encoder(_m: torch.nn.Module, _i, out):
            captured["encoder_hidden"] = out.last_hidden_state

        def hook_projector(_m: torch.nn.Module, _i, out):
            captured["audio_embeds"] = out

        h1 = model.model.encoder.register_forward_hook(hook_encoder)
        h2 = model.model.projector.register_forward_hook(hook_projector)

        try:
            for name in NAMES:
                wav = load_wav(REPO / "tests" / "fixtures" / f"{name}.wav")
                seconds = len(wav) / SAMPLE_RATE
                first_chunk = capture_first_chunk(wav, chunk_samples)
                chunk_seconds = min(seconds, CHUNK_SECONDS)
                torch.cuda.synchronize()
                t0 = time.perf_counter()

                wav_t = torch.from_numpy(first_chunk).unsqueeze(0).contiguous().to("cuda")
                inputs = build_inputs(processor, wav_t)
                prompt_ids = inputs["input_ids"]
                mel160 = inputs["input_features"]
                mel80 = pre_stack_mel(processor.feature_extractor, wav_t)

                with torch.inference_mode():
                    # Encoder + projector fire through the hooks; the merged
                    # embeds are re-derived exactly like the model's
                    # get_merged_audio_embeddings (byte-exact per pipeline.py).
                    enc_hidden = captured["encoder_hidden"]
                    audio_embeds = captured["audio_embeds"]
                    mask = prompt_ids == AUDIO_TOKEN_ID
                    llm_ids = torch.where(mask, 0, prompt_ids)
                    embeds = model.model.language_model.get_input_embeddings()(llm_ids)
                    embeds = embeds.masked_scatter(
                        mask.unsqueeze(-1).expand_as(embeds),
                        audio_embeds.to(embeds.dtype),
                    )
                    # Prefill logits on the last position: lm_head / 8 (the
                    # GraniteForCausalLM epilogue the C++ port mirrors).
                    hidden = model.model(
                        inputs_embeds=embeds, use_cache=False
                    ).last_hidden_state[:, -1:, :]
                    prefill_logits = model.lm_head(hidden) / LOGITS_SCALING

                    # End-to-end ids for this chunk (stock greedy), budget
                    # mirroring the reference json.
                    chunk_ref = ref["fixtures"][name]["chunks"][0]
                    text, ids = pipe.transcribe(
                        mel160,
                        prompt_ids,
                        inputs.get("input_features_mask"),
                        max_new_tokens=chunk_ref["budget"],
                        speculative=False,
                    )
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0

                expected_ids = chunk_ref["ids"]
                assert ids[0].cpu().tolist() == expected_ids, (
                    f"{name}: captured decode diverged from granite_reference.json"
                )
                first_token = int(prefill_logits.argmax(dim=-1).item())
                assert first_token == expected_ids[0], (
                    f"{name}: prefill argmax {first_token} != first token {expected_ids[0]}"
                )

                saved = {
                    "mel": cpu_float(mel80),
                    "mel_stacked": cpu_float(mel160),
                    "encoder_hidden": cpu_float(enc_hidden),
                    "audio_embeds": cpu_float(audio_embeds),
                    "prompt_ids": prompt_ids.detach().to("cpu", torch.int64).contiguous(),
                    "inputs_embeds": cpu_float(embeds),
                    "prefill_logits": cpu_float(prefill_logits),
                }
                for stage, tensor in saved.items():
                    torch.save(tensor, gdir / f"granite_{name}_{stage}.pt")
                    rows.append(
                        (name, stage, str(tuple(tensor.shape)),
                         str(tensor.dtype).removeprefix("torch."))
                    )

                meta = {
                    "model": ref["model"],
                    "fixture": f"tests/fixtures/{name}.wav",
                    "seconds": seconds,
                    "chunk": {
                        "index": 0,
                        "seconds": chunk_seconds,
                        "padded": seconds > CHUNK_SECONDS,
                    },
                    "mel_frontend": {
                        "sample_rate": SAMPLE_RATE,
                        "mel_dim": 80,
                        "n_fft": 512,
                        "win_length": 400,
                        "hop_length": 160,
                    },
                    "n_prompt_tokens": int(prompt_ids.shape[1]),
                    "n_audio_tokens": int((prompt_ids[0] == AUDIO_TOKEN_ID).sum()),
                    "first_generated_token_id": first_token,
                    "reference_ids": expected_ids,
                    "reference_text": chunk_ref["text"],
                    "tensors": {stage: shape_dtype(t) for stage, t in saved.items()},
                }
                (gdir / f"granite_{name}_meta.json").write_text(
                    json.dumps(meta, indent=2, sort_keys=True) + "\n"
                )
                print(
                    f"[granite-components] {name}: ids verified "
                    f"({len(expected_ids)} tok), {elapsed:.1f}s"
                )
        finally:
            h1.remove()
            h2.remove()

    expected = [gdir / f"granite_{n}_{s}.pt" for n in NAMES for s in STAGES]
    expected += [gdir / f"granite_{n}_meta.json" for n in NAMES]
    missing = [str(p) for p in expected if not p.is_file()]
    assert not missing, f"missing goldens: {missing}"
    for p in expected:
        if p.suffix == ".pt":
            v = torch.load(p, map_location="cpu", weights_only=True)
            assert isinstance(v, torch.Tensor), f"{p} did not reload as a tensor"
        else:
            json.loads(p.read_text())

    print("\nfixture  stage            shape                      dtype")
    print("-------  ---------------  --------------------------  -------")
    for name, stage, shape, dtype in rows:
        print(f"{name:<7}  {stage:<15}  {shape:<26}  {dtype}")
    print(f"\n[granite-components] verified {len(rows)} component files reload correctly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
