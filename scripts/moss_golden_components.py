"""Capture staged, self-describing MOSS-Transcribe reference goldens.

This deliberately uses ``starling.moss.reference`` for the audio-encoder and
embedding-merge stages.  The prefill forward below is the prefill portion of
``greedy_generate`` verbatim, so its logits are the distribution whose argmax
seeds that reference decoder.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import soundfile as sf
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

NAMES = ("short", "medium", "long")
STAGES = ("mel", "encoder_hidden", "audio_embeds", "prompt_ids", "inputs_embeds", "prefill_logits")


def load_wav(path: Path) -> tuple[torch.Tensor, int]:
    import numpy as np

    wav, sr = sf.read(str(path))
    if sr != 16000:
        import librosa

        wav = librosa.resample(wav.astype("float32"), orig_sr=sr, target_sr=16000)
        sr = 16000
    if wav.ndim > 1:
        wav = wav.mean(1)
    return torch.from_numpy(wav.astype("float32")), sr


def last_prefill_logits(model: Any, inputs_embeds: torch.Tensor, *, max_cache_len: int = 2048) -> torch.Tensor:
    """Return the exact prefill distribution used by reference.greedy_generate."""
    from transformers.cache_utils import DynamicCache

    lm = model.model.language_model
    T = inputs_embeds.shape[1]
    cache = DynamicCache(config=lm.config)
    device = inputs_embeds.device
    pos = torch.arange(T, device=device).unsqueeze(0)
    cp = torch.arange(T, device=device)
    # attention_mask=None: model-built causal mask, bit-identical to the
    # explicit 0/-inf mask on the eager golden path (see reference.py).
    out = lm(
        inputs_embeds=inputs_embeds,
        attention_mask=None,
        position_ids=pos,
        past_key_values=cache,
        use_cache=True,
        cache_position=cp,
    )
    return model.lm_head(out.last_hidden_state[:, -1:, :])


def cpu_float(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()


def shape_dtype(tensor: torch.Tensor) -> dict[str, Any]:
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype).removeprefix("torch.")}


def verify_saved(gdir: Path) -> None:
    expected = [gdir / f"moss_{name}_{stage}.pt" for name in NAMES for stage in STAGES]
    expected += [gdir / f"moss_{name}_meta.json" for name in NAMES]
    assert len(expected) == 21, f"expected 21 component files, got {len(expected)}"
    missing = [str(path) for path in expected if not path.is_file()]
    assert not missing, f"missing goldens: {missing}"
    for path in expected:
        if path.suffix == ".pt":
            value = torch.load(path, map_location="cpu", weights_only=True)
            assert isinstance(value, torch.Tensor), f"{path} did not reload as a tensor"
        else:
            json.loads(path.read_text())


def main() -> int:
    from starling.moss.loader import load_model_and_processor
    from starling.moss.reference import audio_features, build_inputs_embeds, greedy_generate
    from starling.parakeet.gpu_lock import with_gpu_lock

    gdir = REPO / "golden"
    gdir.mkdir(exist_ok=True)
    rows: list[tuple[str, str, str, str]] = []

    with with_gpu_lock(session="ggml-goldens", model="MOSS-Transcribe-preview-2B", eta_min=15,
                       note="capturing staged MOSS C++ reference goldens"):
        print("[moss-components] loading model ...")
        model, proc = load_model_and_processor()
        for name in NAMES:
            wav, sr = load_wav(REPO / "tests" / "fixtures" / f"{name}.wav")
            seconds = wav.shape[0] / sr
            raw = proc(wav.numpy())
            # Preserve the processor's tensor (bf16) for the reference forward.
            inp = {key: (value.cuda() if isinstance(value, torch.Tensor) else value) for key, value in raw.items()}

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                encoder_hidden = audio_features(model, inp["audio_data"], inp["audio_data_seqlens"])
                inputs_embeds = build_inputs_embeds(model, inp["input_ids"], encoder_hidden, inp["audio_input_mask"])
                # This is exactly the adapter output consumed by masked_scatter_ above.
                audio_embeds = model.model.audio_adapter(encoder_hidden)
                prefill_logits = last_prefill_logits(model, inputs_embeds)
                ids = greedy_generate(model, inputs_embeds, max_new_tokens=200, max_cache_len=2048)
            torch.cuda.synchronize()

            expected_ids = torch.load(gdir / f"moss_{name}_ids.pt", map_location="cpu", weights_only=True)
            assert torch.equal(ids.cpu(), expected_ids), (
                f"{name}: component capture diverged from moss_{name}_ids.pt "
                f"(got {ids.tolist()}, expected {expected_ids.tolist()})"
            )
            first_token = int(prefill_logits.argmax(dim=-1).item())
            assert first_token == int(ids[0, 0]), f"{name}: prefill argmax does not seed greedy decode"

            # The processor feeds bf16 to the bf16 model.  Persist mel as fp32 as
            # the C++ frontend gate, retaining those exact bf16-quantized values.
            saved = {
                "mel": cpu_float(inp["audio_data"]),
                "encoder_hidden": cpu_float(encoder_hidden),
                "audio_embeds": cpu_float(audio_embeds),
                "prompt_ids": inp["input_ids"].detach().to(device="cpu", dtype=torch.int64).contiguous(),
                "inputs_embeds": cpu_float(inputs_embeds),
                "prefill_logits": cpu_float(prefill_logits),
            }
            for stage, tensor in saved.items():
                torch.save(tensor, gdir / f"moss_{name}_{stage}.pt")
                rows.append((name, stage, str(tuple(tensor.shape)), str(tensor.dtype).removeprefix("torch.")))

            meta = {
                "model": "MOSS-Transcribe-preview-2B",
                "fixture": f"tests/fixtures/{name}.wav",
                "seconds": seconds,
                "mel_frontend": {"sample_rate": 16000, "mel_dim": 128, "n_fft": 640, "hop_length": 160},
                "audio_data_seqlens": inp["audio_data_seqlens"].detach().cpu().tolist(),
                "audio_input_mask_sum": int(inp["audio_input_mask"].sum().item()),
                "n_audio_tokens": int(inp["audio_input_mask"].sum().item()),
                "first_generated_token_id": first_token,
                "reference_ids_file": f"moss_{name}_ids.pt",
                "tensors": {stage: shape_dtype(tensor) for stage, tensor in saved.items()},
            }
            (gdir / f"moss_{name}_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
            elapsed = time.perf_counter() - t0
            print(f"[moss-components] {name}: ids verified, {elapsed:.1f}s")

    verify_saved(gdir)
    print("\nfixture  stage            shape                 dtype")
    print("-------  ---------------  --------------------  -------")
    for name, stage, shape, dtype in rows:
        print(f"{name:<7}  {stage:<15}  {shape:<20}  {dtype}")
    print("\n[moss-components] verified 21 component files reload correctly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
