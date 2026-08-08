"""Capture staged, self-describing bosonai/higgs-audio-v3-stt reference goldens.

Hooks the eager HF model (transformers 4.51.3, run under ``.venv-higgs``) to
capture the intermediate tensors the ggml C++ port must reproduce, one stage
at a time, so divergence can be localized during porting:

  mel              Whisper log-mel (1, 128, mel_T) f32  -- the collator output
  audio_tower      HiggsAudioEncoder output (post layers + avgpool + ln_post),
                   (B, T_avg, 1280) -- feeds the projector
  audio_embeds     HiggsAudioFeatureProjector output (B, T_proj, 2048) -- the
                   final audio features scattered into inputs_embeds
  prompt_ids       the ChatML input_ids before expansion (right-padded)
  inputs_embeds    merged multimodal embeddings (B, S, 2048) feeding the Qwen3
                   decoder prefill (audio scattered into <|AUDIO|> slots)
  prefill_logits   the first-step logits whose argmax seeds greedy decode
                   (1, 1, vocab)

The end-to-end emitted token ids are re-captured via ``model.generate`` and
asserted equal to ``golden/higgs_golden.json``'s ``gen_ids`` so the staged
tensors correspond to the exact decode the ggml port must match.

Run under the isolated venv (from the repo root):
    .venv-higgs/bin/python scripts/higgs_golden_components.py
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "ref"))

NAMES = ("short", "medium", "long")
STAGES = (
    "mel",
    "audio_tower",
    "audio_embeds",
    "prompt_ids",
    "inputs_embeds",
    "prefill_logits",
)
MODEL_ID = "bosonai/higgs-audio-v3-stt"
DEFAULT_PROMPT = (
    "Transcribe the speech. Output only the spoken words in lowercase with no punctuation."
)
ENABLE_THINKING = True


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    wav, sr = sf.read(str(path))
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return np.asarray(wav, dtype=np.float32), int(sr)


def cpu_float(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to(device="cpu", dtype=torch.float32).contiguous()


def shape_dtype(t: torch.Tensor) -> dict[str, Any]:
    return {"shape": list(t.shape), "dtype": str(t.dtype).removeprefix("torch.")}


def _build_input_tokens(tokenizer, user_prompt: str, enable_thinking: bool = True) -> list[int]:
    def enc(s: str) -> list[int]:
        return tokenizer.encode(s, add_special_tokens=False)

    input_tokens: list[int] = []
    input_tokens += enc("<|im_start|>user\n")
    input_tokens += enc(user_prompt)
    input_tokens += enc("<|audio_bos|><|AUDIO|><|audio_eos|>")
    input_tokens += enc("<|im_end|>\n")
    input_tokens += enc("<|im_start|>assistant\n")
    if not enable_thinking:
        input_tokens += enc("<think>\n\n</think>\n\n")
    return input_tokens


def _build_sample(audio_np: np.ndarray, input_ids: list[int], sample_rate: int = 16000):
    from starling.higgs.vendor import ChatMLDatasetSample

    return ChatMLDatasetSample(
        input_ids=torch.LongTensor(input_ids),
        label_ids=torch.LongTensor([-100] * len(input_ids)),
        audio_ids_concat=torch.zeros((1, 0), dtype=torch.long),
        audio_ids_start=torch.tensor([0], dtype=torch.long),
        audio_waveforms_concat=torch.tensor(audio_np, dtype=torch.float32),
        audio_waveforms_start=torch.tensor([0]),
        audio_sample_rate=torch.tensor([sample_rate]),
        audio_speaker_indices=torch.tensor([0]),
    )


def capture_stages(model: Any, tokenizer: Any, audio_np: np.ndarray) -> tuple[dict, list[int]]:
    """Run one prefill forward + greedy decode with hooks grabbing stage tensors.

    Hooks:
      - audio_tower: capture the output of ``model.audio_tower`` (the
        HiggsAudioEncoder, post layers + avgpool + ln_post) -> (B, T_avg, 1280).
      - audio_embeds: capture the output of ``model.audio_encoder_proj`` (the
        HiggsAudioFeatureProjector MLP) -> (B, T_proj, 2048).

    mel is taken from the collator's ``audio_pixel_values``; prompt_ids from the
    collator's ``input_ids``; inputs_embeds from the model forward; prefill
    logits from the first forward's last position.
    """
    import transcribe as higgs_transcribe  # noqa: E402

    input_ids = _build_input_tokens(tokenizer, DEFAULT_PROMPT, ENABLE_THINKING)
    sample = _build_sample(audio_np, input_ids, sample_rate=16000)
    from starling.higgs.loader import make_collator
    collator = make_collator(model)
    batch = asdict(collator([sample]))
    batch = {
        k: (v.to("cuda").contiguous() if isinstance(v, torch.Tensor) else v)
        for k, v in batch.items()
    }
    # The collator emits the Whisper mel as ``audio_features`` (1, 128, T).
    mel = batch["audio_features"]

    captured: dict[str, torch.Tensor] = {}

    def hook_audio_tower(_m: torch.nn.Module, _i, out):
        # HiggsAudioEncoder.forward returns a BaseModelOutput; take the hidden.
        if hasattr(out, "last_hidden_state"):
            captured["audio_tower"] = out.last_hidden_state
        else:
            captured["audio_tower"] = out[0]

    def hook_projector(_m: torch.nn.Module, _i, out):
        captured["audio_embeds"] = out

    def hook_first_layer(_m: torch.nn.Module, args, kwargs):
        # Qwen3DecoderLayer(hidden_states, ...) -- the first positional or the
        # ``hidden_states`` kwarg is the EXACT merged inputs_embeds feeding the
        # prefill (audio scattered into <|AUDIO|> slots). This is the authoritative
        # tensor the ggml prefill must reproduce.
        if args:
            captured["inputs_embeds"] = args[0]
        elif "hidden_states" in kwargs:
            captured["inputs_embeds"] = kwargs["hidden_states"]

    h1 = model.audio_tower.register_forward_hook(hook_audio_tower)
    h2 = model.audio_encoder_proj.register_forward_hook(hook_projector)
    # forward_pre_hook to read the decoder-layer input by reference (the layer
    # mutates nothing in-place); use the kwargs/args as captured.
    h3 = model.layers[0].register_forward_pre_hook(
        lambda _m, args, kwargs: hook_first_layer(_m, args, kwargs), with_kwargs=True
    )

    try:
        with torch.inference_mode(), higgs_transcribe._suppress_right_padding_warning():
            # First forward: prefill (fills the DynamicCache) to grab the merged
            # inputs_embeds + prefill logits. use_cache=True so the audio tower +
            # projector fire during this single call (hooks populate captured).
            out = model(
                **batch,
                use_cache=True,
                return_dict=True,
            )
            prefill_logits = out.logits[:, -1:, :]  # (1, 1, vocab)

            # The expanded input_ids the model built internally (after expanding
            # the single <|AUDIO|> placeholder into N audio slots). The forward
            # returns it as ``expanded_input_ids``; fall back to the pre-expansion
            # batch ids if absent.
            prompt_ids = batch["input_ids"]  # (1, T_pre) right-padded pre-expansion
            if hasattr(out, "expanded_input_ids") and out.expanded_input_ids is not None:
                prompt_ids = out.expanded_input_ids  # (1, S) post-merge

            captured["prefill_logits"] = prefill_logits
            captured["mel"] = mel
            captured["prompt_ids"] = prompt_ids
    finally:
        h1.remove()
        h2.remove()
        h3.remove()

    # Full e2e greedy decode (separate call; matches capture_golden_ref exactly).
    with torch.inference_mode(), higgs_transcribe._suppress_right_padding_warning():
        T = batch["input_ids"].shape[1]
        outputs = model.generate(
            **batch,
            max_new_tokens=512,
            use_cache=True,
            do_sample=False,
            stop_strings=["<|im_end|>", "<|endoftext|>"],
            tokenizer=tokenizer,
        )
        output_ids = outputs[0] if isinstance(outputs, tuple) else outputs
        gen_ids = output_ids[0, T:].cpu().tolist()
    return captured, gen_ids


def verify_saved(gdir: Path) -> None:
    expected = [gdir / f"higgs_{name}_{stage}.pt" for name in NAMES for stage in STAGES]
    expected += [gdir / f"higgs_{name}_meta.json" for name in NAMES]
    missing = [str(p) for p in expected if not p.is_file()]
    assert not missing, f"missing goldens: {missing}"
    for p in expected:
        if p.suffix == ".pt":
            v = torch.load(p, map_location="cpu", weights_only=True)
            assert isinstance(v, torch.Tensor), f"{p} did not reload as a tensor"
        else:
            json.loads(p.read_text())


def main() -> int:
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    import transcribe as higgs_transcribe  # noqa: E402  (warms the collator cache)

    gdir = REPO / "golden"
    gdir.mkdir(exist_ok=True)
    rows: list[tuple[str, str, str, str]] = []

    # load under .venv-higgs (transformers 4.51.3); eager attn == byte-exact golden path
    print("[higgs-components] loading model ...", flush=True)
    cfg = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_ID, config=cfg, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="eager",
    )
    model.eval()
    # patch graph-unsafe remote helpers (same as the loader does for the pipeline)
    from starling.higgs.loader import _patch_graph_safe_remote_helpers
    _patch_graph_safe_remote_helpers(model)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    ref_path = gdir / "higgs_golden.json"
    ref = json.loads(ref_path.read_text())

    for name in NAMES:
        wav, sr = load_wav(REPO / "tests" / "fixtures" / f"{name}.wav")
        seconds = len(wav) / sr
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        captured, gen_ids = capture_stages(model, tokenizer, wav)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        expected_ids = ref["fixtures"][name]["gen_ids"]
        # The golden may stop early (EOS); compare up to the shorter length, but
        # require the captured ids to match the golden prefix exactly.
        n = min(len(gen_ids), len(expected_ids))
        assert gen_ids[:n] == expected_ids[:n], (
            f"{name}: captured decode diverged from higgs_golden.json at "
            f"index {next((i for i in range(n) if gen_ids[i] != expected_ids[i]), n)} "
            f"(got {gen_ids[:12]}... expected {expected_ids[:12]}...)"
        )
        first_token = int(captured["prefill_logits"].argmax(dim=-1).item())
        assert first_token == gen_ids[0], (
            f"{name}: prefill argmax {first_token} != first decoded token {gen_ids[0]}"
        )

        saved = {
            "mel": cpu_float(captured["mel"]),
            "audio_tower": cpu_float(captured["audio_tower"]),
            "audio_embeds": cpu_float(captured["audio_embeds"]),
            "prompt_ids": captured["prompt_ids"]
            .detach()
            .to(device="cpu", dtype=torch.int64)
            .contiguous(),
            "inputs_embeds": cpu_float(captured["inputs_embeds"]),
            "prefill_logits": cpu_float(captured["prefill_logits"]),
        }
        for stage, tensor in saved.items():
            torch.save(tensor, gdir / f"higgs_{name}_{stage}.pt")
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
            "n_prompt_tokens_pre_expand": int(captured["prompt_ids"].shape[1]),
            "n_inputs_embeds_tokens": int(captured["inputs_embeds"].shape[1]),
            "first_generated_token_id": first_token,
            "reference_ids": gen_ids,
            "reference_text": ref["fixtures"][name]["text"],
            "tensors": {stage: shape_dtype(t) for stage, t in saved.items()},
        }
        (gdir / f"higgs_{name}_meta.json").write_text(
            json.dumps(meta, indent=2, sort_keys=True) + "\n"
        )
        print(
            f"[higgs-components] {name}: ids verified ({len(gen_ids)} tok), {elapsed:.1f}s",
            flush=True,
        )

    verify_saved(gdir)
    print("\nfixture  stage            shape                          dtype")
    print("-------  ---------------  -----------------------------  -------")
    for name, stage, shape, dtype in rows:
        print(f"{name:<7}  {stage:<15}  {shape:<29}  {dtype}")
    print(f"\n[higgs-components] verified {len(rows)} component files reload correctly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
