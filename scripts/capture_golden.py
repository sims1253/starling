"""Capture byte-exact golden transcripts for higgs-audio-v3-stt.

Runs the stock ``model.generate()`` (eager, the byte-exact reference) on the
test fixtures and persists the generated token ids + decoded text to
``golden/higgs_golden.json``. These are the correctness oracle every megakernel
path must reproduce exactly.

This is the reference "stock transformers" path -- no CUDA graphs, no fused
kernels, just ``HiggsAudio3Model.generate(do_sample=False)``.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer  # imported early to avoid lazy-import races

REPO = Path(__file__).resolve().parents[1]
# The .wav fixtures are gitignored (not checked into worktrees); they live in the
# main repo checkout. Resolve them from either location.
MAIN_REPO = Path("/home/m0hawk/Documents/starling")
FIXTURES_DIR = MAIN_REPO / "tests" / "fixtures" if (MAIN_REPO / "tests" / "fixtures" / "short.wav").exists() else REPO / "tests" / "fixtures"
sys.path.insert(0, str(REPO / "src"))

from starling.higgs.vendor import HiggsAudioSampleCollator, ChatMLDatasetSample  # noqa: E402

MODEL_ID = "bosonai/higgs-audio-v3-stt"
DEFAULT_PROMPT = (
    "Transcribe the speech. Output only the spoken words in lowercase with no punctuation."
)
FIXTURES = [
    ("short", FIXTURES_DIR / "short.wav"),
    ("medium", FIXTURES_DIR / "medium.wav"),
    ("long", FIXTURES_DIR / "long.wav"),
]


def _build_input_tokens(tokenizer, user_prompt, enable_thinking=True):
    def enc(s):
        return tokenizer.encode(s, add_special_tokens=False)

    input_tokens = []
    input_tokens += enc("<|im_start|>user\n")
    input_tokens += enc(user_prompt)
    input_tokens += enc("<|audio_bos|><|AUDIO|><|audio_eos|>")
    input_tokens += enc("<|im_end|>\n")
    input_tokens += enc("<|im_start|>assistant\n")
    if not enable_thinking:
        input_tokens += enc("<think>\n\n</think>\n\n")
    return input_tokens


def _build_sample(audio_np, input_ids, sample_rate=16000):
    return ChatMLDatasetSample(
        input_ids=torch.LongTensor(input_ids),
        label_ids=torch.LongTensor([-100] * len(input_ids)),
        audio_ids_concat=torch.zeros((1, 0), dtype=torch.long),  # ASR-only: no discrete codes
        audio_ids_start=torch.tensor([0], dtype=torch.long),
        audio_waveforms_concat=torch.tensor(audio_np, dtype=torch.float32),
        audio_waveforms_start=torch.tensor([0]),
        audio_sample_rate=torch.tensor([sample_rate]),
        audio_speaker_indices=torch.tensor([0]),
    )


def _create_collator(config):
    from transformers import WhisperProcessor

    whisper_proc = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    return HiggsAudioSampleCollator(
        whisper_processor=whisper_proc,
        audio_in_token_id=config.audio_in_token_idx,
        audio_out_token_id=config.audio_out_token_idx,
        audio_stream_bos_id=config.audio_stream_bos_id,
        audio_stream_eos_id=config.audio_stream_eos_id,
        encode_whisper_embed=config.encode_whisper_embed,
        pad_token_id=config.pad_token_id,
        return_audio_in_tokens=config.encode_audio_in_tokens,
        use_delay_pattern=config.use_delay_pattern,
        round_to=1,
        audio_num_codebooks=config.audio_num_codebooks,
        chunk_size_seconds=getattr(config, "chunk_size_seconds", 30),
        pad_left=False,
    )


def _parse_output(full_text):
    import re

    parts = full_text.split("assistant\n")
    hyp = parts[-1] if len(parts) > 1 else full_text
    hyp = re.sub(r"<think>.*?</think>", "", hyp, flags=re.DOTALL)
    if "<think>" in hyp:
        hyp = hyp[hyp.index("<think>") + len("<think>"):]
    hyp = re.sub(r"<\|.*?\|>", "", hyp)
    return hyp.strip()


def _patch_generation_kwargs() -> None:
    """Shim the removed ``GenerationConfig.generation_kwargs`` for tf 5.x.

    The model's custom ``generate()`` stashes params (``audio_out_bos_token_id``,
    ``ras_win_len``, ...) into ``generation_config.generation_kwargs``, an
    attribute that existed in transformers 4.46 but was removed in 5.x. For ASR
    (text-only) all those params are None/unused, so restoring an empty dict is
    behaviour-preserving -- it just gives their code somewhere to write.
    """
    from transformers import GenerationConfig

    if not hasattr(GenerationConfig, "generation_kwargs"):
        GenerationConfig.generation_kwargs = {}


def main() -> int:
    from dataclasses import asdict

    _patch_generation_kwargs()
    torch.manual_seed(0)
    # Load via the vendored (patched) modeling classes -- not trust_remote_code --
    # so the transformers-5.x patches in starling/higgs/vendor/modeling apply.
    from starling.higgs.vendor.modeling import HiggsAudio3Config, HiggsAudio3Model

    cfg = HiggsAudio3Config.from_pretrained(MODEL_ID)
    print("loading model (vendored, patched)...", flush=True)
    t0 = time.time()
    model = HiggsAudio3Model.from_pretrained(
        MODEL_ID, config=cfg,
        torch_dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="eager",
    )
    model.eval()
    print(f"loaded in {time.time()-t0:.1f}s, VRAM {torch.cuda.memory_allocated()//1024//1024} MB", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    collator = _create_collator(cfg)
    import soundfile as sf

    out = {"model": MODEL_ID, "prompt": DEFAULT_PROMPT, "fixtures": {}}
    for name, path in FIXTURES:
        audio_np, sr = sf.read(str(path))
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=1)
        audio_np = np.asarray(audio_np, dtype=np.float32)
        dur = len(audio_np) / sr
        input_ids = _build_input_tokens(tok, DEFAULT_PROMPT, enable_thinking=True)
        sample = _build_sample(audio_np, input_ids, sample_rate=16000)
        batch = asdict(collator([sample]))
        batch = {k: (v.to("cuda").contiguous() if isinstance(v, torch.Tensor) else v)
                 for k, v in batch.items()}
        t1 = time.time()
        with torch.inference_mode():
            # Manual greedy decode (their generate()/_sample() signatures drifted vs
            # transformers 5.x). ASR greedy uses no logits processors, so this eager
            # loop is the byte-exact reference: prefill -> argmax loop -> EOS.
            eos_ids = {151643, 151645}  # <|endoftext|>, <|im_end|>
            from transformers import DynamicCache
            cache = DynamicCache()
            out = model.forward(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                audio_features=batch["audio_features"],
                audio_feature_attention_mask=batch["audio_feature_attention_mask"],
                past_key_values=cache,
                use_cache=True,
            )
            logits = out.logits[:, -1, :].float()
            next_tok = logits.argmax(dim=-1)  # (B,)
            gen = [int(next_tok.item())]
            cur_ids = batch["input_ids"]
            attn = batch["attention_mask"]
            for _ in range(511):
                if int(next_tok.item()) in eos_ids:
                    break
                cur_ids = torch.cat([cur_ids, next_tok.unsqueeze(-1)], dim=1)
                attn = torch.cat([attn, torch.ones((attn.shape[0], 1), dtype=attn.dtype, device=attn.device)], dim=1)
                out = model.forward(
                    input_ids=next_tok.unsqueeze(-1),
                    attention_mask=attn,
                    past_key_values=cache,
                    use_cache=True,
                )
                logits = out.logits[:, -1, :].float()
                next_tok = logits.argmax(dim=-1)
                gen.append(int(next_tok.item()))
        elapsed = time.time() - t1
        gen_ids = gen
        full_ids = torch.tensor([batch["input_ids"][0].cpu().tolist() + gen_ids], dtype=torch.long)
        full_text = tok.decode(full_ids[0], skip_special_tokens=False)
        text = _parse_output(full_text)
        print(f"[{name}] {dur:.1f}s -> {len(gen_ids)} toks in {elapsed:.2f}s :: {text!r}", flush=True)
        out["fixtures"][name] = {
            "path": str(path), "duration_s": dur, "sample_rate": sr,
            "gen_ids": gen_ids, "text": text, "wall_s": elapsed,
        }

    golden_path = REPO / "golden" / "higgs_golden.json"
    golden_path.parent.mkdir(parents=True, exist_ok=True)
    golden_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {golden_path}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
