#!/usr/bin/env python3
"""Convert the bosonai/higgs-audio-v3-stt safetensors checkpoint to a Starling GGUF.

Higgs-Audio-v3-STT is an audio-encoder + MLP-projector + Qwen3 decoder ASR model.
The audio path is a Whisper-large-v3 encoder (32 layers, d_model 1280, 20 heads,
head_dim 64) using **absolute positional embeddings** (NOT RoPE), followed by a
LayerNorm, an AvgPool1d(2) temporal downsample, and an MLP projector (depthwise
Conv1d temporal stride-2 + Linear 1280->2048 + ReLU + Linear 2048->2048) that
emits ~12.5 tokens/sec. The decoder is a Qwen3-1.7B trunk (28 layers, d2048,
16 query / 8 KV GQA, head_dim 128, SwiGLU intermediate 6144, RMSNorm eps 1e-6,
RoPE theta 1e6, **separate** lm_head — not tied despite the config flag — and
**with** q_norm/k_norm).

This converter mirrors scripts/convert_ark_gguf.py: an explicit, complete
tensor-name map (a checkpoint addition should fail conversion), with the model
config baked as `higgs.*` KV metadata, the Qwen3 BPE tokenizer, the Whisper mel
frontend constants, and the absolute positional embedding table. All weights are
stored BF16 (the checkpoint dtype).
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
import gguf

MODEL_ID = "bosonai/higgs-audio-v3-stt"
DEFAULT_SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub/models--bosonai--higgs-audio-v3-stt/snapshots"
)
DEFAULT_OUTPUT = Path("models/higgs-audio-v3-bf16-exact.gguf")


# ---------------------------------------------------------------------------
# Tensor-name mapping: HF checkpoint name -> Starling GGUF name.
#
# See plans/add-higgs-hojo-ggml.md for the authoritative contract; the C++ loader
# (cpp/higgs/loader.cpp) requires exactly these names.
def gguf_name(name: str) -> str | None:
    # ---- Whisper audio encoder front-end + stack (audio_tower.*) ----
    if name == "audio_tower.conv1.weight":
        return "enc.conv1.weight"
    if name == "audio_tower.conv1.bias":
        return "enc.conv1.bias"
    if name == "audio_tower.conv2.weight":
        return "enc.conv2.weight"
    if name == "audio_tower.conv2.bias":
        return "enc.conv2.bias"
    if name == "audio_tower.embed_positions.weight":
        return "enc.positional_emb.weight"  # ABSOLUTE positional, applied by index
    if name.startswith("audio_tower.layers."):
        p = name.removeprefix("audio_tower.layers.").split(".")
        i, rest = p[0], ".".join(p[1:])
        repl = {
            "self_attn_layer_norm": "attn_norm",
            "final_layer_norm": "ffn_norm",
            "self_attn.q_proj": "attn.q",
            "self_attn.k_proj": "attn.k",
            "self_attn.v_proj": "attn.v",
            "self_attn.out_proj": "attn.o",
            "fc1": "ffn.fc1",
            "fc2": "ffn.fc2",
        }
        for old, new in repl.items():
            if rest.startswith(old + "."):
                return f"enc.blk.{i}.{new}." + rest[len(old) + 1 :]
    if name.startswith("audio_tower.layer_norm."):
        return "enc.ln_post." + name.removeprefix("audio_tower.layer_norm.")

    # ---- MLP projector (audio_encoder_proj.*) ----
    if name.startswith("audio_encoder_proj."):
        rest = name.removeprefix("audio_encoder_proj.")
        return "proj." + rest  # temporal.{weight,bias}, linear1.{w,b}, linear2.{w,b}

    # ---- Qwen3 decoder (top-level, no model. prefix) ----
    if name == "embed_tokens.weight":
        return "llm.embed.weight"
    if name == "norm.weight":
        return "llm.final_norm.weight"
    if name == "audio_decoder_proj.text_lm_head.weight":
        return "llm.lm_head.weight"  # SEPARATE lm_head, not tied
    # Codec / audio-out path tensors — unused for STT.
    if name in (
        "audio_decoder_proj.audio_lm_head.weight",
        "audio_codebook_embeddings.weight",
    ):
        return None
    if name.startswith("layers."):
        p = name.removeprefix("layers.").split(".")
        i, rest = p[0], ".".join(p[1:])
        repl = {
            "input_layernorm": "attn_norm",
            "post_attention_layernorm": "ffn_norm",
            "self_attn.q_proj": "attn.q",
            "self_attn.k_proj": "attn.k",
            "self_attn.v_proj": "attn.v",
            "self_attn.o_proj": "attn.o",
            "self_attn.q_norm": "attn.q_norm",
            "self_attn.k_norm": "attn.k_norm",
            "mlp.gate_proj": "ffn.gate",
            "mlp.up_proj": "ffn.up",
            "mlp.down_proj": "ffn.down",
        }
        for old, new in repl.items():
            if rest.startswith(old + "."):
                return f"llm.blk.{i}.{new}." + rest[len(old) + 1 :]
    raise KeyError(f"no HIGGS GGUF mapping for {name!r}")


def add_metadata(w: gguf.GGUFWriter, prompt_prefix: list[int], prompt_suffix: list[int]) -> None:
    V = gguf.GGUFValueType
    w.add_key_value("starling.format_version", 1, V.UINT32)
    w.add_string("starling.numeric_profile", "bf16_exact")
    # general.architecture is set by GGUFWriter's arch-name arg ("higgs").

    def ints(**xs):
        for k, v in xs.items():
            w.add_key_value("higgs." + k, v, V.UINT32)

    def floats(**xs):
        for k, v in xs.items():
            w.add_key_value("higgs." + k, float(v), V.FLOAT32)

    def strings(**xs):
        for k, v in xs.items():
            w.add_string("higgs." + k, v)

    def bools(**xs):
        for k, v in xs.items():
            w.add_key_value("higgs." + k, bool(v), V.BOOL)

    # Frontend: Whisper feature extractor (n_fft=400, hop=160, 128 bins, 30s).
    ints(
        **{
            "frontend.sample_rate": 16000,
            "frontend.n_fft": 400,
            "frontend.win_length": 400,
            "frontend.hop_length": 160,
            "frontend.n_mels": 128,
            "frontend.power": 2,
            "frontend.nb_max_frames": 3000,
            "frontend.n_samples": 480000,
            "frontend.chunk_length": 30,
        }
    )
    floats(
        **{
            "frontend.mel_floor": 1e-10,
            # Whisper normalization: x = (log10(max(mel,1e-10)); clamp dynamic
            # range 8.0; then (x + 4.0) / 4.0). C++ mel uses (v + offset)/divisor.
            "frontend.normalization_offset": 4.0,
            "frontend.normalization_divisor": 4.0,
            "frontend.dynamic_range": 8.0,
        }
    )
    strings(
        **{
            "frontend.pad_mode": "reflect",
            "frontend.mel_scale": "slaney",
            "frontend.mel_norm": "slaney",
            "frontend.log": "log",
            "frontend.output_dtype": "bf16",
        }
    )

    # Whisper encoder (absolute positional embeddings, NOT RoPE).
    ints(
        **{
            "enc.num_mel_bins": 128,
            "enc.encoder_layers": 32,
            "enc.d_model": 1280,
            "enc.encoder_attention_heads": 20,
            "enc.head_dim": 64,
            "enc.encoder_ffn_dim": 5120,
            "enc.max_source_positions": 1500,
            "enc.conv_kernel": 3,
            "enc.avg_pool_kernel": 2,
        }
    )
    floats(**{"enc.layer_norm_eps": 1e-5})
    ints(**{"enc.use_rope": 0})
    bools(**{"enc.has_positional_embeddings": True})

    # MLP projector: depthwise temporal Conv1d(C,C,3,stride=2,pad=1,groups=C) +
    # Linear(1280->2048) + ReLU + Linear(2048->2048).
    ints(
        **{
            "proj.temporal_kernel": 3,
            "proj.temporal_stride": 2,
            "proj.temporal_groups": 1280,
            "proj.input_size": 1280,
            "proj.hidden_size": 2048,
            "proj.output_size": 2048,
        }
    )
    strings(**{"proj.act": "relu"})

    # LLM (Qwen3-1.7B decoder).
    ints(
        **{
            "llm.hidden_size": 2048,
            "llm.num_layers": 28,
            "llm.num_heads": 16,
            "llm.num_kv_heads": 8,
            "llm.head_dim": 128,
            "llm.intermediate_size": 6144,
            "llm.vocab_size": 151936,
            "llm.max_position_embeddings": 32768,
            "llm.max_cache_len": 4096,
        }
    )
    floats(**{"llm.rope_theta": 1_000_000.0, "llm.rms_norm_eps": 1e-6})
    strings(**{"llm.rope_scaling": "none"})
    bools(**{"llm.tied_embeddings": False, "llm.has_qk_norm": True})

    # Token ids (Qwen3 tokenizer, ChatML).
    ints(
        **{
            "audio_placeholder_id": 151672,  # <|AUDIO|>
            "audio_bos_id": 151669,          # <|audio_bos|>
            "audio_eos_id": 151670,          # <|audio_eos|>
            "im_start_id": 151644,           # <|im_start|>
            "im_end_id": 151645,             # <|im_end|>
            "pad_token_id": 151643,          # <|endoftext|>
            "eos_token_id": 151643,          # <|endoftext|> (dual-EOS: 151643 or 151645)
            "max_new_tokens": 200,
        }
    )
    strings(
        **{
            "default_instruction": "Transcribe the speech. Output only the "
            "spoken words in lowercase with no punctuation."
        }
    )

    # Prompt layout baked as token-id arrays so the C++ builder reproduces the
    # eager ChatML reference exactly. prefix = everything up to and including
    # <|audio_bos|>; then N x <|AUDIO|> (scattered with audio features); suffix =
    # <|audio_eos|> + "\n<|im_end|>\n<|im_start|>assistant\n".
    w.add_key_value("higgs.prompt_prefix", prompt_prefix, V.ARRAY, V.INT32)
    w.add_key_value("higgs.prompt_suffix", prompt_suffix, V.ARRAY, V.INT32)


def tokenizer(w: gguf.GGUFWriter, snapshot: Path) -> None:
    data = json.loads((snapshot / "tokenizer.json").read_text())
    vocab = data["model"]["vocab"]
    tokens = [None] * 151936
    for token, i in vocab.items():
        tokens[i] = token
    special = set()
    for item in data.get("added_tokens", []):
        tokens[item["id"]] = item["content"]
        if item.get("special", False):
            special.add(item["id"])
    for i, token in enumerate(tokens):
        if token is None:
            tokens[i] = "[PAD" + str(i) + "]"
    merges = data["model"]["merges"]
    merges = [" ".join(x) if isinstance(x, list) else x for x in merges]
    w.add_tokenizer_model("gpt2")
    w.add_token_list(tokens)
    w.add_token_scores([0.0] * len(tokens))
    w.add_token_types(
        [
            gguf.TokenType.CONTROL if i in special else gguf.TokenType.NORMAL
            for i in range(len(tokens))
        ]
    )
    w.add_token_merges(merges)
    w.add_bos_token_id(151643)
    w.add_eos_token_id(151645)
    w.add_pad_token_id(151643)


def frontend(snapshot: Path) -> tuple[np.ndarray, np.ndarray]:
    """Whisper mel filterbank + Hann window (n_fft=400, hop=160, 128 bins)."""
    from transformers.models.whisper.feature_extraction_whisper import (
        WhisperFeatureExtractor,
    )
    from transformers.audio_utils import window_function

    fx = WhisperFeatureExtractor(
        feature_size=128, sampling_rate=16000, hop_length=160, n_fft=400
    )
    mel = np.asarray(fx.mel_filters.T, dtype=np.float32)  # [128, 201]
    window = np.asarray(window_function(400, "hann"), dtype=np.float32)
    return mel, window


def build_prompt_arrays(snapshot: Path) -> tuple[list[int], list[int]]:
    """Pre-tokenize the ChatML prompt prefix/suffix via the Qwen3 tokenizer.

    Layout: ``<|im_start|>user\\n{instruction} <|audio_bos|>`` + (N x <|AUDIO|>)
    + ``<|audio_eos|>\\n<|im_end|>\\n<|im_start|>assistant\\n``.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(snapshot), trust_remote_code=True)
    instruction = (
        "Transcribe the speech. Output only the spoken words in lowercase "
        "with no punctuation."
    )
    prefix_str = f"<|im_start|>user\n{instruction} <|audio_bos|>"
    suffix_str = "<|audio_eos|>\n<|im_end|>\n<|im_start|>assistant\n"
    prefix = tok.encode(prefix_str, add_special_tokens=False)
    suffix = tok.encode(suffix_str, add_special_tokens=False)
    return prefix, suffix


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--snapshot",
        type=Path,
        default=None,
        help="Path to the HF snapshot dir (auto-detected if omitted).",
    )
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    snapshot = args.snapshot
    if snapshot is None:
        snaps = sorted(DEFAULT_SNAPSHOT.iterdir()) if DEFAULT_SNAPSHOT.exists() else []
        if not snaps:
            raise SystemExit(
                f"snapshot not found under {DEFAULT_SNAPSHOT}; pass --snapshot"
            )
        snapshot = snaps[-1]

    index = json.loads((snapshot / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    prompt_prefix, prompt_suffix = build_prompt_arrays(snapshot)
    print(f"prompt_prefix ({len(prompt_prefix)} tokens): {prompt_prefix}")
    print(f"prompt_suffix ({len(prompt_suffix)} tokens): {prompt_suffix}")

    w = gguf.GGUFWriter(args.output, "higgs", use_temp_file=True)
    add_metadata(w, prompt_prefix, prompt_suffix)
    tokenizer(w, snapshot)

    shards = {s: snapshot / s for s in sorted(set(weight_map.values()))}

    learned = 0
    dtypes: set[str] = set()
    skipped: list[str] = []
    for shard_name, shard_path in shards.items():
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for source in f.keys():
                target = gguf_name(source)
                if target is None:
                    skipped.append(source)
                    continue
                t = f.get_tensor(source)
                dtypes.add(str(t.dtype))
                if t.dtype is not torch.bfloat16:
                    raise TypeError(f"{source}: expected BF16, found {t.dtype}")
                a = np.ascontiguousarray(t.view(torch.uint16).numpy())
                w.add_tensor(
                    target, a, raw_shape=a.shape, raw_dtype=gguf.GGMLQuantizationType.BF16
                )
                learned += 1

    mel, window = frontend(snapshot)
    w.add_tensor("audio.mel_filters", np.ascontiguousarray(mel.T, dtype=np.float32))
    w.add_tensor("audio.mel_window", np.ascontiguousarray(window, dtype=np.float32))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(
        f"wrote {args.output}: {learned + 2} tensors ({learned} learned), "
        f"skipped {len(skipped)} (codec/unused), source dtypes={sorted(dtypes)}"
    )
    if skipped:
        print(f"skipped: {skipped}")


if __name__ == "__main__":
    main()
