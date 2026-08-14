#!/usr/bin/env python3
"""Convert the HojoAI/Hojo-ASR-V1 merged safetensors checkpoint to a Starling GGUF.

Hojo-ASR-V1 = Whisper-large-v3 mel -> Qwen3-Omni audio tower -> WeNet Conformer
bottleneck -> LayerNorm -> Qwen3-4B decoder (beam-4). 5.19B params, Apache-2.0.

Architecture (verified against hojo-asr-v1.tensors.json + the hojo_asr package):
  - Mel: Whisper-large-v3 extractor, 128 mel, 16kHz, hop 160, n_fft 400.
  - Qwen3-Omni audio tower (`speech_encoder.*`, F32): 3x Conv2d downsample
    (k3/s2/p1, GELU between each) -> flatten freq -> conv_out Linear -> add
    computed SinusoidsPositionEmbedding -> 32 pre-norm LayerNorm transformer
    layers (MHA 20 heads head_dim 64 with bias, bidirectional, GELU FFN) ->
    ln_post -> proj1 GELU proj2. Output [n_speech, 2048].
  - WeNet Conformer bottleneck (`bottleneck.*`, F32): LinearNoSubsampling
    (Linear 2048->2560 + LayerNorm) + RelPositionalEncoding -> 2 ConformerEncoder
    layers (macaron FFN, rel-pos MHA, conv module w/ BatchNorm1d + depthwise k15)
    -> after_norm. -> ln_speech LayerNorm.
  - Qwen3-4B decoder (`decoder_model.*`, BF16): 36 layers, hidden 2560, GQA 32/8,
    head_dim 128, intermediate 9728, vocab 151670, qk_norm, SEPARATE lm_head.

The forward path packs the mel into windows of n_window*2 = 3000 frames
(ceil(mel_T/3000); conv_chunksize batches the conv compute) for the conv2d tower
and runs the 32 tower transformer layers over the full packed sequence with a
block-diagonal attention mask built from cu_seqlens (bidirectional within each
window). The bottleneck then runs over the re-segmented per-window output.
Single-utterance inference (the parity case) has exactly one window's worth of
frames when feat_len <= n_window_infer.

This converter mirrors scripts/convert_higgs_gguf.py: an explicit, complete
tensor-name map (a checkpoint addition should fail conversion), with all model
config baked as `hojo.*` KV metadata, the Qwen3 BPE tokenizer (vocab 151670),
the Whisper mel frontend constants, and the RelPositionalEncoding buffer.

Mixed-dtype discipline: the tower + bottleneck tensors are F32 and are stored
as F32 (no precision loss); the decoder tensors are BF16 and stored as BF16 (no
bloat). The C++ loader handles mixed dtypes per-tensor.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
import gguf

MODEL_ID = "HojoAI/Hojo-ASR-V1"
DEFAULT_WEIGHTS = Path(".hf-cache/hojo-asr-v1/merged_full_model.safetensors")
DEFAULT_TOKENIZER = Path(".hf-cache/hojo-asr-v1/Qwen3-4B-Instruct-2507")
DEFAULT_WHISPER = Path(".hf-cache/hojo-asr-v1/whisper-large-v3")
DEFAULT_OUTPUT = Path("models/hojo-asr-v1.gguf")


# ---------------------------------------------------------------------------
# Tensor-name mapping: HF checkpoint name -> Starling GGUF name.
#
# The C++ loader (cpp/hojo/loader.cpp) requires exactly these names. Every
# checkpoint tensor must map (to a name) or be deliberately skipped (return
# None); an unmapped tensor raises KeyError so a checkpoint change fails loudly.
def gguf_name(name: str) -> str | None:
    # ---- Skip computed / non-persistent buffers (recomputed in C++). ----
    # rotary_emb.inv_freq is computed; the SinusoidsPositionEmbedding for the
    # tower is non-persistent and computed in C++.
    if name.endswith("rotary_emb.inv_freq"):
        return None

    # ================ speech_encoder (Qwen3-Omni audio tower), F32 ==========
    if name.startswith("speech_encoder."):
        rest = name.removeprefix("speech_encoder.")
        # Conv front-end + projections.
        if rest in ("conv2d1.weight", "conv2d1.bias",
                    "conv2d2.weight", "conv2d2.bias",
                    "conv2d3.weight", "conv2d3.bias"):
            return "audio." + rest
        if rest == "conv_out.weight":
            return "audio.conv_out.weight"
        if rest in ("ln_post.weight", "ln_post.bias"):
            return "audio.ln_post." + rest.split(".")[1]
        if rest in ("proj1.weight", "proj1.bias"):
            return "audio.proj1." + rest.split(".")[1]
        if rest in ("proj2.weight", "proj2.bias"):
            return "audio.proj2." + rest.split(".")[1]
        # 32 transformer layers.
        if rest.startswith("layers."):
            p = rest.removeprefix("layers.").split(".")
            i, sub, leaf = p[0], p[1], p[-1]
            if sub == "self_attn_layer_norm":
                return f"audio.blk.{i}.attn_norm.{leaf}"
            if sub == "final_layer_norm":
                return f"audio.blk.{i}.ffn_norm.{leaf}"
            if sub == "self_attn":
                # self_attn.{q,k,v,out}_proj.{weight,bias} -> leaf is weight/bias.
                proj = p[2]  # q_proj / k_proj / v_proj / out_proj
                repl = {"q_proj": "q", "k_proj": "k",
                        "v_proj": "v", "out_proj": "o"}
                if proj in repl:
                    return f"audio.blk.{i}.attn.{repl[proj]}.{leaf}"
            if sub in ("fc1", "fc2"):
                return f"audio.blk.{i}.ffn.{sub}.{leaf}"

    # ================ bottleneck (WeNet Conformer), F32 ====================
    if name.startswith("bottleneck."):
        rest = name.removeprefix("bottleneck.")
        # LinearNoSubsampling: embed.out.0 (Linear), embed.out.1 (LayerNorm).
        if rest.startswith("embed.out.0."):
            return "bottleneck.embed.out.0." + rest.split(".", 3)[-1]
        if rest.startswith("embed.out.1."):
            return "bottleneck.embed.out.1." + rest.split(".", 3)[-1]
        # RelPositionalEncoding buffer: embed.pos_enc.pe [1, 5000, 2560].
        if rest == "embed.pos_enc.pe":
            return "bottleneck.pos_enc.pe"
        # after_norm (LayerNorm).
        if rest.startswith("after_norm."):
            return "bottleneck.after_norm." + rest.split(".", 1)[1]
        # 2 ConformerEncoderLayer blocks: encoders.{i}.*
        if rest.startswith("encoders."):
            p = rest.removeprefix("encoders.").split(".")
            i = p[0]
            sub = p[1]
            tail = ".".join(p[2:])
            # Block norms: encoders.{i}.norm_mha.{weight,bias} (no further nesting).
            if sub in ("norm_mha", "norm_ff", "norm_ff_macaron",
                       "norm_conv", "norm_final"):
                return f"bottleneck.blk.{i}.{sub}.{tail}"
            # Self-attention (RelPositionMultiHeadedAttention):
            # encoders.{i}.self_attn.{linear_q,linear_k,...}.{weight,bias} ->
            # bottleneck.blk.{i}.mha.{proj}.{leaf} (drop proj from tail to
            # avoid a double prefix like mha.linear_q.linear_q.weight).
            if sub == "self_attn":
                proj = p[2]
                leaf = p[3] if len(p) > 3 else ""
                if proj == "linear_pos":
                    # linear_pos has NO bias.
                    return f"bottleneck.blk.{i}.mha.linear_pos.{leaf}"
                if proj in ("linear_q", "linear_k", "linear_v", "linear_out"):
                    return f"bottleneck.blk.{i}.mha.{proj}.{leaf}"
                if proj in ("pos_bias_u", "pos_bias_v"):
                    return f"bottleneck.blk.{i}.mha.{proj}"
            # Feed-forward modules: encoders.{i}.feed_forward.{w_1,w_2}.{weight,bias}.
            if sub == "feed_forward":
                return f"bottleneck.blk.{i}.ffn.{tail}"
            if sub == "feed_forward_macaron":
                return f"bottleneck.blk.{i}.ffn_macaron.{tail}"
            # Convolution module: encoders.{i}.conv_module.{pointwise_conv1,...}.
            #   p = [i, "conv_module", cmsub, weight-or-bias]; the leaf name is
            #   p[3], NOT tail (tail would re-include cmsub -> double prefix).
            if sub == "conv_module":
                cmsub = p[2]
                leaf = p[3] if len(p) > 3 else ""
                if cmsub in ("pointwise_conv1", "pointwise_conv2"):
                    return f"bottleneck.blk.{i}.conv.{cmsub}.{leaf}"
                if cmsub == "depthwise_conv":
                    return f"bottleneck.blk.{i}.conv.depthwise_conv.{leaf}"
                if cmsub == "norm":
                    # BatchNorm1d leaf in {weight,bias,running_mean,
                    # running_var,num_batches_tracked}.
                    return f"bottleneck.blk.{i}.conv.norm.{leaf}"

    # ================ ln_speech (LayerNorm over 2560), F32 =================
    if name.startswith("ln_speech."):
        return "ln_speech." + name.split(".", 1)[1]

    # ================ decoder_model (Qwen3-4B), BF16 ======================
    if name == "decoder_model.model.embed_tokens.weight":
        return "llm.embed.weight"
    if name == "decoder_model.lm_head.weight":
        return "llm.lm_head.weight"  # SEPARATE lm_head, not tied
    if name == "decoder_model.model.norm.weight":
        return "llm.final_norm.weight"
    if name.startswith("decoder_model.model.layers."):
        p = name.removeprefix("decoder_model.model.layers.").split(".")
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
                return f"llm.blk.{i}.{new}." + rest[len(old) + 1:]

    raise KeyError(f"no HOJO GGUF mapping for {name!r}")


def add_metadata(w: gguf.GGUFWriter) -> None:
    V = gguf.GGUFValueType
    w.add_key_value("starling.format_version", 1, V.UINT32)
    w.add_string("starling.numeric_profile", "mixed_f32_bf16_exact")

    def ints(**xs):
        for k, v in xs.items():
            w.add_key_value("hojo." + k, v, V.UINT32)

    def f64s(**xs):
        for k, v in xs.items():
            w.add_key_value("hojo." + k, float(v), V.FLOAT64)

    def strings(**xs):
        for k, v in xs.items():
            w.add_string("hojo." + k, v)

    def bools(**xs):
        for k, v in xs.items():
            w.add_key_value("hojo." + k, bool(v), V.BOOL)

    # ---- Frontend: Whisper feature extractor (n_fft=400, hop=160, 128 bins).
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
    f64s(
        **{
            "frontend.mel_floor": 1e-10,
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
        }
    )

    # ---- Qwen3-Omni audio tower. ----
    ints(
        **{
            "tower.num_mel_bins": 128,
            "tower.d_model": 1280,
            "tower.encoder_layers": 32,
            "tower.encoder_attention_heads": 20,
            "tower.head_dim": 64,
            "tower.encoder_ffn_dim": 5120,
            "tower.downsample_hidden_size": 480,
            "tower.output_dim": 2048,
            "tower.max_source_positions": 1500,
            "tower.conv_kernel": 3,
        }
    )
    f64s(**{"tower.layer_norm_eps": 1e-5})
    # Conv chunking / windowing parameters (hojo_asr_model.py + the omni config).
    ints(
        **{
            "tower.n_window": 1500,
            "tower.n_window_infer": 3000,
            "tower.conv_chunksize": 500,
        }
    )
    strings(**{"tower.activation_function": "gelu"})

    # ---- WeNet Conformer bottleneck. ----
    ints(
        **{
            "bottleneck.input_size": 2048,
            "bottleneck.output_size": 2560,
            "bottleneck.linear_units": 640,
            "bottleneck.num_blocks": 2,
            "bottleneck.attention_heads": 4,
            "bottleneck.cnn_module_kernel": 15,
            "bottleneck.max_len": 5000,
        }
    )
    f64s(**{"bottleneck.norm_eps": 1e-5})
    strings(
        **{
            "bottleneck.input_layer": "linear",
            "bottleneck.pos_enc_layer_type": "rel_pos",
            "bottleneck.selfattention_layer_type": "rel_selfattn",
            "bottleneck.activation_type": "swish",
            "bottleneck.cnn_module_norm": "batch_norm",
        }
    )
    bools(**{"bottleneck.macaron_style": True, "bottleneck.normalize_before": True})

    # ---- LLM (Qwen3-4B decoder). ----
    ints(
        **{
            "llm.hidden_size": 2560,
            "llm.num_layers": 36,
            "llm.num_heads": 32,
            "llm.num_kv_heads": 8,
            "llm.head_dim": 128,
            "llm.intermediate_size": 9728,
            "llm.vocab_size": 151670,
            "llm.max_position_embeddings": 262144,
            "llm.max_cache_len": 4096,
        }
    )
    f64s(**{"llm.rope_theta": 5_000_000.0, "llm.rms_norm_eps": 1e-6})
    strings(**{"llm.rope_scaling": "none"})
    bools(**{"llm.tied_embeddings": False, "llm.has_qk_norm": True})

    # ---- Token ids + decode (beam-4). ----
    ints(
        **{
            "bos_token_id": 151644,  # <|im_start|> (Qwen3 special token)
            "eos_token_id": 151645,  # <|im_end|>
            "pad_token_id": 151645,
            "max_new_tokens": 200,
        }
    )
    # Beam search parameters (hojo config.yaml generate).
    ints(
        **{
            "decode.num_beams": 4,
            "decode.min_length": 1,
        }
    )
    f64s(
        **{
            "decode.repetition_penalty": 2.0,
            "decode.length_penalty": 1.0,
            "decode.temperature": 1.0,
            "decode.top_p": 0.9,
        }
    )
    bools(**{"decode.do_sample": False})


def add_tokenizer(w: gguf.GGUFWriter, tokenizer_dir: Path) -> None:
    data = json.loads((tokenizer_dir / "tokenizer.json").read_text())
    vocab = data["model"]["vocab"]
    vocab_size = 151670
    tokens = [None] * vocab_size
    for token, i in vocab.items():
        if i < vocab_size:
            tokens[i] = token
    special = set()
    for item in data.get("added_tokens", []):
        if item["id"] < vocab_size:
            tokens[item["id"]] = item["content"]
            if item.get("special", False):
                special.add(item["id"])
    for i, token in enumerate(tokens):
        if token is None:
            tokens[i] = "[PAD" + str(i) + "]"
    merges = data["model"].get("merges", [])
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
    w.add_bos_token_id(151644)
    w.add_eos_token_id(151645)
    w.add_pad_token_id(151645)


def frontend(whisper_dir: Path) -> tuple[np.ndarray, np.ndarray]:
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help="Path to merged_full_model.safetensors.",
    )
    ap.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=DEFAULT_TOKENIZER,
        help="Path to the Qwen3-4B tokenizer dir.",
    )
    ap.add_argument(
        "--whisper-dir",
        type=Path,
        default=DEFAULT_WHISPER,
        help="Path to the whisper-large-v3 frontend config dir.",
    )
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    w = gguf.GGUFWriter(args.output, "hojo", use_temp_file=True)
    add_metadata(w)
    add_tokenizer(w, args.tokenizer_dir)

    learned = 0
    dtypes: set[str] = set()
    skipped: list[str] = []
    f32_count = 0
    bf16_count = 0
    with safe_open(str(args.weights), framework="pt", device="cpu") as f:
        for source in f.keys():
            target = gguf_name(source)
            if target is None:
                skipped.append(source)
                continue
            t = f.get_tensor(source)
            dtypes.add(str(t.dtype))
            if t.dtype == torch.float32:
                a = np.ascontiguousarray(t.numpy())
                w.add_tensor(target, a, raw_dtype=gguf.GGMLQuantizationType.F32)
                f32_count += 1
            elif t.dtype == torch.bfloat16:
                # numpy has no native bf16; view as uint16 bytes (same bit pattern).
                a = np.ascontiguousarray(t.view(torch.uint16).numpy())
                w.add_tensor(
                    target, a, raw_shape=tuple(t.shape),
                    raw_dtype=gguf.GGMLQuantizationType.BF16,
                )
                bf16_count += 1
            elif t.dtype == torch.int64:
                # num_batches_tracked I64 scalar.
                a = np.ascontiguousarray(t.numpy())
                w.add_tensor(target, a, raw_dtype=gguf.GGMLQuantizationType.I64)
            else:
                raise TypeError(f"{source}: unhandled dtype {t.dtype}")
            learned += 1

    mel, window = frontend(args.whisper_dir)
    w.add_tensor("audio.mel_filters", np.ascontiguousarray(mel.T, dtype=np.float32))
    w.add_tensor("audio.mel_window", np.ascontiguousarray(window, dtype=np.float32))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(
        f"wrote {args.output}: {learned + 2} tensors ({learned} learned, "
        f"{f32_count} F32 + {bf16_count} BF16 + "
        f"{learned - f32_count - bf16_count} I64), skipped {len(skipped)}, "
        f"source dtypes={sorted(dtypes)}"
    )
    if skipped:
        print(f"skipped: {skipped}")


if __name__ == "__main__":
    main()
