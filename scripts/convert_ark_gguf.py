#!/usr/bin/env python3
"""Convert the pinned ARK-ASR-3B safetensors checkpoint to a Starling GGUF.

ARK-ASR-3B (AutoArk-AI/ARK-ASR-3B) is an audio-encoder + MLP-adapter + Qwen2.5
decoder ASR model. The audio path is a Whisper encoder (32 layers, d_model 1280,
20 heads, head_dim 64) that uses **RoPE** attention (use_rope=True, rope_dim=32,
base=10000) -- NOT absolute positional embeddings -- followed by a LayerNorm and
an MLP adapter that merges every 4 frames into one Qwen2.5 decoder token. The
decoder is a stock Qwen2.5 trunk (36 layers, d2048, 16 query / 2 KV GQA, head_dim
128, SwiGLU intermediate 11008, RMSNorm eps 1e-6, RoPE theta 1e6, tied
embeddings) -- notably WITHOUT the q_norm/k_norm of the Qwen3 family.

This converter mirrors scripts/convert_moss_gguf.py: an explicit, complete
tensor-name map (a checkpoint addition should fail conversion), with the model
config baked as `ark.*` KV metadata, the Qwen2.5 BPE tokenizer, and the Whisper
mel frontend constants (n_fft=400, hop=160, 128 bins). All weights are stored
BF16 (the checkpoint dtype). Audio-token-count + prompt-layout constants are
baked so the C++ side can reproduce the eager reference's prompt exactly.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
import gguf

REVISION = "1e28271b79edc97635783bea65abc89195a09ed3"
DEFAULT_SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub/models--AutoArk-AI--ARK-ASR-3B/snapshots"
    / REVISION
)


# ---------------------------------------------------------------------------
# Tensor-name mapping: HF checkpoint name -> Starling GGUF name.
#
# Audio encoder (Whisper + the ARK post-LayerNorm + MLP adapter):
#   enc.conv1/conv2            Whisper Conv1d front-end (kernel 3; conv2 stride 2)
#   enc.blk.N.{attn_norm,attn.q/k/v/o,ffn_norm,ffn.fc1/fc2}
#                             one Whisper encoder layer (q/v/o have bias; k has NO bias)
#   enc.ln_post                the ARK-added LayerNorm after the Whisper stack
#                             (the Whisper encoder's own layer_norm is replaced
#                             with nn.Identity, so it carries no weight).
#   adapter.fc0/fc2           the adapting MLP: Linear(5120->4096) GELU Linear(4096->2048)
#
# LLM decoder (Qwen2.5 trunk, no q_norm/k_norm):
#   llm.blk.N.{attn_norm,attn.q/k/v/o,ffn_norm,ffn.gate/up/down}
#   llm.embed / llm.final_norm   (lm_head is tied to embed_tokens; not duplicated)
#
# RoPE tables for the encoder attention are baked below as `enc.rope_cos` /
# `enc.rope_sin` because the HF path builds them with a specific (dim=32,
# base=10000) formulation that the C++ port must match byte-for-byte.
def gguf_name(name: str) -> str:
    # ---- audio encoder front-end + stack ----
    if name == "audio_encoder.whisper.conv1.weight": return "enc.conv1.weight"
    if name == "audio_encoder.whisper.conv1.bias":   return "enc.conv1.bias"
    if name == "audio_encoder.whisper.conv2.weight": return "enc.conv2.weight"
    if name == "audio_encoder.whisper.conv2.bias":   return "enc.conv2.bias"
    # embed_positions is unused (use_rope=True); skip deliberately.
    if name.startswith("audio_encoder.whisper.embed_positions"):
        return None
    if name.startswith("audio_encoder.whisper.layers."):
        p = name.removeprefix("audio_encoder.whisper.layers.").split(".")
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
                return f"enc.blk.{i}.{new}." + rest[len(old) + 1:]
    if name.startswith("audio_encoder.layer_norm."):
        return "enc.ln_post." + name.removeprefix("audio_encoder.layer_norm.")
    # ---- MLP adapter (adapting Sequential: [0]=Linear, [1]=GELU, [2]=Linear) ----
    if name == "audio_encoder.adapting.0.weight": return "adapter.fc0.weight"
    if name == "audio_encoder.adapting.0.bias":   return "adapter.fc0.bias"
    if name == "audio_encoder.adapting.2.weight": return "adapter.fc2.weight"
    if name == "audio_encoder.adapting.2.bias":   return "adapter.fc2.bias"
    # ---- LLM decoder (Qwen2.5) ----
    if name == "model.embed_tokens.weight": return "llm.embed.weight"
    if name == "model.norm.weight":         return "llm.final_norm.weight"
    # lm_head.weight is tied to embed_tokens; skip (do not store a duplicate).
    if name == "lm_head.weight": return None
    if name.startswith("model.layers."):
        p = name.removeprefix("model.layers.").split(".")
        i, rest = p[0], ".".join(p[1:])
        repl = {
            "input_layernorm": "attn_norm",
            "post_attention_layernorm": "ffn_norm",
            "self_attn.q_proj": "attn.q",
            "self_attn.k_proj": "attn.k",
            "self_attn.v_proj": "attn.v",
            "self_attn.o_proj": "attn.o",
            "mlp.gate_proj": "ffn.gate",
            "mlp.up_proj": "ffn.up",
            "mlp.down_proj": "ffn.down",
        }
        for old, new in repl.items():
            if rest.startswith(old + "."):
                return f"llm.blk.{i}.{new}." + rest[len(old) + 1:]
    raise KeyError(f"no ARK GGUF mapping for {name!r}")


def add_metadata(w: gguf.GGUFWriter, numeric_profile: str = "bf16_exact") -> None:
    V = gguf.GGUFValueType
    w.add_key_value("starling.format_version", 1, V.UINT32)
    w.add_string("starling.numeric_profile", numeric_profile)
    w.add_string("general.architecture", "ark")

    def ints(**xs):
        for k, v in xs.items():
            w.add_key_value("ark." + k, v, V.UINT32)

    def floats(**xs):
        for k, v in xs.items():
            w.add_key_value("ark." + k, float(v), V.FLOAT32)

    def strings(**xs):
        for k, v in xs.items():
            w.add_string("ark." + k, v)

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
            # Whisper normalization: x = log10(max(mel, 1e-10)); x = max(x,
            # x.max()-8.0); then (x + 4.0) / 4.0. The C++ mel frontend uses the
            # formula (v + offset) / divisor (see moss/mel.cpp), so offset=4.0,
            # divisor=4.0 reproduces Whisper's (x+4)/4.
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

    # Whisper encoder.
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
        }
    )
    w.add_key_value("ark.enc.layer_norm_eps", 1e-5, V.FLOAT32)
    floats(**{"enc.rope_base": 10000.0, "enc.rope_dim": 32.0})
    ints(**{"enc.use_rope": 1, "enc.merge_factor": 4})

    # Adapter: Linear(5120 -> 4096) GELU Linear(4096 -> 2048).
    ints(
        **{
            "adapter.input_size": 5120,
            "adapter.hidden_size": 4096,
            "adapter.output_size": 2048,
            "adapter.merge_factor": 4,
        }
    )
    strings(**{"adapter.act": "gelu"})

    # LLM (Qwen2.5 decoder).
    ints(
        **{
            "llm.hidden_size": 2048,
            "llm.num_layers": 36,
            "llm.num_heads": 16,
            "llm.num_kv_heads": 2,
            "llm.head_dim": 128,
            "llm.intermediate_size": 11008,
            "llm.vocab_size": 151936,
            "llm.max_position_embeddings": 32768,
            "llm.max_cache_len": 4096,
        }
    )
    floats(**{"llm.rope_theta": 1000000.0, "llm.rms_norm_eps": 1e-6})
    strings(**{"llm.rope_scaling": "none"})
    w.add_key_value("ark.llm.tied_embeddings", True, V.BOOL)
    w.add_key_value("ark.llm.has_qk_norm", False, V.BOOL)

    # Token ids.
    ints(
        **{
            "audio_token_id": 151663,
            "begin_audio_id": 151666,
            "end_audio_id": 151667,
            "user_id": 151665,
            "assistant_id": 151668,
            "pad_token_id": 151643,
            "bos_token_id": 151643,
            "eos_token_id": 151645,
            "max_new_tokens": 200,
        }
    )
    strings(**{"default_instruction": "Transcribe the audio to text."})

    # Prompt layout baked as token-id arrays so the C++ builder reproduces the
    # eager reference exactly (empirically captured from the HF processor).
    # prefix = <|user|><|begin_of_audio|>
    w.add_key_value("ark.prompt_prefix", [151665, 151666], V.ARRAY, V.INT32)
    # suffix = <|end_of_audio|> + instruction_tokens + <|assistant|>
    instruction_tokens = [3167, 3114, 279, 7699, 311, 1467, 13]
    w.add_key_value(
        "ark.prompt_suffix", [151667] + instruction_tokens + [151668], V.ARRAY, V.INT32
    )


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
    # Any still-undefined slots (Qwen reserves a tail) get explicit placeholders.
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
    """Whisper mel filterbank + Hann window (n_fft=400, hop=160, 128 bins).

    Uses precisely the Transformers Whisper implementation the ARK feature
    extractor wraps, so the filterbank matches the eager reference exactly.
    """
    from transformers.models.whisper.feature_extraction_whisper import (
        WhisperFeatureExtractor,
    )
    from transformers.audio_utils import window_function

    fx = WhisperFeatureExtractor(
        feature_size=128, sampling_rate=16000, hop_length=160, n_fft=400
    )
    mel = np.asarray(fx.mel_filters.T, dtype=np.float32)  # [B=201, M=128] -> [128,201]
    window = np.asarray(window_function(400, "hann"), dtype=np.float32)
    return mel, window


def rope_tables() -> tuple[np.ndarray, np.ndarray]:
    """Encoder RoPE cos/sin tables matching modeling_audio.RotaryEmbedding.

    The HF path builds inv_freq = 1/(base**(2i/dim)) for i in [0,dim/2) with
    dim=32 (head_dim//2 = 64//2) and base=10000, freqs = outer(arange(seq_len),
    inv_freq), then emb = stack([cos(freqs), sin(freqs)], -1). We precompute
    cos/sin for the max encoder length (3000 frames post-conv = downsampled mel
    ceiling) and store them as [seq_len, dim/2] f32. The C++ attention applies
    them with the HF rotate-half-interleaved formulation.
    """
    dim = 32
    base = 10000.0
    max_len = 3000  # Whisper nb_max_frames; conv2 stride-2 halves mel_T -> 1500 max
    inv_freq = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float32)[: dim // 2] / dim))
    t = np.arange(max_len, dtype=np.float32)
    freqs = np.outer(t, inv_freq)  # [max_len, dim/2]
    return np.cos(freqs).astype(np.float32), np.sin(freqs).astype(np.float32)


# --------------------------------------------------------------------------- #
# Weight-quantization target selection.
#
# The ARK decode (Qwen2.5 autoregressive loop) is memory-bandwidth-bound on the
# decoder linears; quantizing them to q8_0 engages ggml's MMQ/MVQ dequant-GEMM
# kernels and roughly halves the per-token weight traffic. The audio encoder is
# attention-bound (not weight-bound) and is left at bf16 to preserve the
# audio-conditioning path. Norms and biases are tiny and stay bf16/f32. The
# tied lm_head embedding (`llm.embed.weight`) is the single biggest tensor and
# a quantization candidate (it dominates the embedding-lookup / logits matmul).
#
# See docs/ggml-ark-perf.md "Remaining optimization headroom" for the rationale
# and the projected latency win.
# --------------------------------------------------------------------------- #
_QUANT_LINEARS = {
    "attn.q", "attn.k", "attn.v", "attn.o",
    "ffn.gate", "ffn.up", "ffn.down",
}


def is_quant_target(name: str) -> bool:
    """True iff this GGUF tensor name should be quantized for the fast path.

    Matches the decoder linears: ``llm.blk.{N}.{attn.q|attn.k|attn.v|attn.o|
    ffn.gate|ffn.up|ffn.down}.weight`` (7 per layer) plus the tied lm_head
    embedding ``llm.embed.weight``. The dotted linear names (``attn.q``) mean a
    naive split would miscount, so match by suffix.
    """
    if name == "llm.embed.weight":
        return True
    if not name.startswith("llm.blk.") or not name.endswith(".weight"):
        return False
    return any(name.endswith(f"{lin}.weight") for lin in _QUANT_LINEARS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("models/ark-asr-3b-bf16-exact.gguf"),
    )
    ap.add_argument(
        "--quant",
        choices=["bf16", "q8_0"],
        default="bf16",
        help="Weight quantization for the decoder linears (q8_0 engages ggml's "
             "MMQ dequant-GEMM; the encoder/norms/biases stay bf16 either way).",
    )
    args = ap.parse_args()

    quantize = args.quant == "q8_0"
    numeric_profile = "q8_0" if quantize else "bf16_exact"

    index = json.loads((args.snapshot / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    w = gguf.GGUFWriter(args.output, "ark", use_temp_file=True)
    add_metadata(w, numeric_profile=numeric_profile)
    tokenizer(w, args.snapshot)

    # Resolve every shard file once.
    shards = {s: args.snapshot / s for s in sorted(set(weight_map.values()))}

    learned = 0
    quantized = 0
    dtypes = set()
    skipped = []
    # Stream tensors shard-by-shard so each file opens once.
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

                if quantize and is_quant_target(target):
                    # Quantize this decoder linear to q8_0. Cast bf16 -> f32 (value
                    # upcast, NOT a byte view), then quantize. gguf.quantize blocks
                    # along the innermost (C-contiguous) axis, which is the ggml
                    # ne[0] axis -- exactly what ggml_mul_mat expects for a
                    # quantized src0.
                    f32 = np.ascontiguousarray(t.to(torch.float32).numpy())
                    q = gguf.quantize(f32, gguf.GGMLQuantizationType.Q8_0)
                    w.add_tensor(
                        target, q, raw_shape=q.shape, raw_dtype=gguf.GGMLQuantizationType.Q8_0
                    )
                    quantized += 1
                else:
                    # GGUF stores dims innermost-first; keep the BF16 byte stream in
                    # checkpoint row-major order (same convention as moss).
                    a = np.ascontiguousarray(t.view(torch.uint16).numpy())
                    w.add_tensor(
                        target, a, raw_shape=a.shape, raw_dtype=gguf.GGMLQuantizationType.BF16
                    )
                learned += 1

    mel, window = frontend(args.snapshot)
    w.add_tensor("audio.mel_filters", np.ascontiguousarray(mel.T, dtype=np.float32))
    w.add_tensor("audio.mel_window", np.ascontiguousarray(window, dtype=np.float32))
    cos, sin = rope_tables()
    # NOTE: store cos/sin as plain f32 numpy ([seq_len, dim/2] -> ggml reads
    # innermost-first as [dim/2, seq_len]). This is the storage that produces
    # correct short-fixture parity. Medium/long currently hit a ggml view_4d
    # bounds issue in apply_enc_rope (T_enc large) — see docs/ggml-ark-port-status.md.
    w.add_tensor("enc.rope_cos", np.ascontiguousarray(cos, dtype=np.float32))
    w.add_tensor("enc.rope_sin", np.ascontiguousarray(sin, dtype=np.float32))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(
        f"wrote {args.output}: {learned + 4} tensors ({learned} learned"
        + (f", {quantized} q8_0" if quantize else "")
        + f"), skipped {len(skipped)} (tied/unused), source dtypes={sorted(dtypes)}"
    )
    if skipped:
        print(f"skipped: {skipped}")


if __name__ == "__main__":
    main()
