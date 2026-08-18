#!/usr/bin/env python3
"""Convert the pinned Qwen3-ASR-1.7B-hf safetensors checkpoint to a Starling GGUF.

Qwen/Qwen3-ASR-1.7B-hf is a windowed-attention conv encoder + 2-layer MLP
projector + Qwen3 decoder ASR model. The encoder chunks the 128-bin log-mel
into 100-frame (2*n_window) chunks, runs three GELU 3x3/stride-2 conv2d layers
(480 channels) + a bias-free Linear(7680 -> 1024) per chunk, adds a (13, 1024)
sinusoidal position table, packs the valid post-CNN rows and runs 24
windowed-attention layers (full attention within 8-chunk windows of 104 packed
rows) + LayerNorm. The projector is Linear(1024 -> 1024) + GELU +
Linear(1024 -> 2048). The decoder is a bias-free Qwen3 trunk WITH per-head
q_norm/k_norm and a TIED lm_head, stock numerics (no Granite multipliers).

This converter mirrors scripts/convert_granite_gguf.py: an explicit, complete
tensor-name map (a checkpoint addition should fail conversion), the config
baked as `qwen3.*` KV metadata, the byte-level BPE tokenizer, and the mel
frontend constants. All weights are stored BF16 (the checkpoint dtype). The
sinusoidal position table is non-persistent in the checkpoint, so it is
PRECOMPUTED here (float64 math per SinusoidsPositionEmbedding, rounded
straight to BF16 — the dtype the reference buffer holds after torch_dtype
loading) and stored as enc.pos_embed. The chat-template prompt layout is baked
as prefix/suffix token-id arrays around the N audio placeholder copies
(tokenization is invariant to N — every boundary token is special).
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
import gguf

REVISION = "bcd2b5b7f32b480ab5790554cfa8347f246a14f3"
DEFAULT_SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub/models--Qwen--Qwen3-ASR-1.7B-hf/snapshots"
    / REVISION
)

N_LAYERS = 24
VOCAB_SIZE = 151936


# ---------------------------------------------------------------------------
# Tensor-name mapping: HF checkpoint name -> Starling GGUF name.
#
# Encoder (Qwen3ASREncoder):
#   enc.conv1 / conv2 / conv3       Conv2d k3 s2 p1 + GELU stack (480 ch)
#                                  (weights keep the HF [OC,IC,KH,KW] byte
#                                  order, which is ggml's [KW,KH,IC,OC])
#   enc.out                         bias-free Linear(7680 -> 1024)
#   enc.pos_embed                   PRECOMPUTED (13, 1024) sinusoids (bf16)
#   enc.blk.N.attn_norm             pre-attention LayerNorm (biased)
#   enc.blk.N.attn_{q,k,v,o}        windowed MHA projections (all biased)
#   enc.blk.N.ffn_norm              pre-FFN LayerNorm (biased)
#   enc.blk.N.ff_up / ff_down       fc1/fc2 (GELU between, both biased)
#   enc.ln_post                     final LayerNorm (biased)
#
# Projector (Qwen3ASRMultiModalProjector):
#   proj.linear_1                   Linear(1024 -> 1024, bias) + GELU
#   proj.linear_2                   Linear(1024 -> 2048, bias)
#
# LLM decoder (Qwen3 trunk: bias-free projections, q/k norm, TIED lm_head):
#   llm.blk.N.{attn_norm,attn.q/k/v/o,attn.q_norm,attn.k_norm,ffn_norm,
#              ffn.gate/up/down}
#   llm.embed / llm.final_norm      (lm_head == llm.embed; the checkpoint
#                                    carries no separate lm_head)
def gguf_name(name: str) -> str:
    # ---- encoder frontend + layers ----
    if name == "model.audio_tower.conv2d1.weight": return "enc.conv1.weight"
    if name == "model.audio_tower.conv2d1.bias":   return "enc.conv1.bias"
    if name == "model.audio_tower.conv2d2.weight": return "enc.conv2.weight"
    if name == "model.audio_tower.conv2d2.bias":   return "enc.conv2.bias"
    if name == "model.audio_tower.conv2d3.weight": return "enc.conv3.weight"
    if name == "model.audio_tower.conv2d3.bias":   return "enc.conv3.bias"
    if name == "model.audio_tower.conv_out.weight": return "enc.out.weight"
    if name == "model.audio_tower.ln_post.weight": return "enc.ln_post.weight"
    if name == "model.audio_tower.ln_post.bias":   return "enc.ln_post.bias"
    if name.startswith("model.audio_tower.layers."):
        p = name.removeprefix("model.audio_tower.layers.").split(".")
        i, rest = p[0], ".".join(p[1:])
        repl = {
            "self_attn_layer_norm": "attn_norm",
            "self_attn.q_proj": "attn_q",
            "self_attn.k_proj": "attn_k",
            "self_attn.v_proj": "attn_v",
            "self_attn.out_proj": "attn_o",
            "final_layer_norm": "ffn_norm",
            "fc1": "ff_up",
            "fc2": "ff_down",
        }
        for old, new in repl.items():
            if rest.startswith(old + "."):
                return f"enc.blk.{i}.{new}." + rest[len(old) + 1:]
    # ---- projector ----
    if name.startswith("model.multi_modal_projector."):
        return "proj." + name.removeprefix("model.multi_modal_projector.")
    # ---- LLM decoder (Qwen3 trunk) ----
    if name == "model.language_model.embed_tokens.weight": return "llm.embed.weight"
    if name == "model.language_model.norm.weight":         return "llm.final_norm.weight"
    if name.startswith("model.language_model.layers."):
        p = name.removeprefix("model.language_model.layers.").split(".")
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
    raise KeyError(f"no qwen3 GGUF mapping for {name!r}")


def positional_embedding() -> torch.Tensor:
    """(13, 1024) SinusoidsPositionEmbedding table, float64 math -> BF16.

    Mirrors SinusoidsPositionEmbedding.compute_default_singular_positional_embedding
    exactly: the reference computes sin/cos in float64 (np.log is float64 and
    propagates) and the buffer is rounded to the model dtype (bf16) once at
    load, so baking the direct float64->BF16 rounding is bit-exact.
    """
    length, channels, max_timescale = 13, 1024, 10000
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = np.exp(-log_timescale_increment * np.arange(channels // 2))
    scaled_time = np.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    pos = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
    return torch.from_numpy(pos).to(torch.bfloat16)  # (13, 1024)


def add_metadata(w: gguf.GGUFWriter) -> None:
    V = gguf.GGUFValueType
    w.add_key_value("starling.format_version", 1, V.UINT32)
    w.add_string("starling.numeric_profile", "bf16_exact")
    w.add_string("general.architecture", "qwen3")

    def ints(**xs):
        for k, v in xs.items():
            w.add_key_value("qwen3." + k, v, V.UINT32)

    def floats(**xs):
        for k, v in xs.items():
            w.add_key_value("qwen3." + k, float(v), V.FLOAT32)

    def strings(**xs):
        for k, v in xs.items():
            w.add_string("qwen3." + k, v)

    # Frontend: Qwen3ASRFeatureExtractor (torch.stft n_fft=400, hop=160,
    # periodic hann, center/reflect; power=2; slaney mel scale + norm;
    # log10). Normalization: x = log10(clamp(power, 1e-10)); global amax over
    # ALL frames; x = max(x, mx - 8); x = (x + 4) / 4. The trailing STFT frame
    # is dropped (T = S/H, the moss/ark rule) and the mel axis is then
    # right-padded with ZEROS to a multiple of 2*n_window = 100 frames
    # (engine-side, after the shared mel). Clips shorter than min_length
    # samples are zero-padded first (engine-side).
    ints(
        **{
            "frontend.sample_rate": 16000,
            "frontend.n_fft": 400,
            "frontend.win_length": 400,
            "frontend.hop_length": 160,
            "frontend.n_mels": 128,
            "frontend.power": 2,
            "frontend.chunk_length": 30,
            "frontend.min_length": 8000,
            "frontend.n_window": 50,
        }
    )
    floats(
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
            "frontend.log": "log10",
            "frontend.output_dtype": "bf16",
        }
    )

    # Windowed-attention conv encoder.
    ints(
        **{
            "enc.n_mel": 128,
            "enc.hidden": 1024,
            "enc.layers": 24,
            "enc.heads": 16,
            "enc.head_dim": 64,
            "enc.ffn_dim": 4096,
            "enc.downsample_hidden": 480,
            "enc.n_window": 50,
            "enc.n_window_infer": 800,
            "enc.max_pos_emb": 13,
            "enc.output_dim": 2048,
        }
    )
    w.add_key_value("qwen3.enc.layer_norm_eps", 1e-5, V.FLOAT32)

    # Projector (2-layer MLP).
    ints(**{"proj.hidden": 1024, "proj.output_dim": 2048})

    # LLM (Qwen3 trunk, stock numerics: all multipliers 1, default scale).
    ints(
        **{
            "llm.hidden_size": 2048,
            "llm.num_layers": 28,
            "llm.num_heads": 16,
            "llm.num_kv_heads": 8,
            "llm.head_dim": 128,
            "llm.intermediate_size": 6144,
            "llm.vocab_size": 151936,
            "llm.max_position_embeddings": 65536,
            "llm.max_cache_len": 4096,
        }
    )
    floats(**{"llm.rope_theta": 1000000.0, "llm.rms_norm_eps": 1e-6})
    w.add_key_value("qwen3.llm.tied_embeddings", True, V.BOOL)
    w.add_key_value("qwen3.llm.has_qk_norm", True, V.BOOL)

    # Token ids + generation + the serve chunk policy (mirrored by the
    # engine's decode entry so the C++ path matches the Python server
    # byte-for-byte). eos is the serving path's <|im_end|>.
    ints(
        **{
            "audio_token_id": 151676,
            "pad_token_id": 151645,
            "eos_token_id": 151645,
            "max_new_tokens": 200,
        }
    )
    floats(**{"chunk_seconds": 30.0})


# Chat-template prompt layout, empirically captured from the HF processor
# under the reference environment (transformers git pin 957e6032 with the
# qwen3_asr restore; the repo's main .venv). The rendered prompt is
# "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|>" +
# [<|audio_pad|> x N] + "<|audio_end|><|im_end|>\n<|im_start|>assistant\n"
# (empty system text, no language prefill — auto-detect). Every boundary
# token is special, so prefix + [audio]*N + suffix is exact for every N
# (verified for N=97 under the reference tokenizer; the audio-token count
# follows the post-CNN length formula baked into the engine).
PROMPT_PREFIX = [151644, 8948, 198, 151645, 198, 151644, 872, 198, 151669]
PROMPT_SUFFIX = [151670, 151645, 198, 151644, 77091, 198]


def prompt_layout(w: gguf.GGUFWriter) -> None:
    V = gguf.GGUFValueType
    w.add_key_value("qwen3.prompt_prefix", PROMPT_PREFIX, V.ARRAY, V.INT32)
    w.add_key_value("qwen3.prompt_suffix", PROMPT_SUFFIX, V.ARRAY, V.INT32)
    print(f"prompt: prefix={PROMPT_PREFIX} suffix={PROMPT_SUFFIX}")


def tokenizer(w: gguf.GGUFWriter, snapshot: Path) -> None:
    data = json.loads((snapshot / "tokenizer.json").read_text())
    vocab = data["model"]["vocab"]
    size = VOCAB_SIZE
    tokens = [None] * size
    for token, i in vocab.items():
        tokens[i] = token
    special = set()
    for item in data.get("added_tokens", []):
        if item["id"] < size:
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
    w.add_eos_token_id(151645)
    w.add_pad_token_id(151645)


def frontend() -> tuple[np.ndarray, np.ndarray]:
    """Mel filterbank (201 x 128, slaney scale+norm) + periodic Hann window.

    Uses precisely transformers.audio_utils.mel_filter_bank with the
    Qwen3ASRFeatureExtractor arguments, stored freq-major [n_fft/2+1, n_mels]
    -- the layout the shared C++ mel frontend expects.
    """
    from transformers.audio_utils import mel_filter_bank

    mel = mel_filter_bank(
        num_frequency_bins=1 + 400 // 2,
        num_mel_filters=128,
        min_frequency=0.0,
        max_frequency=8000.0,
        sampling_rate=16000,
        norm="slaney",
        mel_scale="slaney",
    )
    mel = np.ascontiguousarray(mel, dtype=np.float32)
    assert mel.shape == (201, 128), mel.shape
    window = torch.hann_window(400)  # periodic, matching torch.stft's window
    return mel, window.numpy().astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("models/qwen3-asr-1.7b-bf16-exact.gguf"),
    )
    args = ap.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    w = gguf.GGUFWriter(args.output, "qwen3", use_temp_file=True)
    add_metadata(w)
    tokenizer(w, args.snapshot)
    prompt_layout(w)

    learned = 0
    dtypes = set()
    skipped = []
    with safe_open(args.snapshot / "model.safetensors", framework="pt", device="cpu") as f:
        for source in sorted(f.keys()):
            target = gguf_name(source)
            if target is None:
                skipped.append(source)
                continue
            t = f.get_tensor(source)
            dtypes.add(str(t.dtype))
            if t.dtype is not torch.bfloat16:
                raise TypeError(f"{source}: expected BF16, found {t.dtype}")
            # GGUF stores dims innermost-first; keep the BF16 byte stream in
            # checkpoint row-major order (the HF [OC,IC,KH,KW] conv layout IS
            # ggml's [KW,KH,IC,OC]).
            shape = tuple(t.shape)
            a = np.ascontiguousarray(t.view(torch.uint16).numpy()).reshape(shape)
            w.add_tensor(
                target, a, raw_shape=shape, raw_dtype=gguf.GGMLQuantizationType.BF16
            )
            learned += 1

    # lm_head is tied to llm.embed.weight (tie_word_embeddings=true) and absent
    # from the checkpoint; the decode stack reuses the embedding table.

    pos = positional_embedding()
    a = np.ascontiguousarray(pos.view(torch.uint16).numpy())
    w.add_tensor(
        "enc.pos_embed", a, raw_shape=(13, 1024),
        raw_dtype=gguf.GGMLQuantizationType.BF16,
    )

    mel, window = frontend()
    w.add_tensor("audio.mel_filters", np.ascontiguousarray(mel, dtype=np.float32))
    w.add_tensor("audio.mel_window", np.ascontiguousarray(window, dtype=np.float32))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(
        f"wrote {args.output}: {learned + 3} tensors ({learned} learned + "
        f"pos_embed/mel/window), skipped {len(skipped)}, "
        f"source dtypes={sorted(dtypes)}"
    )
    if skipped:
        print(f"skipped: {skipped}")


if __name__ == "__main__":
    main()
