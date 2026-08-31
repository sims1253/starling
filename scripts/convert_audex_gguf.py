#!/usr/bin/env python3
"""Convert the pinned Nemotron-Labs-Audex-2B safetensors checkpoint to a Starling GGUF.

nvidia/Nemotron-Labs-Audex-2B (checkpoint_folder_full) is a Qwen2AudioEncoder
(whisper-large-v3 shaped: 128-bin mel, 30 s clips = 3000 frames -> conv
frontend -> learned 1500x1280 positional embedding -> 32 pre-norm layers,
d_model 1280, 20 heads, FFN 5120 -> avg-pooler halving 1500 -> 750) + an
RMSNorm -> fc1 -> relu2 -> fc2 projector (1280 -> 4096 -> 2048) + a
Nemotron-Dense 2B decoder (28 layers, GQA 16Q/8KV head_dim 128, hidden 2048,
intermediate 9216, plain squared-ReLU MLP -- NO gate -- UNTIED lm_head, RoPE
theta 1e8, rms eps 1e-5).

This converter mirrors scripts/convert_qwen3_gguf.py: an explicit, complete
tensor-name map (a checkpoint addition must fail conversion), the config baked
as `audex.*` KV metadata, the byte-level BPE tokenizer, and the mel frontend
constants. The checkpoint safetensors are BF16 (config.json's "dtype: float32" is stale
-- every shard tensor is BF16, which is what the reference loads), so weights
stream through as the raw BF16 bytes. The audio-token count per 30 s clip is
FIXED at 750 (the avg-pooler output), simpler than qwen3's formula. The
ChatML prompt layout is baked as prefix/suffix token-id arrays around the 750
<so_embedding> copies (tokenization is invariant to the count -- every
boundary token is special).
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
import gguf

REVISION = "77b7e1a4de899769f22c1bc074db666601c28907"
# The audex snapshot is a manual download living next to the repos (verified
# sha256 against the pinned revision); fall back to the hub cache layout.
DEFAULT_SNAPSHOT_CANDIDATES = (
    Path.home() / "Documents" / "starling" / ".hf-cache" / "audex-2b" / "checkpoint_folder_full",
    Path.home()
    / ".cache/huggingface/hub/models--nvidia--Nemotron-Labs-Audex-2b/snapshots"
    / REVISION
    / "checkpoint_folder_full",
)

N_ENC_LAYERS = 32
N_LLM_LAYERS = 28
VOCAB_SIZE = 205312


# ---------------------------------------------------------------------------
# Tensor-name mapping: HF checkpoint name -> Starling GGUF name.
#
# Encoder (stock transformers Qwen2AudioEncoder):
#   enc.conv1 / conv2            Conv1d k3 s1/s2 p1 + GELU stack over time
#                                (weights keep the HF [OC, IC, K] byte order,
#                                which is ggml's [K, 1, IC, OC] conv layout)
#   enc.pos_embed                LEARNED (1500, 1280) positional embedding
#   enc.blk.N.attn_norm          pre-attention LayerNorm (biased)
#   enc.blk.N.attn_{q,v,o}       MHA projections (biased)
#   enc.blk.N.attn_k             keys (NO bias -- the one bias-free proj)
#   enc.blk.N.ffn_norm           pre-FFN LayerNorm (biased)
#   enc.blk.N.ff_up / ff_down    fc1/fc2 (erf GELU between, both biased)
#   enc.ln_post                  final LayerNorm after the avg-pooler (biased)
#
# Projector (NemotronDenseAudexProjector, all bias-free):
#   proj.norm                    RMSNorm(1280, eps 1e-5) -> fc1 -> relu2 -> fc2
#
# LLM decoder (Nemotron-Dense trunk: bias-free, NO q/k norm, UNTIED lm_head,
# relu2 MLP = up/down only -- no gate tensor):
#   llm.blk.N.{attn_norm,attn.q/k/v/o,ffn_norm,ffn.up,ffn.down}
#   llm.embed / llm.final_norm / llm.lm_head
def gguf_name(name: str) -> str:
    # ---- encoder frontend ----
    if name == "audio_encoder.conv1.weight": return "enc.conv1.weight"
    if name == "audio_encoder.conv1.bias":   return "enc.conv1.bias"
    if name == "audio_encoder.conv2.weight": return "enc.conv2.weight"
    if name == "audio_encoder.conv2.bias":   return "enc.conv2.bias"
    if name == "audio_encoder.embed_positions.weight": return "enc.pos_embed"
    if name == "audio_encoder.layer_norm.weight": return "enc.ln_post.weight"
    if name == "audio_encoder.layer_norm.bias":   return "enc.ln_post.bias"
    if name.startswith("audio_encoder.layers."):
        p = name.removeprefix("audio_encoder.layers.").split(".")
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
    if name.startswith("audio_projector."):
        return "proj." + name.removeprefix("audio_projector.")
    # ---- LLM decoder (Nemotron-Dense trunk) ----
    if name == "model.embed_tokens.weight": return "llm.embed.weight"
    if name == "model.norm.weight":         return "llm.final_norm.weight"
    if name == "lm_head.weight":            return "llm.lm_head.weight"
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
            "mlp.up_proj": "ffn.up",
            "mlp.down_proj": "ffn.down",
        }
        for old, new in repl.items():
            if rest.startswith(old + "."):
                return f"llm.blk.{i}.{new}." + rest[len(old) + 1:]
    raise KeyError(f"no audex GGUF mapping for {name!r}")


def add_metadata(w: gguf.GGUFWriter) -> None:
    V = gguf.GGUFValueType
    w.add_key_value("starling.format_version", 1, V.UINT32)
    w.add_string("starling.numeric_profile", "bf16_exact")
    w.add_string("general.architecture", "audex")

    def ints(**xs):
        for k, v in xs.items():
            w.add_key_value("audex." + k, v, V.UINT32)

    def floats(**xs):
        for k, v in xs.items():
            w.add_key_value("audex." + k, float(v), V.FLOAT32)

    def strings(**xs):
        for k, v in xs.items():
            w.add_string("audex." + k, v)

    # Frontend: WhisperFeatureExtractor (torch.stft n_fft=400, hop=160,
    # periodic hann, center/reflect; power=2; slaney mel scale + norm;
    # log10). Normalization: x = log10(clamp(mel_spec, 1e-10)); global max
    # over the KEPT frames (the trailing STFT frame is dropped BEFORE the
    # clamp -- T = S/H = 3000 at the fixed 30 s clip); x = max(x, mx - 8);
    # x = (x + 4) / 4. Every clip is zero-padded to n_samples = 480000
    # (padding="max_length", engine-side before the shared mel).
    ints(
        **{
            "frontend.sample_rate": 16000,
            "frontend.n_fft": 400,
            "frontend.win_length": 400,
            "frontend.hop_length": 160,
            "frontend.n_mels": 128,
            "frontend.power": 2,
            "frontend.chunk_length": 30,
            "frontend.n_samples": 480000,
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

    # Full-attention whisper-shaped encoder.
    ints(
        **{
            "enc.n_mel": 128,
            "enc.hidden": 1280,
            "enc.layers": 32,
            "enc.heads": 20,
            "enc.head_dim": 64,
            "enc.ffn_dim": 5120,
            "enc.max_pos_emb": 1500,
            "enc.out_frames": 750,
        }
    )
    w.add_key_value("audex.enc.layer_norm_eps", 1e-5, V.FLOAT32)

    # Projector (RMSNorm -> fc1 -> relu2 -> fc2, bias-free).
    ints(**{"proj.hidden": 1280, "proj.intermediate": 4096,
            "proj.output_dim": 2048})
    w.add_key_value("audex.proj.norm_eps", 1e-5, V.FLOAT32)
    w.add_string("audex.proj.activation", "relu2")

    # LLM (Nemotron-Dense trunk: untied lm_head, no q/k norm, relu2 MLP,
    # stock numerics -- all multipliers 1, default scale).
    ints(
        **{
            "llm.hidden_size": 2048,
            "llm.num_layers": 28,
            "llm.num_heads": 16,
            "llm.num_kv_heads": 8,
            "llm.head_dim": 128,
            "llm.intermediate_size": 9216,
            "llm.vocab_size": 205312,
            "llm.max_position_embeddings": 131072,
            "llm.max_cache_len": 4096,
        }
    )
    floats(**{"llm.rope_theta": 100000000.0, "llm.rms_norm_eps": 1e-5})
    w.add_key_value("audex.llm.tied_embeddings", False, V.BOOL)
    w.add_key_value("audex.llm.has_qk_norm", False, V.BOOL)
    w.add_string("audex.llm.hidden_act", "relu2")

    # Token ids + generation + the serve chunk policy (mirrored by the
    # engine's decode entry so the C++ path matches the Python server
    # byte-for-byte). eos is the serving path's <|im_end|>.
    ints(
        **{
            "audio_token_id": 29,
            "pad_token_id": 0,
            "eos_token_id": 11,
            "sound_start_token_id": 30,
            "sound_end_token_id": 31,
            "sound_embedding_size": 750,
            "max_new_tokens": 200,
        }
    )
    floats(**{"chunk_seconds": 30.0})


# Chat-template prompt layout, captured under the reference tokenizer (the
# repo's main .venv, transformers 5.15.0, AutoTokenizer over the pinned
# snapshot; verified invariant to the placeholder count N for N in
# {1, 13, 750} -- every boundary token is special). The rendered prompt is
# "<|im_start|>user\n<so_start>" + [<so_embedding> x N] + "<so_end>\n" +
# "Transcribe the speech in the input audio.<|im_end|>\n" +
# "<|im_start|>assistant\n<think></think>" (N = 750 per 30 s clip; 23 text
# tokens -> prompt_len 773 for one clip).
PROMPT_PREFIX = [10, 3263, 1010, 30]
PROMPT_SUFFIX = [31, 1010, 6881, 13089, 1278, 16181, 1294, 1278, 4292,
                 16023, 1046, 11, 1010, 10, 1503, 19464, 1010, 12, 13]


def prompt_layout(w: gguf.GGUFWriter) -> None:
    V = gguf.GGUFValueType
    w.add_key_value("audex.prompt_prefix", PROMPT_PREFIX, V.ARRAY, V.INT32)
    w.add_key_value("audex.prompt_suffix", PROMPT_SUFFIX, V.ARRAY, V.INT32)
    print(f"prompt: prefix={PROMPT_PREFIX} suffix={PROMPT_SUFFIX}")


def tokenizer(w: gguf.GGUFWriter, snapshot: Path) -> None:
    import json

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
    w.add_eos_token_id(11)
    w.add_pad_token_id(0)


def frontend() -> tuple[np.ndarray, np.ndarray]:
    """Mel filterbank (201 x 128, slaney scale+norm) + periodic Hann window.

    Uses precisely transformers.audio_utils.mel_filter_bank with the
    WhisperFeatureExtractor arguments (bitwise identical across the repo
    venvs' transformers versions), stored freq-major [n_fft/2+1, n_mels]
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
    ap.add_argument("--snapshot", type=Path, default=None)
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("models/audex-2b-bf16-exact.gguf"),
    )
    args = ap.parse_args()

    snapshot = args.snapshot
    if snapshot is None:
        snapshot = next((p for p in DEFAULT_SNAPSHOT_CANDIDATES if p.exists()), None)
        if snapshot is None:
            raise SystemExit(
                "audex snapshot not found; pass --snapshot (pinned revision "
                f"{REVISION[:12]})"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    w = gguf.GGUFWriter(args.output, "audex", use_temp_file=True)
    add_metadata(w)
    tokenizer(w, snapshot)
    prompt_layout(w)

    learned = 0
    dtypes = set()
    shard_names = sorted(
        set(
            __import__("json").loads(
                (snapshot / "model.safetensors.index.json").read_text()
            )["weight_map"].values()
        )
    )
    for shard in shard_names:
        with safe_open(snapshot / shard, framework="pt", device="cpu") as f:
            for source in sorted(f.keys()):
                target = gguf_name(source)
                t = f.get_tensor(source)
                dtypes.add(str(t.dtype))
                if t.dtype is not torch.bfloat16:
                    raise TypeError(f"{source}: expected BF16, found {t.dtype}")
                # GGUF stores dims innermost-first; keep the BF16 byte stream
                # in checkpoint row-major order (the HF [OC, IC, K] conv
                # layout IS ggml's [K, 1, IC, OC]).
                shape = tuple(t.shape)
                a = np.ascontiguousarray(t.view(torch.uint16).numpy()).reshape(shape)
                w.add_tensor(
                    target, a, raw_shape=shape,
                    raw_dtype=gguf.GGMLQuantizationType.BF16,
                )
                learned += 1
    if learned != 717:
        raise SystemExit(f"expected 717 checkpoint tensors, converted {learned}")

    mel, window = frontend()
    w.add_tensor("audio.mel_filters", np.ascontiguousarray(mel, dtype=np.float32))
    w.add_tensor("audio.mel_window", np.ascontiguousarray(window, dtype=np.float32))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(
        f"wrote {args.output}: {learned + 2} tensors ({learned} learned BF16 "
        f"(from {sorted(dtypes)}) + mel/window), revision {REVISION[:12]}"
    )


if __name__ == "__main__":
    main()
