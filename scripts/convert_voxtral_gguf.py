#!/usr/bin/env python3
"""Convert the Voxtral-Mini-4B-Realtime safetensors checkpoint to a Starling GGUF.

mistralai/Voxtral-Mini-4B-Realtime-2602 is a Whisper-style causal audio encoder
(32 layers, d_model 1280, 32 heads x head_dim 64, sliding window 750, RoPE theta
1e6) feeding a downsample-4 projector (5120 -> 3072 GELU 3072 -> 3072) into a
Ministral-3-class text decoder (26 layers, hidden 3072, GQA 32q/8kv x head_dim
128, sliding window 8192, tied lm_head, per-layer AdaRMSNorm on the MLP branch).

The tensor-name map below is explicit and complete (a checkpoint addition fails
conversion, the ark KeyError pattern). All weights are stored BF16 (the
checkpoint dtype). The checkpoint ships unsharded (one model.safetensors, no
index); tensors stream one at a time so peak RAM stays low.

Mel arithmetic (settled empirically 2026-09-05 against the stock
VoxtralRealtimeFeatureExtractor on tests/fixtures/{short,medium,long}.wav):

* offline pad: padded = ceil(S/1280)*1280 + 49*1280 zeros (32 left-pad + 17
  right-pad tokens of 1280 samples = 8 mel frames each).
* the extractor's STFT runs center=True (1 + padded//160 raw frames) but then
  drops the last TIME frame (stft[..., :-1]), canceling the +1: mel_T is
  padded//160 exactly, always a multiple of 8 (short 1136, medium 2624, long
  7832), giving conv2 lengths divisible by 4 and audio tokens mel_T//8 with no
  remainder (short 142, medium 328, long 979).

Tokenizer: decode-only raw bytes, no merges. tekken.json (parsed with plain
json, not a tokenizer class) holds 1000 specials (ids 0..999, CONTROL type)
plus 150000 rank-ordered tokens; only ids < 131072 are kept (tekken's tail
131072..149999 is unusable). Token strings are latin-1 of the base64-decoded
token_bytes (exact byte round-trip); the C++ side concats bytes and skips
CONTROL ids.
"""
from __future__ import annotations
import argparse, base64, json, math
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
import gguf

REVISION = "2769294da9567371363522aac9bbcfdd19447add"
DEFAULT_SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub/models--mistralai--Voxtral-Mini-4B-Realtime-2602/snapshots"
    / REVISION
)
VOCAB_SIZE = 131072
NUM_SPECIAL = 1000
NUM_DELAY_TOKENS = 6
LEFT_PAD_TOKENS = 32
RIGHT_PAD_TOKENS = 17  # delay 6 + BOS 1 + buffer 10
RAW_SAMPLES_PER_AUDIO_TOK = 1280  # hop 160 * 8 mel frames per audio token


# ---------------------------------------------------------------------------
# Tensor-name mapping: HF checkpoint name -> Starling GGUF name.
#
# Audio embedder (causal convs over 128 mel bins):
#   enc.conv1.{weight,bias}    Conv1d(128->1280, k3 s1, left-pad 2)
#   enc.conv2.{weight,bias}    Conv1d(1280->1280, k3 s2, left-pad 1)
# Audio encoder (32 pre-norm layers; attention projects to 32 heads x head_dim
# 64 = 2048, NOT the hidden 1280):
#   enc.blk.N.attn_norm.weight         self_attn_layer_norm (no bias)
#   enc.blk.N.attn.{q,v,o}.{weight,bias}  (k has weight only, NO bias)
#   enc.blk.N.ffn_norm.weight          final_layer_norm (no bias)
#   enc.blk.N.ffn.{gate,up,down}.weight   (ONLY down has a bias)
#   enc.final_norm.weight              audio_tower.norm (no bias)
# Projector (downsample 4; reshape groups 4x1280 -> 5120; no biases):
#   proj.fc0.weight   Linear(5120 -> 3072)
#   proj.fc2.weight   Linear(3072 -> 3072)
# Text decoder (26 layers, hidden 3072, 32q/8kv x head_dim 128; no biases
# anywhere; per-layer AdaRMSNorm Linear(3072->32) GELU Linear(32->3072)):
#   llm.embed.weight / llm.final_norm.weight  (lm_head tied; skipped)
#   llm.blk.N.{attn_norm,ffn_norm}.weight
#   llm.blk.N.attn.{q,k,v,o}.weight
#   llm.blk.N.ffn.{gate,up,down}.weight
#   llm.blk.N.ada.{fc0,fc2}.weight
def gguf_name(name: str) -> str | None:
    if name == "lm_head.weight":
        return None  # tied to embed_tokens; not duplicated
    if name.startswith("audio_tower.embedder.conv"):
        return "enc." + name.removeprefix("audio_tower.embedder.")
    if name.startswith("audio_tower.layers."):
        p = name.removeprefix("audio_tower.layers.").split(".")
        i, rest = p[0], ".".join(p[1:])
        repl = {
            "self_attn_layer_norm": "attn_norm",
            "self_attn.q_proj": "attn.q",
            "self_attn.k_proj": "attn.k",
            "self_attn.v_proj": "attn.v",
            "self_attn.o_proj": "attn.o",
            "final_layer_norm": "ffn_norm",
            "mlp.gate_proj": "ffn.gate",
            "mlp.up_proj": "ffn.up",
            "mlp.down_proj": "ffn.down",
        }
        for old, new in repl.items():
            if rest.startswith(old + "."):
                return f"enc.blk.{i}.{new}." + rest[len(old) + 1:]
    if name == "audio_tower.norm.weight":
        return "enc.final_norm.weight"
    if name == "multi_modal_projector.linear_1.weight":
        return "proj.fc0.weight"
    if name == "multi_modal_projector.linear_2.weight":
        return "proj.fc2.weight"
    if name == "language_model.model.embed_tokens.weight":
        return "llm.embed.weight"
    if name == "language_model.model.norm.weight":
        return "llm.final_norm.weight"
    if name.startswith("language_model.model.layers."):
        p = name.removeprefix("language_model.model.layers.").split(".")
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
            "ada_rms_norm.linear1": "ada.fc0",
            "ada_rms_norm.linear2": "ada.fc2",
        }
        for old, new in repl.items():
            if rest.startswith(old + "."):
                return f"llm.blk.{i}.{new}." + rest[len(old) + 1:]
    raise KeyError(f"no VOXTRAL GGUF mapping for {name!r}")


def add_metadata(w: gguf.GGUFWriter) -> None:
    V = gguf.GGUFValueType
    w.add_key_value("starling.format_version", 1, V.UINT32)
    w.add_string("starling.numeric_profile", "bf16_exact")
    w.add_string("general.architecture", "voxtral")

    def ints(**xs):
        for k, v in xs.items():
            w.add_key_value("voxtral." + k, v, V.UINT32)

    def floats(**xs):
        for k, v in xs.items():
            w.add_key_value("voxtral." + k, float(v), V.FLOAT32)

    def strings(**xs):
        for k, v in xs.items():
            w.add_string("voxtral." + k, v)

    # Frontend: stock _torch_extract_fbank_features (n_fft/hann 400, hop 160,
    # 128 slaney bins, stft magnitude squared dropping the last freq bin,
    # log10 clamp min 1e-10, GLOBAL fixed log-mel max 1.5 -- streaming-safe,
    # not per-utterance -- floor at max-8, then (x+4)/4). center=True offline.
    ints(
        **{
            "frontend.sample_rate": 16000,
            "frontend.n_fft": 400,
            "frontend.win_length": 400,
            "frontend.hop_length": 160,
            "frontend.n_mels": 128,
            "frontend.center": 1,
            "frontend.unit_samples": RAW_SAMPLES_PER_AUDIO_TOK,
            "frontend.left_pad_tokens": LEFT_PAD_TOKENS,
            "frontend.right_pad_tokens": RIGHT_PAD_TOKENS,
        }
    )
    floats(
        **{
            "frontend.mel_floor": 1e-10,
            "frontend.log_mel_max": 1.5,
            "frontend.normalization_offset": 4.0,
            "frontend.normalization_divisor": 4.0,
            "frontend.dynamic_range": 8.0,
        }
    )
    strings(
        **{
            "frontend.mel_scale": "slaney",
            "frontend.log": "log10",
            "frontend.output_dtype": "bf16",
        }
    )

    # Audio encoder: 32 layers, d_model 1280; attention projects to
    # heads*head_dim = 2048 (NOT the hidden width).
    ints(
        **{
            "enc.num_mel_bins": 128,
            "enc.encoder_layers": 32,
            "enc.d_model": 1280,
            "enc.encoder_attention_heads": 32,
            "enc.head_dim": 64,
            "enc.encoder_ffn_dim": 5120,
            "enc.sliding_window": 750,
            "enc.conv_kernel": 3,
            "enc.conv_left_pad1": 2,
            "enc.conv_left_pad2": 1,
            "enc.conv_stride2": 2,
        }
    )
    floats(**{"enc.rope_theta": 1000000.0, "enc.rms_norm_eps": 1e-5})

    # Projector: reshape groups 4 -> Linear(5120->3072, no bias) GELU
    # Linear(3072->3072, no bias); 8 mel frames per audio token.
    ints(
        **{
            "proj.input_size": 5120,
            "proj.output_size": 3072,
            "proj.downsample": 4,
            "proj.mel_per_token": 8,
        }
    )
    strings(**{"proj.act": "gelu"})

    # Text decoder: 26 layers, hidden 3072, GQA 32q/8kv x head_dim 128,
    # sliding window 8192, RoPE theta 1e6, RMSNorm eps 1e-5, tied lm_head.
    # AdaRMSNorm: Linear(3072->32) GELU Linear(32->3072) on the MLP branch,
    # driven by the sinusoidal TimeEmbedding(dim 3072, theta 10000) of the
    # fixed per-request num_delay_tokens (baked below as llm.t_cond).
    ints(
        **{
            "llm.hidden_size": 3072,
            "llm.num_layers": 26,
            "llm.num_heads": 32,
            "llm.num_kv_heads": 8,
            "llm.head_dim": 128,
            "llm.intermediate_size": 9216,
            "llm.vocab_size": VOCAB_SIZE,
            "llm.sliding_window": 8192,
            "llm.tied": 1,
            "llm.num_delay_tokens": NUM_DELAY_TOKENS,
            "llm.time_embedding_dim": 3072,
            "llm.ada_bottleneck": 32,
            "llm.max_cache_len": 4096,
        }
    )
    floats(**{"llm.rope_theta": 1000000.0, "llm.rms_norm_eps": 1e-5})
    w.add_key_value("voxtral.llm.time_embedding_theta", 10000.0, V.FLOAT32)

    # Token ids (verified against tekken.json specials + the processor).
    ints(
        **{
            "bos_token_id": 1,
            "eos_token_id": 2,
            "pad_token_id": 11,
            "streaming_pad_id": 32,
            "left_pad_tokens": LEFT_PAD_TOKENS,
            "right_pad_tokens": RIGHT_PAD_TOKENS,
            "max_new_tokens": 200,
        }
    )

    # Prompt prefix baked as token ids: mistral-common's offline
    # encode_streaming_tokens output, verified against the stock processor on
    # all three fixtures (P == 39, all streaming-pad after BOS).
    w.add_key_value("voxtral.prompt_prefix", [1] + [32] * 38, V.ARRAY, V.INT32)


def tokenizer(w: gguf.GGUFWriter, snapshot: Path) -> None:
    tek = json.loads((snapshot / "tekken.json").read_text())
    specials = tek["special_tokens"]  # 1000 entries, ids 0..999
    assert len(specials) == NUM_SPECIAL, f"expected 1000 specials, got {len(specials)}"
    tokens: list[str] = [""] * VOCAB_SIZE
    types = [gguf.TokenType.NORMAL] * VOCAB_SIZE
    for i, sp in enumerate(specials):
        tokens[i] = sp["token_str"]
        if sp.get("is_control", False):
            types[i] = gguf.TokenType.CONTROL
    for entry in tek["vocab"]:
        idx = NUM_SPECIAL + entry["rank"]
        if idx < VOCAB_SIZE:  # tekken's tail 131072..149999 is unusable; drop it
            # latin-1 preserves the raw bytes exactly (C++ decodes the same way).
            tokens[idx] = base64.b64decode(entry["token_bytes"]).decode("latin-1")
    for i, t in enumerate(tokens):
        if not t:
            tokens[i] = "[PAD" + str(i) + "]"
    w.add_tokenizer_model("gpt2")
    w.add_token_list(tokens)
    w.add_token_scores([0.0] * len(tokens))
    w.add_token_types(types)
    # Decode-only: no merges (the C++ side concats raw token bytes).
    w.add_bos_token_id(1)
    w.add_eos_token_id(2)
    w.add_pad_token_id(11)


def time_cond() -> np.ndarray:
    """Baked llm.t_cond: TimeEmbedding(num_delay_tokens=6), dim 3072.

    Mirrors VoxtralRealtimeTimeEmbedding (transformers
    modeling_voxtral_realtime.py) through the stock bf16 path bit-exactly:
    the f32 inv_freq buffer is cast to the input dtype (bf16, the model
    dtype), emb = time * inv_freq runs in bf16, and out = [cos, sin] is
    computed on the bf16 tensor -- the same values the pipeline's
    _precompute_ada feeds the ada linears. Stored f32 (exact cast-up).
    """
    dim, theta, t = 3072, 10000.0, 6.0
    inv = np.exp(-math.log(theta) * np.arange(dim // 2, dtype=np.float64) / (dim // 2))
    inv_bf16 = torch.from_numpy(inv.astype(np.float32)).to(torch.bfloat16)
    emb = torch.full((1,), t, dtype=torch.bfloat16) * inv_bf16
    out = torch.cat((emb.cos(), emb.sin()))
    return np.asarray(out.to(torch.float32))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("models/voxtral-mini-4b-realtime-bf16-exact.gguf"),
    )
    args = ap.parse_args()

    # Unsharded checkpoint: one model.safetensors, no index. (The snapshot also
    # carries a mistral-native consolidated.safetensors with different key
    # names; the HF model.safetensors below is the mapped source.)
    shard_path = args.snapshot / "model.safetensors"
    if not shard_path.exists():
        raise FileNotFoundError(f"no model.safetensors under {args.snapshot}")
    shards = [shard_path]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    w = gguf.GGUFWriter(args.output, "voxtral", use_temp_file=True)
    add_metadata(w)
    tokenizer(w, args.snapshot)
    w.add_tensor("llm.t_cond", np.ascontiguousarray(time_cond(), dtype=np.float32))

    learned = 0
    dtypes = set()
    skipped = []
    # Stream tensors one at a time; the 8.9 GB weights never sit in RAM.
    for shard_path in shards:
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

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(
        f"wrote {args.output}: {learned + 1} tensors ({learned} learned), "
        f"skipped {len(skipped)} (tied), source dtypes={sorted(dtypes)}"
    )
    if skipped:
        print(f"skipped: {skipped}")


if __name__ == "__main__":
    main()
