#!/usr/bin/env python3
"""Convert the pinned granite-speech-4.1-2b safetensors checkpoint to a Starling GGUF.

ibm-granite/granite-speech-4.1-2b is a CTC-conformer encoder + BLIP2 Q-Former
projector + Granite-4.0-1b decoder ASR model. The encoder (16 blocks, hidden
1024, 8 heads x 128, block-local Shaw relative-position attention over windows
of 200 frames, depthwise-conv module with BatchNorm, mid-stack self-conditioned
CTC at block 8) consumes 160-dim mel frames (torchaudio MelSpectrogram
n_mels=80, n_fft=512, win_length=400, hop=160, then log10/normalize and a
consecutive-pair stack). The projector windows the encoder output into 15-frame
blocks and cross-attends with 3 learned queries (2 BERT-style qformer layers,
GELU, LayerNorm eps 1e-12), emitting 3 tokens per block into the decoder's 2048
space. The decoder is a bias-free Qwen-family trunk WITHOUT q_norm/k_norm and
an UNTIED lm_head, plus the Granite numerics: embedding_multiplier 12.0,
attention_multiplier 0.0078125 (used directly as the softmax scale),
residual_multiplier 0.22, logits_scaling 8.0.

This converter mirrors scripts/convert_ark_gguf.py: an explicit, complete
tensor-name map (a checkpoint addition should fail conversion), the config
baked as `granite.*` KV metadata, the byte-level BPE tokenizer, and the mel
frontend constants. All weights are stored BF16 (the checkpoint dtype). The
per-layer Shaw rel-pos bias is PRECOMPUTED here (an exact embedding gather over
the deterministic attention_dists table) and stored as a (200, 200, 128) bf16
tensor per layer, so the C++ encoder never re-derives it. The chat-template
prompt layout is baked as prefix/suffix token-id arrays (the single <|audio|>
placeholder expands to N copies at runtime; tokenization is invariant to N --
special tokens are hard BPE boundaries). `out_llm.safetensors` (the
self-speculative CTC draft head) is out of scope and not converted.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
import gguf

REVISION = "de575db64086f84fdc79da4932d1076e965bc546"
DEFAULT_SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub/models--ibm-granite--granite-speech-4.1-2b/snapshots"
    / REVISION
)

# Encoder rel-pos geometry (GraniteSpeechCTCEncoder): attention_dists[i, j] =
# clamp(i - j, -context, context) + max_pos_emb indexes each layer's
# rel_pos_emb table (2*max_pos_emb+1, head_dim).
CONTEXT_SIZE = 200
MAX_POS_EMB = 512
N_LAYERS = 16
DEFAULT_TASK_PROMPT = "transcribe the speech with proper punctuation and capitalization."


# ---------------------------------------------------------------------------
# Tensor-name mapping: HF checkpoint name -> Starling GGUF name.
#
# Encoder (CTC conformer):
#   enc.input_linear            Linear(160 -> 1024, bias)
#   enc.blk.N.ff1_norm / ff1_up / ff1_down     ff half-module (LN + up + SiLU + down)
#   enc.blk.N.attn_norm / attn_q / attn_kv / attn_o   block-local Shaw attention
#                                (to_q/to_kv bias-free, to_out biased)
#   enc.blk.N.rel_pos_bias      PRECOMPUTED (200, 200, 128) Shaw bias (see above)
#   enc.blk.N.conv_norm / conv_up / conv_depth / bn_* / conv_down
#                                conv module (LN, pw-up, GLU, depthwise k15,
#                                BatchNorm1d eval, SiLU, pw-down)
#   enc.blk.N.post_norm         block-closing LayerNorm
#   enc.out / enc.out_mid       mid-stack CTC feedback heads (1024 <-> 348)
#
# Projector (BLIP2 Q-Former):
#   proj.query                 (3, 1024) learned queries
#   proj.qformer_ln            LayerNorm applied to the queries (eps 1e-12)
#   proj.blk.N.self_{q,k,v} / self_out / self_ln          self-attention block
#   proj.blk.N.cross_{q,k,v} / cross_out / cross_ln       cross-attention block
#   proj.blk.N.ff_up / ff_down / ff_ln                    query FFN (erf GELU)
#   proj.out                   Linear(1024 -> 2048, bias)
#
# LLM decoder (Granite-4.0-1b trunk, bias-free, no q/k norm, UNTIED lm_head):
#   llm.blk.N.{attn_norm,attn.q/k/v/o,ffn_norm,ffn.gate/up/down}
#   llm.embed / llm.lm_head / llm.final_norm
def gguf_name(name: str) -> str:
    # ---- encoder ----
    if name == "encoder.input_linear.weight": return "enc.input_linear.weight"
    if name == "encoder.input_linear.bias":   return "enc.input_linear.bias"
    if name == "encoder.out.weight":          return "enc.out.weight"
    if name == "encoder.out.bias":            return "enc.out.bias"
    if name == "encoder.out_mid.weight":      return "enc.out_mid.weight"
    if name == "encoder.out_mid.bias":        return "enc.out_mid.bias"
    if name.startswith("encoder.layers."):
        p = name.removeprefix("encoder.layers.").split(".")
        i, rest = p[0], ".".join(p[1:])
        repl = {
            "ff1.pre_norm": "ff1_norm", "ff1.up_proj": "ff1_up", "ff1.down_proj": "ff1_down",
            "ff2.pre_norm": "ff2_norm", "ff2.up_proj": "ff2_up", "ff2.down_proj": "ff2_down",
            "attn.pre_norm": "attn_norm", "attn.to_q": "attn_q", "attn.to_kv": "attn_kv",
            "attn.to_out": "attn_o",
            "conv.norm": "conv_norm", "conv.up_conv": "conv_up",
            "conv.depth_conv.conv": "conv_depth",
            "conv.batch_norm": "bn",  # weight/bias/running_mean/running_var below
            "conv.down_conv": "conv_down", "post_norm": "post_norm",
        }
        for old, new in repl.items():
            if rest.startswith(old + "."):
                tail = rest[len(old) + 1:]
                if new == "bn":
                    bn = {"weight": "bn_weight", "bias": "bn_bias",
                          "running_mean": "bn_mean", "running_var": "bn_var"}.get(tail)
                    if bn:
                        return f"enc.blk.{i}.{bn}"
                    if tail == "num_batches_tracked":
                        return None  # eval-mode BatchNorm never reads the counter
                    return None
                # conv_up/conv_depth keep the .weight/.bias tail (k=1 convs are
                # stored squeezed to 2D Linear-style weights; depthwise is
                # [C, K] rows of taps).
                return f"enc.blk.{i}.{new}." + tail
    # ---- projector ----
    if name == "projector.query": return "proj.query"
    if name == "projector.linear.weight": return "proj.out.weight"
    if name == "projector.linear.bias":   return "proj.out.bias"
    if name == "projector.qformer.layernorm.weight": return "proj.qformer_ln.weight"
    if name == "projector.qformer.layernorm.bias":   return "proj.qformer_ln.bias"
    if name.startswith("projector.qformer.encoder.layer."):
        p = name.removeprefix("projector.qformer.encoder.layer.").split(".")
        i, rest = p[0], ".".join(p[1:])
        repl = {
            "attention.attention.query": "self_q",
            "attention.attention.key": "self_k",
            "attention.attention.value": "self_v",
            "attention.output.dense": "self_out",
            "attention.output.LayerNorm": "self_ln",
            "crossattention.attention.query": "cross_q",
            "crossattention.attention.key": "cross_k",
            "crossattention.attention.value": "cross_v",
            "crossattention.output.dense": "cross_out",
            "crossattention.output.LayerNorm": "cross_ln",
            "intermediate_query.dense": "ff_up",
            "output_query.dense": "ff_down",
            "output_query.LayerNorm": "ff_ln",
        }
        for old, new in repl.items():
            if rest.startswith(old + "."):
                return f"proj.blk.{i}.{new}." + rest[len(old) + 1:]
    # ---- LLM decoder (Granite-4.0-1b) ----
    if name == "language_model.model.embed_tokens.weight": return "llm.embed.weight"
    if name == "language_model.model.norm.weight":         return "llm.final_norm.weight"
    if name == "language_model.lm_head.weight":            return "llm.lm_head.weight"
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
        }
        for old, new in repl.items():
            if rest.startswith(old + "."):
                return f"llm.blk.{i}.{new}." + rest[len(old) + 1:]
    raise KeyError(f"no granite GGUF mapping for {name!r}")


def attention_dists() -> torch.Tensor:
    """(context, context) int64 relative-distance table (GraniteSpeechCTCEncoder)."""
    seq = torch.arange(CONTEXT_SIZE)
    relpos = seq.view(-1, 1) - seq.view(1, -1)
    return torch.clamp(relpos, -CONTEXT_SIZE, CONTEXT_SIZE) + MAX_POS_EMB


def add_metadata(w: gguf.GGUFWriter) -> None:
    V = gguf.GGUFValueType
    w.add_key_value("starling.format_version", 1, V.UINT32)
    w.add_string("starling.numeric_profile", "bf16_exact")
    w.add_string("general.architecture", "granite")

    def ints(**xs):
        for k, v in xs.items():
            w.add_key_value("granite." + k, v, V.UINT32)

    def floats(**xs):
        for k, v in xs.items():
            w.add_key_value("granite." + k, float(v), V.FLOAT32)

    def strings(**xs):
        for k, v in xs.items():
            w.add_string("granite." + k, v)

    # Frontend: GraniteSpeechFeatureExtractor (torchaudio MelSpectrogram
    # n_mels=80, n_fft=512, win_length=400, hop=160, center/reflect, power=2,
    # htk mel scale, no norm). Normalization: x = log10(max(power, 1e-10));
    # global amax over ALL frames; x = max(x, mx - 8) / 4 + 1 (the C++ mel
    # frontend computes (v + offset) / divisor, and (v + 4) / 4 == v / 4 + 1
    # bit-exactly in f32). The odd trailing frame is dropped and consecutive
    # pairs stacked into 160-dim frames (engine-side, after the shared mel).
    ints(
        **{
            "frontend.sample_rate": 16000,
            "frontend.n_fft": 512,
            "frontend.win_length": 400,
            "frontend.hop_length": 160,
            "frontend.n_mels": 80,
            "frontend.power": 2,
            "frontend.chunk_length": 30,
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
            "frontend.mel_scale": "htk",
            "frontend.mel_norm": "none",
            "frontend.log": "log10",
            "frontend.output_dtype": "bf16",
        }
    )

    # CTC conformer encoder.
    ints(
        **{
            "enc.input_dim": 160,
            "enc.hidden": 1024,
            "enc.layers": 16,
            "enc.heads": 8,
            "enc.head_dim": 128,
            "enc.ffn_dim": 4096,
            "enc.conv_kernel": 15,
            "enc.context_size": 200,
            "enc.max_pos_emb": 512,
            "enc.output_dim": 348,
            "enc.mid_layer": 8,
        }
    )
    w.add_key_value("granite.enc.layer_norm_eps", 1e-5, V.FLOAT32)

    # Projector (BLIP2 Q-Former).
    ints(
        **{
            "proj.window_size": 15,
            "proj.downsample_rate": 5,
            "proj.num_queries": 3,
            "proj.hidden": 1024,
            "proj.qformer_layers": 2,
            "proj.qformer_heads": 16,
            "proj.qformer_intermediate": 4096,
            "proj.output_dim": 2048,
        }
    )
    w.add_key_value("granite.proj.layer_norm_eps", 1e-12, V.FLOAT32)

    # LLM (Granite-4.0-1b trunk + Granite numerics).
    ints(
        **{
            "llm.hidden_size": 2048,
            "llm.num_layers": 40,
            "llm.num_heads": 16,
            "llm.num_kv_heads": 4,
            "llm.head_dim": 128,
            "llm.intermediate_size": 4096,
            "llm.vocab_size": 100353,
            "llm.max_position_embeddings": 4096,
            "llm.max_cache_len": 640,
        }
    )
    floats(
        **{
            "llm.rope_theta": 10000.0,
            "llm.rms_norm_eps": 1e-5,
            "llm.attention_multiplier": 0.0078125,
            "llm.embedding_multiplier": 12.0,
            "llm.residual_multiplier": 0.22,
            "llm.logits_scaling": 8.0,
        }
    )
    w.add_key_value("granite.llm.tied_embeddings", False, V.BOOL)
    w.add_key_value("granite.llm.has_qk_norm", False, V.BOOL)

    # Token ids + generation + the serve chunk policy (mirrored by the engine's
    # decode entry so the C++ path matches the Python server byte-for-byte).
    ints(
        **{
            "audio_token_id": 100352,
            "pad_token_id": 100256,
            "bos_token_id": 100257,
            "eos_token_id": 100257,
            "max_new_tokens": 200,
        }
    )
    floats(**{"chunk_seconds": 30.0})
    strings(**{"default_instruction": DEFAULT_TASK_PROMPT})


# Chat-template prompt layout, empirically captured from the HF processor under
# the reference environment (transformers 5.15; the repo's main .venv). The
# rendered prompt is "USER: <|audio|><task>\n ASSISTANT:" and the processor
# expands the single <|audio|> to N copies. GPT-2 BPE never merges across
# special tokens, so prefix + [audio]*N + suffix is exact for every N (verified
# for N=3 under the reference tokenizer). NOTE: transformers 4.x tokenizes the
# ".\n" boundary differently (merges to token 627) -- do NOT recompute these
# arrays under an older transformers.
PROMPT_PREFIX = [6584, 25, 220]  # USER, :, Ġ
PROMPT_SUFFIX = [
    1485, 3191, 279, 8982, 449, 6300, 62603, 323, 6864, 2065,  # task prompt words
    13, 198, 36660, 3931, 2891, 25,                            # . Ċ ĠASS IST ANT :
]


def prompt_layout(w: gguf.GGUFWriter) -> None:
    V = gguf.GGUFValueType
    w.add_key_value("granite.prompt_prefix", PROMPT_PREFIX, V.ARRAY, V.INT32)
    w.add_key_value("granite.prompt_suffix", PROMPT_SUFFIX, V.ARRAY, V.INT32)
    print(f"prompt: prefix={PROMPT_PREFIX} suffix={PROMPT_SUFFIX}")


def tokenizer(w: gguf.GGUFWriter, snapshot: Path) -> None:
    data = json.loads((snapshot / "tokenizer.json").read_text())
    vocab = data["model"]["vocab"]
    size = 100353
    tokens = [None] * size
    for token, i in vocab.items():
        tokens[i] = token
    special = set()
    for item in data.get("added_tokens", []):
        tokens[item["id"]] = item["content"]
        if item.get("special", False):
            special.add(item["id"])
    # The granite vocab fills all 100353 slots; keep the defensive fill anyway.
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
    w.add_bos_token_id(100257)
    w.add_eos_token_id(100257)
    w.add_pad_token_id(100256)


def frontend(snapshot: Path) -> tuple[np.ndarray, np.ndarray]:
    """Mel filterbank (80 x 257) + Hann window (400 -> padded to 512).

    Uses precisely the torchaudio MelSpectrogram the GraniteSpeech feature
    extractor holds (htk scale, no norm, win_length 400 centered inside the
    512-point frame by torch.stft), so the constants match the eager reference
    exactly. Stored freq-major [n_fft/2+1, n_mels] -- the layout the shared C++
    mel frontend expects.
    """
    import torchaudio

    melspec = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=512, win_length=400, hop_length=160, n_mels=80
    )
    # torchaudio exposes the filterbank via the inner MelScale module ([257, 80]
    # freq-major); instantiate it so the values are exactly the ones the
    # feature extractor's forward uses.
    mel = np.ascontiguousarray(melspec.mel_scale.fb.numpy(), dtype=np.float32)  # [257, 80]
    window = torch.hann_window(400)  # periodic, matching torch.stft's window
    pad = (512 - 400) // 2
    window = torch.nn.functional.pad(window, (pad, 512 - 400 - pad))
    return mel, window.numpy().astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("models/granite-speech-4.1-2b-bf16-exact.gguf"),
    )
    args = ap.parse_args()

    index = json.loads((args.snapshot / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    w = gguf.GGUFWriter(args.output, "granite", use_temp_file=True)
    add_metadata(w)
    tokenizer(w, args.snapshot)
    prompt_layout(w)

    # Resolve every shard file once.
    shards = {s: args.snapshot / s for s in sorted(set(weight_map.values()))}

    learned = 0
    dtypes = set()
    skipped = []
    dists = attention_dists()
    # Stream tensors shard-by-shard so each file opens once.
    for shard_name, shard_path in shards.items():
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for source in f.keys():
                if source.endswith("attn.rel_pos_emb.weight"):
                    # Precompute this layer's Shaw bias: an exact embedding
                    # gather over the deterministic dists table, matching the
                    # reference's per-forward lookup bit-for-bit. The raw
                    # rel_pos_emb table is consumed here, not stored.
                    t = f.get_tensor(source)
                    if t.dtype is not torch.bfloat16:
                        raise TypeError(f"{source}: expected BF16, found {t.dtype}")
                    layer = int(source.split(".")[2])
                    bias = t[dists].contiguous()  # (200, 200, 128) bf16
                    a = np.ascontiguousarray(bias.view(torch.uint16).numpy())
                    w.add_tensor(
                        f"enc.blk.{layer}.rel_pos_bias", a,
                        raw_shape=a.shape, raw_dtype=gguf.GGMLQuantizationType.BF16,
                    )
                    learned += 1
                    skipped.append(source)
                    continue
                target = gguf_name(source)
                if target is None:
                    skipped.append(source)
                    continue
                t = f.get_tensor(source)
                dtypes.add(str(t.dtype))
                if t.dtype is not torch.bfloat16:
                    raise TypeError(f"{source}: expected BF16, found {t.dtype}")
                # conv k=1 weights are stored squeezed to 2D Linear layout; the
                # depthwise [C, 1, K] is stored TRANSPOSED k-major [K, C] so the
                # engine reads each tap as a contiguous row (strided views of
                # the raw [C, K] layout are not expressible in ggml).
                if source.endswith("conv.depth_conv.conv.weight"):
                    t = t.reshape(t.shape[0], t.shape[2]).T.contiguous()  # [C,K] -> [K,C]
                    shape = t.shape
                else:
                    shape = t.shape
                    if len(shape) == 3 and shape[2] == 1:
                        shape = shape[:2]
                # GGUF stores dims innermost-first; keep the BF16 byte stream in
                # checkpoint row-major order (same convention as moss/ark).
                a = np.ascontiguousarray(t.view(torch.uint16).numpy()).reshape(shape)
                w.add_tensor(
                    target, a, raw_shape=shape, raw_dtype=gguf.GGMLQuantizationType.BF16
                )
                learned += 1

    mel, window = frontend(args.snapshot)
    w.add_tensor("audio.mel_filters", np.ascontiguousarray(mel, dtype=np.float32))
    w.add_tensor("audio.mel_window", np.ascontiguousarray(window, dtype=np.float32))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(
        f"wrote {args.output}: {learned + 2} tensors ({learned} learned), "
        f"skipped {len(skipped)} (rel_pos_emb consumed / unused), "
        f"source dtypes={sorted(dtypes)}"
    )
    if skipped:
        print(f"skipped: {skipped}")


if __name__ == "__main__":
    main()
