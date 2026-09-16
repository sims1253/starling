#!/usr/bin/env python3
"""Convert the pinned ARK-ASR-0.6B safetensors checkpoint to a Starling GGUF.

ARK-ASR-0.6B (Audio8/ARK-ASR-0.6B) is the distilled sibling of ARK-ASR-3B:
byte-identical remote modeling code (same tensor layout, same Whisper-large-v3
encoder: 32 layers, d_model 1280, 20 heads, head_dim 64, RoPE dim 32 base
10000), an MLP adapter that now maps 5120 -> 1792 -> 896, and a
Qwen2.5-0.5B-class decoder trunk (24 layers, d896, 14 query / 2 KV GQA,
head_dim 64, SwiGLU intermediate 4864, RMSNorm eps 1e-6, RoPE theta 1e6,
tied embeddings). Special-token ids and the chat template are identical to
the 3B; only the tokenizer is larger (vocab 163958 — the Qwen base 151643
plus 12315 added tokens, most of them audio-codec tokens ASR never emits;
the base vocab is byte-identical, so the baked instruction token ids carry
over unchanged).

The tensor-name map, mel frontend, and encoder RoPE tables are shared with
scripts/convert_ark_gguf.py (imported, not duplicated); this file owns the
0.6B metadata, the 163958-entry tokenizer, and the unsharded checkpoint
layout (a single model.safetensors, no index). All weights are stored BF16
(the checkpoint dtype).
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
import gguf

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from convert_ark_gguf import frontend, gguf_name, rope_tables  # noqa: E402

REVISION = "45776b56d58cdfb2e2eb632f7e110f38684633e0"
DEFAULT_SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub/models--Audio8--ARK-ASR-0.6B/snapshots"
    / REVISION
)
VOCAB_SIZE = 163958


def add_metadata(w: gguf.GGUFWriter) -> None:
    V = gguf.GGUFValueType
    w.add_key_value("starling.format_version", 1, V.UINT32)
    w.add_string("starling.numeric_profile", "bf16_exact")
    # Same engine and architecture family as the 3B GGUF.
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

    # Frontend: identical to the 3B (Whisper mel, n_fft=400, hop=160, 128 bins).
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

    # Whisper encoder: identical to the 3B.
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

    # Adapter: Linear(5120 -> 1792) GELU Linear(1792 -> 896).
    ints(
        **{
            "adapter.input_size": 5120,
            "adapter.hidden_size": 1792,
            "adapter.output_size": 896,
            "adapter.merge_factor": 4,
        }
    )
    strings(**{"adapter.act": "gelu"})

    # LLM (Qwen2.5-0.5B-class decoder).
    ints(
        **{
            "llm.hidden_size": 896,
            "llm.num_layers": 24,
            "llm.num_heads": 14,
            "llm.num_kv_heads": 2,
            "llm.head_dim": 64,
            "llm.intermediate_size": 4864,
            "llm.vocab_size": VOCAB_SIZE,
            "llm.max_position_embeddings": 32768,
            "llm.max_cache_len": 4096,
        }
    )
    floats(**{"llm.rope_theta": 1000000.0, "llm.rms_norm_eps": 1e-6})
    strings(**{"llm.rope_scaling": "none"})
    w.add_key_value("ark.llm.tied_embeddings", True, V.BOOL)
    w.add_key_value("ark.llm.has_qk_norm", False, V.BOOL)

    # Token ids: identical to the 3B (verified against the 0.6B tokenizer;
    # the base vocab is byte-identical).
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
    # eager reference exactly. prefix = <|user|><|begin_of_audio|>;
    # suffix = <|end_of_audio|> + instruction + <|assistant|> (same ids as 3B).
    instruction_tokens = [3167, 3114, 279, 7699, 311, 1467, 13]
    w.add_key_value("ark.prompt_prefix", [151665, 151666], V.ARRAY, V.INT32)
    w.add_key_value(
        "ark.prompt_suffix", [151667] + instruction_tokens + [151668], V.ARRAY, V.INT32
    )


def tokenizer(w: gguf.GGUFWriter, snapshot: Path) -> None:
    data = json.loads((snapshot / "tokenizer.json").read_text())
    vocab = data["model"]["vocab"]
    tokens = [None] * VOCAB_SIZE
    for token, i in vocab.items():
        tokens[i] = token
    special = set()
    for item in data.get("added_tokens", []):
        tokens[item["id"]] = item["content"]
        if item.get("special", False):
            special.add(item["id"])
    # Generation suppression list (the model card's build_bad_words_ids
    # constraint: every special id and every <...>-style added token must be
    # banned from greedy picks so the decoder cannot spiral into chat-template
    # or audio-codec ids; only the EOS id stays emittable). Baked as an INT32
    # array for the engine's logits penalty; returned for the metadata write.
    ban = sorted(
        item["id"]
        for item in data.get("added_tokens", [])
        if item["id"] != 151645
        and (item.get("special", False)
             or (item["content"].startswith("<") and item["content"].endswith(">")))
    )
    assert len(ban) == 12314 and ban[0] == 151643 and ban[-1] == VOCAB_SIZE - 1, (
        f"unexpected 0.6B suppression list: n={len(ban)} "
        f"ends=({ban[0] if ban else None}, {ban[-1] if ban else None})"
    )
    # Any still-undefined slots get explicit placeholders.
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
    w.add_key_value("ark.bad_words_ids", ban, gguf.GGUFValueType.ARRAY,
                    gguf.GGUFValueType.INT32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("models/ark-asr-0.6b-bf16-exact.gguf"),
    )
    args = ap.parse_args()

    # The 0.6B checkpoint ships unsharded: one model.safetensors, no index.
    shards = sorted(args.snapshot.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no safetensors under {args.snapshot}")
    if len(shards) > 1 and (args.snapshot / "model.safetensors.index.json").exists():
        index = json.loads((args.snapshot / "model.safetensors.index.json").read_text())
        shards = sorted(
            {args.snapshot / s for s in set(index["weight_map"].values())}
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    w = gguf.GGUFWriter(args.output, "ark", use_temp_file=True)
    add_metadata(w)
    tokenizer(w, args.snapshot)

    learned = 0
    dtypes = set()
    skipped = []
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

    mel, window = frontend(args.snapshot)
    w.add_tensor("audio.mel_filters", np.ascontiguousarray(mel.T, dtype=np.float32))
    w.add_tensor("audio.mel_window", np.ascontiguousarray(window, dtype=np.float32))
    cos, sin = rope_tables()
    w.add_tensor("enc.rope_cos", np.ascontiguousarray(cos, dtype=np.float32))
    w.add_tensor("enc.rope_sin", np.ascontiguousarray(sin, dtype=np.float32))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(
        f"wrote {args.output}: {learned + 4} tensors ({learned} learned), "
        f"skipped {len(skipped)} (tied/unused), source dtypes={sorted(dtypes)}"
    )
    if skipped:
        print(f"skipped: {skipped}")


if __name__ == "__main__":
    main()
