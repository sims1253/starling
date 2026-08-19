#!/usr/bin/env python3
"""Convert the superwhisper/s1-mini safetensors checkpoint to a Starling GGUF.

S1-mini is a text-to-text normalizer: a plain ``Qwen3ForCausalLM`` (0.6B, 28
layers, hidden 1024, GQA 16Q/8KV, head_dim 128, tied embeddings, per-head
q/k norm) with no audio front-end. This converter mirrors
scripts/convert_qwen3_gguf.py minus every audio-side piece: an explicit
tensor-name map (an unknown checkpoint tensor fails conversion), the config
baked as ``s1.*`` KV metadata, the byte-level BPE tokenizer tables (merges
included — the C++ side encodes, not just decodes), and the chat-template
layout baked as prefix/suffix token-id arrays around the runtime-encoded user
content (system prompt + control line + transcript).

Prompt shape (trained contract, model card):

    <|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n
    <|im_start|>user\n[Styling: s] [Structure: t] [Context: c]\n{transcript}<|im_end|>\n
    <|im_start|>assistant\n<think>\n\n</think>\n\n

``prompt_prefix`` ends at the ``user\n`` newline; ``prompt_suffix`` starts at
the ``<|im_end|>`` after the transcript. Both boundaries are pre-token splits
(the GPT-2 regex puts ``\n`` in its own pre-token class and special tokens
split first), so prefix + encode(user_content) + suffix is token-identical to
encoding the full rendered string — verified against the HF tokenizer for all
fixture transcripts + control combinations at conversion time.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
import gguf

REVISION = "65f84bcda1d13df582c4a8443c1c5aa53c0c66db"
DEFAULT_SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub/models--superwhisper--s1-mini/snapshots"
    / REVISION
)

N_LAYERS = 28
VOCAB_SIZE = 151936

SYSTEM_PROMPT = (
    "You are a text normalizer for speech-to-text transcripts. The input begins "
    "with a control line specifying the styling, structure, and context settings; "
    "clean the transcript to match those settings and output only the cleaned text."
)
STYLING_VALUES = ["casual", "semi-casual", "semi-formal", "formal"]
STRUCTURE_VALUES = ["prose", "lists"]
CONTEXT_VALUES = ["general", "email"]


# ---------------------------------------------------------------------------
# Tensor-name mapping: HF checkpoint name -> Starling GGUF name.
#
# LLM decoder (Qwen3 trunk: bias-free projections, q/k norm, TIED lm_head):
#   llm.blk.N.{attn_norm,attn.q/k/v/o,attn.q_norm,attn.k_norm,ffn_norm,
#              ffn.gate/up/down}
#   llm.embed / llm.final_norm      (lm_head == llm.embed; the checkpoint also
#                                    carries a materialized lm_head copy that
#                                    must be bit-identical — verified, skipped)
def gguf_name(name: str) -> str | None:
    if name == "model.embed_tokens.weight": return "llm.embed.weight"
    if name == "model.norm.weight":         return "llm.final_norm.weight"
    if name == "lm_head.weight":            return None  # tied; verified below
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
            "self_attn.q_norm": "attn.q_norm",
            "self_attn.k_norm": "attn.k_norm",
            "mlp.gate_proj": "ffn.gate",
            "mlp.up_proj": "ffn.up",
            "mlp.down_proj": "ffn.down",
        }
        for old, new in repl.items():
            if rest.startswith(old + "."):
                return f"llm.blk.{i}.{new}." + rest[len(old) + 1:]
    raise KeyError(f"no s1 GGUF mapping for {name!r}")


def add_metadata(w: gguf.GGUFWriter) -> None:
    V = gguf.GGUFValueType
    w.add_key_value("starling.format_version", 1, V.UINT32)
    w.add_string("starling.numeric_profile", "bf16_exact")
    w.add_string("general.architecture", "s1")
    w.add_string("general.name", "S1-mini by Superwhisper")

    def ints(**xs):
        for k, v in xs.items():
            w.add_key_value("s1." + k, v, V.UINT32)

    def floats(**xs):
        for k, v in xs.items():
            w.add_key_value("s1." + k, float(v), V.FLOAT32)

    def strings(**xs):
        for k, v in xs.items():
            w.add_string("s1." + k, v)

    # LLM (Qwen3 trunk, stock numerics: all multipliers 1, default scale).
    ints(
        **{
            "llm.hidden_size": 1024,
            "llm.num_layers": 28,
            "llm.num_heads": 16,
            "llm.num_kv_heads": 8,
            "llm.head_dim": 128,
            "llm.intermediate_size": 3072,
            "llm.vocab_size": 151936,
            "llm.max_position_embeddings": 40960,
            "llm.max_cache_len": 4096,
        }
    )
    floats(**{"llm.rope_theta": 1000000.0, "llm.rms_norm_eps": 1e-6})
    w.add_key_value("s1.llm.tied_embeddings", True, V.BOOL)
    w.add_key_value("s1.llm.has_qk_norm", True, V.BOOL)

    # Token ids + generation (eos list mirrors generation_config.json:
    # stop on <|im_end|> 151645 OR <|endoftext|> 151643; pad is <|endoftext|>).
    ints(
        **{
            "eos_token_id": 151645,
            "eos2_token_id": 151643,
            "pad_token_id": 151643,
            "max_input_tokens": 1000,
        }
    )
    floats(
        **{
            "max_new_tokens_input_factor": 1.3,
            "max_new_tokens_fixed": 32.0,
        }
    )

    # The trained control-space values the engine validates against (the card:
    # values outside the trained sets make the model hallucinate).
    strings(styling_values="|".join(STYLING_VALUES),
            structure_values="|".join(STRUCTURE_VALUES),
            context_values="|".join(CONTEXT_VALUES))


# Chat-template layout, captured from the shipped tokenizer's apply_chat_template
# with enable_thinking=False. prefix = everything through "user\n"; suffix =
# the assistant opening including the empty think block.
def _capture_prompt_layout(snapshot: Path) -> tuple[list[int], list[int]]:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "s1_config", Path(__file__).resolve().parents[1] / "src" / "starling" / "s1" / "config.py"
    )
    cfg = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(cfg)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(snapshot))
    probe = "\x00TRANSCRIPT\x00"  # never merges across special boundaries
    text = tok.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": probe},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    pre, post = text.split(probe)
    prefix = tok(pre, add_special_tokens=False).input_ids
    suffix = tok(post, add_special_tokens=False).input_ids
    full = tok(text, add_special_tokens=False).input_ids
    assert full[: len(prefix)] == prefix and full[-len(suffix):] == suffix, (
        "template boundary is not a clean pre-token split"
    )
    # The user content the engine builds (control line + transcript) must be
    # encodable standalone: prefix ends with the "user\n" newline token.
    assert pre.endswith("user\n"), repr(pre[-12:])
    assert post.startswith("<|im_end|>")
    return prefix, suffix


def prompt_layout(w: gguf.GGUFWriter, prefix: list[int], suffix: list[int]) -> None:
    V = gguf.GGUFValueType
    w.add_key_value("s1.prompt_prefix", prefix, V.ARRAY, V.INT32)
    w.add_key_value("s1.prompt_suffix", suffix, V.ARRAY, V.INT32)
    print(f"prompt: prefix({len(prefix)})={prefix}")
    print(f"prompt: suffix({len(suffix)})={suffix}")


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
    # The C++ side ENCODES (BPE with merge ranks), not just decodes.
    w.add_token_merges(merges)
    w.add_eos_token_id(151645)
    w.add_pad_token_id(151643)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("models/s1-mini-bf16-exact.gguf"),
    )
    args = ap.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    w = gguf.GGUFWriter(args.output, "s1", use_temp_file=True)
    add_metadata(w)
    tokenizer(w, args.snapshot)
    prefix, suffix = _capture_prompt_layout(args.snapshot)
    prompt_layout(w, prefix, suffix)

    learned = 0
    dtypes = set()
    with safe_open(args.snapshot / "model.safetensors", framework="pt", device="cpu") as f:
        # lm_head is tied to llm.embed.weight (tie_word_embeddings=true); the
        # checkpoint stores a materialized copy that must be bit-identical.
        if "lm_head.weight" in f.keys():
            lm = f.get_tensor("lm_head.weight")
            emb = f.get_tensor("model.embed_tokens.weight")
            if not torch.equal(lm, emb):
                raise ValueError("lm_head.weight differs from embed_tokens.weight (not tied?)")
        for source in sorted(f.keys()):
            target = gguf_name(source)
            if target is None:
                continue
            t = f.get_tensor(source)
            dtypes.add(str(t.dtype))
            if t.dtype is not torch.bfloat16:
                raise TypeError(f"{source}: expected BF16, found {t.dtype}")
            shape = tuple(t.shape)
            a = np.ascontiguousarray(t.view(torch.uint16).numpy()).reshape(shape)
            w.add_tensor(
                target, a, raw_shape=shape, raw_dtype=gguf.GGMLQuantizationType.BF16
            )
            learned += 1

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {args.output}: {learned} tensors, source dtypes={sorted(dtypes)}")


if __name__ == "__main__":
    main()
