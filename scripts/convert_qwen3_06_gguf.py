#!/usr/bin/env python3
"""Convert the pinned Qwen3-ASR-0.6B-hf safetensors checkpoint to a Starling GGUF.

Qwen/Qwen3-ASR-0.6B-hf is the smaller sibling of Qwen3-ASR-1.7B-hf: the same
windowed-attention conv encoder + 2-layer MLP projector + Qwen3 decoder
topology, with a narrower audio tower (18 layers, d896, 14 heads, FFN 3584,
projector output 1024) and a 1024-wide decoder trunk with a 3072-wide SwiGLU.
The trunk KEEPS the 1.7B's 16Q/8KV GQA at head_dim 128, so the attention width
(2048) is wider than the trunk (1024) — the shared decode stack supports this
(since voxtral), and the engine loader validates the widths against the actual
tensor shapes instead of assuming hidden == heads*head_dim.

The tensor-name map, mel frontend, tokenizer and chat-template prompt layout
are shared with scripts/convert_qwen3_gguf.py (imported, not duplicated): the
hub tokenizer.json / tokenizer_config.json / chat_template.jinja /
processor_config.json are byte-identical between the two checkpoints (verified
against revisions bcd2b5b7 / 7f1569a4), so the baked prompt prefix/suffix
token ids carry over unchanged. This file owns the 0.6B metadata — read from
the snapshot's own config.json so the GGUF cannot drift from the checkpoint —
and the (13, 896) sinusoidal position table. All weights are stored BF16 (the
checkpoint dtype).

Engine: the GGUF loads under the shared qwen3 engine (registry slug
``qwen3_06``); every dim is qwen3.* GGUF metadata validated at load.
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
from convert_qwen3_gguf import (  # noqa: E402
    PROMPT_SUFFIX,
    PROMPT_PREFIX,
    frontend,
    gguf_name,
    positional_embedding,
    tokenizer,
)
from make_qwen3_golden import MAX_CACHE_LEN  # noqa: E402  # one shared KV-cache policy with the golden captures

REVISION = "7f1569a48a89f3e3f4dc3a5c9d28bddd903bc76c"
DEFAULT_SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub/models--Qwen--Qwen3-ASR-0.6B-hf/snapshots"
    / REVISION
)

VOCAB_SIZE = 151936  # identical tokenizer to the 1.7B (asserted against the snapshot)


def load_config(snapshot: Path) -> tuple[dict, dict]:
    """Parse config.json into the flat dim dict the metadata writer needs,
    returning ``(dims, raw_cfg)`` so callers reuse the parsed config instead
    of re-reading the file.

    head_dim of the audio tower is not explicit in the HF config; it is
    d_model // encoder_attention_heads (896 // 14 = 64), exactly how
    Qwen3ASREncoderConfig consumers derive it.
    """
    cfg = json.loads((snapshot / "config.json").read_text())
    audio, text = cfg["audio_config"], cfg["text_config"]
    d_model = int(audio["d_model"])
    heads = int(audio["encoder_attention_heads"])
    dims = {
        # audio tower
        "enc.hidden": d_model,
        "enc.layers": int(audio["encoder_layers"]),
        "enc.heads": heads,
        "enc.head_dim": d_model // heads,
        "enc.ffn_dim": int(audio["encoder_ffn_dim"]),
        "enc.downsample_hidden": int(audio["downsample_hidden_size"]),
        "enc.n_window": int(audio["n_window"]),
        "enc.n_window_infer": int(audio["n_window_infer"]),
        "enc.max_pos_emb": int(audio["max_position_embeddings"]),
        "enc.output_dim": int(audio["output_dim"]),
        "enc.n_mel": int(audio["num_mel_bins"]),
        # projector: Linear(d_model -> d_model) + GELU + Linear(d_model -> output)
        "proj.hidden": d_model,
        "proj.output_dim": int(audio["output_dim"]),
        # text trunk
        "llm.hidden_size": int(text["hidden_size"]),
        "llm.num_layers": int(text["num_hidden_layers"]),
        "llm.num_heads": int(text["num_attention_heads"]),
        "llm.num_kv_heads": int(text["num_key_value_heads"]),
        "llm.head_dim": int(text["head_dim"]),
        "llm.intermediate_size": int(text["intermediate_size"]),
        "llm.vocab_size": int(text["vocab_size"]),
        "llm.max_position_embeddings": int(text["max_position_embeddings"]),
        "llm.rope_theta": float(text["rope_parameters"]["rope_theta"]),
        "llm.rms_norm_eps": float(text["rms_norm_eps"]),
        # tie_word_embeddings lives at the TOP level of config.json (both
        # pinned revisions carry true); read it so the GGUF cannot drift if a
        # future revision unties the head.
        "llm.tied_embeddings": bool(
            cfg.get("tie_word_embeddings", text.get("tie_word_embeddings", True))
        ),
    }
    return dims, cfg


def add_metadata(w: gguf.GGUFWriter, d: dict) -> None:
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

    # Frontend: byte-identical Qwen3ASRFeatureExtractor to the 1.7B (the hub
    # processor_config.json is identical; see convert_qwen3_gguf.py for the
    # exact stft/mel/normalization policy).
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

    # Windowed-attention conv encoder + projector + Qwen3 trunk, all from the
    # checkpoint's own config.json (dims above).
    ints(
        **{
            "enc.n_mel": d["enc.n_mel"],
            "enc.hidden": d["enc.hidden"],
            "enc.layers": d["enc.layers"],
            "enc.heads": d["enc.heads"],
            "enc.head_dim": d["enc.head_dim"],
            "enc.ffn_dim": d["enc.ffn_dim"],
            "enc.downsample_hidden": d["enc.downsample_hidden"],
            "enc.n_window": d["enc.n_window"],
            "enc.n_window_infer": d["enc.n_window_infer"],
            "enc.max_pos_emb": d["enc.max_pos_emb"],
            "enc.output_dim": d["enc.output_dim"],
        }
    )
    w.add_key_value("qwen3.enc.layer_norm_eps", 1e-5, V.FLOAT32)
    ints(**{"proj.hidden": d["proj.hidden"], "proj.output_dim": d["proj.output_dim"]})
    ints(
        **{
            "llm.hidden_size": d["llm.hidden_size"],
            "llm.num_layers": d["llm.num_layers"],
            "llm.num_heads": d["llm.num_heads"],
            "llm.num_kv_heads": d["llm.num_kv_heads"],
            "llm.head_dim": d["llm.head_dim"],
            "llm.intermediate_size": d["llm.intermediate_size"],
            "llm.vocab_size": d["llm.vocab_size"],
            "llm.max_position_embeddings": d["llm.max_position_embeddings"],
            "llm.max_cache_len": MAX_CACHE_LEN,
        }
    )
    floats(**{"llm.rope_theta": d["llm.rope_theta"], "llm.rms_norm_eps": d["llm.rms_norm_eps"]})
    w.add_key_value("qwen3.llm.tied_embeddings", d["llm.tied_embeddings"], V.BOOL)
    w.add_key_value("qwen3.llm.has_qk_norm", True, V.BOOL)

    # Token ids + generation + the serve chunk policy: identical to the 1.7B
    # (the generation_config/tokenizer_config are byte-identical; eos is the
    # serving path's <|im_end|>).
    ints(
        **{
            "audio_token_id": 151676,
            "pad_token_id": 151645,
            "eos_token_id": 151645,
            "max_new_tokens": 200,
        }
    )
    floats(**{"chunk_seconds": 30.0})


def _fail(msg: str) -> None:
    """Raise instead of `assert` so `python -O` cannot strip the external
    snapshot cross-checks."""
    raise SystemExit(f"convert_qwen3_06: {msg}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("models/qwen3-asr-0.6b-bf16-exact.gguf"),
    )
    args = ap.parse_args()

    dims, top_cfg = load_config(args.snapshot)
    # Cross-check the shared tokenizer before baking it: the prompt layout and
    # the byte-level BPE carry over from the 1.7B only if the vocab matches.
    # Real raises (not asserts) so `python -O` cannot strip them.
    tokenizer_json = json.loads((args.snapshot / "tokenizer.json").read_text())
    tok_vocab = tokenizer_json["model"]["vocab"]
    vocab_max = max(tok_vocab.values()) if tok_vocab else -1
    if not tok_vocab or len(tok_vocab) > VOCAB_SIZE or vocab_max >= VOCAB_SIZE:
        _fail(
            f"tokenizer vocab ({len(tok_vocab)} entries, max id "
            f"{vocab_max}) does not fit the shared {VOCAB_SIZE}-entry "
            "vocab — the shared 1.7B tokenizer/prompt layout cannot be reused"
        )
    if dims["llm.vocab_size"] != VOCAB_SIZE:
        _fail(f"config.json vocab_size {dims['llm.vocab_size']} != shared {VOCAB_SIZE}")

    # Cross-check the baked special-token ids and the shared prompt layout
    # against the snapshot's own configs (same drift doctrine as the vocab
    # check above): a future revision that renumbers its special tokens must
    # fail conversion instead of silently baking stale ids. Absent keys fall
    # through to the baked value; a present-but-different value fails.
    try:
        gen_cfg = json.loads((args.snapshot / "generation_config.json").read_text())
    except FileNotFoundError:
        gen_cfg = {}
    added_ids = {item["id"] for item in tokenizer_json.get("added_tokens", [])}
    if int(top_cfg.get("audio_token_id", 151676)) != 151676:
        _fail(f"config.json audio_token_id {top_cfg['audio_token_id']} != baked 151676")
    for key, want in (("eos_token_id", 151645), ("pad_token_id", 151645)):
        got = gen_cfg.get(key, top_cfg.get(key, want))
        if isinstance(got, (list, tuple)):  # some checkpoints list several ids
            got = got[0] if got else want
        if int(got) != want:
            _fail(f"snapshot {key} {got} != baked {want}")
    baked_ids = [151676, 151645, *PROMPT_PREFIX, *PROMPT_SUFFIX]
    valid_ids = set(tok_vocab.values()) | added_ids
    missing = sorted(i for i in baked_ids if i not in valid_ids)
    if missing:
        _fail(
            f"prompt/special token ids {missing} are absent from the snapshot's "
            "own tokenizer — the shared 1.7B prompt layout cannot be reused"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    w = gguf.GGUFWriter(args.output, "qwen3", use_temp_file=True)
    add_metadata(w, dims)
    tokenizer(w, args.snapshot)
    w.add_key_value("qwen3.prompt_prefix", PROMPT_PREFIX, gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.INT32)
    w.add_key_value("qwen3.prompt_suffix", PROMPT_SUFFIX, gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.INT32)
    print(f"prompt: prefix={PROMPT_PREFIX} suffix={PROMPT_SUFFIX}")

    learned = 0
    with safe_open(args.snapshot / "model.safetensors", framework="pt", device="cpu") as f:
        for source in sorted(f.keys()):
            target = gguf_name(source)
            t = f.get_tensor(source)
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

    pos = positional_embedding(dims["enc.max_pos_emb"], dims["enc.hidden"])
    a = np.ascontiguousarray(pos.view(torch.uint16).numpy())
    w.add_tensor(
        "enc.pos_embed", a, raw_shape=(dims["enc.max_pos_emb"], dims["enc.hidden"]),
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
        f"pos_embed/mel/window), every source tensor torch.bfloat16"
    )
    print(
        f"dims: enc(hidden={dims['enc.hidden']}, layers={dims['enc.layers']}, "
        f"heads={dims['enc.heads']}x{dims['enc.head_dim']}, ffn={dims['enc.ffn_dim']}, "
        f"out={dims['enc.output_dim']}) llm(hidden={dims['llm.hidden_size']}, "
        f"layers={dims['llm.num_layers']}, {dims['llm.num_heads']}Q/"
        f"{dims['llm.num_kv_heads']}KV x{dims['llm.head_dim']}, "
        f"ffn={dims['llm.intermediate_size']})"
    )


if __name__ == "__main__":
    main()
