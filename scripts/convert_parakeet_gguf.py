#!/usr/bin/env python3
"""Convert nvidia/parakeet-tdt-0.6b-v3 to Starling GGUF.

Parakeet-tdt is a FastConformer + TDT (token-and-duration transducer) model.
Unlike the Whisper-mel models (moss, ark, higgs), parakeet has its own mel
frontend (NeMo AudioToMelSpectrogramPreprocessor) and an RNN-T/TDT decoder.

This converter:
  1. Loads the model via ``transformers.AutoModelForTDT`` (same path as the
     starling pipeline), extracts ``state_dict()`` + the mel filterbank/window
     from the processor's feature extractor, and the SentencePiece tokenizer
     pieces.
  2. Remaps every tensor to the GGUF tensor names that the C++ engine
     (``cpp/parakeet/loader.cpp``) reads — the verbatim NeMo-style names.
  3. Writes all parakeet.* KV metadata (config, durations, tokenizer pieces).

The GGUF tensor names are the authoritative contract between this converter and
the C++ loader; they match what ``parakeet.cpp`` reads. See
``cpp/parakeet/config.hpp`` for the config struct and ``cpp/parakeet/loader.cpp``
for the KV key strings.

Usage::

    uv run python scripts/convert_parakeet_gguf.py \\
        --model-id nvidia/parakeet-tdt-0.6b-v3 \\
        --output models/parakeet-tdt-0.6b-v3-bf16-exact.gguf
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    import gguf
except ImportError:
    sys.exit("error: gguf not installed. Run: uv pip install gguf")

REPO = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"
DEFAULT_OUTPUT = REPO / "models" / "parakeet-tdt-0.6b-v3-bf16-exact.gguf"

# ---------------------------------------------------------------------------
# Tensor name mapping: transformers state_dict → GGUF tensor names
# ---------------------------------------------------------------------------
# The C++ loader expects NeMo-style names (encoder.pre_encode.*, encoder.layers.N.*,
# decoder.prediction.*, joint.*). The transformers ParakeetForTDT model wraps the
# NeMo model, so the state_dict keys have a "model." prefix and slightly different
# nesting. This table handles the most common patterns; unmapped keys raise.

def map_key(src: str) -> str | None:
    """Map a transformers state_dict key to the GGUF tensor name.

    Returns None for keys we should skip (non-weight buffers, etc.).
    Raises KeyError for keys that look like weights but have no mapping.
    """
    # Strip common prefixes.
    s = src
    if s.startswith("model."):
        s = s[len("model."):]

    # ---- subsampling (ConvSubsampler) ----
    # transformers: encoder.pre_encoder.conv_layers.model.{i}.weight
    #          or: encoder.pre_encoder.conv.model.{i}.weight
    # GGUF: encoder.pre_encode.conv.{i}.weight
    for pfx in ("encoder.pre_encoder.conv_layers.model.",
                "encoder.pre_encoder.conv.model.",
                "encoder.pre_encoder.conv."):
        if s.startswith(pfx):
            tail = s[len(pfx):]
            return "encoder.pre_encode.conv." + tail

    # Subsampling output projection (Linear).
    # transformers: encoder.pre_encoder.out.weight / .bias
    #               encoder.subsampling.linear.weight / .bias
    if s.startswith("encoder.pre_encoder.out."):
        return "encoder.pre_encode.out." + s[len("encoder.pre_encoder.out."):]
    if s.startswith("encoder.subsampling.linear."):
        return "encoder.pre_encode.out." + s[len("encoder.subsampling.linear."):]
    # transformers: encoder.subsampling.layers.{i}.{weight,bias}
    if s.startswith("encoder.subsampling.layers."):
        return "encoder.pre_encode.conv." + s[len("encoder.subsampling.layers."):]

    # ---- conformer layers ----
    # transformers: encoder.conformer_encoder.layers.{i}.{...}
    #          or: encoder.encoder.layers.{i}.{...}
    # GGUF: encoder.layers.{i}.{...}
    for pfx in ("encoder.conformer_encoder.layers.",
                "encoder.encoder.layers.",
                "encoder.layers."):
        if s.startswith(pfx):
            tail = s[len(pfx):]
            # The HF rel-pos attention and conv BatchNorm names inside each
            # conformer layer, translated to the NeMo names (in parentheses)
            # the C++ engine reads:
            #   q_proj        -> linear_q        k_proj -> linear_k
            #   v_proj        -> linear_v        o_proj -> linear_out
            #   relative_k_proj -> linear_pos
            #   bias_u/bias_v -> pos_bias_u/pos_bias_v
            #   conv.norm.*   -> conv.batch_norm.*
            for old, new in (
                ("self_attn.q_proj.", "self_attn.linear_q."),
                ("self_attn.k_proj.", "self_attn.linear_k."),
                ("self_attn.v_proj.", "self_attn.linear_v."),
                ("self_attn.o_proj.", "self_attn.linear_out."),
                ("self_attn.relative_k_proj.", "self_attn.linear_pos."),
                ("self_attn.bias_u", "self_attn.pos_bias_u"),
                ("self_attn.bias_v", "self_attn.pos_bias_v"),
                ("conv.norm.", "conv.batch_norm."),
            ):
                tail = tail.replace(old, new)
            # The conformer layer internal naming matches NeMo for most tensors.
            # Some transformers versions rename modules; handle known renames:
            # feed_forward1 → feed_forward1, self_attn → self_attn, etc.
            # The NeMo and C++ names are identical for the conformer internals.
            return "encoder.layers." + tail

    # ---- decoder (prediction network) ----
    # transformers: decoder.prediction.embed.weight
    #          or: decoder.blstm.embed.weight  (older naming)
    # GGUF: decoder.prediction.embed.weight
    if s.startswith("decoder.prediction."):
        tail = s[len("decoder.prediction."):]
        # The LSTM weights in transformers are:
        #   dec_rnn.weight_ih_l0, dec_rnn.weight_hh_l0, etc.
        # C++ expects: dec_rnn.lstm.weight_ih_l0, etc.
        if tail.startswith("dec_rnn.") and ".lstm." not in tail:
            tail = "dec_rnn.lstm." + tail[len("dec_rnn."):]
        return "decoder.prediction." + tail

    # Also handle the direct naming (no "prediction." infix in some versions).
    if s.startswith("decoder.embed."):
        return "decoder.prediction.embed." + s[len("decoder.embed."):]

    # The flat HF decoder naming (the engine reads the NeMo-style names):
    #   decoder.embedding.weight          -> decoder.prediction.embed.weight
    #   decoder.lstm.weight_ih_l0 ...      -> decoder.prediction.dec_rnn.lstm.*
    #   decoder.decoder_projector.{w,b}    -> joint.pred.{w,b}     (pred->joint proj)
    #   joint.head.{w,b}                   -> joint.joint_net.2.{w,b} (output proj)
    if s.startswith("decoder.embedding."):
        return "decoder.prediction.embed." + s[len("decoder.embedding."):]
    if s.startswith("decoder.lstm."):
        return "decoder.prediction.dec_rnn.lstm." + s[len("decoder.lstm."):]
    if s.startswith("decoder.decoder_projector."):
        return "joint.pred." + s[len("decoder.decoder_projector."):]
    if s.startswith("joint.head."):
        return "joint.joint_net.2." + s[len("joint.head."):]

    # ---- joint network ----
    # transformers: joint.encoder_joint.{i}.weight / .bias  (encoder proj)
    #          or: joint.enc_to_enc_proj.weight / .bias
    # GGUF: joint.enc.weight / .bias
    if s.startswith("joint.encoder_joint.0."):
        return "joint.enc." + s[len("joint.encoder_joint.0."):]
    if s.startswith("joint.encoder_to_joint_network."):
        return "joint.enc." + s[len("joint.encoder_to_joint_network."):]

    # transformers: joint.joint_network.{i}.weight / .bias
    #          or: joint.joint_net.{i}.weight
    # GGUF: joint.joint_net.{i}.weight / .bias
    if s.startswith("joint.joint_network."):
        return "joint.joint_net." + s[len("joint.joint_network."):]
    if s.startswith("joint.joint_net."):
        return s  # already matches

    # transformers: joint.decoder_joint.{i}.weight / .bias  (pred proj)
    #          or: joint.pred_to_joint_network.weight
    # GGUF: joint.pred.weight / .bias
    if s.startswith("joint.decoder_joint.0."):
        return "joint.pred." + s[len("joint.decoder_joint.0."):]
    if s.startswith("joint.pred_to_joint_network."):
        return "joint.pred." + s[len("joint.pred_to_joint_network."):]

    # ---- encoder pooler / projector (the enc_out projection) ----
    # transformers: encoder_projector.weight / encoder_projector.bias
    #          or: encoder.proj.weight / encoder.proj.bias
    # GGUF: joint.enc.weight / joint.enc.bias (the joint does the encoder projection)
    if s.startswith("encoder_projector."):
        return "joint.enc." + s[len("encoder_projector."):]
    if s.startswith("encoder.proj."):
        return "joint.enc." + s[len("encoder.proj."):]

    # ---- known non-weight keys to skip ----
    # Actually, running_mean/var ARE needed for batch_norm. Don't skip them.
    # Only skip num_batches_tracked.
    if s.endswith(".num_batches_tracked"):
        return None

    # If it starts with known prefixes but we couldn't map it, raise.
    if any(s.startswith(p) for p in ("encoder.", "decoder.", "joint.")):
        raise KeyError(f"no GGUF mapping for state_dict key: {src!r}")

    return None


# ---------------------------------------------------------------------------
# Mel frontend extraction
# ---------------------------------------------------------------------------
def extract_mel_filterbank(processor: Any) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Extract the mel filterbank + STFT window from the NeMo preprocessor.

    Returns (fb [n_mels, n_fft//2+1], window [win_length], frontend_params) as float32.
    """
    import librosa

    fe = getattr(processor, "feature_extractor", None)
    if fe is None:
        raise RuntimeError("processor has no feature_extractor")

    sample_rate = int(getattr(fe, "sampling_rate", getattr(fe, "sample_rate", 16000)))
    n_fft = int(getattr(fe, "n_fft", 512))
    n_mels = int(getattr(fe, "feature_size", getattr(fe, "n_mels", 128)))
    win_length = int(getattr(fe, "win_length", getattr(fe, "n_window", 400)))
    hop_length = int(getattr(fe, "hop_length", getattr(fe, "n_stride", 160)))
    fmin = float(getattr(fe, "f_min", getattr(fe, "fmin", 0.0)))
    fmax = float(getattr(fe, "f_max", getattr(fe, "fmax", sample_rate / 2.0)))
    preemph = float(getattr(fe, "preemph", 0.97))
    mag_power = float(getattr(fe, "mag_power", getattr(fe, "power", 2.0)))
    log_zero_guard = float(getattr(fe, "log_zero_guard", 5.9604645e-08))

    cached_fb = getattr(fe, "filter_banks", None)
    if cached_fb is not None:
        fb = np.asarray(cached_fb, dtype=np.float32)
    else:
        # Mel filterbank (librosa 'slaney' normalization, matching NeMo).
        fb = librosa.filters.mel(
            sr=sample_rate, n_fft=n_fft, n_mels=n_mels,
            fmin=fmin, fmax=fmax, htk=False, norm="slaney",
        ).astype(np.float32)

    cached_win = getattr(fe, "window", None)
    if cached_win is not None and not isinstance(cached_win, str):
        window = np.asarray(cached_win, dtype=np.float32)
    else:
        # Hann window (NeMo default).
        window = np.array(
            [0.5 - 0.5 * np.cos(2 * np.pi * n / win_length)
             for n in range(win_length)],
            dtype=np.float32,
        )

    frontend_params = {
        "sample_rate": sample_rate,
        "n_mels": n_mels,
        "n_fft": n_fft,
        "win_length": win_length,
        "hop_length": hop_length,
        "preemph": preemph,
        "mag_power": mag_power,
        "log_zero_guard": log_zero_guard,
    }

    return fb, window, frontend_params


# ---------------------------------------------------------------------------
# KV metadata
# ---------------------------------------------------------------------------
def resolve_config_attr(config: Any, *keys: str) -> Any:
    # Search the config itself first, then its sub-configs: transformers >= 5
    # moved the encoder/decoder attributes of ParakeetTDTConfig into nested
    # configs (encoder_config.hidden_size etc.) and renamed some
    # (num_decoder_layers for decoder_num_layers).
    sub_configs = [getattr(config, a, None) for a in ("encoder_config", "decoder_config")]
    for cfg in (config, *[sc for sc in sub_configs if sc is not None]):
        for key in keys:
            if hasattr(cfg, key):
                val = getattr(cfg, key)
                if val is not None:
                    return val
    raise AttributeError(f"config missing required attribute (checked: {', '.join(keys)})")


def add_metadata(w: gguf.GGUFWriter, config: Any, frontend: dict[str, Any], profile: str = "bf16_exact") -> None:
    V = gguf.GGUFValueType

    w.add_key_value("starling.format_version", 1, V.UINT32)
    w.add_string("starling.numeric_profile", profile)

    # ---- preprocessor / mel ----
    w.add_key_value("parakeet.preprocessor.sample_rate", frontend["sample_rate"], V.UINT32)
    w.add_key_value("parakeet.preprocessor.n_mels", frontend["n_mels"], V.UINT32)
    w.add_key_value("parakeet.preprocessor.n_fft", frontend["n_fft"], V.UINT32)
    w.add_key_value("parakeet.preprocessor.win_length", frontend["win_length"], V.UINT32)
    w.add_key_value("parakeet.preprocessor.hop_length", frontend["hop_length"], V.UINT32)
    w.add_key_value("parakeet.preprocessor.preemph", frontend["preemph"], V.FLOAT32)
    w.add_key_value("parakeet.preprocessor.mag_power", frontend["mag_power"], V.FLOAT32)
    w.add_string("parakeet.preprocessor.normalize", "per_feature")
    w.add_key_value("parakeet.preprocessor.log_zero_guard", frontend["log_zero_guard"], V.FLOAT32)

    # ---- encoder ----
    d_model = resolve_config_attr(config, "d_model", "hidden_size")
    n_layers = resolve_config_attr(config, "encoder_layers", "num_hidden_layers", "n_layers")
    pred_out = resolve_config_attr(config, "encoder_hidden_size", "output_hidden_size",
                                   "decoder_hidden_size")
    n_heads = resolve_config_attr(config, "encoder_attention_heads", "num_attention_heads")
    ff_dim = resolve_config_attr(config, "encoder_feedforward_dim", "intermediate_size")
    conv_kernel = resolve_config_attr(config, "conv_kernel_size")
    sub_channels = resolve_config_attr(config, "subsampling_conv_channels")
    w.add_key_value("parakeet.encoder.d_model", d_model, V.UINT32)
    w.add_key_value("parakeet.encoder.n_layers", n_layers, V.UINT32)
    w.add_key_value("parakeet.encoder.pred_out", pred_out, V.UINT32)
    w.add_key_value("parakeet.encoder.n_heads", n_heads, V.UINT32)
    w.add_key_value("parakeet.encoder.ff_dim", ff_dim, V.UINT32)
    w.add_key_value("parakeet.encoder.conv_kernel", conv_kernel, V.UINT32)
    w.add_key_value("parakeet.encoder.subsampling_conv_channels", sub_channels, V.UINT32)
    w.add_string("parakeet.encoder.conv_norm_type", "batch_norm")
    w.add_key_value("parakeet.encoder.xscaling", 0, V.INT32)

    # ---- decoder (prediction net) ----
    pred_hidden = resolve_config_attr(config, "decoder_hidden_size", "pred_hidden")
    pred_rnn_layers = resolve_config_attr(config, "decoder_num_layers",
                                          "num_decoder_layers", "pred_rnn_layers")
    w.add_key_value("parakeet.decoder.pred_hidden", pred_hidden, V.UINT32)
    w.add_key_value("parakeet.decoder.pred_rnn_layers", pred_rnn_layers, V.UINT32)

    # ---- joint ----
    joint_hidden = resolve_config_attr(config, "joint_hidden_size", "joint_hidden",
                                       "decoder_hidden_size")
    w.add_key_value("parakeet.joint.joint_hidden", joint_hidden, V.UINT32)
    w.add_string("parakeet.joint.activation", "relu")

    # ---- decoding / vocab ----
    max_symbols = resolve_config_attr(config, "max_symbols_per_step", "max_symbols")
    vocab_size = resolve_config_attr(config, "vocab_size")
    blank_id = resolve_config_attr(config, "blank_token_id", "blank_id")
    # The engine's parakeet.vocab_size EXCLUDES the blank id (vocab_p1 =
    # vocab_size + 1 == embedding rows), while transformers' ParakeetTDTConfig
    # .vocab_size includes it (embed rows == blank_id + 1). The reference GGUF
    # records 8192 for a 8193-row embedding.
    assert blank_id == vocab_size - 1, (blank_id, vocab_size)
    vocab_size = blank_id
    w.add_key_value("parakeet.decoding.max_symbols", max_symbols, V.UINT32)
    w.add_key_value("parakeet.vocab_size", vocab_size, V.UINT32)
    w.add_key_value("parakeet.blank_id", blank_id, V.UINT32)

    # ---- TDT durations ----
    durations = list(resolve_config_attr(config, "durations"))
    w.add_key_value("parakeet.tdt.durations", durations, V.ARRAY, V.INT32)


def extract_tokenizer_pieces(processor: Any) -> list[str]:
    """Extract the SentencePiece tokenizer pieces from the processor.

    Parakeet-tdt uses a SentencePiece model. The processor's tokenizer
    exposes the vocabulary via `get_vocab()` or the sentencepiece model.
    """
    tok = getattr(processor, "tokenizer", processor)
    # Try the sentencepiece interface.
    sp = getattr(tok, "sp_model", None)
    if sp is not None:
        pieces = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]
        return pieces
    # Fallback: build from vocab dict.
    vocab = tok.get_vocab() if hasattr(tok, "get_vocab") else {}
    if vocab:
        size = max(vocab.values()) + 1
        pieces = ["<unk>"] * size
        for token, idx in vocab.items():
            pieces[idx] = token
        return pieces
    raise RuntimeError("cannot extract tokenizer pieces from processor")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Convert nvidia/parakeet-tdt-0.6b-v3 to Starling GGUF.")
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID,
                    help=f"HuggingFace model id (default: {DEFAULT_MODEL_ID})")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                    help=f"Output GGUF path (default: {DEFAULT_OUTPUT.name})")
    ap.add_argument("--dtype", choices=["bf16", "f32"], default="bf16",
                    help="Weight dtype: bf16 (default, byte-exact) or f32")
    args = ap.parse_args()

    print(f"[convert_parakeet] loading {args.model_id} via transformers ...")
    from transformers import AutoModelForTDT, AutoProcessor

    model = AutoModelForTDT.from_pretrained(args.model_id)
    processor = AutoProcessor.from_pretrained(args.model_id)
    config = model.config

    state_dict = model.state_dict()
    print(f"[convert_parakeet] state_dict has {len(state_dict)} tensors")

    # Extract mel filterbank + window.
    print("[convert_parakeet] extracting mel filterbank + window ...")
    fb, window, frontend_params = extract_mel_filterbank(processor)
    print(f"  fb: {fb.shape}, window: {window.shape}")

    # Extract tokenizer pieces.
    print("[convert_parakeet] extracting tokenizer pieces ...")
    pieces = extract_tokenizer_pieces(processor)
    # The engine's detokenizer drops blank emissions because pieces.size() ==
    # vocab_size (blank-EXCLUDED; see capi_parakeet.cpp's decode_ids note), so
    # truncate the piece table at the blank id.
    blank_id = resolve_config_attr(config, "blank_token_id", "blank_id")
    if len(pieces) not in (blank_id, blank_id + 1):
        sys.exit(
            f"unexpected tokenizer piece count {len(pieces)} for blank_id "
            f"{blank_id}: expected {blank_id} (blank excluded) or "
            f"{blank_id + 1} (blank still attached)"
        )
    if len(pieces) > blank_id:
        pieces = pieces[:blank_id]
    print(f"  {len(pieces)} pieces")

    # Write GGUF.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    w = gguf.GGUFWriter(str(args.output), "parakeet_tdt", use_temp_file=True)

    # Metadata.
    profile = "bf16_exact" if args.dtype == "bf16" else "f32_exact"
    add_metadata(w, config, frontend_params, profile)

    # Tokenizer pieces (STRING array KV).
    V = gguf.GGUFValueType
    w.add_key_value("parakeet.tokenizer.pieces", pieces, V.ARRAY, V.STRING)

    # Mel filterbank + window tensors.
    w.add_tensor("preprocessor.featurizer.fb", fb.astype(np.float32))
    w.add_tensor("preprocessor.featurizer.window", window.astype(np.float32))

    # Model weights.
    learned = 0
    for src_name, tensor in sorted(state_dict.items()):
        gguf_name = map_key(src_name)
        if gguf_name is None:
            continue

        # Convert to numpy in the right dtype. Mirroring the reference f16
        # GGUF layout (the layout the engine's conv path is built around —
        # conformer.cpp casts the pointwise conv weights to F16 in-graph):
        #   * conv weights (subsampling / pointwise / depthwise)  -> F16
        #     (ggml_conv_2d im2col types the activations after the weight; a
        #      BF16 weight produces BF16 activations, which the CPU backend's
        #      mul_mat wdata path rejects)
        #   * linear / embedding weights                            -> BF16
        #   * every bias, LayerNorm/BatchNorm parameter, BN running
        #     stat and positional vector                           -> F32
        def tensor_kind(name: str) -> str:
            if (name.endswith(".bias") or ".bias_" in name          # LSTM b_ih/b_hh
                    or ".norm_" in name or name.startswith("norm_")
                    or ".conv.batch_norm." in name or "pos_bias_" in name
                    or name in ("preprocessor.featurizer.fb",
                                "preprocessor.featurizer.window")
                    # PredictionNet::ensure_embed_host_ D2H-reads the embedding
                    # table as F32 (prediction.cpp).
                    or name == "decoder.prediction.embed.weight"):
                return "f32"
            if (".conv." in name or name.endswith("_conv.weight")
                    or ".depthwise_conv." in name):
                return "f16"
            return "bf16"
        kind = tensor_kind(gguf_name)
        if args.dtype == "bf16" and kind == "bf16":
            if tensor.dtype != torch.bfloat16:
                tensor = tensor.to(torch.bfloat16)
            a = np.ascontiguousarray(tensor.view(torch.uint16).numpy())
            w.add_tensor(gguf_name, a, raw_shape=a.shape,
                        raw_dtype=gguf.GGMLQuantizationType.BF16)
        elif kind == "f16":
            a = np.ascontiguousarray(tensor.to(torch.float16).numpy())
            w.add_tensor(gguf_name, a, raw_shape=a.shape,
                        raw_dtype=gguf.GGMLQuantizationType.F16)
        else:
            a = np.ascontiguousarray(tensor.float().numpy())
            w.add_tensor(gguf_name, a)

        learned += 1

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    total = learned + 2  # +2 for fb + window
    print(f"[convert_parakeet] wrote {args.output}: {total} tensors "
          f"({learned} learned), dtype={args.dtype}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
