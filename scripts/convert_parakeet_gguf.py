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

# Conformer-internal renames: transformers names the attention projections
# *_proj and the conv norm `norm`, while the engine (and NeMo) use linear_* /
# pos_bias_* / batch_norm.
_CONFORMER_TAIL_RENAMES = [
    ("self_attn.q_proj.", "self_attn.linear_q."),
    ("self_attn.k_proj.", "self_attn.linear_k."),
    ("self_attn.v_proj.", "self_attn.linear_v."),
    ("self_attn.o_proj.", "self_attn.linear_out."),
    ("self_attn.relative_k_proj.", "self_attn.linear_pos."),
    ("self_attn.bias_u", "self_attn.pos_bias_u"),
    ("self_attn.bias_v", "self_attn.pos_bias_v"),
    ("conv.norm.", "conv.batch_norm."),
]

def _map_conformer_tail(tail: str) -> str:
    for old, new in _CONFORMER_TAIL_RENAMES:
        if old in tail:
            return tail.replace(old, new)
    return tail

def map_key(src: str) -> str | None:
    """Map a transformers state_dict key to the GGUF tensor name.

    Returns None for keys we should skip (non-weight buffers, etc.).
    Raises KeyError for keys that look like weights but have no mapping.
    """
    # Strip common prefixes.
    s = src
    if s.startswith("model."):
        s = s[len("model."):]

    # Bookkeeping buffers are never written.
    if s.endswith(".num_batches_tracked"):
        return None

    # ---- transformers >= 5.x flat naming (ParakeetForTDT) ----
    if s.startswith("encoder.subsampling."):
        tail = s[len("encoder.subsampling."):]
        # The ModuleList is [conv0, ReLU, dw2, pw3, ReLU, dw5, pw6, ReLU,
        # linear]; the Conv2d indices already equal the engine's
        # encoder.pre_encode.conv.{0,2,3,5,6} slots and the ReLUs have no
        # parameters, so a straight index passthrough is exact.
        if tail.startswith("layers."):
            idx, sep, rest = tail[len("layers."):].partition(".")
            if not idx.isdigit():
                return None
            return f"encoder.pre_encode.conv.{idx}.{rest}"
        if tail.startswith("linear."):
            return "encoder.pre_encode.out." + tail[len("linear."):]
        return None
    if s.startswith("encoder.layers."):
        parts = s.split(".", 3)
        if len(parts) < 4:
            return None
        return "encoder.layers." + parts[2] + "." + _map_conformer_tail(parts[3])
    if s.startswith("decoder.embedding."):
        return "decoder.prediction.embed." + s[len("decoder.embedding."):]
    if s.startswith("decoder.lstm."):
        # weight_ih_l0 / weight_hh_l1 / bias_*: same tails the engine reads
        # under decoder.prediction.dec_rnn.lstm.*.
        return "decoder.prediction.dec_rnn.lstm." + s[len("decoder.lstm."):]
    if s.startswith("decoder.decoder_projector."):
        # transformers folds the pred->joint projection into the decoder; the
        # engine applies the same layer as joint.pred on the LSTM output.
        return "joint.pred." + s[len("decoder.decoder_projector."):]
    if s.startswith("encoder_projector."):
        return "joint.enc." + s[len("encoder_projector."):]
    if s.startswith("joint.head."):
        return "joint.joint_net.2." + s[len("joint.head."):]

    # ---- subsampling (ConvSubsampler, older naming) ----
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
    if s.startswith("encoder.pre_encoder.out."):
        return "encoder.pre_encode.out." + s[len("encoder.pre_encoder.out."):]

    # ---- conformer layers ----
    # transformers: encoder.conformer_encoder.layers.{i}.{...}
    #          or: encoder.encoder.layers.{i}.{...}
    # GGUF: encoder.layers.{i}.{...}
    for pfx in ("encoder.conformer_encoder.layers.",
                "encoder.encoder.layers.",
                "encoder.layers."):
        if s.startswith(pfx):
            tail = s[len(pfx):]
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
    # (running_mean/var ARE needed for batch_norm; handled above.)

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
    """Resolve a config value by any of ``keys``.

    transformers >= 5.x nests the encoder hyperparameters in a separate
    ``encoder_config`` dict (hidden_size / num_hidden_layers / ...) instead of
    flat NeMo-style names, so after the flat lookup each key is also tried
    against that nested dict.
    """
    for key in keys:
        if hasattr(config, key):
            val = getattr(config, key)
            if val is not None:
                return val
    enc = getattr(config, "encoder_config", None)
    if enc is not None:
        for key in keys:
            if isinstance(enc, dict):
                if enc.get(key) is not None:
                    return enc[key]
            elif getattr(enc, key, None) is not None:
                return getattr(enc, key)
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
    pred_rnn_layers = resolve_config_attr(config, "decoder_num_layers", "num_decoder_layers",
                                          "pred_rnn_layers")
    w.add_key_value("parakeet.decoder.pred_hidden", pred_hidden, V.UINT32)
    w.add_key_value("parakeet.decoder.pred_rnn_layers", pred_rnn_layers, V.UINT32)

    # ---- joint ----
    joint_hidden = resolve_config_attr(config, "joint_hidden_size", "joint_hidden",
                                       "decoder_hidden_size")
    w.add_key_value("parakeet.joint.joint_hidden", joint_hidden, V.UINT32)
    w.add_string("parakeet.joint.activation", "relu")

    # ---- decoding / vocab ----
    # The engine's joint layout is [vocab_size tokens | blank | durations]:
    # token_count = vocab_size + 1 (blank is its own logit slot past the
    # pieces) and blank_id points at that slot. transformers counts the blank
    # INSIDE config.vocab_size, so subtract it back out.
    max_symbols = resolve_config_attr(config, "max_symbols_per_step", "max_symbols")
    vocab_size = resolve_config_attr(config, "vocab_size")
    blank_id = resolve_config_attr(config, "blank_token_id", "blank_id")
    if blank_id == vocab_size - 1:
        vocab_size = blank_id
    w.add_key_value("parakeet.decoding.max_symbols", max_symbols, V.UINT32)
    w.add_key_value("parakeet.vocab_size", vocab_size, V.UINT32)
    w.add_key_value("parakeet.blank_id", blank_id, V.UINT32)

    # ---- TDT durations ----
    # Written verbatim, INCLUDING a leading 0: the joint head carries one logit
    # per config duration (head size == vocab_size + 1 + len(durations)), and
    # the engine's decode loop handles skip==0 via its max_symbols guard.
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
    print(f"  {len(pieces)} pieces")

    # Write GGUF.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    w = gguf.GGUFWriter(str(args.output), "parakeet_tdt", use_temp_file=True)

    # Metadata.
    profile = "bf16_exact" if args.dtype == "bf16" else "f32_exact"
    add_metadata(w, config, frontend_params, profile)

    # Tokenizer pieces (STRING array KV). The blank must NOT be a piece: the
    # engine's detokenizer emits pieces[id] for any id < len(pieces), and ids
    # >= vocab_size (blank included) must fall through and be dropped.
    V = gguf.GGUFValueType
    blank_id = resolve_config_attr(config, "blank_token_id", "blank_id")
    if pieces and blank_id == len(pieces) - 1:
        pieces = pieces[:-1]
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

        # Convert to numpy in the right dtype.
        if args.dtype == "bf16":
            if tensor.dtype != torch.bfloat16:
                tensor = tensor.to(torch.bfloat16)
            a = np.ascontiguousarray(tensor.view(torch.uint16).numpy())
            w.add_tensor(gguf_name, a, raw_shape=a.shape,
                        raw_dtype=gguf.GGMLQuantizationType.BF16)
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
