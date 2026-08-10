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
    skip_suffixes = (
        ".num_batches_tracked",
        ".running_mean",
        ".running_var",
    )
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
def extract_mel_filterbank(processor: Any) -> tuple[np.ndarray, np.ndarray]:
    """Extract the mel filterbank + STFT window from the NeMo preprocessor.

    Returns (fb [n_mels, n_fft//2+1], window [win_length]) as float32.
    """
    # The NeMo AudioToMelSpectrogramPreprocessor computes filterbanks via
    # librosa.filters.mel. The transformers processor wraps this in its
    # feature_extractor. We reconstruct the filterbank and window here.
    import librosa

    # Parakeet-tdt constants (from config.hpp / NeMo defaults).
    sample_rate = 16000
    n_fft = 512
    n_mels = 128
    win_length = 400
    hop_length = 160
    fmin = 0.0
    fmax = sample_rate / 2.0

    # Mel filterbank (librosa 'slaney' normalization, matching NeMo).
    fb = librosa.filters.mel(
        sr=sample_rate, n_fft=n_fft, n_mels=n_mels,
        fmin=fmin, fmax=fmax, htk=False, norm="slaney",
    ).astype(np.float32)  # [n_mels, n_fft//2+1]

    # Hann window (NeMo default).
    window = np.array(
        [0.5 - 0.5 * np.cos(2 * np.pi * n / win_length)
         for n in range(win_length)],
        dtype=np.float32,
    )

    # If the processor has cached filter_banks/windows, use those instead.
    fe = getattr(processor, "feature_extractor", None)
    if fe is not None:
        # NeMo preprocessor stores filter_banks as an attribute.
        cached_fb = getattr(fe, "filter_banks", None)
        if cached_fb is not None:
            fb = np.asarray(cached_fb, dtype=np.float32)
        cached_win = getattr(fe, "window", None)
        if cached_win is not None:
            window = np.asarray(cached_win, dtype=np.float32)

    return fb, window


# ---------------------------------------------------------------------------
# KV metadata
# ---------------------------------------------------------------------------
def add_metadata(w: gguf.GGUFWriter, config: Any, profile: str = "bf16_exact") -> None:
    V = gguf.GGUFValueType

    w.add_key_value("starling.format_version", 1, V.UINT32)
    w.add_string("starling.numeric_profile", profile)

    # ---- preprocessor / mel ----
    w.add_key_value("parakeet.preprocessor.sample_rate", 16000, V.UINT32)
    w.add_key_value("parakeet.preprocessor.n_mels", 128, V.UINT32)
    w.add_key_value("parakeet.preprocessor.n_fft", 512, V.UINT32)
    w.add_key_value("parakeet.preprocessor.win_length", 400, V.UINT32)
    w.add_key_value("parakeet.preprocessor.hop_length", 160, V.UINT32)
    w.add_key_value("parakeet.preprocessor.preemph", 0.97, V.FLOAT32)
    w.add_key_value("parakeet.preprocessor.mag_power", 2.0, V.FLOAT32)
    w.add_string("parakeet.preprocessor.normalize", "per_feature")
    w.add_key_value("parakeet.preprocessor.log_zero_guard", 5.9604645e-08, V.FLOAT32)

    # ---- encoder ----
    d_model = getattr(config, "d_model", 1024)
    n_layers = getattr(config, "encoder_layers", 24)
    pred_out = getattr(config, "encoder_hidden_size",
                       getattr(config, "output_hidden_size", 640))
    n_heads = getattr(config, "encoder_attention_heads",
                      getattr(config, "num_attention_heads", 8))
    ff_dim = getattr(config, "encoder_feedforward_dim", 4096)
    conv_kernel = getattr(config, "conv_kernel_size", 9)
    sub_channels = getattr(config, "subsampling_conv_channels", 256)
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
    pred_hidden = getattr(config, "decoder_hidden_size",
                          getattr(config, "pred_hidden", 640))
    pred_rnn_layers = getattr(config, "decoder_num_layers",
                              getattr(config, "pred_rnn_layers", 2))
    w.add_key_value("parakeet.decoder.pred_hidden", pred_hidden, V.UINT32)
    w.add_key_value("parakeet.decoder.pred_rnn_layers", pred_rnn_layers, V.UINT32)

    # ---- joint ----
    joint_hidden = getattr(config, "joint_hidden_size",
                           getattr(config, "joint_hidden", 640))
    w.add_key_value("parakeet.joint.joint_hidden", joint_hidden, V.UINT32)
    w.add_string("parakeet.joint.activation", "relu")

    # ---- decoding / vocab ----
    max_symbols = getattr(config, "max_symbols_per_step",
                          getattr(config, "max_symbols", 10))
    vocab_size = getattr(config, "vocab_size", 8193)
    blank_id = getattr(config, "blank_token_id",
                       getattr(config, "blank_id", 8192))
    w.add_key_value("parakeet.decoding.max_symbols", max_symbols, V.UINT32)
    w.add_key_value("parakeet.vocab_size", vocab_size, V.UINT32)
    w.add_key_value("parakeet.blank_id", blank_id, V.UINT32)

    # ---- TDT durations ----
    durations = list(getattr(config, "durations", [0, 1, 2, 3, 4]))
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
    fb, window = extract_mel_filterbank(processor)
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
    add_metadata(w, config, profile)

    # Tokenizer pieces (STRING array KV).
    V = gguf.GGUFValueType
    w.add_key_value("parakeet.tokenizer.pieces", pieces, V.ARRAY, V.STRING)

    # Mel filterbank + window tensors.
    w.add_tensor("preprocessor.featurizer.fb", fb.astype(np.float32))
    w.add_tensor("preprocessor.featurizer.window", window.astype(np.float32))

    # Model weights.
    learned = 0
    unmapped = []
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

    # Check for unmapped weight keys.
    for src_name in sorted(state_dict.keys()):
        if src_name.endswith(".num_batches_tracked"):
            continue
        gguf_name = map_key(src_name)
        if gguf_name is None:
            # Check if it's a known non-weight (e.g. positional buffers).
            if not any(src_name.startswith(p) for p in
                       ("model.encoder.", "model.decoder.", "model.joint.",
                        "encoder.", "decoder.", "joint.")):
                continue
            unmapped.append(src_name)

    if unmapped:
        print(f"[convert_parakeet] WARNING: {len(unmapped)} unmapped weight keys:")
        for k in unmapped[:20]:
            print(f"  {k}")
        if len(unmapped) > 20:
            print(f"  ... and {len(unmapped) - 20} more")

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    total = learned + 2  # +2 for fb + window
    print(f"[convert_parakeet] wrote {args.output}: {total} tensors "
          f"({learned} learned), dtype={args.dtype}")
    if unmapped:
        print(f"[convert_parakeet] WARNING: {len(unmapped)} keys were not mapped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
