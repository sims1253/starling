#!/usr/bin/env python3
"""Re-quantize a Starling GGUF to a smaller ggml quant (q8_0, q5_0, q4_0, ...).

Policy (fail-closed): a tensor is quantized ONLY if it is a 2-D matmul
(mul_mat src0) weight whose name component is ``weight``/``weight_*`` and
whose row length is block-divisible. Everything else -- norms, biases,
BatchNorm stats, positional/rope tables, conv kernels, mel fb/window,
embedding tables (get_rows sources) -- is copied byte-for-byte in its
original dtype, because the C++ engines consume those via ggml_cast-based
f32() helpers, host-side reads, or conv/im2col paths that assume a
non-quantized source.

Any tensor that is neither kept by the patterns below nor a quantizable
2-D weight aborts the run: extend keep_original() consciously per engine
instead of silently quantizing something an engine reads elementwise.

The keep patterns cover all nine engines' tensor names (dry-run verified),
and the numeric_profile is honestly relabeled to the quant — which the
LOADERS gate: only parakeet's loader accepts an arbitrary profile today,
so quantized files currently LOAD on parakeet only. The other engines
(granite/qwen3/audex/hojo/moss/ark/higgs/s1) validate
starling.numeric_profile
and refuse; relaxing those gates is a deliberate engine-side decision.
Validated end-to-end on parakeet (0.0% fixture WER at q8_0/q5_0/q4_0).

Usage:
    uv run python scripts/quantize_gguf_starling.py \
        --input models/parakeet-tdt-0.6b-v3-bf16-exact.gguf \
        --output models/parakeet-tdt-0.6b-v3-q8_0.gguf --quant q8_0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from gguf import GGUFReader, GGUFValueType, GGUFWriter, GGMLQuantizationType
from gguf.lazy import LazyNumpyTensor
from gguf.quants import quantize
from gguf.constants import GGML_QUANT_SIZES

# Substrings covering every non-matmul tensor family across the engines'
# GGUFs (checked against cpp/*/loader.cpp require-lists and the converters,
# then dry-run over all nine engine GGUFs):
# conv kernels (enc.conv1..3, conv_out, audio.conv2d1..3, conv_up/conv_depth/
# conv_down, pointwise/depthwise, subsampling), norms (incl. the GPT-2-style
# *_ln/ln_* LayerNorms of granite's Q-Former and hojo's ln_speech),
# BatchNorm (bn_*, batch_norm), biases (incl. LSTM b_ih/b_hh), positional
# tables (pos_embed, positional_embedding, pos_table, pos_bias_*), rope/sin/
# cos/freq tables, mel filterbank/window, embedding/learned-constant tables
# (get_rows sources, parakeet's host-read F32 embedding, granite's proj.query,
# higgs' proj.temporal).
_KEEP_SUBSTRINGS = (
    "conv", "norm", "ln_", "_ln", "bn_", "batch_norm", "bias", "pos", "rope",
    "sin", "cos", "arange", "freq", "mel", "fb", "window", "embed",
    "query", "temporal",
)


def keep_original(name: str) -> bool:
    """Tensors the engines require in their stored dtype."""
    low = name.lower()
    return any(s in low for s in _KEEP_SUBSTRINGS)


def is_quantizable_weight(name: str, ne_shape: tuple[int, ...]) -> bool:
    """A 2-D matmul weight: mul_mat src0 with a weight[-_*] name component.

    Sequences indexed into the name (parakeet's joint.joint_net.2.weight)
    count as their last non-index component.
    """
    if len(ne_shape) != 2:
        return False
    parts = name.lower().rsplit(".", 2)
    last = parts[-1]
    if last.isdigit() and len(parts) > 1:
        last = parts[-2]
    return last == "weight" or last.startswith("weight_")


def dequant_to_f32(t) -> np.ndarray:
    """Tensor data -> f32 numpy (F32/F16/BF16; quantized sources unsupported)."""
    data = t.data if not isinstance(t, LazyNumpyTensor) else np.frombuffer(
        bytes(t.data), dtype=np.uint8)
    ty = t.tensor_type
    if ty == GGMLQuantizationType.F32:
        return np.frombuffer(bytes(data), dtype=np.float32).copy()
    if ty == GGMLQuantizationType.F16:
        return np.frombuffer(bytes(data), dtype=np.float16).astype(np.float32)
    if ty == GGMLQuantizationType.BF16:
        u16 = np.frombuffer(bytes(data), dtype=np.uint16)
        return (u16.astype(np.uint32) << 16).view(np.float32).astype(np.float32)
    raise SystemExit(
        f"unsupported source dtype {ty!r} for {t.name} (already quantized?)")


def add_raw(writer: GGUFWriter, t) -> None:
    """Re-add a tensor byte-for-byte in its original dtype.

    The reader exposes ne-order shape; GGUFWriter's raw_shape is numpy-order
    (last dim == ne0), so reverse it.
    """
    data = bytes(t.data)
    ty = GGMLQuantizationType(int(t.tensor_type))
    _blck, tsz = GGML_QUANT_SIZES[ty]
    np_shape = tuple(reversed([int(x) for x in t.shape]))
    if tsz in (2, 4) and len(data) % tsz == 0:
        arr = np.frombuffer(data, dtype={2: np.uint16, 4: np.uint32}[tsz])
        writer.add_tensor(t.name, arr.reshape(np_shape), raw_shape=np_shape,
                          raw_dtype=ty)
    else:
        writer.add_tensor(t.name, np.frombuffer(data, dtype=np.uint8),
                          raw_shape=np_shape, raw_dtype=ty)


def add_field(writer: GGUFWriter, name, field) -> None:
    """Copy one KV field from a reader to the writer."""
    ty = field.types[0] if field.types else None
    if ty == GGUFValueType.STRING:
        writer.add_string(name, bytes(field.parts[field.data[0]])
                          .decode("utf-8", "replace"))
    elif ty == GGUFValueType.ARRAY:
        it = field.types[1]
        if not field.data:
            # GGUFWriter.add_array drops empty sequences; preserve that
            # behaviour explicitly (an empty array is lost either way).
            return
        vals = []
        for part in field.data:
            p = np.asarray(field.parts[part]).reshape(-1)
            if it == GGUFValueType.STRING:
                vals.append(bytes(p).decode("utf-8", "replace"))
            else:
                vals.append(p[0].item())
        try:
            # Preserve the stored element subtype (e.g. UINT32) when the
            # writer supports it; inference from Python scalars could pick
            # a different one.
            writer.add_key_value(name, vals, GGUFValueType.ARRAY, sub_type=it)
        except TypeError:
            writer.add_array(name, vals)
    else:
        p = np.asarray(field.parts[field.data[0]]).reshape(-1)
        writer.add_key_value(name, p[0].item(), ty)


def main() -> int:
    quants = sorted(
        m.name.lower() for m in GGMLQuantizationType
        if m.name.lower().startswith(("q", "iq"))
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--quant", default="q8_0", choices=quants,
                    help="target ggml quant")
    args = ap.parse_args()

    qtype = GGMLQuantizationType[args.quant.upper()]
    block = GGML_QUANT_SIZES[qtype][0]
    reader = GGUFReader(str(args.input))

    # Preserve the source file's arch tag (the engine loaders key off the
    # starling.*/<model>.* fields, not general.architecture, but keep it
    # faithful rather than stamping a generic one).
    arch = "starling"
    arch_field = reader.fields.get("general.architecture")
    if arch_field and arch_field.data:
        arch = bytes(arch_field.parts[arch_field.data[0]]).decode("utf-8", "replace")

    writer = GGUFWriter(str(args.output), arch, use_temp_file=True)

    # Copy all KV metadata except the GGUF.* bookkeeping keys, the arch tag
    # (taken by the constructor above), and the numeric profile, which is
    # re-added below to describe the quantized file.
    for name, field in reader.fields.items():
        if (name.startswith("GGUF.") or name == "starling.numeric_profile"
                or name == "general.architecture"):
            continue
        add_field(writer, name, field)
    writer.add_string("starling.numeric_profile", args.quant)

    n_quant = n_keep = 0
    kept_odd_rows: list[str] = []
    for t in reader.tensors:
        ne_shape = tuple(int(x) for x in t.shape)
        if keep_original(t.name):
            add_raw(writer, t)
            n_keep += 1
            continue
        if not is_quantizable_weight(t.name, ne_shape):
            sys.exit(
                f"error: {t.name!r} is neither matched by the keep-list nor a "
                f"2-D 'weight' tensor; refusing to quantize it blindly. "
                f"Extend keep_original() if the engine reads it elementwise/"
                f"host-side, or is_quantizable_weight() if it is a matmul "
                f"src0 with a non-'weight' name."
            )
        if ne_shape[0] % block != 0:
            # A matmul weight whose row width the quant cannot express
            # (e.g. granite's 348-wide enc.out_mid). Keep it unquantized —
            # correctness over compression — and say so.
            kept_odd_rows.append(
                f"{t.name} (row {ne_shape[0]} % block {block} != 0)")
            add_raw(writer, t)
            n_keep += 1
            continue
        np_shape = tuple(reversed(ne_shape))  # (ne1, ne0)
        f32 = dequant_to_f32(t).reshape(np_shape)
        qdata = quantize(f32, qtype)   # (ne1, bytes-per-row) uint8
        writer.add_tensor(t.name, qdata, raw_shape=qdata.shape,
                          raw_dtype=qtype)
        n_quant += 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"[quantize] {args.output}: {n_quant} quantized ({args.quant}), "
          f"{n_keep} kept original dtype")
    for w in kept_odd_rows:
        print(f"  kept unquantized (block-incompatible row): {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
