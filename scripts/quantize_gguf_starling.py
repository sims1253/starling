#!/usr/bin/env python3
"""Re-quantize a Starling GGUF to a smaller ggml quant (q8_0, q5_0, q4_k, ...).

Generic over the engine GGUFs written by scripts/convert_*_gguf.py: it keeps
every tensor the C++ engines read host-side or elementwise in its ORIGINAL
dtype (norms, biases, BN stats, positional vectors, mel fb/window, parakeet's
F32 embedding) and quantizes only the matmul (mul_mat src0) weights.

Usage:
    uv run python scripts/quantize_gguf_starling.py \
        --input models/parakeet-tdt-0.6b-v3-bf16-exact.gguf \
        --output models/parakeet-tdt-0.6b-v3-q8_0.gguf --quant q8_0
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
from gguf import GGUFReader, GGUFValueType, GGUFWriter, GGMLQuantizationType
from gguf.lazy import LazyNumpyTensor
from gguf.quants import quantize
from gguf.constants import GGML_QUANT_SIZES


def dequant_to_f32(t) -> np.ndarray:
    """Tensor data -> f32 numpy (handles BF16/F16/F32; quants not expected)."""
    data = t.data if not isinstance(t, LazyNumpyTensor) else np.frombuffer(
        bytes(t.data), dtype=np.uint8)
    ty = str(t.tensor_type)
    if ty == "0":  # F32
        return np.frombuffer(bytes(data), dtype=np.float32).copy()
    if ty == "1":  # F16
        return np.frombuffer(bytes(data), dtype=np.float16).astype(np.float32)
    if ty == "30":  # BF16 (ggml type index)
        u16 = np.frombuffer(bytes(data), dtype=np.uint16)
        return (u16.astype(np.uint32) << 16).view(np.float32).astype(np.float32)
    raise SystemExit(f"unsupported source dtype {ty} for {t.name}")


def raw_copy(t):
    """Return (bytes, GGMLQuantizationType) for a pass-through tensor."""
    data = bytes(t.data)
    ty = GGMLQuantizationType(int(str(t.tensor_type)))
    return data, ty


def add_raw(writer: GGUFWriter, t) -> None:
    """Re-add a tensor byte-for-byte in its original dtype.

    The reader exposes ne-order shape; GGUFWriter's raw_shape is numpy-order
    (last dim == ne0), so reverse it.
    """
    data, ty = raw_copy(t)
    _blck, tsz = GGML_QUANT_SIZES[ty]
    np_shape = tuple(reversed([int(x) for x in t.shape]))
    if tsz in (2, 4) and len(data) % tsz == 0:
        arr = np.frombuffer(data, dtype={2: np.uint16, 4: np.uint32}[tsz])
        writer.add_tensor(t.name, arr.reshape(np_shape), raw_shape=np_shape,
                          raw_dtype=ty)
    else:
        writer.add_tensor(t.name, np.frombuffer(data, dtype=np.uint8),
                          raw_shape=np_shape, raw_dtype=ty)


def keep_original(name: str) -> bool:
    """Tensors the engines require in their stored dtype."""
    return (
        name.endswith(".bias")
        or ".bias_" in name                       # LSTM b_ih/b_hh
        or ".norm_" in name or name.startswith("norm_")
        or ".batch_norm." in name
        or "pos_bias_" in name
        or "pos_embed" in name or "pos_table" in name
        or "mel_filter" in name or "mel_window" in name
        or ".fb" in name or ".window" in name
        or "embed.weight" in name                 # parakeet host-read embedding
        or ".conv." in name                       # conv weights stay F16 (engine casts)
        or re.search(r"\.(rel_pos|sin|cos|arange|freq)", name)
    )


def add_field(writer: GGUFWriter, name, field) -> None:
    """Copy one KV field from a reader to the writer."""
    ty = field.types[0] if field.types else None
    if ty == GGUFValueType.STRING:
        writer.add_string(name, bytes(field.parts[field.data[0]])
                          .decode("utf-8", "replace"))
    elif ty == GGUFValueType.ARRAY:
        it = field.types[1]
        vals = []
        for part in field.data:
            p = np.asarray(field.parts[part]).reshape(-1)
            if it == GGUFValueType.STRING:
                vals.append(bytes(p).decode("utf-8", "replace"))
            else:
                vals.append(p[0].item())
        writer.add_array(name, vals)
    else:
        p = np.asarray(field.parts[field.data[0]]).reshape(-1)
        writer.add_key_value(name, p[0].item(), ty)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--quant", default="q8_0",
                    help="target ggml quant (q8_0, q5_0, q4_k, q4_k_s, ...)")
    args = ap.parse_args()

    qtype = GGMLQuantizationType[args.quant.upper()]
    reader = GGUFReader(str(args.input))

    writer = GGUFWriter(str(args.output), "starling", use_temp_file=True)

    # Copy all KV metadata except the GGUF.* bookkeeping keys.
    for name, field in reader.fields.items():
        if name.startswith("GGUF."):
            continue
        add_field(writer, name, field)
    # Update the numeric profile so the loaders' metadata gate stays honest.
    try:
        writer.override("starling.numeric_profile", args.quant)
    except Exception:
        pass

    n_quant = n_keep = 0
    for t in reader.tensors:
        if keep_original(t.name):
            add_raw(writer, t)
            n_keep += 1
        else:
            np_shape = tuple(reversed([int(x) for x in t.shape]))  # (ne1, ne0)
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
