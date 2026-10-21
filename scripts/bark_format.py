"""Decode ConwayResearch ``model.bark`` files into a transformers ParakeetForTDT state dict.

Underdog Bark 3.0, Bark 3.0 Plus and Bark 3.0 Clean
(https://huggingface.co/ConwayResearch) are parakeet-tdt-0.6b-v3 fine-tunes
shipped as a single entropy-coded container instead of safetensors. The layout
follows the publisher's ``bark_loader.py`` (CC-BY-4.0, ConwayResearch):

  [u64 LE header length][JSON header][tensor blobs, in header["index"] order]

``header["format"]`` is ``bark-levels-v2`` or ``bark-levels-v3`` (v3 LZMA-wraps
the small side blobs; per-entry ``zl``/``zs``/``zr`` flags say which). Entry kinds:

  levels2   encoder linear weights. Per row, K fp16 magnitudes; each weight is a
            code in [0, 2K]: 0 -> 0, 1..K -> +lev[code-1], K+1..2K -> -lev[code-K-1].
            Codes are ANS-coded column-major per column class (16 classes, one
            categorical model each; the class of each column is a 4-bit id).
  ansintN   other matrices: symmetric N-bit integers (offset 2^(N-1)-1), one ANS
            stream, times an fp16 per-row scale.
  raw       small tensors as fp16 (or int32/int64 when ``dtype`` says so).

Decoding is exact: the result equals what the publisher evaluated. ANS decoding
needs the ``constriction`` package (``uv run --group convert``).
"""
from __future__ import annotations

import json
import lzma
from pathlib import Path

import numpy as np

N_COLUMN_CLASSES = 16
FORMATS = ("bark-levels-v2", "bark-levels-v3")


def _maybe_lzma(blob: bytes, compressed: bool | None) -> bytes:
    return lzma.decompress(blob) if compressed else blob


def _levels_weights(levels: np.ndarray, codes: np.ndarray) -> np.ndarray:
    k = levels.shape[1]
    table = np.concatenate([np.zeros((levels.shape[0], 1), np.float32),
                            levels.astype(np.float32)], axis=1)
    mag = np.where(codes <= k, codes, codes - k).astype(np.int64)
    val = np.take_along_axis(table, mag, axis=1)
    return np.where(codes > k, -val, val)


def _decode_levels_codes(classes: np.ndarray, counts: np.ndarray, words: np.ndarray,
                         rows: int, cols: int) -> np.ndarray:
    import constriction
    coder = constriction.stream.stack.AnsCoder(words)
    codes = np.empty((rows, cols), np.uint8)
    for c in range(len(counts)):
        idx = np.nonzero(classes == c)[0]
        model = constriction.stream.model.Categorical(counts[c] / counts[c].sum(), perfect=False)
        codes[:, idx] = coder.decode(model, rows * len(idx)).reshape(len(idx), rows).T
    return codes


def read_bark(path: str | Path) -> tuple[dict[str, np.ndarray], dict]:
    """Returns (state_dict of numpy arrays, ParakeetTDTConfig dict)."""
    import constriction

    raw = Path(path).read_bytes()
    n = int.from_bytes(raw[:8], "little")
    head = json.loads(raw[8:8 + n])
    if head.get("format") not in FORMATS:
        raise ValueError(f"{path}: unsupported bark format {head.get('format')!r}")
    p = 8 + n

    def take(nbytes: int) -> bytes:
        nonlocal p
        blob = raw[p:p + nbytes]
        if len(blob) != nbytes:
            raise ValueError(f"{path}: truncated at byte {p}")
        p += nbytes
        return blob

    sd: dict[str, np.ndarray] = {}
    for e in head["index"]:
        shape = tuple(e["shape"])
        kind = e["k"]
        if kind == "levels2":
            k, rows, cols = e["K"], shape[0], int(np.prod(shape[1:]))
            levels = np.frombuffer(_maybe_lzma(take(e["lev"]), e.get("zl")), np.float16).reshape(rows, k)
            meta = take(e["meta"])
            nb = (cols * 4 + 7) // 8
            bits = np.unpackbits(np.frombuffer(meta[:nb], np.uint8))[:cols * 4].reshape(cols, 4)
            classes = np.packbits(np.concatenate([np.zeros((cols, 4), np.uint8), bits], 1), axis=1)[:, 0]
            counts = np.frombuffer(meta[nb:], np.uint32).reshape(N_COLUMN_CLASSES, 2 * k + 1).astype(np.float64)
            words = np.frombuffer(take(e["b"]), np.uint32).copy()
            codes = _decode_levels_codes(classes, counts, words, rows, cols)
            sd[e["n"]] = _levels_weights(levels, codes).reshape(shape)
        elif kind.startswith("ansint"):
            nbits = int(kind[len("ansint"):])
            scale = np.frombuffer(_maybe_lzma(take(e["s"]), e.get("zs")), np.float16).astype(np.float32)
            counts = np.frombuffer(take(e["meta"]), np.uint32).astype(np.float64)
            words = np.frombuffer(take(e["b"]), np.uint32).copy()
            model = constriction.stream.model.Categorical(counts / counts.sum(), perfect=False)
            sym = constriction.stream.stack.AnsCoder(words).decode(model, int(np.prod(shape)))
            q = (sym - (2 ** (nbits - 1) - 1)).astype(np.float32)
            sd[e["n"]] = (q.reshape(len(scale), -1) * scale[:, None]).reshape(shape)
        elif kind == "raw":
            dtype = {"torch.int64": np.int64, "torch.int32": np.int32}.get(e.get("dtype"), np.float16)
            sd[e["n"]] = np.frombuffer(_maybe_lzma(take(e["b"]), e.get("zr")), dtype).copy().reshape(shape)
        else:
            raise ValueError(f"{path}: unknown bark entry kind {kind!r} for {e['n']}")
    if p != len(raw):
        raise ValueError(f"{path}: {len(raw) - p} trailing bytes after the last tensor")
    return sd, json.loads(head["config"])
