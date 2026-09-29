"""Test blockwise second-order compensation on actual ggml IQ2_XXS blocks.

This experiment changes one Parakeet tensor only and measures its output
reconstruction on disjoint held-out activations. It does not establish WER.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import shutil
import struct
from pathlib import Path

import numpy as np
from gguf import GGMLQuantizationType, GGUFReader, quants

IQ2 = GGMLQuantizationType.IQ2_XXS
BLOCK = 256


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1 << 20), b""):
            digest.update(part)
    return digest.hexdigest()


def read_trace(path: Path) -> np.ndarray:
    with path.open("rb") as stream:
        header = stream.read(20)
        if len(header) != 20 or header[:8] != b"STLGACT1":
            raise ValueError(f"{path}: invalid activation trace header")
        width, count = struct.unpack("<IQ", header[8:])
        if not (0 < width <= 16384 and 0 < count <= 4096):
            raise ValueError(f"{path}: invalid activation shape")
        data = np.frombuffer(stream.read(), dtype="<f4")
    if data.size != width * count or not np.isfinite(data).all():
        raise ValueError(f"{path}: incomplete or nonfinite activation trace")
    return data.reshape(count, width).astype(np.float64)


def rel_rms(weights: np.ndarray, quantized: np.ndarray, inputs: np.ndarray) -> float:
    reference = inputs @ weights.T
    error = inputs @ (weights - quantized).T
    return float(np.linalg.norm(error) / np.linalg.norm(reference))


def imatrix_values(path: Path, tensor: str) -> np.ndarray:
    def read_exact(stream, size: int, field: str) -> bytes:
        data = stream.read(size)
        if len(data) != size:
            raise ValueError(f"{path}: truncated imatrix {field}")
        return data

    with path.open("rb") as stream:
        if stream.read(8) != b"STLGIMX1":
            raise ValueError(f"{path}: invalid imatrix magic")
        version, entries = struct.unpack("<II", read_exact(stream, 8, "header"))
        if version != 1:
            raise ValueError(f"{path}: unsupported imatrix version")
        for _ in range(entries):
            name_len = struct.unpack("<I", read_exact(stream, 4, "name length"))[0]
            if not (0 < name_len <= 4096):
                raise ValueError(f"{path}: invalid imatrix name length {name_len}")
            try:
                name = read_exact(stream, name_len, "name").decode()
            except UnicodeDecodeError as exc:
                raise ValueError(f"{path}: invalid UTF-8 imatrix name") from exc
            width = struct.unpack("<I", read_exact(stream, 4, "width"))[0]
            if not (0 < width <= 1_000_000):
                raise ValueError(f"{path}: invalid imatrix width {width}")
            read_exact(stream, 8, "observation count")
            values = np.frombuffer(read_exact(stream, width * 4, f"entry {name}"), dtype="<f4").copy()
            if name == tensor:
                return values
    raise ValueError(f"{tensor}: no imatrix entry")


def named_tensor(path: Path, name: str):
    reader = GGUFReader(path)
    tensor = next((item for item in reader.tensors if item.name == name), None)
    if tensor is None:
        raise ValueError(f"{path}: missing tensor {name}")
    return tensor


class IQ2Quantizer:
    def __init__(self, path: Path):
        self.library = ctypes.CDLL(str(path))
        self.library.ggml_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
        self.library.ggml_row_size.restype = ctypes.c_size_t
        self.library.ggml_quantize_chunk.argtypes = [
            ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.POINTER(ctypes.c_float),
        ]
        self.library.ggml_quantize_chunk.restype = ctypes.c_size_t

    def encode(self, values: np.ndarray, importance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rows, width = values.shape
        if width % BLOCK or importance.shape != (width,):
            raise ValueError("IQ2 input width/importance mismatch")
        source = np.ascontiguousarray(values, dtype=np.float32)
        im = np.ascontiguousarray(importance, dtype=np.float32)
        row_bytes = self.library.ggml_row_size(IQ2.value, width)
        packed = np.empty((rows, row_bytes), dtype=np.uint8)
        written = self.library.ggml_quantize_chunk(
            IQ2.value, source.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_void_p(packed.ctypes.data), 0, rows, width,
            im.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        if written != packed.nbytes:
            raise RuntimeError(f"ggml encoded {written}/{packed.nbytes} bytes")
        decoded = quants.dequantize(packed, IQ2).astype(np.float64)
        return packed, decoded


def compensated(weights: np.ndarray, x: np.ndarray, importance: np.ndarray,
                quantizer: IQ2Quantizer, damping: float) -> tuple[np.ndarray, np.ndarray]:
    """Quantize each 256-wide IQ2 block, updating later blocks via H inverse.

    This is GPTQ's block error update with the existing vector codebook. The
    inner 256 weights are rounded by ggml together, so it is not full GPTQ's
    per-column rounding; the distinction matters when interpreting quality.
    """
    covariance = x.T @ x / len(x)
    covariance.flat[::len(covariance) + 1] += damping * np.mean(np.diag(covariance))
    try:
        inverse = np.linalg.inv(covariance)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            f"calibration covariance is singular ({len(x)} vectors, width {x.shape[1]})"
        ) from exc
    current = weights.copy()
    decoded = np.empty_like(weights)
    packed_parts = []
    for start in range(0, weights.shape[1], BLOCK):
        stop = start + BLOCK
        raw, q = quantizer.encode(current[:, start:stop], importance[start:stop])
        packed_parts.append(raw)
        decoded[:, start:stop] = q
        if stop == weights.shape[1]:
            break
        error = current[:, start:stop] - q
        coefficient = np.linalg.solve(inverse[start:stop, start:stop],
                                      inverse[start:stop, stop:])
        current[:, stop:] -= error @ coefficient
        # Schur complement yields the inverse covariance after eliminating
        # the now-fixed block, as in the block form of the GPTQ update.
        inverse[stop:, stop:] -= inverse[stop:, start:stop] @ coefficient
    return np.concatenate(packed_parts, axis=1), decoded


def run(source: Path, baseline: Path, tensor: str, imatrix: Path,
        calibration: Path, validation: Path, ggml_base: Path,
        damping: float, packed_out: Path | None, gguf_out: Path | None) -> dict:
    x, heldout = read_trace(calibration), read_trace(validation)
    source_tensor, baseline_tensor = named_tensor(source, tensor), named_tensor(baseline, tensor)
    if source_tensor.tensor_type != GGMLQuantizationType.F32 or baseline_tensor.tensor_type != IQ2:
        raise ValueError("expected F32 source and IQ2_XXS baseline tensor")
    weights = np.asarray(source_tensor.data, dtype=np.float64)
    importance = imatrix_values(imatrix, tensor)
    if weights.shape[1] != len(importance) or weights.shape[1] != x.shape[1] or x.shape[1] != heldout.shape[1]:
        raise ValueError("source, imatrix and activation widths differ")
    quantizer = IQ2Quantizer(ggml_base)
    direct_raw, direct = quantizer.encode(weights, importance)
    baseline_raw = np.asarray(baseline_tensor.data, dtype=np.uint8)
    if direct_raw.shape != baseline_raw.shape:
        raise ValueError("baseline tensor byte shape differs")
    actual_baseline = quants.dequantize(baseline_raw, IQ2).astype(np.float64)
    repaired_raw, repaired = compensated(weights, x, importance, quantizer, damping)
    # AWQ's distinct path rescales salient input channels before quantizing.
    # The inverse scale must be applied to activations (or folded into an
    # upstream operation) at inference; this arm is layer-only, not a GGUF.
    salience = np.maximum(np.mean(np.abs(x), axis=0), 1e-12)
    salience /= np.exp(np.mean(np.log(salience)))
    awq_candidates = []
    for alpha in (0.25, 0.5, 0.75, 1.0):
        scale = np.clip(salience ** alpha, 0.25, 4.0)
        _, scaled = quantizer.encode(weights * scale[None, :], importance / scale**2)
        adjusted = scaled / scale[None, :]
        awq_candidates.append((rel_rms(weights, adjusted, x), alpha, adjusted))
    awq_train, awq_alpha, awq = min(awq_candidates, key=lambda item: item[0])
    if repaired_raw.shape != baseline_raw.shape:
        raise ValueError("repaired tensor would change GGUF byte size")
    if packed_out:
        packed_out.write_bytes(repaired_raw.tobytes())
    if gguf_out:
        if gguf_out.resolve() == baseline.resolve():
            raise ValueError("candidate GGUF path must differ from baseline")
        shutil.copyfile(baseline, gguf_out)
        with gguf_out.open("r+b") as stream:
            stream.seek(baseline_tensor.data_offset)
            stream.write(repaired_raw.tobytes())
    return {
        "schema": "starling-iq2-block-pilot-v1",
        "source_sha256": sha256(source),
        "baseline_sha256": sha256(baseline),
        "imatrix_sha256": sha256(imatrix),
        "calibration_sha256": sha256(calibration),
        "validation_sha256": sha256(validation),
        "ggml_base_sha256": sha256(ggml_base),
        "tensor": tensor,
        "weight_shape": list(weights.shape),
        "calibration_vectors": len(x),
        "validation_vectors": len(heldout),
        "iq2_block_bytes": quantizer.library.ggml_row_size(IQ2.value, BLOCK),
        "damping_fraction": damping,
        "awq_alpha_selected_on_calibration": awq_alpha,
        "direct_equals_baseline_bytes": bool(np.array_equal(direct_raw, baseline_raw)),
        "changed_tensor_bytes": int(np.count_nonzero(repaired_raw != baseline_raw)),
        "baseline_tensor_sha256": hashlib.sha256(baseline_raw.tobytes()).hexdigest(),
        "repaired_tensor_sha256": hashlib.sha256(repaired_raw.tobytes()).hexdigest(),
        "candidate_gguf_sha256": sha256(gguf_out) if gguf_out else None,
        "relative_output_rms": {
            "calibration": {
                "baseline_file": rel_rms(weights, actual_baseline, x),
                "direct_iq2": rel_rms(weights, direct, x),
                "block_compensated_iq2": rel_rms(weights, repaired, x),
                "awq_scaled_iq2": awq_train,
            },
            "validation": {
                "baseline_file": rel_rms(weights, actual_baseline, heldout),
                "direct_iq2": rel_rms(weights, direct, heldout),
                "block_compensated_iq2": rel_rms(weights, repaired, heldout),
                "awq_scaled_iq2": rel_rms(weights, awq, heldout),
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--tensor", required=True)
    parser.add_argument("--imatrix", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--ggml-base", type=Path, required=True)
    parser.add_argument("--damping", type=float, default=0.01)
    parser.add_argument("--packed-out", type=Path)
    parser.add_argument("--gguf-out", type=Path)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.source, args.baseline, args.tensor, args.imatrix,
                 args.calibration, args.validation, args.ggml_base,
                 args.damping, args.packed_out, args.gguf_out)
    args.json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
