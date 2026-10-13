"""The checkpoint identity of scripts/npu/export_encoder.py against the C++
side (cpp/npu/encoder_contract.hpp): the compile scripts compare the two, so
both must hash the same bytes the same way. cpp/tests/npu_fingerprint_test.cpp
pins the same constant for a GGUF holding the same tensor values.
"""

from __future__ import annotations

import importlib.util
import struct
from pathlib import Path

EXPORTER = Path(__file__).resolve().parents[1] / "scripts" / "npu" / "export_encoder.py"

# kIdentityVector in cpp/tests/npu_fingerprint_test.cpp.
IDENTITY_VECTOR = "0xad085128de48605d"


def _exporter():
    spec = importlib.util.spec_from_file_location("export_encoder", EXPORTER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # stdlib-only at import; torch is imported by export()
    return mod


def test_fnv1a64_is_the_standard_hash():
    ee = _exporter()
    assert ee.FNV_BASIS == 0xCBF29CE484222325
    assert ee.fnv1a64(b"") == 0xCBF29CE484222325
    assert ee.fnv1a64(b"a") == 0xAF63DC4C8601EC8C
    assert ee.fnv1a64(b"foobar") == 0x85944171F73967E8


def test_identity_matches_the_cpp_vector():
    ee = _exporter()
    # Identity tensor k (FINGERPRINT_TENSORS order) holds 4 + k f32 values (16k + i) / 4 - 1.
    values = {ck: [(16 * k + i) * 0.25 - 1.0 for i in range(4 + k)]
              for k, (_, ck) in enumerate(ee.FINGERPRINT_TENSORS)}
    got = ee.checkpoint_identity(lambda name: struct.pack(f"<{len(values[name])}f", *values[name]))
    assert got == IDENTITY_VECTOR


def test_identity_tensors_match_the_cpp_list():
    ee = _exporter()
    header = (EXPORTER.parents[2] / "cpp" / "npu" / "encoder_contract.hpp").read_text()
    body = header.split("kQwen3EncFingerprintTensors[] = {", 1)[1].split("};", 1)[0]
    cpp_names = [s.strip().strip('"') for s in body.replace("\n", " ").split(",") if s.strip()]
    assert cpp_names == [g for g, _ in ee.FINGERPRINT_TENSORS]
