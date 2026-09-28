"""Run the real S1 GGUF with greedy and source-only native copy drafts.

The timed C API call includes BPE encoding, embedding lookup, full target
generation, detokenization, and an identical diagnostic ID write in both arms.
Model load and per-configuration warmups are separate. Paired measured calls
alternate order on each repeat.
The diagnostic ID dump contains verified target output, never proposal IDs.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import runpy
import struct
import tempfile
import time


ROOT = Path(__file__).resolve().parents[2]
PROTECTED_TRANSCRIPT = (
    "um please keep the literal tag ZX-1042 and the URL "
    "https://example.org/manual unchanged in the final note"
)
PROTECTED_SPANS = ("ZX-1042", "https://example.org/manual")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lib", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--golden-dir", type=Path, default=ROOT / "golden/s1")
    parser.add_argument("--cases", nargs="+", default=["short", "medium", "long", "protected"])
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    if any(k < 1 or k > 16 for k in args.ks) or args.repeats < 1:
        parser.error("K must be 1..16 and repeats must be positive")

    transcripts = runpy.run_path(str(ROOT / "tests/fixtures/s1_transcripts.py"))["LENGTH_TIERS"]
    transcripts["protected"] = PROTECTED_TRANSCRIPT
    unknown_cases = [case for case in args.cases if case not in transcripts]
    if unknown_cases:
        parser.error(f"unknown cases: {', '.join(unknown_cases)}")

    import torch
    os.environ["STARLING_GGML_DEVICE"] = "cpu"
    os.environ["STARLING_S1_TIMING"] = "1"
    lib = ctypes.CDLL(str(args.lib.resolve()))
    lib.starling_ggml_load.argtypes = [ctypes.c_int, ctypes.c_char_p]
    lib.starling_ggml_load.restype = ctypes.c_void_p
    lib.starling_ggml_normalize_text.argtypes = [ctypes.c_void_p] + [ctypes.c_char_p] * 4
    lib.starling_ggml_normalize_text.restype = ctypes.c_void_p
    lib.starling_ggml_last_error.argtypes = [ctypes.c_void_p]
    lib.starling_ggml_last_error.restype = ctypes.c_char_p
    lib.starling_ggml_free_string.argtypes = [ctypes.c_void_p]
    lib.starling_ggml_free.argtypes = [ctypes.c_void_p]
    lib.starling_ggml_shutdown.argtypes = []
    lib.starling_ggml_backend_name.argtypes = []
    lib.starling_ggml_backend_name.restype = ctypes.c_char_p

    started = time.perf_counter()
    model = lib.starling_ggml_load(8, os.fsencode(args.gguf.resolve()))
    if not model:
        raise RuntimeError(lib.starling_ggml_last_error(None).decode())
    load_ms = (time.perf_counter() - started) * 1000
    device = lib.starling_ggml_backend_name().decode()
    rows = []
    try:
        with tempfile.TemporaryDirectory(prefix="starling-s1-copy-") as tmp:
            dump = Path(tmp) / "ids.i32"

            def run(transcript: str, k: int | None) -> tuple[list[int], str, float, str]:
                os.environ["STARLING_S1_DUMP_IDS"] = str(dump)
                if k is None:
                    os.environ.pop("STARLING_S1_COPY_DRAFT", None)
                    os.environ.pop("STARLING_S1_COPY_MAX_K", None)
                else:
                    os.environ["STARLING_S1_COPY_DRAFT"] = "1"
                    os.environ["STARLING_S1_COPY_MAX_K"] = str(k)
                dump.unlink(missing_ok=True)
                start = time.perf_counter()
                output = lib.starling_ggml_normalize_text(
                    model, transcript.encode(), None, None, None
                )
                elapsed = (time.perf_counter() - start) * 1000
                if not output:
                    raise RuntimeError(lib.starling_ggml_last_error(model).decode())
                try:
                    text = ctypes.string_at(output).decode("utf-8")
                finally:
                    lib.starling_ggml_free_string(output)
                if not dump.exists():
                    raise RuntimeError("native library did not create the S1 ID dump")
                data = dump.read_bytes()
                if not data or len(data) % 4:
                    raise ValueError("incomplete S1 ID dump")
                ids = [value[0] for value in struct.iter_unpack("=i", data)]
                return ids, text, elapsed, hashlib.sha256(data).hexdigest()

            for case in args.cases:
                transcript = transcripts[case]
                expected_ids = expected_text = golden_hash = None
                if case != "protected":
                    golden = args.golden_dir / f"greedy_ids_{case}.pt"
                    expected_ids = torch.load(golden, map_location="cpu", weights_only=True).reshape(-1).tolist()
                    expected_text = (args.golden_dir / f"greedy_text_{case}.txt").read_text()
                    golden_hash = sha256(golden)
                for k in args.ks:
                    # Warm the exact greedy and K configurations for this
                    # case before comparing them. Neither time is reported.
                    run(transcript, None)
                    run(transcript, k)
                    for repeat in range(args.repeats):
                        order = ("greedy", "copy") if repeat % 2 == 0 else ("copy", "greedy")
                        pair = {
                            mode: run(transcript, None if mode == "greedy" else k)
                            for mode in order
                        }
                        baseline_ids, baseline_text, baseline_ms, baseline_hash = pair["greedy"]
                        ids, text, elapsed, output_hash = pair["copy"]
                        rows.append({
                            "case": case, "k": k, "repeat": repeat + 1,
                            "order": list(order),
                            "source_sha256": hashlib.sha256(transcript.encode()).hexdigest(),
                            "baseline_ms": round(baseline_ms, 2),
                            "copy_ms": round(elapsed, 2),
                            "ratio_copy_over_greedy": (
                                round(elapsed / baseline_ms, 3) if baseline_ms > 0 else None
                            ),
                            "output_tokens": len(ids),
                            "greedy_ids_sha256": baseline_hash,
                            "copy_ids_sha256": output_hash,
                            "ids_match_greedy": ids == baseline_ids,
                            "text_matches_greedy": text == baseline_text,
                            "greedy_ids_match_stock": baseline_ids == expected_ids if expected_ids is not None else None,
                            "copy_ids_match_stock": ids == expected_ids if expected_ids is not None else None,
                            "greedy_text_matches_stock": baseline_text == expected_text if expected_text is not None else None,
                            "greedy_protected_spans": {
                                span: span in baseline_text for span in PROTECTED_SPANS
                            } if case == "protected" else None,
                            "copy_protected_spans": {
                                span: span in text for span in PROTECTED_SPANS
                            } if case == "protected" else None,
                            "golden_sha256": golden_hash,
                        })
    finally:
        os.environ.pop("STARLING_S1_COPY_DRAFT", None)
        os.environ.pop("STARLING_S1_COPY_MAX_K", None)
        os.environ.pop("STARLING_S1_DUMP_IDS", None)
        lib.starling_ggml_free(model)
        lib.starling_ggml_shutdown()

    print(json.dumps({
        "model_sha256": sha256(args.gguf),
        "lib_sha256": sha256(args.lib),
        "model_load_ms": round(load_ms, 2),
        "device": device,
        "warmup": "one unmeasured greedy and one unmeasured copy C API call per case/K",
        "timing_scope": "C API call including tokenization, embedding, generation, detokenization, and same ID dump write",
        "rows": rows,
    }, indent=2))
    passed = all(
        row["ids_match_greedy"] and row["text_matches_greedy"]
        and row["greedy_ids_match_stock"] is not False
        and row["copy_ids_match_stock"] is not False
        and row["greedy_text_matches_stock"] is not False
        and (row["copy_protected_spans"] is None
             or all(row["copy_protected_spans"].values()))
        and (row["greedy_protected_spans"] is None
             or all(row["greedy_protected_spans"].values()))
        for row in rows
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
