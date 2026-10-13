# PARKED (2026-10-13): shared tail of the parked TPU exports in this directory.
"""Shared tail of the TPU segment exports: torch module -> .tflite the phone's
on-device compiler accepts.

Weights: NPU_WEIGHTS=fp16 (default; FLOAT_CASTING) or int8 (weight-only,
per-channel min/max; about half the artifact and RAM). The converter's
keep_stablehlo_constant metadata key, which the phone's compiler rejects, is
renamed in place. Returns the input / output byte sizes (f32 I/O, inputs in
argument order).
"""
import os
import tempfile

import numpy as np


def export(module, args, out_path, weights=None):
    import litert_torch
    from ai_edge_litert import interpreter as tfl
    from ai_edge_quantizer import qtyping, quantizer, recipe_manager
    from ai_edge_quantizer.algorithm_manager import AlgorithmName

    weights = weights or os.environ.get("NPU_WEIGHTS", "fp16")
    with tempfile.TemporaryDirectory() as td:
        f32 = os.path.join(td, "f32.tflite")
        litert_torch.convert(module, args).export(f32)
        rm = recipe_manager.RecipeManager()
        if weights == "int8":
            rm.add_weight_only_config(".*", qtyping.TFLOperationName.ALL_SUPPORTED, 8,
                                      algorithm_key=AlgorithmName.MIN_MAX_UNIFORM_QUANT)
        elif weights == "fp16":
            rm.add_weight_only_config(".*", qtyping.TFLOperationName.ALL_SUPPORTED, 16,
                                      algorithm_key=AlgorithmName.FLOAT_CASTING)
        else:
            raise SystemExit(f"NPU_WEIGHTS={weights}: fp16 or int8")
        src = os.path.join(td, "q.tflite")
        quantizer.Quantizer(f32, rm.get_quantization_recipe()).quantize(None, serialize_to_path=src)
        it = tfl.Interpreter(model_path=src)
        ins = [int(np.prod(d["shape"])) * 4 for d in sorted(it.get_input_details(), key=lambda d: d["name"])]
        outs = [int(np.prod(o["shape"])) * 4 for o in it.get_output_details()]
        b = bytearray(open(src, "rb").read())
        key = b"keep_stablehlo_constant"
        assert b.count(key) <= 1
        open(out_path, "wb").write(b.replace(key, b"x" * len(key)))
    return ins, outs
