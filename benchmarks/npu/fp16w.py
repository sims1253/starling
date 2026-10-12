"""fp16 / int8 weight-only variants of the masked f32 encoder (activations stay float).
python -I fp16w.py OUTDIR"""
import os
import sys

import numpy as np
from ai_edge_quantizer import quantizer, recipe_manager, qtyping
from ai_edge_quantizer.algorithm_manager import AlgorithmName
from ai_edge_litert import interpreter as tfl

OUT = sys.argv[1]
src = f"{OUT}/qwen3_enc_w800_f32mask.tflite"
Op = qtyping.TFLOperationName
for tag, bits, alg in (("fp16wmask", 16, AlgorithmName.FLOAT_CASTING),
                       ("wi8mask", 8, AlgorithmName.MIN_MAX_UNIFORM_QUANT)):
    rm = recipe_manager.RecipeManager()
    rm.add_weight_only_config(".*", Op.ALL_SUPPORTED, bits, algorithm_key=alg)
    q = quantizer.Quantizer(src, rm.get_quantization_recipe())
    path = f"{OUT}/qwen3_enc_w800_{tag}.tflite"
    if os.path.exists(path):
        os.remove(path)
    q.quantize(None, serialize_to_path=path)
    print("wrote", path, os.path.getsize(path) / 1e6, "MB")

ref = np.fromfile(f"{OUT}/ref_w800_mask.f32", np.float32).reshape(-1, 1024)
mel = np.fromfile(f"{OUT}/mel_w800.f32", np.float32).reshape(1, 128, 800)
kb = np.fromfile(f"{OUT}/kbias_w800.f32", np.float32).reshape(1, 1, 1, 104)
for tag in ("f32mask", "fp16wmask", "wi8mask"):
    it = tfl.Interpreter(model_path=f"{OUT}/qwen3_enc_w800_{tag}.tflite", num_threads=4)
    run = it.get_signature_runner()
    names = list(run.get_input_details().keys())
    print(tag, "inputs", names)
    o = list(run(**{names[0]: mel, names[1]: kb}).values())[0].reshape(-1, 1024)
    A = 97
    cos = (ref[:A] * o[:A]).sum(1) / (np.linalg.norm(ref[:A], axis=1) * np.linalg.norm(o[:A], axis=1))
    print(f"host-cpu {tag}: rel-L2 {np.linalg.norm(ref[:A]-o[:A])/np.linalg.norm(ref[:A]):.3e} cos min {cos.min():.6f}")
