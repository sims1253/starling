"""Second probe set: per-tensor int8 weights (NNAPI FULLY_CONNECTED has no per-channel
weights) and lower-rank batch matmuls.  python -I nn_probes2.py OUTDIR"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn

OUT = os.path.join(sys.argv[1], "probes")
torch.manual_seed(0)
torch.set_grad_enabled(False)
import litert_torch  # noqa: E402
from ai_edge_quantizer import quantizer, recipe_manager, qtyping  # noqa: E402
from ai_edge_litert import interpreter as tfl  # noqa: E402

Op = qtyping.TFLOperationName


class BMM3(nn.Module):
    def forward(self, q, k):
        return torch.bmm(q, k.transpose(1, 2))


class BMM2(nn.Module):
    def forward(self, q, k):
        return torch.matmul(q, k.t())


new = {"bmm3": (BMM3(), (torch.randn(14, 104, 64), torch.randn(14, 104, 64))),
       "bmm2": (BMM2(), (torch.randn(104, 64), torch.randn(104, 64)))}
for name, (m, args) in new.items():
    litert_torch.convert(m.eval(), args).export(f"{OUT}/{name}.tflite")


def quant(name, bits, tag):
    path = f"{OUT}/{name}.tflite"
    it = tfl.Interpreter(model_path=path)
    sig = it.get_signature_list()["serving_default"]
    dets = {d["name"]: d for d in it.get_input_details()}
    rm = recipe_manager.RecipeManager()
    rm.add_static_config(".*", Op.ALL_SUPPORTED, bits, 8,
                         weight_granularity=qtyping.QuantGranularity.TENSORWISE)
    q = quantizer.Quantizer(path, rm.get_quantization_recipe())
    cal = []
    for _ in range(4):
        c = {}
        for n in sig["inputs"]:
            shp = [int(s) for s in dets["serving_default_" + n]["shape"]]
            c[n] = np.random.randn(*shp).astype(np.float32)
            if shp == [1, 1, 1, 104]:
                c[n] = np.where(np.arange(104) >= 97, -1e4, 0).astype(np.float32).reshape(shp)
        cal.append(c)
    res = q.calibrate({"serving_default": cal})
    qp = f"{OUT}/{name}_{tag}.tflite"
    if os.path.exists(qp):
        os.remove(qp)
    q.quantize(res, serialize_to_path=qp)
    print("wrote", qp, flush=True)


for name in ("fc", "layer", "layernm", "conv"):
    quant(name, 8, "a8w8t")
for name in ("bmm3", "bmm2"):
    quant(name, 8, "a8w8")
