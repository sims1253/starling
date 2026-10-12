"""Per-variant input files in the model's input dtype + output dequant params.
python -I prep_io.py OUTDIR TAG...  -> OUTDIR/in_TAG_{0,1}.bin, OUTDIR/io_TAG.json"""
import json
import sys

import numpy as np
from ai_edge_litert import interpreter as tfl

OUT = sys.argv[1]
mel = np.fromfile(f"{OUT}/mel_w800.f32", np.float32).reshape(1, 128, 800)
kb = np.fromfile(f"{OUT}/kbias_w800.f32", np.float32).reshape(1, 1, 1, 104)
for tag in sys.argv[2:]:
    it = tfl.Interpreter(model_path=f"{OUT}/qwen3_enc_w800_{tag}.tflite")
    ins = sorted(it.get_input_details(), key=lambda d: d["name"])
    meta = {"inputs": [], "output": None}
    for i, (d, x) in enumerate(zip(ins, (mel, kb))):
        if d["dtype"] != np.float32:
            sc, zp = d["quantization"]
            info = np.iinfo(d["dtype"])
            x = np.clip(np.round(x / sc) + zp, info.min, info.max).astype(d["dtype"])
        x.tofile(f"{OUT}/in_{tag}_{i}.bin")
        meta["inputs"].append({"name": d["name"], "shape": [int(s) for s in d["shape"]],
                               "dtype": str(np.dtype(d["dtype"]))})
    od = it.get_output_details()[0]
    meta["output"] = {"dtype": str(np.dtype(od["dtype"])), "quant": [float(od["quantization"][0]),
                      int(od["quantization"][1])]}
    json.dump(meta, open(f"{OUT}/io_{tag}.json", "w"))
    print(tag, meta)
