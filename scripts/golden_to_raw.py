#!/usr/bin/env python3
"""Export MOSS stage goldens as endian-stable raw arrays plus shape metadata."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
NAMES = ("mel", "encoder_hidden", "audio_embeds", "inputs_embeds", "prefill_logits", "prompt_ids", "ids")
def main() -> None:
    out = ROOT / "golden/raw"; out.mkdir(parents=True, exist_ok=True)
    for length in ("short", "medium", "long"):
        for suffix in NAMES:
            stem = f"moss_{length}_{suffix}"; value = torch.load(ROOT / "golden" / f"{stem}.pt", map_location="cpu", weights_only=True)
            is_ids = suffix in ("prompt_ids", "ids")
            a = value.detach().cpu().to(torch.int64 if is_ids else torch.float32).contiguous().numpy()
            a = a.astype("<i8" if is_ids else "<f4", copy=False)
            ext = "i64" if is_ids else "f32"
            a.tofile(out / f"{stem}.{ext}")
            (out / f"{stem}.json").write_text(json.dumps({"shape": list(a.shape), "dtype": ext}) + "\n")
            print(stem, list(a.shape), ext)
if __name__ == "__main__": main()
