"""Generate the Voxtral Realtime golden reference (``golden/voxtral_reference.json``).

Runs the stock eager ``model.generate`` greedy path on the short/medium/long
fixtures and records the emitted token ids, transcript text, prompt length,
mel shape, and wall-clock ms. ``tests/test_voxtral_ggml_parity.py`` compares
the native engine against this file.

With no explicit length kwargs this is pure stock semantics: total length
bound ``ceil(mel/8)``, EOS from generation_config (id 2). The processor
(mistral-common ``encode_transcription`` + feature extractor) owns the prompt
ids, including the streaming silence pads and the delay-token prefix.

Usage (from the repo root)::

    uv run --group convert python scripts/goldens/voxtral_golden.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "fixtures"
GOLDEN_PATH = REPO / "golden" / "voxtral_reference.json"
MODEL_ID = "mistralai/Voxtral-Mini-4B-Realtime-2602"


def main() -> int:
    from transformers import AutoProcessor, VoxtralRealtimeForConditionalGeneration

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[voxtral-golden] loading model (eager, bf16, {device}) ...")
    model = VoxtralRealtimeForConditionalGeneration.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map=device, attn_implementation="eager")
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    dtype = model.get_input_embeddings().weight.dtype

    out: dict[str, Any] = {
        "model": MODEL_ID,
        "policy": {
            "generate": "stock model.generate (greedy, default length bound)",
            "text": "tokenizer.decode skip_special_tokens=True",
        },
        "fixtures": {},
    }
    for name in ("short", "medium", "long"):
        wav, sr = sf.read(str(FIXTURES / f"{name}.wav"), dtype="float32")
        assert sr == 16000, f"{name}: expected 16 kHz audio, got {sr} Hz"
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        wav = np.ascontiguousarray(wav, dtype=np.float32)

        enc = processor(wav, return_tensors="pt")
        mel = enc["input_features"]
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)
        num_delay = enc.get("num_delay_tokens", 6)
        if isinstance(num_delay, torch.Tensor):
            num_delay = int(num_delay.reshape(-1)[0].item())
        batch = {
            "input_ids": enc["input_ids"].to(torch.int64).to(device),
            "input_features": mel.to(torch.float32).to(device, dtype),
            "num_delay_tokens": int(num_delay),
        }
        prompt_len = int(batch["input_ids"].shape[1])

        with torch.inference_mode():  # warmup once
            model.generate(**batch, max_new_tokens=5)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            gen = model.generate(**batch)
        if device == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        ids = gen[0, prompt_len:].cpu().tolist()
        text = processor.tokenizer.decode(ids, skip_special_tokens=True)

        out["fixtures"][name] = {
            "fixture": f"tests/fixtures/{name}.wav",
            "seconds": round(len(wav) / 16000, 2),
            "prompt_len": prompt_len,
            "mel_T": int(batch["input_features"].shape[-1]),
            "num_delay_tokens": batch["num_delay_tokens"],
            "text": text,
            "ids": ids,
            "n_tokens": len(ids),
            "ms": round(ms, 1),
        }
        print(f"[voxtral-golden] {name}: P={prompt_len} tok={len(ids)} stock_ms={ms:.1f}")

    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_PATH.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"[voxtral-golden] wrote {GOLDEN_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
