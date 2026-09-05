"""Generate the Voxtral Realtime golden reference (``golden/voxtral_reference.json``).

Runs the stock eager ``model.generate`` greedy path (via
``VoxtralPipeline.transcribe_stock``) on the short/medium/long fixtures and
records the emitted token ids, transcript text, prompt length, mel shape,
and wall-clock ms. The GPU-gated parity tests in
``tests/test_voxtral_pipeline.py`` compare against this file.

With no explicit length kwargs this is pure stock semantics: total length
bound ``ceil(mel/8)``, EOS from generation_config (id 2).

The output is gitignored (it requires the ~9 GB model); re-run this after
pulling a new model revision to refresh the reference.

Usage (from the repo root, GPU box):
    uv run python scripts/make_voxtral_golden.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "tests" / "fixtures"
GOLDEN_PATH = REPO_ROOT / "golden" / "voxtral_reference.json"
FIXTURE_NAMES = ("short", "medium", "long")


def main() -> int:
    from starling.voxtral.config import MODEL_ID
    from starling.voxtral.pipeline import VoxtralPipeline

    print("[voxtral-golden] loading model (eager, bf16) ...")
    pipe = VoxtralPipeline.from_pretrained()

    out: dict[str, Any] = {
        "model": MODEL_ID,
        "policy": {
            "generate": "stock model.generate (greedy, default length bound)",
            "text": "tokenizer.decode skip_special_tokens=True",
        },
        "fixtures": {},
    }

    for name in FIXTURE_NAMES:
        path = FIXTURES / f"{name}.wav"
        wav, _sr = pipe._read_wav_or_array(str(path))
        batch = pipe._prepare_batch(wav)
        prompt_len = int(batch["input_ids"].shape[1])
        mel_T = int(batch["input_features"].shape[-1])
        duration = len(wav) / 16000

        import torch

        # warmup once
        with torch.inference_mode():
            pipe.model.generate(**batch, max_new_tokens=5)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        text, ids = pipe.transcribe_stock(str(path))
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000

        gen = ids[0].tolist()
        out["fixtures"][name] = {
            "fixture": f"tests/fixtures/{name}.wav",
            "seconds": round(duration, 2),
            "prompt_len": prompt_len,
            "mel_T": mel_T,
            "num_delay_tokens": batch["num_delay_tokens"],
            "text": text,
            "ids": gen,
            "n_tokens": len(gen),
            "ms": round(ms, 1),
        }
        print(
            f"[voxtral-golden] {name}: dur={duration:.1f}s P={prompt_len} "
            f"mel_T={mel_T} tok={len(gen)} stock_ms={ms:.1f}"
        )

    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_PATH.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"[voxtral-golden] wrote {GOLDEN_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
