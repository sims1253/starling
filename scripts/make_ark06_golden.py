"""Generate the ARK-ASR-0.6B golden reference (``golden/ark06_reference.json``).

Runs the stock eager ``model.generate`` greedy path on the short/medium/long
fixtures and records the emitted token ids, transcript text, and wall-clock ms.
The megakernel correctness tests in ``tests/test_ark06_pipeline.py`` compare
against this file byte-for-byte.

The output is gitignored (it requires the 0.6B model); re-run this after pulling
a new model revision to refresh the reference.

Usage (from the repo root):
    TRUST_REMOTE_CODE=1 uv run python scripts/make_ark06_golden.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

from starling.ark06.config import (
    DEFAULT_INSTRUCTION,
    MODEL_ID,
    build_bad_token_ids,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN_REPO = Path("/home/m0hawk/Documents/starling")
FIXTURES = MAIN_REPO / "tests" / "fixtures"
if not FIXTURES.exists():
    FIXTURES = REPO_ROOT / "tests" / "fixtures"
GOLDEN_PATH = REPO_ROOT / "golden" / "ark06_reference.json"


def main() -> int:
    proc = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True, attn_implementation="eager",
    )
    model.eval()

    # Card-faithful suppression: the 0.6B degenerates into special-token
    # repetition under plain greedy decode (see the card README section
    # "build_bad_words_ids"); the ban set below restores the exact reference
    # transcript. EOS stays emittable so the decode can terminate.
    bad_ids = sorted(build_bad_token_ids(proc.tokenizer))
    bad_words_ids = [[i] for i in bad_ids]
    print(f"bad_words_ids: banning {len(bad_ids)} token ids (eos stays emittable)")

    golden: dict[str, dict] = {}
    for fx in ["short", "medium", "long"]:
        wav, sr = sf.read(str(FIXTURES / f"{fx}.wav"))
        if wav.ndim > 1:
            wav = wav[:, 0]
        wav = wav.astype(np.float32)
        dur = len(wav) / sr
        conv = [{"role": "user", "content": [
            {"type": "audio", "array": wav},
            {"type": "text", "text": DEFAULT_INSTRUCTION},
        ]}]
        data = proc.apply_chat_template(
            conv, audio_torch_dtype=torch.bfloat16, tokenize=True,
            return_tensors="pt", add_generation_prompt=True,
        )
        data = {k: v.to("cuda") for k, v in data.items()}
        T = data["input_ids"].shape[1]
        # warmup once
        with torch.inference_mode():
            model.generate(
                **data, max_new_tokens=5, do_sample=False,
                bad_words_ids=bad_words_ids,
            )
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(
                **data, max_new_tokens=200, do_sample=False,
                bad_words_ids=bad_words_ids,
            )
        ms = (time.perf_counter() - t0) * 1000
        gen = out[0][T:]
        text = proc.tokenizer.decode(gen, skip_special_tokens=True)
        golden[fx] = {
            "ids": gen.cpu().tolist(), "text": text,
            "ms": round(ms, 1), "dur": round(dur, 1),
            "n_tokens": int(len(gen)), "T": int(T), "mel": int(data["audios"].shape[-1]),
        }
        print(f"{fx}: dur={dur:.1f}s T={T} tok={len(gen)} stock_ms={ms:.1f}")

    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(GOLDEN_PATH, "w") as f:
        json.dump(golden, f, indent=2)
    print(f"wrote {GOLDEN_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
