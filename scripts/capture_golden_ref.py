"""Capture byte-exact golden transcripts via the UPSTREAM reference path.

Uses the isolated ``.venv-higgs`` (transformers 4.46.3 + boson-multimodal) and
the model's own ``transcribe()`` (which calls ``model.generate(do_sample=False)``)
-- the unmodified reference decode. This is the correctness oracle: every
starling megakernel path must reproduce these token ids / text exactly.

Run with uv targeting the isolated venv (see src/starling/higgs/UV_NOTES.md):
    uv run --no-project --python .venv-higgs/bin/python python scripts/capture_golden_ref.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MAIN_REPO = Path("/home/m0hawk/Documents/starling")
FIXTURES_DIR = MAIN_REPO / "tests" / "fixtures"
# Use the upstream transcribe.py shipped with the model (copied here). It imports
# the collator from a co-located higgs_audio_collator shim (re-exporting our
# vendored copy, which is transformers-version-independent).
sys.path.insert(0, str(REPO / "scripts" / "ref"))
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import soundfile as sf  # noqa: E402
from transformers import AutoModel, AutoConfig, AutoTokenizer  # noqa: E402

import transcribe as higgs_transcribe  # noqa: E402  (upstream ref)

MODEL_ID = "bosonai/higgs-audio-v3-stt"
FIXTURES = [
    ("short", FIXTURES_DIR / "short.wav"),
    ("medium", FIXTURES_DIR / "medium.wav"),
    ("long", FIXTURES_DIR / "long.wav"),
]


def main() -> int:
    torch.manual_seed(0)
    cfg = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    print("loading model (ref venv, trust_remote_code)...", flush=True)
    t0 = time.time()
    model = AutoModel.from_pretrained(
        MODEL_ID, config=cfg, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="eager",
    )
    model.eval()
    print(f"loaded in {time.time()-t0:.1f}s, VRAM {torch.cuda.memory_allocated()//1024//1024} MB", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)

    out = {"model": MODEL_ID, "path": "transcribe.transcribe (upstream generate)", "fixtures": {}}
    for name, path in FIXTURES:
        audio_np, sr = sf.read(str(path))
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=1)
        audio_np = np.asarray(audio_np, dtype=np.float32)
        dur = len(audio_np) / sr
        t1 = time.time()
        text = higgs_transcribe.transcribe(model, tok, audio_np, sample_rate=sr, max_new_tokens=512)
        elapsed = time.time() - t1
        print(f"[{name}] {dur:.1f}s -> {elapsed:.2f}s :: {text!r}", flush=True)

        # Also capture the exact generated token ids by replaying the same
        # prompt/collator through model.generate() directly (the byte-exact
        # oracle the megakernel must reproduce). build input_ids the same way
        # transcribe() does.
        from dataclasses import asdict
        input_ids = higgs_transcribe._build_input_tokens(
            tok, higgs_transcribe.DEFAULT_PROMPT, enable_thinking=True
        )
        sample = higgs_transcribe._build_sample(audio_np, input_ids, sample_rate=16000)
        collator = higgs_transcribe._cached_collator
        batch = asdict(collator([sample]))
        batch = {
            k: v.to("cuda").contiguous() if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        with torch.inference_mode(), higgs_transcribe._suppress_right_padding_warning():
            outputs = model.generate(
                **batch, max_new_tokens=512, use_cache=True, do_sample=False,
                stop_strings=["<|im_end|>", "<|endoftext|>"], tokenizer=tok,
            )
        output_ids = outputs[0] if isinstance(outputs, tuple) else outputs
        # strip the prompt prefix; keep only the newly-generated tokens
        prompt_len = batch["input_ids"].shape[1]
        gen_ids = output_ids[0, prompt_len:].cpu().tolist()

        out["fixtures"][name] = {
            "path": str(path), "duration_s": dur, "sample_rate": sr,
            "text": text, "wall_s": elapsed, "gen_ids": gen_ids,
            "prompt_len": prompt_len,
        }

    golden_path = REPO / "golden" / "higgs_golden.json"
    golden_path.parent.mkdir(parents=True, exist_ok=True)
    golden_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {golden_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
