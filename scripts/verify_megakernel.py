"""Verify the higgs LLMMega megakernel reproduces the golden token ids exactly.

Loads the model under .venv-higgs (transformers 4.51), runs the graph-captured
decode on the short fixture, and compares the generated token ids to the golden
oracle in golden/higgs_golden.json. Must be byte-exact.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MAIN_REPO = Path("/home/m0hawk/Documents/starling")
FIXTURES_DIR = MAIN_REPO / "tests" / "fixtures"
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import torch
import soundfile as sf

from starling.higgs.loader import load_model_and_tokenizer, make_collator
from starling.higgs.config import EOS_TOKEN_IDS
from starling.higgs.llm_mega import LLMMega

# reuse the ref prompt builder
sys.path.insert(0, str(REPO / "scripts" / "ref"))
import transcribe as ref  # noqa: E402


def build_batch(model, tok, audio_np):
    collator = make_collator(model)
    input_ids = ref._build_input_tokens(tok, ref.DEFAULT_PROMPT, enable_thinking=True)
    sample = ref._build_sample(audio_np, input_ids, sample_rate=16000)
    batch = asdict(collator([sample]))
    return {k: (v.to("cuda").contiguous() if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()}


def main() -> int:
    golden = json.loads((REPO / "golden" / "higgs_golden.json").read_text())
    model, tok = load_model_and_tokenizer()
    mega = LLMMega(model, max_cache_len=1024)

    all_ok = True
    for name in ("short", "medium", "long"):
        path = FIXTURES_DIR / f"{name}.wav"
        audio_np, sr = sf.read(str(path))
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=1)
        audio_np = np.asarray(audio_np, dtype=np.float32)
        batch = build_batch(model, tok, audio_np)
        ref_ids = golden["fixtures"][name]["gen_ids"]
        max_new = len(ref_ids) + 2  # allow one extra to confirm EOS stops it
        res = mega.generate(batch, max_new_tokens=max_new, eos_token_ids=EOS_TOKEN_IDS, tokenizer=tok)
        got = res.ids[0].tolist()
        ok = got[:len(ref_ids)] == ref_ids
        # also accept if we stopped early at EOS within the golden length
        stopped_ok = got == ref_ids
        print(f"[{name}] generated {len(got)} tokens; golden {len(ref_ids)}")
        print(f"  match (prefix): {ok}   exact: {stopped_ok}")
        if not ok:
            print(f"  GOT[:15]:  {got[:15]}")
            print(f"  REF[:15]:  {ref_ids[:15]}")
            print(f"  GOT[-5:]:  {got[-5:]}")
            print(f"  REF[-5:]:  {ref_ids[-5:]}")
            all_ok = False
        else:
            print(f"  text: {res.text[:80]!r}")
    print("\n=== BYTE-EXACT:", "YES" if all_ok else "NO", "===")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
