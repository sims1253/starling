"""Generate the ARK-ASR golden references (``golden/ark_reference.json`` or
``golden/ark06_reference.json``).

Runs the stock eager ``model.generate`` greedy path (``trust_remote_code``)
on the short/medium/long fixtures and records the emitted token ids,
transcript text, and wall-clock ms. The native exact-text gate
(``tests/test_ggml_parity.py::test_starling_ggml_ark_text_parity``) compares
against the 3B file.

The 0.6B degenerates into special-token repetition under plain greedy decode;
the model card's ``build_bad_words_ids`` recipe (ban all special ids and
added ``<...>`` tokens except EOS) restores the reference transcript, so the
``ark06`` capture applies it.

Usage (from the repo root)::

    uv run --group convert python scripts/goldens/ark_golden.py --model ark
    uv run --group convert python scripts/goldens/ark_golden.py --model ark06
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "fixtures"
INSTRUCTION = "Transcribe the audio to text."
MODELS = {
    "ark": ("AutoArk-AI/ARK-ASR-3B", "ark_reference.json"),
    "ark06": ("Audio8/ARK-ASR-0.6B", "ark06_reference.json"),
}


def bad_token_ids(tokenizer) -> list[int]:
    """The card recipe: ``(all_special_ids | added "<...>" tokens) - {eos}``."""
    bad = set(tokenizer.all_special_ids or [])
    for tok_str, tid in (tokenizer.get_added_vocab() or {}).items():
        if len(tok_str) >= 2 and tok_str.startswith("<") and tok_str.endswith(">"):
            bad.add(int(tid))
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if isinstance(vocab_size, int) and vocab_size > 0:
        bad = {i for i in bad if 0 <= i < vocab_size}
    bad.discard(int(tokenizer.eos_token_id))
    return sorted(bad)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=sorted(MODELS), default="ark")
    args = ap.parse_args()
    model_id, out_name = MODELS[args.model]

    from transformers import AutoModelForCausalLM, AutoProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map=device,
        trust_remote_code=True, attn_implementation="eager",
    )
    model.eval()

    gen_kwargs: dict = {"do_sample": False}
    if args.model == "ark06":
        gen_kwargs["bad_words_ids"] = [[i] for i in bad_token_ids(proc.tokenizer)]
        print(f"bad_words_ids: banning {len(gen_kwargs['bad_words_ids'])} token ids")

    golden: dict[str, dict] = {}
    for fx in ("short", "medium", "long"):
        wav, sr = sf.read(str(FIXTURES / f"{fx}.wav"))
        if wav.ndim > 1:
            wav = wav[:, 0]
        wav = wav.astype(np.float32)
        dur = len(wav) / sr
        conv = [{"role": "user", "content": [
            {"type": "audio", "array": wav},
            {"type": "text", "text": INSTRUCTION},
        ]}]
        data = proc.apply_chat_template(
            conv, audio_torch_dtype=torch.bfloat16, tokenize=True,
            return_tensors="pt", add_generation_prompt=True,
        )
        data = {k: v.to(device) for k, v in data.items()}
        T = data["input_ids"].shape[1]
        with torch.inference_mode():  # warmup once
            model.generate(**data, max_new_tokens=5, **gen_kwargs)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(**data, max_new_tokens=200, **gen_kwargs)
        ms = (time.perf_counter() - t0) * 1000
        gen = out[0][T:]
        text = proc.tokenizer.decode(gen, skip_special_tokens=True)
        golden[fx] = {
            "ids": gen.cpu().tolist(), "text": text,
            "ms": round(ms, 1), "dur": round(dur, 1),
            "n_tokens": int(len(gen)), "T": int(T), "mel": int(data["audios"].shape[-1]),
        }
        print(f"{fx}: dur={dur:.1f}s T={T} tok={len(gen)} stock_ms={ms:.1f}")

    path = REPO / "golden" / out_name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(golden, indent=2))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
