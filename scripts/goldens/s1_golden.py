"""Capture the S1-mini text-normalizer goldens under ``golden/s1/``.

Runs the eager stock transformers path, exactly the model-card quickstart
(``apply_chat_template`` with ``enable_thinking=False``, greedy
``model.generate``), on the length-tier transcripts in
``tests/fixtures/s1_transcripts.py``. The greedy budget is the card's
``1.3 x prompt_tokens + 32``. The native engine's exact-text gate
(``tests/test_ggml_parity.py::test_starling_ggml_s1_text_parity``) compares
against ``greedy_text_{tier}.txt``.

Usage (from the repo root)::

    uv run --group convert python scripts/goldens/s1_golden.py [--force]
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
GOLDEN_DIR = REPO / "golden" / "s1"
MODEL_ID = "superwhisper/s1-mini"

# Verbatim from the model card: reworded system prompts hallucinate.
SYSTEM_PROMPT = (
    "You are a text normalizer for speech-to-text transcripts. The input begins "
    "with a control line specifying the styling, structure, and context settings; "
    "clean the transcript to match those settings and output only the cleaned text."
)
CONTROL_LINE = "[Styling: semi-formal] [Structure: prose] [Context: general]"

TIERS = ("short", "medium", "long")
FILES = ("greedy_ids_{tier}.pt", "greedy_text_{tier}.txt", "prompt_len_{tier}.pt",
         "prompt_ids_{tier}.pt")


def max_new_tokens_for(n_input_tokens: int) -> int:
    """Greedy budget per the card: ``1.3 x input + 32``."""
    return int(n_input_tokens * 1.3) + 32


def fixture_transcripts() -> dict[str, str]:
    spec = importlib.util.spec_from_file_location(
        "s1_transcripts", REPO / "tests" / "fixtures" / "s1_transcripts.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return dict(mod.LENGTH_TIERS)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="overwrite existing goldens")
    args = ap.parse_args()

    existing = [GOLDEN_DIR / f.format(tier=t) for t in TIERS for f in FILES]
    if all(p.exists() for p in existing) and not args.force:
        print(f"[s1-golden] all artefacts present in {GOLDEN_DIR}; pass --force to recapture")
        return 0
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[s1-golden] loading eager stock {MODEL_ID} on {device} ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, attn_implementation="eager").to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    for tier, transcript in fixture_transcripts().items():
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{CONTROL_LINE}\n{transcript}"},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        budget = max_new_tokens_for(int(inputs.input_ids.shape[1]))
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=budget, do_sample=False)
        ids = out[0][inputs.input_ids.shape[1]:].cpu()
        decoded = tokenizer.decode(ids, skip_special_tokens=True)
        prompt_ids = inputs.input_ids.cpu()

        torch.save(prompt_ids, GOLDEN_DIR / f"prompt_ids_{tier}.pt")
        torch.save(ids, GOLDEN_DIR / f"greedy_ids_{tier}.pt")
        (GOLDEN_DIR / f"greedy_text_{tier}.txt").write_text(decoded, encoding="utf-8")
        torch.save(torch.tensor(int(prompt_ids.shape[1])), GOLDEN_DIR / f"prompt_len_{tier}.pt")
        print(f"[s1-golden] {tier}: prompt {prompt_ids.shape[1]} tok, gen {ids.shape[0]} tok "
              f"-> {decoded[:60]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
