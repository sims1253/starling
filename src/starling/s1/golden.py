"""Golden reference capture / load for the S1-mini normalization pipeline.

Golden artefacts under ``golden/s1/`` (gitignored) are produced by the **eager
stock transformers** path — exactly the model-card quickstart
(``apply_chat_template`` with ``enable_thinking=False``, ``model.generate``)
— on the transcript fixtures in ``tests/fixtures/s1_transcripts.py``. The
megakernel pipeline and the native GGML engine compare byte-for-byte against
these.

Run ``python -m starling.s1.golden`` to (re)capture.
"""

from __future__ import annotations

import importlib.util
from typing import Any

import torch

from .config import GOLDEN_DIR, MODEL_ID
from .loader import load_model_and_tokenizer

GREEDY_IDS = "greedy_ids_{tier}.pt"
GREEDY_TEXT = "greedy_text_{tier}.txt"
PROMPT_LEN = "prompt_len_{tier}.pt"
PROMPT_IDS = "prompt_ids_{tier}.pt"

TIERS = ("short", "medium", "long")

_ALL_FILES = tuple(
    name.format(tier=tier)
    for tier in TIERS
    for name in (GREEDY_IDS, GREEDY_TEXT, PROMPT_LEN, PROMPT_IDS)
)


def load_golden(name: str) -> torch.Tensor:
    return torch.load(GOLDEN_DIR / name, map_location="cpu")


def load_golden_text(name: str) -> str:
    return (GOLDEN_DIR / name).read_text(encoding="utf-8")


def _fixture_transcripts() -> dict[str, str]:
    spec = importlib.util.spec_from_file_location(
        "s1_transcripts", GOLDEN_DIR.parents[1] / "tests" / "fixtures" / "s1_transcripts.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return dict(mod.LENGTH_TIERS)


def _all_exist() -> bool:
    return all((GOLDEN_DIR / f).exists() for f in _ALL_FILES)


def _stock_normalize(model: Any, tokenizer: Any, transcript: str, max_new_tokens: int):
    """The model-card quickstart, verbatim (tokenize=False path)."""
    from .config import SYSTEM_PROMPT, control_line

    user = f"{control_line()}\n{transcript}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    ids = out[0][inputs.input_ids.shape[1]:]
    return inputs.input_ids, ids.cpu(), tokenizer.decode(ids, skip_special_tokens=True)


def capture_golden(force: bool = False) -> dict[str, Any]:
    """Capture and persist golden artefacts for all length tiers."""
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    if _all_exist() and not force:
        print(f"[golden] all artefacts present in {GOLDEN_DIR}; skipping (force=True to recapture)")
        return {}

    print(f"[golden] loading eager stock model from {MODEL_ID} ...")
    model, tokenizer = load_model_and_tokenizer(attn_impl="eager")

    from .config import SYSTEM_PROMPT, control_line, max_new_tokens_for

    tiers = _fixture_transcripts()
    summary: dict[str, Any] = {}
    for tier, transcript in tiers.items():

        # Budget computed from the actual prompt length, mirroring the pipeline.
        user_text = f"{control_line()}\n{transcript}"
        tmpl = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        budget = max_new_tokens_for(len(tokenizer(tmpl).input_ids))
        prompt_ids, ids, text = _stock_normalize(model, tokenizer, transcript, budget)
        torch.save(prompt_ids.cpu(), GOLDEN_DIR / PROMPT_IDS.format(tier=tier))
        torch.save(ids, GOLDEN_DIR / GREEDY_IDS.format(tier=tier))
        (GOLDEN_DIR / GREEDY_TEXT.format(tier=tier)).write_text(text, encoding="utf-8")
        torch.save(torch.tensor(int(prompt_ids.shape[1])), GOLDEN_DIR / PROMPT_LEN.format(tier=tier))
        summary[tier] = {"prompt_len": int(prompt_ids.shape[1]), "n_gen": int(ids.shape[0]), "text": text}
        print(f"[golden] {tier}: prompt {summary[tier]['prompt_len']} tok, gen {summary[tier]['n_gen']} tok -> {text[:60]!r}")

    del model
    torch.cuda.empty_cache()
    return summary


if __name__ == "__main__":
    capture_golden()
