"""Golden reference capture / load for the Audex-2B ASR pipeline.

The golden artefacts under ``golden/audex/`` (gitignored) are produced by the
**eager** stock transformers pipeline on ``tests/fixtures/short.wav``. Later
megakernel phases compare their outputs against these references.

Run ``python -m starling.audex.golden`` to (re)capture.
"""

from __future__ import annotations

from typing import Any

import torch

from .audio import build_inputs, load_wav
from .config import EOS_TOKEN_ID, GOLDEN_DIR
from .loader import get_components, load_model_and_processor

# ---------------------------------------------------------------------------
# Artefact names (relative to GOLDEN_DIR).
# ---------------------------------------------------------------------------
SAMPLE_AUDIO = "short.wav"
ENCODER_LAST_HIDDEN = "encoder_last_hidden.pt"
AUDIO_EMBEDS = "audio_embeds.pt"
INPUTS_EMBEDS = "inputs_embeds.pt"
GREEDY_IDS = "greedy_ids.pt"
GREEDY_TEXT = "greedy_text.txt"
PROMPT_LEN = "prompt_len.pt"

_ALL_FILES = (
    ENCODER_LAST_HIDDEN,
    AUDIO_EMBEDS,
    INPUTS_EMBEDS,
    GREEDY_IDS,
    GREEDY_TEXT,
    PROMPT_LEN,
)


def load_golden(name: str) -> torch.Tensor:
    return torch.load(GOLDEN_DIR / name, map_location="cpu")


def load_golden_text(name: str = GREEDY_TEXT) -> str:
    return (GOLDEN_DIR / name).read_text(encoding="utf-8")


def _all_exist() -> bool:
    return all((GOLDEN_DIR / f).exists() for f in _ALL_FILES)


def _fixture_wav() -> str:
    from .config import REPO_ROOT

    candidates = [
        REPO_ROOT / "tests" / "fixtures" / SAMPLE_AUDIO,
        REPO_ROOT.parent / "starling" / "tests" / "fixtures" / SAMPLE_AUDIO,
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    raise FileNotFoundError(
        f"audio fixture {SAMPLE_AUDIO!r} not found; tried {[str(c) for c in candidates]}"
    )


def capture_golden(force: bool = False, *, max_new_tokens: int = 200) -> dict[str, Any]:
    """Capture and persist all golden reference artefacts.

    Idempotent: if every artefact already exists and ``force`` is False, this
    is a no-op.
    """
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    if _all_exist() and not force:
        print(f"[golden] all artefacts present in {GOLDEN_DIR}; skipping")
        return _summarise_existing()

    print("[golden] loading eager model + tokenizer + feature extractor ...")
    model, tokenizer, feature_extractor = load_model_and_processor(attn_impl="eager")
    components = get_components(model)
    encoder = components["encoder"]
    projector = components["projector"]

    print(f"[golden] loading sample audio {SAMPLE_AUDIO} ...")
    wav, sr = load_wav(_fixture_wav())
    inputs = build_inputs(tokenizer, feature_extractor, wav)
    input_ids = inputs["input_ids"]
    input_features = inputs["input_features"]
    prompt_len = int(input_ids.shape[1])

    with torch.inference_mode():
        # (1) Encoder last hidden state.
        enc_out = encoder(input_features=input_features, return_dict=True)
        enc_lhs = enc_out.last_hidden_state
        torch.save(enc_lhs.cpu(), GOLDEN_DIR / ENCODER_LAST_HIDDEN)

        # (2) Projector output (audio_embeds).
        audio_embeds = projector(enc_lhs.clone())
        torch.save(audio_embeds.cpu(), GOLDEN_DIR / AUDIO_EMBEDS)

        # (3) Merged inputs_embeds — call prepare_inputs_embeds directly.
        # NemotronDenseAudexForConditionalGeneration._dense_forward_from_embeds
        # bypasses NemotronDenseModel.forward(), so a pre-hook on the inner
        # model never sees inputs_embeds. The top-level prepare_inputs_embeds
        # is the exact merge step.
        inputs_embeds = model.prepare_inputs_embeds(
            input_ids=input_ids,
            input_features=input_features,
        )
        torch.save(inputs_embeds.cpu(), GOLDEN_DIR / INPUTS_EMBEDS)
        torch.save(torch.tensor(prompt_len), GOLDEN_DIR / PROMPT_LEN)

        # (4) Greedy generation (stock transformers generate()).
        gen = model.generate(
            input_ids=input_ids,
            input_features=input_features,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            eos_token_id=EOS_TOKEN_ID,
        )
        gen_new = gen[:, prompt_len:]
        torch.save(gen_new.cpu(), GOLDEN_DIR / GREEDY_IDS)
        raw = tokenizer.decode(gen_new[0], skip_special_tokens=False)
        if "</think>" in raw:
            raw = raw.rsplit("</think>", 1)[-1]
        if "<|im_end|>" in raw:
            raw = raw.split("<|im_end|>", 1)[0]
        text = raw.strip()
        # Extract transcript from conversational wrapper (matches pipeline).
        import re

        m = re.search(r"'(.+)'", text, re.DOTALL)
        if m:
            text = m.group(1).strip()
        (GOLDEN_DIR / GREEDY_TEXT).write_text(text, encoding="utf-8")

    summary = _summarise_existing()
    _print_summary(summary, text)
    return summary


def _summarise_existing() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in _ALL_FILES:
        p = GOLDEN_DIR / name
        if not p.exists():
            out[name] = "<missing>"
            continue
        if name.endswith(".txt"):
            out[name] = f'"{p.read_text(encoding="utf-8")[:60]}..."'
            continue
        try:
            t = torch.load(p, map_location="cpu")
            if isinstance(t, torch.Tensor):
                out[name] = f"{tuple(t.shape)} {t.dtype}" if t.dim() else str(int(t))
            else:
                out[name] = type(t).__name__
        except Exception as exc:
            out[name] = f"<load error: {exc!r}>"
    return out


def _print_summary(summary: dict[str, Any], text: str) -> None:
    print(f"[golden] artefacts in {GOLDEN_DIR}:")
    for name, info in summary.items():
        print(f"  {name:30s} {info}")
    print(f"[golden] greedy_text (first 200 chars):\n{text[:200]!r}")


def main() -> int:
    import sys

    force = "--force" in sys.argv or "-f" in sys.argv
    capture_golden(force=force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
