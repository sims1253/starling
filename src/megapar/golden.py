"""Golden reference capture / load for the Granite-Speech-4.1-2b pipeline.

The golden artefacts under ``golden/`` (gitignored) are produced by the
**eager** stock transformers pipeline on the sample audio. Later Triton-based
phases compare their outputs against these references with the tolerances
declared in :mod:`megapar.config`.

Run ``python -m megapar.golden`` to (re)capture.
"""

from __future__ import annotations

from typing import Any

import torch

from .audio import build_inputs, load_sample_audio
from .config import GOLDEN_DIR, MODEL_ID
from .loader import get_components, load_model_and_processor


# ---------------------------------------------------------------------------
# Artefact names (relative to GOLDEN_DIR). Keep in sync with consumers.
# ---------------------------------------------------------------------------
ENCODER_LAST_HIDDEN = "encoder_last_hidden.pt"
PROJECTOR_OUT = "projector_out.pt"
AUDIO_EMBEDS = "audio_embeds.pt"
INPUTS_EMBEDS = "inputs_embeds.pt"
GREEDY_IDS = "greedy_ids.pt"
GREEDY_TEXT = "greedy_text.txt"
LLM_PREFILL_LOGITS = "llm_prefill_logits.pt"

_ALL_FILES = (
    ENCODER_LAST_HIDDEN,
    PROJECTOR_OUT,
    AUDIO_EMBEDS,
    INPUTS_EMBEDS,
    GREEDY_IDS,
    GREEDY_TEXT,
    LLM_PREFILL_LOGITS,
)


def load_golden(name: str) -> torch.Tensor:
    """Load a tensor artefact from :data:`GOLDEN_DIR` by short name."""
    path = GOLDEN_DIR / name
    return torch.load(path, map_location="cpu")


def load_golden_text(name: str = GREEDY_TEXT) -> str:
    """Load a text artefact from :data:`GOLDEN_DIR` by short name."""
    path = GOLDEN_DIR / name
    return path.read_text(encoding="utf-8")


def _all_exist() -> bool:
    return all((GOLDEN_DIR / f).exists() for f in _ALL_FILES)


def capture_golden(force: bool = False, *, max_new_tokens: int = 200) -> dict[str, Any]:
    """Capture and persist all golden reference artefacts.

    Idempotent: if every artefact already exists and ``force`` is False, this
    is a no-op.

    Returns a dict of {name: shape-or-len} describing what was captured.
    """
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    if _all_exist() and not force:
        print(f"[golden] all artefacts present in {GOLDEN_DIR}; skipping (force=True to recapture)")
        return _summarise_existing()

    print(f"[golden] loading eager model + processor from {MODEL_ID} ...")
    model, processor = load_model_and_processor(attn_impl="eager")
    components = get_components(model)
    encoder = components["encoder"]
    projector = components["projector"]

    print("[golden] loading sample audio ...")
    wav, sr = load_sample_audio()
    inputs = build_inputs(processor, wav)
    input_ids = inputs["input_ids"]
    input_features = inputs["input_features"]
    attention_mask = inputs["attention_mask"]
    input_features_mask = inputs.get("input_features_mask")

    dtype = model.dtype  # bfloat16

    with torch.inference_mode():
        # (1) Encoder last hidden state. The encoder expects bf16 input.
        feats_bf = input_features.to(dtype)
        enc_out = encoder(feats_bf, return_dict=True)
        enc_lhs = enc_out.last_hidden_state
        torch.save(enc_lhs.cpu(), GOLDEN_DIR / ENCODER_LAST_HIDDEN)

        # (2) Projector output.
        proj_out = projector(enc_lhs)
        torch.save(proj_out.cpu(), GOLDEN_DIR / PROJECTOR_OUT)

        # (3) Audio embeds = get_audio_features(...).pooler_output
        audio_out = model.get_audio_features(feats_bf, return_dict=True)
        audio_embeds = audio_out.pooler_output
        torch.save(audio_embeds.cpu(), GOLDEN_DIR / AUDIO_EMBEDS)

        # (4) Merged multimodal inputs_embeds fed to the LLM. Capture via a
        # pre-hook on the LLM so we don't depend on private helpers.
        captured: dict[str, torch.Tensor] = {}

        def _llm_pre_hook(_module, args, kwargs):
            ie = kwargs.get("inputs_embeds", None)
            if ie is None and len(args) >= 1 and isinstance(args[0], torch.Tensor):
                # Positional inputs_embeds (rare for Granite but be safe).
                ie = args[0]
            if ie is not None:
                captured["inputs_embeds"] = ie
            return None

        lm = components["language_model"]
        handle = lm.register_forward_pre_hook(_llm_pre_hook, with_kwargs=True)
        try:
            fwd = model(
                input_ids=input_ids,
                input_features=input_features,
                attention_mask=attention_mask,
                input_features_mask=input_features_mask,
                use_cache=True,
                logits_to_keep=1,
            )
        finally:
            handle.remove()

        if "inputs_embeds" not in captured:
            raise RuntimeError("Failed to capture inputs_embeds from LLM forward")
        inputs_embeds = captured["inputs_embeds"]
        torch.save(inputs_embeds.cpu(), GOLDEN_DIR / INPUTS_EMBEDS)

        # (5) LLM prefill logits at the last position (already computed above
        # with logits_to_keep=1).
        torch.save(fwd.logits.detach().cpu(), GOLDEN_DIR / LLM_PREFILL_LOGITS)

        # (6) Greedy generation.
        gen = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
        )
        torch.save(gen.cpu(), GOLDEN_DIR / GREEDY_IDS)
        text = processor.tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
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
                out[name] = f"{tuple(t.shape)} {t.dtype}"
            else:
                out[name] = type(t).__name__
        except Exception as exc:  # noqa: BLE001
            out[name] = f"<load error: {exc!r}>"
    return out


def _print_summary(summary: dict[str, Any], text: str) -> None:
    print(f"[golden] artefacts in {GOLDEN_DIR}:")
    for name, info in summary.items():
        print(f"  {name:30s} {info}")
    print(f"[golden] greedy_text (first 200 chars):\n{text[:200]!r}")


def main() -> int:
    capture_golden()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
