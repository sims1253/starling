"""Golden reference capture / load for the Qwen3-ASR-1.7B pipeline.

The golden artefacts under ``golden/qwen3/`` (gitignored) are produced by the
**eager** stock transformers pipeline on ``tests/fixtures/short.wav``. Later
megakernel phases compare their outputs against these references.

Run ``python -m starling.qwen3.golden`` to (re)capture.
"""

from __future__ import annotations

from typing import Any

import torch

from .audio import build_inputs, load_wav
from .config import GOLDEN_DIR, MODEL_ID
from .loader import get_components, load_model_and_processor

# ---------------------------------------------------------------------------
# Artefact names (relative to GOLDEN_DIR). Keep in sync with consumers.
# ---------------------------------------------------------------------------
SAMPLE_AUDIO = "short.wav"  # tests/fixtures/short.wav
ENCODER_LAST_HIDDEN = "encoder_last_hidden.pt"
AUDIO_EMBEDS = "audio_embeds.pt"          # projector output (pooler_output)
INPUTS_EMBEDS = "inputs_embeds.pt"
GREEDY_IDS = "greedy_ids.pt"
GREEDY_TEXT = "greedy_text.txt"
PROMPT_LEN = "prompt_len.pt"              # int — input_ids length (decode offset)

_ALL_FILES = (
    ENCODER_LAST_HIDDEN,
    AUDIO_EMBEDS,
    INPUTS_EMBEDS,
    GREEDY_IDS,
    GREEDY_TEXT,
    PROMPT_LEN,
)


def load_golden(name: str) -> torch.Tensor:
    """Load a tensor artefact from :data:`GOLDEN_DIR` by short name."""
    return torch.load(GOLDEN_DIR / name, map_location="cpu")


def load_golden_text(name: str = GREEDY_TEXT) -> str:
    return (GOLDEN_DIR / name).read_text(encoding="utf-8")


def _all_exist() -> bool:
    return all((GOLDEN_DIR / f).exists() for f in _ALL_FILES)


def _fixture_wav() -> str:
    """Locate the shared audio fixture.

    Audio fixtures are gitignored, so they live in the main working tree
    (``starling``) rather than per-worktree. Resolve whichever exists.
    """
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
        print(f"[golden] all artefacts present in {GOLDEN_DIR}; skipping (force=True to recapture)")
        return _summarise_existing()

    print(f"[golden] loading eager model + processor from {MODEL_ID} ...")
    model, processor = load_model_and_processor(attn_impl="eager")
    components = get_components(model)
    encoder = components["encoder"]
    projector = components["projector"]

    print(f"[golden] loading sample audio {SAMPLE_AUDIO} ...")
    wav, sr = load_wav(_fixture_wav())
    inputs = build_inputs(processor, wav, sr=sr)
    input_ids = inputs["input_ids"]
    input_features = inputs["input_features"]
    input_features_mask = inputs.get("input_features_mask")
    prompt_len = int(input_ids.shape[1])

    dtype = model.dtype  # bfloat16

    with torch.inference_mode():
        # (1) Encoder last hidden state (packed valid-only sequence).
        enc_out = encoder(
            input_features=input_features,
            input_features_mask=input_features_mask,
            return_dict=True,
        )
        enc_lhs = enc_out.last_hidden_state
        torch.save(enc_lhs.cpu(), GOLDEN_DIR / ENCODER_LAST_HIDDEN)

        # (2) Projector output = get_audio_features(...).pooler_output.
        audio_out = model.get_audio_features(
            input_features=input_features,
            input_features_mask=input_features_mask,
            return_dict=True,
        )
        audio_embeds = audio_out.pooler_output
        torch.save(audio_embeds.cpu(), GOLDEN_DIR / AUDIO_EMBEDS)

        # (3) Merged multimodal inputs_embeds fed to the LLM. Capture via a
        # pre-hook on the language_model so we don't depend on private helpers.
        captured: dict[str, torch.Tensor] = {}

        def _llm_pre_hook(_module, args, kwargs):
            ie = kwargs.get("inputs_embeds", None)
            if ie is None and len(args) >= 1 and isinstance(args[0], torch.Tensor):
                ie = args[0]
            if ie is not None:
                captured["inputs_embeds"] = ie
            return None

        lm = components["language_model"]
        handle = lm.register_forward_pre_hook(_llm_pre_hook, with_kwargs=True)
        try:
            model(
                input_ids=input_ids,
                input_features=input_features,
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
        torch.save(torch.tensor(prompt_len), GOLDEN_DIR / PROMPT_LEN)

        # (4) Greedy generation (stock transformers generate()).
        gen = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
        )
        gen_new = gen[:, prompt_len:]
        torch.save(gen_new.cpu(), GOLDEN_DIR / GREEDY_IDS)
        try:
            text = processor.decode(gen_new, return_format="transcription_only")[0]
        except Exception:
            text = processor.batch_decode(gen_new, skip_special_tokens=True)[0]
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
