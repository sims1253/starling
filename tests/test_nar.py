"""Correctness gate for the starling.nar (Granite-Speech-4.1-2b-NAR) megakernel.

The NAR pipeline is a single non-autoregressive forward. These tests verify the
``NarMega`` pipeline reproduces the golden reference (captured from the stock
eager ``model.transcribe``) byte-exactly on the short/medium/long fixture tiers,
plus stage-level byte-exactness for the graph-safe encoder trunk.

Run with:  uv run python -m pytest tests/test_nar.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

import torch  # noqa: E402

from starling.nar import NarMega  # noqa: E402
from starling.nar.golden import load_golden, load_golden_json, TIERS  # noqa: E402

# Cache the model + mega + per-tier inputs across tests (one ~2GB model).
_STATE = None

FIXTURES = {
    "short": _REPO_ROOT / "tests" / "fixtures" / "short.wav",
    "medium": _REPO_ROOT / "tests" / "fixtures" / "medium.wav",
    "long": _REPO_ROOT / "tests" / "fixtures" / "long.wav",
}


def _get_state():
    """Load model + processor + NarMega once, build per-tier inputs."""
    global _STATE
    if _STATE is not None:
        return _STATE
    import soundfile as sf
    from transformers import AutoModel, AutoProcessor

    from starling.nar.config import MODEL_ID

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    ).to("cuda")
    model.eval()
    mega = NarMega(model)

    tiers = {}
    for tier, path in FIXTURES.items():
        wav, sr = sf.read(str(path))
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        wav_t = torch.from_numpy(wav).float().to("cuda")
        inputs = processor(audios=wav_t, device="cuda")
        feats = inputs["input_features"].to(torch.bfloat16)
        attn = inputs["attention_mask"]
        tiers[tier] = {"feats": feats, "attn": attn, "dur": len(wav) / sr}

    _STATE = {"model": model, "processor": processor, "mega": mega, "tiers": tiers}
    return _STATE


@pytest.fixture(scope="module")
def state():
    return _get_state()


# --------------------------------------------------------------------------- #
# End-to-end byte-exactness vs the golden reference (all tiers)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("tier", TIERS)
def test_transcribe_byte_exact(tier, state):
    """The full NarMega.transcribe matches the golden token ids exactly."""
    st = state["tiers"][tier]
    mega = state["mega"]
    with torch.inference_mode():
        # Warmup: triggers torch.compile + graph capture (first call is slow).
        mega.transcribe(st["feats"], st["attn"])
        preds, _ = mega.transcribe(st["feats"], st["attn"])
    golden = load_golden(f"{tier}_preds.pt").tolist()
    assert preds[0] == golden, (
        f"{tier}: decoded tokens differ from golden "
        f"(got {len(preds[0])} tok, golden {len(golden)})"
    )


@pytest.mark.parametrize("tier", TIERS)
def test_transcribe_text_matches(tier, state):
    """The decoded transcript string matches the golden text."""
    st = state["tiers"][tier]
    mega = state["mega"]
    processor = state["processor"]
    with torch.inference_mode():
        mega.transcribe(st["feats"], st["attn"])  # ensure captured
        preds, _ = mega.transcribe(st["feats"], st["attn"])
    text = processor.batch_decode([preds[0]])[0]
    assert text == load_golden_json()[tier]["text"]


# --------------------------------------------------------------------------- #
# Encoder trunk byte-exactness (0.0 diff vs stock)
# --------------------------------------------------------------------------- #
def test_encoder_trunk_byte_exact(state):
    """The graph-safe encoder trunk + BPE head is byte-exact vs the stock encoder.

    Verified on the short tier (smallest; the trunk is the model's own ops so
    byte-exactness holds for every shape).
    """
    from starling.nar.mega import _encoder_bpe_head, _encoder_trunk

    st = state["tiers"]["short"]
    encoder = state["model"].encoder
    feats, attn = st["feats"], st["attn"]
    with torch.inference_mode():
        stock = encoder(
            input_features=feats, attention_mask=attn, output_hidden_states=True
        )
        h, _ah, bp = _encoder_trunk(encoder, feats, attn)
        bpe = _encoder_bpe_head(encoder, h, bp, attn)
    assert torch.equal(bpe, stock.logits), "BPE logits differ from stock"
    assert torch.equal(h, stock.last_hidden_state), (
        "encoder last hidden differs from stock"
    )


# --------------------------------------------------------------------------- #
# Pipeline config sanity
# --------------------------------------------------------------------------- #
def test_pipeline_deterministic(state):
    """Two consecutive transcribes (post-warmup) return identical tokens.

    Guards against any latent statefulness in the graph replay path.
    """
    st = state["tiers"]["medium"]
    mega = state["mega"]
    with torch.inference_mode():
        mega.transcribe(st["feats"], st["attn"])  # warmup
        p1, _ = mega.transcribe(st["feats"], st["attn"])
        p2, _ = mega.transcribe(st["feats"], st["attn"])
    assert p1 == p2, "pipeline is non-deterministic across replays"
