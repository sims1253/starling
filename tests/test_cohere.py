"""Correctness tests for the cohere-transcribe-03-2026 megakernel pipeline.

These verify byte-exactness vs the golden reference (captured by
``scripts/cohere_golden.py``) for each stage:

* the graphed encoder matches the stock eager encoder,
* the K-step graphed decoder reproduces the golden greedy token sequence,
* the end-to-end pipeline transcribe matches the golden transcript.

Golden references live in ``golden/cohere_{short,medium,long}_ids.pt`` / ``.txt``
(gitignored). Tests skip if the goldens or fixtures are absent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
GOLDEN_DIR = REPO / "golden"
FIXTURES = REPO / "tests" / "fixtures"


def _have_torch_cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


_SKIP_REASON = (
    None
    if _have_torch_cuda()
    else "CUDA / torch unavailable -- skipping cohere megakernel tests"
)
pytestmark = pytest.mark.skipif(
    _SKIP_REASON is not None, reason=_SKIP_REASON or ""
)

_FIXTURE_OK = (FIXTURES / "short.wav").exists()
_GOLDEN_OK = (GOLDEN_DIR / "cohere_short_ids.pt").exists()

# module-level model cache
_MODEL = None
_PROC = None


def _load():
    global _MODEL, _PROC
    if _MODEL is None:
        import sys

        sys.path.insert(0, str(REPO / "src"))
        from starling.cohere.loader import load_model_and_processor

        _MODEL, _PROC = load_model_and_processor()
    return _MODEL, _PROC


def _load_wav(name: str):
    import soundfile as sf
    import torch

    wav, sr = sf.read(str(FIXTURES / f"{name}.wav"))
    return torch.from_numpy(wav.astype("float32")), sr


def _proc_inputs(name: str, language: str = "en"):
    """Build (feat, amask, dec_in) processor outputs for a fixture (cached on the module)."""
    import torch

    model, proc = _load()
    wav, sr = _load_wav(name)
    inp = proc(wav.numpy(), sampling_rate=sr, language=language, return_tensors="pt")
    feat = inp["input_features"].to(torch.bfloat16).cuda()
    amask = inp["attention_mask"].cuda()
    dec_in = inp["decoder_input_ids"].cuda()
    return feat, amask, dec_in


# --------------------------------------------------------------------------- #
# 1. graphed encoder byte-exactness
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not (_FIXTURE_OK and _GOLDEN_OK), reason="fixtures/golden absent")
def test_encoder_graph_byte_exact():
    import torch
    from starling.cohere.encoder_graph import GraphedEncoder

    model, _ = _load()
    feat, amask, _ = _proc_inputs("short")
    enc = model.model.encoder
    ge = GraphedEncoder(enc, warmup_iters=2)
    # graphed
    g = ge(feat, amask)
    # eager reference
    with torch.inference_mode():
        e = enc(input_features=feat, attention_mask=amask).last_hidden_state
    diff = (g - e).abs().max().item()
    assert diff == 0.0, f"encoder graph diff {diff} != 0.0"


# --------------------------------------------------------------------------- #
# 2. K-step graphed decoder byte-exactness vs golden
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not (_FIXTURE_OK and _GOLDEN_OK), reason="fixtures/golden absent")
@pytest.mark.parametrize("name", ["short", "medium"])
@pytest.mark.parametrize("K", [1, 8])
def test_decoder_graph_byte_exact(name, K):
    import torch
    from starling.cohere.decode_mega import GraphedDecoder
    from starling.cohere.reference import encode

    model, _ = _load()
    feat, amask, dec_in = _proc_inputs(name)
    with torch.inference_mode():
        enc_h, enc_mask = encode(model, feat, amask)
    gd = GraphedDecoder(model, steps_per_replay=K, warmup_iters=2)
    gd.capture(dec_in, enc_h, enc_mask)
    ids = gd.decode(dec_in, enc_h, enc_mask, max_new_tokens=300)
    golden = torch.load(GOLDEN_DIR / f"cohere_{name}_ids.pt")
    # trim/pad to compare (golden is (B, n_golden); ids is (B, n_gen))
    n = min(ids.shape[1], golden.shape[1])
    assert ids.shape[0] == golden.shape[0], f"B mismatch {ids.shape} {golden.shape}"
    for b in range(ids.shape[0]):
        got = ids[b, :n].tolist()
        exp = golden[b, :n].tolist()
        assert got == exp, f"{name} K={K} row {b}: {got[:20]}... != {exp[:20]}..."


# --------------------------------------------------------------------------- #
# 3. reference greedy_generate matches HF generate() (the oracle itself)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _FIXTURE_OK, reason="fixtures absent")
def test_reference_matches_generate():
    """The manual eager reference must reproduce model.generate() byte-for-byte."""
    import torch
    from starling.cohere.reference import encode, greedy_generate

    model, proc = _load()
    feat, amask, dec_in = _proc_inputs("short")
    with torch.inference_mode():
        enc_h, enc_mask = encode(model, feat, amask)
        ids_ref = greedy_generate(model, enc_h, enc_mask, dec_in, max_new_tokens=60)
        # HF generate
        gen = model.generate(
            input_features=feat, attention_mask=amask,
            decoder_input_ids=dec_in, max_length=dec_in.shape[1] + 60,
        )
    # generated = gen minus the prompt prefix
    gen_new = gen[:, dec_in.shape[1]:]
    n = min(ids_ref.shape[1], gen_new.shape[1])
    assert ids_ref[0, :n].tolist() == gen_new[0, :n].tolist(), (
        f"reference {ids_ref[0,:n].tolist()[:20]} != generate {gen_new[0,:n].tolist()[:20]}"
    )


# --------------------------------------------------------------------------- #
# 4. end-to-end pipeline transcribe byte-exactness vs golden transcript
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not (_FIXTURE_OK and _GOLDEN_OK), reason="fixtures/golden absent")
@pytest.mark.parametrize("name", ["short", "medium"])
def test_pipeline_transcribe_byte_exact(name):
    import torch
    from starling.cohere.pipeline import CohereMegaPipeline

    model, proc = _load()
    pipe = CohereMegaPipeline(model, proc, steps_per_replay=8)
    wav, sr = _load_wav(name)
    texts, ids = pipe.transcribe(wav.numpy(), language="en", max_new_tokens=300)
    golden_text = (GOLDEN_DIR / f"cohere_{name}_text.txt").read_text()
    # the first row transcript must match the golden (golden_text is row 0)
    assert texts[0] == golden_text, f"{name}: {texts[0]!r} != {golden_text!r}"
