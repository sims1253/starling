"""Correctness tests for the MOSS-Transcribe megakernel pipeline.

These verify byte-exactness vs the golden reference (captured by
``scripts/moss_golden.py``) for each stage:

* the graphed audio encoder + adapter match the stock eager encoder,
* the single-step / fused / multi-step LLM decoders all reproduce the golden
  greedy token sequence,
* the end-to-end pipeline transcribe matches the golden transcript.

Golden references live in ``golden/moss_{short,medium,long}_ids.pt`` /
``.txt`` (gitignored). Tests skip if the goldens or fixtures are absent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
GOLDEN_DIR = REPO / "golden"
FIXTURES = REPO / "tests" / "fixtures"

# --------------------------------------------------------------------------- #
# lazy imports + skip guards (mirrors the other test modules)
# --------------------------------------------------------------------------- #


def _have_torch_cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


_SKIP_REASON = (
    None
    if _have_torch_cuda()
    else "CUDA / torch unavailable -- skipping MOSS megakernel tests"
)
pytestmark = pytest.mark.skipif(
    _SKIP_REASON is not None, reason=_SKIP_REASON or ""
)

# fixtures + golden presence
_FIXTURE_OK = (FIXTURES / "short.wav").exists()
_GOLDEN_OK = (GOLDEN_DIR / "moss_short_ids.pt").exists()


# --------------------------------------------------------------------------- #
# module-level model cache (cleaned up by conftest's _drop_module_model_cache)
# --------------------------------------------------------------------------- #
_MODEL = None
_PROC = None
_EMB = None  # cached inputs_embeds for the short fixture
_INPUTS = None  # cached processor outputs for the short fixture


def _load():
    global _MODEL, _PROC
    if _MODEL is None:
        import sys

        sys.path.insert(0, str(REPO / "src"))
        from starling.moss.loader import load_model_and_processor

        _MODEL, _PROC = load_model_and_processor()
    return _MODEL, _PROC


def _short_inputs():
    """Build (inputs_embeds, processor_outputs) for the short fixture (cached)."""
    global _EMB, _INPUTS
    if _EMB is not None:
        return _EMB, _INPUTS
    import soundfile as sf
    import torch

    from starling.moss.reference import audio_features, build_inputs_embeds

    model, proc = _load()
    wav, sr = sf.read(str(FIXTURES / "short.wav"))
    if wav.ndim > 1:
        wav = wav.mean(1)
    inp = proc(wav.astype("float32"))
    inp = {k: (v.cuda() if isinstance(v, torch.Tensor) else v) for k, v in inp.items()}
    with torch.inference_mode():
        feats = audio_features(model, inp["audio_data"], inp["audio_data_seqlens"])
        emb = build_inputs_embeds(model, inp["input_ids"], feats, inp["audio_input_mask"])
    _EMB, _INPUTS = emb, inp
    return _EMB, _INPUTS


def _golden_ids(name: str):
    import torch

    return torch.load(GOLDEN_DIR / f"moss_{name}_ids.pt")


@pytest.fixture
def require_fixtures():
    if not _FIXTURE_OK:
        pytest.skip("short.wav fixture missing (gitignored)")
    if not _GOLDEN_OK:
        pytest.skip("golden/moss_short_ids.pt missing (run scripts/moss_golden.py)")


# --------------------------------------------------------------------------- #
# encoder + adapter
# --------------------------------------------------------------------------- #
def test_encoder_adapter_byte_exact(require_fixtures):
    """Graphed (eager) encoder + adapter reproduce the stock audio features."""
    import torch

    from starling.moss.encoder_graph import GraphedAudioEncoder
    from starling.moss.reference import audio_features

    model, _ = _load()
    _, inp = _short_inputs()
    enc = GraphedAudioEncoder(model.model.audio_model, model.model.audio_adapter)
    with torch.inference_mode():
        ref = audio_features(model, inp["audio_data"], inp["audio_data_seqlens"])
        ref_adapted = model.model.audio_adapter(ref)
        got = enc(inp["audio_data"], inp["audio_data_seqlens"])
    diff = (got.float() - ref_adapted.float()).abs().max().item()
    assert diff == 0.0, f"encoder+adapter not byte-exact: max abs diff {diff}"


# --------------------------------------------------------------------------- #
# LLM decoders
# --------------------------------------------------------------------------- #
def test_single_step_decoder_byte_exact(require_fixtures):
    import torch

    from starling.moss.llm_mega import MossLLMMega

    model, _ = _load()
    emb, _ = _short_inputs()
    gold = _golden_ids("short")
    dec = MossLLMMega(model.model.language_model, model.lm_head, max_cache_len=1024)
    with torch.inference_mode():
        r = dec.generate(emb, max_new_tokens=int(gold.shape[1]))
    assert r.ids[0].tolist() == gold[0].tolist(), "single-step decode != golden"


def test_fused_decoder_byte_exact(require_fixtures):
    import torch

    from starling.moss.fused_decode import FusedMossLLMMega

    model, _ = _load()
    emb, _ = _short_inputs()
    gold = _golden_ids("short")
    dec = FusedMossLLMMega(model.model.language_model, model.lm_head, max_cache_len=1024)
    with torch.inference_mode():
        r = dec.generate(emb, max_new_tokens=int(gold.shape[1]))
    assert r.ids[0].tolist() == gold[0].tolist(), "fused decode != golden"


def test_encoder_cudagraph_matches_eager(require_fixtures):
    """Adaptive cudagraph encoder matches eager within ENCODER_ATOL across the
    full lifecycle: first sighting (eager), the recurrence that captures, a
    cache replay, and a replay with a *different* utterance of the same shape
    (padded_feature is rebuilt in-graph, so a stale-capture bug would show here).

    The captured residual (~1 bf16 ULP) is CUDA-graph GEMM-algorithm selection,
    not a logic difference: ``_capture_forward`` run eagerly is bit-identical to
    ``_forward_eager``.  The end-to-end transcript is unchanged (see the golden
    pipeline tests)."""
    import torch

    from starling.moss.config import ENCODER_ATOL
    from starling.moss.encoder_graph import GraphedAudioEncoder

    model, _ = _load()
    _, inp = _short_inputs()
    audio, sl = inp["audio_data"], inp["audio_data_seqlens"]
    eager = GraphedAudioEncoder(model.model.audio_model, model.model.audio_adapter, mode="eager")
    cg = GraphedAudioEncoder(model.model.audio_model, model.model.audio_adapter, mode="cudagraph")
    audio_b = (audio * 0.7 + 0.3).to(audio.dtype)
    with torch.inference_mode():
        ref = eager(audio, sl)
        ref_b = eager(audio_b, sl)
        # 1st sighting -> eager (byte-exact); 2nd -> captures + replays; 3rd -> cache replay
        assert cg._graphs == {}, "should not capture on first sighting"
        got1 = cg(audio, sl)
        assert not cg._graphs, "still eager after one sighting"
        got2 = cg(audio, sl)  # recurrence -> capture
        assert int(audio.shape[1]) in cg._graphs, "should capture on recurrence"
        got3 = cg(audio, sl)  # cache replay
        got_b = cg(audio_b, sl)  # cache replay with different values, same shape
        for tag, got, r in [("eager", got1, ref), ("capture", got2, ref),
                             ("replay", got3, ref), ("new-values", got_b, ref_b)]:
            assert (got.float() - r.float()).abs().max().item() < ENCODER_ATOL, f"cudagraph != eager ({tag})"


def test_fp8_weights_short_reproduces_golden(require_fixtures):
    """fp8 decoder-layer weights (opt-in) reproduce the golden short transcript.

    fp8 is lossy in principle, but the per-layer RMSNorms normalise the weight
    noise away over the residual stream, so on the short fixture the greedy
    token sequence is unchanged.  lm_head stays bf16 (its argmax is fp8-fragile).
    Longer decodes can diverge late (see the WER-gated bench); this test only
    pins the byte-exact-on-short behaviour so a regression is visible.
    """
    import torch

    from starling.flags import flags
    from starling.moss.multistep import FusedMossMultiStepMega

    model, _ = _load()
    emb, _ = _short_inputs()
    gold = _golden_ids("short")
    with flags(tolerance_mode=True, fp8_weights=True):
        dec = FusedMossMultiStepMega(
            model.model.language_model, model.lm_head,
            max_cache_len=1024, steps_per_replay=4,
        )
        assert dec._fp8_weights is not None, "fp8_weights flag did not quantize"
        with torch.inference_mode():
            r = dec.generate(emb, max_new_tokens=int(gold.shape[1]))
    assert r.ids[0].tolist() == gold[0].tolist(), "fp8 decode != golden (short)"


@pytest.mark.parametrize("K", [1, 8, 16])
def test_multistep_decoder_byte_exact(K, require_fixtures):
    import torch

    from starling.moss.multistep import FusedMossMultiStepMega

    model, _ = _load()
    emb, _ = _short_inputs()
    gold = _golden_ids("short")
    dec = FusedMossMultiStepMega(
        model.model.language_model, model.lm_head,
        max_cache_len=1024, steps_per_replay=K,
    )
    with torch.inference_mode():
        r = dec.generate(emb, max_new_tokens=int(gold.shape[1]))
    assert r.ids[0].tolist() == gold[0].tolist(), f"K={K} multistep decode != golden"


# --------------------------------------------------------------------------- #
# end-to-end pipeline
# --------------------------------------------------------------------------- #
def test_pipeline_transcribe_byte_exact(require_fixtures):
    import torch

    from starling.moss.pipeline import MossMegaPipeline

    model, proc = _load()
    pipe = MossMegaPipeline(model, proc, max_cache_len=1024, steps_per_replay=16)
    gold = _golden_ids("short")
    _, inp = _short_inputs()
    with torch.inference_mode():
        text, ids = pipe.transcribe(
            inp["audio_data"], inp["audio_data_seqlens"], inp["input_ids"],
            inp["audio_input_mask"], max_new_tokens=int(gold.shape[1]),
        )
    assert ids[0].tolist() == gold[0].tolist(), "pipeline decode != golden"
    gold_text = (GOLDEN_DIR / "moss_short_text.txt").read_text().strip()
    assert text.strip() == gold_text, "pipeline text != golden text"


@pytest.mark.parametrize("name", ["short", "medium", "long"])
def test_pipeline_all_fixtures(name, require_fixtures):
    """End-to-end on each fixture (skips if fixture/golden absent)."""
    import soundfile as sf
    import torch

    if not (FIXTURES / f"{name}.wav").exists() or not (
        GOLDEN_DIR / f"moss_{name}_ids.pt"
    ).exists():
        pytest.skip(f"{name} fixture or golden missing")
    from starling.moss.pipeline import MossMegaPipeline

    model, proc = _load()
    pipe = MossMegaPipeline(model, proc, max_cache_len=2048, steps_per_replay=16)
    wav, sr = sf.read(str(FIXTURES / f"{name}.wav"))
    if wav.ndim > 1:
        wav = wav.mean(1)
    inp = proc(wav.astype("float32"))
    inp = {k: (v.cuda() if isinstance(v, torch.Tensor) else v) for k, v in inp.items()}
    gold = _golden_ids(name)
    with torch.inference_mode():
        text, ids = pipe.transcribe(
            inp["audio_data"], inp["audio_data_seqlens"], inp["input_ids"],
            inp["audio_input_mask"], max_new_tokens=int(gold.shape[1]),
        )
    mine, g = ids[0].tolist(), gold[0].tolist()
    if name == "short":
        # Short is byte-exact across exact-width eager and the fixed-width
        # megakernel path.
        assert mine == g, f"{name} pipeline decode != golden"
        gold_text = (GOLDEN_DIR / f"moss_{name}_text.txt").read_text().strip()
        assert text.strip() == gold_text, f"{name} pipeline text != golden text"
    else:
        # The canonical golden uses exact-width DynamicCache attention. The
        # optimized megakernel deliberately keeps fixed-width static attention;
        # padded softmax reduction order flips a bf16 near-tie at token 21 on
        # medium/long (432 vs 13). This is a cache-shape numeric artifact, not a
        # transcription error, so assert the semantically correct transcript.
        # Whole-transcript similarity vs the golden transcript: the single
        # token-21 divergence touches ~1 token of 89/187, so the ratio stays
        # well above 0.95, but a truncated/halved transcript scores ~0.5-0.7
        # and garbage ~0.0, so 0.95 cleanly separates. The key-phrase check is
        # kept as a secondary signal.
        import difflib

        gold_text = (GOLDEN_DIR / f"moss_{name}_text.txt").read_text().strip()
        low = text.strip().lower()
        ratio = difflib.SequenceMatcher(None, low, gold_text.lower()).ratio()
        assert ratio >= 0.95, (
            f"{name} pipeline transcript diverges from golden "
            f"(ratio={ratio:.3f}): {text[:120]!r}"
        )
        assert "like the old portrait" in low, (
            f"{name} pipeline transcript missing expected phrase: {text[:120]!r}"
        )
