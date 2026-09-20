"""CPU unit tests + GPU-gated parity tests for the Voxtral Realtime pipeline.

CPU section (runs on any box): conv/frame arithmetic and token accounting,
the stock ``ceil(mel/8)`` bound vs the exact conv-chain count, ada
precompute math vs the per-step formula on tiny scaled-down modules, and
prompt/bookkeeping invariants (slice bounds, generation cap).

GPU section (skipped without CUDA / model / golden): byte-exact
fast-vs-slow loop transcripts and parity vs stock ``generate`` on the
short/medium/long fixtures against ``golden/voxtral_reference.json``.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from starling.voxtral import audio as vaudio
from starling.voxtral import config as vcfg
from starling.voxtral.pipeline import (
    _FrozenAdaMod,
    generation_cap,
    slice_bounds,
)

_GPU = torch.cuda.is_available()

# ---------------------------------------------------------------------------
# CPU: conv / frame arithmetic
# ---------------------------------------------------------------------------


def test_conv1_preserves_length():
    """conv1 (k3 s1, left-pad 2) maps L -> L for the lengths that matter."""
    for mel_T in (1, 2, 7, 8, 9, 16, 17, 100, 1000, 3751):
        assert vaudio.conv1_out_len(mel_T) == mel_T


def test_conv2_halves_floor():
    """conv2 (k3 s2, left-pad 1) maps L -> floor(L / 2)."""
    for l_in, want in ((1, 0), (2, 1), (3, 1), (4, 2), (7, 3), (8, 4), (3751, 1875)):
        assert vaudio.conv2_out_len(l_in) == want


def test_offline_mel_is_multiple_of_eight():
    """Offline mel lengths are multiples of 8 across realistic clip durations.

    The stock extractor's ``stft[..., :-1]`` drops the last time frame,
    cancelling the center=True +1 (verified against the real feature
    extractor; the fixture mel lengths are 1136/2624/7832).
    """
    for seconds in (0.5, 1.0, 7.43, 22.3, 30.0, 74.35, 120.0):
        n = int(round(seconds * vcfg.SAMPLE_RATE))
        assert vaudio.mel_frames(n) % 8 == 0
    for n_samples, mel in ((118960, 1136), (356880, 2624), (1189600, 7832)):
        assert vaudio.mel_frames(n_samples) == mel


def test_offline_padding_matches_mistral_common_formula():
    """Padded length = whole-token body + 32 left + 17 right pad tokens."""
    n = 7 * 16000 + 123
    unit = vcfg.RAW_SAMPLES_PER_AUDIO_TOK
    body = ((n + unit - 1) // unit) * unit
    assert vaudio.offline_padded_samples(n) == body + 49 * unit


@pytest.mark.parametrize("mel_T", [8, 16, 80, 800, 1136, 2624, 7832])
def test_token_accounting_consistent_where_it_matters(mel_T):
    """Conv-chain tokens == mel//8 == the stock bound, exactly.

    Offline mel lengths are multiples of 8: no partial projector group, no
    dropped frames, and the stock ``ceil(mel/8)`` bound equals the exact
    conv-chain count (delta 0). Lengths that are not multiples of 8 are
    synthetic only (never produced offline); the delta is explicit there.
    """
    assert mel_T % 8 == 0
    info = vaudio.check_mel_accounting(mel_T)
    assert info["conv1"] == mel_T
    assert info["conv2"] == mel_T // 2
    assert info["conv2"] % 4 == 0, "no partial group offline: reshape drops nothing"
    assert info["tokens"] == mel_T // 8
    assert info["delta"] == 0


def test_non_multiple_of_eight_delta_is_explicit():
    """Synthetic non-multiple-of-8 lengths show where padding makes counts differ."""
    info = vaudio.check_mel_accounting(100)
    # conv2 = 50 -> 12 tokens (2 frames dropped by the reshape); bound = 13.
    assert (info["tokens"], info["stock_bound"]) == (12, 13)
    assert info["delta"] == 1  # the dropped frames are the risk


# ---------------------------------------------------------------------------
# CPU: ada precompute math vs the per-step formula (tiny scaled-down modules)
# ---------------------------------------------------------------------------


def _tiny_ada_layers(n_layers=2, hidden=16, bottleneck=4):
    """Stock-shaped ada linears (Linear->GELU->Linear, no bias) at toy scale."""
    mods = torch.nn.ModuleList()
    for _ in range(n_layers):
        mods.append(
            torch.nn.Sequential(
                torch.nn.Linear(hidden, bottleneck, bias=False),
                torch.nn.GELU(),
                torch.nn.Linear(bottleneck, hidden, bias=False),
            )
        )
    return mods


def test_ada_precompute_matches_per_step_formula():
    """Frozen precomputed modulation equals recomputing ada(t_cond) per step."""
    torch.manual_seed(0)
    hidden, bottleneck, n_layers = 16, 4, 2
    layers = _tiny_ada_layers(n_layers, hidden, bottleneck)
    t_cond = torch.randn(1, hidden)
    h = torch.randn(1, 1, hidden)

    mods = [ada(t_cond) for ada in layers]
    for ada, mod in zip(layers, mods):
        frozen = _FrozenAdaMod(mod)
        assert torch.equal(frozen(t_cond), ada(t_cond))
        assert torch.equal(h * (1 + frozen(t_cond)), h * (1 + ada(t_cond)))


def test_ada_modulation_scales_mlp_branch_only():
    """Modulation multiplies the post-norm branch; residuals pass through."""
    torch.manual_seed(1)
    h = torch.randn(1, 1, 8)
    mod = torch.randn(1, 8) * 0.1
    out = h * (1 + _FrozenAdaMod(mod)(torch.zeros(1, 8)))
    assert torch.equal(out, h * (1 + mod))
    assert not torch.equal(out, h)


# ---------------------------------------------------------------------------
# CPU: prompt / bookkeeping invariants
# ---------------------------------------------------------------------------


def test_slice_bounds_match_prepare_inputs_for_generation():
    """Token range [c, c+k) reads embeds [4c, 4(c+k)): fixed 4-per-token."""
    assert slice_bounds(0, 10) == (0, 40)
    assert slice_bounds(7, 1) == (28, 32)
    assert slice_bounds(100, 3) == (400, 412)


def test_generation_cap_mirrors_stock_length_logic():
    """None -> pure stock bound; budgets clamp down to it, never exceed."""
    assert generation_cap(38, 3751) == 469  # ceil(3751/8), no user budget
    assert generation_cap(38, 3751, 200) == 238  # prompt + budget under bound
    assert generation_cap(38, 3751, 10000) == 469  # clamped to the bound
    assert generation_cap(38, 3751, 0) == 38  # zero budget stops immediately


def test_config_dims_internally_consistent():
    """Projector/ada/token-rate constants agree with the sub-module dims."""
    assert vcfg.PROJECTOR_IN_DIM == vcfg.ENCODER_HIDDEN * vcfg.DOWNSAMPLE_FACTOR
    assert vcfg.PROJECTOR_HIDDEN == vcfg.LLM_HIDDEN_SIZE == vcfg.TIME_EMBEDDING_DIM
    assert vcfg.RAW_SAMPLES_PER_AUDIO_TOK == vcfg.HOP_LENGTH * vcfg.AUDIO_LENGTH_PER_TOK
    assert (
        vcfg.STREAMING_RIGHT_PAD_TOKENS
        == vcfg.DEFAULT_NUM_DELAY_TOKENS + 1 + vcfg.STREAMING_BUFFER_TOKENS
    )
    assert vcfg.LLM_VOCAB_SIZE == 131072
    assert vcfg.ENCODER_NUM_LAYERS == 32 and vcfg.LLM_NUM_LAYERS == 26


def test_get_components_rejects_partial_models():
    """get_components raises when any required submodule is missing."""
    from starling.voxtral.loader import get_components

    class _Partial:
        pass

    with pytest.raises(AttributeError):
        get_components(_Partial())


def test_prewarm_stub_raises():
    """prewarm is an explicit stub until the graphed decode lands."""
    from starling.voxtral.pipeline import VoxtralPipeline

    pipe = VoxtralPipeline.__new__(VoxtralPipeline)
    with pytest.raises(NotImplementedError):
        pipe.prewarm()


def test_pad_slice_zero_pads_short_reads():
    """Synthetic short embeds are zero-padded to a multiple of 4 frames."""
    from starling.voxtral.pipeline import VoxtralPipeline

    pipe = VoxtralPipeline.__new__(VoxtralPipeline)
    all_embeds = torch.ones(1, 10, 4)
    piece, rows = pipe._pad_slice_to_rows(all_embeds, 8, 16)
    assert piece.shape == (1, 8, 4)
    assert rows == 2
    assert torch.equal(piece[:, :2, :], torch.ones(1, 2, 4))
    assert piece[:, 2:, :].abs().sum().item() == 0.0


# ---------------------------------------------------------------------------
# GPU-gated parity tests (skip cleanly without CUDA / golden / model)
# ---------------------------------------------------------------------------

from pathlib import Path as _Path  # noqa: E402

REPO_ROOT = _Path(__file__).resolve().parents[1]
GOLDEN_PATH = REPO_ROOT / "golden" / "voxtral_reference.json"
FIXTURES = REPO_ROOT / "tests" / "fixtures"

needs_gpu = pytest.mark.skipif(not _GPU, reason="CUDA required for voxtral parity")


def _load_golden():
    import json

    if not GOLDEN_PATH.exists():
        pytest.skip(f"golden {GOLDEN_PATH} missing; run scripts/make_voxtral_golden.py")
    with open(GOLDEN_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def fast_pipe():
    from starling.voxtral.pipeline import VoxtralPipeline

    return VoxtralPipeline.from_pretrained(use_precomputed_ada=True)


@pytest.fixture(scope="module")
def slow_pipe(fast_pipe):
    from starling.voxtral.pipeline import VoxtralPipeline

    # Both wrappers run sequentially; _frozen_ada restores the shared modules.
    return VoxtralPipeline(
        fast_pipe.model, fast_pipe.processor, use_precomputed_ada=False
    )


def _wav(name: str):
    import numpy as np
    import soundfile as sf

    path = FIXTURES / f"{name}.wav"
    if not path.exists():
        pytest.skip(f"fixture {path} not found")
    wav, sr = sf.read(str(path))
    if getattr(wav, "ndim", 1) > 1:
        wav = wav[:, 0]
    assert sr == vcfg.SAMPLE_RATE
    return np.ascontiguousarray(wav, dtype=np.float32)


@needs_gpu
@pytest.mark.parametrize("fixture", ["short", "medium", "long"])
def test_fast_vs_slow_loop_byte_exact(fast_pipe, slow_pipe, fixture):
    """Precomputed-ada fast path matches the stock-forward slow path."""
    wav = _wav(fixture)
    fast_text, fast_ids = fast_pipe.transcribe(wav)
    slow_text, slow_ids = slow_pipe.transcribe(wav)
    assert fast_ids[0].tolist() == slow_ids[0].tolist()
    assert fast_text == slow_text


@needs_gpu
@pytest.mark.parametrize("fixture", ["short", "medium", "long"])
def test_parity_vs_stock_generate(fast_pipe, fixture):
    """Fast loop matches stock generate ids and the golden reference."""
    golden = _load_golden()
    if fixture not in golden.get("fixtures", {}):
        pytest.skip(f"golden has no entry for {fixture!r}")
    wav = _wav(fixture)
    text, ids = fast_pipe.transcribe(wav)
    stock_text, stock_ids = fast_pipe.transcribe_stock(wav)
    assert ids[0].tolist() == stock_ids[0].tolist()
    assert text == stock_text
    entry = golden["fixtures"][fixture]
    assert ids[0].tolist() == entry["ids"]
    assert text == entry["text"]


@pytest.mark.parametrize("fast", [True, False])
@pytest.mark.parametrize("budget", [0, -1])
def test_zero_budget_skips_model_forward(fast, budget):
    import numpy as np
    from starling.voxtral.pipeline import VoxtralPipeline

    pipe = VoxtralPipeline.__new__(VoxtralPipeline)
    pipe.use_precomputed_ada = fast
    pipe.max_cache_len = 4096
    pipe._prepare_batch = lambda wav: {
        "input_ids": torch.ones(1, 39, dtype=torch.int64),
        "input_features": torch.zeros(1, 128, 392),
    }
    class Tokenizer:
        def decode(self, ids, **kwargs):
            assert ids == []
            return ""
    pipe.tokenizer = Tokenizer()
    text, ids = pipe.transcribe(np.zeros(1, dtype=np.float32), max_new_tokens=budget)
    assert text == "" and ids.shape == (1, 0)


def test_golden_capture_with_waveform_reader(monkeypatch, tmp_path):
    """Exercise the capture entry point with its real waveform-only contract."""
    import importlib.util
    import json
    import numpy as np
    from starling.voxtral.pipeline import VoxtralPipeline

    spec = importlib.util.spec_from_file_location(
        "make_voxtral_golden", REPO_ROOT / "scripts" / "make_voxtral_golden.py"
    )
    capture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(capture)

    class FakePipe:
        model = type("Model", (), {"generate": lambda self, **kwargs: None})()
        def _read_wav_or_array(self, path):
            return np.zeros(16000, dtype=np.float32)
        def _prepare_batch(self, wav):
            assert wav.shape == (16000,)
            return {"input_ids": torch.ones(1, 39),
                    "input_features": torch.zeros(1, 128, 496),
                    "num_delay_tokens": 6}
        def transcribe_stock(self, path):
            return " test ", torch.tensor([[123, 2]])

    monkeypatch.setattr(VoxtralPipeline, "from_pretrained", lambda: FakePipe())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    capture.GOLDEN_PATH = tmp_path / "reference.json"
    assert capture.main() == 0
    entries = json.loads(capture.GOLDEN_PATH.read_text())["fixtures"]
    assert set(entries) == {"short", "medium", "long"}
    assert entries["short"]["seconds"] == 1.0
    assert entries["short"]["ids"] == [123, 2]
    assert entries["short"]["text"] == " test "
