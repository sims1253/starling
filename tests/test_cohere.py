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
    from starling.cohere.pipeline import CohereMegaPipeline

    model, proc = _load()
    pipe = CohereMegaPipeline(model, proc, steps_per_replay=8)
    wav, sr = _load_wav(name)
    texts, ids = pipe.transcribe(wav.numpy(), language="en", max_new_tokens=300)
    golden_text = (GOLDEN_DIR / f"cohere_{name}_text.txt").read_text()
    # the first row transcript must match the golden (golden_text is row 0)
    assert texts[0] == golden_text, f"{name}: {texts[0]!r} != {golden_text!r}"


# --------------------------------------------------------------------------- #
# 5. shape bucketing (the leaderboard-RTFx fix)
# --------------------------------------------------------------------------- #
# Both captured graphs are shape-keyed, so diverse clip lengths re-captured on
# every clip (284-2300ms) instead of replaying (9-16ms). Two bucketing stages fix
# that, with different correctness properties:
#
#   cross_attn_bucketing (default ON)  -- pads the encoder OUTPUT S and masks the
#       padding out of cross-attention. Byte-exact: masked keys get exactly zero
#       softmax weight. Shares the decoder graph.
#   shape_bucketing (default OFF)      -- pads the MEL so the encoder graph is
#       shared too. Padding cannot leak into valid frames, but growing S retiles
#       the conformer's bf16 reductions, which flips near-tie greedy tokens on
#       3.4% of real clips (12/350). Byte-exact on these fixtures, NOT in general.


def test_bucket_mel_len_pads_to_grid_and_caps():
    """``_bucket_mel_len`` rounds up to the grid, caps at max_mel_frames, and
    no-ops when aligned/at-cap/disabled. Pure integer logic -- no model load."""
    from starling.cohere.pipeline import CohereMegaPipeline

    p = CohereMegaPipeline.__new__(CohereMegaPipeline)  # bypass __init__
    p.shape_bucketing = True
    p.mel_bucket_frames = 1024
    p.max_mel_frames = 3501

    assert p._bucket_mel_len(744) == 1024, "744 -> next 1024-multiple"
    assert p._bucket_mel_len(2231) == 3072, "2231 -> next 1024-multiple"
    assert p._bucket_mel_len(1024) == 1024, "already on the grid"
    # capped: 3100 would round to 4096, but the processor never emits >3501
    assert p._bucket_mel_len(3100) == 3501, "capped at max_mel_frames"
    assert p._bucket_mel_len(3501) == 3501, "at the cap -> unchanged"

    p.shape_bucketing = False
    assert p._bucket_mel_len(744) == 744, "disabled -> identity"


def test_maybe_bucket_preserves_valid_region():
    """Padding is zeros, the valid region is untouched, and the mask marks the
    padded frames invalid so the encoder masks them out."""
    import torch

    from starling.cohere.pipeline import CohereMegaPipeline

    p = CohereMegaPipeline.__new__(CohereMegaPipeline)
    p.shape_bucketing = True
    p.mel_bucket_frames = 1024
    p.max_mel_frames = 3501

    feat = torch.randn(1, 744, 128)
    amask = torch.ones(1, 744, dtype=torch.long)
    f2, m2 = p._maybe_bucket(feat, amask)
    assert f2.shape == (1, 1024, 128) and m2.shape == (1, 1024)
    assert torch.equal(f2[:, :744], feat), "valid mel must be preserved"
    assert f2[:, 744:].abs().sum().item() == 0.0, "mel padding must be zeros"
    assert int(m2[:, 744:].sum()) == 0, "mask must mark padding invalid"

    # already on the grid -> returned unchanged (no copy)
    on_grid = torch.randn(1, 1024, 128)
    on_mask = torch.ones(1, 1024, dtype=torch.long)
    f3, m3 = p._maybe_bucket(on_grid, on_mask)
    assert f3 is on_grid and m3 is on_mask, "on-grid input must not be copied"

    p.shape_bucketing = False
    f4, m4 = p._maybe_bucket(feat, amask)
    assert f4 is feat and m4 is amask, "disabled -> no copy"


@pytest.mark.skipif(not _FIXTURE_OK, reason="fixtures absent")
@pytest.mark.parametrize("name", ["short", "medium", "long"])
def test_cross_attn_bucketing_is_byte_exact(name):
    """Cross-attention bucketing must not change a single token id.

    Padding the encoder output on ``S`` and masking it with ``-inf`` gives those
    keys exactly zero softmax weight, so the decoder attends to precisely the
    frames it would have seen unbucketed. This is the pipeline's byte-exactness
    guarantee, and it holds on real leaderboard audio (0/100 clips diverged),
    not just on these fixtures.
    """
    import torch

    from starling.cohere.pipeline import CohereMegaPipeline

    model, proc = _load()
    audio = _load_wav(name)[0].numpy()

    pipe_off = CohereMegaPipeline(
        model, proc, steps_per_replay=8, cross_attn_bucketing=False
    )
    text_off, ids_off = pipe_off.transcribe(audio, language="en", max_new_tokens=300)
    del pipe_off
    torch.cuda.empty_cache()

    pipe_on = CohereMegaPipeline(
        model, proc, steps_per_replay=8, cross_attn_bucketing=True
    )
    text_on, ids_on = pipe_on.transcribe(audio, language="en", max_new_tokens=300)

    assert text_on == text_off, (
        f"[{name}] transcript drift:\n  unbucketed: {text_off!r}\n"
        f"  bucketed:   {text_on!r}"
    )
    assert torch.equal(ids_on, ids_off), (
        f"[{name}] token ids drift under cross-attention bucketing"
    )


@pytest.mark.skipif(not _FIXTURE_OK, reason="fixtures absent")
def test_cross_attn_bucketing_collapses_decoder_graphs():
    """Diverse clip lengths must share ONE decoder graph under the default config.

    This is the mechanism behind the RTFx fix: without bucketing, N distinct clip
    lengths capture N decoder graphs (each costing a full capture); with it they
    collapse onto the 128-frame encoder-output grid.
    """
    import numpy as np
    import torch

    from starling.cohere.pipeline import CohereMegaPipeline

    model, proc = _load()
    rng = np.random.default_rng(0)
    # five distinct lengths that all land in the same 128-frame encoder bucket
    clips = [rng.standard_normal(int(s * 16000)).astype("float32") * 0.05
             for s in (3.1, 3.6, 4.2, 5.0, 5.8)]

    pipe_off = CohereMegaPipeline(
        model, proc, steps_per_replay=8, cross_attn_bucketing=False
    )
    for c in clips:
        pipe_off.transcribe(c, language="en", max_new_tokens=8)
    n_dec_off = len(pipe_off._decoders)
    del pipe_off
    torch.cuda.empty_cache()

    pipe_on = CohereMegaPipeline(
        model, proc, steps_per_replay=8, cross_attn_bucketing=True
    )
    for c in clips:
        pipe_on.transcribe(c, language="en", max_new_tokens=8)
    n_dec_on = len(pipe_on._decoders)

    assert n_dec_off == len(clips), f"expected one decoder graph per clip, got {n_dec_off}"
    assert n_dec_on == 1, f"bucketed clips must share ONE decoder graph, got {n_dec_on}"


def _pipeline_inputs(pipe, audio):
    """Drive the pipeline's front half: processor -> encoder -> cross-attn prep."""
    import torch

    inp = pipe.processor(
        audio, sampling_rate=16000, language="en", return_tensors="pt"
    )
    feat = inp["input_features"].to(pipe.dtype).cuda()
    amask = inp["attention_mask"].cuda()
    dec_in = inp["decoder_input_ids"].cuda()
    with torch.inference_mode():
        s_nat = pipe._subsampled_len(int(feat.shape[1]))
        feat, amask = pipe._maybe_bucket(feat, amask)
        enc_h = pipe._encode(feat, amask)
        enc_h, enc_mask = pipe._prepare_cross(enc_h, s_nat)
    return dec_in, enc_h, enc_mask


@pytest.mark.skipif(not _FIXTURE_OK, reason="fixtures absent")
def test_prefill_rewinds_self_cache_so_reused_graphs_do_not_leak():
    """A reused decoder graph must prefill THIS utterance's prompt into [0, T).

    ``CohereAsrSelfAttention`` calls ``past_key_values.update(k, v, layer_idx)``
    with no ``cache_position``, so ``StaticLayer.update`` writes at the layer's
    ``cumulative_length``. After a decode that sits at ``T + n_generated``, so a
    prefill that does not rewind it scribbles the prompt K/V into
    ``[T+n, T+n+T)`` while the decode reads ``[0, T)`` -- which still holds the
    CAPTURE utterance's prompt. Only layer 0 survives (its prompt K/V depend on
    the token ids alone); layers i>0 derive theirs from layer i-1's
    cross-attention output and so carry the wrong clip's encoder state.

    This asserts the invariant directly rather than through decoded text: the
    end-to-end effect is a ~18% clip-level transcript flip on real audio, too
    sparse for three fixtures to catch reliably.
    """
    import torch

    from starling.cohere.pipeline import CohereMegaPipeline

    model, proc = _load()
    med = _load_wav("medium")[0].numpy()
    audio_a = _load_wav("short")[0].numpy()          # 7.4s
    audio_b = med[8 * 16000: 17 * 16000]             # different content, same bucket

    pipe = CohereMegaPipeline(model, proc, steps_per_replay=8)
    dec_in_a, enc_a, mask_a = _pipeline_inputs(pipe, audio_a)
    dec = pipe._get_decoder(dec_in_a, enc_a, mask_a)   # captures on clip A
    dec.decode(dec_in_a, enc_a, mask_a, max_new_tokens=32)  # advances the write head

    dec_in_b, enc_b, mask_b = _pipeline_inputs(pipe, audio_b)
    assert enc_b.shape == enc_a.shape, "test needs both clips in one encoder bucket"
    T = dec_in_b.shape[1]
    with torch.inference_mode():
        dec._prefill(dec_in_b, enc_b, mask_b, dec._cache)

    layers = dec._cache.self_attention_cache.layers
    cums = [int(layer.cumulative_length) for layer in layers]
    assert all(c == T for c in cums), (
        f"prefill left the self-attn write head at {cums[:3]}, expected {T}: "
        f"the prompt K/V did not land in slots [0, T)"
    )
    keys_reused = [layer.keys[:, :, :T].clone() for layer in layers]
    values_reused = [layer.values[:, :, :T].clone() for layer in layers]
    # Release the first decoder and its cached graph state before constructing
    # the fresh pipeline, so no objects from the first capture remain
    # referenced during the second.
    dec._cache = None
    del dec, layers, pipe
    torch.cuda.empty_cache()

    # same clip B, prefilled on a pristine cache -> the ground truth prompt K/V
    fresh = CohereMegaPipeline(model, proc, steps_per_replay=8)
    dec_in_b2, enc_b2, mask_b2 = _pipeline_inputs(fresh, audio_b)
    dec2 = fresh._get_decoder(dec_in_b2, enc_b2, mask_b2)  # captures on clip B
    keys_fresh = [layer.keys[:, :, :T].clone()
                  for layer in dec2._cache.self_attention_cache.layers]
    values_fresh = [layer.values[:, :, :T].clone()
                    for layer in dec2._cache.self_attention_cache.layers]

    for i, (reused, fresh_k) in enumerate(zip(keys_reused, keys_fresh)):
        assert torch.equal(reused, fresh_k), (
            f"layer {i}: prompt K on a reused graph differs from a fresh prefill "
            f"(maxdiff {(reused.float() - fresh_k.float()).abs().max():.3e}) -- "
            f"the previous utterance's decode state is leaking"
        )
    # Also verify cached values: clip A's V must not leak into clip B's decode.
    for i, (reused_v, fresh_v) in enumerate(zip(values_reused, values_fresh)):
        assert torch.equal(reused_v, fresh_v), (
            f"layer {i}: prompt V on a reused graph differs from a fresh prefill "
            f"(maxdiff {(reused_v.float() - fresh_v.float()).abs().max():.3e}) -- "
            f"stale values from the previous utterance are leaking"
        )


@pytest.mark.skipif(not _FIXTURE_OK, reason="fixtures absent")
def test_graphed_encoder_follows_shape_bucketing():
    """The encoder is graphed only when its input shapes are bucketed.

    Graphing an unbucketed encoder captures a fresh graph per clip -- the exact
    pathology this module documents -- so ``use_graphed_encoder`` defaults to
    ``shape_bucketing`` rather than to ``True``.
    """
    from starling.cohere.pipeline import CohereMegaPipeline

    model, proc = _load()
    assert CohereMegaPipeline(model, proc).encoder is None
    assert CohereMegaPipeline(model, proc, shape_bucketing=True).encoder is not None
    # explicit override still wins
    assert CohereMegaPipeline(model, proc, use_graphed_encoder=True).encoder is not None


@pytest.mark.skipif(not _FIXTURE_OK, reason="fixtures absent")
@pytest.mark.parametrize("name", ["short", "medium", "long"])
def test_mel_shape_bucketing_matches_on_fixtures(name):
    """Mel bucketing agrees with the natural-shape path on these fixtures.

    NOTE this is a *fixture-level* check, not a guarantee. Mel bucketing grows the
    post-subsampling length ``S``, which retiles the conformer's bf16 reductions;
    on the wider leaderboard corpus that flips near-tie greedy tokens on 3.4% of
    clips (12/350; per-dataset WER moves <= 0.18, both directions). Use
    ``shape_bucketing=False`` (the default) when byte-exactness matters.
    """
    import torch

    from starling.cohere.pipeline import CohereMegaPipeline

    model, proc = _load()
    audio = _load_wav(name)[0].numpy()

    pipe_off = CohereMegaPipeline(model, proc, steps_per_replay=8, shape_bucketing=False)
    text_off, _ = pipe_off.transcribe(audio, language="en", max_new_tokens=300)
    del pipe_off
    torch.cuda.empty_cache()

    pipe_on = CohereMegaPipeline(model, proc, steps_per_replay=8, shape_bucketing=True)
    text_on, _ = pipe_on.transcribe(audio, language="en", max_new_tokens=300)

    assert text_on == text_off, (
        f"[{name}] mel bucketing changed the transcript on a fixture:\n"
        f"  natural:  {text_off!r}\n  bucketed: {text_on!r}"
    )
