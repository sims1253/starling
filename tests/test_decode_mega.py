"""Byte-exact correctness tests for the parakeet TDT decoders (eager + graphed).

Both the hand-rolled eager decoder (:mod:`decode_eager`) and the CUDA-graph-
captured decoder (:mod:`decode_mega`) must reproduce the deterministic greedy
transcript in ``outputs/oracle.json`` BYTE-FOR-BYTE on the short/medium/long
fixtures.

Run with:  uv run pytest tests/test_decode_mega.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "tests" / "fixtures"))
sys.path.insert(0, str(_REPO_ROOT / "src"))

import make_fixtures as mkfx  # noqa: E402

ORACLE_PATH = _REPO_ROOT / "outputs" / "oracle.json"

# Loading the model is expensive (~25s); cache it across all tests in the module.
_STATE: dict = {}


def _get_model_and_processor():
    if not _STATE:
        import torch  # noqa: WPS433
        from transformers import AutoModelForTDT, AutoProcessor  # noqa: WPS433

        MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        model = AutoModelForTDT.from_pretrained(
            MODEL_ID, dtype=torch.bfloat16, device_map="cuda"
        )
        model.eval()
        _STATE["model"] = model
        _STATE["processor"] = processor
    return _STATE["model"], _STATE["processor"]


def _oracle():
    if not ORACLE_PATH.exists():
        pytest.skip(
            f"oracle missing: {ORACLE_PATH} (run benchmarks/bench_rtf.py first)"
        )
    return {e["name"]: e for e in json.loads(ORACLE_PATH.read_text())}


def _prepare(processor, audio):
    """processor + H2D + bf16 cast (matches the baseline's prepare_inputs)."""
    import torch  # noqa: WPS433

    inputs = processor([audio], sampling_rate=16000).to("cuda")
    inputs["input_features"] = inputs["input_features"].to(torch.bfloat16)
    return inputs


FIXTURE_NAMES = ["short", "medium", "long"]

# steps_per_replay values exercised by the multi-step-capture byte-exactness
# test. K=1 is the reference (one step per replay); {4,16,64} cover sub-chunk,
# the production default, and a full-utterance-in-one-replay case.
K_VALUES = [1, 4, 16, 64]


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_eager_decode_matches_oracle(name):
    """Eager greedy_decode must reproduce the oracle transcript byte-for-byte."""
    from megapar.parakeet.decode_eager import greedy_decode

    oracle = _oracle()
    model, processor = _get_model_and_processor()
    fixtures = mkfx.load_fixtures()
    inputs = _prepare(processor, fixtures[name])

    texts = greedy_decode(
        model,
        inputs["input_features"],
        inputs["attention_mask"],
        processor,
    )
    text = texts[0]
    expected = oracle[name]["text"]
    assert text == expected, (
        f"[eager/{name}] transcript drift:\n  oracle: {expected!r}\n"
        f"  mine:   {text!r}"
    )
    # token count must also match (the decoder emitted the right number of ids)
    # oracle num_tokens counts non-pad ids; our decode emitted exactly the
    # transcript, so a text match already implies the token sequence matches.


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_graphed_decode_matches_oracle(name):
    """CUDA-graph-captured greedy_decode_graphed must match the oracle too."""
    from megapar.parakeet.decode_mega import greedy_decode_graphed

    oracle = _oracle()
    model, processor = _get_model_and_processor()
    fixtures = mkfx.load_fixtures()
    inputs = _prepare(processor, fixtures[name])

    texts = greedy_decode_graphed(
        model,
        inputs["input_features"],
        inputs["attention_mask"],
        processor,
    )
    text = texts[0]
    expected = oracle[name]["text"]
    assert text == expected, (
        f"[graphed/{name}] transcript drift:\n  oracle: {expected!r}\n"
        f"  mine:   {text!r}"
    )


@pytest.mark.parametrize("K", K_VALUES)
@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_graphed_multistep_byte_exact(name, K):
    """steps_per_replay in {1,4,16,64} must be byte-identical to K=1.

    The multi-step capture records K consecutive decode steps into one CUDA
    graph and replays it ceil(max_out/K) times, syncing the host once per K
    steps. Because every step's state (last_token / frame_idx / h_buf / c_buf /
    cc_buf) lives in in-place-mutated static buffers, one K-step replay must
    produce exactly the same per-step tokens and cumulative encoder-frame
    indices as K separate single-step replays. This asserts three things, all
    vs the K=1 reference:

    * the emitted token sequence ``output[:, :out_step]`` is byte-identical;
    * the ``collect_meta`` path (``meta_tokens`` / ``meta_frames`` consumed by
      the frame-aligned chunker) is byte-identical;
    * the decoded text still matches the oracle (catches any drift that
      ``skip_special_tokens`` might otherwise mask).

    Guards both the plain decode and the chunking path against K-regressions.
    """
    import torch  # noqa: WPS433

    from megapar.parakeet.decode_mega import GraphedDecoder  # noqa: WPS433

    oracle = _oracle()
    model, processor = _get_model_and_processor()
    fixtures = mkfx.load_fixtures()
    inputs = _prepare(processor, fixtures[name])
    pad_id = processor.tokenizer.pad_token_id

    # precompute encoder features once for this fixture
    with torch.inference_mode():
        enc = model.get_audio_features(
            input_features=inputs["input_features"],
            attention_mask=inputs["attention_mask"],
        )
        pooler = enc.pooler_output.contiguous()
        valid_lengths = enc.attention_mask.to(torch.long).sum(-1).contiguous()

    # K=1 reference: both the plain output and the collect_meta path.
    g_ref = GraphedDecoder(model, steps_per_replay=1)
    g_ref.capture(pooler, valid_lengths, pad_id, steps_per_replay=1)
    out_step_ref = g_ref._run_loop(pooler, valid_lengths)
    ref_tokens = g_ref.output[:, :out_step_ref].clone()
    out_step_ref_m, meta_tokens_ref, meta_frames_ref = g_ref._run_loop(
        pooler, valid_lengths, collect_meta=True
    )
    assert out_step_ref_m == out_step_ref, (
        f"[ref/{name}] K=1 collect_meta out_step {out_step_ref_m} "
        f"!= plain {out_step_ref}"
    )

    # this K: plain decode, meta path, and text-vs-oracle.
    gd = GraphedDecoder(model, steps_per_replay=K)
    gd.capture(pooler, valid_lengths, pad_id, steps_per_replay=K)

    out_step = gd._run_loop(pooler, valid_lengths)
    got = gd.output[:, :out_step]
    assert out_step == out_step_ref, (
        f"[multistep/{name}/K={K}] out_step {out_step} != K=1 {out_step_ref}"
    )
    assert torch.equal(got, ref_tokens), (
        f"[multistep/{name}/K={K}] emitted token sequence drifted from K=1"
    )

    out_step_m, meta_tokens, meta_frames = gd._run_loop(
        pooler, valid_lengths, collect_meta=True
    )
    assert out_step_m == out_step_ref, (
        f"[multistep-meta/{name}/K={K}] out_step {out_step_m} != K=1 {out_step_ref}"
    )
    assert meta_tokens == meta_tokens_ref, (
        f"[multistep-meta/{name}/K={K}] meta_tokens drifted from K=1"
    )
    assert meta_frames == meta_frames_ref, (
        f"[multistep-meta/{name}/K={K}] meta_frames drifted from K=1"
    )

    text = processor.batch_decode(
        [gd.output[0, :out_step].tolist()], skip_special_tokens=True
    )[0]
    expected = oracle[name]["text"]
    assert text == expected, (
        f"[multistep/{name}/K={K}] transcript drift vs oracle:\n"
        f"  oracle: {expected!r}\n  mine:   {text!r}"
    )
