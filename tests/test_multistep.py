"""Correctness gate for the multi-step CUDA-graph decoder.

The K-step graph captures K consecutive greedy decode steps into one
``torch.cuda.CUDAGraph`` with the argmax chained in-graph (no host sync between
captured steps).  Greedy = greedy, so the emitted token sequence must be
**byte-exact** with the single-step decoder and therefore with the golden
``greedy_ids.pt[:, 271:]``.

This file verifies:
* ``test_multistep_exact_token_match`` -- K=16 reproduces the 100 golden tokens.
* ``test_multistep_matches_single_step`` -- multi-step output == single-step
  FusedLLMMega output for the same inputs (both byte-exact with golden).
* ``test_multistep_various_k`` -- K in {1, 4, 8, 16, 32} all reproduce golden
  (K=1 degenerates to one-step-per-replay, the original behaviour).
* ``test_multistep_is_not_slower`` -- the K-step decoder should not regress
  throughput vs the single-step decoder.

Run with:  uv run pytest tests/test_multistep.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from starling.config import LLM_EOS_TOKEN_ID  # noqa: E402
from starling.golden import load_golden  # noqa: E402
from starling.loader import get_components, load_model_and_processor  # noqa: E402
from starling.llm_mega import FusedLLMMega  # noqa: E402
from starling.multistep import MultiStepLLMMega  # noqa: E402

# Loading the speech model is expensive (~5s); cache across tests.
_MODEL = None
_PROC = None


def _get_model_and_processor():
    global _MODEL, _PROC
    if _MODEL is None:
        _MODEL, _PROC = load_model_and_processor(attn_impl="eager")
    return _MODEL, _PROC


def _golden_inputs_embeds() -> torch.Tensor:
    return load_golden("inputs_embeds.pt").to("cuda", torch.bfloat16)


def _golden_generated() -> torch.Tensor:
    """The 100 greedy-generated token ids (= golden[:, 271:])."""
    ids = load_golden("greedy_ids.pt")[0]
    return ids[271:]


def _build_decoder(K: int) -> MultiStepLLMMega:
    model, _ = _get_model_and_processor()
    comps = get_components(model)
    return MultiStepLLMMega(
        comps["language_model"],
        model.lm_head,
        max_cache_len=640,
        steps_per_replay=K,
    )


@pytest.fixture(scope="module")
def decoder_k16():
    return _build_decoder(K=16)


# --------------------------------------------------------------------------- #
# primary correctness: byte-exact token match vs golden
# --------------------------------------------------------------------------- #
def test_multistep_exact_token_match(decoder_k16):
    """K=16 multi-step decode must reproduce the 100 golden tokens exactly."""
    inputs_embeds = _golden_inputs_embeds()
    golden_gen = _golden_generated()

    res = decoder_k16.generate(
        inputs_embeds,
        max_new_tokens=100,
        eos_token_id=LLM_EOS_TOKEN_ID,
    )
    assert res.n_tokens == 100, f"expected 100 tokens, got {res.n_tokens}"
    assert (res.ids[0] == golden_gen).all(), (
        f"token mismatch: first diff at "
        f"{(res.ids[0] != golden_gen).nonzero()[0].item() if (res.ids[0] != golden_gen).any() else -1}"
    )


def test_multistep_matches_single_step(decoder_k16):
    """Multi-step output must equal single-step FusedLLMMega output."""
    model, _ = _get_model_and_processor()
    comps = get_components(model)
    single = FusedLLMMega(comps["language_model"], model.lm_head, max_cache_len=640)

    inputs_embeds = _golden_inputs_embeds()
    res_single = single.generate(
        inputs_embeds, max_new_tokens=100, eos_token_id=LLM_EOS_TOKEN_ID
    )
    res_multi = decoder_k16.generate(
        inputs_embeds, max_new_tokens=100, eos_token_id=LLM_EOS_TOKEN_ID
    )
    assert (res_multi.ids[0] == res_single.ids[0]).all(), (
        "multi-step output != single-step output"
    )


# --------------------------------------------------------------------------- #
# K sweep: every K value must be byte-exact
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("K", [1, 4, 8, 16, 32])
def test_multistep_various_k(K):
    """Every K in {1,4,8,16,32} must reproduce the golden tokens exactly."""
    decoder = _build_decoder(K=K)
    inputs_embeds = _golden_inputs_embeds()
    golden_gen = _golden_generated()

    res = decoder.generate(
        inputs_embeds,
        max_new_tokens=100,
        eos_token_id=LLM_EOS_TOKEN_ID,
    )
    assert res.n_tokens == 100, f"K={K}: expected 100 tokens, got {res.n_tokens}"
    assert (res.ids[0] == golden_gen).all(), (
        f"K={K}: token mismatch at "
        f"{(res.ids[0] != golden_gen).nonzero()[0].item() if (res.ids[0] != golden_gen).any() else -1}"
    )


# --------------------------------------------------------------------------- #
# throughput: multi-step should not regress
# --------------------------------------------------------------------------- #
def test_multistep_is_not_slower(decoder_k16):
    """The K=16 decoder should be at least as fast as the single-step floor."""
    inputs_embeds = _golden_inputs_embeds()
    rep = decoder_k16.bench(inputs_embeds, max_new_tokens=100, decode_iters=8)
    # Same floor as the single-step test (85 tok/s); multi-step should meet or
    # exceed it.
    assert rep.decode_tok_per_s > 85.0, (
        f"multi-step decode too slow: {rep.decode_tok_per_s:.1f} tok/s "
        f"(expected >85)"
    )


if __name__ == "__main__":
    dec = _build_decoder(K=16)
    test_multistep_exact_token_match(dec)
    print("[manual] test_multistep_exact_token_match PASSED")
    test_multistep_matches_single_step(dec)
    print("[manual] test_multistep_matches_single_step PASSED")
    for K in [1, 4, 8, 16, 32]:
        test_multistep_various_k(K)
        print(f"[manual] test_multistep_various_k[K={K}] PASSED")
    test_multistep_is_not_slower(dec)
    print("[manual] test_multistep_is_not_slower PASSED")
