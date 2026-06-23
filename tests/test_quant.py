"""Correctness gate for the weight-only INT8 quantised decoder.

Verifies that:
* ``quantized_weights=True`` requires ``tolerance_mode=True`` (guard).
* The pipeline wires ``quantized_weights=True`` to :class:`QuantLLMMega`.
* :func:`quantize_linear` round-trips a bf16 weight within INT8 tolerance.
* The fused :func:`w8_linear` dequant-GEMM matches ``nn.functional.linear`` on
  the bf16 reference weight within INT8 rounding error.
* **End-to-end transcript quality**: :class:`QuantLLMMega` decodes the golden
  ``inputs_embeds`` to a transcript that matches the golden transcript text with
  ~0 WER (INT8 rounding breaks byte-exactness but greedy-chaos must not flip the
  transcript) and reproduces the vast majority of golden tokens.

Run with:  uv run pytest tests/test_quant.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from megapar.config import LLM_EOS_TOKEN_ID  # noqa: E402
from megapar.flags import OptFlags  # noqa: E402
from megapar.golden import load_golden, load_golden_text  # noqa: E402
from megapar.loader import get_components, load_model_and_processor  # noqa: E402
from megapar.quant import QuantLLMMega, quantize_linear, w8_linear  # noqa: E402

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
    return load_golden("greedy_ids.pt")[0, 271:]


def _golden_response() -> str:
    txt = load_golden_text().strip()
    return txt.split("ASSISTANT:", 1)[1].strip()


def _wer(ref: str, hyp: str) -> float:
    r, h = ref.lower().split(), hyp.lower().split()
    if not r:
        return 0.0 if not h else 1.0
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    return d[len(r)][len(h)] / len(r)


# --------------------------------------------------------------------------- #
# flag guard
# --------------------------------------------------------------------------- #
def test_quantized_weights_requires_tolerance():
    """quantized_weights=True without tolerance_mode must raise."""
    with pytest.raises(ValueError, match="quantized_weights"):
        OptFlags(quantized_weights=True, tolerance_mode=False)


def test_quantized_weights_with_tolerance_ok():
    """quantized_weights=True WITH tolerance_mode=True is valid + defaults off."""
    assert OptFlags().quantized_weights is False, "defaults OFF (byte-exact)"
    f = OptFlags(quantized_weights=True, tolerance_mode=True)
    assert f.quantized_weights is True


# --------------------------------------------------------------------------- #
# pipeline wiring
# --------------------------------------------------------------------------- #
def test_pipeline_quantized_wiring():
    """quantized_weights=True -> QuantLLMMega (single-stream MegaPipeline)."""
    from megapar.pipeline import MegaPipeline

    model, proc = _get_model_and_processor()
    pipe = MegaPipeline(
        model, proc, flags=OptFlags(quantized_weights=True, tolerance_mode=True)
    )
    assert isinstance(pipe.llm, QuantLLMMega), (
        "quantized_weights=True should use QuantLLMMega"
    )


def test_batched_pipeline_quantized_wiring():
    """quantized_weights=True -> BatchedQuantLLMMega (batched pipeline)."""
    from megapar.batched import BatchedPipeline
    from megapar.quant import BatchedQuantLLMMega

    model, proc = _get_model_and_processor()
    pipe = BatchedPipeline(
        model, proc, max_batch_size=2,
        flags=OptFlags(quantized_weights=True, tolerance_mode=True),
    )
    assert isinstance(pipe.llm, BatchedQuantLLMMega), (
        "quantized_weights=True should use BatchedQuantLLMMega"
    )


# --------------------------------------------------------------------------- #
# quantize_linear + w8_linear unit correctness
# --------------------------------------------------------------------------- #
def test_quantize_linear_roundtrip():
    """quantize_linear + w8_linear must match bf16 linear within INT8 error."""
    torch.manual_seed(0)
    for N, K in [(2048, 2048), (4096, 2048), (1024, 2048)]:
        w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.05
        w_int8, scales = quantize_linear(w)
        assert w_int8.dtype == torch.int8 and w_int8.shape == (N, K)
        assert scales.shape == (N,)
        x = torch.randn(1, K, device="cuda", dtype=torch.bfloat16)
        y_ref = torch.nn.functional.linear(x, w)
        y = w8_linear(x, w_int8, scales)
        diff = (y.float() - y_ref.float()).abs()
        # INT8 channelwise: relative error bounded by 1/127 ~ 0.008 of the
        # weight magnitude; for |w|~0.05 the abs error is well under 1.0.
        assert diff.max().item() < 2.0, f"({N},{K}) max-abs {diff.max().item():.3f} too large"


# --------------------------------------------------------------------------- #
# end-to-end transcript quality (the real correctness bar)
# --------------------------------------------------------------------------- #
def test_quant_decode_matches_golden_transcript():
    """QuantLLMMega decode must match the golden transcript (WER ~0) and most
    golden tokens (INT8 is not byte-exact, but greedy-chaos must not derail)."""
    model, proc = _get_model_and_processor()
    comps = get_components(model)
    qdec = QuantLLMMega(comps["language_model"], model.lm_head, max_cache_len=640)

    inputs_embeds = _golden_inputs_embeds()
    golden_gen = _golden_generated()
    golden_resp = _golden_response()

    res = qdec.generate(
        inputs_embeds, max_new_tokens=100, eos_token_id=LLM_EOS_TOKEN_ID
    )
    assert res.n_tokens == 100, f"expected 100 tokens, got {res.n_tokens}"

    text = proc.tokenizer.decode(res.ids[0], skip_special_tokens=True)
    # WER must be ~0 (transcript text preserved).
    w = _wer(golden_resp, text)
    assert w <= 0.05, f"WER={w:.3f} too high; transcript diverged:\n  gold={golden_resp[:120]!r}\n  quant={text[:120]!r}"

    # Token-level match: INT8 is not byte-exact, but the vast majority should
    # match (greedy-chaos flips only a few tokens that decode to the same text).
    n_match = int((res.ids[0] == golden_gen).sum().item())
    pct = n_match / golden_gen.shape[0] * 100.0
    assert pct >= 80.0, (
        f"token match {pct:.1f}% too low; INT8 quantisation too aggressive"
    )


def test_quant_decode_lm_head_logit_diff_small():
    """The quantised lm_head logits must stay close to the bf16 lm_head logits
    (sanity bound; not a strict tolerance since INT8 is approximate)."""
    model, _ = _get_model_and_processor()
    comps = get_components(model)
    lm = comps["language_model"]
    lm_head = model.lm_head
    qdec = QuantLLMMega(lm, lm_head, max_cache_len=640)
    inputs_embeds = _golden_inputs_embeds()
    with torch.inference_mode():
        pos_ids = torch.arange(inputs_embeds.shape[1], device="cuda").unsqueeze(0)
        out = lm(inputs_embeds=inputs_embeds, position_ids=pos_ids,
                 past_key_values=qdec.cache, use_cache=True)
        hidden = out.last_hidden_state[:, -1:, :]
        # Compare the SCALED logits (post /LLM_LOGITS_SCALING) -- those are what
        # the greedy argmax actually sees.  INT8 lm_head rounding is a few
        # percent of the logit magnitude; argmax is robust (100% token match).
        from megapar.config import LLM_LOGITS_SCALING
        lg_bf = lm_head(hidden) / LLM_LOGITS_SCALING
        lg_q = w8_linear(hidden, qdec._lm_head_int8, qdec._lm_head_scales) / LLM_LOGITS_SCALING
        diff = (lg_q.float() - lg_bf.float()).abs()
    assert diff.max().item() < 0.5, f"scaled lm_head logit diff {diff.max().item():.3f} too large"


if __name__ == "__main__":
    test_quantized_weights_requires_tolerance()
    print("[manual] test_quantized_weights_requires_tolerance PASSED")
    test_quantized_weights_with_tolerance_ok()
    print("[manual] test_quantized_weights_with_tolerance_ok PASSED")
    test_quantize_linear_roundtrip()
    print("[manual] test_quantize_linear_roundtrip PASSED")
    test_pipeline_quantized_wiring()
    print("[manual] test_pipeline_quantized_wiring PASSED")
    test_batched_pipeline_quantized_wiring()
    print("[manual] test_batched_pipeline_quantized_wiring PASSED")
    test_quant_decode_matches_golden_transcript()
    print("[manual] test_quant_decode_matches_golden_transcript PASSED")
    test_quant_decode_lm_head_logit_diff_small()
    print("[manual] test_quant_decode_lm_head_logit_diff_small PASSED")
