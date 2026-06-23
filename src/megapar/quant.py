"""Weight-only INT8 quantised decode path for the Granite-4.0-1b LLM.

This module implements **weight-only quantisation** (Option B from the
diagnosis): every decoder ``Linear`` weight is stored as INT8 with a per-output
row (channelwise) scale, and the per-token matmuls in the decode step are
served by a fused Triton *dequant-GEMM* kernel.  Activations stay bf16; only
the *weight reads* are halved (2 bytes bf16 -> 1 byte int8 per weight element),
which is the single biggest remaining lever for a memory-bandwidth-bound
single-token decode.

Why this exists / honest verdict
--------------------------------
The decode of Granite-4.0-1b at batch=1 is, in theory, memory-bandwidth-bound
on weight reads (~3.0 GB of weights read per token).  Weight-only quantisation
halves that traffic, so it *should* ~2x throughput -- *if* the dequant kernel
saturates the memory bus as well as cuBLAS saturates it for bf16.

Empirically it does **not**.  Measured on the RTX 5090 (sm_120, torch
2.12.1+cu130) across the real 280-GEMV/token decode pattern:

    * FP8 ``torch._scaled_mm`` sustained: **0.56x** bf16 (per-call launch
      overhead dominates the tiny M=1 GEMVs -- the earlier "0.94x isolated"
      finding holds and is in fact worse in the sustained back-to-back case).
    * Triton INT8 dequant-GEMV (M=1): **0.42-0.84x** bf16.
    * Triton INT8 dequant-GEMM (tl.dot, M=1/8/16): **0.37-0.62x** bf16.

cuBLAS bf16 already achieves ~324-426 GB/s effective for these shapes, and the
int8 dequantisation overhead (int8->bf16 cast + the channelwise scale) plus
Triton's lower per-shape bandwidth efficiency for the small decode matmuls
eats the 2x weight-traffic reduction.  **So the path is shipped for
completeness and future re-evaluation (better HW/torch/kernels may flip the
calculus) but is NOT a speedup on current hardware.**  It is gated behind
``OptFlags(quantized_weights=True, tolerance_mode=True)`` and defaults OFF.

Numerics / correctness
----------------------
INT8 channelwise quantisation is approximate (max-abs ~1.5-2.0 per matmul on
random weights), so this path is **not byte-exact** -- it requires
``tolerance_mode=True``.  The decoded transcript is verified against the golden
transcript with a WER tolerance (greedy-chaos token flips that decode to the
same text are allowed).  See ``tests/test_quant.py``.

Public API
----------
``quantize_linear(weight_bf16) -> (w_int8, scales)``  channelwise int8 quant.
``quantize_model(llm, lm_head)``                       quantise in place (stores int8).
``QuantLLMMega`` (subclass of :class:`FusedLLMMega`)   single-stream quantised decode.
``BatchedQuantLLMMega`` (subclass of :class:`BatchedFusedLLMMega`)  batched variant.
"""

from __future__ import annotations

from typing import Any

import torch
import triton
import triton.language as tl

from .config import LLM_LOGITS_SCALING
from .llm_mega import FusedLLMMega, _EMB_MULT, _repeat_kv

# =========================================================================== #
# Triton fused weight-only INT8 dequant-GEMM kernel
# =========================================================================== #
# Computes  C = (A @ W_int8^T) * scale   (bf16 activations, int8 weights)
#
#   A       : (M, K) bf16     (activation)
#   W_int8  : (N, K) int8     (weight, row-major, one row per output)
#   scale   : (N,)   fp16/bf16 (per-output-row / "channelwise" scale)
#   C       : (M, N) bf16
#
# Strategy (channelwise, the standard weight-only-int8 GEMM):
#   * In the K loop the int8 weight tile is cast straight to bf16 (exact: every
#     int8 value is representable in bf16) -- NO per-element scale multiply,
#     which is the expensive part.  The matmul A @ W^T is accumulated in fp32
#     via ``tl.dot``.
#   * The per-output-row scale is applied ONCE at the very end (an M*N outer
#     broadcast), so the dequantisation cost is amortised over the whole matmul
#     instead of being paid per K tile per element.
#
# This is graph-safe: it performs no host syncs and all allocations come from
# the capturing graph's private pool.
# =========================================================================== #


@triton.jit
def _w8_gemm_kernel(
    A_ptr, W_ptr, S_ptr, C_ptr,
    M, N, K: tl.constexpr,
    stride_am, stride_ak,
    stride_wn, stride_wk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m0 = pid_m * BLOCK_M
    n0 = pid_n * BLOCK_N

    offs_m = m0 + tl.arange(0, BLOCK_M)
    offs_n = n0 + tl.arange(0, BLOCK_N)
    mmask = offs_m < M
    nmask = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        kmask = ks < K
        # Activation tile (BLOCK_M, BLOCK_K) bf16.
        a = tl.load(
            A_ptr + offs_m[:, None] * stride_am + ks[None, :] * stride_ak,
            mask=mmask[:, None] & kmask[None,], other=0.0,
        )
        # Weight tile (BLOCK_N, BLOCK_K) int8 -> cast to bf16 (exact, no scale).
        w = tl.load(
            W_ptr + offs_n[:, None] * stride_wn + ks[None, :] * stride_wk,
            mask=nmask[:, None] & kmask[None,], other=0,
        )
        w = w.to(a.dtype)  # int8 -> bf16 (exact; per-row scale applied later)
        # acc += A @ W^T  (fp32 accumulate via tensor cores / tl.dot).
        acc += tl.dot(a, w.trans(), allow_tf32=False)

    # Apply the channelwise scale ONCE at the end.
    scales = tl.load(S_ptr + offs_n, mask=nmask, other=0.0).to(tl.float32)
    c = (acc * scales[None, :]).to(C_ptr.dtype.element_ty)
    tl.store(
        C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        c, mask=mmask[:, None] & nmask[None, :],
    )


def w8_linear(
    x: torch.Tensor,
    w_int8: torch.Tensor,
    scales: torch.Tensor,
    *,
    block_m: int = 16,
    block_n: int = 64,
    block_k: int = 64,
) -> torch.Tensor:
    """Weight-only INT8 matmul: ``y = x @ w_int8^T * scales`` (bf16 out).

    Args:
        x: ``(*, K)`` bf16 activations (any leading dims; flattened to ``(M, K)``).
        w_int8: ``(N, K)`` int8 weight (one row per output feature).
        scales: ``(N,)`` per-output-row scale (fp16/bf16).
        block_m/block_n/block_k: Triton tile sizes (``block_m`` is padded up to
            16 to enable ``tl.dot``; for ``M=1`` the decode case, only the first
            row is real and the rest are masked out).

    Returns:
        ``(*leading, N)`` bf16 output, same leading shape as ``x``.
    """
    leading = x.shape[:-1]
    K = x.shape[-1]
    M = 1
    for d in leading:
        M *= d
    a = x.reshape(M, K)
    N = w_int8.shape[0]
    # tl.dot requires the M-tile >= 16; pad BLOCK_M up to 16 even for M=1.
    BM = max(16, block_m)
    C = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)
    grid = (triton.cdiv(M, BM), triton.cdiv(N, block_n))
    _w8_gemm_kernel[grid](
        a, w_int8, scales, C,
        M, N, K,
        a.stride(0), a.stride(1),
        w_int8.stride(0), w_int8.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BM, BLOCK_N=block_n, BLOCK_K=block_k,
    )
    return C.view(*leading, N)


# =========================================================================== #
# Quantisation helpers
# =========================================================================== #


def quantize_linear(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Channelwise INT8-quantise a bf16/fp16 ``nn.Linear`` weight.

    ``weight`` is ``(out_features, in_features)`` (the ``nn.Linear.weight``
    layout).  Returns ``(w_int8, scales)`` where ``w_int8`` has the same shape
    (int8) and ``scales`` is ``(out_features,)`` fp16 such that
    ``weight ~= w_int8.to(bf16) * scales[:, None]``.

    The scale is ``max(|weight[row]|) / 127`` (symmetric, per-output-row), the
    standard "channelwise" weight-only-int8 scheme (lowest quality risk; matches
    the task's "per-channel (per-output-column) scales" requirement).
    """
    assert weight.ndim == 2, f"expected 2D weight, got {weight.shape}"
    wf = weight.float()
    row_abs_max = wf.abs().amax(dim=1).clamp(min=1e-8)  # (out,)
    scales = (row_abs_max / 127.0).to(torch.float16)    # (out,) fp16
    w_int8 = (wf / scales.float().unsqueeze(1)).round().clamp(-128, 127).to(torch.int8)
    return w_int8, scales


def quantize_model(language_model: Any, lm_head: Any) -> dict[str, dict[str, torch.Tensor]]:
    """Quantise every Granite decoder ``Linear`` weight + ``lm_head`` to INT8.

    Stores the int8 weight + per-row scale on the module under attributes
    ``_wq_int8`` / ``_wq_scales`` and returns a manifest for inspection.  The
    original bf16 ``.weight`` is left untouched so the eager prefill (which uses
    the model's own layers via ``language_model(...)``) stays in bf16 -- prefill
    is compute-bound and quality-sensitive, so per the constraints it is NOT
    quantised.
    """
    manifest: dict[str, dict[str, torch.Tensor]] = {}

    def _q(name: str, linear: Any) -> None:
        w_int8, scales = quantize_linear(linear.weight.data)
        linear._wq_int8 = w_int8      # type: ignore[attr-defined]
        linear._wq_scales = scales    # type: ignore[attr-defined]
        manifest[name] = {
            "w_int8": tuple(w_int8.shape),
            "scales": tuple(scales.shape),
        }

    for i, layer in enumerate(language_model.layers):
        sa = layer.self_attn
        mlp = layer.mlp
        _q(f"layer.{i}.q_proj", sa.q_proj)
        _q(f"layer.{i}.k_proj", sa.k_proj)
        _q(f"layer.{i}.v_proj", sa.v_proj)
        _q(f"layer.{i}.o_proj", sa.o_proj)
        _q(f"layer.{i}.gate_proj", mlp.gate_proj)
        _q(f"layer.{i}.up_proj", mlp.up_proj)
        _q(f"layer.{i}.down_proj", mlp.down_proj)
    _q("lm_head", lm_head)
    return manifest


# =========================================================================== #
# Single-stream quantised decoder
# =========================================================================== #
# NOTE: :class:`QuantLLMMega` subclasses :class:`FusedLLMMega` (imported above
# from :mod:`megapar.llm_mega`).  The batched sibling subclassing
# :class:`BatchedFusedLLMMega` is defined at the bottom of this module to keep
# the top-level import block free of a circular dependency (``batched`` wires
# the batched quant decoder via a *local* import, so it must not top-import
# this module).


class QuantLLMMega(FusedLLMMega):
    """Weight-only INT8 single-stream quantised decoder for the Granite LLM.

    Subclasses :class:`FusedLLMMega` (so it inherits the fused RMSNorm/SiLU/
    residual Triton elementwise kernels and all CUDA-graph capture / generate /
    bench machinery) and overrides :meth:`_decode_step_eager` to route the seven
    per-layer matmuls + ``lm_head`` through :func:`w8_linear` using pre-quantised
    INT8 weights.  Prefill, attention softmax, RoPE and the elementwise glue
    stay identical to the fused bf16 path.

    The weights are quantised ONCE in :meth:`_quantize`; the original bf16
    weights are retained for the eager prefill (prefill is not quantised per the
    design constraints).

    Args:
        language_model: The ``GraniteModel`` decoder trunk.
        lm_head: ``nn.Linear`` lm_head from the top-level speech model.
        block_n/block_k: Triton tile sizes for the dequant-GEMM.  Tuned for the
            Granite shapes (single-token decode, M=1).
        max_cache_len/warmup_iters/device/dtype: forwarded to :class:`FusedLLMMega`.
    """

    def __init__(
        self,
        language_model: Any,
        lm_head: Any,
        *,
        max_cache_len: int = 640,
        warmup_iters: int = 3,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        block_n: int = 64,
        block_k: int = 64,
    ) -> None:
        super().__init__(
            language_model,
            lm_head,
            max_cache_len=max_cache_len,
            warmup_iters=warmup_iters,
            device=device,
            dtype=dtype,
        )
        self._block_n = int(block_n)
        self._block_k = int(block_k)
        # Quantise every decoder Linear + lm_head into INT8 (channelwise).
        self._quant_manifest = quantize_model(self.lm, self.lm_head)
        # Cache per-layer int8 weight/scale handles for the hot decode loop.
        self._q_weights = []
        for layer in self._layers:
            sa = layer.self_attn
            mlp = layer.mlp
            self._q_weights.append({
                "q": (sa.q_proj._wq_int8, sa.q_proj._wq_scales),
                "k": (sa.k_proj._wq_int8, sa.k_proj._wq_scales),
                "v": (sa.v_proj._wq_int8, sa.v_proj._wq_scales),
                "o": (sa.o_proj._wq_int8, sa.o_proj._wq_scales),
                "gate": (mlp.gate_proj._wq_int8, mlp.gate_proj._wq_scales),
                "up": (mlp.up_proj._wq_int8, mlp.up_proj._wq_scales),
                "down": (mlp.down_proj._wq_int8, mlp.down_proj._wq_scales),
            })
        self._lm_head_int8 = self.lm_head._wq_int8
        self._lm_head_scales = self.lm_head._wq_scales

    def _qlinear(self, x: torch.Tensor, qw: torch.Tensor, qs: torch.Tensor) -> torch.Tensor:
        """Weight-only int8 matmul with the cached tile config."""
        return w8_linear(x, qw, qs, block_m=16, block_n=self._block_n, block_k=self._block_k)

    def _decode_step_eager(self) -> None:
        """INT8-quantised single-token decode forward.

        Identical arithmetic structure to
        :meth:`FusedLLMMega._decode_step_eager` but the seven per-layer GEMMs
        and the final ``lm_head`` GEMM run through :func:`w8_linear` on INT8
        weights.  Fused RMSNorm / SwiGLU / residual kernels and RoPE are
        unchanged.
        """
        k = self._k
        hd = self._head_dim
        n_q = self._n_q_heads
        n_kv = self._n_kv_heads
        bn, bk = self._block_n, self._block_k

        # (1) embedding lookup + multiplier (unchanged).
        hidden = self._embed(self.static_input_ids) * _EMB_MULT  # (1, 1, 2048)

        # (2) rotary cos/sin (unchanged).
        cos, sin = self._rotary(hidden, position_ids=self.static_position_ids)
        cos4 = cos.unsqueeze(1)
        sin4 = sin.unsqueeze(1)

        # (3) iterate layers (quantised matmuls).
        for idx, layer in enumerate(self._layers):
            qw = self._q_weights[idx]
            # --- attention block ---
            residual = hidden
            normed = k.fused_rmsnorm(hidden, layer.input_layernorm.weight, self._rms_eps)

            q = self._qlinear(normed, qw["q"][0], qw["q"][1]).view(1, 1, n_q, hd).transpose(1, 2)
            kv = self._qlinear(normed, qw["k"][0], qw["k"][1]).view(1, 1, n_kv, hd).transpose(1, 2)
            v = self._qlinear(normed, qw["v"][0], qw["v"][1]).view(1, 1, n_kv, hd).transpose(1, 2)

            half = hd // 2
            q_rot = torch.cat((-q[..., half:], q[..., :half]), dim=-1)
            kv_rot = torch.cat((-kv[..., half:], kv[..., :half]), dim=-1)
            q = q * cos4 + q_rot * sin4
            kv = kv * cos4 + kv_rot * sin4

            kv, v = self.cache.update(kv, v, idx)
            kv_r = _repeat_kv(kv, self._n_kv_groups)
            v_r = _repeat_kv(v, self._n_kv_groups)

            scores = torch.matmul(q, kv_r.transpose(2, 3)) * self._attn_scale
            scores = scores + self.static_attn_mask
            attn = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(self.dtype)
            attn_out = torch.matmul(attn, v_r)

            attn_out = attn_out.transpose(1, 2).reshape(1, 1, n_q * hd)
            attn_out = self._qlinear(attn_out, qw["o"][0], qw["o"][1])
            hidden = k.fused_residual_scale(residual, attn_out, self._res_mult)

            # --- MLP block ---
            residual = hidden
            normed = k.fused_rmsnorm(hidden, layer.post_attention_layernorm.weight, self._rms_eps)
            gate = self._qlinear(normed, qw["gate"][0], qw["gate"][1])
            up = self._qlinear(normed, qw["up"][0], qw["up"][1])
            act = k.fused_silu_mul(gate, up)
            mlp_out = self._qlinear(act, qw["down"][0], qw["down"][1])
            hidden = k.fused_residual_scale(residual, mlp_out, self._res_mult)

        # (4) final fused RMSNorm + quantised lm_head + logits scaling.
        hidden = k.fused_rmsnorm(hidden, self._final_norm.weight, self._rms_eps)
        logits = self._qlinear(hidden, self._lm_head_int8, self._lm_head_scales) / LLM_LOGITS_SCALING
        self.static_logits.copy_(logits)


# =========================================================================== #
# Batched quantised decoder
# =========================================================================== #
# Imported lazily here (not at module top) to keep the dependency direction
# acyclic: ``batched`` must top-import ``FusedEncoder``/``loader`` but must NOT
# top-import ``quant``; it wires :class:`BatchedQuantLLMMega` via a local import
# inside :class:`BatchedPipeline.__init__`.
from .batched import BatchedFusedLLMMega  # noqa: E402


class BatchedQuantLLMMega(BatchedFusedLLMMega):
    """Weight-only INT8 batched quantised decoder for the Granite LLM.

    Subclasses :class:`BatchedFusedLLMMega` (fused Triton elementwise kernels +
    batched CUDA-graph capture / generate machinery) and overrides
    :meth:`_decode_step_eager` to route the per-layer matmuls through
    :func:`w8_linear` on INT8 weights.  The dequant-GEMM now has ``M=B``, so
    the weight read is amortised over ``B`` tokens -- this is where the
    bandwidth win from quantisation *would* compound most if the kernel could
    saturate the bus (it cannot, per the module docstring, but the batched
    numbers are reported in ``scripts/bench_quant.py`` for completeness).
    """

    def __init__(
        self,
        language_model: Any,
        lm_head: Any,
        *,
        max_cache_len: int = 640,
        max_batch_size: int = 8,
        warmup_iters: int = 3,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        block_n: int = 64,
        block_k: int = 64,
    ) -> None:
        super().__init__(
            language_model,
            lm_head,
            max_cache_len=max_cache_len,
            max_batch_size=max_batch_size,
            warmup_iters=warmup_iters,
            device=device,
            dtype=dtype,
        )
        self._block_n = int(block_n)
        self._block_k = int(block_k)
        self._quant_manifest = quantize_model(self.lm, self.lm_head)
        self._q_weights = []
        for layer in self._layers:
            sa = layer.self_attn
            mlp = layer.mlp
            self._q_weights.append({
                "q": (sa.q_proj._wq_int8, sa.q_proj._wq_scales),
                "k": (sa.k_proj._wq_int8, sa.k_proj._wq_scales),
                "v": (sa.v_proj._wq_int8, sa.v_proj._wq_scales),
                "o": (sa.o_proj._wq_int8, sa.o_proj._wq_scales),
                "gate": (mlp.gate_proj._wq_int8, mlp.gate_proj._wq_scales),
                "up": (mlp.up_proj._wq_int8, mlp.up_proj._wq_scales),
                "down": (mlp.down_proj._wq_int8, mlp.down_proj._wq_scales),
            })
        self._lm_head_int8 = self.lm_head._wq_int8
        self._lm_head_scales = self.lm_head._wq_scales

    def _qlinear(self, x: torch.Tensor, qw: torch.Tensor, qs: torch.Tensor) -> torch.Tensor:
        return w8_linear(x, qw, qs, block_m=16, block_n=self._block_n, block_k=self._block_k)

    def _decode_step_eager(self) -> None:
        """INT8-quantised batched single-token decode forward (mirrors the
        bf16 batched fused decoder but with quantised GEMMs)."""
        k = self._k
        B = self.max_batch_size
        hd = self._head_dim
        n_q = self._n_q_heads
        n_kv = self._n_kv_heads
        half = hd // 2

        hidden = self._embed(self.static_input_ids) * self._emb_mult  # (B, 1, 2048)

        cos, sin = self._rotary(hidden, position_ids=self.static_position_ids)
        cos4 = cos.unsqueeze(1)
        sin4 = sin.unsqueeze(1)

        for idx, layer in enumerate(self._layers):
            qw = self._q_weights[idx]
            sa = layer.self_attn
            mlp = layer.mlp

            residual = hidden
            normed = k.fused_rmsnorm(hidden, layer.input_layernorm.weight, self._rms_eps)

            q = self._qlinear(normed, qw["q"][0], qw["q"][1]).view(B, 1, n_q, hd).transpose(1, 2)
            kv = self._qlinear(normed, qw["k"][0], qw["k"][1]).view(B, 1, n_kv, hd).transpose(1, 2)
            v = self._qlinear(normed, qw["v"][0], qw["v"][1]).view(B, 1, n_kv, hd).transpose(1, 2)

            q_rot = torch.cat((-q[..., half:], q[..., :half]), dim=-1)
            kv_rot = torch.cat((-kv[..., half:], kv[..., :half]), dim=-1)
            q = q * cos4 + q_rot * sin4
            kv = kv * cos4 + kv_rot * sin4

            kv, v = self.cache.update(kv, v, idx)
            kv_r = self._repeat_kv(kv, self._n_kv_groups)
            v_r = self._repeat_kv(v, self._n_kv_groups)

            scores = torch.matmul(q, kv_r.transpose(2, 3)) * self._attn_scale
            scores = scores + self.static_attn_mask
            attn = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(self.dtype)
            attn_out = torch.matmul(attn, v_r)

            attn_out = attn_out.transpose(1, 2).reshape(B, 1, n_q * hd)
            attn_out = self._qlinear(attn_out, qw["o"][0], qw["o"][1])
            hidden = k.fused_residual_scale(residual, attn_out, self._res_mult)

            residual = hidden
            normed = k.fused_rmsnorm(hidden, layer.post_attention_layernorm.weight, self._rms_eps)
            gate = self._qlinear(normed, qw["gate"][0], qw["gate"][1])
            up = self._qlinear(normed, qw["up"][0], qw["up"][1])
            act = k.fused_silu_mul(gate, up)
            mlp_out = self._qlinear(act, qw["down"][0], qw["down"][1])
            hidden = k.fused_residual_scale(residual, mlp_out, self._res_mult)

        hidden = k.fused_rmsnorm(hidden, self._final_norm.weight, self._rms_eps)
        logits = self._qlinear(hidden, self._lm_head_int8, self._lm_head_scales) / LLM_LOGITS_SCALING
        self.static_logits.copy_(logits)
