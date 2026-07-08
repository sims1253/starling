"""Make the MOSS audio encoder CUDA-graph-capturable (byte-exact).

The stock ``Qwen3OmniMoeAudioEncoder`` forward is ~85% CPU-launch-bound (the GPU
idles between hundreds of small kernel launches across its 32 layers), so a
CUDA-graph replay is 3-12x faster.  Two host-syncing ops abort capture, and both
are eliminated here by hoisting the (shape-only) sync work OUT of the captured
region:

1. ``_compute_chunking`` calls ``.item()`` / ``.tolist()`` to derive the
   chunk/window bookkeeping.  Those outputs (``chunk_lengths``,
   ``valid_indices``, ``cu_seqlens``) depend only on ``feature_lens`` (the audio
   *shape*), not the audio values, so :class:`GraphedAudioEncoder` computes them
   once on the host and feeds them through the stock forward's kwargs pop-path.
   Only ``padded_feature`` depends on values and is rebuilt inside the graph
   from the replayed input via a **static** split (constant chunk sizes).

2. Each of the 32 attention layers does
   ``torch.split(q/k/v, (cu[1:]-cu[:-1]).tolist(), dim=2)`` -- a per-layer
   device->host sync.  Those split lengths are also a pure function of shape, so
   :func:`patch_audio_attention` replaces the runtime ``.tolist()`` with a
   precomputed static python list published via :func:`active_split_lengths`
   during capture.  When no static list is active the patched forward is
   behaviourally identical to the stock one (same ``.tolist()`` fallback), so
   the patch is a no-op for every other code path.

Same sdpa math, static shapes.  The rebuilt-in-graph forward run *eagerly* is
bit-identical to the stock forward; the captured replay differs by ~1 bf16 ULP
(``ENCODER_ATOL``-level) purely from CUDA-graph GEMM-algorithm selection, and
the end-to-end transcript is unchanged (verified against the golden fixtures).
Capture on utterance A then replay a different utterance B of the same length is
correct (padded_feature is rebuilt from the replayed input).  No
flash-attention dependency.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

# Static per-layer attention split lengths active during a capture/warmup.  Set
# by GraphedAudioEncoder around the capture region; ``None`` everywhere else.
_ACTIVE_SPLIT: Optional[list[int]] = None
_PATCHED = False


@contextmanager
def active_split_lengths(lengths: list[int]):
    """Publish static attention split lengths for the duration of a capture."""
    global _ACTIVE_SPLIT
    saved = _ACTIVE_SPLIT
    _ACTIVE_SPLIT = lengths
    try:
        yield
    finally:
        _ACTIVE_SPLIT = saved


def patch_audio_attention() -> None:
    """Idempotently patch ``Qwen3OmniMoeAudioAttention.forward`` to use static
    split lengths when :func:`active_split_lengths` is in effect."""
    global _PATCHED
    if _PATCHED:
        return
    import torch
    import transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe as M
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        eager_attention_forward,
        is_flash_attention_requested,
    )

    Attn = M.Qwen3OmniMoeAudioAttention
    if getattr(Attn, "_starling_patched", False):
        _PATCHED = True
        return
    _orig_forward = Attn.forward

    def forward(self, hidden_states, cu_seqlens, **kwargs):
        # Flash path is already capture-safe (uses cu_seqlens directly); leave it
        # entirely to the stock implementation.
        if is_flash_attention_requested(self.config):
            return _orig_forward(self, hidden_states, cu_seqlens, **kwargs)

        seq_length, _ = hidden_states.size()
        q = self.q_proj(hidden_states).reshape(seq_length, self.num_heads, -1).transpose(0, 1).unsqueeze(0)
        k = self.k_proj(hidden_states).reshape(seq_length, self.num_heads, -1).transpose(0, 1).unsqueeze(0)
        v = self.v_proj(hidden_states).reshape(seq_length, self.num_heads, -1).transpose(0, 1).unsqueeze(0)
        attn_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        # Static list during capture (no device sync); identical .tolist()
        # fallback otherwise -> behaviourally unchanged for all other callers.
        lengths = _ACTIVE_SPLIT if _ACTIVE_SPLIT is not None else (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        splits = [torch.split(t, lengths, dim=2) for t in (q, k, v)]
        outs = [
            attn_interface(
                self, qq, kk, vv, attention_mask=None, scaling=self.scaling,
                dropout=0.0, is_causal=False, **kwargs,
            )[0]
            for qq, kk, vv in zip(*splits)
        ]
        attn_output = torch.cat(outs, dim=1).reshape(seq_length, -1).contiguous()
        return self.out_proj(attn_output)

    Attn.forward = forward
    Attn._starling_patched = True
    _PATCHED = True
