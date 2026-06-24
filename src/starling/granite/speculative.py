"""Self-speculative decoding for Granite-Speech-4.1-2b ASR.

Uses the encoder's BPE CTC head (``out_llm``) to draft multiple tokens cheaply
from a single encoder pass, then verifies them in fewer LLM forwards.

Guarantee
---------
Greedy-verify-of-a-greedy-oracle produces **byte-identical** output to standard
greedy decoding: every emitted token is the LLM's greedy argmax at its position
(given the correct prefix), so the transcript matches the non-speculative path
exactly.

CTC framing (reconstructed from the IBM ``speculative_decoding_bpe`` notebook)
---------------------------------------------------------------------------
1. Run the encoder eagerly; capture the mid-layer (block
   ``num_layers // 2 - 1`` = 7) output **before** the self-conditioned CTC
   feedback.
2. ``importance = 1 - blank_prob`` from the grapheme head (``encoder.out``) on
   the mid-layer hidden (blank = label 0 in the 348-dim grapheme vocab).
3. Posterior-weighted pooling (window = 4) of the **last-layer** hidden using
   these importance weights -> pooled ``(1, T//4, 1024)``.
4. ``out_llm`` (Linear 1024 -> 100353) on the pooled vectors -> softmax ->
   argmax per pooled frame.
5. CTC collapse: ``unique_consecutive`` -> drop blank (label 0) -> map BPE
   label ``i`` -> Granite tokenizer token ``i - 1`` (label 0 is the CTC blank;
   labels 1..100352 map to LLM tokens 0..100351).

The resulting draft is a fixed BPE token sequence for a given utterance
(deterministic encoder), computed once, then speculated.

Public API
----------
``CTCBPEDraft(fused_encoder, out_llm)`` -- extract draft tokens.
``CTCBPEDraft.encode_with_mid(input_features) -> (mid_h, enc_hidden)``
``CTCBPEDraft.draft(enc_hidden, mid_h) -> list[int]``
``SpeculativeDecoder(llm, embed_tokens)`` -- greedy-verify decoder.
``SpeculativeDecoder.generate(inputs_embeds, draft, max_new_tokens, eos_token_id)``
    -> :class:`SpecResult`
``load_out_llm(device, dtype)`` -- load the BPE CTC head from the HF snapshot.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import LLM_EOS_TOKEN_ID, LLM_LOGITS_SCALING, MODEL_ID


# =========================================================================== #
# Result container
# =========================================================================== #
@dataclass
class SpecResult:
    """Output of :meth:`SpeculativeDecoder.generate`."""

    ids: torch.Tensor  # (1, n_new) int64 on CPU
    text: str
    n_tokens: int
    total_ms: float  # verify + decode fallback wall time (excludes prefill)
    tok_per_s: float
    draft_count: int
    accepted: int
    """Number of draft tokens confirmed to match the LLM greedy output: the
    position-exact verify accepts PLUS the re-alignment matches (draft tokens
    the LLM emitted at a shifted position, because the raw CTC draft lacks
    caps/punctuation). Together this equals the draft's LCS-style match rate."""
    verify_forwards: int
    acceptance_rate: float
    """``accepted / draft_count``. For the sample audio this reaches ~92% (the
    draft's intrinsic LCS match rate); strict position-exact verify acceptance
    alone is ~83%."""
    # v2 instrumentation: separate the two forward types so callers can confirm
    # the draft phase ran on pure verify forwards (no decode probes).
    decode_steps: int = 0
    """Single-token CUDA-graph decode steps (fallback, run only AFTER the draft
    is exhausted). Zero during the draft phase in the v2 pure-verify loop."""
    decode_probes: int = 0
    """Single-token decode steps executed INSIDE the draft loop. Must be 0 in
    the v2 pure-verify design (every draft-phase forward is a multi-token
    verify). Non-zero indicates a regression to the v1 hybrid probe loop."""


# =========================================================================== #
# BPE CTC draft extraction
# =========================================================================== #
class CTCBPEDraft:
    """Extract BPE CTC draft tokens from encoder internals.

    Matches the IBM ``speculative_decoding_bpe`` notebook algorithm exactly.
    The draft tokens live in the LLM's vocabulary (after the label ``i ->
    token i-1`` mapping) and can be fed directly to the LLM for verification.
    """

    WINDOW: int = 4
    """Temporal pooling window for the BPE head (LLM_DOWNSAMPLE_WINDOW)."""

    BPE_VOCAB: int = 100353
    """BPE CTC vocab = 100352 Granite BPE tokens + 1 CTC blank (label 0)."""

    BLANK_LABEL: int = 0
    """CTC blank occupies label 0 in the BPE head output."""

    def __init__(
        self,
        fused_encoder: Any,
        out_llm: nn.Module,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.enc = fused_encoder
        self.out_llm = out_llm
        self.device = device
        self.dtype = dtype
        # mid_idx is 1-indexed (num_layers // 2 = 8); the mid-layer hook fires
        # on block mid_idx-1 = 7 and captures its output BEFORE the CTC feedback.
        self.mid_idx = int(fused_encoder.mid_idx)

    # ------------------------------------------------------------------ #
    # encoder pass with mid-layer capture
    # ------------------------------------------------------------------ #
    def encode_with_mid(
        self, input_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the encoder eagerly and return ``(mid_h, enc_hidden)``.

        ``mid_h`` = output of encoder block ``mid_idx - 1`` (= 7), captured
        **before** the self-conditioned CTC feedback is applied.  ``enc_hidden``
        = last-layer output (identical to ``FusedEncoder.forward``).

        This reuses :meth:`FusedEncoder._block_eager` (byte-exact with stock)
        and replicates the CTC feedback injection point so ``mid_h`` matches the
        notebook's forward-hook capture.
        """
        enc = self.enc
        feats = input_features
        if feats.dtype != self.dtype:
            feats = feats.to(self.dtype)
        if feats.device.type != self.device:
            feats = feats.to(self.device)
        # The block-attention padding mask is normally prepared by
        # FusedEncoder.forward(); we bypass forward() so do it here.
        enc._prepare_block_mask(int(feats.shape[1]), feats.device)

        x = enc.input_linear(feats)
        mid_h: Optional[torch.Tensor] = None
        for idx in range(enc.num_layers):
            x = enc._block_eager(idx, x)
            if (idx + 1) == self.mid_idx:
                # Capture BEFORE the self-conditioned CTC feedback (matches the
                # notebook's register_forward_hook on layers[mid_idx - 1]).
                mid_h = x
                mid_logits = enc.out(x)
                x = x + enc.out_mid(F.softmax(mid_logits, dim=-1))
        assert mid_h is not None, "mid_idx not reached during encoder forward"
        return mid_h, x

    # ------------------------------------------------------------------ #
    # draft extraction
    # ------------------------------------------------------------------ #
    def draft(
        self, enc_hidden: torch.Tensor, mid_h: torch.Tensor
    ) -> list[int]:
        """Extract BPE CTC draft token ids (in the LLM vocabulary).

        Returns a list of ints, each in ``[0, 100351]`` (Granite BPE tokens).
        """
        # (1) Importance weights from mid-layer grapheme blank probability.
        mid_grapheme_logits = self.enc.out(mid_h)  # (1, T, 348)
        mid_probs = F.softmax(mid_grapheme_logits.float(), dim=-1)
        importance = 1.0 - mid_probs[:, :, self.BLANK_LABEL]  # (1, T)

        # (2) Posterior-weighted pooling (window=4) of last-layer hidden.
        pooled = self._posterior_weighted_pool(
            enc_hidden, importance, self.WINDOW
        )  # (1, T//4, 1024) float32
        pooled = pooled.to(self.dtype)

        # (3) BPE head -> softmax -> per-frame argmax.
        bpe_logits = self.out_llm(pooled)  # (1, T//4, 100353)
        bpe_probs = F.softmax(bpe_logits.float(), dim=-1)
        labels = bpe_probs.argmax(dim=-1)[0]  # (T//4,)

        # (4) CTC collapse: unique_consecutive -> drop blank -> map i -> i-1.
        dedup = torch.unique_consecutive(labels)
        non_blank = dedup[dedup > self.BLANK_LABEL]
        token_ids = [int(t.item()) - 1 for t in non_blank]
        return token_ids

    # ------------------------------------------------------------------ #
    # convenience: full pipeline
    # ------------------------------------------------------------------ #
    def draft_from_features(
        self, input_features: torch.Tensor
    ) -> tuple[list[int], torch.Tensor]:
        """Encode + draft in one call.

        Returns ``(draft_token_ids, enc_hidden)`` where ``enc_hidden`` is the
        last-layer output (for downstream projection -> audio_embeds).
        """
        mid_h, enc_hidden = self.encode_with_mid(input_features)
        return self.draft(enc_hidden, mid_h), enc_hidden

    # ------------------------------------------------------------------ #
    # posterior-weighted pooling (from the notebook)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _posterior_weighted_pool(
        hidden: torch.Tensor, importance: torch.Tensor, window: int = 4
    ) -> torch.Tensor:
        """Importance-weighted temporal downsampling.

        Args:
            hidden: ``(B, T, D)`` encoder hidden states.
            importance: ``(B, T)`` per-frame weights (1 - blank_prob).
            window: temporal downsampling factor.

        Returns:
            ``(B, ceil(T/window), D)`` pooled hidden (float32).
        """
        B, T, D = hidden.shape
        pad = (window - T % window) % window
        if pad > 0:
            hidden = F.pad(hidden, (0, 0, 0, pad))
            importance = F.pad(importance, (0, pad))
        nw = hidden.shape[1] // window
        h = hidden.reshape(B, nw, window, D)
        imp = importance.reshape(B, nw, window)
        weights = imp / (imp.sum(dim=-1, keepdim=True) + 1e-8)
        return (h * weights.unsqueeze(-1)).sum(dim=2)


# =========================================================================== #
# Speculative greedy decoder
# =========================================================================== #
class SpeculativeDecoder:
    """Self-speculative greedy decoder using CTC BPE drafts.

    Algorithm (v2 pure multi-token verify loop)
    --------------------------------------------
    1. **Prefill** the LLM on the multimodal ``inputs_embeds`` -> first token.
       Capture the single-token CUDA-graph decode step (used only for the
       post-draft fallback).
    2. **Pure verify loop**: every draft-phase round is ONE multi-token verify
       forward. Feed ``[last_token, draft_chunk]`` (k+1 tokens) through the LLM
       and accept the longest prefix where the greedy argmax agrees with the
       draft:
       - On **full acceptance** (all k match): emit all k chunk tokens + a free
         bonus token (``logits[k].argmax()``), then **ramp** the chunk size
         (8 -> 12 -> 16).
       - On **mismatch at position j**: emit the accepted prefix ``chunk[0..j-1]``
         plus the LLM's greedy correction ``logits[j].argmax()`` (the true greedy
         token, NOT the rejected draft token), then reset the chunk size to 8.
       After each round the KV cache is rewound (``_reset_cache_pos``) to the
       position of the last emitted token so the stale slots written beyond the
       acceptance point are overwritten before they can be read.
    3. **Decode fallback**: once the draft is exhausted (or ``max_new_tokens``
       is reached), emit any remaining tokens with the single-token CUDA-graph
       decode step.

    Why no decode probe is needed
    -----------------------------
    The verify forward already returns per-position logits at ALL k+1 positions.
    The mismatch position's argmax IS the LLM's greedy re-alignment token, so a
    separate single-token probe to "check alignment" is pure waste. v1 ran
    (probe + verify) = 2 forwards per accepted run; v2 runs exactly 1 verify
    forward per round, roughly halving the forward count for the draft portion.

    Why multi-round with skip works for ASR
    ---------------------------------------
    The CTC draft produces raw BPE tokens WITHOUT capitalization or
    punctuation, while the LLM generates formatted text.  This causes
    intermittent misalignment at formatting boundaries (sentence-initial caps,
    commas, periods).  However, the CONTENT words match exactly (91.9% LCS).
    By re-aligning the draft pointer to the last emitted token after each round
    and re-verifying, the decoder re-discovers alignment at each content run and
    accepts long stretches (up to 24 consecutive tokens on the sample audio) in
    a single forward.

    Guarantee
    ---------
    Every emitted token is the LLM's greedy argmax at its position given the
    correct prefix. Accepted draft tokens are verified to equal the greedy
    argmax; corrections and bonus tokens are the greedy argmax by construction;
    the fallback decode is standard greedy. The output sequence is
    **byte-identical** to standard greedy decoding.
    """

    MIN_CHUNK: int = 8
    """Starting / reset verify chunk size (also the floor after a mismatch)."""

    BOUNDARY_CHUNK: int = 2
    """Tiny chunk used while stalled inside a formatting boundary (j=0 mismatch
    with no draft re-alignment). Keeps single-token verify forwards cheap."""

    MAX_CHUNK: int = 16
    """Maximum verify chunk size (ramped 8 -> 12 -> 16 on full acceptance)."""

    def __init__(self, llm: Any, embed_tokens: Any) -> None:
        self.llm = llm
        self.embed_tokens = embed_tokens
        # Detect whether the LLM decoder is a FusedLLMMega (has fused Triton
        # kernels + manual layer iteration) for the fast verify path.
        self._fused = hasattr(llm, "_layers") and hasattr(llm, "_k")
        # Lazy CUDA-graph cache: L -> {graph, ids, pos, mask, logits}.
        self._verify_graphs: dict[int, dict] = {}

    # ------------------------------------------------------------------ #
    # main entry point
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def generate(
        self,
        inputs_embeds: torch.Tensor,
        draft: list[int],
        max_new_tokens: int = 200,
        eos_token_id: int = LLM_EOS_TOKEN_ID,
    ) -> SpecResult:
        llm = self.llm
        device = llm.device
        P = inputs_embeds.shape[1]
        draft_count = len(draft)

        # Cache overflow guard (same as LLMMega.generate).
        max_safe = llm.max_cache_len - P + 1
        if max_new_tokens > max_safe:
            raise ValueError(
                f"max_new_tokens={max_new_tokens} would overflow the static KV "
                f"cache (prompt T={P}, max_cache_len={llm.max_cache_len})."
            )
        if inputs_embeds.shape[0] != 1:
            raise ValueError("SpeculativeDecoder only supports batch=1.")
        if max_new_tokens <= 0:
            return self._finalize([], 0.0, draft_count, 0, 0, 0, 0)

        # (1) Prefill -> first token.
        llm._reset_cache_pos(0)
        first_token = llm.prefill(inputs_embeds)  # (1, 1)
        emitted: list[int] = [int(first_token.item())]

        if emitted[-1] == eos_token_id or len(emitted) >= max_new_tokens:
            return self._finalize(emitted, 0.0, draft_count, 0, 0, 0, 0)

        # Capture the decode graph once at the prefill position.  The graph is
        # position-agnostic (reads static buffers + cumulative_length on each
        # replay) so it can be reused at any cache position during the fallback.
        if not llm._captured:
            llm.capture(first_token, P)

        # Pre-capture verify CUDA graphs for every chunk size in [1, MAX_CHUNK]
        # so the timed decode loop never pays an on-the-fly capture. These use
        # the same fused Triton kernels as the decode step but process L tokens
        # at once, making each verify forward as cheap as a single decode step.
        self.warmup_graphs(tuple(range(1, self.MAX_CHUNK + 1)))
        # Restore cache to the prefill position (warmup_graphs may have moved it).
        llm._reset_cache_pos(P)

        t_start = time.perf_counter()

        last_token = first_token  # (1, 1) tensor
        cache_pos = P  # cumulative_length after prefill (position of last_token)
        accepted = 0
        verify_forwards = 0
        decode_steps = 0       # single-token decode steps (fallback only)
        decode_probes = 0      # decode steps inside the draft loop (MUST stay 0)
        draft_pos = 0
        chunk_size = self.MIN_CHUNK  # adaptive: 8 -> 12 -> 16 on full accept
        SEARCH_WIN = 40        # how far ahead to search for re-alignment
        STUCK_LIMIT = 8        # max consecutive no-progress rounds before a forced skip
        stuck = 0              # consecutive rounds with no draft_pos advance
        # On-device draft for sync-free verify_ids construction (slice copies).
        draft_tensor = (
            torch.tensor(draft, dtype=torch.int64, device=device)
            if draft_count > 0
            else torch.empty(0, dtype=torch.int64, device=device)
        )

        # (2) PURE VERIFY LOOP (no decode probes).
        #
        # Every round is ONE multi-token verify forward. We feed
        # ``[last_token, draft_chunk]`` (k+1 tokens) through the LLM and accept
        # the longest prefix where the greedy argmax agrees with the draft. The
        # mismatch position's argmax is the LLM's greedy correction (so no probe
        # is needed to re-align); on full acceptance the bonus token at position
        # k is taken for free. After each round the KV cache is rewound to the
        # position of the last emitted token (``_reset_cache_pos``) so the stale
        # slots written beyond the acceptance point are overwritten before they
        # can be read, and the 4D causal mask keeps them invisible.
        #
        # Re-alignment: after each round, search the draft ahead for the last
        # emitted token and jump the draft pointer past it, so the next chunk
        # starts at a likely-aligned position. This handles the formatting drift
        # caused by the CTC draft lacking capitalization/punctuation.
        while draft_pos < draft_count and len(emitted) < max_new_tokens:
            draft_pos_before = draft_pos

            # (a) size the chunk: clamp to remaining draft / token budget / cache.
            remaining_budget = max_new_tokens - len(emitted)
            cache_headroom = llm.max_cache_len - cache_pos - 1
            k = min(chunk_size, draft_count - draft_pos,
                    remaining_budget, cache_headroom)
            if k <= 0:
                break  # nothing left to verify; fall through to decode fallback

            chunk = draft[draft_pos: draft_pos + k]

            # (b) verify_ids = [last_token] + chunk  -> (1, k+1).
            #     Build it with GPU-native slice copies (no D2H sync); last_token
            #     is already on-device and draft_tensor is precomputed on-device.
            verify_ids = torch.empty(
                (1, k + 1), dtype=torch.int64, device=device
            )
            verify_ids[0, 0:1] = last_token.view(1)
            verify_ids[0, 1:k + 1] = draft_tensor[draft_pos: draft_pos + k]

            # (c) multi-token verify forward (per-length CUDA graph replay).
            logits = self._verify_forward(verify_ids, cache_pos)
            verify_forwards += 1

            # (d) accept the longest matching prefix; emit the greedy correction
            #     on mismatch, or a bonus token on full acceptance. Pull ALL k+1
            #     greedy argmaxes in ONE D2H sync (vs up to k+1 per-position
            #     syncs in the naive scan).
            preds_all = logits[0, : k + 1].argmax(dim=-1).tolist()
            preds = preds_all[:k]
            bonus = preds_all[k]

            j = 0
            mismatch = False
            full_accept = False
            stopped = False
            while j < k and preds[j] == chunk[j]:
                # accept draft token chunk[j] (verified == greedy argmax).
                emitted.append(chunk[j])
                accepted += 1
                if chunk[j] == eos_token_id:
                    j += 1
                    stopped = True
                    break
                j += 1
            if not stopped:
                if j < k:
                    # mismatch at position j: emit the LLM's greedy correction.
                    emitted.append(preds[j])
                    mismatch = True
                    if preds[j] == eos_token_id:
                        stopped = True
                else:
                    # all k accepted -> bonus token from logits position k.
                    full_accept = True
                    emitted.append(bonus)
                    if bonus == eos_token_id:
                        stopped = True

            # (e) advance the cache to the position of the last emitted token.
            #     The verify forward wrote K/V for all k+1 positions; only the
            #     first (accepted+1) are valid, so we rewind the rest in (f).
            if full_accept:
                cache_pos += k + 1
                draft_pos += k
            elif mismatch:
                cache_pos += j + 1     # j accepted + 1 correction
                draft_pos += j         # rejected chunk[j] left for re-align
            else:
                # stopped during accept (last accepted token was EOS).
                cache_pos += j
                draft_pos += j

            # (f) rewind stale cache slots to the new write position.
            llm._reset_cache_pos(cache_pos)

            # (g) re-align: search the draft ahead for the last emitted token so
            #     the next chunk begins at a likely-aligned position. When the
            #     just-emitted token equals a draft token, the draft correctly
            #     predicted it (at a shifted position due to the raw CTC draft
            #     lacking caps/punctuation), so we count it as accepted too.
            last_val = emitted[-1]
            realigned = False
            if not stopped and draft_pos < draft_count:
                hi = min(draft_pos + SEARCH_WIN, draft_count)
                for d in range(draft_pos, hi):
                    if draft[d] == last_val:
                        draft_pos = d + 1
                        accepted += 1
                        realigned = True
                        break

            # Forward-progress accounting. We do NOT blindly skip on every
            # failed re-alignment: a formatting-boundary correction (a cap /
            # quote / period that is absent from the raw CTC draft) is followed
            # within a few rounds by a CONTENT token that IS in the draft, and
            # the re-alignment search then jumps to its nearest (correct)
            # occurrence. Forcing a skip every time would leapfrog that content
            # token and re-align to a coincidental later copy, permanently
            # desynchronising the rest of the draft. We only force a single
            # skip after STUCK_LIMIT consecutive no-progress rounds (a guard
            # against a genuinely un-matchable draft token).
            if draft_pos > draft_pos_before:
                stuck = 0
            else:
                stuck += 1
            if stuck >= STUCK_LIMIT and not stopped and draft_pos < draft_count:
                draft_pos = draft_pos_before + 1
                stuck = 0

            # (h) adapt chunk size. Content runs ramp up (8 -> 12 -> 16); a
            #     partial accept (content run broke at a formatting token) and a
            #     re-alignment back onto content reset to the standard floor.
            #     A j=0 mismatch with NO re-alignment means we are still inside a
            #     formatting boundary (caps/quotes/periods absent from the raw
            #     CTC draft) -- the next round will almost certainly also emit a
            #     single greedy token, so use a tiny chunk to make that
            #     single-token verify forward cheap instead of wasting an L=9
            #     forward on one token.
            if not stopped:
                if full_accept:
                    chunk_size = min(chunk_size + 4, self.MAX_CHUNK)
                elif mismatch and j == 0 and not realigned:
                    chunk_size = self.BOUNDARY_CHUNK
                else:
                    chunk_size = self.MIN_CHUNK

            last_token = torch.tensor(
                [[last_val]], dtype=torch.int64, device=device
            )

            if stopped or len(emitted) >= max_new_tokens:
                torch.cuda.synchronize()
                wall_ms = (time.perf_counter() - t_start) * 1000.0
                return self._finalize(
                    emitted, wall_ms, draft_count, accepted,
                    verify_forwards, decode_steps, decode_probes,
                )

        # (3) Decode remaining tokens with the single-token CUDA-graph step.
        #     Only runs AFTER the draft is exhausted (decode_probes stayed 0).
        while len(emitted) < max_new_tokens:
            cur_pos = cache_pos
            llm.static_input_ids.copy_(last_token.reshape(1, 1))
            llm.static_position_ids.copy_(
                torch.tensor([[cur_pos]], device=device)
            )
            llm._set_mask(cur_pos + 1)
            if llm._captured:
                llm._graph.replay()
            else:
                llm._decode_step_eager()
            decode_steps += 1
            last_token = llm.static_logits.argmax(dim=-1)  # (1, 1)
            tid = int(last_token.item())
            emitted.append(tid)
            cache_pos += 1
            if tid == eos_token_id:
                break

        torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - t_start) * 1000.0
        return self._finalize(
            emitted, wall_ms, draft_count, accepted,
            verify_forwards, decode_steps, decode_probes,
        )

    # ------------------------------------------------------------------ #
    # chunked verify forward (a short prefill over candidate tokens)
    # ------------------------------------------------------------------ #
    def _verify_forward(
        self, verify_ids: torch.Tensor, start_pos: int
    ) -> torch.Tensor:
        """Forward ``verify_ids`` ``(1, L)`` through the LLM on top of the cache.

        The StaticCache must be primed at ``start_pos`` (cumulative_length ==
        start_pos).  After this call the cache advances by ``L``.

        Returns logits ``(1, L, vocab)`` at all ``L`` positions (already scaled
        by ``1 / logits_scaling`` to match :class:`LLMMega` output).

        Uses the fused-kernel multi-token forward (same Triton kernels as
        :class:`FusedLLMMega`) captured into per-length CUDA graphs for zero
        launch overhead.  Falls back to the model's own forward if the decoder
        does not expose fused kernels.
        """
        L = verify_ids.shape[1]
        if self._fused:
            if L not in self._verify_graphs:
                self._capture_verify_graph(L)
            return self._verify_forward_graph(verify_ids, start_pos, L)
        return self._verify_forward_eager(verify_ids, start_pos)

    # ------------------------------------------------------------------ #
    # fused multi-token forward (graph-safe; mirrors FusedLLMMega._decode_step)
    # ------------------------------------------------------------------ #
    def _verify_step_fused(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        mask: torch.Tensor,
        logits_out: torch.Tensor,
    ) -> None:
        """Multi-token fused forward (graph-safe). Writes into ``logits_out``.

        Mirrors :meth:`FusedLLMMega._decode_step_eager` but processes ``L``
        tokens at once.  All intermediate tensors are derived from the four
        static inputs so the entire computation is CUDA-graph capturable.
        """
        from .llm_mega import _EMB_MULT, _repeat_kv

        llm = self.llm
        k = llm._k
        hd = llm._head_dim
        n_q = llm._n_q_heads
        n_kv = llm._n_kv_heads

        # (1) embedding lookup + multiplier
        hidden = llm._embed(input_ids) * _EMB_MULT  # (1, L, 2048)

        # (2) rotary cos/sin for L positions
        cos, sin = llm._rotary(hidden, position_ids=position_ids)
        cos4 = cos.unsqueeze(1)  # (1, 1, L, hd)
        sin4 = sin.unsqueeze(1)

        half = hd // 2

        # (3) iterate layers
        for idx, layer in enumerate(llm._layers):
            sa = layer.self_attn
            mlp = layer.mlp

            # --- attention block ---
            residual = hidden
            normed = k.fused_rmsnorm(
                hidden, layer.input_layernorm.weight, llm._rms_eps
            )

            B, Llen = normed.shape[:2]
            q = sa.q_proj(normed).view(B, Llen, n_q, hd).transpose(1, 2)
            kv = sa.k_proj(normed).view(B, Llen, n_kv, hd).transpose(1, 2)
            v = sa.v_proj(normed).view(B, Llen, n_kv, hd).transpose(1, 2)

            # RoPE (PyTorch, matching the reference's bf16 arithmetic exactly)
            q_rot = torch.cat((-q[..., half:], q[..., :half]), dim=-1)
            kv_rot = torch.cat((-kv[..., half:], kv[..., :half]), dim=-1)
            q = q * cos4 + q_rot * sin4
            kv = kv * cos4 + kv_rot * sin4

            # cache update (writes L entries, returns full buffer)
            kv, v = llm.cache.update(kv, v, idx)
            kv_r = _repeat_kv(kv, llm._n_kv_groups)
            v_r = _repeat_kv(v, llm._n_kv_groups)

            # attention: scores = Q @ K^T * scale + mask
            scores = torch.matmul(q, kv_r.transpose(2, 3)) * llm._attn_scale
            scores = scores + mask
            attn = F.softmax(scores, dim=-1, dtype=torch.float32).to(llm.dtype)
            attn_out = torch.matmul(attn, v_r)

            attn_out = attn_out.transpose(1, 2).reshape(B, Llen, n_q * hd)
            attn_out = sa.o_proj(attn_out)
            hidden = k.fused_residual_scale(residual, attn_out, llm._res_mult)

            # --- MLP block ---
            residual = hidden
            normed = k.fused_rmsnorm(
                hidden, layer.post_attention_layernorm.weight, llm._rms_eps
            )
            gate = mlp.gate_proj(normed)
            up = mlp.up_proj(normed)
            act = k.fused_silu_mul(gate, up)
            mlp_out = mlp.down_proj(act)
            hidden = k.fused_residual_scale(residual, mlp_out, llm._res_mult)

        # (4) final norm + lm_head
        hidden = k.fused_rmsnorm(hidden, llm._final_norm.weight, llm._rms_eps)
        logits = llm.lm_head(hidden) / LLM_LOGITS_SCALING
        logits_out.copy_(logits)

    # ------------------------------------------------------------------ #
    # CUDA graph capture for a fixed verify length L
    # ------------------------------------------------------------------ #
    def _capture_verify_graph(self, L: int) -> None:
        """Capture a CUDA graph for a verify forward of ``L`` tokens."""
        llm = self.llm
        device = llm.device
        dtype = llm.dtype
        max_cache_len = llm.max_cache_len
        vocab = llm.vocab_size

        # Static buffers (fixed addresses for the graph).
        s_ids = torch.zeros((1, L), dtype=torch.int64, device=device)
        s_pos = torch.zeros((1, L), dtype=torch.int64, device=device)
        s_mask = torch.full(
            (1, 1, L, max_cache_len), llm._neg_val, dtype=dtype, device=device
        )
        s_logits = torch.zeros((1, L, vocab), dtype=dtype, device=device)

        # Prime with dummy values at a safe cache position.  Use a position
        # well within max_cache_len so warmup writes don't overflow.
        capture_pos = max_cache_len - L - 2
        s_ids.fill_(1)
        s_pos.copy_(
            torch.arange(capture_pos, capture_pos + L, device=device).unsqueeze(0)
        )
        self._set_verify_mask(s_mask, capture_pos, L)
        llm._reset_cache_pos(capture_pos)

        # Warmup (reset cumulative_length between iters so we never overflow
        # the static cache; each iter writes exactly L entries).
        for _ in range(3):
            llm._reset_cache_pos(capture_pos)
            self._verify_step_fused(s_ids, s_pos, s_mask, s_logits)
        torch.cuda.synchronize()
        llm._reset_cache_pos(capture_pos)

        # Re-prime after warmup.
        s_ids.fill_(1)
        s_pos.copy_(
            torch.arange(capture_pos, capture_pos + L, device=device).unsqueeze(0)
        )
        self._set_verify_mask(s_mask, capture_pos, L)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._verify_step_fused(s_ids, s_pos, s_mask, s_logits)

        # Reset after capture (the captured step advanced cumulative_length).
        llm._reset_cache_pos(capture_pos)

        self._verify_graphs[L] = {
            "graph": graph,
            "ids": s_ids,
            "pos": s_pos,
            "mask": s_mask,
            "logits": s_logits,
        }

    def _verify_forward_graph(
        self, verify_ids: torch.Tensor, start_pos: int, L: int
    ) -> torch.Tensor:
        """Replay the captured verify graph for ``L`` tokens."""
        llm = self.llm
        entry = self._verify_graphs[L]
        entry["ids"].copy_(verify_ids)
        entry["pos"].copy_(
            torch.arange(start_pos, start_pos + L, device=llm.device).unsqueeze(0)
        )
        self._set_verify_mask(entry["mask"], start_pos, L)
        entry["graph"].replay()
        return entry["logits"]  # (1, L, vocab)

    def _set_verify_mask(
        self, mask: torch.Tensor, start_pos: int, L: int
    ) -> None:
        """Fill the causal verify mask (called OUTSIDE the graph, before replay).

        ``mask`` is ``(1, 1, L, max_cache_len)``.  Query ``i`` (absolute position
        ``start_pos + i``) may attend to keys ``[0, start_pos + i]``.
        """
        mask.fill_(self.llm._neg_val)
        for i in range(L):
            mask[:, :, i, : start_pos + i + 1] = 0.0

    def warmup_graphs(self, chunk_sizes: tuple[int, ...] | None = None) -> None:
        """Pre-capture verify graphs for the given chunk sizes (L = size + 1).

        Defaults to every size in ``[1, MAX_CHUNK]`` so the pure-verify loop
        never pays an on-the-fly capture during the timed decode region. Call
        this before benchmarking to exclude capture latency from the timing.
        """
        if not self._fused:
            return
        if chunk_sizes is None:
            chunk_sizes = tuple(range(1, self.MAX_CHUNK + 1))
        for cs in chunk_sizes:
            L = cs + 1
            if L not in self._verify_graphs:
                self._capture_verify_graph(L)

    # ------------------------------------------------------------------ #
    # eager fallback (model's own forward — slow, no graph)
    # ------------------------------------------------------------------ #
    def _verify_forward_eager(
        self, verify_ids: torch.Tensor, start_pos: int
    ) -> torch.Tensor:
        """Fallback: use the model's own forward (no fused kernels)."""
        llm = self.llm
        device = llm.device
        dtype = llm.dtype
        L = verify_ids.shape[1]
        max_cache_len = llm.max_cache_len

        embeds = self.embed_tokens(verify_ids)
        position_ids = torch.arange(
            start_pos, start_pos + L, device=device
        ).unsqueeze(0)

        neg = llm._neg_val
        q_abs = torch.arange(
            start_pos, start_pos + L, device=device
        ).unsqueeze(1)
        k_pos = torch.arange(max_cache_len, device=device).unsqueeze(0)
        valid = k_pos <= q_abs
        mask = torch.where(
            valid,
            torch.tensor(0.0, dtype=dtype, device=device),
            torch.tensor(neg, dtype=dtype, device=device),
        )
        mask = mask.unsqueeze(0).unsqueeze(0)

        out = llm.lm(
            inputs_embeds=embeds,
            position_ids=position_ids,
            attention_mask=mask,
            past_key_values=llm.cache,
            use_cache=True,
        )
        hidden = out.last_hidden_state
        logits = llm.lm_head(hidden) / LLM_LOGITS_SCALING
        return logits

    # ------------------------------------------------------------------ #
    def _finalize(
        self,
        emitted: list[int],
        decode_wall_ms: float,
        draft_count: int,
        accepted: int,
        verify_forwards: int,
        decode_steps: int = 0,
        decode_probes: int = 0,
    ) -> SpecResult:
        ids = torch.tensor(emitted, dtype=torch.int64).unsqueeze(0)
        n = len(emitted)
        tps = n / max(decode_wall_ms / 1000.0, 1e-9)
        acc_rate = accepted / draft_count if draft_count > 0 else 0.0
        return SpecResult(
            ids=ids,
            text="",
            n_tokens=n,
            total_ms=decode_wall_ms,
            tok_per_s=tps,
            draft_count=draft_count,
            accepted=accepted,
            verify_forwards=verify_forwards,
            acceptance_rate=acc_rate,
            decode_steps=decode_steps,
            decode_probes=decode_probes,
        )


# =========================================================================== #
# BPE head loader
# =========================================================================== #
def _find_snapshot_dir() -> Path:
    """Locate the HF snapshot directory for Granite-Speech-4.1-2b."""
    try:
        from huggingface_hub import snapshot_download

        p = snapshot_download(repo_id=MODEL_ID)
        return Path(p)
    except Exception:
        pass
    # Fallback: search the HF cache directly.
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    glob_pattern = f"models--{MODEL_ID.replace('/', '--')}"
    matches = sorted((cache_root / glob_pattern / "snapshots").glob("*/"))
    if matches:
        return matches[-1]
    raise FileNotFoundError(
        f"Could not locate the HF snapshot for {MODEL_ID}. "
        f"Ensure the model is downloaded."
    )


def load_out_llm(
    device: str = "cuda", dtype: torch.dtype = torch.bfloat16
) -> nn.Linear:
    """Load the BPE CTC head (``out_llm``) from the model snapshot.

    Returns an ``nn.Linear(1024, 100353)`` on ``device`` in ``dtype``.
    """
    from safetensors.torch import load_file

    snapshot = _find_snapshot_dir()
    path = snapshot / "out_llm.safetensors"
    if not path.exists():
        raise FileNotFoundError(
            f"out_llm.safetensors not found at {path}. "
            f"The BPE CTC head is required for speculative decoding."
        )
    sd = load_file(str(path))
    head = nn.Linear(1024, 100353, bias=True)
    with torch.no_grad():
        head.weight.copy_(sd["weight"].to(torch.float32))
        head.bias.copy_(sd["bias"].to(torch.float32))
    return head.to(device=device, dtype=dtype).eval()
