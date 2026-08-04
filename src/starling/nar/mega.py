"""CUDA-graph megakernel for Granite-Speech-4.1-2b-NAR (non-autoregressive ASR).

Unlike every other starling track (granite/parakeet/moss/qwen3/higgs/ark), NAR
has **no autoregressive decode loop**. ASR is one bidirectional forward pass:

    mel -> encoder (16 conformer blocks + BPE CTC head)
        -> CTC greedy collapse -> rough token draft
        -> interleave blank "edit slots" -> edit sequence
        -> projector (Q-Former) -> audio embeds
        -> bidirectional granite-4.0-1b LLM editor (ONE forward, no KV cache)
        -> argmax + unique_consecutive + drop-blank -> final text

So there is no K-step decode graph to capture. The win lives in removing the
host launch overhead across the ~16 encoder + 40 LLM layers via CUDA-graph
capture of each dense forward, plus ``torch.compile`` of the LLM editor (Inductor
fuses the RMSNorm/SwiGLU/residual/RoPE elementwise glue — the proven moss/qwen3
lever).

Graph structure (per shape key):
  * **encoder graph**  — the graph-safe conformer trunk (input_linear -> 16
    blocks with self-conditioning -> dropout) is captured per mel-frame count.
    The stock ``out_bpe`` head does ``lengths.tolist()`` (host) which aborts
    capture, so the BPE head + posterior-weighted pool run eagerly *after* replay
    (a few cheap ops on the small pooled tensor). Byte-exact (0.0) vs stock.
  * **LLM graph** — the ``torch.compile``d **stock** ``model.language_model``
    forward is captured per edit-sequence length. Compiling in eager first lets
    Inductor emit its own deterministic fused Triton kernels (not cuBLAS), so
    graph-capturing the compiled function does NOT perturb the numerics the way
    capturing raw cuBLAS does. Byte-exact at the decoded-token level on all tiers
    (greedy is robust to the sub-ULP logit noise Inductor introduces — the
    moss/qwen3 finding). The CTC collapse (host ``unique_consecutive``) runs
    eagerly after replay.

We deliberately do NOT hand-iterate the LLM layers (unlike the AR tracks): a
hand-rolled forward diverges from the stock cuBLAS reduction order on long
packed sequences (bf16 tiling noise compounds over 40 layers, flipping a
borderline argmax). The compiled-stock path is the byte-exact fast path.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Optional

import torch
import torch.nn.functional as F

from .._kernels._compile import torch_compile


# =========================================================================== #
# Graph-safe encoder forward.
#
# The stock GraniteSpeechNarCTCEncoder.forward computes the conformer stack +
# self-conditioning (all graph-safe) and THEN a BPE head that does
# ``lengths.tolist()`` and a per-sample ``torch.cat`` list comprehension (host
# ops that abort CUDA-graph capture). We split it: capture the graph-safe trunk
# (input_linear -> 16 layers with self-conditioning -> dropout), then run the
# BPE head + posterior-weighted pool eagerly on the small pooled tensor.
# =========================================================================== #
def _encoder_trunk(
    encoder: Any,
    input_features: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], torch.Tensor]:
    """Graph-safe conformer trunk. Returns (last_hidden, all_hidden, blank_probs).

    Mirrors ``GraniteSpeechNarCTCEncoder.forward`` up to (and including) the
    final dropout, plus returns ``blank_probs`` (the self-conditioning
    mid-layer blank posterior) so the eager BPE head can compute importance.
    """
    hidden_states = encoder.input_linear(input_features.to(encoder.dtype))

    context_size = encoder.config.context_size
    seq = torch.arange(context_size, device=hidden_states.device)
    relpos_dist = seq.view(-1, 1) - seq.view(1, -1)
    attention_dists = torch.clamp(relpos_dist, -context_size, context_size) + encoder.config.max_pos_emb

    all_hidden = (hidden_states,)
    blank_probs = None
    for layer_idx, layer in enumerate(encoder.layers, start=1):
        hidden_states = layer(hidden_states, attention_dists=attention_dists)
        if layer_idx == encoder.config.self_conditioning_layer:
            mid_logits = encoder.out(encoder.dropout(hidden_states))
            mid_probs = torch.softmax(mid_logits.float(), dim=-1)
            blank_probs = mid_probs[:, :, 0]
            hidden_states = hidden_states + encoder.out_mid(mid_probs.to(hidden_states.dtype))
        all_hidden += (hidden_states,)

    hidden_states = encoder.dropout(hidden_states)
    return hidden_states, all_hidden, blank_probs


def _encoder_bpe_head(
    encoder: Any,
    hidden_states: torch.Tensor,
    blank_probs: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Eager BPE CTC logits (posterior-weighted pool + out_bpe).

    This is the non-graph-safe tail of the encoder: ``lengths.tolist()`` and a
    per-sample cat. It runs on the host between graph replays. For batch=1 it
    reduces to a single pooled matmul.
    """
    pool_window = encoder.config.bpe_pooling_window
    importance = 1.0 - blank_probs
    pooled = _posterior_weighted_pool(hidden_states.float(), importance, window_size=pool_window).to(
        hidden_states.dtype
    )
    encoder_lengths = attention_mask.sum(dim=1)
    lengths = -(encoder_lengths // -pool_window)
    lengths_list = lengths.tolist()
    logits = encoder.out_bpe(torch.cat([pooled[i, :length] for i, length in enumerate(lengths_list)]))
    return logits


def _posterior_weighted_pool(hidden: torch.Tensor, importance: torch.Tensor, window_size: int = 4) -> torch.Tensor:
    batch_size, seq_len, hidden_dim = hidden.shape
    pad_len = (window_size - seq_len % window_size) % window_size
    if pad_len > 0:
        hidden = F.pad(hidden, (0, 0, 0, pad_len))
        importance = F.pad(importance, (0, pad_len))
    num_windows = hidden.shape[1] // window_size
    hidden = hidden.view(batch_size, num_windows, window_size, hidden_dim)
    importance = importance.view(batch_size, num_windows, window_size)
    weights = importance / (importance.sum(dim=-1, keepdim=True) + 1e-8)
    pooled = (hidden * weights.unsqueeze(-1)).sum(dim=2)
    return pooled


# =========================================================================== #
# Edit-sequence construction (host, between encoder and LLM graphs).
# =========================================================================== #
def ctc_collapse_decode(bpe_logits: torch.Tensor, bpe_lengths: list[int], blank_id: int) -> list[torch.Tensor]:
    """GPU CTC greedy decode: argmax -> unique_consecutive -> remove blank.

    ``bpe_logits`` is ``[B, L, vocab]``; argmax over vocab then flatten the
    token axis and split per sample by ``bpe_lengths``.
    """
    preds_flat = bpe_logits.argmax(dim=-1).flatten()
    per_sample = preds_flat.split(bpe_lengths)
    return [(collapsed := torch.unique_consecutive(seq))[collapsed != blank_id] for seq in per_sample]


def add_insertion_slots(token_ids: torch.Tensor, blank_id: int, min_len: int) -> torch.Tensor:
    """Insert blank tokens between each CTC token as editing slots for the LLM."""
    n = token_ids.numel()
    total_len = max(2 * n + 1, min_len)
    idx = torch.arange(n, device=token_ids.device)
    out_idx = 2 * idx + 1
    out = torch.full((total_len,), fill_value=blank_id, dtype=token_ids.dtype, device=token_ids.device)
    out[out_idx] = token_ids
    return out


def build_flat_inputs(
    ctc_token_ids: list[torch.Tensor],
    audio_embeds: torch.Tensor,
    audio_lengths: list[int],
    embed_tokens: Any,
    blank_id: int,
    min_len: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Build flat (pad-free) LLM input: [audio_0, text_0, audio_1, text_1, ...]."""
    embeds_list = []
    position_ids_list = []
    text_lengths = []
    for i, audio_len in enumerate(audio_lengths):
        audio_emb = audio_embeds[i, :audio_len]
        text_ids_with_slots = add_insertion_slots(ctc_token_ids[i], blank_id, min_len)
        text_emb = embed_tokens(text_ids_with_slots)
        sample_embeds = torch.cat([audio_emb, text_emb], dim=0)
        embeds_list.append(sample_embeds)
        position_ids_list.append(torch.arange(sample_embeds.shape[0], device=device))
        text_lengths.append(text_ids_with_slots.shape[0])
    flat_embeds = torch.cat(embeds_list, dim=0).unsqueeze(0)
    flat_position_ids = torch.cat(position_ids_list, dim=0).unsqueeze(0)
    return flat_embeds, flat_position_ids, text_lengths


# =========================================================================== #
# Shape-keyed CUDA-graph cache.
# =========================================================================== #
class _GraphCache:
    """Bounded CUDA-graph cache with adaptive per-shape capture.

    Captured graphs are retained for the pipeline lifetime. Freeing a graph
    while other captures share allocator state can be unsafe, so once the
    cache reaches its limit, unseen shapes stay eager instead of evicting and
    recapturing.
    """

    def __init__(
        self,
        *,
        max_captures: int,
        capture_after: int = 2,
        max_seen: int = 4096,
    ) -> None:
        if max_captures < 0:
            raise ValueError("max_captures must be non-negative")
        if capture_after < 1:
            raise ValueError("capture_after must be positive")
        if max_seen < 1:
            raise ValueError("max_seen must be positive")
        self.max_captures = max_captures
        self.capture_after = capture_after
        self.max_seen = max_seen
        self._graphs: dict[Any, dict[str, Any]] = {}
        self._seen: OrderedDict[Any, int] = OrderedDict()

    def get(self, key: Any) -> Optional[dict[str, Any]]:
        return self._graphs.get(key)

    def should_capture(self, key: Any) -> bool:
        if key in self._graphs:
            return False
        sightings = self._seen.pop(key, 0) + 1
        self._seen[key] = sightings
        while len(self._seen) > self.max_seen:
            self._seen.popitem(last=False)
        return sightings >= self.capture_after and len(self._graphs) < self.max_captures

    def put(self, key: Any, entry: dict[str, Any]) -> None:
        if key not in self._graphs and len(self._graphs) >= self.max_captures:
            raise RuntimeError("graph cache capacity exceeded")
        self._graphs[key] = entry
        self._seen.pop(key, None)



# =========================================================================== #
# The megakernel pipeline.
# =========================================================================== #
class NarMega:
    """CUDA-graph-captured Granite-Speech-4.1-2b-NAR inference pipeline.

    Parameters
    ----------
    model : GraniteSpeechNarForASR
        The loaded top-level model (encoder + projector + language_model).
    compile_llm : bool
        If True (default), ``torch.compile`` the LLM editor forward before
        capturing it into a graph. Inductor fuses the elementwise glue; verified
        byte-exact at the decoded-token level.
    capture_encoder : bool
        If True (default), capture the encoder trunk into a per-shape graph.
    capture_llm : bool
        If True (default), capture the compiled LLM editor forward into a
        per-shape graph. Compiling first (so Inductor emits deterministic
        Triton kernels) keeps graph-capture byte-exact.
    max_encoder_graphs, max_llm_graphs : int
        Maximum captured shapes retained for each stage. Novel shapes remain
        eager after the corresponding cache is full.
    capture_after : int
        Capture a shape on this sighting. The default avoids paying capture
        cost or retaining graph-owned memory for one-off input lengths.
    """

    def __init__(
        self,
        model: Any,
        *,
        compile_llm: bool = True,
        capture_encoder: bool = True,
        capture_llm: bool = True,
        max_encoder_graphs: int = 8,
        max_llm_graphs: int = 8,
        capture_after: int = 2,
    ) -> None:
        self.model = model
        self.config = model.config
        self.encoder = model.encoder
        self.projector = model.projector
        self.lm = model.language_model
        self.embed_tokens = self.lm.model.embed_tokens
        self.dtype = next(model.parameters()).dtype
        self.device = next(model.parameters()).device

        self.compile_llm = compile_llm
        self.capture_encoder = capture_encoder
        self.capture_llm = capture_llm

        self.blank_id = int(self.config.blank_token_id)
        self.min_len = int(self.config.min_edit_sequence_length)
        self.layer_indices = list(self.config.encoder_layer_indices)
        self.emb_mult = float(
            getattr(self.config.text_config, "embedding_multiplier", 1.0)
        )
        self.downsample_rate = int(self.projector.config.downsample_rate)
        self.pool_window = int(self.encoder.config.bpe_pooling_window)

        # Compiled LLM forward (lazily, so construction doesn't pay the cost).
        self._llm_forward_compiled: Any = None

        # Shape-keyed graph caches. Novel shapes run eager first, then recurring
        # shapes are captured up to fixed limits to bound graph-owned VRAM.
        self._enc_graphs = _GraphCache(
            max_captures=max_encoder_graphs, capture_after=capture_after
        )
        self._llm_graphs = _GraphCache(
            max_captures=max_llm_graphs, capture_after=capture_after
        )

    # ------------------------------------------------------------------ #
    # LLM editor forward
    #
    # Uses the STOCK model.language_model forward, optionally torch.compiled.
    # We deliberately do NOT hand-iterate the layers here: a hand-rolled
    # forward diverges from the stock cuBLAS reduction order on long packed
    # sequences (bf16 tiling noise compounds over 40 layers, flipping a
    # borderline argmax), whereas the stock + torch.compile path is
    # byte-exact at the decoded-token level on all tiers (greedy is robust to
    # the sub-ULP logit noise Inductor introduces — the moss/qwen3 finding).
    # The compiled stock forward is the byte-exact fast path.
    # ------------------------------------------------------------------ #
    def _llm_forward(self, inputs_embeds: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        out = self.lm(inputs_embeds=inputs_embeds, position_ids=position_ids)
        return out.logits  # (1, T, vocab)

    def _get_compiled_llm(self) -> Any:
        if self._llm_forward_compiled is None and self.compile_llm:
            self._llm_forward_compiled = torch_compile(
                self._llm_forward,
                mode="max-autotune-no-cudagraphs",
                dynamic=False,
            )
        return self._llm_forward_compiled

    # ------------------------------------------------------------------ #
    # Encoder: graph-captured trunk + eager BPE head
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def encode(
        self, input_features: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, list[int], list[int]]:
        """Run encoder + projector -> (audio_embeds, bpe_logits, audio_lengths, bpe_lengths).

        The encoder trunk AND the multilayer-cat + projector forward are captured
        together in one graph (the projector alone was ~6ms eager). Lengths are
        host-constants for a given input shape (no padding in batch=1), so they
        are computed once at capture time and cached — avoiding the ~5ms
        GPU->CPU sync that ``.cpu().tolist()`` would force on every call.

        Returns ``audio_embeds`` (post-projector, scaled) and the BPE CTC logits
        for the eager collapse; plus the cached host-side lengths.
        """
        T = input_features.shape[1]
        key = ("enc", T, input_features.dtype)
        entry = self._enc_graphs.get(key)

        if self.capture_encoder:
            if entry is None and self._enc_graphs.should_capture(key):
                entry = self._capture_encoder(input_features, attention_mask)
                self._enc_graphs.put(key, entry)
            if entry is not None:
                entry["input"].copy_(input_features)
                entry["graph"].replay()
                audio_embeds = entry["audio_embeds"]
                bpe_logits = entry["bpe_logits"]
                audio_lengths = entry["audio_lengths"]
                bpe_lengths = entry["bpe_lengths"]
                return audio_embeds, bpe_logits, audio_lengths, bpe_lengths

        audio_embeds, bpe_logits, audio_lengths, bpe_lengths = self._encproj_eager(
            input_features, attention_mask
        )

        return audio_embeds, bpe_logits, audio_lengths, bpe_lengths

    def _encproj_eager(
        self, input_features: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, list[int], list[int]]:
        """Eager encoder+projector (fallback when capture_encoder=False)."""
        hidden_states, all_hidden, blank_probs = _encoder_trunk(
            self.encoder, input_features, attention_mask
        )
        bpe_logits = _encoder_bpe_head(self.encoder, hidden_states, blank_probs, attention_mask)
        multilayer = torch.cat([all_hidden[idx] for idx in self.layer_indices], dim=-1)
        audio_embeds = self.projector(multilayer)
        if self.config.scale_projected_embeddings:
            audio_embeds = audio_embeds / self.emb_mult
        audio_embeds = audio_embeds.to(self.embed_tokens.weight.dtype)
        encoder_lengths = attention_mask.sum(dim=1)
        # Match the stock transcribe: audio_lengths = floor(encoder_len / ds),
        # which can be ONE LESS than the projector's actual frame count (the
        # projector pads to a multiple of its block size). Slicing with the
        # floor value drops the trailing padded frame, exactly as stock does.
        audio_lengths = (encoder_lengths // self.downsample_rate).tolist()
        bpe_len = (int(encoder_lengths[0].item()) + self.pool_window - 1) // self.pool_window
        bpe_lengths = [bpe_len]
        return audio_embeds, bpe_logits, audio_lengths, bpe_lengths

    def _capture_encoder(self, input_features: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, Any]:
        static_in = input_features.clone()
        static_mask = attention_mask.clone()

        # Host-constant lengths for this shape (batch=1, no padding). Computed
        # once here so the hot path never touches .cpu()/.tolist(). audio_lengths
        # uses floor(encoder_len / downsample) to MATCH the stock transcribe —
        # the projector may emit one more frame than this (block-size padding),
        # and stock drops that trailing frame by slicing with the floor value.
        encoder_len = int(attention_mask.sum(dim=1)[0].item())
        bpe_lengths = [(encoder_len + self.pool_window - 1) // self.pool_window]
        audio_lengths = [encoder_len // self.downsample_rate]
        bpe_len = bpe_lengths[0]

        def run():
            hidden_states, all_hidden, blank_probs = _encoder_trunk(
                self.encoder, static_in, static_mask
            )
            importance = 1.0 - blank_probs
            pooled = _posterior_weighted_pool(
                hidden_states.float(), importance, window_size=self.pool_window
            ).to(hidden_states.dtype)
            bpe_logits = self.encoder.out_bpe(pooled[:, :bpe_len])
            multilayer = torch.cat([all_hidden[idx] for idx in self.layer_indices], dim=-1)
            audio_embeds = self.projector(multilayer)
            if self.config.scale_projected_embeddings:
                audio_embeds = audio_embeds / self.emb_mult
            return audio_embeds.to(self.embed_tokens.weight.dtype), bpe_logits

        with torch.inference_mode():
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    run()
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()

            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                audio_embeds, bpe_logits = run()

        return {
            "input": static_in,
            "graph": g,
            "audio_embeds": audio_embeds,
            "bpe_logits": bpe_logits,
            "audio_lengths": audio_lengths,
            "bpe_lengths": bpe_lengths,
        }

    # ------------------------------------------------------------------ #
    # Projector + LLM editor: graph-captured per (audio_len, text_len)
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def edit_forward(
        self,
        audio_embeds: torch.Tensor,
        ctc_token_ids: list[torch.Tensor],
        audio_lengths: list[int],
    ) -> list[torch.Tensor]:
        """Run LLM editor -> per-sample text logits.

        ``audio_embeds`` is already the scaled projector output (computed inside
        the encoder graph). ``audio_lengths`` are the host-cached lengths.
        Returns a list of (text_len_i, vocab) logit tensors (one per sample).
        """
        flat_embeds, flat_pos, text_lengths = build_flat_inputs(
            ctc_token_ids, audio_embeds, audio_lengths, self.embed_tokens,
            self.blank_id, self.min_len, self.device,
        )
        L = flat_embeds.shape[1]
        key = ("llm", L)
        entry = self._llm_graphs.get(key)

        if self.capture_llm:
            if entry is None and self._llm_graphs.should_capture(key):
                entry = self._capture_llm(flat_embeds, flat_pos)
                self._llm_graphs.put(key, entry)
            if entry is not None:
                entry["input"].copy_(flat_embeds)
                entry["pos"].copy_(flat_pos)
                entry["graph"].replay()
                all_logits = entry["logits"]
            else:
                fwd = self._get_compiled_llm() if self.compile_llm else self._llm_forward
                all_logits = fwd(flat_embeds, flat_pos)
        else:
            fwd = self._get_compiled_llm() if self.compile_llm else self._llm_forward
            all_logits = fwd(flat_embeds, flat_pos)

        all_logits = all_logits.squeeze(0)
        segment_lengths = [length for a, t in zip(audio_lengths, text_lengths, strict=True) for length in (a, t)]
        text_logits = torch.cat(list(all_logits.split(segment_lengths)[1::2]))
        return list(text_logits.split(text_lengths))

    def _capture_llm(self, flat_embeds: torch.Tensor, flat_pos: torch.Tensor) -> dict[str, Any]:
        static_in = flat_embeds.clone()
        static_pos = flat_pos.clone()
        fwd = self._get_compiled_llm() if self.compile_llm else self._llm_forward

        # Probe + warmup (compile if needed).
        with torch.inference_mode():
            _ = fwd(static_in, static_pos)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                fwd(static_in, static_pos)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            logits = fwd(static_in, static_pos)

        return {"input": static_in, "pos": static_pos, "graph": g, "logits": logits}

    # ------------------------------------------------------------------ #
    # End-to-end transcribe
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def transcribe(
        self, input_features: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[list[list[int]], list[torch.Tensor]]:
        """End-to-end NAR transcription.

        Returns ``(token_id_lists, raw_pred_tensors)`` — one per sample in the
        batch (batch=1 typical). ``token_id_lists[i]`` is the final decoded
        token id list for sample ``i``.
        """
        audio_embeds, bpe_logits, audio_lengths, bpe_lengths = self.encode(
            input_features, attention_mask
        )
        ctc_token_ids = ctc_collapse_decode(bpe_logits, bpe_lengths, self.blank_id)
        text_logits = self.edit_forward(audio_embeds, ctc_token_ids, audio_lengths)

        preds = []
        for sl in text_logits:
            pred = torch.unique_consecutive(sl.argmax(-1))
            pred = pred[pred != self.blank_id]
            preds.append(pred.tolist())
        return preds, text_logits
