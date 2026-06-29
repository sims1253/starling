"""CUDA-graph-captured MOSS-Transcribe audio encoder.

The audio encoder (``Qwen3OmniMoeAudioEncoder``) is a 32-layer transformer over
chunked mel features with windowed attention (cu_seqlens).  For a fixed audio
length the chunking, valid_indices, and cu_seqlens are all static, so the whole
encoder + adapter forward can be captured into one ``torch.cuda.CUDAGraph``
replay -- collapsing ~hundreds of per-layer kernel launches into one replay.

Byte-exactness
--------------
The captured forward calls the encoder's *own* modules unchanged (we only
precompute the chunking bookkeeping that the encoder would otherwise recompute
via Python/tensor ops each call, and feed it through the ``kwargs`` pop-path the
stock forward already supports).  Graph replay of the model's own ops is
bit-exact with eager, so the output matches the golden reference exactly.

Public API
----------
``GraphedAudioEncoder(audio_model, audio_adapter)``
``GraphedAudioEncoder.forward(audio_data, seqlens) -> audio_embeds  (1, N, 2048) bf16``
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_feat_extract_output_lengths(input_lengths: torch.Tensor) -> torch.Tensor:
    """Stock module-level CNN output length (deepstack formula).

    Matches ``transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe.
    _get_feat_extract_output_lengths`` -- the one used by ``get_valid_indices``
    and ``get_audio_cu_seqlens`` (NOT the encoder's own method, which is a
    simpler ``(len-1)//2+1`` and is a separate code path).
    """
    input_lengths = input_lengths.long()
    input_lengths_leave = input_lengths % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    return ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13


def _compute_chunking(
    input_features: torch.Tensor,
    feature_lens: torch.Tensor,
    n_window: int,
    n_window_infer: int,
):
    """Precompute padded_feature / chunk_lengths / valid_indices / cu_seqlens.

    Mirrors the stock ``chunk_and_pad_features`` / ``get_valid_indices`` /
    ``get_audio_cu_seqlens`` helpers (from
    ``transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe``) so we can
    hand them to the stock forward via its kwargs pop-path.  All four are pure
    functions of (input_features shape, feature_lens), so for a fixed audio
    length they are constant across calls -- which is what makes the encoder
    graph-capturable.
    """
    device = feature_lens.device
    # ---- chunk_and_pad_features ----
    chunk_num = torch.ceil(feature_lens.float() / (n_window * 2)).long()
    chunk_lengths = torch.full(
        (chunk_num.sum().item(),), n_window * 2, dtype=torch.long, device=device
    )
    tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
    chunk_lengths[tail_chunk_index] = feature_lens % (n_window * 2)
    chunk_lengths = torch.where(chunk_lengths == 0, n_window * 2, chunk_lengths)

    chunk_list = input_features.T.split(chunk_lengths.tolist(), dim=0)
    padded_feature = nn.utils.rnn.pad_sequence(chunk_list, batch_first=True).transpose(
        1, 2
    )

    # ---- get_valid_indices ----
    feature_lens_after_cnn = _get_feat_extract_output_lengths(chunk_lengths)
    max_len_after_cnn = feature_lens_after_cnn.max().item()
    mask = torch.arange(max_len_after_cnn, device=device) < feature_lens_after_cnn.unsqueeze(
        1
    )
    valid_indices = mask.flatten().nonzero().squeeze(-1)

    # ---- get_audio_cu_seqlens ----
    aftercnn_lens = _get_feat_extract_output_lengths(feature_lens)
    n_window_ratio = n_window_infer // (n_window * 2)
    window_aftercnn = max_len_after_cnn * n_window_ratio

    cu_chunk_lens = [0]
    for cnn_len in aftercnn_lens.tolist():
        cu_chunk_lens += [window_aftercnn] * (cnn_len // window_aftercnn)
        remainder = cnn_len % window_aftercnn
        if remainder != 0:
            cu_chunk_lens += [remainder]
    cu_seqlens = torch.tensor(cu_chunk_lens, device=device).cumsum(-1, dtype=torch.int32)

    return padded_feature.contiguous(), chunk_lengths, valid_indices, cu_seqlens


class GraphedAudioEncoder(nn.Module):
    """MOSS audio encoder + adapter, byte-exact.

    Two modes:
    * ``mode="eager"`` (default) -- runs the stock encoder + adapter forward
      eager.  Byte-exact, but re-pays per-layer launch overhead each call.
      Robust (the stock attention path does ``.tolist()`` chunk splits which
      abort CUDA-graph capture).
    * ``mode="cudagraph"`` -- captures the forward into a CUDA graph.  Requires
      a graph-safe attention path (set ``attn_implementation="flash_attention"``
      on the encoder so the cu_seqlens path is used instead of the
      ``.tolist()`` chunk split).  When the captured path is unavailable
      (eager attention) this raises a clear error on first capture.

    Parameters
    ----------
    audio_model : Qwen3OmniMoeAudioEncoder
    audio_adapter : MossGatedMLP
    mode : {"eager","cudagraph"}
    """

    def __init__(self, audio_model, audio_adapter, *, mode: str = "eager") -> None:
        super().__init__()
        if mode not in ("eager", "cudagraph"):
            raise ValueError(f"unknown mode {mode!r}; expected eager/cudagraph")
        self.audio_model = audio_model
        self.audio_adapter = audio_adapter
        cfg = audio_model.config
        self.n_window = int(cfg.n_window)
        self.n_window_infer = int(cfg.n_window_infer)
        self.mode = mode

        self._graph: Optional[torch.cuda.CUDAGraph] = None
        self._static_input: Optional[torch.Tensor] = None
        self._static_seqlens: Optional[torch.Tensor] = None
        self._static_output: Optional[torch.Tensor] = None
        self._captured_for: Optional[tuple] = None  # (mel_len,) cache key

    @torch.inference_mode()
    def _capture(self, audio_data: torch.Tensor, seqlens: torch.Tensor) -> None:
        """Warmup on a side stream then capture the eager encoder+adapter."""
        device = audio_data.device
        self._static_input = torch.empty_like(audio_data)
        self._static_input.copy_(audio_data)
        self._static_seqlens = torch.empty_like(seqlens)
        self._static_seqlens.copy_(seqlens)

        side = torch.cuda.Stream(device=device)
        side.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(side):
            for _ in range(3):
                _ = self._forward_eager(self._static_input, self._static_seqlens)
        torch.cuda.current_stream(device).wait_stream(side)

        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._static_output = self._forward_eager(
                self._static_input, self._static_seqlens
            )

        self._captured_for = (int(audio_data.shape[1]),)

    def _forward_eager(self, audio_data: torch.Tensor, seqlens: torch.Tensor) -> torch.Tensor:
        """Encoder + adapter (stock modules), chunking precomputed per-shape."""
        pf, cl, vi, cu = _compute_chunking(
            audio_data, seqlens, self.n_window, self.n_window_infer
        )
        # The stock forward pops these from kwargs and uses them verbatim.
        feats = self.audio_model(
            input_features=audio_data,
            feature_lens=seqlens,
            padded_feature=pf,
            chunk_lengths=cl,
            valid_indices=vi,
            cu_seqlens=cu,
        ).last_hidden_state
        return self.audio_adapter(feats)

    def forward(self, audio_data: torch.Tensor, seqlens: torch.Tensor) -> torch.Tensor:
        """Run the encoder+adapter. Returns audio_embeds ``(N, 2048)`` bf16.

        In ``cudagraph`` mode re-captures automatically if the mel length changed
        (different audio).
        """
        if audio_data.dtype != self.audio_model.dtype:
            audio_data = audio_data.to(self.audio_model.dtype)

        if self.mode == "eager":
            return self._forward_eager(audio_data, seqlens)

        key = (int(audio_data.shape[1]),)
        if self._graph is None or self._captured_for != key:
            self._capture(audio_data, seqlens)
            return self._static_output.clone()

        if audio_data.shape != self._static_input.shape:
            raise RuntimeError(
                f"cudagraph captured for shape {tuple(self._static_input.shape)} "
                f"but got {tuple(audio_data.shape)}"
            )
        self._static_input.copy_(audio_data)
        # seqlens is fixed per captured shape (single stream); no copy needed.
        self._graph.replay()
        return self._static_output.clone()
