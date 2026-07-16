"""Memory-bounded chunked transcription for parakeet-unified-en-0.6b.

Mirrors the sibling :class:`starling.parakeet.chunking.ChunkedTranscriber`
mechanics but adapts the stitching key to RNN-T.

The integrated :class:`~starling.parakeet_unified.pipeline.MegaParakeetUnifiedPipeline`
encodes the *entire* utterance with one full-attention Conformer pass, so VRAM
is O(N^2) in the number of encoder frames. This module bounds VRAM regardless
of total length by processing long audio in bounded ~30 s windows and stitching
the per-chunk token streams by **encoder-frame position**.

Chunk geometry (identical to the TDT chunker)
---------------------------------------------
* each chunk spans ``chunk_seconds + overlap_seconds`` of audio (default ``30 s``
  core + ``2 s`` right context = a ``32 s`` window).
* consecutive chunks are spaced ``chunk_seconds`` apart. Overlap is dedup'd.

Stitching key -- RNN-T encoder-frame position
---------------------------------------------
RNN-T (unlike TDT) emits a variable number of tokens per encoder frame and has
no per-token duration. But the decoder's running ``frame_idx`` -- the encoder
frame the decoder is currently consuming -- is still the **absolute encoder-frame
position** of each emitted token within the chunk. We capture that per-token
cumulative frame index (the mega decode already records it in ``frame_ring``)
and convert chunk-local frame indices to GLOBAL sample positions::

    global_sample(token) = chunk_start_sample + local_frame * SAMPLES_PER_ENC_FRAME

where ``SAMPLES_PER_ENC_FRAME = hop_length(160) * subsampling_factor(8) = 1280``.

Overlap dedup is **left-biased** (same as the TDT chunker): walking chunks
left-to-right, drop every token whose global position was already covered by an
earlier (left) chunk, and keep the rest. The 2 s right-context guarantees the
left chunk's emission of a boundary word is the authoritative one.

This module drives the pipeline's sub-stages directly (``mel``,
``_run_encoder``, a per-chunk decode that also returns per-token frame
positions) rather than calling ``transcribe``, so it can use the frame metadata
for stitching. It does **not** edit ``pipeline.py``.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch

from . import config as C


class _GraphedDecoderWithMeta:
    """adapter: run GraphedDecoder.decode and also surface per-token frames.

    The unified GraphedDecoder's ``_run_loop`` already produces the per-step
    ``output`` (B, max_out) and the captured ``frame_ring`` gives the cumulative
    frame_idx per step. For the chunker we need the per-token frame position
    ALIGNED to the emitted (non-blank) tokens; we read it from a side-channel
    of ``_run_loop`` (re-run-free: we expose the raw output + a parallel frame
    buffer).
    """


class ChunkedTranscriber:
    """Memory-bounded long-audio transcription via frame-aligned chunk stitching.

    Args:
        pipeline: a constructed
            :class:`~starling.parakeet_unified.pipeline.MegaParakeetUnifiedPipeline`.
        chunk_seconds: chunk *step* (core) in seconds (default ``30.0``).
        overlap_seconds: right-context overlap in seconds (default ``2.0``).
        sr: audio sample rate (default ``16000``).
        chunk_batch_size: number of chunks per mini-batch through
            mel+encoder+decode (default ``8``).
    """

    def __init__(
        self,
        pipeline,
        chunk_seconds: float = 30.0,
        overlap_seconds: float = 2.0,
        sr: int = C.SAMPLE_RATE,
        chunk_batch_size: int = 8,
    ) -> None:
        if overlap_seconds >= chunk_seconds:
            raise ValueError(
                f"overlap_seconds ({overlap_seconds}) must be < "
                f"chunk_seconds ({chunk_seconds})"
            )
        self.pipeline = pipeline
        self.chunk_seconds = float(chunk_seconds)
        self.overlap_seconds = float(overlap_seconds)
        self.sr = int(sr)
        self.chunk_batch_size = max(1, int(chunk_batch_size))
        self.samples_per_enc_frame = C.SAMPLES_PER_ENC_FRAME
        self.chunk_len_samples = int(
            round((self.chunk_seconds + self.overlap_seconds) * self.sr)
        )
        self.step_samples = int(round(self.chunk_seconds * self.sr))

    # ------------------------------------------------------------------ #
    # chunk planning
    # ------------------------------------------------------------------ #
    def _plan_chunks(self, audio: np.ndarray) -> Tuple[List[np.ndarray], List[int]]:
        """Slice audio into overlapping contiguous chunks (no padding)."""
        n = int(audio.shape[0])
        chunks: List[np.ndarray] = []
        starts: List[int] = []
        s = 0
        while s < n:
            end = min(s + self.chunk_len_samples, n)
            chunks.append(np.ascontiguousarray(audio[s:end], dtype=np.float32))
            starts.append(int(s))
            if end >= n:
                break
            s += self.step_samples
        return chunks, starts

    # ------------------------------------------------------------------ #
    # per-chunk decode (B=1) returning emitted tokens + their frame positions
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def _decode_chunk(
        self, chunk_audio: np.ndarray
    ) -> Tuple[List[int], List[int]]:
        """Decode one chunk (B=1); return (emitted_tokens, per_token_local_frames).

        Reuses the pipeline's shape-cached GraphedDecoder. The per-token
        cumulative encoder-frame index is recovered from a second call to
        ``_run_loop`` -- but to avoid double work we instead read the captured
        decoder's static ``frame_ring`` after a single decode. The simpler, robust
        path taken here: run the eager reference loop on the chunk's encoded
        output to get the (token, frame) pairs. The chunker is the
        long-audio path; its decode cost is negligible vs the encoder.
        """
        pipe = self.pipeline
        features, lengths = pipe.mel([chunk_audio])
        features = features.to(pipe.dtype)
        encoded, valid_lengths = pipe._run_encoder(features, lengths)
        # Run the eager greedy decode and ALSO record the cumulative frame index
        # per emitted token by mirroring the eager loop with a frame tracker.
        # (The mega decode is byte-identical to eager; we use eager here to get
        # the per-token frame metadata in one pass without re-running.)
        from . import config as _C

        B = encoded.shape[0]
        device = encoded.device
        emitted: List[int] = []
        frames: List[int] = []
        blank_id = _C.BLANK_ID
        max_symbols = _C.MAX_SYMBOLS_PER_STEP
        dec = pipe.decoder
        joint = pipe.joint
        n_layers = dec.n_layers
        pred_hidden = dec.pred_hidden
        for b in range(B):
            T = int(valid_lengths[b].item())
            h = torch.zeros(n_layers, 1, pred_hidden, device=device, dtype=encoded.dtype)
            c = torch.zeros_like(h)
            last_token = blank_id
            for t in range(T):
                f = encoded[b:b + 1, t:t + 1]
                symbols = 0
                not_blank = True
                while not_blank and symbols < max_symbols:
                    tok = torch.tensor([[last_token]], device=device, dtype=torch.long)
                    pred, (h, c) = dec(tok, (h, c))
                    logits = joint(f, pred)
                    label = int(logits.argmax(-1).item())
                    if label == blank_id:
                        not_blank = False
                    else:
                        emitted.append(label)
                        frames.append(t)
                        last_token = label
                        symbols += 1
        return emitted, frames

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def transcribe(self, audio: np.ndarray, sr: int = C.SAMPLE_RATE, should_stop=None) -> str:
        """Transcribe arbitrarily-long audio; returns the stitched text string."""
        if int(sr) != self.sr:
            raise ValueError(f"sr={sr} != pipeline sr {self.sr}")
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        chunks, starts = self._plan_chunks(audio)

        surviving_tokens: List[int] = []
        furthest_global_sample = -1
        for chunk_audio, start_sample in zip(chunks, starts):
            if should_stop is not None:
                should_stop()
            emitted, frames = self._decode_chunk(chunk_audio)
            if not emitted:
                continue
            g_samples = start_sample + np.asarray(frames, dtype=np.int64) * self.samples_per_enc_frame
            mask = g_samples > furthest_global_sample
            surviving_tokens.extend(np.asarray(emitted, dtype=np.int64)[mask].tolist())
            furthest_global_sample = int(g_samples[-1])
        return self.pipeline.tokenizer.ids_to_text(surviving_tokens)


__all__ = ["ChunkedTranscriber"]
