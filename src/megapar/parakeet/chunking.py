"""Memory-bounded chunked transcription for ``nvidia/parakeet-tdt-0.6b-v3``.

The integrated :class:`~megapar.parakeet.pipeline.MegaParakeetPipeline` encodes
the *entire* utterance with one full-attention Conformer pass, so VRAM is
O(N^2) in the number of encoder frames (and the encoder clifs at
``max_position_embeddings = 5000`` frames, ~6.5 min). On an RTX 5090 a single
clip is feasible only to ~5 min at batch=1 (~2 GB) before VRAM explodes; the
prior worker OOM'd trying 10-15 min clips.

This module bounds VRAM *regardless of total length* by processing long audio
in bounded ~30 s windows and stitching the per-chunk token streams by
frame-aligned TDT durations. VRAM is then a function of the (constant) chunk
size, not the total length, so 1 h+ audio transcribes with the same peak VRAM
as a single 30 s chunk.

Chunk geometry
--------------
Following the standard RNN-T/TDT chunking pattern:

* each chunk spans ``chunk_seconds + overlap_seconds`` of audio (default ``30 s``
  core + ``2 s`` right context = a ``32 s`` window). The right context gives the
  encoder/decoder boundary continuity so a word straddling a chunk edge is not
  truncated.
* consecutive chunks are spaced ``chunk_seconds`` apart (the *step*). So for a
  long clip the windows are ``[0-32s], [30-62s], [60-92s], ...`` -- a ``2 s``
  overlap between every adjacent pair.
* each chunk is transcribed end-to-end through the full pipeline (GPU mel ->
  graphed Conformer encoder -> graphed TDT decode) at its **natural** length
  (only the final, partial chunk is shorter; we never zero-pad, because padded
  frames contaminate the encoder's full self-attention and would break
  byte-exactness on single-chunk inputs).

Stitching key -- TDT durations
------------------------------
A TDT (token-and-duration transducer) emits, at every decode step, *both* a
token and a duration (the encoder-frame advance for that step, in
``[0, 1, 2, 3, 4]`` with a forced ``>= 1`` advance on blank-with-0). The
decoder's running frame pointer is therefore the **absolute encoder-frame
position** of each emitted token *within the chunk*. :meth:`GraphedDecoder.decode_with_durations`
returns this per-token cumulative frame index.

We convert chunk-local frame indices to GLOBAL sample positions::

    global_sample(token) = chunk_start_sample + local_frame * SAMPLES_PER_ENC_FRAME

where ``SAMPLES_PER_ENC_FRAME = hop_length(160) * subsampling_factor(8) = 1280``
audio samples per encoder frame (verified against ``robust_bench.json``:
60 s -> T_enc = 751 = 960000 / 1280 + 1).

Overlap dedup is **left-biased**: walking chunks left-to-right, we drop every
token whose global position was already covered by an earlier (left) chunk
(``global_sample <= furthest_global_sample_so_far``), and keep the rest. The
2 s right-context guarantees the left chunk's emission of a boundary word is the
authoritative one; the right chunk's duplicate of that region is discarded. The
surviving tokens are concatenated and decoded to text.

This module drives the pipeline's sub-stages directly (``mel``,
``_run_encoder``, ``_get_decoder``) rather than calling
``MegaParakeetPipeline.transcribe`` so it can use
:meth:`GraphedDecoder.decode_with_durations` (which returns the per-token frame
positions needed for stitching) instead of the text-only
:meth:`GraphedDecoder.decode`. It does **not** edit ``pipeline.py``.

Memory safety
-------------
Before every chunk's forward pass we probe ``torch.cuda.mem_get_info``; if free
VRAM drops below ``min_free_vram_gb`` (default 24 GB on this 32 GB card) the
chunk is **aborted with a clear error** rather than risk an OOM. Between chunks
``torch.cuda.empty_cache()`` releases the per-chunk intermediates, so peak VRAM
tracks a single chunk (~1.5 GB) no matter how long the audio is.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch

# Encoder-frame geometry, derived once from the live pipeline in __init__ but
# with safe model-specific defaults (parakeet-tdt-0.6b-v3: hop=160, subsample=8).
_DEFAULT_HOP = 160
_DEFAULT_SUBSAMPLE = 8


class ChunkedTranscriber:
    """Memory-bounded long-audio transcription via frame-aligned chunk stitching.

    Args:
        pipeline: a constructed :class:`MegaParakeetPipeline` (model loaded on
            cuda). The chunker drives its ``mel`` / ``_run_encoder`` /
            ``_get_decoder`` sub-stages and its ``processor`` directly.
        chunk_seconds: the chunk *step* (core) in seconds (default ``30.0``).
            Each chunk is ``chunk_seconds + overlap_seconds`` long.
        overlap_seconds: the right-context overlap in seconds (default ``2.0``).
            Adjacent chunks overlap by this much; this region is dedup'd.
        sr: audio sample rate (default ``16000``).
        min_free_vram_gb: memory-safety guard. Before each chunk's forward pass,
            if free VRAM (``torch.cuda.mem_get_info``) is below this, the chunk
            is aborted with :class:`MemoryError` instead of risking an OOM.
            Default ``24.0`` GB (the 32 GB card is shared; we cap our own use at
            ~8 GB and leave headroom for other processes).
    """

    def __init__(
        self,
        pipeline,
        chunk_seconds: float = 30.0,
        overlap_seconds: float = 2.0,
        sr: int = 16000,
        min_free_vram_gb: float = 24.0,
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
        self.min_free_vram_gb = float(min_free_vram_gb)

        # Derive samples-per-encoder-frame from the live pipeline so this is
        # robust to a different mel hop / subsampling factor (rather than
        # hard-coding 1280). Fall back to the parakeet-tdt-0.6b-v3 defaults.
        hop = int(getattr(pipeline.mel, "hop_length", _DEFAULT_HOP))
        enc_cfg = {}
        try:
            enc_cfg = pipeline.model.config.to_dict().get("encoder_config", {}) or {}
        except Exception:
            enc_cfg = {}
        subsample = int(enc_cfg.get("subsampling_factor", _DEFAULT_SUBSAMPLE))
        self.samples_per_enc_frame = hop * subsample
        if self.samples_per_enc_frame <= 0:
            self.samples_per_enc_frame = _DEFAULT_HOP * _DEFAULT_SUBSAMPLE

        # Chunk geometry in samples.
        self.chunk_len_samples = int(
            round((self.chunk_seconds + self.overlap_seconds) * self.sr)
        )
        self.step_samples = int(round(self.chunk_seconds * self.sr))

    # ------------------------------------------------------------------ #
    # chunk planning
    # ------------------------------------------------------------------ #
    def _plan_chunks(self, audio: np.ndarray) -> Tuple[List[np.ndarray], List[int]]:
        """Slice ``audio`` into overlapping contiguous chunks (no padding).

        Returns ``(chunks, start_samples)``. The last chunk may be shorter than
        ``chunk_len_samples`` (a partial tail); every other chunk is full-width.
        We never zero-pad, because padded frames participate in the encoder's
        full self-attention and would perturb the real frames (breaking the
        single-chunk byte-exactness guarantee).
        """
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
    # VRAM guard
    # ------------------------------------------------------------------ #
    def _free_vram_gb(self) -> float:
        free, _total = torch.cuda.mem_get_info()
        return free / (1024.0 ** 3)

    # ------------------------------------------------------------------ #
    # per-chunk decode (drives the pipeline sub-stages directly)
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def _decode_chunk(
        self, chunk_audio: np.ndarray
    ) -> Tuple[str, List[int], List[int], int, dict]:
        """Run mel -> encoder -> graphed decode (B=1) for one chunk.

        Returns ``(text, tokens, local_frames, valid_enc_frames, timing)`` where
        ``tokens`` / ``local_frames`` are the per-token ids and cumulative local
        encoder-frame indices (see :class:`GraphedDecoder.decode_with_durations`),
        and ``timing`` has per-stage ms (mel/encoder/decode/total) from cuda
        events. Raises :class:`MemoryError` if the VRAM guard trips.
        """
        free_gb = self._free_vram_gb()
        if free_gb < self.min_free_vram_gb:
            raise MemoryError(
                f"chunked: free VRAM {free_gb:.2f} GB < "
                f"{self.min_free_vram_gb:.2f} GB guard; aborting chunk "
                f"(reduce chunk_seconds or free GPU memory)"
            )

        pipe = self.pipeline

        def _timed(fn):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            out = fn()
            end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(end), out

        mel_ms, (input_features, attention_mask) = _timed(lambda: pipe.mel([chunk_audio]))
        input_features = input_features.to(pipe.dtype)

        encoder_ms, (pooler, valid_lengths) = _timed(
            lambda: pipe._run_encoder(input_features, attention_mask)
        )

        decoder = pipe._get_decoder(pooler, valid_lengths)
        decode_ms, (texts, meta_tokens, meta_frames) = _timed(
            lambda: decoder.decode_with_durations(
                pooler, valid_lengths, pipe.processor
            )
        )

        timing = {
            "mel_ms": float(mel_ms),
            "encoder_ms": float(encoder_ms),
            "decode_ms": float(decode_ms),
            "total_ms": float(mel_ms + encoder_ms + decode_ms),
        }
        valid_enc = int(valid_lengths[0].item())
        return texts[0], meta_tokens[0], meta_frames[0], valid_enc, timing

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def transcribe(self, audio: np.ndarray, sr: int = 16000) -> str:
        """Transcribe arbitrarily-long audio; returns the stitched text string."""
        return self.transcribe_with_timing(audio, sr=sr)[0]

    @torch.inference_mode()
    def transcribe_with_timing(
        self, audio: np.ndarray, sr: int = 16000
    ) -> Tuple[str, dict]:
        """Transcribe and return ``(text, summary)``.

        ``summary`` contains: ``total_ms`` (wall, cuda-event bracketed over the
        whole multi-chunk run), ``audio_seconds``, ``n_chunks``,
        ``n_tokens_surviving``, ``n_stitches`` (overlap-region tokens dropped),
        ``peak_vram_gb`` (``torch.cuda.max_memory_allocated`` over the whole run,
        reset at the start -- this is the key bounded-memory metric), and
        ``per_chunk`` (per-chunk ms + token counts).
        """
        if int(sr) != self.sr:
            raise ValueError(f"sr={sr} != pipeline sr {self.sr}")
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        chunks, starts = self._plan_chunks(audio)
        assert len(chunks) >= 1, "must produce at least one chunk"

        surviving_tokens: List[int] = []
        # furthest global sample position covered by any KEPT token so far.
        # left-biased dedup: drop any token whose global_sample <= this.
        furthest_global_sample = -1
        per_chunk: List[dict] = []
        n_stitches = 0

        # Peak-VRAM is measured over the whole run (one chunk's worth because
        # we empty_cache() between chunks); reset so it reflects this call only.
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        t_start = torch.cuda.Event(enable_timing=True)
        t_end = torch.cuda.Event(enable_timing=True)
        t_start.record()

        for ci, (chunk_audio, start_sample) in enumerate(zip(chunks, starts)):
            _text, tokens_c, frames_c, valid_enc, timing = self._decode_chunk(chunk_audio)

            # Frame-aligned left-biased dedup. tokens_c / frames_c are in
            # emission order (non-decreasing frame). Convert each to a global
            # sample position; keep those beyond the previously-covered furthest
            # position, drop the (overlap) prefix. Because frames are
            # non-decreasing within a chunk, the kept tokens form a contiguous
            # suffix of the chunk's emission.
            kept_here = 0
            chunk_furthest = furthest_global_sample
            for tok, lf in zip(tokens_c, frames_c):
                g_sample = start_sample + int(lf) * self.samples_per_enc_frame
                if g_sample > furthest_global_sample:
                    surviving_tokens.append(int(tok))
                    kept_here += 1
                    if g_sample > chunk_furthest:
                        chunk_furthest = g_sample
                else:
                    n_stitches += 1
            furthest_global_sample = chunk_furthest

            per_chunk.append(
                {
                    "chunk_idx": ci,
                    "start_sample": int(start_sample),
                    "start_s": start_sample / self.sr,
                    "n_samples": int(chunk_audio.shape[0]),
                    "duration_s": chunk_audio.shape[0] / self.sr,
                    "valid_enc_frames": int(valid_enc),
                    "tokens_emitted": int(len(tokens_c)),
                    "tokens_kept": int(kept_here),
                    **timing,
                }
            )

            # Release per-chunk intermediates so peak VRAM tracks a single chunk.
            torch.cuda.empty_cache()

        t_end.record()
        torch.cuda.synchronize()
        total_ms = float(t_start.elapsed_time(t_end))
        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024.0 ** 3)

        text = self.pipeline.processor.batch_decode(
            [surviving_tokens], skip_special_tokens=True
        )[0]

        summary = {
            "total_ms": total_ms,
            "audio_seconds": audio.shape[0] / self.sr,
            "n_chunks": len(chunks),
            "n_tokens_surviving": int(len(surviving_tokens)),
            "n_stitches": int(n_stitches),
            "peak_vram_gb": float(peak_vram_gb),
            "chunk_seconds": self.chunk_seconds,
            "overlap_seconds": self.overlap_seconds,
            "samples_per_enc_frame": self.samples_per_enc_frame,
            "per_chunk": per_chunk,
        }
        return text, summary
