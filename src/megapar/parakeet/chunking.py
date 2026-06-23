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
Before every mini-batch's forward pass we probe ``torch.cuda.mem_get_info``;
:meth:`_effective_batch_size` shrinks the batch so that
``per_chunk_vram_gb * B + vram_headroom_gb <= free`` (reducing ``B`` rather than
aborting), and a hard ``min_free_vram_gb`` floor aborts only when truly starved.
Between mini-batches ``torch.cuda.empty_cache()`` releases the per-batch
intermediates, so peak VRAM tracks ONE mini-batch
(``~per_chunk_vram_gb * chunk_batch_size``) no matter how long the audio is.

Batching
--------
The mel extractor, Conformer encoder and graphed TDT decoder are all already
batched, so :class:`ChunkedTranscriber` groups the planned chunks into
mini-batches of up to ``chunk_batch_size`` (default ``8``) and runs each through
one set of batched mel+encoder+decode forwards -- turning ~121 sequential B=1
iterations for 1 h of audio into ~16 B=8 iterations, recovering most of the
megakernel pipeline's batched throughput. Per-chunk token+durations streams are
split out of the batched result and stitched exactly as in the sequential path.
A single chunk (audio <= one window) forms a mini-batch of ``B=1`` and is
byte-exact with the direct pipeline path.
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
        min_free_vram_gb: hard memory floor. Before each batch's forward pass,
            if free VRAM (``torch.cuda.mem_get_info``) is below this, the batch
            is aborted with :class:`MemoryError`. Default ``8.0`` GB (only abort
            when genuinely unable to proceed; the adaptive batch-size guard
            :meth:`_effective_batch_size` is the primary OOM defence and reduces
            B before this floor is ever reached).
        chunk_batch_size: number of chunks processed simultaneously per
            mini-batch through mel+encoder+decode (default ``8``). The mel
            extractor, Conformer encoder and graphed TDT decoder are all already
            batched (they take ``(B, ...)`` inputs), so grouping ``B`` chunks
            into one forward recovers the megakernel pipeline's batched
            throughput: ~16 sequential B=8 iterations for 1 h of audio instead
            of ~121 sequential B=1 iterations. The last mini-batch may contain
            fewer than ``chunk_batch_size`` chunks (it runs at its natural
            ``B = remainder``; no dummy padding, see Memory safety / batching).
            ``1`` reproduces the original one-chunk-at-a-time behaviour exactly.
        per_chunk_vram_gb: estimated peak VRAM cost of ONE ~32 s chunk through
            mel+encoder+decode, used by :meth:`_effective_batch_size` to size
            each mini-batch from the live free-VRAM reading (default ``2.0``).
        vram_headroom_gb: headroom (GB) reserved for the resident model + any
            other GPU processes when sizing a mini-batch (default ``4.0``).

    Batching & memory safety
    ------------------------
    Chunks are grouped left-to-right into mini-batches of up to
    ``chunk_batch_size`` (the last group may be smaller -- its natural ``B``).
    Each mini-batch is run through ``pipe.mel(batch)`` -> ``_run_encoder`` ->
    ``GraphedDecoder.decode_with_durations`` in ONE set of batched forwards, so
    ``B`` chunks pay one (amortised) graph capture + one set of per-stage launches
    instead of ``B`` separate ones. The per-chunk token+durations streams are
    then split back out of the batched result and stitched exactly as in the
    sequential path (frame-aligned left-biased dedup by global sample position).

    The mel extractor pads the mini-batch's audio to the longest chunk and emits
    a correct per-element attention mask; the encoder masks padded frames out of
    self-attention; and the decoder's per-element ``valid_lengths`` stop each
    chunk's decode at its own boundary. So a shorter chunk padded within a batch
    decodes identically to its natural length (the decoder never reads past
    ``valid_lengths``). A single chunk (audio <= one window) therefore forms a
    mini-batch of ``B=1`` and is byte-exact with the direct pipeline path.

    Before each mini-batch, :meth:`_effective_batch_size` reads free VRAM and
    shrinks ``B`` so that ``per_chunk_vram_gb * B + vram_headroom_gb <= free``;
    between mini-batches ``torch.cuda.empty_cache()`` releases intermediates so
    peak VRAM tracks ONE mini-batch (``~per_chunk_vram_gb * chunk_batch_size``),
    independent of total audio length.
    """

    def __init__(
        self,
        pipeline,
        chunk_seconds: float = 30.0,
        overlap_seconds: float = 2.0,
        sr: int = 16000,
        min_free_vram_gb: float = 8.0,
        chunk_batch_size: int = 8,
        per_chunk_vram_gb: float = 2.0,
        vram_headroom_gb: float = 4.0,
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
        self.chunk_batch_size = max(1, int(chunk_batch_size))
        self.per_chunk_vram_gb = float(per_chunk_vram_gb)
        self.vram_headroom_gb = float(vram_headroom_gb)

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

    def _effective_batch_size(self, desired_b: int) -> int:
        """Largest mini-batch size ``<= desired_b`` that fits current free VRAM.

        Each ~32 s chunk costs ~``per_chunk_vram_gb`` at peak; we reserve
        ``vram_headroom_gb`` for the resident model + other GPU processes. This
        is the primary OOM defence for the batched path: it *reduces* ``B``
        (rather than aborting) when free VRAM is low. With the defaults
        (``per_chunk_vram_gb=2.0``, ``vram_headroom_gb=4.0``) the full
        ``chunk_batch_size=8`` is used while free VRAM >= ~20 GB and shrinks
        below that, never OOM'ing. At least ``1`` is always returned (the hard
        :attr:`min_free_vram_gb` floor in :meth:`_decode_batch` catches the
        truly-starved case).
        """
        free_gb = self._free_vram_gb()
        max_safe = int((free_gb - self.vram_headroom_gb) / self.per_chunk_vram_gb)
        return max(1, min(int(desired_b), max_safe))

    # ------------------------------------------------------------------ #
    # batched decode (drives the pipeline sub-stages on B chunks at once)
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def _decode_batch(
        self, batch_audio: List[np.ndarray]
    ) -> Tuple[List[str], List[List[int]], List[List[int]], List[int], dict]:
        """Run mel -> encoder -> graphed decode for ``B`` chunks at once.

        The mel extractor, Conformer encoder and graphed TDT decoder are all
        already batched (they take ``(B, ...)`` inputs), so this feeds the whole
        mini-batch through one set of forwards. ``B = len(batch_audio)``; the
        mel extractor pads to the longest chunk in the batch and emits a correct
        per-element attention mask, the encoder masks padded frames, and the
        decoder stops each chunk at its own ``valid_lengths``.

        Returns ``(texts, meta_tokens, meta_frames, valid_enc_list, timing)``
        where each list has ``B`` entries (one per chunk, in input order) and
        ``timing`` carries batch-level ``mel_ms``/``encoder_ms``/``decode_ms``/
        ``total_ms``/``batch_size`` from cuda events. Raises :class:`MemoryError`
        if the VRAM hard floor (:attr:`min_free_vram_gb`) is breached.
        """
        free_gb = self._free_vram_gb()
        if free_gb < self.min_free_vram_gb:
            raise MemoryError(
                f"chunked: free VRAM {free_gb:.2f} GB < hard floor "
                f"{self.min_free_vram_gb:.2f} GB; aborting batch "
                f"(free GPU memory or reduce chunk_batch_size)"
            )

        pipe = self.pipeline
        B = len(batch_audio)

        def _timed(fn):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            out = fn()
            end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(end), out

        mel_ms, (input_features, attention_mask) = _timed(
            lambda: pipe.mel(batch_audio)
        )
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

        valid_enc_list = [int(valid_lengths[b].item()) for b in range(B)]
        timing = {
            "mel_ms": float(mel_ms),
            "encoder_ms": float(encoder_ms),
            "decode_ms": float(decode_ms),
            "total_ms": float(mel_ms + encoder_ms + decode_ms),
            "batch_size": int(B),
        }
        return texts, meta_tokens, meta_frames, valid_enc_list, timing

    @torch.inference_mode()
    def _decode_chunk(
        self, chunk_audio: np.ndarray
    ) -> Tuple[str, List[int], List[int], int, dict]:
        """Single-chunk decode (``B=1``) -- thin wrapper over :meth:`_decode_batch`.

        Kept for backward compatibility; returns ``(text, tokens, local_frames,
        valid_enc_frames, timing)``. At ``B=1`` this is byte-exact with the
        direct ``MegaParakeetPipeline.transcribe`` path (identical mel / encoder
        / decode_with_durations calls on the same single chunk).
        """
        texts, meta_tokens, meta_frames, valid_enc_list, timing = self._decode_batch(
            [chunk_audio]
        )
        return texts[0], meta_tokens[0], meta_frames[0], valid_enc_list[0], timing

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

        Chunks are grouped left-to-right into mini-batches of up to
        :attr:`chunk_batch_size` (sized down by :meth:`_effective_batch_size`
        when free VRAM is low). Each mini-batch is run through the batched
        mel+encoder+decode in one set of forwards; the per-chunk token+durations
        streams are split back out and stitched exactly as in the sequential
        path (frame-aligned left-biased dedup by global sample position).

        ``summary`` contains: ``total_ms`` (wall, cuda-event bracketed over the
        whole run), ``audio_seconds``, ``n_chunks``, ``n_batches``,
        ``chunk_batch_size``, ``n_tokens_surviving``, ``n_stitches`` (overlap-
        region tokens dropped), ``peak_vram_gb``
        (``torch.cuda.max_memory_allocated`` over the whole run, reset at the
        start -- tracks ONE mini-batch, independent of total length), and both
        ``per_batch`` (batch-level per-stage ms + batch size) and ``per_chunk``
        (per-chunk token counts + an evenly-distributed per-stage ms estimate,
        kept for backward compatibility with the sequential summary shape).
        """
        if int(sr) != self.sr:
            raise ValueError(f"sr={sr} != pipeline sr {self.sr}")
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        chunks, starts = self._plan_chunks(audio)
        assert len(chunks) >= 1, "must produce at least one chunk"
        n_chunks = len(chunks)

        surviving_tokens: List[int] = []
        # furthest global sample position covered by any KEPT token so far.
        # left-biased dedup: drop any token whose global_sample <= this.
        furthest_global_sample = -1
        per_batch: List[dict] = []
        per_chunk: List[dict] = []
        n_stitches = 0

        # Peak-VRAM is measured over the whole run (one mini-batch's worth
        # because we empty_cache() between batches); reset to this call only.
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        t_start = torch.cuda.Event(enable_timing=True)
        t_end = torch.cuda.Event(enable_timing=True)
        t_start.record()

        # Group chunks into mini-batches of up to chunk_batch_size (sized down
        # adaptively by _effective_batch_size when free VRAM is low). The last
        # group may contain fewer than chunk_batch_size chunks; it runs at its
        # natural B (no dummy padding -- see the class docstring).
        batch_idx = 0
        ci = 0
        while ci < n_chunks:
            eff_b = self._effective_batch_size(self.chunk_batch_size)
            end_ci = min(ci + eff_b, n_chunks)
            batch_audio = chunks[ci:end_ci]
            batch_starts = starts[ci:end_ci]
            actual_b = len(batch_audio)

            _texts_b, tokens_b, frames_b, valid_enc_b, timing = self._decode_batch(
                batch_audio
            )

            # Stitch each chunk in the batch (in order), using the same
            # frame-aligned left-biased dedup as the sequential path.
            for k in range(actual_b):
                gci = ci + k
                start_sample = batch_starts[k]
                tokens_c = tokens_b[k]
                frames_c = frames_b[k]
                chunk_audio_k = batch_audio[k]

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

                # Per-chunk record: token counts are exact; the per-stage ms are
                # the batch's totals distributed evenly across its chunks (an
                # approximation kept for backward-compat with the sequential
                # summary shape -- use ``per_batch`` for exact batch timings).
                per_chunk.append(
                    {
                        "chunk_idx": int(gci),
                        "batch_idx": int(batch_idx),
                        "start_sample": int(start_sample),
                        "start_s": start_sample / self.sr,
                        "n_samples": int(chunk_audio_k.shape[0]),
                        "duration_s": chunk_audio_k.shape[0] / self.sr,
                        "valid_enc_frames": int(valid_enc_b[k]),
                        "tokens_emitted": int(len(tokens_c)),
                        "tokens_kept": int(kept_here),
                        "mel_ms": timing["mel_ms"] / actual_b,
                        "encoder_ms": timing["encoder_ms"] / actual_b,
                        "decode_ms": timing["decode_ms"] / actual_b,
                        "total_ms": timing["total_ms"] / actual_b,
                    }
                )

            per_batch.append(
                {
                    "batch_idx": int(batch_idx),
                    "batch_size": int(actual_b),
                    "chunk_start_idx": int(ci),
                    "chunk_end_idx": int(end_ci),
                    **timing,
                }
            )

            ci = end_ci
            batch_idx += 1
            # Release per-batch intermediates so peak VRAM tracks one batch.
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
            "n_chunks": n_chunks,
            "n_batches": int(batch_idx),
            "chunk_batch_size": int(self.chunk_batch_size),
            "n_tokens_surviving": int(len(surviving_tokens)),
            "n_stitches": int(n_stitches),
            "peak_vram_gb": float(peak_vram_gb),
            "chunk_seconds": self.chunk_seconds,
            "overlap_seconds": self.overlap_seconds,
            "samples_per_enc_frame": self.samples_per_enc_frame,
            "per_batch": per_batch,
            "per_chunk": per_chunk,
        }
        return text, summary
