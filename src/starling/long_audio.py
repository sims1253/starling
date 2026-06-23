"""Long-audio (1 min - 1 h) chunked transcription for Granite-Speech-4.1-2b.

The fused LLM decoder uses a fixed ``StaticCache`` of 640 tokens.  Audio is
downsampled 10x in total (2x conformer encoder + 5x BLIP2 q-former projector),
so each second of audio consumes ~10 LLM prompt tokens plus ~22 text tokens for
the chat-template prompt.  That caps a single-shot transcribe at roughly 62 s
before the 640-token KV cache overflows (prompt + new tokens > 640).

This module handles arbitrary-length audio (1 min up to 1 h and beyond) by
windowing it into cache-safe ~30 s chunks and transcribing each independently
with a fresh prompt.  Per-chunk transcripts are concatenated.  Chunk boundaries
may cause minor word splits; this is a *speed + memory* benchmark, not a WER
benchmark.

Design notes
------------
* Chunks are taken at the **waveform** level (not mel level).  Each chunk is
  fed through the full processor (mel extraction + tokenization) so it reuses
  all existing input-construction machinery.  Peak RAM/VRAM stays bounded
  because only one chunk's features live at a time.
* The last chunk is zero-padded up to ``chunk_seconds`` so every chunk has an
  identical mel-feature shape.  This is REQUIRED for the non-speculative path:
  the fused encoder is a CUDA graph captured for one static input shape, and a
  differently-sized final chunk would raise.  Padding the tail with silence is
  harmless for a speed benchmark.
* The KV cache (and all CUDA graphs) are reset per chunk, so peak VRAM is
  constant regardless of the total audio length.

Public API
----------
``synthesize_long_audio(target_seconds) -> (wav, sr)``
``chunk_audio(wav, sr, chunk_seconds) -> iterator``
``transcribe_long(pipe, processor, wav, sr, ...) -> LongTranscribeResult``
``transcribe_long_stock(model, processor, wav, sr, ...) -> LongTranscribeResult``
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

import torch

from .config import DEFAULT_TASK_PROMPT

SAMPLE_SR: int = 16000
"""Sample rate of the Granite-Speech feature extractor (16 kHz)."""

DEFAULT_CHUNK_SECONDS: float = 30.0
"""30 s of audio -> ~300 audio tokens + ~22 chat-template tokens = ~322 prompt
tokens, leaving room for ~300 generated tokens inside the 640-token
StaticCache.  Comfortably cache-safe for ``max_new_tokens <= ~300``."""


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class ChunkResult:
    """Per-chunk transcription result."""

    index: int
    start_s: float
    end_s: float
    text: str
    n_tokens: int
    ms: float


@dataclass
class LongTranscribeResult:
    """Aggregated chunked-transcription result."""

    text: str
    chunks: list[ChunkResult] = field(default_factory=list)
    total_ms: float = 0.0
    n_chunks: int = 0
    total_tokens: int = 0
    audio_seconds: float = 0.0
    rtfx: float = 0.0
    """``audio_seconds / total_seconds`` (higher is faster)."""
    speculative: bool = False
    extrapolated: bool = False
    """True if ``total_ms`` was extrapolated from a per-chunk measurement
    (e.g. stock on very long audio) rather than measured end-to-end."""

    @property
    def tokens_per_s(self) -> float:
        return self.total_tokens / max(self.total_ms / 1000.0, 1e-9)

    @property
    def per_chunk_ms(self) -> float:
        return self.total_ms / max(self.n_chunks, 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text_preview": self.text[:200],
            "total_ms": round(self.total_ms, 2),
            "n_chunks": self.n_chunks,
            "total_tokens": self.total_tokens,
            "audio_seconds": round(self.audio_seconds, 3),
            "rtfx": round(self.rtfx, 3),
            "tokens_per_s": round(self.tokens_per_s, 2),
            "per_chunk_ms": round(self.per_chunk_ms, 2),
            "speculative": self.speculative,
            "extrapolated": self.extrapolated,
        }


# ---------------------------------------------------------------------------
# Audio synthesis + chunking
# ---------------------------------------------------------------------------
def synthesize_long_audio(
    target_seconds: float,
    base_wav: Optional[torch.Tensor] = None,
    sr: int = SAMPLE_SR,
) -> tuple[torch.Tensor, int]:
    """Synthesize long audio by tiling the 24.9 s sample, then trimming.

    Keeps content predictable (the repeated multilingual sample) and avoids
    needing new reference data.  For 1 h this is ~3.6 M samples at 16 kHz,
    i.e. ~14.4 MB float32 in RAM.

    Args:
        target_seconds: Desired duration in seconds.
        base_wav: Optional ``(1, N)`` float32 base waveform to tile.  If None,
            the Granite-Speech sample audio is loaded.
        sr: Sample rate (16 kHz).

    Returns:
        ``(wav, sr)`` where ``wav`` is ``(1, target_seconds*sr)`` float32.
    """
    if base_wav is None:
        from .audio import load_sample_audio

        base_wav, sr = load_sample_audio()
    base_samples = int(base_wav.shape[1])
    target_samples = int(round(target_seconds * sr))
    if target_samples <= 0:
        raise ValueError(f"target_seconds must be > 0, got {target_seconds}")
    reps = max(1, (target_samples + base_samples - 1) // base_samples)
    tiled = base_wav.repeat(1, reps)[:, :target_samples]
    return tiled.contiguous(), sr


def chunk_audio(
    wav: torch.Tensor,
    sr: int,
    chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
    *,
    pad_last: bool = True,
) -> Iterator[tuple[torch.Tensor, float, float, int]]:
    """Yield ``(chunk_wav, start_s, end_s, index)`` windows.

    With ``pad_last=True`` the final window is zero-padded up to
    ``chunk_seconds`` so every chunk has an identical mel-feature shape.  This
    is REQUIRED for the non-speculative path (the fused encoder is a CUDA graph
    captured for one static shape).
    """
    chunk_samples = int(round(chunk_seconds * sr))
    total = int(wav.shape[1])
    pos = 0
    idx = 0
    while pos < total:
        end = min(pos + chunk_samples, total)
        chunk = wav[:, pos:end]
        real_end_s = end / sr
        if pad_last and chunk.shape[1] < chunk_samples:
            chunk = torch.nn.functional.pad(
                chunk, (0, chunk_samples - chunk.shape[1])
            )
        yield chunk.contiguous(), pos / sr, real_end_s, idx
        pos = end
        idx += 1


def n_chunks_for(total_seconds: float, chunk_seconds: float) -> int:
    """Number of chunks produced by :func:`chunk_audio` for a given duration."""
    return max(1, int(-(-int(round(total_seconds)) // int(round(chunk_seconds)))))


def _join_chunk_texts(texts: list[str]) -> str:
    """Concatenate per-chunk transcripts, collapsing whitespace."""
    joined = " ".join(t.strip() for t in texts if t and t.strip())
    return " ".join(joined.split())


# ---------------------------------------------------------------------------
# Mega (fused) chunked transcription
# ---------------------------------------------------------------------------
@torch.inference_mode()
def transcribe_long(
    pipe: Any,
    processor: Any,
    wav: torch.Tensor,
    sr: int,
    *,
    chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
    max_new_tokens: int = 200,
    speculative: bool = True,
    dtype: torch.dtype = torch.bfloat16,
    task_prompt: str = DEFAULT_TASK_PROMPT,
) -> LongTranscribeResult:
    """Chunked transcription of arbitrary-length audio with the mega pipeline.

    Each chunk gets a fresh chat-template prompt and is transcribed
    independently; the KV cache is reset for every chunk so peak VRAM is
    constant regardless of total audio length.
    """
    from .audio import build_inputs

    max_cache_len = int(getattr(pipe.llm, "max_cache_len", 640))
    chunks: list[ChunkResult] = []
    texts: list[str] = []
    total_tokens = 0
    t0 = time.perf_counter()
    for chunk_wav, start_s, end_s, idx in chunk_audio(wav, sr, chunk_seconds):
        inputs = build_inputs(processor, chunk_wav, task_prompt=task_prompt)
        feats = inputs["input_features"].to(dtype)
        ids = inputs["input_ids"]
        mask = inputs.get("input_features_mask")
        prompt_len = int(ids.shape[1])
        # Clamp the generation budget so prompt + new tokens stays within the
        # static KV cache (leaves a 1-token safety margin).
        budget = max(1, min(max_new_tokens, max_cache_len - prompt_len - 1))
        c0 = time.perf_counter()
        text, gen_ids = pipe.transcribe(
            feats,
            ids,
            mask,
            max_new_tokens=budget,
            speculative=speculative,
        )
        torch.cuda.synchronize()
        cms = (time.perf_counter() - c0) * 1000.0
        n_tok = int(gen_ids.shape[1])
        chunks.append(ChunkResult(idx, start_s, end_s, text, n_tok, cms))
        texts.append(text)
        total_tokens += n_tok
    torch.cuda.synchronize()
    total_ms = (time.perf_counter() - t0) * 1000.0
    audio_seconds = wav.shape[1] / sr
    full_text = _join_chunk_texts(texts)
    return LongTranscribeResult(
        text=full_text,
        chunks=chunks,
        total_ms=total_ms,
        n_chunks=len(chunks),
        total_tokens=total_tokens,
        audio_seconds=audio_seconds,
        rtfx=audio_seconds / max(total_ms / 1000.0, 1e-9),
        speculative=speculative,
    )


# ---------------------------------------------------------------------------
# Stock transformers chunked transcription
# ---------------------------------------------------------------------------
@torch.inference_mode()
def transcribe_long_stock(
    model: Any,
    processor: Any,
    wav: torch.Tensor,
    sr: int,
    *,
    chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
    max_new_tokens: int = 200,
    dtype: torch.dtype = torch.bfloat16,
    task_prompt: str = DEFAULT_TASK_PROMPT,
) -> LongTranscribeResult:
    """Chunked transcription with the stock transformers ``generate()`` path.

    Stock uses a DynamicCache (grows with usage) so it would not crash on a
    single very long prompt, but the LLM's RoPE max position embeddings (4096)
    and quadratic prefill attention make single-shot long audio both wrong and
    slow.  Chunking keeps it correct and comparable to the mega path.
    """
    from .audio import build_inputs

    chunks: list[ChunkResult] = []
    texts: list[str] = []
    total_tokens = 0
    t0 = time.perf_counter()
    for chunk_wav, start_s, end_s, idx in chunk_audio(wav, sr, chunk_seconds):
        inputs = build_inputs(processor, chunk_wav, task_prompt=task_prompt)
        feats = inputs["input_features"].to(dtype)
        ids = inputs["input_ids"]
        am = inputs["attention_mask"]
        mask = inputs.get("input_features_mask")
        prompt_len = int(ids.shape[1])
        c0 = time.perf_counter()
        gen = model.generate(
            input_ids=ids,
            input_features=feats,
            attention_mask=am,
            input_features_mask=mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
        )
        torch.cuda.synchronize()
        cms = (time.perf_counter() - c0) * 1000.0
        n_new = int(gen.shape[1]) - prompt_len
        text = processor.tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
        chunks.append(ChunkResult(idx, start_s, end_s, text, n_new, cms))
        texts.append(text)
        total_tokens += max(n_new, 0)
    torch.cuda.synchronize()
    total_ms = (time.perf_counter() - t0) * 1000.0
    audio_seconds = wav.shape[1] / sr
    full_text = _join_chunk_texts(texts)
    return LongTranscribeResult(
        text=full_text,
        chunks=chunks,
        total_ms=total_ms,
        n_chunks=len(chunks),
        total_tokens=total_tokens,
        audio_seconds=audio_seconds,
        rtfx=audio_seconds / max(total_ms / 1000.0, 1e-9),
        speculative=False,
    )


def extrapolate_from_chunk(
    per_chunk_ms: float,
    n_chunks: int,
    audio_seconds: float,
    tokens_per_chunk: int,
    *,
    speculative: bool = False,
) -> LongTranscribeResult:
    """Build an extrapolated result from a per-chunk measurement.

    Used for stock (and optionally non-spec mega) on very long audio where an
    end-to-end run would take too long.  The result is flagged
    ``extrapolated=True`` so callers can label it clearly.
    """
    total_ms = per_chunk_ms * n_chunks
    return LongTranscribeResult(
        text="<extrapolated>",
        chunks=[],
        total_ms=total_ms,
        n_chunks=n_chunks,
        total_tokens=tokens_per_chunk * n_chunks,
        audio_seconds=audio_seconds,
        rtfx=audio_seconds / max(total_ms / 1000.0, 1e-9),
        speculative=speculative,
        extrapolated=True,
    )
